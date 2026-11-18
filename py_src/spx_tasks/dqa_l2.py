__all__ = ["DQAL2Config", "DQAL2Task"]

# Standard library imports
import copy
import logging

# Related third party imports
from lsst.pex.config import Field, DictField, RangeField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy.nddata import VarianceUncertainty
from astropy.table import Table
from astropy.units import Unit
from astropy.wcs import WCS

# Local application/library specific imports
from spx_data import SPHERExImage, Level2DQA, Flag, FLAG_NAME_ALIASES
from spx_tasks import add_history
from spx_utils.idqr_utils import get_idqr
from spx_tasks.update_persistence import dqa_check_persistence_flux
from spx_tasks.apply_bad_pixel_mask import dqa_check_hot_cold_pixel_fraction
# from spx_tasks.apply_flat_field_and_photo_calib import dqa_check_flat_field
from spx_tasks.query_reference_catalog import dqa_check_source_count
from spx_tasks.create_source_mask import dqa_check_masked_sources
from spx_tasks.create_source_mask import (dqa_check_ghost_flux,
                                          dqa_check_crosstalk_flux)
from spx_tasks.detect_outlier_pixels import dqa_check_outlier_pixel_count
from spx_utils.dqa_utils import log_dqa_result, nan_variance_flagged, nan_pixels_flagged
from spx_utils.utils import get_refcat_provenance

# ==============================================================================
logger = logging.getLogger(__name__)


def dqa_check_variance_sigma(uncertainty: VarianceUncertainty,
                             expected: float,
                             tolerance: float,
                             flags: np.ndarray = None,
                             mask_flags: int = None) -> dict:
    """Perform a DQA check on the square root of the variance map,
    excluding flagged pixels.

    Parameters
    ----------
    uncertainty : `VarianceUncertainty`
        Variance layer from SPHERExImage.
    expected : `float`
        Expected value for sqrt(variance).
    tolerance : `float`
        Allowed deviation from expected value.
    flags : `np.ndarray`, optional
        Flag array from SPHERExImage (same shape as uncertainty).
    mask_flags : `int`, optional
        Bitmask of flags to exclude from the variance check.

    Returns
    -------
    result : `dict`
        Dictionary with result ('PASS'/'FAIL') and description.

    """
    if not isinstance(uncertainty, VarianceUncertainty):
        raise TypeError(f"Expected VarianceUncertainty, got {type(uncertainty)}")

    expected_unit = Unit("MJy/sr") ** 2
    if uncertainty.unit != expected_unit:
        result = {
            "result": "FAIL",
            "description": f"Variance has incorrect unit: {uncertainty.unit}. Expected {expected_unit}."
        }
        log_dqa_result("variance sigma check", result)
        return result

    sigma = np.sqrt(uncertainty.array)

    if flags is not None and mask_flags is not None:
        valid_mask = (flags & mask_flags) == 0
        sigma = sigma[valid_mask]

    median_sigma = np.nanmedian(sigma)
    lower = expected - tolerance
    upper = expected + tolerance

    if lower <= median_sigma <= upper:
        result = {
            "result": "PASS",
            "description": f"Median sigma {median_sigma:.4e} is within expected range "
                           f"[{lower:.4e}, {upper:.4e}]."
        }
    else:
        result = {
            "result": "FAIL",
            "description": f"Median sigma {median_sigma:.4e} is outside expected range "
                           f"[{lower:.4e}, {upper:.4e}]."
        }

    log_dqa_result("variance sigma check", result)
    return result


def dqa_check_zodi_exists(spximg: SPHERExImage) -> dict:
    """Check if the ZODI layer exists on the SPHERExImage.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image object.

    Returns
    -------
    result : `dict`
        DQA result dictionary with "result" and "description".

    """
    try:
        _ = spximg.zodi
        result = {
            "result": "PASS",
            "description": "ZODI layer is present in the image."
        }
    except AttributeError:
        result = {
            "result": "FAIL",
            "description": "ZODI layer is missing from the image."
        }
    log_dqa_result("check zodi layer", result)
    return result


def dqa_check_all_layers_present(spximg: SPHERExImage) -> dict:
    """Check that all required data layers are present in SPHERExImage.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image to check.

    Returns
    -------
    result : `dict`
        Dictionary with DQA result and description.

    """
    missing = []

    # Check each required attribute
    for attr in ["data", "flags", "uncertainty", "zodi", "psfcube"]:
        try:
            _ = getattr(spximg, attr)
        except AttributeError:
            missing.append(attr)

    if not missing:
        result = {
            "result": "PASS",
            "description": "All required layers (data, flags, uncertainty, zodi, psfcube) are present."
        }
    else:
        result = {
            "result": "FAIL",
            "description": f"Missing layer(s): {', '.join(missing)}"
        }
    log_dqa_result("check L2 layers", result)
    return result


def dqa_check_solid_angle(spximg, ranges: dict, tolerance_pct: float) -> dict:
    """Perform a DQA check on the pixel solid angle solution for a given detector array.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image with header keywords OMEGA_MEDIAN and OMEGA_STDDEV.
    ranges : `dict`
        Dictionary with expected median and std dev ranges per detector ID.
    tolerance_pct : `float`
        Allowed percentage tolerance for the standard deviation check.

    Returns
    -------
    result : `dict`
        Dictionary with DQA "result" and "description".

    """
    arr = spximg.getArrayId()
    if "OMEGA_MEDIAN" not in spximg.meta or "OMEGA_STDDEV" not in spximg.meta:
        return {
            "result": "FAIL",
            "description": "OMEGA_MEDIAN or OMEGA_STDDEV metadata missing. Cannot assess pixel solid angle."
        }

    med = spximg.meta["OMEGA_MEDIAN"]
    std = spximg.meta["OMEGA_STDDEV"]

    med_min = ranges.get(f"{arr}_med_min")
    med_max = ranges.get(f"{arr}_med_max")
    std_ref = ranges.get(f"{arr}_std")
    std_tol = std_ref * (tolerance_pct / 100)

    med_ok = med_min <= med <= med_max
    std_ok = (std_ref - std_tol) <= std <= (std_ref + std_tol)

    result = {
        "result": "PASS" if (med_ok and std_ok) else "FAIL",
        "description": (
            f"Detector {arr} solid angle median = {med:.3f}, "
            f"expected range = [{med_min:.3f}, {med_max:.3f}]; "
            f"stddev = {std:.3f}, expected range = [{std_ref - std_tol:.3f}, {std_ref + std_tol:.3f}]."
        )
    }
    return result


def dqa_check_psf_solid_angle(spximg, expected_fwhm: float = 7.00, tolerance_fraction: float = 0.30) -> dict:
    """Check that the PSF solid angle matches the expected value
       for a Gaussian PSF within a specified tolerance.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image with "HIERARCH.PSF_OMEGA" in header.
    expected_fwhm : `float`, optional
        Expected FWHM of the Gaussian PSF in pixels (default = 7.00).
    tolerance_fraction : `float`, optional
        Allowed fractional difference (e.g., 0.30 for ±30%).

    Returns
    -------
    result : `dict`
        Dictionary with DQA "result" and "description".

    """
    try:
        omega_measured = spximg.header["HIERARCH.PSF_OMEGA"]
    except KeyError:
        result = {
            "result": "FAIL",
            "description": "HIERARCH.PSF_OMEGA keyword missing from header. Cannot assess PSF solid angle."
        }
        log_dqa_result("psf solid angle check", result)
        return result

    # Compute expected solid angle for 2D Gaussian
    sigma = expected_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    omega_expected = 2 * np.pi * sigma**2

    lower = omega_expected * (1 - tolerance_fraction)
    upper = omega_expected * (1 + tolerance_fraction)

    if lower <= omega_measured <= upper:
        result = {
            "result": "PASS",
            "description": (
                f"Measured PSF solid angle = {omega_measured:.4f} "
                f"is within ±{tolerance_fraction*100:.0f}% of "
                f"expected value {omega_expected:.4f} (range: {lower:.4f} – {upper:.4f})."
            )
        }
    else:
        result = {
            "result": "FAIL",
            "description": (
                f"Measured PSF solid angle = {omega_measured:.4f} "
                f"is outside ±{tolerance_fraction*100:.0f}% of "
                f"expected value {omega_expected:.4f} (range: {lower:.4f} – {upper:.4f})."
            )
        }

    log_dqa_result("psf solid angle check", result)
    return result


def dqa_check_srccat_coverage(spximg: SPHERExImage,
                              srccat: Table,
                              required_pad: int = 250) -> dict:
    """Check that the srccat covers the image with at least `required_pad`
    pixels of margin on all four sides, in the image's pixel frame.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The Level-2 image (must have .wcs and .data).
    srccat : `astropy.table.Table`
        Source catalog with RA/Dec columns ('ra'/'dec').
    required_pad : `int`, optional
        Required padding in pixels beyond the image edge on each side.

    Returns
    -------
    result : `dict`
        DQA result dictionary with "result" and "description".

    """
    # get ra, dec columns
    colnames = {c.lower(): c for c in srccat.colnames}
    if "ra" not in colnames or "dec" not in colnames:
        result = {
            "result": "FAIL",
            "description": "srccat coverage check: no RA/Dec columns found in catalog."
        }
        log_dqa_result("srccat coverage check", result)
        return result

    ra_col = colnames["ra"]
    dec_col = colnames["dec"]

    ra = np.array(srccat[ra_col])
    dec = np.array(srccat[dec_col])

    if ra.size == 0:
        result = {
            "result": "FAIL",
            "description": "srccat coverage check: empty catalog."
        }
        log_dqa_result("srccat coverage check", result)
        return result

    w: WCS = spximg.wcs
    nx, ny = spximg.data.shape

    # Project srccat ra/dec to pixel coordinates
    # origin=0 -> 0-based pixel coords
    x, y = w.all_world2pix(ra, dec, 0)

    # Drop any NaNs just in case
    good = np.isfinite(x) & np.isfinite(y)
    if not np.any(good):
        result = {
            "result": "FAIL",
            "description": "srccat coverage check: all projected positions are NaN/Inf."
        }
        log_dqa_result("srccat coverage check", result)
        return result

    x = x[good]
    y = y[good]

    # Compute minimum and maximum projected x and y
    xmin = float(np.min(x))
    xmax = float(np.max(x))
    ymin = float(np.min(y))
    ymax = float(np.max(y))

    # Compute required border padding
    # Catalog must include sources >= 250 pix left of the image
    pad_left = 0.0 - xmin
    # Catalog must include sources >= 250 pix right of the image
    pad_right = xmax - (nx - 1)
    pad_bottom = 0.0 - ymin
    pad_top = ymax - (ny - 1)
    # Means requirement to pass is: (in pixel space)
    # xmin_srccat <= −250
    # xmax_srccat >= nx−1 + 250 (nx = 2040)
    # ymin_srccat <= −250
    # ymax_srccat >= ny−1 + 250 (ny= = 2040)

    pads = {
        "pad_left": pad_left,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "pad_top": pad_top,
    }

    # require coverage on all sides
    all_good = all(p >= required_pad for p in pads.values())

    if all_good:
        result = {
            "result": "PASS",
            "description": (
                "srccat covers image with sufficient margin: "
                f"left={pad_left:.1f}px, right={pad_right:.1f}px, "
                f"bottom={pad_bottom:.1f}px, top={pad_top:.1f}px "
                f"(required ≥ {required_pad}px)."
            ),
        }
    else:
        bad_sides = [name for name, p in pads.items() if p < required_pad]
        result = {
            "result": "FAIL",
            "description": (
                "srccat does not provide sufficient border padding. "
                f"left={pad_left:.1f}px, right={pad_right:.1f}px, "
                f"bottom={pad_bottom:.1f}px, top={pad_top:.1f}px; "
                f"required ≥ {required_pad}px on all sides. "
                f"Sides below requirement: {', '.join(bad_sides)}."
            ),
        }

    # Include the pad values into result for YAML/debug
    result.update(pads)

    log_dqa_result("srccat coverage check", result)
    return result


class DQAL2Connections(pipeBase.PipelineTaskConnections,
                       dimensions=("instrument", "exposure"),
                       defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    srccat = cT.Input(
        name="srccat",
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "dichroicpair", "spsplan", "slewgroup"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output data quality assessment for level 2.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    dqa_l2 = cT.Output(
        name="dqa_l2",
        doc="Level 2 DQA record.",
        storageClass="Level2DQA",
        dimensions=("instrument", "exposure"),
    )
    idqr_obs = cT.Output(
        name="idqr_obs",
        doc="IDQR per observation.",
        storageClass="IdqrObs",
        dimensions=("instrument", "exposure"),
    )


class DQAL2Config(pipeBase.PipelineTaskConfig,
                  pipelineConnections=DQAL2Connections):
    """Configuration for DQAL2 (Level 2 Data Quality Assessment) task."""

    variance_range = DictField(
        doc="Expected range of variance plane values, in MJy/sr.",
        keytype=str,
        itemtype=float,
        default={"value": 1e-3, "range": 1e-3})

    pct_bad_pixels = RangeField(
        doc="Maximum acceptable percentage of bad pixels.",
        dtype=float,
        min=0.,
        max=100.,
        default=1.)

    apply_abs_gain_range = DictField(
        doc="Expected range of pixel fluxes after applying absolute gain"
            " matrix correction, in MJy/sr.",
        keytype=str,
        itemtype=float,
        # TODO: populate default with actual desired values
        default={"value": 0.0, "range": 0.0})

    qrc_factor = Field(
        doc="Factor with which number of Reference Catalog sources in the mini"
            "-catalog are consisent with expected results.",
        dtype=int,
        default=3)

    sso_factor = Field(
        doc="Factor with which number of SSOs in the mini-catalog are "
            "consistent with expected results.",
        dtype=int,
        default=2)

    src_mask_factor = Field(
        doc="Factor with which number of sources in mini-catalog agrees with"
            " the number of pixels masked in the image.",
        dtype=int,
        default=3)

    outlier_factor = Field(
        doc="Factor with which number of additionally masked pixels agrees "
            "with the number of source masked pixels in the image.",
        dtype=int,
        default=2)

    est_abs_gain_pct = RangeField(
        # NOTE: only for Year-1 test - then: may replace with flat-field
        # derived in TVAC during IOC, and with flat-field derived in previous
        # iteration during operations
        doc="Percent agreement between the true flat injected in simulations"
            " and the flat field recovered by the module.",
        dtype=float,
        min=0.,
        max=100.,
        default=10.)

    est_dc_pct = RangeField(
        # NOTE: only for Year-1 test - then: may replace with dark currents
        # derived in TVAC during IOC, and with dark current derived in previous
        # iteration during operations
        doc="Percent agreement between the true dark injected in simulations"
            " and the dark recovered by the module.",
        dtype=float,
        min=0.,
        max=100.,
        default=10.)

    solid_pix_pct = RangeField(
        doc="Percent agreement between pixel solid angle solutions and "
            "expected value.",
        dtype=float,
        min=0.,
        max=100.,
        default=10.)

    solid_pix_ranges = DictField(
        doc="Expected range of single-exposure solid angle pixel medians and "
            " standard deviations.",
        keytype=str,
        itemtype=float,
        default={
            "1_med_min": 37.42, "1_med_max": 38.03, "1_std": 0.23,
            "2_med_min": 37.24, "2_med_max": 37.71, "2_std": 0.17,
            "3_med_min": 37.43, "3_med_max": 38.05, "3_std": 0.23,
            "4_med_min": 37.50, "4_med_max": 38.14, "4_std": 0.24,
            "5_med_min": 37.32, "5_med_max": 37.80, "5_std": 0.17,
            "6_med_min": 37.49, "6_med_max": 38.12, "6_std": 0.23
        })

    nan_flag_mask = Field(
        doc="Bitmask of flags that justify NaN or Inf values",
        dtype=int,
        default=(Flag.MISSING_DATA | Flag.NONFUNC | Flag.SUR_ERROR)
    )


class DQAL2Task(pipeBase.PipelineTask):
    """Implementation of L2 Perform Data Quality Assessment for Level 2.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-025

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "dqa-l2"
    ConfigClass = DQAL2Config
    full_name = "Perform Data Quality Assessment for Level 2"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-025 v1.4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage, srccat: Table) -> pipeBase.Struct:
        # Set result to True (Pass) to begin with;
        # if any checks fail, set to False
        dqa_result = True

        # L2 module dqa residuals checks
        nl_dqa_res = spximg.meta.get("NL_DQA", "FAIL")
        correction_limit = spximg.meta['corlim']
        median_residual = spximg.meta['medres']
        if nl_dqa_res == "FAIL":
            n_over_corrected = spximg.meta['ovrcor']
            max_corr = spximg.meta['maxcor']
            median_corr = spximg.meta['medcor']
            nl_dqa_desc = (f"{n_over_corrected} pixels corrected > {int(correction_limit * 100)}%. "
                           f"Max = {max_corr:.2f}, mean over limit = {median_corr:.2f}, "
                           f"median |residual| = {median_residual:.2f}.")
            spximg.meta.pop('ovrcor', None)
            spximg.meta.pop('maxcor', None)
            spximg.meta.pop('medcor', None)
        else:
            n_corr = spximg.meta['n_corr']
            nl_dqa_desc = (f"All {n_corr} pixels corrected within {int(correction_limit * 100)}% limit. "
                           f"Median |residual|: {median_residual:.2f}")
            spximg.meta.pop("n_corr", None)

        nonlinear_check = {'result': nl_dqa_res,
                           'description': nl_dqa_desc}

        log_dqa_result("nonlinearity dqa", nonlinear_check)
        # Safely remove from header
        spximg.meta.pop("corlim", None)
        spximg.meta.pop("medres", None)
        spximg.meta.pop("NL_DQA", None)

        persistence_check = dqa_check_persistence_flux(spximg, max_fraction=0.2)
        badpix_check = dqa_check_hot_cold_pixel_fraction(spximg, threshold=1.0)

        # flat_field_check =
        # dqa_check_flat_field(spximg, abs_gain_matrix, threshold=0.30)

        # srccat coverage vs image check
        # Only perform this check if FINAST == 0 (good fine astrometry).

        # srccat coverage vs image check
        # Only perform this check if FINAST == 0 (good fine astrometry).
        finast_value = spximg.header.get("FINAST", -1)
        if finast_value == 0:
            try:
                srccat_coverage_check = dqa_check_srccat_coverage(spximg,
                                                                  srccat,
                                                                  required_pad=250)
            except Exception as e:
                # Never crash
                # Log and return a structured DQA failure
                self.log.warning("srccat coverage check failed; treating as DQA failure: %s",
                                 str(e),
                                 exc_info=True)

                srccat_coverage_check = {"result": "FAIL",
                                         "description": ("srccat coverage check failed due to exception "
                                                         f"({type(e).__name__}: {e})"), }
        else:
            srccat_coverage_check = {"result": "PASS",
                                     "description": ("srccat coverage check skipped because FINAST != 0 "
                                                     "(fine astrometry not successful)."), }

        log_dqa_result("srccat coverage check", srccat_coverage_check)

        # Source count check
        source_count_check = dqa_check_source_count(srccat)

        # Source masking checks
        source_mask_check = dqa_check_masked_sources(spximg, srccat)
        ghost_flux_check = dqa_check_ghost_flux(spximg)
        crosstalk_check = dqa_check_crosstalk_flux(spximg, threshold=400.0)

        # Outlier check
        outlier_check = dqa_check_outlier_pixel_count(spximg, tolerance=self.config.outlier_factor)
        zodi_check = dqa_check_zodi_exists(spximg)
        layer_check = dqa_check_all_layers_present(spximg)

        # Variance check in units of MJy^2/sr^2
        variance_cfg = self.config.variance_range
        excluded_flags = (Flag.SOURCE | Flag.NONLINEAR | Flag.OVERFLOW
                          | Flag.SUR_ERROR | Flag.TRANSIENT)

        dqa_variance_check = dqa_check_variance_sigma(
            spximg.uncertainty,
            expected=variance_cfg["value"],
            tolerance=variance_cfg["range"],
            flags=spximg.flags,
            mask_flags=excluded_flags
        )
        log_dqa_result("dqa record check", dqa_variance_check)

        # NaN and Inf check
        nan_result = nan_pixels_flagged(spximg.data,
                                        spximg.flags,
                                        nan_flags=self.config.nan_flag_mask,
                                        check_inf=True)

        nan_check = {
            "result": "PASS" if nan_result else "FAIL",
            "description": "All NaN or Inf pixels are properly flagged."
            if nan_result else "Some NaN or Inf pixels are unflagged."
        }
        log_dqa_result("nan check", nan_check)

        # Variance NaN and Inf check
        variance_nan_result = nan_variance_flagged(
            spximg.uncertainty.array,
            spximg.flags,
            nan_flags=self.config.nan_flag_mask,
            check_inf=True
        )

        nan_variance_check = {
            "result": "PASS" if variance_nan_result else "FAIL",
            "description": "All NaN or Inf values in variance are properly flagged."
            if variance_nan_result else "Some NaN or Inf values in variance are unflagged."
        }
        log_dqa_result("nan variance check", nan_variance_check)

        # Distoriton check with pixel solid angle
        distortion_dict = dqa_check_solid_angle(
            spximg,
            ranges=self.config.solid_pix_ranges,
            tolerance_pct=self.config.solid_pix_pct
        )
        log_dqa_result("distortion dqa check", distortion_dict)

        # PSF solid angle
        _ = dqa_check_psf_solid_angle(spximg)

        # FINAST check
        finast_value = spximg.header.get("FINAST", 0)
        if finast_value != 0:
            dqa_result = False
            finast_check = {
                "result": "FAIL",
                "description": f"FINAST = {finast_value} indicates fine astrometry failure. "
                "Expected FINAST = 0."
            }
        else:
            finast_check = {
                "result": "PASS",
                "description": "FINAST = 0 indicates fine astrometry success."
            }
        log_dqa_result("fine astrometry check", finast_check)

        # Build structured results for YAML serialization
        dqa_record = Level2DQA(
            nonlinear_flag_flux_check=nonlinear_check,
            dqa_variance=dqa_variance_check,
            persistence_flag_flux_check=persistence_check,
            badpix_flag_fraction_check=badpix_check,
            # flat_field_check=flat_field_check,
            source_count_check=source_count_check,
            source_mask_check=source_mask_check,
            ghost_flag_flux_check=ghost_flux_check,
            crosstalk_flag_flux_check=crosstalk_check,
            outlier_flag_check=outlier_check,
            zodi_layer_check=zodi_check,
            layer_presence_check=layer_check,
            unflagged_nan_check=nan_check,
            unflagged_nan_variance_check=nan_variance_check,
            update_field_distortion=distortion_dict,
            # psf_solid_angle_check=psf_solid_angle_check,
            finast_check=finast_check,
            srccat_coverage_check=srccat_coverage_check
        )

        # shallow copy
        spximg_out = copy.copy(spximg)

        # Check for unflagged NaNs in the image
        # if nan_pixels_flagged(spximg.data, spximg.flags) is False:
        # self.log.error("Unflagged NaN pixels detected. Setting dqa_result to False.")
        # dqa_result = False

        # If FINAST passed, check DQA record
        if finast_check["result"] == "PASS":
            fail_count = 0
            total_checks = 0
            for value in dqa_record.model_dump().values():
                if isinstance(value, dict) and "result" in value:
                    total_checks += 1
                    if value["result"] == "FAIL":
                        fail_count += 1
            if fail_count > total_checks / 2:
                dqa_result = False
        else:
            dqa_result = False

        # Update header
        spximg_out.header['L2DQAFLG'] = ('Pass' if dqa_result else 'Fail', 'L2 DQA outcome')

        # Add final result to the record (mutate after construction)
        dqa_record.overall_result = "PASS" if dqa_result else "FAIL"

        L2_FLAGS_TO_COUNT = [
            (Flag.TRANSIENT, "flagged transient pixels"),
            (Flag.OVERFLOW, "overflow pixels"),
            (Flag.SUR_ERROR, "sur error pixels"),
            (Flag.NONFUNC, "nonfunctional pixels"),
            (Flag.DICHROIC, "pixels due to low efficiency"),
            (Flag.MISSING_DATA, "pixels with missing data"),
            (Flag.FULLSAMPLE, "full-sample pixels"),
            (Flag.PHANMISS, "phanmiss pixels"),
            (Flag.SNOWBALL, "snowball artifact pixels"),
            (Flag.BLOOM, "bloom artifact pixels"),
            (Flag.HALO, "transient halo pixels"),
            (Flag.SATELLITE_HALO, "streak halo pixels"),
        ]
        for flag, desc in L2_FLAGS_TO_COUNT:
            count = np.count_nonzero(spximg_out.flags & flag)
            flag_name = FLAG_NAME_ALIASES.get(flag.name, flag.name)
            keyword = f"HIERARCH L2 N_{flag_name}"
            description = f"number of flagged {desc}"
            spximg_out.header[keyword] = (count, description)

        # Flag plane should be unitless
        spximg_out.flag_header["BUNIT"] = ("", "Unit of flag pixel value (dimensionless)")

        idqr = get_idqr(spximg_out)
        # Remove integer target id, only needed for ODIR/IDQR reports
        spximg_out.meta.pop("SPS_TARGET_PEF", None)

        # Extract refcat provenance and write into image header
        _, ref_ver = get_refcat_provenance(srccat)
        if ref_ver:
            spximg_out.header["HIERARCH REFCAT.VERSION"] = (ref_ver, "Reference catalog version")

        return pipeBase.Struct(spximg_out=spximg_out, dqa_l2=dqa_record, idqr_obs=idqr)
