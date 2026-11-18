__all__ = ["CorrectNonLinearityConfig", "CorrectNonLinearityTask"]

# Standard library imports

# Related third party imports
import numpy as np
# import lsst.log as lsstLog
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import Flag, GainFactors, NonLinearityParameters, ReadoutParameters, spx_param, SPHERExImage
from spx_utils import l1_utils, utils
from spx_utils.utils import get_overflow_threshold
from spx_tasks import add_history
from spx_utils.dqa_utils import log_dqa_result

# ==============================================================================


class CorrectNonLinearityConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # default dataset type for input
        doc="Input level 1 image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    nonlin_pars = cT.PrerequisiteInput(
        name="nonlin_pars",
        storageClass="NonLinearityParameters",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    gain_factors = cT.PrerequisiteInput(
        name="gain_factors",
        storageClass="GainFactors",
        dimensions=["instrument"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after nonlinearity correction.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CorrectNonLinearityConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CorrectNonLinearityConnections):
    """Configuration for CorrectNonLinearity task."""

    t_exp = Field(
        "Time the detector accumulated charge, measured in seconds.",
        dtype=float,
        default=1.5349*75)  # TSAMP * SURLIM

    dqa_nonlinear_threshold = Field(
        "DQA on threshold for nonlinearityh",
        dtype=float,
        default=30000)


def dqa_check_nonlinear_flux_with_residuals(spximg_before,
                                            spximg_after,
                                            threshold=30000,
                                            correction_limit=0.5) -> dict:
    """DQA check based only on correction residuals for NONLINEAR pixels.

    Parameters
    ----------
    spximg_before : `SPHERExImage`
        Original SPHEREx image before correction.
    spximg_after : `SPHERExImage`
        Corrected SPHEREx image.
    threshold : `float`, optional
        Flux threshold for info-only diagnostics (default: 30000).
    correction_limit : `float`, optional
        Max allowed fractional correction (default: 0.5 = 50%).

    Returns
    -------
    `dict`
        DQA result with status and message.

    """
    # Select all pixels not flagged NONLINEAR
    corr_mask = (
        spximg_after.flags
        & (Flag.NONLINEAR | Flag.NONFUNC | Flag.SUR_ERROR | Flag.MISSING_DATA)
    ) == 0
    corr_flux_after = spximg_after.data[corr_mask]
    corr_flux_before = spximg_before.data[corr_mask]

    n_corr = corr_flux_after.size

    if n_corr == 0:
        description = "No corrected pixels present.."
        result_qa = {"result": "FAIL", "description": description}
        log_dqa_result("nonlinearity dqa", result_qa)
        return result_qa

    # Informational log (not used for PASS/FAIL logic)
    n_below_thresh = np.nansum(corr_flux_after <= threshold)
    print(f"[INFO] {n_below_thresh} corrected pixels are below "
          f"the {threshold} threshold after correction.")

    # Residual correction DQA (main criterion)
    residual = np.abs(corr_flux_after - corr_flux_before) / (np.abs(corr_flux_before) + 1e-8)
    median_residual = np.nanmedian(residual)
    over_corrected = residual > correction_limit
    n_over_corrected = np.nansum(over_corrected)

    if n_over_corrected > 0:
        max_corr = np.nanmax(residual)
        median_corr = np.nanmedian(residual[over_corrected])
        result_qa = {
            "result": "FAIL",
            "ovrcor": n_over_corrected,
            "corlim": correction_limit,
            "maxcor": max_corr,
            "medcor": median_corr,
            "medres": median_residual
        }
        return result_qa

    result_qa = {
        "result": "PASS",
        "n_corr": n_corr,
        "corlim": correction_limit,
        "medres": median_residual
    }
    return result_qa


class CorrectNonLinearityTask(pipeBase.PipelineTask):
    """Implementation of L2 Correct Non-Linearity.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-010

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "correct-non-linearity"
    ConfigClass = CorrectNonLinearityConfig
    full_name = "Correct Non-Linearity"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-010 v3.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            nonlin_pars: NonLinearityParameters,
            gain_factors: GainFactors,
            ) -> pipeBase.Struct:
        """Apply non-linearity correction to active area pixel flux.

        Parameters
        ----------
        spximg
            Input SPHEREx image.
        nonlin_pars
            Input non-linearity calibration parameters.
        gain_factors
            Per-detector gain factors.

        Returns
        -------
        output: `~lsst.pipe.base.Struct`
            An object that contains non-linearity corrected `~spx_data.SPHERExImage`.

        Raises
        ------
        KeyError
            Raised if keyword is not found in the image header.
        ValueError
            Raised if ``n_trans`` is not in valid range.

        Notes
        -----
        - Non-linearity correction only applied to active pixels.
        - The input parameters will be per detector ID.
          At the moment we expect the inputs to be FITS files.
        - Non-linearity and transient parameters are 3 per detector.
          e.g. there are 18 total non-linearity parameter image arrays.
        - Input calibration parameters in active area shape,
          while input image is in L1 shape.

        """
        spximg_before = spximg.copy()
        self.log.debug("Running createVarianceMap with input image ID: %s",
                       f"SPXIMG ID: {spximg.getArrayId()}")

        rp = ReadoutParameters.from_metadata(spximg.meta)
        det_id = spximg.getArrayId()
        # Get gain map from gain factor.
        read_gain = utils.gains_for_image(gain_factors.get_gains_for_detector(det_id), rp)
        q_nl = nonlin_pars.get_extension_data('Q_nl')
        nl_thresh = nonlin_pars.get_extension_data('Qmax')

        # all extensions must exist in practice
        # (KeyError would have been raised otherwise)
        trnsnt_1 = nonlin_pars.get_extension_data('b1')
        trnsnt_2 = nonlin_pars.get_extension_data('b2')
        trnsnt_3 = nonlin_pars.get_extension_data('b3')

        # Get SUR count at which transient detection begins.
        if "TRANDLY" in spximg.header:
            trandly = spximg.header["TRANDLY"]
        else:
            raise ValueError("Missing required header keyword: TRANDLY")

        if "SURLIM" in spximg.header:
            surlim = spximg.header["SURLIM"]
        else:
            raise ValueError("Missing required header keyword: SURLIM")

        if "SURDLY" in spximg.header:
            surdly = spximg.header["SURDLY"]
        else:
            raise ValueError("Missing required header keyword: SURDLY")

        # Get overflow thresholds in ADU for active pixels.
        ovrflw_thresh = get_overflow_threshold(spximg.header, "KOVR_A")

        tcut = spximg.header.get("TCUT", 0)
        n_trans = (surlim + tcut) / 2

        self.log.info(f"SURLIM = {surlim}")
        self.log.info(f"TCUT = {tcut}")
        self.log.info(f"NTRANS = {n_trans}")
        self.log.info(f"TRANDLY = {trandly}")
        self.log.info(f"OVRFLW THRESH = {ovrflw_thresh}")

        # SUR frame duration in seconds.
        t_frame = spx_param.get_sample_time().value
        self.log.info(f"t_frame = {t_frame} [s]")
        # Exposure duration in seconds.
        self.log.info(f"t_exposure = surlim x t_frame = {surlim*t_frame}")
        t = surlim * t_frame

        # Get active area and initialize
        actv_area = l1_utils.active_area_range(spximg)
        actv_pixls = spximg.data[actv_area]
        actv_flags = spximg.flags[actv_area]
        times = np.full_like(actv_pixls, t, dtype=np.float32)

        # Flag masks
        # Start with all pixels in active region
        all_pix = np.ones_like(actv_pixls, dtype=bool)

        # Pixels flagged with SUR_ERROR are excluded from correction
        mask_err = (actv_flags & Flag.SUR_ERROR) > 0
        eligible = all_pix & ~mask_err  # Only pixels eligible for correction

        # Build mutually exclusive masks
        mask_ovrflw = eligible & ((actv_flags & Flag.OVERFLOW) > 0)
        mask_trnsnt = eligible & ~mask_ovrflw & ((actv_flags & Flag.TRANSIENT) > 0)
        mask_clean = eligible & ~mask_ovrflw & ~mask_trnsnt

        n_total = np.sum(eligible)
        n_ovrflw = np.sum(mask_ovrflw)
        n_trnsnt = np.sum(mask_trnsnt)
        n_clean = np.sum(mask_clean)
        assert n_total == (n_ovrflw + n_trnsnt + n_clean), "Masks overlap or miss pixels!"

        # Compute overflow threshold per pixel
        charge_overflw = ovrflw_thresh * read_gain

        # Counters
        num_pix_corr = 0
        num_by_case = {"err": np.sum(mask_err),
                       "ovrflw": np.sum(mask_ovrflw),
                       "trnsnt": np.sum(mask_trnsnt),
                       "clean": np.sum(mask_clean)}

        for label, mask in [("ovrflw", mask_ovrflw),
                            ("trnsnt", mask_trnsnt),
                            ("clean", mask_clean)]:

            pix = actv_pixls[mask]
            flags = actv_flags[mask]
            q_nl_group = q_nl[mask]
            q_max_group = nl_thresh[mask]
            q_over_group = charge_overflw[mask]

            d_prime = pix.copy()
            nend = np.full_like(pix, surlim, dtype=np.float32)

            if label == "trnsnt":
                # For safe transient pixels, need to update d_prime
                safe_mask = (float(n_trans) * t_frame * pix) <= q_over_group
                unsafe_mask = ~safe_mask
                nend[safe_mask] = n_trans
                # Fix: Apply Equation 4 for safe transient pixels
                if np.any(safe_mask):
                    d_orig = pix[safe_mask]
                    qnl_safe = q_nl_group[safe_mask]
                    nend_safe = nend[safe_mask]
                    sqrt_term = np.sqrt(qnl_safe**2 - 4 * t_frame * (nend_safe + surdly) * d_orig * qnl_safe)
                    denom = 2 * t_frame * (nend_safe + surdly)
                    d_prime[safe_mask] = (qnl_safe - sqrt_term) / denom
                to_corr = unsafe_mask
            else:
                to_corr = np.ones(pix.shape, dtype=bool)

            for _ in range(3):
                qnl = q_nl_group[to_corr]
                qover = q_over_group[to_corr]
                d_corr = d_prime[to_corr]
                pix_corr = pix[to_corr]

                sqrt_term = np.sqrt(qnl**2 - 4 * qover * qnl)
                nend[to_corr] = np.minimum(
                    surlim,
                    np.floor((qnl - sqrt_term) / (2 * t_frame * d_corr))
                ).astype(np.float32)

                sqrt_term2 = np.sqrt(qnl**2 - 4 * t_frame * (nend[to_corr] + surdly) * pix_corr * qnl)
                denom = 2 * t_frame * (nend[to_corr] + surdly)
                d_prime[to_corr] = (qnl - sqrt_term2) / denom

            qaccum = nend * t_frame * pix
            over_qmax = qaccum > q_max_group
            flags[over_qmax] |= Flag.NONLINEAR
            linear_mask = ~over_qmax

            if np.any(linear_mask):
                if label == "trnsnt" and any(np.any(arr) for arr in [trnsnt_1, trnsnt_2, trnsnt_3]):
                    tr1 = trnsnt_1[mask]
                    tr2 = trnsnt_2[mask]
                    tr3 = trnsnt_3[mask]
                    n_corr = nend[linear_mask]
                    offset = (tr1[linear_mask] * n_corr
                              + tr2[linear_mask] * n_corr**2
                              + tr3[linear_mask] * n_corr**3)
                    d_prime[linear_mask] += offset

                pix[linear_mask] = d_prime[linear_mask]
                num_pix_corr += np.sum(linear_mask)

            # Update data and flags
            actv_pixls[mask] = pix
            actv_flags[mask] = flags
            times[mask] = t

        # Write back to image
        spximg.data[actv_area] = actv_pixls
        spximg.setFlagsFromMask(actv_flags & Flag.NONLINEAR > 0, Flag.NONLINEAR, area=actv_area)

        # DQA
        qa_result = dqa_check_nonlinear_flux_with_residuals(spximg_before, spximg,
                                                            threshold=self.config.dqa_nonlinear_threshold)

        spximg.meta["NL_DQA"] = (qa_result["result"], "Nonlinearity DQA status (PASS/FAIL)")
        spximg.meta["corlim"] = (qa_result["corlim"], "Max allowed fractional correction for DQA")
        spximg.meta["medres"] = (qa_result["medres"],
                                 "Median fractional NL correction residual")

        if qa_result["result"] == "PASS":
            spximg.meta["n_corr"] = (qa_result["n_corr"],
                                     "Number of nonlinear pixels corrected")
        else:
            spximg.meta["ovrcor"] = qa_result["ovrcor"]
            spximg.meta["maxcor"] = qa_result["maxcor"]
            spximg.meta["medcor"] = qa_result["medcor"]

        # Header stats
        npix_flagged_nonlinear = np.sum((spximg.flags & Flag.NONLINEAR) > 0)
        spximg.header["HIERARCH L2 N_NONLINEAR"] = (npix_flagged_nonlinear,
                                                    "number of pixels flagged nonlinear")

        # SUR error count
        num_surerror = np.sum(mask_err)

        # Logging
        self.log.info(f"Pixels per case: {num_by_case}")
        self.log.info(f"Corrected pixels: {num_pix_corr}")
        self.log.info(f"SUR_ERROR (skipped): {num_surerror}")
        self.log.info(f"NONLINEAR flagged: {npix_flagged_nonlinear}")

        # Sanity: all pixels accounted for
        num_actv = actv_pixls.size
        num_flagged = np.sum(actv_flags & Flag.NONLINEAR > 0)
        num_surerr = np.sum(mask_err)
        # Verify conservation of pixels, all should be either:
        # sur_error, flagged, or corrected.
        num_accounted = num_pix_corr + num_flagged + num_surerr
        assert num_accounted == num_actv, (
            f"{num_actv - num_accounted} active pixels were neither corrected, "
            "flagged as NONLINEAR, nor marked SUR_ERROR."
        )

        return pipeBase.Struct(spximg_out=spximg,
                               t_xy=times,
                               num_pix_corr=num_pix_corr,
                               num_surerror=num_surerror,
                               num_nonlinear=npix_flagged_nonlinear)
