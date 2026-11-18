__all__ = ["PerformForcedPhotometryConfig", "PerformForcedPhotometryTask"]

# Standard library imports
from itertools import repeat
import copy
import gc
import multiprocessing as mp
import time

# Related third party imports
from astropy.coordinates import SkyCoord
from astropy.io.fits import Header, HDUList
from astropy.table import Table, Column, vstack
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pex.config import ChoiceField, Field, ListField, Config
import numpy as np
import healpy as hp
import astropy.units as u
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import (
    Flag, EPSFModels, ObservationId, PSFZoneData, SPHERExImage, SPSFlag)
from spx_data.spx_param import get_npix_x_active, get_npix_y_active
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_tasks import CreateSourceMaskConfig
from spx_utils.photometry.window_utils import (
    slide_window, run_tractor_window, combine_measurements, combine_fits)
from spx_utils.photometry.tractor_utils import MJysr_to_uJy
from spx_utils import SourcePipelineClass, get_current_rss
from spx_utils.utils import get_refcat_provenance, get_data_release_version


# ==============================================================================
# TODO: Incorporate x/y pixel position info
def calculate_mjd(hdr: Header):
    mjd = np.average([hdr["MJD-BEG"], hdr["MJD-END"]])
    return mjd


def get_deep_flag(hdr: Header):
    if "SFLAG" in hdr:
        # one-year simulated images
        sps_flag_int = SPSFlag.from_string(hdr["SFLAG"])
    elif "SPS_FLAGS" in hdr:
        # SSDC level1 and level2 images
        sps_flag_int = SPSFlag.from_string(hdr["SPS_FLAGS"])
    else:
        sps_flag_int = 0

    return SPSFlag.is_deep_field(sps_flag_int)


def check_cut_side_length(value: int) -> bool:
    """Check that cutout side length (in pixels) is a factor of the number of
    active pixels along a side.
    """
    img_side_length = get_npix_x_active()
    if (img_side_length % value == 0):
        return True
    else:
        return False


def add_srccat_pass_thru(idx_map: np.array, input_cat: Table, col_name: str, output_tab: Table):
    """Add a pass-through column from the input catalog to the output
    photometry measurement table.

    Parameters
    ----------
    idx_map : `numpy.ndarray`
        Reorders the input catalog to match the order of the output table.
    input_cat : `astropy.table.Table`
        Filtered, image-level input source catalog.
    col_name : `str`
        Column name in input source catalog.
    output_tab : `astropy.table.Table`
        Output photometry measurement table.

    """
    # Input catalog columns that can be added to the output table
    pass_thru_lookup = {
        "name": {
            "description": "User-specified name of the source ('Target NNN' if no name provided)",
            "dtype": "U",
            "unit": None,
            "idx": 1
        },
        "spherex_class": {
            "description": "Source origin in reference catalog",
            "dtype": "int32",
            "unit": None,
            "idx": 1
        }
    }

    # NOTE: Whether adding or replacing columns, we avoid copying to save
    # memory. If future module changes require a column's input data to be
    # modified *after* the data has been added to the table, switch to use
    # `copy=True`.
    if col_name in pass_thru_lookup.keys():
        output_tab.add_column(
            col=Column(data=input_cat[col_name][idx_map],
                       name=col_name,
                       unit=pass_thru_lookup[col_name]["unit"],
                       description=pass_thru_lookup[col_name]["description"],
                       dtype=pass_thru_lookup[col_name]["dtype"]),
            index=pass_thru_lookup[col_name]["idx"], copy=False
        )

def modify_ssocat(ssocat: Table, config: Config, srccat: Table):
    """Modify the Solar System Object catalog format to include required
    photometry columns.

    Parameters
    ----------
    ssocat : `astropy.table.Table`
        Solar System Object catalog with sources found in SPHEREx FOV. Must
        include "desig", "RA", and "DEC" columns.
    config : `lsst.pex.config.Config`
        Perform Forced Photometry task configuration.
    srccat : `astropy.table.Table`
        Star/galaxy source catalog.
    """
    # Normalize column names and add units/shape parameters/photometry flag
    ssocat.rename_columns(names=["RA", "DEC"], new_names=["ra", "dec"])
    ssocat["ra"].unit = u.deg
    ssocat["dec"].unit = u.deg
    shape_r = Column(name="shape_r", data=np.full(len(ssocat), fill_value=0.0), dtype="float32")
    phi = Column(name="phi", data=np.full(len(ssocat), fill_value=0.0), dtype="float32")
    ab_ratio = Column(name="ab_ratio", data=np.full(len(ssocat), fill_value=0.0), dtype="float32")
    sersic = Column(name="sersic", data=np.full(len(ssocat), fill_value=0.0), dtype="float32")
    ssocat.add_columns([shape_r, phi, ab_ratio, sersic])
    gp_no_phot = list(CreateSourceMaskConfig().planet_name_keys)
    ssocat["do_phot"] = (~np.isin(ssocat["desig"].data, gp_no_phot)).astype(np.uint8)

    # If using reference position columns in star/galaxy source catalog,
    # populate them with current positions for SSOs
    if "ref_ra" in config.cat_phot_cols and "ref_ra" in srccat.colnames:
        ssocat["ref_ra"] = ssocat["ra"]
    if "ref_dec" in config.cat_phot_cols and "ref_dec" in srccat.colnames:
        ssocat["ref_dec"] = ssocat["dec"]

    # NOTE: To distinguish from stars and galaxies in input source catalog,
    # use negative sentinel values for the "SPHERExRefID" of SSOs. Original
    # SSO catalog indices map to IDs like so: idx 0, 1...N-1 -> ID -1, -2...-N
    ssocat["SPHERExRefID"] = np.arange(-1, -len(ssocat) - 1, -1, dtype=np.int64)

    return

class PerformForcedPhotometryConnections(pipeBase.PipelineTaskConnections,
                                         dimensions=("instrument", "exposure"),
                                         defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    global_bkgd = cT.Input(
        name="global_bkgd",  # input is output from previous module
        doc="Estimated global background.",
        storageClass="NumpyArray",
        dimensions=("instrument", "exposure"),
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "dichroicpair", "spsplan", "slewgroup"),
        deferLoad=True,
    )
    solid_angle_pixel_map = cT.PrerequisiteInput(
        name="solid_angle_pixel_map",
        doc="Input solid angle pixel map.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral WCS with lookup table.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    special_ids = cT.PrerequisiteInput(
        name="special_ids",
        doc="List of filtered source IDs to photometer at the window level.",
        storageClass="NumpyArray",
        dimensions=["instrument"],
        isCalibration=False
    )
    epsf = cT.PrerequisiteInput(
        name="epsf",
        doc="Input average EPSF product used as a template for EPSF binary-table output.",
        storageClass="EPSF",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    ssocat = cT.Input(
        name="ssocat",
        doc="Input Solar System Object catalog with sources found in SPHEREx FOV.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "observation"),
        deferLoad=True
    )
    phot_meas = cT.Output(
        name="phot_meas",
        doc="Output photometry measurement table (per HEALPix).",
        storageClass="AstropyParquet",
        # dimensions should include a table with spatial dimension (exposure)
        # skypix dimension should match commonSkyPix in dimensions.yaml
        dimensions=("instrument", "exposure", "hips6"),
        multiple=True
    )
    phot_exceptions = cT.Output(
        name="phot_exceptions",
        doc="Output photometry measurement table for sources with HEALPix-"
            "related exceptions.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
    )
    phot_meas_combined = cT.Output(
        name="phot_meas_combined",
        doc="Output photometry measurement table (full image).",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
        multiple=False
    )
    tractor_fits = cT.Output(
        name="tractor_fits",
        doc="Tractor results",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "exposure"),
    )
    phot_meas_sso = cT.Output(
        name="phot_meas_sso",
        doc="Output photometry measurement table for Solar System Objects.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
        multiple=False
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.combinedOutput:
            del self.phot_meas
            del self.phot_exceptions
        else:
            del self.phot_meas_combined

        if not config.return_fit:
            del self.tractor_fits

        if not config.filter_by_ids:
            del self.special_ids

        if not config.override_epsf:
            del self.epsf

        if not config.fit_sso:
            del self.ssocat
            del self.phot_meas_sso


class PerformForcedPhotometryConfig(pipeBase.PipelineTaskConfig,
                                    pipelineConnections=PerformForcedPhotometryConnections):
    """Configuration for PerformForcedPhotometry task."""

    hips_order = Field(
        doc="HEALPix order for grouping the measurements",
        dtype=int,
        default=6)

    combinedOutput = Field(
        doc="If True, output the photometry measurements for all sources in an image in one table."
            " Default is False.",
        dtype=bool,
        default=False)

    pass_thru_cols = ListField(
        doc="List of columns to pass through to the output catalog, in order of relative placement in the"
            "output catalog.",
        dtype=str,
        default=[])

    gc_interval = Field(
        doc="Triggers manual garbage collection very N iterations during window loop. Default is 0 (no"
            " garbage collection).",
        dtype=int,
        default=0)

    cat_phot_cols = ListField(
        doc="Minimal list of input source catalog columns required to perform photometry.",
        dtype=str,
        default=["SPHERExRefID", "ra", "dec", "ra_error", "dec_error", "ref_ra", "ref_dec", "sersic",
                 "shape_r", "ab_ratio", "phi", "do_phot"])

    merge_srccat = Field(
        doc="If True, adds `pass_thru_cols` from the input source catalog to the output photometry table."
            " Default is False.",
        dtype=bool,
        default=False)

    filter_by_ids = Field(
        doc="If True, only photometer windows with 1+ targets in the `special_ids` input list. If False,"
            " photometer all windows with 1+ sources flagged for photometry. Default is False.",
        dtype=bool,
        default=False)

    psf_extension = ChoiceField(
        doc="PSF extension name in input SPHERExImage to use for fitting."
            "If 'PSF', use the legacy PSF. If 'EPSF', use the effectivePSF."
            "Default is 'EPSF'.",
        dtype=str,
        allowed={"PSF": "Use the legacy PSF", "EPSF": "Use the effective PSF"},
        default="EPSF")

    override_epsf = Field(
        doc="If True, use input EPSF connection instead of extension from the input SPHERExImage."
            "Default is False.",
        dtype=bool,
        default=False)

    phot_tool_mode = Field(
        doc="If True, this pipeline is being run as part of the photometry tool. Default is False.",
        dtype=bool,
        default=False)

    fit_sso = Field(
        doc="If True, perform forced photometry for Solar System Objects. Default is True.",
        dtype=bool,
        default=True)

    # Sliding window parameters
    cut_side_length = Field(
        doc="Side length (in pixels) of the NxN square cutouts used to step across the detector. Sources "
            "inside the cutout region are included in the Tractor model, and their photometry results are "
            "stored in the output of this module.",
        dtype=int,
        check=check_cut_side_length,
        default=5)

    frame_width = Field(
        doc="Number of extra pixels around a cutout to include in Tractor fitting. Sources in the frame "
            "region are included in the Tractor model, but their photometry results are not recorded by "
            "default. The frame is added along 2 (corner), 3 (edge), or 4 (inner) sides. For example, the "
            "window size for a corner with `cut_side_length`=5 and `frame_width`=3 would be 8x8 pixels.",
        dtype=int,
        default=3)

    parallel_fitting = Field(
        doc="Whether to use parallel window fitting. Only suggested for local testing with single images "
            "using Python scripts/notebooks. Default is False.",
        dtype=bool,
        default=False)

    class_filter = Field(
        doc="Filters target windows to photometer based on `spherex_class` from SPHEREx Reference Catalog. "
            "Default is `SourcePipelineClass.ALL`.",
        dtype=int,
        default=SourcePipelineClass.ALL)

    # Fitting settings
    fit_shapes = Field(
        doc="If True, treat galaxies as shapes during modeling. If False, "
            "treat galaxies as point sources. Default is False.",
        dtype=bool,
        default=False)

    return_fit = Field(
        doc="If True, fit model and residual image are returned. Default is"
            " False.",
        dtype=bool,
        default=False)

    def update(self, **kw):
        """Override default update method to validate upon field changes.
        """
        super().update(**kw)
        self.validate()

    def validate(self):
        """Override default validate method to include inter-field checks.
        """
        super().validate()

        # Frame width is smaller than cutout
        frame_size_check = (self.frame_width < self.cut_side_length)
        if not frame_size_check:
            raise ValueError("frame_width must be less than cut_side_length.")

        # Non-default filter class requires spherex_class column
        class_missing = "spherex_class" not in self.pass_thru_cols
        filtering_enabled = (self.class_filter != SourcePipelineClass.ALL.value)
        if (filtering_enabled and class_missing):
            raise ValueError("Pass-through column 'spherex_class' must be included to perform class "
                             "filtering.")

        # If merging srccat data, 1+ pass-through columns must be specified
        if self.merge_srccat and (len(self.pass_thru_cols) == 0):
            raise ValueError("To merge data from the srccat, 1+ pass-through columns must be specified.")

        # Minimal required catalog columns for photometry
        required_cols = set(["SPHERExRefID", "ra", "dec", "shape_r", "ab_ratio", "phi", "sersic"])
        if not required_cols.issubset(set(self.cat_phot_cols)):
            missing_cols = required_cols - set(self.cat_phot_cols)
            raise ValueError(f"cat_phot_cols is missing required columns: {missing_cols}")

class PerformForcedPhotometryTask(pipeBase.PipelineTask):
    """Implementation of L3 Perform Forced Photometry.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-033

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "perform-forced-photometry"
    ConfigClass = PerformForcedPhotometryConfig
    full_name = "Perform Forced Photometry"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "SSDC-MS-033 3.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        if self.config.combinedOutput:
            super().runQuantum(butlerQC, inputRefs, outputRefs)
        else:
            inputs = butlerQC.get(inputRefs)
            output_ref_map = {name: getattr(outputRefs, name) for name in vars(outputRefs)
                              if name in ["phot_meas", "phot_exceptions", "phot_meas_sso", "tractor_fits"]}
            result = self.run(**inputs)

            if result.phot_meas is not None:
                ref_or_lst = output_ref_map["phot_meas"]
                out_healpix_ids = result.phot_meas.keys()
                expected_healpix_ids = {ref.dataId.get("hips6") for ref in ref_or_lst}
                unexpected_ids = set(out_healpix_ids).difference(expected_healpix_ids)

                # Per-HEALPixel measurement tables predicted by level 2 WCS/
                # exposure region
                for ref in ref_or_lst:
                    healpix_id = ref.dataId.get("hips6")
                    if healpix_id in out_healpix_ids:
                        butlerQC.put(result.phot_meas[healpix_id], ref)
                    else:
                        self.log.warning(f"No sources found for expected healpix ID {healpix_id}.")

                # Per-exposure measurement table for high proper motion sources
                # whose corrected positions shifted the output HEALPix ID
                if len(unexpected_ids) > 0:
                    unexpected_tabs = []
                    for uid in unexpected_ids:
                        self.log.warning(f"Adding sources from unexpected healpix ID {uid} to"
                                         " exceptions table.")
                        unexpected_tabs.append(result.phot_meas[uid])
                    exceptions = vstack(unexpected_tabs)
                    if len(exceptions) > 0:
                        butlerQC.put(exceptions, output_ref_map["phot_exceptions"])

            if self.config.fit_sso and result.phot_meas_sso is not None \
                and len(result.phot_meas_sso) > 0:
                butlerQC.put(result.phot_meas_sso, output_ref_map["phot_meas_sso"])

            supported = {"phot_meas", "phot_exceptions", "phot_meas_sso"}
            for name in output_ref_map:
                if name not in supported:
                    # TODO handle other types of output
                    self.log.warning(f"ERROR: output for connection {name} is not saved")


    def run(self, spximg: SPHERExImage, srccat: DeferredDatasetHandle | Table, global_bkgd: np.ndarray,
            solid_angle_pixel_map: SPHERExImage, spectral_wcs: HDUList, special_ids: np.ndarray = None,
            epsf: SPHERExImage = None, ssocat: DeferredDatasetHandle = None) -> pipeBase.Struct:
        """Perform Forced Photometry.

        Parameters
        ----------
        spximg : `spx_data.SPHERExImage`
            LVF image with catalog of sources to be measured
        srccat : `astropy.table.Table`|`lsst.daf.butler.DeferredDatasetHandle`
            Catalog of sources within or close to the image.
        global_bkgd : `numpy.ndarray`
            Global background estimate.
        solid_angle_pixel_map : `spx_data.SPHERExImage`
            Solid angle pixel map derived by Update Field Distortion module.
        spectral_wcs : `~astropy.io.fits.HDUList`
            Spectral WCS calibration with central wavelength and bandwidth.
        special_ids : `numpy.ndarray`, optional
            Array of filtered source IDs to photometer at the window level. If
            not specified, all sources flagged for photometry in the input
            catalog are photometered at the window level.
        epsf : `spx_data.SPHERExImage`, optional
            EPSF calibration to override the extension stored in the image.
        ssocat : `lsst.daf.butler.DeferredDatasetHandle`, optional
            Deferred `astropy.table.Table` of Solar System Objects to
            photometer.

        Returns
        -------
        return_obj : `pipeBase.Struct`
            Container for output connection fields. By default, only includes
            `phot_meas` (`dict` of `astropy.table.Table` grouped by HEALPix). If
            `self.config.return_fit` is True, includes `tractor_fits` (`dict`
            of `astropy.fits.io.HDUList`). If `self.config.combinedOutput` is
            True, includes `phot_meas_combined` (`astropy.table.Table`). If
            `self.config.combinedOutput` is True, includes `phot_meas_sso`
            (`astropy.table.Table`).

        Raises
        ------
        ValueError
            If configuration settings conflict with input arguments.
        ValueError
            If input source catalog is missing requested column or has negative
            SPHERExRefIDs.
        """
        # =====================================================================
        # Prepare inputs for photometry fitting
        task_t0 = time.time()
        self.log.info("Starting forced photometry...")

        # Prevent silent failure due to mismatch between config and run args
        if self.config.filter_by_ids and special_ids is None:
            raise ValueError("`filter_by_ids` is True but `special_ids` input was not provided.")
        if self.config.override_epsf and epsf is None:
            raise ValueError("`override_epsf` is True but `epsf` input was not provided.")
        if self.config.fit_sso and ssocat is None:
            raise ValueError("`fit_sso` is True but `ssocat` input was not provided.")

        # Only read necessary columns from input source catalog
        if isinstance(srccat, DeferredDatasetHandle):
            read_cols = list(self.config.cat_phot_cols) + list(self.config.pass_thru_cols) if \
                self.config.merge_srccat else list(self.config.cat_phot_cols)
            try:
                srccat = srccat.get(
                    parameters={
                        "columns": read_cols
                    }
                )
            except KeyError as e:
                raise ValueError(f"Requested column {e} is not in the input catalog.")
            else:
                if not np.all(srccat["SPHERExRefID"] >= 0):
                    raise ValueError("Negative SPHERExRefIDs found in srccat.")

        # If fitting SSOs, read catalog
        if self.config.fit_sso and ssocat is not None:
            if isinstance(ssocat, DeferredDatasetHandle):
                try:
                    ssocat = ssocat.get(
                        parameters={
                            "columns": ["desig", "RA", "DEC"]  # only need identifier and positions
                        }
                    )
                except KeyError as e:
                    raise ValueError(f"Requested column {e} is not in the solar system object catalog.")
            modify_ssocat(ssocat, self.config, srccat)  # standardize basic column names and format
            srccat = vstack([srccat, ssocat], join_type="outer",
                            metadata_conflicts="silent")  # temporarily merge all sources

        # Extract refcat provenance (version) from input srccat once
        ref_ver = None
        try:
            _, ref_ver = get_refcat_provenance(srccat)
        except Exception:
            # Optional: try the meta dict if your pipeline stores it there
            try:
                _, ref_ver = get_refcat_provenance(getattr(srccat, "meta", {}))
            except Exception:
                ref_ver = None

        if ref_ver:
            # Record in task metadata (shows up in Butler task metadata)
            self.metadata.add("refcat_version", str(ref_ver))

        # Convert positions/their errors to X, Y pixel coordinates on detector
        srctab = copy.copy(srccat)  # avoid directly modifying input cat
        cat_coords = SkyCoord(ra=srctab["ra"], dec=srctab["dec"], unit=u.deg, frame="icrs")
        src_xdet, src_ydet = (spximg.wcs).world_to_pixel(cat_coords)

        # Select sources whose positions fall within image boundaries
        # NOTE: Subtract 0.5 from edges since WCS.world_to_pixel returns
        # positions assuming zero-based indices.
        img_mask = ((src_xdet > 0 - 0.5)
                    & (src_xdet < get_npix_x_active() - 0.5)
                    & (src_ydet > 0 - 0.5)
                    & (src_ydet < get_npix_y_active() - 0.5))
        src_ximg = src_xdet[img_mask]
        src_yimg = src_ydet[img_mask]
        cat_coords = cat_coords[img_mask]
        srctab = srctab[img_mask]

        if self.config.filter_by_ids and special_ids is not None:
            # Filter special IDs down to those in image
            id_mask = np.isin(srctab["SPHERExRefID"], special_ids)
            if np.any(id_mask):
                special_ids = set(srctab["SPHERExRefID"][id_mask])  # use set (faster lookup in window loop)
            else:
                self.log.warning("No requested `special_ids` are in the image. Skipping photometry.")
                # Set up empty output with correct fields and return early
                kwargs = {}
                if self.config.return_fit:
                    kwargs["tractor_fits"] = None
                if self.config.fit_sso:
                    kwargs["phot_meas_sso"] = None
                if self.config.combinedOutput:
                    kwargs["phot_meas_combined"] = None
                else:
                    kwargs["phot_meas"] = None
                return pipeBase.Struct(**kwargs)

        # If "do_phot" column exists, filter out sources not flagged for L3
        if "do_phot" in srctab.colnames:
            l3phot_mask = (srctab["do_phot"] == 1)
            if self.config.filter_by_ids and special_ids is not None:
                l3phot_mask |= id_mask  # keep sources flagged by selection criteria or requested by ID
            src_ximg = src_ximg[l3phot_mask]
            src_yimg = src_yimg[l3phot_mask]
            cat_coords = cat_coords[l3phot_mask]
            srctab = srctab[l3phot_mask]
        self.log.debug(get_current_rss("Image-filtered catalog created"))

        # Add HEALPix index for each source, using refcat RA/DEC if available
        # to ensure consistency over time
        hips_order = self.config.hips_order
        hips_col_name = f"hips{hips_order}"
        hips_ra = "ref_ra" if (
            "ref_ra" in self.config.cat_phot_cols and "ref_ra" in srctab.colnames) else "ra"
        hips_dec = "ref_dec" if (
            "ref_dec" in self.config.cat_phot_cols and "ref_dec" in srctab.colnames) else "dec"
        col = Column(data=hp.ang2pix(2 ** hips_order, srctab[hips_ra], srctab[hips_dec], nest=True,
                                     lonlat=True),
                     name=hips_col_name, unit="")  # leaving unit None, makes "deg" unit
        # NOTE: We avoid copying columns to save memory. If future module
        # changes require a column's input data to be modified after the data
        # has been added to the table, switch to use `copy=True`.
        srctab.add_column(col, copy=False)

        # Modify pixels that should be ignored by Tractor
        # NOTE: Tractor doesn't support masking, so resetting these values
        #       effectively forces it to discount pixels in the fit.
        var_nanmax_init = np.nanmax(spximg.uncertainty.array)
        var_nanmean_init = np.nanmean(spximg.uncertainty.array)

        # Flagged pixels (in image and variance)
        ignore_pixel_flags = (Flag.SUR_ERROR | Flag.NONFUNC | Flag.MISSING_DATA | Flag.HOT | Flag.COLD
                              | Flag.NONLINEAR | Flag.PERSIST | Flag.BLOOM | Flag.SNOWBALL | Flag.HALO
                              | Flag.SATELLITE_HALO)
        ignore_mask = spximg.getFlagsAsMask(ignore_pixel_flags)
        spximg.data[ignore_mask] = 0
        spximg.uncertainty.array[ignore_mask] = 1e3 * var_nanmax_init

        # Remaining non-finite pixels (in image and variance)
        n_nonfin_img = np.count_nonzero(~np.isfinite(spximg.data))
        if n_nonfin_img > 0:
            spximg.data[~np.isfinite(spximg.data)] = 0
            self.log.info(f"Assigned {n_nonfin_img} non-finite-valued image pixels to 0.")
        n_nonfin_var = np.count_nonzero(~np.isfinite(spximg.uncertainty.array))
        if n_nonfin_var > 0:
            spximg.uncertainty.array[~np.isfinite(spximg.uncertainty.array)] = 1e3 * var_nanmax_init
            self.log.info(f"Assigned {n_nonfin_var} non-finite-valued variance pixels to initial max "
                          "variance value.")

        # Zeros (in variance)
        n_zeros = np.count_nonzero(spximg.uncertainty.array == 0)
        if n_zeros > 0:
            spximg.uncertainty.array[spximg.uncertainty.array == 0] = var_nanmean_init
            self.log.info(f"Assigned {n_zeros} zero-valued variance pixels to initial mean variance value.")

        # Subtract global spectral background if suitable
        no_glob_bkg = np.all(np.isnan(global_bkgd))
        if no_glob_bkg:
            self.log.info("No good spectral background estimate available...")
        else:
            spximg.data -= global_bkgd
            self.log.info("Subtracting global spectral background estimate...")

        # Save arguments for data, unit, variance, etc. in memory
        data = spximg.data
        units = spximg.unit
        variance = spximg.uncertainty
        flags = spximg.flags
        wcs = spximg.wcs
        detector = spximg.getArrayId()
        hdr = spximg.header
        # NOTE: We use exposure ID as a component of the deterministic
        # per-source flux-initialization seed. Re-runs reproduce, while each
        # (source, image) measurement still seeds independently.
        exp_id_int = ObservationId(hdr["OBSID"]).pointing_id + hdr["DETECTOR"]

        # Set up desired PSF model
        if self.config.override_epsf or self.config.psf_extension == "EPSF":
            # Use effective PSF (EPSF) format; set legacy PSF to None
            epsf_hdu = spximg.layers["EPSF"] if not self.config.override_epsf else epsf.layers["EPSF"]
            psf = EPSFModels.from_hdu(epsf_hdu)
            psf_msg = "EPSF from level2 image extension" if not self.config.override_epsf else \
                "EPSF override from input calibration"
        elif self.config.psf_extension == "PSF":
            # Use legacy PSF format; set EPSF to None
            psf = PSFZoneData.from_hdu(spximg.layers["PSF"])
            psf_msg = "legacy PSF"
        self.log.info(f"Using {psf_msg} for photometry fitting.")

        # Only pass the columns relevant for photometry fitting
        relevant_cols = ["SPHERExRefID", "ra", "dec", "shape_r", "ab_ratio", "phi", "sersic",
                         hips_col_name]
        if (
            (self.config.class_filter != SourcePipelineClass.ALL.value)
            and ("spherex_class" in self.config.pass_thru_cols)
        ):
            relevant_cols += ["spherex_class"]
        fit_tab = srctab[relevant_cols]

        # =====================================================================
        # Perform fitting
        # Step across the detector, storing bounds for "windows" or subregions
        img_windows = slide_window(spximg, self.config)
        self.log.debug(get_current_rss("Array of window objects created"))

        # Set up photometry measurement array/fitting results dictionary
        combined_arr = np.empty(len(srctab), dtype=[("SPHERExRefID", "int64"),
                                                    ("ra", "float64"),
                                                    ("dec", "float64"),
                                                    ("x_image", "float32"),
                                                    ("y_image", "float32"),
                                                    ("mjd", "float64"),
                                                    ("lambda", "float32"),
                                                    ("lambda_width", "float32"),
                                                    ("flux", "float32"),
                                                    ("flux_err", "float32"),
                                                    ("flux_bkg", "float32"),
                                                    ("ext_flg", "bool"),
                                                    ("flags", "int64"),
                                                    ("fit_ql", "float32"),
                                                    ("deep_flg", "bool"),
                                                    ("lvf_id", "int64"),
                                                    ("det_id", "uint8"),
                                                    (hips_col_name, "int32")])
        tractor_fits = {} if self.config.return_fit else None
        start = 0
        end = 0

        self.log.info("Starting window fitting with Tractor...")
        fitting_t0 = time.time()
        fit_tab_np = fit_tab.as_array()
        self.log.debug(get_current_rss("Empty fitting array created"))
        if not self.config.parallel_fitting:
            # Fit windows sequentially
            for i, window in enumerate(img_windows):
                result = run_tractor_window(
                    img_data=data, img_units=units, img_wcs=wcs, img_var=variance, img_flags=flags,
                    window_params=window, src_ximg=src_ximg, src_yimg=src_yimg, src_cat=fit_tab_np,
                    cat_coords=cat_coords, config=self.config, psf=psf, detector=detector,
                    keep_frame_sources=False, special_ids=special_ids, exp_id=exp_id_int)
                start, end = combine_measurements(combined_arr, result, start, end)
                if self.config.return_fit:
                    tractor_fits = combine_fits(tractor_fits, result)
                if (self.config.gc_interval > 0) and (i % self.config.gc_interval == 0):
                    n_objs = gc.collect()
                    self.log.info(f"{n_objs} objects garbage collected")
        else:
            # Fit windows in parallel (for small datasets/quick local tests)
            iterator = zip(repeat(data), repeat(units), repeat(wcs), repeat(variance), repeat(flags),
                           img_windows, repeat(src_ximg), repeat(src_yimg), repeat(fit_tab_np),
                           repeat(cat_coords), repeat(self.config), repeat(psf), repeat(detector),
                           repeat(False), repeat(special_ids), repeat(exp_id_int))
            mp.freeze_support()
            with mp.pool.Pool() as pool:
                for result in pool.starmap(run_tractor_window, iterator):
                    start, end = combine_measurements(combined_arr, result, start, end)
                    if self.config.return_fit:
                        tractor_fits = combine_fits(tractor_fits, result)
        fitting_t1 = time.time()
        # Truncate unfilled elements
        combined_arr = combined_arr[:end]

        # Convert specific intensity to spectral flux density
        pix_map = solid_angle_pixel_map.data << solid_angle_pixel_map.unit  # use << to avoid copying
        ypos_idx, xpos_idx = (spximg.wcs).world_to_array_index(
            SkyCoord(combined_arr["ra"], combined_arr["dec"], unit=u.deg, frame="icrs"))  # keep array order
        # NOTE: We pass an oversample factor of 1 because the Tractor
        # result flux was concentrated in a single 6.2" x 6.2" pixel.
        combined_arr["flux"] = MJysr_to_uJy(combined_arr["flux"]*(u.MJy/u.sr), 1, pix_map, xpos_idx, ypos_idx)
        combined_arr["flux_err"] = MJysr_to_uJy(combined_arr["flux_err"]*(u.MJy/u.sr), 1, pix_map,
                                                xpos_idx, ypos_idx)

        # To save memory and avoid creating more cutouts, populate the flux_bkg
        # column in the global background case all at once
        if not no_glob_bkg:
            combined_arr["flux_bkg"] += global_bkgd[ypos_idx, xpos_idx]  # sum spectral with local background

        # =====================================================================
        # Add back in values for non-fitting parameters
        # Get central wavelengths and bandwidths
        # Use zero-based position indices initialized in order of combined_array
        central_wls = spectral_wcs["CWAVE"].data[ypos_idx, xpos_idx]  # row-major: x is columns, y is rows
        central_bws = spectral_wcs["CBAND"].data[ypos_idx, xpos_idx]

        # Quantities based on image/detector data
        combined_arr["mjd"] = calculate_mjd(hdr)
        combined_arr["lambda"] = central_wls
        combined_arr["lambda_width"] = central_bws
        combined_arr["deep_flg"] = get_deep_flag(hdr)
        combined_arr["lvf_id"] = exp_id_int
        combined_arr["det_id"] = detector

        # Construct output table
        out_fmt_arr = np.array([
            ("source_id", "Unique SPHEREx ID", "", "int64"),
            ("ra", "Right ascension in input source catalog", u.deg, "float64"),
            ("dec", "Declination in input source catalog", u.deg, "float64"),
            ("x_image", "x pixel position on LVF image used for fitting", u.pixel, "float32"),
            ("y_image", "y pixel position on LVF image used for fitting", u.pixel, "float32"),
            ("mjd", "Modified Julian Date", u.day, "float64"),
            ("lambda", "SPHEREx wavelengths", u.um, "float32"),
            ("lambda_width", "SPHEREx wavelength widths", u.um, "float32"),
            ("flux", "SPHEREx observed flux", u.uJy, "float32"),
            ("flux_err", "SPHEREx observed flux error", u.uJy, "float32"),
            ("flux_bkg", "SPHEREx background surface brightness at source position", u.MJy/u.sr, "float32"),
            ("ext_flg", "True if treated as extended source", "", "bool"),
            ("flags", "Photometry flags", "", "int64"),
            ("fit_ql", "Photometry metric", "", "float32"),
            ("deep_flg", "True if in Deep Field", "", "bool"),
            ("lvf_id", "ID of LVF exposure used for photometry", "", "int64"),
            ("det_id", "Detector ID", "", "uint8"),
            (hips_col_name, f"HEALPix tile (order {self.config.hips_order})", "", "int32")],
            dtype=[("name", "U20"), ("parameter", "U60"), ("unit", "U10"), ("format", "U10")]
        )
        combined_tab = Table(data=combined_arr,
                             names=out_fmt_arr["name"].tolist(),
                             descriptions=out_fmt_arr["parameter"].tolist(),
                             units=out_fmt_arr["unit"].tolist(),
                             copy=False)

        # Unmerge SSOs if they were photometered together with stars/galaxies
        if self.config.fit_sso and ssocat is not None:
            sso_mask = combined_tab["source_id"] < 0  # SSO IDs were assigned negative values
            sso_results = combined_tab[sso_mask]
            combined_tab = combined_tab[~sso_mask]  # drop SSO rows from main table

            # Add SSO designations back in and clean up unnecessary columns
            og_sso_idx = (-sso_results["source_id"]) - 1  # SSO IDs were assigned as -(idx+1)
            sso_results.add_column(col=Column(data=ssocat["desig"][og_sso_idx], name="desig",
                                              dtype=ssocat["desig"].dtype,
                                              description="JPL Horizons name of object", unit=None))
            cols_to_remove = ["source_id", hips_col_name]
            cols_to_remove += ["ref_ra"] if "ref_ra" in sso_results.colnames else []
            cols_to_remove += ["ref_dec"] if "ref_dec" in sso_results.colnames else []
            sso_results.remove_columns(cols_to_remove)
            sso_results.sort("desig")  # for easier gather operation later

            # Add version metadata
            sso_results.meta["version_ssocat"] = ssocat.meta.get("user_metadata", {}).get("version", "")
            sso_results.meta["version_level2"] = spximg.header["VERSION"]
            sso_results.meta["version_phot_meas_sso"] = SSDC_PIPELINES_VERSION

        # Add requested pass-through columns from srccat
        if self.config.merge_srccat and len(self.config.pass_thru_cols) > 0:
            srctab.sort(keys="SPHERExRefID")
            combined_tab.sort(keys="source_id")
            if (
                (self.config.class_filter == SourcePipelineClass.ALL.value)
                and (len(srctab) == len(combined_tab))
            ):
                # Use contiguous indexing to avoid copying
                index_map = slice(None)
            else:
                # Manually find corresponding indices (table lengths unequal)
                index_map = [np.nonzero(srctab["SPHERExRefID"] == combined_tab["source_id"][idx])[0][0] for
                             idx in range(len(combined_tab))]

            for col_name in self.config.pass_thru_cols:
                add_srccat_pass_thru(index_map, srctab, col_name, combined_tab)

        # =====================================================================
        # Add metadata and create return object based on config parameters
        combined_tab.meta["version_phot_meas"] = SSDC_PIPELINES_VERSION
        combined_tab.meta["version_level2"] = spximg.header["VERSION"]
        if self.config.phot_tool_mode:
            spx_header_version = get_data_release_version(spximg)
            combined_tab.add_column(
                Column(
                    data=np.full(len(combined_tab), spx_header_version),
                    name="data_collection",
                    description="Level 2 pipeline version for this measurement",
                    unit=None,
                )
            )
        if ref_ver:
            combined_tab.meta["refcat_version"] = str(ref_ver)

        # Star/galaxy photometry measurement table keyword args
        if self.config.combinedOutput:
            # Group photometry measurements by image
            kwargs = {"phot_meas_combined": combined_tab}
        else:
            # Group photometry measurements by HEALPix
            cat_dict = {}
            grouped_table = combined_tab.group_by(hips_col_name)
            for key, group in zip(grouped_table.groups.keys[hips_col_name], grouped_table.groups):
                group.meta = dict(combined_tab.meta)
                cat_dict[key] = group
            kwargs = {"phot_meas": cat_dict}

        # Fitting results and SSO photometry measurement table keyword args
        if self.config.return_fit:
            kwargs["tractor_fits"] = tractor_fits
        if self.config.fit_sso:
            kwargs["phot_meas_sso"] = sso_results

        # Create return object
        return_obj = pipeBase.Struct(**kwargs)
        task_t1 = time.time()
        self.log.info("Fitting time in seconds (minutes): %s (%s)", f"{(fitting_t1-fitting_t0):.2f}",
                      f"{((fitting_t1-fitting_t0)/60.):.2f}")
        self.log.info("Task run time in seconds (minutes): %s (%s)", f"{(task_t1-task_t0):.2f}",
                      f"{((task_t1-task_t0)/60.):.2f}")
        return return_obj
