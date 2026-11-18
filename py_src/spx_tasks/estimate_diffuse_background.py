__all__ = ["EstimateDiffuseBackgroundConfig", "EstimateDiffuseBackgroundTask"]

# Standard library imports
import logging

# Related third party imports
from astropy.io.fits import HDUList
from scipy.interpolate import make_smoothing_spline
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_tasks.estimate_abs_gain_matrix import (
    compute_avg_flux_and_wavelength, define_spectral_bins, EstimateAbsGainMatrixConfig)
from spx_utils import (
    get_current_rss, read_airglow_fit_from_header, reconstruct_p_fit_from_results, subtract_airglow_he,
    fit_airglow_he)
from spx_utils.template_utils import HE_TEMP_PAR_KWS, HE_FIT_BKG_KWS


# ============================================================================
def add_he_airglow(global_bkgd: np.ndarray, logger: logging.Logger, compute_fn: callable) -> bool:
    """Add He I airglow component to the global background estimate. Create
    the He I airglow array outside the main ``run`` method to make it eligible
    sooner for garbage collection and reduce temporary memory usage.

    Parameters
    ----------
    global_bkgd : `numpy.ndarray`
        Global background estimate.
    logger : `logging.logger`
        Logger instance.
    compute_fn : `callable`
        Function to compute the He I airglow component.

    Returns
    -------
    `bool`
        True if the component was added successfully, False otherwise.

    """
    he_component = compute_fn()
    if np.all(np.isnan(he_component)):
        logger.info("He I airglow fit failed.")
        return False
    else:
        global_bkgd[:] = 0.0
        global_bkgd += he_component
        logger.info("He I airglow component added to global background estimate.")
        return True


# =============================================================================
# Pipeline subclasses
class EstimateDiffuseBackgroundConnections(pipeBase.PipelineTaskConnections,
                                           dimensions=("instrument", "exposure"),
                                           defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is from previous module
        doc="Input image from Correct Dark Current.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral WCS with lookup table.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    ),
    he_template = cT.PrerequisiteInput(
        name="he_template",
        doc="He I Airglow Template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0
    )
    global_bkgd = cT.Output(
        name="global_bkgd",
        doc="Estimated global background.",
        storageClass="NumpyArray",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.est_he_airglow:
            del self.he_template


class EstimateDiffuseBackgroundConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=EstimateDiffuseBackgroundConnections):
    """Configuration for EstimateDiffuseBackground task."""

    est_he_airglow = Field(
        doc="If True, estimates the He I airglow component. Default is True.",
        dtype=bool,
        default=True)


class EstimateDiffuseBackgroundTask(pipeBase.PipelineTask):
    """Implementation of L3 Estimate Diffuse Background.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-032

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "estimate-diffuse-background"
    ConfigClass = EstimateDiffuseBackgroundConfig
    full_name = "Estimate Diffuse Background"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "SSDC-MS-032 v3.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, spximg: SPHERExImage, spectral_wcs: HDUList,
            he_template: SPHERExImage = None) -> pipeBase.Struct:
        """Estimate the global spectral background in the image.

        Parameters
        ----------
        spximg : `spx_data.SPHERExImage`
            A 2-D science image.
        spectral_wcs : `~astropy.io.fits.HDUList`, optional
            Spectral WCS calibration with central wavelength and bandwidth.
            If not specified, Simulator's wavelength model is used.
        he_template : `spx_data.SPHERExImage`, optional
            He I airglow template. If not specified, no airglow component will
            be estimated.

        Returns
        -------
        `lsst.pipe.base.Struct`
            Container for output connections. The global background estimate is
            stored in the ``global_bkgd`` attribute.

        Notes
        -----
        - Creates a mask from sources and bad pixels.
        """
        # To save memory, use one array and sum components as needed
        global_bkgd = np.full(spximg.data.shape, np.nan, dtype=spximg.data.dtype)
        has_components = False

        # Estimate He I airglow component for D1/D2 images
        detector = spximg.getArrayId()
        if he_template and (detector in [1, 2]) and self.config.est_he_airglow:
            params_airglow_helium = {
                "WL_AIR": he_template.header[HE_TEMP_PAR_KWS["ag_wavelength"]],
                "HW_AIR": he_template.header[HE_TEMP_PAR_KWS["ag_halfwidth"]],
                "HW_BKG": he_template.header[HE_TEMP_PAR_KWS["ag_halfwidth_bkg"]],
                "WL_TOP": [
                    he_template.header[HE_TEMP_PAR_KWS["wl_top_1"]],
                    he_template.header[HE_TEMP_PAR_KWS["wl_top_2"]],
                ],
            }

            if not all(f"HE_AG_{kw}" in spximg.header for kw in HE_FIT_BKG_KWS.values()):
                self.log.warning("He I airglow fit parameters not found in image header. Calculating from"
                                 " He I template...")
                results = fit_airglow_he(
                    spximg, detector, he_template, spectral_wcs, params_airglow_helium, use_variance=True,
                    fit_xslope=True, verbose=False)
            else:
                self.log.info("Reading He I airglow fit parameters from image header...")
                results = read_airglow_fit_from_header(spximg.header, prefix="HE_AG_")

            results["_p_fit"] = reconstruct_p_fit_from_results(results)
            has_components = add_he_airglow(
                global_bkgd, self.log,
                lambda: subtract_airglow_he(spximg, he_template, spectral_wcs, params_airglow_helium,
                                            results, return_model=True)
            )

        # Get wavelength mesh and spectral bins for the detector
        lambda_mesh = spectral_wcs["CWAVE"].data
        spec_config = EstimateAbsGainMatrixConfig()
        _, region_msk_iter_fine = define_spectral_bins(lambda_mesh, spec_config)

        # Compute the average flux and wavelength in each bin
        flags = (Flag.OVERFLOW | Flag.SUR_ERROR | Flag.NONFUNC | Flag.MISSING_DATA | Flag.HOT | Flag.COLD
                 | Flag.NONLINEAR | Flag.PERSIST | Flag.OUTLIER | Flag.CROSSTALK | Flag.SOURCE | Flag.GHOST
                 | Flag.GHOST_FPA | Flag.GHOST_EXT | Flag.STREAK | Flag.BLOOM | Flag.SNOWBALL | Flag.HALO
                 | Flag.SATELLITE_HALO)
        mask = spximg.getFlagsAsMask(flags)
        img_masked = np.ma.array(
            spximg.data - global_bkgd if has_components else spximg.data,
            mask=mask)
        fluxes_fine, wls_fine = compute_avg_flux_and_wavelength(region_msk_iter_fine, img_masked, lambda_mesh)

        # Attempt to compute 1D spectral model
        try:
            spl_fit = make_smoothing_spline(wls_fine, fluxes_fine, lam=0)
            self.log.debug(get_current_rss("Spectral background estimated"))
        # If failure, ``global_bkgd`` contains all zeros or only He I component
        except ValueError:
            self.log.info("Spectral background estimation failed.")
        # If success, continue estimating spectral background
        else:
            # Compute 2D spectral model
            if not has_components:
                global_bkgd[:] = 0.0

            # NOTE: To save memory, we call spline object in chunks.
            for i in range(lambda_mesh.shape[0]):
                global_bkgd[i] += spl_fit(lambda_mesh[i])
            self.log.info("Spectral component added to global background estimate.")

        return pipeBase.Struct(global_bkgd=global_bkgd)
