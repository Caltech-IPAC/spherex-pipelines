__all__ = ["EstimateDiffuseBackgroundConfig", "EstimateDiffuseBackgroundTask"]

# Standard library imports

# Related third party imports
from astropy.io.fits import HDUList
from astropy.stats import SigmaClip, sigma_clipped_stats
from lsst.pex.config import Field, ListField
from photutils.background import Background2D
from scipy.interpolate import make_smoothing_spline
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, NullGlobal, get_wavelength_mesh
from spx_data.spx_param import get_npix_x_active, get_npix_y_active
from spx_tasks.estimate_abs_gain_matrix import (compute_avg_flux_and_wavelength, define_spectral_bins,
                                                EstimateAbsGainMatrixConfig)


# ==============================================================================
def check_box_size(value: list[int, int]) -> bool:
    """Check that the box size for global background estimation is a factor of
       the number of active pixels along X and Y.
    """
    if (get_npix_x_active() % value[0] == 0) & (get_npix_y_active() % value[1] == 0):
        return True
    else:
        return False


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
    )
    global_bkgd = cT.Output(
        name="global_bkgd",
        doc="Estimated global background.",
        storageClass="NumpyArray",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.use_simulator_bandpass:
            self.inputs.discard("spectral_wcs")


class EstimateDiffuseBackgroundConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=EstimateDiffuseBackgroundConnections):
    sc_sigma_lower = Field(
        doc="Number of standard deviations to use for lower limit.",
        dtype=float,
        default=3)

    sc_sigma_upper = Field(
        doc="Number of standard deviations to use for upper limit.",
        dtype=float,
        default=3)

    sc_maxiters = Field(
        doc="Maximum number of sigma-clipping iterations to perform.",
        dtype=int,
        default=10)

    box_size = ListField(
        doc="The box size for subregions for image for background analysis."
        "Should be a divisible value.",
        dtype=int,
        listCheck=check_box_size,
        default=(40, 40))

    filter_size = ListField(
        doc="Box size of median filter to apply to low-res background."
        "A value of 1 or (1, 1) disables filtering",
        dtype=int,
        default=(3, 3))

    exclude_percentile = Field(
        doc="Percentage of masked pixels allowed in a box. If the percent of"
            " masked pixels exceeds this threshold, the box will be excluded"
            " from the low-resolution map.",
        dtype=float,
        default=75.0)

    pct_good_regions = Field(
        doc="Desired minimum percentage of 'good' regions to be used for "
            "background estimation. We define 'good' regions as those with a "
            "percentage of masked pixels below the exclude_percentile value.",
        dtype=float,
        default=90.0)

    sc_median_thresh = Field(
        doc="The upper bound for the median of sigma-clipped, unmasked pixels "
        "in the spectral and diffuse-subtracted image. The median value should "
        "be close to zero if background removal is working well.",
        dtype=float,
        default=2e-3)

    use_simulator_bandpass = Field(
        doc="Use simulator wavelength model.",
        dtype=bool,
        default=False)


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

    def run(self, spximg: SPHERExImage, spectral_wcs: HDUList = None) -> pipeBase.Struct:
        """Estimate median background and return the background as output.

        Parameters
        ----------
        spximg : `spx_data.SPHERExImage`
            A 2-D science image.
        spectral_wcs : `~astropy.io.fits.HDUList`, optional
            Spectral WCS calibration file with center wavelength and center bandwidth.
            If not specified, Simulator's wavelength model is used.

        Returns
        -------
        `lsst.pipe.base.Struct`
            Container for output connections. The global background estimate is
            stored in the ``global_bkgd`` attribute.


        Notes
        -----
        - Creates a mask from sources and bad pixels.
        """
        # =====================================================================
        # Get spectral bins for the detector
        spec_config = EstimateAbsGainMatrixConfig()
        region_msk_iter, region_msk_iter_fine = define_spectral_bins(spximg, spec_config)
        # assert region_msk_iter.num_regions == spec_config.num_regions
        # assert region_msk_iter_fine.num_regions == spec_config.num_regions_fine

        # Compute the average flux and wavelength in each bin
        if spectral_wcs is None:
            # using Simulator bandpass model
            lambda_mesh = get_wavelength_mesh(array_id=spximg.getArrayId(), npix_x=spximg.data.shape[1],
                                              npix_y=spximg.data.shape[0])
        else:
            lambda_mesh = spectral_wcs["CWAVE"].data

        flags = (Flag.OVERFLOW | Flag.SUR_ERROR | Flag.NONFUNC | Flag.MISSING_DATA | Flag.HOT | Flag.COLD
                 | Flag.NONLINEAR | Flag.PERSIST | Flag.OUTLIER | Flag.CROSSTALK | Flag.SOURCE | Flag.GHOST
                 | Flag.GHOST_FPA | Flag.GHOST_EXT | Flag.STREAK)
        mask = spximg.getFlagsAsMask(flags)
        img_masked = np.ma.array(spximg.data, mask=mask)
        fluxes_fine, wls_fine = compute_avg_flux_and_wavelength(region_msk_iter_fine, img_masked, lambda_mesh)

        # Attempt to compute 1D spectral model
        try:
            spl_fit = make_smoothing_spline(wls_fine, fluxes_fine, lam=0)
        # If failure, pass null global BG to photometry
        except ValueError:
            global_bkgd = np.full(spximg.data.shape, np.nan)
            self.log.info("Spectral background estimation failed; passing null global background to "
                          "photometry.")
        # If success, continue estimating spectral and diffuse components
        else:
            # Compute 2D spectral model
            spec_bkgd = spl_fit(lambda_mesh)

            # =================================================================
            # Check number of "good" regions (regions where % of masked pixels
            # below exclude_percentile parameter passed to Background2D)
            num_good_regions = 0.0
            num_rows = spximg.data.shape[0] // self.config.box_size[0]
            num_cols = spximg.data.shape[1] // self.config.box_size[1]
            for row in range(num_rows):
                for col in range(num_cols):
                    r_low = row * self.config.box_size[0]
                    r_high = r_low + self.config.box_size[0]
                    c_low = col * self.config.box_size[1]
                    c_high = c_low + self.config.box_size[1]
                    masked_pix = np.count_nonzero(mask[r_low:r_high, c_low:c_high])
                    frac_masked_pix = masked_pix / (self.config.box_size[0] * self.config.box_size[1])
                    if frac_masked_pix * 100. < self.config.exclude_percentile:
                        num_good_regions += 1
            total_regions = num_rows * num_cols
            frac_good_regions = num_good_regions / total_regions

            # Set up sigma clipper
            sigma_clip = SigmaClip(sigma_lower=self.config.sc_sigma_lower,
                                   sigma_upper=self.config.sc_sigma_upper, maxiters=self.config.sc_maxiters)

            # Attempt diffuse BG estimation on spectral-subtracted image
            spec_removed = spximg.data - spec_bkgd
            if frac_good_regions * 100. >= self.config.pct_good_regions:
                try:
                    diff_bkgd = Background2D(data=spec_removed, mask=mask, box_size=self.config.box_size,
                                             sigma_clip=sigma_clip, filter_size=self.config.filter_size,
                                             exclude_percentile=self.config.exclude_percentile)
                except ValueError:
                    # In the event Background2D estimate still fails
                    diff_bkgd = NullGlobal(data=spec_removed, box_size=self.config.box_size,
                                           sigma_clip=sigma_clip, filter_size=self.config.filter_size,
                                           exclude_percentile=self.config.exclude_percentile)
            else:
                diff_bkgd = NullGlobal(data=spec_removed, box_size=self.config.box_size,
                                       sigma_clip=sigma_clip, filter_size=self.config.filter_size,
                                       exclude_percentile=self.config.exclude_percentile)

            # Determine if global or local BG estimate is suitable
            if isinstance(diff_bkgd, NullGlobal):
                global_bkgd = diff_bkgd.background
                self.log.info("Background2D estimate failed; passing null global background to photometry.")
            else:
                # Check summary statistics
                mean, median, stddev = sigma_clipped_stats(
                    data=spec_removed - diff_bkgd.background, mask=mask,
                    sigma_lower=self.config.sc_sigma_lower, sigma_upper=self.config.sc_sigma_upper,
                    maxiters=self.config.sc_maxiters)

                if abs(median) < self.config.sc_median_thresh:
                    # Global BG removal reduces noise effectively
                    global_bkgd = spec_bkgd + diff_bkgd.background
                    self.log.info("Global background estimation succeeded; passing sum of spectral and "
                                  "diffuse components to photometry.")
                else:
                    # Global BG removal does NOT reduce noise effectively
                    global_bkgd = np.full(spximg.data.shape, np.nan)
                    self.log.info(f"Sigma-clipped median of unmasked pixels in subtracted image >= "
                                  f"{self.config.sc_median_thresh}; passing null global background to "
                                  f"photometry.")

        return pipeBase.Struct(global_bkgd=global_bkgd)
