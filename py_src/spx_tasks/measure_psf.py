__all__ = ["MeasurePSFConfig", "MeasurePSFTask"]

# Standard library imports
import time
import logging
import random
import string
import multiprocessing
import traceback

# Related third party imports
from astropy.table import Table
import astropy.units as u
from lsst.pex.config import Field, ListField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from joblib import Parallel, delayed

# Local application/library specific imports
from spx_data import SPHERExImage, PSFZoneData
from spx_utils.psf.jdk import photutils_jdk
from spx_utils.psf.superresolution_psf import stack_psf, deconvolve_psf
from spx_utils.psf.superresolution_psf import convolve_psf_with_jitter
from spx_utils.psf.psf_parameters import PSFConstants
from spx_tasks import add_history
from spx_utils.psf.psf_fwhm import calculate_fwhm, plot_fwhm_histogram, process_psf_zone_data
from spx_utils.psf.psf_plots import plot_psf
from spx_data.spx_param import get_pix_scale
# ==============================================================================


def _process_psf_bin_measure(ibin, spximg, srccat, array_id, config, plt, verbose):
    """Process a single PSF bin for MeasurePSFTask."""
    array_id = spximg.getArrayId()
    try:
        stack, super_psf, stack_delete, cat_down = stack_psf(
            spximg=spximg,
            cat=srccat,
            iterableImgCat=None,
            imageHandles=None,
            catalogHandles=None,
            sip=None,
            multi_exprs=False,
            bins_x=config.bins_x,
            bins_y=config.bins_y,
            bins=config.bins,
            ibin=ibin,
            plt=plt,
            verbose=verbose,
            array_id=array_id,
            config=config)

        super_psf.rldIterations = config.rldIterations
        super_psf.rldIterations_decon = config.rldIterations_decon
        super_psf.stackDeconIterations = config.stackDeconIterations
        super_psf.damper_factor = config.damper_factor

        psf_bin, _ = deconvolve_psf(stack, super_psf, stack_delete, cat_down, plt, verbose,
                                    config.plot_path, ibin, array_id)

        # Return super_psf from bin 0 only
        return ibin, psf_bin, super_psf
    except Exception as e:
        print(f"[Parallel Error] Bin {ibin} failed with exception: {e}")
        traceback.print_exc()
        return ibin, None, None


def calculate_psf_solid_angle(psf: np.ndarray) -> float:
    """Calculate the PSF solid angle.

    Parameters
    ----------
    psf : `np.ndarray`
        PSF data for all zones

    Returns
    -------
    `float`
        PSF solid angle.
    """
    # Prevent division by zero by checking for an empty or all-NaN PSF
    maximum_flux = np.nanmax(psf)
    print(f'maximum_flux = {maximum_flux}')
    if np.nansum(psf) == 0 or maximum_flux == 0:
        return np.nan * u.arcsec**2  # Return NaN with proper units

    # Normalize the PSF
    normalized_psf = psf / maximum_flux
    # Integrate the normalized PSF over all pixels
    integrated_psf = np.nansum(normalized_psf)
    print(f'integrated psf = {integrated_psf}')

    # Get pixel solid angle
    pixel_scale_arcsec = get_pix_scale()
    # Ensure units are arcsec^2
    pixel_solid_angle = (pixel_scale_arcsec**2).to(u.arcsec**2)
    print(f'pixel solid angle = {pixel_solid_angle.to_value()}')

    # Calculate the PSF solid angle
    psf_solid_angle = integrated_psf * pixel_solid_angle
    print(f"psf solid angle = {psf_solid_angle.to_value(u.arcsec**2)}")

    return psf_solid_angle


class MeasurePSFConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "exposure"),
                            defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    psf_avg = cT.PrerequisiteInput(
        name="average_psf",
        doc="Input psf library.",
        storageClass="PSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalogs.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output after measure per-exposure PSF.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class MeasurePSFConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=MeasurePSFConnections):
    """Configuration parameters for Estimate PSF task."""

    # NOTE
    # ``llx`` and ``lly`` obtained from fake_PSF() in fake_products.py
    # # Get centers of the zones (represented in FITS convention).
    # zone_x_centers = np.asarray([ii*npx_x/nbr_x_zones + (npx_x/nbr_x_zones + 1)/2
    #                              for ii in range(nbr_x_zones)])
    # zone_y_centers = np.asarray([ii*npx_y/nbr_y_zones + (npx_y/nbr_y_zones + 1)/2
    #                              for ii in range(nbr_y_zones)])
    # TRY 3x3 because not enough sources for one exposure !!!

    llx = ListField(
        doc="X offsets of the effective PSF bins.",
        dtype=float,
        default=[340.5, 1020.5, 1700.5])

    lly = ListField(
        doc="Y offsets of the effective PSF bins.",
        dtype=float,
        default=[340.5, 1020.5, 1700.5])

    oversample = Field(
        doc="Oversample factor for super-resolution PSF.",
        dtype=int,
        default=10)

    bands = Field(
        doc="Number of detectors (aka bands).",
        dtype=int,
        default=1)  # Default should be 6.

    exp = Field(
        doc="Number of exposures.",
        dtype=int,
        default=1)

    bins = Field(
        doc="Number of PSF zones."
        "Number of bins to slice each grid into for stacking",
        dtype=int,
        default=9)

    bins_x = Field(
        doc="Number of PSF zones."
        "Number of bins to slice each grid into for stacking",
        dtype=int,
        default=3)

    bins_y = Field(
        doc="Number of PSF zones."
        "Number of bins to slice each grid into for stacking",
        dtype=int,
        default=3)

    rldIterations = Field(
        doc="Number of Richardson-Lucy Deconvolution iterations.",
        dtype=int,
        default=20
    )

    rldIterations_decon = Field(
        doc="Number of Richardson-Lucy Deconvolution iterations for decon.",
        dtype=int,
        default=15
    )

    stackDeconIterations = Field(
        doc="Number of iterations for stack deconvolution.",
        dtype=int,
        default=3
    )

    damper_factor = Field(
        doc="Damper factor for damping deconvolution artifacts.",
        dtype=float,
        default=0.2
    )

    trim_pixels = Field(
        doc="Trim pixel edges for Jitter Difference Kernel.",
        dtype=int,
        default=10
    )

    window = Field(
        doc="create matching kernel window parameter: "
            "tophat, cosine, hanning, splitcosine, tukey ",
        dtype=str,
        default="hanning"
    )

    alpha = Field(
        doc="alpha parameter for window parameter in create_matching_kernel ",
        dtype=float,
        default=0.5
    )

    beta = Field(
        doc="beta parameter for window parameter in create_matching_kernel ",
        dtype=float,
        default=0.2
    )

    plot_path = Field(
        doc="Path to save plots.",
        dtype=str,
        default=""
    )

    smooth_jdk = Field(
        doc="Do smoothing sigma for JDK.",
        dtype=bool,
        default=False
    )

    sigma_jdk = Field(
        doc="Smoothing sigma for JDK.",
        dtype=float,
        default=1.0
    )

    b = Field(
        doc="Galactic latitdue cut.",
        dtype=float,
        default=0.0
    )

    do_jitter = Field(
        doc="Deconvolve with jitter",
        dtype=bool,
        default=False
    )

    num_cores = Field(
        doc="Number of cores",
        dtype=int,
        default=4
    )

    run_parallel = Field(
        doc="Run PSF zone processing in parallel.",
        dtype=bool,
        default=True
    )

    do_psf_omega = Field(
        doc="Add PSF solid angle to header.",
        dtype=bool,
        default=False
    )


class MeasurePSFTask(pipeBase.PipelineTask):
    """Implementation of L2 Measure PSF.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-023

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "measure-psf"
    ConfigClass = MeasurePSFConfig
    full_name = "Measure Per-Exposure PSF"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-023 v3.3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            srccat: Table,
            psf_avg: PSFZoneData,
            plt: bool = False,
            verbose: bool = False) -> pipeBase.Struct:
        """Calculate per-exposure effective PSF.
        This involves calculating per-exposure Jitter Difference Kernel
        and applying it to exposure-averaged effective PSF computed from
        multiple images to obtain per-exposure PSF.

        Parameters
        ----------
        spximg : SPHERExImage
        psf_avg : exposure-averaged PSF

        Returns
        -------
        output : Output
            Output with image containing per-exposure effective PSF
        """
        c = self.config

        if c.do_jitter is True:
            array_id = spximg.getArrayId()
            # # NOTE Temporary workaroud to handle NaN PSF zone.
            # Iterate over each 2D slice in the 3D array
            for i in range(1, psf_avg.data.shape[0]):
                if np.isnan(psf_avg.data[i]).any():
                    # Replace the current slice with the previous one
                    logging.warning("PSF zone is NaN copying previous zone.")
                    psf_avg.data[i] = psf_avg.data[i - 1].copy()

            try:
                obsid = spximg.header["OBSID"]
            except KeyError:
                char_set = string.ascii_letters + string.digits
                obsid = ''.join(random.choice(char_set) for _ in range(8))

            deconvolved_stack_bins = np.zeros((c.bins, PSFConstants.stackSize, PSFConstants.stackSize))

            num_cores = min(multiprocessing.cpu_count(), c.num_cores)

            if c.run_parallel:
                results = Parallel(n_job=num_cores, backend="loky")(
                    delayed(_process_psf_bin_measure)(ibin, spximg, srccat, array_id, c, plt, verbose)
                    for ibin in range(c.bins)
                )
            else:
                results = [
                    _process_psf_bin_measure(ibin, spximg, srccat, array_id, c, plt, verbose)
                    for ibin in range(c.bins)
                ]

            super_psf = None
            failed_bins = []

            for ibin, psf_bin, spf in results:
                if psf_bin is not None:
                    deconvolved_stack_bins[ibin] = psf_bin
                else:
                    failed_bins.append(ibin)

                if super_psf is None and spf is not None:
                    super_psf = spf  # Use the first valid one, not just ibin == 0

            if failed_bins:
                logging.warning(f"[MeasurePSF] PSF measurement failed in bins: {failed_bins}")

            if super_psf is None:
                raise RuntimeError("[MeasurePSF] No valid PSF found in any bin. Unable to proceed.")

            # Find Jitter PSF with photutils.
            jdktime_0 = time.time()
            if verbose is True:
                print("Solving for JDK...")
            jitter_psf, group_sizes = photutils_jdk(deconvolved_stack_bins, psf_avg.data,
                                                    super_psf.halfStampSizeUp, c.bins,
                                                    c.trim_pixels, c.window, c.alpha, c.beta,
                                                    verbose, plt)
            jdktime = time.time() - jdktime_0
            if verbose is True:
                print(f"JDK time: {jdktime}")

            # PSF_exposure = PSF_avg * jdk
            # Convolve psf avg with jitter PSF.
            tpsf = time.time()
            if verbose is True:
                print("Obtaining Effective PSF...")

            # Effective PSF
            eff_psf = convolve_psf_with_jitter(psf_avg.data, jitter_psf, group_sizes,
                                               smooth_jdk=self.config.smooth_jdk,
                                               sigma_jdk=self.config.sigma_jdk)
            psftime = time.time() - tpsf

            if verbose is True:
                print('PSF Sim Time:'+str(psftime))

            eff_psf *= (np.nansum(psf_avg.data) / np.nansum(eff_psf))

        else:
            eff_psf = psf_avg.data
            jitter_psf = None
            deconvolved_stack_bins = None

        llxy = [93.22727273, 278.68181818, 464.13636364, 649.59090909, 835.04545455,
                1020.5, 1205.95454545, 1391.40909091, 1576.86363636, 1762.31818182,
                1947.77272727]

        # Return measured psf per-exposure.
        psf = PSFZoneData(data=eff_psf,
                          x_centers=llxy,
                          y_centers=llxy)

        spximg.psfcube = psf
        fwhms = calculate_fwhm(psf)
        fwhm_median = np.nanmedian(fwhms['fwhms_arcsec'])
        # Check for NaN and set to 0.0 if so
        if np.isnan(fwhm_median):
            fwhm_median = 0.0
        spximg.header["PSF_FWHM"] = fwhm_median
        spximg.header.comments["PSF_FWHM"] = " [arcsec] PSF FWHM Median"

        if plt is True:
            process_psf_zone_data(psf, output_dir=c.plot_path, obsid=obsid)
            plot_fwhm_histogram(fwhms['fwhms_arcsec'], 1, output_dir=c.plot_path, obsid=obsid)
            plot_psf(spximg.psfcube.data, output_dir=c.plot_path, obsid=obsid)

        if self.config.do_psf_omega:
            psf_omega = calculate_psf_solid_angle(spximg.psfcube.data)
            # Check if it's NaN and set to zero if so
            if np.isnan(psf_omega):
                psf_omega = 0.0 * u.arcsec**2  # preserve units
            spximg.header["HIERARCH PSF_OMEGA"] = (psf_omega.to_value(u.arcsec**2),
                                                   "[arcsec^2] PSF solid angle")

        return pipeBase.Struct(spximg_out=spximg,
                               jitter_psf=jitter_psf,
                               deconvolved_stack_bins=deconvolved_stack_bins,
                               fwhm_median=fwhm_median)
