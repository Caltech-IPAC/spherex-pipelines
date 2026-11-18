__all__ = ["CreateVarianceMapConfig", "CreateVarianceMapTask", "calculate_poisson_term"]

# Standard library imports

# Related third party imports
import numpy as np
from astropy.units import Unit
from astropy.nddata import VarianceUncertainty
from lsst.pex.config import DictField, Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import matplotlib.pyplot as plt
import copy

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, GainFactors, ReadoutParameters, ReadNoiseParameters
from spx_utils import l1_utils, utils
from spx_utils.utils import get_overflow_threshold
from spx_tasks import add_history
# ==============================================================================


class CreateVarianceMapConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "exposure"),
                                   defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    gain_factors = cT.PrerequisiteInput(
        name="gain_factors",
        storageClass="GainFactors",
        dimensions=["instrument"],
        isCalibration=True
    )
    read_noise = cT.PrerequisiteInput(
        name="read_noise",
        doc="Read noise map.",
        storageClass="ReadNoiseParameters",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after variance map created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CreateVarianceMapConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=CreateVarianceMapConnections):
    """Configuration for CreateVarianceMap task."""

    n_lab = Field(
        doc="Lab measured n-reads",
        dtype=float,
        default=71.0)

    frac_uncertainty_boost = DictField(
        doc=("Additional fractional flux uncertainty to add in quadrature "
             "to the variance map, detector specific."
             "e.g. 0.03 gives variance += (0.03 * image)^2."),
        keytype=int,
        itemtype=float,
        default={1: 0.03, 2: 0.03, 3: 0.03, 4: 0.03, 5: 0.03, 6: 0.03})

    apply_variance_boost = Field(
        doc="If True, apply additional fractional variance boost for bright sources.",
        dtype=bool,
        default=False,
    )

def calculate_poisson_term(n: int | np.ndarray) -> float | np.ndarray:
    """Calculate the Poisson term for a given value of n.

    Parameters
    ----------
    n : `int` or `~numpy.ndarray`
        The input value for which the Poisson term is calculated.

    Returns
    -------
    result : `float` or `~numpy.ndarray`
        The Poisson term calculated for the given value(s) of ``n``.

    """
    if type(n) is int:
        if n <= 1:
            raise ValueError(f"Number of reads, {n}, must be greater than 1.")
    if type(n) is np.ndarray:
        # If n reads less than or equal to 1, set indices values to nan.
        n[n <= 1] = np.nan
    return 6 / 5 * (n**2 + 1) / (n**2 - 1)


class CreateVarianceMapTask(pipeBase.PipelineTask):
    """Implementation of L2 Create Variance Map.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-011

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "create-variance-map"
    ConfigClass = CreateVarianceMapConfig
    full_name = "Create Variance Map"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-011 v4.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            gain_factors: GainFactors,
            read_noise: ReadNoiseParameters,
            plot: bool = False) -> pipeBase.Struct:
        """Create a maximum likelihood estimate of the per-pixel variance from the
           processed on-board flux image or down-linked ramp.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input SPHEREx image.
        gain_factors : `~spx_data.GainFactors`
            Per detector gain factors.
        read_noise : `~spx_data.ReadNoiseParameters`
            Electronic noise map; in shape of active + ref pix.
        plot : `bool`, optional
            If True, generate diagnostic plots. Default is False.

        Returns
        -------
        output : `~lsst.pipe.base.Struct`
            An object with an `~spx_data.SPHERExImage` ``spximg_out`` attribute.
            The ``spximg_out`` attribute contains the input image with a
            ``VarianceUncertainty`` layer from the calculated variance.

        Raises
        ------
        ValueError
            Raised if calculated overflow slope is less than or equal to zero.

        Notes
        -----
        Line-fitting method by Robberto 2007 (WFC3 ISR 2007-12, page 25)
        Title: Analysis of the sampling schemes for WFC3-IR by Massimo Robberto
        Link: https://www.stsci.edu/files/live/sites/www/files/home/hst/
        instrumentation/wfc3/documentation/instrument-science-reports-isrs/
        _documents/2007/WFC3-2007-12.pdf

        """
        # Input calib noise maps in shape 2048 x 2048.
        # Input SPHEREx image in l1 shape 2176 x 2072.
        rp = ReadoutParameters.from_metadata(spximg.meta)
        det_id = spximg.getArrayId()
        t_sample = spximg.meta["TSAMP"]

        gains_1 = gain_factors.get_gains_for_detector(det_id)
        # Get gain map from gain factor.
        gain_map = utils.gains_for_image(gains=gains_1, rp=rp, extra_refs=True)

        # Variance calculated in shape of active with ref pixels (w/ extra ref reads).
        actv_ref_area = l1_utils.active_area_range(spximg, include_refs=True)

        variance = np.zeros_like(spximg.data[actv_ref_area])
        if variance.shape != (rp.Jactive + rp.jref * rp.nref, rp.Kactive + rp.kref):
            raise ValueError(f"Varince shape not accurate with active and nrefs, {variance.shape}")

        read_noise_full = read_noise.get_extension_data(ReadNoiseParameters.full)

        # Propagate extra reference row reads to noise maps.
        # NOTE  read noise is specified for all physical pixels.
        read_noise_n = l1_utils.with_extra_refs(rp, read_noise_full)

        # Case 1: Calculate nominal variance.
        # Will subsequently handle pixels flagged overflow, transient, and bad pixels.
        n_nominal = spximg.meta["SURLIM"] - (spximg.meta["SURDLY"] - 1)
        t_intg = n_nominal * t_sample
        poisson_term = calculate_poisson_term(n=n_nominal)

        # Scale electronic noise variance. Lab noise is for N=71 reads.
        N_LAB = self.config.n_lab
        # The poisson term (shot noise) variance
        # is calculated from the total signal in electrons.
        # Signal in electrons = signal_rate (e/s) * integration_time (s)

        elec_var_scaled = read_noise_n * (t_intg**2) * (N_LAB / n_nominal)

        # Start with electronic/read-noise variance everywhere,
        # including reference pixels.
        variance = elec_var_scaled.copy()

        # Add Poisson/shot-noise term only for non-reference pixels.
        # Reference pixels are not photosensitive,
        # so they should not get a photon-noise term.
        ref_mask = (spximg.flags[actv_ref_area] & Flag.REFERENCE) > 0
        nonref_mask = ~ref_mask

        variance[nonref_mask] += (
            poisson_term * spximg.data[actv_ref_area][nonref_mask] * t_intg
        )

        variance /= t_intg**2

        # Variance correction.
        # Separate handling for overflow and transient.
        ovrflw_pix = np.where(
            ((spximg.flags[actv_ref_area] & Flag.OVERFLOW) > 0) & nonref_mask
        )

        # CASE 2: Overflow. Poisson term dominates.
        # Get overflow thresholds in ADU for active pixels.
        ovrflw_thresh_actv = get_overflow_threshold(spximg.header, "KOVR_A")
        # Get overflow thresholds in ADU for reference pixels.
        ovrflw_thresh_ref = get_overflow_threshold(spximg.header, "KOVR_R")

        # Create a robust validity mask and the final set of valid indices
        valid_mask = spximg.data[actv_ref_area][ovrflw_pix] > 0.0

        valid_ovrflw_pix = (ovrflw_pix[0][valid_mask], ovrflw_pix[1][valid_mask])

        n_ovrflw = np.array([])

        # Overflow thesholds in correct shape.
        ovrflw_thresh = np.empty_like(spximg.data[actv_ref_area])
        ovrflw_thresh[:] = ovrflw_thresh_ref
        ovrflw_thresh[rp.jref//2: rp.jref//2 + rp.Jactive, rp.kref//2:-rp.kref//2] = ovrflw_thresh_actv

        # Convert overflow threshold to overflow time using the corrected indices
        # This doesn't yet reflect discrete possible values of t_ovrflw,
        # it gives the total duration from reset frame instead of the integration time.
        t_ovrflw = ((ovrflw_thresh[valid_ovrflw_pix] * gain_map[valid_ovrflw_pix])
                    / spximg.data[actv_ref_area][valid_ovrflw_pix])
        # Estimate number of reads, ``n``, enforces integer values of n
        n_ovrflw = t_ovrflw // t_sample
        # Subtract the 2 ignored frames.
        n_ovrflw -= (spximg.meta["SURDLY"] - 1)
        # Minimum valid exposure.
        n_ovrflw[n_ovrflw < 2] = 2
        # Overflow-flagged integration can't be longer than nominal.
        n_ovrflw[n_ovrflw > n_nominal] = n_nominal

        # CHECK THIS?
        # The (discrete-valued) integration time.
        t_ovrflw = n_ovrflw * t_sample

        # CHECK INDEXING?
        elec_var_ovrflw = read_noise_n[valid_ovrflw_pix] * (t_ovrflw**2) * (N_LAB / n_ovrflw)

        n = copy.copy(n_ovrflw)  # DO WE NEED THIS?
        poisson_term_ovrflw = calculate_poisson_term(n=n)

        # Electronic noise negligible in comparison to source shot noise.
        # Use valid_ovrflw_pix for assignment to correctly update the variance array.
        # Include electronic noise, could be significant contributor in some situations
        variance[valid_ovrflw_pix] = (elec_var_ovrflw + poisson_term_ovrflw
                                      * spximg.data[actv_ref_area][valid_ovrflw_pix] * t_ovrflw)

        variance[valid_ovrflw_pix] /= t_ovrflw**2

        # CASE 2: Transients.
        trnsnt_mask = ((spximg.flags[actv_ref_area] & Flag.TRANSIENT) > 0) & nonref_mask
        if np.any(trnsnt_mask):
            # Case 2a: Early transient (SUR_ERROR > 0)
            early_mask = trnsnt_mask & ((spximg.flags[actv_ref_area] & Flag.SUR_ERROR) > 0)
            if np.any(early_mask):
                n_early = spximg.meta["TRANDLY"] - spximg.meta["SURDLY"]
                t_intg_early = n_early * t_sample
                elec_var_early = (read_noise_n[early_mask]) * (t_intg_early**2) * (N_LAB / n_early)
                poisson_term_early = calculate_poisson_term(n=n_early)
                variance[early_mask] = (elec_var_early + poisson_term_early
                                        * spximg.data[actv_ref_area][early_mask] * t_intg_early)
                variance[early_mask] /= t_intg_early**2

            # Case 2b: Late transient (SUR_ERROR == 0)
            late_mask = trnsnt_mask & ((spximg.flags[actv_ref_area] & Flag.SUR_ERROR) == 0)
            if np.any(late_mask):
                tcut = spximg.header.get("TCUT")
                if tcut is None:
                    raise ValueError("TCUT keyword not found, cannot process late transients.")
                n_late = int(tcut - (spximg.meta["SURDLY"] - 1))
                if n_late > 1:
                    t_intg_late = n_late * t_sample
                    elec_var_late = (read_noise_n[late_mask]) * (t_intg_late**2) * (N_LAB / n_late)
                    poisson_term_late = calculate_poisson_term(n=n_late)
                    variance[late_mask] = (elec_var_late + poisson_term_late
                                           * spximg.data[actv_ref_area][late_mask] * t_intg_late)
                    variance[late_mask] /= t_intg_late**2
                else:
                    self.log.warning(f"Invalid n_late={n_late} from TCUT={tcut}. Setting variance to NaN.")
                    variance[late_mask] = np.nan

        # Update bad pixel mask. SUR_ERROR is now handled under transients.
        variance[np.where((spximg.flags[actv_ref_area] & Flag.MISSING_DATA) > 0)] = np.nan

        # Optional variance boost for bright-source flux uncertainty.
        frac = self.config.frac_uncertainty_boost[det_id]
        if frac < 0:
            raise ValueError(
                f"frac_uncertainty_boost must be non-negative, got {frac}"
            )

        # Add variance boost to account for
        # excess scatter observed in bright-source fluxes.
        # Assumes an additional fractional uncertainty on flux:
        #   sigma_extra = f * image
        # which contributes in quadrature to the variance:
        #   variance += (f * image)^2
        if self.config.apply_variance_boost:
            if frac > 0:
                image_actv_ref = spximg.data[actv_ref_area]
                variance += (frac * image_actv_ref) ** 2

        # Set phantom pixel range in variance to nan.
        phans = np.empty((rp.Jactive + rp.jref * rp.nref, rp.Mphan * rp.nchannel))
        phans[:] = np.nan

        # Variance map output in L1 format.
        variance_l1 = np.hstack((phans, variance), dtype=np.float32)

        # Analyze reference pixels DQA
        self.analyze_reference_pixels(spximg, variance_l1, plot)

        spximg_unit = Unit(spximg.unit)  # variance map in (electron / s)**2
        spximg.uncertainty = VarianceUncertainty(variance_l1, unit=spximg_unit**2)

        return pipeBase.Struct(spximg_out=spximg)


    def analyze_reference_pixels(self, spximg: SPHERExImage, variance_l1, plot: bool,
                                tolerance: float = 0.5,
                                median_atol: float = 1e-1):
        """Analyze the distribution of reference pixels
        and verify statistical properties.

        Parameters
        ----------
        spximg : `SPHERExImage`
            Input SPHEREx image.
        variance_l1 : `np.ndarray`
            Variance map in l1 format.
        plot : `bool`
            Flag to display reference distribution.
        tolerance : `float`
            Tolerance to compare sqrt variance to reference std.
            Default is 50%.
        median_atol : `float`
            Absolute tolerance for checking that reference-pixel median is near zero.
            Default is 0.1.

        """
        # Extract reference pixels
        ref_mask = (spximg.flags & Flag.REFERENCE) > 0

        # Exclude bad/problematic reference pixels
        bad_mask = (
            ((spximg.flags & Flag.MISSING_DATA) > 0)
            | ((spximg.flags & Flag.SUR_ERROR) > 0)
            | ((spximg.flags & Flag.TRANSIENT) > 0)
            | ((spximg.flags & Flag.OVERFLOW) > 0)
        )

        good_ref_mask = ref_mask & ~bad_mask

        ref_pixels = spximg.data[good_ref_mask]
        ref_variance = variance_l1[good_ref_mask]

        if ref_pixels.size == 0:
            self.log.warning("No good reference pixels available for reference-pixel analysis.")
            return

        if plot is True:
            plot_reference_pixel_distribution(ref_pixels)

        ref_mean = np.nanmean(ref_pixels)
        ref_median = np.nanmedian(ref_pixels)
        ref_std = np.nanstd(ref_pixels)

        self.log.info(f"Reference pixels mean: {ref_mean:.4f}")
        self.log.info(f"Reference pixels median: {ref_median:.4f}")
        self.log.info(f"Reference pixel std dev: {ref_std:.4f}")

        # Check that median of reference pixels is close to zero.
        # Reference pixels do not produce photoelectric signal.
        if not np.isclose(ref_median, 0.0, atol=median_atol):
            self.log.warning(
                f"Reference pixels median {ref_median:.4f} is not close to zero "
                f"(atol={median_atol:.4f})."
            )

        mean_ref_variance = np.nanmean(ref_variance)

        if not np.isfinite(mean_ref_variance) or mean_ref_variance <= 0:
            self.log.warning(
                f"Cannot compare reference-pixel std dev to variance: "
                f"mean reference-pixel variance is {mean_ref_variance}."
            )
            return

        expected_ref_std = np.sqrt(mean_ref_variance)

        ratio = ref_std / expected_ref_std
        frac_diff = np.abs(ref_std - expected_ref_std) / expected_ref_std

        self.log.info(f"mean reference-pixel variance: {mean_ref_variance:.4f}")
        self.log.info(f"sqrt(mean reference-pixel variance): {expected_ref_std:.4f}")
        self.log.info(f"Reference std / expected std: {ratio:.3f}")
        self.log.info(f"Reference fractional std difference: {frac_diff:.3f}")

        if not np.isclose(ref_std, expected_ref_std, rtol=tolerance):
            self.log.warning(
                f"Reference pixel std dev {ref_std:.4f} does not match "
                f"expected sqrt reference-pixel variance {expected_ref_std:.4f} "
                f"(ratio={ratio:.3f}, frac_diff={frac_diff:.3f}, tolerance={tolerance:.2f})."
            )

        self.log.info("Reference pixel analysis completed.")


def plot_reference_pixel_distribution(ref_pixels):
    """Plot the distribution of reference pixels with an optimized range.

    Parameters
    ----------
    ref_pixels : `numpy.ndarray`
        Array of reference pixel values.

    """
    # Just for plotting
    # Remove NaN values
    ref_pixels = ref_pixels[~np.isnan(ref_pixels)]

    # Compute statistics
    ref_mean = np.mean(ref_pixels)
    ref_median = np.median(ref_pixels)
    ref_std = np.std(ref_pixels)

    # Define range: focus on \pm 3 standard deviations from mean
    min_clip = max(np.min(ref_pixels), ref_mean - 3 * ref_std)
    max_clip = min(np.max(ref_pixels), ref_mean + 3 * ref_std)

    # Identify outliers beyond 3 sigma
    num_outliers = np.sum((ref_pixels < min_clip) | (ref_pixels > max_clip))

    # Plot histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ref_pixels, bins=50, alpha=0.7, color='b',
             range=(min_clip, max_clip), label="Reference Pixels")

    # Add mean, median, and standard deviation indicators
    plt.axvline(ref_mean, color='r', linestyle='dashed',
                linewidth=2, label=f"Mean: {ref_mean:.4f}")
    plt.axvline(ref_median, color='g', linestyle='dotted',
                linewidth=2, label=f"Median: {ref_median:.4f}")
    plt.axvline(ref_mean + ref_std, color='k', linestyle='dashdot',
                linewidth=2, label=f"Sigma: {ref_std:.4f}")
    plt.axvline(ref_mean - ref_std, color='k', linestyle='dashdot', linewidth=2)

    plt.xlabel("Reference Pixel Value")
    plt.ylabel("Count")
    plt.title("Distribution of Reference Pixels")
    plt.legend()
    plt.grid()

    # Display details about outliers, if found.
    if num_outliers > 0:
        plt.figtext(0.15, 0.85, f"Outliers beyond +/- 3 sigma: {num_outliers}", fontsize=10, color='red')

    plt.show()
