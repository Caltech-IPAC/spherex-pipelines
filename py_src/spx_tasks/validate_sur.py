__all__ = ["ValidateSURConfig", "ValidateSURTask"]

# Standard library imports
import copy
import warnings

# Related third party imports
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.time import Time
from astropy.nddata import VarianceUncertainty
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, ReadoutParameters

# ==============================================================================


class ValidateSURConnections(pipeBase.PipelineTaskConnections,
                             dimensions=("instrument", "exposure"),
                             defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    sur_diag = cT.Output(
        name="sur_diag",
        doc="The SUR diagnostics: image with variance and diagnostic ratio image,"
            "which shows how the approximated data reconstructed from full sample data"
            "are different from L0 data actually received.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure")
    )


class ValidateSURConfig(pipeBase.PipelineTaskConfig,
                        pipelineConnections=ValidateSURConnections):
    pass


# ==============================================================================

def get_full_sample_cols(col_sect: int, rp: ReadoutParameters) -> tuple[slice, slice, slice]:
    """Get SUR full sample area column numbers from *L1 image*.

    Parameters
    ----------
    col_sect  : `int`
        Column section of SUR area.
    rp : `~spx_data.ReadoutParameters`
        Readout parameter values.

    Returns
    -------
    phans : `slice`
        Phantom column start and stop numbers.
    refs : `slice`
        Reference column start and stop numbers.
    actvs : `slice`
        Active columns start and stop numbers.

    """
    phans: slice = None
    refs: slice = None
    actvs: slice = None

    sur_area_actv_cols_even = (rp.wchannel - rp.Mphan) // 2
    sur_area_actv_cols_odd = (rp.wchannel + rp.Mphan) // 2
    # First column section of first channel, 4 phan cols, 4 ref cols, 26 actv cols.
    if col_sect == 0:
        phan_start = 0
        phan_end = phan_start + rp.Mphan
        ref_start = rp.Mphan * rp.nchannel
        ref_end = rp.Mphan * rp.nchannel + rp.kref // 2
        actv_start = rp.Mphan * rp.nchannel + rp.kref // 2
        actv_end = actv_start + sur_area_actv_cols_even - rp.kref // 2
        phans = slice(phan_start, phan_end)
        refs = slice(ref_start, ref_end)
        actvs = slice(actv_start, actv_end)
    # Second column section of last channel, 30 active cols, 4 ref cols.
    elif col_sect == 63:
        actv_start = rp.wchannel * (col_sect // 2) + (rp.wchannel
                                                      - rp.kref // 2) // 2 + rp.Mphan * rp.nchannel
        actv_end = actv_start + sur_area_actv_cols_odd - rp.kref // 2
        ref_start = actv_end
        ref_end = actv_end + rp.kref // 2
        refs = slice(ref_start, ref_end)
        actvs = slice(actv_start, actv_end)
    # Middle even columns: 4 phan cols, 30 active cols.
    elif col_sect % 2 == 0:
        phan_start = col_sect // 2 * rp.Mphan
        phan_end = phan_start + rp.Mphan
        actv_start = rp.wchannel*(col_sect // 2) + rp.Mphan * rp.nchannel
        actv_end = actv_start + sur_area_actv_cols_even
        phans = slice(phan_start, phan_end)
        actvs = slice(actv_start, actv_end)
    # Middle odd columns: 34 active cols.
    else:
        actv_start = rp.Mphan * rp.nchannel + (col_sect // 2) * rp.wchannel + sur_area_actv_cols_even
        actv_end = actv_start + sur_area_actv_cols_odd
        actvs = slice(actv_start, actv_end)

    return phans, refs, actvs


def l0_fullsample_area(spximg: SPHERExImage, row_start: int, col_sect: int,
                       rp: ReadoutParameters) -> np.ndarray:
    """Gets full sample area flags.

    Parameters
    ----------
    data : ~numpy.ndarray
        L0 image data.
    row_start : `int`
        Row starting number for raw data window.
    col_sect : `int`
        Column section for raw data window.
    rp : `~spx_data.ReadoutParameters`
        Readout paramter values.

    Returns
    -------
    raw_data : ~numpy.ndarray
        L0 data in fully sampled area.

    Notes
    -----
    - Starting row number for fully sampled area: 5 - 2037.
    - Column section number for fully sampled area: 0 - 63.
    """
    phans, refs, actvs = get_full_sample_cols(col_sect=col_sect, rp=rp)

    # Row start is one-based, while column section is zero-based.
    sur_area_rows = 8
    row_start -= 1
    row_end = row_start + sur_area_rows

    # NOTE Column section 0 and 63 need to be handled differently
    # because there are reference columns in these sections.

    # First column section: 4 phan, 4 ref, and 30 active cols.
    if col_sect == 0:
        l0_window_data = np.column_stack((spximg.data[row_start:row_end, phans],
                                          spximg.data[row_start:row_end, refs],
                                          spximg.data[row_start:row_end, actvs]))
        l0_window_flgs = np.column_stack((spximg.flags[row_start:row_end, phans],
                                          spximg.flags[row_start:row_end, refs],
                                          spximg.flags[row_start:row_end, actvs]))
    # Last column section: 30 active and 4 ref cols.
    elif col_sect == 63:
        l0_window_data = np.column_stack((spximg.data[row_start:row_end, actvs],
                                          spximg.data[row_start:row_end, refs]))
        l0_window_flgs = np.column_stack((spximg.flags[row_start:row_end, actvs],
                                          spximg.flags[row_start:row_end, refs]))
    # Middle even columns: 4 phan and 30 active cols.
    elif col_sect % 2 == 0:
        l0_window_data = np.column_stack((spximg.data[row_start:row_end, phans],
                                          spximg.data[row_start:row_end, actvs]))
        l0_window_flgs = np.column_stack((spximg.flags[row_start:row_end, phans],
                                          spximg.flags[row_start:row_end, actvs]))
    # Middle odd columns: 34 active cols.
    else:
        l0_window_data = spximg.data[row_start:row_end, actvs]
        l0_window_flgs = spximg.flags[row_start:row_end, actvs]

    return l0_window_data, l0_window_flgs


def clip_outliers(data: np.ndarray, num_clipping_passes: int, sigma: float) -> np.ndarray:
    """Sigma clipping outliers from the readout data.

    Parameters
    ----------
    data : ~numpy.ndarray
        The input data to be clipped.
    num_clipping_passes : `int`
        The maximum number of iterations to perform clipping.
    sigma : `float`, optional. Default = 3.0
        The number of standard deviations to use for clipping limit.


    Returns
    -------
    clipped_data : ~numpy.ndarray
        The clipped data with outliers masked.
    """
    # Perform sigma clipping
    clipped_data = sigma_clip(data,
                              sigma=sigma,
                              maxiters=num_clipping_passes,
                              cenfunc='median',
                              stdfunc='std')

    return clipped_data


def diagnostic_image(sur_frames: np.ndarray,
                     surdly: int,
                     l0_image: np.ndarray,
                     num_clipping_passes: int,
                     sigma_clipping: int) -> tuple[np.ndarray, np.ndarray]:
    """Diagnostic image.

    Paramters
    ---------
    sur_frames : ~numpy.ndarray
        Input SUR readout frames.
    l0_image : ~numpy.ndarray
        Level 0 input image.
    num_clipping_passes : `int`.
        Number of sigma clipping passes.
    sigma_clipping : `int`.
        Sigma value in which to perform clipping.

    Returns
    -------
    diagnostic_img : ~numpy.ndarray
        Diagnostic image L0'.
    """

    # Calculate N-1 photocurrent measurements distribution.
    # Measure photocurrents only when SUR sum accumulation begins.
    readouts_diff = np.diff(sur_frames[surdly:], axis=0)

    # Number of clipping passes specified by user.
    # photocurrent_measurements = readouts_diff
    clipped_data = clip_outliers(readouts_diff, num_clipping_passes, sigma_clipping)

    # Raw data frames with outliers masked.
    unclipped_data = np.where(clipped_data.mask, np.nan, clipped_data.data)
    # Outliers
    outliers = np.ma.array(readouts_diff, mask=clipped_data.mask)
    n_outliers = np.nansum(clipped_data.mask)
    nan_in_unclipped = np.sum(np.isnan(unclipped_data))
    assert n_outliers == nan_in_unclipped

    # Calculate the median and standard deviation of the clipped data.
    median = np.nanmedian(unclipped_data, axis=0)
    sigma = np.nanstd(unclipped_data, axis=0)

    # Form the diagnostic image L0' = (L0 - M) / sigma.
    diagnostic_img = (l0_image - median) / sigma

    return diagnostic_img, median, sigma, unclipped_data, outliers, n_outliers


def handle_blank_planes(data: np.ndarray, blank_value: int) -> np.ndarray:
    """Replace planes with all values equal to BLANK with NaNs or some other
       specified value to indicate missing data.

    Parameters
    ----------
    data : np.ndarray
        The input data cube.
    blank_value : int
        The value used in the data to indicate missing data (BLANK).

    Returns
    -------
    np.ndarray
        The data cube with missing planes replaced.
    """
    # Identify planes where all values are equal to the BLANK value
    blank_plane = np.all(data == blank_value, axis=(1, 2))

    # Replace these planes with NaNs (or another appropriate value)
    data[blank_plane, :, :] = np.nan  # or use a different value as needed

    return data


class ValidateSURTask(pipeBase.PipelineTask):
    """Implementation of L1 Validate SUR.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-008

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "validate-sur"
    ConfigClass = ValidateSURConfig
    full_name = "Validate SUR on Fully-Sampled Region"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-008 v2.6"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self,
            spximg: SPHERExImage,
            num_clipping_passes: int = 1,
            sigma_clipping: int = 3,
            framet: float = 1.5349) -> pipeBase.Struct:
        """Calculate photocurrent measurements in SUR window
           for comparison to on-board SUR algorithm.

        Parameters
        ----------
        spximg
            Input SPHEREx image with full set of SUR readouts.
        num_clipping_passes
        sigma_clipping
        framet

        Returns
        -------
        output : `~lsst.pipe.base.Struct`
            An `~lsst.pipe.base.Struct` object as a `~spx_data.SPHERExImage` where
            the data layer is the sur-diagnostic image, L0'. The diagnostic layer is the
            diagnostic ratio, and the variance set as sigma squared.
        """
        sur = spximg.sur

        if sur is None or sur.data is None:
            warnings.warn("Data is missing SUR data set.", UserWarning)
            # Skip module task if sur data is not available.
            sur_diag = SPHERExImage(data=np.empty((8, 34)) * np.nan, unit=u.electron)
            return pipeBase.Struct(sur_diag=sur_diag)

        try:
            rp = ReadoutParameters.from_metadata(spximg.meta)
        except ValueError:
            rp = ReadoutParameters()

        # SURDLY is value at which SUR sums accumulation begins.
        try:
            surdly = spximg.header["SURDLY"]
        except KeyError:
            surdly = 3  # default

        sur_data = spximg.sur.data

        blank_value = spximg.header.get('BLANK', None)
        if blank_value is not None:
            sur_data = handle_blank_planes(sur_data, blank_value)

        onboard_data, onboard_flgs = l0_fullsample_area(spximg=spximg,
                                                        row_start=sur.row_start,
                                                        col_sect=sur.col_sectn,
                                                        rp=rp)
        if onboard_data.shape != (8, 34):
            print(f"Onboard data shape is not (8, 34), it's: {onboard_data.shape}."
                  " Returning sur_diag is None.")
            return pipeBase.Struct(sur_diag=None)
        # Mask for "bad pixels"
        bad_pix_msk = Flag.SUR_ERROR | Flag.OVERFLOW | Flag.MISSING_DATA
        bad_pix = np.where((onboard_flgs & bad_pix_msk) > 0)
        received_data = copy.deepcopy(onboard_data)
        # Apply bad pixel mask.
        received_data[bad_pix] = np.nan

        # Generate the diagnostic image
        # framet should be applied to raw data in ingest.
        diagnostic_img, median, sigma, unclipped_data, outliers, n_outliers =\
            diagnostic_image(sur_data/framet, surdly, received_data, num_clipping_passes, sigma_clipping)

        l0_data = copy.deepcopy(received_data)
        diagnostic_data = copy.deepcopy(diagnostic_img)
        diagnostic_ratio = (l0_data - diagnostic_data) / l0_data

        # SUR diagnostic output product.
        sur_diag = SPHERExImage(data=diagnostic_img, unit=sur.unit)
        # Diagnostic ratio.
        sur_diag.diagnostic_ratio = diagnostic_ratio
        # Variance.
        sur_diag.uncertainty = VarianceUncertainty(sigma**2, unit=sur.unit**2)
        # Median.
        sur_diag.diagnostic_median = median
        # Standard deviation.
        sur_diag.diagnostic_sigma = sigma

        # Trending values.
        sur_diag.header["NCLIPP"] = n_outliers  # Total clipped pixels in raw data.
        # Total onboard transients in l0 image 8x34 window.
        n_transients = np.nansum((onboard_flgs & Flag.TRANSIENT) > 0)
        sur_diag.header["NTRNST"] = n_transients
        sur_diag.header["DATE"] = Time.now().to_value(format="fits")
        sur_diag.header.comments["DATE"] = "HDU creation time (UTC)"
        sur_diag.header["MJD-BEG"] = spximg.header["MJD-BEG"]
        sur_diag.header["MJD-END"] = spximg.header["MJD-END"]

        output = pipeBase.Struct(sur_diag=sur_diag, unclipped_data=unclipped_data, outliers=outliers)

        return output
