# Standard library imports
import math

# Related third party imports
from dataclasses import dataclass, field
import astropy.units as u
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.wcs import Sip
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import interpn
from scipy.ndimage import gaussian_filter
try:
    from SPHEREx_Simulator_Tools import data_filename
    from SPHEREx_InstrumentSimulator import PSF
    hasSPHERExSimulator = True
except ImportError:
    hasSPHERExSimulator = False

# Local application/library specific imports
from spx_data import (
    GainFactors,
    NonLinearityParameters,
    PSFZoneData,
    ReadoutParameters,
    SPHERExImage,
    spx_param,
    PersistenceState,
    PersistenceStateModelPars,
    ReadNoiseParameters,
    NonfuncData
)

from spx_utils import spectral_wcs_lookup_data, get_data_file
from spx_utils.utils import add_detcoord


@dataclass
class ArrayCalValues:
    """CAL values for instrument to physical unit conversion.

    Notes
    -----
    Coded here because we will not have access
    to these numbers with in-orbit data taking.
    """
    cal_values: dict = field(default_factory=lambda: {
        1: 8.286700087641215,
        2: 8.38949569816677,
        3: 7.896646949035173,
        4: 9.867400104358904,
        5: 2.0516901757300006,
        6: 1.682176698588182
    })


def initial_distortion() -> Table:
    """Returns the intial csv ray-tracing data."""
    dist_file = get_data_file("data/distorted_pixels_20240425.csv", package="spx_utils")
    return Table.read(dist_file)


def fake_persistence_state() -> PersistenceState:
    """Fake persistence state for pipeline. (mutable)"""
    # Initialize PersistenceData with empty state.
    rp = ReadoutParameters()

    empty_persist_model = PersistenceStateModelPars(
        a_n=np.zeros((rp.Jframe, rp.Kframe, 6), dtype=np.float32),
        b_prev2=np.zeros((rp.Jframe, rp.Kframe, 6), dtype=np.float32),
        b_prev1_data=np.empty((2, rp.Jframe, rp.Kframe)),
        b_prev1_flgs=np.empty((rp.Jframe, rp.Kframe)),
    )
    primary_hdu = fits.Header()
    fits_content = [
        fits.PrimaryHDU(header=primary_hdu),
        fits.ImageHDU(data=empty_persist_model.a_n, header=fits.Header({"EXTNAME": "a_n"})),
        fits.ImageHDU(data=empty_persist_model.b_prev2, header=fits.Header({"EXTNAME": "b_prev2"})),
        fits.ImageHDU(data=empty_persist_model.b_prev1_data, header=fits.Header({"EXTNAME": "b_prev1_data"})),
        fits.ImageHDU(data=empty_persist_model.b_prev1_flgs, header=fits.Header({"EXTNAME": "b_prev1_flgs"})),
    ]
    persistence_state = PersistenceState(hdulist=fits.HDUList(fits_content),
                                         detector=1, num_prev=0, prev1_end_time=0.0)
    return persistence_state


def fake_ssocat() -> Table:
    """Fake Solar System Objects catalog"""
    names = ["object", "ra", "dec", "vmag", "x", "y"]
    units = ["", "deg", "deg", "V", "pixel", "pixel"]
    dtype = [str, float, float, float, float, float]
    rows = [["fake", 0.0, 0.0, 0.0, 0.0, 0.0]]
    return Table(rows=rows, names=names, dtype=dtype, units=units)


def fake_nonfunc_map(detector_id: int) -> SPHERExImage:
    """Fake Nonfunctional Pixels map"""
    npix_x = spx_param.get_npix_x()
    npix_y = spx_param.get_npix_y()
    data = np.zeros([npix_x, npix_y], dtype=np.uint8)
    np.fill_diagonal(data, 1)
    img = SPHERExImage(data, unit=u.dimensionless_unscaled, dtype_data=data.dtype)
    img.setArrayId(detector_id)
    # add detector coordinate system
    add_detcoord(img, device_coord=True)
    return img


def fake_nonfunc_v2_map(detector: int, seed: int | None = 42) -> NonfuncData:
    """Fake Nonfunctional v2 product, returned as a NonfuncData instance.

    Builds a multi-extension FITS with:
      - COMBINED (1=bad, 0=good)
      - DARKCURR, HIGHDRIFT, SPECCAL, TRANSIENT, OVERFLOW (each 1=bad, 0=good)

    Parameters
    ----------
    detector : `int`
        Detector ID written into the PrimaryHDU header.

    Returns
    -------
    NonfuncData
        Parsed wrapper around the generated HDUList.
    """
    rng = np.random.default_rng(seed)

    # Match how fake_nonlin_inputs gets its shape
    ny, nx = spx_param.get_npix_x(), spx_param.get_npix_y()

    def sparse_mask(count: int) -> np.ndarray:
        """Create a sparse binary mask (1=bad) with ~count bad pixels."""
        m = np.zeros((ny, nx), dtype=np.uint8)
        if count <= 0:
            return m
        count = min(count, ny * nx)
        idx = rng.choice(ny * nx, size=count, replace=False)
        m.flat[idx] = 1
        return m

    darkcurr = sparse_mask(300)
    highdrift = sparse_mask(250)
    speccal = sparse_mask(120)
    transient = sparse_mask(7000)
    overflow = sparse_mask(60)

    # Add a couple structured features for easy QA
    np.fill_diagonal(speccal, 1)      # thin diagonal line
    box = 24
    darkcurr[:box, :box] |= 1         # small UL box

    combined = (darkcurr | highdrift | speccal | transient | overflow).astype(np.uint8)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DETECTOR"] = detector
    primary_hdu.header["BPCONV"] = (1, "Bad-pixel convention: 1=bad, 0=good")

    combined_hdu = fits.ImageHDU(data=combined, header=fits.Header({"EXTNAME": "COMBINED"}))
    darkcurr_hdu = fits.ImageHDU(data=darkcurr, header=fits.Header({"EXTNAME": "DARKCURR"}))
    highdrift_hdu = fits.ImageHDU(data=highdrift, header=fits.Header({"EXTNAME": "HIGHDRIFT"}))
    speccal_hdu = fits.ImageHDU(data=speccal, header=fits.Header({"EXTNAME": "SPECCAL"}))
    transient_hdu = fits.ImageHDU(data=transient, header=fits.Header({"EXTNAME": "TRANSIENT"}))
    overflow_hdu = fits.ImageHDU(data=overflow, header=fits.Header({"EXTNAME": "OVERFLOW"}))

    hdul = fits.HDUList([
        primary_hdu,
        combined_hdu,
        darkcurr_hdu,
        highdrift_hdu,
        speccal_hdu,
        transient_hdu,
        overflow_hdu,
    ])

    # Optional integrity check
    union = (darkcurr | highdrift | speccal | transient | overflow).astype(bool)
    cmb = (combined == 1)
    if np.any(union & ~cmb) or np.any(cmb & ~union):
        raise ValueError("COMBINED is not the union of category planes.")

    return NonfuncData(hdul, detector=detector)


def fake_dichroic_cut_map(detector_id: int) -> SPHERExImage:
    """Fake Dichroic Cut Pixels map"""
    npix_x = spx_param.get_npix_x_active()
    npix_y = spx_param.get_npix_y_active()
    data = np.zeros([npix_x, npix_y], dtype=np.uint8)
    # dichroic cut only affects detectors 3 and 4
    if detector_id == 3:
        data[:50, :] = 1
    elif detector_id == 4:
        data[-50:, :] = 1
    img = SPHERExImage(data, unit=u.dimensionless_unscaled, dtype_data=data.dtype)
    img.setArrayId(detector_id)
    return img


def fake_ff_calib(detector_id: int) -> SPHERExImage:
    """Fake Flat Field Calibration"""
    rp = ReadoutParameters()
    data = np.ones((rp.Jactive, rp.Kactive), dtype=np.float32)
    img = SPHERExImage(data, unit=u.dimensionless_unscaled)
    img.setArrayId(detector_id)
    # add detector coordinate system
    add_detcoord(img)
    return img


def fake_dc_calib(detector_id: int) -> SPHERExImage:
    """Fake Dark Current Calibration"""
    npix_x = spx_param.get_npix_x_active()
    npix_y = spx_param.get_npix_y_active()
    data = np.zeros([npix_x, npix_y], dtype=np.float32)
    img = SPHERExImage(data, unit=u.Unit("electron/s"))
    img.setArrayId(detector_id)
    # add detector coordinate system
    add_detcoord(img)
    return img


def fake_abs_gain_matrix(detector_id: int) -> SPHERExImage:
    """Fake Absolute Gain Matrix"""
    cal_value = ArrayCalValues().cal_values.get(detector_id)
    npix_x = spx_param.get_npix_x_active()
    npix_y = spx_param.get_npix_y_active()
    data = np.ones([npix_x, npix_y], dtype=np.float32) / cal_value
    img = SPHERExImage(data, unit=u.Unit("MJy/sr") / u.Unit("electron/s"))
    img.setArrayId(detector_id)
    # add detector coordinate system
    add_detcoord(img)
    return img


def fake_sip(detector_id: int) -> Sip:
    """Fake Distortion / SIP Coefficients"""
    npix_x = spx_param.get_npix_x_active()
    npix_y = spx_param.get_npix_y_active()
    a = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)

    b = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                  [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    crpix = np.array([npix_y / 2, npix_x / 2])
    return Sip(a, b, None, None, crpix)


def fake_gain_factors() -> GainFactors:
    """Generates a sample `~spx_data.GainFactors`.

    Returns
    -------
    `~spx_data.GainFactors`
        Sample gain factors for each array id.
    """
    rp = ReadoutParameters()

    g_fake = [1.] * rp.nchannel

    gain_factors = GainFactors(gains_1=g_fake, gains_2=g_fake,
                               gains_3=g_fake, gains_4=g_fake,
                               gains_5=g_fake, gains_6=g_fake,
                               num_channels=rp.nchannel)
    return gain_factors


def fake_read_noise_map(detector: int) -> ReadNoiseParameters:
    """Generates sample per-pixel readout noise map matching Simulator's implementation
    based on Robberto 2007, page 25.

    Parameters
    ----------
    detector : `int`, optional
        Detector id number, defaults to 1.

    Returns
    -------
    `ReadNoiseParameters`
        Per-pixel readout noise map with random values.

    Notes
    -----
    - The map will be in the shape of the active area with reference pixels,
      where nref = 1.
    - There will be a separate map for each of the six detectors.
    - Read noise maps are in units of electrons.
    """
    rp = ReadoutParameters()

    # noise for single read: standard deviation in electrons
    noise_sigma_CDS = spx_param.get_cds_read_noise(detector)

    # read noise for a single read
    readnoise_1 = np.full((rp.Jactive+rp.jref, rp.Kactive+rp.kref),
                          noise_sigma_CDS,
                          dtype=np.float32)

    # CDS sampling noise for full ramp
    n = spx_param.get_num_samples()
    read_coeff = math.sqrt(6 * (n - 1) / (n * (n + 1)))
    readnoise_n = np.full((rp.Jactive + rp.jref, rp.Kactive + rp.kref),
                          read_coeff * noise_sigma_CDS,
                          dtype=np.float32)

    # Mock input calibration as FITS.
    fits_content = [
        fits.PrimaryHDU(),
        fits.ImageHDU(
            data=readnoise_1,
            header=fits.Header({'EXTNAME': ReadNoiseParameters.single, 'BUNIT': 'electron'})),
        fits.ImageHDU(
            data=readnoise_n,
            header=fits.Header({'EXTNAME': ReadNoiseParameters.full, 'BUNIT': 'electron'})),
    ]
    return ReadNoiseParameters(data=fits.HDUList(fits_content), detector=detector)


def fake_PSF(detector: int = 1, nbr_x_zones: int = 11, nbr_y_zones: int = 11) -> PSFZoneData:
    """Creates a fake Point Spread Function.

    Parameters
    ----------
    detector : `int`, optional
        The detector ID (1-6) to generate a PSF for, by default 1.

    nbr_x_zones : `int`, optional
        The number of PSF x zones, by default 11.

    nbr_y_zones : `int`, optional
        The number of PSF y zones, by default 11.

    Returns
    -------
    `PSFZoneData`
        Fake PSF data structure.
    """
    psf_size = 101  # In oversampled pixels
    psf_oversampling = 10  # In times SPHEREx pixels
    fast_jitter_sigma = 1.6  # Fast jitter (gaussian) in arcsec
    rp = ReadoutParameters()
    npx_x = rp.Jactive
    npx_y = rp.Kactive

    # PSFs are oversampled by factor of 20 for arrays 1-3 and 10 for arrays 4-6.
    downsample_by_array = {  # NOTE why this division for downsample?
        1: int(20/psf_oversampling),
        2: int(20/psf_oversampling),
        3: int(20/psf_oversampling),
        4: int(10/psf_oversampling),
        5: int(10/psf_oversampling),
        6: int(10/psf_oversampling)
    }
    trim_by_array = {1: psf_size, 2: psf_size, 3: psf_size, 4: psf_size, 5: psf_size, 6: psf_size}
    # Use Simulator to generate PSF downsampled and trimmed.
    # Simulator PSFs are arranged in a 3x3 grid.
    if not hasSPHERExSimulator:
        raise ImportError("SPHEREx Simulator is not available")
    psf = PSF(data_filename("psf/simulated_PSF_database.fits"), downsample_by_array=downsample_by_array,
              trim_by_array=trim_by_array)

    # Data structure to hold Simulator 3x3 grid.
    # 0 and 1: zone (0...2)
    # 2 and 3: pixels per PSF
    psf_library = np.zeros((3, 3, psf_size, psf_size))

    # Get fast jitter for this exposure.
    np.random.seed(123)
    fast_jitter = np.random.chisquare(fast_jitter_sigma)

    # Create the grid of PSFs.
    for ii in range(3):
        for jj in range(3):

            # Get the PSF closest to the grid point.
            this_psf = psf.psf(ii*npx_x//3 + npx_x//(3*2), jj*npx_y//3 + npx_y//(3*2), array=detector)

            # Get the platescale and convert jitter from arcsec to pixel units.
            platescale = np.array(psf.psf_platescale)[(psf.psf_array == detector)][0]
            jitter_pixel = fast_jitter / platescale

            this_psf_jitter = gaussian_filter(this_psf, jitter_pixel)

            # Normalize PSF.
            this_psf_jitter = this_psf_jitter / np.nanmax(this_psf_jitter)

            psf_library[ii, jj, :, :] = this_psf_jitter

    # Now interpolate the 3x3 grid library on NxN grid.
    # 0 and 1: zone (0...N-1)
    # 2 and 3: pixels per PSF
    z1 = np.arange(0, 3)  # original coordinates
    z2 = np.arange(0, 3)  # original coordinates
    z1_interp = np.linspace(0, 3-1, nbr_x_zones)  # interpolated coordinates
    z2_interp = np.linspace(0, 3-1, nbr_y_zones)  # interpolated coordinates

    # Inteprolate the PSF 3x3 grid for each of the arrays.
    data_cube = np.zeros((nbr_x_zones * nbr_y_zones, psf_size, psf_size), dtype=np.float32)
    for iy in range(nbr_y_zones):
        for ix in range(nbr_x_zones):
            img_interp = interp_images_2d(imgs=psf_library[:, :, :, :], z1=z1, z2=z2,
                                          z1_interp=z1_interp[ix], z2_interp=z2_interp[iy])

            # Normalize PSF
            img_interp = img_interp / np.nanmax(img_interp)
            data_cube[iy * nbr_x_zones + ix] = img_interp

    # Get centers of the zones (represented in FITS convention).
    zone_x_centers = np.asarray([ii*npx_x/nbr_x_zones + (npx_x/nbr_x_zones + 1)/2
                                 for ii in range(nbr_x_zones)])
    zone_y_centers = np.asarray([ii*npx_y/nbr_y_zones + (npx_y/nbr_y_zones + 1)/2
                                 for ii in range(nbr_y_zones)])

    psf_zone_data = PSFZoneData(data_cube, zone_x_centers, zone_y_centers, psf_oversampling)
    add_detcoord(psf_zone_data)
    return psf_zone_data


def interp_images_2d(imgs: ArrayLike, z1: np.ndarray, z2: np.ndarray, z1_interp: float,
                     z2_interp: float) -> ArrayLike:
    """Interpolates an array of images in two dimensions (4D space).

    Parameters
    ----------
    imgs : `~numpy.typing.ArrayLike`
        An array of NxN images of shape (``z1``, ``z2``, N, N), where ``z1``
        and ``z2`` are the 3rd and 4th dimension (aka number of images).
    z1 : `~numpy.ndarray`
        The axis of 3rd dimension.
    z2 : `~numpy.ndarray`
        The axis of 4th dimension
    z1_interp : `float`
        Where to interpolate in 3rd dimension (relative to ``z1``).
    z2_interp : `float`
        Where to interpolate in 4th dimension (relative to ``z2``).

    Returns
    -------
    `~numpy.typing.ArrayLike`
        The interpolated NxN image.
    """
    # Get points
    n = imgs.shape[2]
    points = (z1, z2, np.arange(n), np.arange(n))

    # Get interpolated points
    xi = np.rollaxis(np.mgrid[:n, :n], 0, 3).reshape((n**2, 2))
    xi = np.c_[np.repeat(z1_interp, n**2), np.repeat(z2_interp, n**2), xi]

    # Interpolate
    img_interp = interpn(points, imgs, xi, method="linear")
    img_interp = img_interp.reshape((n, n))

    return img_interp


def fake_nonlin_inputs(det_id: int) -> NonLinearityParameters:
    """Fake input parameters for Correct Nonlinearity module.

    Parameters
    ----------
    det_id : `int`
        Detector ID.

    Returns
    `NonLinearityParameters`
        Nonlinearity paramters in primary and extensions of fits header data unit list.

    """
    rp = ReadoutParameters()
    # Create data arrays with test values.
    b1_data = b2_data = b3_data = np.zeros((rp.Jactive, rp.Kactive), dtype=np.float32)
    qnl_data = np.full((rp.Jactive, rp.Kactive), 1.2E5, dtype=np.float32)
    qmax_data = np.full((rp.Jactive, rp.Kactive), 1.2E5, dtype=np.float32)

    # Create a Primary HDU.
    primary_hdu = fits.PrimaryHDU()

    # Create HDUs for nonlinearity, transient, and threshold parameters.
    qnl_hdu = fits.ImageHDU(data=qnl_data,
                            header=fits.Header({"EXTNAME": "Q_nl", "BUNIT": "electron"}))
    qmax_hdu = fits.ImageHDU(data=qmax_data,
                             header=fits.Header({"EXTNAME": "Qmax", "BUNIT": "electron"}))
    b1_hdu = fits.ImageHDU(data=b1_data, header=fits.Header({"EXTNAME": "b1"}))
    b2_hdu = fits.ImageHDU(data=b2_data, header=fits.Header({"EXTNAME": "b2"}))
    b3_hdu = fits.ImageHDU(data=b3_data, header=fits.Header({"EXTNAME": "b3"}))

    # Create an HDUList object and add the HDUs.
    hdul = fits.HDUList([primary_hdu, qnl_hdu, qmax_hdu, b1_hdu, b2_hdu, b3_hdu])

    # Instantiate NonLinearityParameters class with the created HDUList.
    non_linearity_params = NonLinearityParameters(hdul, detector=det_id)
    return non_linearity_params


def fake_spectral_wcs(detector_id: int) -> HDUList:
    """Get HDUList with central wavelength and bandwidth for each active pixel
    and lookup table, as defined in "Representations of spectral coordinates in FITS",
    Greisen et al. [2006], section 6.

    Parameters
    ----------
    detector_id : int
        Detector id.
    Returns
    -------

    """
    hdul = spectral_wcs_lookup_data(detector=detector_id, wl_tolerance=1 * u.nm)
    hdul = add_detcoord(hdul)

    return hdul


def fake_binned_calib_data(detector_id: int, num_bins: int = 64) -> Table:
    """Fake binned calibration data output from
       Photometric Calibration module.

    Parameters
    ----------
    detector_id : `int`

    Returns
    -------
    `astropy.table.Table`
        Ballpark values of binned calibration data.
    """
    # Define approximate ranges for `a_bin` based on detector ID
    a_bin_ranges = {
        1: (0.119, 0.122),  # 1 / 8.2867 = ~0.1206
        2: (0.118, 0.120),  # 1 / 8.38949 = ~0.1191
        3: (0.125, 0.128),  # 1 / 7.8966 = ~0.12663
        4: (0.100, 0.103),  # 1 / 9.8674 = ~0.1013
        5: (0.485, 0.490),  # 1 / 2.05169 = ~0.4874
        6: (0.590, 0.600),  # 1 / 1.68217 = ~0.5944
    }

    # Set `a_bin` range based on detector ID
    a_bin_min, a_bin_max = a_bin_ranges.get(detector_id, (0.09, 0.12))

    data = {
        'wl_min': np.random.uniform(0.73, 0.8, num_bins),
        'wl_max': np.random.uniform(0.73, 0.8, num_bins),
        'a_bin': np.random.uniform(a_bin_min, a_bin_max, num_bins),
        'z_bin': np.random.uniform(0.99, 1.01, num_bins),
        'omega_bin': np.random.uniform(8.9e-10, 9.0e-10, num_bins)
    }

    # Sort wl_min and wl_max so that wl_min < wl_max for each row
    data['wl_min'], data['wl_max'] = (np.minimum(data['wl_min'], data['wl_max']),
                                      np.maximum(data['wl_min'], data['wl_max']))

    # Create a sample fake binned calibration data table
    spx_binned_cal = Table(data)
    return spx_binned_cal


def fake_solid_angle_pixel_map(detector_id: int) -> SPHERExImage:
    """Fake solid angle pixel map.

    Parameters
    ----------
    detector_id : `int`
        SPHEREx detector id.

    Returns
    -------
    `SPHERExImage`
        Fake solid angle pixel map.
    """
    rp = ReadoutParameters()
    data = np.full((rp.Jactive, rp.Kactive), 6.15*6.15, dtype=np.float32)
    img = SPHERExImage(data, unit=u.arcsec**2)
    img.setArrayId(detector_id)
    return img


def sband_subset_table():
    data = {
        "El.": [
            "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX", "SPX"
        ],
        "Mnemonic": [
            "ADST1SANG", "ADCSTTMS", "ADSTATE", "ADATTERRX", "ADATTERR", "ADMSUNVX", "ADBFAQ1", "ADGYDQ",
            "ADRW1SDIR", "ADSTCNT", "STMEASSTARMIN", "ADGPSDATST", "TSA1T", "RA_TS_V8_T"
        ],
        "Description": [
            "Tracker 1 Sun Angle",
            "ADCS Time Tag, Milliseconds of Day",
            "Current ADCS State",
            "Attitude Error X",
            "Attitude Error Detected",
            "Measured Sun Vector X",
            "Body Frame Attitude Q1",
            "Gyro Data Quality",
            "Wheel 1 Speed/Direction",
            "ST Tracked Star Count",
            "Star Tracker Number of Measured Stars Too Small",
            "GPS Data Status",
            "SA SENSE1 Temp",
            "RT6, Video8, top"
        ],
        "S/S": [
            "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "ADC", "THM", "PYLD"
        ],
        "Type": [
            "U16", "U32", "D3", "S32", "D1", "S16", "S24", "D2",
            "F64", "U8", "U1", "D8", "U12", "F32"
        ],
        "Packet": [
            "APID0110", "APID0110", "APID0100", "APID0100", "APID0100", "APID0100", "APID0110", "APID0110",
            "APID0110", "APID0110", "APID0110", "APID0100", "APID0130", ""
        ],
        "(byte:bit)": [
            "/0179", "/0016", "/0022:0-2", "/0023", "/0035:0", "/0165", "/1194", "/0855:2-3",
            "/0803", "/0175", "/0451:6", "/0071", "/0056:0-11", ""
        ],
        "Notes": [
            "", "", "", "Control Frame Attitude Error", "", "", "", "Rate_Sensor_Status", "", "",
            "SubAddress 13, AAM", "", "PCM External PRT 17 Sense, 500KOhm", "thermistor"
        ],
        "Units": [
            "", "", "", "deg", "", "", "", "", "RPM", "", "", "", "C", "C"
        ],
        "Value/State": [
            "c0=0.0   c1=57.29578e-4", "", "0/Wait 1/Detumble 2/AcqSun 3/Point 4/DeltaV 5/Earth",
            "c0=0.0   c1=57.29578e-7",
            "0/Clear 1/Error", "c0=0.0   c1=3.2e-5", "c0=0.0   c1=1.25e-7", "0/Off   1/Pos  3/Neg",
            "c0=0.0 c1=9.5493", "0/Good 1/Coarse 2/Bad", "",
            "0/OK 1/BadStat 2/Pos 3/Vel 4/PosJump 5/VelJump 6/SampleDT 7/RawBad 8/TimeRNG 9/PropDT 10/Stale",
            "c0=-246.377961 c1=0.324219724 c2=-0.00010361 c3=1.38713E-07 c4=-7.03566e-11 c5=1.29697E-14", ""
        ],
        "Limits (RL YL YH RH)": [
            "", "",
            "0/Wait/red  1/Detumble/yel 2/AcqSun/yel 5/Earth/yel [1]", "RL=-2.8 YL=-2.3 YH=2.3 RH=2.8 [1]",
            "1/Error/yel [1]", "", "", "", "1/Coarse/yel 2/Bad/red [1]", "", "", "", "",
            "RL=-25.0 YL=-20.0 YH=50.0 RH=55.0 [1]"
        ],
        "Fmt": [
            "", "", "", "", "", "", "", "", "", "", "", "", "", ""
        ],
    }

    sband_mnemonics = Table(data)
    return sband_mnemonics


def fake_xtalk_coupling(detector_id: int) -> SPHERExImage:
    """Fake crosstalk coupling calibration values.

    Parameters
    ----------
    detector_id : `int`
        SPHEREx detector id.

    Returns
    -------
    `SPHERExImage`
        Fake crosstalk coupling values.
    """
    data_shape = (32, 32, 3)
    dummy_array = np.ones(data_shape)
    xtalk_coupling = SPHERExImage(data=dummy_array, unit=u.dimensionless_unscaled)
    xtalk_coupling.setArrayId(detector_id)
    return xtalk_coupling


def sgt_table() -> Table:
    data = {
        "Date_Time": ["2025-086T17:07:40.000", "2025-086T17:08:00.000", "2025-086T17:08:20.000",
                      "2025-086T17:08:40.000", "2025-086T17:09:00.000", "2025-086T17:09:20.000",
                      "2025-086T17:09:40.000", "2025-086T17:10:00.000", "2025-086T17:10:20.000"],
        "Latitude": [-65.290814, -66.443149, -67.588177, -68.724732, -69.851391, -70.96641, -72.067633,
                     -73.152367, -74.217228],
        "Longitude": [355.289568, 354.190719, 352.994041, 351.684482, 350.243966, 348.650673, 346.878129,
                      344.894091, 342.659176],
        "GCD2SAA": [16.359751, 16.987838, 17.612186, 18.23245, 18.848342, 19.459633, 20.066159, 20.667819,
                    21.264587],
        "Height": [674.817131, 675.30008, 675.769354, 676.224387, 676.664635, 677.089582, 677.498731,
                   677.891618, 678.2678]
    }
    sgt = Table(data)
    return sgt
