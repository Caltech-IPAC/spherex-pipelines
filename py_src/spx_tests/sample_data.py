__all__ = ["sample_psf", "get_sample_psf", "get_sample_ssocat",
           "get_sample_sur", "get_sample_zodi", "get_sample_sip",
           "get_sample_hdulist", "get_sample_spx_img", "get_sample_background",
           "get_sample_pyarrow_table", "get_sample_gain_factors",
           "get_sample_level1_img", "get_sample_level1b_img", "get_sample_persistence_img",
           "get_sample_read_noise_map",
           "get_5x5sample_spectral_wcs", "get_6x6sample_spectral_wcs",
           "get_sample_nonlinearity_parameters",
           "get_sample_planned_obs_data",
           "get_sample_image_metadata", "get_sample_image_metadata_stats", "add_gradient",
           "get_sample_global_bkgd", "get_sample_pix_map", "get_fake_nonfunc_v2_map"]

from typing import Optional
from datetime import datetime, timedelta

# Related third party imports
import astropy.units as u
from astropy.coordinates import Angle
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import Sip, WCS
from photutils.background import Background2D, MedianBackground
from photutils.datasets import make_noise_image
import pyarrow as pa
import numpy as np
from numpy.typing import ArrayLike
from lsst.daf.butler import Timespan

from spx_data import (SPHERExImage, SPSFlag, ImageMetadata, ImageMetadataStats,
                      PSFZoneData, SUR, Zodi, Flag, BackgroundMesh, spx_param,
                      GainFactors, ReadoutParameters, NonLinearityParameters, ReadNoiseParameters,
                      PersistenceState, PersistenceStateModelPars, NonfuncData,
                      set_metadata_provenance)
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_utils import l1_utils, spectral_wcs_tab_header


def get_sample_hdulist() -> fits.HDUList:
    """Return a sample FITS HDUList for testing."""
    hdu1 = fits.PrimaryHDU()
    hdu2 = fits.ImageHDU(data=np.zeros([5, 5]))
    hdulist = fits.HDUList([hdu1, hdu2])
    return hdulist


def get_sample_background(shape=None, box_size=None) -> BackgroundMesh:
    """Create a sample background with a gradient.

    Parameters
    ----------
    shape : tuple, optional
        The shape of the background array to generate, by default None.
    box_size : tuple, optional
        The box size for subregions to calculate background, by default None.

    Returns
    -------
    bkg : `~spx_data.BackgroundMesh`
        A `~spx_data.BackgroundMesh` object with all the background information

    """
    if shape is None:
        shape = (50, 50)
    if box_size is None:
        box_size = (5, 5)
    # Generate gradient
    ny, nx = shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / (shape[0] * 10.)
    # Create image with background of mean 10
    img = make_noise_image(shape, distribution="gaussian", mean=10., stddev=5.)
    img = img + gradient

    spximg = get_sample_spx_img(data=img)
    bkg = Background2D(data=spximg, box_size=box_size,
                       bkg_estimator=MedianBackground())
    bkg = BackgroundMesh.from_background2d(bkg)
    return bkg


def get_sample_spx_img(data=None, shape=None):
    """Return a sample SPHERExImage with default layers for testing."""
    if shape is None:
        shape = (5, 5)
    if data is None:
        data = np.zeros(shape, dtype=np.float32)
    spx_img = SPHERExImage(data, unit="adu")
    # Build flags
    flags = np.zeros(spx_img.data.shape, dtype=np.int32)
    flags[0, 0] = flags[2, 2] = Flag.OVERFLOW
    mask_saturated = flags & Flag.OVERFLOW > 0
    flags[2, 2] |= Flag.TRANSIENT
    flags[1, 1] = Flag.TRANSIENT
    mask_cosmic_ray = flags & Flag.TRANSIENT > 0
    spx_img.setFlagsFromMask(mask_saturated, Flag.OVERFLOW)
    spx_img.setFlagsFromMask(mask_cosmic_ray, Flag.TRANSIENT)

    # Add additional layers
    psf = get_sample_psf()
    spx_img.psfcube = psf
    ssocat = get_sample_ssocat(asteroids_only=True)  # exclude None values/object dtype from GPs
    spx_img.ssocat = ssocat
    sur = get_sample_sur()
    spx_img.sur = sur
    zodi = get_sample_zodi(spx_img.data.shape)
    spx_img.zodi = zodi
    wcs = get_sample_wcs()
    spx_img.wcs = wcs
    sip = get_sample_sip(spx_img)
    spx_img.sip = sip
    spx_img.wcs.sip = sip
    catalog = get_sample_catalog()
    spx_img.catalog = catalog

    # Add in Sip headers
    spx_img.wcs.wcs.ctype = ["-SIP", "-SIP"]
    spx_img.wcs.wcs.crpix = spx_img.wcs.sip.crpix

    return spx_img


def get_sample_wcs():
    return WCS()


def sample_psf(nx: int = 5, ny: int = 5, sigma: float = 1.0) -> ArrayLike:
    """Return test PSF as 2d array.

    Parameters
    ----------
    nx : `int`
        Number of pixels along X will be (2*nx+1).
    ny : `int`
        Number of pixels along Y will be (2*ny+1).
    sigma : `float`
        Parameter controlling PSF size.

    Returns
    -------
    `~numpy.ndarray`
        Array of shape [(2*``ny``+1), (2*``nx``+1)] with sample PSF.

    """
    # Create the data
    x = np.arange(-nx, nx + 1)
    y = np.arange(-ny, ny + 1)
    xv, yv = np.meshgrid(x, y)
    r = np.sqrt(xv ** 2 + yv ** 2)
    z = np.exp(-r / (2 * sigma * sigma))
    z = z / z.sum()

    return z


def get_sample_psf() -> PSFZoneData:
    """Return test PSF object.

    Returns
    -------
    `~spx_data.PSFZoneData`
        PSF object.

    """
    npx = 484  # Total pixel size
    nbr_x_zones = 11
    nbr_y_zones = 11
    # Calculate centers of PSF
    zone_x_centers = np.asarray([ii*npx/nbr_x_zones + (npx/nbr_x_zones + 1)/2 for ii in range(nbr_x_zones)])
    zone_y_centers = np.asarray([ii*npx/nbr_y_zones + (npx/nbr_y_zones + 1)/2 for ii in range(nbr_y_zones)])

    nx = ny = 5  # PSF size in high-res pixels will be (2*nx+1)x(2*ny+1)
    psf2d_arr = []
    for n in range(121):
        psf2d = sample_psf(nx=nx, ny=ny, sigma=1.0 + 0.1 * n)
        psf2d_arr.append(psf2d)
    psf = np.stack(psf2d_arr)
    return PSFZoneData(data=psf.astype(np.float32), x_centers=zone_x_centers, y_centers=zone_y_centers)


def get_sample_sur() -> SUR:
    """Return test SUR object.

    Returns
    -------
    sur : `~spx_data.SUR`

    """
    nsamp = 10  # number of samples
    shape = [2, 3]
    # fake data: adding one electron per sample
    sur2d_arr = []
    for n in range(nsamp):
        sample_data = np.full(shape, n * 1.1)
        sample_data[0, 0] = n + 1
        sur2d_arr.append(sample_data)
    sur_data = np.stack(sur2d_arr)
    return SUR(data=sur_data.astype(np.float64), row_start=0, col_sectn=1)


def get_sample_zodi(shape) -> Zodi:
    """Return a sample Zodi object for testing."""
    zodi_data = np.fromfunction(lambda x, y: 10 * x + y, shape)
    return Zodi(zodi_data, "Fake zodi data")


def get_sample_ssocat(asteroids_only=False) -> Table:
    """Generate sample Solar System Object catalog for photometry testing.

    Parameters
    ----------
    asteroids_only : `bool`, optional
        If True, only include asteroids in the catalog and avoid mixed dtype
        object columns. Default is False, which includes both asteroids and
        giant planets (have None for some quantities).

    Returns
    -------
    ssocat : `astropy.table.Table`
        Contains 2 giant planets and 3 asteroids, with only 2
        asteroids positioned in photometry unit test image boundaries.
    """
    non_gp_orb_data = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    gp_orb_data = np.array([None, None])
    orb_data = np.concatenate([non_gp_orb_data, gp_orb_data])
    ssocat = Table({
        "H": orb_data,
        "G": orb_data,
        "dmax": orb_data,
        "alpha": orb_data,
        "rh": orb_data,
        "ro": orb_data,
        "V": orb_data,
        "Tmax": orb_data,
        "AB5_brightest": orb_data,
        "RA": [350.000, 354.305, 354.295, 354.310, 354.300],
        "DEC": [10.000, 12.874, 12.876, 12.878, 12.880],
        "desig": ["1234", "5678", "9101", "saturn barycenter", "phoebe"]
    })
    ssocat.meta["user_metadata"] = {
        "version": SSDC_PIPELINES_VERSION
    }

    if asteroids_only:
        ssocat = ssocat[0:3]  # only asteroids (GPs have None values)
        for col in ssocat.colnames:
            if ssocat[col].dtype == object:
                ssocat[col] = ssocat[col].astype("float64")  # remove object dtype associated w/ None values

    return ssocat


def get_sample_catalog() -> Table:
    names = ["a", "b"]
    rows = [["a1", "a2"], ["b1", "b2"]]
    return Table(rows=rows, names=names)


def get_sample_sip(img) -> Sip:
    """Return a sample SIP distortion object for testing."""
    # A 4th order polynomial requires a 5x5 array for coefficients.
    shape = (5, 5)
    # Forward corefficients
    a = np.zeros(shape, dtype=np.float64)
    b = np.zeros(shape, dtype=np.float64)
    # Inverse coefficients.
    ap = np.zeros(shape, dtype=np.float64)
    bp = np.zeros(shape, dtype=np.float64)
    # CRPIX from image header
    imghdr = img.wcs.to_header(relax=True)
    crpix = np.array([imghdr["CRPIX1"], imghdr["CRPIX2"]])
    return Sip(a, b, ap, bp, crpix)


def get_fake_nonfunc_v2_map(detector: int = 1, seed: int | None = 42) -> NonfuncData:
    """Return fake Nonfunctional v2 product as a NonfuncData instance.

    Builds a multi-extension FITS with:
      - COMBINED (1=bad, 0=good)
      - DARKCURR, HIGHDRIFT, SPECCAL, TRANSIENT, OVERFLOW, OPTEFF, OOS
        (each 1=bad, 0=good)

    Parameters
    ----------
    detector : `int`
        Detector ID written into the PrimaryHDU header.
    seed : `int` or None, optional
        Random seed for reproducibility. Default is 42.

    Returns
    -------
    NonfuncData
        Parsed wrapper around the generated HDUList.
    """
    rng = np.random.default_rng(seed)

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
    opteff = sparse_mask(400)
    oos = sparse_mask(500)

    # Add a couple structured features for easy QA
    np.fill_diagonal(speccal, 1)
    box = 24
    darkcurr[:box, :box] |= 1

    category_planes = {
        "DARKCURR": darkcurr,
        "HIGHDRIFT": highdrift,
        "SPECCAL": speccal,
        "TRANSIENT": transient,
        "OVERFLOW": overflow,
        "OPTEFF": opteff,
        "OOS": oos,
    }

    combined = np.zeros((ny, nx), dtype=np.uint8)
    for mask in category_planes.values():
        combined |= mask

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DETECTOR"] = detector
    primary_hdu.header["BPCONV"] = (1, "Bad-pixel convention: 1=bad, 0=good")

    hdus = [
        primary_hdu,
        fits.ImageHDU(data=combined, header=fits.Header({"EXTNAME": "COMBINED"})),
    ]

    for extname, mask in category_planes.items():
        hdus.append(
            fits.ImageHDU(data=mask, header=fits.Header({"EXTNAME": extname}))
        )

    hdul = fits.HDUList(hdus)

    # Integrity check: COMBINED should be the union of category planes.
    union = np.zeros((ny, nx), dtype=bool)
    for mask in category_planes.values():
        union |= mask.astype(bool)

    cmb = combined.astype(bool)
    if not np.array_equal(cmb, union):
        raise ValueError("COMBINED is not the union of category planes.")

    return NonfuncData(hdul, detector=detector)


def get_sample_nonlinearity_parameters() -> NonLinearityParameters:
    """Return sample non-linearity correction parameters for testing.

    Notes
    -----
    Q_nl must be large enough that the discriminant in the quadratic formula
    is non-negative: qnl**2 - 4 * qover * qnl >= 0, i.e. qnl >= 4 * qover.
    With overflow threshold KOVR_A=1000 and potential gain variations,
    qover could be higher than expected, so we use Q_nl in [50000, 100000]
    to ensure valid discriminants with plenty of margin.
    """
    rng = np.random.default_rng(seed=42)
    # Q_nl needs to be >= 4 * overflow_charge for valid sqrt discriminant
    # Using large values to account for any gain variations
    q_nl = rng.integers(50000, 100001, size=(28, 28)).astype(np.float32)
    # Qmax should be larger than typical accumulated charge
    q_max = rng.integers(80000, 150001, size=(28, 28)).astype(np.float32)
    # Transient correction coefficients (small values)
    b1 = rng.integers(1, 51, size=(28, 28)).astype(np.float32) * 1e-6
    b2 = rng.integers(1, 51, size=(28, 28)).astype(np.float32) * 1e-9
    b3 = rng.integers(1, 51, size=(28, 28)).astype(np.float32) * 1e-12

    fits_content = [
        fits.PrimaryHDU(
            header=fits.Header({"DETECTOR": 1})),
        fits.ImageHDU(
            data=q_nl,
            header=fits.Header({'EXTNAME': 'Q_nl'})),
        fits.ImageHDU(
            data=q_max,
            header=fits.Header({'EXTNAME': 'Qmax'})),
        fits.ImageHDU(
            data=b1,
            header=fits.Header({'EXTNAME': 'b1'})),
        fits.ImageHDU(
            data=b2,
            header=fits.Header({'EXTNAME': 'b2'})),
        fits.ImageHDU(
            data=b3,
            header=fits.Header({'EXTNAME': 'b3'})),
    ]

    return NonLinearityParameters(data=fits.HDUList(fits_content))


def get_sample_persistence_state(spximg: SPHERExImage | None = None) -> PersistenceState:
    """Return fake persistence state for pipeline (mutable)."""
    if spximg is not None:
        rp = ReadoutParameters.from_metadata(spximg.meta)
    else:
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
    p_state = PersistenceState(hdulist=fits.HDUList(fits_content),
                               detector=1, num_prev=0, prev1_end_time=0.0, prev2_end_time=0.0)
    return p_state


def get_sample_planned_obs_data() -> dict:
    """Return sample data to construct planned observation for testing.

    The obs_id is generated based on the current time in SPS format
    (e.g., "2025W52_2A_0923_1") to allow extraction of spsplan_id.
    """
    now = datetime.now()
    begin_time = Time(now)
    end_time = begin_time + timedelta(seconds=1)

    # Generate realistic obs_id based on current time
    # Long format:
    #   {year}W{week}_{plan_number}{plan_version}_{large_slew}_{small_slew}_{total}
    year = now.year
    week = now.isocalendar()[1]  # ISO week number
    plan_number = 2  # Day of week indicator (1 or 2)
    plan_version = "A"
    large_slew = 486  # Example slew group
    small_slew = 1  # Small slew counter
    total_slews = 4  # Total number of pointings per small slew group (informational)
    obs_id = f"{year}W{week:02d}_{plan_number}{plan_version}_{large_slew:04d}_{small_slew}_{total_slews}"

    planned_obs_data = {
        'ra': Angle(23, unit="deg"),  # Angle object
        'dec': Angle(-78, unit="deg"),  # Angle object
        'pa_eq': Angle(10, unit="deg"),  # Angle object
        'lat_ecl': 60.2,
        'lon_ecl': 17.7,
        'pa_ecl': 180.0,
        'timespan': Timespan(begin=begin_time, end=end_time),  # Timespan
        'obs_id': obs_id,
        'target_id': 789,
        'flags': SPSFlag.ALL_SKY | SPSFlag.SAA,  # Make sure this matches expected bitwise flags
        'quaternions': {'q1': 0.1, 'q2': 0.2, 'q3': 0.3, 'q4': 0.4},
        'zenith': [1.1, 1.2, 1.3],
        'moon': [2.1, 2.2, 2.3],
        'sun': [3.1, 3.2, 3.3],
        'vel': [4.1, 4.2, 4.3],
        'in_sband': False,
        'non_survey': True
    }
    return planned_obs_data


def get_sample_planned_obs_data_with_target_name() -> dict:
    """Return sample data with optional target name fields for testing."""
    data = get_sample_planned_obs_data()
    data['regularized_target_id'] = 789
    data['target_name'] = 'SPS_000789_S12'
    return data


def get_sample_image_metadata() -> ImageMetadata:
    """Return sample ImageMetadata for testing."""
    meta2 = ImageMetadata(
        start_time_days=24108,
        start_time_ms=69119379,
        start_time_us=0,
        rob_temp_start=np.array([37.33568, 34.041824, 35.039505, 37.851242,
                                 37.788605, 38.00203, 36.01027, 36.110382], dtype=np.float32),
        fpa_temp_start=np.array([63.00000136, 45.49999942, 62.99976906, 45.433284]),
        ceb_temp_start=np.array([34.931446, 39.01967574, 34.46210695, 36.00192234, 34.36571054,
                                 37.90897632, 37.99996456, 31.56104367, 32.94969356]),
        aux_temp_start=np.array(
            [59.83515974, 49.87638869, 43.71516857, 61.73660837, 59.74546702, 65.65259662,
             45.2871295, 62.62744016, 62.63896953, 45.27447894, 37.9199655, 61.6250975,
             55.51916131, 53.0520222, 53.63350712, 46.55234575, 52.88014164, 51.062733,
             40.51699721, 51.53395171, 53.49895852, 50.82612206, 50.37728299, 54.69907363,
             47.19583471]),
        rob_temp_end=np.array([37.426956, 34.041916, 35.0737, 37.858902, 37.79628, 38.002518, 36.010513,
                               36.293167], dtype=np.float32),
        fpa_temp_end=np.array([62.99999961, 45.49999703, 62.99976062, 45.43327123]),
        ceb_temp_end=np.array([35.02977622, 39.2643305, 34.48618917, 36.00391058, 34.37176214, 37.90949312,
                               38.00184964, 31.52833588, 32.94788733]),
        aux_temp_end=np.array([59.83354113, 49.87784952, 43.71609466, 61.73702459, 59.74642438, 65.65231506,
                               45.28608364, 62.62428728, 62.63913975, 45.27321671, 37.92191231, 61.62734793,
                               55.52131557, 53.05460639, 53.63549411, 46.55359315, 52.8819968, 51.06335502,
                               40.51792377, 51.53571484, 53.50003965, 50.82656448, 50.37872034, 54.70043185,
                               47.19778267]),
        regpix_binwidth=348,
        refpix_binwidth=348,
        phanpix_binwidth=348,
        avg_phanpix_slopes=np.array(
            [-2784515, -3213858, 2331482, 1458013, 703547, 1300229, -3085072, 2552164,
             1559410, 3098360, -2475975, 1556572, 1375126, -6268745, -705576, -196354,
             1620735, -2518086, 2515572, -1229603, -1934537, -139206, 3848779, -3464557,
             303889, -934574, 160601, -1401580, 130672, -1030932, 9082884, -1550933]),
        surlim=74,
        nskip=32,
        nref=4,
        ken=np.array([-15536, 25000, 500]),
        ksn=np.array([-56, 0, 0]),
        kovr=np.array([25000, 25000, 25000]),
        rstdly_surcnt=1,
        surdly_surcnt=3,
        trandly_surcnt=7,
        ovrdly_surcnt=5,
        tcut_surcnt=0,
        raw_row_start=1024,
        raw_col_section=32,
        framet=153500,
        block_size=235
    )
    set_metadata_provenance(meta2, None, observation=202537201048620, detector=1)
    return meta2


def get_sample_image_metadata_stats() -> ImageMetadataStats:
    """Return sample ImageMetadataStats for testing."""
    meta3 = ImageMetadataStats(
        start_time_days=24108,
        start_time_ms=71231401,
        start_time_us=0,
        regpix_flagstats=np.array([4158204, 806, 3, 0, 0, 2555, 30, 2]),
        refpix_flagstats=np.array([81819, 14, 0, 0, 0, 23, 0, 0]),
        phanpix_flagstats=np.array([265215, 0, 0, 0, 0, 1, 0, 0]),
        nbits_stats=np.array([394, 750, 287, 197, 15985, 0, 0, 0]),
        blocklen_stats=np.array([0, 1036, 14461, 801, 339, 551, 425, 0]),
        obswords_stats=1573663,
        msecs_stats=73913,
        block_size=45
    )
    set_metadata_provenance(None, meta3, observation=202537201048620, detector=1)
    return meta3


def get_sample_pyarrow_table() -> pa.Table:
    """Generate a sample `pyarrow.Table` for testing."""
    row = {"flux_bkg": 0, "flux": 10000000, "flux_err": 10.01, "flag": True, "status": "Testing",
           "calibration": [0.1, 0.2, 3, 10]}
    # Create duplicates of row
    rows = []
    for _ in range(3):
        rows.append(row)
    # Create metadata
    metadata = {"flux_bkg": "This is always 0, maybe bug?"}
    # Setup the schema
    lfloat64 = pa.list_(pa.field('element', pa.float64()))
    schema = pa.schema([
        ("flux_bkg", pa.int8()),
        ("flux", pa.int64()),
        ("flux_err", pa.float32()),
        ("flag", pa.bool_()),
        ("status", pa.string()),
        ("calibration", lfloat64)],
        metadata=metadata)
    table = pa.Table.from_pylist(rows, schema=schema)

    return table


def get_sample_gain_factors(spximg: SPHERExImage = None) -> GainFactors:
    """Generate a sample `~spx_data.GainFactors`.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`, optional
        Input spherex image.

    Returns
    -------
    `~spx_data.GainFactors`
        Sample gain factors for each array id.

    """
    if spximg is not None:
        rp = ReadoutParameters.from_metadata(spximg.meta)
    else:
        rp = ReadoutParameters()
    # Instrument gain factors
    # g_fake = [float(0.85) for x in range(rp.nchannel)]
    g_fake = np.ones(rp.nchannel)  # gain factors for simulator.

    gain_factors = GainFactors(gains_1=g_fake, gains_2=g_fake,
                               gains_3=g_fake, gains_4=g_fake,
                               gains_5=g_fake, gains_6=g_fake,
                               num_channels=rp.nchannel)
    return gain_factors


def get_sample_level1_img(rp: Optional[ReadoutParameters] = None) -> SPHERExImage:
    """Generate level 0 image with PHANTOM and REFERENCE flags only.

    Parameters
    ----------
    rp : `~spx_data.ReadoutParameters`, optional
        Readout parameters. If None, uses default test parameters.

    Returns
    -------
    SPHERExImage
        The level 0 image.

    """
    # small test L1 image.
    phan_val = 1
    ref_val = 2
    pix_val = 3

    if rp is None:
        # create an image with 28x28 active pixels and reference pixels.
        rp = ReadoutParameters(nref=2, Mphan=2, nchannel=3, wchannel=10,
                               kref=2, jref=2, Jactive=28, Kactive=28)

    phan_data = np.full((rp.Jphan, rp.Kphan), phan_val)
    phan_flg = np.full(phan_data.shape, Flag.PHANTOM)
    det_data = np.full((rp.Jframe, rp.Kframe-rp.Kphan), pix_val)
    det_flg = np.full(det_data.shape, 0)
    # extra ref rows have higher row indexes
    det_flg[:rp.jref//2, :] = Flag.REFERENCE
    det_flg[rp.jref//2 + rp.Jactive:, :] = Flag.REFERENCE
    det_flg[:, :rp.kref//2] = Flag.REFERENCE
    det_flg[:, -rp.kref//2:] = Flag.REFERENCE
    det_data[det_flg == Flag.REFERENCE] = ref_val

    data = np.hstack((phan_data, det_data)).astype(np.float32)
    flags = np.hstack((phan_flg, det_flg)).astype(np.int16)
    hdr = fits.Header(rp.to_metadata())
    hdr["JD"] = 2460480.499855344
    hdr["TINT"] = 112.5  # total exposure time in seconds
    # Set array ID
    hdr["ARRAY"] = 1
    w = get_sample_wcs()
    # Update wcs
    x = rp.Mphan * rp.nchannel + rp.kref//2 + rp.Kactive/2 + 0.5
    y = rp.jref//2 + rp.Jactive/2 + 0.5
    w.wcs.crpix = [x, y]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = np.repeat(spx_param.get_pix_scale().to(u.deg).value, 2)

    spximg = SPHERExImage(data=data.astype(np.dtype("float32")),
                          flags=flags.astype(np.dtype("int16")),
                          meta=hdr, unit="electron / s",
                          wcs=w)
    # Sample SUR raw data window.
    spximg.sur = get_sample_sur()

    return spximg


def get_sample_level1b_img(seed_number: int | None = 42,
                           bad_flags: bool | None = False,
                           no_refs: bool | None = False) -> SPHERExImage:
    """Generate level 1 image with flags.

    Parameters
    ----------
    seed_number : `int` | `None`, optional
        Seed number for reproducible randomness.
    bad_flags : `bool` | `None`, optional
        Make random missing data and checksum error flags.
    no_refs : `bool` | `None`, optional
        Make LVF image; only active pixel area.

    Returns
    -------
    `~spx_data.SPHERExImage`
        Level 1 image with reference, phantom, and active pixels;
        includes some random transient and overflow flags.

    """
    # small test L2 image.
    phan_val = 1
    ref_val = 2
    pix_val = 3

    # create an image with 28x28 active pixels and reference pixels.
    rp = ReadoutParameters(nref=2, Mphan=2, nchannel=3, wchannel=10, kref=2, jref=2, Jactive=28, Kactive=28)

    det_data = np.full((rp.Jactive + rp.jref * rp.nref, rp.Kactive + rp.kref),
                       pix_val,
                       dtype=np.float32)
    det_flg = np.full(det_data.shape, 0)
    # Phantoms
    phan_data = np.full((rp.Jphan, rp.Mphan * rp.nchannel), phan_val)
    phan_flg = np.full(phan_data.shape, Flag.PHANTOM)
    # References
    det_flg[:rp.jref // 2, :] = Flag.REFERENCE
    det_flg[:, :rp.kref // 2] = Flag.REFERENCE
    det_flg[(rp.jref // 2) * rp.nref - rp.jref // 2 + rp.Jactive:, :] = Flag.REFERENCE
    det_flg[:, rp.kref // 2 + rp.Kactive:] = Flag.REFERENCE
    det_data[det_flg == Flag.REFERENCE] = ref_val

    # Update the function parameters to take a seed number but default 42
    rng = np.random.default_rng(seed_number)

    # Reference pixels
    # A transient and an overflow in bottom.
    row = rng.integers(0, rp.jref // 2)
    col = rng.integers(0, rp.Kactive + rp.kref)
    det_flg[row][col] = Flag.OVERFLOW | Flag.REFERENCE
    row = rng.integers(0, rp.jref // 2)
    col = rng.integers(0, rp.Kactive + rp.kref)
    det_flg[row][col] = Flag.TRANSIENT | Flag.REFERENCE
    # A transient and an overflow in LHS.
    row = rng.integers(0, rp.jref + rp.Jactive)
    col = rng.integers(0, rp.kref // 2)
    det_flg[row][col] = Flag.OVERFLOW | Flag.REFERENCE
    row = rng.integers(0, rp.jref + rp.Jactive)
    col = rng.integers(0, rp.kref // 2)
    det_flg[row][col] = Flag.TRANSIENT | Flag.REFERENCE
    # A transient and an overflow in RHS.
    row = rng.integers(0, rp.jref + rp.Jactive)
    col = rng.integers(rp.kref // 2 + rp.Kactive, rp.kref + rp.Kactive)
    det_flg[row][col] = Flag.OVERFLOW | Flag.REFERENCE
    row = rng.integers(0, rp.jref + rp.Jactive)
    col = rng.integers(rp.kref // 2 + rp.Kactive, rp.kref + rp.Kactive)
    det_flg[row][col] = Flag.TRANSIENT | Flag.REFERENCE
    # A transient and a overflow in top.
    row = rng.integers(rp.jref // 2 + rp.Jactive, rp.jref // 2 * (rp.nref + 1) + rp.Jactive)
    col = rng.integers(0, rp.kref + rp.Kactive)
    det_flg[row][col] = Flag.OVERFLOW | Flag.REFERENCE
    row = rng.integers(rp.jref // 2 + rp.Jactive, rp.jref // 2 * (rp.nref + 1) + rp.Jactive)
    col = rng.integers(0, rp.kref + rp.Kactive)
    det_flg[row][col] = Flag.TRANSIENT | Flag.REFERENCE

    # Active pixels
    for i in range(3):
        row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
        col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
        det_flg[row][col] = Flag.OVERFLOW

    for i in range(3):
        row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
        col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
        det_flg[row][col] = Flag.TRANSIENT

    row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
    col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
    det_flg[row][col] = Flag.OVERFLOW | Flag.TRANSIENT

    if bad_flags:
        row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
        col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
        det_flg[row][col] = Flag.SUR_ERROR

        row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
        col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
        det_flg[row][col] = Flag.SUR_ERROR | Flag.TRANSIENT

        row = rng.integers(rp.jref // 2, rp.Jactive + rp.jref // 2)
        col = rng.integers(rp.kref // 2, rp.Kactive + rp.kref // 2)
        det_flg[row][col] = Flag.MISSING_DATA

        det_data[det_flg & (Flag.MISSING_DATA | Flag.SUR_ERROR) > 0] = np.nan

    # Define SUR region, assume starting row and column section are 0.
    sur_nrows, sur_ncols = (2, 3)
    for i in range(rp.jref // 2, rp.jref // 2 + sur_nrows):
        for j in range(rp.kref // 2, rp.kref // 2 + sur_ncols):
            det_flg[i][j] = Flag.FULLSAMPLE

    # Set a random SURERROR flag in available area.
    row = rng.integers(rp.jref // 2, rp.jref // 2 + sur_nrows)
    col = rng.integers(rp.kref // 2, rp.kref // 2 + sur_ncols)
    det_flg[row][col] = Flag.SUR_MISMATCH

    data = np.hstack((phan_data, det_data)).astype(np.float32)
    flags = np.hstack((phan_flg, det_flg)).astype(np.int32)
    w = get_sample_wcs()

    hdr = fits.Header(rp.to_metadata())
    hdr["MJD-BEG"] = 60480.499855344
    hdr["DATE-BEG"] = "2024-06-19T11:59:47.502"  # ISO date corresponding to MJD-BEG
    hdr["TINT"] = 112.5  # total exposure time in seconds
    hdr["TELAPSE"] = 115.5  # total elapsed time
    hdr["TSAMP"] = 1.5349  # frame time in seconds

    # Sample overflow threshold for reference and active pixels.
    hdr["KOVR_R"] = 1000  # reference
    hdr["KOVR_A"] = 1000  # active

    # Sample SUR metadata information.
    hdr["SURLIM"] = 74  # Total number of SUR frames.
    hdr["SURDLY"] = 3  # First frame when SUR accumulation begins.
    hdr["TRANDLY"] = 7  # SUR count value at which transient detection begins.

    # Set array ID
    hdr["ARRAY"] = 1

    spximg = SPHERExImage(data=data, flags=flags, meta=hdr, unit="electron / s", wcs=w)
    spximg.sur = get_sample_sur()
    # Update wcs
    x = rp.Mphan * rp.nchannel + rp.kref // 2 + rp.Kactive/2 + 0.5
    y = rp.jref // 2 + rp.Jactive / 2 + 0.5
    w.wcs.crpix = [x, y]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.cdelt = np.repeat(spx_param.get_pix_scale().to(u.deg).value, 2)

    # If LVF imaged wanted, trim reference pixels.
    if no_refs is True:
        row_range, col_range = l1_utils.active_area_range(spximg)
        spximg = spximg[row_range.start:row_range.stop, col_range.start:col_range.stop]
    return spximg


def get_sample_read_noise_map(spximg: SPHERExImage | None = None,
                              mu: int = 0,
                              sigma: int = 0.1) -> ReadNoiseParameters:
    """Generate sample per-pixel readout noise map with random values
    sampled from a Gaussian with one sigma equal to 0.1 and centered at 0.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage` or `None`, optional.
        Input spherex image.
    mu : `int`
        Mean of random Gaussian sampling, default is 0.
    sigma : `int`
        Standard deviation of random Gaussian sampling, default is 0.1.

    Returns
    -------
    `~spx_data.SPHERExImage`
        Per-pixel readout noise map with random values.

    Notes
    -----
    - The map will be in the shape of the active area with reference pixels,
      where nref = 1.
    - There will be a separate map for each of the six detectors.
    - Read noise maps are in units of electrons.

    """
    if spximg is not None:
        rp = ReadoutParameters.from_metadata(spximg.meta)
    else:
        rp = ReadoutParameters()

    # Toy map to use on sample L2 image.
    rng = np.random.default_rng(seed=123)

    # Fake read noise for active and reference pixel area.
    var_map_fake = abs(rng.normal(mu, sigma,
                                  size=(rp.Jactive+rp.jref, rp.Kactive+rp.kref)).astype(np.float32))
    # Mock input calibration as Fits.
    fits_content = [
        fits.PrimaryHDU(
            header=fits.Header({"DETECTOR": 1})),
        fits.ImageHDU(
            data=var_map_fake,
            header=fits.Header({'EXTNAME': ReadNoiseParameters.single, 'BUNIT': 'electron'})),
        fits.ImageHDU(
            data=var_map_fake,
            header=fits.Header({'EXTNAME': ReadNoiseParameters.full, 'BUNIT': 'electron'})),
    ]
    return ReadNoiseParameters(data=fits.HDUList(fits_content))


def get_sample_ff(spximg: SPHERExImage) -> SPHERExImage:
    """Make a fake flat field with random values.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`
        Input SHPEREx image to obtain shape template.

    Returns
    -------
    `~spx_data.SPHERExImage`
        Sample flat field as data in a SPHEREx image.

    """
    rng = np.random.default_rng(seed=321)

    ff_vals = rng.uniform(0.9, 1.1, size=spximg.shape)

    if ff_vals.shape != spximg.shape:
        raise ValueError("Input image and sample_ff image shapes are inconsistent.")

    sample_ff = SPHERExImage(data=ff_vals, unit="")
    sample_ff.header["ARRAY"] = 1
    return sample_ff


def get_sample_persistence_img() -> SPHERExImage:
    """Get sample persistence image from l1 image.

    Returns
    -------
    `SPHERExImage`
    Persistence image with data, flags, variance, and nonlinearity correction.
    """
    from spx_tasks import CreateVarianceMapTask, CorrectNonLinearityTask, UpdatePersistenceTask
    spximg = get_sample_level1b_img(bad_flags=True)

    spximg.header["TCUT"] = 70
    spximg.header["TSAMP"] = 1.5349
    spximg.header["SURLIM"] = 77
    spximg.header["TRANDLY"] = 7
    spximg.header["SURDLY"] = 3

    gain_factors = get_sample_gain_factors(spximg)
    read_noise = get_sample_read_noise_map(spximg)
    nonlinear_pars = get_sample_nonlinearity_parameters()

    task = CreateVarianceMapTask()
    result = task.run(spximg, gain_factors, read_noise=read_noise)
    spximg = result.spximg_out

    task = CorrectNonLinearityTask()
    result = task.run(spximg, nonlinear_pars, gain_factors)
    spximg = result.spximg_out

    task = UpdatePersistenceTask()
    result = task.run(spximg, None)
    persistence_image = result.persistence_image

    return persistence_image


def get_sample_l2_img() -> SPHERExImage:
    """Get sample l2 image from l1 image, after variance map made and ref pix
    removed.

    Returns
    -------
    L2 image with data, flags, and variance.

    """
    from spx_tasks import CreateVarianceMapTask
    task = CreateVarianceMapTask()
    # Get sample Level 1b image.
    spximg = get_sample_level1b_img(bad_flags=True)
    spximg.header["TCUT"] = 70
    spximg.header["TSAMP"] = 1.5349
    spximg.header["SURLIM"] = 77
    spximg.header["TRANDLY"] = 7
    spximg.header["SURDLY"] = 3
    # Get sample gain factors for array detector 1.
    gain_factors = get_sample_gain_factors(spximg)
    # Read noise map.
    read_noise = get_sample_read_noise_map(spximg)
    # Run variance map task to check that uncertainty array changes as
    # expected.
    output = task.run(spximg, gain_factors, read_noise=read_noise)

    actv_area = l1_utils.active_area_range(output.spximg_out)
    spximg_out = SPHERExImage(data=output.spximg_out.data[actv_area],
                              flags=output.spximg_out.flags[actv_area],
                              meta=spximg.meta,
                              uncertainty=spximg.uncertainty[actv_area],
                              unit="electron/s", wcs=spximg.wcs)
    return spximg_out


def get_5x5sample_spectral_wcs() -> tuple[fits.HDUList, np.ndarray]:
    """Sample spectral WCS for 5x5 image.

    Returns
    -------
    result : `tuple` [`~astropy.io.fits.HDUList`, `~numpy.ndarray`]
        Tuple of two elements:
        1. spectral wcs python representation
        2. expected values at each point of 5x5 image as int32 numpy array

    """
    index_x = [1, 3, 5]
    index_y = [1, 5]
    coord_data = [[[200, 2000], [160, 1600], [240, 2400]], [[100, 1000], [60, 600], [140, 1000]]]

    expected_vals = np.array([
        [[200, 2000], [180, 1800], [160, 1600], [200, 2000], [240, 2400]],
        [[175, 1750], [155, 1550], [135, 1350], [175, 1550], [215, 2150]],
        [[150, 1500], [130, 1300], [110, 1100], [150, 1500], [210, 2100]],
        [[125, 1250], [105, 1050], [85, 850], [125, 1250], [165, 1650]],
        [[100, 1000], [80, 800], [60, 600], [100, 1000], [140, 1400]],
    ], dtype=np.int32)

    expected_vals_split = np.squeeze(np.split(expected_vals, 2, axis=-1))
    expected_vals1 = expected_vals_split[0]
    expected_vals2 = expected_vals_split[1]

    alt = " "
    hdr = spectral_wcs_tab_header(alt=alt)

    tbl = Table({
        hdr[f"PS1_2{alt}"]: [index_x],
        hdr[f"PS2_2{alt}"]: [index_y],
        hdr[f"PS1_1{alt}"]: [coord_data]})
    # HDU with lookup table
    tbl_hdu = fits.BinTableHDU(data=tbl, name=hdr[f"PS2_0{alt}"])
    # Create an HDUList from the list of HDUs
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(expected_vals1, header=hdr),
                         fits.ImageHDU(expected_vals2, header=hdr),
                         tbl_hdu])
    return hdul, expected_vals


def get_6x6sample_spectral_wcs() -> tuple[fits.HDUList, np.ndarray]:
    """Sample spectral WCS for 6x6 (minimal level 1) image.

    Returns
    -------
    result : `tuple` [`~astropy.io.fits.HDUList`, `~numpy.ndarray`]
        Tuple of two elements:
        1. spectral wcs python representation
        2. expected values at each point of 5x5 image as int32 numpy array

    """
    index_x = [1, 3, 4, 6]
    index_y = [1, 6]
    coord_data = [[[225, 2250], [185, 1850], [185, 1850], [225, 2250]],
                  [[100, 1000], [60, 600], [60, 600], [100, 1000]]]

    expected_vals = np.array([
        [[225, 2250], [205, 2050], [185, 1850], [185, 1850], [205, 2050], [225, 2250]],
        [[200, 2000], [180, 1800], [160, 1600], [160, 1600], [180, 1800], [200, 2000]],
        [[175, 1750], [155, 1550], [135, 1350], [135, 1350], [155, 1550], [175, 1750]],
        [[150, 1500], [130, 1300], [110, 1100], [110, 1100], [130, 1300], [150, 1500]],
        [[125, 1250], [105, 1050], [85, 850], [85, 850], [105, 1050], [125, 1250]],
        [[100, 1000], [80, 800], [60, 600], [60, 600], [80, 800], [100, 1000]],
    ], dtype=np.int32)

    expected_vals_split = np.squeeze(np.split(expected_vals, 2, axis=-1))
    expected_vals1 = expected_vals_split[0]
    expected_vals2 = expected_vals_split[1]

    alt = " "
    hdr = spectral_wcs_tab_header(alt=alt)

    tbl = Table({
        hdr[f"PS1_2{alt}"]: [index_x],
        hdr[f"PS2_2{alt}"]: [index_y],
        hdr[f"PS1_1{alt}"]: [coord_data]})
    # HDU with lookup table
    tbl_hdu = fits.BinTableHDU(data=tbl, name=hdr[f"PS2_0{alt}"])
    # Create an HDUList from the list of HDUs
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(expected_vals1, header=hdr),
                         fits.ImageHDU(expected_vals2, header=hdr),
                         tbl_hdu])
    return hdul, expected_vals


def make_fake_spectral_wcs_l2_img(
    spximg_l2,
    cwave_center_um: float = 1.50,
    cwave_step_x_um: float = 1e-4,
    cwave_step_y_um: float = -7e-5,
    include_dwave: bool = True,
    dwave_um: float = 1e-4,
) -> fits.HDUList:
    """Create a fake spectral_wcs HDUList with a 'CWAVE' image extension.

    This is intended for unit tests where code does:
        spectral_wcs["CWAVE"].data

    Parameters
    ----------
    spximg_l2 : `~spx_data.SPHERExImage`
        L2 image that defines output shape (ny, nx).
    cwave_center_um : `float`, optional
        Central wavelength value at image center (microns).
    cwave_step_x_um : `float`, optional
        Linear gradient in +x (microns/pixel).
    cwave_step_y_um : `float`, optional
        Linear gradient in +y (microns/pixel).
    include_dwave : `bool`, optional
        If True, add a 'DWAVE' extension as well.
    dwave_um : `float`, optional
        Constant delta-wavelength value (microns).

    Returns
    -------
    hdul : `~astropy.io.fits.HDUList`
        HDUList with at least:
          - PrimaryHDU
          - ImageHDU named 'CWAVE' (float32)
          - optionally ImageHDU named 'DWAVE' (float32)

    """
    ny, nx = spximg_l2.data.shape[-2], spximg_l2.data.shape[-1]

    yy, xx = np.mgrid[0:ny, 0:nx]
    x0 = (nx - 1) / 2.0
    y0 = (ny - 1) / 2.0

    # Fake CWAVE map (microns): simple plane
    cwave = (
        cwave_center_um
        + cwave_step_x_um * (xx - x0)
        + cwave_step_y_um * (yy - y0)
    ).astype(np.float32)

    # Minimal header is usually enough for tests
    cwave_hdr = fits.Header()
    cwave_hdr["BUNIT"] = "um"

    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(data=cwave, header=cwave_hdr, name="CWAVE"),
    ]

    if include_dwave:
        dwave_arr = np.full((ny, nx), dwave_um, dtype=np.float32)
        dwave_hdr = fits.Header()
        dwave_hdr["BUNIT"] = "um"
        hdus.append(fits.ImageHDU(data=dwave_arr, header=dwave_hdr, name="DWAVE"))

    return fits.HDUList(hdus)


def add_gradient(data: np.ndarray) -> np.ndarray:
    """Add a gradient going from bottom left to upper right.

    Parameters
    ----------
    data : `numpy.ndarray`
        Data array to add gradient to.

    Returns
    -------
    data_with_gradient : `numpy.ndarray`
        The data array with the gradient added in.

    """
    ny, nx = data.shape
    y, x = np.mgrid[:ny, :nx]
    gradient = x * y / 2500.
    return data + gradient


def get_sample_global_bkgd(data: np.ndarray, null=False) -> np.ndarray:
    """Generate a sample global background estimate.

    Parameters
    ----------
    data : `numpy.ndarray`
        Sample image data.
    null: `bool`, optional
        If True, return a global background estimate of all NaNs. If False,
        return a global background estimate to be subtracted during photometry.
        Default is False.

    Returns
    -------
    bkgd : `numpy.ndarray`
        Global background estimate.

    """
    if not null:
        bkgd = add_gradient(data)
    else:
        bkgd = np.full_like(data, np.nan)
    return bkgd


def get_sample_pix_map(data: np.ndarray) -> SPHERExImage:
    """Generate a sample solid angle pixel map.

    Parameters
    ----------
    data : `numpy.ndarray`
        Sample image data.

    Returns
    -------
    `spx_data.SPHERExImage`
        Solid angle pixel map.

    """
    solid_angle_data = np.ones(data.shape, dtype=np.float32)
    return SPHERExImage(solid_angle_data, unit=u.arcsec**2)


def fake_he_template(spximg: SPHERExImage) -> SPHERExImage:
    """Return fake He I Airglow Template."""
    npix_x, npix_y = spximg.data.shape
    data = np.ones([npix_x, npix_y], dtype=np.float32)/100
    img = SPHERExImage(data, unit=u.Unit("MJy/sr"))
    return img
