# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import EstimateFineAstrometryConfig, EstimateFineAstrometryTask
from spx_tests import sample_data
from spx_tasks.estimate_fine_astrometry import apply_aberration_correction

test_root = Path(__file__).parent.parent


def get_sim_img() -> SPHERExImage:
    """Get input simulated SPHEREx level 2 image.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.

    """
    sample_file = test_root / "data/cutout_L2.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file, unit="electron / s")
    # Mock the flags attribute, which is None because the test FITS file
    # doesn't have a FLAGS extension. This is needed for the n_transients calculation.
    if spximg.flags is None:
        spximg.flags = np.zeros(spximg.data.shape, dtype=np.uint32)
    sip = sample_data.get_sample_sip(spximg)
    # Fake initial SIP, all zero.
    spximg.wcs.sip = sip
    spximg.wcs.wcs.ctype = [s + "-SIP" for s in spximg.wcs.wcs.ctype]
    spximg.header["OBSID"] = "2025W15_1A_0075_4"
    return spximg


def get_cat() -> Table:
    """Get input catalog.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.

    """
    cat_file = test_root / "data/catalog_L2.fits"
    with fits.open(cat_file) as hdul:
        cat = Table(hdul[1].data)
    return cat


def validate_poly_coeff(a):
    """Validate SIP coefficients.

    Parameters
    ----------
    a : ndarray
        Representation of polynomial coefficients

    """
    # the array should have 2 dimensions
    if a.ndim != 2:
        raise ValueError("Polynomial coefficients should be represented by 2-d array")
    # the number of elements in each dimension should be the same
    (d1, d2) = a.shape
    if d1 != d2:
        raise ValueError("Polynomial coefficients have invalid shape")
    # validate sparseness
    for (p, q), value in np.ndenumerate(a):
        if (p + q >= d1 or p + q < 2) and value != 0:
            raise ValueError(f"Polynomial coefficients have invalid value "
                             f"{value} at {p},{q}.")


def test_EstimateFineAstrometry():
    # Test Case 1: Successful run
    config = EstimateFineAstrometryConfig()
    config.min_ncalibrators = 100
    config.min_sources = 100
    task = EstimateFineAstrometryTask(config=config)
    spximg = get_sim_img()
    crpix_in = spximg.wcs.wcs.crpix
    cat = get_cat()
    spximg.catalog = cat
    sip = sample_data.get_sample_sip(spximg)
    index_dir = test_root / "data/astronet_indexfiles/"
    # Run task
    output = task.run(spximg, cat, index_dir, sip)
    # out_sip = output.spximg_out.wcs.sip
    # TODO replace when conversion back to SIP
    # assert out_sip is not None
    # validate_poly_coeff(out_sip.a)
    # validate_poly_coeff(out_sip.b)
    assert spximg.data.shape == output.spximg_out.data.shape
    crpix_out = output.spximg_out.wcs.wcs.crpix
    # Check that CRPIX is unchanged.
    assert crpix_in.all() == crpix_out.all()
    assert output.spximg_out.header['HIERARCH OMEGA_MEDIAN'] is not None, "solid angle median is missing."
    assert output.spximg_out.header['HIERARCH OMEGA_STDDEV'] is not None, "solid angle stddev is missing."

    # Check header updates
    header = output.spximg_out.header
    assert "RADESYS" in header, "RADESYS keyword is missing from header."
    assert header["RADESYS"] == "ICRS", f"Expected RADESYS='ICRS', found {header['RADESYS']}"
    assert "EQUINOX" not in header, "EQUINOX keyword should have been removed from header."

    # Test Case 2: Failed run (FINAST != 0)
    # Should return input WCS
    fail_config = EstimateFineAstrometryConfig()
    # Force a failure
    fail_config.min_ncalibrators = 10000
    fail_task = EstimateFineAstrometryTask(config=fail_config)
    fail_spximg = get_sim_img()

    # Save the original WCS for comparison
    # Run task
    original_wcs_header = fail_spximg.wcs.to_header().copy()
    fail_output = fail_task.run(fail_spximg, cat, index_dir, sip)
    output_wcs_header = fail_output.spximg_out.wcs.to_header()

    # Check that the run was not successful
    assert fail_output.spximg_out.header["FINAST"] != 0, "Expected a failed run"
    # Check that the output WCS is the same as the input WCS
    for key in original_wcs_header:
        if isinstance(original_wcs_header[key], float):
            assert np.isclose(original_wcs_header[key], output_wcs_header[key])
        else:
            assert original_wcs_header[key] == output_wcs_header[key]


def test_aberration_correction():
    spximg = get_sim_img()
    cat = get_cat()
    spximg.header["EXPSTART"] = 60200.5  # MJD

    # Inject spacecraft position and velocity headers (in km and km/s)
    spximg.header["X_SC"] = 7000.0  # km
    spximg.header["Y_SC"] = 0.0
    spximg.header["Z_SC"] = 0.0
    spximg.header["VX_SC"] = 0.0  # km/s
    spximg.header["VY_SC"] = 7.5
    spximg.header["VZ_SC"] = 0.0

    # Run the function under test
    corrected = apply_aberration_correction(spximg, cat)

    # Validate output columns exist
    assert "ra_aberr" in corrected.colnames
    assert "dec_aberr" in corrected.colnames


# The SIP Convention for Representing Distortion in FITS Image Headers
# https://ui.adsabs.harvard.edu/abs/2005ASPC..347..491S/abstract
# The distortion is represented by 4th order polinomial fit,
# which would translate into A_ORDER = 4, B_ORDER = 4, 12 of A_p_q, and 12 of B_p_q
# parameters for x and y. (See eq. 2 in the paper.)
# Do we also want the reverse parametrization (AP_1_p and BP_q_p)? This is
# usually represented by higher order (in the example below - 9)
#
# Example SIP parameters:
# A_ORDER =                    4
# B_ORDER =                    4
# A_0_2   = 2.65025220036833E-06
# A_0_3   = 3.84480481275673E-11
# A_0_4   = -1.5527942495722E-16
# A_1_1   = 5.83258771725766E-06
# A_1_2   = 7.51347340152890E-10
# A_1_3   = 1.38226924343502E-14
# A_2_0   = 3.41752717715302E-06
# A_2_1   = 2.28445153682875E-10
# A_2_2   = -6.6713248924751E-15
# A_3_0   = 6.29653031516014E-10
# A_3_1   = 2.56154392549919E-14
# A_4_0   = -1.3269173483521E-20
# B_0_2   = 3.31524086792763E-06
# B_0_3   = 5.53805605479185E-10
# B_0_4   = 3.65483914977568E-15
# B_1_1   = 1.95853772621017E-06
# B_1_2   = 1.08650545013240E-10
# B_1_3   = -5.8576470484154E-15
# B_2_0   = 7.98486261413883E-06
# B_2_1   = 7.46257364779495E-10
# B_2_2   = 2.53382532729581E-14
# B_3_0   = -6.5933570103105E-11
# B_3_1   = -6.5248501389361E-15
# B_4_0   = 1.39013897660421E-21
# SIPMXERR= 1.55366567488992E-07 / Max diff from GWCS (equiv pix).
# SIPIVERR=   0.2624046558616521 / Max diff for inverse (pixels)
# AP_0_1  = 5.20028036889805E-06
# AP_0_2  = -2.6143384695884E-06
# AP_0_3  = -1.4190331999932E-12
# AP_0_4  = 3.02792158781354E-15
# AP_0_5  = 2.57962736738238E-23
# AP_0_6  = -1.3462931699925E-33
# AP_0_7  = 4.52168416128143E-42
# AP_0_8  = -1.6882629176404E-52
# AP_0_9  = 4.81333594531297E-61
# AP_1_0  = 0.000230542268848650
# AP_1_1  = -5.8191461764361E-06
# AP_1_2  = -6.3835136569126E-10
# AP_1_3  = 5.64074396399449E-15
# AP_1_4  = 2.99408151789023E-23
# AP_1_5  = 3.68231025960678E-33
# AP_1_6  = 5.70864561403744E-42
# AP_1_7  = 7.87955317634650E-52
# AP_1_8  = 6.52371214324070E-61
# AP_2_0  = -3.2934155707025E-06
# AP_2_1  = -1.1224308662762E-10
# AP_2_2  = 2.61954328530305E-14
# AP_2_3  = -4.9638186488761E-25
# AP_2_4  = 3.14547958466288E-33
# AP_2_5  =  8.0398598108575E-44
# AP_2_6  = 1.97910128906322E-52
# AP_2_7  = 1.11119644656348E-62
# AP_3_0  = -5.6519397012637E-10
# AP_3_1  = 4.43307476087560E-15
# AP_3_2  = -2.3388571141437E-25
# AP_3_3  = 1.08059278660225E-33
# AP_3_4  = 7.92296732637393E-44
# AP_3_5  = 1.10006575380117E-52
# AP_3_6  = 1.47269831180305E-62
# AP_4_0  = -1.3525869954857E-18
# AP_4_1  = -8.6635130839439E-26
# AP_4_2  = 2.91798369146160E-35
# AP_4_3  = -2.0855869958674E-44
# AP_4_4  = 2.17031943858858E-54
# AP_4_5  = -1.8344273835126E-63
# AP_5_0  = 7.52346320372043E-29
# AP_5_1  = 5.28876655431726E-36
# AP_5_2  = -9.0781503426317E-47
# AP_5_3  = 1.22582280291444E-54
# AP_5_4  = 9.52029001226056E-65
# AP_6_0  = -2.4178733356598E-39
# AP_6_1  = -8.3728479953505E-47
# AP_6_2  = 1.90838889213620E-56
# AP_6_3  = -1.9121686578878E-65
# AP_7_0  = 1.10211556231479E-49
# AP_7_1  = 3.87981008444097E-57
# AP_7_2  = 3.00477417012293E-68
# AP_8_0  = -1.2264729522130E-57
# AP_8_1  = -5.5016253501508E-68
# AP_9_0  = 7.81042134646028E-66
# BP_0_1  = -9.4311182950939E-06
# BP_0_2  = -3.3331316271362E-06
# BP_0_3  = -5.0849416364687E-10
# BP_0_4  = 1.26816356578127E-14
# BP_0_5  = -7.4969272891167E-24
# BP_0_6  = 8.01229164420434E-34
# BP_0_7  = -2.6629327621735E-42
# BP_0_8  = -2.4850371712756E-52
# BP_0_9  = -4.8723176001879E-61
# BP_1_0  = 0.000444386548254929
# BP_1_1  = -1.9065344205615E-06
# BP_1_2  = -4.2400195322002E-11
# BP_1_3  = 1.45329970270047E-14
# BP_1_4  = 7.54334623906621E-23
# BP_1_5  = 1.32606232405881E-32
# BP_1_6  = 1.43152387772277E-41
# BP_1_7  = 2.62296372258263E-51
# BP_1_8  = 1.65778894101597E-60
# BP_2_0  = -7.8373420017531E-06
# BP_2_1  = -5.7266321997763E-10
# BP_2_2  = 1.30713866401503E-14
# BP_2_3  = 1.00320151456150E-24
# BP_2_4  = -4.7819416809406E-34
# BP_2_5  = -4.7909658244555E-44
# BP_2_6  = -2.4040914914054E-52
# BP_2_7  = -3.2032539953836E-62
# BP_3_0  = 9.61650104459101E-11
# BP_3_1  = 1.20736996444320E-14
# BP_3_2  = -3.2814502648757E-25
# BP_3_3  = 2.93885817764721E-33
# BP_3_4  = 2.52693555958042E-43
# BP_3_5  =  3.0280202047959E-52
# BP_3_6  = 4.24953175093384E-62
# BP_4_0  = -3.8349830103054E-18
# BP_4_1  = -2.3797985319665E-25
# BP_4_2  = 1.60360843850797E-35
# BP_4_3  = -5.5100248317261E-44
# BP_4_4  = -3.3093292698562E-54
# BP_4_5  = -5.2950117129855E-63
# BP_5_0  = 2.01753612387034E-28
# BP_5_1  = 1.44119348749468E-35
# BP_5_2  = -1.0369599746402E-46
# BP_5_3  = 3.34458426571175E-54
# BP_5_4  = 2.72775058659568E-64
# BP_6_0  = -6.7647187172386E-39
# BP_6_1  = -2.3095825612971E-46
# BP_6_2  = 7.95063507165357E-57
# BP_6_3  = -5.1387291034077E-65
# BP_7_0  = 3.58104292057522E-49
# BP_7_1  = 1.05707974863746E-56
# BP_7_2  = 9.13715649564130E-68
# BP_8_0  = -1.7478805260934E-57
# BP_8_1  = -1.5248396453707E-67
# BP_9_0  = 1.33392269332861E-65
# AP_ORDER=                    9
# BP_ORDER=                    9
