# Standard library imports

# Related third party imports
import numpy as np
from astropy.io import fits

# Local application/library specific imports
from spx_data import NonLinearityParameters, ReadoutParameters, Flag
from spx_tasks import CorrectNonLinearityConfig, CorrectNonLinearityTask
from spx_tests import sample_data
from spx_utils import l1_utils

def sample_inputs(shape: tuple[int, int]) -> NonLinearityParameters:
    """Create test inputs using Q_nl (nonlinearity parameter) and Qmax.
    Optional transient coefficients b1, b2, b3 are also included.
    """
    q_nl_data = np.full(shape, 40000.0)  # Arbitrary Q_nl values
    qmax_data = np.full(shape, 20000.0)  # Threshold at 20k e-

    b1_data = np.full(shape, 0.01)  # Small transient offset terms
    b2_data = np.zeros(shape)
    b3_data = np.zeros(shape)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header['DETECTOR'] = '1'

    qnl_hdu = fits.ImageHDU(data=q_nl_data, header=fits.Header({'EXTNAME': 'Q_nl'}))
    qmax_hdu = fits.ImageHDU(data=qmax_data, header=fits.Header({'EXTNAME': 'Qmax'}))

    b1_hdu = fits.ImageHDU(data=b1_data, header=fits.Header({'EXTNAME': 'b1'}))
    b2_hdu = fits.ImageHDU(data=b2_data, header=fits.Header({'EXTNAME': 'b2'}))
    b3_hdu = fits.ImageHDU(data=b3_data, header=fits.Header({'EXTNAME': 'b3'}))

    hdul = fits.HDUList([primary_hdu, qnl_hdu, qmax_hdu, b1_hdu, b2_hdu, b3_hdu])
    return NonLinearityParameters(hdul)


def sample_inputs_nonzero_transient(shape):
    """Create test inputs using Q_nl (nonlinearity parameter) and Qmax.
    Optional transient coefficients b1, b2, b3 are also included.
    """
    q_nl_data = np.full(shape, 40000.0)
    qmax_data = np.full(shape, 1.0e9)   # keep well above accumulated charge

    # Make all three nonzero so the transient branch is eligible
    b1_data = np.full(shape, 0.01)
    b2_data = np.full(shape, 1.0e-4)
    b3_data = np.full(shape, 1.0e-6)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DETECTOR"] = "1"

    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(data=q_nl_data, header=fits.Header({"EXTNAME": "Q_nl"})),
        fits.ImageHDU(data=qmax_data, header=fits.Header({"EXTNAME": "Qmax"})),
        fits.ImageHDU(data=b1_data, header=fits.Header({"EXTNAME": "b1"})),
        fits.ImageHDU(data=b2_data, header=fits.Header({"EXTNAME": "b2"})),
        fits.ImageHDU(data=b3_data, header=fits.Header({"EXTNAME": "b3"})),
    ])
    return NonLinearityParameters(hdul)


def make_nonlin_inputs(shape, b1=0.0, b2=0.0, b3=0.0):
    q_nl_data = np.full(shape, 40000.0)
    qmax_data = np.full(shape, 1.0e9)

    primary_hdu = fits.PrimaryHDU()
    primary_hdu.header["DETECTOR"] = "1"

    hdul = fits.HDUList([
        primary_hdu,
        fits.ImageHDU(data=q_nl_data, header=fits.Header({"EXTNAME": "Q_nl"})),
        fits.ImageHDU(data=qmax_data, header=fits.Header({"EXTNAME": "Qmax"})),
        fits.ImageHDU(data=np.full(shape, b1), header=fits.Header({"EXTNAME": "b1"})),
        fits.ImageHDU(data=np.full(shape, b2), header=fits.Header({"EXTNAME": "b2"})),
        fits.ImageHDU(data=np.full(shape, b3), header=fits.Header({"EXTNAME": "b3"})),
    ])
    return NonLinearityParameters(hdul)


def test_CorrectNonLinearity():
    config = CorrectNonLinearityConfig()
    task = CorrectNonLinearityTask(config=config)

    spximg_unmod = sample_data.get_sample_level1b_img(bad_flags=True)
    spximg = sample_data.get_sample_level1b_img(bad_flags=True)
    spximg.header["TCUT"] = 0.0

    rp = ReadoutParameters.from_metadata(spximg.meta)
    shape = (rp.Jactive, rp.Kactive)

    non_linearity_params = sample_inputs(shape)
    gain_factors = sample_data.get_sample_gain_factors(spximg)

    output = task.run(spximg=spximg, nonlin_pars=non_linearity_params, gain_factors=gain_factors)

    assert spximg_unmod.data.shape == output.spximg_out.data.shape

    # Accounting check:
    # all active pixels must be either corrected, flagged NONLINEAR, or marked SUR_ERROR
    actv_area = l1_utils.active_area_range(spximg)
    flags = output.spximg_out.flags[actv_area]

    num_actv_pix = flags.size
    num_corrected = output.num_pix_corr
    num_nonlinear = np.sum((flags & Flag.NONLINEAR) > 0)
    num_sur_error = np.sum((flags & Flag.SUR_ERROR) > 0)

    total_accounted = num_corrected + num_nonlinear + num_sur_error

    assert total_accounted == num_actv_pix, (
        f"{num_actv_pix - total_accounted} active pixels were unaccounted for!"
    )


def test_CorrectNonLinearity_transient_polynomial_path():
    config = CorrectNonLinearityConfig()
    task = CorrectNonLinearityTask(config=config)

    spximg_base = sample_data.get_sample_level1b_img(bad_flags=True)
    spximg_base.header["TCUT"] = 0.0

    rp = ReadoutParameters.from_metadata(spximg_base.meta)
    shape = (rp.Jactive, rp.Kactive)

    gain_factors = sample_data.get_sample_gain_factors(spximg_base)

    actv_area = l1_utils.active_area_range(spximg_base)
    actv_data = spximg_base.data[actv_area].copy()
    actv_flags = spximg_base.flags[actv_area].copy()

    y, x = 10, 10
    actv_flags[y, x] = 0
    actv_flags[y, x] |= Flag.TRANSIENT
    actv_data[y, x] = 1.0

    spximg_base.data[actv_area] = actv_data
    spximg_base.flags[actv_area] = actv_flags

    # Clone identical images for the two runs
    spximg_zero = spximg_base.copy()
    spximg_nonzero = spximg_base.copy()

    pars_zero = make_nonlin_inputs(shape, b1=0.0, b2=0.0, b3=0.0)
    pars_nonzero = make_nonlin_inputs(shape, b1=0.01, b2=1.0e-4, b3=1.0e-6)

    out_zero = task.run(
        spximg=spximg_zero,
        nonlin_pars=pars_zero,
        gain_factors=gain_factors,
    )

    out_nonzero = task.run(
        spximg=spximg_nonzero,
        nonlin_pars=pars_nonzero,
        gain_factors=gain_factors,
    )

    value_zero = out_zero.spximg_out.data[actv_area][y, x]
    value_nonzero = out_nonzero.spximg_out.data[actv_area][y, x]
    extra_delta = value_nonzero - value_zero
    surlim = spximg_base.header["SURLIM"]
    tcut = spximg_base.header.get("TCUT", 0.0)
    n_trans = (surlim + tcut) / 2.0

    expected_offset = 0.01 * n_trans + 1.0e-4 * n_trans**2 + 1.0e-6 * n_trans**3
    print("SURLIM =", spximg_base.header["SURLIM"])
    print("TCUT =", spximg_base.header.get("TCUT", 0.0))
    print("n_trans =", n_trans)
    print("value_zero =", value_zero)
    print("value_nonzero =", value_nonzero)
    print("extra_delta =", extra_delta)
    assert np.isclose(extra_delta, expected_offset, rtol=1e-6, atol=1e-6)

