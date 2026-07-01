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
    """
    Create test inputs using Q_nl (nonlinearity parameter) and Qmax.
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
