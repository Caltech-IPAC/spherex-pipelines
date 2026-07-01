from spx_data import SUR
from spx_tests import get_sample_sur


def test_sur():
    sur_in = get_sample_sur()
    hdu = sur_in.to_hdu('SUR_DATA')
    sur_out = SUR.from_hdu(hdu)
    assert sur_in == sur_out
