# Standard library imports
import unittest

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_utils import l1_utils
from spx_tests import sample_data
from spx_data import ReadoutParameters


class TestL1Utils(unittest.TestCase):
    spximg = sample_data.get_sample_level1b_img()
    rp = ReadoutParameters.from_metadata(spximg.meta)

    def test_active_row_col(self):
        row_mid, col_mid = l1_utils.active_row_col(self.rp)
        assert row_mid == 16
        assert col_mid == 20

    def test_active_area_range(self):
        actv_rows, actv_cols = l1_utils.active_area_range(self.spximg)
        assert actv_rows.stop - actv_rows.start == 28
        assert actv_cols.stop - actv_cols.start == 28
        # Including nref reference rows.
        actv_ref_rows, actv_ref_cols = l1_utils.active_area_range(self.spximg, include_refs=True)
        assert actv_ref_rows.stop - actv_ref_rows.start == 32
        assert actv_ref_cols.stop - actv_ref_cols.start == 30

    def test_with_extra_refs(self):
        rp = ReadoutParameters(nref=3, Mphan=2, nchannel=4, wchannel=8,
                               kref=4, jref=4, Jactive=28, Kactive=28)
        calib = np.zeros((rp.Jactive + rp.jref, rp.Kactive + rp.kref), dtype=np.int8)
        np.fill_diagonal(calib, 1)
        for i in range(1, (rp.Jactive + rp.jref)+1):
            calib[i-1] *= i
        calib_xtra_nrefs = l1_utils.with_extra_refs(rp, calib)
        assert calib_xtra_nrefs.shape == (rp.Jactive + rp.jref * rp.nref, rp.Kactive + rp.kref)
        assert np.array_equal(calib_xtra_nrefs[-8], calib_xtra_nrefs[-7])
        assert np.array_equal(calib_xtra_nrefs[-6], calib_xtra_nrefs[-5])
        assert np.array_equal(calib_xtra_nrefs[-4], calib_xtra_nrefs[-3])
        assert np.array_equal(calib_xtra_nrefs[-2], calib_xtra_nrefs[-1])


if __name__ == '__main__':
    unittest.main()
