from spx_tests import get_sample_level1_img
from spx_utils import flags_summary, log_threshold_test, nan_pixels_flagged
from spx_data import ReadoutParameters, Flag
from pathlib import Path
import logging
import unittest
from astropy.io import fits
# Related third party imports
import numpy as np
# Local application/library specific imports
test_root = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


class TestDQAUtils(unittest.TestCase):
    """Test DQA utility functions."""

    rp = ReadoutParameters(nref=2, Mphan=2, nchannel=3, wchannel=10,
                           kref=2, jref=2, Jactive=28, Kactive=28)
    spximg = get_sample_level1_img(rp)
    spximg.header = fits.Header(rp.to_metadata())
    spximg.flag_defs = {'TRANSIENT': 0,
                        'OVERFLOW': 1,
                        'SUR_ERROR': 2,
                        'PHANTOM': 4,
                        'REFERENCE': 5}

    def test_flags_summary(self):
        # Calculate stats
        mean = np.nanmean(self.spximg.data)
        median = np.nanmedian(self.spximg.data)
        rms = np.sqrt(np.sum(np.power(self.spximg.data, 2)) / len(self.spximg.data.flatten()))

        img_stats_dict = {
            "MEAN": f"{mean:.4f} {self.spximg.unit}",
            "MEDIAN": f"{median:.4f} {self.spximg.unit}",
            "RMS": f"{rms:.4f}",
        }

        # Calculate flags and add to dict
        flags_summary(self.spximg, img_stats_dict)
        expected_keys = ["MEAN", "MEDIAN", "RMS", "TRANSIENT FLAGS", "OVERFLOW FLAGS",
                         "SUR_ERROR FLAGS", "PHANTOM FLAGS", "REFERENCE FLAGS"]

        assert list(img_stats_dict.keys()) == expected_keys, "Image stats dictionary is missing keys."
        ref_flags = self.rp.jref*self.rp.nref * \
            (self.rp.Kframe-self.rp.Kphan)+self.rp.kref*(self.rp.Jframe - self.rp.jref * self.rp.nref)
        phantom_flags = self.rp.Jphan*self.rp.Kphan
        assert img_stats_dict[expected_keys[-1]] == ref_flags
        assert img_stats_dict[expected_keys[-2]] == phantom_flags

    def test_log_threshold_test_pass(self):
        """Test threshold test logger for passing tests."""
        message = "Data array has shape (2072, 2176) and element type np.float32."
        data_test_1 = {
            "result": "PASS",
            "description": "Data array has the correct shape (2072, 2176) and bit depth."
        }
        with self.assertLogs(logger, logging.DEBUG) as captured:
            log_threshold_test(logger, data_test_1, message)
        self.assertEqual(len(captured.records), 1)  # only info message is logged
        self.assertEqual(captured.records[0].getMessage(),
                         f"{data_test_1['result']}: {data_test_1['description']}")

    def test_log_threshold_test_fail(self):
        """Test threshold test logger for failing tests."""
        message = "Data array has shape (2072, 2176) and element type np.int16"
        data_test_1 = {
            "result": "FAIL",
            "description": "Data array has the correct shape (2072, 2176) and bit depth."
        }
        with self.assertLogs(logger, logging.DEBUG) as captured:
            log_threshold_test(logger, data_test_1, message)
        self.assertEqual(len(captured.records), 2)  # info and warning messages logged
        self.assertEqual(captured.records[0].getMessage(),
                         f"{data_test_1['result']}: {data_test_1['description']}")
        self.assertEqual(captured.records[1].getMessage(), message)

    def test_check_nan_pixels_flagged(self):
        """Test check_nan_pixels_flagged utility."""
        data = np.array([[1.0, np.nan], [3.0, 4.0]])
        flags_good = np.array([[0, Flag.MISSING_DATA], [0, 0]], dtype=np.int32)
        flags_bad = np.array([[0, 0], [0, 0]], dtype=np.int32)
        nan_flags = Flag.MISSING_DATA  # only treat MISSING_DATA as a valid reason

        # Test for properly flagged NaN
        self.assertTrue(nan_pixels_flagged(data, flags_good, nan_flags))

        # Test for unflagged NaN
        self.assertFalse(nan_pixels_flagged(data, flags_bad, nan_flags))


if __name__ == "__main__":
    unittest.main()
