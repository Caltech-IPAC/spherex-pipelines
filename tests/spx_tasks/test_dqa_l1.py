# Standard library imports
import unittest

# Related third party imports
from astropy.io import fits
import numpy as np
from astropy.time import Time

# Local application/library specific imports
from spx_data import ReadoutParameters, SPHERExImage, Flag
from spx_tests import get_sample_level1_img
from spx_tasks import DQAL1Config, DQAL1Task


def fake_flags_metadata(spximg: SPHERExImage, rp: dict):
    """Add flag definitions and fake header metadata to sample image."""
    # Create a base header with all required keys for DQA L1 and get_odir
    header = fits.Header(rp.to_metadata())
    header["OBSID"] = "2025W37_2A_0486_2"  # Valid SPS format
    header["DETECTOR"] = 1
    header["HIERARCH SPS_TARGET"] = 54321
    header["HIERARCH SPS_FLAGS"] = "all_sky"
    start_time = Time("2025-09-15T12:00:00")
    end_time = Time("2025-09-15T12:05:00")
    header["MJD-BEG"] = (start_time.mjd, "Exposure start time in MJD")
    header["MJD-END"] = (end_time.mjd, "Exposure end time in MJD")
    header["DATE-BEG"] = (start_time.iso, "Exposure start time in ISO")
    header["DATE-END"] = (end_time.iso, "Exposure end time in ISO")

    spximg.header = header
    spximg.flag_defs = {'TRANSIENT': 0,
                        'OVERFLOW': 1,
                        'SUR_ERROR': 2,
                        'PHANTOM': 4,
                        'REFERENCE': 5,
                        'MISSING_DATA': 6}


class TestL1DQA(unittest.TestCase):
    """Test L1 DQA Task."""

    config = DQAL1Config()
    # Set a lower transient threshold for easier testing
    config.max_transient_pix = 50
    task = DQAL1Task(config=config)
    rp = ReadoutParameters()

    def test_pass_case(self):
        """Test the case where all DQA checks pass."""
        spximg = get_sample_level1_img(self.rp)
        fake_flags_metadata(spximg, self.rp)
        output = self.task.run(spximg)

        # Check main DQA flag
        self.assertEqual(output.spximg_out.header["L1DQAFLG"], "Pass")
        self.assertIsNotNone(output.stats.q95)

        # Check ODIR output
        self.assertIsNotNone(output.odir_obs)
        obs = output.odir_obs.observations["2025W37_2A_0486_2"]
        self.assertEqual(obs.images[1].status, "good")
        self.assertTrue(obs.survey_plan_indicator)

    def test_fail_case_bad_dtype(self):
        """Test a fail case due to incorrect data type."""
        spximg = get_sample_level1_img(self.rp)
        fake_flags_metadata(spximg, self.rp)
        # Populate data attribute with zeros, incorrect data type
        spximg.data = np.zeros(spximg.data.shape, dtype=np.int16)
        output = self.task.run(spximg)

        # Check main DQA flag
        self.assertEqual(output.spximg_out.header["L1DQAFLG"], "Fail")
        self.assertIsNotNone(output.stats.q95)

        # ODIR should still be generated and have a 'good' status
        self.assertIsNotNone(output.odir_obs)
        obs = output.odir_obs.observations["2025W37_2A_0486_2"]
        self.assertEqual(obs.images[1].status, "good")
        self.assertTrue(obs.survey_plan_indicator)

    def test_odir_reobserve_flag_triggered(self):
        """Test that the ODIR correctly flags for re-observation due to transients."""
        spximg = get_sample_level1_img(self.rp)
        fake_flags_metadata(spximg, self.rp)

        # Inject transient flags above the threshold
        spximg.flags = np.zeros(spximg.data.shape, dtype=np.int16)
        spximg.flags[50:60, 50:60] = Flag.TRANSIENT  # 100 transients

        output = self.task.run(spximg)

        # Main DQA might still pass as it doesn't check transients for this flag
        self.assertEqual(output.spximg_out.header["L1DQAFLG"], "Pass")

        # Check ODIR output for the "bad" status and re-observe flag
        self.assertIsNotNone(output.odir_obs)
        obs = output.odir_obs.observations["2025W37_2A_0486_2"]
        self.assertEqual(obs.images[1].status, "bad")
        self.assertFalse(obs.survey_plan_indicator)

    def test_nan_pixel_check(self):
        """Test case where the image contains NaN values exceeding the threshold."""
        spximg = get_sample_level1_img(self.rp)
        fake_flags_metadata(spximg, self.rp)
        # Inject NaN values into the data
        spximg.data[:-100, :-100] = np.nan

        # Run the task and check DQA result
        output = self.task.run(spximg)
        self.assertEqual(output.spximg_out.header["L1DQAFLG"], "Fail")

        # Check NaN percentage
        nan_check_pass, nan_percentage = self.task.check_nan_percent(spximg)
        self.assertFalse(nan_check_pass, "NaN percentage check should fail.")
        self.assertGreater(nan_percentage, self.config.nan_percentage_threshold)

        # ODIR should still be generated and have a 'good' status
        self.assertIsNotNone(output.odir_obs)
        obs = output.odir_obs.observations["2025W37_2A_0486_2"]
        self.assertEqual(obs.images[1].status, "good")

    def test_nan_check_below_threshold(self):
        """Test case where the image contains NaN values below the threshold."""
        spximg = get_sample_level1_img(self.rp)
        fake_flags_metadata(spximg, self.rp)
        # Introduce a small number of NaNs
        spximg.data[0:10, 0:10] = np.nan

        nan_check_pass, nan_percentage = self.task.check_nan_percent(spximg)
        self.assertTrue(nan_check_pass, "NaN percentage check should pass.")
        self.assertLess(nan_percentage, self.config.nan_percentage_threshold)

        # Run the full task to check the output
        output = self.task.run(spximg)
        self.assertEqual(output.spximg_out.header["L1DQAFLG"], "Pass")

        # ODIR should be good
        self.assertIsNotNone(output.odir_obs)
        obs = output.odir_obs.observations["2025W37_2A_0486_2"]
        self.assertEqual(obs.images[1].status, "good")


if __name__ == "__main__":
    unittest.main()
