import unittest
import numpy as np
from astropy.time import Time
from spx_data import Flag
from spx_utils.odir_utils import get_odir


class MockSPHERExImage:
    """A mock class to simulate SPHERExImage for unit testing."""

    def __init__(self, header, flags):
        self.header = header
        self.flags = flags
        self.wcs = None  # Not needed for get_odir, so we can ignore it.

    def getExposureStartTime(self):
        return Time(self.header["DATE-BEG"])

    def getExposureEndTime(self):
        return Time(self.header["DATE-END"])

    def getArrayId(self):
        return self.header["ARRAYID"]


class TestGetOdir(unittest.TestCase):
    """Unit tests for the get_odir function."""

    def setUp(self):
        """Set up common data and mock objects for tests."""
        self.valid_obs_id = "2025W37_2A_0486_2"
        self.base_header = {
            "OBSID": self.valid_obs_id,
            "ARRAYID": 1,
            "SPS_TARGET": "SPS_012345_S12",
            "SPS_TARGET_PEF": 12345,
            "SPS_FLAGS": "all_sky",
            "DATE-BEG": "2023-01-01T12:00:00",
            "DATE-END": "2023-01-01T12:05:00",
        }
        self.transient_threshold = 100

    def test_status_good_below_threshold(self):
        """Test that status is 'good' when transient count is below threshold."""
        flags = np.zeros((10, 10), dtype=int)
        flags[0, 0] = Flag.TRANSIENT  # 1 transient pixel
        mock_image = MockSPHERExImage(self.base_header, flags)

        odir = get_odir(mock_image, self.transient_threshold)
        image_status = odir.observations[self.valid_obs_id].images[1].status
        self.assertEqual(image_status, "good")

    def test_status_bad_above_threshold(self):
        """Test that status is 'bad' when transient count is above threshold."""
        flags = np.full((11, 11), Flag.TRANSIENT)  # 121 transient pixels
        mock_image = MockSPHERExImage(self.base_header, flags)

        odir = get_odir(mock_image, self.transient_threshold)
        image_status = odir.observations[self.valid_obs_id].images[1].status
        self.assertEqual(image_status, "bad")

    def test_survey_plan_indicator_false_for_bad_all_sky(self):
        """Test indicator is False for a 'bad' all_sky observation not in SAA."""
        flags = np.full((11, 11), Flag.TRANSIENT)  # Above threshold
        mock_image = MockSPHERExImage(self.base_header, flags)

        odir = get_odir(mock_image, self.transient_threshold)
        indicator = odir.observations[self.valid_obs_id].survey_plan_indicator
        self.assertFalse(indicator)

    def test_survey_plan_indicator_true_for_good_all_sky(self):
        """Test indicator is True for a 'good' all_sky observation."""
        flags = np.zeros((10, 10), dtype=int)  # Below threshold
        mock_image = MockSPHERExImage(self.base_header, flags)

        odir = get_odir(mock_image, self.transient_threshold)
        indicator = odir.observations[self.valid_obs_id].survey_plan_indicator
        self.assertTrue(indicator)

    def test_survey_plan_indicator_true_in_saa(self):
        """Test indicator is True for a 'bad' observation in the SAA."""
        header = self.base_header.copy()
        header["SPS_FLAGS"] = "all_sky,saa"
        flags = np.full((11, 11), Flag.TRANSIENT)  # Above threshold
        mock_image = MockSPHERExImage(header, flags)

        odir = get_odir(mock_image, self.transient_threshold)
        indicator = odir.observations[self.valid_obs_id].survey_plan_indicator
        self.assertTrue(indicator)

    def test_merge_observations(self):
        """Test that images from the same observation are merged correctly."""
        header1 = self.base_header.copy()
        header1["ARRAYID"] = 1
        flags1 = np.zeros((10, 10), dtype=int)  # Good image
        mock_image1 = MockSPHERExImage(header1, flags1)

        header2 = self.base_header.copy()
        header2["ARRAYID"] = 2
        flags2 = np.full((11, 11), Flag.TRANSIENT)  # Bad image
        mock_image2 = MockSPHERExImage(header2, flags2)

        # Process a list of images
        odir = get_odir([mock_image1, mock_image2], self.transient_threshold)

        # Check that there is only one observation
        self.assertEqual(len(odir.observations), 1)

        obs = odir.observations[self.valid_obs_id]
        self.assertEqual(len(obs.images), 2)
        self.assertIn(1, obs.images)
        self.assertIn(2, obs.images)
        self.assertEqual(obs.images[1].status, "good")
        self.assertEqual(obs.images[2].status, "bad")

        # Check that the survey indicator for the merged observation is False
        self.assertFalse(obs.survey_plan_indicator)


if __name__ == '__main__':
    unittest.main()
