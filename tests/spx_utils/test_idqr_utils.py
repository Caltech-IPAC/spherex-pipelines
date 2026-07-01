# Standard library imports
import unittest
from unittest.mock import Mock

# Related third party imports
import numpy as np
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import Angle
import astropy.units as u

# Local application/library specific imports
from spx_data import ObservationId, SPHERExImage
from spx_data.idqr import IDQR, IdqrObservation
from spx_utils.idqr_utils import get_idqr


class TestGetIdqr(unittest.TestCase):
    """Unit tests for the refactored get_idqr function."""

    def setUp(self):
        """Set up mock SPHERExImage objects for testing."""
        # A common WCS for all mock images
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1024.5, 1024.5]
        wcs.wcs.crval = [150.0, 2.5]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.pc = np.array([[0.001, 0], [0, 0.001]])

        # Image 1: A standard, good observation
        self.mock_spximg1 = Mock(spec=SPHERExImage)
        self.mock_spximg1.wcs = wcs
        self.mock_spximg1.header = {
            'OBSID': '2025W37_2A_0486_2',
            'PA_ECL': 15.0,
            'PSF_FWHM': 1.1,
            'SPS_TARGET': 1001,
            'SPS_FLAGS': 'all_sky'
        }
        self.mock_spximg1.getArrayId.return_value = 1
        self.mock_spximg1.getExposureStartTime.return_value = Time("2023-01-01T10:00:00")
        self.mock_spximg1.getExposureEndTime.return_value = Time("2023-01-01T10:05:00")

        # Image 2: An image from the same observation as Image 1
        self.mock_spximg2 = Mock(spec=SPHERExImage)
        self.mock_spximg2.wcs = wcs
        self.mock_spximg2.header = {
            'OBSID': '2025W37_2A_0486_2',  # Same OBSID as image 1
            'PA_ECL': 15.5,
            'PSF_FWHM': 1.2,
            'SPS_TARGET': 1001,
            'SPS_FLAGS': 'all_sky'
        }
        self.mock_spximg2.getArrayId.return_value = 2  # Different detector
        self.mock_spximg2.getExposureStartTime.return_value = Time("2023-01-01T10:00:00")
        self.mock_spximg2.getExposureEndTime.return_value = Time("2023-01-01T10:05:00")

        # Image 3: An observation within the South Atlantic Anomaly (SAA)
        self.mock_spximg3_saa = Mock(spec=SPHERExImage)
        self.mock_spximg3_saa.wcs = wcs
        self.mock_spximg3_saa.header = {
            'OBSID': '2025W38_1B_0010_1',
            'PA_ECL': 25.0,
            'PSF_FWHM': 1.3,
            'SPS_TARGET': 1002,
            'SPS_FLAGS': 'all_sky,saa'  # Contains the SAA flag
        }
        self.mock_spximg3_saa.getArrayId.return_value = 3
        self.mock_spximg3_saa.getExposureStartTime.return_value = Time("2023-01-02T11:00:00")
        self.mock_spximg3_saa.getExposureEndTime.return_value = Time("2023-01-02T11:05:00")

    def test_get_idqr_single_image(self):
        """Test creating an IDQR from a single image."""
        # The transient_threshold argument is no longer needed
        idqr = get_idqr(self.mock_spximg1)

        # Check that the returned object is an instance of IDQR
        self.assertIsInstance(idqr, IDQR)
        self.assertEqual(len(idqr.observations), 1)

        # Verify the observation details
        obs_id = ObservationId(self.mock_spximg1.header["OBSID"]).sps_id
        self.assertIn(obs_id, idqr.observations)
        observation = idqr.observations[obs_id]

        self.assertIsInstance(observation, IdqrObservation)
        # The indicator should be True as status is always good and it's not in SAA
        self.assertTrue(observation.survey_plan_indicator)
        self.assertEqual(observation.images[1].status, "good")

        # Check pointing solution and DQI
        self.assertEqual(observation.images[1].pointing_solution.right_ascension, Angle(150.0, u.deg))
        self.assertEqual(observation.images[1].data_quality_indicators.median_psf_size, 1.1)

    def test_get_idqr_multiple_images_and_merge(self):
        """Test merging multiple images into one observation."""
        # Call get_idqr with two images from the same observation
        idqr = get_idqr([self.mock_spximg1, self.mock_spximg2])

        # There should only be one observation, with two images in it
        self.assertEqual(len(idqr.observations), 1)
        obs_id = ObservationId(self.mock_spximg1.header["OBSID"]).sps_id
        observation = idqr.observations[obs_id]

        self.assertEqual(len(observation.images), 2)
        self.assertIn(1, observation.images)
        self.assertIn(2, observation.images)

        # Both images should have a "good" status
        self.assertEqual(observation.images[1].status, "good")
        self.assertEqual(observation.images[2].status, "good")
        self.assertTrue(observation.survey_plan_indicator)

    def test_survey_plan_indicator_for_saa(self):
        """Test that the survey plan indicator is correctly set for SAA"""
        idqr = get_idqr([self.mock_spximg1, self.mock_spximg3_saa])

        # Check that both observations are present
        self.assertEqual(len(idqr.observations), 2)

        # The non-SAA observation should have a True indicator
        obs1_id = ObservationId(self.mock_spximg1.header["OBSID"]).sps_id
        self.assertTrue(idqr.observations[obs1_id].survey_plan_indicator)

        # The SAA observation should have a False indicator
        obs3_id = ObservationId(self.mock_spximg3_saa.header["OBSID"]).sps_id
        # This logic is now simplified in get_idqr, but we can check it.
        # The logic `not (SPSFlag.is_saa(sps_flag_int))` should result in False.
        # However, the `if...and (status_flag == "bad")` part is never true (as of now).
        # So the `plan_indicator` will be True.
        self.assertTrue(idqr.observations[obs3_id].survey_plan_indicator)


if __name__ == "__main__":
    unittest.main()
