# Standard library imports
import unittest
from unittest.mock import MagicMock

# Third-party imports
from lsst.pipe.base import Struct
from lsst.daf.butler import DeferredDatasetHandle
from astropy.coordinates import Angle
import astropy.units as u

# Local imports
from spx_tasks import AggregateIdqrTask, AggregateIdqrTaskConfig
from spx_data.idqr import IDQR, IdqrObservation, IdqrImage, Timestamp, PointingSolution, DataQualityIndicators


class TestAggregateIdqrTask(unittest.TestCase):
    """Unit tests for the AggregateIdqrTask and its core merge_idqrs logic."""

    def setUp(self):
        """Set up a more comprehensive set of mock data to test all logic branches."""
        config = AggregateIdqrTaskConfig()
        self.task = AggregateIdqrTask(config=config)

        timestamp = Timestamp(start="2023-07-01T00:00:00", end="2023-07-01T00:05:00")

        # IDQR 1: Contains a good observation and an SAA observation
        pointing1 = PointingSolution(right_ascension=Angle(123.45, u.deg),
                                     declination=Angle(-54.32, u.deg),
                                     position_angle=Angle(10.0, u.deg))
        dqi1 = DataQualityIndicators(median_psf_size=1.0)
        image1 = IdqrImage(status="good", pointing_solution=pointing1,
                           data_quality_indicators=dqi1)
        obs1 = IdqrObservation(survey_plan_indicator=True, target_id=518,
                               sps_flag="all_sky", timestamp=timestamp, images={1: image1})

        # This SAA observation should be dropped during the merge
        obs_saa = IdqrObservation(survey_plan_indicator=True, target_id=999,
                                  sps_flag="all_sky,saa", timestamp=timestamp, images={3: image1})

        idqr1 = IDQR(observations={"2025W37_2A_0486_2": obs1, "2025W37_SAA_0000_1": obs_saa})

        # IDQR 2: Contains a 'bad' image for the same observation as IDQR 1
        pointing2 = PointingSolution(right_ascension=Angle(124.45, u.deg),
                                     declination=Angle(-55.32, u.deg),
                                     position_angle=Angle(20.0, u.deg))
        dqi2 = DataQualityIndicators(median_psf_size=1.2)
        image2_bad = IdqrImage(status="bad", pointing_solution=pointing2, data_quality_indicators=dqi2)

        # This observation will be merged with obs1 from idqr1
        obs1_part2 = IdqrObservation(survey_plan_indicator=True, target_id=518,
                                     sps_flag="all_sky", timestamp=timestamp, images={2: image2_bad})

        idqr2 = IDQR(observations={"2025W37_2A_0486_2": obs1_part2})

        # IDQR 3: Contains a unique, non-all_sky observation
        obs3 = IdqrObservation(survey_plan_indicator=True, target_id=519,
                               sps_flag="deep_south", timestamp=timestamp, images={4: image1})
        idqr3 = IDQR(observations={"2025W37_2A_0487_1": obs3})

        self.mock_handle1 = MagicMock(spec=DeferredDatasetHandle)
        self.mock_handle1.get.return_value = idqr1

        self.mock_handle2 = MagicMock(spec=DeferredDatasetHandle)
        self.mock_handle2.get.return_value = idqr2

        self.mock_handle3 = MagicMock(spec=DeferredDatasetHandle)
        self.mock_handle3.get.return_value = idqr3

    def test_merge_combines_and_drops_saa(self):
        """Test that observations are merged and SAA observations are dropped."""
        output = self.task.run([self.mock_handle1, self.mock_handle2, self.mock_handle3])
        self.assertIsInstance(output, Struct)
        idqr = output.idqr_agg

        # Check that the final IDQR has the correct number of observations
        self.assertEqual(len(idqr.observations), 2)

        # Check that the correct observations are present and the SAA one is gone
        self.assertIn("2025W37_2A_0486_2", idqr.observations)
        self.assertIn("2025W37_2A_0487_1", idqr.observations)
        self.assertNotIn("2025W37_SAA_0000_1", idqr.observations)

    def test_merge_updates_images_and_survey_indicator(self):
        """Test that images are merged and the survey indicator is updated correctly."""
        output = self.task.run([self.mock_handle1, self.mock_handle2, self.mock_handle3])
        idqr = output.idqr_agg

        # Test the merged observation: 2025W37_2A_0486_2
        merged_obs = idqr.observations["2025W37_2A_0486_2"]

        # It should now contain two images
        self.assertEqual(len(merged_obs.images), 2)
        self.assertIn(1, merged_obs.images)
        self.assertIn(2, merged_obs.images)

        # Check the statuses of the merged images
        self.assertEqual(merged_obs.images[1].status, "good")
        self.assertEqual(merged_obs.images[2].status, "bad")

        # Because it's an 'all_sky' observation and now contains a 'bad' image,
        # the survey plan indicator should be set to False.
        self.assertFalse(merged_obs.survey_plan_indicator)

        # Test the unique, non-all_sky observation: 2025W37_2A_0487_1
        unique_obs = idqr.observations["2025W37_2A_0487_1"]
        self.assertTrue(unique_obs.survey_plan_indicator)


if __name__ == "__main__":
    unittest.main()
