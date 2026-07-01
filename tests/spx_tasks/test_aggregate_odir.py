import unittest
from astropy.time import Time
from spx_data.odir import ODIR, OdirObservation, OdirImage, Timestamp
from spx_tasks.aggregate_odir import merge_odirs


class TestMergeOdirs(unittest.TestCase):
    """Unit tests for the merge_odirs function."""

    def setUp(self):
        """Set up common data and mock ODIR objects for tests."""
        # Common timestamp for all test data
        start_time = Time("2023-01-01T12:00:00")
        end_time = Time("2023-01-01T12:05:00")
        self.timestamp = Timestamp.from_time(start_time, end_time)

        obs1_img1 = OdirImage(status="good")
        obs1 = OdirObservation(
            survey_plan_indicator=True,
            target_id=101,
            sps_flag="all_sky",
            timestamp=self.timestamp,
            images={1: obs1_img1}
        )
        # This observation should be dropped because it's in the SAA
        obs_saa = OdirObservation(
            survey_plan_indicator=True,
            target_id=999,
            sps_flag="all_sky,saa",
            timestamp=self.timestamp,
            images={3: OdirImage(status="good")}
        )
        self.odir1 = ODIR(observations={"SPS101": obs1, "SPS999": obs_saa})

        # This image will be merged into SPS101 from odir1
        obs1_img2 = OdirImage(status="bad")
        obs1_part2 = OdirObservation(
            survey_plan_indicator=True,  # This will be ignored and re-evaluated on merge
            target_id=101,
            sps_flag="all_sky",
            timestamp=self.timestamp,
            images={2: obs1_img2}
        )
        # A unique observation for odir2
        obs2 = OdirObservation(
            survey_plan_indicator=True,
            target_id=202,
            sps_flag="deep_south",
            timestamp=self.timestamp,
            images={4: OdirImage(status="bad")}
        )
        self.odir2 = ODIR(observations={"SPS101": obs1_part2, "SPS202": obs2})

    def test_merge_combines_observations(self):
        """Test that unique observations are kept and shared ones are merged."""
        merged_odir = merge_odirs([self.odir1, self.odir2])

        # Should contain SPS101 (merged) and SPS202 (unique), but not SPS999 (SAA)
        self.assertIn("SPS101", merged_odir.observations)
        self.assertIn("SPS202", merged_odir.observations)
        self.assertEqual(len(merged_odir.observations), 2)

    def test_saa_observation_is_dropped(self):
        """Test that an observation with the 'saa' flag is correctly dropped."""
        merged_odir = merge_odirs([self.odir1, self.odir2])
        self.assertNotIn("SPS999", merged_odir.observations)

    def test_images_are_merged_for_shared_id(self):
        """Test that images from a shared observation ID are correctly combined."""
        merged_odir = merge_odirs([self.odir1, self.odir2])
        merged_obs1 = merged_odir.observations["SPS101"]

        # The final observation should have images from both original ODIRs
        self.assertEqual(len(merged_obs1.images), 2)
        self.assertIn(1, merged_obs1.images)
        self.assertIn(2, merged_obs1.images)
        self.assertEqual(merged_obs1.images[1].status, "good")
        self.assertEqual(merged_obs1.images[2].status, "bad")

    def test_survey_indicator_becomes_false_for_bad_all_sky(self):
        """Test indicator becomes False for a merged 'all_sky' obs with a bad image."""
        merged_odir = merge_odirs([self.odir1, self.odir2])
        merged_obs1 = merged_odir.observations["SPS101"]

        # Because SPS101 is 'all_sky' and one of its merged images is 'bad',
        # the final indicator should be False.
        self.assertFalse(merged_obs1.survey_plan_indicator)

    def test_survey_indicator_stays_true_for_non_all_sky(self):
        """Test indicator stays True for a non-'all_sky' obs even with a bad image."""
        merged_odir = merge_odirs([self.odir1, self.odir2])
        merged_obs2 = merged_odir.observations["SPS202"]

        # Because SPS202 is not 'all_sky', its indicator should remain True
        # even though its image has a 'bad' status.
        self.assertTrue(merged_obs2.survey_plan_indicator)

    def test_empty_input_list(self):
        """Test that merging an empty list results in an empty ODIR."""
        merged_odir = merge_odirs([])
        self.assertEqual(len(merged_odir.observations), 0)


if __name__ == '__main__':
    unittest.main()
