import unittest
from astropy.time import Time
from spx_data.odir import ODIR, OdirObservation, OdirImage, Timestamp


class TestODIR(unittest.TestCase):
    """Unit tests for the ODIR Pydantic models."""

    def setUp(self):
        """Set up common data for all tests."""
        self.start_time = Time("2023-01-01T12:00:00")
        self.end_time = Time("2023-01-01T12:05:00")

        self.timestamp = Timestamp.from_time(self.start_time, self.end_time)

        self.image1 = OdirImage(status="good")
        self.image2 = OdirImage(status="bad")

        self.observation_data = {
            "survey_plan_indicator": True,
            "target_id": 12345,
            "sps_flag": "all_sky",
            "timestamp": self.timestamp,
            "images": {1: self.image1, 2: self.image2}
        }

    def test_create_odir_image(self):
        """Test the successful creation of an OdirImage instance."""
        self.assertEqual(self.image1.status, "good")
        self.assertEqual(self.image2.status, "bad")
        with self.assertRaises(ValueError):
            # Pydantic should raise a validation error for missing fields
            OdirImage()

    def test_create_odir_observation(self):
        """Test the successful creation of an OdirObservation instance."""
        obs = OdirObservation(**self.observation_data)
        self.assertTrue(obs.survey_plan_indicator)
        self.assertEqual(obs.target_id, 12345)
        self.assertEqual(obs.sps_flag, "all_sky")
        self.assertEqual(len(obs.images), 2)
        self.assertEqual(obs.images[1].status, "good")

    def test_create_odir(self):
        """Test the successful creation of the top-level ODIR instance."""
        obs = OdirObservation(**self.observation_data)
        odir_data = {"observations": {"SPS123": obs}}
        odir = ODIR(**odir_data)

        self.assertIn("SPS123", odir.observations)
        self.assertEqual(len(odir.observations["SPS123"].images), 2)
        self.assertEqual(odir.observations["SPS123"].target_id, 12345)

    def test_odir_serialization(self):
        """Test that the ODIR model serializes to a dictionary correctly."""
        obs = OdirObservation(**self.observation_data)
        odir = ODIR(observations={"SPS123": obs})

        # Use Pydantic's model_dump method for serialization
        odir_dict = odir.model_dump()

        expected_dict = {
            "observations": {
                "SPS123": {
                    "survey_plan_indicator": True,
                    "target_id": 12345,
                    "sps_flag": "all_sky",
                    "timestamp": {
                        "start": self.start_time.iso,
                        "end": self.end_time.iso
                    },
                    "images": {
                        1: {"status": "good"},
                        2: {"status": "bad"}
                    }
                }
            }
        }

        self.assertDictEqual(odir_dict, expected_dict)


if __name__ == '__main__':
    unittest.main()
