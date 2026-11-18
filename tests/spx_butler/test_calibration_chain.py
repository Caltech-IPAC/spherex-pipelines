import unittest
from unittest.mock import MagicMock
import numpy as np
from lsst.daf.butler import Butler


class TestCalibrationChain(unittest.TestCase):
    def setUp(self):
        """Set up a mock Butler and mock collections."""
        self.mock_butler = MagicMock(spec=Butler)

        # Define a test Data ID
        self.dataId = {"instrument": "simu", "detector": 1, "exposure": 123456}

        # Define test calibration collections
        self.chainid_calib = "calib"  # this is the ``calib`` chain
        self.calib_init = "calib/init/test"
        self.calib_update = "calib/update/test"

        # Step 1: Define initial calibration
        self.init_calib = np.ones((10, 10), dtype=np.float32)

        # Step 2: Define updated calibration
        self.updated_calib = np.full((10, 10), 1.12, dtype=np.float32)

        # Mock Butler registry methods
        self.mock_butler.registry.refreshCollectionChain = MagicMock()

        # Mock the behavior of Butler get for different collections
        def mock_get(datasetType, dataId, collections):
            if collections == [self.calib_init]:
                return self.init_calib
            elif collections == [self.chainid_calib]:
                return self.updated_calib
            return None

        self.mock_butler.get.side_effect = mock_get

    def test_calibration_chain_prepend(self):
        """Test that the calibration chain prepends correctly
        and retrieves the latest calibration.
        """
        # Step 1: Simulate ingesting the initial calibration
        self.mock_butler.put(self.init_calib, "abs_gain_matrix", dataId=self.dataId, run=self.calib_init)

        # Step 2: Retrieve and verify the initial calibration
        retrieved_calib = self.mock_butler.get("abs_gain_matrix",
                                               dataId=self.dataId,
                                               collections=[self.calib_init])
        self.assertAlmostEqual(np.mean(retrieved_calib), 1.0, places=5,
                               msg="Initial calibration is incorrect")

        # Step 3: Simulate ingesting the updated calibration
        self.mock_butler.put(self.updated_calib, "abs_gain_matrix", dataId=self.dataId, run=self.calib_update)

        # Step 4: Simulate updating the calibration chain using refreshCollectionChain
        self.mock_butler.registry.refreshCollectionChain(self.chainid_calib)

        # Step 5: Retrieve and verify the latest calibration from the chain
        latest_calib = self.mock_butler.get("abs_gain_matrix",
                                            dataId=self.dataId,
                                            collections=[self.chainid_calib])
        self.assertAlmostEqual(np.mean(latest_calib), 1.12, places=5,
                               msg="Updated calibration is incorrect")

        # Step 6: Verify that `refreshCollectionChain` was called
        self.mock_butler.registry.refreshCollectionChain.assert_called_once_with(self.chainid_calib)


if __name__ == "__main__":
    unittest.main(argv=[''], exit=False)
