import unittest
from astropy.io import fits
import numpy as np
from spx_data import SPHERExImage, Flag
from spx_tasks.source_bloom_and_cosmic_ray_halo_flagging import (SourceBloomAndCosmicRayHaloFlaggingTask,
                                                                 SourceBloomAndCosmicRayHaloFlaggingConfig)


class TestSourceBloomAndCosmicRayHaloFlagging(unittest.TestCase):

    def setUp(self):
        """Set up a mock SPHERExImage designed to trigger the halo bug."""
        self.shape = (100, 100)
        self.data = np.zeros(self.shape, dtype=np.float32)
        self.flags = np.zeros(self.shape, dtype=np.int32)

        # Mock header.
        # TELAPSE=1.0 makes fluence=data.
        # DETECTOR=1 sets min_overflow threshold to 5e4.
        self.header = fits.Header()
        self.header["TELAPSE"] = (1.0, "Mock exposure time")
        self.header["DETECTOR"] = (1, "Mock detector ID")

        # Set fluence > min_overflow (5e4) for overflow pixels
        overflow_fluence = 6.0e4

        # Pixel A: snow_type=0 center (early, transient, overflow)
        # This will be detected in the first loop of get_snowball_flag
        # and (for a 1-pixel event) create a 3x3 halo with value 2.
        self.loc_A = (20, 20)
        self.flags[self.loc_A] = Flag.TRANSIENT | Flag.OVERFLOW | Flag.SUR_ERROR
        self.data[self.loc_A] = overflow_fluence

        # Add an "outer ring" pixel for A. This is required by the
        # n_outer > 0 check to trigger the snowball/halo logic.
        # It must be (transient, early) but NOT (overflow).
        self.loc_A_ring = (19, 20)
        self.flags[self.loc_A_ring] = Flag.TRANSIENT | Flag.SUR_ERROR
        self.data[self.loc_A_ring] = 100.0  # Not overflow

        # Pixel B: snow_type=1 center (late, transient, overflow)
        # This pixel is *inside* the 3x3 halo of Pixel A.
        # In the second loop, adding another halo (value 2) on top.
        self.loc_B = (20, 21)
        self.flags[self.loc_B] = Flag.TRANSIENT | Flag.OVERFLOW
        self.data[self.loc_B] = overflow_fluence

        # Pixel C: Simple transient
        # This will be found by mark_transient_halo and create a halo (value 1).
        # Placed far away to avoid overlap with A and B.
        self.loc_C = (40, 40)
        self.flags[self.loc_C] = Flag.TRANSIENT
        self.data[self.loc_C] = 100.0

        # Create the mock SPHERExImage
        self.spximg = SPHERExImage(data=self.data, flags=self.flags, meta=self.header, unit='electron/s')

        # Configure the Task
        self.config = SourceBloomAndCosmicRayHaloFlaggingConfig()
        # Must write diagnostics to check the HALO_FLAG layer
        self.config.write_diagnostics = True
        # Use a predictable halo method
        self.config.bsh.satellite_halomethod = "dilation"
        # Ensure the BSHConfig defaults are loaded
        self.config.bsh.setDefaults()

    def test_halo_multiplication(self):
        """
        This test creates a scenario where a snowball center (Pixel B)
        is located inside the halo of another snowball (Pixel A) that
        is detected in an earlier loop.
        """
        # Instantiate and run the task
        task = SourceBloomAndCosmicRayHaloFlaggingTask(config=self.config)
        result = task.run(self.spximg)

        # Check that diagnostic output was created
        self.assertIsNotNone(result.bsh_diag_img, "Diagnostic image was not created.")
        self.assertIn("HALO_FLAG", result.bsh_diag_img._layers.keys(), "HALO_FLAG layer is missing.")

        # Get the HALO_FLAG diagnostic layer
        halo_diag_layer = result.bsh_diag_img._layers["HALO_FLAG"].data

        # Halo for Pixel C (simple transient) should be 1
        # (mark_transient_halo gives value 1)
        self.assertEqual(halo_diag_layer[self.loc_C], 1, "Simple transient center (loc_C) should be 1")
        self.assertEqual(halo_diag_layer[40, 41], 1, "Simple transient neighbor should be 1")

        # Halo for Pixel A (snowball) should be 2 + 1 (transient)
        self.assertEqual(halo_diag_layer[self.loc_A], 3, "Snowball A center (loc_A) should be 3")

        # Check the value at Pixel B.
        self.assertEqual(halo_diag_layer[self.loc_B], 3, "Pixel B (in A's halo) should be 3.")


if __name__ == '__main__':
    unittest.main()
