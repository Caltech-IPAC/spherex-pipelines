# Standard library imports
from pathlib import Path
import unittest

# Related third party imports
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage, PSFZoneData
from spx_tasks import EstimatePSFConfig, EstimatePSFTask
from spx_tests.mock_dataset import MockDeferredImageHandle, MockDeferredCatalogHandle

test_root = Path(__file__).parent.parent


def get_sim_img() -> SPHERExImage:
    """
    Gets input simulated SPHEREx level 2 image.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.
    """
    sample_file = test_root / "data/simplified_psf.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file, hdu_layers=[])
    cat_file = test_root / "data/simplified_cat.fits"
    with fits.open(cat_file) as hdul_cat:
        cat = Table(hdul_cat[1].data)
        cat.rename_column('ABmag', 'mag_det1')
    spximg.catalog = cat

    return spximg


class TestEstimatePSF(unittest.TestCase):
    """Test the updated estimatePSF task."""

    def setUp(self):
        """Set up the test case with a new config and task instance."""
        # Configuration
        self.config = EstimatePSFConfig()
        self.config.bins_x = 2
        self.config.bins_y = 2
        self.config.run_parallel = False  # Ensure tests run serially
        self.config.do_plots = False  # Disable plotting for tests

        self.task = EstimatePSFTask(config=self.config)

        # Mock data
        self.spximg = get_sim_img()
        # Ensure the mock image passes the quality checks in the task
        self.spximg.header["FINAST"] = 0
        self.spximg.header["SPS_FLAGS"] = "survey"

    def mockDataId(self, i: int):
        """Create a mock dataId."""
        return {"spsplan": 111, "slewgroup": i, "dichroicpair": 0, "exposure": i, "detector": 1}

    def test_task_run(self):
        # Create a list of mock deferred handles for images and catalogs
        image_handles = \
            [MockDeferredImageHandle(image=self.spximg, dataid=self.mockDataId(i)) for i in range(2)]
        catalog_handles = [MockDeferredCatalogHandle(h) for h in image_handles]

        # Run the task with the updated method signature
        out = self.task.run(image_handles, catalog_handles)

        self.assertIsNotNone(out.psf, "Final PSF output is None.")
        self.assertIsNotNone(out.psf_stack, "Stacked PSF output is None.")
        self.assertIsNotNone(out.fwhm_dist, "FWHM distribution output is None.")
        self.assertIsInstance(out.psf, PSFZoneData, "Final PSF is not a PSFZoneData object.")

        # Check that the output dimensions are correct
        num_zones = self.config.bins_x * self.config.bins_y
        final_psf_size = 101  # The new task hard-codes the final cropped size
        expected_shape = (num_zones, final_psf_size, final_psf_size)

        self.assertEqual(out.psf.data.shape, expected_shape,
                         f"Final PSF data shape mismatch. Expected: {expected_shape}, "
                         f"got: {out.psf.data.shape}")
        self.assertEqual(out.psf_stack.data.shape, expected_shape,
                         f"Stacked PSF data shape mismatch. Expected: {expected_shape}, "
                         f"got: {out.psf_stack.data.shape}")

        # Verify history is added to the PSF header
        self.assertIn("HISTORY", out.psf.header, "No HISTORY entry found in PSF header.")


if __name__ == '__main__':
    unittest.main()
