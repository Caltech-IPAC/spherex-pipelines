# Standard library imports
from pathlib import Path
import logging
import unittest

# Related third party imports
from astropy.table import Table
from lsst.pipe.base import Struct
import astropy.units as u
import numpy as np

# Local application/library specific imports
from spx_data import spherex_image_reader, SPHERExImage
from spx_tasks import EstimateDiffuseBackgroundConfig, L3aConfig, L3aTask, PerformForcedPhotometryConfig
from spx_tests import get_sample_pix_map


test_root = Path(__file__).parent.parent


def get_sample_file() -> SPHERExImage:
    sample_file = test_root / "data/photometry_point_sources.fits"  # NOTE: image shape is 15x15
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def fake_invalid_cat() -> Table:
    invalid_cat = Table()
    invalid_cat["SPHERExRefID"] = np.int64([12345, 23456, 34567, 45678])
    invalid_cat["ra"] = np.float64([30.437813, 321.385428, 197.486377, 246.971852]) * u.deg
    invalid_cat["declination"] = np.float32([-85.156436, 2.178935, 54.940162, -12.617344]) * u.deg
    invalid_cat["shape_r"] = np.float32([0.0, 0.0, -9999.0, 1.5]) * u.deg
    invalid_cat["ab_ratio"] = np.float32([0.0, 0.0, -9999.0, 13.45])
    invalid_cat["phi"] = np.float32([0.0, 0.0, -9999.0, 27.429])
    invalid_cat["sersic"] = np.float32([0.0, 0.0, 0.0, 4.2])
    return invalid_cat


# Little hack to give a way to pass in generated tables without filepath
class MockDeferredDatasetHandle:
    def __init__(self, table: Table):
        self.table = table

    def get(self, parameters=None) -> Table:
        if parameters is not None:
            return self.table[parameters["columns"]]
        else:
            return self.table


class TestL3aTask(unittest.TestCase):
    """Test L3a Task."""
    # Inputs
    ps_img = spherex_image_reader(get_sample_file(), hdu_layers=["PSF", "CATALOG"])
    pix_map = get_sample_pix_map(ps_img.data)
    valid_minimal = MockDeferredDatasetHandle(
        ps_img.catalog["SPHERExRefID", "ra", "dec", "shape_r", "ab_ratio", "phi", "sersic"])
    invalid_srccat = MockDeferredDatasetHandle(fake_invalid_cat())

    # Setup
    small_edb_config = EstimateDiffuseBackgroundConfig(box_size=(5, 5))  # test image is 15x15

    # IRSA tool mode
    minimal_pfp_config = PerformForcedPhotometryConfig(
        hips_order=12,
        cat_phot_cols=["SPHERExRefID", "ra", "dec", "shape_r", "ab_ratio", "phi", "sersic"])
    minimal_l3a_config = L3aConfig(
        estimate_diffuse_background=small_edb_config,
        perform_forced_photometry=minimal_pfp_config)
    minimal_l3a_task = L3aTask(config=minimal_l3a_config)

    def test_valid_srccat_minimal(self):
        """Test L3a run method with valid input source catalog, including
        minimal columns."""
        self.ps_img.header["FINAST"] = 0

        output = self.minimal_l3a_task.run(spximg=self.ps_img,
                                           srccat=self.valid_minimal,
                                           solid_angle_pixel_map=self.pix_map)
        self.assertIsInstance(output, Struct)
        assert len(output.phot_meas.keys()) == 3  # test grouping

    def test_invalid_srccat(self):
        """Test L3a run method with invalid input source catalog."""
        self.ps_img.header["FINAST"] = 0

        with self.assertRaises(ValueError):
            self.minimal_l3a_task.run(spximg=self.ps_img,
                                      srccat=self.invalid_srccat,
                                      solid_angle_pixel_map=self.pix_map)

    def test_no_finast(self):
        """Test L3a run method with level 2 image without FINAST keyword.
        """
        if "FINAST" in self.ps_img.header:
            self.ps_img.header.remove("FINAST")

        with (self.assertLogs(self.minimal_l3a_task.log.name, level=logging.WARNING) as captured):
            result = self.minimal_l3a_task.run(spximg=self.ps_img,
                                               srccat=self.valid_minimal,
                                               solid_angle_pixel_map=self.pix_map)
            self.assertTrue(result.phot_meas is None)
        # Check for warning message
        self.assertEqual(captured.records[0].getMessage(),
                         "Input level 2 image failed fine astrometry; skipping level 3 processing.")

    def test_finast_fail(self):
        """Test L3a run method with level 2 image that failed fine astrometry.
        """
        self.ps_img.header["FINAST"] = 1

        with (self.assertLogs(self.minimal_l3a_task.log.name, level=logging.WARNING) as captured):
            result = self.minimal_l3a_task.run(spximg=self.ps_img,
                                               srccat=self.valid_minimal,
                                               solid_angle_pixel_map=self.pix_map)
            self.assertTrue(result.phot_meas is None)
        # Check for warning message
        self.assertEqual(captured.records[0].getMessage(),
                         "Input level 2 image failed fine astrometry; skipping level 3 processing.")
