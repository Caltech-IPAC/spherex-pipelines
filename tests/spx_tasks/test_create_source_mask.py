import unittest
from unittest.mock import MagicMock
import numpy as np
from astropy.table import Table
from spx_data import SPHERExImage
from spx_tasks import CreateSourceMaskTask, CreateSourceMaskConfig
from spx_tests.sample_data import get_sample_level1b_img
import astropy.units as u


class TestCreateSourceMaskTask(unittest.TestCase):
    def setUp(self):
        """Set up mock data for testing."""
        # Mock SPHERExImage
        self.spximg = get_sample_level1b_img(seed_number=42, bad_flags=True, no_refs=True)
        self.spximg.data = np.random.uniform(low=10, high=1000, size=self.spximg.data.shape)
        # Mock Absolute Gain Matrix
        gm_data = np.ones(self.spximg.data.shape, dtype=np.float32)
        self.gain_matrix = SPHERExImage(gm_data, unit=u.Unit("MJy / sr") / u.Unit("electron / s"))

        # Mock catalog
        nx, ny = self.spximg.data.shape
        self.catalog = Table({
            "ix": [10, nx//2, nx - 10],
            "iy": [10, nx//2, ny - 10],
            "ra": [150.0, 151.0, 152.0],
            "dec": [-10.0, -10.5, -11.0],
            "mag_det1": [2.0, 3.0, 4.0],  # Brightness for sources
            "dQ": [80000, 70000, 10000],  # High enough values to exceed thresholds
        })

        # Mock crosstalk matrix
        self.xtalk_matrix = MagicMock()
        self.xtalk_matrix.data = np.random.rand(4, 4, 4)

        # Configuration
        self.config = CreateSourceMaskConfig()
        self.config.minmag = 0.0
        self.config.maxmag = 20.0
        self.config.dmag = 1.0
        # Lower threshold for crosstalk
        self.config.limiting_xtalk_current = 400.0
        # Increase likelihood of coupling
        self.config.xtalk_amplitude_thresh = 1e-4
        self.config.ghost_radius = 2
        self.config.edge_mag_limit = 11.0
        self.config.edge_offset_min = 10.0
        self.config.edge_offset_max = 45.0
        self.config.dbs_params_3 = [-0.0051, 0.0013, -7.4, 0.0, 255.0, 0.0]
        self.config.dbs_params_4 = [-0.0040, 0.0013, -8.9, 0.0, 255.0, 0.0]
        self.config.dbs_params_5 = [-0.0042, 0.0005, 0.1, 0.0, 255.0, 0.0]
        self.config.dbs_params_6 = [-0.0043, 0.0013, 9.4, 0.0, 225.0, 0.0]

    def test_mask_sources(self):
        """Test the mask_sources function."""
        from spx_tasks.create_source_mask import mask_sources

        params_dict = {
            1: self.config.radius_model_params_det1,
            2: self.config.radius_model_params_det2,
            3: self.config.radius_model_params_det3,
            4: self.config.radius_model_params_det4,
            5: self.config.radius_model_params_det5,
            6: self.config.radius_model_params_det6,
        }

        source_mask = mask_sources(
            spximg=self.spximg,
            cat=self.catalog,
            minmag=self.config.minmag,
            maxmag=self.config.maxmag,
            dmag=self.config.dmag,
            params_dict=params_dict,
            plot_this=False,
        )

        # Assertions
        self.assertIsInstance(source_mask, np.ndarray)

    def test_mask_crosstalk(self):
        """Test the mask_crosstalk function."""
        from spx_tasks.create_source_mask import mask_crosstalk

        crosstalk_mask = mask_crosstalk(
            spximg=self.spximg,
            cat=self.catalog,
            limiting_xtalk_current=self.config.limiting_xtalk_current,
            t_int=self.spximg.header["TINT"],
            xtalk_coupling_matrix=self.xtalk_matrix.data,
            xtalk_amplitude_thresh=self.config.xtalk_amplitude_thresh,
            abs_gain_matrix=self.gain_matrix
        )

        self.assertIsInstance(crosstalk_mask, np.ndarray)
        # Ensure some mask applied
        self.assertTrue(np.any(crosstalk_mask > 0))

    def test_run_task(self):
        """Test the full pipeline task."""
        task = CreateSourceMaskTask(config=self.config)

        # Run the task
        output = task.run(
            spximg=self.spximg,
            srccat=self.catalog,
            ssocat=None,
            # xtalk_coupling_matrix=self.xtalk_matrix,
            abs_gain_matrix=self.gain_matrix,
            plot=False,
            verbose=False,
        )

        # Validate the output
        spximg_out = output.spximg_out
        self.assertIsInstance(spximg_out, SPHERExImage)

    def test_mask_ghosts(self):
        """Test the mask_ghosts function."""
        from spx_tasks.create_source_mask import mask_ghosts
        self.spximg.setArrayId(array_id=3)
        # Mock catalog
        nx, ny = self.spximg.data.shape
        self.catalog = Table({
            "ix": [10, nx//2, nx - 10],
            "iy": [10, nx//2, ny - 10],
            "mag_det3": [13.0, 5.0, 15.0],  # Brightness for sources
            "dQ": [800, 700, 100],  # High enough values to exceed thresholds
            "WISE_W1": [2e8, 1e8, 3e8],  # WISE W1 flux in μJy
            "detector": [3, 4, 5],  # Detectors for testing GHOST_FPA
        })
        # DBS parameters for each detector
        dbs_params = {
            3: self.config.dbs_params_3,
            4: self.config.dbs_params_4,
            5: self.config.dbs_params_5,
            6: self.config.dbs_params_6,
        }

        ghost_mask = mask_ghosts(
            spximg=self.spximg,
            cat=self.catalog,
            ghost_radius=self.config.ghost_radius,
            dbs_params=dbs_params,
            edge_mag_limit=self.config.edge_mag_limit,
            edge_offset=(self.config.edge_offset_min, self.config.edge_offset_max),
            plot_this=False,
        )

        # Assertions
        self.assertIsInstance(ghost_mask, np.ndarray)
        self.assertTrue(np.any(ghost_mask > 0), "Ghost mask is empty.")


if __name__ == "__main__":
    unittest.main()
