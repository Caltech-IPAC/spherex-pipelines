import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from astropy.table import Table
from spx_data import SPHERExImage, Flag
from spx_tasks import CreateSourceMaskTask, CreateSourceMaskConfig
from spx_tests.sample_data import get_sample_level1b_img
import astropy.units as u
from spx_tasks.create_source_mask import mask_dbs_ghosts, dqa_check_dbs_ghost_ext_corners


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
        """Test the mask_sources function (blended-exponential only)."""
        from spx_tasks.create_source_mask import mask_sources

        blended_params_dict = {
            1: self.config.blended_radius_params_det1,
            2: self.config.blended_radius_params_det2,
            3: self.config.blended_radius_params_det3,
            4: self.config.blended_radius_params_det4,
            5: self.config.blended_radius_params_det5,
            6: self.config.blended_radius_params_det6,
        }

        # Ensure we’re testing a known band
        self.spximg.setArrayId(array_id=1)

        with patch(
            "spx_tasks.create_source_mask.blended_exponential_radius",
            return_value=3.0,  # constant radius makes behavior predictable
        ) as mock_blended:
            source_mask = mask_sources(
                spximg=self.spximg,
                cat=self.catalog,
                minmag=self.config.minmag,
                maxmag=self.config.maxmag,
                dmag=self.config.dmag,
                blended_params_dict=blended_params_dict,
                plot_this=False,
            )

        self.assertIsInstance(source_mask, np.ndarray)

        # New check: blended model was actually used
        self.assertGreater(mock_blended.call_count, 0)

        # Sanity check it was called with mags from the current band
        band = self.spximg.getArrayId()
        mags = np.array(self.catalog[f"mag_det{band}"])
        # confirm first positional arg of first call is one of those mags
        first_called_mag = float(mock_blended.call_args[0][0])
        self.assertIn(first_called_mag, mags.tolist())

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

    def test_mask_dbs_ghosts(self):
        """Test the mask_dbs_ghosts function."""
        # Override the setUp image with a larger one
        mock_data = np.zeros((512, 512))
        mock_header = self.spximg.meta.copy()
        mock_header["JACTIVE"] = 512
        mock_header["KACTIVE"] = 512
        self.spximg.data = mock_data
        self.spximg.meta = mock_header

        # Use Band 6: Pair (6, 3), H-proxy is 'mag_det3'
        self.spximg.setArrayId(array_id=6)

        # Define the (x, y) coordinates our mock WCS will return.
        mock_xs = np.array([100.0, 100.0, 100.0, 100.0])
        mock_ys = np.array([100.0, 250.0, 3000.0, 100.0])

        # The B_lim for Band 6 is 10000.0 Jy
        def mock_mag_to_flux(mags):
            fluxes = []
            for m in mags:
                if m == -2.0:
                    fluxes.append(20000.0)  # > 10000 (Bright)
                elif m == -1.5:
                    fluxes.append(15000.0)  # > 10000 (Bright)
                elif m == 20.0:
                    fluxes.append(0.1)  # < 10000 (Faint)
                else:
                    fluxes.append(np.nan)
            return np.array(fluxes)

        # For Band 6:
        # H-proxy is 'mag_det3'
        # 'this' band is 'mag_det6'
        self.catalog = Table({
            'ra': [1.0, 2.0, 3.0, 4.0],  # Dummy values
            'dec': [1.0, 2.0, 3.0, 4.0],  # Dummy values
            'mag_det3': [-2.0, -1.5, -2.0, 20.0],
            'mag_det6': [8.0, np.nan, 8.0, 10.0]
        })

        # These are the real parameters the function expects,
        # which our 512x512 image is set up to test.
        mock_params_edge = {
            3: [100.0, -0.0051, 0.0013, -5.0, 0.0, 0.0, 225.0, 52],
            4: [100.0, 0.0, 0.0, 10.0, 0.0, 0.0, 221.0, 52],
        }
        mock_params_main = {
            4: [3000.0, 0.0, 0.0, 5.0, 0.0, 0.0, 221.0, 52],
            5: [10000.0, -0.0042, 0.0005, 0.1, 0.0, 0.0, 225.0, 52],
            6: [10000.0, -0.0043, 0.0013, 9.4, 0.0, 0.0, 225.0, 52],  # testing this
        }
        mock_roi_edge = {
            3: [-52, 2092, -250, 175],
            4: [-52, 2092, 1760, 2040],
        }
        mock_roi_main = {
            4: [-52, 2092, 52, 2261],
            5: [-52, 2092, 52, 2265],
            6: [-52, 2092, 52, 2265],  # testing this
        }
        mock_bleed_flux_thresh = 3000.0
        mock_bleed_roi = [-52, 2092, 1920, 2048]

        # Patch WCS and flux conversion
        with (
            patch.object(
                self.spximg.wcs,
                "world_to_pixel_values",
                return_value=(mock_xs, mock_ys),
            ) as mock_wcs_call,
            patch(
                "spx_tasks.create_source_mask.refcat_utils.mag_to_flux",
                side_effect=mock_mag_to_flux,
            ) as mock_flux_call,
        ):
            ghost_mask = mask_dbs_ghosts(
                spximg=self.spximg,
                cat=self.catalog,
                params_edge=mock_params_edge,
                params_main=mock_params_main,
                roi_edge=mock_roi_edge,
                roi_main=mock_roi_main,
                bleed_flux_thresh=mock_bleed_flux_thresh,
                bleed_roi=mock_bleed_roi,
            )

        # Check that our WCS mock was called once.
        mock_wcs_call.assert_called_once()
        args, kwargs = mock_wcs_call.call_args
        np.testing.assert_array_equal(args[0], self.catalog['ra'])
        np.testing.assert_array_equal(args[1], self.catalog['dec'])

        # Check that our flux mock was called once
        mock_flux_call.assert_called_once()
        np.testing.assert_array_equal(mock_flux_call.call_args[0][0], self.catalog['mag_det3'])

        # Assertions
        self.assertIsInstance(ghost_mask, np.ndarray)
        self.assertEqual(ghost_mask.dtype, np.int32)
        self.assertEqual(ghost_mask.shape, self.spximg.data.shape)

        # Check that the correct flags were set
        total_ghost_pixels = np.count_nonzero((ghost_mask & Flag.GHOST) > 0)
        total_ghost_ext_pixels = np.count_nonzero((ghost_mask & Flag.GHOST_EXT) > 0)

        # Source 1 should create GHOST pixels
        self.assertGreater(total_ghost_pixels, 0, "No GHOST pixels were flagged.")

        # No source should create GHOST_EXT
        self.assertEqual(total_ghost_ext_pixels, 0, "GHOST_EXT pixels were flagged unexpectedly.")

    def test_dqa_check_dbs_ghost_ext_corners_pass(self):
        """Test that dqa_check_dbs_ghost_ext_corners passes when there are
        external sources in both edge ROI corners.
        """
        # Use Band 3 (detector 3) since it has an edge ROI defined
        self.spximg.setArrayId(array_id=3)
        ny, nx = self.spximg.data.shape

        # Use the same kind of ROI as the config for band 3:
        roi_edge_band = [-52, 2092, -250, 175]  # [x_min, x_max, y_min, y_max]

        # Two catalog sources (no mag cut applied here)
        catalog = Table({
            "ra": [1.0, 2.0],
            "dec": [1.0, 2.0],
        })

        # Construct pixel coordinates that:
        #  - Lie inside the ROI corners
        #  - Are *external* to the detector (so they count as GHOST_EXT candidates)
        # First corner near (x_min, y_min): external via xi < 0
        xi1 = -10.0
        yi1 = -200.0  # within [y_min, y_min + corner_box_size] = [-250, -200] for size=50

        # Second corner near (x_max, y_max): external via xi >= nx
        # x_max = 2092, so "x_max - 50" = 2042; we choose 2050 ∈ [2042, 2092]
        xi2 = 2050.0
        yi2 = 150.0   # within [y_max - 50, y_max] = [125, 175] for size=50

        mock_ix = np.array([xi1, xi2])
        mock_iy = np.array([yi1, yi2])

        with (
            patch.object(
                self.spximg.wcs,
                "world_to_pixel_values",
                return_value=(mock_ix, mock_iy),
            ) as mock_wcs_call,
            patch(
                "spx_tasks.create_source_mask.log_dqa_result"
            ) as mock_log_dqa,
        ):
            result = dqa_check_dbs_ghost_ext_corners(
                spximg=self.spximg,
                srccat=catalog,
                roi_edge_band=roi_edge_band,
                corner_box_size=50.0,
            )

        # WCS and logging should each be called once
        mock_wcs_call.assert_called_once()
        mock_log_dqa.assert_called_once()

        # We expect a PASS because we have one external source in each corner box
        self.assertEqual(result["result"], "PASS")
        self.assertIn("found", result["description"])
        self.assertIn("corner", result["description"])

    def test_dqa_check_dbs_ghost_ext_corners_fail_missing_one_corner(self):
        """Test that dqa_check_dbs_ghost_ext_corners fails when only one
        corner has external sources.
        """
        self.spximg.setArrayId(array_id=3)
        roi_edge_band = [-52, 2092, -250, 175]

        catalog = Table({
            "ra": [1.0],
            "dec": [1.0],
        })

        # Only first corner populated; second corner empty
        xi = np.array([-10.0])   # inside first corner box in x
        yi = np.array([-200.0])  # inside first corner box in y

        with (
            patch.object(
                self.spximg.wcs,
                "world_to_pixel_values",
                return_value=(xi, yi),
            ) as mock_wcs_call,
            patch(
                "spx_tasks.create_source_mask.log_dqa_result"
            ) as mock_log_dqa,
        ):
            result = dqa_check_dbs_ghost_ext_corners(
                spximg=self.spximg,
                srccat=catalog,
                roi_edge_band=roi_edge_band,
                corner_box_size=50.0,
            )

        mock_wcs_call.assert_called_once()
        mock_log_dqa.assert_called_once()

        self.assertEqual(result["result"], "FAIL")
        self.assertIn("no external sources found", result["description"])

    def test_run_task_wcs_failure_finast_nonzero_skips_masks(self):
        """If masking throws WCS error and FINAST!=0,
           task should skip masking and return cleanly."""
        task = CreateSourceMaskTask(config=self.config)

        # Ensure FINAST != 0 to activate the skip branch
        self.spximg.header["FINAST"] = 1

        # Make skipping meaningful
        self.config.mask_ghosts = True
        self.config.mask_xtalk = False

        with patch(
            "spx_tasks.create_source_mask.mask_sources",
            side_effect=RuntimeError("boom"),
        ) as mock_mask_sources, patch(
            "spx_tasks.create_source_mask.mask_crosstalk"
        ) as mock_mask_xtalk, patch(
            "spx_tasks.create_source_mask.mask_dbs_ghosts"
        ) as mock_mask_ghosts:
            output = task.run(
                spximg=self.spximg,
                srccat=self.catalog,
                abs_gain_matrix=self.gain_matrix,
                plot=False,
                verbose=False,
            )

        spximg_out = output.spximg_out
        self.assertIsInstance(spximg_out, SPHERExImage)

        # Only mask_sources attempted; others should not run because we return early
        mock_mask_sources.assert_called_once()
        mock_mask_xtalk.assert_not_called()
        mock_mask_ghosts.assert_not_called()

        # No flags should have been added by this task
        self.assertEqual(int(np.count_nonzero((spximg_out.flags & Flag.SOURCE) > 0)), 0)
        self.assertEqual(int(np.count_nonzero((spximg_out.flags & Flag.CROSSTALK) > 0)), 0)
        self.assertEqual(int(np.count_nonzero((spximg_out.flags & Flag.GHOST) > 0)), 0)
        self.assertEqual(int(np.count_nonzero((spximg_out.flags & Flag.GHOST_FPA) > 0)), 0)
        self.assertEqual(int(np.count_nonzero((spximg_out.flags & Flag.GHOST_EXT) > 0)), 0)

        # Checkheaders
        self.assertEqual(spximg_out.header.get("PXSRCMSK", 0.0), 0.0)
        self.assertEqual(spximg_out.header.get("HIERARCH L2 N_SOURCE", 0), 0)
        self.assertEqual(spximg_out.header.get("HIERARCH L2 N_GHOST", 0), 0)
        self.assertEqual(spximg_out.header.get("HIERARCH L2 N_GHOST_EXT", 0), 0)

    def test_run_task_wcs_failure_finast_zero_reraises(self):
        """If masking throws and FINAST==0, the task should raise error,
           and treat as real bug."""
        task = CreateSourceMaskTask(config=self.config)

        # Ensure FINAST == 0
        self.spximg.header["FINAST"] = 0
        # Normal configs
        self.config.mask_xtalk = False
        self.config.mask_ghosts = True

        with patch(
            "spx_tasks.create_source_mask.mask_sources",
            side_effect=RuntimeError("fail"),
        ) as mock_mask_sources:
            with self.assertRaises(RuntimeError):
                task.run(
                    spximg=self.spximg,
                    srccat=self.catalog,
                    abs_gain_matrix=self.gain_matrix,
                    plot=False,
                    verbose=False,
                )

        mock_mask_sources.assert_called_once()


if __name__ == "__main__":
    unittest.main()
