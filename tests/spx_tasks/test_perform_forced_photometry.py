# Standard library imports
from pathlib import Path
import logging
import unittest
import tempfile
from unittest.mock import patch

# Related third party imports
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage, spherex_image_reader
from spx_tasks import PerformForcedPhotometryConfig, PerformForcedPhotometryTask
from spx_tests import get_sample_global_bkgd, get_sample_pix_map
from spx_utils import SourcePipelineClass

test_root = Path(__file__).parent.parent


def check_order(output_tab, input_tab):
    """Check that relative order of output flux table rows is the same as the
    relative order of input source catalog rows."""
    shared_cols = {
        "coord_src": {
            "final_name": "external_survey",
        },
        "external_source_id": {
            "final_name": "external_source_id",
        },
        "parallax": {
            "final_name": "parallax",
        },
        "pmdec": {
            "final_name": "pm_dec",
        },
        "pmra": {
            "final_name": "pm_ra",
        },
        "spherex_class": {
            "final_name": "spherex_class",
        },
        "ref_dec": {
            "final_name": "dec"
        },
        "ref_ra": {
            "final_name": "ra",
        }
    }

    input_cols = list(shared_cols.keys())
    output_cols = [shared_cols[col]["final_name"] for col in input_cols]

    for in_col, out_col in zip(input_cols, output_cols):
        for idx, unique_id in enumerate(output_tab["source_id"]):
            output_val = output_tab[out_col][idx]
            input_val = input_tab[in_col][input_tab["SPHERExRefID"] == unique_id]
            msg = f"Shared column {out_col} does not match original for source_id {unique_id}."

            if in_col in ["parallax", "pmdec", "pmra"]:
                # For proper motions and parallaxes, native format is float64
                # and output format is float32
                np.testing.assert_allclose(input_val, output_val, atol=1e-15)
            else:
                # Use strict comparison for other columns
                assert input_val == output_val, msg
    return


def prepare_mock_lvf_fits(array_id=1, shape=(2040, 2040)):
    mock_data = np.random.uniform(0.75, 4.6, shape).astype(np.float32)
    tmpdir = tempfile.TemporaryDirectory()
    fits_path = Path(tmpdir.name) / f"20250202_SSDC_CENTER_Band{array_id}.FITS"
    fits.writeto(fits_path, mock_data, overwrite=True)
    return tmpdir, fits_path


def get_sample_file(type: str) -> SPHERExImage:
    if type == "point source":
        sample_file = test_root / "data/photometry_point_sources.fits"  # NOTE: image shape is 15x15
    elif type == "galaxy":
        sample_file = test_root / "data/photometry_galaxy.fits"   # NOTE: image shape is 10x10
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def get_sample_spectral_wcs(data: np.ndarray) -> fits.HDUList:
    primary_hdu = fits.PrimaryHDU()
    cwave_hdu = fits.ImageHDU(data=np.full(data.shape, 0.123), name="CWAVE")
    cband_hdu = fits.ImageHDU(data=np.full(data.shape, 0.001), name="CBAND")
    return fits.HDUList([primary_hdu, cwave_hdu, cband_hdu])


def get_num_meas(cat_dict) -> int:
    nmeas = 0
    for hips_idx in cat_dict.keys():
        nmeas += len(cat_dict[hips_idx])
    return nmeas


def check_cols(flux_tab: Table, global_bkgd: bool, pipe_pass_thru: bool, tool_pass_thru: bool):
    """Check photometry measurement table column formatting.

    Parameters
    ----------
    flux_tab : `astropy.table.Table`
        Table of photometry measurements.
    global_bkgd : `bool`
        True if global background estimate was subtracted, False if local
        background estimates were subtracted.
    pipe_pass_thru : `bool`
        True if "pipeline"/"production"-mode pass-through columns
        ("spherex_class", "pm_ra", "pm_dec", "parallax", "external_survey",
        "external_source_id") were added to the table.
    tool_pass_thru : `bool`
        True if "spectrophotometry tool"-mode pass-through columns ("name")
        were added to the table.
    """
    # Check that expected columns exist and have proper units
    expected = np.array([
        ("source_id", None),
        ("ra", u.deg),
        ("dec", u.deg),
        ("x_image", u.pixel),
        ("y_image", u.pixel),
        ("mjd", u.day),
        ("lambda", u.um),
        ("lambda_width", u.um),
        ("flux", u.uJy),
        ("flux_err", u.uJy),
        ("flux_bkg", u.MJy/u.sr),
        ("local_bkg_flg", None),
        ("ext_flg", None),
        ("flags", None),
        ("fit_ql", None),
        ("deep_flg", None),
        ("lvf_id", None),
        ("det_id", None)
    ])
    if pipe_pass_thru:
        np.append(expected, [("spherex_class", None), ("pm_ra", u.mas/u.year), ("pm_dec", u.mas/u.year),
                             ("parallax", u.mas), ("external_survey", None), ("external_source_id", None)])
    if tool_pass_thru:
        np.append(expected, [("name", None)])
    for exp_name, exp_unit in zip(expected[:, 0], expected[:, 1]):
        assert exp_name in flux_tab.colnames, f"Column {exp_name} not found in output photometry table."
        assert flux_tab[exp_name].unit == exp_unit, f"Column {exp_name} does not have units {exp_unit}."
        # Verify string column is not blank
        if exp_name == "external_source_id":
            assert np.all(flux_tab[exp_name] != ""), "Column external_source_id is blank for some sources."

    # Verify flux_bkg and local_bkg_flg are properly populated
    if global_bkgd:
        zero_bkg_msg = "Column flux_bkg is zero for all sources in the global background case."
        assert np.sum(flux_tab["flux_bkg"]) != 0, zero_bkg_msg
        global_msg = "Column local_bkg_flg is True for 1+ sources in the global background case."
        assert np.count_nonzero(flux_tab["local_bkg_flg"]) == 0, global_msg
    else:
        local_msg = "Column local_bkg_flg is False for 1+ sources in the local background case."
        assert np.all(flux_tab["local_bkg_flg"]), local_msg


class TestPerformForcedPhotometry(unittest.TestCase):
    """Test Perform Forced Photometry Task."""
    def setUp(self):
        # Point sources
        ps_img = spherex_image_reader(get_sample_file("point source"), hdu_layers=["PSF", "CATALOG"])
        self.ps_img = ps_img
        self.ps_bkgd = get_sample_global_bkgd(ps_img.data)
        self.ps_null_bkgd = get_sample_global_bkgd(ps_img.data, null=True)
        self.ps_pix_map = get_sample_pix_map(ps_img.data)
        self.ps_spectral_wcs = get_sample_spectral_wcs(ps_img.data)

        # Galaxy
        gal_img = spherex_image_reader(get_sample_file("galaxy"), hdu_layers=["PSF", "CATALOG"])
        self.gal_img = gal_img
        self.gal_bkgd = get_sample_global_bkgd(gal_img.data)
        self.gal_null_bkgd = get_sample_global_bkgd(gal_img.data, null=True)
        self.gal_pix_map = get_sample_pix_map(gal_img.data)
        self.gal_spectral_wcs = get_sample_spectral_wcs(gal_img.data)

    def test_dictionary_output(self):
        tmpdir, _ = prepare_mock_lvf_fits()
        with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
            """Test photometry dictionary output (key per HEALPix index)."""
            # NOTE: Only test with the point source sample image since it
            # includes multiple sources.
            # HIPS order 6 + good global background solution
            config1 = PerformForcedPhotometryConfig(hips_order=6, pass_thru_cols=["name"])
            task1 = PerformForcedPhotometryTask(config=config1)

            # Add a few non-finite pixels to check pixel filtering
            self.ps_img.data[0, 0] = np.inf
            self.ps_img.data[9, 14] = np.nan
            self.ps_img.data[14, 14] = np.nan

            # Check for filtered pixel logs and metadata
            with self.assertLogs(task1.log.name, level=logging.INFO) as captured:
                ps_output1 = task1.run(self.ps_img, self.ps_img.catalog,
                                       self.ps_bkgd, self.ps_pix_map, self.ps_spectral_wcs)
            logs = [rec.getMessage() for rec in captured.records]
            assert "Assigned 3 non-finite-valued image pixels to 0." in logs
            assert "version_level2" in (list(ps_output1.phot_meas.values())[0]).meta.keys()
            assert "version_phot_meas" in (list(ps_output1.phot_meas.values())[0]).meta.keys()

            # Check expected output and formatting
            assert len(ps_output1.phot_meas.keys()) == 1  # test grouping
            assert get_num_meas(ps_output1.phot_meas) == len(self.ps_img.catalog)  # check for all sources
            check_cols(list(ps_output1.phot_meas.values())[0], global_bkgd=True, pipe_pass_thru=False,
                       tool_pass_thru=True)

            # HIPS order 12 + no good global background solution
            config2 = PerformForcedPhotometryConfig(hips_order=12)
            task2 = PerformForcedPhotometryTask(config=config2)
            ps_output2 = task2.run(self.ps_img, self.ps_img.catalog, self.ps_null_bkgd, self.ps_pix_map)
            assert len(ps_output2.phot_meas.keys()) == 3  # test grouping
            assert get_num_meas(ps_output2.phot_meas) == len(self.ps_img.catalog)  # check for all sources
            check_cols(list(ps_output2.phot_meas.values())[0], global_bkgd=False, pipe_pass_thru=True,
                       tool_pass_thru=False)
            check_order(output_tab=list(ps_output2.phot_meas.values())[0], input_tab=self.ps_img.catalog)

    def test_combined_output(self):
        """Test photometry single-table output (for a full image)."""
        config_ps = PerformForcedPhotometryConfig(hips_order=6, combinedOutput=True)
        task_ps = PerformForcedPhotometryTask(config=config_ps)
        # Point sources + good global background solution
        ps_output1 = task_ps.run(self.ps_img, self.ps_img.catalog, self.ps_bkgd, self.ps_pix_map)
        assert len(ps_output1.phot_meas_combined) == len(self.ps_img.catalog)  # check for all sources
        check_cols(ps_output1.phot_meas_combined, global_bkgd=True, pipe_pass_thru=True, tool_pass_thru=False)
        check_order(output_tab=ps_output1.phot_meas_combined, input_tab=self.ps_img.catalog)
        # Point sources + no good global background solution
        ps_output2 = task_ps.run(self.ps_img, self.ps_img.catalog, self.ps_null_bkgd, self.ps_pix_map)
        assert len(ps_output2.phot_meas_combined) == len(self.ps_img.catalog)  # check for all sources
        check_cols(ps_output2.phot_meas_combined, global_bkgd=False, pipe_pass_thru=True,
                   tool_pass_thru=False)
        check_order(output_tab=ps_output2.phot_meas_combined, input_tab=self.ps_img.catalog)
        assert "version_level2" in ps_output2.phot_meas_combined.meta.keys()
        assert "version_phot_meas" in ps_output2.phot_meas_combined.meta.keys()

        config_gal = PerformForcedPhotometryConfig(hips_order=6, combinedOutput=True, cut_side_length=2,
                                                   frame_width=1, bkg_side_length=4, bkg_box_size=1,
                                                   pass_thru_cols=["name"])
        task_gal = PerformForcedPhotometryTask(config=config_gal)
        # Extended source + good global background solution
        gal_output1 = task_gal.run(self.gal_img, self.gal_img.catalog,
                                   self.gal_bkgd, self.gal_pix_map, self.gal_spectral_wcs)
        assert len(gal_output1.phot_meas_combined) == len(self.gal_img.catalog)  # check for all sources
        check_cols(gal_output1.phot_meas_combined, global_bkgd=True, pipe_pass_thru=False,
                   tool_pass_thru=True)
        # Extended source + no good global background solution
        gal_output2 =\
            task_gal.run(self.gal_img, self.gal_img.catalog, self.gal_null_bkgd, self.gal_pix_map)
        assert len(gal_output2.phot_meas_combined) == len(self.gal_img.catalog)  # check for all sources
        check_cols(gal_output2.phot_meas_combined, global_bkgd=False, pipe_pass_thru=False,
                   tool_pass_thru=True)

    def test_config_checks(self):
        """Tests that config checks work for faulty setups."""
        # Frame width > cut side length
        with self.assertRaises(ValueError):
            PerformForcedPhotometryConfig(cut_side_length=5, frame_width=10)

        # Background cut side length < image cut side length
        with self.assertRaises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=4, cut_side_length=5)

        # Difference in BG cut side length and image cut side length is odd
        with self.assertRaises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=12, cut_side_length=5)

        # Background box size is not a factor of background side length
        with self.assertRaises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=15, bkg_box_size=4)

        # Non-"ALL" class filter without "spherex_class" column
        with self.assertRaises(ValueError):
            PerformForcedPhotometryConfig(pass_thru_cols=[], class_filter=SourcePipelineClass.COSMOLOGY)


if __name__ == "__main__":
    unittest.main()
