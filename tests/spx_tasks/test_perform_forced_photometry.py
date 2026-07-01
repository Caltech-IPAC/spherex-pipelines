# Standard library imports
from pathlib import Path
import logging
import tempfile

# Related third party imports
import numpy as np
import astropy.units as u
import lsst.pipe.base as pipeBase
import pytest
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage, spherex_image_reader
from spx_tasks import PerformForcedPhotometryConfig, PerformForcedPhotometryTask
from spx_tasks.perform_forced_photometry import get_deep_flag
from spx_tests import get_sample_global_bkgd, get_sample_pix_map
from spx_utils import SourcePipelineClass

test_root = Path(__file__).parent.parent


# Helper functions ============================================================
def check_pass_thru_order(output_tab: Table, input_tab: Table, pass_thru_cols: list[str]):
    """Check that pass-through columns shared between the input source catalog
    and the output flux table have the same relative row ordering.

    Parameters
    ----------
    output_tab : `astropy.table.Table`
        Table of photometry measurements.
    input_tab : `astropy.table.Table`
        Input source catalog.
    pass_thru_cols : `list` of `str`
        List of input catalog columns passed through to the photometry table.
    """
    supported_pass_through = ["spherex_class", "name"]
    # Check that values match for each source ID
    for col in pass_thru_cols:
        if col in supported_pass_through:
            for idx, unique_id in enumerate(output_tab["source_id"]):
                try:
                    output_val = output_tab[col][idx]
                    input_val = input_tab[col][input_tab["SPHERExRefID"] == unique_id]
                    msg = f"Shared column '{col}' does not match original for source_id {unique_id}."
                    assert input_val == output_val, msg
                except KeyError:
                    raise ValueError(f"Pass-through column '{col}' not present in both input source catalog"
                                     f" and output photometry measurement table. Input columns:"
                                     f" {input_tab.colnames}, output columns: {output_tab.colnames}.")
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


def check_cols(flux_tab: Table, global_bkgd: bool, pass_thru_cols: list[str]):
    """Check photometry measurement table column formatting.

    Parameters
    ----------
    flux_tab : `astropy.table.Table`
        Table of photometry measurements.
    global_bkgd : `bool`
        True if global background estimate was subtracted, False if local
        background estimates were subtracted.
    pass_thru_cols : `list` of `str`
        List of input catalog columns passed through to the photometry table.
    """
    # Start with minimal set of columns
    expected = [
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
    ]

    # Add pass-through columns if specified
    if len(pass_thru_cols) > 0:
        for col in pass_thru_cols:
            if col == "name":
                expected.insert(1, ("name", None))
            elif col == "spherex_class":
                expected.insert(1, ("spherex_class", None))

    # Check order, name, and units of columns
    for i, exp_col in enumerate(expected):
        assert exp_col[0] == flux_tab.colnames[i], \
            f"Column '{exp_col[0]}' not found or not at expected column index {i} in output photometry table."
        assert flux_tab[exp_col[0]].unit == exp_col[1], \
            f"Column '{exp_col[0]}' does not have units {exp_col[1]}."

    # Verify flux_bkg and local_bkg_flg are properly populated
    if global_bkgd:
        zero_bkg_msg = "Column 'flux_bkg' is zero for all sources in the global background case."
        assert np.sum(flux_tab["flux_bkg"]) != 0, zero_bkg_msg
        global_msg = "Column 'local_bkg_flg' is True for 1+ sources in the global background case."
        assert np.count_nonzero(flux_tab["local_bkg_flg"]) == 0, global_msg
    else:
        local_msg = "Column 'local_bkg_flg' is False for 1+ sources in the local background case."
        assert np.all(flux_tab["local_bkg_flg"]), local_msg


def _extract_refcat_from_srccat(srccat: Table) -> tuple[str | None, str | None]:
    """Return (run, version) from srccat.meta if present; otherwise return
    (None, None)."""
    um = dict(srccat.meta.get("user_metadata") or {})
    ver = um.get("refcat_version") or srccat.meta.get("refcat_version")
    run = um.get("refcat_run") or srccat.meta.get("refcat_run")
    return run, ver


def _assert_refcat_version_on_table_if_expected(out_tab: Table, expected_ver: str | None):
    """If expected_ver is provided, assert it is present in out_tab.meta."""
    if expected_ver:
        assert "refcat_version" in out_tab.meta, "Missing refcat_version in output table metadata."
        assert out_tab.meta["refcat_version"] == expected_ver, (
            f"refcat_version mismatch: got {out_tab.meta['refcat_version']!r}, expected {expected_ver!r}"
        )


def validate_output(input_catalog: Table, output: pipeBase.Struct, global_bkgd: bool,
                    pass_thru_cols: list[str]):
    """Validate photometry output against input catalog.

    Parameters
    ----------
    input_catalog : `astropy.table.Table`
        Input source catalog.
    output : `lsst.pipe.base.Struct`
        Output object containing photometry measurement table(s).
    global_bkgd : `bool`
        True if global background estimate was subtracted, False if local
        background estimates were subtracted.
    pass_thru_cols : `list` of `str`
        List of input catalog columns passed through to the photometry table.
    """
    # Determine output type and number of sources
    fields = output.getDict().keys()
    # Dictionary output
    if "phot_meas" in fields:
        msg = "Number of sources in output phot_meas tables does not match input catalog."
        n_src_output = get_num_meas(output.phot_meas)
        sample_tab = list(output.phot_meas.values())[0]
    # Combined output
    elif "phot_meas_combined" in fields:
        msg = "Number of sources in output phot_meas_combined table does not match input catalog."
        n_src_output = len(output.phot_meas_combined)
        sample_tab = output.phot_meas_combined

    # Determine expected number of sources
    if "for_l3phot" in input_catalog.colnames:
        expected_srcs = len(np.nonzero(input_catalog["for_l3phot"] == 1)[0])
    else:
        expected_srcs = len(input_catalog)

    # Verify number of sources matches input catalog
    assert n_src_output == expected_srcs, msg

    # Check column format and row ordering
    check_cols(sample_tab, global_bkgd, pass_thru_cols)
    if len(pass_thru_cols) > 0:
        check_pass_thru_order(output_tab=sample_tab, input_tab=input_catalog, pass_thru_cols=pass_thru_cols)

    # Check version metadata
    _, exp_ver = _extract_refcat_from_srccat(input_catalog)
    _assert_refcat_version_on_table_if_expected(sample_tab, exp_ver)
    assert "version_level2" in sample_tab.meta.keys()
    assert "version_phot_meas" in sample_tab.meta.keys()


# Fixtures ===================================================================
@pytest.fixture()
def point_source():
    ps_img = spherex_image_reader(get_sample_file("point source"), hdu_layers=["PSF", "CATALOG"])
    return dict(
        img=ps_img,
        global_bkgd=get_sample_global_bkgd(ps_img.data),
        local_bkgd=get_sample_global_bkgd(ps_img.data, null=True),
        pix_map=get_sample_pix_map(ps_img.data),
        spectral_wcs=get_sample_spectral_wcs(ps_img.data)
    )


@pytest.fixture()
def galaxy():
    gal_img = spherex_image_reader(get_sample_file("galaxy"), hdu_layers=["PSF", "CATALOG"])
    return dict(
        img=gal_img,
        global_bkgd=get_sample_global_bkgd(gal_img.data),
        local_bkgd=get_sample_global_bkgd(gal_img.data, null=True),
        pix_map=get_sample_pix_map(gal_img.data),
        spectral_wcs=get_sample_spectral_wcs(gal_img.data)
    )


# Test cases ==================================================================
class TestPerformForcedPhotometry():
    """Test Perform Forced Photometry Task."""
    def test_pixel_filtering_and_debug_options(self, point_source, caplog):
        """Test that non-finite and zero-valued pixels are properly filtered
        out before photometry. Test that config options used for debugging
        work."""
        config = PerformForcedPhotometryConfig(combinedOutput=True, gc_interval=2, return_fit=True)
        task = PerformForcedPhotometryTask(config=config)

        # Image layer
        point_source["img"].data[0, 0] = np.inf
        point_source["img"].data[9, 14] = np.nan
        point_source["img"].data[14, 14] = np.nan

        # Variance layer
        point_source["img"].uncertainty.array[5, 5] = np.inf
        point_source["img"].uncertainty.array[6, 6] = np.nan
        point_source["img"].uncertainty.array[7, 7] = 0.0

        with caplog.at_level(logging.INFO, logger=task.log.name):
            output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                              point_source["pix_map"], point_source["spectral_wcs"])

        # Check logs
        logs = caplog.messages
        # Pixel filtering
        assert "Assigned 3 non-finite-valued image pixels to 0." in logs
        assert "Assigned 2 non-finite-valued variance pixels to initial max variance value." in logs
        assert "Assigned 1 zero-valued variance pixels to initial mean variance value." in logs
        # Garbage collection
        gc_logs = [log for log in logs if "objects garbage collected" in log]
        assert len(gc_logs) == 5, f"Expected 5 garbage collection log entries, got {len(gc_logs)}."

        # Check for fit results
        assert len(list(output.tractor_fits.keys())) == 3, \
            f"Expected fit results for 3 windows with 1+ sources to photometer, got" \
            f" {len(list(output.tractor_fits.keys()))}."

    # NOTE: Only the point source sample includes 1+ sources that could be
    #       in different HEALPix.
    @pytest.mark.parametrize("hips_order, n_healpix_tabs, fake_high_pm", [
        (6, 1, False), (12, 3, False), (12, 3, True)])
    def test_healpix_ordering(self, point_source, hips_order, n_healpix_tabs, fake_high_pm):
        """Test that HEALPix indexing and ordering is correct."""
        if fake_high_pm and hips_order == 12:
            catalog = point_source["img"].catalog
            # Move source 3 (HiPS12 idx 80465472) corrected position to
            # the reference position of source 5 (HiPS12 idx 80465131)
            hi_pm_mask = catalog["SPHERExRefID"] == 3
            catalog["ra"][hi_pm_mask] = catalog["ref_ra"][catalog["SPHERExRefID"] == 5]
            catalog["dec"][hi_pm_mask] = catalog["ref_dec"][catalog["SPHERExRefID"] == 5]

        config = PerformForcedPhotometryConfig(hips_order=hips_order)
        task = PerformForcedPhotometryTask(config=config)
        ps_output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                             point_source["pix_map"], point_source["spectral_wcs"])
        assert len(ps_output.phot_meas.keys()) == n_healpix_tabs  # test grouping
        validate_output(point_source["img"].catalog, ps_output, global_bkgd=True,
                        pass_thru_cols=config.pass_thru_cols)

        if fake_high_pm and hips_order == 12:
            # Check for proper HEALPix assignment
            assert 3 in ps_output.phot_meas[80465472]["source_id"], (
                "High proper motion source not found in the tile corresponding to its reference position.")
            assert 3 not in ps_output.phot_meas[80465131]["source_id"], (
                "High proper motion source misplaced in the tile corresponding to its corrected position.")

            # Check that photometry was performed on the corrected position
            src3_mask = ps_output.phot_meas[80465472]["source_id"] == 3
            src3_ra = ps_output.phot_meas[80465472]["ra"][src3_mask]
            src3_dec = ps_output.phot_meas[80465472]["dec"][src3_mask]
            assert src3_ra == catalog["ref_ra"][catalog["SPHERExRefID"] == 5] and \
                   src3_dec == catalog["ref_dec"][catalog["SPHERExRefID"] == 5], (
                "High proper motion source not photometered at its corrected position.")

    @pytest.mark.parametrize("bkgd_type", ["global_bkgd", "local_bkgd"])
    def test_background_subtraction(self, point_source, bkgd_type):
        """Test that global and local background subtraction work."""
        config = PerformForcedPhotometryConfig(hips_order=6)
        task = PerformForcedPhotometryTask(config=config)
        ps_output = task.run(point_source["img"], point_source["img"].catalog, point_source[bkgd_type],
                             point_source["pix_map"], point_source["spectral_wcs"])
        validate_output(point_source["img"].catalog, ps_output, global_bkgd=(bkgd_type == "global_bkgd"),
                        pass_thru_cols=config.pass_thru_cols)

    @pytest.mark.parametrize("source_type, cut_side_length, frame_width, bkg_side_length, bkg_box_size",
                             [("point source", 5, 3, 9, 3), ("galaxy", 2, 1, 4, 1)])
    def test_source_fitting(self, source_type, cut_side_length, frame_width, bkg_side_length,
                            bkg_box_size, point_source, galaxy):
        """Test that point source and galaxy fitting work."""
        config = PerformForcedPhotometryConfig(
            hips_order=6, cut_side_length=cut_side_length, frame_width=frame_width,
            bkg_side_length=bkg_side_length, bkg_box_size=bkg_box_size)
        task = PerformForcedPhotometryTask(config=config)

        if source_type == "point source":
            inputs = point_source
        elif source_type == "galaxy":
            inputs = galaxy
        output = task.run(inputs["img"], inputs["img"].catalog, inputs["global_bkgd"],
                          inputs["pix_map"], inputs["spectral_wcs"])
        validate_output(inputs["img"].catalog, output, global_bkgd=True, pass_thru_cols=config.pass_thru_cols)

    # NOTE: 'spherex_class' and 'name' are the only supported pass-through
    #       columns with lookup parameters. We add 'pmra' to test that adding
    #       a non-lookup column still results in a standard output.
    @pytest.mark.parametrize(
        "pass_thru_cols",
        [["spherex_class"], ["name"], ["pmra"]])
    def test_pass_thru_columns_and_srccat_read(self, point_source, pass_thru_cols):
        """Test that pass-through columns are properly added to output."""
        config = PerformForcedPhotometryConfig(hips_order=6, merge_srccat=True,
                                               pass_thru_cols=pass_thru_cols)
        task = PerformForcedPhotometryTask(config=config)

        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"])
        validate_output(point_source["img"].catalog, output, global_bkgd=True,
                        pass_thru_cols=pass_thru_cols)

    def test_config_checks(self):
        """Tests that config checks work for faulty setups."""
        # Cut side length is a factor of # of active pixels
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(cut_side_length=7)

        # Frame width > cut side length
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(cut_side_length=5, frame_width=10)

        # Background cut side length < image cut side length
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=4, cut_side_length=5)

        # Difference in BG cut side length and image cut side length is odd
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=12, cut_side_length=5)

        # Background box size is not a factor of background side length
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(bkg_side_length=15, bkg_box_size=4)

        # Non-"ALL" class filter without "spherex_class" column
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(pass_thru_cols=[], class_filter=SourcePipelineClass.COSMOLOGY)

        # Add pass-through columns without specifying any
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(pass_thru_cols=[], merge_srccat=True)

    @pytest.mark.parametrize("keyword, flag_str, is_deep", [
        ("SFLAG", "deep_north", True),
        ("SPS_FLAGS", "deep_south", True),
        (None, None, False)
    ])
    def test_deep_flag(self, point_source, keyword, flag_str, is_deep):
        """Test that deep field flag is correctly read from header."""
        if keyword is not None:
            point_source["img"].header[keyword] = flag_str
        else:
            if "SFLAG" in point_source["img"].header:
                point_source["img"].header.remove("SFLAG")
            elif "SPS_FLAGS" in point_source["img"].header:
                point_source["img"].header.remove("SPS_FLAGS")

        deep_flag = get_deep_flag(point_source["img"].header)
        assert deep_flag == is_deep

    @pytest.mark.parametrize("class_bit, n_phot_sources", [
        (SourcePipelineClass.ALL, 4),
        (SourcePipelineClass.COSMOLOGY, 3),
        (SourcePipelineClass.CLUSTER, 0)])
    def test_class_filtering(self, point_source, class_bit, n_phot_sources):
        """Test that source class filtering works."""
        config = PerformForcedPhotometryConfig(hips_order=6, combinedOutput=True, class_filter=class_bit,
                                               pass_thru_cols=["spherex_class"], merge_srccat=True)
        task = PerformForcedPhotometryTask(config=config)
        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"])

        # Check only sources of requested class with for_l3phot=1 and their
        # window mates have flux measurements
        assert len(output.phot_meas_combined) == n_phot_sources, \
            f"Number of photometered sources {len(output.phot_meas_combined)} does not match expected" \
            f" ({n_phot_sources}."
