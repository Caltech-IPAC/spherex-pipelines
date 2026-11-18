# Standard library imports
from pathlib import Path
from unittest.mock import patch, MagicMock
import contextlib
import logging
import re
import tempfile

# Related third party imports
import numpy as np
import astropy.units as u
import healpy as hp
import lsst.pipe.base as pipeBase
import pytest
from astropy.io import fits
from astropy.table import Table
from lsst.daf.butler import DeferredDatasetHandle

# Local application/library specific imports
from spx_butler.defaults import fake_products
from spx_data import SPHERExImage, spherex_image_reader
from spx_tasks import PerformForcedPhotometryConfig, PerformForcedPhotometryTask
from spx_tasks.perform_forced_photometry import get_deep_flag
from spx_tests.sample_data import get_sample_global_bkgd, get_sample_pix_map, get_sample_ssocat
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


def check_cols(flux_tab: Table, pass_thru_cols: list[str]):
    """Check photometry measurement table column formatting.

    Parameters
    ----------
    flux_tab : `astropy.table.Table`
        Table of photometry measurements.
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


def _extract_refcat_from_srccat(srccat: Table) -> tuple[str | None, str | None]:
    """Return (run, version) from srccat.meta if present; otherwise return
    (None, None).
    """
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


def validate_output(input_catalog: Table, output: pipeBase.Struct, pass_thru_cols: list[str]):
    """Validate photometry output against input catalog.

    Parameters
    ----------
    input_catalog : `astropy.table.Table`
        Input source catalog.
    output : `lsst.pipe.base.Struct`
        Output object containing photometry measurement table(s).
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
    if "do_phot" in input_catalog.colnames:
        expected_srcs = len(np.nonzero(input_catalog["do_phot"] == 1)[0])
    else:
        expected_srcs = len(input_catalog)

    # Verify number of sources matches input catalog
    assert n_src_output == expected_srcs, msg

    # Check column format and row ordering
    check_cols(sample_tab, pass_thru_cols)
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
    ps_img = spherex_image_reader(get_sample_file("point source"))
    return dict(
        img=ps_img,
        global_bkgd=get_sample_global_bkgd(ps_img.data),
        fail_bkgd=get_sample_global_bkgd(ps_img.data, null=True),
        pix_map=get_sample_pix_map(ps_img.data),
        spectral_wcs=get_sample_spectral_wcs(ps_img.data),
        ssocat=get_sample_ssocat()
    )


@pytest.fixture()
def galaxy():
    gal_img = spherex_image_reader(get_sample_file("galaxy"))
    return dict(
        img=gal_img,
        global_bkgd=get_sample_global_bkgd(gal_img.data),
        fail_bkgd=get_sample_global_bkgd(gal_img.data, null=True),
        pix_map=get_sample_pix_map(gal_img.data),
        spectral_wcs=get_sample_spectral_wcs(gal_img.data),
        ssocat=get_sample_ssocat()
    )


# Test cases ==================================================================
class TestPerformForcedPhotometry():
    """Test Perform Forced Photometry Task."""

    @pytest.mark.parametrize("config_override, match", [
        ({"filter_by_ids": True, "fit_sso": False}, "`special_ids` input was not provided"),
        ({"override_epsf": True, "fit_sso": False}, "`epsf` input was not provided"),
        ({"fit_sso": True}, "`ssocat` input was not provided")
    ])
    def test_silent_failures(self, config_override, match, point_source):
        """Test that excluding expected run inputs raises errors."""
        config = PerformForcedPhotometryConfig(**config_override)
        task = PerformForcedPhotometryTask(config=config)
        with pytest.raises(ValueError, match=re.escape(match)):
            task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                     point_source["pix_map"], point_source["spectral_wcs"])

    @pytest.mark.parametrize("srccat_missing, ids_negative, ssocat_missing, match", [
        (False, False, False, None),
        (True, False, False, "Requested column 'ab_ratio' is not"),
        (False, True, False, "Negative SPHERExRefIDs found"),
        (False, False, True, "not in the solar system object catalog")
    ])
    def test_deferred_read(self, srccat_missing: bool, ids_negative: bool, ssocat_missing: bool, match: str,
                           point_source: dict):
        """Test that only requested columns are read from deferred catalogs."""
        # Set up config overrides and mock objects
        config = PerformForcedPhotometryConfig(merge_srccat=True, pass_thru_cols=["spherex_class"])
        task = PerformForcedPhotometryTask(config=config)
        task.perform_forced_photometry = MagicMock()

        mock_srccat = MagicMock(spec=DeferredDatasetHandle)
        srccat_return = point_source["img"].catalog
        if srccat_missing:
            srccat_return.remove_column("ab_ratio")
            mock_srccat.get.side_effect = KeyError("ab_ratio")
        if ids_negative:
            srccat_return["SPHERExRefID"] = -1 * srccat_return["SPHERExRefID"]
        mock_srccat.get.return_value = srccat_return

        mock_ssocat = MagicMock(spec=DeferredDatasetHandle)
        ssocat_return = point_source["ssocat"]
        if ssocat_missing:
            ssocat_return.remove_column("desig")
            mock_ssocat.get.side_effect = KeyError("desig")
        mock_ssocat.get.return_value = ssocat_return

        # Check for errors and filtered read
        ctx = pytest.raises(ValueError, match=re.escape(match)) if match else contextlib.nullcontext()
        with ctx:
            task.run(point_source["img"], mock_srccat, point_source["global_bkgd"], point_source["pix_map"],
                     point_source["spectral_wcs"], ssocat=mock_ssocat)

        if not match:
            expected_cols = list(config.cat_phot_cols) + ["spherex_class"]
            mock_srccat.get.assert_called_once_with(parameters={"columns": expected_cols})
            mock_ssocat.get.assert_called_once_with(parameters={"columns": ["desig", "RA", "DEC"]})

    @pytest.mark.parametrize("meta_fmt", ["dict", "no_provenance"])
    def test_refcat_ver_extraction(self, meta_fmt: str, point_source: dict):
        config = PerformForcedPhotometryConfig(combinedOutput=True, fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)

        # Manipulate reference catalog metadata
        srccat = point_source["img"].catalog
        if meta_fmt == "dict":
            srccat.meta["user_metadata"] = {
                "refcat_run": "refcat_v11",
                "refcat_version": "SPHEREx_RefCat_v1.1"
            }
        elif meta_fmt == "no_provenance":
            srccat.meta = {}

        # Check expected output
        output = task.run(point_source["img"], srccat, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"])
        if meta_fmt == "dict":
            assert output.phot_meas_combined.meta["refcat_version"] == "SPHEREx_RefCat_v1.1"
        elif meta_fmt == "no_provenance":
            assert "refcat_version" not in output.phot_meas_combined.meta

    def test_pixel_filtering_and_debug_options(self, point_source, caplog):
        """Test that non-finite and zero-valued pixels are properly filtered
        out before photometry. Test that config options used for debugging
        work.
        """
        config = PerformForcedPhotometryConfig(combinedOutput=True, gc_interval=2, return_fit=True,
                                               fit_sso=False)
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
    @pytest.mark.parametrize("hips_order, n_healpix_tabs, fake_high_pm, use_ref", [
        (6, 1, False, True), (12, 3, False, True), (12, 3, True, True), (6, 1, False, False)])
    def test_healpix_ordering(self, point_source, hips_order: int, n_healpix_tabs: int, fake_high_pm: bool,
                              use_ref: bool):
        """Test that HEALPix indexing and ordering is correct."""
        # Set up catalog
        catalog = point_source["img"].catalog
        if fake_high_pm and hips_order == 12:
            # Move source 3 (HiPS12 idx 80465472) corrected position to
            # the reference position of source 5 (HiPS12 idx 80465131)
            hi_pm_mask = catalog["SPHERExRefID"] == 3
            catalog["ra"][hi_pm_mask] = catalog["ref_ra"][catalog["SPHERExRefID"] == 5]
            catalog["dec"][hi_pm_mask] = catalog["ref_dec"][catalog["SPHERExRefID"] == 5]

        # Set up task
        config = PerformForcedPhotometryConfig(hips_order=hips_order, fit_sso=False)
        if not use_ref:
            catalog.remove_columns([c for c in ("ref_ra", "ref_dec") if c in catalog.colnames])
        exp_ra_col = "ref_ra" if (use_ref and "ref_ra" in catalog.colnames) else "ra"
        exp_dec_col = "ref_dec" if (use_ref and "ref_dec" in catalog.colnames) else "dec"
        task = PerformForcedPhotometryTask(config=config)

        # Check task uses expected RA/Dec columns and returns proper output
        with patch("healpy.ang2pix", wraps=hp.ang2pix) as mock_ang2pix:
            ps_output = task.run(point_source["img"], catalog, point_source["global_bkgd"],
                                 point_source["pix_map"], point_source["spectral_wcs"])
            args, kwargs = mock_ang2pix.call_args
            assert np.all(np.isin(args[1], catalog[exp_ra_col])), \
                f"Expected lon values to contain values from column '{exp_ra_col}'."
            assert np.all(np.isin(args[2], catalog[exp_dec_col])), \
                f"Expected lat values to contain values from column '{exp_dec_col}'."
        assert len(ps_output.phot_meas.keys()) == n_healpix_tabs  # test grouping
        validate_output(point_source["img"].catalog, ps_output, pass_thru_cols=config.pass_thru_cols)

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

    def test_flux_background(self, point_source):
        """Test that reported flux background is higher when global spectral
        background is included.
        """
        config = PerformForcedPhotometryConfig(hips_order=6, fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)

        # Generate outputs with and without spectral background included
        out_w_spec = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                              point_source["pix_map"], point_source["spectral_wcs"])
        out_no_spec = task.run(point_source["img"], point_source["img"].catalog, point_source["fail_bkgd"],
                               point_source["pix_map"], point_source["spectral_wcs"])

        # Ensure that flux_bkg is higher when spectral background is included
        tab_w_spec = list(out_w_spec.phot_meas.values())[0]
        tab_no_spec = list(out_no_spec.phot_meas.values())[0]
        for bkg_w_spec, bkg_no_spec in zip(tab_w_spec["flux_bkg"], tab_no_spec["flux_bkg"]):
            assert bkg_w_spec >= bkg_no_spec, (
                "Global spectral background was not properly added to the local diffuse background."
                f" flux_bkg with spectral: {bkg_w_spec}, flux_bkg without spectral: {bkg_no_spec}.")

    @pytest.mark.parametrize("source_type, cut_side_length, frame_width, psf_extension, do_phot",
                             [("point source", 5, 3, "EPSF", True), ("galaxy", 2, 1, "EPSF", True),
                              ("point source", 5, 3, "PSF", False), ("galaxy", 2, 1, "PSF", False)])
    def test_source_fitting(self, source_type, cut_side_length, frame_width, psf_extension, do_phot,
                            point_source, galaxy):
        """Test that point source and galaxy fitting work."""
        config = PerformForcedPhotometryConfig(
            hips_order=6, cut_side_length=cut_side_length, frame_width=frame_width,
            fit_shapes=True if source_type == "galaxy" else False,
            psf_extension=psf_extension, fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)

        if source_type == "point source":
            inputs = point_source
        elif source_type == "galaxy":
            inputs = galaxy
        srccat = inputs["img"].catalog
        if not do_phot and "do_phot" in srccat.colnames:
            srccat.remove_column("do_phot")
        output = task.run(inputs["img"], srccat, inputs["global_bkgd"], inputs["pix_map"],
                          inputs["spectral_wcs"])
        validate_output(srccat, output, pass_thru_cols=config.pass_thru_cols)

    # NOTE: 'spherex_class' and 'name' are the only supported pass-through
    #       columns with lookup parameters. We add 'pmra' to test that adding
    #       a non-lookup column still results in a standard output.
    @pytest.mark.parametrize(
        "pass_thru_cols",
        [["spherex_class"], ["name"], ["pmra"]])
    def test_pass_thru_columns_and_srccat_read(self, point_source, pass_thru_cols):
        """Test that pass-through columns are properly added to output."""
        config = PerformForcedPhotometryConfig(hips_order=6, merge_srccat=True,
                                               pass_thru_cols=pass_thru_cols, fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)

        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"])
        validate_output(point_source["img"].catalog, output, pass_thru_cols=pass_thru_cols)

    def test_config_checks(self):
        """Tests that config checks work for faulty setups."""
        # Cut side length is a factor of # of active pixels
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(cut_side_length=7)

        # Frame width > cut side length
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(cut_side_length=5, frame_width=10)

        # Non-"ALL" class filter without "spherex_class" column
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(pass_thru_cols=[], class_filter=SourcePipelineClass.COSMOLOGY)

        # Add pass-through columns without specifying any
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(pass_thru_cols=[], merge_srccat=True)

        # Missing required photometry columns
        with pytest.raises(ValueError):
            PerformForcedPhotometryConfig(cat_phot_cols=["ra", "dec", "shape_r", "ab_ratio", "phi"])

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
                                               pass_thru_cols=["spherex_class"], merge_srccat=True,
                                               fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)
        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"])

        # Check only sources of requested class with do_phot=1 and their
        # window mates have flux measurements
        assert len(output.phot_meas_combined) == n_phot_sources, \
            f"Number of photometered sources {len(output.phot_meas_combined)} does not match expected" \
            f" ({n_phot_sources}."

    @pytest.mark.parametrize("special_ids, n_phot_sources, config_overrides", [
        (np.array([4, 10]), 1,  # ID 1 in catalog & do_phot=1, but no flagged neighbors -> 1 source
         {"combinedOutput": True}),
        (np.array([2, 10]), 2,  # ID 2 in catalog & do_phot=0, but neighbor ID 3 has do_phot=1 -> 2 sources
         {"combinedOutput": True}),
        (np.array([5, 20]), 2,  # ID 5 in catalog & do_phot=1, but neighbor ID 6 has do_phot=1 -> 2 sources
         {"combinedOutput": True}),
        (np.array([999, 1000]), 0,  # neither ID is in catalog -> 0 sources
         {"combinedOutput": True, "fit_sso": True, "return_fit": True}),
        (np.array([888, 2000]), 0,  # neither ID is in catalog -> 0 sources
         {"combinedOutput": False, "fit_sso": False, "return_fit": False})
    ])
    def test_id_filtering(self, point_source, special_ids, n_phot_sources, config_overrides):
        """Test that source ID filtering works."""
        config = PerformForcedPhotometryConfig(hips_order=6, filter_by_ids=True, **config_overrides)
        task = PerformForcedPhotometryTask(config=config)
        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"], special_ids=special_ids,
                          ssocat=point_source["ssocat"] if config.fit_sso else None)

        # Check only sources of requested IDs with do_phot=1 and their
        # window mates have flux measurements
        if n_phot_sources == 0:
            # Ensure output connections are empty
            if config.combinedOutput:
                assert output.phot_meas_combined is None
            else:
                assert output.phot_meas is None
            if config.fit_sso:
                assert output.phot_meas_sso is None
            if config.return_fit:
                assert output.tractor_fits is None
        else:
            assert len(output.phot_meas_combined) == n_phot_sources, \
                f"Expected {n_phot_sources} sources, but got {len(output.phot_meas_combined)}."

    @pytest.mark.parametrize("psf_extension, override_epsf, log_msg", [
        ("EPSF", False, "Using EPSF from level2 image extension for photometry fitting."),
        ("PSF", False, "Using legacy PSF for photometry fitting."),
        ("EPSF", True, "Using EPSF override from input calibration for photometry fitting.")
    ])
    @pytest.mark.optional_deps  # builds fake_PSF()/fake_epsf(), which need the simulator
    def test_psf_extensions(self, point_source, psf_extension, override_epsf, log_msg, caplog):
        """Test that PSF extensions are properly selected."""
        # Set up appropriate config/inputs
        config = PerformForcedPhotometryConfig(hips_order=6, cut_side_length=5, frame_width=3,
                                               fit_shapes=False, psf_extension=psf_extension,
                                               override_epsf=override_epsf, fit_sso=False)
        task = PerformForcedPhotometryTask(config=config)
        epsf_calib = fake_products.fake_epsf(detector=1)

        # Run and check logs
        with caplog.at_level(logging.INFO, logger=task.log.name):
            task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                     point_source["pix_map"], point_source["spectral_wcs"],
                     epsf=epsf_calib if config.override_epsf else None)
        msgs = caplog.messages
        assert log_msg in msgs, f"Expected log message '{log_msg}' not found in logs: {msgs}."


    @pytest.mark.parametrize("include_ref_pos", [True, False])
    def test_fit_sso(self, point_source: dict, include_ref_pos: bool):
        """Test that SSO photometry works."""
        config = PerformForcedPhotometryConfig(combinedOutput=True, fit_sso=True)
        if not include_ref_pos:
            config.cat_phot_cols = [col for col in config.cat_phot_cols if col not in ("ref_ra", "ref_dec")]
        task = PerformForcedPhotometryTask(config=config)
        output = task.run(point_source["img"], point_source["img"].catalog, point_source["global_bkgd"],
                          point_source["pix_map"], point_source["spectral_wcs"],
                          ssocat=point_source["ssocat"])

        # Check format and membership of SSO and star/galaxy photometry tables
        assert np.all(output.phot_meas_combined["source_id"] >= 0), \
            "Negative source_id sentinel values from SSOs found in combined photometry table."
        non_sso_cols = set(["source_id", "ref_ra", "ref_dec", "hips6"])
        assert non_sso_cols.isdisjoint(set(output.phot_meas_sso.colnames)), \
            ("SSO photometry table contains non-SSO columns:"
             f" {non_sso_cols.intersection(set(output.phot_meas_sso.colnames))}.")
        assert len(output.phot_meas_sso) == 2 # only 2 are non-GPs in detector area
        assert "version_level2" in output.phot_meas_sso.meta.keys()
        assert "version_phot_meas_sso" in output.phot_meas_sso.meta.keys()


    def test_flux_init(self, point_source):
        """Test per-source deterministic flux initialization."""
        config = PerformForcedPhotometryConfig(hips_order=6, fit_sso=True, combinedOutput=True)
        task = PerformForcedPhotometryTask(config=config)

        # Use fresh inputs for each run to ensure no side effects
        ps_img = spherex_image_reader(get_sample_file("point source"))
        output1 = task.run(ps_img, ps_img.catalog, get_sample_global_bkgd(ps_img.data),
                           get_sample_pix_map(ps_img.data), get_sample_spectral_wcs(ps_img.data),
                           ssocat=get_sample_ssocat())
        ps_img = spherex_image_reader(get_sample_file("point source"))
        output2 = task.run(ps_img, ps_img.catalog, get_sample_global_bkgd(ps_img.data),
                           get_sample_pix_map(ps_img.data), get_sample_spectral_wcs(ps_img.data),
                           ssocat=get_sample_ssocat())

        # Ensure bitwise identical flux and flux err
        tab1 = output1.phot_meas_combined
        tab2 = output2.phot_meas_combined
        assert np.array_equal(tab1["flux"], tab2["flux"]), "Flux values differ between runs."
        assert np.array_equal(tab1["flux_err"], tab2["flux_err"]), "Flux error values differ between runs."
