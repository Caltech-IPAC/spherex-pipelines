# Standard library imports
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch, MagicMock
import re
import tempfile
import warnings

# Related third party imports
from astropy.io import fits
from astropy.table import Table, meta, vstack
from astropy.utils.exceptions import AstropyWarning
from lsst.pex.config.config import FieldValidationError
from lsst.pipe.base import Struct, QuantumContext
import astropy.units as u
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Local application/library specific imports
from spx_butler.defaults import fake_products
from spx_data import SPHERExImage
from spx_tasks import (CreateAllSkyCatalogConfig, CreateAllSkyCatalogTask,
                       PerformForcedPhotometryTask,
                       PerformForcedPhotometryConfig)
from spx_tasks.create_all_sky_catalog import (hips6_to_hips5,
                                              compute_nyquist_metric,
                                              compute_nyquist_diagnostics,
                                              add_meta_yaml,
                                              META_YAML_TYPE_TAG)
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_tests import get_sample_global_bkgd, get_sample_pix_map

test_root = Path(__file__).parent.parent

MINIMAL_COLUMNS = {
    "source_id", "ra", "dec", "x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",
    "flux_bkg", "ext_flg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id", "nyquist_comp"
}

FLUX_COLUMNS = ["flux", "flux_err", "flux_bkg"]


# Helper functions ===========================================================
class MockDeferredAstropyParquet:
    def __init__(self, table: Table):
        self.table = table

    def get(self, parameters=None) -> Table:
        if parameters is not None:
            return self.table[parameters["columns"]]
        else:
            return self.table


def get_sample_file() -> str:
    sample_file = test_root / "data/photometry_point_sources.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def get_sample_spectral_wcs(data: np.ndarray) -> fits.HDUList:
    primary_hdu = fits.PrimaryHDU()
    cwave_hdu = fits.ImageHDU(data=np.full(data.shape, 0.123), name="CWAVE")
    cband_hdu = fits.ImageHDU(data=np.full(data.shape, 0.001), name="CBAND")
    return fits.HDUList([primary_hdu, cwave_hdu, cband_hdu])


def check_spectral_catalog_schema(arrow_tab: pa.table):
    """Double checks order, name, and type for each column in the Spectral
    Catalog Schema.
    """
    expected = [
        ("source_id", pa.int64()),
        ("external_survey", pa.uint8()),
        ("spherex_class", pa.int32()),
        ("ra", pa.float64()),
        ("dec", pa.float64()),
        ("pm_ra", pa.float32()),
        ("pm_dec", pa.float32()),
        ("parallax", pa.float32()),
        ("x_image", pa.list_(pa.float32())),
        ("y_image", pa.list_(pa.float32())),
        ("mjd", pa.list_(pa.float64())),
        ("lambda", pa.list_(pa.float32())),
        ("lambda_width", pa.list_(pa.float32())),
        ("flux", pa.list_(pa.float32())),
        ("flux_err", pa.list_(pa.float32())),
        ("flux_err_tractor", pa.list_(pa.float32())),  # if variance boost applied
        ("flux_bkg", pa.list_(pa.float32())),
        ("ext_flg", pa.bool_()),
        ("flags", pa.list_(pa.int64())),
        ("fit_ql", pa.list_(pa.float32())),
        ("deep_flg", pa.list_(pa.bool_())),
        ("lvf_id", pa.list_(pa.int64())),
        ("det_id", pa.list_(pa.uint8())),
        ("nyquist_comp", pa.float32()),
    ]
    for i, (name, pa_type) in enumerate(zip(arrow_tab.schema.names, arrow_tab.schema.types)):
        exp_name, exp_type = expected[i]
        assert name == exp_name, f"Pyarrow table does not have expected column '{exp_name}' at index {i}."
        assert pa_type == exp_type, f"Pyarrow table column '{name}' does not have expected type {exp_type}."


def check_astropy_read(path: str, expected_columns: set):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=AstropyWarning)
        tab = Table.read(path, format="parquet")
        ap_warnings = [warning.message for warning in w if issubclass(warning.category, AstropyWarning)]
        assert len(ap_warnings) == 0, f"Astropy raised warning(s) while reading parquet file: {ap_warnings}"

    # Check columns
    astropy_columns = set(tab.colnames)
    missing_columns = expected_columns - astropy_columns
    astropy_msg = f"Astropy table is missing expected columns: {missing_columns}. Please double check the "\
                  f"field and schema metadata in pyarrow."
    assert not missing_columns, astropy_msg

    # Check metadata
    expected_versions = ["version_phot_meas", "version_level2", "version_allsky_cat"]
    for key in expected_versions:
        assert key in tab.meta.keys(), f"Version keyword {key} is missing from catalog metadata."


def check_field_meta(arrow_tab: pa.table):
    for name in arrow_tab.schema.names:
        msg = f"Pyarrow table is missing field-level metadata for field {name}."
        assert (arrow_tab.schema.field(name).metadata) is not None, msg


def l3a_output(merge_srccat: bool, pass_thru_cols: list, inject_nans: bool, inject_mask: bool) -> list:
    """Run the ``PerformForcedPhotometryTask`` twice for a single small image.

    Fake multiple measurements of the same sources by running twice on the
    same HEALPixel.

    Parameters
    ----------
    merge_srccat : `bool`
        If True, adds pass-through columns to the output photometry measurement
        tables.
    pass_thru_cols : `list`
        List of columns to pass through from input source catalog to output
        photometry measurement tables.
    inject_nans : `bool`
        If True, simulate fit errors by injecting NaNs into the 'flux',
        'flux_err', and 'fit_ql' columns.
    inject_mask : `bool`
        If True, simulate masked values in the 'flux', 'flux_err', and 'fit_ql'
        columns.

    Returns
    -------
    l3a_output : `dict`
        Dictionary with HEALPixel ID key and `astropy.table.Table` value.

    """
    # Configure the task
    pfp_config = PerformForcedPhotometryConfig(combinedOutput=False, merge_srccat=merge_srccat,
                                               pass_thru_cols=pass_thru_cols, fit_sso=False)
    pfp_task = PerformForcedPhotometryTask(config=pfp_config)

    healpixes = {}
    # Generate data for the same image twice to simulate having multiple obs
    for i in range(2):
        spx_img = SPHERExImage.read(get_sample_file())
        bkgd = get_sample_global_bkgd(spx_img.data)
        pix_map = get_sample_pix_map(spx_img.data)
        spectral_wcs = get_sample_spectral_wcs(spx_img.data)
        output = pfp_task.run(spximg=spx_img, srccat=spx_img.catalog, global_bkgd=bkgd,
                              solid_angle_pixel_map=pix_map, spectral_wcs=spectral_wcs)

        # Modify metadata and data if needed, then combine tables
        for healpix, table in output.phot_meas.items():
            if "refcat_version" not in table.meta:
                table.meta["refcat_version"] = "SPHEREx_RefCat_v0.TEST"
                um = dict(table.meta.get("user_metadata") or {})
                um["refcat_version"] = "SPHEREx_RefCat_v0.TEST"
                um["refcat_run"] = "refcat_vTEST"
                table.meta["user_metadata"] = um

            # Fake different array-sortable values
            # 'fit_ql' is ordered depending on 'mjd' or 'lambda' sort
            table["mjd"] += i * 2
            table["lambda"] -= i * 0.1
            table["fit_ql"] -= i * 0.05

            if inject_nans:
                table["flux"][table["source_id"] == 1] = np.nan
                table["flux_err"][table["source_id"] == 1] = np.nan
                table["fit_ql"][table["source_id"] == 1] = np.nan

            if inject_mask:
                table = Table(table, masked=True, copy=False)  # convert to masked table
                table["flux"].mask[table["source_id"] == 3] = True
                table["flux_err"].mask[table["source_id"] == 3] = True
                table["fit_ql"].mask[table["source_id"] == 3] = True

            if healpix not in healpixes:
                # start with a copy of the first table to KEEP its .meta
                healpixes[healpix] = table.copy(copy_data=True)
            else:
                # append while preserving existing meta
                healpixes[healpix] = vstack(
                    [healpixes[healpix], table],
                    metadata_conflicts="silent"  # keep meta from first table
                )
    return healpixes


def check_refcat_meta_in_parquet(parquet_path: Path):
    tbl = pq.read_table(parquet_path)
    md = tbl.schema.metadata or {}
    assert b"refcat_version" in md, "Missing 'refcat_version' in Parquet schema metadata."
    val = md[b"refcat_version"].decode("utf-8", errors="ignore").strip()
    assert val, "'refcat_version' is present but empty."


def _fake_nyquist_params(use_quantities=False):
    lmin = np.array([1.0, 2.0, 3.0])
    lmean = np.array([1.5, 2.5, 3.5])
    lmax = np.array([2.0, 3.0, 4.0])
    r = np.array([10.0, 10.0, 10.0])

    if use_quantities:
        lmin = lmin * u.micron
        lmean = lmean * u.micron
        lmax = lmax * u.micron

    return {
        "spectral_channel": {
            "lmin": {"quantity": lmin},
            "lmean": {"quantity": lmean},
            "lmax": {"quantity": lmax},
            "R": {"quantity": r},
        }
    }

def test_compute_nyquist_diagnostics_current_thresholds():
    params = _fake_nyquist_params(use_quantities=False)

    wavelengths = np.array([
        1.10, 1.20,  # spread = 0.10; channel_size = 1.5 / 10 = 0.15; spread_fraction = 0.667
        2.10, 2.12,  # spread = 0.02; channel_size = 2.5 / 10 = 0.25; spread_fraction = 0.08
        3.10,        # one measurement only
    ])

    diag = compute_nyquist_diagnostics(wavelengths, params)

    assert diag["frac_bins_with_1plus"] == pytest.approx(3 / 3)
    assert diag["frac_bins_with_2plus"] == pytest.approx(2 / 3)
    assert diag["frac_bins_with_spread_005"] == pytest.approx(2 / 3)
    assert diag["frac_bins_with_spread_015"] == pytest.approx(1 / 3)
    assert diag["frac_bins_with_spread_033"] == pytest.approx(1 / 3)
    assert diag["frac_bins_with_spread_050"] == pytest.approx(1 / 3)

    assert "frac_bins_with_spread_001" not in diag
    assert "frac_bins_with_spread_010" not in diag


@pytest.mark.parametrize("use_quantities", [False, True])
def test_compute_nyquist_metric_counts_channels_with_two_measurements(use_quantities):
    params = _fake_nyquist_params(use_quantities=use_quantities)

    wavelengths = np.array([
        1.10, 1.11,  # channel 0 has 2 measurements
        2.20,        # channel 1 has 1 measurement
        3.20,        # channel 2 has 1 measurement
    ])

    result = compute_nyquist_metric(wavelengths, params)

    assert result == pytest.approx(1 / 3)


def test_compute_nyquist_metric_ignores_out_of_range_wavelengths():
    params = _fake_nyquist_params(use_quantities=False)

    wavelengths = np.array([
        0.5,         # below first lmin, ignored
        1.10, 1.20, # channel 0 has 2 measurements
        4.0,         # exactly at final lmax, ignored because upper edge is exclusive
        4.5,         # above final lmax, ignored
    ])

    result = compute_nyquist_metric(wavelengths, params)

    assert result == pytest.approx(1 / 3)


def test_compute_nyquist_metric_no_longer_requires_wavelength_spread():
    params = _fake_nyquist_params(use_quantities=False)

    wavelengths = np.array([
        1.10, 1.1001,  # very small spread, but still 2 measurements in channel 0
    ])

    result = compute_nyquist_metric(wavelengths, params)

    assert result == pytest.approx(1 / 3)


@pytest.mark.parametrize("wavelengths", [[], [1.5]])
def test_compute_nyquist_metric_requires_at_least_two_measurements(wavelengths):
    params = _fake_nyquist_params(use_quantities=False)

    result = compute_nyquist_metric(wavelengths, params)

    assert result == 0.0

@pytest.mark.parametrize(
    "phot_meas, enable_diagnostics, expected_extra_cols",
    [
        ({"mode": "pipeline"}, False, set()),
        ({"mode": "pipeline"}, True, {
            "nyquist_frac_1plus",
            "nyquist_frac_2plus",
            "nyquist_spread_005",
            "nyquist_spread_015",
            "nyquist_spread_033",
            "nyquist_spread_050",
        }),
    ],
    indirect=["phot_meas"],
)
def test_nyquist_diagnostic_columns_toggle(phot_meas, enable_diagnostics, expected_extra_cols):
    config = CreateAllSkyCatalogConfig(
        merge_refcat=False,
        pass_thru_cols=[],
        enable_nyquist_diagnostics=enable_diagnostics,
    )
    task = CreateAllSkyCatalogTask(config=config)

    output = task.run(phot_meas["meas_table_list"], refcat=None)
    actual_columns = set(output.allsky_cat.schema.names)

    for col in expected_extra_cols:
        assert col in actual_columns

    old_cols = {
        "nyquist_spread_001",
        "nyquist_spread_010",
    }
    assert actual_columns.isdisjoint(old_cols)


class MockDeferredSPHERExImage():
    def __init__(self, image: SPHERExImage):
        self.image = image

    def get(self) -> SPHERExImage:
        return self.image


class MockDeferredAstropyTable():
    def __init__(self, table: Table):
        self.table = table

    def get(self) -> Table:
        return self.table


# Fixtures ====================================================================
@pytest.fixture()
def phot_meas(request: pytest.FixtureRequest) -> dict:
    """Generate sample HEALPixel dimension and measurement tables for
    photometry output.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `dict` of
        `str`. Key "mode" maps to values "minimal", "pipeline", and "tool";
        key "inject_nans" maps to `bool` indicating whether to simulate fit
        errors; key "inject_mask" maps to `bool` indicating whether to simulate
        masked values.

    Returns
    -------
    phot_meas : `dict`
        Key "healpix_id" maps to `int` HEALPixel ID; key "meas_table_list" maps
        to a `list` of `MockDeferredAstropyParquet` objects containing
        photometry measurement tables.
    """
    # Determine config parameters based on requested mode
    param_dict = request.param
    if param_dict["mode"] == "minimal":
        merge_srccat = False
        pass_thru_cols = []
    elif param_dict["mode"] == "pipeline":
        merge_srccat = True
        pass_thru_cols = ["spherex_class"]
    elif param_dict["mode"] == "tool":
        merge_srccat = True
        pass_thru_cols = ["name"]

    # Run Perform Forced Photometry and return relevant output
    phot_meas = l3a_output(merge_srccat=merge_srccat, pass_thru_cols=pass_thru_cols,
                           inject_nans=param_dict.get("inject_nans", False),
                           inject_mask=param_dict.get("inject_mask", False))
    return dict(
        healpix_id=next(iter(phot_meas)),
        meas_table_list=[MockDeferredAstropyParquet(phot_meas[next(iter(phot_meas))])]
    )


@pytest.fixture()
def refcat():
    spx_img = SPHERExImage.read(get_sample_file())
    refcat = spx_img.catalog
    refcat["spherex_class"] = refcat["spherex_class"].astype(np.int32)
    return [MockDeferredAstropyTable(refcat)]


@pytest.fixture()
def list_items(request) -> list[str]:
    """Fixture to initialize the `list_items` config parameter.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `bool` value.
        If True, "flux_err_tractor" is included in the list of items; if False,
        it is excluded.

    Returns
    -------
    list_items : `list` of `str`
        List-valued field names to use in the output PyArrow table.
    """
    include_tractor_err = request.param
    list_items = ["x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",
                  "flux_err_tractor", "flux_bkg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id"]
    if not include_tractor_err:
        list_items.remove("flux_err_tractor")
    return list_items

# Test cases =================================================================
class TestCreateAllSkyCatalogTask():
    """Test Create All Sky Catalog Task."""

    @pytest.mark.parametrize("config_overrides, phot_meas, empty_cat", [
        ({"merge_refcat": True, "apply_flux_corrections": True, "filter_by_ids": True},
         {"mode": "pipeline"}, False),
        ({"merge_refcat": False, "apply_flux_corrections": False, "filter_by_ids": False},
         {"mode": "pipeline"}, True)
    ], indirect=["phot_meas"])
    def test_runQuantum(self, config_overrides: dict, phot_meas: dict, refcat: list, empty_cat: bool):
        """Test runQuantum method."""
        # Set up mock input arguments
        inputRefs = MagicMock()
        flux_corr_handle = MagicMock()
        flux_corr_handle.ref.run = "test_run"
        get_map = {inputRefs.meas_table_list: phot_meas["meas_table_list"]}
        if config_overrides["apply_flux_corrections"]:
            get_map[inputRefs.l3_flux_corrections[0]] = flux_corr_handle
            get_map[inputRefs.l3_flux_corrections] =  [
                MockDeferredSPHERExImage(fake_products.fake_l3_flux_corrections(i)) for i in range(1, 7)]
        if config_overrides["merge_refcat"]:
            get_map[inputRefs.refcat] = refcat
        if config_overrides["filter_by_ids"]:
            get_map[inputRefs.special_ids] = np.array([1, 2, 3], dtype=np.int64)

        butlerQC = MagicMock(spec=QuantumContext)
        butlerQC.get.side_effect = lambda ref: get_map[ref]

        # Run task and verify calls
        config = CreateAllSkyCatalogConfig(**config_overrides)
        task = CreateAllSkyCatalogTask(config=config)
        result = Struct(allsky_cat=None if empty_cat else MagicMock())
        with patch.object(task, "run", return_value=result) as mock_run:
            task.runQuantum(butlerQC, inputRefs, outputRefs=MagicMock())

        call_kwargs = mock_run.call_args.kwargs
        if config_overrides["merge_refcat"] and config_overrides["apply_flux_corrections"] \
            and config_overrides["filter_by_ids"]:
                assert call_kwargs["l3_flux_corrections"] == get_map[inputRefs.l3_flux_corrections]
                assert call_kwargs["flux_corr_run"] == flux_corr_handle.ref.run
                assert call_kwargs["refcat"] == get_map[inputRefs.refcat]
                np.testing.assert_array_equal(call_kwargs["special_ids"], get_map[inputRefs.special_ids])
        else:
            assert call_kwargs["l3_flux_corrections"] is None
            assert call_kwargs["flux_corr_run"] is None
            assert call_kwargs["refcat"] is None
            assert call_kwargs["special_ids"] is None

        if empty_cat:
            butlerQC.put.assert_not_called()
        else:
            butlerQC.put.assert_called_once()
            assert butlerQC.put.call_args.args[0] == result.allsky_cat

    @pytest.mark.parametrize("phot_meas, sort_on",
                             [({"mode": "minimal"}, "mjd"), ({"mode": "minimal"}, "lambda")],
                             indirect=["phot_meas"])
    def test_sort_on(self, phot_meas, refcat, sort_on):
        """Test that output catalog is sorted on specified column."""
        # Configure and run task
        config = CreateAllSkyCatalogConfig(sort_on=sort_on)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat=refcat)
        allsky_cat_df = output.allsky_cat.to_pandas()  # for easier indexing/comparison

        # Check <sort_on> is in ascending order
        input_phot = phot_meas["meas_table_list"][0].get()
        test_id = input_phot["source_id"][0]
        in_sort_on = input_phot[input_phot["source_id"] == test_id][sort_on].data
        out_sort_on = allsky_cat_df[allsky_cat_df["source_id"] == test_id][sort_on][0]
        expected_sort_on = in_sort_on[np.argsort(in_sort_on)]
        np.testing.assert_array_equal(actual=out_sort_on, desired=expected_sort_on,
                                      err_msg=f"Sorting column '{sort_on}' is not in ascending order."
                                              f" Expected {expected_sort_on}, got {out_sort_on}.")

        # Check another array column (test w/ 'fit_ql') is sorted accordingly
        in_fit_ql = input_phot[input_phot["source_id"] == test_id]["fit_ql"].data
        out_fit_ql = allsky_cat_df[allsky_cat_df["source_id"] == test_id]["fit_ql"][0]
        expected_fit_ql = in_fit_ql[np.argsort(in_sort_on)]
        np.testing.assert_array_equal(actual=out_fit_ql, desired=expected_fit_ql,
                                      err_msg=f"Array column 'fit_ql' not sorted in order of {sort_on}."
                                              f" Expected {expected_fit_ql}, got {out_fit_ql}.")

    @pytest.mark.parametrize(
        "phot_meas, filter_spherex_class, flag_list, n_sources",
        [({"mode": "pipeline"}, 0, [], 4),  # no filter
         ({"mode": "pipeline"}, 32, [32], 2),  # (bit 5) COSMOLOGY
         ({"mode": "pipeline"}, 4 | 16 | 32, [4, 16, 32], 3),  # (bits 2|4|5) PHOTOMETRIC + BRIGHT + COSMOLOGY
         ({"mode": "pipeline"}, 1024, [1024], 0)],  # (bit 10) CLUSTER
        indirect=["phot_meas"])
    def test_spx_class_filter(self, phot_meas, refcat, filter_spherex_class, flag_list, n_sources):
        """Test that output catalog is filtered on spherex_class bitmask."""
        # Configure and run task
        config = CreateAllSkyCatalogConfig(
            filter_spherex_class=filter_spherex_class,
            pass_thru_cols=["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"],
            merge_refcat=True)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat=refcat)

        # Verify output
        if n_sources == 0:
            assert output.allsky_cat is None, "Expected no output catalog when no sources pass filter."
        else:
            allsky_cat_df = output.allsky_cat.to_pandas()  # for easier indexing
            assert len(allsky_cat_df) == n_sources
            if config.filter_spherex_class > 0:
                # If sources exist, ensure they match the filter criteria
                if len(allsky_cat_df) > 0:
                    assert all(allsky_cat_df["spherex_class"].isin(flag_list)), \
                        (f"Catalog contains unexpected sources for bitmask filter"
                         f" {config.filter_spherex_class}:\n{allsky_cat_df['spherex_class']}")

    @pytest.mark.parametrize("phot_meas, merge_refcat, pass_thru_cols, final_names", [
        ({"mode": "minimal"}, False, [], []),
        ({"mode": "pipeline"}, True, ["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"],
         ["external_survey", "spherex_class", "pm_ra", "pm_dec", "parallax", "ra", "dec"]),
        ({"mode": "tool"}, False, ["name"], ["name"])],
        indirect=["phot_meas"]
    )
    def test_pass_thru_columns(self, phot_meas, refcat, merge_refcat, pass_thru_cols, final_names):
        """Test that requested pass-through columns are included in the output
        catalog.
        """
        # Configure and run task
        config = CreateAllSkyCatalogConfig(merge_refcat=merge_refcat, pass_thru_cols=pass_thru_cols)
        task = CreateAllSkyCatalogTask(config=config)
        input_params = {"meas_table_list": phot_meas["meas_table_list"]}
        if config.merge_refcat:
            input_params["refcat"] = refcat
        output = task.run(**input_params)

        # Determine expected columns
        actual_columns = set(output.allsky_cat.schema.names)
        expected_columns = deepcopy(MINIMAL_COLUMNS)
        for col_name in final_names:
            expected_columns.add(col_name)

        # Check for missing columns, field-level metadata
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, f"Missing expected columns: {missing_columns}"
        check_field_meta(output.allsky_cat)

        # If pipeline-mode, check order follows Spectral Catalog Schema
        if pass_thru_cols == ["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"]:
            check_spectral_catalog_schema(output.allsky_cat)

        # Save to a temp directory for verification
        with tempfile.TemporaryDirectory() as dirpath:
            parquet_path = Path(dirpath) / f"{phot_meas['healpix_id']}.parquet"
            pq.write_table(output.allsky_cat, parquet_path)
            assert parquet_path.is_file(), "Parquet file failed to write."
            check_astropy_read(parquet_path, expected_columns)
            check_refcat_meta_in_parquet(parquet_path)

    @pytest.mark.parametrize("phot_meas", [{"mode": "pipeline"}], indirect=["phot_meas"])
    def test_pass_thru_values_unsorted_refcat(self, phot_meas, refcat):
        """Test that pass-through values are matched to the correct source
        when the reference catalog rows are not sorted by SPHERExRefID.

        Regression test: matched indices are computed against the sorted view
        of the refcat, so column values must be taken from that same view.
        """
        rc = refcat[0].get()
        expected = {int(sid): (float(ra), float(dec), float(par))
                    for sid, ra, dec, par in zip(
                        rc["SPHERExRefID"], rc["ra"], rc["dec"], rc["parallax"])}

        # Reorder rows by descending ID -- a valid refcat, just not sorted
        rc_unsorted = rc[np.argsort(rc["SPHERExRefID"].data)[::-1]]

        config = CreateAllSkyCatalogConfig(merge_refcat=True)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"],
                          refcat=[MockDeferredAstropyTable(rc_unsorted)])

        df = output.allsky_cat.to_pandas()
        assert len(df) > 0, "Expected a non-empty output catalog."
        for _, row in df.iterrows():
            exp_ra, exp_dec, exp_par = expected[int(row["source_id"])]
            assert np.isclose(row["ra"], exp_ra) and np.isclose(row["dec"], exp_dec), (
                f"Pass-through ra/dec for source {int(row['source_id'])} do not match refcat.")
            assert np.isclose(row["parallax"], exp_par, equal_nan=True), (
                f"Pass-through parallax for source {int(row['source_id'])} does not match refcat.")

    @pytest.mark.parametrize("has_meta", [False, True])
    def test_add_meta_yaml(self, has_meta):
        """Test that add_meta_yaml places field entries in the 'datatype'
        list and version entries under a top-level 'meta' key, creating the
        meta section when the input table has no table-level metadata.
        """
        table = Table({"ra": [1.0], "dec": [2.0]})
        if has_meta:
            table.meta["origin"] = "TEST"
        meta_yaml = meta.get_yaml_from_table(table)
        assert (META_YAML_TYPE_TAG in meta_yaml) == has_meta

        add_meta_yaml(meta_yaml, mode="field",
                      field_dict={"name": "nyquist_comp", "datatype": "float32",
                                  "description": "Nyquist flag"})
        add_meta_yaml(meta_yaml, mode="version")

        # Round-trip through the astropy YAML parser: the version entry must
        # land under the top-level 'meta' key, not in the 'datatype' list
        header = meta.get_header_from_yaml(meta_yaml)
        colnames = [col["name"] for col in header["datatype"]]
        assert colnames == ["ra", "dec", "nyquist_comp"], \
            f"Unexpected columns in 'datatype': {colnames}"
        assert header["meta"]["version_allsky_cat"] == SSDC_PIPELINES_VERSION, \
            "Missing or wrong 'version_allsky_cat' in table-level meta."
        if has_meta:
            assert header["meta"]["origin"] == "TEST", \
                "Pre-existing table-level meta was not preserved."

    @pytest.mark.parametrize("hips6, hips5", [(10371, 2592), (42, 10)])
    def test_hips6_to_hips5(self, hips6, hips5):
        """Test conversion from HiPS6 to HiPS5 HEALPixel IDs."""
        converted = hips6_to_hips5(hips6)
        assert converted == hips5, f"Converted HiPS5 ID {converted} does not match expected {hips5}."

    @pytest.mark.parametrize("err_type, config_kwargs", [
        (FieldValidationError, {"sort_on": "prohibited_column"}),
        (ValueError, {"merge_refcat": True, "pass_thru_cols": []}),
        (ValueError, {"apply_variance_boost": True, "var_boost_frac": 0}),
        (ValueError, {"apply_variance_boost": True, "list_items": ["flux", "flux_err", "flux_bkg"]}),
        (ValueError, {"apply_variance_boost": False, "list_items": ["flux_err", "flux_err_tractor"]}),
        (ValueError, {"merge_refcat": True, "pass_thru_cols": ["unsupported_column"]})
    ])
    def test_config_checks(self, err_type, config_kwargs):
        """Test that invalid configuration parameters raise errors."""
        # Sorting on a column not in the allowed set
        with pytest.raises(err_type):
            CreateAllSkyCatalogConfig(**config_kwargs)

    @pytest.mark.parametrize("phot_meas, fault_type, exc_msg", [
        ({"mode": "pipeline"}, "duplicate",
         r"^Duplicate \'SPHERExRefID\' values found in reference catalog: \[1\]$"),
        ({"mode": "pipeline"}, "missing",
         r"^1\+ photometered sources were not found in the reference catalog: \[np\.int64\(1\)\]$")],
        indirect=["phot_meas"])
    def test_faulty_merge(self, phot_meas, refcat, fault_type, exc_msg):
        """Check that faulty reference catalog matching raises errors."""
        config = CreateAllSkyCatalogConfig(merge_refcat=True)
        task = CreateAllSkyCatalogTask(config=config)

        # Modify refcat based on fault type
        refcat_table = refcat[0].get()
        if fault_type == "duplicate":
            # Introduce duplicate source_id
            dup_row = refcat_table[0]
            refcat_table = vstack([refcat_table, dup_row])
        elif fault_type == "missing":
            # Remove a source_id present in photometry measurements
            src_id_to_remove = phot_meas["meas_table_list"][0].get()["source_id"][0]
            refcat_table = refcat_table[refcat_table["SPHERExRefID"] != src_id_to_remove]

        with pytest.raises(ValueError, match=exc_msg):
            task.run(phot_meas["meas_table_list"], refcat=[MockDeferredAstropyTable(refcat_table)])

    @pytest.mark.parametrize(
        "phot_meas, list_items, var_boost, flux_corrections, detector, factor, meas_changed", [
            ({"mode": "pipeline"}, True, True, True, 1, 5.6, True),  # meas detector, "flux_err_tractor" col
            ({"mode": "pipeline"}, False, False, True, 1, 5.6, True),  # meas detector, "flux_err" col
            ({"mode": "pipeline"}, True, True, True, 2, 3.2, False),  # other detector
            ({"mode": "pipeline"}, False, False, False, None, None, False)],  # no correction
        indirect=["list_items", "phot_meas"])
    def test_apply_flux_corrections(self, phot_meas: dict, list_items: bool, var_boost: bool,
                                    flux_corrections: bool, detector: int, factor: float, meas_changed: bool):
        """Test that applying flux corrections modifies the output catalog
        accordingly.
        """
        # Set up config, task, and input flux corrections
        config = CreateAllSkyCatalogConfig(apply_flux_corrections=flux_corrections,
                                           apply_variance_boost=var_boost, list_items=list_items,
                                           merge_refcat=False)
        task = CreateAllSkyCatalogTask(config=config)
        if flux_corrections:
            l3_flux_corrections = []
            for i in range(1, 7):
                l3_flux_corrections.append(MockDeferredSPHERExImage(fake_products.fake_l3_flux_corrections(
                    i, factor=factor if i == detector else 1.0)))
            flux_corr_run = f"fake_run_d{detector}_f{factor}"
        else:
            l3_flux_corrections = None
            flux_corr_run = None

        # Keep minimum flux values from input photometry
        min_in = {}
        for col in FLUX_COLUMNS:
            min_in[col] = np.nanmin(
                np.concatenate([tab.get()[col].data for tab in phot_meas["meas_table_list"]]))

        # Run task and check for corrections
        output = task.run(phot_meas["meas_table_list"], l3_flux_corrections=l3_flux_corrections,
                          flux_corr_run=flux_corr_run)
        out_cat = output.allsky_cat
        for col in FLUX_COLUMNS:
            expected = min_in[col] * factor if meas_changed else min_in[col]
            if col == "flux_err" and var_boost:
                col = "flux_err_tractor"
            min_out = np.nanmin(np.concatenate(out_cat[col].to_numpy()))
            input_det = set(np.concatenate(
                [tab.get()["det_id"].data for tab in phot_meas["meas_table_list"]]))

            assert min_out == expected, \
                (f"Minimum output {col} does not match expected value. Expected {expected}, got {min_out}."
                 f" Debugging: apply_flux_corrections={flux_corrections}, detector for flux"
                 f" corrections={detector}, factor for flux corrections={factor}, input measurement"
                 f" detector(s)={input_det}, minimum input {col}={min_in[col]}")

    @pytest.mark.parametrize("phot_meas, apply_variance_boost, additional_cols, list_items", [
        ({"mode": "pipeline"}, True, ["flux_err_tractor"], True),
        ({"mode": "pipeline"}, False, [], False)
    ], indirect=["phot_meas", "list_items"])
    def test_apply_variance_boost(self, phot_meas, apply_variance_boost: bool, additional_cols: list[str],
                                  list_items):
        config = CreateAllSkyCatalogConfig(apply_variance_boost=apply_variance_boost, merge_refcat=False,
                                           pass_thru_cols=[], list_items=list_items)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat=None)

        # Determine expected columns
        actual_columns = set(output.allsky_cat.schema.names)
        expected_columns = deepcopy(MINIMAL_COLUMNS)
        for col_name in additional_cols:
            expected_columns.add(col_name)

        # Check for missing columns, field-level metadata
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, f"Missing expected columns: {missing_columns}"
        check_field_meta(output.allsky_cat)

    @pytest.mark.parametrize("phot_meas, n_meas, missing_ids", [
        ({"mode": "pipeline", "inject_nans": True, "inject_mask": False}, 8, []),  # ID 1 meas excluded
        ({"mode": "pipeline", "inject_nans": False, "inject_mask": True}, 6, [3]),  # ID 3 meas excluded
        ({"mode": "pipeline", "inject_nans": False, "inject_mask": False}, 8, [])  # no source meas excluded
    ], indirect=["phot_meas"])
    def test_mask_nans(self, phot_meas: pytest.fixture, n_meas: int, missing_ids: list[int]):
        """Test that NaN and masked values in input photometry measurement
        tables are excluded in output catalog.
        """
        config = CreateAllSkyCatalogConfig(merge_refcat=False, apply_flux_corrections=False)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat=None)
        allsky_cat_df = output.allsky_cat.to_pandas()

        # Check that sources with NaN or masked measurements are excluded
        for src_id in missing_ids:
            assert src_id not in allsky_cat_df["source_id"].values, \
                f"Source ID {src_id} found in output catalog."

        # Check that number of measurements matches expected
        assert allsky_cat_df["flux"].apply(len).sum() == n_meas, \
            f"Number of measurements in output catalog does not match expected {n_meas}."

    @pytest.mark.parametrize("phot_meas, config_override, match", [
        ({"mode": "pipeline"}, {"apply_flux_corrections": True, "merge_refcat": False},
         "`l3_flux_corrections` input was not provided"),
        ({"mode": "pipeline"}, {"merge_refcat": True}, "`refcat` input was not provided"),
        ({"mode": "pipeline"}, {"filter_by_ids": True, "merge_refcat": False},
         "`special_ids` input was not provided")
    ], indirect=["phot_meas"])
    def test_silent_failures(self, phot_meas: pytest.fixture, config_override: dict, match: str):
        """Test that excluding expected run inputs raises errors."""
        config = CreateAllSkyCatalogConfig(**config_override)
        task = CreateAllSkyCatalogTask(config=config)
        with pytest.raises(ValueError, match=re.escape(match)):
            task.run(phot_meas["meas_table_list"])

    @pytest.mark.parametrize("phot_meas, special_ids, n_sources", [
        ({"mode": "pipeline"}, np.array([1, 2, 3, 4], dtype=np.int64), 2),
        ({"mode": "pipeline"}, np.array([7, 8, 9, 10], dtype=np.int64), 0)
    ], indirect=["phot_meas"])
    def test_filter_by_ids(self, phot_meas: pytest.fixture, special_ids: np.ndarray, n_sources: int,
                           refcat: pytest.fixture):
        """Test that filtering by special IDs works as expected."""
        # Configure task to filter by specific source IDs
        config = CreateAllSkyCatalogConfig(filter_by_ids=True)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat=refcat, special_ids=special_ids)

        # Verify output
        if n_sources == 0:
            assert output.allsky_cat is None, "Expected no output catalog when no sources pass filter."
        else:
            allsky_cat_df = output.allsky_cat.to_pandas()  # for easier indexing
            assert len(allsky_cat_df) == n_sources
            assert all(allsky_cat_df["source_id"].isin(special_ids)), \
                (f"Sources in filtered catalog not found in special IDs array ({special_ids}):\n"
                 f"{allsky_cat_df['source_id']}")
