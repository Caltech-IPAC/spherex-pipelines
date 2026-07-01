# Standard library imports
from pathlib import Path
from copy import deepcopy
import tempfile
import warnings

# Related third party imports
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from astropy.table import Table, vstack
from astropy.utils.exceptions import AstropyWarning
from lsst.pex.config.config import FieldValidationError

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import (CreateAllSkyCatalogConfig, CreateAllSkyCatalogTask,
                       PerformForcedPhotometryTask,
                       PerformForcedPhotometryConfig)
from spx_tasks.create_all_sky_catalog import hips6_to_hips5
from spx_tests import get_sample_global_bkgd, get_sample_pix_map

test_root = Path(__file__).parent.parent

MINIMAL_COLUMNS = {
    "source_id", "ra", "dec", "x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",
    "flux_bkg", "ext_flg", "local_bkg_flg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id", "nyquist_comp"
}


# Helper functions ===========================================================
# Little hack to give a way to pass in generated tables without filepath
class MockDeferredDatasetHandle:
    def __init__(self, table: Table):
        self.table = table

    def get(self) -> Table:
        return self.table


def get_sample_file() -> str:
    sample_file = test_root / "data/photometry_point_sources.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def check_spectral_catalog_schema(arrow_tab: pa.table):
    """Double checks order, name, and type for each column in the Spectral
    Catalog Schema."""
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
        ("flux_bkg", pa.list_(pa.float32())),
        ("local_bkg_flg", pa.list_(pa.bool_())),
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


def l3a_output(merge_srccat: bool, pass_thru_cols: list) -> list:
    """Runs the ``PerformForcedPhotometryTask`` twice for a single small
    image/HEALPixel to fake multiple measurements of the same sources.

    Parameters
    ----------
    merge_srccat : `bool`
        If True, adds pass-through columns to the output photometry measurement
        tables.
    pass_thru_cols : `list`
        List of columns to pass through from input source catalog to output
        photometry measurement tables.

    Returns
    -------
    l3a_output : `dict`
        Dictionary with HEALPixel ID key and `astropy.table.Table` value.
    """
    # Configure the task
    pfp_config = PerformForcedPhotometryConfig(combinedOutput=False, merge_srccat=merge_srccat,
                                               pass_thru_cols=pass_thru_cols)
    pfp_task = PerformForcedPhotometryTask(config=pfp_config)

    healpixes = {}
    # Generate data for the same image twice to simulate having multiple obs
    for i in range(2):
        spx_img = SPHERExImage.read(get_sample_file())
        bkgd = get_sample_global_bkgd(spx_img.data)
        pix_map = get_sample_pix_map(spx_img.data)
        output = pfp_task.run(spx_img, spx_img.catalog, bkgd, pix_map)

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


# Fixtures ====================================================================
@pytest.fixture()
def phot_meas(request):
    # Determine config parameters based on requested mode
    mode = request.param
    if mode == "minimal":
        merge_srccat = False
        pass_thru_cols = []
    elif mode == "pipeline":
        merge_srccat = True
        pass_thru_cols = ["spherex_class"]
    elif mode == "tool":
        merge_srccat = True
        pass_thru_cols = ["name"]

    # Run Perform Forced Photometry and return relevant output
    phot_meas = l3a_output(merge_srccat=merge_srccat, pass_thru_cols=pass_thru_cols)
    return dict(
        healpix_id=next(iter(phot_meas)),
        meas_table_list=[MockDeferredDatasetHandle(phot_meas[next(iter(phot_meas))])]
    )


@pytest.fixture()
def refcat():
    spx_img = SPHERExImage.read(get_sample_file())
    refcat = spx_img.catalog["SPHERExRefID", "coord_src", "spherex_class", "ra", "dec", "pmra",
                             "pmdec", "parallax"]
    refcat["spherex_class"] = refcat["spherex_class"].astype(np.int32)
    return [MockDeferredDatasetHandle(refcat)]


# Test cases =================================================================
class TestCreateAllSkyCatalogTask():
    """Test Create All Sky Catalog Task."""
    @pytest.mark.parametrize("phot_meas, sort_on", [("minimal", "mjd"), ("minimal", "lambda")],
                             indirect=["phot_meas"])
    def test_sort_on(self, phot_meas, refcat, sort_on):
        """Test that output catalog is sorted on specified column."""
        # Configure and run task
        config = CreateAllSkyCatalogConfig(sort_on=sort_on)
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat)
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

    # 0: no filter; 4: PHOTOMETRIC; 4|8|256: PHOTOMETRIC + CALSPEC + SECONDARY
    @pytest.mark.parametrize(
        "phot_meas, filter_spherex_class, flag_list",
        [("pipeline", 0, []), ("pipeline", 4, [4]), ("pipeline", 4 | 8 | 256, [4, 8, 256])],
        indirect=["phot_meas"])
    def test_spx_class_filter(self, phot_meas, refcat, filter_spherex_class, flag_list):
        """Test that output catalog is filtered on spherex_class bitmask."""
        # Configure and run task
        config = CreateAllSkyCatalogConfig(
            filter_spherex_class=filter_spherex_class,
            pass_thru_cols=["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"])
        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(phot_meas["meas_table_list"], refcat)
        allsky_cat_df = output.allsky_cat.to_pandas()  # for easier indexing

        # In no-filter case, all sources should be present
        if config.filter_spherex_class == 0:
            # In no-filter case, all sources should be present
            assert len(allsky_cat_df) > 0, "Output table should not be empty when no filter is applied."
        else:
            # If sources exist, ensure they match the filter criteria
            if len(allsky_cat_df) > 0:
                assert all(allsky_cat_df["spherex_class"].isin(flag_list)), \
                    (f"Catalog contains unexpected sources for bitmask filter {config.filter_spherex_class}:"
                     f"\n{allsky_cat_df['spherex_class']}")
            # If no sources match, ensure an empty table is written
            if (allsky_cat_df["spherex_class"] & config.filter_spherex_class == 0).all():
                assert len(output.allsky_cat) == 0, "Output table should be empty when no sources match."

    @pytest.mark.parametrize("phot_meas, merge_refcat, pass_thru_cols, final_names", [
        ("minimal", False, [], []),
        ("pipeline", True, ["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"],
         ["external_survey", "spherex_class", "pm_ra", "pm_dec", "parallax", "ra", "dec"]),
        ("tool", False, ["name"], ["name"])],
        indirect=["phot_meas"]
    )
    def test_pass_thru_columns(self, phot_meas, refcat, merge_refcat, pass_thru_cols, final_names):
        """Test that requested pass-through columns are included in the output
        catalog."""
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

    @pytest.mark.parametrize("hips6, hips5", [(10371, 2592), (42, 10)])
    def test_hips6_to_hips5(self, hips6, hips5):
        """Test conversion from HiPS6 to HiPS5 HEALPixel IDs."""
        converted = hips6_to_hips5(hips6)
        assert converted == hips5, f"Converted HiPS5 ID {converted} does not match expected {hips5}."

    def test_config_checks(self):
        """Test that invalid configuration parameters raise errors."""
        # Sorting on unsupported column
        with pytest.raises(FieldValidationError):
            CreateAllSkyCatalogConfig(sort_on="invalid_column")

        # Merging refcat but not requesting any pass-through columns
        with pytest.raises(ValueError):
            CreateAllSkyCatalogConfig(merge_refcat=True, pass_thru_cols=[])

    @pytest.mark.parametrize("phot_meas, fault_type", [("pipeline", "duplicate"), ("pipeline", "missing")],
                             indirect=["phot_meas"])
    def test_faulty_merge(self, phot_meas, refcat, fault_type):
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

        with pytest.raises(ValueError):
            task.run(phot_meas["meas_table_list"], [MockDeferredDatasetHandle(refcat_table)])
