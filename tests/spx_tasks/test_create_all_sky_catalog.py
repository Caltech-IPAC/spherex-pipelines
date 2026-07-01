# Standard library imports
from pathlib import Path
from copy import deepcopy
import tempfile
import warnings

# Related third party imports
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from unittest.mock import patch

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import (CreateAllSkyCatalogConfig, CreateAllSkyCatalogTask,
                       PerformForcedPhotometryTask,
                       PerformForcedPhotometryConfig)
from spx_tests import get_sample_global_bkgd, get_sample_pix_map

test_root = Path(__file__).parent.parent

pfp_config = PerformForcedPhotometryConfig(combinedOutput=False)
pfp_task = PerformForcedPhotometryTask(config=pfp_config)


MINIMAL_COLUMNS = {
    "source_id", "ra", "dec", "x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",
    "flux_bkg", "ext_flg", "local_bkg_flg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id", "nyquist_comp"
}


def prepare_mock_lvf_fits(array_id=1, shape=(2040, 2040)):
    mock_data = np.random.uniform(0.75, 4.6, shape).astype(np.float32)
    tmpdir = tempfile.TemporaryDirectory()
    fits_path = Path(tmpdir.name) / f"20250202_SSDC_CENTER_Band{array_id}.FITS"
    fits.writeto(fits_path, mock_data, overwrite=True)
    return tmpdir, fits_path


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


def check_astropy_read(path: str, expected_columns: set):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=AstropyWarning)
        tab = Table.read(path, format="parquet")
        ap_warnings = [warning for warning in w if issubclass(warning.category, AstropyWarning)]
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


def run_pfp(fits_files, runs=2):
    """Runs the ``PerformForcedPhotometryTask`` ``runs`` amount of times and
    aggregates results into one dictionary with an entry for each healpix. All
    source measurements per healpix are combined into its respective healpix
    index.

    Parameters
    ----------
    fits_files : `list`
        List of paths to the fits files to use in PFP.
    runs : `int`, Optional
        The amount of times to generate results from PFP per fits file

    Returns
    -------
    healpixes : `dict`
        Dictionary of healpixes with lists for each source measurements.
    """
    healpixes = {}
    # Generate data for the same images twice to simulate having multiple obs
    for _ in range(runs):
        for img in fits_files:
            spx_img = SPHERExImage.read(img)
            bkgd = get_sample_global_bkgd(spx_img.data)
            pix_map = get_sample_pix_map(spx_img.data)
            output = pfp_task.run(spx_img, spx_img.catalog, bkgd, pix_map)

            # Want to combine it all together so combining healpixes
            for healpix, table in output.phot_meas.items():
                if healpix not in healpixes.keys():
                    healpixes[healpix] = Table()
                healpixes[healpix] = vstack([healpixes[healpix], table])

    return healpixes


# TEST 1: Run task with NO filter (should return all sources)
def test_CreateAllSkyCatalog_no_filter():
    tmpdir, _ = prepare_mock_lvf_fits()
    with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
        sample_file = get_sample_file()
        healpixes = run_pfp([sample_file], runs=1)

        healpix_id = next(iter(healpixes))
        healpix1 = deepcopy(healpixes[healpix_id])

        # Ensure `spherex_class` is int32
        healpix1["spherex_class"] = healpix1["spherex_class"].astype(np.int32)

        # Create a list of deferred handles
        deferred_meas_table_list = [MockDeferredDatasetHandle(healpix1)]

        config = CreateAllSkyCatalogConfig(sort_on="source_id")
        config.filter_spherex_class = 0  # No filtering applied

        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(deferred_meas_table_list)

        assert len(output.allsky_cat) > 0, \
            "Output table should not be empty when no filter is applied."

        # Ensure expected columns exist for pass-through case
        actual_columns = set(output.allsky_cat.schema.names)
        expected_columns = deepcopy(MINIMAL_COLUMNS)
        for col_name in (config.pass_thru_cols):
            expected_columns.add(col_name)
        missing_columns = expected_columns - actual_columns
        assert not missing_columns, f"Missing expected columns: {missing_columns}"
        check_field_meta(output.allsky_cat)
        str_msg = "Missing string length keyword for 'external_source_id' column"
        assert b"table::len::external_source_id" in output.allsky_cat.schema.metadata.keys(), str_msg

        # Save to a temp directory for verification
        with tempfile.TemporaryDirectory() as dirpath:
            parquet_path = Path(dirpath) / f"{healpix_id}.parquet"
            pq.write_table(output.allsky_cat, parquet_path)
            assert parquet_path.is_file(), "Parquet file failed to write."
            check_astropy_read(parquet_path, expected_columns)


# TEST 2: Run task with minimal columns and NO filter
def test_CreateAllSkyCatalog_minimal_columns():
    tmpdir, _ = prepare_mock_lvf_fits()
    with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
        sample_file = get_sample_file()
        healpixes = run_pfp([sample_file], runs=1)

        healpix_id = next(iter(healpixes))
        healpix1 = deepcopy(healpixes[healpix_id])

        # Ensure `spherex_class` is int32
        healpix1["spherex_class"] = healpix1["spherex_class"].astype(np.int32)

        # Create a list of deferred handles
        deferred_meas_table_list = [MockDeferredDatasetHandle(healpix1)]

        config = CreateAllSkyCatalogConfig(sort_on="source_id", pass_thru_cols=[])
        config.filter_spherex_class = 0  # No filtering applied

        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(deferred_meas_table_list)

        assert len(output.allsky_cat) > 0, \
            "Output table should not be empty when no filter is applied."

        # Ensure expected columns exist for minimal case
        actual_columns = set(output.allsky_cat.schema.names)
        missing_columns = MINIMAL_COLUMNS - actual_columns
        assert not missing_columns, f"Missing expected columns: {missing_columns}"
        check_field_meta(output.allsky_cat)

        # Save to a temp directory for verification
        with tempfile.TemporaryDirectory() as dirpath:
            parquet_path = Path(dirpath) / f"{healpix_id}.parquet"
            pq.write_table(output.allsky_cat, parquet_path)
            assert parquet_path.is_file(), "Parquet file failed to write."
            check_astropy_read(parquet_path, MINIMAL_COLUMNS)


# TEST 3: Run task with PRIMARY CALIBRATOR filter only
def test_CreateAllSkyCatalog_primary_calibrators():
    tmpdir, _ = prepare_mock_lvf_fits()
    with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
        sample_file = get_sample_file()
        healpixes = run_pfp([sample_file], runs=1)

        healpix_id = next(iter(healpixes))
        healpix1 = deepcopy(healpixes[healpix_id])

        PHOTOMETRIC_FLAG = 1 << 2  # 4 = Primary Photometric Calibrator

        # Ensure `spherex_class` is int32 and assign at least one source to
        # Primary Photometric Calibrator (bit 2 = 4)
        healpix1["spherex_class"] = healpix1["spherex_class"].astype(np.int32)
        healpix1["spherex_class"][0] = PHOTOMETRIC_FLAG

        deferred_meas_table_list = [MockDeferredDatasetHandle(healpix1)]

        config = CreateAllSkyCatalogConfig(sort_on="source_id")
        config.filter_spherex_class = PHOTOMETRIC_FLAG  # Only primary calibrators

        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(deferred_meas_table_list)

        # If sources exist, ensure they match the filter criteria
        if len(output.allsky_cat) > 0:
            output_table = output.allsky_cat.to_pandas()
            assert all(output_table["spherex_class"] == 4), "Filtered catalog contains non-primary sources."

        # If no sources match, ensure an empty table is written
        if (healpix1["spherex_class"] & 4 == 0).all():
            assert len(output.allsky_cat) == 0, "Output table should be empty when no sources match."

        with tempfile.TemporaryDirectory() as dirpath:
            parquet_path = Path(dirpath) / f"{healpix_id}_primary.parquet"
            pq.write_table(output.allsky_cat, parquet_path)
            assert parquet_path.is_file(), "Filtered Parquet file failed to write."


# TEST 4: Run task with PRIMARY, SECONDARY, and CALSPEC CALIBRATORS
def test_CreateAllSkyCatalog_multiple_classes():
    tmpdir, _ = prepare_mock_lvf_fits()
    with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
        sample_file = get_sample_file()
        healpixes = run_pfp([sample_file], runs=1)

        healpix_id = next(iter(healpixes))
        healpix1 = deepcopy(healpixes[healpix_id])

        # Define SPHEREx class flags
        PHOTOMETRIC_FLAG = 1 << 2  # Bit 2 = 4  Primary calibrators
        CALSPEC_FLAG = 1 << 3      # Bit 3 = 8  Calspec sources
        SECONDARY_FLAG = 1 << 8    # Bit 8 = 256  Secondary calibrators

        # Ensure `spherex_class` is int32
        # and assign different sources to different classes
        healpix1["spherex_class"] = healpix1["spherex_class"].astype(np.int32)
        healpix1["spherex_class"][0] = PHOTOMETRIC_FLAG  # Primary
        healpix1["spherex_class"][1] = CALSPEC_FLAG      # Calspec
        healpix1["spherex_class"][2] = SECONDARY_FLAG    # Secondary

        deferred_meas_table_list = [MockDeferredDatasetHandle(healpix1)]

        config = CreateAllSkyCatalogConfig(sort_on="source_id")
        config.filter_spherex_class = PHOTOMETRIC_FLAG | CALSPEC_FLAG | SECONDARY_FLAG  # Combine filters

        task = CreateAllSkyCatalogTask(config=config)
        output = task.run(deferred_meas_table_list)

        # Ensure all sources in output match one of the selected class filters
        output_table = output.allsky_cat.to_pandas()
        assert all(output_table["spherex_class"].isin([4, 8, 256])), \
            "Filtered catalog contains unexpected sources."

        with tempfile.TemporaryDirectory() as dirpath:
            parquet_path = Path(dirpath) / f"{healpix_id}_multi.parquet"
            pq.write_table(output.allsky_cat, parquet_path)
            assert parquet_path.is_file(), "Filtered Parquet file failed to write."
