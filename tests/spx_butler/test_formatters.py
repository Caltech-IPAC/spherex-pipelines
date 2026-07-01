import logging
import unittest
from astropy.io import fits
from astropy.table import Table
from astropy import wcs
import numpy as np
from pathlib import Path
import pyarrow as pa
import pytest
import yaml
from lsst.daf.butler import (CollectionType, DatasetRef, FileDataset, Timespan,
                             DatasetType)
from lsst.daf.butler.formatters.yaml import YamlFormatter
from lsst.daf.butler.formatters.json import JsonFormatter
from lsst.daf.butler.formatters.astropyTable import AstropyTableFormatter

from spx_butler.tests import temp_butler
from spx_butler.defaults.dataset_types import DatasetTypes
from spx_data import (SPHERExImage, ImageMetadata, ImageMetadataStats,
                      PSFZoneData, BackgroundMesh, GainFactors,
                      NonLinearityParameters, ReadNoiseParameters,
                      PersistenceState, PlannedObservation)
from spx_butler.formatters import (SPHERExImageFormatter, AstropyImageFormatter,
                                   AstropyParquetFormatter,
                                   SPHERExPSFFormatter, AstropySIPFormatter,
                                   BackgroundMeshFormatter, PyarrowTableFormatter,
                                   NonlinearityParametersFormatter, ReadNoiseParametersFormatter,
                                   PersistenceStateFormatter)
from spx_tests import sample_data

TEST_ROOT = Path(__file__).parent.parent.absolute()
BUTLER_CONFIG = "resource://spx_butler/configs"
BUTLER_YAML = "butler.yaml"
DIMENSION_YAML = "dimensions.yaml"
INSTRUMENT_NAME = "simu"
DATA_ID = {"instrument": "simu", "detector": 1}


def get_storage_class(butler, storage_class):
    return butler.storageClasses.getStorageClass(storage_class)


def get_formatter_class(butler, storage_class):
    if hasattr(storage_class, "name"):
        storage_class = storage_class.name
    return butler.datastore.formatterFactory.getFormatterClass(storage_class)


planned_observation_data = sample_data.get_sample_planned_obs_data()
planned_obs = PlannedObservation(**planned_observation_data)


FORMATTERS = [
    {
        "formatter_class": SPHERExImageFormatter,
        "dataset_type": "test_spx_img",
        "storage_class": "SPHERExImage",
        "storage_pytype": SPHERExImage,
        "get_pyobject": sample_data.get_sample_spx_img
    },
    {
        "formatter_class": AstropyImageFormatter,
        "dataset_type": "test_astropy_img",
        "storage_class": "AstropyImage",
        "storage_pytype": fits.HDUList,
        "get_pyobject": sample_data.get_sample_hdulist
    },
    {
        "formatter_class": AstropyParquetFormatter,
        "dataset_type": "test_cat",
        "storage_class": "AstropyParquet",
        "storage_pytype": Table,
        "get_pyobject": sample_data.get_sample_catalog
    },
    {
        "formatter_class": SPHERExPSFFormatter,
        "dataset_type": "test_psf",
        "storage_class": "PSF",
        "storage_pytype": PSFZoneData,
        "get_pyobject": sample_data.get_sample_psf
    },
    {
        "formatter_class": AstropySIPFormatter,
        "dataset_type": "test_sip",
        "storage_class": "AstropySIP",
        "storage_pytype": wcs.Sip,
        "get_pyobject": lambda: sample_data.get_sample_spx_img().wcs.sip
    },
    {
        "formatter_class": BackgroundMeshFormatter,
        "dataset_type": "test_spx_background",
        "storage_class": "BackgroundMesh",
        "storage_pytype": BackgroundMesh,
        "get_pyobject": sample_data.get_sample_background
    },
    {
        "formatter_class": PyarrowTableFormatter,
        "dataset_type": "test_pyarrow_table",
        "storage_class": "PyarrowTable",
        "storage_pytype": pa.Table,
        "get_pyobject": sample_data.get_sample_pyarrow_table
    },
    {
        "formatter_class": YamlFormatter,
        "dataset_type": "test_gain_factors",
        "storage_class": "GainFactors",
        "storage_pytype": GainFactors,
        "get_pyobject": sample_data.get_sample_gain_factors
    },
    {
        "formatter_class": SPHERExImageFormatter,
        "dataset_type": "test_spx_l1_img",
        "storage_class": "SPHERExL1Image",
        "storage_pytype": SPHERExImage,
        "get_pyobject": sample_data.get_sample_level1_img
    },
    {
        "formatter_class": NonlinearityParametersFormatter,
        "dataset_type": "test_nonlin_param",
        "storage_class": "NonLinearityParameters",
        "storage_pytype": NonLinearityParameters,
        "get_pyobject": sample_data.get_sample_nonlinearity_parameters
    },
    {
        "formatter_class": ReadNoiseParametersFormatter,
        "dataset_type": "test_readnoise_param",
        "storage_class": "ReadNoiseParameters",
        "storage_pytype": ReadNoiseParameters,
        "get_pyobject": sample_data.get_sample_read_noise_map
    },
    {
        "formatter_class": PersistenceStateFormatter,
        "dataset_type": "test_persist_state",
        "storage_class": "PersistenceState",
        "storage_pytype": PersistenceState,
        "get_pyobject": sample_data.get_sample_persistence_state
    },
    {
        "formatter_class": AstropyTableFormatter,
        "dataset_type": "test_gds_data",
        "storage_class": "GdsData",
        "storage_pytype": Table,
        "get_pyobject": lambda: Table([
            {'obs_id': '2025W52_2A_0942_1', 'Position': 'pos1', 'Velocity': 'vel1'},
            {'obs_id': '2025W52_2A_0942_2', 'Position': 'pos2', 'Velocity': 'vel2'}])
    },
    {
        "formatter_class": JsonFormatter,
        "dataset_type": "test_pef_data",
        "storage_class": "PredictedObservation",
        "storage_pytype": PlannedObservation,
        "get_pyobject": lambda: planned_obs
    },
    {
        "formatter_class": YamlFormatter,
        "dataset_type": "test_img_metadata",
        "storage_class": "ImageMetadata",
        "storage_pytype": ImageMetadata,
        "get_pyobject": sample_data.get_sample_image_metadata
    },
    {
        "formatter_class": YamlFormatter,
        "dataset_type": "test_img_metadata_stats",
        "storage_class": "ImageMetadataStats",
        "storage_pytype": ImageMetadataStats,
        "get_pyobject": sample_data.get_sample_image_metadata_stats
    },
]
# Avoid pytest warning about file closure
MEMMAP_FORMATTERS = [AstropyImageFormatter]  # NonlinearityParametersFormatter,
# PersistenceStateFormatter

log = logging.getLogger(__name__)
dt = DatasetTypes()


class FormattersTests(unittest.TestCase):
    butler = None
    root = None

    @classmethod
    def setUpClass(cls):
        """Create a new butler only once."""
        butler, root = temp_butler.create_temp_butler()

        dimension_universe = butler.registry.dimensions
        dimensions = dimension_universe.conform(("instrument", "detector"))

        # register calibration dataset types
        for formatter in FORMATTERS:
            dataset_type = DatasetType(name=formatter["dataset_type"],
                                       dimensions=dimensions,
                                       storageClass=formatter["storage_class"],
                                       universe=dimension_universe)
            # idempotent dataset type registration
            butler.registry.registerDatasetType(dataset_type)

        # register collection
        butler.registry.registerCollection("calib", type=CollectionType.RUN)
        butler.registry.registerCollection("raw", type=CollectionType.RUN)
        butler.registry.registerCollection("test", type=CollectionType.RUN)

        cls.butler = butler
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        if cls.root is not None:
            temp_butler.destroy_temp_butler(cls.root)

    def _test_formatters(self, formatters: list):
        data_id = {"instrument": INSTRUMENT_NAME, "detector": 1}

        for formatter in formatters:
            inmem_obj = formatter["get_pyobject"]()
            self.assertIsInstance(inmem_obj, formatter["storage_pytype"])

            dataset_type = formatter["dataset_type"]
            self.butler.put(inmem_obj, dataset_type, data_id, run="calib")

            # Get butler dataset
            out = self.butler.get(dataset_type, data_id, collections="calib")
            self.assertIsInstance(out, formatter["storage_pytype"])

    def test_copy_formatters(self):
        # test formatters maintain copy of the data
        self._test_formatters([f for f in FORMATTERS if f["formatter_class"] not in MEMMAP_FORMATTERS])

    @pytest.mark.filterwarnings("ignore::ResourceWarning")
    def test_memmap_formatters(self):
        # test formatters known to memory map and keeping the file open
        self._test_formatters([f for f in FORMATTERS if f["formatter_class"] in MEMMAP_FORMATTERS])

    def test_ingest(self):
        formatter = FORMATTERS[0]
        filepath = TEST_ROOT / "data" / "small.fits"
        dataset_type_name = "test_ingest"

        # Image dimensions include observation, instrument, and detector
        dimension_universe = self.butler.registry.dimensions
        dimensions = dimension_universe.conform(("instrument", "detector", "spsplan",
                                                 "slewgroup", "slewgroup_counter"))

        # Create a new dataset for this test
        dataset_type = DatasetType(name=dataset_type_name,
                                   dimensions=dimensions,
                                   storageClass=formatter["storage_class"],
                                   universe=dimension_universe)
        # idempotent dataset type registration
        self.butler.registry.registerDatasetType(dataset_type)

        datasets = []
        fmt = formatter["formatter_class"]
        plan = 202537201
        plan_name = "2025W37_2A"
        slewgroup = 486
        run = "raw"
        for slewgroup_counter in range(1, 5):
            # Detector equal to 0 throws an error
            for detector in range(1, 7):
                # Use the same fits to test ingest
                data_id = {"instrument": INSTRUMENT_NAME, "detector": detector, "spsplan": plan,
                           "slewgroup": slewgroup, "slewgroup_counter": slewgroup_counter}
                ref = DatasetRef(dataset_type, data_id, run=run)
                datasets.append(FileDataset(refs=ref,
                                            path=filepath,
                                            formatter=fmt))

        self.butler.registry.registerCollection(run, type=CollectionType.RUN)

        with self.butler.transaction():
            self.butler.registry.insertDimensionData(
                "spsplan",
                {"instrument": INSTRUMENT_NAME, "id": plan, "name": plan_name})
            self.butler.registry.insertDimensionData(
                "slewgroup",
                {"instrument": INSTRUMENT_NAME, "id": slewgroup})
            for slewgroup_counter in range(1, 5):
                t = Timespan(begin=None, end=None)
                obs_id = slewgroup*10+slewgroup_counter
                self.butler.registry.insertDimensionData(
                    "observation",
                    {"instrument": INSTRUMENT_NAME,
                     "id": obs_id,
                     "spsplan": plan,
                     "slewgroup": slewgroup,
                     "slewgroup_counter": slewgroup_counter,
                     "timespan": t})
            self.butler.ingest(*datasets, transfer="symlink")

        # Verify that files were ingested, should be 24 files
        # 6 detectors, 4 observations each
        refsSet = set(self.butler.query_datasets(dataset_type_name,
                                                 collections=[run]))

        error = f"Collection {run} should have 24 elements after ingest"
        self.assertEqual(len(refsSet), 24, error)

        # Verify that data id is present
        data_id = {"instrument": INSTRUMENT_NAME, "detector": 5,
                   "spsplan": plan, "slewgroup": slewgroup, "slewgroup_counter": 3}

        refsList = list(self.butler.query_datasets(dataset_type_name,
                                                   collections=[run],
                                                   data_id=data_id))
        error = f"Collection {run} should have 1 element with {data_id}"
        self.assertEqual(len(refsList), 1, error)

    def test_sip(self):
        spx_img = sample_data.get_sample_spx_img()
        sip = sample_data.get_sample_sip(spx_img)
        # Insert fake item
        self.butler.put(sip, "test_sip", dataId=DATA_ID, run="test")
        # Get fake item
        out = self.butler.get("test_sip", dataId=DATA_ID, collections="test")

        # Verify in and out are the same
        error = [
            "SIP crpix are not equal",
            "SIP A are not equal",
            "SIP B are not equal",
            "SIP AP are not equal",
            "SIP BP are not equal",
            "SIP A order are not equal",
            "SIP AP order are not equal",
            "SIP B order are not equal",
            "SIP BP order are not equal"
        ]
        self.assertTrue(np.array_equal(sip.crpix, out.crpix), error[0])
        self.assertTrue(np.array_equal(sip.a, out.a), error[1])
        self.assertTrue(np.array_equal(sip.b, out.b), error[2])
        self.assertEqual(sip.ap, out.ap, error[3])
        self.assertEqual(sip.bp, out.bp, error[4])
        self.assertEqual(sip.a_order, out.a_order, error[5])
        self.assertEqual(sip.ap_order, out.ap_order, error[6])
        self.assertEqual(sip.b_order, out.b_order, error[7])
        self.assertEqual(sip.bp_order, out.bp_order, error[8])

    def test_psf(self):
        psf = sample_data.get_sample_psf()
        # Insert fake item
        self.butler.put(psf, "test_psf", dataId=DATA_ID, run="test")
        # Get fake item
        out = self.butler.get("test_psf", dataId=DATA_ID, collections="test")
        error = [
            "PSF objects are not equal"
        ]
        self.assertTrue(psf == out, error[0])

    def test_spx_img(self):
        img = sample_data.get_sample_spx_img()
        # Set write parameters for dtype change
        read_params = {
            "dtype_flags": np.dtype("uint8"),
            "dtype_data": np.dtype("float16"),
        }
        self.assertTrue(img.flags.dtype != read_params["dtype_flags"],
                        f"flags dtype are the same {img.flags.dtype}")
        self.assertTrue(img.data.dtype != read_params["dtype_data"],
                        f"data dtype are the same {img.data.dtype}")

        # Insert fake item
        self.butler.put(img, "test_spx_img", dataId=DATA_ID, run="test")

        # Get fake item
        out = self.butler.get("test_spx_img", dataId=DATA_ID, collections="test", parameters=read_params)

        # Verify in and out have the desired dtypes
        self.assertTrue(out.flags.dtype == read_params["dtype_flags"])
        self.assertTrue(out.data.dtype == read_params["dtype_data"])

        # Verify in and out data are the same after changing dtypes
        error = [
            "SPHERExImage data are not equal"
        ]
        self.assertTrue(np.allclose(img.data.astype(read_params["dtype_data"]), out.data), error[0])

    def test_spx_l1_img(self):
        img = sample_data.get_sample_level1_img()
        # Insert fake item
        self.butler.put(img, "test_spx_l1_img", dataId=DATA_ID, run="test")

        # Get fake item
        out = self.butler.get("test_spx_l1_img", dataId=DATA_ID, collections="test")
        # Verify in and out have the desired dtypes
        self.assertTrue(img.flags.dtype != out.flags.dtype,
                        f"flags dtype unchanged: {img.flags.dtype} {out.flags.dtype} ")

        # Verify in and out data are the same after changing dtypes
        error = [
            "SPHERExImage data are not equal"
        ]
        self.assertTrue(np.allclose(img.data.astype("int16"), out.data), error[0])

    def test_nonlin_param(self):
        test_in = sample_data.get_sample_nonlinearity_parameters()
        # Insert fake item
        self.butler.put(test_in, "test_nonlin_param", dataId=DATA_ID, run="test")

        # Get fake item
        test_out = self.butler.get("test_nonlin_param", dataId=DATA_ID, collections="test")
        # Verify in and out have the desired dtypes
        for i in range(len(test_in.hdulist)):
            self.assertTrue(np.all(test_out.hdulist[i].data == test_in.hdulist[i].data),
                            "HDUList data changed")
            self.assertTrue(np.all(test_out.hdulist[i].header == test_in.hdulist[i].header),
                            "HDUList header changed")

    def test_readnoise_param(self):
        test_in = sample_data.get_sample_read_noise_map()
        # Insert fake item
        self.butler.put(test_in, "test_readnoise_param", dataId=DATA_ID, run="test")
        # Get fake item
        test_out = self.butler.get("test_readnoise_param", dataId=DATA_ID, collections="test")
        # Check if "READNOISE-1" extension exists
        readnoise_1_exists = any(hdu.header.get('EXTNAME') == ReadNoiseParameters.single
                                 for hdu in test_out.hdulist)
        self.assertTrue(readnoise_1_exists, "Extension 'READNOISE-1' does not exist")
        # Check if "READNOISE-N" extension exists
        readnoise_n_exists = any(hdu.header.get('EXTNAME') == ReadNoiseParameters.full
                                 for hdu in test_out.hdulist)
        self.assertTrue(readnoise_n_exists, "Extension 'READNOISE-N' does not exist")
        # Verify in and out have the desired dtypes
        for i in range(len(test_in.hdulist)):
            self.assertTrue(np.all(test_out.hdulist[i].data == test_in.hdulist[i].data),
                            "HDUList data changed")
            self.assertTrue(np.all(test_out.hdulist[i].header == test_in.hdulist[i].header),
                            "HDUList header changed")

    def test_persist_state(self):
        persist_state = sample_data.get_sample_persistence_state()
        # Insert fake item
        self.butler.put(persist_state, "test_persist_state", dataId=DATA_ID, run="test")
        # Get fake item
        test_out = self.butler.get("test_persist_state", dataId=DATA_ID, collections="test")
        # Verify in and out have the desired dtypes
        for i in range(len(persist_state.hdulist)):
            self.assertTrue(np.all(test_out.hdulist[i].data == persist_state.hdulist[i].data),
                            "HDUList data changed")
            self.assertTrue(np.all(test_out.hdulist[i].header == persist_state.hdulist[i].header),
                            "HDUList header changed")

    def test_img(self):
        hdulist = sample_data.get_sample_hdulist()
        # Insert fake item
        self.butler.put(hdulist, "test_astropy_img",
                        dataId=DATA_ID, run="test")
        # Get fake item
        hdulist_out = self.butler.get("test_astropy_img", dataId=DATA_ID,
                                      collections="test")
        # Verify in and out are the same
        error = [
            "Astropy image data are not equal",
            "Astropy image headers are not equal"
        ]
        self.assertTrue(np.array_equal(hdulist[1].data, hdulist_out[1].data), error[0])
        self.assertTrue(hdulist[0].header == hdulist_out[0].header)
        hdulist_out.close()

    def test_background(self):
        bkg = sample_data.get_sample_background()

        # Insert fake item
        self.butler.put(bkg, "test_spx_background",
                        dataId=DATA_ID, run="test")
        # Get fake item
        out = self.butler.get("test_spx_background", dataId=DATA_ID,
                              collections="test")
        # Verify in and out are the same
        error = [
            "BackgroundMesh are not equal",
            "BackgroundMesh NDData high-resolution background not equal",
            "BackgroundMesh NDData high-resolution uncertainty not equal"
        ]
        nddata_bkg = bkg.get_background()
        nddata_out = out.get_background()
        self.assertTrue(bkg == out, error[0])
        self.assertTrue(np.array_equal(nddata_bkg.data, nddata_out.data),
                        error[1])
        self.assertTrue(np.array_equal(nddata_bkg.uncertainty.array,
                                       nddata_out.uncertainty.array), error[2])

    def test_pyarrow_table(self):
        table = sample_data.get_sample_pyarrow_table()
        table_columns = table.schema.names

        # Insert pyarrow table and write as parquet
        self.butler.put(table, "test_pyarrow_table", dataId=DATA_ID, run="test")
        # Get out pyarrow table from parquet
        out1 = self.butler.get("test_pyarrow_table", dataId=DATA_ID, collections="test")
        # Verify in == out
        self.assertTrue(table.equals(out1, check_metadata=True))
        # Test different ways of getting column or columns
        out2 = self.butler.get("test_pyarrow_table", dataId=DATA_ID, collections="test",
                               parameters={"columns": table_columns[0]})
        self.butler.get("test_pyarrow_table", dataId=DATA_ID, collections="test",
                        parameters={"columns": table_columns[:2]})
        self.butler.get("test_pyarrow_table", dataId=DATA_ID, collections="test",
                        parameters={"columns": set(table_columns[:2])})
        # Verify partial table is equal to main table row
        self.assertTrue(table[table_columns[0]].equals(out2[table_columns[0]]))
        # Verify only one column was returned
        self.assertTrue(len(out2.schema.names) == 1)
        # Test errors
        with self.assertRaises(ValueError):
            self.butler.get("test_pyarrow_table", dataId=DATA_ID, collections="test",
                            parameters={"columns": ["not_in_table"]})

    def test_yaml(self):
        yaml_file = TEST_ROOT / "data" / "gain_factors.yaml"
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        GainFactors(**data)

    def test_gdsdata_formatter(self):
        """Test the AstropyTableFormatter for GDS data with a sample Table."""
        sample_gds_data = Table([
            {'obs_id': '2025W52_2A_0942_1', 'Position': 'pos1', 'Velocity': 'vel1'},
            {'obs_id': '2025W52_2A_0942_2', 'Position': 'pos2', 'Velocity': 'vel2'}])
        # Insert fake item
        self.butler.put(sample_gds_data, "test_gds_data", dataId=DATA_ID, run="test")
        # Get fake item
        out = self.butler.get("test_gds_data", dataId=DATA_ID, collections="test")
        assert len(out) == 2
        assert out.colnames == sample_gds_data.colnames, "Retrieved GDS columns do not match."
        assert all(np.all(out[name] == sample_gds_data[name]) for name in sample_gds_data.colnames)

        gds_img_data = out[out['obs_id'] == '2025W52_2A_0942_1']

        # Extract the position column
        position = gds_img_data['Position'][0]
        velocity = gds_img_data['Velocity'][0]
        assert position == 'pos1', "Retrieved Position does not match."
        assert velocity == 'vel1', "Retrieved Velocity does not match."

    def test_pefdata_formatter(self):
        """Test the JsonFormatter for PEF with a sample list of dictionaries."""
        # Insert fake item
        self.butler.put(planned_obs, "test_pef_data", dataId=DATA_ID, run="test")
        # Get fake item
        out = self.butler.get("test_pef_data", dataId=DATA_ID, collections="test")
        error = [
            "PEF data are not equal"
        ]
        self.assertTrue(planned_obs == out, error[0])


if __name__ == '__main__':
    unittest.main()
