# Standard library imports
from pathlib import Path
from unittest.mock import patch, MagicMock, call
import logging
import pytest
import importlib.resources
import tempfile

# Related third party imports
from astropy.io import fits
from astropy.table import Table
from lsst.daf.butler import Butler, Config
from lsst.pipe.base import Struct, testUtils, PipelineTask, QuantumContext
import astropy.units as u
import lsst.daf.butler.tests as butlerTests
import numpy as np

# Local application/library specific imports
from spx_data import spherex_image_reader, SPHERExImage
from spx_tasks import EstimateDiffuseBackgroundConfig, L3aConfig, L3aTask, PerformForcedPhotometryConfig
from spx_tests.sample_data import get_sample_pix_map, get_sample_ssocat
from spx_butler.defaults.fake_products import fake_epsf, fake_he_template


test_root = Path(__file__).parent.parent

# Helper functions ============================================================
def get_sample_spectral_wcs(data: np.ndarray) -> fits.HDUList:
    primary_hdu = fits.PrimaryHDU()
    # Create wavelength gradient along y-axis (like real LVF)
    ny, nx = data.shape
    wl_gradient = np.linspace(0.75, 1.0, ny)
    wl_array = np.tile(wl_gradient[:, np.newaxis], (1, nx)).astype(np.float32)
    cwave_hdu = fits.ImageHDU(data=wl_array, name="CWAVE")
    cband_hdu = fits.ImageHDU(data=np.full(data.shape, 0.001), name="CBAND")
    return fits.HDUList([primary_hdu, cwave_hdu, cband_hdu])

def get_sample_file() -> str:
    sample_file = test_root / "data/photometry_point_sources.fits"  # NOTE: image shape is 15x15
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)

# Little hack to give a way to pass in generated tables without filepath
class MockDeferredDatasetHandle:
    def __init__(self, table: Table):
        self.table = table

    def get(self, parameters=None) -> Table:
        if parameters is not None:
            return self.table[parameters["columns"]]
        else:
            return self.table

def mock_hips_ref(hips_id):
    """Create fake HiPS tile dataset references."""
    ref = MagicMock()
    ref.dataId.get.side_effect = lambda key: hips_id if key == "hips6" else None
    return ref

# Fixtures ====================================================================
@pytest.fixture()
def common_inputs(request) -> dict:
    """Create common L3a inputs shared across multiple tests.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `str` value.
        To introduce different fine astrometry cases: "missing" for absent
        keyword, "partial" for FINAST=1 (failed), "good" for FINAST=0 (passed).

    Returns
    -------
    common_inputs : `dict`
        Keys "img", "srccat", "pix_map", and "spec_wcs" map to objects to pass
        to `spx_tasks.L3aTask.run`.
    """
    ps_img = spherex_image_reader(get_sample_file())
    finast_hdr = request.param
    if finast_hdr == "missing":
        if "FINAST" in ps_img.header:
            ps_img.header.remove("FINAST")
    elif finast_hdr == "partial":
        ps_img.header["FINAST"] = 1
    elif finast_hdr == "good":
        ps_img.header["FINAST"] = 0
    return dict(
        img=ps_img,
        srccat=MockDeferredDatasetHandle(ps_img.catalog),
        pix_map=get_sample_pix_map(ps_img.data),
        spec_wcs=get_sample_spectral_wcs(ps_img.data)
    )

@pytest.fixture()
def fake_invalid_cat(request) -> Table:
    """Create a fake invalid source catalog.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `str` value.
        To introduce different errors: "format" for wrong data type, "unit"
        for wrong units, "missing" for missing required column.

    Returns
    -------
    invalid_cat : `astropy.table.Table`
        A table with the specified error introduced.
    """
    invalid_cat = Table()
    invalid_cat["SPHERExRefID"] = np.int64([12345, 23456, 34567, 45678])
    invalid_cat["ra"] = np.float64([30.437813, 321.385428, 197.486377, 246.971852]) * u.deg
    invalid_cat["dec"] = np.float64([-85.156436, 2.178935, 54.940162, -12.617344]) * u.deg
    invalid_cat["shape_r"] = np.float32([0.0, 0.0, -9999.0, 1.5]) * u.arcsec
    invalid_cat["ab_ratio"] = np.float32([0.0, 0.0, -9999.0, 13.45])
    invalid_cat["phi"] = np.float32([0.0, 0.0, -9999.0, 27.429])
    invalid_cat["sersic"] = np.float32([0.0, 0.0, 0.0, 4.2])

    error_type = request.param
    if error_type == "format":
        invalid_cat["SPHERExRefID"] = invalid_cat["SPHERExRefID"].astype(np.int32)
    elif error_type == "unit":
        invalid_cat["ra"] = invalid_cat["ra"].value
    elif error_type == "missing":
        invalid_cat.remove_column("sersic")
    elif error_type == "neg_id":
        invalid_cat["SPHERExRefID"] = -1 * invalid_cat["SPHERExRefID"]

    return MockDeferredDatasetHandle(invalid_cat)

@pytest.fixture()
def optional_connection(request) -> SPHERExImage | Table | np.ndarray:
    """Create optional input connection objects.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `str` value.
        Supported inputs: "epsf" for EPSF calibration image, "he_template" for
        Helium I airglow template calibration image, "special_ids" for array of
        source IDs, "ssocat" for Solar System Object catalog.

    Returns
    -------
    `spx_data.SPHERExImage` | `astropy.table.Table` | `numpy.ndarray`
        Example objects to pass as optional connections to L3a task run method.
        If "epsf" or "he_template", returns an image. If "special_ids", returns
        an array of integers. If "ssocat", returns a deferred table.
    """
    cnxn = request.param
    if cnxn == "epsf":
        return fake_epsf()
    elif cnxn == "he_template":
        return fake_he_template()
    elif cnxn == "special_ids":
        return np.arange(1, 10, 1)
    elif cnxn == "ssocat":
        return MockDeferredDatasetHandle(get_sample_ssocat())

@pytest.fixture()
def minimal_l3a_task() -> L3aTask:
    """Create an L3aTask with a minimal config for testing."""
    minimal_pfp_config = PerformForcedPhotometryConfig(
        fit_sso=False, hips_order=12,
        cat_phot_cols=["SPHERExRefID", "ra", "dec", "shape_r", "ab_ratio", "phi", "sersic"])
    minimal_l3a_config = L3aConfig(
        estimate_diffuse_background=EstimateDiffuseBackgroundConfig(),
        perform_forced_photometry=minimal_pfp_config)
    minimal_l3a_task = L3aTask(config=minimal_l3a_config)
    return minimal_l3a_task

@pytest.fixture()
def temp_repo() -> Butler:
    """Create a temporary Butler repository for testing."""
    _tmpdir = tempfile.TemporaryDirectory()
    config = Config(importlib.resources.files("spx_butler").joinpath("configs/butler.yaml"))
    dim_config = importlib.resources.files("spx_butler").joinpath("configs/dimensions.yaml")
    repo = butlerTests.makeTestRepo(_tmpdir.name, config=config, dimensionConfig=dim_config)

    # Register dimensions
    butlerTests.addDataIdValue(repo, "instrument", "spx")
    butlerTests.addDataIdValue(repo, "exposure", 202517101000126)
    butlerTests.addDataIdValue(repo, "observation", 202517101000120)
    butlerTests.addDataIdValue(repo, "spsplan", 202517101)
    butlerTests.addDataIdValue(repo, "slewgroup", 1)
    butlerTests.addDataIdValue(repo, "dichroicpair", 3)
    butlerTests.addDataIdValue(repo, "detector", 6)
    butlerTests.addDataIdValue(repo, "hips6", 123)
    butlerTests.addDataIdValue(repo, "hips6", 456)

    # Register dataset types
    butlerTests.addDatasetType(repo, "level2", {"instrument", "exposure"}, "SPHERExImage")
    butlerTests.addDatasetType(repo, "srccat", {"instrument", "dichroicpair", "spsplan", "slewgroup"},
                               "AstropyParquet")
    butlerTests.addDatasetType(repo, "solid_angle_pixel_map", {"instrument", "detector"}, "SPHERExImage")
    butlerTests.addDatasetType(repo, "spectral_wcs", {"instrument", "detector"}, "AstropyImage")
    butlerTests.addDatasetType(repo, "he_template", {"instrument", "detector"}, "SPHERExImage")
    butlerTests.addDatasetType(repo, "special_ids", {"instrument"}, "NumpyArray")
    butlerTests.addDatasetType(repo, "epsf", {"instrument", "detector"}, "EPSF")
    butlerTests.addDatasetType(repo, "ssocat", {"instrument", "observation"}, "AstropyParquet")
    butlerTests.addDatasetType(repo, "phot_meas_combined", {"instrument", "exposure"}, "AstropyParquet")
    butlerTests.addDatasetType(repo, "phot_meas", {"instrument", "exposure", "hips6"}, "AstropyParquet")
    butlerTests.addDatasetType(repo, "phot_exceptions", {"instrument", "exposure"}, "AstropyParquet")
    butlerTests.addDatasetType(repo, "phot_meas_sso", {"instrument", "exposure"}, "AstropyParquet")

    yield repo
    _tmpdir.cleanup()


# Test cases ==================================================================
class TestL3aTask():
    """Test L3a Task."""

    @pytest.mark.parametrize("common_inputs", ["good"], indirect=["common_inputs"])
    def test_read_srccat(self, common_inputs: dict, minimal_l3a_task: L3aTask):
        """Test that only requested columns are read from the source catalog.
        """
        # Set up config overrides and mock objects
        minimal_l3a_task.config.perform_forced_photometry.merge_srccat = True
        minimal_l3a_task.config.perform_forced_photometry.pass_thru_cols = ["spherex_class"]
        mock_srccat = MagicMock()
        mock_srccat.get.return_value = common_inputs["srccat"].get()
        minimal_l3a_task.estimate_diffuse_background = MagicMock()
        minimal_l3a_task.perform_forced_photometry = MagicMock()
        minimal_l3a_task.estimate_diffuse_background.return_value.global_bkgd = MagicMock()

        # Check selected columns to read
        minimal_l3a_task.run(spximg=common_inputs["img"], srccat=mock_srccat,
                             solid_angle_pixel_map=common_inputs["pix_map"],
                             spectral_wcs=common_inputs["spec_wcs"])
        expected_cols = list(minimal_l3a_task.config.perform_forced_photometry.cat_phot_cols) + \
                        ["spherex_class"]
        mock_srccat.get.assert_called_once_with(parameters={"columns": expected_cols})


    @pytest.mark.parametrize("fake_invalid_cat, common_inputs, match", [
        ("missing", "good", "Requested column 'sersic' is not"),
        ("format", "good", "does not have expected type"),
        ("unit", "good", "does not have expected unit"),
        ("neg_id", "good", "Negative SPHERExRefIDs found")],
        indirect=["fake_invalid_cat", "common_inputs"])
    def test_invalid_srccat(self, common_inputs: dict, fake_invalid_cat: Table, match: str, minimal_l3a_task):
        """Test that invalid source catalogs trigger an error."""
        with pytest.raises(ValueError, match=match):
            minimal_l3a_task.run(spximg=common_inputs["img"], srccat=fake_invalid_cat,
                                 solid_angle_pixel_map=common_inputs["pix_map"],
                                 spectral_wcs=common_inputs["spec_wcs"])


    @pytest.mark.parametrize("common_inputs, optional_connection", [
        ("good", "ssocat"),
    ], indirect=["common_inputs", "optional_connection"])
    def test_invalid_ssocat(self, common_inputs: dict, optional_connection: object,
                            minimal_l3a_task: L3aTask):
        """Test that invalid SSO catalog input raises an error."""
        bad_ssocat = optional_connection.get()
        bad_ssocat.rename_column("desig", "__index_level_0__")
        bad_ssocat = MockDeferredDatasetHandle(bad_ssocat)
        minimal_l3a_task.config.perform_forced_photometry.fit_sso = True
        with pytest.raises(ValueError, match="not in the solar system object catalog"):
            minimal_l3a_task.run(spximg=common_inputs["img"], srccat=common_inputs["srccat"],
                                 solid_angle_pixel_map=common_inputs["pix_map"],
                                 spectral_wcs=common_inputs["spec_wcs"],
                                 ssocat=bad_ssocat)

    @pytest.mark.parametrize("common_inputs, combinedOutput, fit_sso, msg", [
        ("missing", False, True, "Input level 2 image failed fine astrometry; skipping level 3 processing."),
        ("partial", False, True, "Input level 2 image failed fine astrometry; skipping level 3 processing."),
        ("missing", True, False, "Input level 2 image failed fine astrometry; skipping level 3 processing."),
        ("partial", True, False, "Input level 2 image failed fine astrometry; skipping level 3 processing.")
        ], indirect=["common_inputs"])
    def test_bad_finast(self, common_inputs: dict, combinedOutput: bool, fit_sso: bool, msg: str, caplog,
                        minimal_l3a_task):
        """Test that missing and partial astronomy yield no measurements."""
        if combinedOutput:
            minimal_l3a_task.config.perform_forced_photometry.combinedOutput = True
        if fit_sso:
            minimal_l3a_task.config.perform_forced_photometry.fit_sso = True
        with caplog.at_level(logging.INFO, logger=minimal_l3a_task.log.name):
            result = minimal_l3a_task.run(
                spximg=common_inputs["img"], srccat=common_inputs["srccat"],
                solid_angle_pixel_map=common_inputs["pix_map"], spectral_wcs=common_inputs["spec_wcs"],
                ssocat=MockDeferredDatasetHandle(get_sample_ssocat()) if fit_sso else None)

        assert result.phot_meas is None if not combinedOutput else result.phot_meas_combined is None, \
            "Expected no photometry measurements for partial or missing fine astrometry."
        if fit_sso:
            assert result.phot_meas_sso is None, \
                "Expected no SSO photometry measurements for partial or missing fine astrometry."
        logs = caplog.messages
        assert msg in logs, "Missing warning log for partial or missing fine astrometry."


    @patch('spx_tasks.estimate_diffuse_background.EstimateAbsGainMatrixConfig')
    @pytest.mark.parametrize("common_inputs", ["good"], indirect=True)
    def test_valid_srccat_minimal(self, mock_config_class, common_inputs, minimal_l3a_task):
        """Test L3a run method with valid input source catalog, including
        minimal columns.
        """
        # Configure mock to return config with smaller region counts
        # for 15x15 test image
        mock_config = mock_config_class.return_value
        mock_config.num_regions = 5
        mock_config.num_regions_fine = 10
        mock_config.region_list = None
        mock_config.region_list_fine = None

        output = minimal_l3a_task.run(
            spximg=common_inputs["img"], srccat=common_inputs["srccat"],
            solid_angle_pixel_map=common_inputs["pix_map"], spectral_wcs=common_inputs["spec_wcs"])
        assert isinstance(output, Struct)
        assert len(output.phot_meas.keys()) == 3  # test grouping


    @pytest.mark.parametrize("common_inputs, config_overrides, optional_connection", [
        ("good", {"est_he_airglow": True, "filter_by_ids": False, "override_epsf": False, "fit_sso": False},
         "he_template"),
        ("good", {"est_he_airglow": False, "filter_by_ids": True, "override_epsf": False, "fit_sso": False},
         "special_ids"),
        ("good", {"est_he_airglow": False, "filter_by_ids": False, "override_epsf": True, "fit_sso": False},
         "epsf"),
        ("good", {"est_he_airglow": False, "filter_by_ids": False, "override_epsf": False, "fit_sso": True},
         "ssocat"),
    ], indirect=["common_inputs", "optional_connection"])
    def test_opt_cnxn_run(self, common_inputs: dict, config_overrides: dict, optional_connection: object):
        """Test that subtasks are called with expected inputs depending on
        config.
        """
        # Set up config and task
        edb_config = EstimateDiffuseBackgroundConfig()
        edb_config.est_he_airglow = config_overrides["est_he_airglow"]
        pfp_config = PerformForcedPhotometryConfig()
        pfp_config.filter_by_ids = config_overrides["filter_by_ids"]
        pfp_config.override_epsf = config_overrides["override_epsf"]
        pfp_config.fit_sso = config_overrides["fit_sso"]
        l3a_config = L3aConfig(estimate_diffuse_background=edb_config, perform_forced_photometry=pfp_config)
        task = L3aTask(config=l3a_config)

        # Replace subtasks with mocks
        task.estimate_diffuse_background = MagicMock()
        task.perform_forced_photometry = MagicMock()
        task.estimate_diffuse_background.return_value.global_bkgd = MagicMock()
        task.run(spximg=common_inputs["img"], srccat=common_inputs["srccat"],
                 solid_angle_pixel_map=common_inputs["pix_map"], spectral_wcs=common_inputs["spec_wcs"],
                 epsf=optional_connection if config_overrides["override_epsf"] else None,
                 he_template=optional_connection if config_overrides["est_he_airglow"] else None,
                 special_ids=optional_connection if config_overrides["filter_by_ids"] else None,
                 ssocat=optional_connection if config_overrides["fit_sso"] else None)

        # Check subtask calls
        edb_kwargs = task.estimate_diffuse_background.run.call_args[1]
        pfp_kwargs = task.perform_forced_photometry.run.call_args[1]
        if config_overrides["est_he_airglow"]:
            assert edb_kwargs["he_template"] is optional_connection, "he_template not passed to subtask"
        if config_overrides["filter_by_ids"]:
            assert pfp_kwargs["special_ids"] is optional_connection, "special_ids not passed to subtask"
        if config_overrides["override_epsf"]:
            assert pfp_kwargs["epsf"] is optional_connection, "epsf not passed to subtask"
        if config_overrides["fit_sso"]:
            sso_tab = optional_connection.get(parameters={"columns": ["desig", "RA", "DEC"]})
            assert pfp_kwargs["ssocat"].values_equal(sso_tab), "ssocat not passed to subtask"

    @pytest.mark.parametrize("combinedOutput, common_inputs, sso_result", [
        (True, "good", "populated"),  # parent runQuantum
        (False, "good", "populated"),  # custom runQuantum; populated ssocat -> written with put()
        (False, "good", "empty"),  # custom runQuantum; zero-row ssocat -> skipped
        (False, "good", "none"),  # custom runQuantum; no ssocat -> skipped
        ], indirect=["common_inputs"])
    def test_runQuantum(self, temp_repo: Butler, minimal_l3a_task: L3aTask, combinedOutput: bool,
                        common_inputs: dict, sso_result: str):
        """Test runQuantum method."""
        # Adjust config and input/output data IDs based on optional connections
        minimal_l3a_task.config.perform_forced_photometry.fit_sso = True # test all outputs
        task_id = {"instrument": "spx", "exposure": 202517101000126}
        calib_id = {"instrument": "spx", "detector": 6}
        ioDataIds = {
            "spximg": task_id,
            "srccat": {"instrument": "spx", "dichroicpair": 3, "spsplan": 202517101, "slewgroup": 1},
            "solid_angle_pixel_map": calib_id,
            "spectral_wcs": calib_id,
            "he_template": calib_id,
            "special_ids": {"instrument": "spx"},
            "epsf": calib_id,
            "ssocat": {"instrument": "spx", "observation": 202517101000120},
            "phot_meas_sso": task_id
        }

        if combinedOutput:
            minimal_l3a_task.config.perform_forced_photometry.combinedOutput = True
            ioDataIds["phot_meas_combined"] = task_id
        else:
            minimal_l3a_task.config.perform_forced_photometry.combinedOutput = False
            ioDataIds["phot_meas"] = [
                {"instrument": "spx", "exposure": 202517101000126, "hips6": 123},
                {"instrument": "spx", "exposure": 202517101000126, "hips6": 456}
            ]
            ioDataIds["phot_exceptions"] = task_id

        # Create quantum
        butler = butlerTests.makeTestCollection(temp_repo)
        input_objs = {
            "spximg": common_inputs["img"],
            "srccat": common_inputs["srccat"].get(),
            "solid_angle_pixel_map": common_inputs["pix_map"],
            "spectral_wcs": common_inputs["spec_wcs"],
            "he_template": fake_he_template(),
            "special_ids": np.arange(1, 10, 1),
            "epsf": fake_epsf(),
            "ssocat": get_sample_ssocat(asteroids_only=True)
        }
        for (key, value) in input_objs.items():
            butler.put(value, key if key != "spximg" else "level2", ioDataIds[key])
        quantum = testUtils.makeQuantum(minimal_l3a_task, butler, task_id, ioDataIds=ioDataIds)

        # Test runQuantum call
        if combinedOutput:
            # Check defers to parent method
            with patch.object(PipelineTask, "runQuantum") as mock_super:
                testUtils.runTestQuantum(minimal_l3a_task, butler, quantum, mockRun=True)
                mock_super.assert_called_once()
        else:
            # Check custom method builds and writes expected outputs
            butlerQC = MagicMock(spec=QuantumContext)
            butlerQC.get.return_value = {}

            # Fake output dataset refs
            ref_123 = mock_hips_ref(123)
            ref_456 = mock_hips_ref(456)
            ref_exceptions = MagicMock()
            ref_phot_meas_sso = MagicMock()
            outputRefs = MagicMock()
            outputRefs.phot_meas = [ref_123, ref_456]
            outputRefs.phot_exceptions = ref_exceptions
            outputRefs.phot_meas_sso = ref_phot_meas_sso

            # Fake output return objects
            if sso_result == "populated":
                phot_meas_sso = Table({"name": ["sso"]})
            elif sso_result == "empty":
                # Zero rows but a valid schema, as produced when no SSOs
                # survive the window-fitting/matching step.
                phot_meas_sso = Table({"name": np.array([], dtype="U10")})
            else:
                phot_meas_sso = None
            result = Struct(
                phot_meas={
                    123: Table({"name": ["123"]}),
                    456: Table({"name": ["456"]}),
                    789: Table({"name": ["789"]})  # not a phot_meas ref, should go to exceptions
                },
                phot_meas_sso=phot_meas_sso
            )

            # Mock custom runQuantum execution
            with patch.object(minimal_l3a_task, "run", return_value=result):
                with patch("builtins.vars", return_value={
                    "phot_meas": outputRefs.phot_meas,
                    "phot_exceptions": outputRefs.phot_exceptions,
                    "phot_meas_sso": outputRefs.phot_meas_sso
                }):
                    minimal_l3a_task.runQuantum(butlerQC, MagicMock(), outputRefs)

            # phot_meas and phot_exceptions
            put_calls = butlerQC.put.call_args_list
            assert call(result.phot_meas[123], outputRefs.phot_meas[0]) in put_calls
            assert call(result.phot_meas[456], outputRefs.phot_meas[1]) in put_calls
            assert call(result.phot_meas[789], outputRefs.phot_exceptions) in put_calls

            # phot_meas_sso write depends on table content
            sso_put_calls = [c for c in put_calls
                             if len(c.args) > 1 and c.args[1] is outputRefs.phot_meas_sso]
            if sso_result == "populated":
                assert len(sso_put_calls) == 1, "Expected phot_meas_sso to be written out."
                assert sso_put_calls[0] == call(result.phot_meas_sso, outputRefs.phot_meas_sso)
            else:
                assert len(sso_put_calls) == 0, \
                    f"Expected no phot_meas_sso output for sso_result={sso_result!r}."
