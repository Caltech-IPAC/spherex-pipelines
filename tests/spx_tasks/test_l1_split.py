from unittest.mock import MagicMock, patch

import lsst.pipe.base as pipeBase

from spx_tasks.l1_split import (
    L1OdirOnlyConfig,
    L1OdirOnlyTask,
    L1WithSbandNoOdirConfig,
    L1WithSbandNoOdirTask,
)


def _make_inputs():
    """Create a minimal input dict for calling run()/runQuantum()."""
    return dict(
        spximg=MagicMock(name="spximg"),
        planned_obs=MagicMock(name="planned_obs"),
        gain_factors=MagicMock(name="gain_factors"),
        nonfunc_map_v2=MagicMock(name="nonfunc_map_v2"),
        dichroic_cut_map=MagicMock(name="dichroic_cut_map"),
        spectral_wcs=MagicMock(name="spectral_wcs"),
        initial_distortion_table=MagicMock(name="initial_distortion_table"),
        sgt=MagicMock(name="sgt"),
        spk=MagicMock(name="spk"),
        sband=MagicMock(name="sband"),
        sband_mnemonics=MagicMock(name="sband_mnemonics"),
    )


def _make_full_super_run_outputs():
    """Return what the parent L1Task.run() would have returned."""
    return pipeBase.Struct(
        level1_img=MagicMock(name="level1_img"),
        dqa_l1=MagicMock(name="dqa_l1"),
        sur_diag=MagicMock(name="sur_diag"),
        odir_obs=MagicMock(name="odir_obs"),
    )


def test_l1_odir_only_config_defaults_disable_sband():
    config = L1OdirOnlyConfig()
    assert config.gds_image_association.useSband is False


def test_l1_with_sband_no_odir_config_defaults_enable_sband():
    config = L1WithSbandNoOdirConfig()
    assert config.gds_image_association.useSband is True


@patch("spx_tasks.l1_split.L1Task.run")
def test_l1_odir_only_run_returns_only_expected_outputs(mock_super_run):
    mock_super_run.return_value = _make_full_super_run_outputs()

    task = L1OdirOnlyTask(config=L1OdirOnlyConfig())
    outputs = task.run(**_make_inputs())

    assert hasattr(outputs, "dqa_l1")
    assert hasattr(outputs, "sur_diag")
    assert hasattr(outputs, "odir_obs")

    assert not hasattr(outputs, "level1_img")

    assert outputs.dqa_l1 is mock_super_run.return_value.dqa_l1
    assert outputs.sur_diag is mock_super_run.return_value.sur_diag
    assert outputs.odir_obs is mock_super_run.return_value.odir_obs

    mock_super_run.assert_called_once()


@patch("spx_tasks.l1_split.L1Task.run")
def test_l1_with_sband_no_odir_run_returns_only_expected_outputs(mock_super_run):
    mock_super_run.return_value = _make_full_super_run_outputs()

    task = L1WithSbandNoOdirTask(config=L1WithSbandNoOdirConfig())
    outputs = task.run(**_make_inputs())

    assert hasattr(outputs, "level1_img")
    assert hasattr(outputs, "dqa_l1")
    assert hasattr(outputs, "sur_diag")

    assert not hasattr(outputs, "odir_obs")

    assert outputs.level1_img is mock_super_run.return_value.level1_img
    assert outputs.dqa_l1 is mock_super_run.return_value.dqa_l1
    assert outputs.sur_diag is mock_super_run.return_value.sur_diag

    mock_super_run.assert_called_once()


def test_l1_odir_only_runquantum_puts_outputs_without_provenance():
    task = L1OdirOnlyTask(config=L1OdirOnlyConfig())

    inputs = _make_inputs()
    outputs = pipeBase.Struct(
        dqa_l1=MagicMock(name="dqa_l1"),
        sur_diag=MagicMock(name="sur_diag"),
        odir_obs=MagicMock(name="odir_obs"),
    )

    butlerQC = MagicMock()
    butlerQC.get.return_value = inputs

    inputRefs = MagicMock()
    outputRefs = MagicMock()

    task.run = MagicMock(return_value=outputs)

    task.runQuantum(butlerQC, inputRefs, outputRefs)

    butlerQC.get.assert_called_once_with(inputRefs)
    task.run.assert_called_once_with(**inputs)
    butlerQC.put.assert_called_once_with(outputs, outputRefs)


@patch("spx_tasks.l1_split.add_calibration_provenance")
@patch("spx_tasks.l1_split.get_calibration_inputs")
def test_l1_with_sband_no_odir_runquantum_adds_provenance(
    mock_get_calibration_inputs,
    mock_add_calibration_provenance,
):
    task = L1WithSbandNoOdirTask(config=L1WithSbandNoOdirConfig())

    inputs = _make_inputs()
    outputs = pipeBase.Struct(
        level1_img=MagicMock(name="level1_img"),
        dqa_l1=MagicMock(name="dqa_l1"),
        sur_diag=MagicMock(name="sur_diag"),
    )

    butlerQC = MagicMock()
    butlerQC.get.return_value = inputs

    inputRefs = MagicMock()
    outputRefs = MagicMock()

    mock_get_calibration_inputs.return_value = {"dummy": "calib-id"}
    task.run = MagicMock(return_value=outputs)

    task.runQuantum(butlerQC, inputRefs, outputRefs)

    butlerQC.get.assert_called_once_with(inputRefs)
    task.run.assert_called_once_with(**inputs)

    mock_get_calibration_inputs.assert_called_once_with(inputRefs)
    mock_add_calibration_provenance.assert_called_once_with(
        outputs.level1_img,
        {"dummy": "calib-id"},
    )

    butlerQC.put.assert_called_once_with(outputs, outputRefs)

