from unittest.mock import Mock, patch
from astropy.io.fits import Header
from spx_tasks.l2a import L2aTask


def make_task(fail_on_bad_l1_dqa=False):
    task = object.__new__(L2aTask)

    task.config = Mock()
    task.config.fail_on_bad_l1_dqa = fail_on_bad_l1_dqa

    task.gds_image_association = Mock()
    task.source_bloom_and_cosmic_ray_halo_flagging = Mock()
    task.correct_non_linearity = Mock()
    task.create_variance_map = Mock()
    task.update_persistence = Mock()

    return task


def make_spximg():
    spximg = Mock()
    spximg.header = Header()
    return spximg


def test_run_calls_subtasks_in_order_and_returns_outputs():
    task = make_task()

    spximg0 = make_spximg()
    spximg1 = make_spximg()
    spximg2 = make_spximg()
    spximg3 = make_spximg()
    spximg4 = make_spximg()
    spximg5 = make_spximg()

    gain_factors = Mock()
    nonlinear_pars = Mock()
    readnoise_pars = Mock()
    persistence_state_in = Mock()

    bsh_diag_img = Mock()
    persistence_state_out = Mock()
    persistence_image = Mock()

    task.gds_image_association.run.return_value = Mock(spximg_out=spximg1)
    task.source_bloom_and_cosmic_ray_halo_flagging.run.return_value = Mock(
        spximg_out=spximg2,
        bsh_diag_img=bsh_diag_img,
    )
    task.correct_non_linearity.run.return_value = Mock(spximg_out=spximg3)
    task.create_variance_map.run.return_value = Mock(spximg_out=spximg4)
    task.update_persistence.run.return_value = Mock(
        spximg_out=spximg5,
        persistence_state_out=persistence_state_out,
        persistence_image=persistence_image,
    )

    result = task.run(
        spximg=spximg0,
        gain_factors=gain_factors,
        nonlinear_pars=nonlinear_pars,
        readnoise_pars=readnoise_pars,
        persistence_state_in=persistence_state_in,
        spk="spk",
        sgt="sgt",
        sband="sband",
        sband_mnemonics="mnemonics",
    )

    task.gds_image_association.run.assert_called_once_with(
        spximg0,
        None,
        "sgt",
        "sband",
        "spk",
        "mnemonics",
    )

    task.source_bloom_and_cosmic_ray_halo_flagging.run.assert_called_once_with(
        spximg1
    )
    task.correct_non_linearity.run.assert_called_once_with(
        spximg2,
        nonlinear_pars,
        gain_factors,
    )
    task.create_variance_map.run.assert_called_once_with(
        spximg3,
        gain_factors,
        readnoise_pars,
    )
    task.update_persistence.run.assert_called_once_with(
        spximg4,
        persistence_state_in,
    )

    assert result.spximg_out is spximg5
    assert result.bsh_diag_img is bsh_diag_img
    assert result.persistence_state_out is persistence_state_out
    assert result.persistence_image is persistence_image

    assert "DATE" in spximg5.header
    assert spximg5.header.comments["DATE"] == "HDU creation time (UTC)"

@patch("spx_tasks.l2a.check_l1_dqa")
def test_run_calls_l1_dqa_check_when_configured(mock_check_l1_dqa):
    task = make_task(fail_on_bad_l1_dqa=True)

    spximg0 = make_spximg()
    spximg1 = make_spximg()
    spximg2 = make_spximg()
    spximg3 = make_spximg()
    spximg4 = make_spximg()
    spximg5 = make_spximg()

    task.gds_image_association.run.return_value = Mock(spximg_out=spximg1)
    task.source_bloom_and_cosmic_ray_halo_flagging.run.return_value = Mock(
        spximg_out=spximg2,
        bsh_diag_img=Mock(),
    )
    task.correct_non_linearity.run.return_value = Mock(spximg_out=spximg3)
    task.create_variance_map.run.return_value = Mock(spximg_out=spximg4)
    task.update_persistence.run.return_value = Mock(
        spximg_out=spximg5,
        persistence_state_out=Mock(),
        persistence_image=Mock(),
    )

    task.run(
        spximg=spximg0,
        gain_factors=Mock(),
        nonlinear_pars=Mock(),
        readnoise_pars=Mock(),
    )

    mock_check_l1_dqa.assert_called_once_with(spximg0)


@patch("spx_tasks.l2a.check_l1_dqa")
def test_run_does_not_call_l1_dqa_check_by_default(mock_check_l1_dqa):
    task = make_task(fail_on_bad_l1_dqa=False)

    spximg0 = make_spximg()
    spximg1 = make_spximg()
    spximg2 = make_spximg()
    spximg3 = make_spximg()
    spximg4 = make_spximg()
    spximg5 = make_spximg()

    task.gds_image_association.run.return_value = Mock(spximg_out=spximg1)
    task.source_bloom_and_cosmic_ray_halo_flagging.run.return_value = Mock(
        spximg_out=spximg2,
        bsh_diag_img=Mock(),
    )
    task.correct_non_linearity.run.return_value = Mock(spximg_out=spximg3)
    task.create_variance_map.run.return_value = Mock(spximg_out=spximg4)
    task.update_persistence.run.return_value = Mock(
        spximg_out=spximg5,
        persistence_state_out=Mock(),
        persistence_image=Mock(),
    )

    task.run(
        spximg=spximg0,
        gain_factors=Mock(),
        nonlinear_pars=Mock(),
        readnoise_pars=Mock(),
    )

    mock_check_l1_dqa.assert_not_called()


@patch("spx_tasks.l2a.add_calibration_provenance")
@patch("spx_tasks.l2a.get_calibration_inputs")
@patch("spx_tasks.l2a.ObservationId")
def test_run_quantum_adds_missing_obsid_and_writes_outputs(
    mock_observation_id,
    mock_get_calibration_inputs,
    mock_add_calibration_provenance,
):
    task = object.__new__(L2aTask)

    spximg = make_spximg()

    butlerQC = Mock()
    butlerQC.get.return_value = {
        "spximg": spximg,
        "gain_factors": Mock(),
        "nonlinear_pars": Mock(),
        "readnoise_pars": Mock(),
    }

    inputRefs = Mock()
    inputRefs.spximg.dataId = {"exposure": 12345}

    outputRefs = Mock()

    outputs = Mock()
    outputs.spximg_out = Mock()

    task.run = Mock(return_value=outputs)

    obsid = Mock()
    obsid.sps_id = "2026W01_0001_1D1"
    mock_observation_id.return_value = obsid

    mock_get_calibration_inputs.return_value = {"gain_factors": "abc"}

    task.runQuantum(butlerQC, inputRefs, outputRefs)

    assert spximg.header["OBSID"] == "2026W01_0001_1D1"

    mock_observation_id.assert_called_once_with(12345)
    task.run.assert_called_once_with(**butlerQC.get.return_value)

    mock_get_calibration_inputs.assert_called_once_with(inputRefs)
    mock_add_calibration_provenance.assert_called_once_with(
        outputs.spximg_out,
        {"gain_factors": "abc"},
    )
    butlerQC.put.assert_called_once_with(outputs, outputRefs)


@patch("spx_tasks.l2a.add_calibration_provenance")
@patch("spx_tasks.l2a.get_calibration_inputs")
@patch("spx_tasks.l2a.ObservationId")
def test_run_quantum_does_not_overwrite_existing_obsid(
    mock_observation_id,
    mock_get_calibration_inputs,
    mock_add_calibration_provenance,
):
    task = object.__new__(L2aTask)

    spximg = make_spximg()
    spximg.header["OBSID"] = "existing-obsid"

    butlerQC = Mock()
    butlerQC.get.return_value = {
        "spximg": spximg,
        "gain_factors": Mock(),
        "nonlinear_pars": Mock(),
        "readnoise_pars": Mock(),
    }

    inputRefs = Mock()
    outputRefs = Mock()

    outputs = Mock()
    outputs.spximg_out = Mock()

    task.run = Mock(return_value=outputs)
    mock_get_calibration_inputs.return_value = {}

    task.runQuantum(butlerQC, inputRefs, outputRefs)

    assert spximg.header["OBSID"] == "existing-obsid"
    mock_observation_id.assert_not_called()
    mock_add_calibration_provenance.assert_called_once_with(
        outputs.spximg_out,
        {},
    )
    butlerQC.put.assert_called_once_with(outputs, outputRefs)
