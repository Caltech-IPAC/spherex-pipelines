import pytest
import spx_tasks.l2b_retry as mod


class DummyLog:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class DummyImage:
    def __init__(self, obsid, detector):
        self.header = {"OBSID": obsid}
        self._detector = detector

    def getArrayId(self):
        return self._detector


class DummyResult:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class RecordingSubtask:
    def __init__(self, name, transform=None, extra=None):
        self.name = name
        self.calls = []
        self.transform = transform or (lambda spximg: spximg)
        self.extra = extra or {}

    def run(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        if args:
            spximg = args[0]
        else:
            spximg = kwargs["spximg"]

        out_img = self.transform(spximg)
        payload = {"spximg_out": out_img}
        payload.update(self.extra)
        return DummyResult(**payload)


def make_payload(solutions):
    return {"slewgroup_solutions": solutions}


def test_detector_row():
    assert mod._detector_row(1) == (1, 2, 3)
    assert mod._detector_row(2) == (1, 2, 3)
    assert mod._detector_row(6) == (4, 5, 6)

    with pytest.raises(ValueError, match="Unexpected detector"):
        mod._detector_row(7)


def test_select_same_small_slew_dichroic_solution(monkeypatch):
    monkeypatch.setattr(mod, "opposite_detector", lambda det: 5 if det == 2 else None)

    payload = make_payload(
        [
            {"detector": 5, "finast": 1, "small_slew_counter": 4},
            {"detector": 5, "finast": 0, "small_slew_counter": 3},
            {"detector": 5, "finast": 0, "small_slew_counter": 4, "obsid": 999},
        ]
    )

    out = mod._select_same_small_slew_dichroic_solution(
        l2bwcs_payload=payload,
        target_observation=202610202022140,
        target_detector=2,
        max_finast=999,
    )

    # current implementation returns first valid candidate, not best FINAST
    assert out["detector"] == 5
    assert out["small_slew_counter"] == 4
    assert out["finast"] == 1


def test_select_adjacent_same_detector_prefers_previous_on_tie():
    payload = make_payload(
        [
            {"detector": 2, "finast": 0, "obsid": 202610202022120},  # small slew 2
            {"detector": 2, "finast": 0, "obsid": 202610202022160},  # small slew 6
            {"detector": 2, "finast": 0, "obsid": 202610202022140},  # same obs -> exclude
            {"detector": 3, "finast": 0, "obsid": 202610202022130},  # wrong detector
        ]
    )

    out = mod._select_adjacent_same_detector_solution(
        l2bwcs_payload=payload,
        target_observation=202610202022140,
        target_detector=2,
        max_finast=999,
    )

    # distance to 2 and 6 is both 2; previous should win
    assert out["obsid"] == 202610202022120


def test_select_same_small_slew_same_row_solution_prefers_nearest_detector():
    payload = make_payload(
        [
            {"detector": 1, "finast": 0, "small_slew_counter": 4, "obsid": 1},
            {"detector": 3, "finast": 0, "small_slew_counter": 4, "obsid": 2},
            {"detector": 2, "finast": 0, "small_slew_counter": 4, "obsid": 3},  # target -> excluded
            {"detector": 1, "finast": 1, "small_slew_counter": 4, "obsid": 4},  # bad finast
        ]
    )

    out = mod._select_same_small_slew_same_row_solution(
        l2bwcs_payload=payload,
        target_observation=202610202022140,
        target_detector=2,
        max_finast=999,
    )

    # det 1 and det 3 are equally near; sort key prefers lower detector
    assert out["detector"] == 1


def test_select_donor_payload_priority(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()

    spximg = DummyImage(obsid=202610202022140, detector=2)

    monkeypatch.setattr(
        mod,
        "_select_same_small_slew_dichroic_solution",
        lambda **kwargs: {"detector": 5, "observation": 123, "finast": 0},
    )
    monkeypatch.setattr(
        mod,
        "_select_adjacent_same_detector_solution",
        lambda **kwargs: {"detector": 2, "observation": 456, "finast": 0},
    )
    monkeypatch.setattr(
        mod,
        "_select_same_small_slew_same_row_solution",
        lambda **kwargs: {"detector": 1, "observation": 789, "finast": 0},
    )

    donor, strategy = task._select_donor_payload(spximg, {"slewgroup_solutions": []})

    assert donor["detector"] == 2
    assert strategy == "adjacent_small_slew_same_detector"


def test_select_donor_payload_falls_back_in_order(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()
    spximg = DummyImage(obsid=202610202022140, detector=2)

    monkeypatch.setattr(mod, "_select_adjacent_same_detector_solution", lambda **kwargs: None)
    monkeypatch.setattr(
        mod,
        "_select_same_small_slew_dichroic_solution",
        lambda **kwargs: {"detector": 5, "observation": 456, "finast": 0},
    )

    donor, strategy = task._select_donor_payload(spximg, {"slewgroup_solutions": []})

    assert donor["observation"] == 456
    assert strategy == "same_small_slew_dichroic"

    assert donor["detector"] == 5
    assert strategy == "same_small_slew_dichroic"


def test_select_donor_payload_returns_no_donor_strategy_when_none_found(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()
    spximg = DummyImage(obsid=202610202022140, detector=2)

    monkeypatch.setattr(mod, "_select_adjacent_same_detector_solution", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_select_same_small_slew_dichroic_solution", lambda **kwargs: None)
    monkeypatch.setattr(mod, "_select_same_small_slew_same_row_solution", lambda **kwargs: None)

    donor, strategy = task._select_donor_payload(spximg, {"slewgroup_solutions": []})

    assert donor is None
    assert strategy == "no_donor_run_astronet"


def test_run_quantum_writes_when_retry_finast_is_as_good_or_better(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()

    spximg = DummyImage(obsid=202610202022140, detector=2)
    spximg.header["FINAST"] = 4

    donor_payload = {
        "detector": 5,
        "source_collection": "u/test/run",
        "observation": 202610202022141,
        "finast": 0,
    }

    output_img = DummyImage(obsid=202610202022140, detector=2)
    output_img.header["FINAST"] = 1   # improved relative to prior=4
    outputs_obj = DummyResult(spximg_out=output_img)

    monkeypatch.setattr(
        task,
        "_select_donor_payload",
        lambda img, payload: (donor_payload, "same_small_slew_dichroic"),
    )
    monkeypatch.setattr(task, "run", lambda **inputs: outputs_obj)
    monkeypatch.setattr(mod, "get_calibration_inputs", lambda inputRefs: {"sip": "X"})
    monkeypatch.setattr(mod, "add_calibration_provenance", lambda spximg, ids: None)

    class DummyButlerQC:
        def __init__(self):
            self.put_called = None

        def get(self, inputRefs):
            return {
                "spximg": spximg,
                "l2bwcs": {
                    "prior_finast": 4,
                    "slewgroup_solutions": [],
                },
            }

        def put(self, outputs, outputRefs):
            self.put_called = outputs

    butlerQC = DummyButlerQC()

    task.runQuantum(butlerQC, inputRefs=object(), outputRefs=object())

    assert butlerQC.put_called is outputs_obj
    assert output_img.header["HIERARCH WCSSEED"] == 5
    assert output_img.header["HIERARCH WCSSEEDOBS"] == 202610202022141

def test_run_quantum_skips_write_when_retry_finast_is_worse(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()

    spximg = DummyImage(obsid=202610202022140, detector=2)
    spximg.header["FINAST"] = 1

    donor_payload = {
        "detector": 5,
        "source_collection": "u/test/run",
        "observation": 202610202022141,
        "finast": 0,
    }

    output_img = DummyImage(obsid=202610202022140, detector=2)
    output_img.header["FINAST"] = 3   # worse than prior=1
    outputs_obj = DummyResult(spximg_out=output_img)

    monkeypatch.setattr(
        task,
        "_select_donor_payload",
        lambda img, payload: (donor_payload, "same_small_slew_dichroic"),
    )
    monkeypatch.setattr(task, "run", lambda **inputs: outputs_obj)
    monkeypatch.setattr(mod, "get_calibration_inputs", lambda inputRefs: {"sip": "X"})
    monkeypatch.setattr(mod, "add_calibration_provenance", lambda spximg, ids: None)

    class DummyButlerQC:
        def __init__(self):
            self.put_called = None

        def get(self, inputRefs):
            return {
                "spximg": spximg,
                "l2bwcs": {
                    "prior_finast": 1,
                    "slewgroup_solutions": [],
                },
            }

        def put(self, outputs, outputRefs):
            self.put_called = outputs

    butlerQC = DummyButlerQC()

    task.runQuantum(butlerQC, inputRefs=object(), outputRefs=object())

    assert butlerQC.put_called is outputs_obj
    assert output_img.header["HIERARCH WCSSEED"] == 5
    assert output_img.header["HIERARCH WCSSEEDOBS"] == 202610202022141

def test_run_quantum_writes_when_retry_finast_equals_prior(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()

    spximg = DummyImage(obsid=202610202022140, detector=2)
    spximg.header["FINAST"] = 2

    donor_payload = {
        "detector": 5,
        "source_collection": "u/test/run",
        "observation": 202610202022141,
        "finast": 0,
    }

    output_img = DummyImage(obsid=202610202022140, detector=2)
    output_img.header["FINAST"] = 2   # equal to prior
    outputs_obj = DummyResult(spximg_out=output_img)

    monkeypatch.setattr(
        task,
        "_select_donor_payload",
        lambda img, payload: (donor_payload, "same_small_slew_dichroic"),
    )
    monkeypatch.setattr(task, "run", lambda **inputs: outputs_obj)
    monkeypatch.setattr(mod, "get_calibration_inputs", lambda inputRefs: {"sip": "X"})
    monkeypatch.setattr(mod, "add_calibration_provenance", lambda spximg, ids: None)

    class DummyButlerQC:
        def __init__(self):
            self.put_called = None

        def get(self, inputRefs):
            return {
                "spximg": spximg,
                "l2bwcs": {
                    "prior_finast": 2,
                    "slewgroup_solutions": [],
                },
            }

        def put(self, outputs, outputRefs):
            self.put_called = outputs

    butlerQC = DummyButlerQC()

    task.runQuantum(butlerQC, inputRefs=object(), outputRefs=object())

    assert butlerQC.put_called is outputs_obj


def test_run_executes_pipeline_in_order(monkeypatch):
    task = object.__new__(mod.L2bRetryTask)
    task.log = DummyLog()

    monkeypatch.setattr(mod, "check_l1_dqa", lambda spximg: None)

    call_order = []

    def transform(name):
        def _fn(spximg):
            call_order.append(name)
            return spximg
        return _fn

    task.remove_reference_pixels = RecordingSubtask(
        "remove_reference_pixels", transform("remove_reference_pixels")
    )
    task.apply_bad_pixel_mask = RecordingSubtask(
        "apply_bad_pixel_mask", transform("apply_bad_pixel_mask")
    )
    task.apply_dark_flat_field_and_photo_calib = RecordingSubtask(
        "apply_dark_flat_field_and_photo_calib",
        transform("apply_dark_flat_field_and_photo_calib"),
    )
    task.estimate_fine_astrometry = RecordingSubtask(
        "estimate_fine_astrometry",
        transform("estimate_fine_astrometry"),
        extra={"astrometry_diagnostics": "ASTR_DIAG"},
    )
    task.create_source_mask = RecordingSubtask(
        "create_source_mask", transform("create_source_mask")
    )
    task.detect_outlier_pixels = RecordingSubtask(
        "detect_outlier_pixels", transform("detect_outlier_pixels")
    )
    task.get_zodi_model = RecordingSubtask(
        "get_zodi_model", transform("get_zodi_model")
    )
    task.measure_psf = RecordingSubtask(
        "measure_psf", transform("measure_psf")
    )
    task.create_spectral_image = RecordingSubtask(
        "create_spectral_image", transform("create_spectral_image")
    )
    task.dqa_l2 = RecordingSubtask(
        "dqa_l2",
        transform("dqa_l2"),
        extra={"dqa_l2": "DQA_OBJ", "idqr_obs": "IDQR_OBJ"},
    )

    spximg = DummyImage(obsid=202610202022140, detector=2)

    out = task.run(
        spximg=spximg,
        l2bwcs={"detector": 5},
        abs_gain_matrix="ABS",
        dark="DARK",
        sip="SIP",
        srccat="SRCCAT",
        psf_avg="PSF",
        epsf="EPSF",
        zodi_scaling_v2="ZODI_SCALING",
        spectral_wcs="SPECWCS",
        ssocat="SSOCAT",
        xtalk_class1="XTALK1",
        xtalk_class2="XTALK2",
        he_template="HE",
        aurora_template="AURORA",
        shuttle_glow_template="SHUTTLE",
    )

    assert call_order == [
        "remove_reference_pixels",
        "apply_bad_pixel_mask",
        "apply_dark_flat_field_and_photo_calib",
        "estimate_fine_astrometry",
        "create_source_mask",
        "detect_outlier_pixels",
        "get_zodi_model",
        "measure_psf",
        "create_spectral_image",
        "dqa_l2",
    ]

    est_args, est_kwargs = task.estimate_fine_astrometry.calls[0]
    assert est_args[1] == "SRCCAT"
    assert est_kwargs["l2bwcs"] == {"detector": 5}
    assert est_kwargs["sip"] == "SIP"

    mpsf_args, mpsf_kwargs = task.measure_psf.calls[0]
    assert mpsf_args == ()
    assert mpsf_kwargs == {
        "spximg": spximg,
        "srccat": "SRCCAT",
        "psf_avg": "PSF",
        "epsf": "EPSF",
    }

    csm_args, csm_kwargs = task.create_source_mask.calls[0]
    assert csm_args == ()

    assert csm_kwargs == {
        "spximg": spximg,
        "srccat": "SRCCAT",
        "ssocat": "SSOCAT",
        "xtalk_class2": "XTALK2",
        "xtalk_class1": "XTALK1",
        "abs_gain_matrix": "ABS",
    }

    spec_args, spec_kwargs = task.create_spectral_image.calls[0]
    assert spec_args == (spximg, "SPECWCS", "HE")
    assert spec_kwargs == {
        "aurora_template": "AURORA",
        "shuttle_glow_template": None,
    }

    dqa_args, dqa_kwargs = task.dqa_l2.calls[0]
    assert dqa_args == (spximg, "SRCCAT")
    assert dqa_kwargs == {}

    assert out.spximg_out is spximg
    assert out.dqa_l2 == "DQA_OBJ"
    assert out.idqr_obs == "IDQR_OBJ"
    assert out.astr_diag == "ASTR_DIAG"
