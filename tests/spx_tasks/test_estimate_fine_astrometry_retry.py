import copy
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import Sip, WCS

import spx_tasks.estimate_fine_astrometry_retry as mod


def make_basic_wcs(crpix=(1020.5, 1020.5), crval=(10.0, 20.0), with_sip=False):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = list(crpix)
    w.wcs.crval = list(crval)
    w.wcs.cd = np.array([[-1.0 / 3600.0, 0.0], [0.0, 1.0 / 3600.0]])

    if with_sip:
        a = np.zeros((3, 3))
        b = np.zeros((3, 3))
        ap = np.zeros((2, 2))
        bp = np.zeros((2, 2))
        w.sip = Sip(a, b, ap, bp, crpix)

    return w


class DummyLog:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass


class DummyImage:
    def __init__(self, array_id=4, wcs=None, obsid=202610202022142):
        self._array_id = array_id
        self.wcs = wcs or make_basic_wcs()
        self.header = {"OBSID": obsid}
        self.meta = {}

    def getArrayId(self):
        return self._array_id


def make_task():
    task = object.__new__(mod.EstimateFineAstrometryRetryTask)
    task.config = SimpleNamespace(
        dichroic_pair_crpix2_offset=11.0,
        keep_target_crval=True,
        keep_target_crpix=False,
        pv_to_sip=False,
        do_write_diagnostics=False,
    )
    task.log = DummyLog()
    task.metadata = {}
    task._fullName = "estimate_fine_astrometry_retry"
    return task


def test_get_retry_pair_crpix2_shift_valid_pairs():
    task = make_task()

    assert task._get_retry_pair_crpix2_shift(4, 1, 11.0) == -11.0
    assert task._get_retry_pair_crpix2_shift(1, 4, 11.0) == 11.0
    assert task._get_retry_pair_crpix2_shift(5, 2, 11.0) == -11.0
    assert task._get_retry_pair_crpix2_shift(2, 5, 11.0) == 11.0
    assert task._get_retry_pair_crpix2_shift(6, 3, 11.0) == -11.0
    assert task._get_retry_pair_crpix2_shift(3, 6, 11.0) == 11.0


def test_get_retry_pair_crpix2_shift_invalid_pair_raises():
    task = make_task()

    with pytest.raises(ValueError, match="Invalid retry donor/target pair"):
        task._get_retry_pair_crpix2_shift(2, 1, 11.0)


def test_seed_from_donor_wcs_same_detector_uses_donor_crpix_plus_slew_shift(monkeypatch):
    task = make_task()
    target_wcs = make_basic_wcs(crpix=(100.0, 200.0))
    spximg = DummyImage(array_id=2, wcs=target_wcs)

    donor_wcs = make_basic_wcs(crpix=(300.0, 400.0), with_sip=False)
    donor_payload = {
        "detector": 2,
        "observation": 202610202022142,
        "obsid": 202610202022142,
        "finast": 0,
        "retry_small_slew_shift_x": 3.0,
        "retry_small_slew_shift_y": -5.0,
    }

    target_sip = make_basic_wcs(with_sip=True).sip

    monkeypatch.setattr(mod, "payload_to_wcs", lambda payload: donor_wcs)

    out = task._seed_from_donor_wcs(spximg, donor_payload, target_sip)

    assert out is spximg
    assert np.allclose(out.wcs.wcs.crpix, [303.0, 395.0])
    assert out.wcs.sip is target_sip

    assert out.header["HIERARCH RETRY_DONOR"] == 2
    assert out.header["HIERARCH RETRY_DONOROBS"] == 202610202022142
    assert out.header["HIERARCH RETRY_DONORFIN"] == 0
    assert out.header["HIERARCH RETRY_WCSMODE"] == "adjacent_small_slew_same_detector"
    assert out.header["HIERARCH RETRY_SLEWDX"] == 3.0
    assert out.header["HIERARCH RETRY_SLEWDY"] == -5.0
    assert out.header["HIERARCH RETRY_DICHDY"] == 0.0
    assert out.header["HIERARCH ASTRONETSOLVED"] is False
    assert out.header["HIERARCH RETRY_SIPSRC"] == "input_sip"

    assert "SIP" in out.wcs.wcs.ctype[0]
    assert "SIP" in out.wcs.wcs.ctype[1]


def test_seed_from_donor_wcs_dichroic_uses_target_linear_terms_and_applies_crpix2_shift(monkeypatch):
    task = make_task()

    target_wcs = make_basic_wcs(
        crpix=(1000.0, 1001.0),
        crval=(30.0, 40.0),
    )
    target_wcs.wcs.pc = np.array([[0.9, 0.1], [-0.1, 0.9]])
    target_wcs.wcs.cdelt = np.array([-1.0 / 3600.0, 1.0 / 3600.0])

    spximg = DummyImage(array_id=4, wcs=target_wcs)

    donor_wcs = make_basic_wcs(
        crpix=(500.0, 600.0),
        crval=(10.0, 20.0),
        with_sip=False,
    )

    donor_payload = {
        "detector": 1,
        "observation": 202610202022141,
        "obsid": 202610202022141,
        "finast": 0,
    }

    target_sip = make_basic_wcs(with_sip=True).sip

    monkeypatch.setattr(mod, "payload_to_wcs", lambda payload: donor_wcs)

    out = task._seed_from_donor_wcs(spximg, donor_payload, target_sip)

    assert out is spximg

    # New behavior: dichroic pair uses target CRPIX, then applies CRPIX2 shift.
    assert np.allclose(out.wcs.wcs.crpix, [1000.0, 990.0])
    assert np.allclose(out.wcs.wcs.crval, [30.0, 40.0])
    assert out.wcs.sip is target_sip

    assert out.header["HIERARCH RETRY_DONOR"] == 1
    assert out.header["HIERARCH RETRY_DONOROBS"] == 202610202022141
    assert out.header["HIERARCH RETRY_DONORFIN"] == 0
    assert out.header["HIERARCH RETRY_WCSMODE"] == "same_small_slew_dichroic"
    assert out.header["HIERARCH RETRY_SLEWDX"] == 0.0
    assert out.header["HIERARCH RETRY_SLEWDY"] == 0.0
    assert out.header["HIERARCH RETRY_DICHDY"] == -11.0
    assert out.header["HIERARCH ASTRONETSOLVED"] is False
    assert out.header["HIERARCH RETRY_SIPSRC"] == "input_sip"

    assert "SIP" in out.wcs.wcs.ctype[0]
    assert "SIP" in out.wcs.wcs.ctype[1]


def test_seed_from_donor_wcs_same_row_uses_target_crpix(monkeypatch):
    task = make_task()

    spximg = DummyImage(
        array_id=2,
        wcs=make_basic_wcs(crpix=(111.0, 222.0), crval=(33.0, 44.0)),
    )
    donor_wcs = make_basic_wcs(crpix=(777.0, 888.0), crval=(11.0, 22.0), with_sip=False)

    donor_payload = {
        "detector": 3,
        "observation": 202610202022143,
        "obsid": 202610202022143,
        "finast": 0,
    }

    target_sip = make_basic_wcs(with_sip=True).sip

    monkeypatch.setattr(mod, "payload_to_wcs", lambda payload: donor_wcs)

    out = task._seed_from_donor_wcs(spximg, donor_payload, target_sip)

    assert out is spximg
    assert np.allclose(out.wcs.wcs.crpix, [111.0, 222.0])
    assert np.allclose(out.wcs.wcs.crval, [33.0, 44.0])
    assert out.wcs.sip is target_sip

    assert out.header["HIERARCH RETRY_DONOR"] == 3
    assert out.header["HIERARCH RETRY_DONOROBS"] == 202610202022143
    assert out.header["HIERARCH RETRY_DONORFIN"] == 0
    assert out.header["HIERARCH RETRY_WCSMODE"] == "same_small_slew_same_row"
    assert out.header["HIERARCH RETRY_DICHDY"] == 0.0
    assert out.header["HIERARCH ASTRONETSOLVED"] is False
    assert out.header["HIERARCH RETRY_SIPSRC"] == "input_sip"

    assert "SIP" in out.wcs.wcs.ctype[0]
    assert "SIP" in out.wcs.wcs.ctype[1]


def test_seed_from_donor_wcs_invalid_combination_raises(monkeypatch):
    task = make_task()

    spximg = DummyImage(array_id=1, wcs=make_basic_wcs())
    donor_payload = {
        "detector": 5,
        "observation": 1,
        "obsid": 1,
        "finast": 0,
    }

    monkeypatch.setattr(mod, "payload_to_wcs", lambda payload: make_basic_wcs())

    with pytest.raises(ValueError, match="Unsupported retry donor/target combination"):
        task._seed_from_donor_wcs(spximg, donor_payload, sip=make_basic_wcs(with_sip=True).sip)


def test_seed_from_donor_wcs_requires_sip(monkeypatch):
    task = make_task()

    spximg = DummyImage(array_id=2, wcs=make_basic_wcs())
    donor_payload = {
        "detector": 2,
        "observation": 202610202022142,
        "obsid": 202610202022142,
        "finast": 0,
    }

    monkeypatch.setattr(mod, "payload_to_wcs", lambda payload: make_basic_wcs())

    with pytest.raises(ValueError, match="requires input sip dataset"):
        task._seed_from_donor_wcs(spximg, donor_payload, sip=None)


def test_run_with_adjacent_same_detector_does_not_force_astronet(monkeypatch):
    task = make_task()

    spximg = DummyImage(array_id=2, wcs=make_basic_wcs())
    srccat = Table({"spherex_class": np.array([1, 0, 1])})
    donor_payload = {"detector": 2, "observation": 123, "obsid": 123, "finast": 0}

    seeded_img = copy.deepcopy(spximg)
    seeded_img.header["HIERARCH RETRY_WCSMODE"] = "adjacent_small_slew_same_detector"

    captured = {}

    def fake_seed(img, donor, sip):
        captured["seed_called"] = True
        assert img is spximg
        assert donor == donor_payload
        assert sip == "SIP"
        return seeded_img

    def fake_base_run(self, **kwargs):
        captured["base_kwargs"] = kwargs
        return SimpleNamespace(spximg_out=kwargs["spximg"])

    monkeypatch.setattr(task, "_seed_from_donor_wcs", fake_seed)
    monkeypatch.setattr(mod.EstimateFineAstrometryTask, "run", fake_base_run)

    out = task.run(
        spximg=spximg,
        srccat=srccat,
        l2bwcs=donor_payload,
        sip="SIP",
        donor_spximg="DONOR_IMAGE",
        verbose=False,
        plot=False,
    )

    assert out.spximg_out is seeded_img
    assert captured["seed_called"] is True
    assert seeded_img.header["HIERARCH RETRY_RUNASTRO"] is False

    kwargs = captured["base_kwargs"]
    assert kwargs["spximg"] is seeded_img
    assert kwargs["srccat"] is srccat
    assert kwargs["index_dir"] is None
    assert kwargs["sip"] == "SIP"
    assert kwargs["donor_spximg"] == "DONOR_IMAGE"
    assert kwargs["max_sep"] == 1.00
    assert kwargs["naxis_1"] == 2040
    assert kwargs["naxis_2"] == 2040
    assert kwargs["verbose"] is False
    assert kwargs["plot"] is False


def test_run_with_dichroic_pair_forces_astronet(monkeypatch):
    task = make_task()

    spximg = DummyImage(array_id=4, wcs=make_basic_wcs())
    srccat = Table({"spherex_class": np.array([1, 0, 1])})
    donor_payload = {"detector": 1, "observation": 123, "obsid": 123, "finast": 0}

    seeded_img = copy.deepcopy(spximg)
    seeded_img.header["HIERARCH RETRY_WCSMODE"] = "same_small_slew_dichroic"

    captured = {}

    def fake_seed(img, donor, sip):
        captured["seed_called"] = True
        return seeded_img

    def fake_base_run(self, **kwargs):
        captured["base_kwargs"] = kwargs
        return SimpleNamespace(spximg_out=kwargs["spximg"])

    monkeypatch.setattr(task, "_seed_from_donor_wcs", fake_seed)
    monkeypatch.setattr(mod.EstimateFineAstrometryTask, "run", fake_base_run)

    out = task.run(
        spximg=spximg,
        srccat=srccat,
        l2bwcs=donor_payload,
        sip="SIP",
        verbose=False,
        plot=False,
    )

    assert out.spximg_out is seeded_img
    assert captured["seed_called"] is True
    assert seeded_img.header["HIERARCH RETRY_RUNASTRO"] is True

    kwargs = captured["base_kwargs"]
    assert kwargs["spximg"] is seeded_img
    assert kwargs["srccat"] is srccat
    assert kwargs["sip"] == "SIP"
    assert kwargs["verbose"] is False
    assert kwargs["plot"] is False


def test_run_without_l2bwcs_runs_astronet_path(monkeypatch):
    task = make_task()

    spximg = DummyImage(array_id=4, wcs=make_basic_wcs())
    srccat = Table({"spherex_class": np.array([1, 0, 1])})

    captured = {}

    def fake_base_run(self, **kwargs):
        captured["base_kwargs"] = kwargs
        return SimpleNamespace(spximg_out=kwargs["spximg"])

    monkeypatch.setattr(mod.EstimateFineAstrometryTask, "run", fake_base_run)

    out = task.run(
        spximg=spximg,
        srccat=srccat,
        l2bwcs=None,
        sip="SIP",
        verbose=False,
        plot=False,
    )

    assert out.spximg_out is spximg

    assert spximg.header["HIERARCH RETRY_WCSMODE"] == "no_donor_run_astronet"
    assert spximg.header["HIERARCH RETRY_RUNASTRO"] is True

    kwargs = captured["base_kwargs"]
    assert kwargs["spximg"] is spximg
    assert kwargs["srccat"] is srccat
    assert kwargs["sip"] == "SIP"
    assert kwargs["verbose"] is False
    assert kwargs["plot"] is False
