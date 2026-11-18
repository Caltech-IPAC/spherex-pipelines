import numpy as np
import pytest
from astropy.wcs import WCS
import spx_utils.l2bwcs_utils as mod


def dummy_header(obsid, detector, finast=0, wcs=None):
    wcs = wcs or make_wcs()
    hdr = wcs.to_header(relax=True)
    hdr["OBSID"] = obsid
    hdr["FINAST"] = finast
    hdr["DETECTOR"] = detector
    return hdr


def make_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crpix = [1000.5, 1001.5]
    w.wcs.crval = [12.34, -45.67]
    w.wcs.cd = np.array([[-1 / 3600.0, 0.0], [0.0, 1 / 3600.0]])
    return w


def test_opposite_detector_valid():
    assert mod.opposite_detector(1) == 4
    assert mod.opposite_detector(4) == 1
    assert mod.opposite_detector(2) == 5
    assert mod.opposite_detector(6) == 3


def test_opposite_detector_invalid():
    with pytest.raises(ValueError, match="Unsupported detector"):
        mod.opposite_detector(99)


def test_to_builtin_numpy_scalar():
    assert mod._to_builtin(np.int64(5)) == 5
    assert isinstance(mod._to_builtin(np.int64(5)), int)

    assert mod._to_builtin(np.float32(1.25)) == pytest.approx(1.25)


def test_to_builtin_python_scalar_passthrough():
    assert mod._to_builtin(7) == 7
    assert mod._to_builtin("abc") == "abc"


def test_build_l2bwcs_payload(monkeypatch):
    class DummyObservationId:
        def __init__(self, obsid):
            self.pointing_id = 202610202022140
            self.large_slew_counter = 221
            self.small_slew_counter = 4
            self.slewgroup_id = 202610202221
            self.slewgroup_counter_id = 202610202221004
            self.plan_name = "2026W10_2B"
            self.sps_id = "SPS_TEST"

    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)

    header = dummy_header(obsid="202610202022140", detector=2, finast=3)

    payload = mod.build_l2bwcs_payload(header, source_collection="u/test/run")

    assert payload["observation"] == 202610202022140
    assert payload["obsid"] == "202610202022140"
    assert payload["detector"] == 2
    assert payload["dichroicpair"] == 2
    assert payload["finast"] == 3
    assert payload["source_collection"] == "u/test/run"
    assert payload["large_slew_counter"] == 221
    assert payload["small_slew_counter"] == 4
    assert payload["slewgroup_id"] == 202610202221
    assert payload["slewgroup_counter_id"] == 202610202221004
    assert payload["plan_name"] == "2026W10_2B"
    assert payload["sps_id"] == "SPS_TEST"

    hdr = payload["wcs_header"]
    assert hdr["CTYPE1"] == "RA---TAN"
    assert hdr["CTYPE2"] == "DEC--TAN"
    assert "CRPIX1" in hdr
    assert "CRVAL1" in hdr


def test_build_l2bwcs_payload_default_finast(monkeypatch):
    class DummyObservationId:
        def __init__(self, obsid):
            self.pointing_id = 1
            self.large_slew_counter = 2
            self.small_slew_counter = 3
            self.slewgroup_id = 4
            self.slewgroup_counter_id = 5
            self.plan_name = "PLAN"
            self.sps_id = "SPS"

    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)

    header = dummy_header(obsid="202610202022140", detector=1)
    header.pop("FINAST")

    payload = mod.build_l2bwcs_payload(header, source_collection="u/test/run")

    assert payload["finast"] == -999


def test_payload_to_wcs_round_trip(monkeypatch):
    class DummyObservationId:
        def __init__(self, obsid):
            self.pointing_id = 202610202022140
            self.large_slew_counter = 221
            self.small_slew_counter = 4
            self.slewgroup_id = 202610202221
            self.slewgroup_counter_id = 202610202221004
            self.plan_name = "2026W10_2B"
            self.sps_id = "SPS_TEST"

    monkeypatch.setattr(mod, "ObservationId", DummyObservationId)

    original = make_wcs()
    header = dummy_header(obsid="202610202022140", detector=5, finast=0, wcs=original)

    payload = mod.build_l2bwcs_payload(header, source_collection="u/test/run")
    rebuilt = mod.payload_to_wcs(payload)

    assert rebuilt.wcs.ctype[0] == original.wcs.ctype[0]
    assert rebuilt.wcs.ctype[1] == original.wcs.ctype[1]
    assert rebuilt.wcs.crpix[0] == pytest.approx(original.wcs.crpix[0])
    assert rebuilt.wcs.crpix[1] == pytest.approx(original.wcs.crpix[1])
    assert rebuilt.wcs.crval[0] == pytest.approx(original.wcs.crval[0])
    assert rebuilt.wcs.crval[1] == pytest.approx(original.wcs.crval[1])
