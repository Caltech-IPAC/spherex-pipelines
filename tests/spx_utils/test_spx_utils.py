# Standard library
from contextlib import contextmanager
from pathlib import Path
# Third party
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
# Local
from spx_utils.utils import (
    add_detcoord,
    is_device_detcoord,
    get_data_file,
    gains_for_image,
    get_overflow_threshold,
    get_refcat_provenance,
)


# Tests for add_detcoord / is_device_detcoord
def _make_hdulist():
    # Primary + one image HDU, no DETCOORD initially
    prim = fits.PrimaryHDU()
    img = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32))
    return fits.HDUList([prim, img])


def test_add_detcoord_hdulist_sets_sky_comment():
    hdul = _make_hdulist()
    add_detcoord(hdul, device_coord=False)
    # Primary should be annotated
    assert hdul[0].header["DETCOORD"] == "sky"
    # And the comment is applied
    assert "MWIR pixels flipped on X" in hdul[0].header.comments["DETCOORD"]
    # ImageHDU too
    assert hdul[1].header["DETCOORD"] == "sky"


def test_add_detcoord_hdulist_sets_device_comment():
    hdul = _make_hdulist()
    add_detcoord(hdul, device_coord=True)
    assert hdul[0].header["DETCOORD"] == "device"
    assert "pixel coordinate system matching instrument" in hdul[0].header.comments["DETCOORD"]


def test_add_detcoord_unsupported_type_raises():
    with pytest.raises(TypeError):
        add_detcoord(object(), device_coord=True)


def test_is_device_detcoord_default_true_when_missing_key():
    # No DETCOORD present -> should default to 'device' (True)
    hdul = _make_hdulist()
    assert is_device_detcoord(hdul) is True


def test_is_device_detcoord_respects_sky_and_device():
    hdul = _make_hdulist()
    # Mark explicitly 'sky'
    hdul[0].header["DETCOORD"] = "sky"
    assert is_device_detcoord(hdul) is False
    # Now mark explicitly 'device'
    hdul[0].header["DETCOORD"] = "device"
    assert is_device_detcoord(hdul) is True


# Tests for get_data_file (monkeypatch importlib.resources)
@contextmanager
def _as_file_ctx(path: Path):
    # Simple stand-in for importlib.resources.as_file(...) that yields a Path.
    yield path


def test_get_data_file_success(monkeypatch, tmp_path):
    # Create a fake package directory with one file
    pkg_root = tmp_path / "fakepkg"
    pkg_root.mkdir(parents=True)
    fname = "hello.txt"
    fpath = pkg_root / fname
    fpath.write_text("hi")

    # Make importlib.resources.files("fakepkg") return our Path
    monkeypatch.setattr("spx_utils.utils.resources.files",
                        lambda _pkg: pkg_root, raising=True)

    out = get_data_file(fname, package="fakepkg")
    assert out == fpath
    assert out.is_file()


def test_get_data_file_not_found(monkeypatch, tmp_path):
    pkg_root = tmp_path / "fakepkg"
    pkg_root.mkdir(parents=True)

    # Make importlib.resources.files("fakepkg") return our Path
    monkeypatch.setattr("spx_utils.utils.resources.files",
                        lambda _pkg: pkg_root, raising=True)

    with pytest.raises(FileNotFoundError):
        get_data_file("missing.txt", package="fakepkg")


# Tests for gains_for_image
class _MockRP:
    # Choose parameters that satisfy both branches’ broadcast shapes
    # gains = [10, 20], wchannel=2 -> repeated len=4
    # extra_refs=False: slice by jref//2=1 on both ends -> length 2 -> Kactive=2
    # extra_refs=True: columns Kactive + kref = 2 + 2 = 4 -> matches repeated len
    wchannel = 2
    Jactive = 3
    Kactive = 2
    jref = 2
    kref = 2
    nref = 1


def test_gains_for_image_no_extra_refs_shape_and_values():
    rp = _MockRP()
    gains = [10.0, 20.0]
    arr = gains_for_image(gains, rp, extra_refs=False)
    # Shape (Jactive, Kactive) = (3, 2)
    assert arr.shape == (rp.Jactive, rp.Kactive)
    # Repeated gains -> [10,10,20,20], slice [1:-1] -> [10,20]
    expected_row = np.array([10.0, 20.0], dtype=np.float64)
    np.testing.assert_allclose(arr[0], expected_row)


def test_gains_for_image_with_extra_refs_shape_and_values():
    rp = _MockRP()
    gains = [10.0, 20.0]
    arr = gains_for_image(gains, rp, extra_refs=True)
    # Shape (Jactive + jref*nref, Kactive + kref) = (5, 4)
    assert arr.shape == (rp.Jactive + rp.jref * rp.nref, rp.Kactive + rp.kref)
    # Repeated gains -> [10,10,20,20] (no slicing)
    expected_row = np.array([10.0, 10.0, 20.0, 20.0], dtype=np.float64)
    np.testing.assert_allclose(arr[0], expected_row)


# Tests for get_overflow_threshold
def test_get_overflow_threshold_uses_primary_key_times_4():
    hdr = fits.Header()
    hdr["PRIMKEY"] = 123
    val = get_overflow_threshold(hdr, "PRIMKEY")
    assert val == 123 * 4


def test_get_overflow_threshold_fallback_overflow_direct():
    hdr = fits.Header()
    # No primary key -> use OVERFLOW directly
    hdr["OVERFLOW"] = 456
    val = get_overflow_threshold(hdr, "MISSING")
    assert val == 456


def test_get_overflow_threshold_hardcoded_default():
    hdr = fits.Header()  # neither primary nor OVERFLOW
    val = get_overflow_threshold(hdr, "MISSING")
    assert val == 30000


# Test for get_refcat_provenance
def test_get_refcat_provenance_present(caplog):
    t = Table()
    t.meta["user_metadata"] = {
        "refcat_run": "refcat_v073",
        "refcat_version": "SPHEREx_RefCat_v0.73",
    }
    run, ver = get_refcat_provenance(t)
    assert run == "refcat_v073"
    assert ver == "SPHEREx_RefCat_v0.73"
    # No warning in this case
    assert not any("Reference catalog provenance not found" in rec.message for rec in caplog.records)


def test_get_refcat_provenance_missing_warns(caplog):
    t = Table()
    with caplog.at_level("WARNING"):
        run, ver = get_refcat_provenance(t)
    assert run is None and ver is None
    assert any("Reference catalog provenance not found" in rec.message for rec in caplog.records)
