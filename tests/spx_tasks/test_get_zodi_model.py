import pytest
from pathlib import Path
import numpy as np
from astropy.table import Table
from spx_data import SPHERExImage, Zodi, spherex_image_reader
from spx_tasks import GetZodiModelConfig, GetZodiModelTask, hasSPHERExSimulator


test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


class DummyHDU:
    def __init__(self, data):
        self.data = data


def make_dummy_spectral_wcs(shape, cwave_um=1.0):
    return {"CWAVE": DummyHDU(np.full(shape, cwave_um, dtype=float))}


def make_zodi_scaling_table(scale=1.0, offset=0.0):
    return Table(
        {
            "wavelength_um": np.array([0.5, 1.0, 1.5], dtype=float),
            "ZodiScale": np.array([scale, scale, scale], dtype=float),
            "ZodiOffset_nWm2sr": np.array([offset, offset, offset], dtype=float),
        }
    )


@pytest.mark.skipif(not hasSPHERExSimulator, reason="SPHEREx Simulator is not installed")
def test_GetZodiModel_basic():
    config = GetZodiModelConfig()
    config.zodi_subsample = 1
    config.sparse_bandpass = False

    task = GetZodiModelTask(config=config)
    spximg = get_sample_img()

    zodi_scaling = make_zodi_scaling_table(scale=1.0, offset=0.0)
    spectral_wcs = make_dummy_spectral_wcs(spximg.data.shape, cwave_um=1.0)

    output = task.run(
        spximg=spximg,
        zodi_scaling=zodi_scaling,
        spectral_wcs=spectral_wcs,
    )

    zodi = output.spximg_out.zodi
    assert isinstance(zodi, Zodi)
    assert spximg.data.shape == output.spximg_out.data.shape
    assert spximg.data.shape == zodi.zodi_data.shape
    assert np.all(np.isfinite(zodi.zodi_data))


@pytest.mark.skipif(not hasSPHERExSimulator, reason="SPHEREx Simulator is not installed")
def test_GetZodiModel_run_uses_build_zodi_correction_maps(monkeypatch):
    config = GetZodiModelConfig()
    config.zodi_subsample = 1
    config.sparse_bandpass = False

    task = GetZodiModelTask(config=config)
    spximg = get_sample_img()
    spectral_wcs = make_dummy_spectral_wcs(spximg.data.shape, cwave_um=1.0)
    zodi_scaling = make_zodi_scaling_table(scale=1.0, offset=0.0)

    def fake_maps_raise(zodi_scaling, spectral_wcs):
        raise RuntimeError("patch hit")

    wrapped_run = GetZodiModelTask.run.__wrapped__
    monkeypatch.setitem(
        wrapped_run.__globals__,
        "_build_zodi_correction_maps",
        fake_maps_raise,
    )

    with pytest.raises(RuntimeError, match="patch hit"):
        task.run(
            spximg=spximg,
            zodi_scaling=zodi_scaling,
            spectral_wcs=spectral_wcs,
        )
