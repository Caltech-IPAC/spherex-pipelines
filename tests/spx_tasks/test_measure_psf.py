# Standard library imports
from pathlib import Path
import contextlib
import os
import re
import tempfile

# Related third party imports
import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage, PSFZoneData, EPSFModels
from spx_tasks.measure_psf import MeasurePSFConfig, MeasurePSFTask, fwhm_neff
from spx_data.psf import spx_psf_reader
from spx_butler.defaults import fake_products


test_root = Path(__file__).parent.parent

# Helper functions & fixtures =================================================
def get_sim_img() -> tuple[SPHERExImage, Table]:
    """Get input simulated SPHEREx level 2 image."""
    sample_file = test_root / "data/simplified_psf.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file, hdu_layers=[])

    cat_file = test_root / "data/simplified_cat.fits"
    with fits.open(cat_file) as hdul_cat:
        cat = Table(hdul_cat[1].data)
        if "ABmag" in cat.colnames and "mag_det1" not in cat.colnames:
            cat.rename_column("ABmag", "mag_det1")

    return spximg, cat

@pytest.fixture
def psf_calib(request) -> PSFZoneData | SPHERExImage:
    """Fixture to provide PSF data.

    Parameters
    ----------
    request : `pytest.FixtureRequest`
        Contains test request context via `param` attribute with `str` value.
        If "psf", returns legacy PSF. If "epsf", returns effective PSF.

    Returns
    -------
    `spx_data.PSFZoneData` | `spx_data.SPHERExImage`
        Image containing PSF data, accessible via `PSF` (`EPSF`) layer or
        attribute `psfcube` (`epsf_table`).
    """
    if request.param == "psf":
        sample_file = test_root / "data/psf_lib.fits"
        assert sample_file.exists(), f"Sample file {sample_file} does not exist."
        return spx_psf_reader(sample_file)
    elif request.param == "epsf":
        return fake_products.fake_epsf(detector=1)


# Test cases ==================================================================
def test_fwhm_neff():
    """Test Gaussian FWHM calculation."""
    neff = 42.5
    fwhm = fwhm_neff(neff)
    assert isinstance(fwhm, u.Quantity)
    assert fwhm.unit == u.arcsec


class TestMeasurePSFTask():
    """Test Measure PSF task."""

    @pytest.mark.parametrize("psf_calib, write_epsf_bintable", [("psf", False), ("epsf", True)],
                             indirect=["psf_calib"])
    @pytest.mark.optional_deps  # psf_calib builds fake_PSF(), which needs the simulator
    def test_psf_extensions(self, psf_calib: PSFZoneData | SPHERExImage, write_epsf_bintable: bool):
        # Set up config and run task
        config = MeasurePSFConfig()
        config.write_epsf_bintable = write_epsf_bintable
        task = MeasurePSFTask(config=config)
        spximg, srccat = get_sim_img()
        output = task.run(spximg=spximg, srccat=srccat,
                          psf_avg=psf_calib if not write_epsf_bintable else None,
                          epsf=psf_calib if write_epsf_bintable else None)

        # Check general outputs
        assert output.spximg_out is not None
        assert output.spximg_out.data.shape == spximg.data.shape
        assert output.jitter_psf is None  # Simplified task returns these as None
        assert output.deconvolved_stack_bins is None  # Simplified task returns these as None

        # Check PSF extensions
        if write_epsf_bintable:
            assert output.spximg_out.epsf_table is not None
            assert isinstance(output.spximg_out.epsf_table, fits.BinTableHDU)
            assert output.spximg_out.epsf_table.name == "EPSF"
            tab = output.spximg_out.epsf_table.data
            assert "EPSF" in tab.names
            assert len(tab) == len(psf_calib.epsf_table.data)
            loaded_psf = EPSFModels.from_hdu(output.spximg_out.layers["EPSF"])
        else:
            assert output.spximg_out.psfcube is not None
            assert isinstance(output.spximg_out.psfcube, PSFZoneData)
            assert output.spximg_out.psfcube.data.shape == psf_calib.data.shape
            assert output.spximg_out.epsf_table is None
            loaded_psf = PSFZoneData.from_hdu(output.spximg_out.layers["PSF"])

        assert loaded_psf is not None

    @pytest.mark.parametrize("psf_calib, write_epsf_bintable, match", [
        ("epsf", False, "but psf_avg input is None"),
        ("psf", True, "but epsf input is None")],
        indirect=["psf_calib"])
    @pytest.mark.optional_deps  # psf_calib builds fake_PSF(), which needs the simulator
    def test_opt_cnxn(self, psf_calib: PSFZoneData | SPHERExImage, write_epsf_bintable: bool, match: str):
        # Set up config and task
        config = MeasurePSFConfig()
        config.write_epsf_bintable = write_epsf_bintable
        task = MeasurePSFTask(config=config)
        spximg, srccat = get_sim_img()

        # No silent failures should occur if input is missing
        with pytest.raises(ValueError, match=re.escape(match)):
            task.run(spximg=spximg, srccat=srccat,
                     psf_avg=psf_calib if write_epsf_bintable else None,
                     epsf=psf_calib if not write_epsf_bintable else None)


    @pytest.mark.parametrize("psf_calib, write_epsf_bintable, do_psf_omega", [
        ("epsf", True, True),
        ("psf", False, True)],
        indirect=["psf_calib"])
    @pytest.mark.optional_deps  # psf_calib builds fake_PSF(), which needs the simulator
    def test_hdr_products(self, psf_calib: PSFZoneData | SPHERExImage, write_epsf_bintable: bool,
                          do_psf_omega: bool):
        # Set up config and run task
        config = MeasurePSFConfig()
        config.write_epsf_bintable = write_epsf_bintable
        config.do_psf_omega = do_psf_omega
        task = MeasurePSFTask(config=config)
        spximg, srccat = get_sim_img()
        output = task.run(spximg=spximg, srccat=srccat,
                          psf_avg=psf_calib if not write_epsf_bintable else None,
                          epsf=psf_calib if write_epsf_bintable else None)

        # Check general outputs
        assert output.spximg_out is not None
        assert output.spximg_out.data.shape == spximg.data.shape
        assert output.jitter_psf is None  # Simplified task returns these as None
        assert output.deconvolved_stack_bins is None  # Simplified task returns these as None

        # Check header products
        assert "PSF_FWHM" in output.spximg_out.header
        assert "PSF_OMEGA" in output.spximg_out.header
        assert np.isfinite(output.spximg_out.header["PSF_FWHM"])
        assert np.isfinite(output.spximg_out.header["PSF_OMEGA"])

    @pytest.mark.parametrize("psf_calib, write_epsf_bintable, match", [
        ("epsf", True, "not yet supported for effective PSF"),
        ("psf", False, None)],
        indirect=["psf_calib"])
    @pytest.mark.optional_deps  # psf_calib builds fake_PSF(), which needs the simulator
    def test_plotting(self, psf_calib: PSFZoneData | SPHERExImage, write_epsf_bintable: bool, match: str):
        tmpdir = tempfile.TemporaryDirectory()
        plot_path = Path(tmpdir.name) / "test_plots"

        # Raise error if using EPSF; otherwise, generate plots
        ctx = pytest.raises(NotImplementedError, match=re.escape(match)) if match \
            else contextlib.nullcontext()
        with ctx:
            config = MeasurePSFConfig(write_epsf_bintable=write_epsf_bintable,
                                      plot_path=str(plot_path))

        if match is None:
            task = MeasurePSFTask(config=config)
            spximg, srccat = get_sim_img()
            task.run(spximg=spximg, srccat=srccat,
                     psf_avg=psf_calib if not write_epsf_bintable else None,
                     epsf=psf_calib if write_epsf_bintable else None,
                     plt=True)
            assert len(os.listdir(plot_path)) > 0
