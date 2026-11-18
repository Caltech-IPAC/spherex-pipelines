# Standard library imports
from pathlib import Path
import astropy.units as u
import numpy as np
import copy

# Related third party imports
from astropy.io import fits
from astropy.units import Unit
from astropy.wcs import WCS
from astropy.nddata import VarianceUncertainty
from unittest.mock import patch

# Local application/library specific imports
from spx_data import SPHERExImage, Layer, SpectralWCS
from spx_tasks import CreateSpectralImageConfig, CreateSpectralImageTask
from spx_tests import sample_data, get_5x5sample_spectral_wcs

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small_with_cat.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file)
    # Fake variance for this image
    mu, sigma = 0, 0.1  # Fake mean and standard deviation.
    variance = np.random.normal(mu, sigma, size=spximg.shape)
    spximg_unit = Unit(spximg.unit)
    spximg.uncertainty = VarianceUncertainty(variance, unit=spximg_unit**2)
    # Covert from e/s to MJy / sr
    calib = SPHERExImage(np.full(spximg.data.shape, .1),
                         unit="MJy s / (electron sr)")
    # Convert image data unit from e/s to MJy/sr
    spximg.data *= calib.data
    spximg.unit *= calib.unit
    # Convert variance map unit from (e/s)**2 to (MJy/sr)**2
    variance_esp = spximg.uncertainty.array
    spximg.uncertainty = VarianceUncertainty(
        variance_esp * calib.data**2, unit=spximg.uncertainty.unit * calib.unit**2)

    # Add zodi.
    zodi = sample_data.get_sample_zodi(spximg.data.shape)
    spximg.zodi = zodi

    # Add sample catalog.
    cat = sample_data.get_sample_catalog()
    spximg.catalog = cat

    return spximg


def test_CreateSpectralImage():
    spximg_in = get_sample_img()
    (spectral_wcs_data, spectral_coord_vals) = get_5x5sample_spectral_wcs()
    # TODO: Add Layer.PSF after PSF module is implemented
    config1 = CreateSpectralImageConfig(layers=[Layer.CATALOG, Layer.ZODI])

    task1 = CreateSpectralImageTask(config=config1)

    output = task1.run(copy.deepcopy(spximg_in), spectral_wcs=spectral_wcs_data,
                       he_template=None, aurora_template=None, shuttle_glow_template=None)
    spximg = output.spximg_out

    # > Check basic products
    # Image data
    assert spximg_in.data.shape == spximg.data.shape
    # Variance map
    assert spximg.uncertainty is not None
    # Scientific units
    assert spximg.unit is not None
    # Image has scientific units
    assert output.spximg_out.unit == u.Unit("MJy / sr"), (
        f'Wrong image unit: {output.image.unit}')
    # Variance map has scientific units
    variance = output.spximg_out.uncertainty
    assert variance.unit == u.Unit("MJy / sr")**2, (
        f'Wrong variance unit: {variance.unit}')
    # WCS
    assert spximg.wcs is not None
    # Flag bitmap
    assert spximg.flags is not None

    # > Check for presence of layers
    assert spximg.catalog is not None
    # TODO: assert spximg.psf is not None
    assert spximg.zodi is not None

    # spectral WCS
    assert isinstance(spximg.spectral_wcs, SpectralWCS)
    spectral_wcs = spximg.spectral_wcs
    hdul = fits.HDUList([fits.PrimaryHDU(), spectral_wcs.tab])
    hdr = spectral_wcs.hdr
    wcs = WCS(hdr, hdul, key=spectral_wcs.WCS_KEY)
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    coord_vals = np.stack(
        (wcs.pixel_to_world_values(x, y)[0].astype(np.int32),
         wcs.pixel_to_world_values(x, y)[1].astype(np.int32),
         ), axis=-1
    )
    np.array_equal(coord_vals, spectral_coord_vals)

    # > Check for absence of layers
    config2 = CreateSpectralImageConfig(layers=[])
    task2 = CreateSpectralImageTask(config=config2)
    output = task2.run(copy.deepcopy(spximg_in), None, None, None, None)
    spximg = output.spximg_out
    assert spximg.catalog is None
    # TODO: assert spximg.psf is None
    assert spximg.zodi is None

    # > the presence of the required metadata will be checked in DQA module

    # Check for header keyword checks for He / Aurora / Shuttle glow
    # Make deterministic “templates” (exact content doesn't matter for this test)
    shape = spximg_in.data.shape
    he_template_img = SPHERExImage(np.zeros(shape, dtype=float), unit=spximg_in.unit)
    he_template_img.header["HE_WLAIR"] = 1.083
    he_template_img.header["HE_HWAIR"] = 0.027 + 0.007
    he_template_img.header["HE_HWBKG"] = 0.035 + 0.010
    he_template_img.header["HIERARCH HE_WLTOP1"] = 1.0737
    he_template_img.header["HIERARCH HE_WLTOP2"] = 1.0915
    aurora_template_img = SPHERExImage(np.zeros(shape, dtype=float), unit=spximg_in.unit)
    shuttle_template_img = SPHERExImage(np.zeros(shape, dtype=float), unit=spximg_in.unit)

    # Helper: force detector id without depending on sample file headers
    def _force_band(img: SPHERExImage, band: int) -> SPHERExImage:
        img2 = copy.deepcopy(img)
        img2.getArrayId = lambda: band  # monkeypatch instance method
        return img2

    spximg_det1 = _force_band(spximg_in, 1)
    spximg_det5 = _force_band(spximg_in, 5)

    def _fake_fit_airglow_he(spximg, band, he_template, spectral_wcs, params, **kwargs):
        assert np.isclose(params["WL_AIR"], 1.083)
        assert np.isclose(params["HW_AIR"], 0.027 + 0.007)
        assert np.isclose(params["HW_BKG"], 0.035 + 0.010)
        assert np.allclose(params["WL_TOP"], [1.0737, 1.0915])

        return {"ampl": 7.89, "xslope": 1.0e-4, "chi2r": 1.11}

    # Patch airglow/correlation helpers so this test only checks header-writing behavior
    def _fake_attach_airglow_fit_to_header(header, fit_params):
        # Minimal set of representative keywords your DQA expects to exist
        header["HIERARCH HEAG_AMPL"] = (float(fit_params.get("ampl", 1.23)),
                                        "He airglow model amplitude")
        header["HIERARCH HEAG_XSLOPE"] = (float(fit_params.get("xslope", 4.56e-4)),
                                          "He airglow x-gradient")
        header["HIERARCH HEAG_CHI2R"] = (float(fit_params.get("chi2r", 2.34)),
                                         "Reduced chi^2 of He airglow fit")

    with patch("spx_tasks.create_spectral_image.fit_airglow_he",
               side_effect=_fake_fit_airglow_he), \
         patch("spx_tasks.create_spectral_image.attach_airglow_fit_to_header",
               side_effect=_fake_attach_airglow_fit_to_header), \
         patch("spx_tasks.create_spectral_image.correlate_image",
               return_value=0.321):

        # Detector 1: He + Aurora
        out1 = task1.run(copy.deepcopy(spximg_det1),
                         spectral_wcs=spectral_wcs_data,
                         he_template=he_template_img,
                         aurora_template=aurora_template_img,
                         shuttle_glow_template=None)
        hdr1 = out1.spximg_out.header

        # He-fit keywords exist
        for k in ("HIERARCH HEAG_AMPL", "HIERARCH HEAG_XSLOPE", "HIERARCH HEAG_CHI2R"):
            assert k in hdr1, f"Missing expected He airglow header keyword {k} for detector 1"

        # Aurora/Shuttle correlation keywords exist (AURORA computed, SHUTT default)
        assert "HIERARCH AURORA_CORR" in hdr1, "Missing HIERARCH AURORA_CORR for detector 1"
        str1 = "Missing HIERARCH SHUTTLE_GLOW_CORR default for detector 1"
        assert "HIERARCH SHUTTLE_GLOW_CORR" in hdr1, str1

        # Sanity: aurora is finite number
        assert np.isfinite(hdr1["HIERARCH AURORA_CORR"]), "AURORA_CORR is not finite for detector 1"

        # Detector 5: Shuttle glow
        out5 = task1.run(copy.deepcopy(spximg_det5),
                         spectral_wcs=spectral_wcs_data,
                         he_template=None,
                         aurora_template=None,
                         shuttle_glow_template=shuttle_template_img)
        hdr5 = out5.spximg_out.header

        # Correlation keywords exist (SHUTT computed, AURORA default)
        assert "HIERARCH SHUTTLE_GLOW_CORR" in hdr5, "Missing HIERARCH SHUTTLE_GLOW_CORR for detector 5"
        assert "HIERARCH AURORA_CORR" in hdr5, "Missing HIERARCH AURORA_CORR default for detector 5"

        # Sanity: shuttle corr is finite number
        str2 = "SHUTTLE_GLOW_CORR is not finite for detector 5"
        assert np.isfinite(hdr5["HIERARCH SHUTTLE_GLOW_CORR"]), str2


def _run_with_headers(headers):
    spximg_in = get_sample_img()
    for header in headers:
        spximg_in.header[header] = 1

    config = CreateSpectralImageConfig(layers=[Layer.CATALOG, Layer.ZODI])
    task = CreateSpectralImageTask(config=config)

    output = task.run(copy.deepcopy(spximg_in), spectral_wcs=None,
                      he_template=None, aurora_template=None,
                      shuttle_glow_template=None)
    return output.spximg_out


def test_retained_l2b_bsh_headers():
    retained = [
        'SATSTRKS',
        'NBLMSRCS',
        'NSNOWSRC',
        'NBLM5SCS',
        'SNW7-5CT',
        'SNW3-1CT',
        'SNW2-5CT',
        'SNW6-5CT',
    ]
    spximg = _run_with_headers(retained)
    for header in retained:
        assert header in spximg.header, f"Expected '{header}' to be retained"


def test_removed_l2b_headers():
    removed = [
        "SNWCIRCL",
        "HIGHTRNS",
        "MXTRNSFR",
        "XTENDRAD",
        "MINOVRFL",
        "WTAVGCTR",
        "MAXOVRLP",
        "MAXSNWLP",
        "SATHALOM",
        "SATHALOF",
        "BRDRMIND",
    ]
    spximg = _run_with_headers(removed)
    for header in removed:
        assert header not in spximg.header, f"Expected '{header}' to be removed"
