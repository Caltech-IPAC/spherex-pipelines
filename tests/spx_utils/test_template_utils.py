# Standard library imports
import pytest

# Related third party imports
from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.modeling.models import Polynomial2D
from astropy.nddata import VarianceUncertainty
import numpy as np
from astropy.wcs import WCS
from types import SimpleNamespace

# Local application/library specific imports
from spx_data import SPHERExImage, Flag


from spx_utils.template_utils import (create_masks,
                                      fit_airglow_he,
                                      subtract_airglow_he,
                                      attach_airglow_fit_to_header,
                                      read_airglow_fit_from_header,
                                      reconstruct_p_fit_from_results,
                                      correlate_image)
from spx_utils import template_utils as tu

SOURCE_BIT = Flag.SOURCE.value


def _patch_spximg_zodi_getter(monkeypatch, zodi_data_factory=None):
    """Monkeypatch SPHERExImage.zodi property getter.

    Return an object with `.zodi_data` without invoking the real setter
    (which expects a Zodi model with to_hdu()).
    """
    if zodi_data_factory is None:
        def zodi_data_factory(spximg):
            return np.zeros_like(spximg.data, dtype=np.float32)

    def _fake_zodi_getter(self):
        return SimpleNamespace(zodi_data=zodi_data_factory(self))

    # Replace the property on the class (getter-only)
    monkeypatch.setattr(SPHERExImage, "zodi", property(_fake_zodi_getter))


def _make_spectral_wcs_cwave(shape: tuple[int, int],
                             wl_air: float,
                             hw_bkg: float) -> HDUList:
    """Minimal spectral_wcs HDUList containing an ImageHDU named 'CWAVE'.
    CWAVE varies with row so wavelength windows select non-empty regions.
    """
    ny, nx = shape

    # Make sure CWAVE spans beyond the background window.
    # This ensures mask_bkg and mask_top can be non-empty.
    span = 3.0 * hw_bkg
    y = np.linspace(wl_air - span, wl_air + span, ny, dtype=np.float64)
    wmap = np.repeat(y[:, None], nx, axis=1).astype(np.float32)

    hdr = fits.Header()
    hdr["EXTNAME"] = "CWAVE"
    return fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(wmap, header=hdr, name="CWAVE")])


def _make_synthetic_scene(shape: tuple[int, int] = (120, 96),
                          wl_air: float = 1.0830,
                          hw_air: float = 0.01,
                          hw_bkg: float = 0.06,
                          rng_seed: int = 123) -> tuple[SPHERExImage, SPHERExImage, HDUList, dict, dict]:
    """Build a synthetic scene for testing.

    Create scene where:
      data = background_poly(xx, wmap) + k_true * template(wmap,xx) + noise

    Designed to avoid:
      - empty background masks
      - empty WL_TOP masks (mask_top.sum()==0 -> AG_RMS NaN)
    """
    rng = np.random.default_rng(rng_seed)
    ny, nx = shape

    spectral_wcs = _make_spectral_wcs_cwave(shape, wl_air=wl_air, hw_bkg=hw_bkg)
    wmap = spectral_wcs["CWAVE"].data

    # WL_TOP must be inside the CWAVE span AND (usually) outside the line.
    # Put it on the high-wavelength side but still within HW_BKG.
    wl_top_lo = wl_air - 0.5 * hw_air
    wl_top_hi = wl_air + 0.5 * hw_air

    params = {
        "WL_AIR": float(wl_air),
        "HW_AIR": float(hw_air),
        "HW_BKG": float(hw_bkg),
        "WL_TOP": (float(wl_top_lo), float(wl_top_hi)),
    }

    xx, _ = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))

    # True background polynomial in (x, wavelength)
    p_true = Polynomial2D(
        degree=2,
        c0_0=5.0,
        c1_0=0.01,
        c0_1=-1.5,
        c2_0=8.0e-5,
        c1_1=2.0e-4,
        c0_2=0.3,
    )

    # Template: Gaussian line profile in wavelength, mild x modulation
    line_profile = np.exp(-0.5 * ((wmap - wl_air) / (0.35 * hw_air)) ** 2)
    x_mod = 1.0 + 0.10 * (xx - (nx / 2.0)) / (nx / 2.0)
    template = (line_profile * x_mod).astype(np.float32)

    k_true = 7.5

    sigma = 0.10
    noise = rng.normal(0.0, sigma, size=shape).astype(np.float32)

    background = p_true(xx, wmap).astype(np.float32)
    data = background + (k_true * template) + noise

    # Flags: mostly good, sprinkle SOURCE_BIT pixels
    flags = np.zeros(shape, dtype=np.int32)
    src_mask = rng.random(shape) < 0.03
    flags[src_mask] = SOURCE_BIT

    # a few arbitrary bad flags (excluded by default)
    bad_mask = (rng.random(shape) < 0.01)
    flags[bad_mask] = 123

    spximg = SPHERExImage(data=data, flags=flags, unit="MJy/sr")
    spximg.uncertainty = VarianceUncertainty(np.full(shape, sigma**2, dtype=np.float32))

    he_template_img = SPHERExImage(data=template, flags=np.zeros_like(flags), unit="MJy/sr")

    mask_air, mask_good, mask_bkg, mask_all = create_masks(
        wmap, flags, WL_AIR=params["WL_AIR"], HW_AIR=params["HW_AIR"], HW_BKG=params["HW_BKG"]
    )

    truth = {
        "k_true": k_true,
        "mask_good": mask_good,
        "p_true": p_true
    }

    return spximg, he_template_img, spectral_wcs, params, truth


def test_create_masks_flag_logic():
    ny, nx = 10, 12
    wl_air = 1.0
    hw_air = 0.05
    hw_bkg = 0.20

    wmap = np.repeat(np.linspace(wl_air - 0.5, wl_air + 0.5, ny)[:, None], nx, axis=1)
    flags = np.zeros((ny, nx), dtype=np.int32)
    flags[0, 0] = SOURCE_BIT
    flags[1, 1] = 123

    mask_air, mask_good, mask_bkg, mask_all = create_masks(
        wmap, flags, WL_AIR=wl_air, HW_AIR=hw_air, HW_BKG=hw_bkg
    )

    assert mask_air.shape == (ny, nx)
    assert mask_good.shape == (ny, nx)
    assert mask_bkg.shape == (ny, nx)
    assert mask_all.shape == (ny, nx)

    # mask_all should include mask_airglow
    assert np.all(mask_air <= mask_all)

    # Bad-flag pixel must be excluded (use "not" / "==" not "is")
    assert not mask_good[1, 1]
    assert not mask_bkg[1, 1]

    # SOURCE_BIT allowed when mask_source=False
    # Whether it lands in mask_good depends on wavelength
    # Assert it isn't rejected by flag logic
    if mask_air[0, 0]:
        assert mask_good[0, 0]
    else:
        # If it's outside airglow, it could be in bkg annulus
        assert (mask_bkg[0, 0] == (mask_all[0, 0] and (not mask_air[0, 0])))

    # SOURCE_BIT rejected when mask_source=True
    _, mask_good2, mask_bkg2, _ = create_masks(
        wmap, flags, mask_source=True, WL_AIR=wl_air, HW_AIR=hw_air, HW_BKG=hw_bkg
    )
    assert not mask_good2[0, 0]
    assert not mask_bkg2[0, 0]


@pytest.mark.parametrize("fit_back,use_variance,fit_xslope", [
    (True, False, False),
    (True, True, False),
    (False, False, False),
])
def test_fit_airglow_he_recovers_amplitude_and_serializes_background(fit_back, use_variance, fit_xslope):
    spximg, he_template_img, spectral_wcs, params, truth = _make_synthetic_scene()

    results = fit_airglow_he(
        spximg=spximg,
        band=1,
        he_template_img=he_template_img,
        spectral_wcs=spectral_wcs,
        params_airglow_helium=params,
        fit_back=fit_back,
        use_variance=use_variance,
        fit_xslope=fit_xslope,
        verbose=False,
    )

    for k in ("AG_AMPL", "AG_XSLP", "AG_CHI2M", "AG_RMS", "AG_BACK", "AG_BKG", "_NX"):
        assert k in results

    # Ensure rms is finite
    assert np.isfinite(results["AG_RMS"]), "AG_RMS became NaN; mask_top likely empty."

    k_hat = float(results["AG_AMPL"])
    assert np.isfinite(k_hat)

    if fit_back:
        # When background is subtracted, we should recover the true amplitude.
        assert np.isclose(k_hat, truth["k_true"], rtol=0.10, atol=0.5)
    else:
        # When background is NOT subtracted,
        # the optimal k is biased by background/template correlation.
        # The code uses compute_proportional_coefficient(),
        # whose first-iteration solution is:
        #   k = sum(A*B) / sum(A^2)   over mask_good (unit weights)
        A = he_template_img.data
        B = spximg.data
        w = truth["mask_good"]  # return this from _make_synthetic_scene (see below)

        k_expected = np.sum(A[w] * B[w]) / np.sum(A[w] * A[w])

        # Should match closely
        # Sigma clipping should not change much in this synthetic setup
        assert np.isclose(k_hat, k_expected, rtol=0.02, atol=0.1)

    bkg = results["AG_BKG"]
    assert int(bkg["degree"]) == 2
    assert isinstance(bkg["params"], dict)
    for name in ("c0_0", "c1_0", "c0_1", "c2_0", "c1_1", "c0_2"):
        assert name in bkg["params"]
        assert np.isfinite(bkg["params"][name])


def test_attach_read_reconstruct_roundtrip():
    spximg, he_template_img, spectral_wcs, params, _ = _make_synthetic_scene()

    results = fit_airglow_he(
        spximg=spximg,
        band=1,
        he_template_img=he_template_img,
        spectral_wcs=spectral_wcs,
        params_airglow_helium=params,
        fit_back=True,
        use_variance=False,
        fit_xslope=False,
        verbose=False,
    )

    # Must be finite for FITS header writing
    assert np.isfinite(results["AG_RMS"])

    hdr = fits.Header()
    attach_airglow_fit_to_header(hdr, results, prefix="HE_AG_")

    out = read_airglow_fit_from_header(hdr, prefix="HE_AG_")
    for k in ("AG_AMPL", "AG_XSLP", "AG_CHI2M", "AG_RMS", "AG_BACK"):
        assert np.isfinite(out[k])

    p_fit = reconstruct_p_fit_from_results(out)
    assert isinstance(p_fit, Polynomial2D)
    assert int(p_fit.degree) == 2
    for name, val in out["AG_BKG"]["params"].items():
        assert np.isclose(getattr(p_fit, name).value, float(val), rtol=0, atol=0)


def test_subtract_airglow_he_reduces_airglow_residual_in_line_region():
    spximg, he_template_img, spectral_wcs, params, truth = _make_synthetic_scene()

    results = fit_airglow_he(
        spximg=spximg,
        band=1,
        he_template_img=he_template_img,
        spectral_wcs=spectral_wcs,
        params_airglow_helium=params,
        fit_back=True,
        use_variance=False,
        fit_xslope=False,
        verbose=False,
    )

    # Provide _p_fit as required by subtract_airglow_he
    results2 = dict(results)
    results2["_p_fit"] = reconstruct_p_fit_from_results({"AG_BKG": results["AG_BKG"]})

    wmap = spectral_wcs["CWAVE"].data
    mask_airglow, mask_good, _, _ = create_masks(wmap, spximg.flags, **params)

    ny, nx = spximg.data.shape
    xx, _ = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    background_true = truth["p_true"](xx, wmap).astype(np.float32)

    resid_before = (spximg.data - background_true)[mask_airglow & mask_good]
    rms_before = float(np.sqrt(np.mean(resid_before**2)))

    data_sub = subtract_airglow_he(
        spximg=spximg,
        he_template_img=he_template_img,
        spectral_wcs=spectral_wcs,
        params_airglow_helium=params,
        results=results2,
    )

    resid_after = (data_sub - background_true)[mask_airglow & mask_good]
    rms_after = float(np.sqrt(np.mean(resid_after**2)))

    assert rms_after < 0.5 * rms_before


def test_subtract_airglow_he_raises_without_reconstructed_model():
    spximg, he_template_img, spectral_wcs, params, _ = _make_synthetic_scene()

    results = {"AG_AMPL": 1.0, "AG_XSLP": 0.0}  # intentionally missing "_p_fit"
    with pytest.raises(ValueError, match="missing reconstructed _p_fit"):
        subtract_airglow_he(
            spximg=spximg,
            he_template_img=he_template_img,
            spectral_wcs=spectral_wcs,
            params_airglow_helium=params,
            results=results,
        )


def _nanmedian_block_reduce(arr: np.ndarray, block_size: int, func):
    """Minimal block_reduce replacement for tests.
    Assumes arr.shape is divisible by block_size.
    Returns nanmedian over block_size x block_size tiles.
    """
    bs = int(block_size)
    ny, nx = arr.shape
    assert ny % bs == 0 and nx % bs == 0, "test array shape must be divisible by downsamp"
    reshaped = arr.reshape(ny // bs, bs, nx // bs, bs)
    # nanmedian over the within-block axes
    return np.nanmedian(reshaped, axis=(1, 3))


def _make_spximg_with_crval(data: np.ndarray, flags: np.ndarray,
                            ra_deg: float, dec_deg: float) -> SPHERExImage:
    """Build a SPHERExImage with a minimal WCS containing CRVAL.
    """
    spximg = SPHERExImage(data=data, flags=flags, unit="MJy/sr")
    w = WCS(naxis=2)
    w.wcs.crval = [float(ra_deg), float(dec_deg)]
    # Ensure SkyCoord(...) path works
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    spximg.wcs = w
    return spximg


def test_fit_background_ignores_nan_pixels_within_mask_bkg():
    """Regression test for a bug where `zunc = z * 0 + sigma_clipped_stats(z)[2]`
    silently produced NaN weights wherever `z` was NaN (since `nan * 0 == nan`,
    not 0). Even a single NaN pixel inside `mask_bkg` corrupted the weighted
    design matrix passed to astropy's LinearLSQFitter and crashed the fit.
    """
    rng = np.random.default_rng(42)
    ny, nx = 150, 100
    xx, yy = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))

    p_true = Polynomial2D(degree=2, c0_0=2.0, c1_0=0.05, c0_1=0.03,
                          c2_0=0.001, c1_1=0.0005, c0_2=0.0008)
    data = (p_true(xx, yy) + rng.normal(0, 0.01, size=(ny, nx))).astype(np.float64)

    mask_bkg = np.ones((ny, nx), dtype=bool)

    # A single NaN pixel (~0.0067% of the image) reproduces the bug.
    nan_row, nan_col = ny // 2, nx // 2
    data[nan_row, nan_col] = np.nan
    nan_fraction = 1 / data.size
    assert nan_fraction < 0.0001  # well under the 0.01 percent that triggered it

    p_fit, _ = tu.fit_background(data, xx, yy, mask_bkg)

    for name in p_fit.param_names:
        fitted_value = getattr(p_fit, name).value
        assert np.isfinite(fitted_value), f"{name} is not finite: {fitted_value}"
        assert np.isclose(fitted_value, getattr(p_true, name).value, atol=0.05)


def test_fit_background_raises_clear_error_when_mask_bkg_too_small():
    """Regression test: a `mask_bkg` selecting too few pixels (e.g. because
    the background annulus is too narrow, or fully flagged out for this
    exposure) used to crash deep inside astropy's LinearLSQFitter with an
    unrelated-looking "could not broadcast (5,) into (6,)" shape-mismatch
    error instead of a clear, actionable message.
    """
    ny, nx = 60, 50
    xx, yy = np.meshgrid(np.arange(nx, dtype=np.float64), np.arange(ny, dtype=np.float64))
    data = np.ones((ny, nx), dtype=np.float64)

    mask_bkg_empty = np.zeros((ny, nx), dtype=bool)
    with pytest.raises(ValueError, match="0 valid pixel"):
        tu.fit_background(data, xx, yy, mask_bkg_empty)

    # Polynomial2D(degree=2) has 6 free parameters; 3 points is not enough.
    mask_bkg_too_few = np.zeros((ny, nx), dtype=bool)
    mask_bkg_too_few[:3, 0] = True
    with pytest.raises(ValueError, match="3 valid pixel"):
        tu.fit_background(data, xx, yy, mask_bkg_too_few)


def test_correlate_image_low_galactic_lat_returns_nan(monkeypatch):
    # Patch block_reduce used by correlate_image
    monkeypatch.setattr(tu, "block_reduce", _nanmedian_block_reduce)
    _patch_spximg_zodi_getter(monkeypatch)

    shape = (40, 40)
    data = np.ones(shape, dtype=np.float32)
    flags = np.zeros(shape, dtype=np.int32)
    template = SPHERExImage(data=np.ones(shape, dtype=np.float32),
                            flags=np.zeros(shape, dtype=np.int32), unit="MJy/sr")

    # Use Galactic Center (b ~ 0 deg) => should trigger b < 15 threshold
    ra_gc, dec_gc = 266.4051, -28.936175
    spximg = _make_spximg_with_crval(data.copy(), flags, ra_gc, dec_gc)

    out = correlate_image(spximg, template)
    assert np.isnan(out)


def test_correlate_image_high_galactic_lat_proportional_returns_near_one(monkeypatch):
    monkeypatch.setattr(tu, "block_reduce", _nanmedian_block_reduce)
    _patch_spximg_zodi_getter(monkeypatch)
    downsamp = 20
    shape = (40, 40)
    rng = np.random.default_rng(0)

    # Full-res data/template field
    t_full = (1.0 + 0.01 * rng.normal(size=shape)).astype(np.float32)
    t_full[t_full <= 0] = 1.0
    data_full = (3.0 * t_full + 0.001 * rng.normal(size=shape)).astype(np.float32)
    flags = np.zeros(shape, dtype=np.int32)

    # correlate_image downsamples DATA to (2,2); template must match that
    t_ds = _nanmedian_block_reduce(t_full, block_size=downsamp, func=np.nanmedian).astype(np.float32)
    template = SPHERExImage(data=t_ds, flags=np.zeros_like(t_ds, dtype=np.int32), unit="MJy/sr")

    ra_ngp, dec_ngp = 192.85948, 27.12825
    spximg = _make_spximg_with_crval(data_full.copy(), flags, ra_ngp, dec_ngp)

    out = correlate_image(spximg, template)
    assert np.isfinite(out)
    assert out > 0.99


def test_correlate_image_returns_zero_if_too_many_nans_after_masking(monkeypatch):
    monkeypatch.setattr(tu, "block_reduce", _nanmedian_block_reduce)
    _patch_spximg_zodi_getter(monkeypatch)
    downsamp = 20
    shape = (40, 40)
    t_full = np.ones(shape, dtype=np.float32)
    data_full = (2.0 * t_full).astype(np.float32)

    flags = np.zeros(shape, dtype=np.int32)
    flags[:, :] |= int(Flag.SOURCE.value)   # mask EVERYTHING => dat becomes all-NaN

    t_ds = _nanmedian_block_reduce(t_full, block_size=downsamp, func=np.nanmedian).astype(np.float32)
    template = SPHERExImage(data=t_ds, flags=np.zeros_like(t_ds, dtype=np.int32), unit="MJy/sr")

    ra_ngp, dec_ngp = 192.85948, 27.12825
    spximg = _make_spximg_with_crval(data_full.copy(), flags, ra_ngp, dec_ngp)

    out = correlate_image(spximg, template)
    assert out == 0.0


def test_correlate_image_masks_sur_source_nonfunc_pixels(monkeypatch):
    monkeypatch.setattr(tu, "block_reduce", _nanmedian_block_reduce)
    _patch_spximg_zodi_getter(monkeypatch)
    downsamp = 20
    shape = (40, 40)

    # Use a non-constant template so masking changes correlation in a measurable way
    yy, xx = np.indices(shape)
    t_full = (1.0 + 0.001 * (xx + 2 * yy)).astype(np.float32)
    data_full = (5.0 * t_full).astype(np.float32)

    flags = np.zeros(shape, dtype=np.int32)

    # Make three disjoint masked regions:
    flags[0:10, 0:10] |= 1  # SUR-like bit (flags & 7) > 0
    flags[10:20, 0:10] |= Flag.SOURCE.value
    flags[20:30, 0:10] |= Flag.NONFUNC.value

    # Downsampled template (2,2)
    t_ds = _nanmedian_block_reduce(t_full, block_size=downsamp, func=np.nanmedian).astype(np.float32)
    template = SPHERExImage(data=t_ds, flags=np.zeros_like(t_ds, dtype=np.int32), unit="MJy/sr")

    ra_ngp, dec_ngp = 192.85948, 27.12825
    spximg = _make_spximg_with_crval(data_full.copy(), flags, ra_ngp, dec_ngp)

    out = correlate_image(spximg, template)
    assert np.isfinite(out)
    assert 0.0 < out < 1.0
