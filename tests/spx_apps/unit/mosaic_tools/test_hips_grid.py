"""Tests for HiPS tile grid creation (hips_grid.py).

The ground truth throughout is healpy: a HiPS tile at order k, NESTED
pixel npix, imaged as tile_size x tile_size pixels, must have its pixel
(i, j) centered on the healpy center of the corresponding NESTED sub-cell
at order k + log2(tile_size).

Face 6 (the base facet centered on longitude 180) is exercised explicitly
because the HPX projection plane splits it across the +/-180 boundary:
tile headers there must still be valid (pix2world direction), and the
HEALPix-based sky-to-pixel mapping must be exact where wcslib's
world2pix is not.
"""

import numpy as np
import healpy as hp
import pytest

from spx_apps.mosaic_tool.hips_grid import (
    HIPS_TILE_SIZE,
    hips_tile_header,
    create_hips_tile_grid,
    compute_healpix_pixel_mapping,
    hips_tile_polygon,
    npix_for_coordinate,
    hips_pixel_scale_arcsec,
)


def subpixel_truth(order, npix, tile_size):
    """Healpy ground truth for every pixel of a tile.

    Returns
    -------
    ix, iy : ndarray of int
        0-based tile pixel indices (FITS axis 1, axis 2 order).
    ra, dec : ndarray of float
        healpy centers of the corresponding NESTED sub-cells, degrees.

    """
    nside = 1 << order
    nside_sub = nside * tile_size
    sub = np.arange(tile_size * tile_size) + npix * tile_size * tile_size
    xs, ys, _ = hp.pix2xyf(nside_sub, sub, nest=True)
    x_face, y_face, _ = (int(v) for v in hp.pix2xyf(nside, npix, nest=True))
    tx = xs - x_face * tile_size
    ty = ys - y_face * tile_size
    ix = ty
    iy = (tile_size - 1) - tx
    ra, dec = hp.pix2ang(nside_sub, sub, nest=True, lonlat=True)
    return ix, iy, ra, dec


def angular_separation_arcsec(ra1, dec1, ra2, dec2):
    """Great-circle separation in arcseconds."""
    ra1, dec1, ra2, dec2 = map(np.radians, (ra1, dec1, ra2, dec2))
    cos_sep = (np.sin(dec1) * np.sin(dec2)
               + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(cos_sep, -1.0, 1.0))) * 3600.0


ORDER = 3
NSIDE = 1 << ORDER

# Representative order-3 tiles: one per base facet, plus geometric
# corner cases — tiles containing the poles, tiles touching the RA=0
# seam, and face-6 tiles on and off the +/-180 plane-split diagonal.
SAMPLE_TILES = (
    [(f"face{f}", f * NSIDE * NSIDE + 27) for f in range(12)]
    + [
        ("north-pole", int(hp.xyf2pix(NSIDE, NSIDE - 1, NSIDE - 1, 0, nest=True))),
        ("south-pole", int(hp.xyf2pix(NSIDE, 0, 0, 8, nest=True))),
        ("ra0-seam", int(hp.ang2pix(NSIDE, 0.0, 0.0, nest=True, lonlat=True))),
        ("face6-split-diagonal", int(hp.xyf2pix(NSIDE, 3, 3, 6, nest=True))),
        ("face6-corner", int(hp.xyf2pix(NSIDE, 0, 0, 6, nest=True))),
        ("face6-east-of-split", int(hp.xyf2pix(NSIDE, 5, 2, 6, nest=True))),
        ("face6-west-of-split", int(hp.xyf2pix(NSIDE, 2, 5, 6, nest=True))),
    ]
)
SAMPLE_IDS = [t[0] for t in SAMPLE_TILES]
SAMPLE_NPIX = [t[1] for t in SAMPLE_TILES]

# Face-6 tiles on the split diagonal (within-face x == y) straddle the
# +/-180 plane boundary; wcslib returns NaN for the out-of-domain half of
# their pixels (see hips_grid module docstring).
STRADDLING_NPIX = {int(hp.xyf2pix(NSIDE, d, d, 6, nest=True)) for d in range(NSIDE)}


class TestHipsTileHeader:
    def test_header_cards(self):
        header = hips_tile_header(order=5, npix=1000)
        s = 45.0 / (32 * 512)
        assert header['CTYPE1'] == 'RA---HPX'
        assert header['CTYPE2'] == 'DEC--HPX'
        assert header['CRVAL1'] == 0.0
        assert header['CRVAL2'] == 0.0
        assert header['PV2_1'] == 4.0
        assert header['PV2_2'] == 3.0
        assert header['NAXIS1'] == HIPS_TILE_SIZE
        assert header['NAXIS2'] == HIPS_TILE_SIZE
        assert header['ORDER'] == 5
        assert header['NPIX'] == 1000
        # hipsgen CD sign pattern: 45-degree rotation with RA flip
        assert header['CD1_1'] == pytest.approx(-s, rel=1e-12)
        assert header['CD1_2'] == pytest.approx(-s, rel=1e-12)
        assert header['CD2_1'] == pytest.approx(s, rel=1e-12)
        assert header['CD2_2'] == pytest.approx(-s, rel=1e-12)

    def test_crpix_half_integer(self):
        # hipsgen CRPIX values are exact half-integers (e.g. -2047.5)
        for npix in (0, 1000, 12287):
            header = hips_tile_header(order=5, npix=npix)
            assert (2 * header['CRPIX1']) == int(2 * header['CRPIX1'])
            assert header['CRPIX1'] != int(header['CRPIX1'])
            assert (2 * header['CRPIX2']) == int(2 * header['CRPIX2'])

    def test_nominal_pixel_scale(self):
        # HiPS standard nominal pixel size at order 5: ~12.88 arcsec
        assert hips_pixel_scale_arcsec(5) == pytest.approx(12.88, abs=0.01)
        # each order halves the pixel size
        assert hips_pixel_scale_arcsec(6) == pytest.approx(
            hips_pixel_scale_arcsec(5) / 2, rel=1e-12
        )

    def test_validation_errors(self):
        with pytest.raises(ValueError, match="order"):
            hips_tile_header(order=-1, npix=0)
        with pytest.raises(ValueError, match="npix"):
            hips_tile_header(order=3, npix=12 * 64)  # one past the end
        with pytest.raises(ValueError, match="npix"):
            hips_tile_header(order=3, npix=-1)
        with pytest.raises(ValueError, match="tile_size"):
            hips_tile_header(order=3, npix=0, tile_size=500)
        with pytest.raises(ValueError, match="NSIDE"):
            hips_tile_header(order=25, npix=0)


class TestPixToWorld:
    """The header must reproduce healpy sub-cell centers via pix2world."""

    @pytest.mark.parametrize("npix", SAMPLE_NPIX, ids=SAMPLE_IDS)
    def test_matches_healpy(self, npix):
        tile_size = 16
        wcs, shape = create_hips_tile_grid(ORDER, npix, tile_size)
        assert shape == (tile_size, tile_size)
        ix, iy, ra, dec = subpixel_truth(ORDER, npix, tile_size)
        ra_w, dec_w = wcs.wcs_pix2world(ix, iy, 0)
        valid = np.isfinite(ra_w) & np.isfinite(dec_w)
        if npix in STRADDLING_NPIX:
            # tiles straddling the +/-180 plane split: wcslib cannot
            # evaluate the out-of-domain half, but must get the rest right
            assert 0.4 < valid.sum() / valid.size < 0.6
        else:
            assert valid.all()
        sep = angular_separation_arcsec(
            ra_w[valid] % 360, dec_w[valid], ra[valid] % 360, dec[valid]
        )
        # tolerance is ~4e-4 of a tile pixel at this order
        assert sep.max() < 0.05

    def test_matches_healpy_full_size(self):
        # One full-size (512x512) tile at the nominal order 5
        npix = npix_for_coordinate(5, ra=83.63, dec=22.01)  # Crab, face check
        wcs, shape = create_hips_tile_grid(5, npix)
        assert shape == (HIPS_TILE_SIZE, HIPS_TILE_SIZE)
        ix, iy, ra, dec = subpixel_truth(5, npix, HIPS_TILE_SIZE)
        ra_w, dec_w = wcs.wcs_pix2world(ix, iy, 0)
        sep = angular_separation_arcsec(ra_w % 360, dec_w, ra % 360, dec)
        assert sep.max() < 0.05


class TestHealpixPixelMapping:
    @pytest.mark.parametrize("npix", SAMPLE_NPIX, ids=SAMPLE_IDS)
    def test_recovers_all_subpixels(self, npix):
        tile_size = 16
        ix, iy, ra, dec = subpixel_truth(ORDER, npix, tile_size)
        ix_m, iy_m, bounds = compute_healpix_pixel_mapping(
            ra, dec, ORDER, npix, tile_size
        )
        assert bounds.all()
        np.testing.assert_array_equal(ix_m, ix)
        np.testing.assert_array_equal(iy_m, iy)
        assert ix_m.dtype == np.int32
        assert iy_m.dtype == np.int32

    def test_exact_where_wcslib_fails(self):
        """Regression guard for the facet-6 +/-180 plane split.

        Tiles straddling the split diagonal have no representation fully
        inside wcslib's projection domain: wcslib fails for part of their
        pixels in both directions. The HEALPix mapping must be unaffected.
        """
        tile_size = 16
        npix = int(hp.xyf2pix(NSIDE, 3, 3, 6, nest=True))
        ix, iy, ra, dec = subpixel_truth(ORDER, npix, tile_size)
        wcs, _ = create_hips_tile_grid(ORDER, npix, tile_size)
        with np.errstate(invalid='ignore'):
            x_w, y_w = wcs.wcs_world2pix(ra, dec, 0)
            wcslib_correct = (np.abs(x_w - ix) < 0.25) & (np.abs(y_w - iy) < 0.25)
        assert not wcslib_correct.all()  # documents the wcslib limitation
        ix_m, iy_m, bounds = compute_healpix_pixel_mapping(
            ra, dec, ORDER, npix, tile_size
        )
        assert bounds.all()
        np.testing.assert_array_equal(ix_m, ix)
        np.testing.assert_array_equal(iy_m, iy)

    def test_east_of_split_uses_in_domain_representation(self):
        """Face-6 tiles east of the split must be written on the -180
        side, where wcslib handles both transform directions.
        """
        tile_size = 16
        npix = int(hp.xyf2pix(NSIDE, 5, 2, 6, nest=True))
        ix, iy, ra, dec = subpixel_truth(ORDER, npix, tile_size)
        wcs, _ = create_hips_tile_grid(ORDER, npix, tile_size)
        ra_w, dec_w = wcs.wcs_pix2world(ix, iy, 0)
        assert np.isfinite(ra_w).all()
        x_w, y_w = wcs.wcs_world2pix(ra, dec, 0)
        assert (np.abs(x_w - ix) < 0.25).all()
        assert (np.abs(y_w - iy) < 0.25).all()

    def test_outside_tile_masked(self):
        tile_size = 16
        npix, other = SAMPLE_NPIX[0], SAMPLE_NPIX[1]
        _, _, ra, dec = subpixel_truth(ORDER, other, tile_size)
        _, _, bounds = compute_healpix_pixel_mapping(ra, dec, ORDER, npix, tile_size)
        assert not bounds.any()

    def test_nan_and_shape_handling(self):
        tile_size = 16
        _, _, ra, dec = subpixel_truth(ORDER, SAMPLE_NPIX[0], tile_size)
        ra = ra.reshape(tile_size, tile_size).copy()
        dec = dec.reshape(tile_size, tile_size).copy()
        ra[0, 0] = np.nan
        dec[3, 5] = np.nan
        ix, iy, bounds = compute_healpix_pixel_mapping(
            ra, dec, ORDER, SAMPLE_NPIX[0], tile_size
        )
        assert ix.shape == (tile_size, tile_size)
        assert bounds.shape == (tile_size, tile_size)
        assert not bounds[0, 0]
        assert not bounds[3, 5]
        assert bounds.sum() == tile_size * tile_size - 2

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_healpix_pixel_mapping(
                np.zeros(3), np.zeros(4), ORDER, 0, 16
            )

    @pytest.mark.parametrize(
        "npix",
        [SAMPLE_NPIX[4], SAMPLE_NPIX[0],
         int(hp.xyf2pix(NSIDE, 5, 2, 6, nest=True))],
        ids=["face4", "face0", "face6-east-of-split"],
    )
    def test_consistent_with_tile_wcs(self, npix):
        """Mapping and header must describe the same pixelization.

        For random points inside the tile, pix2world of the mapped pixel
        index must land in the same HEALPix sub-cell as the point itself.
        This ties compute_healpix_pixel_mapping to hips_tile_header
        independently of healpy internals.
        """
        tile_size = 16
        nside_sub = NSIDE * tile_size
        rng = np.random.default_rng(12345)
        # random points strictly inside random sub-cells of the tile
        _, _, ra_c, dec_c = subpixel_truth(ORDER, npix, tile_size)
        sel = rng.integers(0, ra_c.size, 200)
        # jitter well within a sub-cell (~0.34 deg at this resolution)
        ra = ra_c[sel] + rng.uniform(-0.05, 0.05, sel.size)
        dec = dec_c[sel] + rng.uniform(-0.05, 0.05, sel.size)

        ix, iy, bounds = compute_healpix_pixel_mapping(ra, dec, ORDER, npix, tile_size)
        assert bounds.all()

        wcs, _ = create_hips_tile_grid(ORDER, npix, tile_size)
        ra_pix, dec_pix = wcs.wcs_pix2world(ix, iy, 0)
        cell_of_point = hp.ang2pix(nside_sub, ra, dec, nest=True, lonlat=True)
        cell_of_pixel = hp.ang2pix(nside_sub, ra_pix, dec_pix, nest=True, lonlat=True)
        np.testing.assert_array_equal(cell_of_point, cell_of_pixel)


class TestNpixForCoordinate:
    def test_tile_center_roundtrip(self):
        for order in (1, 3, 5):
            nside = 1 << order
            for npix in (0, 5, 12 * nside * nside - 1):
                ra, dec = hp.pix2ang(nside, npix, nest=True, lonlat=True)
                assert npix_for_coordinate(order, ra, dec) == npix

    def test_validation(self):
        with pytest.raises(ValueError, match="order"):
            npix_for_coordinate(-2, 0.0, 0.0)


class TestHipsTilePolygon:
    def test_format(self):
        poly = hips_tile_polygon(ORDER, SAMPLE_NPIX[0], points_per_edge=4)
        parts = poly.split()
        assert parts[0] == "POLYGON"
        values = [float(x) for x in parts[1:]]
        assert len(values) == 2 * 4 * 4  # 4 edges x points_per_edge vertices
        ra = np.array(values[0::2])
        dec = np.array(values[1::2])
        assert ((ra >= 0) & (ra < 360)).all()
        assert ((dec >= -90) & (dec <= 90)).all()

    def test_unbuffered_vertices_on_cell_boundary(self):
        npix = SAMPLE_NPIX[0]
        poly = hips_tile_polygon(ORDER, npix, buffer_deg=0.0)
        values = [float(x) for x in poly.split()[1:]]
        vec = np.array(hp.boundaries(NSIDE, npix, step=4, nest=True)).T
        ra_b, dec_b = hp.vec2ang(vec, lonlat=True)
        np.testing.assert_allclose(values[0::2], ra_b, atol=1e-6)
        np.testing.assert_allclose(values[1::2], dec_b, atol=1e-6)

    @pytest.mark.parametrize("npix", SAMPLE_NPIX, ids=SAMPLE_IDS)
    def test_buffer_expands_from_center(self, npix):
        buffer_deg = 0.5
        ra_c, dec_c = hp.pix2ang(NSIDE, npix, nest=True, lonlat=True)

        def center_dist(poly):
            values = np.array([float(x) for x in poly.split()[1:]])
            return angular_separation_arcsec(
                values[0::2], values[1::2], ra_c, dec_c
            ) / 3600.0

        dist0 = center_dist(hips_tile_polygon(ORDER, npix, buffer_deg=0.0))
        dist1 = center_dist(hips_tile_polygon(ORDER, npix, buffer_deg=buffer_deg))
        # atol reflects arccos round-off in the separation computation
        np.testing.assert_allclose(dist1 - dist0, buffer_deg, atol=1e-5)

    def test_validation(self):
        with pytest.raises(ValueError, match="buffer_deg"):
            hips_tile_polygon(ORDER, 0, buffer_deg=-1.0)
        with pytest.raises(ValueError, match="points_per_edge"):
            hips_tile_polygon(ORDER, 0, points_per_edge=0)
