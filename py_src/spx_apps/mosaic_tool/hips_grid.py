"""HiPS tile grid (HPX WCS) creation and HEALPix pixel mapping.

A HiPS tile at order ``k`` is one HEALPix NESTED cell at NSIDE = 2^k,
imaged as a ``tile_size`` x ``tile_size`` grid whose pixels are the NESTED
sub-cells at order ``k + log2(tile_size)`` (HiPS 1.0 standard, IVOA REC).
The tile WCS uses the HPX projection of Calabretta & Roukema (2007),
"Mapping on the HEALPix grid", Section 3 (H=4, K=3), with the same header
conventions as the CDS ``hipsgen`` tool: CRVAL = (0, 0), PV2_1 = 4,
PV2_2 = 3, a CD matrix encoding a 45-degree rotation, and ``ORDER`` /
``NPIX`` keywords identifying the tile.

The HPX projection is exactly linear in the HEALPix grid within a base
facet, so the tile WCS is an exact description of the HEALPix cell
geometry: ``wcs.pix2world`` of a tile pixel center reproduces the healpy
sub-cell center to numerical precision.

The +/-180 plane split (base facet 6)
-------------------------------------
The HPX projection plane splits the base facet centered on longitude 180
(face 6) across the +/-180 boundary, and wcslib enforces that domain: it
returns NaN (pix2world) or wrong-side coordinates (world2pix) for plane
positions beyond it. The projection is periodic in the equatorial zone,
so tiles lying entirely east of the split are written using the
equivalent in-domain representation on the -180 side. Only the 2^order
face-6 tiles straddling the split diagonal have no single in-domain
representation; for those, astropy/wcslib transforms return NaN for the
half of the tile beyond the boundary (a wcslib domain limitation — the
header still describes the correct geometry and HiPS clients with their
own HPX implementations read it fine).

Because of this, sky-to-pixel mapping must not use ``wcs.world2pix``;
use :func:`compute_healpix_pixel_mapping`, which bins sky coordinates by
HEALPix cell containment. This is exact on the whole sphere, including
the straddling tiles, and is equivalent to nearest-pixel-center binning
in the tile frame (HEALPix cells are exact squares in the HPX plane),
i.e. it matches the histogram reprojection semantics and hipsgen's own
binning.
"""

__all__ = [
    'HIPS_TILE_SIZE',
    'hips_tile_header',
    'create_hips_tile_grid',
    'compute_healpix_pixel_mapping',
    'hips_tile_polygon',
    'hips_tile_relpath',
    'npix_for_coordinate',
    'hips_pixel_scale_arcsec',
]

import os

import numpy as np
import healpy as hp
from astropy.io import fits
from astropy.wcs import WCS

# Standard HiPS tile width in pixels (HiPS 1.0 standard).
HIPS_TILE_SIZE = 512

# healpy supports NSIDE up to 2^29; the sub-pixel grid NSIDE is
# 2^order * tile_size and must not exceed this.
_MAX_NSIDE = 1 << 29

# Position of each base facet in the HPX projection plane, in units of
# 45 degrees: plane coordinates of the facet corner at within-face
# healpy coords (x, y) = (0, 0). For the north polar facets (0-3) this
# corner is the facet's equatorial vertex; for the equatorial facets
# (4-7) the southernmost vertex; for the south polar facets (8-11) the
# south pole vertex. Facet 6 is centered on longitude 180 and is the one
# split across the +/-180 plane boundary.
_FACET_X = (1, 3, -3, -1, 0, 2, 4, -2, 1, 3, -3, -1)
_FACET_Y = (0, 0, 0, 0, -1, -1, -1, -1, -2, -2, -2, -2)


def _validate_tile(order: int, npix: int, tile_size: int) -> None:
    """Validate HiPS tile parameters, raising ValueError on failure."""
    if order < 0:
        raise ValueError(f"order must be non-negative, got {order}")
    if tile_size < 1 or (tile_size & (tile_size - 1)) != 0:
        raise ValueError(
            f"tile_size must be a positive power of 2, got {tile_size}"
        )
    if (1 << order) * tile_size > _MAX_NSIDE:
        raise ValueError(
            f"order {order} with tile_size {tile_size} exceeds the maximum "
            f"HEALPix NSIDE of 2^29"
        )
    npix_max = 12 * (1 << (2 * order))
    if not 0 <= npix < npix_max:
        raise ValueError(
            f"npix must be in [0, {npix_max}) for order {order}, got {npix}"
        )


def hips_pixel_scale_arcsec(order: int, tile_size: int = HIPS_TILE_SIZE) -> float:
    """Tile pixel size in arcseconds at a given HiPS order.

    This is the side length of the (square) HEALPix sub-cell in the HPX
    projection plane, e.g. ~12.88 arcsec at order 5 with 512-pixel tiles.

    Parameters
    ----------
    order : int
        HiPS tile order.
    tile_size : int, optional
        Tile width in pixels. Default is 512.

    Returns
    -------
    float
        Pixel size in arcseconds.

    """
    nside_pix = (1 << order) * tile_size
    # mean pixel size = sqrt(cell area), the conventional HiPS figure
    # (e.g. 12.88 arcsec at order 5); the cell side in the HPX projection
    # plane is the slightly larger 45*sqrt(2)/nside_pix.
    return np.degrees(np.sqrt(np.pi / 3.0) / nside_pix) * 3600.0


def hips_tile_header(
    order: int,
    npix: int,
    tile_size: int = HIPS_TILE_SIZE
) -> fits.Header:
    """Create a hipsgen-style FITS header for a HiPS tile.

    The header uses the HPX projection (CTYPE ``RA---HPX`` / ``DEC--HPX``,
    PV2_1 = 4, PV2_2 = 3, CRVAL = 0,0) with a CD matrix encoding the
    45-degree rotation between the HEALPix cell diagonal and the HPX plane
    axes, CRPIX locating the tile within the global HPX plane, and the
    ``ORDER`` and ``NPIX`` keywords identifying the tile as written by the
    CDS hipsgen tool.

    Axis convention (reproduces the hipsgen CD sign pattern
    ``CD1_1 = CD1_2 = -s``, ``CD2_1 = s``, ``CD2_2 = -s``): FITS axis 1
    increases with the healpy within-tile y coordinate and FITS axis 2
    decreases with the healpy within-tile x coordinate.

    Parameters
    ----------
    order : int
        HiPS tile order. HEALPix NSIDE of the tile grid is 2^order.
    npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^order.
    tile_size : int, optional
        Tile width in pixels (power of 2). Default is 512.

    Returns
    -------
    astropy.io.fits.Header
        Header with complete spatial WCS plus ORDER and NPIX cards.

    """
    _validate_tile(order, npix, tile_size)

    nside = 1 << order
    nside_pix = nside * tile_size
    s = 45.0 / nside_pix

    x_face, y_face, face = (
        int(v) for v in hp.pix2xyf(nside, npix, nest=True)
    )

    facet_x = _FACET_X[face]
    if face == 6 and x_face > y_face:
        # The tile lies entirely east of the +/-180 plane split; use the
        # periodic-equivalent representation on the -180 side so all its
        # pixels are within wcslib's formal projection domain (see module
        # docstring).
        facet_x = -4

    # Plane coordinates of the tile origin sub-pixel center (within-tile
    # healpy coords tx = ty = 0), in units of s (exact integers):
    # plane = s * [[1, -1], [1, 1]] @ (xs, ys) + facet corner + (0, s)
    u = (x_face - y_face) * tile_size + facet_x * nside_pix
    v = (x_face + y_face) * tile_size + _FACET_Y[face] * nside_pix + 1

    # CRPIX from matching CD @ (P - CRPIX) to the plane coordinates with
    # 1-based FITS coords P1 = ty + 1, P2 = tile_size - tx. Exact
    # half-integer arithmetic (u - v is odd, u + v is even).
    crpix1 = 1 + (u - v) / 2
    crpix2 = tile_size + (u + v) / 2

    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = tile_size
    header['NAXIS2'] = tile_size
    header['CRPIX1'] = (crpix1, 'Reference pixel in global HPX plane')
    header['CRPIX2'] = (crpix2, 'Reference pixel in global HPX plane')
    header['CD1_1'] = (-s, 'Transformation matrix (45 deg rotation)')
    header['CD1_2'] = (-s, '')
    header['CD2_1'] = (s, '')
    header['CD2_2'] = (-s, '')
    header['CTYPE1'] = ('RA---HPX', 'HPX projection (Calabretta&Roukema 2007)')
    header['CTYPE2'] = ('DEC--HPX', 'HPX projection (Calabretta&Roukema 2007)')
    header['CRVAL1'] = (0.0, '[deg] Reference RA')
    header['CRVAL2'] = (0.0, '[deg] Reference Dec')
    header['PV2_1'] = (4.0, 'HPX H parameter (longitude)')
    header['PV2_2'] = (3.0, 'HPX K parameter (latitude)')
    header['ORDER'] = (order, 'HiPS tile order (HEALPix NSIDE = 2^ORDER)')
    header['NPIX'] = (npix, 'HEALPix NESTED index of this tile')
    return header


def create_hips_tile_grid(
    order: int,
    npix: int,
    tile_size: int = HIPS_TILE_SIZE
) -> tuple[WCS, tuple[int, int]]:
    """Create WCS and shape for a HiPS tile output grid.

    HiPS counterpart of :func:`~spx_apps.mosaic_tool.grid.create_mosaic_grid`.
    Note that ``WCS.to_header()`` drops the non-WCS ``ORDER`` and ``NPIX``
    cards; use :func:`hips_tile_header` to recover them when writing output
    files.

    Parameters
    ----------
    order : int
        HiPS tile order.
    npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^order.
    tile_size : int, optional
        Tile width in pixels (power of 2). Default is 512.

    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS object for the tile grid.
    shape : tuple of (int, int)
        Shape of the tile array as (ny, nx) = (tile_size, tile_size).

    """
    header = hips_tile_header(order, npix, tile_size)
    return WCS(header), (tile_size, tile_size)


def compute_healpix_pixel_mapping(
    ra,
    dec,
    order: int,
    npix: int,
    tile_size: int = HIPS_TILE_SIZE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map sky coordinates to HiPS tile pixel indices via HEALPix binning.

    Bins each coordinate into the HEALPix NESTED sub-cell at order
    ``order + log2(tile_size)`` and converts the cell index to tile pixel
    indices. Cell containment equals nearest-pixel-center binning in the
    tile frame, so this is a drop-in replacement for the
    ``wcs.all_world2pix`` + rounding step of
    ``MosaicBuilder._compute_pixel_mapping`` — and unlike wcslib's inverse
    HPX transform it is exact for tiles in the base facet split across the
    +/-180 plane boundary (see module docstring).

    Parameters
    ----------
    ra, dec : array_like
        Sky coordinates in ICRS degrees. Any shape; may contain NaN
        (mapped to ``bounds_mask = False``).
    order : int
        HiPS tile order.
    npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^order.
    tile_size : int, optional
        Tile width in pixels (power of 2). Default is 512.

    Returns
    -------
    ix_idx : numpy.ndarray of int32
        Tile x (FITS axis 1) pixel indices, 0-based, same shape as `ra`.
        Only valid where ``bounds_mask`` is True (0 elsewhere).
    iy_idx : numpy.ndarray of int32
        Tile y (FITS axis 2) pixel indices, 0-based, same shape as `ra`.
        Only valid where ``bounds_mask`` is True (0 elsewhere).
    bounds_mask : numpy.ndarray of bool
        True where the coordinate falls inside this tile.

    """
    _validate_tile(order, npix, tile_size)

    ra = np.asarray(ra, dtype=np.float64)
    dec = np.asarray(dec, dtype=np.float64)
    if ra.shape != dec.shape:
        raise ValueError(
            f"ra and dec must have the same shape, got {ra.shape} and {dec.shape}"
        )

    nside = 1 << order
    nside_sub = nside * tile_size
    # Bits occupied by the within-tile part of the NESTED sub-cell index
    shift = 2 * (tile_size.bit_length() - 1)

    finite = np.isfinite(ra) & np.isfinite(dec)

    ipix = np.zeros(ra.shape, dtype=np.int64)
    if finite.any():
        ipix[finite] = hp.ang2pix(
            nside_sub,
            ra[finite],
            np.clip(dec[finite], -90.0, 90.0),
            nest=True,
            lonlat=True
        )

    bounds_mask = finite & ((ipix >> shift) == npix)

    ix_idx = np.zeros(ra.shape, dtype=np.int32)
    iy_idx = np.zeros(ra.shape, dtype=np.int32)
    if bounds_mask.any():
        xs, ys, _ = hp.pix2xyf(nside_sub, ipix[bounds_mask], nest=True)
        x_face, y_face, _ = (
            int(v) for v in hp.pix2xyf(nside, npix, nest=True)
        )
        tx = xs - x_face * tile_size
        ty = ys - y_face * tile_size
        # FITS axis 1 follows healpy y; axis 2 runs against healpy x
        # (see hips_tile_header)
        ix_idx[bounds_mask] = ty.astype(np.int32)
        iy_idx[bounds_mask] = ((tile_size - 1) - tx).astype(np.int32)

    return ix_idx, iy_idx, bounds_mask


def hips_tile_polygon(
    order: int,
    npix: int,
    buffer_deg: float = 0.0,
    points_per_edge: int = 4
) -> str:
    """Compute a sky polygon covering a HiPS tile, for Butler region queries.

    Samples the HEALPix cell boundary and optionally pushes each vertex
    radially away from the tile center by `buffer_deg` along the great
    circle. The boundary is sampled with several points per edge because
    HEALPix cell edges are not great-circle arcs; the buffer must exceed
    the small sag between samples (well under an arcminute per edge for
    order >= 2 with the default sampling).

    Parameters
    ----------
    order : int
        HiPS tile order.
    npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^order.
    buffer_deg : float, optional
        Angular margin added around the tile in degrees. Default is 0.
    points_per_edge : int, optional
        Boundary samples per cell edge. Default is 4.

    Returns
    -------
    str
        Polygon in the ``POLYGON ra1 dec1 ra2 dec2 ...`` format accepted
        by Butler ``REGION()`` queries (vertices in cell-boundary order).

    """
    _validate_tile(order, npix, tile_size=1)
    if buffer_deg < 0:
        raise ValueError(f"buffer_deg must be non-negative, got {buffer_deg}")
    if points_per_edge < 1:
        raise ValueError(
            f"points_per_edge must be positive, got {points_per_edge}"
        )

    nside = 1 << order
    # Unit vectors of boundary samples, shape (n_points, 3)
    vec = hp.boundaries(nside, npix, step=points_per_edge, nest=True).T

    if buffer_deg > 0:
        ra_c, dec_c = hp.pix2ang(nside, npix, nest=True, lonlat=True)
        center = np.asarray(hp.ang2vec(ra_c, dec_c, lonlat=True), dtype=float)
        cos_theta = vec @ center
        theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
        # Unit tangent at the center, pointing toward each vertex
        tangent = vec - np.outer(cos_theta, center)
        tangent /= np.linalg.norm(tangent, axis=1)[:, np.newaxis]
        theta_new = theta + np.radians(buffer_deg)
        vec = (np.outer(np.cos(theta_new), center)
               + tangent * np.sin(theta_new)[:, np.newaxis])

    ra, dec = hp.vec2ang(vec, lonlat=True)
    coords = " ".join(f"{r:.7f} {d:.7f}" for r, d in zip(ra, dec))
    return f"POLYGON {coords}"


def hips_tile_relpath(order: int, npix: int, suffix: str = '.fits') -> str:
    """Relative path of a tile file following the HiPS directory convention.

    HiPS 1.0 groups tiles as ``Norder{order}/Dir{D}/Npix{npix}``, where
    ``D = (npix // 10000) * 10000``.

    Parameters
    ----------
    order : int
        HiPS tile order.
    npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^order.
    suffix : str, optional
        Filename suffix appended after ``Npix{npix}``. Default is '.fits'.

    Returns
    -------
    str
        Relative path, e.g. ``Norder5/Dir0/Npix1000_cube.fits`` for
        ``suffix='_cube.fits'``.

    """
    _validate_tile(order, npix, tile_size=1)
    dir_index = (npix // 10000) * 10000
    return os.path.join(f"Norder{order}", f"Dir{dir_index}", f"Npix{npix}{suffix}")


def npix_for_coordinate(order: int, ra: float, dec: float) -> int:
    """Find the HiPS tile (NESTED pixel index) containing a sky coordinate.

    Parameters
    ----------
    order : int
        HiPS tile order.
    ra, dec : float
        Sky coordinate in ICRS degrees.

    Returns
    -------
    int
        HEALPix NESTED pixel index at NSIDE = 2^order.

    """
    _validate_tile(order, npix=0, tile_size=1)
    return int(hp.ang2pix(1 << order, ra, dec, nest=True, lonlat=True))
