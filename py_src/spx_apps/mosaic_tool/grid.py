"""Output mosaic grid (WCS) creation."""

from astropy.wcs import WCS
from typing import Optional


def create_mosaic_grid(
    ra_center: float,
    dec_center: float,
    width_deg: float,
    height_deg: float,
    pixel_size_arcsec: float = 9.0,
    rotation_deg: Optional[float] = None,
    projection: str = 'TAN'
) -> tuple[WCS, tuple[int, int]]:
    """Create WCS and shape for output mosaic grid.

    This function replicates the behavior of map_tools.get_mosaic_wcs()
    from SPHEREx_L4GF_map_maker for compatibility.

    Parameters
    ----------
    ra_center : float
        Right ascension of mosaic center in degrees.
    dec_center : float
        Declination of mosaic center in degrees.
    width_deg : float
        Mosaic width in degrees.
    height_deg : float
        Mosaic height in degrees.
    pixel_size_arcsec : float, optional
        Pixel size in arcseconds. Default is 9.0.
    rotation_deg : float or None, optional
        Rotation angle (roll) in degrees (counterclockwise from North).
        None means no rotation.
    projection : str, optional
        WCS projection type (e.g., 'SIN', 'TAN'). Default is 'TAN'.

    Returns
    -------
    wcs : astropy.wcs.WCS
        WCS object for the mosaic grid.
    shape :
        Shape of the mosaic array as (ny, nx).

    Notes
    -----
    The pixel count is forced to be odd in each dimension so that
    CRPIX falls exactly on the center pixel.

    Examples
    --------
    >>> wcs, shape = create_mosaic_grid(
    ...     ra_center=322.0, dec_center=-6.0,
    ...     width_deg=1.0, height_deg=1.0,
    ...     pixel_size_arcsec=9.0
    ... )
    >>> print(shape)
    (401, 401)

    """
    # Convert pixel size to arcseconds for calculation
    width_arcsec = width_deg * 3600.0
    height_arcsec = height_deg * 3600.0

    # Calculate number of pixels - force odd number
    nx = int(width_arcsec / pixel_size_arcsec) | 1  # Force odd
    ny = int(height_arcsec / pixel_size_arcsec) | 1  # Force odd

    # Convert pixel size to degrees
    pixel_size_deg = pixel_size_arcsec / 3600.0

    # Create WCS
    w = WCS(naxis=2)

    # Reference pixel at center (FITS 1-based indexing)
    # This matches map_tools: crpix = [n/2 + 0.5]
    w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]

    # Reference sky coordinates
    w.wcs.crval = [ra_center, dec_center]

    # Coordinate type
    w.wcs.ctype = [f'RA---{projection}', f'DEC--{projection}']

    # Pixel scale (RA increases to the left = negative)
    w.wcs.cdelt = [-pixel_size_deg, pixel_size_deg]

    # Rotation using CROTA (matching map_tools convention)
    if rotation_deg is not None:
        w.wcs.crota = [0.0, rotation_deg]
    else:
        w.wcs.crota = [0.0, 0.0]

    # Shape in (ny, nx) convention for numpy arrays
    shape = (ny, nx)

    return w, shape
