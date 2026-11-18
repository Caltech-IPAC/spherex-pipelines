__all__ = ['MosaicConfig']

from dataclasses import dataclass
from typing import Optional


@dataclass
class MosaicConfig:
    """Configuration for mosaic creation.

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
        Output pixel size in arcseconds. Default is 6.2.
    rotation_deg : float or None, optional
        Rotation angle (roll) in degrees. None means no rotation.
    reproject_method : str, optional
        Reprojection method name. One of 'reproject_exact' or 'reproject_hist'.
        Default is 'reproject_exact'.
    projection : str, optional
        WCS projection type. Default is 'SIN'.
    bad_pixel_flags : int, optional
        Bitwise OR of bad pixel flags. Default is None.
    propagated_flags : int, optional
        Bitwise OR of pixel flags to be propagated. Default is None.

    Examples
    --------
    >>> config = MosaicConfig(
    ...     ra_center=322.0,
    ...     dec_center=-6.0,
    ...     width_deg=1.0,
    ...     height_deg=1.0,
    ...     pixel_size_arcsec=6.15,
    ...     reproject_method='reproject_exact'
    ... )

    """

    ra_center: float
    dec_center: float
    width_deg: float
    height_deg: float
    pixel_size_arcsec: float = 6.2
    rotation_deg: Optional[float] = None
    reproject_method: str = 'reproject_exact'
    projection: str = 'SIN'
    bad_pixel_flags: Optional[int] = None
    propagated_flags: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if not 0 <= self.ra_center < 360:
            raise ValueError(f"ra_center must be in [0, 360), got {self.ra_center}")
        if not -90 <= self.dec_center <= 90:
            raise ValueError(f"dec_center must be in [-90, 90], got {self.dec_center}")
        if self.width_deg <= 0:
            raise ValueError(f"width_deg must be positive, got {self.width_deg}")
        if self.height_deg <= 0:
            raise ValueError(f"height_deg must be positive, got {self.height_deg}")
        if self.pixel_size_arcsec <= 0:
            raise ValueError(f"pixel_size_arcsec must be positive, got {self.pixel_size_arcsec}")
