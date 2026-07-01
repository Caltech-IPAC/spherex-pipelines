__all__ = ["ExposureBoundary", "ra_dec_to_unit_vector",
           "get_exposure_boundary_info", "add_exposure_boundary_info"]

import math
import numpy as np

from pydantic import BaseModel
from astropy.wcs import WCS
from lsst.pipe.base import TaskMetadata
from lsst.sphgeom import ConvexPolygon, Region, UnitVector3d


class ExposureBoundary(BaseModel):
    """Information needed to calculate polygon region covered by SPHEREx exposure.

    Attributes
    ----------
    expid : `int`
        Exposure ID
    center : `tuple[float`, float]`
        RA and Dec in degrees of the image center.
    bounds : `list[tuple[float, float]]`
        RA and Dec in degrees of points defining exposure boundaries counter-clockwise:
        bottom-left, bottom-right, top-right, top-left.
    """
    expid: int
    center: tuple[float, float]
    bounds: list[tuple[float, float]]

    def to_sphgeom_region(self) -> Region:
        """Convert to sphgeom region."""
        unit_vectors: list[UnitVector3d] = [ra_dec_to_unit_vector(ra, dec) for ra, dec in self.bounds]
        return ConvexPolygon.convexHull(unit_vectors)


def ra_dec_to_unit_vector(ra_deg, dec_deg) -> UnitVector3d:
    """Convert RA/Dec in degrees to a UnitVector3d.

    Parameters
    ----------
    ra_deg :
        RA in degrees.
    dec_deg :
        Dec in degrees.

    Returns
    -------
        Unit vector representing RA and Dec.
    """
    # Convert degrees to radians
    ra_rad = math.radians(ra_deg)
    dec_rad = math.radians(dec_deg)

    # Convert RA/Dec to Cartesian coordinates
    x = math.cos(dec_rad) * math.cos(ra_rad)
    y = math.cos(dec_rad) * math.sin(ra_rad)
    z = math.sin(dec_rad)

    return UnitVector3d(x, y, z)


def get_exposure_boundary_info(wcs: WCS, corners_only: bool = False) -> (
        tuple)[float, float, list[float], list[float]]:
    """Get center and boundary points of an image with the given WCS.

    Parameters
    ----------
    wcs :
        WCS object with `array_shape` attribute set to `(n_pixels_y, n_pixels_x)`
    corners_only :
        If True, boundary points will only include four corners, otherwise they will
        include also the middle point of each edge.

    Returns
    -------
    """
    if wcs.array_shape:
        npix_y, npix_x = wcs.array_shape
    else:
        raise ValueError("WCS passed to add_exposure_boundary_info must have an array_shape.")

    # Calculate coordinates along image boundary and in the middle of the array
    if corners_only:
        # corners counterclockwise, the last point is mid-array
        img_xx = np.array([0, npix_x, npix_x, 0, npix_x // 2]) - 0.5
        img_yy = np.array([0, 0, npix_y, npix_y, npix_y // 2]) - 0.5
    else:
        # 8 points counterclockwise, the last point is mid-array
        img_xx = np.array([0, npix_x//2, npix_x, npix_x, npix_x, npix_x//2, 0, 0, npix_x//2]) - 0.5
        img_yy = np.array([0, 0, 0, npix_y//2, npix_y, npix_y, npix_y, npix_y//2, npix_y//2]) - 0.5
    ra_arr, dec_arr = wcs.all_pix2world(img_xx, img_yy, 0)
    return ra_arr[-1], dec_arr[-1], ra_arr[:-1].tolist(), dec_arr[:-1].tolist()


def add_exposure_boundary_info(metadata: TaskMetadata, wcs: WCS) -> None:
    """Add exposure boundary information to the task metadata.

    Parameters
    ----------
    metadata :
        TaskMetadata instance.
    wcs :
        WCS object with `array_shape` attribute set to `(n_pixels_y, n_pixels_x)`
    """
    center_ra, center_dec, ra_lst, dec_lst = get_exposure_boundary_info(wcs)
    metadata.add("center_ra", center_ra)
    metadata.add("center_dec", center_dec)
    metadata.add("boundaries_ra", ra_lst)
    metadata.add("boundaries_dec", dec_lst)
