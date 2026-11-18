__all__ = ["getSkyPixIds", "getDatasetRefsBySkypixIds", "REFCAT_SKYPIX_NAME", "ra_dec_to_unit_vector"]

import numpy as np
from lsst.daf.butler import Butler
from lsst.sphgeom import ConvexPolygon, UnitVector3d, HealpixPixelization
from spx_data import SPHERExImage

# Settings for Gaia/DR2 - CatWISE2020 match catalogs
# HEALPix ring scheme
# from .HEALPixelization import HEALPixelization
# REFCAT_HIPS_LEVEL = 4
# REFCAT_PIXELIZATION = HEALPixelization(REFCAT_HIPS_LEVEL)

# Settings for Example Reference Catalog v0.4:
#    Gaia_EDR3.PS1DR1.CatWISE.AllWISE.2MASS_NSIDE32_xxxxxx.fits
# HEALPix nested scheme
REFCAT_HIPS_LEVEL = 5
REFCAT_PIXELIZATION = HealpixPixelization(REFCAT_HIPS_LEVEL)

REFCAT_SKYPIX_NAME = f"hips{REFCAT_HIPS_LEVEL}"


def getSPHERExImageRegion(image: SPHERExImage, npixMargin: int = 0):
    """Get region that represents image with a margin.

    Parameters
    ----------
    image : `SPHERExImage`
        The SPHEREx image to get the region for.
    npixMargin : `int`, optional
        Margin in pixels, can be negative. Default is 0.

    Returns
    -------
    `lsst.sphgeom.ConvexPolygon`
        The convex polygon region representing the image footprint.

    """
    npixX, npixY = image.data.shape
    npixMarginX, npixMarginY = (npixMargin, npixMargin)
    if abs(npixMarginX) >= npixX / 2 or abs(npixMarginY) >= npixY / 2:
        raise ValueError(f"Pixel margin {npixMargin} exceeds half-image pixel size")
    x1, x2 = (0 - npixMarginX, (npixX - 1) + npixMarginX)
    y1, y2 = (0 - npixMarginY, (npixY - 1) + npixMarginY)
    cornersX = [x1, x2, x2, x1]
    cornersY = [y1, y1, y2, y2]
    cornersCoords = image.wcs.pixel_to_world(cornersX, cornersY)
    vertices = cornersCoords.cartesian.xyz.value.T
    # vertices as a list of unit vectors
    vlist = [UnitVector3d(c[0], c[1], c[2]) for c in vertices]
    return ConvexPolygon(vlist)


def getSkyPixIds(image: SPHERExImage, npixMargin: int = 0) -> list[int]:
    """Get sky pixel indexes that overlap with a given image.

    Parameters
    ----------
    image : `SPHERExImage`
        The SPHEREx image to get sky pixel IDs for.
    npixMargin : `int`, optional
        Margin in pixels, can be negative. Default is 0.

    Returns
    -------
    skyPixIds : list
        Sky pixel indexes that overlap with the image

    """
    region = getSPHERExImageRegion(image, npixMargin)
    # commonSkyPix = butler.registry.dimensions.commonSkyPix
    # pixelization = commonSkyPix.pixelization
    pixelization = REFCAT_PIXELIZATION
    rs = pixelization.envelope(region)
    skyPixIds: list[int] = []
    for begin, end in rs:
        skyPixIds.extend(range(begin, end))
    return skyPixIds


def getDatasetRefsBySkypixIds(butler: Butler,
                              datasetType: str,
                              collections: str,
                              skyPixIds: list[int]):
    """Query dataset references by sky pixel IDs from the butler."""
    # commonSkyPix = butler.registry.dimensions.commonSkyPix
    # where = f"{commonSkyPix.name} in ({str(skyPixIds)[1:-1]})"
    where = f"{REFCAT_SKYPIX_NAME} in ({str(skyPixIds)[1:-1]})"
    # Example:
    # butler.query_datasets(
    #     "catwise2020_gaiadr2_05202020",
    #     "refcats",
    #     where="hips4 in (10,11)"
    # )
    dsrefs = butler.query_datasets(
        datasetType,
        collections,
        where=where
    )
    return list(dsrefs)


def ra_dec_to_unit_vector(ra_deg: float, dec_deg: float) -> UnitVector3d:
    """Convert RA/DEc in degrees to a UnitVector3d for HEALPix indexing.

    Parameters
    ----------
    ra_deg : `float`
        Right Ascension in degrees.
    dec_deg : `float`
        Declination in degrees.

    Returns
    -------
    `UnitVector3d`
        A 3D unit vector representing the sky position.

    """
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)

    x = np.cos(ra_rad) * np.cos(dec_rad)
    y = np.sin(ra_rad) * np.cos(dec_rad)
    z = np.sin(dec_rad)

    return UnitVector3d(x, y, z)
