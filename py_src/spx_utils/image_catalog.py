__all__ = ["SourcePipelineClass", "ImageCatalogIterable", "preprocess_catalog"]

import logging
from enum import IntFlag
from typing import Optional
from collections.abc import Iterator, Iterable

from astropy.table import Table
from lsst.daf.butler import DeferredDatasetHandle

from spx_data import SPHERExImage


class SourcePipelineClass(IntFlag):
    """Per-source bitwise flag for SPHEREx Pipeline."""

    UNDEFINED = 0
    ASTROMETRIC = 1 << 0  # astrometric calibrators
    PSF = 1 << 1  # PSF calibrators
    PHOTOMETRIC = 1 << 2  # primary spectro-photometric calibrators (N = 17)
    CALSPEC = 1 << 3  # CALSPEC stars (N = 99)
    BRIGHT = 1 << 4  # Very bright stars
    COSMOLOGY = 1 << 5  # L4 Cosmology targets
    ICES = 1 << 6  # L4 Ices targets
    NO_ICES = 1 << 7  # Sources within 7.6 srcsec from SPLICES targets
    SECONDARY = 1 << 8  # reserved for secondary photometric standard stars
    STELLAR = 1 << 9  # LC Stellar
    CLUSTER = 1 << 10  # LC Galaxy Cluster
    SNR_CUT = 1 << 11  # Signal-to-Noise cut flag (SNR > 5 per survey)
    ALL = (UNDEFINED | ASTROMETRIC | PSF | PHOTOMETRIC | CALSPEC | BRIGHT | COSMOLOGY | ICES | NO_ICES
           | SECONDARY | STELLAR | CLUSTER | SNR_CUT)

    def get_names_as_list(self):
        # No bit flag case
        if self.value == 0:
            return ["UNDEFINED"]

        # Combination of bit flags (does not account for UNDEFINED)
        names = [flag.name for flag in SourcePipelineClass if flag in self]
        return names


def _preprocess_per_catalog(cat: Table,
                            srcPipeFlag: SourcePipelineClass = SourcePipelineClass.UNDEFINED) -> Table:
    if srcPipeFlag > 0:
        # keep only the flagged sources
        msk = cat["spherex_class"] & srcPipeFlag == srcPipeFlag
        cat = cat[msk]
    return cat


def _preprocess_per_image(cat: Table,
                          img: SPHERExImage,
                          addPixelCoords: bool = True,
                          pixelCoordsWithDistortion: bool = True,
                          filterSourcesOutsideImage: bool = True,
                          filterPixMargin: float = 0.0) -> Table:
    logging.info("==============================================================")
    logging.info(f"CATALOG: {len(cat)} sources.")
    # add pixel coordinates ix and iy
    if addPixelCoords or filterSourcesOutsideImage:
        wcs = img.wcs
        if not pixelCoordsWithDistortion:
            wcs.sip = None
        cat["ix"], cat["iy"] = wcs.world_to_pixel_values(cat["ra"], cat["dec"])

    # filter sources outside image boundaries
    if filterSourcesOutsideImage:
        # image boundaries
        xnpix, ynpix = img.shape
        # if negative margin is greater than half image, clear it out
        # this is needed for testing with small images
        if (-filterPixMargin > xnpix / 2) or (-filterPixMargin > ynpix / 2):
            filterPixMargin = 0
        (xmin, xmax) = (-0.5 - filterPixMargin, xnpix - 0.5 + filterPixMargin)
        (ymin, ymax) = (-0.5 - filterPixMargin, ynpix - 0.5 + filterPixMargin)
        msk = (cat["iy"] > ymin) & (cat["iy"] < ymax) & (cat["ix"] > xmin) & (cat["ix"] < xmax)
        cat = cat[msk]
        logging.info(f"CATALOG: {len(cat)} sources within ({xmin}-{xmax}, {ymin}-{ymax}).")
        logging.info("==============================================================")

    return cat


def preprocess_catalog(cat: Table,
                       img: SPHERExImage,
                       srcPipeFlag: SourcePipelineClass = SourcePipelineClass.UNDEFINED,
                       addPixelCoords: bool = True,
                       pixelCoordsWithDistortion: bool = False,
                       filterSourcesOutsideImage: bool = True,
                       filterPixMargin: float = 0.0) -> Table:
    """Preprocess catalog for a given image.

    Parameters
    ----------
    cat : `Table`
        Catalog of sources that contains "ra", "dec", and "spherex_class" columns.
    img : `~spx_data.SPHERExImage`
        SPHEREx Image.
    srcPipeFlag : `SourcePipelineClass`, optional
        SPHEREx Pipeline Class from reference catalog.
        Default is ``SourcePipelineClass.UNDEFINED`` or no filtering on "spherex_class".
    addPixelCoords : `bool`, optional
        If True, adds "ix" and "iy" columns with pixel coordinates for each source.
        Default is True
    pixelCoordsWithDistortion : `bool`, optional
        If True, SIP parameters are considered when evaluating pixel coordinates.
        Default is False.
    filterSourcesOutsideImage : `bool`, optional
        If True, the sources outside image are filtered out. Default is True.
    filterPixMargin : `float`, optional
        Pixel margin for filtering out the sources. Default is 0.0.
        To remove the sources at image edges, use negative margin.

    Returns
    -------
    result : Table
        Catalog table preprocessed according to the parameters.

    """
    return _preprocess_per_image(
        _preprocess_per_catalog(cat, srcPipeFlag),
        img,
        addPixelCoords,
        pixelCoordsWithDistortion,
        filterSourcesOutsideImage,
        filterPixMargin)


def sortHandles(handles: list[DeferredDatasetHandle]) -> list[DeferredDatasetHandle]:
    return sorted(handles,
                  key=lambda r: (r.dataId["spsplan"], r.dataId["slewgroup"], r.dataId["dichroicpair"]))


class ImageCatalogIterable(Iterable):
    """Iterable for image and catalog pairs.

    The catalog is identified by ``spsplan``, ``slewgroup``, and ``dichroicpair``,
    while image has an additional ``slewgroup_counter`` dimension.
    Hence, a catalog might be shared by several images.

    This iterable can be used by calibration pipeline tasks, which
    accept multiple images and catalogs as input and need to pair
    image with the corresponding catalog, preprocessing catalog
    for each image.

    Parameters
    ----------
    imageHandles : `list[~lsst.daf.butler.DeferredDatasetHandle]`
    catalogHandles : `list[~lsst.daf.butler.DeferredDatasetHandle]`
    srcPipeFlag : `SourcePipelineClass`, optional
        SPHEREx Pipeline Class from reference catalog.
        Default is ``SourcePipelineClass.UNDEFINED`` or no filtering on "spherex_class".
    addPixelCoords : `bool`, optional
        If True, adds "ix" and "iy" columns with pixel coordinates for each source.
        Default is True
    pixelCoordsWithDistortion : `bool`, optional
        If True, SIP parameters are not considered when evaluating pixel coordinates.
        Default is False.
    filterSourcesOutsideImage : `bool`, optional
        If True, the sources outside image are filtered out. Default is True.
    filterPixMargin : `float`, optional
        Pixel margin for filtering out the sources. Default is 0.0.
        To remove the sources at image edges, use negative margin.

    Notes
    -----
    ``preprocess_catalog`` provides equivalent catalog preprocessing functionality
    for a single image and catalog pair.

    """

    def __init__(self,
                 imageHandles: list[DeferredDatasetHandle],
                 catalogHandles: list[DeferredDatasetHandle],
                 *,
                 srcPipeFlag: SourcePipelineClass = SourcePipelineClass.UNDEFINED,
                 addPixelCoords: bool = True,
                 pixelCoordsWithDistortion: bool = True,
                 filterSourcesOutsideImage: bool = True,
                 filterPixMargin=0):
        self.imageHandles = sortHandles(imageHandles)
        self.catalogHandles = sortHandles(catalogHandles)
        self.numCatalogs = len(catalogHandles)
        self.srcPipeFlag = srcPipeFlag
        self.addPixelCoords = addPixelCoords
        self.pixelCoordsWithDistortion = pixelCoordsWithDistortion
        self.filterSourcesOutsideImage = filterSourcesOutsideImage
        self.filterPixMargin = filterPixMargin

    def __iter__(self) -> Iterator[tuple[SPHERExImage, Table]]:
        catIdx = -1
        cat: Optional[Table] = None
        catDataId = (-1, -1)
        for imgHand in self.imageHandles:
            imgDataId = (
                imgHand.dataId["spsplan"],
                imgHand.dataId["slewgroup"],
                imgHand.dataId["dichroicpair"]
            )
            img = imgHand.get()

            if cat is None or catDataId != imgDataId:
                # get matching catalog
                while catDataId != imgDataId:
                    catIdx += 1
                    if catIdx >= self.numCatalogs:
                        raise ValueError(f"Catalog for spsplan, slewgroup, dichroicpair {imgDataId} "
                                         f"is not found in {self.numCatalogs} catalogs.")

                    # next catalog handle and id
                    catHand = self.catalogHandles[catIdx]
                    catDataId = (
                        catHand.dataId["spsplan"],
                        catHand.dataId["slewgroup"],
                        catHand.dataId["dichroicpair"]
                    )

                    if catDataId == imgDataId:
                        # found matching catalog
                        cat = self.catalogHandles[catIdx].get()
                        cat = self._preprocessPerCatalog(cat.copy())
                        break

                if catDataId != imgDataId:
                    raise ValueError("Catalog for spsplan, slewgroup, dichroicpair"
                                     f"  {imgDataId} is not found.")

                if cat is None:
                    raise ValueError(f"Catalog for spsplan, slewgroup, dichroicpair"
                                     f"  {imgDataId} is None.")

            yield img, self._preprocessPerImage(cat.copy(), img)

    def _preprocessPerCatalog(self, cat: Table) -> Table:
        return _preprocess_per_catalog(cat, self.srcPipeFlag)

    def _preprocessPerImage(self, cat: Table, img: SPHERExImage) -> Table:
        return _preprocess_per_image(cat, img,
                                     addPixelCoords=self.addPixelCoords,
                                     pixelCoordsWithDistortion=self.pixelCoordsWithDistortion,
                                     filterSourcesOutsideImage=self.filterSourcesOutsideImage,
                                     filterPixMargin=self.filterPixMargin)
