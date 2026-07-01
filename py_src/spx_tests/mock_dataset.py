__all__ = ["MockDeferredImageHandle", "MockDeferredCatalogHandle", "mockImageCatalogHandles"]

# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.table import Table
from lsst.daf.butler import DeferredDatasetHandle, DataCoordinate

# Local application/library specific imports
from spx_data import SPHERExImage


class MockDeferredImageHandle(DeferredDatasetHandle):

    def __init__(self,
                 filepath: Path = None,
                 image: SPHERExImage = None,
                 dataid: dict = None):
        """Create mock image handle.

        Either ``filepath`` or ``image`` must be given.
        If both are given, ``image`` is used.

        Parameters
        ----------
        filepath : `Path`, optional
            File with SPHEREx image.
        image : `~spx_data.SPHERExImage`, optional
            SPHEREx image
        dataid : `dict`, optional
            Data coordinate of this dataset.
        """
        if filepath is None and image is None:
            raise ValueError(f"Either filepath or image required to create {self.__class__.__name__}")
        params = {"filepath": filepath, "image": image, "dataid": dataid}
        super().__init__(None, None, parameters=params)

    def __repr__(self):
        return f'''MockDeferredImageHandle('{self.parameters["filepath"]}')'''

    def get(self, **kwargs) -> SPHERExImage:
        """Retrieves the dataset pointed to by this handle"""
        if self.parameters["image"] is None:
            spximg = SPHERExImage.read(self.parameters["filepath"])
            self.parameters["image"] = spximg
        return self.parameters["image"]

    @property
    def dataId(self) -> DataCoordinate:
        """The data ID associated with the image."""
        return self.parameters["dataid"]


class MockDeferredCatalogHandle(DeferredDatasetHandle):

    def __init__(self,
                 imghandle: MockDeferredImageHandle):
        super().__init__(None, None, parameters={"imghandle": imghandle})

    def __repr__(self):
        return f'''MockDeferredCatalogHandle('{self.parameters["imghandle"]}')'''

    def get(self, **kwargs) -> Table:
        """Retrieves catalog"""
        return self.parameters["imghandle"].get().catalog

    @property
    def dataId(self) -> DataCoordinate:
        """The data ID associated with the image."""
        return self.parameters["imghandle"].dataId


class MockDeferredSSOCatalogHandle(DeferredDatasetHandle):

    def __init__(self,
                 imghandle: MockDeferredImageHandle):
        super().__init__(None, None, parameters={"imghandle": imghandle})

    def __repr__(self):
        return f'''MockDeferredSSOCatalogHandle('{self.parameters["imghandle"]}')'''

    def get(self, **kwargs) -> Table:
        """Retrieves catalog"""
        return self.parameters["imghandle"].get().ssocat

    @property
    def dataId(self) -> DataCoordinate:
        """The data ID associated with the image."""
        return self.parameters["imghandle"].dataId


def mockImageCatalogHandles(datapath: Path) -> tuple[list[DeferredDatasetHandle],
                                                     list[DeferredDatasetHandle]]:
    """Get image and catalog handles for simulated images with CATALOG extension.

    Parameters
    ----------
    datapath
        Directory with the simulated FITS files.

    Returns
    -------
    result : tuple [`list`, `list`]
        Returns a tuple composed of two lists with
        `~lsst.daf.butler.DeferredDatasetHandle` objects:
        the first list contains handles for the deferred images,
        the second list contains handles for the deferred catalogs.
    """

    fits_files = sorted(datapath.glob("*.fits*"))  # also supports fits.gz

    def mockDataId(i: int):
        return {"slewgroup": i, "dichroicpair": 0}

    images = [MockDeferredImageHandle(p, dataid=mockDataId(i)) for i, p in enumerate(fits_files)]
    catalogs = [MockDeferredCatalogHandle(h) for h in images]

    return images, catalogs
