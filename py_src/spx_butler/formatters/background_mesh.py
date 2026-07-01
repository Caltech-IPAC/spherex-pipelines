__all__ = ["BackgroundMeshFormatter"]

from typing import Any, Optional, Type
from lsst.daf.butler.formatters.file import FileFormatter

from spx_data import BackgroundMesh


class BackgroundMeshFormatter(FileFormatter):
    """Interface for reading and writing background model files."""
    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})

    unsupportedParameters = None

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> BackgroundMesh:
        """Read a file from the path in FITS format.

        Parameters
        ----------
        path : `str`
            Path to use to open FITS file.
        pytype : `class`, optional
            Not used by this implementation, by default None.

        Returns
        -------

        """
        try:
            data = BackgroundMesh.read(path)

        except FileNotFoundError:
            data = None

        return data

    def _writeFile(self, inMemoryDataset: BackgroundMesh) -> None:
        """Write in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset :

        Raises
        ------
        NotImplementedError
            Unable to write this representation of FITS into a file.
        """
        if not isinstance(inMemoryDataset, BackgroundMesh):
            raise NotImplementedError("Unable to write this representation of FITS into a file.")

        inMemoryDataset.write(self.fileDescriptor.location.path)
