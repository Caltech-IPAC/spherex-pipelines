__all__ = ["SPHERExPSFFormatter"]

from typing import (
    Any,
    Optional,
    Type,
)

from lsst.daf.butler.formatters.file import FileFormatter

from spx_data import PSFZoneData


class SPHERExPSFFormatter(FileFormatter):
    """Interface for reading and writing SPHEREx PSF
    data to and from FITS files.
    """

    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})

    unsupportedParameters = None
    """This formatter does not support any parameters (`frozenset`).
    Read parameters are passed when butler's get is called. They
    might be used for getting a slice of an image, for example."""

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> Any:
        """Read a file from the path in FITS format.

        Parameters
        ----------
        path : `str`
            Path to use to open image file.
        pytype : `class`, optional
            Not used by this implementation.

        Returns
        -------
        data : `object`
            Either data as Python object read from JSON file, or None
            if the file could not be opened.

        """
        try:
            data = PSFZoneData.read(path)
        except FileNotFoundError:
            data = None

        return data

    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset : `object`
            Object to serialize.

        Raises
        ------
        Exception
            The file could not be written.

        """
        if not isinstance(inMemoryDataset, PSFZoneData):
            raise NotImplementedError("Unable to write this representation of FITS into a file.")
        inMemoryDataset.write(self.fileDescriptor.location.path)
