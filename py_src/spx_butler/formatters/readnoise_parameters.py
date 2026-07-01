
__all__ = ["ReadNoiseParametersFormatter"]

from astropy.io import fits
from spx_data import ReadNoiseParameters
from typing import (
    Any,
    Optional,
    Type,
)

from lsst.daf.butler.formatters.file import FileFormatter


class ReadNoiseParametersFormatter(FileFormatter):
    """Interface for reading and writing astropy
    image objects to and from FITS files.
    """

    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})

    unsupportedParameters = None
    """This formatter does not support any parameters (`frozenset`)"""

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> Any:
        """Read a file from the path in FITS format.

        Parameters
        ----------
        path : `str`
            Path to use to open image file.
        pytype : `~typing.Any` | `None`, optional
            Not used by this implementation.

        Returns
        -------
        `~spx_data.ReadNoiseParameters` | `None`
            Either data as `ReadNoiseParameters` object read from fits file,
            or `None` if the file could not be opened.
        """
        try:
            data = ReadNoiseParameters(fits.open(path))
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
        if not isinstance(inMemoryDataset, ReadNoiseParameters):
            raise NotImplementedError(
                "Unable to write this representation of NonlinearityParameters into a file.")
        try:
            inMemoryDataset.validate()
        except ValueError as ve:
            raise ValueError(f"Unable to write representation of NonlinearityParameters as file\
                             became invalid after loading: {ve}")
        inMemoryDataset.hdulist.writeto(self.fileDescriptor.location.path)
