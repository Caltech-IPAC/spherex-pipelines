__all__ = ["PersistenceParametersFormatter", "PersistenceStateFormatter"]

from deprecated.sphinx import deprecated
from astropy.io import fits
from spx_data import PersistenceParameters, PersistenceState
from typing import (
    Any,
    Optional,
    Type,
)

from lsst.daf.butler.formatters.file import FileFormatter


class PersistenceParametersFormatter(FileFormatter):
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
        `~spx_data.PersistenceParameters` | `None`
            Either data as `PersistenceParameters` object read from fits file,
            or `None` if the file could not be opened.
        """
        try:
            data = PersistenceParameters(fits.open(path))
        except FileNotFoundError:
            data = None
        return data

    @deprecated(
        "Deprecated in favor of module configuration parameters. Will be removed.",
        version="6.2",
        category=FutureWarning,
    )
    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset
            Object to serialize.

        Raises
        ------
        Exception
            The file could not be written.
        """
        if not isinstance(inMemoryDataset, PersistenceParameters):
            raise NotImplementedError(
                "Unable to write this representation of PersistenceParameters into a file.")
        try:
            inMemoryDataset.validate()
        except ValueError as ve:
            raise ValueError(f"Unable to write representation of PersistenceParameters as file\
                             became invalid after loading: {ve}")
        inMemoryDataset.hdulist.writeto(self.fileDescriptor.location.path)


class PersistenceStateFormatter(FileFormatter):
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
        `~spx_data.PersistenceState` | `None`
            Either data as `PersistenceState` object read from fits file,
            or `None` if the file could not be opened.
        """
        try:
            data = PersistenceState(fits.open(path))
        except FileNotFoundError:
            data = None
        return data

    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write persistence state in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset : `object`
            Object to serialize.

        Raises
        ------
        Exception
            The file could not be written.
        """
        if not isinstance(inMemoryDataset, PersistenceState):
            raise NotImplementedError(
                "Unable to write this representation of PersistenceState into a file.")
        try:
            inMemoryDataset.validate()
        except ValueError as ve:
            raise ValueError(f"Unable to write representation of PersistenceState as file\
                             became invalid after loading: {ve}")
        inMemoryDataset.hdulist.writeto(self.fileDescriptor.location.path)
