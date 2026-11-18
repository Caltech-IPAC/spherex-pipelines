__all__ = ["PersistenceLookupTableFormatter"]

from deprecated.sphinx import deprecated
from astropy.io import fits
from typing import Optional, Type, Any
from lsst.daf.butler.formatters.file import FileFormatter
from spx_data import PersistenceLookupTable


class PersistenceLookupTableFormatter(FileFormatter):
    """Interface for reading astropy table objects from FITS files
    and combining specific extensions.
    """

    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})

    unsupportedParameters = None
    """This formatter does not support any parameters (`frozenset`)"""

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> Any:
        """Read specific extensions from a FITS file, combine them,
        and return as an astropy table.

        Parameters
        ----------
        path : str
            Path to the FITS file to read.
        pytype : Optional[Type[Any]], optional
            Not used in this implementation.

        Returns
        -------
        astropy.table.Table
            A combined table from the first and third extensions of the FITS file,
            or None if the file could not be opened.

        """
        try:
            data = PersistenceLookupTable(fits.open(path))
        except FileNotFoundError:
            data = None
        return data

    @deprecated(
        "Deprecated in favor of module configuration parameters. Will be removed.",
        version="6.2",
        category=FutureWarning,
    )
    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write the in-memory dataset to a file on disk.

        Parameters
        ----------
        inMemoryDataset : object
            The dataset to serialize.

        """
        if not isinstance(inMemoryDataset, PersistenceLookupTable):
            raise NotImplementedError(
                "Unable to write this representation of LookupTable into a file.")
        try:
            inMemoryDataset.validate()
        except ValueError as ve:
            raise ValueError(f"Unable to write representation of LookupTable as file\
                             became invalid after loading: {ve}")
        # fits.HDUList to file not LookupTable()
        inMemoryDataset.hdulist.writeto(self.fileDescriptor.location.path)
