__all__ = ["NonfuncDataFormatter"]

from astropy.io import fits
from spx_data import NonfuncData
from typing import Any, Optional, Type

from lsst.daf.butler.formatters.file import FileFormatter


class NonfuncDataFormatter(FileFormatter):
    """Formatter for reading/writing NonfuncData to FITS."""

    extension = ".fits"
    supportedExtensions = frozenset({".fits", ".fits.gz"})

    unsupportedParameters = frozenset()  # no supported parameters
    """This formatter does not support any parameters (`frozenset`)."""

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> NonfuncData:
        """Read NonfuncData from a FITS file.

        Parameters
        ----------
        path : `str`
            Path to the FITS file.
        pytype : `Type[Any]` | `None`
            Unused (kept for API parity).

        Returns
        -------
        `spx_data.NonfuncData`
            NonfuncData object.

        Raises
        ------
        FileNotFoundError
            If the FITS file is missing.

        """
        try:
            data = NonfuncData(fits.open(path))
            return data
        except FileNotFoundError as e:
            raise FileNotFoundError(f"NonfuncData file not found: {path}") from e

    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write NonfuncData as a FITS file.

        Parameters
        ----------
        inMemoryDataset : `object`
            Object to serialize; must be `NonfuncData`.

        Raises
        ------
        NotImplementedError
            If the provided object is not a NonfuncData.
        ValueError
            If validation fails before writing.

        """
        if not isinstance(inMemoryDataset, NonfuncData):
            raise NotImplementedError(
                "Unable to write this representation of NonfuncData into a file."
            )
        try:
            # Ensure required EXTNAMEs exist, detector header present, etc.
            inMemoryDataset.validate()
        except ValueError as ve:
            raise ValueError(
                "Unable to write representation of NonfuncData as file; "
                f"object became invalid after loading: {ve}"
            )
        # Persist the underlying HDUList (same pattern as your nonlinearity formatter)
        inMemoryDataset.hdulist.writeto(self.fileDescriptor.location.path, overwrite=True)
