__all__ = ["SPHERExImageFormatter"]

from typing import Any
from lsst.daf.butler.formatters.file import FileFormatter
from spx_data import SPHERExImage


class SPHERExImageFormatter(FileFormatter):
    """Interface for reading and writing astropy
    image objects to and from FITS files.
    """

    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})
    supportedWriteParameters = frozenset({"dtype_flags", "dtype_data", "extname"})

    def _readFile(self, path: str, pytype: Any | None = None) -> SPHERExImage | None:
        """Read a file from the path in FITS format.

        Parameters
        ----------
        path : `str`
            Path to use to open image file.
        pytype : `~typing.Any` | `None`, optional
            Not used by this implementation.

        Returns
        -------
        result : `~spx_data.SPHERExImage` or `None`
            Either data as `~spx_data.SPHERExImage` object read from fits file,
            or `None` if the file could not be opened.
        """
        read_kwargs = self.fileDescriptor.parameters or {}

        try:
            data = SPHERExImage.read(path, **read_kwargs)
        except FileNotFoundError:
            data = None

        return data

    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset : `Any`
            Object to serialize.

        Raises
        ------
        Exception
            The file could not be written.
        """
        if not isinstance(inMemoryDataset, SPHERExImage):
            raise NotImplementedError("Unable to write this representation of FITS into a file.")
        # Write to precision wanted.
        inMemoryDataset.dtype_flags = self.writeParameters.get("dtype_flags",
                                                               inMemoryDataset.dtype_flags)
        inMemoryDataset.dtype_data = self.writeParameters.get("dtype_data",
                                                              inMemoryDataset.dtype_data)
        # Write extname if not default.
        extname = self.writeParameters.get("extname", "IMAGE")

        inMemoryDataset.write(self.fileDescriptor.location.path, extname=extname)
