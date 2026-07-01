__all__ = ["AstropySIPFormatter"]

from astropy.io import fits
from astropy import wcs
from typing import (
    Any,
    Optional,
    Type,
)

from lsst.daf.butler.formatters.file import FileFormatter


class AstropySIPFormatter(FileFormatter):
    """Interface for reading and writing astropy WCS SIP file"""
    extension = ".fits"
    supportedExtensions = frozenset({".fits.gz", ".fits"})

    unsupportedParameters = None

    def _readFile(self, path: str, pytype: Optional[Type[Any]] = None) -> Any:
        """Read a file from the path in FITS format.

        Parameters
        ----------
        path : `str`
            Path to use to open WCS/SIP file.
        pytype : `class`, optional
            Not used by this implementation, by default None.

        Returns
        -------
        data : `object`
            WCS object, or None if the file could not be opened.
        """
        try:
            with fits.open(path) as hdulist:
                header = hdulist[0].header
                # Hack to get around Astropy warning
                header.pop("NAXIS")

                w = wcs.WCS(header)
                w = w.sip

        except FileNotFoundError:
            w = None

        return w

    def _writeFile(self, inMemoryDataset: Any) -> None:
        """Write in memory dataset to file on disk.

        Parameters
        ----------
        inMemoryDataset : `object`
            Object to serialize.

        Raises
        ------
        NotImplementedError
            Unable to write this representation of FITS into a file.
        """
        if not isinstance(inMemoryDataset, wcs.Sip):
            raise NotImplementedError("Unable to write this representation of"
                                      " SIP into a file.")

        # Need to create a WCS to store the SIP coefficients
        w = wcs.WCS()
        w.sip = inMemoryDataset
        w.wcs.crpix = w.sip.crpix
        # Specify SIP coefficients exist to supress Astropy warning
        w.wcs.ctype = ["X-SIP", "Y-SIP"]

        # relax=True required for writing SIP information to header
        header = w.to_header(relax=True)
        header['COMMENT'] = "Only SIP information is usable."

        hdu = fits.PrimaryHDU(header=header)

        hdu.writeto(self.fileDescriptor.location.path)
