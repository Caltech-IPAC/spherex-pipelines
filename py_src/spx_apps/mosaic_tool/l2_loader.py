"""L2 FITS file loading."""

from dataclasses import dataclass
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import numpy as np

from spx_data import Flag


@dataclass
class L2Image:
    """Container for L2 image data.

    Attributes
    ----------
    data : numpy.ndarray
        2D image data with units, shape (ny, nx).
    wcs : astropy.wcs.WCS
        World Coordinate System for the image.
    array : int
        Detector array number (1-6).
    filename : str
        Source filename.
    flags : numpy.ndarray
        2D array of propagated flags (int32), shape (ny, nx).

    """

    data: np.ndarray
    wcs: WCS
    array: int
    filename: str
    flags: np.ndarray


class L2FileLoader:
    """Load L2 FITS files.

    This loader reads SPHEREx L2 FITS files and returns the data
    in native FITS orientation (ny, nx) without transposition.

    Zodi is subtracted.

    Parameters
    ----------
    bad_pixel_flags : `spx_data.Flag`
        Bitwise OR of pixel flags indicating bad pixels.
    propagated_flags : `spx_data.Flag`
        Bitwise OR of pixel flags to propagate to mosaic output.
    default_unit : astropy.units.Unit, optional
        Default unit if BUNIT header is missing or unrecognized.
        Default is MJy/sr.

    Examples
    --------
    >>> loader = L2FileLoader()
    >>> image = loader.load('/path/to/l2_file.fits', array=1)
    >>> print(image.data.shape)
    (2040, 2040)

    """

    def __init__(self, bad_pixel_flags: Flag = None,
                 propagated_flags: Flag = None,
                 default_unit: u.Unit = u.MJy / u.sr):
        self.bad_pixel_flags = bad_pixel_flags
        self.propagated_flags = propagated_flags
        self.default_unit = default_unit

    def load(
        self,
        filepath: str,
        array: int
    ) -> L2Image:
        """Load an L2 FITS file.

        Parameters
        ----------
        filepath : str
            Path to the FITS file.
        array : int
            Detector array number (1-6) for metadata.

        Returns
        -------
        L2Image
            Container with data, WCS, and metadata.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file cannot be read or has invalid format.

        """
        with fits.open(filepath) as hdul:
            # Find the science extension
            if "IMAGE" in hdul and "ZODI" in hdul:
                hdu = hdul["IMAGE"]
                hdu_z = hdul["ZODI"]
                hdu_f = hdul["FLAGS"]
            else:
                raise ValueError("IMAGE or ZODI extensions not found")

            wcs = WCS(hdu.header)

            # both IMAGE and ZODI are in the same units MJy/sr
            data = hdu.data - hdu_z.data
            flags = hdu_f.data

            # Set bad pixel values to NaN
            bad_pixel_mask = None
            if self.bad_pixel_flags is not None:
                bad_pixel_mask = (flags & self.bad_pixel_flags) > 0
                data[bad_pixel_mask] = np.nan

            # Filter flags to retain only those that should be propagated.
            # Propagation rules:
            # - OVERFLOW: propagated only when NOT accompanied by SUR_ERROR
            # - All propagated flags are combined via bitwise OR across contributors
            # - Bad pixels (including pixels with SUR_ERROR) don't contribute
            propagated_flags = (flags & self.propagated_flags).astype(np.int32)

            # Clear all flags for bad pixels since they're excluded from mosaic
            if bad_pixel_mask is not None:
                propagated_flags[bad_pixel_mask] = 0

            # Determine units
            bunit = hdu.header.get('BUNIT', '').strip()
            if bunit:
                try:
                    unit = u.Unit(bunit)
                except Exception:
                    if 'e-' in bunit.lower():
                        unit = u.electron / u.s
                    else:
                        raise ValueError(f"Invalid FITS unit: {bunit}")
            else:
                unit = self.default_unit

        return L2Image(
            data=data * unit,
            wcs=wcs,
            array=array,
            filename=filepath,
            flags=propagated_flags
        )
