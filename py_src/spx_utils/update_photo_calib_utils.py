__all__ = ["CalspecPredict"]

# Standard library imports
import logging

# Related third party imports
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u
import numpy as np
from scipy import interpolate

# Local application/library specific imports


class CalspecPredict(object):
    """A class to predict source fluxes based on high-resolution spectral templates.

    Attributes
    ----------
    Names : `list`
        List of source names.
    Coordinates : `dict`
        Dictionary mapping source names to their coordinates.
    Spectra : `dict`
        Dictionary mapping source names to their high-resolution spectral data.
    interp : `dict`
        Dictionary mapping source names to interpolation functions for their spectra.
    flux_unit : `dict`
        Dictionary mapping source names to the units of their flux values.
    nsources : `int`
        Counter for the number of loaded sources.
    logger : `logging.Logger`
        Logger instance for logging messages.

    Parameters
    ----------
    Logger : `logging.Logger`, optional
        Logger instance for logging messages.
        If not provided, the root logger will be used.

    """

    def __init__(self, Logger=None):
        self.Names = []
        self.Coordinates = {}
        self.Spectra = {}
        self.interp = {}
        self.flux_unit = {}
        self.nsources = 0
        if Logger:
            self.logger = Logger
        else:
            self.logger = logging.getLogger()

    def load_catalog(self,
                     inputpath: str,
                     band_data_file: str = "photometric_bands.fits") -> None:
        """Load a catalog of sources and their associated spectral data.



        Parameters
        ----------
        inputpath : `str`
            Path to the catalog file containing source information.
        band_data_file : `str`, optional
            Path to the file with photometric band information.
            Default is "photometric_bands.fits".

        Returns
        -------
        `None`

        Notes
        -----
        This method reads a catalog file with source metadata and photometric band data,
        and constructs high-resolution spectral templates for each source.
        It then creates an interpolation function for each source's spectrum.

        """
        Bandinfo = Table.read(band_data_file)
        # Make a lookup of band name to wavelength
        band_wl = {}
        for ii, field in enumerate(Bandinfo["band"]):
            band_wl[field] = Bandinfo["wl"][ii]

        T = Table.read(inputpath)
        for row in T:
            s_rows = []
            for band in band_wl.keys():
                if band in row.colnames:
                    if not np.isnan(row[band]):
                        s_rows.append([band_wl[band], row[band] / 1e3])
            name = f'{row["source_id"]}'
            self.Names.append(name)
            self.Coordinates[name] = SkyCoord(row["ra"] * u.deg,
                                              row["dec"] * u.deg,
                                              frame="icrs")
            spectrum = Table(rows=s_rows, names=["lambda", "FLUX"])
            spectrum.sort("lambda")
            spectrum["lambda"].unit = u.um
            spectrum["FLUX"].unit = u.mJy
            self.Spectra[name] = spectrum
            self.flux_unit[name] = u.mJy

            self.interp[name] = interpolate.interp1d(self.Spectra[name]["lambda"],
                                                     self.Spectra[name]["FLUX"],
                                                     kind="linear",
                                                     bounds_error=False,
                                                     fill_value=0.0)

            self.logger.info(f"Added source {name} with coordinate {self.Coordinates[name]}."
                             f"Spectrum is in {self.flux_unit[name]}")

    def load_single(self, name: str, ra: float, dec: float, inputpath: str) -> None:
        """Load an individual source's spectrum from a file.


        Parameters
        ----------
        name : `str`
            Name of the source.
        ra : `float`
            Right ascension of the source in degrees.
        dec : `float`
            Declination of the source in degrees.
        inputpath : `str`
            Path to the FITS file containing the source's spectral data.

        Returns
        -------
        `None`

        Notes
        -----
        This method reads a FITS file containing high-resolution spectral data
        for a single source, adds the source's name and coordinates, and
        creates an interpolation function for the spectrum.

        """
        self.Names.append(name)
        self.Coordinates[name] = SkyCoord(ra, dec, frame="icrs")
        self.Spectra[name] = Table.read(inputpath)
        self.interp[name] = interpolate.interp1d(self.Spectra[name]["lambda"],
                                                 self.Spectra[name]["FLUX"],
                                                 kind="linear",
                                                 bounds_error=False,
                                                 fill_value=0.0)
        self.flux_unit[name] = self.Spectra[name]["FLUX"].unit
        # TODO: do better checking of the spectrum input,
        # make sure its one we can convert to millijansky
        self.logger.info(f"Added source {name} with coordinate {self.Coordinates[name]}."
                         f"Spectrum is in {self.flux_unit[name]}")
