__all__ = ["ReadoutParameters"]

from astropy.io import fits
from dataclasses import dataclass, field
from typing import TypeVar
from .spx_param import readout_params

TReadoutParameters = TypeVar("TReadoutParameters", bound="ReadoutParameters")


@dataclass
class ReadoutParameters:
    """Parameters controlling the order and number of readout pixels.

    Attributes
    ----------
    nref : `int`
        Number of visits to each reference row.
    nskip : `int`
        Number of rows to skip.
    Mphan : `int`
        Number of phantom columns reads.
    nchannel : `int`
        Number of readout channels.
    wchannel : `int`
        Number of columns in each readout channel.
    jref : `int`
        Number of physical reference rows.
    kref : `int`
        Number of physical reference columns.
    Jactive : `int`
        Number of active (light-sensitive) rows.
    Kactive : `int`
        Number of active (light-sensitive) columns.
    Nvisit_col : `int`
        Number of visits o each reference column.
    Nvisit_row : `int`
        Derived. Number of visits to each reference row.
        Same as ``nref``.
    Jref : `int`
        Derived. Number of reference rows in the data.
    Kref : `int`
        Derived. Number of reference columns in the data.
    Jphan : `int`
        Derived. Number of phantom rows in the data.
        Same as ``Jframe``.
    Kphan : `int`
        Derived. Number of phantom columns in the data.
    Jframe : `int`
        Derived. Total number of rows in a data frame.
    Kframe : `int`
        Derived. Total number of columns in a data frame.

    """

    nref: int = readout_params.Nvisit_row
    nskip: int = readout_params.Nskip
    Mphan: int = readout_params.Mphan
    nchannel: int = readout_params.n_readout_channels
    wchannel: int = readout_params.wchannel
    jref: int = readout_params.jref
    kref: int = readout_params.kref
    Jactive: int = readout_params.Jactive
    Kactive: int = readout_params.Kactive
    Nvisit_col: int = readout_params.Nvisit_col
    Nvisit_row: int = field(init=False)
    Jref: int = field(init=False)
    Kref: int = field(init=False)
    Jphan: int = field(init=False)
    Kphan: int = field(init=False)
    Jframe: int = field(init=False)
    Kframe: int = field(init=False)

    def __post_init__(self):
        self.Nvisit_row = self.nref
        self.Jref = self.jref * self.Nvisit_row
        self.Kref = self.kref * self.Nvisit_col
        self.Jphan = self.Jactive + self.Jref
        self.Kphan = self.nchannel * self.Mphan
        self.Jframe = self.Jactive + self.Jref
        self.Kframe = self.Kactive + self.Kref + self.Kphan

        if self.nchannel != (self.Kactive + self.kref) / self.wchannel:
            raise ValueError(f"The number of readout channels nchannel={self.nchannel}\
                              and the number of columns in each readout channel wchannel={self.wchannel}\
                              are not consistent.")

    def to_metadata(self) -> list[tuple[str, int, str]]:
        """Get the list of readout parameters that need to be saved with an image.

        The parameters are saved in the FITS header, those that are longer than
        8 chars need HIERARCH prefix.

        Returns
        -------
        result : `list of tuples`
            List of (name, value, description) tuples describing the parameters.

        """
        return [
            ("NSKIP", self.nskip, "Number of rows skipped after a full row is read"),
            ("HIERARCH NVISITROW", self.Nvisit_row, "Number of visits to each reference row"),
            ("HIERARCH NVISITCOL", self.Nvisit_col, "Number of visits to each reference col"),
            ("MPHAN", self.Mphan, "Number of phantom columns reads"),
            ("NCHANNEL", self.nchannel, "Number of readout channels"),
            ("WCHANNEL", self.wchannel, "Number of columns in each readout channel"),
            ("JREF", self.jref, "Number of physical reference rows on H2RG"),
            ("KREF", self.kref, "Number of physical reference cols on H2RG"),
            ("JACTIVE", self.Jactive, "Number of light-sensitive pixel rows"),
            ("KACTIVE", self.Kactive, "Number of light-sensitive pixel cols")
        ]

    @classmethod
    def from_metadata(cls, meta: fits.Header) -> TReadoutParameters:
        """Restore ReadoutParameters from FITS headers.

        Parameters
        ----------
        meta : `~astropy.io.fits.Header`
            Dictionary-like object with metadata.

        Returns
        -------
        readout_parameters : `~spx_data.ReadoutParameters`

        Raises
        ------
        ValueError
            Raised if invalid parameters or missing parameters.

        Notes
        -----
        Either the number of readout channels or the number of columns
        in a readout channel must be in the metadata.
        Throws and exception if both are not availabe.

        """
        nchannel = meta.get("NCHANNEL")
        wchannel = meta.get("WCHANNEL")

        if nchannel is None and wchannel is not None:
            nchannel = (meta["KACTIVE"] + meta["KREF"]) / wchannel
            if int(nchannel) != nchannel:
                Kactive = meta["KACTIVE"]
                Kref = meta["Kref"]
                raise ValueError(f"Invalid readout parameters: KACTIVE={Kactive}\
                                   , KREF={Kref}, WCHANNEL={wchannel}.")
            meta["NCHANNEL"] = nchannel
        elif nchannel is not None and wchannel is None:
            wchannel = (meta["KACTIVE"] + meta["KREF"]) / nchannel
            if int(wchannel) != wchannel:
                Kactive = meta["KACTIVE"]
                Kref = meta["Kref"]
                raise ValueError(f"Invalid readout parameters: KACTIVE={Kactive}\
                                   , KREF={Kref}, NCHANNEL={nchannel}.")
            meta["WCHANNEL"] = wchannel
        elif nchannel is None and wchannel is None:
            raise ValueError("NCHANNEL and WCHANNEL are missing.")

        return ReadoutParameters(nref=meta["NVISITROW"],
                                 nskip=meta["NSKIP"],
                                 Mphan=meta["MPHAN"],
                                 nchannel=int(nchannel),
                                 wchannel=int(wchannel),
                                 jref=meta["JREF"],
                                 kref=meta["KREF"],
                                 Jactive=meta["JACTIVE"],
                                 Kactive=meta["KACTIVE"],
                                 Nvisit_col=meta["NVISITCOL"])
