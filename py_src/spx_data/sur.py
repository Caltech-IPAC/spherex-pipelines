__all__ = ['SUR']

from typing import TypeVar, Any

import numpy as np
from numpy.typing import ArrayLike

import astropy.units as u
from astropy.io import fits

from .spx_param import format_unit

TSUR = TypeVar("TSUR", bound="SUR")


class SUR:
    """Full sample data in electrons from a small region of the detector.

    The data are 3d array, where first dimension is the sample number.
    """

    def __init__(self, data: ArrayLike | None, row_start: int, col_sectn: int | None = None, unit=u.electron):
        """

        Parameters
        ----------
        data : `numpy.typing.ArrayLike`
            3D array containing data with `float` type
        row_start : int
            Row offset of the full-sample area relative to the parent image.
        col_sectn :int
            Column section full-sample area relative to the parent image.
        """
        if hasattr(data, "value"):
            self.data = data.value
            self.unit = data.unit
        else:
            self.data = data
            self.unit = unit

        self.row_start = row_start
        self.col_sectn = col_sectn

    def to_hdu(self, name: str) -> fits.ImageHDU:
        hdr = fits.Header()
        hdr["BUNIT"] = format_unit(self.unit), "Unit of pixel value"
        hdr["ROWSTART"] = (self.row_start, "X offset relative to parent image")
        hdr["COLSECTN"] = (self.col_sectn, "Column section number")
        return fits.ImageHDU(data=self.data, header=hdr, name=name)

    @classmethod
    def from_hdu(cls, hdu) -> TSUR:
        if isinstance(hdu, fits.ImageHDU):
            data = hdu.data
            hdr = hdu.header
            if "BUNIT" in hdr:
                data <<= u.Unit(hdr["BUNIT"])
            try:
                row_start = hdr["ROWSTART"]
                col_sectn = hdr["COLSECTN"]
            except KeyError:
                # Set sur data to None if expected headers are missing.
                # Will skip ValidateSUR module in the case sur data is None.
                return SUR(data=None, row_start=0, col_sectn=0)
            return SUR(data, row_start, col_sectn)
        else:
            raise ValueError("Unexpected hdu type for image layer")

    def __eq__(self, other: Any):
        return isinstance(other, SUR) and \
            np.array_equal(self.data, other.data)
