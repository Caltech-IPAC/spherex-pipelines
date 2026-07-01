__all__ = ["Zodi", "TZodi"]

import numpy as np
import astropy.units as u
from astropy.io import fits
from typing import TypeVar, Any
from .spx_param import format_unit

TZodi = TypeVar("TZodi", bound="Zodi")


class Zodi:
    """Modelled zodi light"""

    def __init__(self, zodi_data, zodi_model_name, unit=u.Unit("MJy / sr")):
        if hasattr(zodi_data, "value"):
            self.zodi_data = zodi_data.value
            self.unit = zodi_data.unit
        else:
            self.zodi_data = zodi_data
            self.unit = unit
        self.zodi_model_name = zodi_model_name

    def to_hdu(self, name: str) -> fits.ImageHDU:
        hdr = fits.Header()
        hdr["ZODIMDL"] = (self.zodi_model_name, "Zodi model name")
        hdr["BUNIT"] = format_unit(self.unit), "Unit of pixel value"
        return fits.ImageHDU(data=self.zodi_data.astype(np.float32, copy=False), header=hdr, name=name)

    @classmethod
    def from_hdu(cls, hdu) -> TZodi:
        if isinstance(hdu, fits.ImageHDU):
            zodi_data = hdu.data
            hdr = hdu.header
            zodi_model_name = hdr["ZODIMDL"]
            if "BUNIT" in hdr:
                zodi_data <<= u.Unit(hdr["BUNIT"])
            return Zodi(zodi_data, zodi_model_name)
        else:
            raise ValueError("Unexpected hdu type for image layer")

    def __eq__(self, other: Any):
        return isinstance(other, Zodi) and \
            self.zodi_model_name == other.zodi_model_name and \
            self.unit == other.unit and \
            np.array_equal(self.zodi_data, other.zodi_data)
