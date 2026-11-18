__all__ = ["SPHERExImageConfig"]

# Standard library imports

# Related third party imports
import lsst.pex.config as pexConfig

# Related third party imports
from spx_data import Layer

EXTNAMES = tuple([v for k, v in Layer.__dict__.items() if not k.startswith("_")])


class SPHERExImageConfig(pexConfig.Config):
    """Configuration for SPHERExImage FITS extension mapping."""

    unit = pexConfig.Field(
        doc="Units of the image data.",
        dtype=str,
        optional=True
    )
    hdu_uncertainty = pexConfig.Field(
        doc="FITS extension from which the uncertainty should be initialized.",
        dtype=str,
        default="VARIANCE"
    )
    hdu_mask = pexConfig.Field(
        doc="FITS extension from which the mask should be initialized.",
        dtype=str,
        default="MASK"
    )
    hdu_flags = pexConfig.Field(
        doc="FITS extension from which the flags should be initialized.",
        dtype=str,
        default="FLAGS"
    )
    hdu_layers = pexConfig.ListField(
        doc="FITS extensions from which the layers should be initialized.",
        dtype=str,
        default=EXTNAMES
    )
    key_uncertainty_type = pexConfig.Field(
        doc="The header key name where the class name of the uncertainty is "
            "stored in the hdu of the uncertainty.",
        dtype=str,
        default="UTYPE"
    )
    dtype_flags = pexConfig.Field(
        doc="The desired numpy data type for flags.",
        dtype=str,
        default="int32"
    )
    dtype_data = pexConfig.Field(
        doc="The desired numpy data type for data and uncertainty.",
        dtype=str,
        default="float32"
    )
    hdu = pexConfig.Field(
        doc="FITS extension from which the data should be initialized.",
        dtype=int,
        default=1
    )
