# SPHEREx Image will be stored in a FITS file with header-only primary hdu (to
# allow compressed images) and the following extensions:
#    image extension
#    flags extension
#    variance extension

__all__ = ["SPHERExImage", "Flag", "ensure_uncertainty_parentdata",
           "spherex_image_reader", "spherex_image_writer", "Layer"]

# Standard library imports
import logging
from copy import copy, deepcopy
from functools import reduce
from typing import Final, Optional
from dataclasses import dataclass

# Related third party imports
import numpy as np
from numpy.typing import ArrayLike
import astropy.units as u
from astropy.io import fits, registry
from astropy.table import Table
from astropy.nddata import CCDData, FlagCollection, StdDevUncertainty
from astropy.nddata import NDArithmeticMixin, NDDataArray, NDSlicingMixin
from astropy.nddata.ccddata import (_unc_name_to_cls,
                                    _generate_wcs_and_update_header,
                                    _KEEP_THESE_KEYWORDS_IN_HEADER)
from astropy.time import Time, TimeDelta
from astropy import wcs
from astropy.wcs import WCS

# Local application/library specific imports
from .spectral_wcs import SpectralWCS
from .zodi import Zodi
from .psf import PSFZoneData
from spx_data.spx_flags import Flag
from .spx_version import SSDC_PIPELINES_VERSION
from .sur import SUR
from . import spx_param
from .spx_flags import flag_description
# from spx_utils.utils import add_detcoord

ExtensionHDU = fits.hdu.base.ExtensionHDU

# define custom units that could be used in SPHERExImage
u.add_enabled_units([u.def_unit("SURstep")])

logger = logging.getLogger(__name__)

# expand default list of headers to keep
for hdr_key in ("MJD-BEG", "MJD-END", "MJD-OBS", "MJDREF", "MJD-AVG",
                "DATE-BEG", "DATE-END", "DATE-OBS", "DATE-AVG",
                "XPOSURE", "TELAPSE", "RADESYS", "TIMESYS"):
    if hdr_key not in _KEEP_THESE_KEYWORDS_IN_HEADER:
        _KEEP_THESE_KEYWORDS_IN_HEADER.append(hdr_key)


@dataclass(frozen=True)
class Layer:
    """Additional image layers"""

    # Fixed sources falling within the image.
    CATALOG: Final[str] = "CATALOG"
    # Diagnostic ratio.
    DIAGNOSTIC_RATIO: Final[str] = "DIAGNOSTIC_RATIO"
    # Diagnostic median.
    DIAGNOSTIC_MEDIAN: Final[str] = "DIAGNOSTIC_MEDIAN"
    # Diagnostic sigma.
    DIAGNOSTIC_SIGMA: Final[str] = "DIAGNOSTIC_SIGMA"
    # PSF data.
    PSF: Final[str] = "PSF"
    # Full sample data.
    RAWDATA: Final[str] = "RAWDATA"
    # Full sample data flags.
    CUBEFLGS: Final[str] = "CUBEFLGS"
    # Table with raw readout order.
    ROW_READOUT_ORDER: Final[str] = "ROW_READOUT_ORDER"
    # Solar system objects falling within the image.
    SSOCAT: Final[str] = "SSOCAT"
    # Zodi model.
    ZODI: Final[str] = "ZODI"
    # GDS data table
    GDS: Final[str] = "GDSDATA"


# extension names corresponding to additional image layers
EXTNAMES = tuple([v for k, v in Layer.__dict__.items() if not k.startswith("_")])


class SPHERExSlicingMixin(NDSlicingMixin):
    """Extension of `astropy.nddata.NDSlicingMixin` for slicing of SPHEREx
    flags and copying flag definitions.

    Notes
    -----
    Be aware that changing values of the sliced instance will change the values
    of the original.
    """
    def _slice(self, item):
        """Slices and collects the attributes and passes them back as a `dict`.

        Parameters
        ----------
        item : `tuple`
            The slice passed to ``__getitem__``.

        Returns
        -------
        kwargs : `dict`
            Contains all the attributes except layers after slicing.

        Notes
        -----
        Passes uncertainty, mask and wcs to their appropriate ``_slice_*``
        method, while ``meta``, ``unit``, and ``flag_defs`` are simply taken
        from the original. The data is assumed to be sliceable and is sliced
        directly.

        When possible the return should not be a copy of the data but
        a reference.
        """
        kwargs = super()._slice(item)
        # Handle flags for SPHEREx
        if hasattr(self, "flags"):
            kwargs["flags"] = self._slice_flags(item)
        # Attributes which are copied and not intended to be sliced
        if hasattr(self, "flag_defs"):
            kwargs["flag_defs"] = self.flag_defs

        return kwargs

    def _slice_flags(self, item):
        """Slices the flags array to match the data slice.

        Parameters
        ----------
        item : `tuple`
            The slice passed to ``__getitem__``.

        Returns
        -------
        flags : ``
            Sliced array of flags.

        Raises
        ------
        TypeError
            Raised if flags cannot be sliced.

        Notes
        -----
        See `astropy.nddata.NDSlicingMixin._slice_mask`
        """
        if self.flags is None:
            return None
        try:
            return self.flags[item]
        except TypeError:
            logger.info("flags cannot be sliced")
        return self.flags


class SPHERExArithmeticMixin(NDArithmeticMixin):
    """Extension of `astropy.nddata.NDArithmeticMixin` to preserve flags, flag
    definitions and layers during arithmetic.
    """
    def _arithmetic(self, operation, operand,
                    propagate_uncertainties=True, handle_mask=np.logical_or,
                    handle_meta=None, uncertainty_correlation=0,
                    compare_wcs="first_found", handle_flags=None,
                    handle_layers=None, **kwds):
        """Base method which calculates the result of the arithmetic operation.

        Parameters
        ----------
        operation : callable
            The operation that is performed on the `~astropy.nddata.NDData`.
            Supported are `numpy.add`, `numpy.subtract`, `numpy.multiply` and
            `numpy.true_divide`.
        operand : same type (class) as self
            see :meth:`NDArithmeticMixin.add`
        propagate_uncertainties : `bool` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`
        handle_mask : callable, ``"first_found"`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`
        handle_meta : callable, ``"first_found"`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`
        compare_wcs : callable, ``"first_found"`` or ``None``, optional
            see :meth:`NDArithmeticMixin.add`
        uncertainty_correlation : ``Number`` or `~numpy.ndarray`, optional
            see :meth:`NDArithmeticMixin.add`
        handle_flags : callable, ``"first_found"`` or ``None``, optional
            copies SPHEREx flags and flag_defs to new object
        handle_layers : callable, ``"first_found"`` or ``None``, optional
            copies SPHEREx layers to new object
        kwargs :
            Any other parameter that should be passed to the different
            :meth:`NDArithmeticMixin._arithmetic_mask` (or wcs, ...)
            methods.

        Returns
        -------
        result : ndarray or `~astropy.units.Quantity`
            The resulting data as array (in case both operands were without
            unit) or as quantity if at least one had a unit.
        kwargs : `dict`
            The kwargs should contain all the other attributes (besides data
            and unit) needed to create a new instance for the result. Creating
            the new instance is up to the calling method, for example
            :meth:`NDArithmeticMixin.add`.

        Notes
        -----
        This method determines the result of the arithmetic operation on the
        ``data`` including their units and then forwards to other methods
        to calculate the other properties for the result (like uncertainty).
        """
        # Pass to super with appropriate parameters
        result, kwargs = super()._arithmetic(
            operation=operation, operand=operand,
            propagate_uncertainties=propagate_uncertainties,
            handle_mask=handle_mask, handle_meta=handle_meta,
            uncertainty_correlation=uncertainty_correlation,
            compare_wcs=compare_wcs, **kwds)
        # Handles flags and flag_defs
        if handle_flags is None:
            kwargs["flags"] = None
            kwargs["flag_defs"] = None
        elif handle_flags in ["ff", "first_found"]:
            if self.flags is None:
                kwargs["flags"] = deepcopy(operand.flags)
            else:
                kwargs["flags"] = deepcopy(self.flags)

            if self.flag_defs is None:
                kwargs["flag_defs"] = deepcopy(operand.flag_defs)
            else:
                kwargs["flag_defs"] = deepcopy(self.flag_defs)
        else:
            raise NotImplementedError("Callables not supported for SPHEREx attribute flags or flag_defs")

        # SPHEREx specific layers
        if handle_layers is None:
            kwargs["layers"] = {}
        elif handle_layers in ["ff", "first_found"]:
            if not self.layers:
                kwargs["layers"] = deepcopy(operand.layers)
            else:
                kwargs["layers"] = deepcopy(self.layers)
        else:
            raise NotImplementedError("Callables not supported for SPHEREx attribute layers")

        # Wrap the individual results into a new instance of the same class.
        return result, kwargs


class SPHERExDataArray(NDDataArray, SPHERExArithmeticMixin,
                       SPHERExSlicingMixin):
    """Extension of `astropy.nddata.NDDataArray` for layers and flag
    definitions and to inherit SPHEREx slicing and arithmetic.

    Parameters
    ----------
    data : `~numpy.ndarray` or `~astropy.nddata.NDData`
        The actual data contained in this `~astropy.nddata.NDData` object.
        Note that this will always be copies by *reference* , so you should make copy
        the ``data`` before passing it in if that's the  desired behavior.
    uncertainty : `~astropy.nddata.NDUncertainty`, optional
        Uncertainties on the data.
    mask : array-like, optional
        Mask for the data, given as a boolean Numpy array or any object that
        can be converted to a boolean Numpy array with a shape
        matching that of the data. The values must be ``False`` where
        the data is *valid* and ``True`` when it is not (like Numpy
        masked arrays). If ``data`` is a numpy masked array, providing
        ``mask`` here will causes the mask from the masked array to be
        ignored.
    flags : array-like or `~astropy.nddata.FlagCollection`, optional
        Flags giving information about each pixel. These can be specified
        either as a Numpy array of any type (or an object which can be
        converted to a Numpy array) with a shape matching that of the
        data, or as a `~astropy.nddata.FlagCollection` instance which has a
        shape matching that of the data.
    wcs : None, optional
        WCS-object containing the world coordinate system for the data.
        .. warning::
            This is not yet defined because the discussion of how best to
            represent this class's WCS system generically is still under
            consideration. For now just leave it as None
    meta : `dict`-like object, optional
        Metadata for this object.  "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object.  e.g., creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.
    unit : `~astropy.units.UnitBase` instance or str, optional
        The units of the data.
    flag_defs : `dict`, optional
        SPHEREx flag defintions for flags

    Raises
    ------
    ValueError :
        If the `uncertainty` or `mask` inputs cannot be broadcast (e.g., match
        shape) onto ``data``.
    """

    def __init__(self, data, *args, flag_defs=None, **kwargs):
        # Remove SPHEREx specific layers from kwargs to pass to parent
        layers: dict[str, ExtensionHDU] = {}
        if "layers" in kwargs:
            layers = kwargs.pop("layers")
        super().__init__(data, *args, **kwargs)

        if isinstance(data, NDDataArray):
            if flag_defs is None:
                flag_defs = data.flag_defs
            else:
                logger.info("Overwriting SPHERExImage's "
                            "current flag_defs with "
                            "specified flag_defs")
            if data.layers is not None:
                layers = data.layers

        self._flag_defs = flag_defs
        self._layers = layers

    # Additional HDU layers, such as zodi, psf, etc.
    # A layer is lazily converted to an object when it is accessed
    @property
    def layers(self) -> dict[str, ExtensionHDU]:
        """Do not use outside this module"""
        return self._layers

    @layers.setter
    def layers(self, value: dict[str, ExtensionHDU]):
        """Do not use outside this module"""
        self._layers = value

    @property
    def flag_defs(self):
        return self._flag_defs

    @flag_defs.setter
    def flag_defs(self, value):
        self._flag_defs = value


class SPHERExImage(CCDData, SPHERExDataArray):
    """Container for SPHEREx image components such as ccd data, flags, etc.
    and provides support for reading and writing SPHERExImage from/to FITS.

    Parameters
    ----------
    data : `~astropy.nddata.NDData`-like or `numpy.ndarray`-like
        CCD data contained in this object.
        Note that the data will always be saved by *reference*, so you should
        make a copy of the ``data`` before passing it in if that's the desired
        behavior.
    flag_defs : dict-like object or None, optional
        Flag definitions, which map flag name to a bit in the flags array.
    flags : `numpy.ndarray` or None, optional
        Flags giving information about each pixel. These can be specified
        as a Numpy array of any type with a shape matching that of the
        data.
        Default is ``None``.
    wcs : `~astropy.wcs.WCS` or None, optional
        WCS-object containing the world coordinate system for the data.
        Default is ``None``.
    meta : dict-like object or None, optional
        Metadata for this object. "Metadata" here means all information that
        is included with this object but not part of any other attribute
        of this particular object, e.g. creation date, unique identifier,
        simulation parameters, exposure time, telescope name, etc.
    unit : `~astropy.units.Unit` or str, optional
        The units of the data. Default is ``None``. If the unit is ``None``
        or not otherwise specified it will raise a ``ValueError``.
    dtype_data : `~numpy.dtype` or str, optional
        Desirable data type for data and uncertainty.
        Data and uncertainty will saved with this type.
        Default is ``"float32"``.
    dtype_flags : `~numpy.dtype` or str, optional
        Desirable data type for flags.
        Flags will be saved with this type.
        Default is ``"int32"``.

    Methods
    -------
    read(*args, **kwargs)
        Class method to create an SPHERExImage instance based on a ``FITS``
        file. This method uses :func:`spherex_fits_reader` with the provided
        parameters.
    write(*args, **kwargs)
        Writes the contents of the SPHERExImage instance into a new ``FITS``
        file. This method uses :func:`spherex_fits_writer` with the provided
        parameters.
    """
    def __init__(self, data, *args, flag_header=None, **kwargs):
        if "flags" in kwargs and isinstance(kwargs["flags"], FlagCollection):
            raise NotImplementedError("Flag collection is not supported")

        # store metadata in ~astropy.io.fits.Header object by default
        # it facilitates setting/getting meta with comments
        if "meta" not in kwargs and "header" not in kwargs and not isinstance(data, SPHERExImage):
            kwargs["meta"] = fits.Header()
            if "unit" in kwargs and kwargs["unit"] == u.dimensionless_unscaled:
                # BUNIT must be set is the data are dimensionless
                kwargs["meta"]["BUNIT"] = ("", "Unit of pixel value (dimensionless)")

        # default data types
        if "dtype_data" in kwargs:
            self.dtype_data = kwargs.pop("dtype_data")
        else:
            self.dtype_data = np.float32
        if "dtype_flags" in kwargs:
            self.dtype_flags = kwargs.pop("dtype_flags")
        else:
            self.dtype_flags = np.int32

        # Initialize with the parent...
        super().__init__(data, *args, **kwargs)
        self._spectral_wcs = None
        # Store flag header if provided
        self._flag_header = flag_header if flag_header is not None else fits.Header()
        # Support for copy
        if isinstance(data, SPHERExImage):
            self._layers = copy(data.layers)
            self._flag_defs = copy(data.flag_defs)
            self._flags = copy(data.flags)
            self._spectral_wcs = data._spectral_wcs
            self.dtype_data = data.dtype_data
            self.dtype_flags = data.dtype_flags
            self.flag_header = copy(data.flag_header)

    @property
    def flag_header(self):
        return self._flag_header

    @flag_header.setter
    def flag_header(self, value):
        """Set the flag header."""
        if not isinstance(value, fits.Header):
            raise TypeError("flag_header must be an instance of astropy.io.fits.Header")
        self._flag_header = value

    @property
    def catalog(self) -> Optional[Table]:
        if Layer.CATALOG in self._layers:
            return Table.read(self._layers[Layer.CATALOG])

    @catalog.setter
    def catalog(self, value: Table):
        hdu = fits.BinTableHDU(data=value, name=Layer.CATALOG)
        self._layers[Layer.CATALOG] = hdu

    @property
    def diagnostic_ratio(self) -> Optional[ArrayLike]:
        if Layer.DIAGNOSTIC_RATIO in self._layers:
            hdu = self._layers[Layer.DIAGNOSTIC_RATIO]
            if isinstance(hdu, fits.ImageHDU):
                return hdu.data
            else:
                raise ValueError("Unexpected hdu type for image layer")

    @diagnostic_ratio.setter
    def diagnostic_ratio(self, value: ArrayLike):
        hdu = fits.ImageHDU(data=value, name=Layer.DIAGNOSTIC_RATIO)
        self._layers[Layer.DIAGNOSTIC_RATIO] = hdu

    @property
    def diagnostic_median(self) -> Optional[ArrayLike]:
        if Layer.DIAGNOSTIC_MEDIAN in self._layers:
            hdu = self._layers[Layer.DIAGNOSTIC_MEDIAN]
            if isinstance(hdu, fits.ImageHDU):
                return hdu.data
            else:
                raise ValueError("Unexpected hdu type for image layer")

    @diagnostic_median.setter
    def diagnostic_median(self, value: ArrayLike):
        hdu = fits.ImageHDU(data=value, name=Layer.DIAGNOSTIC_MEDIAN)
        hdu.header["BUNIT"] = "electron"
        self._layers[Layer.DIAGNOSTIC_MEDIAN] = hdu

    @property
    def diagnostic_sigma(self) -> Optional[ArrayLike]:
        if Layer.DIAGNOSTIC_SIGMA in self._layers:
            hdu = self._layers[Layer.DIAGNOSTIC_SIGMA]
            if isinstance(hdu, fits.ImageHDU):
                return hdu.data
            else:
                raise ValueError("Unexpected hdu type for image layer")

    @diagnostic_sigma.setter
    def diagnostic_sigma(self, value: ArrayLike):
        hdu = fits.ImageHDU(data=value, name=Layer.DIAGNOSTIC_SIGMA)
        hdu.header["BUNIT"] = "electron"
        self._layers[Layer.DIAGNOSTIC_SIGMA] = hdu

    @property
    def psfcube(self) -> Optional[PSFZoneData]:
        if Layer.PSF in self._layers:
            return PSFZoneData.from_hdu(self._layers[Layer.PSF])

    @psfcube.setter
    def psfcube(self, value: PSFZoneData):
        self._layers[Layer.PSF] = value.to_hdu(Layer.PSF)

    @property
    def spectral_wcs(self) -> Optional[SpectralWCS]:
        return self._spectral_wcs

    @spectral_wcs.setter
    def spectral_wcs(self, value: SpectralWCS):
        self._spectral_wcs = value

    @property
    def alt_wcs(self) -> Optional[dict[str, WCS]]:
        """Property getter for alternative WCS.

        Returns
        -------
        `dict[str, WCS]`
            Dictionary of WCS objects, with the keys being the WCS suffix.
        """
        # Fix to not parse spectral WCS as alternative header.
        # Make a copy of the header to modify
        modified_header = deepcopy(self.meta)
        _remove_spectral_wcs_keys(modified_header)
        # Now do find all wcs.
        wcs_list = wcs.find_all_wcs(modified_header.tostring(padding=False))
        # wcs_list = wcs.find_all_wcs(self.meta.tostring(padding=False))
        alt_wcs_dict = {}

        # Loop through each WCS object.
        for wcs_obj in wcs_list:
            if wcs_obj.wcs.alt == " ":
                continue
            suffix = wcs_obj.wcs.alt
            alt_wcs_dict[suffix] = wcs_obj

        return alt_wcs_dict if alt_wcs_dict else None

    @alt_wcs.setter
    def alt_wcs(self, alt_wcs_dict: dict[str, WCS]):
        """Property setter for alternative WCS.

        Parameters
        ----------
        `dict[str, WCS]`
            Dictionary of WCS objects, with the keys being the WCS suffix.
        """
        for suffix, wcsa in alt_wcs_dict.items():
            wcsa_header = wcsa.to_header()
            for key, val in wcsa_header.items():
                if key == "W":
                    continue
                self.meta[key] = val

    @property
    def sur(self) -> Optional[SUR]:
        if Layer.RAWDATA in self._layers:
            return SUR.from_hdu(self._layers[Layer.RAWDATA])

    @sur.setter
    def sur(self, value: SUR):
        self._layers[Layer.RAWDATA] = value.to_hdu(Layer.RAWDATA)

    @property
    def ssocat(self) -> Optional[Table]:
        if Layer.SSOCAT in self._layers:
            return Table.read(self._layers[Layer.SSOCAT])

    @ssocat.setter
    def ssocat(self, value: Table):
        hdu = fits.BinTableHDU(data=value, name=Layer.SSOCAT)
        self._layers[Layer.SSOCAT] = hdu

    @property
    def zodi(self) -> Optional[Zodi]:
        if Layer.ZODI in self._layers:
            return Zodi.from_hdu(self._layers[Layer.ZODI])

    @zodi.setter
    def zodi(self, value: Zodi):
        self._layers[Layer.ZODI] = value.to_hdu(Layer.ZODI)

    @property
    def gds(self) -> Optional[Table]:
        if Layer.GDS in self._layers:
            return Table.read(self._layers[Layer.GDS])

    @gds.setter
    def gds(self, value: Table):
        hdu = fits.BinTableHDU(data=value, name=Layer.GDS)
        self._layers[Layer.GDS] = hdu

    def ensure_layers(self, layers: list[str]):
        """Make sure the required layers and only them
        are present in the image.

        Parameters
        ----------
        layers : `list` [`str`]
            List of image layers
        """
        existing_layers = self._layers.keys()
        # verify that requested layers are part of the image
        for name in layers:
            if name not in EXTNAMES:
                raise ValueError(f"Unknown layer {name}")
            elif name not in existing_layers:
                raise RuntimeError(f"Layer {name} is not set")
        # remove unnecessary layers
        to_remove = set(existing_layers) - set(layers)
        for name in to_remove:
            self._layers.pop(name)

    def getArrayId(self) -> int:
        """Get SPHEREx detector array id.

        Returns
        -------
        array_id : `int`
            SPHEREx detector array id from 1 to 6
        """
        if "ARRAY" in self.header:
            return self.header["ARRAY"]
        elif "DETECTOR" in self.header:
            return self.header["DETECTOR"]
        else:
            raise RuntimeError("SPHEREx Detector ID is not found in the header.")

    def setArrayId(self, array_id: int):
        """Set SPHEREx detector array id.

        Parameters
        ----------
        array_id : `int`
            SPHEREx detector array id from 1 to 6
        """
        self.header["ARRAY"] = (array_id, "SPHEREx detector array")

    def setExposureStartTime(self, time: Time):
        """Set exposure start time in image meta.

        Parameters
        ----------
        time : `astropy.time.Time`
            Exposure start time
        """
        self.meta["MJD-OBS"] = time.mjd
        self.meta["MJD-BEG"] = time.mjd

    def getExposureStartTime(self) -> Time:
        """Get exposure start as an astropy Time.

        Returns
        -------
        exposure_start_time : `astropy.time.Time`
        """
        if "MJD-BEG" in self.meta:
            return Time(self.meta["MJD-BEG"], format="mjd", scale="utc")
        elif "EXPSTART" in self.meta:
            logger.warning("EXPSTART is deprecated, use MJD-BEG instead.")
            return Time(self.meta["EXPSTART"], format="mjd", scale="utc")
        elif "JD" in self.meta:
            return Time(self.meta["JD"], format="jd", scale="utc")
        else:
            raise NotImplementedError("Start exposure time is not found.")

    def getExposureEndTime(self) -> Time:
        """Get exposure end time as an astropy Time.

        Returns
        -------
        exposure_end_time : `astropy.time.Time`
        """
        if "MJD-END" in self.meta:
            return Time(self.meta["MJD-END"], format="mjd", scale="utc")
        elif "EXPSTOP" in self.meta:
            logger.warning("EXPSTOP is deprecated, use MJD-END instead.")
            return Time(self.meta["EXPSTOP"], format="mjd", scale="utc")
        else:
            return self.getExposureStartTime() + self.getExposureDuration()

    def getFlagsAsMask(self, flag_val: int, check_all=False):
        """Get flag mask by providing flag value.
        Use ``|`` operator to combine flags.
        Example: Flag.COSMICRAY | Flag.SATURATED

        Parameters
        ----------
        flag_val : `int`
            Combined flags value.
        check_all : `bool`
            If set, the returned mask reflects the presence of *all* flags.
            Otherwise, `True` mask value means at least one of the flags is
            set.
            Default is ``False``.

        Returns
        -------
        mask : `numpy.ndarray`
            Mask for the data, given as a boolean Numpy array with a shape
            matching that of the data. The values are `False` where
            the data is *not flagged* and `True` when it is.
        """
        if self.flags is None:
            raise RuntimeError("flags are not set")
        if check_all:
            ret = np.bitwise_and(self.flags, flag_val) == flag_val
        else:
            ret = np.bitwise_and(self.flags, flag_val) > 0
        return ret

    def setFreeFormatHeaderCard(self, name: str, value: float, digits: int, comment: str) -> None:
        """Set free format header card, which minimizes header line size

        Parameters
        ----------
        name :
            Header name
        value :
            Float value
        digits :
            Number of digits after the decimal point
        comment :
            Header comment

        Returns
        -------
        card : `fits.Card`
        """
        if len(name) > 8:
            raise ValueError(f"FITS keyword '{name}' exceeds 8 characters")

        fmt = f"{{0:.{digits}f}}"
        val_str = fmt.format(value)

        card_str = f"{name: <8}= {val_str} / {comment}"
        if len(card_str) > 80:
            raise ValueError(f"FITS header card exceeds 80 characters: {len(card_str)}")

        card = fits.Card.fromstring(card_str)
        return self.header.append(card)

    def setFlagsFromMask(self, mask: np.ndarray, flag_val: int, area: tuple[slice, slice] = None):
        """Set flags defined by the ``flag_val`` for pixels defined by the
        ``mask``.

        Parameters
        ----------
        mask : `numpy.ndarray`
            Mask for the data, given as a boolean Numpy array with a shape
            matching that of the data. The values are `False` where the flags
            value is *not changing* and `True` where the requested flags
            should be set.
        flag_val : `int`
            Combined flags value. For example: Flag.HOT | Flag.NONFUNC
        area : `tuple[slice, slice]`

        """
        if mask.dtype != np.bool_:
            # mask must be boolean numpy array
            mask = mask > 0
            logger.warning(f"Unexpected mask type {mask.dtype} instead of {np.bool_} for {flag_val}")

        # decompose flag value into individual flags
        decomposed_flags = [flg for flg in Flag if flg & flag_val]

        # make sure flags are in FLAG_DEFS
        combined_val = reduce(lambda x, y: x | y, decomposed_flags)
        if combined_val != flag_val:
            raise ValueError(f"Undefined flag bits: {bin(flag_val ^ combined_val)}")

        # make sure all flags used are added to flag_def
        if len(decomposed_flags) > 0:
            if self.flag_defs is None:
                flag_defs = {}
            else:
                flag_defs = deepcopy(self.flag_defs)
            for flag in decomposed_flags:
                # we are only interested in pure flags
                if np.ceil(np.log2(flag.value)) != np.floor(np.log2(flag.value)):
                    continue
                # check for conflicts with existing flag definitions
                if flag.name in flag_defs.keys() and self.flag_defs[flag.name] != np.log2(flag.value):
                    raise RuntimeError(f"Conflicting flag definitions for {flag.name}:"
                                       f" {bin(1 << self.flag_defs[flag.name])} before"
                                       f" {bin(flag_val)} now.")
                flag_defs[flag.name] = int(np.log2(flag.value))
            # update flag definitions
            self.flag_defs = flag_defs

        if self.flags is None:
            self.flags = np.zeros(self.data.shape, dtype=self.dtype_flags)

        if area is None:
            area = (slice(None), slice(None))

        if self.flags[area].shape != mask.shape:
            raise ValueError(f"Error setting flags: mask shape {mask.shape}"
                             f" is not matching flags shape {self.flags.shape}")

        self.flags[area] |= (flag_val * mask.astype(self.dtype_flags, copy=False))

    def setExposureDuration(self, exp_duration: TimeDelta):
        """Set exposure start time in image meta.

        Parameters
        ----------
        exp_duration : `astropy.time.TimeDelta`
            Exposure duration
        """
        self.meta["TINT"] = exp_duration.sec

    def getExposureDuration(self) -> TimeDelta:
        """Get exposure duration as an astropy Time.

        Returns
        -------
        exposure_duration : `astropy.time.TimeDelta`
            Total time of taken exposure.
        """
        # exposure duration is named in header TINT.
        if "TINT" in self.meta:
            return TimeDelta(self.meta["TINT"] * u.s)
        else:
            raise NotImplementedError("Exposure duration is not available.")

    def add_provenance(self, add_detcoord_flag=True):
        """Add standard provenance metadata to the image header.

        Parameters
        ----------
        add_detcoord_flag : bool, optional
            If True, calls `add_detcoord(self)` to add detector coordinate WCS.
        """
        now = Time.now()
        hdr = self.meta  # or self.header; they are aliases

        hdr["VERSION"] = (SSDC_PIPELINES_VERSION, "SSDC Pipelines software version")
        hdr["DATE"] = (now.utc.isot, "HDU creation time (UTC)")
        # TODO check keyword and comments
        hdr["ORIGIN"] = ("SSDC", "Organization responsible for data")
        hdr["TELESCOP"] = ("SPHEREx", "Telescope name")
        hdr["INSTRUME"] = ("SPHEREx", "Instrument Name")


def _get_flag_defs(header: fits.Header):
    """Get flag definition dictionary from the the header
    Parameters
    ----------
    header : dict-like object
    Returns
    -------
    flag_defs : dict-like object
    """
    flag_defs = {}
    for key, val in header.items():
        if key.startswith("MP_"):
            try:
                intval = int(val)
                if intval > 31:
                    raise ValueError(f"bit value {intval} for flag {key}"
                                     " is too large")
                flagkey = key[3:]
                flag_defs[flagkey] = intval
            except ValueError:
                pass
    if len(flag_defs) > 0:
        return flag_defs


def _add_flag_defs(spherex_image: SPHERExImage, header: fits.Header) -> None:
    """Add image flag definitions to the header
    Parameters
    ----------
    spherex_image : `~spherex.data.SPHERExImage`
    header : dict-like object
    """

    # check for None or empty dictionary
    if not spherex_image.flag_defs:
        return

    # convention understood by Firefly:
    # EXTTYPE=MASK and flag bit values in keywords starting with "MP_"
    header["EXTTYPE"] = ("MASK", "Extension can be used as a mask overlay")
    # sort flags numerically by value
    for key, val in sorted(spherex_image.flag_defs.items(), key=lambda x: x[1]):
        prefix = "MP_" if len(key) < 6 else "HIERARCH MP_"
        header[f"{prefix}{key.upper()}"] = val
        flag_name = key.upper()
        header_key = f"{prefix}{flag_name}"
        # Get flag description
        flag_enum = getattr(Flag, flag_name, None)
        description = flag_description(flag_enum) if flag_enum else "No description"
        # Add to header with comment
        header[header_key] = (val, description)


def _remove_spectral_wcs_keys(hdr):
    # Remove spectral WCS from header.
    for key in SpectralWCS.getSpectralWCSKeywords():
        hdr.remove(key, ignore_missing=True, remove_all=True)


def ensure_uncertainty_parentdata(spx_image: SPHERExImage) -> None:
    # Workaround for None in _parent_nddata field of uncertainty.
    # _parent_nddata is set in uncertainty setter of CCDData.
    # It is a weak reference and might be lost when the original
    # image is garbage collected.
    spx_image.uncertainty = spx_image.uncertainty


def spherex_image_reader(filename, hdu=0, unit=None,
                         hdu_uncertainty="VARIANCE", hdu_mask="MASK",
                         hdu_flags="FLAGS", hdu_layers=EXTNAMES,
                         key_uncertainty_type="UTYPE",
                         dtype_data=np.float32,
                         dtype_flags=np.int32,
                         **kwd) -> SPHERExImage:
    """
    Generate a SPHERExImage object from a FITS file.

    When flags and variance are present, they are expected to be in
    the second and the third extensions.

    This function is almost identical to Astropy 5.1
    `astropy.nddata.ccddata.fits_ccddata_reader` however it was necessary to
    copy that code because astropy does not support flags nor layers.
    Without this modification, the file would need to be read twice rather than
    once.

    Note, that we do not usually use MASK layer in SPHEREx products.
    Bitmask flags are stored in FLAGS HDU.

    Parameters
    ----------
    filename : `str` or `~pathlib.Path`
        Name of fits file.
    hdu : `str` or `int`, optional
        FITS extension from which the data should be initialized. If zero
        and no data in the primary extension, it will search for the first
        extension with data. The header will be added to the primary header.
        Default is ``0``.
    unit : `~astropy.units.Unit`, optional
        Units of the image data. If this argument is provided and there is a
        unit for the image in the FITS header (the keyword ``BUNIT`` is used
        as the unit, if present), this argument is used for the unit.
        Default is ``None``.
    hdu_uncertainty : `str` or `int` or `None`, optional
        FITS extension from which the uncertainty should be initialized. If the
        extension does not exist the uncertainty of the SPHERExImage is
        ``None``.
        Default is ``VARIANCE``.
    hdu_mask : `str` or `int` or `None`, optional
        FITS extension from which the mask should be initialized. If the
        extension does not exist the mask of the SPHERExImage is ``None``.
        Default is ``MASK``. (Not used in SPHEREx.)
    hdu_flags : `str` or `int` or `None`, optional
        FITS extension from which the flags should be initialized. If the
        extension does not exist the flags of the SPHERExImage is ``None``.
        Default is ``FLAGS``.
    hdu_layers : `list` of `str` or `None`, optional
        FITS extensions from which the layers should be initialized. If the
        extensions do not exist the corresponding layer is ``None``
    key_uncertainty_type : `str`, optional
        The header key name where the class name of the uncertainty  is stored
        in the hdu of the uncertainty (if any).
        Default is ``UTYPE``.
    dtype_data : `str` or `~numpy.dtype`, optional
        Numpy data type for data and uncertainty.
        Default is ``float32``.
    dtype_flags : `str` or `~numpy.dtype`, optional
        Numpy data type for flags.
        Default is ``int32``.
    kwd :
        Any additional keyword parameters are passed through to the FITS reader
        in :mod:`astropy.io.fits`; see Notes for additional discussion.

    Returns
    -------
    spherex_image : `~spx_data.SPHERExImage`

    Notes
    -----
    FITS files that contained scaled data (e.g. unsigned integer images) will
    be scaled and the keywords used to manage scaled data in
    :mod:`astropy.io.fits` are disabled.
    """
    unsupport_open_keywords = {
        "do_not_scale_image_data": "Image data must be scaled.",
        "scale_back": "Scale information is not preserved."
    }
    for key, msg in unsupport_open_keywords.items():
        if key in kwd:
            prefix = f"unsupported keyword: {key}."
            raise TypeError(" ".join([prefix, msg]))

    with fits.open(filename, **kwd) as hdus:
        hdr = hdus[hdu].header

        if hdu_uncertainty is not None and hdu_uncertainty in hdus:
            unc_hdu = hdus[hdu_uncertainty]
            stored_unc_name = unc_hdu.header.get(key_uncertainty_type, "None")
            # For compatibility reasons the default is standard deviation
            # uncertainty because files could have been created before the
            # uncertainty type was stored in the header.
            unc_type = _unc_name_to_cls.get(stored_unc_name, StdDevUncertainty)
            uncertainty = unc_type(unc_hdu.data.astype(dtype_data, copy=False))
        else:
            uncertainty = None

        if hdu_mask is not None and hdu_mask in hdus:
            # Mask is saved as uint but we want it to be boolean.
            mask = hdus[hdu_mask].data.astype(np.bool_)
        else:
            mask = None

        # search for the first instance with data if
        # the primary header is empty.
        if hdu == 0 and hdus[hdu].data is None:
            # The snippet below is workaround for  astropy/astropy#17810,
            # It can be removed when astropy 7.1 is released.
            for an_hdu in hdus:
                if isinstance(an_hdu, fits.ImageHDU) and an_hdu.data is not None:
                    hdu = an_hdu
            # for i in range(len(hdus)):
            #     if (hdus.info(hdu)[i][3] == "ImageHDU"
            #             and hdus.fileinfo(i)["datSpan"] > 0):
            #         hdu = i
                    comb_hdr = hdus[hdu].header.copy()
                    # Add header values from the primary header that aren't
                    # present in the extension header.
                    comb_hdr.extend(hdr, unique=True)
                    hdr = comb_hdr
                    logger.info(f"first HDU with data is extension {hdu}.")
                    break

        #############################
        # Begin SPHEREx modifications
        #############################

        if SpectralWCS.hasSpectralWCS(hdus):
            spectral_wcs = SpectralWCS(hdus, SpectralWCS.WCS_KEY)
            # spectral WCS requires table, and leaving its keys
            # in the header creates inconsistency
            _remove_spectral_wcs_keys(hdr)
        else:
            spectral_wcs = None

        # Build flags and flag definitions
        flags = None
        flag_defs = None
        flag_header = fits.Header()  # Default empty header

        if hdu_flags is not None and hdu_flags in hdus:
            flags_hdu = hdus[hdu_flags]
            flags = flags_hdu.data.astype(dtype_flags, copy=False)
            flag_header = flags_hdu.header  # Store the header
            flag_defs = _get_flag_defs(flag_header)

        # Handle SPHEREx layers
        layers = {}
        if hdu_layers is not None:
            for name in hdu_layers:
                if name in hdus:
                    layer = hdus[name]
                    # Make sure hdu is accessible after file is closed
                    layers[name] = type(layer)(layer.data, layer.header)

        ###########################
        # End SPHEREx modifications
        ###########################

        if "bunit" in hdr:
            fits_unit_string = hdr["bunit"]
            # patch to handle FITS files using ADU for the unit instead of the
            # standard version of "adu"
            if fits_unit_string.strip().lower() == "adu":
                fits_unit_string = fits_unit_string.lower()
        else:
            fits_unit_string = None

        if fits_unit_string:
            if unit is None:
                # Convert the BUNIT header keyword to a unit and if that's not
                # possible raise a meaningful error message.
                try:
                    kifus = CCDData.known_invalid_fits_unit_strings
                    if fits_unit_string in kifus:
                        fits_unit_string = kifus[fits_unit_string]
                    fits_unit_string = u.Unit(fits_unit_string)
                except ValueError:
                    raise ValueError(
                        "The Header value for the key BUNIT ({}) cannot be "
                        "interpreted as valid unit. To successfully read the "
                        "file as CCDData you can pass in a valid `unit` "
                        "argument explicitly or change the header of the FITS "
                        "file before reading it."
                        .format(fits_unit_string))
            else:
                logger.info("using the unit {} passed to the FITS reader instead "
                            "of the unit {} in the FITS file."
                            .format(unit, fits_unit_string))

        use_unit = unit or fits_unit_string or u.dimensionless_unscaled

        # this used to remove MJD-BEG, MJD-END, DATE-BEG, and DATE-END
        # extended _KEEP_THESE_KEYWORDS_IN_HEADER to keep these keywords
        hdr, wcs = _generate_wcs_and_update_header(hdr)

        spherex_image = SPHERExImage(hdus[hdu].data.astype(dtype_data),
                                     uncertainty=uncertainty,
                                     mask=mask, wcs=wcs, meta=hdr,
                                     unit=use_unit, flags=flags,
                                     flag_defs=flag_defs, layers=layers,
                                     flag_header=flag_header,
                                     dtype_data=dtype_data, dtype_flags=dtype_flags)
        if spectral_wcs is not None:
            spherex_image.spectral_wcs = spectral_wcs

    return spherex_image


def spherex_image_writer(spherex_image: SPHERExImage, fileobj, extname="IMAGE",
                         hdu_mask="MASK", hdu_uncertainty="VARIANCE", hdu_flags="FLAGS",
                         wcs_relax=True, key_uncertainty_type="UTYPE",
                         add_version=True, **kwd):
    """Write `~spx_data.SPHERExImage` to a file.

    We do not usually use MASK layer in SPHEREx products.
    Bitmask flags are stored in FLAGS HDU.

    Parameters
    ----------
    spherex_image : `~spx_data.SPHERExImage`
    fileobj : file-like object or filename
    hdu_mask, hdu_uncertainty, hdu_flags : `str` or `None`, optional
        If it is a string append this attribute to the HDUList as
        `~astropy.io.fits.ImageHDU` with the string as extension name.
        Default is ``"MASK"`` for mask, ``"VARIANCE"`` for uncertainty and
        ``"FLAGS"`` for flags
    wcs_relax : `bool`
        Value of the ``relax`` parameter to use in converting the WCS to a
        FITS header using `~astropy.wcs.WCS.to_header`. The common
        ``CTYPE`` ``RA---TAN-SIP`` and ``DEC--TAN-SIP`` requires
        ``relax=True`` for the ``-SIP`` part of the ``CTYPE`` to be
        preserved.
    key_uncertainty_type : `str`, optional
        The header key name for the class name of the uncertainty (if any)
        that is used to store the uncertainty type in the uncertainty hdu.
        Default is ``UTYPE``.
    add_version : `bool`
        If True, adds VERSION header with software version.
    kwd : `dict`
        Keyword parameters passed to :meth:`~astropy.io.fits.HDUList.writeto` method.
    Returns
    -------
    """

    # ensure data types
    ensure_dtype(spherex_image)

    # create primary hdu
    # minimum header with EXTEND will be provided if header is None
    primary_header = None
    if add_version:
        # VERSION header is populated in the PrimaryHDU header on write
        # Remove previous VERSION to prevent it getting into the IMAGE header
        spherex_image.header.remove("VERSION", ignore_missing=True)
        # Add software version to the PrimaryHDU header
        primary_header = fits.Header()
        primary_header["VERSION"] = (SSDC_PIPELINES_VERSION, "SSDC Pipelines software version")
    primary_hdu = fits.PrimaryHDU(data=None, header=primary_header)

    # to_hdu does not support flags at the moment
    # to_hdu puts image data into PrimaryHDU
    hdulist = spherex_image.to_hdu(hdu_mask=hdu_mask,
                                   hdu_uncertainty=hdu_uncertainty,
                                   wcs_relax=wcs_relax,
                                   key_uncertainty_type=key_uncertainty_type)
    if spherex_image.uncertainty and spherex_image.uncertainty.unit:
        # populate missing unit in the extension header
        variance_hdu = hdulist[hdu_uncertainty]
        variance_hdu.header["BUNIT"] = spx_param.format_unit(spherex_image.uncertainty.unit)

    # insert primary hdu
    hdulist.insert(0, primary_hdu)
    image_hdu = hdulist[1]
    image_hdu.name = extname

    # add flags hdu - 2nd extension after the image data
    if hdu_flags and spherex_image.flags is not None:
        # For Firefly to relate to flag mask.
        image_hdu.header["EXTTYPE"] = ("IMAGE", "Extension supports mask overlay")

        # hdr_flags = fits.Header()
        hdr_flags = deepcopy(spherex_image.flag_header)
        _add_flag_defs(spherex_image, hdr_flags)

        hdu = fits.ImageHDU(spherex_image.flags.data, hdr_flags,
                            name=hdu_flags)
        hdulist.insert(2, hdu)

    for name, layer in spherex_image.layers.items():
        hdulist.append(layer)

    # Making combined WCS header, starting with primary WCS header if available.
    wcs_header = spherex_image.wcs.to_header(relax=wcs_relax) if spherex_image.wcs else fits.Header()

    # Add alternative WCS to the combined header if available.
    alt_wcs = spherex_image.alt_wcs
    if alt_wcs is not None:
        for suffix, wcsa in alt_wcs.items():
            alt_wcs_hdr = wcsa.to_header(relax=wcs_relax, key=suffix)
            # update image header
            # make sure alternative WCS headers are added after the main WCS
            for key in alt_wcs_hdr:
                image_hdu.header.remove(key, ignore_missing=True, remove_all=True)
            image_hdu.header.extend(alt_wcs_hdr, useblanks=False, update=True)
            # update wcs header
            wcs_header.extend(alt_wcs_hdr, useblanks=False, update=True)

    # Add spectral WCS to the combined header if available.
    spectral_wcs = spherex_image.spectral_wcs
    if spectral_wcs is not None:
        for hdu in hdulist:
            if hdu.name == "IMAGE":
                spectral_wcs.extendImageHeader(hdu.header)
        spectral_wcs.extendImageHeader(wcs_header)

    # Update the relevant extensions with the combined WCS header.
    for hdu in hdulist:
        if hdu.name in [hdu_uncertainty, hdu_flags, "ZODI"]:
            hdu.header.extend(wcs_header, useblanks=False, update=True)

    # Add the spectral WCS coordinate lookup table as an extension if available.
    if spectral_wcs is not None:
        hdulist.append(spherex_image.spectral_wcs.tab)

    hdulist.writeto(fileobj, **kwd)


def _convert_dtype(array: np.ndarray, dtype_desired: np.dtype, name: str) -> np.ndarray:
    """Convert the array to the desired data type if needed.

    Logs a warning if the array data type is different from the desired data type.

    Parameters
    ----------
    array : `~numpy.ndarray`
        The array to be converted to the desired data type.
    dtype_desired : `~numpy.dtype`
        The desired data type for the array.
    name : `str`
        The name of the data array, used for logging purposes.

    Returns
    -------
    array : `~numpy.ndarray`
        The array with the desired data type.
    """
    dtype_array = array.dtype
    if dtype_array != dtype_desired:
        logger.warning(f"Inadequate dtype of {name} array: "
                       f"{dtype_array} instead of {dtype_desired}.")
        array = array.astype(dtype_desired, copy=False)
    return array


def ensure_dtype(spherex_image: SPHERExImage) -> None:
    """Convert data, uncertainty, and flags arrays in a `~spx_data.SPHERExImage` to the
    desired data types.

    Parameters
    ----------
    spherex_image : `~spx_data.SPHERExImage`
        The SPHERExImage object containing the data, uncertainty, and flags
        arrays to be converted to the desired data types.
    """
    dtype_desired = spherex_image.dtype_data
    spherex_image.data = _convert_dtype(spherex_image.data, dtype_desired, "data")

    if spherex_image.uncertainty is not None:
        spherex_image.uncertainty.array = _convert_dtype(spherex_image.uncertainty.array, dtype_desired,
                                                         "uncertainty")

    if spherex_image.flags is not None:
        dtype_desired = spherex_image.dtype_flags
        spherex_image.flags = _convert_dtype(spherex_image.flags, dtype_desired, "flags")


# Register read/write methods for SPHERExImage
with registry.delay_doc_updates(SPHERExImage):
    registry.register_reader(data_format="fits", data_class=SPHERExImage,
                             function=spherex_image_reader)
    registry.register_writer(data_format="fits", data_class=SPHERExImage,
                             function=spherex_image_writer)
    registry.register_identifier(data_format="fits", data_class=SPHERExImage,
                                 identifier=fits.connect.is_fits)
