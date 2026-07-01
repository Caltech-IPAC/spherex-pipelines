__all__ = ["add_detcoord", "is_device_detcoord", "get_data_file", "get_overflow_threshold"]

# Standard library imports
from pathlib import Path
import importlib.resources as resources

# Related third party imports
import numpy as np
from astropy.io import fits

# Local application/library specific imports
from spx_data import ReadoutParameters
from spx_data import SPHERExImage, PSFZoneData
from astropy.io.fits import HDUList


def add_detcoord(obj, device_coord: bool = False):
    """
    Add DETCOORD keyword to headers of SPHERExImage, HDUList (e.g., spectral WCS),
    or PSFZoneData.

    Parameters
    ----------
    obj : `SPHERExImage`, `HDUList`, or `PSFZoneData`
        The object whose header(s) will be updated.
    device_coord : `bool`
        Indicator for "sky" or "device" coordinate system.
        Default is False, (sky coordinate).

    Returns
    -------
    obj : same as input
        Updated object with DETCOORD keyword in header(s).
    """
    if device_coord:
        value = "device"
    else:
        value = "sky"

    comment = ("MWIR pixels flipped on X to match SWIR view" if value == "sky"
               else "pixel coordinate system matching instrument")

    if isinstance(obj, SPHERExImage):
        obj.header["DETCOORD"] = value
        obj.header.comments["DETCOORD"] = comment
    elif isinstance(obj, HDUList):
        for hdu in obj:
            if isinstance(hdu, fits.ImageHDU) or isinstance(hdu, fits.PrimaryHDU):
                hdu.header["DETCOORD"] = (value, comment)
    elif isinstance(obj, PSFZoneData):
        obj.header["DETCOORD"] = value
        obj.header.comments["DETCOORD"] = comment
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")

    return obj


def is_device_detcoord(obj) -> bool:
    """Determine whether the input object is in 'device' detector coordinate system.

    This function checks the value of the DETCOORD header keyword and returns True
    if it equals 'device'. If the keyword is missing, it assumes the default value
    is 'device'.

    Parameters
    ----------
    obj : SPHERExImage, HDUList, or PSFZoneData
        An object containing a FITS header to inspect. Supported types are:
        - SPHERExImage (with a .header attribute)
        - astropy.io.fits.HDUList (the first image HDU is checked)
        - PSFZoneData (with a .header attribute)

    Returns
    -------
    bool
        True if DETCOORD is 'device' or missing (assumed to be 'device'),
        False otherwise.

    Raises
    ------
    TypeError
        If the input object is not one of the supported types.
    """
    device_detcoord_value = "device"

    def get_detcoord(header):
        return header.get("DETCOORD", device_detcoord_value) == device_detcoord_value

    if isinstance(obj, SPHERExImage):
        return get_detcoord(obj.header)
    elif isinstance(obj, HDUList):
        for hdu in obj:
            if isinstance(hdu, (fits.ImageHDU, fits.PrimaryHDU)):
                return get_detcoord(hdu.header)
    elif isinstance(obj, PSFZoneData):
        return get_detcoord(obj.header)
    else:
        raise TypeError(f"Unsupported object type: {type(obj)}")


def get_data_file(rel_filename: str, package: str = "spx_tests") -> Path:
    """Returns path to package file.

    Parameters
    ----------
    rel_filename : `str`
        The filename.
    package : `str`, optional
        The package to look in, by default "spx_tests".

    Returns
    -------
    `~pathlib.Path`
        The path to the package file.

    Raises
    ------
    FileNotFoundError
        Raised if the file is not found.
    """
    pkg = resources.files(package)
    resource = pkg / rel_filename
    if not resource.is_file():
        raise FileNotFoundError(f"{rel_filename} not found.")
    else:
        with resources.as_file(resource) as file_path:
            return file_path


def gains_for_image(gains: list[float], rp: ReadoutParameters, extra_refs: bool = False) -> np.ndarray:
    """Generates a sample map of gain factors for readout parameters.

    Parameters
    ----------
    gains : `list` [`float`]
        Gain factors (per channel) for a detector (i).
    rp : `~spx_data.ReadoutParameters`
        Readout parameters from SPHEREx image.
    extra_refs : `bool` (Optional)
        Include reference pixels with extra reads in output.

    Returns
    -------
    gain_map : `~numpy.ndarray`
        Gain factors in shape of level2 image.

    """
    if extra_refs is True:
        arow = np.repeat(gains, rp.wchannel)
        gain_map = np.broadcast_to(arow, (rp.Jactive + rp.jref * rp.nref, rp.Kactive + rp.kref))
    else:
        arow = np.repeat(gains, rp.wchannel)[rp.jref // 2: -rp.jref // 2]
        gain_map = np.broadcast_to(arow, (rp.Jactive, rp.Kactive))

    return gain_map


def get_overflow_threshold(header: fits.header, primary_key: str) -> int:
    """Attempts to retrieve a overflow threshold value from the image header.

    Parameters
    ----------
    header: `fits.header`
        SPHEREx image header.
    primary_key: `str`
        Key from instrument to get overflow threshold,
        which has units of ADU.

    Returns
    -------
    overflow_thresh: `int`
        The overflow threshold value in ADU.

    Notes
    -----
    This function checks for the overflow threshold as specified by the primary key.
    The primary key is taken from the instrument value of overflow threshold.
    If the primary key is not found, which implies that the image is simulated,
    it falls back to the simulated data's handling:

    - For images simulated post-05/15/2024, the function retrieves the threshold
      from the 'OVERFLOW' header, which directly provides the threshold in electrons.
      Given that the simulator uses a gain factor of 1 e-/ADU, this value is
      equivalent to ADU.

    - For images simulated before 05/15/2024, the overflow threshold is defaulted
      to 30000 electrons (from the 'nonlinear_flag_threshold' parameter), which,
      under the assumption of 1 e-/ADU gain, is directly used as the value in ADU.

    - TODO: After new simulations are made, post 05/15/2024, change handling for
            the case when no primary or fallback key is found to an exception.
    """
    try:
        # Attempt to retrieve the primary key value directly from the header.
        return header[primary_key] * 4
    except KeyError:
        # Handling for simulated images if the primary key is not found.
        # Check for the existence of the OVERFLOW key which is used post-2024/05/15.
        if "OVERFLOW" in header:
            return header["OVERFLOW"]
        else:
            # Fallback to default value used prior to 2024/05/15 for simulated images.
            return 30000  # Value in electrons, equivalent to ADU with a gain of 1 e-/ADU.
