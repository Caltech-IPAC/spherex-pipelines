__all__ = ["set_active_pix_wcs"]
from astropy.io.fits import Header


def set_active_pix_wcs(header: Header, key='A', crpix: tuple = (1, 1), crval: tuple = (0, 0)) -> None:
    """Add 0-based active pixel coordinates.

    Parameters
    ----------
    header : `~astropy.io.fits.Header`
        FITS Header
    key :
        WCS key, default is 'A'
    crpix :
        Image coordinates of the reference pixel, default is (1., 1.)
    crval :
        Coordinate value at the reference pixel, default is (0., 0.)

    """
    header[f"WCSNAME{key}"] = "0-based active pixel coordinates"
    header[f"WCSAXES{key}"] = 2
    header[f"CTYPE1{key}"] = "LINEAR"
    header[f"CTYPE2{key}"] = "LINEAR"
    header[f"CRPIX1{key}"] = crpix[0]
    header[f"CRPIX2{key}"] = crpix[1]
    header[f"CUNIT1{key}"] = "pixel"
    header[f"CUNIT2{key}"] = "pixel"
    header[f"CRVAL1{key}"] = crval[0]
    header[f"CRVAL2{key}"] = crval[1]
