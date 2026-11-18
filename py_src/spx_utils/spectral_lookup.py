__all__ = ["spectral_wcs_lookup_data", "spectral_wcs_tab_header"]
import numpy as np

from astropy.io import fits
from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.time import Time
from astropy.units import Quantity
from astropy.wcs import WCS

from spx_data import DetectorSpectralInfo, SpectralWCS, spx_param
from .utils import add_detcoord


def _get_tab_data(spectral_info: DetectorSpectralInfo,
                  n: int = 7,
                  start: int = 0,
                  end: int = spx_param.get_npix_x_active()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get lookup table content for nxn points on regular grid.

    For 2040 active pixels, the step or approximate number of pixels between
    the adjacent points is 2040/(n-1):
        step = 340    # 7x7 points
        step = 255    # 9x9 points
        step = 204    # 11x11 points
        step = 170    # 13x13 points
        step = 145.7  # 15x15 points

    Parameters
    ----------
    spectral_info : `spx_data.DetectorSpectralInfo`
        Detector spectral info.
    n : `int`
        Odd number of equidistant points along x or y, default is 7.
    start: `int`
        Start pixel index, default is 0.
    end: `int`
        Index just beyond the last pixel index, default is the number of active pixels.

    Returns
    -------
    result : tuple
        A tuple containing 3 numpy arrays:
        - index_x, 1d array of integer indexes in X direction (1-based)
        - index_y, 1d array of indexes in Y direction (1-based)
        - coordinate_data, 6x6x2 array of coordinate values for wavelength and bandwidth

    """
    if n % 2 == 0:
        raise ValueError(f"n={n} is not odd")

    step = end / (n - 1)

    ix = np.append(np.round(np.arange(start, end - 1, step)), end - 1).astype(np.int32)
    iy = ix.copy()
    x, y = np.meshgrid(ix, iy)

    # get central wavelength and bandwidth for each grid point
    wl, bw = spectral_info.get_wl_mid_fwhm(x, y)

    coord_data = np.dstack((wl, bw))

    # FITS pixel indexes are 1-based
    return ix + 1, iy + 1, coord_data


def spectral_wcs_tab_header(alt="W") -> fits.Header:
    """Get header keywords defining alternate WCS with wavelength and bandwidth coords.

    The original header for IMAGE extensions with WCS information needs to be extended
    with this header to define wavelength and bandwidth coordinates.

    Parameters
    ----------
    alt : string
        alt WCS char or " " for primary WCS

    Returns
    -------
    hdr : `~astropy.io.fits.Header`
        FITS header keywords to add alternate WCS with spectral coordinates

    """
    hdr = fits.Header()

    hdr[f"WCSAXES{alt}"] = (2, "Two output axes: wavelength and bandwidth")
    hdr[f"WCSNAME{alt}"] = ("Approximated per-pixel wavelength and bandwidth", "WCS label")

    hdr[f"CTYPE1{alt}"] = ("WAVE-TAB", "Bilinear lookup table, FITS Paper 3, Sect 6.1")
    hdr[f"CTYPE2{alt}"] = ("WAVE-TAB", "Bilinear lookup table, FITS Paper 3, Sect 6.1")

    hdr[f"CNAME1{alt}"] = ("Wavelength", "First output axis is central wavelength")
    hdr[f"CNAME2{alt}"] = ("Bandwidth", "Second output axis is bandwidth")

    hdr[f"CUNIT1{alt}"] = ("um", "Wavelength is reported in microns")
    hdr[f"CUNIT2{alt}"] = ("um", "Bandpass is reported in microns")

    hdr[f"CRPIX1{alt}"] = (1, "Lookup table is in raw FITS coordinates")
    hdr[f"CRPIX2{alt}"] = (1, "Lookup table is in raw FITS coordinates")

    hdr[f"CRVAL1{alt}"] = (1, "[pixel] Lookup table is in raw FITS coordinates")
    hdr[f"CRVAL2{alt}"] = (1, "[pixel] Lookup table is in raw FITS coordinates")

    hdr[f"CDELT1{alt}"] = (1, "[pixel] Lookup table is in raw FITS coordinates")
    hdr[f"CDELT2{alt}"] = (1, "[pixel] Lookup table is in raw FITS coordinates")

    # i_j: output_input, all defaults
    hdr[f"PC1_1{alt}"] = 1
    hdr[f"PC1_2{alt}"] = 0
    hdr[f"PC2_1{alt}"] = 0
    hdr[f"PC2_2{alt}"] = 1

    # Set up lookup table header keywords
    # -TAB logic: with an indexing vector, PC matrix
    # Table extension name (EXTNAME) for the first coord
    hdr[f"PS1_0{alt}"] = (SpectralWCS.LKPTBL_EXTNAME, "Extension with wavelength lookup table")
    # Table extension name (EXTNAME) for the second coord
    hdr[f"PS2_0{alt}"] = (SpectralWCS.LKPTBL_EXTNAME, "Extension with bandwidth lookup table")
    # Column name for the coordinate array for the first coord
    hdr[f"PS1_1{alt}"] = ("VALUES", "Column with values for wavelength")
    # Column name for the coordinate array for the second coord
    hdr[f"PS2_1{alt}"] = ("VALUES", "Column with values for bandwidth")
    # Column name for the indexing vector for the first coord
    hdr[f"PS1_2{alt}"] = ("X", "Column with control points for detector in X")
    # Column name for the indexing vector for the first coord
    hdr[f"PS2_2{alt}"] = ("Y", "Column with control points for detector in Y")
    # Axis number on coordinate values array for the first coord
    hdr[f"PV1_3{alt}"] = (1, "Index of wavelength in VALUES array")
    # Axis number on coordinate values array for the second coord
    hdr[f"PV2_3{alt}"] = (2, "Index of bandwidth in VALUES array")

    return hdr


def spectral_wcs_lookup_data(detector: int, wl_tolerance: Quantity) -> HDUList:
    """Get HDUList with central wavelength, bandwidth, and lookup table.

    The algorithm to get spectral coordinates by table lookup is described in section 6
    of Greisen, E. W., Calabretta, M. R., Valdes, F. G., & Allen, S. L. (2006).
    Representations of spectral coordinates in FITS. Astronomy & Astrophysics,
    446(2), 747-771. DOI: 10.1051/0004-6361:20053437

    Parameters
    ----------
    detector : int
        detector array number
    wl_tolerance : `Quantity`
        max wavelength error

    Returns
    -------
    result : ~astropy.io.fits.HDUList

    """
    spectral_info = DetectorSpectralInfo.create(detector)
    wl_true, bw_true = (spectral_info.central_wl, spectral_info.central_bw)
    if bw_true is None:
        bw_true = wl_true / spectral_info.spectral_r

    hdu_hdr = fits.Header([
        ("DETECTOR", detector, "1-3: SWIR, 4-6: MWIR"),
        ("BUNIT", "um", "Unit"),
    ])

    alt = ""
    tab_hdr = spectral_wcs_tab_header(alt)
    hdu_hdr.extend(tab_hdr, useblanks=True)
    wl_hdu = fits.ImageHDU(data=wl_true, header=hdu_hdr, name="CWAVE")
    bw_hdu = fits.ImageHDU(data=bw_true, header=hdu_hdr, name="CBAND")

    # indexes for getting lookup coordinates - full image
    start = 0
    end = spx_param.get_npix_x_active()
    ix = np.arange(start, end)
    iy = np.arange(start, end)
    x, y = np.meshgrid(ix, iy)

    # n along x-axis must be odd to adequately reflect smile shape
    for n in range(5, 100, 2):
        index_x, index_y, coord_data = _get_tab_data(spectral_info, n, start, end)

        tbl = Table({
            tab_hdr[f"PS1_2{alt}"]: [index_x],
            tab_hdr[f"PS2_2{alt}"]: [index_y],
            tab_hdr[f"PS1_1{alt}"]: [coord_data]})

        # HDU with lookup table
        tbl_hdu = fits.BinTableHDU(data=tbl, name=tab_hdr[f"PS2_0{alt}"])
        # Create a list of HDU objects
        hdus = [fits.PrimaryHDU(), wl_hdu, bw_hdu, tbl_hdu]
        # Create an HDUList from the list of HDUs
        hdul = fits.HDUList(hdus)

        # get lookup wavelength and bandwidth for each grid point
        wcs = WCS(hdul[1], hdul)  # is using alt WCS, provide key='W'
        wl_lkp, bw_lkp = wcs.wcs_pix2world(x, y, 0)

        max_wl_error = np.max(np.abs(wl_lkp - wl_true))
        max_bp_error = np.max(np.abs(bw_lkp - bw_true))

        if max_wl_error < wl_tolerance.to(spectral_info.wl_unit).value:
            hdus[0].header.extend([
                ("DATE", Time.now().to_value(format="fits"), "HDU creation time (UTC)"),
                ("ORIGIN", "SPHEREx SSDC Team"),
                ("TELESCOP", "SPHEREx"),
                ("INSTRUME", "SPHEREx"),
                ("WLERR", max_wl_error, "Maximum error (um) in wavelength lookup"),
                ("BPERR", max_bp_error, "Maximum error (um) in bandwidth lookup")
            ])
            # The MWIR detectors data are flipped in get_wavelength_mesh
            add_detcoord(hdul)

            hdus[0].header["COMMENT"] = "Central wavelength based on TVAC data model"
            hdus[0].header["COMMENT"] = "Central bandwidth derived from central wavelength"
            hdus[0].header.add_history("Version 1.0.1")

            return hdul

    raise ValueError(f"Failed to create wavelength lookup table with error < {wl_tolerance}")
