__all__ = ["SpectralWCS"]

import numpy as np
from astropy.io import fits
from astropy.table import Table


class SpectralWCS:
    """Object that holds spectral WCS data.

    Attributes
    ----------
    hdr : `~astropy.io.fits.Header`
        FITS Header with the spectral coordinates WCS.
    tab : `~astropy.io.fits.BinTableHDU`
        Spectral coordinate lookup table HDU.

    """

    LKPTBL_EXTNAME: str = "WCS-WAVE"
    """Lookup table extension name."""
    WCS_KEY: str = "W"
    SPECTRAL_WCS_KEYWORDS = [
        "WCSAXES",
        "WCSNAME",
        "CNAME1", "CNAME2",
        "CRPIX1", "CRPIX2",
        "CDELT1", "CDELT2",
        "CUNIT1", "CUNIT2",
        "CTYPE1", "CTYPE2",
        "CRVAL1", "CRVAL2",
        "PV1_3", "PV2_3",
        "PS1_0", "PS2_0",
        "PS1_1", "PS2_1",
        "PS1_2", "PS2_2",
        "LATPOLE"
    ]

    def __init__(self,
                 hdul: fits.HDUList,
                 key: str,
                 *,
                 imgshape: tuple[int, int] = None,
                 actv_area: tuple[slice, slice] = None,
                 xflip: bool = False):
        """Create Spectral WCS object.

        Parameters
        ----------
        hdul : `~astropy.io.fits.HDUList`
            FITS HDU List that contains at least 2 extensions: image with
            spectral WCS and coordinate lookup table.
        key : `str`
            Image WCS key containing spectral coords, ``" "`` if main WCS is used.
        imgshape : `tuple[int, int]`, optional
            Shape of the target image: number of rows and number of columns.
            Used when the original spectral lookup data is provided for a smaller
            area and needs to be modified.
        actv_area : `tuple[slice, slice]`, optional
            Active area. Used when the original spectral lookup data is provided
            for a smaller area and needs to be modified.
        xflip : `bool`, optional
            If True, spectral wcs lookup table needs to be flipped along X axis.
            Default is False.

        """
        # assuming the first extension is an image with spectral WCS
        # create a header with spectral WCS keywords
        image_header = hdul[1].header
        hdr = fits.Header()  # spectral header
        for keyword in self.SPECTRAL_WCS_KEYWORDS:
            if keyword+key in image_header:
                card = image_header.cards[keyword+key]
                hdr[keyword+self.WCS_KEY] = (card.value, card.comment)

        tab = hdul[SpectralWCS.LKPTBL_EXTNAME]
        # make sure we use a copy of BinTableHDU
        if imgshape is not None and actv_area is not None:
            tab = _modify_lookup_data(imgshape, actv_area, tab, hdr, self.WCS_KEY, xflip)
        else:
            tab = fits.BinTableHDU(data=Table.read(tab), name=hdr[f"PS2_0{self.WCS_KEY}"])
        self.hdr = hdr
        self.tab = tab

    def extendImageHeader(self, imghdr: fits.Header):
        imghdr.extend(self.hdr, useblanks=False, update=True)

    @classmethod
    def hasSpectralWCS(cls, hdul: fits.HDUList):
        for hdu in hdul:
            if hdu.name == SpectralWCS.LKPTBL_EXTNAME:
                return True
        return False

    @classmethod
    def getSpectralWCSKeywords(cls) -> list[str]:
        """Spectral WCS keywords as they appear in the SPHERExImage header."""
        return [keyword+cls.WCS_KEY for keyword in cls.SPECTRAL_WCS_KEYWORDS]


def _modify_lookup_data(img_shape: tuple[int, int],
                        actv_area: tuple[slice, slice],
                        tbl_hdu: fits.BinTableHDU,
                        hdr: fits.Header,
                        alt: str,
                        xflip: bool):
    """Modify spectral coordinate lookup data to account for pixels outside active area.

    Parameters
    ----------
    img_shape : `tuple[int, int]`
        Shape of the target image: number of rows and number of columns.
    actv_area : `tuple[slice, slice]`
        Active area as slices in rows and columns
    tbl_hdu : `astropy.io.fits.BinTableHDU`
        HDU with the coordinate lookup data for active pixels only
    hdr : `~astropy.io.fits.Header`
        FITS Header object defining spectral coordinates
    alt : `str`
        Alternate WCS key, used in ``hdr``.
    xflip : `bool`
        If True, spectral wcs lookup table needs to be flipped along X axis.

    Returns
    -------
    result : `astropy.io.fits.BinTableHDU`
        Binary Table HDU with modified lookup data matching image shape

    """
    img_nrows, img_ncols = img_shape
    actv_rows, actv_cols = actv_area  # zero-based indexes
    actv_nrows, actv_ncols = abs(actv_rows.stop - actv_rows.start), abs(actv_cols.stop - actv_cols.start)
    actv_shape = (actv_nrows, actv_ncols)
    if img_nrows < actv_nrows or img_ncols < actv_ncols:
        raise ValueError("Image is smaller than active area.")
    if not xflip and actv_shape == img_shape:
        return tbl_hdu
    ix_col = hdr[f"PS1_2{alt}"]  # one-based indexes
    iy_col = hdr[f"PS2_2{alt}"]  # one-based indexes
    coord_col = hdr[f"PS1_1{alt}"]
    tbl = Table(tbl_hdu.data)
    ix, iy, coords = tbl[ix_col].data[0], tbl[iy_col].data[0], tbl[coord_col].data[0]

    if xflip:
        # reverse the X direction (horizontal flip)
        coords = coords[:, ::-1, :]

    ix += actv_cols.start  # one-based indexes
    iy += actv_rows.start  # one-based indexes

    if actv_cols.start < 2 or actv_rows.start < 2 or \
            img_ncols-actv_cols.stop < 2 or img_nrows-actv_rows.stop < 2:
        # if less than 2 non-active pixels on any side, the indexes are shifted,
        # but there is no change in coordinate values
        cvals = coords
    else:
        # updated indexes: include index extremas for non-active area pixels
        ix = np.concatenate(([1, actv_cols.start], ix, [actv_cols.stop + 1, img_ncols]))
        iy = np.concatenate(([1, actv_rows.start], iy, [actv_rows.stop + 1, img_nrows]))

        # coordinate values for non-active pixels are 0
        cvals = np.zeros(shape=(len(iy), len(ix), 2), dtype=coords.dtype)  # 2 is the number of coordinates
        cvals[2:-2, 2:-2, :] = coords  # 2 indexes were on each end

    # lookup table with updated indices and values
    tbl_out = Table({
        hdr[f"PS1_2{alt}"]: [ix],
        hdr[f"PS2_2{alt}"]: [iy],
        hdr[f"PS1_1{alt}"]: [cvals]})

    # HDU with updated lookup table
    return fits.BinTableHDU(data=tbl_out, name=hdr[f"PS2_0{alt}"])
