__all__ = ["active_row_col", "active_area_range", "with_extra_refs"]

# Standard library imports

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_data import ReadoutParameters, SPHERExImage


def active_row_col(rp: ReadoutParameters) -> tuple[int, int]:
    """Determine the indices of a row and column that cross the L1 image active area.

    Parameters
    ----------
    rp : `~spx_data.ReadoutParameters`
        Readout parameters defining the L1 image format.

    Returns
    -------
    row_idx, col_idx : `tuple[int, int]`
        Middle row and column of photosensitive/h2rg pixels.
    """
    # Origin at bottom LHS.
    row_actv = rp.jref // 2 * (rp.nref + 1)
    row_mid = row_actv + rp.Jactive // 2
    col_actv = int(rp.Mphan * rp.nchannel + rp.kref / 2)
    col_mid = col_actv + rp.Kactive // 2
    row_mid -= 1
    col_mid -= 1
    return row_mid, col_mid


def active_area_range(spximg: SPHERExImage, include_refs: bool = False) -> tuple[slice, slice]:
    """Given an image in L1 format, determine the range
    of the active (active + nref reads) pixel area.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`
        Input image in L1 format.

    include_refs : `bool` (Optional)
        Include reference rows with extra reads.


    Returns
    -------
    `tuple[slice, slice]`
        Range of rows and columns of active (active + nrefs reads) pixels.
    """
    rp = ReadoutParameters.from_metadata(spximg.meta)

    if include_refs is True:
        # Active + nref reads area using parameters (origin in bottom left).
        col_actv = int(rp.Mphan * rp.nchannel)
        row_range = slice(0, int(rp.Jactive + rp.jref * rp.nref))
        col_range = slice(col_actv, col_actv + rp.Kactive + rp.kref)
    else:
        # Active area using parameters (origin in bottom left).
        row_actv = rp.jref // 2
        col_actv = int(rp.Mphan * rp.nchannel) + rp.kref // 2
        row_range = slice(row_actv, row_actv + rp.Jactive)
        col_range = slice(col_actv, col_actv + rp.Kactive)

    return row_range, col_range


def with_extra_refs(rp: ReadoutParameters, calib_data: np.ndarray) -> np.ndarray:
    """Propagates extra reference row reads for calibration data.

    Parameters
    ----------
    rp : `~spx_data.ReadoutParameters`
        Input image readout parameters.
    calib_data : `~numpy.ndarray`
        Calibration data that includes active and reference pixels
        corresponding to physical pixels only.

    Returns
    -------
    `~numpy.ndarray`
    Calibration data with reference rows and extra reference rows
    set to the same values as physical reference rows.

    Notes
    -----
    A couple modules (l1.10 and l2.2) are given input calibration data
    with active and reference pixels (2048x2048); these data are applied
    to the *l1 image*. This function provides handling of extra reference
    row reads by propagating the needed rows in the expected order.
    """

    extra_refs = np.zeros(((rp.nref-1)*rp.jref, rp.Kactive + rp.kref), dtype=np.int32)
    irx = 0
    for n in range(rp.nref - 1):
        for row in range(rp.jref):
            if row > rp.jref // 2 - 1:
                row += rp.Jactive
            i, j = np.divmod(irx, rp.jref)
            row_idx = j * (rp.nref-1) + i
            # Appending extra reference reads.
            extra_refs[row_idx] = calib_data[row]
            irx += 1
    calib_data_extra_nrefs = np.concatenate((calib_data, extra_refs), axis=0)
    return calib_data_extra_nrefs
