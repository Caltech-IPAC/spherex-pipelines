import numpy as np
from astropy.table import Table
from spx_data import SpectralWCS
from spx_tests import get_5x5sample_spectral_wcs


def test_spherex_wcs():
    hdul, _ = get_5x5sample_spectral_wcs()

    # spectral WCS without modification
    swcs1: SpectralWCS = SpectralWCS(hdul, key=" ")
    assert Table.read(swcs1.tab) == Table.read(hdul[SpectralWCS.LKPTBL_EXTNAME])
    hdr = swcs1.hdr
    alt = SpectralWCS.WCS_KEY
    ix_col = hdr[f"PS1_2{alt}"]  # one-based indexes
    iy_col = hdr[f"PS2_2{alt}"]  # one-based indexes
    coord_col = hdr[f"PS1_1{alt}"]
    tbl = Table(swcs1.tab.data)
    # the table is one row, each cell is an array
    ix_in, iy_in, coords_in = tbl[ix_col][0], tbl[iy_col][0], tbl[coord_col][0]

    # spectral WCS with lookup table modification
    # Example: the original spectral WCS is defined for active area,
    # but we have a spectral WCS for a larger area with non-active pixels on the edges,
    # non-active pixels should show with  0 values for spectral coords
    img_shape = (9, 10)
    actv_area = (slice(2, 7), slice(3, 8))
    actv_y, actv_x = actv_area
    swcs2: SpectralWCS = SpectralWCS(hdul, key=" ", imgshape=img_shape, actv_area=actv_area, xflip=False)
    tbl_out = Table(swcs2.tab.data)
    # the table is one row, each cell is an array
    ix, iy, coords = tbl_out[ix_col][0], tbl_out[iy_col][0], tbl_out[coord_col][0]
    # active area lookup data should not change
    assert np.array_equal(ix[2:-2] - actv_x.start, ix_in)
    assert np.array_equal(iy[2:-2] - actv_y.start, iy_in)
    assert np.array_equal(coords[2:-2, 2:-2, :], coords_in)

    swcs3: SpectralWCS = SpectralWCS(hdul, key=" ", imgshape=img_shape, actv_area=actv_area, xflip=True)
    tbl_out = Table(swcs3.tab.data)
    # the table is one row, each cell is an array
    ix, iy, coords = tbl_out[ix_col][0], tbl_out[iy_col][0], tbl_out[coord_col][0]
    # active area lookup data should not change
    assert np.array_equal(ix[2:-2] - actv_x.start, ix_in)
    assert np.array_equal(iy[2:-2] - actv_y.start, iy_in)
    assert np.array_equal(coords[2:-2, 2:-2, :], coords_in[:, ::-1, :])
