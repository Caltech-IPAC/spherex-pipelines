from astropy.io.fits import Header
from astropy.wcs import WCS
from spx_utils import set_active_pix_wcs


def test_set_active_pix_wcs():
    hdr = Header()
    key = "A"
    set_active_pix_wcs(hdr, key=key)
    wcs = WCS(hdr, key=key)
    pixel_coords = [(1, 2), (15, 25)]
    # the second parameter is the origin (1 = 1-based indexing)
    world_coords = wcs.all_pix2world(pixel_coords, 1)

    # verify that the world coordinates are 0-based pixel coordinates
    for pix, world in zip(pixel_coords, world_coords):
        assert pix[0] == world[0]+1 and pix[1] == world[1]+1
