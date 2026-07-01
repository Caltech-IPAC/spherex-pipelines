import numpy as np
from astropy import wcs
from astropy.io import fits

from spx_data import SPHERExImage, Flag, flag2bit

# generate 5x5 small image

data = np.array(
    [[6.791, 6.283, 6.387, 6.373, 6.536],
     [7.158, 8.099, 7.449, 7.803, 8.76],
     [7.003, 7.270, 8.327, 11.317, 22.316],
     [6.333, 7.333, 10.855, 75.292, 1322.66],
     [6.538, 7.143, 8.695, 25.802, 1922.44]], dtype=np.float32)

flags = np.array(
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, Flag.OVERFLOW | Flag.TRANSIENT],
     [0, 0, 0, 0, Flag.OVERFLOW]], dtype=np.int32)

flag_defs = {
    "TRANSIENT": flag2bit(Flag.TRANSIENT),
    "OVERFLOW": flag2bit(Flag.OVERFLOW)
}

# WCS
w = wcs.WCS(naxis=2)
w.wcs.crpix = [2, 2]
w.wcs.cdelt = np.array([0.00172, 0.00172])
w.wcs.crval = [277.70231, -4.62191]
w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
w.wcs.pc = [
    [0.99640353, -0.08473489],
    [0.08473489, 0.99640353]]

hdr = fits.Header()
hdr["EXTNAME"] = "IMAGE"
hdr["EXTYPE"] = "IMAGE"
hdr.add_comment("Generated test data")

img = SPHERExImage(data=data,
                   flag_defs=flag_defs,
                   flags=flags,
                   wcs=w,
                   meta=hdr,
                   unit="electron / s")

small_path = "data/small.fits"
img.write(small_path, overwrite=True)
