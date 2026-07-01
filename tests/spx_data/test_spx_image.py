import os
import shutil
import tempfile
import unittest

import numpy as np

from astropy.io import fits
from astropy.nddata import CCDData
from astropy.table import Table, setdiff
from pathlib import Path
from spx_data import SPHERExImage, Flag, Layer, spherex_image_reader, Zodi
from spx_tests import get_sample_psf, get_sample_ssocat, get_sample_sur, get_sample_zodi

test_root = Path(__file__).parent.parent


class TestSPHERExImage(unittest.TestCase):
    root = None

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp(dir=test_root)

    @classmethod
    def tearDownClass(cls):
        if cls.root is not None:
            shutil.rmtree(cls.root, ignore_errors=True)

    def test_read_write(self):
        # small_with_cat_wcs.fits contains 3 extensions: image, flags, and catalog
        # also contains wcs in flags extension
        file_path = os.path.join(test_root, "data", "small_with_cat_wcs.fits")
        # read multi-extension FITS file
        spherex_image = SPHERExImage.read(file_path)

        self.assertTrue(isinstance(spherex_image, SPHERExImage))
        self.assertTrue(isinstance(spherex_image, CCDData))
        self.assertTrue(spherex_image.data is not None)
        self.assertTrue(spherex_image.flags is not None)
        self.assertTrue(spherex_image.catalog is not None)
        # 3 flags are defined in the test file
        self.assertEqual(len(spherex_image.flag_defs), 3)

        # convert SPHERExImage object into multi-extension FITS file
        out_path = os.path.join(self.root, "small_with_cat1.fits")
        spherex_image.write(out_path, wcs_relax=True, key_uncertainty_type="UTYPE", add_version=True)

        # catalog EXTNAME comment field might be different
        fd = fits.FITSDiff(file_path, out_path, ignore_keywords=["VERSION"],
                           ignore_comments=["VERSION", "EXTTYPE"])
        self.assertTrue(fd.identical)

        with fits.open(out_path) as hdulist:
            # Primary extension - header only
            # first extension - data
            # second extension - flags
            # third extension - catalog
            self.assertEqual(len(hdulist), 4)

        # test layers read/write
        ssocat_in = get_sample_ssocat()
        spherex_image.ssocat = ssocat_in
        sur_in = get_sample_sur()
        spherex_image.sur = sur_in
        zodi_in = get_sample_zodi(spherex_image.data.shape)
        spherex_image.zodi = zodi_in
        psf_in = get_sample_psf()
        spherex_image.psfcube = psf_in
        spherex_image_layers = [k for k in spherex_image.layers.keys()]

        path1 = os.path.join(self.root, "small_with_layers.fits")
        spherex_image.write(path1, wcs_relax=True, key_uncertainty_type="UTYPE")
        spherex_image_with_layers = SPHERExImage.read(path1)
        psf_out = spherex_image_with_layers.psfcube
        ssocat_out = spherex_image_with_layers.ssocat
        ssocat_diff: Table = setdiff(ssocat_in, ssocat_out)
        sur_out = spherex_image_with_layers.sur
        zodi_out = spherex_image_with_layers.zodi
        spherex_image_read_layers = [k for k in spherex_image_with_layers.layers.keys()]
        self.assertTrue(spherex_image_layers == spherex_image_read_layers)
        self.assertTrue(psf_in == psf_out)
        self.assertTrue(ssocat_diff.colnames == ssocat_in.colnames)
        self.assertTrue(len(ssocat_diff) == 0)
        self.assertTrue(sur_in == sur_out)
        self.assertTrue(zodi_in == zodi_out)

        # test reading gz compression
        file_path = test_root / "data/small_with_cat.fits.gz"
        spherex_image = SPHERExImage.read(file_path)
        self.assertTrue(isinstance(spherex_image, SPHERExImage))
        self.assertTrue(spherex_image.data is not None)

    def test_flags(self):
        data = np.zeros([3, 3])
        spherex_image = SPHERExImage(data, unit="adu")
        with self.assertRaises(RuntimeError):
            spherex_image.getFlagsAsMask(Flag.OVERFLOW)

        flags = np.zeros(data.shape, dtype=np.uint32)
        flags[0, 0] = flags[2, 2] = Flag.OVERFLOW
        mask_sat = flags & Flag.OVERFLOW > 0

        spherex_image.setFlagsFromMask(mask_sat, Flag.OVERFLOW)
        self.assertTrue(spherex_image.flags is not None)
        self.assertTrue(np.array_equal(spherex_image.flags, flags))
        self.assertTrue(Flag.OVERFLOW.name in spherex_image.flag_defs.keys())

        flags[2, 2] |= Flag.TRANSIENT
        flags[1, 1] = Flag.TRANSIENT
        mask_cr = flags & Flag.TRANSIENT > 0
        spherex_image.setFlagsFromMask(mask_cr, Flag.TRANSIENT)
        self.assertTrue(np.array_equal(spherex_image.flags, flags))
        self.assertTrue(Flag.TRANSIENT.name in spherex_image.flag_defs.keys())

        mask_sat_or_cp = spherex_image.getFlagsAsMask(Flag.OVERFLOW | Flag.TRANSIENT)
        self.assertTrue(np.array_equal(mask_sat_or_cp, mask_sat | mask_cr))

        mask_sat_and_cp = spherex_image.getFlagsAsMask(Flag.OVERFLOW | Flag.TRANSIENT, check_all=True)
        self.assertTrue(mask_sat_and_cp[2, 2])
        self.assertFalse(mask_sat_and_cp[1, 1])
        self.assertFalse(mask_sat_and_cp[0, 0])

    def test_ensure_layers(self):
        spherex_image = SPHERExImage(np.full([3, 3], 5), unit="adu")
        # adding two layers
        zodi_data = np.full(spherex_image.data.shape, 0.2)
        spherex_image.zodi = Zodi(zodi_data, "Test Model")
        cat_data = Table([["a1", "a2"], ["b1", "b2"]], names=("a", "b"))
        spherex_image.catalog = cat_data
        # test two layers
        spherex_image.ensure_layers([Layer.CATALOG, Layer.ZODI])
        self.assertTrue(spherex_image.catalog is not None and spherex_image.zodi is not None)
        # test one layer
        spherex_image.ensure_layers([Layer.ZODI])
        self.assertTrue(spherex_image.catalog is None and spherex_image.zodi is not None)
        # test no layers
        spherex_image.ensure_layers([])
        self.assertTrue(spherex_image.catalog is None and spherex_image.zodi is None)
        # test non existing layer
        with self.assertRaises(ValueError):
            spherex_image.ensure_layers(["NONEXISTING"])

    def test_preservation_layers_flags(self):
        spximg = SPHERExImage(np.full([3, 3], 5), unit="adu")
        # adding two layers
        zodi_data = np.full(spximg.data.shape, 0.2)
        spximg.zodi = Zodi(zodi_data, "Test Model")
        cat_data = Table([["a1", "a2"], ["b1", "b2"]], names=("a", "b"))
        spximg.catalog = cat_data
        # Add flags and flag_defs to spximg to check persistance
        flags = np.zeros(spximg.data.shape, dtype=np.uint32)
        flags[0, 0] = flags[2, 2] = Flag.OVERFLOW
        mask_sat = flags & Flag.OVERFLOW > 0
        spximg.setFlagsFromMask(mask_sat, Flag.OVERFLOW)
        flags[2, 2] |= Flag.TRANSIENT
        flags[1, 1] = Flag.TRANSIENT
        mask_cr = flags & Flag.TRANSIENT > 0
        spximg.setFlagsFromMask(mask_cr, Flag.TRANSIENT)
        # Creating ff image to do arithmetic to
        ff_img = SPHERExImage(np.full([3, 3], 2), unit="adu")
        ff_reduced = spximg.divide(ff_img, handle_flags="ff", handle_meta="ff", handle_layers="ff")
        self.assertTrue(np.array_equal(spximg.flags, ff_reduced.flags))
        self.assertTrue(np.array_equal(spximg.zodi, ff_reduced.zodi))
        self.assertTrue(ff_reduced.layers)
        # Check flag defs and flags for persistance
        self.assertTrue(spximg.flags is not None)
        self.assertTrue(spximg.flag_defs is not None)
        self.assertTrue(np.array_equal(spximg.flags, ff_reduced.flags))
        self.assertTrue(spximg.flag_defs == ff_reduced.flag_defs)
        # Make sure layers don"t persist for images without
        self.assertFalse(ff_img.layers)
        testimg = SPHERExImage(np.full([3, 3], 5), unit="adu")
        self.assertFalse(testimg.layers)

    def test_slicing_flags_and_defs(self):
        sample_file = test_root / "data/small.fits"
        spximg = spherex_image_reader(sample_file)
        slicedimg = spximg[2:, :]
        self.assertTrue(np.array_equal(spximg.flags[2:, :], slicedimg.flags), "flags are not in sliced array")
        self.assertTrue(slicedimg.flag_defs is not None, "flag defs are missing")
        self.assertTrue(spximg.flag_defs == slicedimg.flag_defs, "flag defs are not consistent")

    def test_wcs_in_layers(self):
        sample_file = test_root / "data/small.fits"
        # WCS headers to check for
        wcs_headers = ["CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CTYPE1", "CTYPE2",
                       "CUNIT1", "CUNIT2", "LATPOLE", "LONPOLE", "MJDREF", "PC1_1", "PC1_2", "PC2_1",
                       "PC2_2", "RADESYS", "WCSAXES"]
        alt_wcs_headers = ["WCSNAMEA", "WCSAXESA",
                           "CTYPE1A", "CTYPE2A",
                           "CRPIX1A", "CRPIX2A",
                           "CUNIT1A", "CUNIT2A"]
        spximg = SPHERExImage.read(sample_file)
        # Write to include wcs in layers
        root_path = Path(self.root)
        spximg.write(root_path / "small1.fits")
        # Read back in and check wcs in headers
        with fits.open(root_path / "small1.fits") as f:
            for hdu in f:
                if hdu.name in ["IMAGE", "VARIANCE", "FLAGS", "ZODI", "PSF"]:
                    # Verify headers are there for wcs
                    hdu_header = list(hdu.header.keys())
                    for wcs_header in wcs_headers:
                        self.assertTrue(wcs_header in hdu_header, f"WCS was not written to layer {hdu.name}")
                    # Check alternative wcs in headers
                    for alt_wcs_header in alt_wcs_headers:
                        self.assertTrue(alt_wcs_header in hdu_header,
                                        f"Alternative WCS was not written to layer {hdu.name}")

    # def test_raw(self):
    #     raw_file = test_root / "data/exp_0_0_raw_2023353T000443.fits"
    #     img = SPHERExImage.read(raw_file)
    #     print(img.wcs)
    #     print(img.alt_wcs["A"])


if __name__ == "__main__":
    unittest.main()
