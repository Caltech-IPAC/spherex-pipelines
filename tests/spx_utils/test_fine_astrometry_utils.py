import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from spx_utils.astrometry.fine_astrometry_utils import se_run, scamp_run, astronet_run
import numpy as np


class TestFineAstroUtils(unittest.TestCase):

    @patch('astropy.io.fits.open')
    @patch('subprocess.run')
    def test_se_run(self, mock_run, mock_fits_open):
        mock_run.return_value = MagicMock(
            stdout="SExtractor simulated",
            stderr="Objects: detected 3500",
            returncode=0
        )

        # Mock fits.open to simulate catalog with sources
        mock_hdul = [None, None, MagicMock()]
        mock_hdul[2].data = [1, 2, 3]  # Fake data for catalog
        mock_fits_open.return_value.__enter__.return_value = mock_hdul

        img_path = Path("/mock/path/image.fits")
        img_ldac = Path("/mock/path/image.ldac")

        se_run(img_path, img_ldac, num_ref_cat=3000, verbose=False)

        mock_run.assert_called()
        self.assertIn("detected 3500", mock_run.return_value.stderr)

    @patch('subprocess.run')
    def test_scamp_run(self, mock_run):
        mock_run.return_value = MagicMock(
            stdout="SCAMP run simulated",
            stderr=(
                '----- 3175 detections loaded\n'
                'Group  1:     8498 standards in file (band 1)\n'
                '2025W16_2A_0044_4_ A1  P1  +0.00 deg   6.131"  0.958    -0.74"    +3.1"   1.34\n'
                '----- Astrometric stats (external):\n'
                'Group 1:   1.23"   2.34"\n'
                'Group 1:   1.68"   2.54" 8.3e+02    2296\n'
            ),
            returncode=0
        )

        img_ldac = Path("/mock/path/img.ldac")
        refcat_ldac = Path("/mock/path/refcat.ldac")
        scamp_out = Path("/mock/path/fulloutcat.fits")
        (contrast, detections, standards,
         ra_shift, dec_shift,
         daxis1, daxis2, nstars) = scamp_run(img_ldac, refcat_ldac,
                                             scamp_out, verbose=False)

        self.assertAlmostEqual(contrast, 1.34)
        self.assertEqual(detections, 3175)
        self.assertEqual(standards, 8498)
        self.assertAlmostEqual(ra_shift, -0.74)
        self.assertAlmostEqual(dec_shift, 3.1)
        self.assertAlmostEqual(daxis1, 1.23)
        self.assertAlmostEqual(daxis2, 2.34)
        self.assertEqual(nstars, 2296)

    @patch("astropy.io.fits.PrimaryHDU.writeto")
    @patch("astropy.io.fits.open")
    @patch("subprocess.run")
    @patch("os.path.exists", return_value=True)
    def test_astronet_run(self, mock_exists, mock_run, mock_fits_open, mock_writeto):
        # -- Mock subprocess.run --
        mock_run.return_value = MagicMock(stdout="solve-field completed", returncode=0)

        # -- Realistic fake image array --
        fake_data = np.random.rand(10, 10).astype(np.float32)
        fake_header = {'CRVAL1': 100.0, 'CRVAL2': -30.0}

        # -- Properly mimic fits.open()[1].data and .header --
        mock_hdu = MagicMock()
        mock_hdu.data = fake_data
        mock_hdu.header = fake_header
        mock_fits_open.return_value.__enter__.return_value = [None, mock_hdu]

        # Inputs
        input_image = Path("/mock/input.fits")
        stripped_image = Path("/mock/stripped.fits")
        wcs_output = Path("/mock/output.wcs")
        index_dir = "/mock/index"

        # Run the function
        result = astronet_run(
            input_image=input_image,
            stripped_image=stripped_image,
            wcs_output=wcs_output,
            index_dir=index_dir,
            verbose=False
        )

        # Assertions
        mock_run.assert_called_once()
        mock_writeto.assert_called_once()
        assert result == (wcs_output, stripped_image)


if __name__ == "__main__":
    unittest.main()
