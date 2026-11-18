import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.wcs import WCS
from lsst.pipe.base import Struct

from spx_data import PSFZoneData, SPHERExImage
from spx_tasks.l2b import L2bConfig, L2bTask
from spx_tests import sample_data


def generate_dark(spx_img: SPHERExImage) -> SPHERExImage:
    """Create dark current image to use for testing."""
    size = spx_img.data.shape
    dc_data = np.random.normal(size=size, loc=1.0, scale=0.1)
    dc_img = SPHERExImage(dc_data, unit="electron / s")
    return dc_img


class TestL2bTask(unittest.TestCase):

    def setUp(self):
        self.task_config = L2bConfig()
        self.task = L2bTask(config=self.task_config)

        # Mock input SPHERExImage
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [1024, 1024]
        wcs.wcs.cdelt = np.array([-0.066667, 0.066667])
        wcs.wcs.crval = [0, 0]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        data = np.random.rand(2048, 2048)
        meta = {"RA": 180.0, "DEC": 45.0}
        self.spximg = SPHERExImage(data=data, wcs=wcs, meta=meta, unit="electron/s")
        self.spximg.getArrayId = MagicMock(return_value=3)

        self.dark = generate_dark(self.spximg)

        # Mock calibration/image inputs
        self.abs_gain_matrix = MagicMock(spec=SPHERExImage)
        self.xtalk_class1 = Table({
            "Channel": np.arange(1, 33, dtype=int),
            "Coupling Amplitude": np.zeros(32, dtype=float),
        })
        self.xtalk_class2 = Table({
            str(i): np.zeros(32, dtype=float) for i in range(32)
        })
        self.he_template = MagicMock(spec=SPHERExImage)
        self.aurora_template = MagicMock(spec=SPHERExImage)
        self.shuttle_glow_template = MagicMock(spec=SPHERExImage)

        # New EPSF calibration input
        self.epsf = MagicMock(spec=SPHERExImage)

        # Mock SIP
        self.sip = sample_data.get_sample_sip(self.spximg)

        # Mock source catalog
        self.srccat = Table()
        self.srccat["ra"] = [180.1, 180.2]
        self.srccat["dec"] = [45.1, 45.2]
        self.srccat["spherex_class"] = [1, 1]

        # Mock zodi scaling table
        self.zodi_scaling = Table()
        self.zodi_scaling["wavelength_um"] = [0.5, 1.0, 1.5]
        self.zodi_scaling["ZodiScale"] = [1.0, 1.0, 1.0]
        self.zodi_scaling["ZodiOffset_nWm2sr"] = [0.0, 0.0, 0.0]

        # Mock PSFZoneData
        self.psf_avg = MagicMock(spec=PSFZoneData)

        # Mock Spectral WCS
        self.spectral_wcs = MagicMock(spec=HDUList)

        # Optional inputs
        self.ssocat = None

    @patch("spx_tasks.l2b.check_l1_dqa")
    def test_run(self, mock_check_l1_dqa):
        self.task.remove_reference_pixels.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.apply_bad_pixel_mask.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.apply_dark_flat_field_and_photo_calib.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )

        mock_diagnostics = {"FINAST": 0, "coarse_astrometry": {}}
        self.task.estimate_fine_astrometry.run = MagicMock(
            return_value=Struct(
                spximg_out=self.spximg,
                astrometry_diagnostics=mock_diagnostics,
            )
        )

        self.task.create_source_mask.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.detect_outlier_pixels.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.get_zodi_model.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.measure_psf.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.create_spectral_image.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg)
        )
        self.task.dqa_l2.run = MagicMock(
            return_value=Struct(
                spximg_out=self.spximg,
                dqa_l2=None,
                idqr_obs=None,
            )
        )

        result = self.task.run(
            spximg=self.spximg,
            abs_gain_matrix=self.abs_gain_matrix,
            dark=self.dark,
            sip=self.sip,
            srccat=self.srccat,
            psf_avg=self.psf_avg,
            epsf=self.epsf,
            zodi_scaling=self.zodi_scaling,
            spectral_wcs=self.spectral_wcs,
            ssocat=self.ssocat,
            he_template=self.he_template,
            aurora_template=self.aurora_template,
            shuttle_glow_template=self.shuttle_glow_template,
        )

        self.assertIsInstance(result, Struct)
        self.assertTrue(hasattr(result, "spximg_out"))
        self.assertTrue(hasattr(result, "dqa_l2"))
        self.assertTrue(hasattr(result, "idqr_obs"))
        self.assertTrue(hasattr(result, "astr_diag"))

        mock_check_l1_dqa.assert_called_once_with(self.spximg)

        self.task.remove_reference_pixels.run.assert_called_once_with(self.spximg)
        self.task.apply_bad_pixel_mask.run.assert_called_once_with(self.spximg)
        self.task.apply_dark_flat_field_and_photo_calib.run.assert_called_once_with(
            self.spximg,
            self.abs_gain_matrix,
            self.dark,
        )
        self.task.estimate_fine_astrometry.run.assert_called_once_with(
            self.spximg,
            self.srccat,
            sip=self.sip,
        )
        self.task.create_source_mask.run.assert_called_once_with(
            spximg=self.spximg,
            srccat=self.srccat,
            ssocat=self.ssocat,
            xtalk_class2=None,
            xtalk_class1=None,
            abs_gain_matrix=self.abs_gain_matrix
        )
        self.task.detect_outlier_pixels.run.assert_called_once_with(self.spximg)
        self.task.get_zodi_model.run.assert_called_once_with(
            self.spximg,
            self.zodi_scaling,
            self.spectral_wcs,
        )

        # Updated assertion: MeasurePSF now gets both PSF and EPSF inputs
        self.task.measure_psf.run.assert_called_once_with(
            spximg=self.spximg,
            srccat=self.srccat,
            psf_avg=self.psf_avg,
            epsf=self.epsf,
        )

        # getArrayId() mocked to return 3, so final else branch
        self.task.create_spectral_image.run.assert_called_once_with(
            self.spximg,
            self.spectral_wcs,
            he_template=None,
            aurora_template=None,
            shuttle_glow_template=None,
        )

        self.task.dqa_l2.run.assert_called_once_with(self.spximg, self.srccat)

    @patch("spx_tasks.l2b.check_l1_dqa")
    def test_run_with_no_epsf(self, mock_check_l1_dqa):
        self.task.remove_reference_pixels.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.apply_bad_pixel_mask.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.apply_dark_flat_field_and_photo_calib.run =\
            MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.estimate_fine_astrometry.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg, astrometry_diagnostics={})
        )
        self.task.create_source_mask.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.detect_outlier_pixels.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.get_zodi_model.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.measure_psf.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.create_spectral_image.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.dqa_l2.run = MagicMock(
            return_value=Struct(spximg_out=self.spximg, dqa_l2=None, idqr_obs=None)
        )

        _ = self.task.run(
            spximg=self.spximg,
            abs_gain_matrix=self.abs_gain_matrix,
            dark=self.dark,
            sip=self.sip,
            srccat=self.srccat,
            ssocat=None,
            xtalk_class1=self.xtalk_class1,
            xtalk_class2=self.xtalk_class2,
            psf_avg=self.psf_avg,
            epsf=None,
            zodi_scaling=self.zodi_scaling,
            spectral_wcs=self.spectral_wcs,
            he_template=self.he_template,
            aurora_template=self.aurora_template,
            shuttle_glow_template=self.shuttle_glow_template,
        )

        mock_check_l1_dqa.assert_called_once_with(self.spximg)
        self.task.remove_reference_pixels.run.assert_called_once_with(self.spximg)
        self.task.apply_bad_pixel_mask.run.assert_called_once_with(self.spximg)
        self.task.apply_dark_flat_field_and_photo_calib.run.assert_called_once_with(self.spximg,
                                                                                    self.abs_gain_matrix,
                                                                                    self.dark)
        self.task.estimate_fine_astrometry.run.assert_called_once_with(
            self.spximg,
            self.srccat,
            sip=self.sip
            )
        self.task.create_source_mask.run.assert_called_once_with(
            spximg=self.spximg,
            srccat=self.srccat,
            ssocat=self.ssocat,
            xtalk_class2=self.xtalk_class2,
            xtalk_class1=self.xtalk_class1,
            abs_gain_matrix=self.abs_gain_matrix
            )


if __name__ == '__main__':
    unittest.main()
