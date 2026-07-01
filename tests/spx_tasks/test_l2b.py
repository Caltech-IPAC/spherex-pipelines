import unittest
from unittest.mock import MagicMock, patch
from astropy.table import Table
from astropy.io.fits import HDUList
from astropy.wcs import WCS, Sip
import numpy as np
from lsst.pipe.base import Struct
from spx_data import SPHERExImage, PSFZoneData
from spx_tasks.l2b import L2bTask, L2bConfig


def generate_dark(spx_img: SPHERExImage) -> SPHERExImage:
    """Creates dark current image to use for testing.

    Parameters
    ----------
    spx_img : SPHERExImage
        Science image to generate dark current size.

    Returns
    -------
    dc_img : SPHERExImage
        The generated dark current image with data and uncertainty.

    Notes
    -----
    Tests and validates the generated data for:
    - ``dc_img`` has uncertainty populated.
    """
    # Generate sample data
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
        meta = {'RA': 180.0, 'DEC': 45.0}
        self.spximg = SPHERExImage(data=data, wcs=wcs, meta=meta, unit="electron/s")
        self.spximg.getArrayId = MagicMock(return_value=1)
        self.dark = generate_dark(self.spximg)

        # Mock abs_gain_matrix, xtalk_coupling_matrix
        self.abs_gain_matrix = MagicMock(spec=SPHERExImage)
        self.xtalk_coupling_matrix = MagicMock(spec=SPHERExImage)

        # Mock SIP
        self.sip = MagicMock(spec=Sip)

        # Mock Source catalog
        self.srccat = Table()
        self.srccat['ra'] = [180.1, 180.2]
        self.srccat['dec'] = [45.1, 45.2]
        self.srccat['spherex_class'] = [1, 1]

        # Mock PSFZoneData
        self.psf_avg = MagicMock(spec=PSFZoneData)

        # Mock Spectral WCS
        self.spectral_wcs = MagicMock(spec=HDUList)

    @patch("spx_tasks.l2b.check_l1_dqa")
    def test_run(self, mock_check_l1_dqa):

        # Mock each subtask on the instance directly
        self.task.remove_reference_pixels.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.apply_bad_pixel_mask.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.apply_dark_flat_field_and_photo_calib.run =\
            MagicMock(return_value=Struct(spximg_out=self.spximg))

        self.task.estimate_fine_astrometry.run =\
            MagicMock(return_value=Struct(spximg_out=self.spximg, astr_err=0.2))
        self.task.create_source_mask.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.detect_outlier_pixels.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.get_zodi_model.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.measure_psf.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.create_spectral_image.run = MagicMock(return_value=Struct(spximg_out=self.spximg))
        self.task.dqa_l2.run = MagicMock(return_value=Struct(spximg_out=self.spximg,
                                                             dqa_l2=None, idqr_obs=None))

        # Now run your task with mocked subtasks
        result = self.task.run(
            spximg=self.spximg,
            abs_gain_matrix=self.abs_gain_matrix,
            dark=self.dark,
            # sip=self.sip,
            srccat=self.srccat,
            # xtalk_coupling_matrix=self.xtalk_coupling_matrix,
            psf_avg=self.psf_avg,
            spectral_wcs=self.spectral_wcs
        )

        # Assertions
        self.assertIsInstance(result, Struct)
        self.assertTrue(hasattr(result, 'spximg_out'))
        self.assertTrue(hasattr(result, 'dqa_l2'))
        self.assertTrue(hasattr(result, 'idqr_obs'))

        mock_check_l1_dqa.assert_called_once_with(self.spximg)
        self.task.remove_reference_pixels.run.assert_called_once_with(self.spximg)
        self.task.apply_bad_pixel_mask.run.assert_called_once_with(self.spximg)
        self.task.apply_dark_flat_field_and_photo_calib.run.assert_called_once_with(self.spximg,
                                                                                    self.abs_gain_matrix,
                                                                                    self.dark)
        self.task.estimate_fine_astrometry.run.assert_called_once_with(self.spximg, self.srccat)


if __name__ == '__main__':
    unittest.main()
