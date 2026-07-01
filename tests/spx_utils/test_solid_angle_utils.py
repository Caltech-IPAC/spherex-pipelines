import unittest
import numpy as np
from astropy.wcs import WCS
from astropy.io import fits
from astropy.wcs import Sip
from spx_utils.solid_angle_utils import calculate_solid_angle, calculate_median_std_solid_angle


class TestSolidAngleCalculations(unittest.TestCase):

    def setUp(self):
        # Set up a simple WCS object with SIP distortion for testing
        header = fits.Header()
        header['CRPIX1'] = 1020.5
        header['CRPIX2'] = 1020.5
        header['CRVAL1'] = 0.0
        header['CRVAL2'] = 0.0
        header['CTYPE1'] = 'RA---TAN-SIP'
        header['CTYPE2'] = 'DEC--TAN-SIP'
        header['CD1_1'] = -0.0002777777778
        header['CD1_2'] = 0.0
        header['CD2_1'] = 0.0
        header['CD2_2'] = 0.0002777777778

        # SIP distortion coefficients
        a_order = 3
        b_order = 3
        a_coeff = np.zeros((a_order + 1, a_order + 1))
        b_coeff = np.zeros((b_order + 1, b_order + 1))

        # Mock distortion values
        a_coeff[1, 1] = 1.25e-5
        a_coeff[2, 0] = 2.5e-6
        b_coeff[1, 1] = -1.1e-5
        b_coeff[2, 0] = 1.6e-6

        # SIP object with distortion coefficients
        sip = Sip(a_coeff, b_coeff, np.zeros_like(a_coeff), np.zeros_like(b_coeff), [1020.5, 1020.5])

        # Initialize the WCS with the header and the SIP distortion
        self.test_wcs = WCS(header)
        self.test_wcs.sip = sip

        # Mock solid angle map for testing
        self.mock_solid_angle_map = np.random.uniform(1.0, 2.0, (2040, 2040))

    def test_calculate_solid_angle(self):
        solid_angle = calculate_solid_angle(self.test_wcs, array_id=1, plot=False)
        # Check type
        self.assertIsInstance(solid_angle, np.ndarray)
        # Check dimensions
        self.assertEqual(solid_angle.shape, (2040, 2040))
        # Check solid angle is positive
        self.assertGreater(np.mean(solid_angle), 0, "The mean solid angle should be positive")

    def test_calculate_median_std_solid_angle(self):
        median, std_dev = calculate_median_std_solid_angle(self.mock_solid_angle_map)
        # Check if both median and std_dev are floats
        self.assertIsInstance(median, float)
        self.assertIsInstance(std_dev, float)
        # Check if the median and std deviation are in reasonable ranges
        self.assertGreater(median, 1.0, "Median should be greater than 1.0")
        self.assertLess(median, 2.0, "Median should be less than 2.0")
        self.assertGreater(std_dev, 0, "Standard deviation should be positive")


if __name__ == '__main__':
    unittest.main()
