# Standard library imports
import unittest

# Related third party imports
from astropy.wcs import WCS
from astropy.modeling import models
import numpy as np
from astropy.modeling.models import Polynomial2D


# Local application/library specific imports
from spx_utils.update_distortion_utils import (sip_coefficients,
                                               initialize_polynomial_from_sip)
from spx_tasks.update_field_distortion import (calculate_solid_angle,
                                               update_wcs_with_sip)


def create_test_wcs():
    # Define a minimal FITS header with the provided WCS details
    from astropy.io import fits
    header = fits.Header()
    header['WCSAXES'] = 2
    header['CRPIX1'] = 1020.5
    header['CRPIX2'] = 1020.5
    header['PC1_1'] = 0.0014856218059215
    header['PC1_2'] = 0.00082742916780765
    header['PC2_1'] = -0.00082554954462085
    header['PC2_2'] = 0.0014890665714916
    header['CDELT1'] = 1.0
    header['CDELT2'] = 1.0
    header['CRVAL1'] = 162.9266220324
    header['CRVAL2'] = 49.53707669978
    header['CTYPE1'] = 'RA---TAN-SIP'
    header['CTYPE2'] = 'DEC--TAN-SIP'
    # SIP coefficients
    header['A_ORDER'] = 3
    header['A_0_2'] = -3.5431756234769E-10
    header['A_0_3'] = 1.20169973465429E-15
    header['A_1_1'] = 4.38766638735491E-06
    header['A_1_2'] = 7.1807507945561E-10
    header['A_2_0'] = -9.2370548865629E-10
    header['A_2_1'] = 1.58425430006051E-15
    header['A_3_0'] = 6.24859629361623E-10
    header['B_ORDER'] = 3
    header['B_0_2'] = 2.19167105580446E-06
    header['B_0_3'] = 5.45413509135254E-10
    header['B_1_1'] = -6.9902693464759E-10
    header['B_1_2'] = 5.21099452910304E-15
    header['B_2_0'] = 7.26451447196016E-06
    header['B_2_1'] = 7.28646487715075E-10
    header['B_3_0'] = 4.52664054796037E-15
    header['AP_ORDER'] = 3
    header['AP_0_1'] = 8.41440804624229E-11
    header['AP_0_2'] = 3.52944342945549E-10
    header['AP_0_3'] = -4.3606179673634E-15
    header['AP_1_0'] = -1.9026943658083E-06
    header['AP_1_1'] = -4.3591335640205E-06
    header['AP_1_2'] = -6.8548074392435E-10
    header['AP_2_0'] = 9.19966340315233E-10
    header['AP_2_1'] = -2.1882856045183E-14
    header['AP_3_0'] = -5.9004606953669E-10
    header['BP_ORDER'] = 3
    header['BP_0_1'] = -2.1391909720378E-06
    header['BP_0_2'] = -2.1779789575975E-06
    header['BP_0_3'] = -5.3279224326151E-10
    header['BP_1_0'] = 5.22455026719184E-11
    header['BP_1_1'] = 6.95537070465801E-10
    header['BP_1_2'] = -1.7956149745671E-14
    header['BP_2_0'] = -7.2417840703483E-06
    header['BP_2_1'] = -6.294613764517E-10
    header['BP_3_0'] = -2.2966619791405E-14
    # Create the WCS object
    wcs = WCS(header)
    return wcs


class TestWCSFunctions(unittest.TestCase):
    def setUp(self):
        self.wcs = create_test_wcs()
        self.poly_order = 3
        self.a_coeffs, self.b_coeffs, self.ap_coeffs, self.bp_coeffs =\
            sip_coefficients(self.wcs, self.poly_order)
        self.p_init_x, self.p_init_y, self.p_init_xp, self.p_init_yp =\
            initialize_polynomial_from_sip(self.wcs, self.poly_order)

    def test_sip_coefficients(self):
        # Test if the coefficients arrays are correctly sized
        self.assertEqual(self.a_coeffs.shape, (self.poly_order + 1, self.poly_order + 1))
        self.assertEqual(self.b_coeffs.shape, (self.poly_order + 1, self.poly_order + 1))
        self.assertEqual(self.ap_coeffs.shape, (self.poly_order + 1, self.poly_order + 1))
        self.assertEqual(self.bp_coeffs.shape, (self.poly_order + 1, self.poly_order + 1))

    def test_initialize_polynomial_from_sip(self):
        # Test if the polynomial models are correctly initialized
        self.assertIsInstance(self.p_init_x, models.Polynomial2D)
        self.assertIsInstance(self.p_init_y, models.Polynomial2D)
        self.assertIsInstance(self.p_init_xp, models.Polynomial2D)
        self.assertIsInstance(self.p_init_yp, models.Polynomial2D)

        # Test if the polynomials are of the correct order
        self.assertEqual(self.p_init_x.degree, self.poly_order)
        self.assertEqual(self.p_init_y.degree, self.poly_order)
        self.assertEqual(self.p_init_xp.degree, self.poly_order)
        self.assertEqual(self.p_init_yp.degree, self.poly_order)

        # Coefficient checks for all indices
        for i in range(self.poly_order + 1):
            for j in range(self.poly_order + 1):
                # Check coefficients within the polynomial order
                if i + j <= self.poly_order:
                    self.assertAlmostEqual(getattr(self.p_init_x, f'c{i}_{j}'),
                                           self.a_coeffs[i, j], places=15)
                    self.assertAlmostEqual(getattr(self.p_init_y, f'c{i}_{j}'),
                                           self.b_coeffs[i, j], places=15)
                    self.assertAlmostEqual(getattr(self.p_init_xp, f'c{i}_{j}'),
                                           self.ap_coeffs[i, j], places=15)
                    self.assertAlmostEqual(getattr(self.p_init_yp, f'c{i}_{j}'),
                                           self.bp_coeffs[i, j], places=15)


class TestCalculateSolidAngle(unittest.TestCase):
    """Calculates the solid angle per pixel"""

    def setUp(self):
        self.wcs = create_test_wcs()

    def test_solid_angle_calculation(self):
        """Tests calculation of solid angles"""
        solid_angle = calculate_solid_angle(self.wcs, array_id=1, plot=False)
        self.assertIsInstance(solid_angle, np.ndarray)


class TestUpdateWCSWithSIP(unittest.TestCase):
    """Tests correct assignment of calculated SIP coefficients to the WCS."""

    def setUp(self):
        # Create polynomial models with mock coefficients
        self.poly_x = Polynomial2D(degree=3)
        self.poly_y = Polynomial2D(degree=3)
        self.poly_xp = Polynomial2D(degree=3)
        self.poly_yp = Polynomial2D(degree=3)

        # Set example coefficients for the test, using random values for simplicity
        # Forward transformation coefficients
        self.poly_x.c0_2 = 0.1
        self.poly_x.c0_3 = 0.2
        self.poly_x.c1_1 = 0.3
        self.poly_x.c1_2 = 0.4
        self.poly_x.c2_0 = 0.5
        self.poly_x.c2_1 = 0.6
        self.poly_x.c3_0 = 0.7
        self.poly_y.c0_2 = 0.8
        self.poly_y.c0_3 = 0.9
        self.poly_y.c1_1 = 1.0
        self.poly_y.c1_2 = 1.1
        self.poly_y.c2_0 = 1.2
        self.poly_y.c2_1 = 1.3
        self.poly_y.c3_0 = 1.4

        # Inverse transformation coefficients
        self.poly_xp.c0_1 = 1.5
        self.poly_xp.c0_2 = 1.6
        self.poly_xp.c0_3 = 1.7
        self.poly_xp.c1_0 = 1.8
        self.poly_xp.c1_1 = 1.9
        self.poly_xp.c1_2 = 2.0
        self.poly_xp.c2_0 = 2.1
        self.poly_xp.c2_1 = 2.2
        self.poly_xp.c3_0 = 2.3
        self.poly_yp.c0_1 = 2.4
        self.poly_yp.c0_2 = 2.5
        self.poly_yp.c0_3 = 2.6
        self.poly_yp.c1_0 = 2.7
        self.poly_yp.c1_1 = 2.8
        self.poly_yp.c1_2 = 2.9
        self.poly_yp.c2_0 = 3.0
        self.poly_yp.c2_1 = 3.1
        self.poly_yp.c3_0 = 3.2

        # Setup a WCS object with a mock SIP attribute
        self.wcs = WCS()
        self.wcs.wcs.crpix = [1020.5, 1020.5]
        self.wcs.sip = None  # SIP not set initially

    def test_update_wcs_with_sip(self):
        """Check that each matrix is properly constructed
           and assigned to the WCS SIP attribute."""
        updated_wcs = update_wcs_with_sip(self.poly_x, self.poly_y, self.poly_xp, self.poly_yp, self.wcs)

        # Check that the WCS SIP is not None
        self.assertIsNotNone(updated_wcs.sip)

        # Verify that coefficients are correctly assigned to the SIP object
        # Forward transformation (poly_x to sip_coeff_a, poly_y to sip_coeff_b)
        np.testing.assert_almost_equal(updated_wcs.sip.a[0, 2], self.poly_x.c0_2)
        np.testing.assert_almost_equal(updated_wcs.sip.a[0, 3], self.poly_x.c0_3)
        np.testing.assert_almost_equal(updated_wcs.sip.a[1, 1], self.poly_x.c1_1)
        np.testing.assert_almost_equal(updated_wcs.sip.a[1, 2], self.poly_x.c1_2)
        np.testing.assert_almost_equal(updated_wcs.sip.a[2, 0], self.poly_x.c2_0)
        np.testing.assert_almost_equal(updated_wcs.sip.a[2, 1], self.poly_x.c2_1)
        np.testing.assert_almost_equal(updated_wcs.sip.a[3, 0], self.poly_x.c3_0)

        np.testing.assert_almost_equal(updated_wcs.sip.b[0, 2], self.poly_y.c0_2)
        np.testing.assert_almost_equal(updated_wcs.sip.b[0, 3], self.poly_y.c0_3)
        np.testing.assert_almost_equal(updated_wcs.sip.b[1, 1], self.poly_y.c1_1)
        np.testing.assert_almost_equal(updated_wcs.sip.b[1, 2], self.poly_y.c1_2)
        np.testing.assert_almost_equal(updated_wcs.sip.b[2, 0], self.poly_y.c2_0)
        np.testing.assert_almost_equal(updated_wcs.sip.b[2, 1], self.poly_y.c2_1)
        np.testing.assert_almost_equal(updated_wcs.sip.b[3, 0], self.poly_y.c3_0)

        # Inverse transformation (poly_xp to sip_coeff_ap, poly_yp to sip_coeff_bp)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[0, 1], self.poly_xp.c0_1)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[0, 2], self.poly_xp.c0_2)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[0, 3], self.poly_xp.c0_3)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[1, 0], self.poly_xp.c1_0)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[1, 1], self.poly_xp.c1_1)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[1, 2], self.poly_xp.c1_2)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[2, 0], self.poly_xp.c2_0)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[2, 1], self.poly_xp.c2_1)
        np.testing.assert_almost_equal(updated_wcs.sip.ap[3, 0], self.poly_xp.c3_0)

        np.testing.assert_almost_equal(updated_wcs.sip.bp[0, 1], self.poly_yp.c0_1)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[0, 2], self.poly_yp.c0_2)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[0, 3], self.poly_yp.c0_3)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[1, 0], self.poly_yp.c1_0)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[1, 1], self.poly_yp.c1_1)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[1, 2], self.poly_yp.c1_2)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[2, 0], self.poly_yp.c2_0)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[2, 1], self.poly_yp.c2_1)
        np.testing.assert_almost_equal(updated_wcs.sip.bp[3, 0], self.poly_yp.c3_0)


if __name__ == '__main__':
    unittest.main()
