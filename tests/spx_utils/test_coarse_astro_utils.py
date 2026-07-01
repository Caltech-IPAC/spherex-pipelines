import unittest
import numpy as np
from astropy.wcs import WCS
from spx_utils.coarse_astro_utils import get_ra_dec_lon_lat_from_quat
from spx_utils import pc_to_cd, cd_to_pc
from spx_utils.coarse_astro_utils import (position_angle_from_cd)


# Quaternion (in scalar-last format) from c-kernel
Q = np.array([0.34589122, -0.38147555, -0.80224746, 0.30205081])

# PEF data for respective observation
# PlannedObservation(
# ra=<Angle 152.83174269 deg>, dec=<Angle 28.0142005 deg>,
# pa_eq=<Angle -21.56766724 deg>,
# timespan=Timespan(begin=astropy.time.Time
# ('2025-12-27 15:38:53.189000', scale='tai', format='iso'),
# end=astropy.time.Time('2025-12-27 15:40:48.306500', scale='tai', format='iso')),
# obs_id='2025W52_2V_0573_1_4',
# target_id=8132,
# flags=1,
# quaternions={
# 'q1': 0.34587240937553293, 'q2': -0.38147956544676936,
# 'q3': -0.8022558250498873, 'q4': 0.3020450442447821},
# lat_ecl=15.698000000000002, lon_ecl=144.6728971962617, pa_ecl=0.0,
# zenith=[-6117.90660445299, -376.0192924033395, 3436.9140271955066],
# moon=[381990.149634427, 18834.64954011774, 18831.40633999084],
# sun=[14821396.75384142, -134300388.8024592, -58220330.6563158],
# vel=[-26.532983363999243, -1.4260828311212352, 5.2415277132694245],
# in_sband=False),


class TestCoarseAstroUtils(unittest.TestCase):
    ra_eq = 0.
    dec_eq = 0.
    pa_eq = 0.

    # Mock WCS
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [1020.5, 1020.5]  # Reference pixel
    wcs.wcs.crval = [290.8570984668, 21.05484201822]   # Reference RA/Dec
    wcs.wcs.cdelt = [1.0, 1.0]  # Pixel scale in degrees
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.pc = np.array([[0.0016590823771923, -0.00044634750461639],
                           [0.00038180932408143, 0.0016461105796416]])

    cdelt = np.array([6.15/3600, 6.15/3600])
    pc = np.array([[-0.00154728, 0.0007241],
                   [-0.0007241, -0.00154728]])
    cd = np.array([[-2.64327000e-06, 1.23700417e-06],
                   [-1.23700417e-06, -2.64327000e-06]])

    def test_get_ra_dec_lon_lat_from_quat(self):
        # Function to get RA, Dec, Lon, Lat, PA
        self.ra_eq, self.dec_eq, self.pa_eq = get_ra_dec_lon_lat_from_quat(Q)

        # Expected values from PEF
        expected_ra_eq = 152.83174269
        expected_dec_eq = 28.0142005
        expected_pa_eq = 21.56766724

        # Assert that the values are within reasonable tolerances
        self.assertAlmostEqual(self.ra_eq, expected_ra_eq, delta=0.2)
        self.assertAlmostEqual(self.dec_eq, expected_dec_eq, delta=0.2)
        self.assertAlmostEqual(self.pa_eq%360., expected_pa_eq, delta=0.1)

    def test_pc_to_cd(self):
        # Test conversion from PC to CD formalism
        calculated_cd = pc_to_cd(self.pc, self.cdelt)
        assert np.allclose(calculated_cd, self.cd, rtol=1e-11, atol=1e-11), "Arrays are not equal to 1e-11"

    def test_cd_to_pc(self):
        # Test conversion from CD to PC formalism
        calculated_pc = cd_to_pc(self.cd, self.cdelt)
        assert np.allclose(calculated_pc, self.pc, rtol=1e-11, atol=1e-11), "Arrays are not equal to 1e-11"

    def test_position_angle_from_cd(self):
        # Test WCS using given PC matrix
        pc_matrix = np.array([[-0.00113601066195, -0.00123308205054],
                              [-0.00127880132542, 0.00118297318887]])
        # Expected value from reference
        expected_pa = 46.18814118828724
        # Call the function
        pa = position_angle_from_cd(pc_matrix)
        # Check result
        self.assertAlmostEqual(pa, expected_pa, places=5,
                               msg=f"Expected PA={expected_pa} deg, got {pa} deg")


if __name__ == "__main__":
    unittest.main()
