import unittest
from spx_data import Flag, flag2bit, flag_description, PhotometryFlag, phot_flag_description


class TestFlagModule(unittest.TestCase):
    def test_flag_enum_values(self):
        """Test that the flag values match the expected bit shifts."""
        self.assertEqual(Flag.TRANSIENT, 1 << 0)
        self.assertEqual(Flag.OVERFLOW, 1 << 1)
        self.assertEqual(Flag.SUR_ERROR, 1 << 2)
        self.assertEqual(Flag.PHANTOM, 1 << 4)
        self.assertEqual(Flag.GHOST_EXT, 1 << 24)
        self.assertEqual(Flag.STREAK, 1 << 25)
        self.assertEqual(PhotometryFlag.CONTAINS_BAD_PIXEL, 1 << 32)
        self.assertEqual(PhotometryFlag.FIT_ERROR, 1 << 33)

    def test_flag2bit(self):
        """Test the flag2bit function."""
        self.assertEqual(flag2bit(Flag.TRANSIENT), 0)
        self.assertEqual(flag2bit(Flag.OVERFLOW), 1)
        self.assertEqual(flag2bit(Flag.SUR_ERROR), 2)
        self.assertEqual(flag2bit(Flag.GHOST_EXT), 24)
        self.assertEqual(flag2bit(Flag.STREAK), 25)
        self.assertEqual(flag2bit(PhotometryFlag.CONTAINS_BAD_PIXEL), 32)
        self.assertEqual(flag2bit(PhotometryFlag.FIT_ERROR), 33)

        # Test with integer input
        self.assertEqual(flag2bit(1), 0)
        self.assertEqual(flag2bit(4), 2)
        self.assertEqual(flag2bit(32), 5)
        self.assertEqual(flag2bit(4294967296), 32)
        self.assertEqual(flag2bit(8589934592), 33)

    def test_flag_description(self):
        """Test the flag_description function."""
        self.assertEqual(flag_description(Flag.TRANSIENT), "Transient detected during SUR")
        self.assertEqual(flag_description(Flag.OVERFLOW), "Overflow detected during SUR")
        self.assertEqual(flag_description(Flag.SUR_ERROR), "Error in onboard processing")
        self.assertEqual(flag_description(Flag.GHOST_EXT), "Pixel affected by a bright source outside FOV")
        self.assertEqual(flag_description(Flag.STREAK), "Pixel affected by a streak")

        # Test invalid flag
        with self.assertRaises(ValueError):
            flag_description(999999)  # Invalid flag

    def test_phot_flag_description(self):
        """Test the phot_flag_description function."""
        self.assertEqual(phot_flag_description(PhotometryFlag.CONTAINS_BAD_PIXEL),
                         "1 or more of the source pixels have a harmful flag set")
        self.assertEqual(phot_flag_description(PhotometryFlag.FIT_ERROR),
                         "Tractor fit failed to return a valid flux or flux error")

        # Test invalid flag
        with self.assertRaises(ValueError):
            flag_description(999999)  # Invalid flag

    def test_flag_combination(self):
        """Test combinations of flags using bitwise operations."""
        combined_flag = Flag.TRANSIENT | Flag.OVERFLOW
        self.assertTrue(combined_flag & Flag.TRANSIENT)
        self.assertTrue(combined_flag & Flag.OVERFLOW)
        self.assertFalse(combined_flag & Flag.SUR_ERROR)

        phot_combined_flag = Flag.STREAK | PhotometryFlag.CONTAINS_BAD_PIXEL
        self.assertTrue(phot_combined_flag & Flag.STREAK)
        self.assertTrue(phot_combined_flag & PhotometryFlag.CONTAINS_BAD_PIXEL)
        self.assertFalse(phot_combined_flag & Flag.OVERFLOW)


if __name__ == "__main__":
    unittest.main()
