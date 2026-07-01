import unittest
from spx_data import SPSFlag  # Replace 'your_module' with the actual module name


class TestSPSFlag(unittest.TestCase):

    def test_enum_values(self):
        self.assertEqual(SPSFlag.DEEP_SOUTH, 1 << 0)
        self.assertEqual(SPSFlag.DEEP_NORTH, 1 << 1)
        self.assertEqual(SPSFlag.ALL_SKY, 1 << 2)
        self.assertEqual(SPSFlag.DOWNLINK, 1 << 3)
        self.assertEqual(SPSFlag.SAA, 1 << 4)
        self.assertEqual(SPSFlag.EARTH_STRAYLIGHT, 1 << 5)
        self.assertEqual(SPSFlag.MOON_STRAYLIGHT, 1 << 6)
        self.assertEqual(SPSFlag.SPECIAL, 1 << 7)

    def test_from_string_valid(self):
        self.assertEqual(SPSFlag.from_string("deep_south"), SPSFlag.DEEP_SOUTH)
        self.assertEqual(SPSFlag.from_string("all_sky,saa"), SPSFlag.ALL_SKY | SPSFlag.SAA)
        self.assertEqual(SPSFlag.from_string("special"), SPSFlag.SPECIAL)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError):
            SPSFlag.from_string("invalid_flag")
        with self.assertRaises(ValueError):
            SPSFlag.from_string("deep_south,unknown")

    def test_to_string(self):
        self.assertEqual(SPSFlag.to_string(SPSFlag.MOON_STRAYLIGHT), "moon_straylight")
        self.assertEqual(SPSFlag.to_string(SPSFlag.ALL_SKY | SPSFlag.SAA), "all_sky,saa")

    def test_round_trip_conversion(self):
        flags_str = "deep_south,saa"
        flags_int = SPSFlag.from_string(flags_str)
        self.assertEqual(SPSFlag.to_string(flags_int), flags_str)

    def test_is_deep_field(self):
        expected = [
            (SPSFlag.ALL_SKY, False),
            (SPSFlag.DEEP_SOUTH, True),
            (SPSFlag.DEEP_NORTH, True),
            (SPSFlag.DEEP_SOUTH | SPSFlag.DOWNLINK, True),
        ]
        for f, v in expected:
            assert SPSFlag.is_deep_field(f) == v

    def test_is_saa(self):
        expected = [
            (SPSFlag.SAA, True),
            (SPSFlag.DEEP_SOUTH, False),
            (SPSFlag.DEEP_NORTH, False),
            (SPSFlag.DOWNLINK, False),
            (SPSFlag.SPECIAL, False),
            (SPSFlag.ALL_SKY | SPSFlag.SAA, True),
        ]
        for f, v in expected:
            assert SPSFlag.is_saa(f) == v
