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
        self.assertEqual(SPSFlag.GO_BACK_S12, 1 << 8)
        self.assertEqual(SPSFlag.GO_BACK_S34, 1 << 9)
        self.assertEqual(SPSFlag.GO_BACK_S56, 1 << 10)

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

    def test_go_back_flags_from_string(self):
        # go_back flags should also set ALL_SKY
        self.assertEqual(
            SPSFlag.from_string("go_back_s12"),
            SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY
        )
        self.assertEqual(
            SPSFlag.from_string("go_back_s34"),
            SPSFlag.GO_BACK_S34 | SPSFlag.ALL_SKY
        )
        self.assertEqual(
            SPSFlag.from_string("go_back_s56"),
            SPSFlag.GO_BACK_S56 | SPSFlag.ALL_SKY
        )

    def test_is_go_back(self):
        expected = [
            (SPSFlag.ALL_SKY, False),
            (SPSFlag.GO_BACK_S12, True),
            (SPSFlag.GO_BACK_S34, True),
            (SPSFlag.GO_BACK_S56, True),
            (SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY, True),
            (SPSFlag.ALL_SKY | SPSFlag.SAA, False),
        ]
        for f, v in expected:
            self.assertEqual(SPSFlag.is_go_back(f), v)

    def test_get_go_back_survey_id(self):
        self.assertIsNone(SPSFlag.get_go_back_survey_id(SPSFlag.ALL_SKY))
        self.assertEqual(SPSFlag.get_go_back_survey_id(SPSFlag.GO_BACK_S12), "S12")
        self.assertEqual(SPSFlag.get_go_back_survey_id(SPSFlag.GO_BACK_S34), "S34")
        self.assertEqual(SPSFlag.get_go_back_survey_id(SPSFlag.GO_BACK_S56), "S56")
        # With combined flags, GO_BACK_S12 takes precedence
        self.assertEqual(
            SPSFlag.get_go_back_survey_id(SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY),
            "S12"
        )
