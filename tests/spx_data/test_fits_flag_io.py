"""Tests for fits_flag_io module."""
import unittest
from astropy.io import fits

from spx_data import Flag
from spx_data.fits_flag_io import (
    FlagHeaderFormat,
    write_flag_headers,
    read_flag_headers,
    read_flag_headers_as_bitmask,
)


class TestWriteFlagHeaders(unittest.TestCase):
    """Tests for write_flag_headers function."""

    def test_write_none_writes_nothing(self):
        """flags=None should write nothing (no flags to propagate)."""
        header = fits.Header()
        write_flag_headers(header, format=FlagHeaderFormat.MP)  # flags=None by default

        # No EXTTYPE, no MP_ keys
        self.assertNotIn("EXTTYPE", header)
        mp_keys = [k for k in header if "MP_" in k]
        self.assertEqual(len(mp_keys), 0)

    def test_write_all_flags_explicit(self):
        """Write all flags by passing list(Flag)."""
        header = fits.Header()
        write_flag_headers(header, list(Flag))

        # Check EXTTYPE is set
        self.assertEqual(header["EXTTYPE"], "MASK")

        # Check that all flags are written (MSK format uses triplets)
        flag_count = len(list(Flag))
        msk_name_keys = [k for k in header if k.startswith("MSKN")]
        self.assertEqual(len(msk_name_keys), flag_count)

    def test_write_from_bitmask(self):
        """Write flags from int bitmask."""
        header = fits.Header()
        bitmask = Flag.HOT | Flag.COLD | Flag.OVERFLOW

        write_flag_headers(header, bitmask)

        # Should have exactly 3 flags (MSK format)
        msk_name_keys = [k for k in header if k.startswith("MSKN")]
        self.assertEqual(len(msk_name_keys), 3)

        # Check specific flags in MSK format
        # MSK keyword number matches bit position:
        # OVERFLOW is bit 1, HOT is bit 10, COLD is bit 11
        self.assertEqual(header["MSKN0001"], "OVERFLOW")
        self.assertEqual(header["MSKM0001"], 1 << 1)   # 2
        self.assertEqual(header["MSKN0010"], "HOT")
        self.assertEqual(header["MSKM0010"], 1 << 10)  # 1024
        self.assertEqual(header["MSKN0011"], "COLD")
        self.assertEqual(header["MSKM0011"], 1 << 11)  # 2048

    def test_write_from_dict(self):
        """Write flags from dict (MP format)."""
        header = fits.Header()
        flag_defs = {"HOT": 10, "COLD": 11}

        write_flag_headers(header, flag_defs, format=FlagHeaderFormat.MP)

        self.assertEqual(header["MP_HOT"], 10)
        self.assertEqual(header["MP_COLD"], 11)

    def test_write_from_list(self):
        """Write flags from list of Flag enums (MP format)."""
        header = fits.Header()
        write_flag_headers(header, [Flag.HOT, Flag.COLD], format=FlagHeaderFormat.MP)

        self.assertEqual(header["MP_HOT"], 10)
        self.assertEqual(header["MP_COLD"], 11)

    def test_write_empty_bitmask(self):
        """Empty bitmask (0) should not write anything."""
        header = fits.Header()
        write_flag_headers(header, 0, format=FlagHeaderFormat.MP)

        # No EXTTYPE, no MP_ keys
        self.assertNotIn("EXTTYPE", header)
        mp_keys = [k for k in header if "MP_" in k]
        self.assertEqual(len(mp_keys), 0)

    def test_write_no_exttype(self):
        """set_exttype=False should skip EXTTYPE."""
        header = fits.Header()
        write_flag_headers(header, Flag.HOT, set_exttype=False)

        self.assertNotIn("EXTTYPE", header)
        # MSK format is default - HOT is bit 10, so keyword is MSKN0010
        self.assertIn("MSKN0010", header)
        self.assertEqual(header["MSKN0010"], "HOT")

    def test_hierarch_for_long_names(self):
        """HIERARCH should be added for keywords > 8 characters (MP format)."""
        header = fits.Header()
        # MISSING_DATA has 12 chars, so MP_MISSING_DATA = 15 chars -> needs HIERARCH
        write_flag_headers(header, Flag.MISSING_DATA, format=FlagHeaderFormat.MP)

        # astropy automatically handles HIERARCH, check the value is there
        self.assertEqual(header["MP_MISSING_DATA"], 9)

    def test_write_msk_format(self):
        """Write MSK format triplets."""
        header = fits.Header()
        write_flag_headers(header, Flag.HOT | Flag.COLD, format=FlagHeaderFormat.MSK)

        # Check EXTTYPE
        self.assertEqual(header["EXTTYPE"], "MASK")

        # Check MSK triplets exist
        # MSK keyword number matches bit position: HOT is bit 10, COLD is bit 11
        self.assertEqual(header["MSKN0010"], "HOT")
        self.assertEqual(header["MSKM0010"], 1 << 10)  # 1024

        self.assertEqual(header["MSKN0011"], "COLD")
        self.assertEqual(header["MSKM0011"], 1 << 11)  # 2048

    def test_flag_descriptions_in_comments(self):
        """Flag descriptions should be written as comments (MP format)."""
        header = fits.Header()
        write_flag_headers(header, Flag.HOT, format=FlagHeaderFormat.MP)

        # Check comment contains description
        comment = header.comments["MP_HOT"]
        self.assertEqual(comment, "Hot pixel")


class TestReadFlagHeaders(unittest.TestCase):
    """Tests for read_flag_headers function."""

    def test_read_mp_format(self):
        """Read MP_ keywords."""
        header = fits.Header()
        header["MP_HOT"] = (10, "Hot pixel")
        header["MP_COLD"] = (11, "Cold pixel")

        flag_defs = read_flag_headers(header)

        self.assertEqual(flag_defs["HOT"], 10)
        self.assertEqual(flag_defs["COLD"], 11)

    def test_read_with_hierarch(self):
        """Read flags that use HIERARCH prefix."""
        header = fits.Header()
        header["MP_HOT"] = 10
        header["HIERARCH MP_MISSING_DATA"] = 9  # Long name requires HIERARCH

        flag_defs = read_flag_headers(header)

        self.assertEqual(flag_defs["HOT"], 10)
        self.assertEqual(flag_defs["MISSING_DATA"], 9)

    def test_read_msk_format(self):
        """Read MSK format triplets."""
        header = fits.Header()
        # MSK keyword number matches bit position: HOT is bit 10, COLD is bit 11
        header["MSKN0010"] = "HOT"
        header["MSKM0010"] = 1024  # 2^10
        header["MSKD0010"] = "Hot pixel"
        header["MSKN0011"] = "COLD"
        header["MSKM0011"] = 2048  # 2^11
        header["MSKD0011"] = "Cold pixel"

        flag_defs = read_flag_headers(header, format=FlagHeaderFormat.MSK)

        self.assertEqual(flag_defs["HOT"], 10)
        self.assertEqual(flag_defs["COLD"], 11)

    def test_read_msk_format_with_gaps(self):
        """Read MSK format with gaps in numbering (bit positions are not contiguous)."""
        header = fits.Header()
        # Simulate flags at bit 1, bit 10, and bit 20 (gaps in numbering)
        header["MSKN0001"] = "OVERFLOW"
        header["MSKM0001"] = 1 << 1  # 2
        header["MSKD0001"] = "Overflow"
        header["MSKN0010"] = "HOT"
        header["MSKM0010"] = 1 << 10  # 1024
        header["MSKD0010"] = "Hot pixel"
        header["MSKN0020"] = "BADSTRIPE"
        header["MSKM0020"] = 1 << 20
        header["MSKD0020"] = "Bad stripe"

        flag_defs = read_flag_headers(header, format=FlagHeaderFormat.MSK)

        self.assertEqual(len(flag_defs), 3)
        self.assertEqual(flag_defs["OVERFLOW"], 1)
        self.assertEqual(flag_defs["HOT"], 10)
        self.assertEqual(flag_defs["BADSTRIPE"], 20)

    def test_auto_detect_mp_format(self):
        """Auto-detect MP_ format."""
        header = fits.Header()
        header["MP_HOT"] = 10

        flag_defs = read_flag_headers(header)  # No format specified
        self.assertEqual(flag_defs["HOT"], 10)

    def test_auto_detect_msk_format(self):
        """Auto-detect MSK format."""
        header = fits.Header()
        # MSK keyword number matches bit position: HOT is bit 10
        header["MSKN0010"] = "HOT"
        header["MSKM0010"] = 1024

        flag_defs = read_flag_headers(header)  # No format specified
        self.assertEqual(flag_defs["HOT"], 10)

    def test_read_empty_header(self):
        """Empty header returns empty dict."""
        header = fits.Header()
        flag_defs = read_flag_headers(header)

        self.assertEqual(flag_defs, {})

    def test_read_empty_header_with_remove(self):
        """Empty header with remove_from_header=True returns empty dict."""
        header = fits.Header()
        header["CUSTOM"] = "value"  # Non-flag keyword

        flag_defs = read_flag_headers(header, remove_from_header=True)

        self.assertEqual(flag_defs, {})
        # Non-flag keyword should be preserved
        self.assertEqual(header["CUSTOM"], "value")


class TestReadFlagHeadersAsBitmask(unittest.TestCase):
    """Tests for read_flag_headers_as_bitmask function."""

    def test_read_as_bitmask(self):
        """Read flags and convert to bitmask."""
        header = fits.Header()
        header["MP_HOT"] = 10
        header["MP_COLD"] = 11

        bitmask = read_flag_headers_as_bitmask(header)

        self.assertTrue(bitmask & Flag.HOT)
        self.assertTrue(bitmask & Flag.COLD)
        self.assertFalse(bitmask & Flag.OVERFLOW)

    def test_read_empty_as_bitmask(self):
        """Empty header returns 0."""
        header = fits.Header()
        bitmask = read_flag_headers_as_bitmask(header)

        self.assertEqual(bitmask, 0)

    def test_unknown_flags_ignored(self):
        """Unknown flag names should be ignored."""
        header = fits.Header()
        header["MP_HOT"] = 10
        header["HIERARCH MP_UNKNOWN_FLAG"] = 99  # Not a valid Flag (needs HIERARCH)

        bitmask = read_flag_headers_as_bitmask(header)

        # Only HOT should be in the bitmask
        self.assertTrue(bitmask & Flag.HOT)
        self.assertEqual(bitmask, Flag.HOT)


class TestRoundTrip(unittest.TestCase):
    """Tests for write then read roundtrip."""

    def test_roundtrip_mp_format_bitmask(self):
        """Write then read should preserve bitmask in MP_ format."""
        original = Flag.HOT | Flag.COLD | Flag.TRANSIENT

        header = fits.Header()
        write_flag_headers(header, original)
        recovered = read_flag_headers_as_bitmask(header)

        self.assertEqual(original, recovered)

    def test_roundtrip_mp_format_dict(self):
        """Write then read should preserve dict in MP_ format."""
        original = {"HOT": 10, "COLD": 11, "TRANSIENT": 0}

        header = fits.Header()
        write_flag_headers(header, original)
        recovered = read_flag_headers(header)

        self.assertEqual(original, recovered)

    def test_roundtrip_msk_format(self):
        """Write then read should preserve flags in MSK format."""
        original = Flag.HOT | Flag.COLD

        header = fits.Header()
        write_flag_headers(header, original, format=FlagHeaderFormat.MSK)
        recovered = read_flag_headers_as_bitmask(header, format=FlagHeaderFormat.MSK)

        self.assertEqual(original, recovered)

    def test_roundtrip_all_flags(self):
        """Roundtrip all flags."""
        header = fits.Header()
        write_flag_headers(header, list(Flag))  # Explicitly pass all flags

        recovered_dict = read_flag_headers(header)

        # Should have all Flag enum members
        for flag in Flag:
            self.assertIn(flag.name, recovered_dict)


class TestFormatCompatibility(unittest.TestCase):
    """Tests for format compatibility and header cleanup."""

    def test_same_data_different_formats(self):
        """Same flags written in different formats should read the same."""
        flags = Flag.HOT | Flag.COLD | Flag.OVERFLOW

        # Write in MP_ format
        header_mp = fits.Header()
        write_flag_headers(header_mp, flags, format=FlagHeaderFormat.MP)

        # Write in MSK format
        header_msk = fits.Header()
        write_flag_headers(header_msk, flags, format=FlagHeaderFormat.MSK)

        # Read both
        result_mp = read_flag_headers_as_bitmask(header_mp)
        result_msk = read_flag_headers_as_bitmask(header_msk)

        # Should be the same
        self.assertEqual(result_mp, result_msk)
        self.assertEqual(result_mp, flags)

    def test_remove_from_header_msk(self):
        """remove_from_header=True should remove MSK keywords after reading."""
        header = fits.Header()
        header["CUSTOM"] = "value"
        header["MSKN0010"] = "HOT"
        header["MSKM0010"] = 1024
        header["MSKD0010"] = "Hot pixel"

        flag_defs = read_flag_headers(header, remove_from_header=True)

        # Flag definitions should be read correctly
        self.assertEqual(flag_defs["HOT"], 10)

        # MSK keywords should be removed
        self.assertNotIn("MSKN0010", header)
        self.assertNotIn("MSKM0010", header)
        self.assertNotIn("MSKD0010", header)

        # Non-flag keywords should be preserved
        self.assertEqual(header["CUSTOM"], "value")

    def test_remove_from_header_mp(self):
        """remove_from_header=True should remove MP_ keywords after reading."""
        header = fits.Header()
        header["CUSTOM"] = "value"
        header["MP_HOT"] = 10
        header["MP_COLD"] = 11

        flag_defs = read_flag_headers(header, remove_from_header=True)

        # Flag definitions should be read correctly
        self.assertEqual(flag_defs["HOT"], 10)
        self.assertEqual(flag_defs["COLD"], 11)

        # MP_ keywords should be removed
        self.assertNotIn("MP_HOT", header)
        self.assertNotIn("MP_COLD", header)

        # Non-flag keywords should be preserved
        self.assertEqual(header["CUSTOM"], "value")


if __name__ == "__main__":
    unittest.main()
