# Standard library imports
import unittest

# Related third party imports

# Local application/library specific imports
from spx_utils.utils import get_overflow_threshold


class TestGetOverflowThreshold(unittest.TestCase):
    def test_primary_key_found(self):
        """Test that the primary key value is returned when it exists."""
        header = {'KOVR_A': 15000}
        result = get_overflow_threshold(header, 'KOVR_A')
        self.assertEqual(result, 60000)

    def test_primary_key_missing_fallback_found(self):
        """Test that the fallback key value is returned when no primary key."""
        header = {'OVERFLOW': 20000}
        result = get_overflow_threshold(header, 'OVERFLOW')
        self.assertEqual(result, 80000)

    def test_all_keys_missing(self):
        """Test that the default value is returned when both keys are missing."""
        header = {}
        result = get_overflow_threshold(header, 'KOVR_A')
        self.assertEqual(result, 30000)


# Running the tests
if __name__ == '__main__':
    unittest.main()
