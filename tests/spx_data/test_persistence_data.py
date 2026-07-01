# Standard library imports
import unittest

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_data import PersistenceState
from spx_tests.sample_data import get_sample_persistence_state


class TestPersistenceData(unittest.TestCase):

    # Setup test and create a PersistenceData instance.
    persistence_state: PersistenceState = get_sample_persistence_state()

    def test_update_state_extensions(self):
        # Retrieve the original data of the ``a_n`` state extension
        original_state_data = self.persistence_state.get_extension_data("a_n").copy()

        # Modify the data (example: add 1 to every element)
        updated_state_data = original_state_data + 0.1

        # Update the PersistenceData instance with the modified data
        self.persistence_state.update_state_extension(updated_state_data, "a_n")

        # Verify that the state extension data has been modified
        new_state_data = self.persistence_state.get_extension_data("a_n")
        self.assertTrue(np.array_equal(new_state_data, updated_state_data))

        # Verify that data in other (non-state) extensions remains unchanged
        for ext in self.persistence_state.required_extensions:
            if ext != "a_n":
                with self.subTest(ext=ext):
                    original_data = self.persistence_state.get_extension_data(ext)
                    unchanged_data = self.persistence_state.hdulist[ext].data
                    self.assertTrue(np.array_equal(original_data, unchanged_data))


if __name__ == '__main__':
    unittest.main()
