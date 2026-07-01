# Standard library imports
import unittest

# Related third party imports
import numpy as np
from pathlib import Path

# Local application/library specific imports
from spx_data import ReadoutParameters
from spx_tests import sample_data
from spx_tasks.update_persistence import (PersistenceModel, PersistenceModelState,
                                          UpdatePersistenceConfig)

test_root = Path(__file__).parent.parent


class TestPersistenceModel(unittest.TestCase):

    def initialize(self):
        # Create a sample SPHERExImage.
        self.spximg = sample_data.get_sample_level1b_img(bad_flags=True)
        rp = ReadoutParameters.from_metadata(self.spximg.meta)

        # Dummy initial values for testing.
        c_n = np.zeros((rp.Jframe, rp.Kframe, 6), dtype=np.float32)
        tau_n = np.ones((rp.Jframe, rp.Kframe, 6), dtype=np.float32)
        i_thresh = np.double(500.0)
        i_reset = np.double(0.1)

        # Config parameters
        config = UpdatePersistenceConfig()

        # Define the initial state for the PersistenceModel.
        initial_model_state = PersistenceModelState(
            a_n=np.zeros((rp.Jframe, rp.Kframe, 6)),
            b_prev2=np.zeros((rp.Jframe, rp.Kframe, 6)),
            spximg_prev=np.empty((2, rp.Jframe, rp.Kframe)),
            spximg_prev_flg=np.empty((rp.Jframe, rp.Kframe)),
            num_prev=0,
            prev1_end_time=0.,
            prev2_end_time=0.)

        self.persistence_model = PersistenceModel(c_n,
                                                  tau_n,
                                                  i_thresh,
                                                  i_reset,
                                                  initial_model_state,
                                                  config)

    def test_compute_persistence_current(self):
        """Simple test for computing persistence current from model."""
        # Initialize only once.
        self.initialize()

        # Run module: No previous exposure.
        persist_current_0, persist_flg_0 = self.persistence_model.compute_persistence_current(self.spximg,
                                                                                              t_slew_2_1=0.0,
                                                                                              t_int=1.)
        # No previous exposure means no persistence current, and no flags set.
        self.assertTrue(persist_current_0.shape == self.spximg.data.shape)
        self.assertEqual(persist_flg_0, None)

        # Run module: One previous exposure.
        self.persistence_model._state.prev1_end_time = 0.7
        persist_current_1, persist_flg_1 = self.persistence_model.compute_persistence_current(self.spximg,
                                                                                              t_slew_2_1=0.0,
                                                                                              t_int=1.)
        self.assertTrue(persist_current_1.shape == self.spximg.data.shape)
        self.assertTrue(persist_flg_1.shape == self.spximg.data.shape)

        # Run module: Two previous exposures.
        self.persistence_model._state.prev1_end_time = 0.5
        self.persistence_model._state.prev2_end_time = 0.7
        persist_current_2, persist_flg_2 = self.persistence_model.compute_persistence_current(self.spximg,
                                                                                              t_slew_2_1=0.7,
                                                                                              t_int=1.)
        self.assertTrue(persist_current_2.shape == self.spximg.data.shape)
        self.assertTrue(persist_flg_2.shape == self.spximg.data.shape)


if __name__ == '__main__':
    unittest.main()
