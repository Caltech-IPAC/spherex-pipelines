# Standard library imports
import copy

# Related third party imports
from astropy.time import TimeDelta
import astropy.units as u
import numpy as np

# Local application/library specific imports
from spx_data import PersistenceState
from spx_tasks import UpdatePersistenceConfig, UpdatePersistenceTask
from spx_tests import sample_data


def test_UpdatePersistence():
    config = UpdatePersistenceConfig()
    task = UpdatePersistenceTask(config=config)
    spximg = sample_data.get_sample_level1b_img()

    # Sample persistence state.
    persistences_init_state: PersistenceState = sample_data.get_sample_persistence_state(spximg)

    ########################
    # No Previous Exposure.
    ########################
    output_0 = task.run(spximg, persistences_init_state)
    # Module should only change the flags of processed SPHEREx image.
    assert output_0 is not None
    assert np.array_equal(spximg.data, output_0.spximg_out.data)
    assert spximg.data.shape == output_0.persistence_state_out._persist_model.b_prev1_data[0].shape
    assert output_0.persistence_state_out._persist_model.a_n.all() == 0  # No persistence initially.
    a_n0 = output_0.persistence_state_out._persist_model.a_n
    assert output_0.persistence_image.data.all() == 0

    ########################
    # One Previous Exposure.
    ########################
    persistence_state_0 = output_0.persistence_state_out

    # Create a new image for the next step in the time sequence.
    spximg_next = copy.deepcopy(spximg)

    # Calculate a new, later start time.
    # We'll use the end time of the previous exposure plus a 2-second slew time.
    new_start_time = spximg.getExposureEndTime() + TimeDelta(2.0 * u.s)

    # Update the metadata of the new image with the new start time.
    # The getExposureStartTime() method likely reads from 'MJD-BEG'.
    spximg_next.meta['MJD-BEG'] = new_start_time.mjd

    # Now, run the task with the new image.
    output_1 = task.run(spximg_next, persistence_state_0)

    assert np.any(np.not_equal(a_n0, output_1.persistence_state_out._persist_model.a_n))
