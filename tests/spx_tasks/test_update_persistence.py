# Standard library imports
import copy

# Related third party imports
from astropy.time import TimeDelta
import astropy.units as u
import numpy as np

# Local application/library specific imports
from spx_data import PersistenceState, Flag  # <-- 1. IMPORT Flag
from spx_tasks import UpdatePersistenceConfig, UpdatePersistenceTask
from spx_tests import sample_data
from spx_utils import l1_utils


def test_UpdatePersistence():
    config = UpdatePersistenceConfig()
    task = UpdatePersistenceTask(config=config)
    spximg = sample_data.get_sample_level1b_img()

    # Sample persistence state.
    persistences_init_state: PersistenceState = sample_data.get_sample_persistence_state(spximg)

    # Test for Step 3 logic (bloom/snowball) in _overwrite_b_n.
    # This logic is triggered on the *next* run (output_1) based on the
    # flags from the *current* run (output_0, which uses spximg).
    # We'll set a pixel (10, 10) in the *first* image to trigger
    # one of the new rules (Rule 3: ~Ftr, ~Fov, Ferr, ~Fbloom, Fsnow).
    # Only set this flag if it exists, otherwise the test would fail
    # when the fallback logic is (correctly) being used.
    test_flag = Flag.SUR_ERROR | Flag.SNOWBALL
    spximg.flags[10, 10] = test_flag
    # Also set data to 0, so persistence is only non-zero *because* of the flag
    spximg.data[10, 10] = 0.0
    # Set a control pixel to 0 data and 0 flags
    spximg.flags[11, 11] = 0
    spximg.data[11, 11] = 0.0
    # run_new_logic_test = True

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

    # Check that persistence only applied within the active area

    # Active-area mask
    actv_area = l1_utils.active_area_range(spximg_next, include_refs=False)
    active_mask = np.zeros_like(spximg_next.data, dtype=bool)
    active_mask[actv_area] = True

    # Should have NO PERSIST flags outside active area
    persist_flags = (output_1.spximg_out.flags & Flag.PERSIST) > 0
    assert not np.any(persist_flags[~active_mask]), \
        "PERSIST flags found outside active area"

    # Active area mask shape of true values should match persistence image shape
    persist_current = output_1.persistence_image.data
    expected_active_rows = np.sum(np.any(active_mask, axis=1))
    expected_active_cols = np.sum(np.any(active_mask, axis=0))

    assert persist_current.shape == (expected_active_rows, expected_active_cols), \
        "Active area does not match expected shape in outputted persistence image."
