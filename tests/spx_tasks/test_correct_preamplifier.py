# Standard library imports
from pathlib import Path

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_tasks import CorrectPreamplifierConfig, CorrectPreamplifierTask
from spx_tests import get_sample_level1b_img
from spx_data import SPHERExImage, Flag
from spx_tests import get_sample_level1_img

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = "processed_raw_202304251454_20230420200431_A.fits"
    return SPHERExImage.read(sample_file)


def test_CorrectPreamplifier():
    config = CorrectPreamplifierConfig()
    task = CorrectPreamplifierTask(config=config)
    # spximg = get_sample_level1_img()
    spximg = get_sample_level1b_img()
    output = task.run(spximg)
    truth_data = np.full(spximg.shape, 1, dtype=np.int32)
    assert np.allclose(output.spximg_out.data, truth_data)


def test_zero_pixel_nan_handling():
    # Get a small fake sample image with metadata
    spximg = get_sample_level1_img()

    # Modify data and flags for a few specific test pixels
    data = spximg.data
    flags = spximg.flags

    # (0,0): SUR_ERROR only => NaN unconditionally
    data[0, 0] = 0
    flags[0, 0] = np.uint32(Flag.SUR_ERROR)

    # (0,1): SUR_ERROR + TRANSIENT => NaN if zero
    data[0, 1] = 0
    flags[0, 1] = np.uint32(Flag.SUR_ERROR | Flag.TRANSIENT)

    # (0,2): SUR_ERROR + TRANSIENT + OVERFLOW => NaN only if zero
    data[0, 2] = 0
    flags[0, 2] = np.uint32(Flag.SUR_ERROR | Flag.TRANSIENT | Flag.OVERFLOW)

    # (0,3): SUR_ERROR + OVERFLOW => NaN unconditionally
    data[0, 3] = 456
    flags[0, 3] = np.uint32(Flag.SUR_ERROR | Flag.OVERFLOW)

    # (1,0): data without SUR_ERROR => should NOT become NaN
    data[1, 0] = 1
    flags[1, 0] = np.uint32(Flag.TRANSIENT)

    # Run the task
    config = CorrectPreamplifierConfig()
    config.no_ref_correction = True
    task = CorrectPreamplifierTask(config=config)
    result = task.run(spximg)
    corrected = result.spximg_out.data

    # Assertions
    assert np.isnan(corrected[0, 0]), "SUR_ERROR pixel should be NaN"
    assert np.isnan(corrected[0, 1]), "SUR_ERROR + TRANSIENT with zero should be NaN"
    assert np.isnan(corrected[0, 2]), "SUR_ERROR + TRANSIENT + OVERFLOW should be NaN"
    assert np.isnan(corrected[0, 3]), "SUR_ERROR + OVERFLOW should be NaN"
    assert corrected[1, 0] == 1, "Unflagged pixel should remain unmodified"
