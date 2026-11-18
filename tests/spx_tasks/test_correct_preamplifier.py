# Standard library imports
from pathlib import Path

# Related third party imports
import numpy as np
import pytest

# Local application/library specific imports
from spx_tasks import CorrectPreamplifierConfig, CorrectPreamplifierTask
from spx_tests import get_sample_level1b_img
from spx_data import SPHERExImage, Flag
from spx_tests import get_sample_level1_img
from spx_utils import active_area_range

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


def test_missing_reference_pixels_tolerated():
    """Reference pixels lost in transmission (Flag.MISSING_DATA, NaN data)
    pass the reference flag check and are excluded from the reference
    medians, so the correction is unchanged everywhere else.
    """
    config = CorrectPreamplifierConfig()
    baseline = CorrectPreamplifierTask(config=config).run(
        get_sample_level1b_img()).spximg_out.data

    spximg = get_sample_level1b_img()
    irows, icols = active_area_range(spximg)
    # a few pixels in the top and bottom reference row bands
    missing = [(0, icols.start + 3),
               (1, icols.start + 8),
               (irows.stop, icols.start + 3)]
    for r, c in missing:
        spximg.flags[r, c] = np.uint32(Flag.MISSING_DATA)
        spximg.data[r, c] = np.nan

    output = CorrectPreamplifierTask(config=config).run(spximg).spximg_out.data

    # A NaN leaking into a median would poison whole readout-channel
    # columns; only the missing pixels themselves may differ.
    diff = ~np.isclose(output, baseline, equal_nan=True)
    assert set(zip(*np.nonzero(diff))) == set(missing)


def test_unflagged_reference_pixel_raises():
    """A reference pixel with neither REFERENCE nor MISSING_DATA still
    fails the flag check.
    """
    spximg = get_sample_level1b_img()
    _, icols = active_area_range(spximg)
    spximg.flags[0, icols.start + 5] = np.uint32(0)

    task = CorrectPreamplifierTask(config=CorrectPreamplifierConfig())
    with pytest.raises(ValueError, match="Ref pixels"):
        task.run(spximg)


def test_all_missing_references_flag_phanmiss():
    """When every reference pixel is missing, the uncorrectable active
    pixels are set to NaN and flagged PHANMISS, matching the phantom
    correction behavior.
    """
    spximg = get_sample_level1b_img()
    irows, icols = active_area_range(spximg)
    for rows in (slice(None, irows.start), slice(irows.stop, None)):
        spximg.flags[rows, icols.start:icols.stop] = np.uint32(Flag.MISSING_DATA)
        spximg.data[rows, icols.start:icols.stop] = np.nan

    task = CorrectPreamplifierTask(config=CorrectPreamplifierConfig())
    output = task.run(spximg).spximg_out

    active_flags = output.flags[irows.start:irows.stop, icols.start:icols.stop]
    assert np.all((active_flags & Flag.PHANMISS) > 0), \
        "Uncorrectable active pixels must be flagged PHANMISS"
    active_data = output.data[irows.start:irows.stop, icols.start:icols.stop]
    assert np.isnan(active_data).all(), \
        "Uncorrectable active pixels must be set to NaN"
