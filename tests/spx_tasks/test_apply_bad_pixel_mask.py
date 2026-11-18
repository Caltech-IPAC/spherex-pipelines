# Standard library imports

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_tasks import (ApplyBadPixelMaskConfig, ApplyBadPixelMaskTask)
from spx_tests import sample_data

hot_testpix = []
cold_testpix = []


def make_random_cold(spximg: SPHERExImage) -> SPHERExImage:
    """Randomly sets cold pixels.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Input SPHEREx image.

    Returns
    -------
    `SPHERExImage`
        Image with 3 random pixels less than -2 eps.

    """
    rng = np.random.default_rng(193)
    x, y = spximg.shape
    for i in range(3):
        row = rng.integers(0, x)
        col = rng.integers(0, y)
        spximg.data[row][col] *= -10
        cold_testpix.append((row, col))
    return spximg


def make_random_hot(spximg: SPHERExImage) -> SPHERExImage:
    """Randomly sets hot pixels.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Input SPHEREx image.

    Returns
    -------
    `SPHERExImage`
        Image with 3 random pixels greater than 10000 eps.

    """
    rng = np.random.default_rng(199)
    x, y = spximg.shape
    for i in range(3):
        row = rng.integers(0, x)
        col = rng.integers(0, y)
        spximg.data[row][col] *= 10000
        hot_testpix.append((row, col))
    return spximg


def test_ApplyBadPixelMask():
    config = ApplyBadPixelMaskConfig()
    task = ApplyBadPixelMaskTask(config=config)
    spximg = sample_data.get_sample_l2_img()
    make_random_cold(spximg)
    make_random_hot(spximg)
    # Run task.
    output = task.run(spximg)
    assert spximg.data.shape == output.spximg_out.data.shape
    # Check number of flags match number of hot/cold pix.
    assert len(output.spximg_out.flags[np.where(output.spximg_out.flags & Flag.COLD)]) == 3
    assert len(output.spximg_out.flags[np.where(output.spximg_out.flags & Flag.HOT)]) == 3
    # Check output flagged indices are the ones we explicitly set.
    in_cold_pix = np.array(cold_testpix)
    out_cold_pix = np.argwhere(output.spximg_out.flags & Flag.COLD)
    assert np.array_equiv(np.sort(in_cold_pix, axis=0), np.sort(out_cold_pix, axis=0))
    in_hot_pix = np.array(hot_testpix)
    out_hot_pix = np.argwhere(output.spximg_out.flags & Flag.HOT)
    assert np.array_equiv(np.sort(in_hot_pix, axis=0), np.sort(out_hot_pix, axis=0))
