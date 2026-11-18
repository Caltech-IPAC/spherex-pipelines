# Standard library imports
from pathlib import Path

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_tasks import DetectOutlierPixelsConfig, DetectOutlierPixelsTask
from spx_tests import sample_data
from spx_data import Flag

test_root = Path(__file__).parent.parent


def test_DetectOutlierPixels():
    config = DetectOutlierPixelsConfig()
    task = DetectOutlierPixelsTask(config=config)
    spximg = sample_data.get_sample_l2_img()

    # Run task.
    output = task.run(spximg)
    spximg_out = output.spximg_out

    assert spximg.data.shape == spximg_out.data.shape

    # Check num. of outlier flagged pixels is equal & reported in both headers
    img_hdr = spximg_out.header

    assert "L2 N_OUTLIER" in list(img_hdr.keys()), "Keyword L2 N_OUTLIER missing from image header."

    # New streak keywords should also be present and agree
    assert "L2 SATSTRKS_PXS" in list(img_hdr.keys()), "Keyword L2 SATSTRKS_PXS missing from image header."
    assert "L2 SATSTRKS" in list(img_hdr.keys()), "Keyword L2 SATSTRKS missing from image header."

    # Header counts should match actual flag counts
    n_outlier = int(np.sum((spximg_out.flags & Flag.OUTLIER) > 0))
    n_streak = int(np.sum((spximg_out.flags & Flag.STREAK) > 0))

    assert img_hdr["L2 N_OUTLIER"] == n_outlier
    assert img_hdr["L2 SATSTRKS_PXS"] == n_streak

    # SATSTRKS should be a non-negative integer-like value
    assert img_hdr["L2 SATSTRKS"] >= 0

    # STREAK pixels should not also be marked as OUTLIER by this task
    overlap = ((spximg_out.flags & Flag.OUTLIER) > 0) & ((spximg_out.flags & Flag.STREAK) > 0)
    assert not np.any(overlap), "Pixels were flagged as both OUTLIER and STREAK."
