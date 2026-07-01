# Standard library imports
from pathlib import Path

# Related third party imports

# Local application/library specific imports
from spx_tasks import DetectOutlierPixelsConfig, DetectOutlierPixelsTask
from spx_tests import sample_data

test_root = Path(__file__).parent.parent


def test_DetectOutlierPixels():
    config = DetectOutlierPixelsConfig()
    task = DetectOutlierPixelsTask(config=config)
    spximg = sample_data.get_sample_l2_img()
    # Run task.
    output = task.run(spximg)
    assert spximg.data.shape == output.spximg_out.data.shape

    # Check num. of outlier flagged pixels is equal & reported in both headers
    img_hdr = output.spximg_out.header
    flg_hdr = output.spximg_out.flag_header
    assert "L2 N_OUTLIER" in list(img_hdr.keys()), "Keyword L2 N_OUTLIER missing from image header."
    assert "L2 N_OUTLIER" in list(flg_hdr.keys()), "Keyword L2 N_OUTLIER missing from flag header."
    ct_msg = f"L2 N_OUTLIER in image header ({img_hdr['L2 N_OUTLIER']}) is not equal to L2 N_OUTLIER in "\
             f"flag header ({flg_hdr['L2 N_OUTLIER']})."
    assert img_hdr['L2 N_OUTLIER'] == flg_hdr['L2 N_OUTLIER'], ct_msg
