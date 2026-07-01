# Standard library imports
from pathlib import Path

# Related third party imports

# Local application/library specific imports
from spx_data import SPHERExImage, spherex_image_reader
from spx_tasks import EstimateSubpixelGainConfig, EstimateSubpixelGainTask


test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


def test_EstimateSubpixelGain():
    config = EstimateSubpixelGainConfig()
    task = EstimateSubpixelGainTask(config=config)
    spximg = get_sample_img()
    output = task.run(spximg)
    assert spximg.data.shape == output.image.data.shape
