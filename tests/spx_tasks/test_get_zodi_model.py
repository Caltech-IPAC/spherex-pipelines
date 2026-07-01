import pytest
# Standard library imports
from pathlib import Path

# Related third party imports

# Local application/library specific imports
from spx_data import SPHERExImage, Zodi, spherex_image_reader
from spx_tasks import GetZodiModelConfig, GetZodiModelTask, hasSPHERExSimulator

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


@pytest.mark.skipif(not hasSPHERExSimulator, reason="SPHEREx Simulator is not installed")
def test_GetZodiModel():
    config = GetZodiModelConfig()
    config.zodi_subsample = 1
    config.sparse_bandpass = False
    task = GetZodiModelTask(config=config)
    spximg = get_sample_img()
    output = task.run(spximg)
    zodi = output.spximg_out.zodi
    assert isinstance(zodi, Zodi)
    assert spximg.data.shape == output.spximg_out.data.shape
    assert spximg.data.shape == zodi.zodi_data.shape
