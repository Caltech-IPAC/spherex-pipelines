# Standard library imports
import numpy as np

# Related third party imports

# Local application/library specific imports
from spx_tasks import FilterByMaskConfig, FilterByMaskTask
from spx_tests import sample_data
from spx_data import Flag, SPHERExImage


# Little hack to give a way to pass in generated SPHERExImages without filepath
class MockDeferredDatasetHandle:
    def __init__(self, spx_img: SPHERExImage):
        self.spx_img = spx_img

    @property
    def shape(self):
        return self.spx_img.shape

    def get(self) -> SPHERExImage:
        return self.spx_img


def test_filter_by_mask():
    # Has flags for SATURATED and COSMICRAY
    # Need at least 2 images for testing
    spx_imgs = []
    for _ in range(2):
        spx_imgs.append(MockDeferredDatasetHandle(sample_data.get_sample_spx_img()))
    # Add in more flags to trigger threshold
    flags = np.full(spx_imgs[0].shape, Flag.TRANSIENT | Flag.SOURCE)
    mask_cosmicray = flags & Flag.TRANSIENT > 0
    mask_source = flags & Flag.SOURCE > 0

    # Set the flags from the mask in the image
    spx_imgs[0].get().setFlagsFromMask(mask_cosmicray, Flag.TRANSIENT)
    spx_imgs[1].get().setFlagsFromMask(mask_source, Flag.SOURCE)

    config1 = FilterByMaskConfig()
    config1.mask = Flag.SOURCE
    task1 = FilterByMaskTask(config=config1)
    config2 = FilterByMaskConfig()
    config2.mask = Flag.TRANSIENT
    task2 = FilterByMaskTask(config=config2)
    config3 = FilterByMaskConfig()
    config3.mask_frac_threshold = .75
    config3.mask = Flag.OVERFLOW | Flag.NONLINEAR
    task3 = FilterByMaskTask(config=config3)

    output1 = task1.run(spx_imgs)
    output2 = task2.run(spx_imgs)
    output3 = task3.run(spx_imgs)

    # Verify the output
    assert task1.config.mask == Flag.SOURCE
    assert len(output1.filtered_dataset) == 1
    assert task2.config.mask == Flag.TRANSIENT
    assert len(output2.filtered_dataset) == 1
    assert task3.config.mask == Flag.OVERFLOW | Flag.NONLINEAR
    assert task3.config.mask_frac_threshold == .75
    assert len(output3.filtered_dataset) == 2
