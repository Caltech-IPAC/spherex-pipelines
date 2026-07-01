__all__ = ["FilterByMaskTask", "FilterByMaskConfig"]

# Standard library imports
import numpy as np
from alive_progress import alive_bar

# Related third party imports
import lsst.pipe.base as pipeBase
from lsst.pex.config import Config, Field
from lsst.daf.butler import DeferredDatasetHandle

# Local application/library specific imports
from spx_data import Flag


class FilterByMaskConfig(Config):
    mask_frac_threshold = Field(
        doc="Exclude images with more than this fraction of pixels masked.",
        dtype=float,
        default=.15)
    mask = Field(
        doc="Mask to filter by, check `spx_data.spx_flags.Flag` for available masks.",
        dtype=int,
        default=Flag.SOURCE)


class FilterByMaskTask(pipeBase.Task):
    _DefaultName = "filter_by_mask"
    ConfigClass = FilterByMaskConfig

    def run(self, dataset: list[DeferredDatasetHandle]):
        """Filters dataset if image is some percentage masked.

        Parameters
        ----------
        dataset
            Deferred dataset of images.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Output with component:

            - ``filtered_dataset`` : `list` [`lsst.daf.butler.DeferredDatasetHandle`]
                Deferred dataset list filtered by specified mask threshold.

        """
        # This function is taken from the task estimate_flat_field with some changes
        filtered_dataset = []
        num_removed = 0
        with alive_bar(len(dataset), title='Filtering masked images: ') as bar:
            for img in dataset:
                spximg = img.get()
                # Exclude images with more than .15 fraction of pixels masked
                mask = spximg.getFlagsAsMask(self.config.mask)
                mask_frac = np.sum(mask) / mask.size

                if mask_frac < self.config.mask_frac_threshold:
                    filtered_dataset.append(img)
                else:
                    num_removed += 1
                bar()

        print(f"Removed {num_removed} images because over mask threshold")

        # Metadata is stored in butler repository
        self.metadata.add("num_start_images", len(dataset))
        self.metadata.add("num_removed", num_removed)
        output = pipeBase.Struct(
            filtered_dataset=filtered_dataset
        )

        return output
