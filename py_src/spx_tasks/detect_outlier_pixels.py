__all__ = ["DetectOutlierPixelsConfig", "DetectOutlierPixelsTask"]

# Standard library imports
import numpy as np

# Related third party imports
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from scipy.ndimage import median_filter

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, ensure_uncertainty_parentdata
from spx_tasks import add_history
from spx_utils import log_dqa_result
# ==============================================================================


def dqa_check_outlier_pixel_count(spximg: SPHERExImage, tolerance: float = 2.0) -> dict:
    """Check that the number of OUTLIER flagged pixels is within a factor
    of `tolerance` relative to the number of SOURCE, HOT, and COLD pixels.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Processed image containing flag plane.

    tolerance : `float`, optional
        Tolerance factor (default is 2.0).

    Returns
    -------
    result : `dict`
        Dictionary with "result" and "description" for Level2DQA.
    """
    num_outliers = np.sum((spximg.flags & Flag.OUTLIER) > 0)
    num_source = np.sum((spximg.flags & Flag.SOURCE) > 0)
    num_hot = np.sum((spximg.flags & Flag.HOT) > 0)
    num_cold = np.sum((spximg.flags & Flag.COLD) > 0)
    total_bad = num_source + num_hot + num_cold

    if total_bad == 0:
        result = {
            "result": "FAIL",
            "description": "No SOURCE, HOT, or COLD pixels found to validate against."
        }

    upper = total_bad * tolerance

    if num_outliers <= upper:
        result = {
            "result": "PASS",
            "description": f"OUTLIER pixel count ({num_outliers}) is within "
                           f"a factor {tolerance} of SOURCE+HOT+COLD pixels ({total_bad})."
        }
    else:
        result = {
            "result": "FAIL",
            "description": f"OUTLIER pixel count ({num_outliers}) is outside "
                           f"factor {tolerance} of SOURCE+HOT+COLD pixels ({total_bad})."
        }
    log_dqa_result("outliers dqa", result)
    return result


class DetectOutlierPixelsConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after detected outlier pixels.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class DetectOutlierPixelsConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=DetectOutlierPixelsConnections):

    n_channels = Field(
        doc="Number of channels a H2RG detector is divided into.",
        dtype=int,
        default=32)

    n_sigma = Field(
        "Variance threshold n * std",
        dtype=float,
        default=2.)

    kernel = Field(
        "Size of kernel window size",
        dtype=int,
        default=5)

    dqa_tolerance = Field(
        doc="Tolerance for number of flagged pixels.",
        dtype=float,
        default=2.0)


class DetectOutlierPixelsTask(pipeBase.PipelineTask):
    """Implementation of L2 Detect Outlier Pixels.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-020

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "detect-outlier-pixels"
    ConfigClass = DetectOutlierPixelsConfig
    full_name = "Detect Outlier Pixels"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-020 v1.4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:

        # Step 1: Apply 2D median filter to remove mean and low-frequency signal
        # Kernel size parameter means each pixel is replaced
        # by the median within a nxn pixel neighborhood
        median_filtered = median_filter(spximg.data, size=self.config.kernel)  # 5x5 kernel size
        filtered_image = spximg.data - median_filtered

        # Step 2: Apply the source and bad pixel mask (SOURCE, HOT, COLD flags)
        mask = spximg.flags.copy()
        # Exclude "bad" pixels
        combined_mask = (
            ((mask & Flag.SOURCE) == 0)  # Exclude source pixels
            & ((mask & Flag.HOT) == 0)  # Exclude hot pixels
            & ((mask & Flag.COLD) == 0)  # Exclude cold pixels
        )

        # Step 3: Select pixels exceeding n*sigma threshold based on variance map
        ensure_uncertainty_parentdata(spximg)
        variance_map = spximg.uncertainty.array
        std_map = np.sqrt(variance_map)
        threshold = self.config.n_sigma * std_map
        # Detect outliers among "good" pixels
        outliers = (filtered_image > threshold) & combined_mask

        # Step 4: Flag selected pixels as OUTLIER
        spximg_out = spximg.copy()
        spximg_out.setFlagsFromMask(outliers, Flag.OUTLIER)

        # DQA check to ensure the number of flagged pixels is reasonable
        # Use distinct calculations to independently check outlier count
        n_outlier = len(spximg_out.data[(spximg_out.flags & Flag.OUTLIER) > 0])
        numpix_outlier = np.sum((spximg_out.flags & Flag.OUTLIER) > 0)
        spximg_out.header["HIERARCH L2 N_OUTLIER"] = (n_outlier, "number of flagged outlier pixels")
        spximg_out.flag_header["HIERARCH L2 N_OUTLIER"] = (numpix_outlier,
                                                           "number of flagged outlier pixels")
        # TODO Include N_STREAK when implemented.
        # spximg_out.header["HIERARCH L2 N_STREAK"] =
        # ("not implemented", "number of pixels flagged streak")
        self.log.info(f"Number of pixels flagged as outliers: {n_outlier}")

        # Perform additional validation
        _ = dqa_check_outlier_pixel_count(spximg_out)

        return pipeBase.Struct(spximg_out=spximg_out)
