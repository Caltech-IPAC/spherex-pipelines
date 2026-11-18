__all__ = ["ApplyBadPixelMaskConfig", "ApplyBadPixelMaskTask"]

# Standard library imports

# Related third party imports
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_tasks import add_history
from spx_utils import log_dqa_result

# ==============================================================================


def dqa_check_hot_cold_pixel_fraction(spximg: SPHERExImage, threshold: float = 1) -> dict:
    """Check that the fraction of hot/cold (BAD) pixels is below a given threshold.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The input SPHEREx image.
    threshold : `float`
        Maximum acceptable percentage of bad pixels (0-20%).

    Returns
    -------
    result_dict : `dict`
        Dictionary with DQA result and string description.

    """
    # Define what constitutes a "bad" pixel
    bad_mask = (spximg.flags & (Flag.NONFUNC | Flag.HOT | Flag.COLD)) > 0
    n_bad = np.sum(bad_mask)
    n_total = np.prod(spximg.data.shape)

    percent_bad = 100 * n_bad / n_total
    passed = percent_bad <= threshold

    result = "PASS" if passed else "FAIL"
    description = (
        f"{n_bad} bad pixels ({percent_bad:.2f}%) found. "
        f"Threshold is {threshold:.2f}%."
    )

    result_dict = {"result": result, "description": description}
    log_dqa_result("bad pixel fraction dqa", result_dict)
    return result_dict


class ApplyBadPixelMaskConnections(pipeBase.PipelineTaskConnections,
                                   dimensions=("instrument", "exposure"),
                                   defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after bad pixel mask applied.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class ApplyBadPixelMaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=ApplyBadPixelMaskConnections):
    """Configuration for ApplyBadPixelMask task."""

    neg_threshold = Field(
        "Negative cutoff threshold. Unphysical.",
        dtype=float,
        default=-2)
    pos_threshold = Field(
        "Positive cutoff threshold. Caused by hot pixels.",
        dtype=float,
        default=1e4)
    dqa_threshold = Field(
        "Percentage threshold or dqa.",
        dtype=float,
        default=1.0)


class ApplyBadPixelMaskTask(pipeBase.PipelineTask):
    """Implementation of L2 Apply Bad Pixel Mask.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-013

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "apply-bad-pixel-mask"
    ConfigClass = ApplyBadPixelMaskConfig
    full_name = "Apply Bad Pixel Mask"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-013 v1.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage) -> pipeBase.Struct:

        # Flag NEGATIVE pixels: < neg cut th = -2 eps
        neg_mask = spximg.data < self.config.neg_threshold
        spximg.setFlagsFromMask(neg_mask, Flag.COLD)
        # Flag HOT pixels: > pos th = 1e4 eps but not saturated
        pos_mask = spximg.data > self.config.pos_threshold
        pos_mask = pos_mask & ~spximg.getFlagsAsMask(Flag.OVERFLOW)
        spximg.setFlagsFromMask(pos_mask, Flag.HOT)

        # Check the fraction of hot/cold pixels
        _ = dqa_check_hot_cold_pixel_fraction(spximg)
        # Add statistics to header
        npix_flagged_hot = np.sum((spximg.flags & Flag.HOT) > 0)
        spximg.header["HIERARCH L2 N_HOT"] = (npix_flagged_hot,
                                              "number of pixels flagged hot")
        npix_flagged_cold = np.sum((spximg.flags & Flag.COLD) > 0)
        spximg.header["HIERARCH L2 N_COLD"] = (npix_flagged_cold,
                                               "number of pixels flagged cold")

        # Define the "bad" pixel mask (matching DQA function)
        bad_mask = (spximg.flags & (Flag.NONFUNC | Flag.HOT | Flag.COLD)) > 0
        not_bad_mask = ~bad_mask

        # Get masks for transient and persistence
        transient_mask = (spximg.flags & Flag.TRANSIENT) > 0
        persistence_mask = (spximg.flags & Flag.PERSIST) > 0

        # Count "transient" pixels that are "not bad"
        n_transient_not_bad = np.count_nonzero(transient_mask & not_bad_mask)

        # Count "persistence" pixels that are "not bad"
        n_persistence_not_bad = np.count_nonzero(persistence_mask & not_bad_mask)

        # Add counts to the header
        spximg.header["HIERARCH L2 N_TRAN_NB"] = (n_transient_not_bad,
                                                  "number of not bad transient pixels")
        spximg.header["HIERARCH L2 N_PERS_NB"] = (n_persistence_not_bad,
                                                  "number of persistence and not bad pixels")

        return pipeBase.Struct(spximg_out=spximg)
