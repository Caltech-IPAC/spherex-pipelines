__all__ = ["ApplyDarkFlatFieldAndPhotoCalibConfig", "ApplyDarkFlatFieldAndPhotoCalibTask"]

# Standard library imports
import logging

# Related third party imports
from dataclasses import dataclass, field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, ensure_uncertainty_parentdata
from spx_tasks import add_history
from spx_utils import log_dqa_result
# ==============================================================================


def dqa_check_flat_field(spximg: SPHERExImage,
                         abs_gain_matrix: SPHERExImage,
                         threshold: float = 0.30) -> dict:
    """Check that the median of the flat-field matches expected calibration.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Image that received the photometric calibration.
    abs_gain_matrix : `SPHERExImage`
        Flat-field used in calibration (units: MJy/sr per e-/s).
    threshold : `float`
        Allowed fractional deviation from expected 1/cal_value.

    Returns
    -------
    dqa_result : `dict`
        Result dictionary with 'result' and 'description' for Level2DQA.
    """
    array_id = spximg.getArrayId()
    median = np.median(abs_gain_matrix.data)
    sigma = np.std(abs_gain_matrix.data)

    # Expected 1/calibration value per array
    cal_value = ArrayCalValues().cal_values.get(array_id)
    if cal_value is None:
        msg = f"Calibration value not found for array_id {array_id}"
        logging.error("DQA L2: " + msg)
        return {"result": "FAIL", "description": msg}

    expected_ratio = 1 / cal_value
    lower_bound = expected_ratio - (sigma * threshold)
    upper_bound = expected_ratio + (sigma * threshold)

    if lower_bound <= median <= upper_bound:
        msg = (f"Median flat-field = {median:.5f}, within {threshold*100:.0f}% of "
               f"expected {expected_ratio:.5f}.")
        result = {"result": "PASS", "description": msg}
    else:
        msg = (f"Median flat-field = {median:.5f} outside {threshold*100:.0f}% "
               f"range of expected {expected_ratio:.5f}.")
        result = {"result": "FAIL", "description": msg}
    log_dqa_result("appluy_gain_matrix", result)
    return result


@dataclass
class ArrayCalValues:
    """CAL values for instrument to physical unit conversion.

    Notes
    -----
    Coded here because we will not have access
    to these numbers with in-orbit data taking.
    """
    cal_values: dict = field(default_factory=lambda: {
        1: 8.286700087641215,
        2: 8.38949569816677,
        3: 7.896646949035173,
        4: 9.867400104358904,
        5: 2.0516901757300006,
        6: 1.682176698588182
    })


class ApplyDarkFlatFieldAndPhotoCalibConnections(pipeBase.PipelineTaskConnections,
                                                 dimensions=("instrument", "exposure"),
                                                 defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    abs_gain_matrix = cT.PrerequisiteInput(
        name="abs_gain_matrix",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    dark = cT.PrerequisiteInput(
        name="dark",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after photometric calibration applied.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class ApplyDarkFlatFieldAndPhotoCalibConfig(pipeBase.PipelineTaskConfig,
                                            pipelineConnections=ApplyDarkFlatFieldAndPhotoCalibConnections):
    dqa_threshold = Field(
        "Percentage tolerance for dqa test",
        dtype=float,
        default=.30)


class ApplyDarkFlatFieldAndPhotoCalibTask(pipeBase.PipelineTask):
    """Implementation of L2 Apply Photometric Calibration.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-014

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "apply-photometric-calibration"
    ConfigClass = ApplyDarkFlatFieldAndPhotoCalibConfig
    # shortened from "Apply Diffuse Flat Field Correction and Photometric Calibration"
    # to fit into FITS HISTORY headers for module versions applied.
    full_name = "Apply Flat Field and Photometric Calibration"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-014 v1.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage,
            abs_gain_matrix: SPHERExImage,
            dark: SPHERExImage) -> pipeBase.Struct:

        ensure_uncertainty_parentdata(spximg)
        spx_reduced = spximg.subtract(operand=dark,  # e-/s
                                      propagate_uncertainties=None,
                                      handle_flags="ff", handle_meta="ff",
                                      handle_layers="ff")
        spx_reduced.uncertainty = spximg.uncertainty

        # Scale image with calibration.
        spximg_out = spx_reduced.multiply(operand=abs_gain_matrix.data,  # MJy/sr / e-/s
                                          handle_flags="ff",
                                          handle_meta="ff",
                                          handle_layers="ff")
        # Convert image data unit from e/s to MJy/sr
        spximg_out.unit *= abs_gain_matrix.unit
        # Convert variance unit from (e/s)**2 to (MJy/sr)**2
        spximg_out.uncertainty.unit *= abs_gain_matrix.unit**2
        # Output SPHERExImage

        # L2 DQA
        _ = dqa_check_flat_field(spximg_out, abs_gain_matrix)
        spximg_out.header["BUNIT"] = "MJy/sr"

        return pipeBase.Struct(spximg_out=spximg_out)
