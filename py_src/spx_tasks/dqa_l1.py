__all__ = ["DQAL1Config", "DQAL1Task", "check_l1_dqa"]

# Standard library imports
import copy
import logging

# Related third party imports
from lsst.pex.config import Field, DictField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, Stats, Level1DQA, Flag, FLAG_NAME_ALIASES
from spx_tasks import add_history
from spx_utils import active_area_range, flags_summary, log_threshold_test, nan_pixels_flagged
from spx_utils.odir_utils import get_odir

# ==============================================================================


class DQAL1Connections(pipeBase.PipelineTaskConnections,
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
        doc="Output image after variance map created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    dqa_l1 = cT.Output(
        name="dqa_l1",
        doc="Level 1 DQA record.",
        storageClass="Level1DQA",
        dimensions=("instrument", "exposure"),
    )
    odir_obs = cT.Output(
        name="odir_obs",
        doc="ODIR per observation.",
        storageClass="OdirObs",
        dimensions=("instrument", "exposure"),
    )


class DQAL1Config(pipeBase.PipelineTaskConfig, pipelineConnections=DQAL1Connections):
    """Configuration for DQAL1 (Level 1 Data Quality Assessment) task."""

    metadata_ranges = DictField(
        doc="Acceptable ranges of metadata values.",
        keytype=str,
        itemtype=float,
        # TODO: populate default with actual desired values
        default={}
    )

    img_stats_ranges = DictField(
        doc="Acceptable ranges for the mean, median, and RMS.",
        keytype=str,
        itemtype=float,
        # TODO: populate default with actual desired values
        default={}
    )

    ncr_threshold = Field(
        doc="Acceptable number of cosmic rays.",
        dtype=float,
        # TODO: populate default with actual desired values
        default=0.0
    )

    flags_thresholds = DictField(
        doc="Acceptable number of flags raised.",
        keytype=str,
        itemtype=float,
        # TODO: populate default with actual desired values
        default={}
    )

    nan_percentage_threshold = Field(
        doc="Percentage of NaN pixels threshold.",
        dtype=float,
        default=80.0
    )

    nan_flag_mask = Field(
        doc="Bitmask of flags that justify NaN or Inf values in L1.",
        dtype=int,
        default=(Flag.MISSING_DATA | Flag.NONFUNC | Flag.SUR_ERROR | Flag.PHANMISS)
    )

    max_transient_pix = Field(
        doc="Maximum number of transient pixels allowed in all-sky images "
            "before trigerring IDQR re-observe flag.",
        dtype=int,
        default=435000)


class DQAL1Task(pipeBase.PipelineTask):
    """Implementation of L1 Perform Data Quality Assessment for Level 1.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-009

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "dqa-l1"
    ConfigClass = DQAL1Config
    full_name = "Perform Data Quality Assessment for Level 1"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-009 v1.3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:
        # Select photosensitive slice of data
        photo_range = active_area_range(spximg)
        photo_img = spximg[photo_range]
        # Set result to True (Pass) to begin with; if any checks fail, set to False
        dqa_result = True

        # Check for unflagged NaNs in the photo_img
        nan_result = nan_pixels_flagged(photo_img.data,
                                        spximg.flags[photo_range],
                                        nan_flags=self.config.nan_flag_mask,
                                        check_inf=True)

        nan_check = {
            "result": "PASS" if nan_result else "FAIL",
            "description": "All NaN or Inf pixels are properly flagged."
            if nan_result else "Some NaN or Inf pixels are unflagged."
        }
        log_threshold_test(self.log.logger, nan_check, warning_message="NaN/Inf flag mismatch in image data.")
        if nan_check["result"] == "FAIL":
            dqa_result = False

        # TODO: Implement Summary of Metadata

        # SUMMARY OF IMAGE STATISTICS -----------------------------------------
        # Calculate stats
        mean = np.nanmean(photo_img.data)
        median = np.nanmedian(photo_img.data)
        rms = np.sqrt(np.sum(np.power(photo_img.data, 2)) / len(photo_img.data.flatten()))

        img_stats_dict = {
            "MEAN": f"{mean:.4f} {photo_img.unit}",
            "MEDIAN": f"{median:.4f} {photo_img.unit}",
            "RMS": f"{rms:.4f}",
        }

        # Calculate flags and add to dict
        flags_summary(spximg, img_stats_dict)

        # Log results
        for key, value in img_stats_dict.items():
            self.log.info("%s: %s", key, value)

        # THRESHOLD TESTS -----------------------------------------------------

        # Data array has the correct size and bit depth
        shape_cond = (spximg.data.shape == (2072, 2176))
        float_cond = (spximg.data.dtype == np.float32)
        message = f"Data array has shape {spximg.data.shape} and element type "\
                  f"{spximg.data.dtype.name}."
        data_test_1 = {
            "result": "PASS" if (shape_cond & float_cond) else "FAIL",
            "description": "Data array has the correct shape (2072, 2176) and bit depth (32)."
        }
        log_threshold_test(self.log.logger, data_test_1, warning_message=message)
        if not (shape_cond & float_cond):
            dqa_result = False

        # Data array is neither empty (all 0) nor contains fully saturated
        # positive or negative values
        num_nonzero = np.count_nonzero(photo_img.data)
        nonzero_cond = (num_nonzero != 0)
        num_saturated = img_stats_dict["OVERFLOW FLAGS"]
        saturated_cond = (num_saturated != len(photo_img.data.flatten()))
        message = f"Data array has {len(photo_img.data.flatten()) - num_nonzero} zero values and "\
                  f"{len(photo_img.data.flatten()) - num_saturated} fully saturated values."
        data_test_2 = {
            "result": "PASS" if (nonzero_cond & saturated_cond) else "FAIL",
            "description": "Data is neither empty nor contains fully saturated positive or negative values."
        }
        log_threshold_test(self.log.logger, data_test_2, warning_message=message)
        if not (nonzero_cond & saturated_cond):
            dqa_result = False

        # Flag array has the correct size and bit depth
        shape_cond = (spximg.flags.shape == (2072, 2176))
        bit_cond = (spximg.flags.dtype == np.int16)
        message = f"Flag array has shape {spximg.flags.shape} and "\
                  f"element type {spximg.flags.dtype.name}."
        flags_test_1 = {
            "result": "PASS" if (shape_cond & bit_cond) else "FAIL",
            "description": "Flag array has the correct shape (2072, 2176) and bit depth (16)."
        }
        log_threshold_test(self.log.logger, flags_test_1, warning_message=message)
        if not (shape_cond & bit_cond):
            dqa_result = False

        nan_check_pass, nan_percentage = DQAL1Task().check_nan_percent(spximg)
        if nan_check_pass is False:
            dqa_result = False
            logging.error(f"Bad input: NaN % = {nan_percentage:3.1f}%")

        # TODO: Implement checks on fully sampled pixel subset
        # TODO: Check metadata for completion and values in expected ranges
        # TODO: Check image stats are within acceptable ranges
        # TODO: Check number of cosmic rays is below a configurable threshold
        # TODO: Check numbers of different flags are below configurable
        #       thresholds

        # Add all results to record object
        dqa_record = Level1DQA(metadata_summary={}, image_statistics=img_stats_dict,
                               threshold_tests=[data_test_1, data_test_2, flags_test_1, nan_check])

        spximg_out = copy.copy(spximg)
        if dqa_result:
            spximg_out.header["L1DQAFLG"] = ("Pass", "L1 DQA outcome")
        else:
            spximg_out.header["L1DQAFLG"] = ("Fail", "L1 DQA outcome")

        # DQA statistics for flagged pixels
        L1_FLAGS_TO_COUNT = [
            (Flag.TRANSIENT, "transient pixels"),
            (Flag.OVERFLOW, "overflow pixels"),
            (Flag.SUR_ERROR, "sur error pixels"),
            (Flag.PHANTOM, "phantom pixels"),
            (Flag.REFERENCE, "reference pixels"),
            (Flag.NONFUNC, "nonfunctional pixels"),
            (Flag.DICHROIC, "pixels due to low efficiency"),
            (Flag.MISSING_DATA, "pixels with missing data"),
            (Flag.FULLSAMPLE, "full-sample pixels"),
            (Flag.PHANMISS, "phanmiss pixels"),
        ]
        for flag, desc in L1_FLAGS_TO_COUNT:
            count = np.count_nonzero(spximg_out.flags & flag)
            flag_name = FLAG_NAME_ALIASES.get(flag.name, flag.name)
            keyword = f"HIERARCH L1 N_{flag_name}"
            description = f"number of flagged {desc}"
            spximg_out.header[keyword] = (count, description)

        # Flag plane should be unitless
        spximg_out.flag_header["BUNIT"] = ("", "Unit of flag pixel value (dimensionless)")

        # get ntrans reobservation
        odir = get_odir(spximg_out, self.config.max_transient_pix)
        return pipeBase.Struct(spximg_out=spximg_out, dqa_l1=dqa_record, odir_obs=odir,
                               stats=Stats(spximg_out))

    def check_nan_percent(self, spximg: SPHERExImage) -> tuple[bool, float]:
        """Check if NaN percentage exceeds threshold.

        Parameters
        ----------
        spximg : SPHERExImage
            The SPHEREx image object.

        Returns
        -------
        bool
            True if NaN percentage is below threshold, False otherwise.
        float
            Percentage of NaN values in the image.

        """
        total_elements = spximg.data.size
        num_nan = np.sum(np.isnan(spximg.data))
        nan_percentage = (num_nan / total_elements) * 100
        if nan_percentage > self.config.nan_percentage_threshold:
            return False, nan_percentage
        else:
            return True, nan_percentage


def check_l1_dqa(spximg: SPHERExImage) -> None:
    """Minimum DQA on level 1 simulated images.

    Raises ValueError if either L1DQAFLG header indicates failed DQA or
    the percentage of NaN values exceeds the threshold.

    Parameters
    ----------
    spximg :
        SPHEREx image

    """
    nan_check_pass, nan_percentage = DQAL1Task().check_nan_percent(spximg)

    if "L1DQAFLG" in spximg.header:
        if spximg.header["L1DQAFLG"] == "Fail":
            raise ValueError("Bad input: Failed L1 DQA.")
    elif nan_check_pass is False:
        # Raise error if the percentage of NaN values exceeds the threshold
        raise ValueError(f"Bad input: NaN % = {nan_percentage:3.1f}%")
