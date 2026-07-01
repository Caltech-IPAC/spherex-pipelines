__all__ = ["flags_summary", "log_threshold_test", "nan_pixels_flagged", "log_dqa_result"]

# Standard library imports
import logging
from logging import Logger
# Related third party imports
# Local application/library specific imports
from spx_data import SPHERExImage
import numpy as np


def log_dqa_result(module: str, result_dict: dict) -> None:
    """Log DQA result uniformly for all checks.

    Parameters
    ----------
    module : `str`
        Name of the DQA module or check (e.g., 'source_count', 'hot_cold_pixel').
    result_dict : `dict`
        Dictionary with 'result' and 'description'.
    """
    prefix = f"DQA L2 [{module}]"
    msg = result_dict["description"]

    if result_dict["result"] == "FAIL":
        logging.error(f"{prefix}: {msg}")
    else:
        logging.info(f"{prefix}: {msg}")


def flags_summary(spximg: SPHERExImage, image_stats_dict: dict):
    """Outputs flag counts.

    Parameters
    ----------
    spximg : `spx_data.SPHERExImage`
        Level 1 LVF image.
    """
    for flag in spximg.flag_defs:
        bit_num = spximg.flag_defs[flag]
        mask = spximg.getFlagsAsMask(1 << bit_num)
        num_flags = np.count_nonzero(mask)
        flag_str = f"{flag} FLAGS"
        image_stats_dict[flag_str] = num_flags


def log_threshold_test(logger: Logger, test_dict: dict, warning_message: str):
    """Outputs results of threshold tests.

    Parameters
    ----------
    logger : `logging.Logger`
        Logger used inside DQA module.
    test_dict : `dict`
        Includes result (pass/fail) and test description.
    warning_message : `str`
        Warning output to user.
    """
    logger.info("%s: %s", test_dict["result"], test_dict["description"])
    if test_dict["result"] == "FAIL":
        logger.warning(warning_message)


def _nan_values_flagged(array: np.ndarray,
                        flags: np.ndarray,
                        nan_flags: int,
                        check_inf: bool = True) -> bool:
    """Helper function to verify that all NaN or Inf values in an array
       are justified by flags.

    Parameters
    ----------
    array : `numpy.ndarray`
        Array of values to check (e.g., image data or variance).
    flags : `numpy.ndarray`
        Bitmask flags corresponding to each pixel in the array.
    nan_flags : `int`
        Bitmask of flag values that are considered valid justifications
        for NaN/Inf values.
    check_inf : `bool`, optional
        If True, check for both NaNs and IEEE infinite values.
        If False, only check for NaNs. Default is True.

    Returns
    -------
    result : `bool`
        True if all NaN and Inf values in the array are accompanied
        by valid flags; False otherwise.
    """
    problem_mask = np.isnan(array)
    if check_inf:
        problem_mask |= np.isinf(array)

    flagged_mask = (flags & nan_flags) > 0
    unflagged_problem_mask = problem_mask & ~flagged_mask

    return not np.any(unflagged_problem_mask)


def nan_pixels_flagged(data: np.ndarray,
                       flags: np.ndarray,
                       nan_flags: int,
                       check_inf: bool = True) -> bool:
    """Check if all NaN or Inf pixels in the image data are flagged with valid reasons.

    Parameters
    ----------
    data : `numpy.ndarray`
        The image data array to check for NaN or Inf values.
    flags : `numpy.ndarray`
        The corresponding image flag array with bitmask values.
    nan_flags : `int`
        Bitmask of flags considered valid explanations for NaN or Inf values.
    check_inf : `bool`, optional
        If True, include Inf values in the check. Default is True.

    Returns
    -------
    result : `bool`
        True if all NaN and Inf pixels are accompanied by a valid flag;
        False otherwise.
    """
    return _nan_values_flagged(data, flags, nan_flags, check_inf)


def nan_variance_flagged(variance: np.ndarray,
                         flags: np.ndarray,
                         nan_flags: int,
                         check_inf: bool = True) -> bool:
    """Check if all NaN or Inf pixels in the variance layer
       are flagged with valid reasons.

    Parameters
    ----------
    variance : `numpy.ndarray`
        SPHEREx uncertainty variance array.
    flags : `numpy.ndarray`
        The corresponding image flag array with bitmask values.
    nan_flags : `int`
        Bitmask of flags considered valid explanations for NaN or Inf values.
    check_inf : `bool`, optional
        If True, include Inf values in the check. Default is True.

    Returns
    -------
    result : `bool`
        True if all NaN and Inf values in the variance array are accompanied
        by a valid flag; False otherwise.
    """
    return _nan_values_flagged(variance, flags, nan_flags, check_inf)
