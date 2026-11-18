"""Survey identifier utilities for SPHEREx observations.

This module provides functions to determine the survey identifier (S12, S34, S56)
based on the spsplan value and to create standardized target names.

The survey identifier is used to group observations into survey phases:
- S12: Surveys 1 and 2
- S34: Surveys 3 and 4
- S56: Surveys 5 and 6

"""

__all__ = [
    "get_survey_id_from_spsplan",
    "create_target_name",
    "derive_target_name",
    "SurveyIdentifierError",
]

from .sps_flags import SPSFlag


class SurveyIdentifierError(ValueError):
    """Exception raised when survey identifier cannot be determined."""

    pass


# Survey boundaries based on spsplan ID values.
# These boundaries define which spsplan IDs belong to which survey.
# Format: spsplan IDs < S12_END belong to S12, etc.
# The spsplan ID format is YYYYWWPVV where:
#   YYYY = year, WW = week, P = plan number, VV = plan version
#
# Boundaries are set to None when not yet known. The code will raise
# SurveyIdentifierError for spsplan IDs beyond known boundaries, forcing
# us to update these values as the mission progresses.

# S12 ends before the first planning period of S34 (2026W20_2A = 202620201)
S12_END: int = 202620200  # End of S12 (exclusive), start of S34

# S34 ends before the first planning period of S56 (2027W22_1A = 202722101)
# Last planning period of S34 is 2027W21_2A
S34_END: int = 202722100  # End of S34 (exclusive), start of S56 (extended mission)

# S56 boundary is not yet known (extended mission)
S56_END: int | None = None  # End of S56 (exclusive) - TBD


def get_survey_id_from_spsplan(spsplan_id: int) -> str:
    """Determine the survey identifier from the spsplan ID.

    This function uses spsplan boundaries to determine which survey phase
    (S12, S34, or S56) a given spsplan belongs to.

    Parameters
    ----------
    spsplan_id : `int`
        The spsplan ID (integer format, e.g., 202537201).

    Returns
    -------
    survey_id : `str`
        The survey identifier: "S12", "S34", or "S56".

    Raises
    ------
    SurveyIdentifierError
        If the spsplan_id is beyond known survey boundaries.
        This is intentional to force review of boundaries as the mission
        progresses to later surveys.

    Examples
    --------
    >>> get_survey_id_from_spsplan(202525101)
    'S12'

    """
    if spsplan_id < 0:
        raise SurveyIdentifierError(
            f"Invalid spsplan ID {spsplan_id}: must be non-negative"
        )

    if spsplan_id < S12_END:
        return "S12"

    if spsplan_id < S34_END:
        return "S34"

    # Beyond S34 - check if S56 boundary is defined
    if S56_END is None:
        raise SurveyIdentifierError(
            f"spsplan ID {spsplan_id} is beyond S34 (ends at {S34_END}), "
            "but S56 boundary is not yet defined. Please update S56_END "
            "in survey_identifier.py."
        )

    if spsplan_id < S56_END:
        return "S56"

    # Beyond all known surveys
    raise SurveyIdentifierError(
        f"spsplan ID {spsplan_id} is beyond all known survey boundaries "
        f"(S56 ends at {S56_END}). Please update survey_identifier.py "
        "for extended mission support."
    )


def create_target_name(regularized_target_id: int, survey_id: str) -> str:
    """Create a standardized target name string.

    The target name is formatted as "SPS_NNNNNN_SXX" where:
    - NNNNNN is the 6-digit zero-padded regularized target ID
    - SXX is the survey identifier (S12, S34, or S56)

    Non-standard target ids were used in IOC, for example, in 2025W17_2A.
    If regularized_target_id is negative or greater than 999999,
    the regularized target ID is assumed to be 0.

    Parameters
    ----------
    regularized_target_id : `int`
        The regularized target ID (with go-backs' IDs reset to originals).
    survey_id : `str`
        The survey identifier ("S12", "S34", or "S56").

    Returns
    -------
    target_name : `str`
        The formatted target name string, e.g., "SPS_000123_S12".

    Raises
    ------
    ValueError
        If regularized_target_id is negative or greater than 999999.
        If survey_id is not one of "S12", "S34", "S56".

    Examples
    --------
    >>> create_target_name(123, "S12")
    'SPS_000123_S12'
    >>> create_target_name(999999, "S34")
    'SPS_999999_S34'

    """
    if regularized_target_id < 0 or regularized_target_id > 999999:
        # non-standard target IDs were used in IOC campaigns
        regularized_target_id = 0
    if survey_id not in ("S12", "S34", "S56"):
        raise ValueError(
            f"survey_id must be 'S12', 'S34', or 'S56', got '{survey_id}'"
        )

    return f"SPS_{regularized_target_id:06d}_{survey_id}"


def derive_target_name(sps_flags: int, spsplan_id: int, target_id: int) -> str:
    """Derive target name from observation metadata.

    For go-back observations (identified by GO_BACK_S12, GO_BACK_S34, or
    GO_BACK_S56 flags), the survey identifier is derived from the go-back
    flag. For other observations, it's derived from the spsplan_id.

    Parameters
    ----------
    sps_flags : `int`
        SPS flags for the observation. Used to check for go-back flags.
    spsplan_id : `int`
        The spsplan ID (used for non-go-back observations to determine survey).
    target_id : `int`
        The regularized target ID to use in the target name.

    Returns
    -------
    target_name : `str`
        Target name in format "SPS_NNNNNN_SXX", e.g., "SPS_000123_S12".

    Raises
    ------
    SurveyIdentifierError
        If the survey identifier cannot be determined from spsplan_id
        (for non-go-back observations).
    ValueError
        If target_id is invalid (negative or > 999999).

    Examples
    --------
    >>> # Non-go-back observation in S12
    >>> derive_target_name(SPSFlag.ALL_SKY, 202525101, 123)
    'SPS_000123_S12'

    >>> # Go-back observation from S12 (executed in S34)
    >>> derive_target_name(SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY, 202650101, 456)
    'SPS_000456_S12'

    """
    # Check if this is a go-back observation
    survey_id = SPSFlag.get_go_back_survey_id(sps_flags)

    if survey_id is None:
        # Not a go-back observation, derive survey from spsplan
        survey_id = get_survey_id_from_spsplan(spsplan_id)

    return create_target_name(target_id, survey_id)
