"""Pydantic representation for a single observation.

The following is the json output.

.. code-block:: json

{
  "title": "PlannedObservation",
  "description": "Pydantic Representation of Single Observation.
  "type": "object",
  "properties": {
    "boresight": {
      "$ref": "#/definitions/BoresightType"
    },
    "ra": {
      "$ref": "#/definitions/AstropyAngleType"
    },
    "dec": {
      "$ref": "#/definitions/AstropyAngleType"
    },
    "position_angle": {
      "$ref": "#/definitions/AstropyAngleType"
    },
    "start_time": {
      "$ref": "#/definitions/AstropyMJDType"
    },
    "stop_time": {
      "$ref": "#/definitions/AstropyMJDType"
    },
    "obs_id": {
      "title": "Obs Id",
      "type": "integer"
    },
    "target_id": {
      "title": "Target Id",
      "type": "integer"
    },
    "flags": {
      "title": "Flags",
      "description": "Observation flags, one bit per flag.",
      "type": "integer"
    }
  },
  "required": [
    "ra",
    "dec",
    "position_angle",
    "start_time",
    "stop_time",
    "obs_id",
    "target_id",
    "flags"
  ],
  "definitions": {
    "BoresightType": {
      "title": "BoresightType",
      "description": "Custom field using `astropy.coordinates.SkyCoord`.
                      Takes the input ``ra`` and ``dec`` from the `dict`,
                      validates, then builds a `SkyCoord`.",
      "type": "object",
      "properties": {}
    },
    "AstropyAngleType": {
      "title": "AstropyAngleType",
      "description": "Custom field using astropy coordinates.",
      "type": "object",
      "properties": {}
    },
    "AstropyMJDType": {
      "title": "AstropyMJDType",
      "description": "Custom field with astropy type time in MJD format.",
      "type": "object",
      "properties": {}
    }
  }
}


Notes
-----
  boresight: The planned telescope boresight in (ra, dec) at the start of the
             observation (ICRS equatorial coordinates in degrees)
  ra: right ascension
  dec: declination
  position_angle: The position angle of the observation at the boresight at the
                  start of the observation (degrees east of North, 0-360)
  start_time: The planned start and stop times for the observation (see below)
  stop_time: The observation ID (probably in its integer form)
  target_id: The "target ID" from the SPS
  flags: derived from the SPS metadata",

"""

__all__ = ["PlannedObservation"]

# Standard library imports
from typing import Optional

# Related third party imports
from pydantic import field_validator, ConfigDict, BaseModel, Field
from astropy import units as u
from lsst.daf.butler import Timespan

# Local application/library specific imports
from .models import AstropyAngleType
from .observation_id import ObservationId
from .sps_flags import SPSFlag
from .survey_identifier import derive_target_name


class PlannedObservation(BaseModel):
    """Pydantic Representation of Single Observation.

    Attributes
    ----------
    ra : `~astropy.coordinates.Angle`
        Right ascension at the start of the observation.
    dec : `~astropy.coordinates.Angle`
        Declination at the start of the observation.
    pa_eq : `~astropy.coordinates.Angle`
        The position angle of the observation at the boresight at the
        start of the observation (degrees east of North, 0-360),
        equatorial coordinates.
    timespan : `~lsst.daf.butler.Timespan`
      The time validity interval for a planned observation.
    obs_id : `str`
      Survey Planning Software Observation ID.
    target_id : `int`
        The "target ID" from the SPS
    flags : `int`
        Derived from the SPS metadata, see
        https://confluence.ipac.caltech.edu/pages/viewpage.action?pageId=620828797
    quaternions : `dict`
      Quaternions for planned observation from the SPS.
    lat_ecl : `float`
      Ecliptic latitude for planned observation from the SPS.
    lon_ecl : `float`
      Ecliptic longitude for planned observation from the SPS.
    pa_ecl : `float`
      Ecliptic position angle for planned observation from the SPS.
    zenith : `list[float]`
      Zenith for planned observation from SPS.
    moon : `list[float]`
      Moon for planned observation from SPS.
    sun : `list[float]`
      Sun for planned observation from SPS.
    vel : `list[float]`
      Velocity for planned observation from SPS.
    is_sband : `bool`
      Observation is taken during S-Band transmission, as noted in PEF.
      If True, the observation might have significant electrical noise
      in detector pixel data and should be excluded from calibrations.
    non_survey : `bool`
      Observation is a non-survey test observation, as noted in PEF.
    regularized_target_id : `int`, optional
      Regularized SPS target number. For go-back observations, this is the
      original target ID from the first survey when the target was observed
      (looked up from the GDS go-backs file). For non-go-back observations,
      this is the same as target_id.
    target_name : `str`, optional
      Target name string for use in SPS_TARGET header and CAOM2/ObsCore.
      Format: "SPS_NNNNNN_SXX" where NNNNNN is the 6-digit zero-padded
      regularized target ID and SXX is the survey identifier (S12, S34, S56).

    """

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    ra: AstropyAngleType
    dec: AstropyAngleType
    pa_eq: AstropyAngleType
    timespan: Timespan
    obs_id: str
    target_id: int
    flags: int = Field(description="Observation SPS flags, one bit per flag.")
    quaternions: dict
    lat_ecl: float
    lon_ecl: float
    pa_ecl: float
    zenith: list[float]  # zenithx, zenithy, zenithz
    moon: list[float]  # moonx, moony, moonz
    sun: list[float]  # sunx, suny, sunz
    vel: list[float]  # velx, vely, velz
    in_sband: bool
    non_survey: Optional[bool] = None
    regularized_target_id: Optional[int] = Field(
        default=None,
        description="Regularized target ID (original ID for go-backs, same as target_id otherwise)"
    )
    target_name: Optional[str] = Field(
        default=None,
        description="Target name string (e.g., 'SPS_000123_S12')"
    )

    @field_validator("dec")
    @classmethod
    def check_dec(cls, v):
        if not u.Quantity("-90 deg") <= v <= u.Quantity("90 deg"):
            raise ValueError("Declination is outside range.")
        return v

    @field_validator("ra")
    @classmethod
    def check_ra(cls, v):
        if not u.Quantity("0 deg") <= v <= u.Quantity("360 deg"):
            raise ValueError("Right ascension is outside range")
        return v

    @field_validator("flags")
    @classmethod
    def check_obsflags(cls, v):
        allowed_nflags = 31
        num_flags = v.bit_length()
        if num_flags > allowed_nflags:
            raise ValueError(f"The number of flags {num_flags} exceed {allowed_nflags}")
        return v

    def get_flag_names(self) -> str:
        """Convert the integer `flags` into a single string listing all active flags.

        Returns
        -------
        result : `str`
            A comma-separated string of all active flag names.

        """
        active_flags = SPSFlag.to_string(self.flags)
        if len(active_flags) == 0:
            active_flags = "No flags set"
        return active_flags

    def get_target_name(self) -> str:
        """Get the extended SPS target ID string (target_name).

        Returns the target_name in format "SPS_NNNNNN_SXX" (e.g., "SPS_000123_S12").
        If target_name is already set, returns it directly. Otherwise, derives it
        from the observation metadata for non-go-back observations.

        Returns
        -------
        target_name : `str`
            The extended SPS target ID string.

        Raises
        ------
        ValueError
            If target_name is not set and the observation is a go-back observation.
            Go-back observations require regularized_target_id to be looked up
            from the GDS go-backs file and target_name to be set explicitly.

        """
        if self.target_name is not None:
            return self.target_name

        # Check if this is a go-back observation
        if SPSFlag.is_go_back(self.flags):
            raise ValueError(
                "Go-back observation target name is not available. "
                "The regularized_target_id must be looked up from the GDS "
                "go-backs file and target_name must be set explicitly."
            )

        # Non-go-back observation: derive target_name from obs_id and target_id
        obs = ObservationId(self.obs_id)
        spsplan_id = obs.plan_id
        target_id = self.regularized_target_id or self.target_id
        return derive_target_name(self.flags, spsplan_id, target_id)
