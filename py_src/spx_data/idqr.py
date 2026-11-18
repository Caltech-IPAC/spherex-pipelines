__all__ = ["IDQR", "Timestamp"]

# Standard library imports
import json
from warnings import warn

# Related third party imports
from pydantic import ConfigDict, BaseModel
from pydantic.functional_validators import field_validator
from astropy.coordinates.angles import Angle
from astropy import units as u

# Local application/library specific imports
from .models import AstropyAngleType
from astropy.time import Time


class PointingSolution(BaseModel):
    """Pydantic model to store the pointing solution.

    Attributes
    ----------
    right_ascension : `~astropy.coordinates.Angle`
        The right ascension angle in degrees.
    declination : `~astropy.coordinates.Angle`
        The declination angle in degrees.
    position_angle : `~astropy.coordinates.Angle`
        The position angle in degrees.

    """

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    right_ascension: AstropyAngleType
    declination: AstropyAngleType
    position_angle: AstropyAngleType

    @field_validator("declination")
    @classmethod
    def check_dec(cls, v: Angle, field: str):
        """Validate the declination value.

        Parameters
        ----------
        v : `~astropy.coordinates.Angle`
            Declination angle.
        field : `str`
            Name of the field.

        Returns
        -------
        v : `~astropy.coordinates.Angle`
            The same value if it is valid.

        Warns
        -----
        UserWarning
            Raised if the value is outside the valid range (-90, 90).

        """
        if not u.Quantity("-90 deg") <= v <= u.Quantity("90 deg"):
            warn(f"\"{field}\" value {v.deg} is outside valid range (-90, 90).", UserWarning)
        return v

    @field_validator("right_ascension", "position_angle")
    @classmethod
    def check_angle(cls, v: Angle, field: str):
        """Validate the angle values.

        Parameters
        ----------
        v : `~astropy.coordinates.Angle`
            Angle value.
        field : `str`
            Name of the field.

        Returns
        -------
        v : `~astropy.coordinates.Angle`
            The same value if it is valid.

        Warns
        -----
        UserWarning
            Raised if the value is outside the valid range (0, 360).

        """
        if not u.Quantity("-360 deg") <= v <= u.Quantity("360 deg"):
            warn(f"\"{field}\" value {v.deg} is outside valid range (0, 360).", UserWarning)
        return v


class DataQualityIndicators(BaseModel):
    """Pydantic model to store the data quality indicators.

    Attributes
    ----------
    median_psf_size : `float`
        The median point spread function size.

    """

    median_psf_size: float


class IdqrImage(BaseModel):
    """Pydantic model to store the image data.

    Attributes
    ----------
    status: `str`
        The overall status of the image.
    pointing_solution : `PointingSolution`
        The pointing solution for the image.
    data_quality_indicators : `DataQualityIndicators`
        The data quality indicators for the image.

    """

    status: str
    pointing_solution: PointingSolution | None = None
    data_quality_indicators: DataQualityIndicators | None = None

    @field_validator("pointing_solution", "data_quality_indicators")
    @classmethod
    def check_status(cls, v: dict, values: dict):
        """Check the status before validating fields.

        If the status is "bad", the function returns the value, which can be
        `None`. If the status is not "bad", it raises an error if the value is
        `None`.

        Parameters
        ----------
        v : `dict`
            Value of the field being validated.
        values : `dict`
            A dictionary containing field values of the model instance up to
            this point.
        field : `ModelField`
            An object containing information about the field being validated.

        Returns
        -------
        v : `dict`
            The same value if the conditions are met.

        Warns
        -----
        UserWarning
            Raised if the value is `None` when status is not "bad".

        """
        if "status" in values.data and values.data["status"] == "bad":
            return v
        elif v is None:
            warn(f"Missing value for {v} when status is not 'bad'.", UserWarning)
        return v

    def dict(self, **kwargs) -> dict:
        """Overridden Pydantic `dict` method to exclude "pointing_solution" and
        "data_quality_indicators" when status is "bad".

        Returns
        -------
        `dict`
            A dictionary representation of the model instance with possibly
            excluded "pointing_solution" and "data_quality_indicators".

        """
        base_dict = super().dict(**kwargs)
        if self.status == "bad":
            base_dict.pop("pointing_solution", None)
            base_dict.pop("data_quality_indicators", None)
        return base_dict

    def json(self, **kwargs) -> str:
        """Overridden Pydantic `json` method to exclude "pointing_solution" and
        "data_quality_indicators" when status is "bad".

        Returns
        -------
        `str`
            A JSON string representation of the model instance with possibly
            excluded "pointing_solution" and "data_quality_indicators".

        """
        return json.dumps(self.dict(), **kwargs)


class Timestamp(BaseModel):
    """Pydantic model to store the timestamp data.

    Attributes
    ----------
    start : `~astropy.time.Time`
        The start time of the observation.
    end : `~astropy.time.Time`
        The end time of the observation.

    """

    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
    start: str
    end: str

    @classmethod
    def from_time(cls, start_time: Time, end_time: Time):
        # Convert astropy Time objects to ISO strings for JSON compatibility
        return cls(start=start_time.iso, end=end_time.iso)

    def to_time(self):
        # Converts stored ISO format strings back to Time objects if needed
        return Time(self.start, format="iso"), Time(self.end, format="iso")


class IdqrObservation(BaseModel):
    """Pydantic model to store the observation data.

    Attributes
    ----------
    survey_plan_indicator : `bool`
        An overall "counts toward the survey plan" indication for use as
        feedback to the survey Coverage-to-Date file.
    target_id : `int`
        The SPS target ID.
    sps_flag : `str`
        The SPS flag(s). For multiple SPS flags, they are comma-separated
        with no spaces, e.g. "all_sky,saa".
    timestamp : `Timestamp`
        The timestamp for the observation.
    images : `dict[int, IdqrImage]`
        The images related to the observation.

    """

    survey_plan_indicator: bool
    target_id: int
    sps_flag: str
    timestamp: Timestamp
    images: dict[int, IdqrImage]


class IDQR(BaseModel):
    """Pydantic model to store the Image Data Quality Report (IDQR).

    Attributes
    ----------
    observations : `dict[str, IdqrObservation]`
        The observations data.

    """

    observations: dict[str, IdqrObservation]
    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
