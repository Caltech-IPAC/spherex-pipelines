__all__ = ["AstropyAngleType", "AstropyMJDType", "AstropyYdayType"]

# Standard library imports
from pydantic import PlainSerializer, WithJsonSchema, BeforeValidator
from typing_extensions import Annotated

# Related third party imports
# Note: core packages are imported as type checking necessitates
# true library imports and not __init__ wildcard imports
from astropy.time.core import Time
from astropy.coordinates.angles import Angle
from astropy import units as u


# Local application/library specific imports


def angle_validate(v) -> Angle:
    return Angle(v, unit=u.deg)


AstropyAngleType = Annotated[Angle,
                             BeforeValidator(func=angle_validate),
                             PlainSerializer(lambda v: f"{v.value} {v.unit}", return_type="str"),
                             WithJsonSchema({"type": "string"}, mode="serialization")]


def time_validate(value) -> Time:
    """Custom field using astropy coordinates."""
    # handle "xxxx mjd" format
    if isinstance(value, str) and "mjd" in value:
        return Time(value.split()[0], format="mjd")
    return Time(value, format="mjd")


AstropyMJDType = Annotated[Time, BeforeValidator(func=time_validate),
                           PlainSerializer(lambda v: f"{v.value} {v.format}", return_type="str"),
                           WithJsonSchema({"type": "string"}, mode="serialization")]


def astropy_ydaytype_validate(v) -> Time:
    # Format the timestamp into something Astropy recognizes.
    v = v.replace("-", ":").replace("T", ":")
    return Time(v, format="yday")


AstropyYdayType = Annotated[Time, BeforeValidator(func=astropy_ydaytype_validate),
                            PlainSerializer(lambda v: v.strftime("%Y:%jT%H:%M:%S.%f"), return_type="str"),
                            WithJsonSchema({"type": "string"}, mode="serialization")]
