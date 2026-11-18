from astropy.coordinates.angles import Angle
from astropy.time.core import Time
from spx_data.models import AstropyAngleType, AstropyMJDType, AstropyYdayType
from pydantic import BaseModel, ConfigDict


def test_astropy_angle_type_validator():
    # Create testing pydantic class
    class TestAngle(BaseModel):
        model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
        angle: AstropyAngleType
    # assert type checking
    assert isinstance(TestAngle(**{"angle": 45}).angle, Angle)


def test_astropy_mjd_type_validator():
    # Create testing pydantic class
    class TestMJD(BaseModel):
        model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
        mjd_time: AstropyMJDType
    # assert type checking
    mjd_instance = TestMJD(**{"mjd_time": 59215.0}).mjd_time
    assert isinstance(mjd_instance, Time)
    # check format of time
    assert mjd_instance.format == "mjd"


def test_astropy_yday_type_validator():
    # Create testing pydantic class
    time_str = "2023-180T12:00:00.000"

    class TestTime(BaseModel):
        model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
        time: AstropyYdayType
    # assert type checking
    astropy_time = TestTime(**{"time": time_str}).time
    assert isinstance(astropy_time, Time)
    # check format of time
    assert astropy_time.format == "yday"
