__all__ = ["ODIR", "OdirImage", "OdirObservation"]

# Standard library imports

# Related third party imports
from pydantic import ConfigDict, BaseModel

# Local application/library specific imports
from spx_data.idqr import Timestamp


# Initial *minimal* Image model specifically for ODIR.
# This model only includes the 'status' field as requested.
class OdirImage(BaseModel):
    """Pydantic model for the simplified image data in an ODIR.

    Attributes
    ----------
    status: `str`
        The overall status of the image.
    """

    status: str


class OdirObservation(BaseModel):
    """Pydantic model for an observation within an ODIR.

    This is similar to the base Observation model but uses the simplified
    OdirImage model for its image data.

    Attributes
    ----------
    survey_plan_indicator : `bool`
        An overall "counts toward the survey plan" indication.
    target_id : `int`
        The SPS target ID.
    sps_flag : `str`
        The SPS flag(s). For multiple SPS flags, they are comma-separated.
    timestamp : `Timestamp`
        The timestamp for the observation, reused from the IDQR definition.
    images : `dict[int, OdirImage]`
        A dictionary of simplified image data, keyed by an integer ID.
    """

    survey_plan_indicator: bool
    target_id: int
    sps_flag: str
    timestamp: Timestamp
    images: dict[int, OdirImage]


class ODIR(BaseModel):
    """Pydantic model to store the Observation Data Integrity Report (ODIR).

    This is a simplified version of the IDQR, containing only the status
    for each image and omitting detailed pointing and quality indicators.

    Attributes
    ----------
    observations : `dict[str, OdirObservation]`
        A dictionary of observation data, keyed by a string identifier.
    """

    observations: dict[str, OdirObservation]
    model_config = ConfigDict(validate_default=True, arbitrary_types_allowed=True)
