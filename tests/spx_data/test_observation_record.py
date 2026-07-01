# Standard library imports
import json
import os
import tempfile

# Related third party imports
import pytest
from pydantic import ValidationError
from astropy.coordinates import Angle
from lsst.daf.butler import Timespan

# Local application/library specific imports
from spx_data import SPSFlag, PlannedObservation
from spx_tests import sample_data


# Example data dictionary
data1 = sample_data.get_sample_planned_obs_data()


def test_observation_serialization():
    json_dir = tempfile.TemporaryDirectory()
    json_file = os.path.join(json_dir.name, "test.json")
    obs1 = PlannedObservation(**data1)
    # dump and load json, ensure consistency with the original
    with open(json_file, "w") as f:
        json.dump(obs1.model_dump(by_alias=True), f)
    with open(json_file, "r") as f:
        data = json.load(f)
    assert PlannedObservation(**data) == obs1
    json_dir.cleanup()


def test_observation_record():
    # using pydantic representation
    # valid observation

    obs1 = PlannedObservation(**data1)
    # asserts for valid observation...
    assert isinstance(obs1.ra, Angle)
    assert isinstance(obs1.dec, Angle)
    assert isinstance(obs1.pa_eq, Angle)
    assert isinstance(obs1.timespan, Timespan)
    assert isinstance(obs1.obs_id, str)
    assert isinstance(obs1.target_id, int)
    assert isinstance(obs1.flags, int)
    assert isinstance(obs1.quaternions, dict)
    assert isinstance(obs1.zenith, list)
    assert isinstance(obs1.moon, list)
    assert isinstance(obs1.sun, list)
    assert isinstance(obs1.vel, list)
    assert isinstance(obs1.in_sband, bool)
    assert SPSFlag.to_string(obs1.flags) == "all_sky,saa"
    assert obs1.non_survey

    # Check for validation errors with incorrect declination
    with pytest.raises(ValidationError):
        data2 = data1.copy()
        # Invalid declination
        data2['dec'] = Angle(-10010, unit="deg")
        PlannedObservation(**data2)

    # Check for validation errors with incorrect flags
    with pytest.raises(ValueError):
        data3 = data1.copy()
        # Invalid flags
        data3['flags'] = 10294967295
        PlannedObservation(**data3)
