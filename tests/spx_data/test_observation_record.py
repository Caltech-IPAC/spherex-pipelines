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


def test_backward_compatibility_without_new_fields():
    """Test that PlannedObservation can be created without new optional fields.

    This ensures backward compatibility with existing JSON files that don't
    have regularized_target_id or target_name fields.
    """
    obs = PlannedObservation(**data1)
    # New optional fields should default to None
    assert obs.regularized_target_id is None
    assert obs.target_name is None


def test_observation_with_new_fields():
    """Test PlannedObservation with the new optional fields."""
    data_with_new_fields = sample_data.get_sample_planned_obs_data_with_target_name()
    obs = PlannedObservation(**data_with_new_fields)

    assert obs.regularized_target_id == 789
    assert obs.target_name == 'SPS_000789_S12'


def test_serialization_with_new_fields():
    """Test JSON serialization/deserialization with new optional fields."""
    data_with_new_fields = sample_data.get_sample_planned_obs_data_with_target_name()
    obs1 = PlannedObservation(**data_with_new_fields)

    json_dir = tempfile.TemporaryDirectory()
    json_file = os.path.join(json_dir.name, "test_new_fields.json")

    with open(json_file, "w") as f:
        json.dump(obs1.model_dump(by_alias=True), f)
    with open(json_file, "r") as f:
        data = json.load(f)

    obs2 = PlannedObservation(**data)
    assert obs2 == obs1
    assert obs2.regularized_target_id == 789
    assert obs2.target_name == 'SPS_000789_S12'

    json_dir.cleanup()


def test_backward_compatibility_serialization():
    """Test that JSON without new fields can still be deserialized."""
    # Simulate an old JSON file without the new fields
    obs1 = PlannedObservation(**data1)

    json_dir = tempfile.TemporaryDirectory()
    json_file = os.path.join(json_dir.name, "test_old_format.json")

    # Serialize and then manually remove the new fields to simulate old format
    with open(json_file, "w") as f:
        data_dict = obs1.model_dump(by_alias=True)
        # Remove new fields to simulate old JSON format
        data_dict.pop('regularized_target_id', None)
        data_dict.pop('target_name', None)
        json.dump(data_dict, f)

    with open(json_file, "r") as f:
        old_format_data = json.load(f)

    # Should be able to create PlannedObservation from old format
    obs2 = PlannedObservation(**old_format_data)
    assert obs2.regularized_target_id is None
    assert obs2.target_name is None
    # Other fields should still match
    assert obs2.target_id == obs1.target_id
    assert obs2.flags == obs1.flags

    json_dir.cleanup()


def test_get_target_name_when_set():
    """Test get_target_name() returns stored value when target_name is set."""
    data = sample_data.get_sample_planned_obs_data_with_target_name()
    obs = PlannedObservation(**data)

    assert obs.get_target_name() == 'SPS_000789_S12'


def test_get_target_name_derived_for_non_go_back():
    """Test get_target_name() derives value for non-go-back observations."""
    data = sample_data.get_sample_planned_obs_data()
    obs = PlannedObservation(**data)

    # target_name is None, should be derived from obs_id and target_id
    target_name = obs.get_target_name()

    # Should be in format SPS_NNNNNN_SXX
    assert target_name.startswith('SPS_')
    assert '_S' in target_name
    # Target ID 789 should be zero-padded
    assert '000789' in target_name


def test_get_target_name_raises_for_go_back():
    """Test get_target_name() raises ValueError for go-back obs without target_name."""
    data = sample_data.get_sample_planned_obs_data()
    # Set go-back flag
    data['flags'] = SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY
    obs = PlannedObservation(**data)

    with pytest.raises(ValueError, match="Go-back observation target name is not available"):
        obs.get_target_name()


def test_get_target_name_go_back_with_target_name_set():
    """Test get_target_name() returns stored value even for go-back observations."""
    data = sample_data.get_sample_planned_obs_data()
    data['flags'] = SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY
    data['regularized_target_id'] = 123
    data['target_name'] = 'SPS_000123_S12'
    obs = PlannedObservation(**data)

    # Should return the stored value without raising
    assert obs.get_target_name() == 'SPS_000123_S12'
