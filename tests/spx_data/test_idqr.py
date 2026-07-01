from spx_data.idqr import IDQR, PointingSolution, IdqrImage, DataQualityIndicators
import json
import re
import pytest
from pathlib import Path


@pytest.fixture
def sample_idqr_filepath():
    return Path(__file__).absolute().parent.parent / "data" / "test.idqr"


@pytest.fixture
def sample_idqr(sample_idqr_filepath):
    with open(sample_idqr_filepath, "r") as f:
        data = json.load(f)
    return IDQR(**data)


def test_read(sample_idqr):
    assert isinstance(sample_idqr, IDQR)


def test_image():
    # Test case where status is "bad".
    IdqrImage(status="bad", pointing_solution=None, data_quality_indicators=None)
    IdqrImage(status="bad",
              pointing_solution=PointingSolution(right_ascension=10, declination=10, position_angle=10),
              data_quality_indicators=DataQualityIndicators(median_psf_size=1.))

    # Test case when status is "good" and values are not supplied.
    with pytest.warns(UserWarning):
        IdqrImage(status="good", pointing_solution=None, data_quality_indicators=None)

    # Test case where status is "good" and other fields are provided.
    IdqrImage(status="good",
              pointing_solution=PointingSolution(right_ascension=10, declination=10, position_angle=10),
              data_quality_indicators=DataQualityIndicators(median_psf_size=1.))


def test_pointing_solution():
    # Test valid input
    ra = 180
    dec = 45.
    rot = 90.

    # Test invalid declination
    with pytest.warns(UserWarning):
        PointingSolution(right_ascension=ra, declination=100, position_angle=rot)

    # Test invalid right ascension
    with pytest.warns(UserWarning):
        PointingSolution(right_ascension=400, declination=dec, position_angle=rot)

    # Test invalid position angle
    with pytest.warns(UserWarning):
        PointingSolution(right_ascension=ra, declination=dec, position_angle=400)


def test_json(sample_idqr):
    # Define a regular expression pattern for the timestamp string.
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+"

    # Convert the IDQR instance back into a JSON string and load it as a dictionary.
    idqr_json_out = sample_idqr.model_dump_json()
    idqr_dict_out = json.loads(idqr_json_out)

    # For each observation in the output dictionary...
    for observation in idqr_dict_out["observations"].values():
        # Check whether the timestamp strings match the pattern.
        assert re.match(pattern, observation["timestamp"]["start"]) is not None
        assert re.match(pattern, observation["timestamp"]["end"]) is not None
