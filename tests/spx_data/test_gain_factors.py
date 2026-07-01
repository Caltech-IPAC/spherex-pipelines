import pytest
from pydantic import ValidationError
from spx_data import GainFactors, ReadoutParameters
from spx_tests.sample_data import get_sample_gain_factors, get_sample_level1_img


def test_valid_gain_factors():
    num_channels = ReadoutParameters.nchannel
    gains = [float(x+1.1) for x in range(num_channels)]

    gain_factors = GainFactors(
        gains_1=gains,
        gains_2=gains,
        gains_3=gains,
        gains_4=gains,
        gains_5=gains,
        gains_6=gains,
        num_channels=num_channels,
    )

    assert gain_factors.gains_1 == gains
    assert gain_factors.gains_2 == gains
    assert gain_factors.gains_3 == gains
    assert gain_factors.gains_4 == gains
    assert gain_factors.gains_5 == gains
    assert gain_factors.gains_6 == gains
    assert gain_factors.num_channels == num_channels


def test_invalid_gain_factors():
    num_channels = ReadoutParameters.nchannel
    gains = [float(x+1.1) for x in range(num_channels - 1)]

    with pytest.raises(ValidationError):
        GainFactors(
            gains_1=gains,
            gains_2=gains,
            gains_3=gains,
            gains_4=gains,
            gains_5=gains,
            gains_6=gains,
            num_channels=num_channels,
        )


def test_get_gains_for_detector():
    gain_factors = get_sample_gain_factors()
    for detector in range(1, 7):
        assert isinstance(gain_factors.get_gains_for_detector(detector), list)


def test_get_sample_gain_factors():
    gain_factors = get_sample_gain_factors()

    assert isinstance(gain_factors, GainFactors)
    assert gain_factors.num_channels == ReadoutParameters.nchannel

    # Test using a sample level 1 image
    spx_img = get_sample_level1_img()
    gain_factors = get_sample_gain_factors(spx_img)
    assert isinstance(gain_factors, GainFactors)
