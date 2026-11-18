import numpy as np
import pandas as pd
from astropy.time import Time
from unittest.mock import Mock, patch

from spx_tasks.compute_solar_system_objects import (
    _hgphi12,
    iau_hg_model,
    calculate_midpoint_time,
    split_dataframe_by_mjd,
    create_dataframe_from_headers,
    propagate_orbit_elements,
)


def test_hgphi12_alpha_zero_returns_one():
    phi1, phi2 = _hgphi12(0.0)

    assert np.isclose(phi1, 1.0)
    assert np.isclose(phi2, 1.0)


def test_hgphi12_accepts_array():
    alpha = np.array([0.0, 10.0, 30.0])

    phi1, phi2 = _hgphi12(alpha)

    assert phi1.shape == alpha.shape
    assert phi2.shape == alpha.shape
    assert np.all(np.isfinite(phi1))
    assert np.all(np.isfinite(phi2))


def test_iau_hg_model_alpha_zero_is_one():
    assert np.isclose(iau_hg_model(0.0), 1.0)


def test_iau_hg_model_negative_alpha_matches_positive_alpha():
    neg = iau_hg_model(-20.0)
    pos = iau_hg_model(20.0)

    assert np.isclose(neg, pos)


def test_iau_hg_model_accepts_array():
    alpha = np.array([0.0, 10.0, 20.0])

    result = iau_hg_model(alpha, gpar=0.15)

    assert result.shape == alpha.shape
    assert np.all(np.isfinite(result))


def test_calculate_midpoint_time():
    df = pd.DataFrame(
        {
            "EXPSTART": [60000.0, 60001.0],
            "EXPSTOP": [60000.5, 60001.5],
        }
    )

    result = calculate_midpoint_time(df)

    expected_midpoint = np.array([60000.25, 60001.25])
    expected = Time(expected_midpoint, format="mjd", scale="utc").tdb.jd

    np.testing.assert_allclose(result, expected)


def test_split_dataframe_by_mjd_single_group():
    df = pd.DataFrame(
        {
            "EXPSTART": [60000.0, 60001.0, 60002.5],
            "value": [1, 2, 3],
        }
    )

    chunks = split_dataframe_by_mjd(df, max_diff_days=3)

    assert len(chunks) == 1
    assert list(chunks[0]["value"]) == [1, 2, 3]
    assert "HUMAN_DATE" in chunks[0].columns


def test_split_dataframe_by_mjd_multiple_groups():
    df = pd.DataFrame(
        {
            "EXPSTART": [60000.0, 60001.0, 60003.0, 60004.0, 60007.1],
            "value": [1, 2, 3, 4, 5],
        }
    )

    chunks = split_dataframe_by_mjd(df, max_diff_days=3)

    assert len(chunks) == 3
    assert list(chunks[0]["value"]) == [1, 2]
    assert list(chunks[1]["value"]) == [3, 4]
    assert list(chunks[2]["value"]) == [5]


def test_split_dataframe_by_mjd_sorts_input():
    df = pd.DataFrame(
        {
            "EXPSTART": [60002.0, 60000.0, 60001.0],
            "value": [3, 1, 2],
        }
    )

    chunks = split_dataframe_by_mjd(df, max_diff_days=3)

    assert list(chunks[0]["value"]) == [1, 2, 3]


def test_create_dataframe_from_headers():
    spximg = Mock()
    spximg.header = {
        "X_SC": 1.0,
        "VX_SC": 2.0,
        "OBSID": "obs-1",
        "DETECTOR": 5,
    }
    spximg.wcs = "fake-wcs"
    spximg.meta = {"fake": "meta"}

    exp_start = Mock()
    exp_start.value = 60000.0
    exp_stop = Mock()
    exp_stop.value = 60000.5

    spximg.getExposureStartTime.return_value = exp_start
    spximg.getExposureEndTime.return_value = exp_stop

    loaded_handle = Mock()
    loaded_handle.get.return_value = spximg

    butlerQC = Mock()
    butlerQC.get.return_value = loaded_handle

    handle = Mock()

    df = create_dataframe_from_headers(
        handles=[handle],
        column_names=["X_SC", "VX_SC", "MISSING"],
        butlerQC=butlerQC,
        max_workers=1,
    )

    assert len(df) == 1
    assert df.loc[0, "X_SC"] == 1.0
    assert df.loc[0, "VX_SC"] == 2.0
    assert df.loc[0, "MISSING"] is None
    assert df.loc[0, "EXPSTART"] == 60000.0
    assert df.loc[0, "EXPSTOP"] == 60000.5
    assert df.loc[0, "OBSID"] == "obs-1"
    assert df.loc[0, "DETECTOR"] == 5
    assert df.loc[0, "WCS"] == "fake-wcs"
    assert df.loc[0, "META"] == {"fake": "meta"}


@patch("spx_tasks.compute_solar_system_objects.kete.propagate_n_body")
def test_propagate_orbit_elements_uses_first_fov_observer_jd(mock_propagate):
    states = Mock()

    fov0 = Mock()
    fov0.observer.jd = 2450000.5

    fov1 = Mock()
    fov1.observer.jd = 2450001.5

    fovs = [fov0, fov1]

    mock_propagate.return_value = "propagated-states"

    result = propagate_orbit_elements(states, fovs)

    assert result == "propagated-states"
    mock_propagate.assert_called_once_with(
        states,
        2450000.5,
        include_asteroids=False,
    )
