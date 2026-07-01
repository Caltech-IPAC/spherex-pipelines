import numpy as np
from spx_butler import ra_dec_to_unit_vector


def test_ra_dec_to_unit_vector_equator_and_poles():
    # Equator, RA = 0
    v = ra_dec_to_unit_vector(0.0, 0.0)
    assert np.allclose([v.x(), v.y(), v.z()], [1.0, 0.0, 0.0], atol=1e-8)

    # Equator, RA = 90
    v = ra_dec_to_unit_vector(90.0, 0.0)
    assert np.allclose([v.x(), v.y(), v.z()], [0.0, 1.0, 0.0], atol=1e-8)

    # North Pole
    v = ra_dec_to_unit_vector(123.0, 90.0)  # RA irrelevant at pole
    assert np.allclose([v.x(), v.y(), v.z()], [0.0, 0.0, 1.0], atol=1e-8)

    # South Pole
    v = ra_dec_to_unit_vector(45.0, -90.0)
    assert np.allclose([v.x(), v.y(), v.z()], [0.0, 0.0, -1.0], atol=1e-8)


def test_ra_dec_to_unit_vector_norm():
    # Verify unit length for a general point
    v = ra_dec_to_unit_vector(123.456, -22.334)
    norm = np.sqrt(v.x()**2 + v.y()**2 + v.z()**2)
    assert np.isclose(norm, 1.0, atol=1e-10)
