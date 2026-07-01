import math
from astropy.wcs import WCS
from lsst.sphgeom import UnitVector3d, Region
from spx_data import ExposureBoundary, ra_dec_to_unit_vector, get_exposure_boundary_info


def test_ra_dec_to_unit_vector():
    """Test ra_dec_to_unit_vector with known RA/Dec values."""
    tolerance = 0.1e-10

    ra_deg = 0.0
    dec_deg = 0.0
    result = ra_dec_to_unit_vector(ra_deg, dec_deg)
    assert isinstance(result, UnitVector3d)
    assert math.isclose(result.x(), 1.0, abs_tol=tolerance)
    assert math.isclose(result.y(), 0.0, abs_tol=tolerance)
    assert math.isclose(result.z(), 0.0, abs_tol=tolerance)

    ra_deg = 90.0
    dec_deg = 0.0
    result = ra_dec_to_unit_vector(ra_deg, dec_deg)
    assert math.isclose(result.x(), 0.0, abs_tol=tolerance)
    assert math.isclose(result.y(), 1.0, abs_tol=tolerance)
    assert math.isclose(result.z(), 0.0, abs_tol=tolerance)

    ra_deg = 0.0
    dec_deg = 90.0
    result = ra_dec_to_unit_vector(ra_deg, dec_deg)
    assert math.isclose(result.x(), 0.0, abs_tol=tolerance)
    assert math.isclose(result.y(), 0.0, abs_tol=tolerance)
    assert math.isclose(result.z(), 1.0, abs_tol=tolerance)


def test_exposure_boundary_to_sphgeom_region():
    """Test ExposureBoundary.to_sphgeom_region."""
    bounds = [
        (0.0, 0.0),
        (90.0, 0.0),
        (90.0, 90.0),
        (0.0, 90.0),
    ]
    exposure_boundary = ExposureBoundary(
        expid=1,
        center=(45.0, 45.0),
        bounds=bounds
    )
    region = exposure_boundary.to_sphgeom_region()
    assert isinstance(region, Region)
    unit_vectors = [ra_dec_to_unit_vector(ra, dec) for ra, dec in bounds]
    for v in unit_vectors:
        assert region.contains(v)


def test_get_exposure_boundary_info():
    """Test get_exposure_boundary_info with a mock WCS object."""
    # Create a mock WCS object
    wcs = WCS(naxis=2)
    wcs.wcs.crval = [0, 0]  # Reference RA/Dec
    wcs.wcs.crpix = [1020.5, 1020.5]  # Reference pixel position
    wcs.wcs.cdelt = [0.1, 0.1]  # Pixel scale (deg/pixel)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # Coordinate types
    wcs.array_shape = (2040, 2040)

    # Call the function
    ra_mid, dec_mid, ra_bounds, dec_bounds = get_exposure_boundary_info(wcs)

    # Assert the mid-array RA/Dec
    assert isinstance(ra_mid, float)
    assert isinstance(dec_mid, float)

    # Assert boundary lengths
    assert len(ra_bounds) == 8
    assert len(dec_bounds) == 8

    # Assert boundary points are within plausible ranges
    assert all(0 <= ra <= 360 for ra in ra_bounds)
    assert all(-90 <= dec <= 90 for dec in dec_bounds)

    # Assert mid-array coordinates are within plausible ranges
    assert 0 <= ra_mid <= 360
    assert -90 <= dec_mid <= 90
