import pytest
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS
from spx_apps.mosaic_tool.temp_butler import compute_polygon_wcs_angular
from matplotlib.path import Path


def parse_polygon_to_array(poly_str):
    """
    Parses 'POLYGON ra1 dec1 ...' into a numpy array of shape (N, 2).
    """
    content = poly_str.replace("POLYGON ", "")
    vals = [float(x) for x in content.split()]
    # Reshape into [[ra1, dec1], [ra2, dec2], ...]
    return np.array(vals).reshape(-1, 2)


def parse_polygon(poly_str):
    """Helper to parse 'POLYGON ra1 dec1 ...' into a SkyCoord object."""
    content = poly_str.replace("POLYGON ", "")
    vals = [float(x) for x in content.split()]
    ra = vals[::2]
    dec = vals[1::2]
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg)


@pytest.mark.parametrize(
    "center_ra, center_dec, location_name",
    [
        (266.4168, -29.0078, "Galactic Center"),  # Sgr A*
        (10.0, 85.0, "Near North Pole"),  # High declination test
    ],
)
@pytest.mark.parametrize("rotation_deg", [0, 90])
def test_polygon_geometry(center_ra, center_dec, location_name, rotation_deg):
    """
    Validates polygon generation for:
    1. Center containment.
    2. Diagonal symmetry (checking the shape is roughly rectangular).
    3. Orientation based on rotation angle.
    """
    width = 1.0
    height = 0.5

    # Generate Polygon
    poly_str = compute_polygon_wcs_angular(
        center_ra, center_dec, width, height, rotation_deg
    )
    corners = parse_polygon(poly_str)

    # Define Center
    center = SkyCoord(ra=center_ra * u.deg, dec=center_dec * u.deg)

    # --- 1. Check Center Containment ---
    # We check that the separation from center to any corner is reasonably close
    # to the expected half-diagonal (plus buffer).
    # Since it's a rectangle, center should be equidistant to all corners (roughly).
    seps = center.separation(corners)

    # Verify all corners are roughly equidistant from center
    # Tolerance is slightly loose to account for spherical projection
    # distortion and pixel snapping
    assert np.allclose(
        seps[0].value, seps[1].value, atol=0.01
    ), "Top corners not symmetric to center"
    assert np.allclose(
        seps[2].value, seps[3].value, atol=0.01
    ), "Bottom corners not symmetric to center"

    # Check that the polygon is large enough to contain the requested width
    # Min distance from center to corner must be > width/2
    assert np.min(seps).deg > (
        min(width, height) / 2.0
    ), "Polygon is too small to contain requested region"

    # --- 2. Check Diagonal Equality ---
    # TL(0) to BR(2) should equal TR(1) to BL(3)
    diag1 = corners[0].separation(corners[2])
    diag2 = corners[1].separation(corners[3])
    assert np.isclose(
        diag1.value, diag2.value, rtol=1e-3
    ), "Diagonals are not equal length"

    # --- 3. Check Orientation (Rotation) ---
    # Note: Corners order in function is TL, TR, BR, BL (based on projection plane)

    tl, tr, br, bl = corners[0], corners[1], corners[2], corners[3]

    if rotation_deg == 0:
        # Standard Orientation: Top corners should have higher Dec than Bottom corners
        # (Unless we are going over the pole, which 85 deg might do if height is huge,
        # but 0.5 deg height is safe at 85 deg dec).

        # Check Dec relationships
        assert (
            tl.dec > center.dec and tr.dec > center.dec
        ), "Rot=0: Top corners should be North of center"
        assert (
            bl.dec < center.dec and br.dec < center.dec
        ), "Rot=0: Bottom corners should be South of center"

        # Check RA relationships (only valid if not crossing RA=0/360 boundary)
        # We skip RA check for Pole test to avoid 0/360 wrapping
        # complexity in simple assert
        if "Pole" not in location_name:
            # In standard projection (CDELT=[1,1]), +x is +RA (East)
            # So TR (positive x) should be East of TL (negative x)
            # RA increases to the East.
            assert tr.ra.deg > tl.ra.deg, "Rot=0: TR should be East of TL"

    elif rotation_deg == 90:
        # Rotated 90 degrees (Top points East).
        # We can predict the exact quadrant for each corner.

        pa_tl = center.position_angle(tl).deg
        pa_tr = center.position_angle(tr).deg

        # TL should be in the North-East quadrant (0 to 90)
        assert 0 < pa_tl < 90, f"Rot=90: TL should be North-East, got PA={pa_tl}"

        # TR should be in the South-East quadrant (90 to 180)
        assert 90 < pa_tr < 180, f"Rot=90: TR should be South-East, got PA={pa_tr}"


def test_buffer_expansion():
    """Ensure buffer actually increases the polygon size."""
    w, h, rot = 1.0, 1.0, 0

    # No buffer
    poly_no_buff = compute_polygon_wcs_angular(0, 0, w, h, rot, buffer_deg=0)
    c_nb = parse_polygon(poly_no_buff)

    # With buffer
    poly_buff = compute_polygon_wcs_angular(0, 0, w, h, rot, buffer_deg=0.1)
    c_b = parse_polygon(poly_buff)

    # Distance between TL corners
    dist_nb = SkyCoord(0 * u.deg, 0 * u.deg).separation(c_nb[0])
    dist_b = SkyCoord(0 * u.deg, 0 * u.deg).separation(c_b[0])

    assert dist_b > dist_nb, "Buffer did not increase polygon size"


@pytest.mark.parametrize(
    "center_ra, center_dec",
    [
        (266.4168, -29.0078),  # Galactic Center
        (10.0, 85.0),  # Near North Pole (Stress test)
        (359.9, 0.0),  # RA Boundary crossing
    ],
)
def test_contains_center(center_ra, center_dec):
    """
    Verifies that the generated polygon strictly contains the input center point.
    Uses a local WCS projection to handle spherical geometry/wrapping correctly.
    """
    width = 0.5
    height = 0.5
    rot = 45  # arbitrary rotation

    # 1. Generate Polygon
    poly_str = compute_polygon_wcs_angular(center_ra, center_dec, width, height, rot)
    corners_world = parse_polygon_to_array(poly_str)

    # 2. Define a Local Test WCS
    # We project everything onto a flat plane centered on our test point
    # to perform a valid 2D point-in-polygon check.
    w_test = WCS(naxis=2)
    w_test.wcs.crval = [center_ra, center_dec]
    w_test.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # 3. Project Polygon Corners to Pixels
    corners_pix = w_test.wcs_world2pix(corners_world, 0)

    # 4. Project Center to Pixels (Should be 0,0
    # if CRPIX is default, but let's calculate it)
    center_pix = w_test.wcs_world2pix([[center_ra, center_dec]], 0)[0]

    # 5. Check Containment using Matplotlib Path
    # radius=0 ensures we check strict containment, not just "on the edge"
    poly_path = Path(corners_pix)
    is_inside = poly_path.contains_point(center_pix, radius=0)

    assert is_inside, (
        f"The center point ({center_ra}, {center_dec}) was not found inside "
        f"the generated polygon for width={width}, height={height}."
    )
