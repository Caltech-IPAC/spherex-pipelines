import pytest
import numpy as np
from astropy import units as u
from astropy.wcs import WCS

from spx_data import Flag
from spx_apps.mosaic_tool.reprojection import (
    ReprojectionMethod,
    ExactReprojection,
    HistogramReprojection,
    InterpReprojection,
    get_method,
    list_methods,
)


@pytest.fixture
def simple_wcs():
    """Create a simple WCS for testing."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [16, 16]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def offset_wcs():
    """Create an offset WCS for testing reprojection."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [16, 16]
    wcs.wcs.crval = [0.05, 0.05]
    wcs.wcs.cdelt = [0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


@pytest.fixture
def sample_data():
    """Create sample data array with some structure and NaN values."""
    data = np.ones((32, 32), dtype=np.float32)
    data[10:20, 10:20] = 5.0
    data[5:8, 5:8] = np.nan
    return data


@pytest.fixture
def sample_flags():
    """Create sample flags array with different flag patterns."""
    flags = np.zeros((32, 32), dtype=np.int32)
    # Set some OVERFLOW flags
    flags[10:15, 10:15] = Flag.OVERFLOW
    # Set some SOURCE flags
    flags[15:20, 15:20] = Flag.SOURCE
    # Set both flags in overlap region (cast to int32 for in-place OR)
    flags[12:18, 12:18] |= np.int32(Flag.OVERFLOW | Flag.SOURCE)
    return flags


def test_list_and_get_methods():
    """Test method listing and retrieval."""
    methods = list_methods()
    assert set(methods) == {"reproject_exact", "reproject_hist", "reproject_interp"}

    assert isinstance(get_method("reproject_exact"), ExactReprojection)
    assert isinstance(get_method("reproject_hist"), HistogramReprojection)
    assert isinstance(get_method("reproject_interp"), InterpReprojection)

    with pytest.raises(ValueError, match="Unknown reprojection method"):
        get_method("invalid_method")


@pytest.mark.parametrize("method_class", [
    ExactReprojection,
    HistogramReprojection,
    InterpReprojection,
])
def test_reproject_method(method_class, sample_data, sample_flags, simple_wcs, offset_wcs):
    """Test reprojection method output shape, units, and interface."""
    method = method_class()
    shape_out = (24, 24)
    data_with_units = sample_data * u.MJy / u.sr

    # Test without flags
    reprojected, nhits, out_flags = method.reproject(
        data_with_units, simple_wcs, offset_wcs, shape_out
    )

    # Check interface
    assert isinstance(method, ReprojectionMethod)
    assert isinstance(method.name, str)
    assert isinstance(method.supports_origin, bool)

    # Check output
    assert reprojected.shape == shape_out
    assert nhits.shape == shape_out
    assert hasattr(reprojected, "unit")
    assert reprojected.unit == u.MJy / u.sr

    # Check nhits_dtype property matches actual nhits dtype
    assert nhits.dtype == method.nhits_dtype

    # Without flags input, output flags should be None
    assert out_flags is None

    # Test with flags
    reprojected, nhits, out_flags = method.reproject(
        data_with_units, simple_wcs, offset_wcs, shape_out, flags=sample_flags
    )

    # Check flags output
    assert out_flags is not None
    assert out_flags.shape == shape_out
    assert out_flags.dtype == np.int32


def test_interp_orders(sample_data, sample_flags, simple_wcs, offset_wcs):
    """Test InterpReprojection with different interpolation orders."""
    assert InterpReprojection().order == "nearest-neighbor"

    with pytest.raises(ValueError, match="Invalid order"):
        InterpReprojection(order="invalid")

    for order in ["nearest-neighbor", "bilinear", "biquadratic", "bicubic"]:
        method = InterpReprojection(order=order)
        assert method.order == order
        reprojected, nhits, out_flags = method.reproject(
            sample_data, simple_wcs, offset_wcs, (24, 24), flags=sample_flags
        )
        assert reprojected.shape == (24, 24)
        # Verify nhits dtype matches nhits_dtype property
        assert nhits.dtype == method.nhits_dtype
        # Verify flags output
        assert out_flags.shape == (24, 24)
        assert out_flags.dtype == np.int32


@pytest.mark.parametrize("order", ["bilinear", "bicubic"])
def test_interp_nan_no_bleeding(order):
    """Test that NaN values do not bleed into neighboring pixels during interpolation.

    Higher-order interpolation methods (bilinear, biquadratic, bicubic) sample
    multiple input pixels. Without proper handling, a single NaN pixel would
    contaminate all neighboring output pixels. This test verifies that the
    fill-and-renormalize pattern prevents NaN bleeding.

    Note: An isolated NaN pixel may be "healed" by interpolation from valid
    neighbors - this is expected. The key is that NaN doesn't SPREAD outward.
    To test that NaN is preserved where appropriate, we use a larger NaN region
    where the center has no valid neighbors to interpolate from.
    """
    # Create uniform data with a 5x5 block of NaN pixels in the center
    # The center of this block should remain NaN (no valid neighbors nearby)
    data = np.ones((21, 21), dtype=np.float32)
    nan_center_y, nan_center_x = 10, 10
    data[8:13, 8:13] = np.nan  # 5x5 NaN block

    # Create flags - set flags in the NaN region and outside
    flags = np.zeros((21, 21), dtype=np.int32)
    flags[8:13, 8:13] = Flag.OVERFLOW  # OVERFLOW in NaN region
    flags[3:6, 3:6] = Flag.SOURCE  # SOURCE in valid region

    # Create identity WCS (no transformation, just tests interpolation behavior)
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [11, 11]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    method = InterpReprojection(order=order)
    reprojected, nhits, out_flags = method.reproject(data, wcs, wcs, (21, 21), flags=flags)

    # The center of the NaN block should remain NaN (no valid data can reach it)
    assert np.isnan(reprojected[nan_center_y, nan_center_x]), \
        f"Center of NaN block should be NaN for {order}"

    # Pixels well outside the NaN block should NOT be NaN
    # These would be contaminated without proper NaN handling
    far_pixels = [
        (3, 3),
        (3, 17),
        (17, 3),
        (17, 17),
    ]
    for fy, fx in far_pixels:
        assert not np.isnan(reprojected[fy, fx]), \
            f"Far pixel ({fy}, {fx}) should NOT be NaN for {order} - NaN bleeding detected"
        assert np.isclose(reprojected[fy, fx], 1.0, atol=0.01), \
            f"Far pixel ({fy}, {fx}) should be ~1.0 for {order}, got {reprojected[fy, fx]}"

    # nhits at center of NaN block should be 0 or very small
    assert nhits[nan_center_y, nan_center_x] < 0.1, \
        f"nhits at NaN block center should be ~0 for {order}, got {nhits[nan_center_y, nan_center_x]}"

    # nhits at valid locations far from NaN should be ~1.0
    assert nhits[3, 3] > 0.9, \
        f"nhits at valid location should be ~1.0 for {order}, got {nhits[3, 3]}"

    # Flags should be reprojected - check SOURCE flag in valid region
    assert out_flags[4, 4] == Flag.SOURCE, \
        f"SOURCE flag should be preserved at (4,4), got {out_flags[4, 4]}"

    # Flags in NaN region should still be reprojected (flags use nearest-neighbor)
    assert out_flags[nan_center_y, nan_center_x] == Flag.OVERFLOW, \
        f"OVERFLOW flag should be preserved at NaN center, got {out_flags[nan_center_y, nan_center_x]}"


@pytest.mark.parametrize("method_class", [
    ExactReprojection,
    HistogramReprojection,
    InterpReprojection,
])
def test_flags_bitwise_or_accumulation(method_class):
    """Test that flags from multiple input pixels are combined via bitwise OR.

    When multiple input pixels map to the same output pixel, their flags
    should be combined using bitwise OR, preserving all set flag bits.
    """
    # Create a larger input that will be downsampled to smaller output
    # This ensures multiple input pixels map to each output pixel
    data = np.ones((20, 20), dtype=np.float32)

    # Create flags with different patterns in different quadrants
    flags = np.zeros((20, 20), dtype=np.int32)
    # Top-left quadrant: OVERFLOW only
    flags[0:10, 0:10] = Flag.OVERFLOW
    # Top-right quadrant: SOURCE only
    flags[0:10, 10:20] = Flag.SOURCE
    # Bottom-left quadrant: both flags
    flags[10:20, 0:10] = Flag.OVERFLOW | Flag.SOURCE
    # Bottom-right quadrant: no flags
    flags[10:20, 10:20] = 0

    # Input WCS covers 0.2 x 0.2 degrees
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = [10.5, 10.5]
    wcs_in.wcs.crval = [0.0, 0.0]
    wcs_in.wcs.cdelt = [0.01, 0.01]
    wcs_in.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Output WCS has same coverage but fewer pixels (downsampling)
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = [5.5, 5.5]
    wcs_out.wcs.crval = [0.0, 0.0]
    wcs_out.wcs.cdelt = [0.02, 0.02]  # 2x larger pixels
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    method = method_class()
    shape_out = (10, 10)
    _, _, out_flags = method.reproject(data, wcs_in, wcs_out, shape_out, flags=flags)

    assert out_flags is not None
    assert out_flags.dtype == np.int32

    # Check that flags are preserved in each quadrant
    # Top-left should have OVERFLOW
    assert out_flags[2, 2] & Flag.OVERFLOW, "Top-left should have OVERFLOW flag"
    # Top-right should have SOURCE
    assert out_flags[2, 7] & Flag.SOURCE, "Top-right should have SOURCE flag"
    # Bottom-left should have both
    assert out_flags[7, 2] & Flag.OVERFLOW, "Bottom-left should have OVERFLOW flag"
    assert out_flags[7, 2] & Flag.SOURCE, "Bottom-left should have SOURCE flag"
    # Bottom-right should have no flags
    assert out_flags[7, 7] == 0, f"Bottom-right should have no flags, got {out_flags[7, 7]}"


def test_histogram_flags_or_multiple_contributors():
    """Test bitwise OR when multiple input pixels contribute to same output pixel.

    This specifically tests the HistogramReprojection method where we have
    precise control over which input pixels map to which output pixels.
    """
    # Create input where we know exactly how pixels will map
    data = np.ones((4, 4), dtype=np.float32)

    # Create flags where adjacent pixels have different flag bits
    # When they map to the same output pixel, flags should be OR'd
    flags = np.zeros((4, 4), dtype=np.int32)
    flags[0, 0] = Flag.OVERFLOW
    flags[0, 1] = Flag.SOURCE
    flags[1, 0] = Flag.OVERFLOW
    flags[1, 1] = Flag.SOURCE

    # Input WCS
    wcs_in = WCS(naxis=2)
    wcs_in.wcs.crpix = [2.5, 2.5]
    wcs_in.wcs.crval = [0.0, 0.0]
    wcs_in.wcs.cdelt = [0.01, 0.01]
    wcs_in.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    # Output WCS with 2x larger pixels (4 input pixels -> 1 output pixel)
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.crpix = [1.5, 1.5]
    wcs_out.wcs.crval = [0.0, 0.0]
    wcs_out.wcs.cdelt = [0.02, 0.02]
    wcs_out.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    method = HistogramReprojection()
    _, _, out_flags = method.reproject(data, wcs_in, wcs_out, (2, 2), flags=flags)

    # The top-left output pixel should have both OVERFLOW and SOURCE
    # because it receives contributions from input pixels with different flags
    combined = out_flags[0, 0]
    assert combined & Flag.OVERFLOW, f"Should have OVERFLOW flag, got {combined}"
    assert combined & Flag.SOURCE, f"Should have SOURCE flag, got {combined}"
    expected = Flag.OVERFLOW | Flag.SOURCE
    assert combined == expected, f"Should have exactly OVERFLOW|SOURCE, got {combined}"
