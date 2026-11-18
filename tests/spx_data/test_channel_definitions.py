"""Unit tests for spectral channel definitions.

Uses synthetic 50x50 data to test all channel definition types
without requiring external dependencies like SPHEREx_InstrumentSimulator.
"""

import numpy as np
import pytest
import astropy.units as u

from spx_data import (
    ArrayWavelengthModel,
    LogSpacedChannelDefinition,
    LinSpacedChannelDefinition,
    LVFContourChannelDefinition,
    LVFChannelDefinitionV2,
)


# Test data dimensions
NY, NX = 50, 50
N_CHANNELS = 5

# Wavelength range for test data (microns)
WL_MIN = 1.0
WL_MAX = 2.0


# --- Module-scoped fixtures for reuse across all tests ---


@pytest.fixture(scope="module")
def synthetic_wavelength_array():
    """Create synthetic wavelength array varying linearly in y (along columns)."""
    y_indices = np.arange(NY)
    wl_1d = WL_MIN + (WL_MAX - WL_MIN) * y_indices / (NY - 1)
    wl_array = np.broadcast_to(wl_1d[:, np.newaxis], (NY, NX)).astype(np.float32)
    return wl_array.copy()


@pytest.fixture(scope="module")
def synthetic_wavelength_model(synthetic_wavelength_array):
    """Create ArrayWavelengthModel with synthetic data for array 1."""
    return ArrayWavelengthModel({1: synthetic_wavelength_array})


@pytest.fixture(scope="module")
def synthetic_contour_map():
    """Create synthetic LVF contour map with 5 channels."""
    contour_map = np.zeros((NY, NX), dtype=np.int32)
    rows_per_channel = NY // N_CHANNELS
    for channel in range(1, N_CHANNELS + 1):
        y_start = (channel - 1) * rows_per_channel
        y_end = channel * rows_per_channel
        contour_map[y_start:y_end, :] = channel
    return contour_map


@pytest.fixture(scope="module")
def log_spaced_channel_def(synthetic_wavelength_model):
    """Create LogSpacedChannelDefinition for reuse."""
    return LogSpacedChannelDefinition(
        synthetic_wavelength_model, n_channels=N_CHANNELS, arrays=[1]
    )


@pytest.fixture(scope="module")
def lin_spaced_channel_def(synthetic_wavelength_model):
    """Create LinSpacedChannelDefinition for reuse."""
    return LinSpacedChannelDefinition(
        synthetic_wavelength_model, n_channels=N_CHANNELS, arrays=[1]
    )


@pytest.fixture(scope="module")
def lvf_contour_channel_def(synthetic_wavelength_model, synthetic_contour_map):
    """Create LVFContourChannelDefinition for reuse."""
    return LVFContourChannelDefinition(
        synthetic_contour_map, synthetic_wavelength_model, arrays=[1]
    )


# --- Tests ---


class TestArrayWavelengthModel:
    """Tests for ArrayWavelengthModel with synthetic data."""

    def test_wavelength_array(self, synthetic_wavelength_model):
        """Test wavelength array shape, range, and gradient."""
        wl_array = synthetic_wavelength_model.get_wavelength_array(1)
        assert wl_array.shape == (NY, NX)

        wl_min, wl_max = synthetic_wavelength_model.get_wavelength_range(1)
        assert np.isclose(wl_min, WL_MIN, rtol=1e-5)
        assert np.isclose(wl_max, WL_MAX, rtol=1e-5)

        # Wavelength increases along y
        assert np.all(wl_array[1:, 0] >= wl_array[:-1, 0])

    def test_missing_array_raises(self, synthetic_wavelength_model):
        """Test KeyError raised for missing array."""
        with pytest.raises(KeyError, match="No wavelength array provided"):
            synthetic_wavelength_model.get_wavelength_array(2)


class TestLogSpacedChannelDefinition:
    """Tests for LogSpacedChannelDefinition."""

    def test_get_all_channels(self, log_spaced_channel_def):
        """Test get_all_channels returns correct tuples."""
        channels = log_spaced_channel_def.get_all_channels()
        assert len(channels) == N_CHANNELS
        assert all(a == 1 for a, c in channels)
        assert [c for _, c in channels] == list(range(N_CHANNELS))

    def test_masks_partition_and_wavelengths_monotonic(self, log_spaced_channel_def):
        """Test masks partition array and wavelengths increase with channel."""
        combined_mask = np.zeros((NY, NX), dtype=bool)
        prev_wl = 0
        for channel in range(N_CHANNELS):
            mask = log_spaced_channel_def.get_channel_mask(1, channel, (NY, NX))
            assert mask.dtype == bool
            assert not np.any(combined_mask & mask), f"Overlap in channel {channel}"
            combined_mask |= mask

            wl = log_spaced_channel_def.get_channel_info(1, channel).wavelength.value
            assert wl > prev_wl
            prev_wl = wl

        assert np.all(combined_mask)

    def test_channel_wavelength_type_and_range(self, log_spaced_channel_def):
        """Test wavelength returns Quantity and range bounds are valid."""
        channel_info = log_spaced_channel_def.get_channel_info(1, 2)
        wl = channel_info.wavelength
        assert isinstance(wl, u.Quantity)
        assert wl.unit == u.um

        wl_min, wl_max = channel_info.wavelength_min, channel_info.wavelength_max
        assert isinstance(wl_min, u.Quantity)
        assert wl_min < wl_max

    def test_iter_channels(self, log_spaced_channel_def):
        """Test iter_channels yields correct channels."""
        results = list(log_spaced_channel_def.iter_channels(1, (NY, NX)))
        assert len(results) == N_CHANNELS
        assert all(isinstance(c, int) and m.shape == (NY, NX) for c, m in results)


class TestLinSpacedChannelDefinition:
    """Tests for LinSpacedChannelDefinition."""

    def test_linear_spacing_and_partition(self, lin_spaced_channel_def):
        """Test equal channel widths and masks partition array."""
        combined_mask = np.zeros((NY, NX), dtype=bool)
        widths = []

        for channel in range(N_CHANNELS):
            mask = lin_spaced_channel_def.get_channel_mask(1, channel, (NY, NX))
            assert not np.any(combined_mask & mask)
            combined_mask |= mask

            channel_info = lin_spaced_channel_def.get_channel_info(1, channel)
            widths.append((channel_info.wavelength_max - channel_info.wavelength_min).value)

        assert np.all(combined_mask)
        # All widths equal within tolerance
        assert np.allclose(widths, widths[0], rtol=1e-10)
        expected_width = (WL_MAX - WL_MIN) / N_CHANNELS
        assert np.isclose(widths[0], expected_width, rtol=1e-4)

    def test_channel_list_filtering(self, synthetic_wavelength_model):
        """Test channel_list parameter filters channels."""
        channel_def = LinSpacedChannelDefinition(
            synthetic_wavelength_model, n_channels=10, arrays=[1], channel_list=[2, 5, 7]
        )
        channels = channel_def.get_all_channels()
        assert [c for _, c in channels] == [2, 5, 7]

    def test_iter_channels_with_filter(self, lin_spaced_channel_def):
        """Test iter_channels with channel filter."""
        results = list(lin_spaced_channel_def.iter_channels(1, (NY, NX), channels=[1, 3]))
        assert [c for c, _ in results] == [1, 3]


class TestLVFContourChannelDefinition:
    """Tests for LVFContourChannelDefinition."""

    def test_channels_and_masks(self, lvf_contour_channel_def, synthetic_contour_map):
        """Test channel detection, mask matching, and partition."""
        channels = lvf_contour_channel_def.get_all_channels()
        assert [c for _, c in channels] == list(range(1, N_CHANNELS + 1))

        combined_mask = np.zeros((NY, NX), dtype=bool)
        prev_wl = 0

        for channel in range(1, N_CHANNELS + 1):
            mask = lvf_contour_channel_def.get_channel_mask(1, channel, (NY, NX))
            expected_mask = synthetic_contour_map == channel
            assert np.array_equal(mask, expected_mask)

            assert not np.any(combined_mask & mask)
            combined_mask |= mask

            channel_info = lvf_contour_channel_def.get_channel_info(1, channel)
            wl = channel_info.wavelength
            wl_min, wl_max = channel_info.wavelength_min, channel_info.wavelength_max
            assert wl_min <= wl <= wl_max
            assert wl.value > prev_wl
            prev_wl = wl.value

        assert np.all(combined_mask)


class TestCropToShape:
    """Tests for cropping behavior when data shapes differ."""

    def test_crop_to_smaller_shape(self, lin_spaced_channel_def, lvf_contour_channel_def):
        """Test masks work with smaller target shape."""
        small_shape = (40, 40)
        mask1 = lin_spaced_channel_def.get_channel_mask(1, 2, small_shape)
        mask2 = lvf_contour_channel_def.get_channel_mask(1, 3, small_shape)
        assert mask1.shape == small_shape
        assert mask2.shape == small_shape


class TestHelperMethods:
    """Tests for get_n_channels and get_channels_for_array."""

    def test_get_n_channels(self, log_spaced_channel_def,
                            synthetic_wavelength_model, synthetic_wavelength_array):
        """Test get_n_channels for various configurations."""
        assert log_spaced_channel_def.get_n_channels() == N_CHANNELS

        # With channel_list filter
        filtered = LinSpacedChannelDefinition(
            synthetic_wavelength_model, n_channels=64, arrays=[1], channel_list=[10, 20, 30]
        )
        assert filtered.get_n_channels() == 3

        # Multiple arrays
        wl_model = ArrayWavelengthModel({1: synthetic_wavelength_array,
                                         2: synthetic_wavelength_array})
        multi = LogSpacedChannelDefinition(wl_model, n_channels=5, arrays=[1, 2])
        assert multi.get_n_channels() == 10

    def test_get_channels_for_array(self, log_spaced_channel_def):
        """Test getting channels for specific array."""
        channels = log_spaced_channel_def.get_channels_for_array(1)
        assert len(channels) == N_CHANNELS
        assert all(a == 1 for a, c in channels)

        # Nonexistent array returns empty
        assert log_spaced_channel_def.get_channels_for_array(2) == []


class TestSubdivideContourMap:
    """Tests for LVFContourChannelDefinition.subdivide_contour_map static method."""

    @pytest.fixture
    def simple_contour_map(self):
        """Create a simple contour map with 3 channels for testing."""
        # 30x10 map with 3 channels, each 10 rows
        contour = np.zeros((30, 10), dtype=np.uint8)
        contour[0:10, :] = 1   # Channel 1: rows 0-9
        contour[10:20, :] = 2  # Channel 2: rows 10-19
        contour[20:30, :] = 3  # Channel 3: rows 20-29
        return contour

    def test_default_subdivisions(self, simple_contour_map):
        """Test subdivision with default n_subdivisions=4."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map, channels=[1, 2, 3], n_subdivisions=4
        )

        assert result.shape == simple_contour_map.shape
        assert result.dtype == np.uint8

        # Should have subchannels 1-12 (3 channels x 4 subdivisions)
        unique_values = np.unique(result)
        assert set(unique_values) == set(range(1, 13))

        # Max subchannel should be 12
        assert result.max() == 12

    def test_subchannel_ordering_within_channel(self, simple_contour_map):
        """Test that subchannels numbering within each channel."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map, channels=[1, 2, 3], n_subdivisions=4
        )

        # Channel 1 (rows 0-9) should map to subchannels 1-4
        # Higher y (row 9) should have subchannel 1,
        # lower y (row 0) should have subchannel 4
        ch1_subchannels = result[0:10, 0]

        # Subchannels should be in order 4,4,3,3,2,2,1,1,1,1 or similar
        # (exact distribution depends on rounding)
        # Key check: values should be in range 1-4 and generally decrease with y
        assert set(ch1_subchannels).issubset({1, 2, 3, 4})

        # First rows (low y) should have higher subchannel numbers
        # Last rows (high y) should have lower subchannel numbers
        assert ch1_subchannels[0] > ch1_subchannels[-1]

        # Channel 2 (rows 10-19) should map to subchannels 5-8
        ch2_subchannels = result[10:20, 0]
        assert set(ch2_subchannels).issubset({5, 6, 7, 8})
        assert ch2_subchannels[0] > ch2_subchannels[-1]

    def test_channels_subset(self, simple_contour_map):
        """Test subdivision with only a subset of channels."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map, channels=[2], n_subdivisions=4
        )

        # Only channel 2 should be subdivided (subchannels 1-4)
        # Channels 1 and 3 should be 0
        assert np.all(result[0:10, :] == 0)   # Channel 1 region -> 0
        assert np.all(result[20:30, :] == 0)  # Channel 3 region -> 0

        # Channel 2 region should have subchannels 1-4
        ch2_values = np.unique(result[10:20, :])
        assert set(ch2_values) == {1, 2, 3, 4}

    def test_n_subdivisions_1(self, simple_contour_map):
        """Test with n_subdivisions=1 (no actual subdivision)."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map, channels=[1, 2, 3], n_subdivisions=1
        )

        # Each channel maps to a single subchannel
        assert np.all(result[0:10, :] == 1)
        assert np.all(result[10:20, :] == 2)
        assert np.all(result[20:30, :] == 3)

    def test_n_subdivisions_2(self, simple_contour_map):
        """Test with n_subdivisions=2."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map, channels=[1, 2, 3], n_subdivisions=2
        )

        # Should have subchannels 1-6 (3 channels x 2 subdivisions)
        assert result.max() == 6
        unique_values = np.unique(result)
        assert set(unique_values) == set(range(1, 7))

    def test_invalid_n_subdivisions(self, simple_contour_map):
        """Test that invalid n_subdivisions raises ValueError."""
        with pytest.raises(ValueError, match="n_subdivisions must be between 1 and 7"):
            LVFContourChannelDefinition.subdivide_contour_map(
                simple_contour_map, n_subdivisions=0
            )

        with pytest.raises(ValueError, match="n_subdivisions must be between 1 and 7"):
            LVFContourChannelDefinition.subdivide_contour_map(
                simple_contour_map, n_subdivisions=8
            )

    def test_pixels_outside_channels_are_zero(self):
        """Test that pixels outside requested channels are set to 0."""
        # Create map with channels 1-3 and some 0 regions
        contour = np.zeros((40, 10), dtype=np.uint8)
        contour[5:15, :] = 1   # Channel 1
        contour[20:30, :] = 2  # Channel 2
        # Rows 0-4, 15-19, 30-39 are 0 (no channel)

        result = LVFContourChannelDefinition.subdivide_contour_map(
            contour, channels=[1, 2], n_subdivisions=4
        )

        # Original 0 regions should remain 0
        assert np.all(result[0:5, :] == 0)
        assert np.all(result[15:20, :] == 0)
        assert np.all(result[30:40, :] == 0)

        # Channel regions should have non-zero values
        assert np.all(result[5:15, :] > 0)
        assert np.all(result[20:30, :] > 0)

    def test_output_file_creation(self, simple_contour_map, tmp_path):
        """Test that output file is created when output_dir is provided."""
        result = LVFContourChannelDefinition.subdivide_contour_map(
            simple_contour_map,
            channels=[1, 2, 3],
            n_subdivisions=4,
            output_dir=str(tmp_path)
        )

        # Check file was created
        expected_file = tmp_path / "LVF_contour_12Subchannels.fits.gz"
        assert expected_file.exists()

        # Verify file contents
        from astropy.io import fits
        with fits.open(expected_file) as hdul:
            assert hdul[0].data.shape == simple_contour_map.shape
            assert np.array_equal(hdul[0].data, result)
            assert hdul[0].header['NSUBDIV'] == 4
            assert hdul[0].header['NCHAN'] == 3
            assert hdul[0].header['MAXSUB'] == 12

    def test_default_channels(self):
        """Test that default channels is [1, 2, ..., 17]."""
        # Create a map with channels 1-17
        contour = np.zeros((170, 10), dtype=np.uint8)
        for ch in range(1, 18):
            contour[(ch - 1) * 10:ch * 10, :] = ch

        result = LVFContourChannelDefinition.subdivide_contour_map(
            contour, n_subdivisions=4
        )

        # Should have 17 * 4 = 68 subchannels
        assert result.max() == 68


@pytest.mark.skip(reason="Reads the bundled LVF contour FITS file")
class TestSubdivideBundledContourMap:
    """Tests for LVFContourChannelDefinition.subdivide_bundled_contour_map."""

    def test_matches_manual_subdivision(self):
        """Test result matches subdivide_contour_map on the loaded bundled file."""
        from astropy.io import fits
        from spx_data.spectral_channel import get_lvf_contour_path

        with fits.open(get_lvf_contour_path()) as hdul:
            base_map = hdul[0].data

        expected = LVFContourChannelDefinition.subdivide_contour_map(
            base_map, channels=list(range(1, 18)), n_subdivisions=2
        )
        result = LVFContourChannelDefinition.subdivide_bundled_contour_map(
            n_subdivisions=2
        )

        assert result.shape == expected.shape
        assert np.array_equal(result, expected)

    def test_default_n_subdivisions(self):
        """Test default n_subdivisions=4 gives 17*4=68 max subchannel."""
        result = LVFContourChannelDefinition.subdivide_bundled_contour_map()
        assert result.max() == 68

    def test_n_subdivisions_1_matches_standard_channels(self):
        """Test n_subdivisions=1 gives standard channels 1-17."""
        result = LVFContourChannelDefinition.subdivide_bundled_contour_map(
            n_subdivisions=1
        )
        assert result.max() == 17
        assert set(np.unique(result)) <= set(range(0, 18))


class TestChannelNumbersFromMap:
    """Tests for LVFContourChannelDefinition.channel_numbers_from_map."""

    @pytest.fixture
    def half_map(self):
        """20x10 map: channel 1 in rows 0-9, channel 2 in rows 10-19,
        each subdivided into 2 subchannels (subchannel 1 at high y, 2 at low y
        within each channel, per subdivide_contour_map's convention).
        """
        contour = np.zeros((20, 10), dtype=np.uint8)
        contour[0:10, :] = 1
        contour[10:20, :] = 2
        return LVFContourChannelDefinition.subdivide_contour_map(
            contour, channels=[1, 2], n_subdivisions=2
        )

    def test_matches_direct_indexing(self, half_map):
        """Test lookup at exact integer pixel coordinates matches direct indexing."""
        y = np.array([2, 7, 12, 17])
        x = np.array([3, 5, 1, 8])
        result = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=x.astype(np.float32), y=y.astype(np.float32)
        )
        expected = half_map[y, x]
        assert np.array_equal(result, expected.astype(np.int32))

    def test_default_dtype_is_int32(self, half_map):
        """Test default output dtype is int32."""
        result = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=np.array([3.0]), y=np.array([2.0])
        )
        assert result.dtype == np.int32

    def test_custom_dtype(self, half_map):
        """Test custom dtype produces identical values in a narrower type."""
        x = np.array([3.0, 5.0, 1.0])
        y = np.array([2.0, 7.0, 12.0])
        default_result = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=x, y=y
        )
        u8_result = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=x, y=y, dtype=np.uint8
        )
        assert u8_result.dtype == np.uint8
        assert np.array_equal(u8_result.astype(np.int32), default_result)

    def test_no_channel_returns_zero(self):
        """Test pixels outside any channel return 0."""
        contour = np.zeros((10, 10), dtype=np.uint8)
        contour[2:5, :] = 1
        result = LVFContourChannelDefinition.channel_numbers_from_map(
            contour, x=np.array([0.0]), y=np.array([8.0])
        )
        assert result[0] == 0

    def test_rounding_at_half_integer_boundary(self, half_map):
        """Test ceil(v - 0.5) convention: v = i + 0.5 maps to pixel i, not i + 1."""
        y0, x0 = 2, 3
        exact = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=np.array([float(x0)]), y=np.array([float(y0)])
        )
        at_half = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=np.array([x0 + 0.5]), y=np.array([float(y0)])
        )
        assert at_half[0] == exact[0]

    def test_clip_out_of_bounds(self, half_map):
        """Test far out-of-range coordinates clip to the nearest edge pixel."""
        exact = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=np.array([0.0]), y=np.array([2.0])
        )
        out_of_range = LVFContourChannelDefinition.channel_numbers_from_map(
            half_map, x=np.array([-500.0]), y=np.array([2.0])
        )
        assert out_of_range[0] == exact[0]


class TestGlobalChannelNumbers:
    """Tests for LVFContourChannelDefinition.global_channel_numbers."""

    def test_matches_manual_formula(self):
        """Test result matches (array_id - 1) * n_channels_per_array + channel."""
        array_ids = np.array([1, 1, 2, 3, 6])
        channels = np.array([1, 34, 1, 17, 34])
        n_channels_per_array = 34

        result = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, channels, n_channels_per_array
        )
        expected = (array_ids - 1) * n_channels_per_array + channels
        assert np.array_equal(result, expected)

    def test_zero_channel_preserved_across_all_detectors(self):
        """Test channel=0 (no channel) stays 0 regardless of detector id."""
        array_ids = np.array([1, 2, 3, 4, 5, 6])
        channels = np.zeros(6, dtype=np.int32)

        result = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, channels, n_channels_per_array=34
        )
        assert np.array_equal(result, np.zeros(6))

    def test_default_dtype_is_int32(self):
        """Test default output dtype is int32."""
        result = LVFContourChannelDefinition.global_channel_numbers(
            np.array([1]), np.array([5]), n_channels_per_array=34
        )
        assert result.dtype == np.int32

    def test_custom_dtype(self):
        """Test custom dtype produces identical values in a wider type."""
        array_ids = np.array([1, 2, 3, 4, 5, 6])
        channels = np.array([1, 34, 17, 1, 20, 34])
        n_channels_per_array = 34

        default_result = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, channels, n_channels_per_array
        )
        u16_result = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, channels, n_channels_per_array, dtype=np.uint16
        )
        assert u16_result.dtype == np.uint16
        assert np.array_equal(u16_result.astype(np.int32), default_result)

    def test_range_spans_all_detectors(self):
        """Test global channel numbers span 1..(n_detectors * n_channels_per_array)."""
        n_channels_per_array = 34
        n_detectors = 6
        array_ids = np.repeat(np.arange(1, n_detectors + 1), n_channels_per_array)
        channels = np.tile(np.arange(1, n_channels_per_array + 1), n_detectors)

        result = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, channels, n_channels_per_array
        )
        assert result.min() == 1
        assert result.max() == n_detectors * n_channels_per_array
        assert len(np.unique(result)) == n_detectors * n_channels_per_array


# --- LVFChannelDefinitionV2 fixtures ---


@pytest.fixture(scope="module")
def v2_wavelength_model_2arrays(synthetic_wavelength_array):
    """ArrayWavelengthModel covering arrays 1 and 2 for V2 override tests."""
    return ArrayWavelengthModel(
        {1: synthetic_wavelength_array, 2: synthetic_wavelength_array}
    )


@pytest.fixture
def v2_sparse(synthetic_contour_map, synthetic_wavelength_model):
    """LVFChannelDefinitionV2 built with the default (sparse) stats_resolutions.

    n_subchannels=2 evenly divides the fixture's rows_per_channel (10), and
    the only precomputed stats resolution is 1 (the `build()` default).
    """
    return LVFChannelDefinitionV2.build(
        synthetic_contour_map, synthetic_wavelength_model, n_subchannels=2, arrays=[1],
        channels=list(range(1, N_CHANNELS + 1)),
    )


class TestLVFChannelDefinitionV2Build:
    """Tests for LVFChannelDefinitionV2.build()'s resolution/stats decoupling."""

    def test_sparse_stats_by_default(self, v2_sparse):
        """Test build() with no stats_resolutions only computes resolution 1."""
        assert v2_sparse.available_resolutions == [2, 1]
        assert v2_sparse.stats_resolutions == [1]

    def test_standard_resolution_stats_match_v1(
        self, v2_sparse, lvf_contour_channel_def
    ):
        """Test resolution-1 stats/masks match LVFContourChannelDefinition."""
        for channel in range(1, N_CHANNELS + 1):
            mask = v2_sparse.get_channel_mask(1, channel, (NY, NX))
            expected_mask = lvf_contour_channel_def.get_channel_mask(1, channel, (NY, NX))
            assert np.array_equal(mask, expected_mask)

            info = v2_sparse.get_channel_info(1, channel)
            expected_info = lvf_contour_channel_def.get_channel_info(1, channel)
            assert np.isclose(info.wavelength.value, expected_info.wavelength.value)

    def test_finer_resolution_geometry_without_stats(self, v2_sparse):
        """Test at_resolution(2) masks/get_global_channel_numbers work with no stats."""
        quarter = v2_sparse.at_resolution(2)
        assert quarter.spectral_stats is None
        assert quarter.n_channels_per_array == 2 * N_CHANNELS

        combined_mask = np.zeros((NY, NX), dtype=bool)
        for channel in range(1, quarter.n_channels_per_array + 1):
            mask = quarter.get_channel_mask(1, channel, (NY, NX))
            assert not np.any(combined_mask & mask)
            combined_mask |= mask
        assert np.all(combined_mask)

        y = np.array([1.0, 6.0])
        x = np.array([0.0, 0.0])
        channels = quarter.get_global_channel_numbers(
            array_ids=np.array([1, 1]), x=x, y=y, shape=(NY, NX)
        )
        assert channels[0] != 0 and channels[1] != 0

    def test_finer_resolution_info_is_nan_fallback(self, v2_sparse):
        """Test get_channel_info at a stats-less resolution returns NaN fallback."""
        quarter = v2_sparse.at_resolution(2)
        info = quarter.get_channel_info(1, 1)
        assert info.R is None
        assert info.R_std is None
        assert info.bandwidth is None
        assert np.isnan(info.wavelength.value)

    def test_finer_resolution_all_channels_geometric_fallback(self, v2_sparse):
        """Test get_all_channels() falls back to geometric enumeration."""
        quarter = v2_sparse.at_resolution(2)
        all_channels = quarter.get_all_channels()
        assert all_channels == [(1, c) for c in range(1, quarter.n_channels_per_array + 1)]

    def test_invalid_stats_resolutions_raises(
        self, synthetic_contour_map, synthetic_wavelength_model
    ):
        """Test stats_resolutions not dividing n_subchannels raises ValueError."""
        with pytest.raises(ValueError, match="not divisors"):
            LVFChannelDefinitionV2.build(
                synthetic_contour_map,
                synthetic_wavelength_model,
                n_subchannels=2,
                arrays=[1],
                channels=list(range(1, N_CHANNELS + 1)),
                stats_resolutions=[3],
            )

    def test_full_stats_resolutions_opt_in(
        self, synthetic_contour_map, synthetic_wavelength_model
    ):
        """Test passing every divisor precomputes stats at every resolution."""
        v2 = LVFChannelDefinitionV2.build(
            synthetic_contour_map,
            synthetic_wavelength_model,
            n_subchannels=2,
            arrays=[1],
            channels=list(range(1, N_CHANNELS + 1)),
            stats_resolutions=[2, 1],
        )
        assert v2.stats_resolutions == [2, 1]
        assert v2.at_resolution(2).spectral_stats is not None


@pytest.fixture
def v2_full(synthetic_contour_map, synthetic_wavelength_model):
    """LVFChannelDefinitionV2 at full-width (n_subchannels=1) resolution only.

    Models a calibration file that only ever stores standard channels --
    the case `at_resolution()`'s subdivide fallback is meant to bridge.
    """
    return LVFChannelDefinitionV2.build(
        synthetic_contour_map, synthetic_wavelength_model, n_subchannels=1, arrays=[1],
        channels=list(range(1, N_CHANNELS + 1)),
    )


class TestLVFChannelDefinitionV2AtResolutionSubdivide:
    """Tests for at_resolution()'s subdivide fallback from a full-width instance."""

    def test_subdivides_when_not_already_available(self, v2_full):
        """Test at_resolution(2) on a full-width instance derives new geometry."""
        assert v2_full.available_resolutions == [1]
        half = v2_full.at_resolution(2)
        assert half.available_resolutions == [2, 1]
        assert half.stats_resolutions == [1]
        assert half.spectral_stats is None

    def test_subdivided_masks_match_independent_build(
        self, v2_full, synthetic_contour_map, synthetic_wavelength_model
    ):
        """Test subdivide-fallback masks match building at that resolution directly."""
        half = v2_full.at_resolution(2)
        expected = LVFChannelDefinitionV2.build(
            synthetic_contour_map, synthetic_wavelength_model, n_subchannels=2,
            arrays=[1], channels=list(range(1, N_CHANNELS + 1)),
        ).at_resolution(2)

        combined_mask = np.zeros((NY, NX), dtype=bool)
        for channel in range(1, half.n_channels_per_array + 1):
            mask = half.get_channel_mask(1, channel, (NY, NX))
            expected_mask = expected.get_channel_mask(1, channel, (NY, NX))
            assert np.array_equal(mask, expected_mask)
            assert not np.any(combined_mask & mask)
            combined_mask |= mask
        assert np.all(combined_mask)

    def test_switch_back_to_full_width_restores_stats(self, v2_full):
        """Test at_resolution(1) after subdividing returns real stats again."""
        half = v2_full.at_resolution(2)
        standard = half.at_resolution(1)
        assert standard.spectral_stats is not None
        # Cheap view-switch back to a resolution half already supports:
        # the stats table is shared, not recomputed.
        assert standard.spectral_stats is v2_full.spectral_stats

        for channel in range(1, N_CHANNELS + 1):
            info = standard.get_channel_info(1, channel)
            expected_info = v2_full.get_channel_info(1, channel)
            assert np.isclose(info.wavelength.value, expected_info.wavelength.value)
            assert info.R == expected_info.R

    def test_already_available_resolution_does_not_subdivide(self, v2_full):
        """Test requesting an already-available resolution takes the cheap path."""
        view = v2_full.at_resolution(1)
        assert view.available_resolutions == [1]
        # Cheap view switch: shares the same stats table object, not rebuilt.
        assert view.spectral_stats is v2_full.spectral_stats

    def test_already_subdivided_instance_cannot_subdivide_further(self, v2_full):
        """Test at_resolution() on an already-subdivided instance raises clearly."""
        half = v2_full.at_resolution(2)
        with pytest.raises(ValueError, match="already subdivided"):
            half.at_resolution(4)


class TestLVFChannelDefinitionV2FromHdul:
    """Tests for LVFChannelDefinitionV2.from_hdul()'s backward compatibility."""

    @staticmethod
    def _make_stats_table(n_channels=N_CHANNELS):
        from astropy.table import Table

        return Table({
            "DETECTOR": np.ones(n_channels, dtype=np.uint8),
            "SUBCHAN": np.arange(1, n_channels + 1, dtype=np.uint8),
            "WAVELENGTH": np.linspace(1.0, 2.0, n_channels).astype(np.float32),
            "WL_MIN": np.linspace(0.9, 1.9, n_channels).astype(np.float32),
            "WL_MAX": np.linspace(1.1, 2.1, n_channels).astype(np.float32),
            "R": np.full(n_channels, 40.0, dtype=np.float32),
            "R_STD": np.full(n_channels, 1.0, dtype=np.float32),
            "BANDWIDTH": np.full(n_channels, 0.02, dtype=np.float32),
        })

    def test_old_bare_format(self, synthetic_contour_map):
        """Test the plain CHANNEL_MAP + SPECTRAL_CHANNELS format (no NSUBCHN header)."""
        from astropy.io import fits

        primary = fits.PrimaryHDU()
        channel_map = fits.ImageHDU(data=synthetic_contour_map, name="CHANNEL_MAP")
        stats = fits.BinTableHDU(self._make_stats_table(), name="SPECTRAL_CHANNELS")
        hdul = fits.HDUList([primary, channel_map, stats])

        assert "NSUBCHN" not in hdul[0].header

        v2 = LVFChannelDefinitionV2.from_hdul(hdul, arrays=[1])
        assert v2.available_resolutions == [1]
        assert v2.n_base_channels == N_CHANNELS
        mask = v2.get_channel_mask(1, 3, (NY, NX))
        assert np.array_equal(mask, synthetic_contour_map == 3)

    def test_bundled_format(self, synthetic_contour_map):
        """Test raw primary-HDU image data with no CHANNEL_MAP extension name."""
        from astropy.io import fits

        primary = fits.PrimaryHDU(data=synthetic_contour_map)
        stats = fits.BinTableHDU(self._make_stats_table(), name="SPECTRAL_CHANNELS")
        hdul = fits.HDUList([primary, stats])

        v2 = LVFChannelDefinitionV2.from_hdul(hdul, arrays=[1])
        assert v2.available_resolutions == [1]
        mask = v2.get_channel_mask(1, 2, (NY, NX))
        assert np.array_equal(mask, synthetic_contour_map == 2)

    def test_nsubchn_nchan_roundtrip_with_sparse_stats(
        self, synthetic_contour_map, synthetic_wavelength_model
    ):
        """Test NSUBCHN/NCHAN headers preserve geometric resolution across round-trip.

        Regression test: before reading these headers, n_subchannels was
        inferred as max(stats), which would silently collapse to 1 once
        stats_resolutions defaults to [1], losing at_resolution(2)'s
        geometric capability even though CHANNEL_MAP still holds the finer
        data.
        """
        v2 = LVFChannelDefinitionV2.build(
            synthetic_contour_map, synthetic_wavelength_model, n_subchannels=2, arrays=[1],
            channels=list(range(1, N_CHANNELS + 1)),
        )
        assert v2.available_resolutions == [2, 1]
        assert v2.stats_resolutions == [1]

        hdul = v2.to_hdul()
        assert "NSUBCHN" in hdul[0].header
        assert "NCHAN" in hdul[0].header

        reloaded = LVFChannelDefinitionV2.from_hdul(hdul, arrays=[1])
        assert reloaded.available_resolutions == [2, 1]
        assert reloaded.stats_resolutions == [1]

        quarter = reloaded.at_resolution(2)
        expected_quarter = v2.at_resolution(2)
        for channel in range(1, quarter.n_channels_per_array + 1):
            assert np.array_equal(
                quarter.get_channel_mask(1, channel, (NY, NX)),
                expected_quarter.get_channel_mask(1, channel, (NY, NX)),
            )

    def test_missing_channel_map_raises(self):
        """Test an HDUList with no image data at all raises a clear error."""
        from astropy.io import fits

        hdul = fits.HDUList([fits.PrimaryHDU()])
        with pytest.raises(ValueError, match="CHANNEL_MAP"):
            LVFChannelDefinitionV2.from_hdul(hdul)

    def test_missing_stats_raises(self, synthetic_contour_map):
        """Test an HDUList with a channel map but no stats extension raises."""
        from astropy.io import fits

        hdul = fits.HDUList([
            fits.PrimaryHDU(),
            fits.ImageHDU(data=synthetic_contour_map, name="CHANNEL_MAP"),
        ])
        with pytest.raises(ValueError, match="SPECTRAL_CHANNELS"):
            LVFChannelDefinitionV2.from_hdul(hdul)


class TestLVFChannelDefinitionV2GetGlobalChannelNumbers:
    """Tests for LVFChannelDefinitionV2.get_global_channel_numbers()."""

    @staticmethod
    def _reference_positions():
        rng = np.random.default_rng(0)
        n = 200
        array_ids = rng.integers(1, 3, size=n).astype(np.int32)
        x = rng.uniform(-0.5, NX - 0.5, size=n).astype(np.float32)
        y = rng.uniform(-0.5, NY - 0.5, size=n).astype(np.float32)
        return array_ids, x, y

    def test_matches_independent_lookup_no_override(
        self, synthetic_contour_map, v2_wavelength_model_2arrays
    ):
        """Test get_global_channel_numbers matches an independent per-array lookup.

        No per-detector override -- exercises the broadcast (stride-0)
        contour stack path and the group_size==1 fast path (n_subchannels=1,
        so the map's own values already are the final channel numbers).
        Validated against `LVFContourChannelDefinition.channel_numbers_from_map`
        -- a completely independent code path (direct indexing on the raw
        map, not the stacked/broadcast lookup) -- combined with
        `global_channel_numbers`.
        """
        v2 = LVFChannelDefinitionV2.build(
            synthetic_contour_map, v2_wavelength_model_2arrays, n_subchannels=1,
            arrays=[1, 2],
        )
        array_ids, x, y = self._reference_positions()

        per_detector = LVFContourChannelDefinition.channel_numbers_from_map(
            synthetic_contour_map, x=x, y=y
        )
        expected = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, per_detector, n_channels_per_array=v2.n_channels_per_array
        )
        fused = v2.get_global_channel_numbers(array_ids, x, y, shape=(NY, NX))
        assert np.array_equal(fused, expected)

    def test_matches_independent_lookup_with_override(
        self, synthetic_contour_map, v2_wavelength_model_2arrays
    ):
        """Test get_global_channel_numbers matches an independent lookup with override.

        Forces the materialized (non-broadcast) contour stack path: array 2
        looks up its own override map, array 1 the default map, combined
        exactly like `global_channel_numbers` does.
        """
        override_map = np.rot90(synthetic_contour_map).copy()
        v2 = LVFChannelDefinitionV2.build(
            synthetic_contour_map, v2_wavelength_model_2arrays, n_subchannels=1,
            arrays=[1, 2], detector_contour_maps={2: override_map},
        )
        array_ids, x, y = self._reference_positions()

        per_detector = np.where(
            array_ids == 2,
            LVFContourChannelDefinition.channel_numbers_from_map(override_map, x=x, y=y),
            LVFContourChannelDefinition.channel_numbers_from_map(
                synthetic_contour_map, x=x, y=y
            ),
        )
        expected = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, per_detector, n_channels_per_array=v2.n_channels_per_array
        )
        fused = v2.get_global_channel_numbers(array_ids, x, y, shape=(NY, NX))
        assert np.array_equal(fused, expected)
        # Sanity check the override actually took effect (results differ from
        # what a no-override build would have produced for array 2).
        assert np.any(per_detector[array_ids == 2] > 0)

    def test_matches_independent_lookup_at_coarser_resolution(
        self, synthetic_contour_map, synthetic_wavelength_model
    ):
        """Test get_global_channel_numbers on the general (group_size != 1) branch.

        Built with n_subchannels=2 but left at the default resolution 1
        (group_size=2), so the divmod/bucket0 arithmetic path runs, not the
        group_size==1 fast path exercised by the other tests in this class.
        """
        v2 = LVFChannelDefinitionV2.build(
            synthetic_contour_map, synthetic_wavelength_model, n_subchannels=2,
            arrays=[1], channels=list(range(1, N_CHANNELS + 1)),
        )
        assert v2.resolution == 1
        array_ids = np.ones(50, dtype=np.int32)
        rng = np.random.default_rng(1)
        x = rng.uniform(-0.5, NX - 0.5, size=50).astype(np.float32)
        y = rng.uniform(-0.5, NY - 0.5, size=50).astype(np.float32)

        fine = LVFContourChannelDefinition.channel_numbers_from_map(
            v2._contour_maps[None], x=x, y=y
        )
        # Standard-channel number at resolution 1 from a resolution-2 fine map:
        # base_idx0 = (fine - 1) // 2, channel = base_idx0 + 1.
        expected_per_detector = np.where(fine > 0, (fine - 1) // 2 + 1, 0)
        expected = LVFContourChannelDefinition.global_channel_numbers(
            array_ids, expected_per_detector, n_channels_per_array=v2.n_channels_per_array
        )
        fused = v2.get_global_channel_numbers(array_ids, x, y, shape=(NY, NX))
        assert np.array_equal(fused, expected)

    def test_out_of_range_fine_values_map_to_zero(self):
        """Test fine values beyond n_base_channels*n_subchannels are treated as invalid.

        Regression test: loaded/raw contour maps aren't guaranteed to already
        be clipped to this instance's declared channel count (e.g. the real
        bundled LVF template stores a channel 18 for excluded corner regions
        that `from_reference()` keeps as-is) -- such out-of-range positive
        values must resolve to global channel 0, not a bogus valid channel.
        """
        contour = np.zeros((NY, NX), dtype=np.int32)
        rows_per_channel = NY // N_CHANNELS
        for ch in range(1, N_CHANNELS + 1):
            contour[(ch - 1) * rows_per_channel:ch * rows_per_channel, :] = ch
        # Out-of-range value the instance's n_base_channels doesn't know about.
        contour[0:2, 0:2] = N_CHANNELS + 1

        from astropy.table import Table

        stats_table = Table({
            "DETECTOR": np.array([1], dtype=np.uint8),
            "SUBCHAN": np.array([1], dtype=np.uint8),
        })
        v2 = LVFChannelDefinitionV2(
            contour_maps={None: contour},
            stats={1: stats_table},
            n_subchannels=1,
            n_base_channels=N_CHANNELS,
            arrays=[1],
        )

        result = v2.get_global_channel_numbers(
            array_ids=np.array([1]), x=np.array([0.0]), y=np.array([0.0]),
            shape=(NY, NX),
        )
        assert result[0] == 0

        # A genuinely in-range pixel still resolves correctly.
        result_valid = v2.get_global_channel_numbers(
            array_ids=np.array([1]),
            x=np.array([0.0]),
            y=np.array([float(rows_per_channel + 1)]),
            shape=(NY, NX),
        )
        assert result_valid[0] == 2


@pytest.mark.skip(reason="Reads the bundled LVF contour FITS file")
class TestLVFChannelDefinitionV2FromReference:
    """Tests for LVFChannelDefinitionV2.from_reference()."""

    def test_matches_make_reference_table(self):
        """Test resolution-1 stats match make_reference_table() exactly."""
        from spx_data.spectral_channels_reference import make_reference_table

        v2 = LVFChannelDefinitionV2.from_reference()
        ref = make_reference_table()

        assert v2.available_resolutions == [1]
        assert v2.stats_resolutions == [1]
        assert v2.spectral_stats.colnames == ref.colnames
        for col in ref.colnames:
            assert np.array_equal(
                np.asarray(v2.spectral_stats[col]), np.asarray(ref[col])
            )

    def test_n_subchannels_greater_than_1_is_geometry_only(self):
        """Test n_subchannels > 1 subdivides geometry with no finer-resolution stats."""
        v2 = LVFChannelDefinitionV2.from_reference(n_subchannels=2)
        assert v2.available_resolutions == [2, 1]
        assert v2.stats_resolutions == [1]

        half = v2.at_resolution(2)
        assert half.spectral_stats is None
        mask = half.get_channel_mask(1, 1, shape=(2040, 2040))
        assert mask.any()
