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
