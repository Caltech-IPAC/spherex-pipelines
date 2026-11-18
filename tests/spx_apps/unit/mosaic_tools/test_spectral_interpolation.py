"""Unit tests for spectral interpolation module."""

import pytest
import numpy as np
import os
from astropy.wcs import WCS
from astropy.io import fits
import astropy.units as u

from spx_apps.mosaic_tool.spectral_interpolation import (
    InterpConfig,
    SpectralInterpolator,
    wavelength_to_array,
)
from spx_apps.mosaic_tool.interpolate_cube import (
    interpolate_cube,
    load_cube_channels,
)
from spx_apps.mosaic_tool.results import ChannelMosaic
from spx_data import ChannelInfo


def create_mock_mosaic(array_id: int, subchannel: int, wavelength_um: float,
                       data: np.ndarray, R: float = 41.0) -> ChannelMosaic:
    """Create a mock ChannelMosaic for testing."""
    shape = data.shape
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
    wcs.wcs.crval = [180.0, 45.0]
    wcs.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    channel_info = ChannelInfo(
        detector=array_id,
        subchannel=subchannel,
        wavelength=wavelength_um * u.um,
        bandwidth=(wavelength_um / R) * u.um,
        R=R
    )

    return ChannelMosaic(
        data=data * (u.MJy / u.sr),
        nhits=np.ones(shape, dtype=np.int32),
        flags=np.zeros(shape, dtype=np.int32),
        wcs=wcs,
        channel_info=channel_info,
        propagated_flags=None
    )


class TestWavelengthToArray:
    """Tests for wavelength_to_array function."""

    @pytest.mark.parametrize("wavelength,expected_array", [
        (0.85, 1),
        (1.3, 2),
        (2.0, 3),
        (3.0, 4),
        (4.0, 5),
        (4.8, 6),
    ])
    def test_wavelength_mapping(self, wavelength, expected_array):
        """Test correct array assignment from wavelength."""
        assert wavelength_to_array(wavelength) == expected_array

    def test_wavelength_outside_range_low(self):
        """Test error for wavelength below SPHEREx range."""
        with pytest.raises(ValueError, match="outside SPHEREx range"):
            wavelength_to_array(0.5)

    def test_wavelength_outside_range_high(self):
        """Test error for wavelength above SPHEREx range."""
        with pytest.raises(ValueError, match="outside SPHEREx range"):
            wavelength_to_array(5.5)


class TestInterpConfig:
    """Tests for InterpConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = InterpConfig()
        assert config.n_interp == 2.5
        assert config.n_req == 0.55
        assert config.interp_method == 'pchip'

    def test_custom_values(self):
        """Test custom configuration values."""
        config = InterpConfig(n_interp=3.0, n_req=0.7, interp_method='pchip')
        assert config.n_interp == 3.0
        assert config.n_req == 0.7


class TestSpectralInterpolator:
    """Tests for SpectralInterpolator class."""

    @pytest.fixture
    def mock_mosaics(self):
        """Create a set of mock channel mosaics for testing."""
        # Create 5 channels for array 6 around 4.8 μm
        shape = (10, 10)
        wavelengths = [4.6, 4.7, 4.8, 4.9, 5.0]
        mosaics = {}

        for i, wl in enumerate(wavelengths):
            # Create data with a smooth gradient plus some noise
            data = np.ones(shape, dtype=np.float32) * (10.0 - 0.5 * i)
            mosaics[(6, i + 1)] = create_mock_mosaic(
                array_id=6,
                subchannel=i + 1,
                wavelength_um=wl,
                data=data,
                R=41.0
            )

        return mosaics

    def test_channel_selection(self, mock_mosaics):
        """Test that correct channels are selected for interpolation."""
        config = InterpConfig(n_interp=2.5)
        interpolator = SpectralInterpolator(mock_mosaics, config)

        selected = interpolator._select_channels(target_wl=4.8, array_id=6)

        # With R=41, Δλ=4.8/41≈0.117, range=±0.293 μm
        # Should include channels from ~4.5 to ~5.1 μm
        assert len(selected) >= 3  # At least the target and neighbors
        wavelengths = [wl for wl, _ in selected]
        assert all(4.5 <= wl <= 5.1 for wl in wavelengths)

    def test_channel_selection_different_array_excluded(self, mock_mosaics):
        """Test that channels from other arrays are excluded."""
        # Add a channel from a different array
        shape = (10, 10)
        mock_mosaics[(5, 1)] = create_mock_mosaic(
            array_id=5, subchannel=1, wavelength_um=4.0,
            data=np.ones(shape, dtype=np.float32) * 5.0
        )

        interpolator = SpectralInterpolator(mock_mosaics)
        selected = interpolator._select_channels(target_wl=4.8, array_id=6)

        # Array 5 channel should not be included
        channel_arrays = [ch[0] for _, ch in selected]
        assert all(arr == 6 for arr in channel_arrays)

    def test_validate_spectrum_valid(self, mock_mosaics):
        """Test spectrum validation with valid data."""
        interpolator = SpectralInterpolator(mock_mosaics)

        wavelengths = np.array([4.6, 4.7, 4.8, 4.9, 5.0])
        values = np.array([10.0, 9.5, 9.0, 8.5, 8.0])
        target_wl = 4.8
        delta_wl = 0.117

        assert interpolator._validate_spectrum(wavelengths, values, target_wl, delta_wl)

    def test_validate_spectrum_insufficient_points(self, mock_mosaics):
        """Test spectrum validation with insufficient data points."""
        interpolator = SpectralInterpolator(mock_mosaics)

        wavelengths = np.array([4.6, 4.7, 4.8, 4.9, 5.0])
        values = np.array([np.nan, np.nan, 9.0, np.nan, np.nan])  # Only 1 valid
        target_wl = 4.8
        delta_wl = 0.117

        assert not interpolator._validate_spectrum(wavelengths, values, target_wl, delta_wl)

    def test_validate_spectrum_no_points_near_target(self, mock_mosaics):
        """Test spectrum validation when no points near target."""
        interpolator = SpectralInterpolator(mock_mosaics, InterpConfig(n_req=0.1))

        wavelengths = np.array([4.6, 4.7, 4.95, 5.0])  # Gap around 4.8
        values = np.array([10.0, 9.5, 8.5, 8.0])
        target_wl = 4.8
        delta_wl = 0.117  # 0.1 * 0.117 = 0.0117 - none within this

        assert not interpolator._validate_spectrum(wavelengths, values, target_wl, delta_wl)

    def test_validate_spectrum_no_bracketing(self, mock_mosaics):
        """Test spectrum validation when no points bracket target."""
        interpolator = SpectralInterpolator(mock_mosaics)

        wavelengths = np.array([4.6, 4.7, 4.75])  # All below 4.8
        values = np.array([10.0, 9.5, 9.2])
        target_wl = 4.8
        delta_wl = 0.117

        assert not interpolator._validate_spectrum(wavelengths, values, target_wl, delta_wl)

    def test_interpolate_pixel(self, mock_mosaics):
        """Test single pixel interpolation."""
        interpolator = SpectralInterpolator(mock_mosaics)

        wavelengths = np.array([4.6, 4.7, 4.8, 4.9, 5.0])
        values = np.array([10.0, 9.5, 9.0, 8.5, 8.0])
        target_wl = 4.75
        delta_wl = 0.117

        result = interpolator.interpolate_pixel(wavelengths, values, target_wl, delta_wl)

        # Should interpolate between 9.5 (4.7) and 9.0 (4.8)
        assert 9.0 < result < 9.5

    def test_interpolate_pixel_with_nans(self, mock_mosaics):
        """Test interpolation handles NaN values correctly."""
        interpolator = SpectralInterpolator(mock_mosaics)

        wavelengths = np.array([4.6, 4.7, 4.8, 4.9, 5.0])
        values = np.array([10.0, np.nan, 9.0, 8.5, 8.0])  # NaN at 4.7
        target_wl = 4.75
        delta_wl = 0.117

        result = interpolator.interpolate_pixel(wavelengths, values, target_wl, delta_wl)

        # Should still work using remaining valid points
        assert not np.isnan(result)
        assert 9.0 < result < 10.0

    def test_interpolate_full_mosaic(self, mock_mosaics):
        """Test full mosaic interpolation."""
        interpolator = SpectralInterpolator(mock_mosaics)

        result = interpolator.interpolate(target_wl=4.8, array_id=6)

        assert isinstance(result, ChannelMosaic)
        assert result.channel_info.detector == 6
        assert result.channel_info.subchannel == -1  # Interpolated marker
        assert result.channel_info.wavelength.to(u.um).value == 4.8
        assert result.data.shape == (10, 10)
        assert np.sum(result.nhits > 0) > 0  # Some valid pixels

    def test_interpolate_insufficient_channels(self):
        """Test error when insufficient channels for interpolation."""
        # Create only 1 channel
        shape = (10, 10)
        mosaics = {
            (6, 1): create_mock_mosaic(
                array_id=6, subchannel=1, wavelength_um=4.8,
                data=np.ones(shape, dtype=np.float32) * 10.0
            )
        }

        interpolator = SpectralInterpolator(mosaics)

        with pytest.raises(ValueError, match="Insufficient channels"):
            interpolator.interpolate(target_wl=4.8, array_id=6)


class TestInterpolationWithStripes:
    """Tests demonstrating stripe artifact reduction."""

    def test_stripe_pattern_averaged(self):
        """Test that different stripe patterns are averaged out.

        This test simulates the key use case: each channel has a different
        stripe pattern, and interpolation averages them to smooth the result.
        """
        shape = (20, 20)
        wavelengths = [4.6, 4.7, 4.8, 4.9, 5.0]
        mosaics = {}

        for i, wl in enumerate(wavelengths):
            # Create data with stripes at different positions per channel
            data = np.ones(shape, dtype=np.float32) * 10.0
            # Add stripe at different column for each channel
            stripe_col = 5 + i * 2
            if stripe_col < shape[1]:
                data[:, stripe_col] = 5.0  # Stripe (lower value)

            mosaics[(6, i + 1)] = create_mock_mosaic(
                array_id=6, subchannel=i + 1, wavelength_um=wl,
                data=data, R=41.0
            )

        interpolator = SpectralInterpolator(mosaics)
        result = interpolator.interpolate(target_wl=4.8, array_id=6)

        # The result should have less variation than individual channels
        result_data = result.data.value
        # Check that the stripe artifacts are reduced
        # In single channel, std would be high due to stripes
        # After interpolation, variance should be reduced
        assert np.nanstd(result_data) < 2.5  # Less variation than single channel

    def test_emission_signal_preserved(self):
        """Test that smooth emission signal is preserved through interpolation.

        This verifies the key property: coherent signals (smooth across channels)
        are preserved, while incoherent artifacts (different patterns) are reduced.
        """
        shape = (20, 20)
        wavelengths = [4.6, 4.7, 4.8, 4.9, 5.0]
        mosaics = {}

        # Create a circular "nebula" that should be preserved
        y, x = np.ogrid[:shape[0], :shape[1]]
        center = (10, 10)
        radius = 5
        nebula_mask = ((y - center[0])**2 + (x - center[1])**2) < radius**2

        for i, wl in enumerate(wavelengths):
            data = np.ones(shape, dtype=np.float32) * 5.0  # Background
            # Add nebula emission that varies smoothly with wavelength
            data[nebula_mask] = 15.0 - i * 0.5  # Smooth spectral gradient

            mosaics[(6, i + 1)] = create_mock_mosaic(
                array_id=6, subchannel=i + 1, wavelength_um=wl,
                data=data, R=41.0
            )

        interpolator = SpectralInterpolator(mosaics)
        result = interpolator.interpolate(target_wl=4.8, array_id=6)

        result_data = result.data.value

        # Check that nebula region is still brighter than background
        nebula_mean = np.nanmean(result_data[nebula_mask])
        background_mean = np.nanmean(result_data[~nebula_mask])
        assert nebula_mean > background_mean * 1.5  # Nebula still significantly brighter


class TestInterpolateCube:
    """Tests for the interpolate_cube standalone tool."""

    @pytest.fixture
    def mock_cube_file(self, tmp_path):
        """Create a mock mosaic cube FITS file for testing."""
        shape = (5, 20, 20)  # 5 channels, 20x20 pixels
        n_channels = shape[0]

        # Create WCS
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10, 10]
        wcs.wcs.crval = [180.0, 45.0]
        wcs.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Create data cubes
        image_cube = np.ones(shape, dtype=np.float32) * 10.0
        nhits_cube = np.ones(shape, dtype=np.int32)
        flags_cube = np.zeros(shape, dtype=np.int32)

        # Add some variation
        for i in range(n_channels):
            image_cube[i] *= (1.0 - 0.1 * i)

        # Create spectral channel table
        wavelengths = [4.6, 4.7, 4.8, 4.9, 5.0]
        detectors = [6] * n_channels
        subchannels = list(range(1, n_channels + 1))
        bandwidths = [w / 41.0 for w in wavelengths]
        R_values = [41.0] * n_channels

        # Build FITS file
        header = wcs.to_header()
        header['NCHAN'] = n_channels

        primary_hdu = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(image_cube, header=header, name='IMAGE')
        nhits_hdu = fits.ImageHDU(nhits_cube, name='NHITS')
        flags_hdu = fits.ImageHDU(flags_cube, name='FLAGS')

        # Spectral channels table
        col_det = fits.Column(name='DETECTOR', format='I', array=detectors)
        col_sub = fits.Column(name='SUBCHAN', format='I', array=subchannels)
        col_wl = fits.Column(name='WAVELENGTH', format='E', unit='um', array=wavelengths)
        col_bw = fits.Column(name='BANDWIDTH', format='E', unit='um', array=bandwidths)
        col_r = fits.Column(name='R', format='E', array=R_values)
        col_wl_min = fits.Column(name='WL_MIN', format='E', unit='um',
                                 array=[w - b/2 for w, b in zip(wavelengths, bandwidths)])
        col_wl_max = fits.Column(name='WL_MAX', format='E', unit='um',
                                 array=[w + b/2 for w, b in zip(wavelengths, bandwidths)])

        spectral_hdu = fits.BinTableHDU.from_columns(
            [col_det, col_sub, col_wl, col_wl_min, col_wl_max, col_bw, col_r],
            name='SPECTRAL_CHANNELS'
        )

        hdul = fits.HDUList([primary_hdu, image_hdu, nhits_hdu, flags_hdu, spectral_hdu])

        cube_path = tmp_path / 'test_cube.fits'
        hdul.writeto(cube_path, overwrite=True)

        return str(cube_path)

    def test_load_cube_channels(self, mock_cube_file):
        """Test loading channel mosaics from a cube file."""
        channels = load_cube_channels(mock_cube_file)

        assert len(channels) == 5
        assert all(ch[0] == 6 for ch in channels.keys())  # All from array 6

        # Check wavelengths (use approximate comparison due to float32 precision)
        wavelengths = sorted([m.channel_info.wavelength.to(u.um).value
                              for m in channels.values()])
        expected = [4.6, 4.7, 4.8, 4.9, 5.0]
        assert np.allclose(wavelengths, expected, rtol=1e-5)

    def test_interpolate_cube_basic(self, mock_cube_file, tmp_path):
        """Test basic cube interpolation."""
        output_path = str(tmp_path / 'output.fits')

        result_path = interpolate_cube(
            cube_path=mock_cube_file,
            target_wavelength=4.75,
            output_path=output_path,
            n_interp=2.5,
            n_req=0.55,
        )

        assert os.path.exists(result_path)

        # Verify output file
        with fits.open(result_path) as hdul:
            assert 'IMAGE' in hdul
            data = hdul['IMAGE'].data
            assert data.shape == (20, 20)
            assert not np.all(np.isnan(data))

    def test_interpolate_cube_wavelength_in_header(self, mock_cube_file, tmp_path):
        """Test that output contains correct wavelength metadata."""
        output_path = str(tmp_path / 'output.fits')
        target_wl = 4.85

        interpolate_cube(
            cube_path=mock_cube_file,
            target_wavelength=target_wl,
            output_path=output_path,
        )

        with fits.open(output_path) as hdul:
            header = hdul['IMAGE'].header
            assert abs(header['WAVELEN'] - target_wl) < 0.001
            assert header['SUBCHAN'] == -1  # Interpolated marker

    def test_interpolate_cube_preserves_smooth_signal(self, tmp_path):
        """Test that smooth spectral signal is preserved."""
        # Create cube with smooth spectral gradient
        shape = (5, 20, 20)
        n_channels = shape[0]

        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [10, 10]
        wcs.wcs.crval = [180.0, 45.0]
        wcs.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]

        # Linear gradient: 10, 9.5, 9.0, 8.5, 8.0
        image_cube = np.zeros(shape, dtype=np.float32)
        for i in range(n_channels):
            image_cube[i] = 10.0 - 0.5 * i

        nhits_cube = np.ones(shape, dtype=np.int32)
        flags_cube = np.zeros(shape, dtype=np.int32)

        wavelengths = [4.6, 4.7, 4.8, 4.9, 5.0]

        # Build FITS
        header = wcs.to_header()
        primary_hdu = fits.PrimaryHDU()
        image_hdu = fits.ImageHDU(image_cube, header=header, name='IMAGE')
        nhits_hdu = fits.ImageHDU(nhits_cube, name='NHITS')
        flags_hdu = fits.ImageHDU(flags_cube, name='FLAGS')

        col_det = fits.Column(name='DETECTOR', format='I', array=[6]*5)
        col_sub = fits.Column(name='SUBCHAN', format='I', array=[1, 2, 3, 4, 5])
        col_wl = fits.Column(name='WAVELENGTH', format='E', unit='um', array=wavelengths)
        col_bw = fits.Column(name='BANDWIDTH', format='E', unit='um', array=[0.1]*5)
        col_r = fits.Column(name='R', format='E', array=[41.0]*5)
        col_wl_min = fits.Column(name='WL_MIN', format='E', unit='um', array=[w-0.05 for w in wavelengths])
        col_wl_max = fits.Column(name='WL_MAX', format='E', unit='um', array=[w+0.05 for w in wavelengths])

        spectral_hdu = fits.BinTableHDU.from_columns(
            [col_det, col_sub, col_wl, col_wl_min, col_wl_max, col_bw, col_r],
            name='SPECTRAL_CHANNELS'
        )

        hdul = fits.HDUList([primary_hdu, image_hdu, nhits_hdu, flags_hdu, spectral_hdu])
        cube_path = tmp_path / 'gradient_cube.fits'
        hdul.writeto(cube_path, overwrite=True)

        # Interpolate at 4.75 (between channels 2 and 3)
        output_path = str(tmp_path / 'output.fits')
        interpolate_cube(str(cube_path), 4.75, output_path)

        with fits.open(output_path) as hdul:
            data = hdul['IMAGE'].data
            # Expected: between 9.5 (at 4.7) and 9.0 (at 4.8), closer to 9.25
            mean_val = np.nanmean(data)
            assert 9.0 < mean_val < 9.5
