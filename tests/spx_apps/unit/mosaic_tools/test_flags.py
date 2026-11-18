"""Tests for FLAGS propagation in mosaic tool."""

import pytest
import numpy as np
from astropy import units as u
from astropy.wcs import WCS
from astropy.io import fits

from spx_data import Flag, ChannelInfo
from spx_apps.mosaic_tool.l2_loader import L2Image
from spx_apps.mosaic_tool.results import ChannelMosaic


@pytest.fixture
def simple_wcs():
    """Create a simple WCS for testing."""
    wcs = WCS(naxis=2)
    wcs.wcs.crpix = [16, 16]
    wcs.wcs.crval = [0.0, 0.0]
    wcs.wcs.cdelt = [0.01, 0.01]
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return wcs


class TestL2ImageFlags:
    """Test L2Image dataclass with flags field."""

    def test_l2image_has_flags_field(self, simple_wcs):
        """Test that L2Image dataclass includes flags field."""
        data = np.ones((32, 32)) * u.MJy / u.sr
        flags = np.zeros((32, 32), dtype=np.int32)
        flags[10, 10] = Flag.OVERFLOW

        image = L2Image(data=data, wcs=simple_wcs, array=1, filename="test.fits", flags=flags)

        assert image.flags is not None
        assert image.flags.dtype == np.int32
        assert image.flags[10, 10] == Flag.OVERFLOW
        assert image.flags[0, 0] == 0

    def test_l2image_flags_shape_matches_data(self, simple_wcs):
        """Test that flags array has same shape as data."""
        shape = (64, 48)
        data = np.ones(shape) * u.MJy / u.sr
        flags = np.zeros(shape, dtype=np.int32)

        image = L2Image(data=data, wcs=simple_wcs, array=2, filename="test.fits", flags=flags)

        assert image.flags.shape == image.data.shape


class TestChannelMosaicFlags:
    """Test ChannelMosaic with flags field."""

    def test_channel_mosaic_has_flags_field(self, simple_wcs):
        """Test that ChannelMosaic includes flags field."""
        data = np.ones((32, 32)) * u.MJy / u.sr
        nhits = np.ones((32, 32), dtype=np.int32)
        flags = np.zeros((32, 32), dtype=np.int32)
        flags[5:10, 5:10] = Flag.SOURCE

        info = ChannelInfo(detector=1, subchannel=1, wavelength=1.0*u.um)
        mosaic = ChannelMosaic(data=data, nhits=nhits, flags=flags, wcs=simple_wcs, channel_info=info)

        assert mosaic.flags is not None
        assert np.all(mosaic.flags[5:10, 5:10] == Flag.SOURCE)
        assert mosaic.flags[0, 0] == 0

    def test_channel_mosaic_write_includes_flags(self, simple_wcs, tmp_path):
        """Test that ChannelMosaic.write() includes FLAGS extension."""
        data = np.ones((32, 32)) * u.MJy / u.sr
        nhits = np.ones((32, 32), dtype=np.int32)
        flags = np.zeros((32, 32), dtype=np.int32)
        flags[10:20, 10:20] = Flag.OVERFLOW | Flag.SOURCE

        info = ChannelInfo(detector=1, subchannel=5, wavelength=1.5*u.um)
        mosaic = ChannelMosaic(data=data, nhits=nhits, flags=flags, wcs=simple_wcs, channel_info=info)

        output_path = tmp_path / "test_mosaic.fits"
        mosaic.write(str(output_path))

        with fits.open(output_path) as hdul:
            assert "FLAGS" in hdul
            assert hdul["FLAGS"].data.shape == (32, 32)
            assert hdul["FLAGS"].data.dtype.kind == 'i'  # integer
            assert hdul["FLAGS"].data.dtype.itemsize == 4  # 32-bit
            assert np.all(hdul["FLAGS"].data[10:20, 10:20] == (Flag.OVERFLOW | Flag.SOURCE))
            assert hdul["FLAGS"].data[0, 0] == 0
