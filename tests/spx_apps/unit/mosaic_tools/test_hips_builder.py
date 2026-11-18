"""Tests for HiPS tile mosaic building (hips_builder.py).

Covers HipsTileConfig validation, the HipsMosaicBuilder grid and pixel
mapping wiring, end-to-end accumulation with synthetic L2 images (on a
face-6 tile east of the +/-180 HPX plane split, where wcslib world2pix
would fail), and ORDER/NPIX propagation into channel mosaic files and
the combined cube.
"""

import healpy as hp
import numpy as np
import pytest
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

from spx_data import ChannelInfo
from spx_apps.mosaic_tool.config import MosaicConfig
from spx_apps.mosaic_tool.hips_builder import (
    HipsMosaicBuilder,
    HipsTileConfig,
)
from spx_apps.mosaic_tool.hips_grid import (
    HIPS_TILE_SIZE,
    create_hips_tile_grid,
    hips_tile_relpath,
)
from spx_apps.mosaic_tool.l2_loader import L2Image
from spx_apps.mosaic_tool.mosaic_builder import MosaicBuilder
from spx_apps.mosaic_tool.results import ChannelMosaic, write_mosaic_cube

DETECTOR_SHAPE = (2040, 2040)

# Order-3 tile east of the face-6 split diagonal: the hardest case for
# the WCS machinery (wcslib world2pix is invalid there), so accumulation
# succeeding here proves the HEALPix mapping path is used.
ORDER = 3
NPIX = int(hp.xyf2pix(1 << ORDER, 5, 2, 6, nest=True))

# Masked detector region used by the fake channel definition
MASK_SLICE = (slice(1000, 1060), slice(980, 1040))
TEST_FLAG = 4
FLAGGED_PIXEL = (1010, 990)  # (y, x), inside MASK_SLICE


class FakeChannelDef:
    """Single-channel definition with a small rectangular mask."""

    def get_all_channels(self):
        return [(1, 1)]

    def get_channel_mask(self, array, subchannel, shape):
        mask = np.zeros(shape, dtype=bool)
        mask[MASK_SLICE] = True
        return mask

    def get_channel_info(self, array, subchannel):
        return ChannelInfo(detector=array, subchannel=subchannel,
                           wavelength=1.0 * u.um)


class FakeLoader:
    """Return synthetic L2 images with constant data per 'file'."""

    def __init__(self, wcs, values_by_file):
        self.wcs = wcs
        self.values_by_file = values_by_file

    def load(self, filepath, array):
        data = np.full(DETECTOR_SHAPE, self.values_by_file[filepath],
                       dtype=np.float32)
        flags = np.zeros(DETECTOR_SHAPE, dtype=np.int32)
        if filepath == "file_a.fits":
            flags[FLAGGED_PIXEL] = TEST_FLAG
        return L2Image(data=data * u.MJy / u.sr, wcs=self.wcs, array=array,
                       filename=filepath, flags=flags)


def make_input_wcs():
    """TAN WCS placing the masked detector region at the tile center."""
    ra_c, dec_c = hp.pix2ang(1 << ORDER, NPIX, nest=True, lonlat=True)
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [float(ra_c), float(dec_c)]
    # 1-based CRPIX at the center of the masked region
    wcs.wcs.crpix = [1010.5, 1030.5]
    wcs.wcs.cdelt = [-6.15 / 3600, 6.15 / 3600]
    return wcs


class TestHipsTileConfig:
    def test_valid(self):
        config = HipsTileConfig(hips_order=5, hips_npix=1000)
        assert config.hips_order == 5
        assert config.hips_npix == 1000

    @pytest.mark.parametrize("order", [2, 7, -1])
    def test_order_out_of_range(self, order):
        with pytest.raises(ValueError, match="hips_order"):
            HipsTileConfig(hips_order=order, hips_npix=0)

    @pytest.mark.parametrize("npix", [-1, 12 * 4**5])
    def test_npix_out_of_range(self, npix):
        with pytest.raises(ValueError, match="hips_npix"):
            HipsTileConfig(hips_order=5, hips_npix=npix)


class TestHipsMosaicBuilder:
    @pytest.fixture
    def builder(self):
        config = HipsTileConfig(hips_order=ORDER, hips_npix=NPIX)
        loader = FakeLoader(
            make_input_wcs(),
            {"file_a.fits": 2.0, "file_b.fits": 4.0},
        )
        return HipsMosaicBuilder(FakeChannelDef(), config, loader=loader)

    def test_grid_is_hpx_tile(self, builder):
        assert builder.shape_mosaic == (HIPS_TILE_SIZE, HIPS_TILE_SIZE)
        assert list(builder.wcs_mosaic.wcs.ctype) == ['RA---HPX', 'DEC--HPX']

    def test_extra_header_cards(self, builder):
        assert builder._extra_header['ORDER'] == ORDER
        assert builder._extra_header['NPIX'] == NPIX

    def test_end_to_end_accumulation(self, builder, tmp_path):
        """Two synthetic L2 images accumulate onto the HiPS tile grid."""
        mosaic = builder.build_channel(
            ["file_a.fits", "file_b.fits"], array=1, subchannel=1
        )
        assert mosaic is not None
        assert mosaic.nhits.shape == (HIPS_TILE_SIZE, HIPS_TILE_SIZE)

        # Expected hit pattern: reproduce the builder's own pixel mapping
        # over the mask bounding box (mapping-vs-healpy exactness is
        # covered in test_hips_grid) and histogram the masked pixels.
        mask = FakeChannelDef().get_channel_mask(1, 1, DETECTOR_SHAPE)
        y0, y1, x0, x1 = MosaicBuilder._get_bounding_box(mask)
        ix, iy, bounds = builder._compute_pixel_mapping(
            make_input_wcs(), (y0, x0), (y1 - y0, x1 - x0)
        )
        crop_mask = mask[y0:y1, x0:x1]
        valid = crop_mask & bounds
        assert valid.sum() == mask.sum()  # whole region lands in the tile

        expected_hits = np.zeros(builder.shape_mosaic, dtype=mosaic.nhits.dtype)
        np.add.at(expected_hits, (iy[valid], ix[valid]), 2)  # 2 files
        np.testing.assert_array_equal(mosaic.nhits, expected_hits)

        # Data is the per-cell mean over contributions: (2.0 + 4.0) / 2
        data = mosaic.data.value
        hit = expected_hits > 0
        np.testing.assert_allclose(data[hit], 3.0, rtol=1e-6)
        assert np.isnan(data[~hit]).all()

        # Flag from file_a propagated by bitwise OR to the mapped cell
        fy, fx = FLAGGED_PIXEL[0] - y0, FLAGGED_PIXEL[1] - x0
        assert mosaic.flags[iy[fy, fx], ix[fy, fx]] == TEST_FLAG
        assert mosaic.flags.sum() == TEST_FLAG  # nowhere else

        # ORDER/NPIX cards are written to the output file
        out = tmp_path / "tile_ch01.fits"
        mosaic.write(str(out))
        with fits.open(out) as hdul:
            assert hdul['IMAGE'].header['ORDER'] == ORDER
            assert hdul['IMAGE'].header['NPIX'] == NPIX


class TestMosaicBuilderRegression:
    """The template-method refactor must not change the rectangular path."""

    def test_rectangular_grid_unchanged(self):
        config = MosaicConfig(
            ra_center=322.0, dec_center=-6.0,
            width_deg=1.0, height_deg=1.0,
            pixel_size_arcsec=9.0, projection='TAN'
        )
        builder = MosaicBuilder(FakeChannelDef(), config)
        assert builder.shape_mosaic == (401, 401)
        assert list(builder.wcs_mosaic.wcs.ctype) == ['RA---TAN', 'DEC--TAN']
        assert builder._extra_header is None


class TestCubeOrderNpixPropagation:
    def test_cube_headers_carry_order_npix(self, tmp_path):
        tile_size = 32
        wcs, shape = create_hips_tile_grid(ORDER, NPIX, tile_size)
        extra = fits.Header()
        extra['ORDER'] = (ORDER, 'HiPS tile order (HEALPix NSIDE = 2^ORDER)')
        extra['NPIX'] = (NPIX, 'HEALPix NESTED index of this tile')

        paths = []
        for subchannel, wavelength in [(1, 1.0), (2, 1.1)]:
            mosaic = ChannelMosaic(
                data=np.ones(shape, dtype=np.float32) * u.MJy / u.sr,
                nhits=np.ones(shape, dtype=np.int32),
                flags=np.zeros(shape, dtype=np.int32),
                wcs=wcs,
                channel_info=ChannelInfo(detector=1, subchannel=subchannel,
                                         wavelength=wavelength * u.um),
                extra_header=extra
            )
            path = tmp_path / f"tile_ch{subchannel:02d}.fits"
            mosaic.write(str(path))
            paths.append(str(path))

        cube_path = tmp_path / "Npix_cube.fits"
        write_mosaic_cube(paths, str(cube_path), compress=False)
        with fits.open(cube_path) as hdul:
            for extname in ('IMAGE', 'NHITS', 'FLAGS'):
                assert hdul[extname].header['ORDER'] == ORDER
                assert hdul[extname].header['NPIX'] == NPIX
            # Spatial WCS still the HPX tile
            assert hdul['IMAGE'].header['CTYPE1'] == 'RA---HPX'


class TestHipsTileRelpath:
    def test_convention(self):
        assert hips_tile_relpath(5, 1000, suffix='_cube.fits') == \
               "Norder5/Dir0/Npix1000_cube.fits"
        assert hips_tile_relpath(5, 12287) == "Norder5/Dir10000/Npix12287.fits"
        assert hips_tile_relpath(6, 45123, suffix='_cube.fits') == \
               "Norder6/Dir40000/Npix45123_cube.fits"

    def test_validation(self):
        with pytest.raises(ValueError, match="npix"):
            hips_tile_relpath(3, 12 * 64)
