"""HiPS tile mosaic building.

The HiPS production path is separate from the user-facing Mosaic Tool
surface (`MosaicConfig`, the `create-mosaic` CLI command) but shares its
accumulation core: `HipsMosaicBuilder` subclasses `MosaicBuilder`,
overriding only the output grid creation (HPX tile WCS instead of a
rectangular grid) and the sky-to-pixel mapping (HEALPix cell containment
instead of ``wcs.world2pix``, which is not valid across the whole sphere
for HPX — see the `hips_grid` module docstring).

One job builds one HiPS tile: a 512x512 spectral cube whose spatial
boundaries match the HEALPix NESTED cell `hips_npix` at order
`hips_order`, written under the HiPS directory convention
``Norder{order}/Dir{D}/Npix{npix}_cube.fits``.
"""

__all__ = [
    'HipsTileConfig',
    'HipsMosaicBuilder',
    'build_hips_tile_from_temp_butler',
]

import logging
import os
from dataclasses import dataclass
from typing import Optional

from astropy.io import fits
from spx_data import ChannelDefinition

from .l2_loader import L2FileLoader
from .hips_grid import (
    create_hips_tile_grid,
    compute_healpix_pixel_mapping,
    hips_pixel_scale_arcsec,
    hips_tile_relpath,
)
from .mosaic_builder import (
    BAD_PIX_FLAGS,
    PROPAGATED_FLAGS,
    MosaicBuilder,
    _build_and_write_cube,
    _find_l2_files,
    _get_channel_definition,
    _select_channels,
)
from ..resource_usage import log_resource_usage

# Allowed HiPS tile orders for the mosaic tool. Order 5 gives the nominal
# ~12.88 arcsec HiPS pixel; order 6 gives ~6.44 arcsec, still above the
# native 6.15 arcsec scale (no super-resolution); orders below 3 make
# tiles too large for the input query machinery (> ~7 degrees across).
HIPS_ORDER_MIN = 3
HIPS_ORDER_MAX = 6


@dataclass
class HipsTileConfig:
    """Configuration for building one HiPS tile.

    The tile fully determines the output grid (HPX projection, 512x512
    pixels), so there are no center/size/pixel-scale/rotation parameters.

    Parameters
    ----------
    hips_order : int
        HiPS tile order (HEALPix NSIDE = 2^order). Allowed range is
        [HIPS_ORDER_MIN, HIPS_ORDER_MAX].
    hips_npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^hips_order.
    bad_pixel_flags : int, optional
        Bitwise OR of bad pixel flags. Default is None.
    propagated_flags : int, optional
        Bitwise OR of pixel flags to be propagated. Default is None.

    Examples
    --------
    >>> config = HipsTileConfig(hips_order=5, hips_npix=1000)

    """

    hips_order: int
    hips_npix: int
    bad_pixel_flags: Optional[int] = None
    propagated_flags: Optional[int] = None

    def __post_init__(self):
        """Validate configuration."""
        if not HIPS_ORDER_MIN <= self.hips_order <= HIPS_ORDER_MAX:
            raise ValueError(
                f"hips_order must be in [{HIPS_ORDER_MIN}, {HIPS_ORDER_MAX}], "
                f"got {self.hips_order}"
            )
        npix_max = 12 * (1 << (2 * self.hips_order))
        if not 0 <= self.hips_npix < npix_max:
            raise ValueError(
                f"hips_npix must be in [0, {npix_max}) for order "
                f"{self.hips_order}, got {self.hips_npix}"
            )


class HipsMosaicBuilder(MosaicBuilder):
    """Build per-channel mosaics on a HiPS tile grid.

    Shares the file-centric accumulation core of `MosaicBuilder`; only
    the output grid (HPX tile WCS) and the sky-to-pixel mapping (HEALPix
    cell containment) differ. Every output mosaic header carries the
    HiPS ``ORDER`` and ``NPIX`` cards.

    Parameters
    ----------
    channel_definition : ChannelDefinition
        Defines how detector pixels map to spectral channels.
    config : HipsTileConfig
        Tile selection and processing configuration.
    loader : L2FileLoader, optional
        Custom file loader. Default creates new L2FileLoader.

    Examples
    --------
    >>> config = HipsTileConfig(hips_order=5, hips_npix=1000)
    >>> builder = HipsMosaicBuilder(channel_def, config)
    >>> mosaics = builder.build_all(l2_files)

    """

    def __init__(
        self,
        channel_definition: ChannelDefinition,
        config: HipsTileConfig,
        loader: Optional[L2FileLoader] = None
    ):
        super().__init__(channel_definition, config, loader)
        self._extra_header = fits.Header()
        self._extra_header['ORDER'] = (
            config.hips_order, 'HiPS tile order (HEALPix NSIDE = 2^ORDER)'
        )
        self._extra_header['NPIX'] = (
            config.hips_npix, 'HEALPix NESTED index of this tile'
        )

    def _create_grid(self, config: HipsTileConfig):
        """Create the HPX tile grid for the configured HiPS tile."""
        return create_hips_tile_grid(config.hips_order, config.hips_npix)

    def _compute_pixel_mapping(self, wcs_in, origin, crop_shape):
        """Map a cropped input region to tile pixels via HEALPix binning.

        Same contract as `MosaicBuilder._compute_pixel_mapping`, but the
        sky-to-pixel step bins by HEALPix cell containment instead of
        ``wcs.world2pix`` (wcslib's inverse HPX transform is not valid
        across the whole sphere; see the `hips_grid` module docstring).
        """
        ra, dec = self._input_sky_coords(wcs_in, origin, crop_shape)
        return compute_healpix_pixel_mapping(
            ra, dec, self.config.hips_order, self.config.hips_npix
        )


def build_hips_tile_from_temp_butler(
    temp_butler_root: str,
    hips_order: int,
    hips_npix: int,
    output_dir: str,
    channel_mosaics_dir: str,
    channel_range: Optional[tuple[float, float]] = None,
    compress: bool = False,
    user_channels: Optional[tuple[float, float]] = None
) -> str:
    """Build one HiPS tile spectral cube from a populated temp butler.

    HiPS counterpart of `build_mosaics_from_temp_butler`, sharing its
    workflow: find L2 files, load channel definitions, build per-channel
    mosaics on the tile grid, and combine them into a cube. The cube is
    written under the HiPS directory convention
    ``{output_dir}/Norder{order}/Dir{D}/Npix{npix}_cube.fits``.

    Parameters
    ----------
    temp_butler_root : str
        Path to the populated temp butler root directory.
    hips_order : int
        HiPS tile order (HEALPix NSIDE = 2^order).
    hips_npix : int
        HEALPix NESTED pixel index of the tile at NSIDE = 2^hips_order.
    output_dir : str
        Root directory for the output; the Norder/Dir hierarchy is
        created beneath it.
    channel_mosaics_dir : str
        Directory for individual channel mosaic files.
    channel_range : tuple[float, float], optional
        Wavelength range (min, max) in microns to filter channels.
        If None, all standard channels are processed.
    compress : bool, optional
        Whether to compress the output cube with gzip. Default is False.
    user_channels : tuple[float, float], optional
        Wavelength range (min, max) in microns for tophat user channels
        as an alternative to standard channels.

    Returns
    -------
    str
        Path to the output cube file.

    Raises
    ------
    ValueError
        If no L2 files are found or no mosaics are created.

    """
    log_resource_usage()
    l2_files_by_array = _find_l2_files(temp_butler_root)
    log_resource_usage()

    channel_def = _get_channel_definition(temp_butler_root, user_channels)
    log_resource_usage()

    channels_to_process = _select_channels(
        channel_def, l2_files_by_array, channel_range
    )
    logging.info(f"Processing {len(channels_to_process)} channels...")

    config = HipsTileConfig(
        hips_order=hips_order,
        hips_npix=hips_npix,
        bad_pixel_flags=BAD_PIX_FLAGS,
        propagated_flags=PROPAGATED_FLAGS
    )

    builder = HipsMosaicBuilder(channel_def, config)
    logging.info(
        f"HiPS tile grid: order {hips_order}, npix {hips_npix}, "
        f"{builder.shape_mosaic[0]} x {builder.shape_mosaic[1]} pixels, "
        f"{hips_pixel_scale_arcsec(hips_order):.2f} arcsec/pixel"
    )

    cube_filepath = os.path.join(
        output_dir, hips_tile_relpath(hips_order, hips_npix, suffix='_cube.fits')
    )
    return _build_and_write_cube(
        builder=builder,
        l2_files_by_array=l2_files_by_array,
        channels_to_process=channels_to_process,
        channel_mosaics_dir=channel_mosaics_dir,
        cube_filepath=cube_filepath,
        compress=compress
    )
