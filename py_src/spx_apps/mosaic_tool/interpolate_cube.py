"""Standalone tool for spectral interpolation of existing mosaic cubes.

This tool reads an existing mosaic cube FITS file and produces a single
interpolated mosaic at a target wavelength. This reduces stripe artifacts
from extended sources with strong spectral features by averaging across
neighboring spectral channels.

Usage:
    python -m spx_apps.mosaic_tool.interpolate_cube input_cube.fits 4.80 -o output.fits

"""

__all__ = ["interpolate_cube", "load_cube_channels"]

import logging
import os
import sys
from typing import Optional
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
import astropy.units as u
import click

from spx_data import ChannelInfo
from .results import ChannelMosaic
from .spectral_interpolation import (
    InterpConfig,
    SpectralInterpolator,
    wavelength_to_array,
)


def load_cube_channels(cube_path: str) -> dict[tuple[int, int], ChannelMosaic]:
    """Load channel mosaics from an existing mosaic cube FITS file.

    Parameters
    ----------
    cube_path : str
        Path to the mosaic cube FITS file.

    Returns
    -------
    dict
        Mapping of (array_id, subchannel) to ChannelMosaic objects.

    Raises
    ------
    ValueError
        If the file format is invalid or missing required extensions.

    """
    logging.info(f"Loading mosaic cube from: {cube_path}")

    with fits.open(cube_path) as hdul:
        # Check required extensions
        required_exts = ["IMAGE", "NHITS", "FLAGS", "SPECTRAL_CHANNELS"]
        for ext in required_exts:
            if ext not in hdul:
                raise ValueError(f"Missing required extension: {ext}")

        # Load data cubes
        image_cube = hdul["IMAGE"].data
        nhits_cube = hdul["NHITS"].data
        flags_cube = hdul["FLAGS"].data

        # Get WCS from header
        wcs = WCS(hdul["IMAGE"].header, hdul, naxis=2)

        # Load spectral channel metadata
        spectral_table = hdul["SPECTRAL_CHANNELS"].data
        n_total = len(spectral_table)

        # Build ChannelMosaic objects for each channel
        channel_mosaics = {}
        n_skipped = 0

        for i in range(n_total):
            detector = int(spectral_table["DETECTOR"][i])
            subchannel = int(spectral_table["SUBCHAN"][i])

            # Skip interpolated channels (SUBCHAN=-1) - they shouldn't be used
            # as input for interpolation
            if subchannel < 0:
                logging.debug(f"Skipping interpolated channel (SUBCHAN={subchannel})")
                n_skipped += 1
                continue

            wavelength = float(spectral_table["WAVELENGTH"][i])

            # Get optional fields with defaults
            bandwidth = (
                spectral_table["BANDWIDTH"][i] if "BANDWIDTH" in spectral_table.dtype.names else np.nan
            )
            R = spectral_table["R"][i] if "R" in spectral_table.dtype.names else np.nan
            wl_min = spectral_table["WL_MIN"][i] if "WL_MIN" in spectral_table.dtype.names else np.nan
            wl_max = spectral_table["WL_MAX"][i] if "WL_MAX" in spectral_table.dtype.names else np.nan

            # Handle NaN values
            bandwidth = bandwidth if not np.isnan(bandwidth) else wavelength / R if not np.isnan(R) else None
            R = R if not np.isnan(R) else 41.0  # Default R if missing

            channel_info = ChannelInfo(
                detector=detector,
                subchannel=subchannel,
                wavelength=wavelength * u.um,
                bandwidth=bandwidth * u.um if bandwidth else None,
                R=R,
                wavelength_min=wl_min * u.um if not np.isnan(wl_min) else None,
                wavelength_max=wl_max * u.um if not np.isnan(wl_max) else None,
            )

            mosaic = ChannelMosaic(
                data=image_cube[i] * (u.MJy / u.sr),
                nhits=nhits_cube[i],
                flags=flags_cube[i],
                wcs=wcs,
                channel_info=channel_info,
                propagated_flags=None,
            )

            channel_mosaics[(detector, subchannel)] = mosaic

        n_loaded = len(channel_mosaics)
        if n_skipped > 0:
            logging.info(
                f"Loaded {n_loaded} channels from cube (skipped {n_skipped} interpolated), "
                f"shape: {image_cube.shape}"
            )
        else:
            logging.info(f"Loaded cube with {n_loaded} channels, shape: {image_cube.shape}")

        return channel_mosaics


def interpolate_cube(
    cube_path: str,
    target_wavelength: float,
    output_path: str,
    n_interp: float = 2.5,
    n_req: float = 0.55,
    array_id: Optional[int] = None,
) -> str:
    """Interpolate an existing mosaic cube to a target wavelength.

    This function reads a mosaic cube, selects channels near the target
    wavelength, and performs PCHIP interpolation to produce a single
    mosaic at the target wavelength. This reduces stripe artifacts from
    extended sources with strong spectral features.

    Parameters
    ----------
    cube_path : str
        Path to the input mosaic cube FITS file.
    target_wavelength : float
        Target wavelength in microns.
    output_path : str
        Path for the output interpolated mosaic FITS file.
    n_interp : float, optional
        Interpolation range in resolution widths. Channels within
        ±n_interp×Δλ of target are used. Default 2.5.
    n_req : float, optional
        Required contribution distance in resolution widths. Default 0.55.
    array_id : int, optional
        Force use of specific array (1-6). If None, determined from
        target wavelength automatically.

    Returns
    -------
    str
        Path to the output file.

    Examples
    --------
    >>> interpolate_cube(
    ...     'mosaic_cube.fits',
    ...     target_wavelength=4.80,
    ...     output_path='mosaic_interp_4.800um.fits',
    ...     n_interp=2.5,
    ... )

    """
    # Load channel mosaics from cube
    channel_mosaics = load_cube_channels(cube_path)

    # Determine array from wavelength if not specified
    if array_id is None:
        array_id = wavelength_to_array(target_wavelength)
    logging.info(f"Target wavelength {target_wavelength} μm -> Array {array_id}")

    # Get wavelength info for logging
    array_channels = [(ch, m) for ch, m in channel_mosaics.items() if ch[0] == array_id]
    if not array_channels:
        raise ValueError(f"No channels found for array {array_id} in cube")

    wavelengths = [m.channel_info.wavelength.to(u.um).value for _, m in array_channels]
    logging.info(
        f"Array {array_id} has {len(array_channels)} channels: "
        f"[{min(wavelengths):.4f}, {max(wavelengths):.4f}] μm"
    )

    # Get representative R for logging
    R = array_channels[0][1].channel_info.R
    delta_wl = target_wavelength / R
    wl_min = target_wavelength - n_interp * delta_wl
    wl_max = target_wavelength + n_interp * delta_wl
    logging.info(
        f"Interpolation range: [{wl_min:.4f}, {wl_max:.4f}] μm "
        f"(n_interp={n_interp}, R={R:.1f}, Δλ={delta_wl:.4f} μm)"
    )

    # Perform interpolation
    config = InterpConfig(n_interp=n_interp, n_req=n_req)
    interpolator = SpectralInterpolator(channel_mosaics, config)

    result = interpolator.interpolate(target_wavelength, array_id)

    # Save result
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result.write(output_path)
    logging.info(f"Interpolated mosaic saved to: {output_path}")

    # Log statistics
    valid_fraction = result.valid_fraction * 100
    logging.info(f"Valid pixels: {valid_fraction:.1f}%")

    return output_path


# CLI interface
@click.command()
@click.argument("cube_path", type=click.Path(exists=True))
@click.argument("target_wavelength", type=float)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(),
    help="Output file path. Default: mosaic_interp_<wavelength>um.fits",
)
@click.option(
    "--n-interp", type=float, default=2.5, show_default=True, help="Interpolation range in resolution widths."
)
@click.option(
    "--n-req",
    type=float,
    default=0.55,
    show_default=True,
    help="Required contribution distance in resolution widths.",
)
@click.option(
    "--array",
    "array_id",
    type=int,
    default=None,
    help="Force specific array (1-6). Default: auto-detect from wavelength.",
)
@click.option("-v", "--verbose", is_flag=True, help="Enable verbose logging.")
def main(cube_path, target_wavelength, output_path, n_interp, n_req, array_id, verbose):
    r"""Interpolate a mosaic cube to a target wavelength.

    Reads CUBE_PATH (a mosaic cube FITS file) and creates an interpolated
    mosaic at TARGET_WAVELENGTH (in microns). This reduces stripe artifacts
    from extended sources with strong spectral features.

    Examples
    --------
        # Interpolate to 4.80 μm
        python -m spx_apps.mosaic_tool.interpolate_cube mosaic_cube.fits 4.80

        # With custom output path and parameters
        python -m spx_apps.mosaic_tool.interpolate_cube mosaic_cube.fits 4.80 \\
            -o nebula_4.80um.fits --n-interp 3.0

    """
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s", stream=sys.stdout)

    # Default output path
    if output_path is None:
        output_path = f"mosaic_interp_{target_wavelength:.3f}um.fits"

    # Validate wavelength range
    if not (0.75 <= target_wavelength <= 5.0):
        raise click.BadParameter(
            f"Target wavelength {target_wavelength} μm outside SPHEREx range [0.75, 5.0]"
        )

    try:
        interpolate_cube(
            cube_path=cube_path,
            target_wavelength=target_wavelength,
            output_path=output_path,
            n_interp=n_interp,
            n_req=n_req,
            array_id=array_id,
        )
        click.echo(f"Output written to: {output_path}")
    except Exception as e:
        logging.error(f"Error: {e}")
        raise click.ClickException(str(e))


if __name__ == "__main__":
    main()
