#!/usr/bin/env python
"""Plot a summary grid of all spectral channels from a mosaic cube.

Creates a multi-panel figure showing all spectral channels from a mosaic_cube.fits
file, with wavelength labels and WCS coordinate axes.

Usage:
    python plot_mosaic_cube.py mosaic_cube.fits
    python plot_mosaic_cube.py mosaic_cube.fits --layer nhits
"""

import argparse
import math
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


def load_mosaic_cube(filepath, layer='image'):
    """Load mosaic cube FITS file.

    Parameters
    ----------
    filepath : str
        Path to mosaic_cube.fits file.
    layer : str
        Data layer to load: 'image' or 'nhits'.

    Returns
    -------
    data : numpy.ndarray
        3D data cube (nchannels, ny, nx).
    wcs_2d : astropy.wcs.WCS
        2D WCS for spatial axes.
    wavelengths : numpy.ndarray
        Wavelength array in microns from WCS-TAB lookup.
    """
    layer = layer.upper()
    if layer not in ('IMAGE', 'NHITS'):
        raise ValueError(f"Invalid layer '{layer}'. Must be 'image' or 'nhits'.")

    with fits.open(filepath) as hdul:
        # Get data from specified extension
        data = hdul[layer].data
        # Use IMAGE header for WCS (NHITS has same spatial WCS)
        header = hdul['IMAGE'].header

        # Extract 2D spatial WCS (drop spectral axes)
        wcs_full = WCS(header, hdul)
        wcs_2d = wcs_full.celestial

        # Get wavelengths from WCS-TAB lookup table
        if 'WCS-WAVE' in hdul:
            wave_table = hdul['WCS-WAVE'].data
            # WAVELENGTHS column contains wavelengths for each plane
            wavelengths = wave_table['WAVELENGTHS'].flatten()
        elif 'WAVELENGTH' in hdul:
            # Fallback to WAVELENGTH table
            wave_table = hdul['WAVELENGTH'].data
            wavelengths = wave_table['WAVELENGTH']
        else:
            # Generate indices if no wavelength info available
            n_channels = data.shape[0]
            wavelengths = np.arange(n_channels)

    return data, wcs_2d, wavelengths


def calculate_grid_layout(n_channels, max_cols=17):
    """Calculate optimal grid layout for panels.

    Parameters
    ----------
    n_channels : int
        Number of spectral channels.
    max_cols : int
        Maximum number of columns.

    Returns
    -------
    nrows : int
        Number of rows.
    ncols : int
        Number of columns.
    """
    ncols = min(n_channels, max_cols)
    nrows = math.ceil(n_channels / ncols)
    return nrows, ncols


def get_vlim_log(data, q_low=0.01, q_high=0.99):
    """Calculate color limits for log-scale display.

    Uses data-driven quantiles to determine appropriate color scale range.

    Parameters
    ----------
    data : numpy.ndarray
        Data array.
    q_low : float
        Lower quantile for vmin (default: 1%).
    q_high : float
        Upper quantile for vmax (default: 99%).

    Returns
    -------
    vmin : float
        Minimum value for colorbar (log10).
    vmax : float
        Maximum value for colorbar (log10).
    """
    # Filter out non-positive values for log scale
    valid = data[np.isfinite(data) & (data > 0)]
    if len(valid) == 0:
        return -1, 1

    # Use quantiles of the positive data
    vmin_lin = np.quantile(valid, q_low)
    vmax_lin = np.quantile(valid, q_high)

    # Convert to log scale
    vmin = np.log10(vmin_lin)
    vmax = np.log10(vmax_lin)

    # Ensure reasonable dynamic range (at least 0.5 decades, at most 3 decades)
    if vmax - vmin < 0.5:
        mid = (vmax + vmin) / 2
        vmin = mid - 0.25
        vmax = mid + 0.25
    elif vmax - vmin > 3:
        vmin = vmax - 3

    return vmin, vmax


def get_vlim_linear(data, q_low=0.01, q_high=0.99):
    """Calculate color limits for linear-scale display.

    Parameters
    ----------
    data : numpy.ndarray
        Data array.
    q_low : float
        Lower quantile for vmin (default: 1%).
    q_high : float
        Upper quantile for vmax (default: 99%).

    Returns
    -------
    vmin : float
        Minimum value for colorbar.
    vmax : float
        Maximum value for colorbar.
    """
    valid = data[np.isfinite(data)]
    if len(valid) == 0:
        return 0, 1

    vmin = np.quantile(valid, q_low)
    vmax = np.quantile(valid, q_high)

    # Ensure vmin is at least 0 for count data
    vmin = max(0, vmin)

    return vmin, vmax


def get_coordinate_range(wcs_2d, ny, nx):
    """Get RA and Dec range from WCS.

    Parameters
    ----------
    wcs_2d : astropy.wcs.WCS
        2D WCS for spatial axes.
    ny, nx : int
        Image dimensions.

    Returns
    -------
    ra_range : tuple
        (ra_min, ra_max) in degrees.
    dec_range : tuple
        (dec_min, dec_max) in degrees.
    """
    # Get corner coordinates
    corners_pix = np.array([
        [0, 0],
        [nx - 1, 0],
        [0, ny - 1],
        [nx - 1, ny - 1]
    ])
    corners_world = wcs_2d.pixel_to_world_values(corners_pix[:, 0], corners_pix[:, 1])
    ra_corners = corners_world[0]
    dec_corners = corners_world[1]

    ra_min, ra_max = np.min(ra_corners), np.max(ra_corners)
    dec_min, dec_max = np.min(dec_corners), np.max(dec_corners)

    return (ra_min, ra_max), (dec_min, dec_max)


def format_coord_range(ra_range, dec_range):
    """Format RA and Dec range as a string.

    Parameters
    ----------
    ra_range : tuple
        (ra_min, ra_max) in degrees.
    dec_range : tuple
        (dec_min, dec_max) in degrees.

    Returns
    -------
    str
        Formatted coordinate range string.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    # Convert to sexagesimal
    coord_min = SkyCoord(ra=ra_range[0] * u.deg, dec=dec_range[0] * u.deg)
    coord_max = SkyCoord(ra=ra_range[1] * u.deg, dec=dec_range[1] * u.deg)

    ra_str = (f"RA: {coord_min.ra.to_string(u.hour, precision=0)} \u2013"
              f" {coord_max.ra.to_string(u.hour, precision=0)}")
    dec_str = (f"Dec: {coord_min.dec.to_string(u.deg, precision=0)} \u2013"
               f" {coord_max.dec.to_string(u.deg, precision=0)}")

    return f"{ra_str}    {dec_str}"


def plot_mosaic_cube(filepath, output_path=None, max_cols=17, figsize_per_panel=1.0,
                     fontsize_wavelength=8, log=None, layer='image',
                     vlim=None, cmap=None, show=False):
    """Create summary plot of mosaic cube.

    Parameters
    ----------
    filepath : str
        Path to mosaic_cube.fits file.
    output_path : str, optional
        Output file path. If None, uses input name with .png extension.
    max_cols : int
        Maximum number of columns in the grid.
    figsize_per_panel : float
        Figure size per panel in inches.
    fontsize_wavelength : int
        Font size for wavelength labels.
    log : bool, optional
        Use log scale for data. If None, defaults to True for IMAGE, False for NHITS.
    layer : str
        Data layer to plot: 'image' or 'nhits'.
    vlim : tuple, optional
        (vmin, vmax) for color scale. If None, computed automatically.
    cmap : str, optional
        Colormap name. If None, defaults to 'inferno' for IMAGE, 'viridis' for NHITS.
    show : bool
        Whether to display the plot interactively.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    """
    import matplotlib.pyplot as plt

    # Load data
    data, wcs_2d, wavelengths = load_mosaic_cube(filepath, layer=layer)
    n_channels, ny, nx = data.shape

    # Set layer-specific defaults
    layer_upper = layer.upper()
    if log is None:
        log = (layer_upper == 'IMAGE')
    if cmap is None:
        cmap = 'inferno' if layer_upper == 'IMAGE' else 'viridis'

    # Calculate grid layout
    nrows, ncols = calculate_grid_layout(n_channels, max_cols)

    # Calculate figure size
    figwidth = ncols * figsize_per_panel
    figheight = nrows * figsize_per_panel + 0.8  # Extra space for colorbar and header

    # Set up colormap
    colormap = plt.get_cmap(cmap)
    colormap.set_bad('gray', alpha=0.5)

    # Calculate color limits
    if vlim is None:
        if log:
            vmin, vmax = get_vlim_log(data)
        else:
            vmin, vmax = get_vlim_linear(data)
    else:
        vmin, vmax = vlim

    # Create figure
    fig = plt.figure(figsize=(figwidth, figheight))

    # Add space for colorbar at bottom and coordinate label at top
    gs = fig.add_gridspec(nrows, ncols, wspace=0.02, hspace=0.35,
                          left=0.01, right=0.99, top=0.95, bottom=0.06)

    # Get and display coordinate range
    ra_range, dec_range = get_coordinate_range(wcs_2d, ny, nx)
    coord_label = format_coord_range(ra_range, dec_range)
    fig.suptitle(coord_label, fontsize=10, y=0.99)

    # Plot each channel
    axes = []
    for i in range(n_channels):
        row = i // ncols
        col = i % ncols

        # Create subplot (simple axes, no WCS projection for cleaner look)
        ax = fig.add_subplot(gs[row, col])
        axes.append(ax)

        # Get channel data
        channel_data = data[i]

        # Apply log transform
        if log:
            plot_data = np.log10(np.where(channel_data > 0, channel_data, np.nan))
        else:
            plot_data = channel_data

        # Plot image
        im = ax.imshow(plot_data, origin='lower', cmap=colormap,
                       vmin=vmin, vmax=vmax, aspect='equal')

        # Add wavelength label as title
        wavelength = wavelengths[i]
        ax.set_title(f'{wavelength:.2f} \u03bcm', fontsize=fontsize_wavelength, pad=2)

        # Remove axes ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])

    # Fill remaining grid cells if needed
    for i in range(n_channels, nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')

    # Add colorbar at the bottom
    cbar_ax = fig.add_axes([0.25, 0.02, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    if layer_upper == 'IMAGE':
        if log:
            cbar.set_label(r'$\log_{10}$(MJy / sr)', fontsize=10)
        else:
            cbar.set_label('MJy / sr', fontsize=10)
    else:  # NHITS
        if log:
            cbar.set_label(r'$\log_{10}$(N hits)', fontsize=10)
        else:
            cbar.set_label('N hits', fontsize=10)

    # Save figure
    if output_path is None:
        suffix = f'_{layer.lower()}_summary.png'
        output_path = filepath.removesuffix(".fits.gz").removesuffix(".fits") + suffix

    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Saved summary plot to: {output_path}')

    if show:
        plt.show()

    return fig


def main():
    """Create summary plot of mosaic cube spectral channels."""
    parser = argparse.ArgumentParser(
        description='Create summary plot of mosaic cube spectral channels.')
    parser.add_argument('filepath', help='Path to mosaic_cube.fits file')
    parser.add_argument('-o', '--output', help='Output file path (default: <input>_<layer>_summary.png)')
    parser.add_argument('--layer', choices=['image', 'nhits'], default='image',
                        help='Data layer to plot (default: image)')
    parser.add_argument('--max-cols', type=int, default=17,
                        help='Maximum number of columns in the grid (default: 17)')
    parser.add_argument('--panel-size', type=float, default=1.0,
                        help='Size per panel in inches (default: 1.0)')
    parser.add_argument('--vmin', type=float, help='Minimum value for color scale')
    parser.add_argument('--vmax', type=float, help='Maximum value for color scale')
    parser.add_argument('--log', action='store_true', dest='log',
                        help='Use log color scale (default for image)')
    parser.add_argument('--linear', action='store_true', dest='linear',
                        help='Use linear color scale (default for nhits)')
    parser.add_argument('--cmap', default=None,
                        help='Matplotlib colormap (default: inferno for image, viridis for nhits)')
    parser.add_argument('--show', action='store_true',
                        help='Display plot interactively')

    args = parser.parse_args()

    vlim = None
    if args.vmin is not None or args.vmax is not None:
        vlim = (args.vmin, args.vmax)

    # Determine log scale: explicit flags override, otherwise None for auto
    if args.log:
        log = True
    elif args.linear:
        log = False
    else:
        log = None  # Use layer-specific default

    plot_mosaic_cube(
        args.filepath,
        output_path=args.output,
        max_cols=args.max_cols,
        figsize_per_panel=args.panel_size,
        log=log,
        layer=args.layer,
        vlim=vlim,
        cmap=args.cmap,
        show=args.show
    )


if __name__ == '__main__':
    main()
