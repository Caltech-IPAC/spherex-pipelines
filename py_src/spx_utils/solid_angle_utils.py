__all__ = ["calculate_solid_angle", "calculate_median_std_solid_angle"]

# Standard library imports

# Related third party imports
from astropy.wcs import WCS
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord

# Local application/library specific imports


def calculate_solid_angle(input_wcs: WCS, array_id: int, plot: bool) -> np.ndarray:
    """Calculate the solid angle per pixel.

    Parameters
    ----------
    input_wcs: `WCS`
        WCS object with the distortion SIP solution.
    array_id: `int`
        Detector ID.
    plot: `bool`
        Make plot of solid angle pixel map for visualization.

    Returns
    -------
    `np.ndarray`: Per-pixel solid angle map.
    """
    # xx, yy = np.meshgrid(np.arange(4, 2044) - 0.5, np.arange(4, 2044) - 0.5)
    xx, yy = np.meshgrid(np.arange(0, 2041), np.arange(0, 2041))
    ra, dec = input_wcs.all_pix2world(xx, yy, 0)
    ra[ra > 180.] -= 360.
    C = SkyCoord(ra, dec, frame='icrs', unit='deg')
    # Define pixel corners
    low_left = C[..., :-1, :-1]
    low_right = C[..., 1:, :-1]
    up_left = C[..., :-1, 1:]
    up_right = C[..., 1:, 1:]
    low = low_left.separation(low_right)
    left = low_left.separation(up_left)
    up = up_left.separation(up_right)
    right = low_right.separation(up_right)

    # Compute enclosed angles
    angle_low_right = low_right.position_angle(up_right) - low_right.position_angle(low_left)
    angle_up_left = up_left.position_angle(up_right) - low_left.position_angle(up_left)

    # Compute area assuming a planar triangle
    area_low_right = 0.5 * low * right * np.sin(angle_low_right.rad)
    area_up_left = 0.5 * up * left * np.sin(angle_up_left.rad)
    solid_angle = np.abs(area_low_right + area_up_left).to(u.arcsec**2)

    if plot:
        fig = plt.figure(figsize=(15, 7))
        # Pixel space plot
        ax1 = fig.add_subplot(121)
        cax1 = ax1.imshow(solid_angle.value, origin='lower', cmap='viridis')
        fig.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
        ax1.set_title(f'Solid Angle Map in Pixel Space: Detector {array_id}')
        ax1.set_xlabel('Pixel X')
        ax1.set_ylabel('Pixel Y')
        # RA/Dec space plot
        ax2 = fig.add_subplot(122, projection=input_wcs)
        cax2 = ax2.imshow(solid_angle.value, origin='lower', cmap='viridis',
                          extent=(0, solid_angle.shape[1], 0, solid_angle.shape[0]))
        ax2.coords[0].set_axislabel('Right Ascension')
        ax2.coords[1].set_axislabel('Declination')
        ax2.set_title(f'Solid Angle Map in RA/Dec Space: Detector {array_id}')
        fig.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
        # Overlay grid
        ax2.coords.grid(True, color='white', ls='solid', alpha=0.5)
        plt.tight_layout()
        plt.show()

    return solid_angle.value


def calculate_median_std_solid_angle(solid_angle_map: np.ndarray) -> tuple[float, float]:
    """Calculate the median and standard deviation of the solid angle pixel map.

    Parameters
    ----------
    solid_angle_map: `np.ndarray`
        Solid angle pixel map, as calculated by the `calculate_solid_angle` function.

    Returns
    -------
    `tuple[float, float]`
        The median and standard deviation of the solid angle map.
    """
    # Flatten the solid angle map to apply statistics on all pixels
    solid_angle_flat = solid_angle_map.flatten()

    # Calculate the median
    median_solid_angle = np.median(solid_angle_flat)

    # Calculate the standard deviation
    std_solid_angle = np.std(solid_angle_flat)

    return median_solid_angle, std_solid_angle
