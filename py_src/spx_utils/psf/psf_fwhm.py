__all__ = ["process_psf_zone_data", "calculate_fwhm", "plot_fwhm_histogram"]

# Standard library imports
import os
import logging

# Related third party imports
import matplotlib.pyplot as plt
import numpy as np
from photutils.profiles import RadialProfile

# Local application/library specific imports
from spx_data import PSFZoneData
from spx_data.spx_param import get_pix_scale


def process_psf_zone_data(psf_zone_data: PSFZoneData,
                          output_dir: str,
                          obsid: str = "",
                          pixsize: float = 6.15):
    """Process and visualizes PSFZoneData for each detector.

    Parameters
    ----------
    psf_zone_data : `PSFZoneData`
        PSF zone data object containing PSF information.
    output_dir : `str`
        Directory to save output figures.
    obsid : `str`, optional
        Observation ID for labeling. Default is "".
    pixsize : `float`, optional
        Size of the pixel in arcseconds for scaling. Default is 6.15.

    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data and headers
    psfs = psf_zone_data.data
    pixscale = psf_zone_data._get_header()["CDELT1"]  # arcsec/px

    # Create the figure and subplots
    fig = plt.figure(figsize=(15, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    axs = [fig.add_subplot(11, 11, ii + 1) for ii in range(psfs.shape[0])]

    # Iterate over PSFs in the cube
    for ii in range(psfs.shape[0]):
        # Display the PSF
        axs[ii].imshow(psfs[ii, :, :], origin="lower",
                       norm="linear", cmap="pink", interpolation="None")

        # Add zone index as text
        axs[ii].text(0.1, 0.9, ii, va="top", ha="left",
                     transform=axs[ii].transAxes, fontsize=9, color="white")

        # Overlay the pixel size rectangle
        xx = np.asarray([-0.5, 0.5, 0.5, -0.5]) * pixsize / pixscale + psfs.shape[1] // 2
        xx = np.concatenate((xx, [xx[0]]))
        yy = np.asarray([-0.5, -0.5, 0.5, 0.5]) * pixsize / pixscale + psfs.shape[2] // 2
        yy = np.concatenate((yy, [yy[0]]))
        axs[ii].plot(xx, yy, "-", linewidth=0.5, color="white")

        # Configure subplot appearance
        axs[ii].set_xticklabels([])
        axs[ii].set_yticklabels([])
        axs[ii].tick_params(axis="both", which="both", color="white")
        axs[ii].set_xlim(psfs[ii, :, :].shape[0] // 2 - 21, psfs[ii, :, :].shape[0] // 2 + 21)
        axs[ii].set_ylim(psfs[ii, :, :].shape[1] // 2 - 21, psfs[ii, :, :].shape[1] // 2 + 21)

    plt.suptitle(f"{psf_zone_data.x_zones*psf_zone_data.y_zones} zones", y=0.9)

    # Save the figure
    outname = f"{psf_zone_data.x_zones*psf_zone_data.y_zones}_zones_{obsid}.pdf"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()


def calculate_fwhm(psf_zone_data: PSFZoneData, pixscale: float = None) -> dict:
    """Calculate the FWHM for a given PSFZoneData object.

    Parameters
    ----------
    psf_zone_data : `PSFZoneData`
        The PSFZoneData object containing the PSF grid data.
    pixscale : `float`, optional
        Pixel scale in arcseconds per pixel. If None, use default pixel scale.

    Returns
    -------
    `dict`
        A dictionary containing the FWHMs in arcseconds
        and in pixels for the PSF grid zones:
        "fwhms_arcsec": list[float],
        "fwhms_pix": list[float]

    """
    if pixscale is None:
        pixscale = get_pix_scale()

    psfs = psf_zone_data.data  # PSF grid data
    fwhms_arcsec = []
    fwhms_pix = []

    for ii in range(psfs.shape[0]):
        img = psfs[ii, :, :]  # Extract the PSF image for this zone

        # Check if the PSF image has too many NaN pixels
        if np.isnan(img).sum() > (img.size * 0.5):
            logging.info(f"Failed to measure FWHM for zone {ii}, "
                         f"{np.isnan(img).sum() / img.size * 100:.2f}% NaN pixels.")
            fwhms_arcsec.append(np.nan)
            fwhms_pix.append(np.nan)
            continue

        # Guard against degenerate / non-finite PSFs (prevents gaussian_fit w/ NaNs)
        if (not np.isfinite(img).any()) or (np.nanmax(img) <= 0) or (np.nanstd(img) == 0):
            logging.info(f"Failed to measure FWHM for zone {ii}: degenerate PSF image.")
            fwhms_arcsec.append(np.nan)
            fwhms_pix.append(np.nan)
            continue

        # Assume aperture center is the center of the image
        aper_center = [img.shape[0] // 2, img.shape[1] // 2]

        # Define radii for curve of growth
        rs = np.arange(0.1, 30, 1)
        # Compute radial profile
        rp = RadialProfile(img, aper_center, rs, mask=None)

        # Calculate FWHM
        fwhms_arcsec.append(rp.gaussian_fwhm * pixscale)
        fwhms_pix.append(rp.gaussian_fwhm)

    return {"fwhms_arcsec": fwhms_arcsec, "fwhms_pix": fwhms_pix}


def plot_fwhm_histogram(fwhms_arcsec1: list, detector_id: int, output_dir: str, obsid: str = ""):
    """Plot a histogram of FWHM values for a single detector.

    Parameters
    ----------
    fwhms_arcsec1 : `list`
        List of FWHM values in arcseconds for the given detector.
    detector_id : `int`
        ID of the detector (e.g., 1, 2, 3, ...).
    output_dir : `str`
        Directory to save the output plot.
    obsid : `str`, optional
        Observation ID for labeling. Default is "".

    """
    # Histogram settings
    bins = np.arange(0, 10, 0.5)
    xlims = [-0.5, 8.5]
    ylims = [0, 55]
    yspan = ylims[1] - ylims[0]
    color = "blue"

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 5))

    # Plot histogram
    ax.hist(fwhms_arcsec1, bins=bins, facecolor="none", edgecolor=color, hatch="/////", label="Histogram")

    # Calculate median and errors
    tmp = np.nanpercentile(fwhms_arcsec1, q=(16, 50, 84))
    Xerr = [[np.abs(tmp[1] - tmp[0])], [np.abs(tmp[2] - tmp[1])]]

    # Plot median
    ax.errorbar(tmp[1], ylims[0] + yspan * 0.2, xerr=Xerr, fmt="o", markersize=8,
                markerfacecolor="white", elinewidth=1, linewidth=1, capsize=4,
                markeredgecolor="black", ecolor="black", zorder=1000, label="Median")
    ax.text(0.07, 0.82, f"Median: {tmp[1]:.1f}''", va="top", ha="left", fontsize=8, color="black",
            transform=ax.transAxes)

    # Add requirement line
    ax.axvline(7, linestyle="--", linewidth=1, color="gray")
    ax.text(7.3, ylims[0] + yspan * 0.4, "Requirement (7'')", color="gray", rotation=90,
            fontsize=9, va="center", ha="left")

    # Labels, ticks, and limits
    ax.set_xlabel("FWHM ['']", labelpad=1)
    ax.set_ylabel("Number of PSFs", labelpad=1)
    ax.set_xticks(np.arange(0, 9, 1))
    ax.set_yticks(np.arange(5, 55, 10))
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # Add legend
    ax.legend(loc="best", fontsize=8, framealpha=1)

    # Add detector label
    ax.text(0.07, 0.92, f"Detector {detector_id}", va="top", ha="left", fontsize=12, color="black",
            transform=ax.transAxes)

    # Save and show the plot
    outname = f"FWHM_Detector_{detector_id}_{obsid}.pdf"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, bbox_inches="tight")
    plt.show()
