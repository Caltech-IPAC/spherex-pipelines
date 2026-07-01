__all__ = ["plot_mult_stacks", "plot_decon_set"]

# Standard library imports
import os

# Related third party imports
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local application/library specific imports

halfStampSizeUp = 50
FLG_stackDeconIterateEnable = 1


def plot_mult_stacks(stack, stack5, stack50, stack100,
                     numSources, stack_deleteArray, plot_path, ibin, array_id):
    """Plot PSF stacks of varying number of sources: 5, 50, 500, all."""
    factor_up = 10
    xAxisTicks = np.arange(0, halfStampSizeUp + factor_up, factor_up)  # Sets x axis ticks
    yAxisTicks = xAxisTicks  # Same axis ticks

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    fig.set_tight_layout(True)
    divider = make_axes_locatable(ax[0, 0])  # Prep to add an axis
    cax = divider.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im = ax[0, 0].pcolormesh(stack5[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Scaled Flux [dimensionless]')
    ax[0, 0].axis('scaled')
    ax[0, 0].set_xlabel('Fine Grid Pixels')
    ax[0, 0].set_ylabel('Fine Grid Pixels')
    ax[0, 0].set_title('5 Sources', y=.85, color='white')
    ax[0, 0].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[0, 0].set_yticks(yAxisTicks)  # Set y axis ticks

    divider2 = make_axes_locatable(ax[0, 1])  # Prep to add an axis
    cax2 = divider2.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im2 = ax[0, 1].pcolormesh(stack50[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('Scaled Flux [dimensionless]')
    ax[0, 1].axis('scaled')
    ax[0, 1].set_xlabel('Fine Grid Pixels')
    ax[0, 1].set_ylabel('Fine Grid Pixels')
    ax[0, 1].set_title('50 Sources', y=.85, color='white')
    ax[0, 1].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[0, 1].set_yticks(yAxisTicks)  # Set y axis ticks

    divider3 = make_axes_locatable(ax[1, 0])  # Prep to add an axis
    cax3 = divider3.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im3 = ax[1, 0].pcolormesh(stack100[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
    cbar3.set_label('Scaled Flux [dimensionless]')
    ax[1, 0].axis('scaled')
    ax[1, 0].set_xlabel('Fine Grid Pixels')
    ax[1, 0].set_ylabel('Fine Grid Pixels')
    ax[1, 0].set_title('100 Sources', y=.85, color='white')
    ax[1, 0].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[1, 0].set_yticks(yAxisTicks)  # Set y axis ticks

    divider4 = make_axes_locatable(ax[1, 1])  # Prep to add an axis
    cax4 = divider4.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im4 = ax[1, 1].pcolormesh(stack[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))):halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar4 = fig.colorbar(im4, cax=cax4, orientation='vertical')
    cbar4.set_label('Scaled Flux [dimensionless]')
    ax[1, 1].axis('scaled')
    ax[1, 1].set_xlabel('Fine Grid Pixels')
    ax[1, 1].set_ylabel('Fine Grid Pixels')
    ax[1, 1].set_title(
        "{:,.0f}".format(numSources - np.sum(stack_deleteArray)) + ' Catalog Sources', y=.85,
        color='white')
    ax[1, 1].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[1, 1].set_yticks(yAxisTicks)  # Set y axis ticks
    plt.show()
    plt.savefig(f"{plot_path}/{array_id}/5-50-200-All-Sources_exposure_band{array_id}_bin{ibin}.png")


def plot_decon_set(stack, Deconvolved_Stack, factor_up, psf_pix_padded,
                   rld_decon, plot_path, ibin, array_id,
                   fwhm_arcsec: float = 0., fwhm_rld_arcsec: float = 0., obsid: str = ""):
    """Plot PSF stack, PSF RLD, PSF IBP, and PSF True."""
    print(f"INSIDE PLOT_DECON_SET np.sum(stack) = {np.sum(stack)}")
    xAxisTicks = np.arange(0, halfStampSizeUp + factor_up, factor_up)
    yAxisTicks = xAxisTicks  # Same axis ticks

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
    # fig.set_tight_layout(True)
    divider = make_axes_locatable(ax[0, 0])  # Prep to add an axis
    cax = divider.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im = ax[0, 0].pcolormesh(stack[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    # im = ax[0, 0].pcolormesh(stack, cmap='nipy_spectral', antialiased=True)
    cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    cbar.set_label('Scaled Flux [dimensionless]')
    ax[0, 0].axis('scaled')
    ax[0, 0].set_xlabel('Fine Grid Pixels')
    ax[0, 0].set_ylabel('Fine Grid Pixels')
    ax[0, 0].set_title('$P_{S}$', y=.85, color='white')
    ax[0, 0].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[0, 0].set_yticks(yAxisTicks)  # Set y axis ticks

    divider2 = make_axes_locatable(ax[0, 1])  # Prep to add an axis
    cax2 = divider2.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im2 = ax[0, 1].pcolormesh(psf_pix_padded[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    # Create a colorbar using the prev. defined cax
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('Scaled Flux [dimensionless]')
    ax[0, 1].axis('scaled')
    ax[0, 1].set_xlabel('Fine Grid Pixels')
    ax[0, 1].set_ylabel('Fine Grid Pixels')
    ax[0, 1].set_title('$P_{grid}$', y=.85, color='white')
    ax[0, 1].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[0, 1].set_yticks(yAxisTicks)  # Set y axis ticks

    divider3 = make_axes_locatable(ax[1, 0])  # Prep to add an axis
    cax2 = divider3.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im2 = ax[1, 0].pcolormesh(rld_decon[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation='vertical')
    cbar2.set_label('Scaled Flux [dimensionless]')
    ax[1, 0].axis('scaled')
    ax[1, 0].set_xlabel('Fine Grid Pixels')
    ax[1, 0].set_ylabel('Fine Grid Pixels')
    ax[1, 0].set_title('$P_{RLD}$', y=.85, color='white')
    ax[1, 0].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[1, 0].set_yticks(yAxisTicks)  # Set y axis ticks

    divider4 = make_axes_locatable(ax[1, 1])  # Prep to add an axis
    cax3 = divider4.append_axes('right', size='2.0%', pad=0.35)  # Make a color bar axis
    im3 = ax[1, 1].pcolormesh(Deconvolved_Stack[halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2))), halfStampSizeUp - np.int64(
        np.floor(np.int64(halfStampSizeUp / 2))): halfStampSizeUp + np.int64(
        np.ceil(np.int64(halfStampSizeUp / 2)))], cmap='nipy_spectral', antialiased=True)
    cbar3 = fig.colorbar(im3, cax=cax3, orientation='vertical')
    cbar3.set_label('Scaled Flux [dimensionless]')
    ax[1, 1].axis('scaled')
    ax[1, 1].set_xlabel('Fine Grid Pixels')
    ax[1, 1].set_ylabel('Fine Grid Pixels')
    if (FLG_stackDeconIterateEnable == 0):
        ax[1, 1].set_title('$P_{RLD}$', y=.85, color='white')
    else:
        ax[1, 1].set_title('$P_{IBP}$', y=.85, color='white')
    # Add FWHM annotation (bottom-left panel)
    fwhm_rld_arcsec = float(fwhm_rld_arcsec)
    ax[1, 0].text(0.05, 0.95,
                  f"FWHM = {fwhm_rld_arcsec:.2f}\"",
                  transform=ax[1, 0].transAxes,
                  fontsize=10,
                  color="white",
                  ha="left",
                  va="top",
                  bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3', alpha=0.5))
    # Add FWHM annotation (bottom-right panel)
    fwhm_arcsec = float(fwhm_arcsec)
    ax[1, 1].text(0.05, 0.95,
                  f"FWHM = {fwhm_arcsec:.2f}\"",
                  transform=ax[1, 1].transAxes,
                  fontsize=10,
                  color="white",
                  ha="left",
                  va="top",
                  bbox=dict(facecolor='black', edgecolor='white', boxstyle='round,pad=0.3', alpha=0.5))
    ax[1, 1].set_xticks(xAxisTicks)  # Set x axis ticks
    ax[1, 1].set_yticks(yAxisTicks)  # Set y axis ticks
    plt.show()
    plt.savefig(f"{plot_path}/{array_id}/P_s-P_grid-P_rld-P_ibp_exposure_band{array_id}_bin{ibin}.png")


def plot_zones(spximg, binedge_x, binedge_y, plot_path: str = None):
    """Plot PSF zones with sources."""
    plt.clf()
    plt.imshow(spximg, cmap='nipy_spectral', vmax=2040, origin="lower")
    for xc in binedge_x:
        plt.axvline(x=xc, color='r')
    for yc in binedge_y:
        plt.axhline(y=yc, color='r')
    plt.title(('PSF Zone Bins'))
    plt.xlim(0, 2040)
    plt.ylim(0, 2040)
    plt.show()
    if plot_path is not None:
        plt.savefig('psf-plots/stack_bin_boundaries.png')


def plot_psf(psf: np.ndarray, output_dir: str, obsid: str):
    """Plot grid of solved PSFs."""
    N = psf.shape[0]
    grid_size = math.ceil(math.sqrt(N))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(40, 40))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    for i in range(N):
        ax = axes[i]
        im = ax.imshow(psf[i], cmap='viridis', origin='lower')
        ax.set_xlabel('pix x')
        ax.set_ylabel('pix y')
        fig.colorbar(im, ax=ax, orientation='vertical')

    # Hide any unused subplots
    for j in range(N, grid_size * grid_size):
        fig.delaxes(axes[j])

    outname = f"PSF_{obsid}.pdf"
    output_path = os.path.join(output_dir, outname)
    plt.savefig(output_path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()
