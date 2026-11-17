__all__ = ["EstimatePSFConnections", "EstimatePSFConfig", "EstimatePSFTask"]

# Standard library imports
import multiprocessing
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Related third party imports
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import sigma_clipped_stats
from astropy.nddata import Cutout2D
from astropy.wcs import Sip
from joblib import Parallel, delayed
from lsst.pex.config import Field, DictField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from scipy import ndimage
from skimage import restoration
from tqdm import tqdm
from photutils.profiles import RadialProfile

# Local application/library specific imports
from spx_data import PSFZoneData
from spx_tasks import add_psf_history
from spx_utils.utils import add_detcoord
from spx_utils import ImageCatalogIterable
from spx_utils import SourcePipelineClass

# ==============================================================================
# Helper Functions (Adapted from Andreas Faisst's starstacking.ipynb)
# ==============================================================================


def _measure_fwhm(data, oversample=10):
    """
    Measures the FWHM of a PSF image using a radial profile.
    This is a more robust method than 2D Gaussian fitting.
    """
    try:
        # Find the center of the PSF using the brightest pixel
        y_cen, x_cen = np.unravel_index(np.argmax(data), data.shape)

        # Create a radial profile object
        # We define radii from the center out to the edge of the image.
        max_radius = min(x_cen, y_cen, data.shape[0] - y_cen, data.shape[1] - x_cen)
        radii = np.arange(1, int(max_radius))
        rp = RadialProfile(data, (x_cen, y_cen), radii)

        # Find the Half-Width at Half-Maximum (HWHM)
        # Find the value of the profile at its peak (the first point)
        profile_max = rp.profile[0]
        half_max = profile_max / 2.0

        # Use interpolation to find the precise radius
        # where the profile crosses the half-max value
        hwhm_pix_oversampled = np.interp(half_max, rp.profile[::-1], rp.radius[::-1])

        # FWHM is 2 * HWHM
        fwhm_pix_oversampled = 2 * hwhm_pix_oversampled

        # Convert to arcseconds
        SPHEREX_PIXEL_SCALE_ARCSEC = 6.15
        fwhm_arcsec = (fwhm_pix_oversampled / oversample) * SPHEREX_PIXEL_SCALE_ARCSEC
        return fwhm_arcsec

    except Exception as e:
        # If any part of this fails, return nan
        print(f" FWHM calculation failed: {e}")
        return np.nan


def _psf_grid(factor_up, stacksize):
    """Creates the pixel grid kernel for deconvolution.

    """
    sourceSize = factor_up * 2 - 1
    xx, yy = np.meshgrid(np.arange(sourceSize), np.arange(sourceSize))
    xx = abs(xx - (sourceSize / 2 - 0.5))
    yy = abs(yy - (sourceSize / 2 - 0.5))

    psf_pix = (factor_up - xx) * (factor_up - yy)
    psf_pix[(xx >= (factor_up + 0.5))] = 0
    psf_pix[(yy >= (factor_up + 0.5))] = 0
    psf_pix = psf_pix / np.sum(psf_pix)

    pad_val = stacksize - psf_pix.shape[0]
    psf_pix_padded = np.pad(psf_pix, ((int(np.floor(pad_val / 2)), int(np.ceil(pad_val / 2))),
                                      (int(np.floor(pad_val / 2)), int(np.ceil(pad_val / 2)))),
                            'constant', constant_values=0)
    return psf_pix_padded


def _create_upsampled_cutout(img, x_pos, y_pos, upsample, cutout_size):
    """Creates a cutout, upsamples it, and centers the star at the sub-pixel level.
    """

    # Create cutout at the nearest integer pixel.
    x_cut = round(x_pos)
    y_cut = round(y_pos)
    try:
        cutout = Cutout2D(img, position=(x_cut, y_cut), size=(cutout_size, cutout_size), mode="strict").data
    except Exception:
        return None

    # Background subtract.
    _, median, _ = sigma_clipped_stats(cutout, sigma=3)
    cutout = cutout - median

    # Calculate the "true" sub-pixel position in the small cutout.
    center_cutout_y = (cutout.shape[0] - 1) / 2.0
    center_cutout_x = (cutout.shape[1] - 1) / 2.0
    dx = x_pos - x_cut
    dy = y_pos - y_cut
    x_true_cutout = center_cutout_x + dx
    y_true_cutout = center_cutout_y + dy

    # Upsample using nearest-neighbor interpolation.
    cutout_up = ndimage.zoom(cutout, zoom=upsample, order=0, mode="constant", cval=0)

    # Pad the upsampled image to get an odd size (e.g., 110x110 -> 111x111).
    if cutout_up.shape[0] % 2 == 0 or cutout_up.shape[1] % 2 == 0:
        cutout_up = np.pad(cutout_up, pad_width=((0, 1), (0, 1)), mode="constant", constant_values=0)

    # Find the star's new position in the upsampled grid.
    center_cutout_up_y = (cutout_up.shape[0] - 1) / 2.0
    center_cutout_up_x = (cutout_up.shape[1] - 1) / 2.0
    y_true_cutout_up = y_true_cutout * upsample + 0.5 * upsample
    x_true_cutout_up = x_true_cutout * upsample + 0.5 * upsample

    # Calculate the final shift and apply it.
    dy_shift = center_cutout_up_y - y_true_cutout_up
    dx_shift = center_cutout_up_x - x_true_cutout_up
    cutout_up_shift = ndimage.shift(cutout_up, shift=(dy_shift, dx_shift), order=0)

    # Normalize and return.
    if np.nansum(cutout_up_shift) > 0 and not np.isnan(cutout_up_shift).any():
        return cutout_up_shift / np.nansum(cutout_up_shift)
    else:
        return None

# ==============================================================================
# Pipeline Task Definition
# ==============================================================================


class EstimatePSFConnections(pipeBase.PipelineTaskConnections,
                             dimensions={"instrument", "detector"},
                             defaultTemplates={}):
    """Connections definition for EstimatePSFTask."""
    imageHandles = cT.Input(
        name="intype",
        doc="Input calibrated exposures.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "exposure", "detector"],
        deferLoad=True,
        multiple=True
    )
    catalogHandles = cT.Input(
        name="srccat",
        doc="Input source catalogs.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
        deferLoad=True,
        multiple=True
    )
    psf = cT.Output(
        name='average_psf',
        doc="Output final estimated PSF (deconvolved).",
        storageClass="PSF",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    psf_stack = cT.Output(
        name='psf_stack',
        doc="Output median-stacked, super-resolved PSF before deconvolution.",
        storageClass="PSF",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    psf_rld = cT.Output(
        name='psf_rld',
        doc="Output Richardson-Lucy Deconvolved PSF.",
        storageClass="PSF",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    fwhm_dist = cT.Output(
        name='fwhm_dist',
        doc="Distribution of FWHM values for each PSF zone.",
        storageClass="NumpyArray",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )


class EstimatePSFConfig(pipeBase.PipelineTaskConfig,
                        pipelineConnections=EstimatePSFConnections):
    """Configuration parameters for EstimatePSFTask."""
    bins_x = Field(doc="Number of PSF zones along the x-axis.",
                   dtype=int,
                   default=11)

    bins_y = Field(doc="Number of PSF zones along the y-axis.",
                   dtype=int,
                   default=11)

    cutout_size = Field(doc="Size of the postage stamp cutout for each star (pixels). Should be odd.",
                        dtype=int,
                        default=11)

    oversample = Field(doc="Oversampling factor for super-resolution.",
                       dtype=int,
                       default=10)

    # Initial values chosen from 07/22 min loss study
    rld_iterations = DictField(doc="Number of Richardson-Lucy Deconvolution iterations.",
                               keytype=int,
                               itemtype=int,
                               default={1: 7, 2: 6, 3: 7, 4: 7, 5: 6, 6: 7})

    plot_path = Field(doc="Path to save output plots.",
                      dtype=str,
                      default="psf-plots/")

    run_parallel = Field(doc="Run PSF estimation for bins in parallel.",
                         dtype=bool,
                         default=True)

    num_cores = Field(doc="Number of cores for parallel processing. 0 means use all available.",
                      dtype=int,
                      default=4)

    do_plots = Field(doc="If True, generate and save diagnostic plots.",
                     dtype=bool,
                     default=True)

    min_mag = Field(doc="Minimum magnitude for a star to be used for PSF estimation.",
                    dtype=float,
                    default=11.0)

    max_mag = Field(doc="Maximum magnitude for a star to be used for PSF estimation.",
                    dtype=float,
                    default=15.0)


def _process_psf_bin(ibin, config, imageHandles, catalogHandles, spherex_image_shape, num_rld_iterations):
    """
    Worker function to process all stars within a single PSF bin.
    """
    ny, nx = spherex_image_shape
    bin_width_x, bin_width_y = nx / config.bins_x, ny / config.bins_y
    # Rows-first ordering
    # ibin = iy * bins_x +ix
    iy, ix = np.unravel_index(ibin, (config.bins_y, config.bins_x))
    x_min, x_max = ix * bin_width_x, (ix + 1) * bin_width_x
    y_min, y_max = iy * bin_width_y, (iy + 1) * bin_width_y

    all_cutouts = []
    total_sources_found = 0

    image_catalog_pairs = ImageCatalogIterable(
        imageHandles,
        catalogHandles,
        srcPipeFlag=SourcePipelineClass.PSF,
        addPixelCoords=True,
        pixelCoordsWithDistortion=True,
        filterSourcesOutsideImage=True
    )

    for spximg, catalog in image_catalog_pairs:

        # The catalog is already filtered for PSF stars and has pixel coordinates.
        # We just need to select the ones in this specific bin.
        sources_in_bin = catalog[
            (catalog['ix'] >= x_min) & (catalog['ix'] < x_max)
            & (catalog['iy'] >= y_min) & (catalog['iy'] < y_max)
        ]

        total_sources_found += len(sources_in_bin)

        for star in sources_in_bin:
            cutout = _create_upsampled_cutout(
                img=spximg.data, x_pos=star['ix'], y_pos=star['iy'],
                upsample=config.oversample, cutout_size=config.cutout_size
            )
            if cutout is not None:
                all_cutouts.append(cutout)

    # DEBUG: Print a summary after processing all images for this bin.
    print(f" Bin {ibin}: Found {total_sources_found} total source(s) across all images. "
          f"Created {len(all_cutouts)} usable cutouts for stacking.")

    if not all_cutouts:
        # DEBUG: Explicitly state that the bin is being skipped.
        print(f" Bin {ibin}: No usable stars found. Skipping bin.")
        return ibin, None, None, np.nan, np.nan

    stacked_psf = np.nanmedian(np.array(all_cutouts), axis=0)
    fwhm_stack = _measure_fwhm(stacked_psf, config.oversample)

    final_size = stacked_psf.shape[0]

    # Perform Deconvolution
    pixel_kernel = _psf_grid(config.oversample, final_size)
    _, _, stddev = sigma_clipped_stats(stacked_psf, sigma=3.0)
    deconvolved_psf = restoration.richardson_lucy(
        stacked_psf, pixel_kernel, num_iter=num_rld_iterations,
        filter_epsilon=stddev * 1e-1, clip=True
    )
    deconvolved_psf /= np.nansum(deconvolved_psf)
    fwhm_decon = _measure_fwhm(deconvolved_psf, config.oversample)

    # DEBUG: Print the final result for the bin.
    print(f" Bin {ibin}: Stacking & Deconvolution complete. "
          f"Stack FWHM: {fwhm_stack:.3f}\" -> Deconvolved FWHM: {fwhm_decon:.3f}\"")

    return ibin, stacked_psf, deconvolved_psf, fwhm_stack, fwhm_decon


class EstimatePSFTask(pipeBase.PipelineTask):
    """Task to estimate a binned, exposure-averaged PSF."""
    _DefaultName = "estimatePSF"
    ConfigClass = EstimatePSFConfig
    ConnectionsClass = EstimatePSFConnections

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fitter = LevMarLSQFitter()

    @add_psf_history()
    def run(self, imageHandles, catalogHandles, sip: Sip = None) -> pipeBase.Struct:
        c = self.config
        self.log.info(f"Starting PSF estimation on {len(imageHandles)} "
                      f"exposures for {c.bins_x*c.bins_y} zones.")

        good_image_handles = []
        for handle in tqdm(imageHandles, desc="Pre-filtering Images"):
            # We need to defer load to check the header.
            img = handle.get()

            # FINAST check
            if img.header.get("FINAST", 0) != 0:
                continue

            # SAA check
            sps_flags = img.header.get("SPS_FLAGS", "")
            if "saa" in [f.strip().lower() for f in sps_flags.split(",")]:
                continue

            good_image_handles.append(handle)

        self.log.info(f"Found {len(good_image_handles)} good-quality images to process.")

        # Create the ImageCatalogIterable with the good handles.
        # It will handle pairing and on-the-fly catalog preprocessing.
        image_catalog_pairs = list(ImageCatalogIterable(
            good_image_handles,
            catalogHandles,
            srcPipeFlag=SourcePipelineClass.PSF,
            addPixelCoords=True,
            pixelCoordsWithDistortion=True,  # Uses SIP
            filterSourcesOutsideImage=True
        ))

        if not image_catalog_pairs:
            raise ValueError("No valid image-catalog pairs found after filtering.")

        spherex_image = imageHandles[0].get()
        image_shape = spherex_image.shape
        detector_id = spherex_image.getArrayId()

        try:
            num_rld_iterations = c.rld_iterations[detector_id]
        except KeyError:
            raise ValueError(f"Detector ID {detector_id} not found in rld_iterations config. "
                             f"Please define it. Available detectors are: {list(c.rld_iterations.keys())}")

        self.log.info(f"Using {num_rld_iterations} RLD iterations for Detector {detector_id}.")

        num_bins = c.bins_x * c.bins_y

        final_psf_size = (101, 101)
        psf_stacks_all = np.zeros((num_bins, final_psf_size[0], final_psf_size[1]))
        psf_decon_all = np.zeros((num_bins, final_psf_size[0], final_psf_size[1]))

        # Create arrays to store both sets of FWHM values.
        fwhms_stack_all = np.full(num_bins, np.nan)
        fwhms_decon_all = np.full(num_bins, np.nan)

        if c.run_parallel:
            num_cores = c.num_cores if c.num_cores > 0 else multiprocessing.cpu_count()
            self.log.info(f"Running in parallel on {num_cores} cores.")

            # Pass the lightweight HANDLES, not the loaded data, to the workers.
            results = Parallel(n_jobs=num_cores)(
                delayed(_process_psf_bin)(i, c, good_image_handles, catalogHandles,
                                          image_shape, num_rld_iterations)
                for i in tqdm(range(num_bins), desc="Processing PSF Bins")
            )
        else:
            results = [_process_psf_bin(i, c, good_image_handles, catalogHandles,
                                        image_shape, num_rld_iterations)
                       for i in tqdm(range(num_bins), desc="Processing PSF Bins")]

        for ibin, stack, decon, fwhm_stack, fwhm_decon in results:
            if stack is not None and decon is not None:
                # Crop the stacked PSF from 111x111 to 101x101
                stack_center = (stack.shape[0] // 2, stack.shape[1] // 2)
                psf_stacks_all[ibin] = Cutout2D(stack,
                                                position=stack_center,
                                                size=final_psf_size,
                                                mode='partial').data
                # Crop the deconvolved PSF from 111x111 to 101x101
                decon_center = (decon.shape[0] // 2, decon.shape[1] // 2)
                psf_decon_all[ibin] = Cutout2D(decon,
                                               position=decon_center,
                                               size=final_psf_size,
                                               mode='partial').data
                # Store the FWHM values
                fwhms_stack_all[ibin] = fwhm_stack
                fwhms_decon_all[ibin] = fwhm_decon

        x_centers, y_centers = self._get_bin_centers(image_shape, c.bins_x, c.bins_y)
        psf_final = PSFZoneData(data=psf_decon_all, x_centers=x_centers, y_centers=y_centers)
        psf_stack = PSFZoneData(data=psf_stacks_all, x_centers=x_centers, y_centers=y_centers)
        add_detcoord(psf_final, spherex_image)
        add_detcoord(psf_stack, spherex_image)

        if c.do_plots:
            self.log.info(f"Generating plots and saving to {c.plot_path}")
            os.makedirs(c.plot_path, exist_ok=True)
            self._generate_plots(psf_stack, psf_final, fwhms_stack_all, fwhms_decon_all,
                                 c.plot_path, c.bins_x, c.bins_y, detector_id)

        return pipeBase.Struct(
            psf=psf_final, psf_rld=psf_final, psf_stack=psf_stack, fwhm_dist=fwhms_decon_all
        )

    def _get_bin_centers(self, spherex_image_shape, bins_x, bins_y):
        ny, nx = spherex_image_shape
        x_centers = (np.arange(bins_x) + 0.5) * (nx / bins_x)
        y_centers = (np.arange(bins_y) + 0.5) * (ny / bins_y)
        return x_centers, y_centers

    def _generate_plots(self, psf_stack_data, psf_decon_data, fwhms_stack, fwhms_decon,
                        output_dir, bins_x, bins_y, detector):
        # Plot FWHM distribution
        plt.figure(figsize=(10, 8))
        plt.hist(fwhms_decon[~np.isnan(fwhms_decon)], bins=20, color='royalblue', edgecolor='black')
        plt.title("Distribution of Deconvolved PSF FWHM", fontsize=16)
        plt.xlabel("FWHM (arcseconds)", fontsize=12)
        plt.ylabel("Number of PSF Zones", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_dir, "fwhm_distribution.png"), bbox_inches='tight')
        plt.close()

        # Plot stacked PSFs
        stack_filename = f"psf_stack_binx-{bins_x}_biny-{bins_y}D{detector}.png"
        self._plot_psf(psf_stack_data.data, "Stacked PSF",
                       os.path.join(output_dir, stack_filename), bins_x, bins_y, detector,
                       fwhms=fwhms_stack)

        # Plot deconvolved PSFs
        decon_filename = f"psf_deconvolved_binx-{bins_x}_biny-{bins_y}D{detector}.png"
        self._plot_psf(psf_decon_data.data, "Deconvolved PSF",
                       os.path.join(output_dir, decon_filename), bins_x, bins_y, detector,
                       fwhms=fwhms_decon)

    def _plot_psf(self, data, title, filename, bins_x, bins_y, detector_id, fwhms=None):
        fig, axes = plt.subplots(bins_y, bins_x, figsize=(3 * bins_x, 3 * bins_y),
                                 squeeze=False)

        fig.suptitle(title, fontsize=20, y=0.98)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        valid_data = data[np.sum(data, axis=(1, 2)) > 0]
        if len(valid_data) == 0:
            self.log.warn(f"No valid data to plot for {title}")
            plt.close(fig)
            return

        vmin, vmax = np.min(valid_data), np.max(valid_data)

        im = None

        # Loop through the subplot grid by row and column
        for iy in range(bins_y):          # iy=0 is bottom row in detector coords
            for ix in range(bins_x):      # ix increases left->right
                i = iy * bins_x + ix      # x-fast then y
                row = (bins_y - 1) - iy   # map bottom->top for the figure
                col = ix

                ax = axes[row, col]

                if np.sum(data[i]) > 0:
                    im = ax.imshow(
                        data[i],
                        origin="lower",          # keep PSF image origin consistent
                        cmap="nipy_spectral",
                        norm=colors.Normalize(vmin=vmin, vmax=vmax),
                    )

                ax.set_title(f"(ix={ix}, iy={iy})  i={i}  D{detector_id}", fontsize=10)
                ax.set_xlabel("Pixels (oversampled)", fontsize=8)
                ax.set_ylabel("Pixels (oversampled)", fontsize=8)

                if fwhms is not None and np.isfinite(fwhms[i]):
                    ax.text(
                        0.05, 0.95, f'FWHM: {fwhms[i]:.2f}"',
                        transform=ax.transAxes,
                        fontsize=9, color="white",
                        va="top",
                        bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.5),
                    )

        if im is not None:
            fig.subplots_adjust(right=0.9)
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im, cax=cbar_ax)

        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
