__all__ = ["EstimateAbsGainMatrixConfig", "EstimateAbsGainMatrixConfig",
           "EstimateAbsGainMatrixTask"]

# Standard library imports
import copy
import logging
import os
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

# Related third party imports
from astropy import log
from astropy.table import Table
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pex.config import Field, ListField, ConfigurableField, Config
from lsst.daf.butler import DeferredDatasetHandle
from astropy.stats import sigma_clipped_stats
from astropy.coordinates import SkyCoord
from sklearn.cluster import DBSCAN
from astropy.modeling import models, fitting
from astropy.stats import sigma_clip

# Local application/library specific imports
from spx_data import SPHERExImage, get_wavelength_mesh, Flag
from spx_tasks.filter_by_mask import FilterByMaskTask
from spx_utils.solar_elongation import compute_tilt_function, tilt_correct
# from spx_data.spx_param import get_pix_scale
from spx_tasks import add_history
from ._provenance import add_calibration_provenance
from spx_utils.utils import add_detcoord
# ==============================================================================
# Fitting Tools


@dataclass
class ArrayCalValues:
    cal_values: dict = field(default_factory=lambda: {
        1: 8.286700087641215,
        2: 8.38949569816677,
        3: 7.896646949035173,
        4: 9.867400104358904,
        5: 2.0516901757300006,
        6: 1.682176698588182
    })


def poly_22(xy, p00, p01, p02, p10, p11, p12, p20, p21, p22):
    x, y = xy
    output = (p00 + p01 * y**1 + p02 * y**2
              + p10 * x**1 + p11 * x**1 * y**1 + p12 * x**1 * y**2
              + p20 * x**2 + p21 * x**2 * y**1 + p22 * x**2 * y**2)
    return output


def fit_curve(x_data, y_data, z_data, fit_func):
    good_msk = ~np.isnan(z_data) & ~np.isinf(z_data)
    x_data = x_data[good_msk]
    y_data = y_data[good_msk]
    z_data = z_data[good_msk]

    xy_data = np.vstack((x_data, y_data))
    popt, pcov = curve_fit(fit_func, xy_data, z_data)
    return popt, pcov


def query_fit_data(x_data, y_data, z_data, fit_func=poly_22):
    popt, pcov = fit_curve(x_data.copy(), y_data.copy(), z_data.copy(), fit_func)
    xy_data = np.vstack((x_data, y_data))
    fit = fit_func(xy_data, *popt)
    return fit


# ==============================================================================
# Flat-Field Algorithm

class RegionMaskIterable:
    """Iterable that yields (region_id, mask) pairs on each pass."""

    def __init__(self,
                 array_id: int,
                 npix_x: int,
                 npix_y: int,
                 num_regions: int,
                 region_list: Optional[list[int]] = None):
        # Heavy work done **once**
        self.lambda_mesh = get_wavelength_mesh(array_id, npix_x, npix_y)
        lmin, lmax = np.min(self.lambda_mesh), np.max(self.lambda_mesh)
        self.edges = np.linspace(lmin, lmax, num_regions + 1)

        self.num_regions = num_regions
        self.region_list = set(region_list) if region_list else None

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """
        Lazily generate (region_id, mask) pairs that partition an image into
        wavelength intervals.

        Return a fresh generator each time the iterable is looped over.

        Yields
        ------
        (region_id, mask) : tuple[int, np.ndarray[bool, 2-D]]
        * `region_id` – integer index of the wavelength bin
        * `mask`      – boolean array, True where the pixel belongs to that bin
        """
        mesh = self.lambda_mesh
        edges = self.edges
        rlist = self.region_list
        for region_id in range(self.num_regions):
            if rlist is not None and region_id not in rlist:
                continue
            mask = (mesh >= edges[region_id]) & (mesh < edges[region_id + 1])
            yield region_id, mask


class ImageFiltering:
    @staticmethod
    def filter_galactic_lat(masked_images: list,
                            latitudes: list,
                            longitudes: list,
                            obstimes: list,
                            wcss: list,
                            config: Config,
                            verbose: bool) -> tuple:
        """Filter galactic latitude to avoid regions of excessive DGL.

        Parameters
        ----------
        masked_images : list[np.ma.MaskedArray]
            Input list of SPHEREx images.
        latitudes : list[np.ndarray]
            List of latitude arrays corresponding to the images.
        longitudes : list[np.ndarray]
            List of longitude arrays corresponding to the images.
        obstimes : list[np.ndarray]
            List of observation times corresponding to the images.
        wcss : list[np.ndarray]
            List of wcs corresponding to the images.
        config : `Config`
            Configuration for estimate flat field module.
        verbose : `bool`
            Condition to print out processing details.

        Returns
        -------
        filtered_dataset : list[np.ma.MaskedArray]
            Filtered list of SPHEREx images with galactic latitude threshold imposed.
        filtered_latitudes : list[np.ndarray]
            Corresponding latitudes for filtered images.
        filtered_longitudes : list[np.ndarray]
            Corresponding longitudes for filtered images.
        filtered_obstimes : list[np.ndarray]
            Corresponding observation times for filtered images.
        """
        filtered_dataset = []
        filtered_latitudes = []
        filtered_longitudes = []
        filtered_obstimes = []
        filtered_wcs = []

        with alive_bar(len(masked_images), title='Filtering crowded images: ') as bar:
            num_removed_galactic = 0
            for spximg, lat, lon, t, w in zip(masked_images, latitudes, longitudes, obstimes, wcss):
                array_id = spximg.getArrayId()
                # Exclude images in the galactic plane.
                crpix = spximg.wcs.wcs.crpix
                coord = spximg.wcs.pixel_to_world(crpix[0], crpix[1])
                b = coord.transform_to('galactic').b.value
                if abs(b) <= config.galactic_lat_threshold[array_id-1]:
                    num_removed_galactic += 1
                else:
                    filtered_dataset.append(spximg.data)  # only keep masked arrays
                    filtered_latitudes.append(lat)
                    filtered_longitudes.append(lon)
                    filtered_obstimes.append(t)
                    filtered_wcs.append(w)
                bar()
        if verbose:
            print(f"Removed {num_removed_galactic} because over galactic lat threshold.\n"
                  f"Remaining images = {len(masked_images) - num_removed_galactic}.")

        return filtered_dataset, filtered_latitudes, filtered_longitudes, filtered_obstimes, filtered_wcs


def define_spectral_bins(spximg: SPHERExImage, config: Config, verbose: bool = False) -> (
        tuple)[Iterable[tuple[int, np.ndarray]], Iterable[tuple[int, np.ndarray]]]:
    """Define spectral bins based on LVF information.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`
        Input SPHEREx image.
    config : `Config`
        Configuration for estimate flat field module.
    verbose :`bool`
        Condition to printout processing details.

    Returns
    -------
    result :
        Region mask dictionaries for nominal and fine resolution.
    Notes
    -----
    Get information from one of the images and define the region dictionary
    corresponding to the 64 / 256 wavelength bins.
    """
    region_msk_iter = RegionMaskIterable(spximg.getArrayId(),
                                         spximg.data.shape[0],
                                         spximg.data.shape[1],
                                         config.num_regions,  # default is 64.
                                         region_list=config.region_list)
    if verbose is True:
        print('region_msk_iter created.')

    # Finer dictionary of wavelengths split into 256 regions
    region_msk_iter_fine = RegionMaskIterable(spximg.getArrayId(),
                                              spximg.data.shape[0],
                                              spximg.data.shape[1],
                                              config.num_regions_fine,  # 256.
                                              region_list=config.region_list_fine)
    if verbose is True:
        print('region_msk_iter_fine created.')

    return region_msk_iter, region_msk_iter_fine


def stack_and_normalize_images_batch(filtered_dataset: list, batch_size: int = 10) -> np.ndarray:
    """Stack and normalize images in batches to reduce memory usage.

    Parameters
    ----------
    filtered_dataset : list[np.ma.MaskedArray]
        Input list of masked SPHEREx images.
    batch_size : int
        Number of images to stack per batch.

    Returns
    -------
    normalized_stack : np.ma.MaskedArray
        Mean-normalized image stack.
    """
    image_shape = filtered_dataset[0].shape
    stack_sum = np.zeros(image_shape, dtype=np.float32)
    stack_num = np.zeros(image_shape, dtype=np.float32)

    num_batches = int(np.ceil(len(filtered_dataset) / batch_size))

    for i in range(num_batches):
        minidx = i * batch_size
        maxidx = min((i + 1) * batch_size, len(filtered_dataset))

        stacked_images_sub = np.ma.stack(filtered_dataset[minidx:maxidx], axis=0)
        stack_sum += stacked_images_sub.sum(axis=0)
        stack_num += (~stacked_images_sub.mask).sum(axis=0)

    with np.errstate(divide='ignore', invalid='ignore'):
        stack_mean = np.divide(stack_sum, stack_num)
        stack_mean[~np.isfinite(stack_mean)] = np.nan

    image_mean = np.nanmean(stack_mean)
    normalized_stack = np.divide(stack_mean, image_mean)
    normalized_stack = np.ma.masked_invalid(normalized_stack)

    return normalized_stack


def stack_and_normalize_images(filtered_dataset: list) -> np.ndarray:
    """Stacking and normalizing of images.

    Parameters
    ----------
    filtered_dataset : `list`
        Input filtered set of masked arrays of SPHEREx data.

    Returns
    -------
    normalized_stack : ~numpy.ndarray
        Normalized SPHEREx stacked images.
    """
    image_shape = filtered_dataset[0].shape
    stack_sum = np.zeros(image_shape, dtype=np.float32)
    stack_num = np.zeros(image_shape, dtype=np.float32)

    # Stack all images
    stacked_images = np.ma.stack(filtered_dataset, axis=0)
    # Compute the sum, automatically ignoring masked elements
    stack_sum = stacked_images.sum(axis=0)
    # Compute the count of unmasked elements along the new axis
    stack_num = (~stacked_images.mask).sum(axis=0)

    # Use alive_bar for progress visualization
    with alive_bar(len(filtered_dataset), title="Stacking and normalizing images.") as bar:
        for _ in filtered_dataset:
            bar()

    # Get mean of each pixel, taking into account
    # how many non-masked pixels went into it
    logging.info("stack sum")
    stack_sum = np.ma.divide(stack_sum, stack_num)
    # stack_sum /= stack_num
    logging.info("mean_stack")
    mean_stack = np.nanmean(stack_sum)  # Mean over the full image.
    logging.info("normalized stack 1")
    normalized_stack = stack_sum / mean_stack  # Normalize by the mean.
    logging.info("normalized stack 2")
    normalized_stack = np.ma.divide(stack_sum, mean_stack)

    return normalized_stack


def compute_avg_flux_and_wavelength(region_msk_iter: Iterable[tuple[int, np.ndarray]],
                                    normalized_stack: np.ndarray,
                                    lambda_mesh: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the average flux and wavelength values in defined spectral bins.


    Parameters
    ----------
    region_msk_iter :
        Mask iterator for wavelength region.
    normalized_stack : ~numpy.ndarray
        Normalized and stacked SPHEREx image data.
    lambda_mesh : ~numpy.ndarray
        Two dimensional wavelength mesh with central wavelength at any position.

    Returns
    -------
    `tuple[~numpy.ndarray, ~numpy.ndarray]`
        Average fluxes and wavelengths.
    """
    flux_list = []
    wav_list = []

    for region_id, region_msk in region_msk_iter:
        if not np.any(region_msk):  # empty region
            print(f"[WARN] Region {region_id} has no valid pixels, skipping.")
            continue

        # Masked selection
        flux_region = normalized_stack[region_msk]
        lambda_region = lambda_mesh[region_msk]

        # Compute statistics directly
        flux_list.append(np.ma.median(flux_region))   # Start with median, may change to mean.
        wav_list.append(np.nanmean(lambda_region))    # Average wavelength corresponding to flux.

    return np.array(flux_list), np.array(wav_list)


def compute_fine_grid_spectral_response(region_msk_iter_fine: dict,
                                        wvs: np.ndarray,
                                        fluxes: np.ndarray,
                                        gf: np.ndarray,
                                        lambda_mesh: np.ndarray,
                                        apply_gf: bool = True) -> np.ndarray:
    """Compute the spectral response on a finer grid.

    Parameters
    ----------
    region_msk_iter_fine : `dict`
        Mask dictionary for fine resolution wavelength region.
    wvs : `~numpy.ndarray`
        List of wavelengths.
    fluxes : `~numpy.ndarray`
        List of fluxes.
    gf : `np.ndarray`
        Estimated 1D gain function from binned calibration data.
    lambda_mesh : ~numpy.ndarray
        Two dimensional wavelength mesh with central wavelength at any position.

    Returns
    -------
    spectral_response_fine : ~numpy.ndarray
        Smooth functional fit to response variation across stacked images.
    """
    # First, get the central wavelengths on the 256 wavelength region split.
    wvs_fine = []
    wvs = np.array(wvs)
    # NOTE dividing by gf is scaling fluxes in unexpected way.
    # 11/13 currently debuggig algorithm.
    if apply_gf is True:
        fluxes = fluxes / gf

    for region_id in region_msk_iter_fine:
        region_msk = region_msk_iter_fine[region_id]
        # Average wavelength in finely divided region, used for fitting 1D response.
        thiswave = np.mean(lambda_mesh[region_msk])
        wvs_fine.append(thiswave)

    # Ensure finite values in fluxes before interpolation
    valid_indices = np.isfinite(fluxes)
    if not np.all(valid_indices):
        # Optionally handle or warn if there are non-finite values
        print("Warning: Non-finite values detected in fluxes. Using only finite values for interpolation.")
        wvs = wvs[valid_indices]
        fluxes = fluxes[valid_indices]

    # Interpolate with finer grid wavelengths.
    spl = CubicSpline(wvs, fluxes)
    spectral_response_fine = spl(np.array(wvs_fine))
    # This should give a smooth functional fit to the response variation
    # across the stacked, normalized 2D image.
    return spectral_response_fine


def divide_out_spectral_response(filtered_dataset: list,
                                 region_msk_iter_fine: dict,
                                 spectral_response_fine: np.ndarray,
                                 use_linear_regression: bool) -> None:
    """Divide the derived spectral response out of images in filtered dataset.

    Parameters
    ----------
    filtered_dataset : `list`
        Input filtered and SPHEREx images.
    region_msk_iter : `dict`
        Dictionary for wavelength region masks.
    spectral_response_fine : ~numpy.ndarray
        Smooth spectral response across image.

    Returns
    -------
    `None`
        Modifies the `filtered_dataset` in-place.
    """
    # for image in filtered_dataset:
    #     cnt = 0
    #     for region_id in region_msk_iter_fine:
    #         region_msk = region_msk_iter_fine[region_id]
    #         image_divided = copy.deepcopy(image.get().data)
    #         image_divided[region_msk] /= spectral_response_fine[cnt]
    #         cnt += 1
    #         image.get().data /= np.mean(image_divided)  # Normalizing.
    # The following code is equivalent to above,
    # but does in-place modification, and is faster.
    for image_data in filtered_dataset:
        # Iterate over each region mask and corresponding spectral response directly
        for region_id, region_msk in region_msk_iter_fine:
            # Apply the spectral response correction directly to the masked regions
            correction_factor = spectral_response_fine[region_id]
            # Safely divide only where mask is False (i.e., unmasked)
            image_data[region_msk] = np.ma.divide(image_data[region_msk], correction_factor)

        # Normalize if not using slope fit method
        if use_linear_regression is False:
            # After applying all corrections, normalize the image data by its mean
            # Compute mean excluding masked elements, infs, and NaNs
            valid_mean = np.ma.mean(np.ma.masked_invalid(image_data))

            if valid_mean is not np.ma.masked and valid_mean != 0:
                # Normalize the image data using the valid mean
                image_data /= valid_mean
            else:
                # Handle case where no valid data exists
                raise ValueError("No valid data to compute the mean or mean is zero.")


def cut_images_with_excessive_signal_variation(filtered_dataset: list,
                                               sigma: int,
                                               percent: float,
                                               verbose: bool,
                                               return_indices: bool = False) -> tuple:
    """
    Cut images with excessive signal variation and optionally return surviving indices.

    Parameters
    ----------
    filtered_dataset : `list`
        Input filtered SPHEREx images (or masked arrays).
    sigma : `int`
        Sigma threshold.
    percent : `float`
        Fractional threshold for allowed outlier pixels.
    verbose : `bool`
        Print debug info.
    return_indices : `bool`, optional
        If True, also return the indices of kept images.

    Returns
    -------
    filtered_images : list
        Subset of input list with valid images.
    indices (optional) : list[int]
        Indices of kept images within input list.
    """
    filtered_images = []
    kept_indices = []

    for i, image in enumerate(filtered_dataset):
        if not _has_excessive_signal_variation(image, sigma, percent, verbose):
            filtered_images.append(image)
            kept_indices.append(i)

    print(f"Number of images cut with excessive noise: {len(filtered_dataset) - len(filtered_images)}")
    print(f"Number of images remaining = {len(filtered_images)}")

    if len(filtered_images) == 0:
        if verbose:
            print("cut all images !!!, resetting to filtered dataset without cutting images.")
        filtered_images = filtered_dataset
        kept_indices = list(range(len(filtered_dataset)))

    if return_indices:
        return filtered_images, kept_indices
    else:
        return filtered_images


def _has_excessive_signal_variation(image: SPHERExImage, sigma: int, percent: float, verbose: bool) -> bool:
    """Check if the image has more than 5% of pixels over 2-sigma from the mean.

    Parameters
    ----------
    image : `~spx_data.SPHERExImage`
        Input SPHEREx image.
    sigma : `int`
        Simga value at which to apply excessive signal check.
    percent : `float`
        Threshold of percentage of pixels with excessive noise.

    Returns
    -------
    `bool`
        True if execssive signal in image is found.
    """
    # `image' is a numpy masked array
    # img_mean = image.mean()
    # img_std = image.std()
    img_mean = np.ma.mean(image)
    img_std = np.ma.std(image)
    # Find pixels over the threshold
    over_threshold = image > (img_mean + sigma * img_std)
    # Count these pixels, ignoring the masked elements
    num_over_threshold = np.count_nonzero(over_threshold)
    # Calculate the fraction of pixels over the threshold
    # Using image.count() to get the number of non-masked elements
    frac = num_over_threshold / image.count()

    # Check if more than 5% of the pixels are over the threshold
    if verbose is True:
        if frac > 0.05:
            print("More than 5% of the pixels are over 2-sigma from the mean.")
        else:
            print("Less than 5% of the pixels are over 2-sigma from the mean.")

    return frac > percent  # Default is 5%.


def compute_ff_delta_streaming(filtered_images: list, chunk_size: int = 10) -> np.ndarray:
    """Compute ff_delta using a streaming approach to avoid large memory usage.

    Parameters
    ----------
    filtered_images : list[np.ma.MaskedArray]
        The list of spectrally flattened images.
    chunk_size : int
        Number of images per chunk to reduce memory load.

    Returns
    -------
    ff_delta : np.ndarray
        Mean-normalized flat-field delta image.
    """
    image_shape = filtered_images[0].shape
    stacked_mean = np.zeros(image_shape, dtype=np.float32)
    stacked_mean_num = 0

    num_chunks = int(np.ceil(len(filtered_images) / chunk_size))
    for i in range(num_chunks):
        minidx = i * chunk_size
        maxidx = min((i + 1) * chunk_size, len(filtered_images))
        stacked_images_sub = np.ma.stack(filtered_images[minidx:maxidx], axis=0)

        mean, _, _ = sigma_clipped_stats(stacked_images_sub, axis=0, sigma=2)
        stacked_mean += mean
        stacked_mean_num += 1

    mean = stacked_mean / stacked_mean_num
    mean = np.nan_to_num(mean, nan=1.0)
    ff_delta = mean / np.mean(mean)

    return ff_delta


def compute_ff_delta(filtered_images: list) -> np.ndarray:
    """Compute the mean of the spectrally flattened, mean-normalized images.

    Parameters
    ----------
    filtered_images : `list`
        Input list of filtered SPHEREx images.

    Returns
    -------
    ff_delta : ~numpy.ndarray
        Per-pixel delta on flat-field image.
    """
    stacked_images = np.ma.stack(filtered_images, axis=0)
    # ff_delta = sigma_clipped_stats(stacked_images, axis=0, sigma=2)[0]
    # Apply a sigma-clipped mean, ignoring NaNs
    # `sigma_clipped_stats` with `np.nanmean` to handle NaNs
    mean, _, _ = sigma_clipped_stats(stacked_images, axis=0, sigma=2)
    # Check for NaNs in the mean
    if np.isnan(mean).any():
        logging.info("Warning: NaNs detected in ff_delta mean values. Replacing NaNs with 1.0.")

    ff_delta = np.nan_to_num(mean, nan=1.0)  # Replace NaNs with a 1
    return ff_delta


def estimate_1D_gain_function(binned_calibration_data: Table) -> np.ndarray:
    """Estimate the 1D gain function Rk based on the binned calibration data.

    Parameters
    ----------
    binned_calibration_data : `astropy.table.Table`
        The table containing the binned calibration data with gain factors.
        It should include columns for 'wl_min', 'wl_max', and 'a_bin' (gain factor).
        Determined from Update Photometric Calibration.

    spectral_bins : `np.ndarray`
        The central wavelengths of the spectral bins.

    Returns
    -------
    gf : `np.ndarray`
        The 1D gain function Rk for each spectral bin.
    """
    # Extract the calibration data for gain factors in each spectral bin
    cal = binned_calibration_data['a_bin']
    zodi_cc = binned_calibration_data['z_bin']
    omega_pix = binned_calibration_data['omega_bin']
    # pix_scale = get_pix_scale()
    # omega_ref = (pix_scale * u.arcsec) ** 2
    omega_ref = (6.15*u.arcsec)**2

    # Calculate 1D gain function, Rk (eq.13).
    gf: np.ndarray = cal * zodi_cc * (omega_ref.to(u.sr)) / omega_pix
    logging.info(f"cal = {cal}")
    logging.info(f"zodi_cc = {zodi_cc}")
    logging.info(f"omega_pix = {omega_pix}")
    logging.info(f"omega_ref.to(u.sr) = {omega_ref.to(u.sr)}")
    logging.info(f"gf = {gf}")

    return gf


def initialize_regression_fitparam(shape):
    return {
        "n": np.zeros(shape, dtype=np.float32),
        "Mx": np.zeros(shape, dtype=np.float32),
        "My": np.zeros(shape, dtype=np.float32),
        "Mxx": np.zeros(shape, dtype=np.float32),
        "Mxy": np.zeros(shape, dtype=np.float32),
    }


def compute_regression_fit(fit_param):
    """
    Compute slope and intercept from regression accumulators.

    Parameters
    ----------
    fit_param : dict
        Dictionary of accumulators: n, Mx, My, Mxx, Mxy.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Slope (b = flat-field), Intercept (a = dark current)
    """
    n = fit_param["n"]
    Mx = fit_param["Mx"]
    My = fit_param["My"]
    Mxx = fit_param["Mxx"]
    Mxy = fit_param["Mxy"]
    # errstate ignores divide by zero and infinite warnings
    # during the calculation
    with np.errstate(divide='ignore', invalid='ignore'):
        denominator = n * Mxx - Mx**2
        b = np.where(denominator != 0.0,
                     (n * Mxy - Mx * My) / denominator,
                     1.0)
        a = np.where(n != 0.0,
                     (My - b * Mx) / n,
                     0.0)

    # Handling in case NaN or infinite values,
    # so it doesn't get propagated into flat
    b[~np.isfinite(b)] = 1.0
    a[~np.isfinite(a)] = 0.0

    return b, a


def initialize_average_param(shape):
    return {
        "yx": np.zeros(shape, dtype=np.float32),
        "n": np.zeros(shape, dtype=np.float32)
    }


def initialize_sigma_param(shape):
    return {
        "sqdiff": np.zeros(shape, dtype=np.float32),
        "n": np.zeros(shape, dtype=np.float32)
    }


def update_average_param_linear_scaled(avg_param, y, region_msk_iter_fine):
    """
    Update average parameters using zodi_data as x, scaled per fine region
    by median(y_region) / median(x_region).
    """
    temp_mask = ~np.isnan(y.data) & ~y.mask
    y_data = np.where(temp_mask, y.data, 0.0)

    # Set x to be the linear fit to the signal
    x = np.zeros_like(y_data)
    for region_id, region_mask in region_msk_iter_fine:
        valid = region_mask & temp_mask
        if np.count_nonzero(valid) < 10:
            continue

        y_region = y_data[valid]
        yclip = sigma_clip(y_region)

        y_indices, x_indices = np.where(valid)
        coeffs = np.polyfit(x_indices[np.where(~yclip.mask)], y_region[np.where(~yclip.mask)], 1)
        slope, intercept = coeffs

        y_indices_region, x_indices_region = np.where(region_mask)
        for yind, xind in zip(y_indices_region, x_indices_region):
            x[yind, xind] = slope * xind + intercept

    x = np.where(temp_mask, x, 0.0)

    # Compute ratio
    valid = temp_mask & (x != 0)
    ratio = np.zeros_like(y_data)
    ratio[valid] = y_data[valid] / x[valid]

    avg_param["yx"] += ratio
    avg_param["n"] += np.where(valid, 1.0, 0.0)

    return avg_param


def update_sigma_param_linear_scaled(sig_param, y, average, region_msk_iter_fine):
    temp_mask = ~np.isnan(y.data) & ~y.mask
    y_data = np.where(temp_mask, y.data, 0.0)

    # Set x to be the linear fit to the signal
    x = np.zeros_like(y_data)
    for region_id, region_mask in region_msk_iter_fine:
        valid = region_mask & temp_mask
        if np.count_nonzero(valid) < 10:
            continue

        y_region = y_data[valid]
        yclip = sigma_clip(y_region)

        y_indices, x_indices = np.where(valid)
        coeffs = np.polyfit(x_indices[np.where(~yclip.mask)], y_region[np.where(~yclip.mask)], 1)
        slope, intercept = coeffs

        y_indices_region, x_indices_region = np.where(region_mask)
        for yind, xind in zip(y_indices_region, x_indices_region):
            x[yind, xind] = slope * xind + intercept

    x = np.where(temp_mask, x, 0.0)

    # Compute ratio and residual
    ratio = np.zeros_like(y_data)
    valid = temp_mask & (x != 0)
    ratio[valid] = y_data[valid] / x[valid]

    sqdiff = (ratio - average) ** 2
    sig_param["sqdiff"] += np.where(valid, sqdiff, 0.0)
    sig_param["n"] += np.where(valid, 1.0, 0.0)

    return sig_param


def update_regression_fitparam_linear_scaled(fit_param, y, average, sigma,
                                             region_msk_iter_fine):
    temp_mask = ~np.isnan(y.data) & ~y.mask
    y_data = np.where(temp_mask, y.data, 0.0)

    # Set x to be the linear fit to the signal
    x = np.zeros_like(y_data)
    for region_id, region_mask in region_msk_iter_fine.items():
        valid = region_mask & temp_mask
        if np.count_nonzero(valid) < 10:
            continue

        y_region = y_data[valid]
        yclip = sigma_clip(y_region)

        y_indices, x_indices = np.where(valid)
        coeffs = np.polyfit(x_indices[np.where(~yclip.mask)], y_region[np.where(~yclip.mask)], 1)
        slope, intercept = coeffs

        y_indices_region, x_indices_region = np.where(region_mask)
        for yind, xind in zip(y_indices_region, x_indices_region):
            x[yind, xind] = slope * xind + intercept

    x = np.where(temp_mask, x, 0.0)

    # Compute ratio for outlier rejection
    ratio = np.zeros_like(y_data)
    valid = temp_mask & (x != 0)
    ratio[valid] = y_data[valid] / x[valid]
    good = np.abs(ratio - average) < 1.5 * sigma
    combined = valid & good

    # Final accumulation
    y_data = np.where(combined, y.data, 0.0)
    x = np.where(combined, x, 0.0)
    n = np.where(combined, 1.0, 0.0)

    fit_param["n"] += n
    fit_param["Mx"] += x
    fit_param["My"] += y_data
    fit_param["Mxx"] += x**2
    fit_param["Mxy"] += x * y_data

    return fit_param


def detect_cluster_dbscan(data, threshold=6, min_cluster_pixels=100, eps=10, debug=False):
    """
    Detects large bright clusters using DBSCAN on unmasked bright pixels.

    Parameters
    ----------
    data : np.ma.MaskedArray
        Masked image data.
    threshold : float
        Threshold in std-dev above median.
    min_cluster_pixels : int
        Minimum cluster size to flag image.
    eps : float
        DBSCAN spatial radius.
    debug : bool
        If True, print cluster stats.

    Returns
    -------
    bool : True if large bright cluster detected
    """
    # Ensure masked pixels are excluded from stats
    if not np.ma.isMaskedArray(data):
        data = np.ma.masked_invalid(data)

    # Use only unmasked pixels
    valid_data = data.compressed()
    if valid_data.size == 0:
        if debug:
            print("All pixels are masked.")
        return False

    median = np.ma.median(data)
    std = np.ma.std(data)

    norm = (data - median) / std
    bright_mask = (norm > threshold) & ~data.mask

    # Extract (y, x) coords of bright pixels
    bright_coords = np.column_stack(np.where(bright_mask))

    if bright_coords.shape[0] == 0:
        if debug:
            print("No bright pixels above threshold.")
        return False

    # Cluster bright pixels spatially
    db = DBSCAN(eps=eps, min_samples=5).fit(bright_coords)
    labels = db.labels_

    # Count cluster sizes (ignore noise = -1)
    unique, counts = np.unique(labels[labels != -1], return_counts=True)

    if debug:
        print("Cluster sizes:", dict(zip(unique, counts)))

    return np.any(counts >= min_cluster_pixels)


def fit_background(data, xx, yy, mask_bkg):
    """
    fit background with 2nd-order polynomials
    """

    x, y = xx[mask_bkg], yy[mask_bkg]
    z = data[mask_bkg]
    zunc = z*0 + sigma_clipped_stats(z)[2]

    p_init = models.Polynomial2D(degree=2)
    fitter = fitting.FittingWithOutlierRemoval(fitting.LinearLSQFitter(),
                                               sigma_clip, niter=10, sigma=3)

    return fitter(p_init, x, y, z, weights=1.0/zunc**2)


def compute_proportional_coefficient(imageA, imageB, mask=None,
                                     lsigma=5, hsigma=3, max_iter=10,
                                     return_mask=True, verbose=False):
    """
    Computes the proportional coefficient 'k' that minimizes the chi-square difference
    between imageB and k * imageA.

    Parameters:
    - imageA (np.ndarray): The reference 2D image.     = template
    - imageB (np.ndarray): The target 2D image to fit  = data
    - mask (np.ndarray, optional): A boolean mask to
      specify which pixels to include in the fit.

    Returns:
    - k (float): The best-fit proportional coefficient.
    - chi2 (float): The minimized chi-square value.
    """

    if mask is None:
        mask = np.ones_like(imageA, dtype=np.bool)

    for i in range(max_iter):
        k = np.sum(imageA[mask] * imageB[mask]) / np.sum(imageA[mask] * imageA[mask])

        # Compute chi-square
        chi2 = np.sum((imageB[mask] - k * imageA[mask]) ** 2)
        rms = np.sqrt(chi2/mask.sum())

        # Update mask
        mask_clipped = np.abs(imageB - k * imageA) < 5*rms
        mask_clipped = ((imageB - k * imageA) > -lsigma*rms) & ((imageB - k * imageA) < hsigma*rms)
        mask = mask & mask_clipped

        if verbose:
            if i == 0:
                print(f"{'iter':>4s}  {'ampl':>10s}  {'chi2':>10s}  {'rms':>8s}")
            print(f"{i:4d}  {k:10.4f}  {chi2:10.4e}  {rms:8.4f}")

    if return_mask:
        return k, chi2, mask
    else:
        return k, chi2


# ==============================================================================
# Visualization Tools
# matplotlib.use('Agg')


def visualize_region(region_msk, values, name, pair_idx, region_id):
    if pair_idx < 2 and region_id < 2 and False:
        img = np.full(region_msk.shape, np.nan)
        img[region_msk] = values
        visualize_img(img, name)


def visualize_img(img, name, out_dir, vmin=None, vmax=None, cmap='jet'):
    out_dir.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=100)
    im = ax.imshow(img, origin='lower', vmin=vmin, vmax=vmax, cmap=cmap)
    divider = make_axes_locatable(ax)
    cbar_ax = divider.new_horizontal(size="5%", pad=0.15, title=7 * ' ')
    ax.get_figure().add_axes(cbar_ax)
    plt.colorbar(im, cax=cbar_ax)
    plt.savefig(out_dir / (name + '.png'))
    plt.close(fig)
    print(f'Plot saved at {out_dir}')

# ==============================================================================
# Module API


class EstimateAbsGainMatrixConnections(pipeBase.PipelineTaskConnections,
                                       dimensions=("instrument", "detector"),
                                       defaultTemplates={}):
    dataset = cT.Input(
        name="intype",  # default dataset type for input
        doc="Input images.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
        deferLoad=True,
        multiple=True
    )
    photo_calib = cT.PrerequisiteInput(
        name='photo_calib',
        doc="Input Binned Photometric Calibration data.",
        storageClass="SpxPhotoCalib",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    abs_cal = cT.Input(
        name='abs_cal',
        doc="Output relative flat field estimate.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    ff_est = cT.Output(
        name='ff_est',
        doc="Output flat field estimate.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    dark_est = cT.Output(
        name='dark_est',
        doc="Output flat field estimate.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    abs_gain_matrix = cT.Output(
        name='abs_gain_matrix',
        doc="Absolute gain matrix estimate.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class EstimateAbsGainMatrixSharedConfig(Config):
    num_iter = Field(
        doc="Number of flat field estimation/correction iterations: 1 or 2."
        " 2 is the minimum number of iterations required."
        " There will be ``n - 1`` polynomial fits.",
        dtype=int,
        default=2)
    num_regions = Field(
        doc="Number of wavelength intervals in a band with "
        "equal wavelength spacing. "
        "The detector will be dissected into wavelength regions and"
        " each wavelength strip will be fitted with a 2-D polynomial"
        " independent of other regions.",
        dtype=int,
        default=64)
    num_regions_fine = Field(
        doc="Number of wavelength intervals in a band with "
        "equal wavelength fine spacing. "
        "The detector will be dissected into wavelength regions and"
        " each wavelength strip will be fitted with a 2-D polynomial"
        " independent of other regions.",
        dtype=int,
        default=256)
    region_list = ListField(
        doc="If set, estimations will be performed on listed regions."
        " This is useful for faster testing and validation."
        " All elements must be in ``range(num_regions)``.",
        dtype=int,
        default=None,
        optional=True)
    region_list_fine = ListField(
        doc="If set, estimations will be performed on fine listed regions."
        " This is useful for faster testing and validation."
        " All elements must be in ``range(num_regions_fine)``.",
        dtype=int,
        default=None,
        optional=True)
    median_capacity = Field(
        doc="Maximum amount of flat-field estimates to store to calculate the"
        " median before taking the average of medians.",
        dtype=int,
        default=16)
    quantile_cut = Field(
        doc="Pixels above this quantile will be set to np.nan.",
        dtype=float,
        default=.98)
    dgl_qntl_threshold = Field(
        doc="Exclude images with a DGL background above this value.",
        dtype=float,
        default=.88)
    galactic_lat_threshold = ListField(
        doc="Exclude images in the galactic plane, for a detector.",  # In order by det ID.
        dtype=float,
        default=[20., 30., 30., 30., 30., 30.])
    min_images = Field(
        doc="Minimum number of images needed after filtering to estimate the "
        " flat field.",
        dtype=int,
        default=2)
    filter_images = ConfigurableField(
        doc="Filter Dataset By Mask",
        target=FilterByMaskTask)
    dqa_threshold = Field(
        doc="Threshold difference between prior and calculated flat field",
        dtype=float,
        default=.10)
    use_linear_regression = Field(
        doc="If True, use slope-fitting method (regression) "
        "to estimate flat-field and dark current.",
        dtype=bool,
        default=False,
    )


class EstimateAbsGainMatrixConfig(EstimateAbsGainMatrixSharedConfig,
                                  pipeBase.PipelineTaskConfig,
                                  pipelineConnections=EstimateAbsGainMatrixConnections):
    pass


class EstimateAbsGainMatrixTask(pipeBase.PipelineTask):
    """Implementation of L2 Estimate Pipetask Flat-Field Correction.

    Attributes
    ----------
    ConfigClass : `lsst.pipe.base.PipelineTask`
        Pipeline task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-026

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `pipeBase.Struct` object.
    """

    _DefaultName = "estimate-flat-field"
    ConfigClass = EstimateAbsGainMatrixConfig
    full_name = "Estimate Flat-Field Correction"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-026 v5.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.array_id: int
        self.npix_x: int
        self.npix_y: int

    def runQuantum(self, butlerQC, inputRefs, outputRefs) -> None:
        inputs = butlerQC.get(inputRefs)

        outputs = self.run(**inputs)

        # Filter only DatasetRef objects with a calibration datasetType
        def safe_calib_inputs(input_refs):
            return [
                (ref.datasetType.name, ref.id)
                for _, ref in input_refs
                if getattr(ref, "datasetType", None) is not None
                and hasattr(ref.datasetType, "isCalibration")
                and ref.datasetType.isCalibration()
            ]

        input_ids = safe_calib_inputs(inputRefs)
        print(f"input ids = {input_ids}")
        add_calibration_provenance(outputs.abs_gain_matrix, input_ids)

        butlerQC.put(outputs, outputRefs)

    @add_history(fld_name="abs_gain_matrix")
    @add_history(fld_name="ff_est")
    def run(self,
            dataset: list[DeferredDatasetHandle],
            abs_cal: SPHERExImage,
            photo_calib: Table,
            sigma_cut: int = 4,
            percent_cut: float = 0.05,
            verbose: bool = False,
            corr_solar_elong: bool = False) -> pipeBase.Struct:
        """Estimate relative flat field and return as an ``hdul`` image.

        The array_id and flat-field shape are determined from the first image
        and the user is responsible for ensuring all images are
        from the same detector and have the same shape.

        Parameters
        ----------
        dataset : `list[~lsst.daf.butler.DeferredDatasetHandle]`
            List of deferred SPHEREx images.
        ff_prev : `~spx_data.SPHERExImage`
            Initial flat field estimate.
        sigma_cut : `int`, default = 2
            Sigma value at which to apply excessive variation cut.
        percent_cut : `float`, default 0.05
            Percentage threshold at which to apply excessive variation cut.
        verbose : `bool`, default = False
            Print out processing details.

        Returns
        -------
        output : `lsst.pipe.base.Struct`
            Result with components:

            - ``array_id`` : `int`
                The detector id, 1-6.
            - ``flat_field_img`` : `~spx_data.SPHERExImage`
                The estimated relative flat-field image.

        Notes
        -----
        The user is responsible for ensuring all data have the same array_id.
        Inputs to the run method (or must at least be available to module):
          - Input image dataset (~3500 images from the previous ~week for a detector).
          - Source mask for each one (module operates on source-masked images).
          - The most recent FF solution used to correct the input images
            for this detector.
        """

        log.disabled = True
        t0_total = time.time()
        timings = {}

        # Get detector ID from first image, all images input are from same detector.
        self.array_id = dataset[0].get().getArrayId()
        self.npix_x, self.npix_y = dataset[0].get().data.shape

        # -------------------
        # FF Algorithm Steps
        # -------------------
        # Step 0: Mask point sources; the only sky signals are diffuse background.
        t0 = time.time()
        masked_images = []
        latitudes = []
        longitudes = []
        obstimes = []
        wcss = []
        # zodi_models = []
        print(f"length dataset = {len(dataset)}")

        for deferred_img in dataset:
            spximg_i = deferred_img.get()

            array_id = copy.copy(spximg_i.getArrayId())
            data = copy.copy(spximg_i.data)
            flags = copy.copy(spximg_i.flags)
            header = dict(spximg_i.header)
            # img_zodi = copy.copy(spximg_i.zodi)
            wcs = (spximg_i.wcs.deepcopy() if hasattr(spximg_i.wcs, "deepcopy")
                   else copy.deepcopy(spximg_i.wcs))

            spximg = SPHERExImage(data=data, header=header, unit="MJy/sr")
            spximg.flags = flags
            spximg.wcs = wcs

            del spximg_i   # now safe to delete the original memory-mapped object!

            try:
                if spximg.header["FINAST"] != 0:
                    continue
            except KeyError:
                continue
            try:
                sps_flags = spximg.header.get("SPS_FLAGS", "")
                if "saa" in [f.strip().lower() for f in sps_flags.split(",")]:
                    print(f"skipping obs in saa: {spximg.header['OBSID']}D{array_id}")
                    continue
            except KeyError:
                continue
            # The *data is now a masked array*.
            mask_flags = Flag.SOURCE | Flag.TRANSIENT | Flag.OVERFLOW | Flag.SUR_ERROR
            # Bring images back to L1
            result = np.divide(spximg.data, abs_cal.data,
                               out=np.ones_like(spximg.data),
                               where=(abs_cal.data != 0))
            spximg.data = np.ma.array(result, mask=spximg.getFlagsAsMask(mask_flags))

            # Check for NaNs after masking
            if np.isnan(spximg.data).any():
                if verbose:
                    print(f"Skipping image due to NaNs: {spximg.header['OBSID']}D{array_id}")
                    continue

            # Check for bright source cluster that survived masking
            has_cluster = detect_cluster_dbscan(spximg.data,
                                                threshold=10,            # Bright pixel threshold (in sigma)
                                                min_cluster_pixels=200,  # Require 200-pixel clusters
                                                eps=20,                  # Cluster pixels within 20 pixels
                                                debug=False
                                                )
            if has_cluster:
                print(f"[WARN] Large bright cluster found in {spximg.header['OBSID']}D{array_id}, skipping.")
                continue

            masked_images.append(spximg)
            # zodi_models.append(img_zodi)

            # Extract and store latitude and longitude arrays
            try:
                latitudes.append(spximg.header['LAT_ECL'])
                longitudes.append(spximg.header['LON_ECL'])
            except KeyError:  # simulated images do not have LAT_ECL and LON_ECL
                crval1_ra = spximg.wcs.wcs.crval[0]
                crval2_dec = spximg.wcs.wcs.crval[1]
                equatorial_coord = SkyCoord(ra=crval1_ra * u.deg, dec=crval2_dec * u.deg, frame='icrs')
                # Convert to Ecliptic coordinates
                ecliptic_coord = equatorial_coord.barycentrictrueecliptic
                # Extract Ecliptic latitude and longitude
                ecliptic_lat = ecliptic_coord.lat
                ecliptic_lon = ecliptic_coord.lon
                latitudes.append(ecliptic_lat)
                longitudes.append(ecliptic_lon)

            obstimes.append(spximg.getExposureStartTime())
            wcss.append(spximg.wcs)

        timings["Load & mask images"] = time.time() - t0

        # Step 1: Filter by galactic latitude.
        t0 = time.time()
        # filtered dataset is a list of masked arrays.
        filtered_dataset, filtered_lats, filtered_longs, filtered_obstimes, filtered_wcs =\
            ImageFiltering.filter_galactic_lat(
                masked_images, latitudes, longitudes, obstimes, wcss, self.config, verbose)
        # Validate images meet requirements before proceeding.
        num_img = len(filtered_dataset)
        if num_img < self.config.min_images:
            raise ValueError(f"Need at least {self.config.min_images} images, only have "
                             f"{num_img} images after filtering for galactic latitude > 30.")
        print(f"Images after galactic latitude filter: {num_img}")
        timings["Filter galactic latitude"] = time.time() - t0

        # Step 2: Solar Elongation Correction
        # Compute tilt function and correct images.
        # Parameters for the tilt correction
        if corr_solar_elong is True:
            t0 = time.time()
            # Compute the tilt function
            tilt_function_dict = compute_tilt_function(filtered_dataset,
                                                       filtered_lats,
                                                       filtered_longs,
                                                       filtered_obstimes)

            # Apply tilt correction to the images
            corrected_images = tilt_correct(filtered_dataset,
                                            filtered_lats,
                                            filtered_longs,
                                            filtered_obstimes,
                                            filtered_wcs,
                                            tilt_function_dict)
            filtered_dataset = corrected_images
            timings["Tilt correction"] = time.time() - t0

        # Step 3: Define spectral bins.
        t0 = time.time()
        print("Define spectral bins")
        region_msk_iter, region_msk_iter_fine = define_spectral_bins(dataset[0].get(), self.config)
        # assert region_msk_iter.num_regions == self.config.num_regions
        # assert region_msk_iter_fine.num_regions == self.config.num_regions_fine
        lambda_mesh: np.ndarray = get_wavelength_mesh(self.array_id, self.npix_x, self.npix_y)
        timings["Define spectral bins"] = time.time() - t0

        # Correct for He I line
        if self.array_id in (1, 2):
            # Use env variables for He templates
            if self.array_id == 1:
                template = os.getenv('HE_I_TEMPLATE_D1')
            if self.array_id == 2:
                template = os.getenv('HE_I_TEMPLATE_D2')

            template = template[4:2044, 132:2172]

            # Should run the below on filtered_dataset
            # after we define spectral bins (Step 3)
            # Key parameters
            # BAND = 1
            AIRGLOW_WAVELENGTH = 1.083   # airglow wavelength
            AIRGLOW_WIDTH = 0.027        # airglow half-width (wavelength direction)
            BACKGROUND_WIDTH = 0.035     # half-width of background region (in terms of wavelength)
            # N_WBINS = 50  # number of wavelength bins for airglow profile
            # N_KNOTS = 20  # number of knots for BSpline fitting
            # LSIGMA, HSIGMA = 3.0, 2.5
            FIT_BACKGROUND = True

            corrected_dataset = []

            for filtered_dataset_element in filtered_dataset:
                data0 = filtered_dataset_element.data
                mask0 = filtered_dataset_element.mask.astype(np.uint8)
                # print(type(data0))
                NY, NX = data0.shape
                xv, yv = np.arange(NX), np.arange(NY)
                xx, yy = np.meshgrid(xv, yv)
                wmap = lambda_mesh

                mask_airglow = np.abs(wmap - AIRGLOW_WAVELENGTH) < AIRGLOW_WIDTH
                mask_good = mask_airglow & (mask0 == 0)
                mask_bkg = ((np.abs(wmap - AIRGLOW_WAVELENGTH) > AIRGLOW_WIDTH)
                            & (np.abs(wmap - AIRGLOW_WAVELENGTH) < BACKGROUND_WIDTH))

                p_fit, _ = fit_background(data0, xx, yy, mask_bkg)

                if FIT_BACKGROUND:
                    data0[mask_airglow] = data0[mask_airglow] - p_fit(xx[mask_airglow], yy[mask_airglow])

                template_array = np.asarray(template.data)  # or template.data.filled(0.0) if it's masked

                k, chi2, mask_clipped = compute_proportional_coefficient(
                    template_array, data0, mask=mask_good,
                    return_mask=True, verbose=True
                )
                fringe_amp = k
                mask = mask_airglow

                data_sub = data0.copy()
                data_sub[mask] = data_sub[mask] - template_array[mask] * fringe_amp
                data_sub[mask] = data_sub[mask] + p_fit(xx[mask], yy[mask])
                # data_sub is what we want to use going forward in the algorithm

                # Append data_sub to corrected_dataset list
                corrected_dataset.append(np.ma.array(data_sub, mask=filtered_dataset_element.mask))

        if self.config.use_linear_regression is False:
            # Step 4: Stack and normalize filtered images.
            t0 = time.time()
            print("Stack and normalize images")
            normalized_stack = stack_and_normalize_images(filtered_dataset)
            print("compute avg flux and wavelength")
            # Step 5: Compute average flux and wavelength in spectral bins.
            spectral_response_fine, wvs_fine =\
                compute_avg_flux_and_wavelength(region_msk_iter_fine,
                                                normalized_stack,
                                                lambda_mesh)
            timings["Stack & avg flux"] = time.time() - t0

            # Step 6: Compute spectral response on a finer grid.
            # logging.info("compute fine grid spectral response")
            # spectral_response_fine =\
            # compute_fine_grid_spectral_response(region_msk_iter_fine,
            # wvs, fluxes, gf, lambda_mesh)
            # Step 7:
            # Divide out the spectral response for each image in filtered dataset.
            print("divide out spectral response")
            t0 = time.time()
            divide_out_spectral_response(filtered_dataset, region_msk_iter_fine, spectral_response_fine,
                                         self.config.use_linear_regression)
            timings["Divide spectral response"] = time.time() - t0

        # Step 8: Cut images with excessive signal variation based on specs.
        print("cut images with excessive signal variation")
        t0 = time.time()
        if self.array_id in (1, 2):
            filtered_dataset = corrected_dataset
        filtered_images, valid_indices = cut_images_with_excessive_signal_variation(filtered_dataset,
                                                                                    sigma_cut,
                                                                                    percent_cut,
                                                                                    verbose,
                                                                                    return_indices=True)
        # filtered_zodi_models = [zodi_models[i] for i in valid_indices]
        timings["Cut noisy images"] = time.time() - t0

        # Step 9: Compute the mean of spectrally flattened, mean-normalized images.

        # Optional slope fitting algorithm
        if self.config.use_linear_regression is True:
            # Compute per-pixel average and sigma (outlier rejection)
            shape = filtered_images[0].shape
            avg_param = initialize_average_param(shape)

            for img in filtered_images:
                avg_param = update_average_param_linear_scaled(
                    avg_param, img, region_msk_iter_fine
                )

            # per pixel average ratio image/median(image)
            with np.errstate(invalid='ignore', divide='ignore'):
                average = np.divide(avg_param["yx"], avg_param["n"])
                average[~np.isfinite(average)] = 0.0

            sig_param = initialize_sigma_param(shape)
            for img in filtered_images:
                sig_param = update_sigma_param_linear_scaled(
                    sig_param, img, average, region_msk_iter_fine
                )
            with np.errstate(invalid='ignore', divide='ignore'):
                ratio_var = np.divide(sig_param["sqdiff"], sig_param["n"])
                ratio_var[~np.isfinite(ratio_var)] = 0.0
                sigma = np.sqrt(ratio_var)

            fit_param = initialize_regression_fitparam(shape)

            for img in filtered_images:
                fit_param = update_regression_fitparam_linear_scaled(
                    fit_param, img, average, sigma, region_msk_iter_fine
                )

            ff_est, dark_est = compute_regression_fit(fit_param)

        if self.config.use_linear_regression is False:
            ff_est = compute_ff_delta(filtered_images)
            dark_est = np.zeros_like(ff_est)

        # Normalize ff_est
        ff_est /= np.nanmedian(ff_est[np.isfinite(ff_est)])

        # Estimate gain function
        cal_values = ArrayCalValues()
        gf = np.ones(self.config.num_regions_fine) / cal_values.cal_values[self.array_id]
        abs_gain_matrix = np.zeros_like(ff_est)
        for region_id, fine_mask in region_msk_iter_fine:
            abs_gain_matrix[fine_mask] = gf[region_id] / ff_est[fine_mask]

        # DQA: check flat-field within 10% of prior
        # diff = np.abs(ff_est - ff_prev.data) / ff_prev.data
        # # Check if any pixel exceeds the 10% threshold
        # if np.any(diff > self.config.dqa_threshold):
        #     num_exceeding = np.sum(diff > self.config.dqa_threshold)
        #     max_diff = np.max(diff)
        #     # Log the DQA result but do not stop processing.
        #     print(f"{num_exceeding} pixels exceed the 10% threshold.
        #     Max deviation: {max_diff:.2%}")

        # ---------------------------------
        # Build output for pipebase struct
        # Flat field estimate (ff_est)
        flat_field_img = SPHERExImage(ff_est, unit=u.dimensionless_unscaled)
        flat_field_img.setArrayId(self.array_id)
        flat_field_img.header['SPXDTYPE'] = ('Relative Flat-Field',
                                             'Data type for estimated flat-field')

        flat_field_img.header['IMG0'] = ('Unknown', 'Earliest Image ID')
        # Dark estimate (dark_est)
        dark_img = SPHERExImage(dark_est, unit="electron/s")

        # Metadata is stored in butler
        self.metadata.add("array_id", self.array_id)

        abs_gain_matrix_image = SPHERExImage(abs_gain_matrix,
                                             unit=u.Unit("MJy/sr") / u.Unit("electron/s"))

        for k, v in timings.items():
            print(f"{k:>30}: {v:.2f} sec")
        print(f"{'Total runtime':>30}: {time.time() - t0_total:.2f} sec\n")

        add_detcoord(flat_field_img)
        add_detcoord(dark_img)
        add_detcoord(abs_gain_matrix_image)

        output = pipeBase.Struct(
            array_id=self.array_id,
            ff_est=flat_field_img,
            dark_est=dark_img,
            abs_gain_matrix=abs_gain_matrix_image,
            region_msk_iter=region_msk_iter,
            region_msk_iter_fine=region_msk_iter_fine,
            filtered_images=filtered_images
        )

        return output
