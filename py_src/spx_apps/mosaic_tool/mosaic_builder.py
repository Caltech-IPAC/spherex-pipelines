"""Main mosaic builder orchestration."""

import gc
import logging
import os
from collections import defaultdict
from glob import glob
from typing import Optional, Callable
import numpy as np
import astropy.units as u
import concurrent.futures
import threading
from spx_data import (
    ChannelDefinition, ArrayWavelengthModel, LVFContourChannelDefinition, Flag
)
from spx_data.spectral_channel import get_lvf_contour_path
from spx_data.spectral_channels_reference import get_reference_spectral_channel_defs
from .config import MosaicConfig
from .results import ChannelMosaic, write_mosaic_cube
from .l2_loader import L2FileLoader
from .grid import create_mosaic_grid
from .reprojection import HistogramReprojection
from .spectral_interpolation import SpectralInterpolator, InterpConfig, wavelength_to_array
from ..resource_usage import log_resource_usage
from numba import njit

# Bad pixel flags. Input image pixels with these flags
# are set to NaN and excluded from mosaic.
BAD_PIX_FLAGS = (Flag.TRANSIENT
                 | Flag.SUR_ERROR
                 | Flag.NONFUNC
                 | Flag.DICHROIC
                 | Flag.MISSING_DATA
                 | Flag.HOT
                 | Flag.COLD
                 | Flag.PHANMISS
                 | Flag.NONLINEAR
                 | Flag.PERSIST
                 | Flag.CROSSTALK
                 | Flag.GHOST
                 | Flag.GHOST_FPA
                 | Flag.GHOST_EXT
                 | Flag.STREAK
                 | Flag.BLOOM
                 | Flag.SNOWBALL
                 | Flag.HALO
                 | Flag.SATELLITE_HALO)


# Propagated pixel flags should be propagated from
# input images to the output mosaic.
PROPAGATED_FLAGS = Flag.OVERFLOW | Flag.SOURCE | Flag.OUTLIER


@njit(nogil=True, cache=True)
def _numba_accumulate_pixels(
    data_in, flags_in, mask,
    ix_idx, iy_idx, bounds_mask,
    out_acc, out_nhits, out_flags
):
    """
    Zero-allocation kernel. Maps valid input pixels directly into the output mosaic.
    Runs natively outside the Python GIL.
    """
    ny, nx = data_in.shape
    for y in range(ny):
        for x in range(nx):
            # 1. Skip if outside the subchannel LVF contour or out of mosaic bounds
            if not mask[y, x] or not bounds_mask[y, x]:
                continue
            # Get pre-computed output coordinates
            ox = ix_idx[y, x]
            oy = iy_idx[y, x]

            # 2. Accumulate flags (Done BEFORE the NaN check to match original logic)
            out_flags[oy, ox] |= flags_in[y, x]

            # 3. Process Data
            val = data_in[y, x]
            # Skip NaNs for data and hit counting
            if np.isnan(val):
                continue

            # 4. Accumulate data natively directly into memory
            out_acc[oy, ox] += val
            out_nhits[oy, ox] += 1


class MosaicBuilder:
    """Build per-channel mosaics from L2 files.

    This class orchestrates the mosaic creation process using a file-centric
    approach that minimizes I/O:
    1. Creates output grid based on configuration
    2. Groups requested channels by detector array
    3. For each array, opens each L2 file once and processes all subchannels
    4. Each subchannel is masked, cropped to bounding box, and reprojected
    5. Results are accumulated and returned after all files are processed

    This approach reduces file I/O by 17x compared to opening each file
    once per subchannel.

    Parameters
    ----------
    channel_definition : ChannelDefinition
        Defines how detector pixels map to spectral channels.
    config : MosaicConfig
        Mosaic grid and processing configuration.
    loader : L2FileLoader, optional
        Custom file loader. Default creates new L2FileLoader.

    Examples
    --------
    >>> from spx_data import (
    ...     LogSpacedChannelDefinition, TabularWavelengthModel
    ... )
    >>>
    >>> # Setup
    >>> wl_model = TabularWavelengthModel()
    >>> channel_def = LogSpacedChannelDefinition(wl_model)
    >>> config = MosaicConfig(
    ...     ra_center=322.0, dec_center=-6.0,
    ...     width_deg=1.0, height_deg=1.0,
    ... )
    >>>
    >>> # Build mosaics
    >>> builder = MosaicBuilder(channel_def, config)
    >>> l2_files = {1: ['file1.fits', 'file2.fits'], 2: [...], ...}
    >>>
    >>> for mosaic in builder.build_all(l2_files):
    ...     info = mosaic.channel_info
    ...     print(f"Array {info.detector}, Ch {info.subchannel}: {info.wavelength:.3f}")
    ...     mosaic.write(f'mosaic_a{info.detector}_ch{info.subchannel:02d}.fits')

    """

    def __init__(
        self,
        channel_definition: ChannelDefinition,
        config: MosaicConfig,
        loader: Optional[L2FileLoader] = None
    ):
        self.channel_def = channel_definition
        self.config = config
        self.loader = loader or L2FileLoader(
            bad_pixel_flags=config.bad_pixel_flags,
            propagated_flags=config.propagated_flags
        )

        # The file-centric accumulation kernel (_numba_accumulate_pixels)
        # implements histogram (nearest-pixel-bin) semantics unconditionally;
        # HistogramReprojection is instantiated only to source nhits_dtype
        # from a single place in the reprojection package.
        self._reprojection = HistogramReprojection()

        # Extra FITS cards attached to every output mosaic header.
        # None for rectangular mosaics; subclasses may set a fits.Header
        # (e.g. ORDER/NPIX for HiPS tiles).
        self._extra_header = None

        # Create output grid
        self.wcs_mosaic, self.shape_mosaic = self._create_grid(config)

    def _create_grid(self, config: MosaicConfig):
        """Create the output grid; grid-creation hook for subclasses.

        Returns
        -------
        tuple of (astropy.wcs.WCS, tuple of (int, int))
            Output WCS and array shape (ny, nx).

        """
        return create_mosaic_grid(
            ra_center=config.ra_center,
            dec_center=config.dec_center,
            width_deg=config.width_deg,
            height_deg=config.height_deg,
            pixel_size_arcsec=config.pixel_size_arcsec,
            rotation_deg=config.rotation_deg,
            projection=config.projection
        )

    def build_all(
        self,
        l2_files_by_array: dict[int, list[str]],
        channels: Optional[list[tuple[int, int]]] = None,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        output_callback: Optional[Callable[[ChannelMosaic], None]] = None
    ) -> list[ChannelMosaic]:
        """Build mosaics for all (or specified) channels.

        This method uses a file-centric approach: each L2 file is opened once
        and all relevant subchannels are processed before moving to the next file.
        This reduces I/O by 17x compared to opening each file once per subchannel.

        Parameters
        ----------
        l2_files_by_array : dict
            Mapping of array number to list of L2 file paths.
            Example: {1: ['file1.fits', 'file2.fits'], 2: [...], ...}
        channels : list of (array, subchannel), optional
            Specific channels to process. If None, processes all channels
            defined by the channel_definition.
        progress_callback : callable, optional
            Called with (current_file_index, total_files, status_message)
            for progress reporting.
        output_callback : callable, optional
            Called with each completed ChannelMosaic. When provided, mosaics
            are passed to this callback immediately upon completion and NOT
            accumulated in memory. This reduces memory usage by ~560 MB for
            102 channels. The callback typically saves the mosaic to disk.
            When output_callback is provided, this method returns an empty list.

        Returns
        -------
        list of ChannelMosaic
            Mosaic results for all successfully processed channels.
            Returns empty list if output_callback is provided (mosaics are
            passed to callback instead of being accumulated).

        Examples
        --------
        >>> # Process all channels (accumulate in memory)
        >>> mosaics = builder.build_all(l2_files)
        >>> for mosaic in mosaics:
        ...     info = mosaic.channel_info
        ...     mosaic.write(f'output/mosaic_{info.detector}_{info.subchannel}.fits')

        >>> # Process with incremental saving (low memory)
        >>> def save_mosaic(mosaic):
        ...     info = mosaic.channel_info
        ...     mosaic.write(f'output/mosaic_{info.detector}_{info.subchannel}.fits')
        >>> builder.build_all(l2_files, output_callback=save_mosaic)

        >>> # With progress reporting
        >>> def report(i, n, msg):
        ...     print(f"[{i}/{n}] {msg}")
        >>> mosaics = builder.build_all(l2_files, progress_callback=report)

        """
        if channels is None:
            channels = self.channel_def.get_all_channels()

        # Group channels by array for efficient processing
        channels_by_array: dict[int, list[int]] = defaultdict(list)
        for array, subchannel in channels:
            channels_by_array[array].append(subchannel)

        # Count total files for progress reporting
        total_files = sum(
            len(l2_files_by_array.get(array, []))
            for array in channels_by_array
        )

        # Process each array
        all_mosaics = []
        file_counter = 0

        for array in sorted(channels_by_array.keys()):
            subchannels = channels_by_array[array]
            l2_files = l2_files_by_array.get(array, [])

            if not l2_files:
                continue

            # Build all subchannels for this array
            mosaics = self._build_array_channels(
                l2_files=l2_files,
                array=array,
                subchannels=subchannels,
                progress_callback=progress_callback,
                file_counter=file_counter,
                total_files=total_files
            )

            # Either pass to callback or accumulate
            if output_callback:
                for mosaic in mosaics:
                    output_callback(mosaic)
                # Explicitly clear references and force garbage collection
                # to free memory from this array before processing the next
                del mosaics
                gc.collect()
            else:
                all_mosaics.extend(mosaics)

            file_counter += len(l2_files)

        return all_mosaics

    def _compute_union_bbox(
        self,
        array: int,
        subchannels: list[int],
        detector_shape: tuple[int, int] = (2040, 2040)
    ) -> tuple[int, int, int, int]:
        """Compute the union bounding box across all subchannels of an array.

        The union bbox covers every pixel that belongs to any subchannel,
        allowing a single WCS transform per file rather than one per subchannel.
        Because LVF contours overlap heavily (union/sum ratio ~0.65), the union
        region is smaller than the sum of individual bboxes — so this is strictly
        cheaper than the per-subchannel approach.

        Parameters
        ----------
        array : int
            Detector array number (1-6).
        subchannels : list of int
            Subchannel numbers to union over.
        detector_shape : tuple of (int, int)
            Shape of the detector in (ny, nx). Default is (2040, 2040).

        Returns
        -------
        tuple of (y_min, y_max, x_min, x_max)
            Union bounding box suitable for slicing as [y_min:y_max, x_min:x_max].

        """
        union_y_min, union_y_max = detector_shape[0], 0
        union_x_min, union_x_max = detector_shape[1], 0

        for sc in subchannels:
            mask = self.channel_def.get_channel_mask(array, sc, detector_shape)
            if np.any(mask):
                y0, y1, x0, x1 = self._get_bounding_box(mask)
                union_y_min = min(union_y_min, y0)
                union_y_max = max(union_y_max, y1)
                union_x_min = min(union_x_min, x0)
                union_x_max = max(union_x_max, x1)

        return union_y_min, union_y_max, union_x_min, union_x_max

    def _compute_pixel_mapping(
        self,
        wcs_in,
        origin: tuple[int, int],
        crop_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute output pixel indices for a cropped input region.

        This is the WCS-transform-only portion extracted from reproject(),
        operating on the union bbox crop region. Called once per file rather
        than once per subchannel, giving a ~17x reduction in WCS work.

        The pixel mapping depends only on the spatial WCS of the input file
        and the output mosaic grid — it is identical for all subchannels that
        share the same file, since subchannels differ only in their data mask,
        not in their spatial coordinates.

        Parameters
        ----------
        wcs_in : astropy.wcs.WCS
            Input image WCS.
        origin : tuple of (int, int)
            (y_offset, x_offset) of the crop region within the full detector.
        crop_shape : tuple of (int, int)
            Shape (ny, nx) of the cropped region.

        Returns
        -------
        ix_idx : np.ndarray of int32, shape (ny, nx)
            Output x pixel indices (may be out of bounds — check bounds_mask).
        iy_idx : np.ndarray of int32, shape (ny, nx)
            Output y pixel indices (may be out of bounds — check bounds_mask).
        bounds_mask : np.ndarray of bool, shape (ny, nx)
            True where the output pixel falls within the mosaic grid.

        """
        ra, dec = self._input_sky_coords(wcs_in, origin, crop_shape)

        # Transform sky coordinates to output pixel coordinates
        ix_out, iy_out = self.wcs_mosaic.all_world2pix(ra, dec, 0)
        del ra, dec

        # Round to nearest integer pixel (bin index)
        # 2D output required for dynamic slicing in the Numba kernel
        ny_out, nx_out = self.shape_mosaic
        ix_idx = np.floor(ix_out + 0.5).astype(np.int32)
        iy_idx = np.floor(iy_out + 0.5).astype(np.int32)
        del ix_out, iy_out

        # Bounds mask: pixels that land within the output mosaic grid
        bounds_mask = (
            (ix_idx >= 0) & (ix_idx < nx_out)
            & (iy_idx >= 0) & (iy_idx < ny_out)
        )

        return ix_idx, iy_idx, bounds_mask

    @staticmethod
    def _input_sky_coords(
        wcs_in,
        origin: tuple[int, int],
        crop_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sky coordinates of pixel centers in a cropped input region.

        Parameters
        ----------
        wcs_in : astropy.wcs.WCS
            Input image WCS.
        origin : tuple of (int, int)
            (y_offset, x_offset) of the crop region within the full detector.
        crop_shape : tuple of (int, int)
            Shape (ny, nx) of the cropped region.

        Returns
        -------
        ra, dec : numpy.ndarray
            Sky coordinates in degrees, shape (ny, nx).

        """
        ny_in, nx_in = crop_shape
        y_offset, x_offset = origin

        # Build pixel coordinate grids with origin offset applied.
        # float32 is sufficient for pixel coordinates at detector scales.
        iy_in, ix_in = np.indices((ny_in, nx_in), dtype=np.float32)
        iy_in += y_offset
        ix_in += x_offset

        # Transform input pixel coordinates to sky coordinates
        return wcs_in.all_pix2world(ix_in, iy_in, 0)

    def _build_array_channels(
        self,
        l2_files: list[str],
        array: int,
        subchannels: list[int],
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        file_counter: int = 0,
        total_files: int = 0) -> list[ChannelMosaic]:
        """Build mosaics for multiple subchannels of one array natively via Numba.

        Each file is opened once via multithreading
        and all subchannels are processed before
        moving to the next file.

        WCS transforms are computed once per file over the union bounding box
        of all subchannels, then reused for each subchannel. Because LVF
        contours overlap heavily (union/sum ~0.65), this reduces WCS work by
        ~17x with no loss of accuracy.

        Parameters
        ----------
        l2_files : list of str
            List of L2 file paths for this array.
        array : int
            Detector array number (1-6).
        subchannels : list of int
            List of subchannel numbers to process.
        progress_callback : callable, optional
            Progress reporting callback.
        file_counter : int
            Starting file count for progress reporting.
        total_files : int
            Total file count for progress reporting.

        Returns
        -------
        list of ChannelMosaic
            Mosaic results for all subchannels with valid data.

        """
        # Initialize accumulators for all subchannels
        # Using float32 for data to match L2 input data precision and reduce memory.
        # nhits dtype matches the reprojection method.
        # flags use int32 to match L2 input flags.
        nhits_dtype = self._reprojection.nhits_dtype
        # 1. Initialize master accumulators AND Threading Locks
        accumulators: dict[int, dict] = {}
        channel_locks: dict[int, threading.Lock] = {}

        for subchannel in subchannels:
            accumulators[subchannel] = {
                'acc_data': np.zeros(self.shape_mosaic, dtype=np.float32),
                'nhits': np.zeros(self.shape_mosaic, dtype=nhits_dtype),
                'acc_flags': np.zeros(self.shape_mosaic, dtype=np.int32),
                'n_valid': 0
            }
            channel_locks[subchannel] = threading.Lock()
        # 2. Pre-compute union bounding box and subchannel metadata
        union_bbox = self._compute_union_bbox(array, subchannels)
        union_y_min, union_y_max, union_x_min, union_x_max = union_bbox
        union_origin = (union_y_min, union_x_min)
        union_shape = (union_y_max - union_y_min, union_x_max - union_x_min)

        subchannel_meta: dict[int, tuple] = {}
        detector_shape = (2040, 2040)

        for sc in subchannels:
            full_mask = self.channel_def.get_channel_mask(array, sc, detector_shape)
            union_mask = full_mask[union_y_min:union_y_max, union_x_min:union_x_max]
            if np.any(union_mask):
                # Get the tight bounding box *within* the union bbox
                y0, y1, x0, x1 = self._get_bounding_box(union_mask)
                tight_mask = union_mask[y0:y1, x0:x1]
                # Store the tight mask and its relative coordinates
                subchannel_meta[sc] = (tight_mask, y0, y1, x0, x1)

        # 3. Define the Zero-Allocation worker function
        def process_single_file(index: int, filepath: str):
            # Load file
            image = self.loader.load(filepath, array)
            data_values = image.data.value if hasattr(image.data, 'value') else image.data

            # Compute 2D WCS pixel mapping ONCE for the union bbox region.
            ix_idx_2d, iy_idx_2d, bounds_mask_2d = self._compute_pixel_mapping(
                image.wcs, union_origin, union_shape
            )

            # Process all subchannels using Zero-Allocation Numba Kernel
            for subchannel in subchannels:
                if subchannel not in subchannel_meta:
                    continue

                tight_mask, y0, y1, x0, x1 = subchannel_meta[subchannel]

                # Absolute coordinates on the original detector
                abs_y0, abs_y1 = union_y_min + y0, union_y_min + y1
                abs_x0, abs_x1 = union_x_min + x0, union_x_min + x1

                # Create Zero-Copy Views (No memory is allocated here)
                tight_data = data_values[abs_y0:abs_y1, abs_x0:abs_x1]
                tight_flags = image.flags[abs_y0:abs_y1, abs_x0:abs_x1]

                ix_view = ix_idx_2d[y0:y1, x0:x1]
                iy_view = iy_idx_2d[y0:y1, x0:x1]
                bounds_view = bounds_mask_2d[y0:y1, x0:x1]
                # Accumulate natively (protected by per-subchannel
                # lock to prevent race conditions)
                acc = accumulators[subchannel]
                with channel_locks[subchannel]:
                    _numba_accumulate_pixels(
                        tight_data, tight_flags, tight_mask,
                        ix_view, iy_view, bounds_view,
                        acc['acc_data'], acc['nhits'], acc['acc_flags']
                    )
                    acc['n_valid'] += 1

            # Cleanup references
            del image, data_values, ix_idx_2d, iy_idx_2d, bounds_mask_2d
            return index, filepath

        # 4. Execute threads concurrently
        # Cap workers to prevent RAM spikes if you have hundreds of files.
        max_threads = min(int(os.getenv("SPXTOOL_NCORES", 6)), max(1, len(l2_files)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {
                executor.submit(process_single_file, i, fp): (i, fp)
                for i, fp in enumerate(l2_files)
            }
            for future in concurrent.futures.as_completed(futures):
                i, filepath = future.result()
                if progress_callback:
                    progress_callback(
                        file_counter + i + 1,
                        total_files,
                        f"Array {array}: {filepath.split('/')[-1]}"
                    )

        # 5. Finalize mosaics
        unit = u.MJy / u.sr
        mosaics = []

        for subchannel in subchannels:
            acc = accumulators[subchannel]
            if acc['n_valid'] == 0:
                continue

            # Compute final weighted average
            with np.errstate(divide='ignore', invalid='ignore'):
                mosaic_data = np.where(
                    acc['nhits'] > 0,
                    acc['acc_data'] / acc['nhits'],
                    np.nan
                )

            channel_info = self.channel_def.get_channel_info(array, subchannel)

            mosaics.append(ChannelMosaic(
                data=mosaic_data * unit,
                nhits=acc['nhits'],
                flags=acc['acc_flags'],
                wcs=self.wcs_mosaic,
                channel_info=channel_info,
                propagated_flags=self.config.propagated_flags,
                extra_header=self._extra_header
            ))

        accumulators.clear()
        log_resource_usage(f"Det. {array} mosaics created")

        return mosaics

    def build_channel(
        self,
        l2_files: list[str],
        array: int,
        subchannel: int
    ) -> Optional[ChannelMosaic]:
        """Build mosaic for a single channel.

        Note: For processing multiple subchannels, use build_all() which
        opens each file only once for all subchannels.

        Parameters
        ----------
        l2_files : list of str
            List of L2 file paths for this array.
        array : int
            Detector array number (1-6).
        subchannel : int
            Subchannel number.

        Returns
        -------
        ChannelMosaic or None
            Mosaic result, or None if no valid data found.

        """
        mosaics = self._build_array_channels(l2_files, array, [subchannel])
        return mosaics[0] if mosaics else None

    @staticmethod
    def _get_bounding_box(mask: np.ndarray, padding: int = 1) -> tuple[int, int, int, int]:
        """Get bounding box of True values in a boolean mask.

        Parameters
        ----------
        mask : np.ndarray
            Boolean mask array.
        padding : int
            Padding to add around the bounding box.

        Returns
        -------
        tuple of (y_min, y_max, x_min, x_max)
            Bounding box indices (suitable for slicing as [y_min:y_max, x_min:x_max]).

        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        y_indices = np.where(rows)[0]
        x_indices = np.where(cols)[0]

        y_min = max(0, y_indices[0] - padding)
        y_max = min(mask.shape[0], y_indices[-1] + 1 + padding)
        x_min = max(0, x_indices[0] - padding)
        x_max = min(mask.shape[1], x_indices[-1] + 1 + padding)

        return y_min, y_max, x_min, x_max


def _find_files_by_array(
    base_dir: str,
    pattern: str,
    file_type: str = "files"
) -> dict[int, list[str]]:
    """Find files organized by detector array.

    Parameters
    ----------
    base_dir : str
        Base directory containing files.
    pattern : str
        Glob pattern with {array} placeholder for array number (1-6).
        Example: '**/{array}/*.fits' or '**/{array}/*_D{array}_*.fits'
    file_type : str
        Description for logging (e.g., "L2 files", "wavelength files").

    Returns
    -------
    dict
        Mapping of array number (1-6) to list of file paths.

    """
    files_by_array = {}
    for array in range(1, 7):
        full_pattern = os.path.join(base_dir, pattern.format(array=array))
        files = sorted(glob(full_pattern, recursive=True))
        files_by_array[array] = files
        logging.info(f"Array {array}: Found {len(files)} {file_type}")
    return files_by_array


def _load_wavelength_model_from_files(
    wavelength_files_by_array: dict[int, list[str]]
) -> ArrayWavelengthModel:
    """Load ArrayWavelengthModel from discovered wavelength FITS files.

    Parameters
    ----------
    wavelength_files_by_array : dict
        Mapping of array number to list of wavelength FITS file paths.
        Expects exactly one file per array.

    Returns
    -------
    ArrayWavelengthModel
        Wavelength model loaded from the FITS files.

    Raises
    ------
    ValueError
        If no wavelength file found for an array, or multiple files found.

    Notes
    -----
    If the FITS files contain a CBAND extension (pixel-center bandwidth),
    it will be loaded alongside the wavelength data (CWAVE extension).

    """
    from astropy.io import fits

    wl_arrays = {}
    bw_arrays = {}
    detcoord = {}

    for array_id, files in wavelength_files_by_array.items():
        if not files:
            continue
        if len(files) > 1:
            raise ValueError(
                f"Multiple wavelength files found for array {array_id}: {files}"
            )
        filepath = files[0]
        logging.info(f"Loading wavelength array {array_id} from: {filepath}")
        with fits.open(filepath) as hdul:
            wl_hdu = hdul["CWAVE"]  # pixel center wavelengths
            wl_arrays[array_id] = wl_hdu.data
            detcoord[array_id] = wl_hdu.header.get('DETCOORD', 'device')

            # Load bandwidth if available
            if "CBAND" in hdul:
                bw_arrays[array_id] = hdul["CBAND"].data
                logging.info(f"  Loaded bandwidth (CBAND) for array {array_id}")

    if not wl_arrays:
        raise ValueError("No wavelength files found")

    return ArrayWavelengthModel(
        wl_arrays,
        bandwidth_arrays=bw_arrays if bw_arrays else None,
        detcoord=detcoord
    )


def _find_l2_files(temp_butler_root: str) -> dict[int, list[str]]:
    """Find L2 files in a temp butler, by detector array.

    Parameters
    ----------
    temp_butler_root : str
        Path to the populated temp butler root directory.

    Returns
    -------
    dict
        Mapping of array number (1-6) to list of L2 file paths.

    Raises
    ------
    ValueError
        If no L2 files are found.

    """
    l2_base_dir = temp_butler_root

    # L2 file pattern (array in directory name):
    # level2/.../1/level2_...D1_....fits
    l2_pattern = '**/level2/**/{array}/*.fits'

    logging.info("Finding L2 files...")
    l2_files_by_array = _find_files_by_array(l2_base_dir, l2_pattern, "L2 files")
    total_files = sum(len(f) for f in l2_files_by_array.values())
    if total_files == 0:
        raise ValueError(f"No L2 files found in {l2_base_dir}")
    logging.info(f"Total L2 files found: {total_files}")
    return l2_files_by_array


def _get_channel_definition(
    temp_butler_root: str,
    user_channels: Optional[tuple[float, float]] = None
) -> ChannelDefinition:
    """Load the channel definition for a populated temp butler.

    Without user channels, standard spectral channel definitions are read
    from the temp butler's spectral_channels dataset (falling back to the
    bundled reference definitions). With user channels, a channel
    definition is built from the wavelength calibration files.

    Parameters
    ----------
    temp_butler_root : str
        Path to the populated temp butler root directory.
    user_channels : tuple[float, float], optional
        Wavelength range (min, max) in microns for tophat user channels
        as an alternative to standard channels.

    Returns
    -------
    ChannelDefinition
        Channel definition to build mosaics with.

    """
    if user_channels is None:
        # Use standard spectral channels

        # Find and load standard channel definitions
        logging.info("Getting standard channel definitions...")
        try:
            spectral_channels_files = []
            for pattern in ("*.fits", "*.fits.gz"):
                fits_pattern = os.path.join(temp_butler_root, "**/spectral_channels/**", pattern)
                files = sorted(glob(fits_pattern, recursive=True))
                spectral_channels_files.extend(files)
            logging.info(f"Total spectral_channels files found: {len(spectral_channels_files)}")
            if len(spectral_channels_files) > 0:
                channel_def = LVFContourChannelDefinition.from_fits(
                    spectral_channels_files[0],
                    channels=list(range(1, 18))
                )
        except Exception:
            spectral_channels_files = []
        if len(spectral_channels_files) < 1:
            logging.info("Using reference spectral channel definitions.")
            channel_def = get_reference_spectral_channel_defs()
    else:
        wavelength_fits_dir = temp_butler_root
        # Wavelength file pattern: **/spectral_wcs/.../1/spectral_wcs_D1_....fits
        wl_pattern = '**/spectral_wcs/**/{array}/*_D{array}_*.fits'

        # Find and load wavelength calibration files
        logging.info("Finding wavelength calibration files...")
        wl_files_by_array = _find_files_by_array(
            wavelength_fits_dir, wl_pattern, "wavelength files"
        )
        wl_model = _load_wavelength_model_from_files(wl_files_by_array)
        log_resource_usage()

        # Create channel definition (bundled LVF contour)
        logging.info("Creating channel definition...")
        # TODO: implement custom channel definitions
        channel_def = LVFContourChannelDefinition.from_bundled(wl_model)
        del wl_model  # wavelength model is only needed for spectral channels stats

    return channel_def


def _select_channels(
    channel_def: ChannelDefinition,
    l2_files_by_array: dict[int, list[str]],
    channel_range: Optional[tuple[float, float]] = None
) -> list[tuple[int, int]]:
    """Determine the (array, subchannel) pairs to process.

    Filters to arrays that have L2 files (e.g., when --detector is used)
    and, optionally, to a wavelength range.

    Parameters
    ----------
    channel_def : ChannelDefinition
        Channel definition providing the available channels.
    l2_files_by_array : dict
        Mapping of array number to list of L2 file paths.
    channel_range : tuple[float, float], optional
        Wavelength range (min, max) in microns to filter channels.

    Returns
    -------
    list of (int, int)
        Channels to process as (array, subchannel) pairs.

    """
    available_arrays = set(l2_files_by_array.keys())
    all_channels = [
        (a, s) for a, s in channel_def.get_all_channels()
        if a in available_arrays
    ]
    if channel_range is None:
        return all_channels

    wl_min, wl_max = channel_range
    channels_to_process = []
    for array_id, subchannel in all_channels:
        wl = channel_def.get_channel_info(array_id, subchannel).wavelength.to('um').value
        if wl_min <= wl <= wl_max:
            channels_to_process.append((array_id, subchannel))
    logging.info(
        f"Filtered to {len(channels_to_process)} channels "
        f"in range [{wl_min}, {wl_max}] um"
    )
    return channels_to_process


def _build_and_write_cube(
    builder: MosaicBuilder,
    l2_files_by_array: dict[int, list[str]],
    channels_to_process: list[tuple[int, int]],
    channel_mosaics_dir: str,
    cube_filepath: str,
    compress: bool = False
) -> str:
    """Build all channel mosaics and combine them into an output cube.

    Each completed mosaic is written to `channel_mosaics_dir` and freed
    immediately; the cube is then assembled from the saved files.

    Parameters
    ----------
    builder : MosaicBuilder
        Configured builder (grid and reprojection already set up).
    l2_files_by_array : dict
        Mapping of array number to list of L2 file paths.
    channels_to_process : list of (int, int)
        Channels to process as (array, subchannel) pairs.
    channel_mosaics_dir : str
        Directory for individual channel mosaic files.
    cube_filepath : str
        Output cube file path (parent directories are created).
    compress : bool, optional
        Whether to compress the output cube with gzip. Default is False.

    Returns
    -------
    str
        Path to the output cube file.

    Raises
    ------
    ValueError
        If no mosaics are created.

    """
    n_channels = len(channels_to_process)

    # Setup output directories
    os.makedirs(channel_mosaics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(cube_filepath) or '.', exist_ok=True)

    # Track saved mosaic file paths for cube creation.
    # Paths are appended in output_callback and used after build_all() returns,
    # at which point all build-phase memory (accumulators, reprojection buffers,
    # L2 image data) has already been freed. Only then are the on-disk mosaics
    # loaded back one array at a time to write the cube.
    saved_mosaic_paths = []
    n_processed = 0

    def progress_callback(file_index: int, total_files_count: int, message: str):
        logging.info(f"[{file_index}/{total_files_count}] {message}")

    def output_callback(mosaic: ChannelMosaic):
        """Write each mosaic to disk immediately and release its memory.

        Mosaics are NOT kept in the saved_mosaic_paths list as objects —
        only their file paths are tracked. This prevents all 102 channel
        arrays (data + nhits + flags each) from accumulating in RAM
        alongside the active per-array accumulators and thread buffers.
        """
        nonlocal n_processed
        n_processed += 1

        # Write the completed mosaic to disk immediately so its arrays
        # can be freed before the next array's build phase begins.
        info = mosaic.channel_info
        filename = f'mosaic_array{info.detector}_ch{info.subchannel:02d}.fits'
        output_file = os.path.join(channel_mosaics_dir, filename)
        mosaic.write(output_file)
        saved_mosaic_paths.append(output_file)

        logging.info(
            f"  [{n_processed}/{n_channels}] Array {info.detector}, "
            f"Ch {info.subchannel:02d}: {info.wavelength.to('um'):.4f}, "
            f"{mosaic.valid_fraction * 100:.1f}% valid -> {filename}"
        )

        # Do NOT append the mosaic object itself. Returning here without
        # holding a reference allows build_all()'s `del mosaics; gc.collect()`
        # to actually free the underlying numpy arrays.

    # Build mosaics — each completed mosaic is written to disk and freed
    # before the next array's accumulators are allocated.
    logging.info("Building mosaics...")
    builder.build_all(
        l2_files_by_array,
        channels=channels_to_process,
        progress_callback=progress_callback,
        output_callback=output_callback
    )

    logging.info(f"Individual mosaics saved to: {channel_mosaics_dir}/")

    if not saved_mosaic_paths:
        raise ValueError("No mosaics were created")

    # All build-phase memory is now free. Load the saved channel mosaics
    # from disk sequentially to assemble the output cube. Peak RAM here
    # is bounded by the number of channels × one mosaic footprint, which
    # is far smaller than holding mosaics alongside active accumulators.
    logging.info(f"Writing combined cube with {len(saved_mosaic_paths)} channels...")
    cube_filepath = write_mosaic_cube(saved_mosaic_paths, cube_filepath, compress=compress)
    logging.info(f"Cube saved to: {cube_filepath}")

    return cube_filepath


def build_mosaics_from_temp_butler(
    temp_butler_root: str,
    center: tuple[float, float],
    size: tuple[float, float],
    pixel_scale: float,
    rotation: float,
    output_dir: str,
    channel_mosaics_dir: str,
    channel_range: Optional[tuple[float, float]] = None,
    compress: bool = False,
    user_channels: Optional[tuple[float, float]] = None
) -> str:
    """Build mosaics from a populated temp butler.

    This function orchestrates the complete mosaic building workflow:
    1. Finds L2 files in temp_butler_root/level2
    2. Loads spectral calibration from temp_butler_root/spectral_channels
       or wavelength calibration from temp_butler_root/spectral_wcs
    3. Creates channel definition using bundled LVF contours
    4. Builds mosaics for all (or filtered) channels
    5. Saves individual channel mosaics to channel_mosaics_dir as they complete
    6. Writes combined cube to output_dir by loading saved channel files

    Parameters
    ----------
    temp_butler_root : str
        Path to the populated temp butler root directory.
    center : tuple[float, float]
        Mosaic center as (ra, dec) in degrees.
    size : tuple[float, float]
        Mosaic size as (width, height) in degrees.
    pixel_scale : float
        Output pixel scale in arcseconds.
    rotation : float
        Rotation angle in degrees.
    output_dir : str
        Directory for the output cube file.
    channel_mosaics_dir : str
        Directory for individual channel mosaic files.
    channel_range : tuple[float, float], optional
        Wavelength range (min, max) in microns to filter channels.
        If None, all standard channels are processed.
    compress : bool, optional
        Whether to compress the output cube with gzip. Default is False.
    user_channels : tuple[float, float], optional
        Wavelength range (min, max) in microns to filter channels.
        Used for tophat user channels as an alternative to standard channels.

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

    # Create mosaic configuration
    config = MosaicConfig(
        ra_center=center[0],
        dec_center=center[1],
        width_deg=size[0],
        height_deg=size[1],
        pixel_size_arcsec=pixel_scale,
        rotation_deg=rotation if rotation != 0 else None,
        projection='TAN',
        bad_pixel_flags=BAD_PIX_FLAGS,
        propagated_flags=PROPAGATED_FLAGS
    )

    # Create builder
    builder = MosaicBuilder(channel_def, config)
    logging.info(
        f"Mosaic grid: {builder.shape_mosaic[0]} x {builder.shape_mosaic[1]} pixels"
    )

    cube_filepath = os.path.join(output_dir, 'mosaic_cube.fits')
    return _build_and_write_cube(
        builder=builder,
        l2_files_by_array=l2_files_by_array,
        channels_to_process=channels_to_process,
        channel_mosaics_dir=channel_mosaics_dir,
        cube_filepath=cube_filepath,
        compress=compress
    )


def build_interpolated_mosaic(
    temp_butler_root: str,
    center: tuple[float, float],
    size: tuple[float, float],
    pixel_scale: float,
    rotation: float,
    output_dir: str,
    target_wavelength: float,
    n_interp: float = 2.5,
    n_req: float = 0.55,
    n_subdivisions: int = 4,
    compress: bool = False,
) -> str:
    """Build a single mosaic at target wavelength using spectral interpolation.

    This function reduces stripe artifacts in mosaics by interpolating across
    neighboring spectral channels. Stripe artifacts occur when extended sources
    with strong spectral features are observed at different roll angles, causing
    the LVF wavelength-position mapping to create inconsistent sampling patterns.

    The interpolation:
    1. Determines which channels are needed based on target wavelength and n_interp
    2. Optionally subdivides channels for finer spectral sampling
    3. Builds mosaics for those channels
    4. Interpolates per-pixel to produce a smooth mosaic at target wavelength

    This preserves the emission signal while averaging out the stripe artifacts.

    Parameters
    ----------
    temp_butler_root : str
        Path to the populated temp butler root directory.
    center : tuple[float, float]
        Mosaic center as (ra, dec) in degrees.
    size : tuple[float, float]
        Mosaic size as (width, height) in degrees.
    pixel_scale : float
        Output pixel scale in arcseconds.
    rotation : float
        Rotation angle in degrees.
    output_dir : str
        Directory for the output file.
    target_wavelength : float
        Target wavelength in microns.
    n_interp : float, optional
        Interpolation range in resolution widths. Channels within
        ±n_interp×Δλ of target are used. Default 2.5.
    n_req : float, optional
        Required contribution distance in resolution widths. At least
        one channel must be within this distance of target. Default 0.55.
    n_subdivisions : int, optional
        Number of subdivisions per standard channel (1-7). Default 4.
        When > 1, creates finer spectral sampling by dividing
        each standard channel into n_subdivisions subchannels. Standard
        channels whose wavelength range overlaps the interpolation range
        are subdivided. Requires wavelength calibration files in
        temp_butler_root/spectral_wcs.
    compress : bool, optional
        Whether to compress the output cube with gzip. Default is False.

    Returns
    -------
    str
        Path to output FITS file.

    Raises
    ------
    ValueError
        If no L2 files are found, insufficient channels for interpolation,
        or wavelength is outside SPHEREx range.

    Examples
    --------
    >>> # Create interpolated mosaic at 4.80 μm with standard channels
    >>> output = build_interpolated_mosaic(
    ...     temp_butler_root='/path/to/butler',
    ...     center=(314.5, 44.0),
    ...     size=(3.5, 3.5),
    ...     pixel_scale=6.15,
    ...     rotation=0,
    ...     output_dir='/path/to/output',
    ...     target_wavelength=4.80,
    ...     n_interp=2.5,
    ...     n_req=0.55,
    ... )

    >>> # With subdivided channels (4x finer spectral sampling)
    >>> output = build_interpolated_mosaic(
    ...     temp_butler_root='/path/to/butler',
    ...     center=(314.5, 44.0),
    ...     size=(3.5, 3.5),
    ...     pixel_scale=6.15,
    ...     rotation=0,
    ...     output_dir='/path/to/output',
    ...     target_wavelength=4.80,
    ...     n_interp=2.5,
    ...     n_req=0.55,
    ...     n_subdivisions=4,
    ... )

    """
    from astropy.io import fits as astropy_fits

    log_resource_usage()

    # Determine array from wavelength
    array_id = wavelength_to_array(target_wavelength)
    logging.info(f"Target wavelength {target_wavelength} μm -> Array {array_id}")

    # Setup paths
    l2_base_dir = temp_butler_root
    l2_pattern = '**/level2/**/{array}/*.fits'

    # Find L2 files
    logging.info("Finding L2 files...")
    l2_files_by_array = _find_files_by_array(l2_base_dir, l2_pattern, "L2 files")
    total_files = sum(len(f) for f in l2_files_by_array.values())
    if total_files == 0:
        raise ValueError(f"No L2 files found in {l2_base_dir}")
    logging.info(f"Total L2 files found: {total_files}")

    # Get standard channel definitions (always needed to determine channel range)
    logging.info("Getting standard channel definitions...")
    try:
        spectral_channels_files = []
        for pattern in ("*.fits", "*.fits.gz"):
            fits_pattern = os.path.join(temp_butler_root, "spectral_channels/**", pattern)
            files = sorted(glob(fits_pattern, recursive=True))
            spectral_channels_files.extend(files)
        if len(spectral_channels_files) > 0:
            std_channel_def = LVFContourChannelDefinition.from_fits(
                spectral_channels_files[0],
                channels=list(range(1, 18))
            )
    except Exception:
        spectral_channels_files = []
    if len(spectral_channels_files) < 1:
        logging.info("Using reference spectral channel definitions.")
        std_channel_def = get_reference_spectral_channel_defs()

    # Get all channels for the target array
    all_channels = std_channel_def.get_all_channels()
    target_array_channels = [(a, s) for a, s in all_channels if a == array_id]

    if not target_array_channels:
        raise ValueError(f"No channels found for array {array_id}")

    # Determine which channels to use and calculate interpolation range
    if n_subdivisions > 1:
        # For subdivided channels:
        # 1. First create subdivided channels for all standard channels in target array
        # 2. Use nearest subchannel to determine R and interpolation range
        # 3. Filter subchannels to those within range

        # Load wavelength model for computing subchannel statistics
        wavelength_fits_dir = os.path.join(temp_butler_root, 'spectral_wcs')
        wl_pattern = '**/{array}/*_D{array}_*.fits'
        logging.info("Loading wavelength calibration for subchannel statistics...")
        wl_files_by_array = _find_files_by_array(
            wavelength_fits_dir, wl_pattern, "wavelength files"
        )
        # Only need the target array
        wl_files_filtered = {array_id: wl_files_by_array.get(array_id, [])}
        if not wl_files_filtered[array_id]:
            raise ValueError(
                f"No wavelength calibration files found for array {array_id} in "
                f"{wavelength_fits_dir}. Wavelength files are required for "
                f"n_subdivisions > 1."
            )
        wl_model = _load_wavelength_model_from_files(wl_files_filtered)

        # Subdivide all standard channels for the target array
        std_channels_to_include = sorted([ch[1] for ch in target_array_channels])
        logging.info(
            f"Subdividing standard channels {std_channels_to_include} "
            f"into {n_subdivisions} subchannels each"
        )

        # Load bundled contour map and create subdivided version
        logging.info("Creating subdivided contour map...")
        with astropy_fits.open(get_lvf_contour_path()) as hdul:
            std_contour_map = hdul[0].data

        subdivided_contour = LVFContourChannelDefinition.subdivide_contour_map(
            std_contour_map,
            channels=std_channels_to_include,
            n_subdivisions=n_subdivisions
        )

        # Create channel definition with subdivided contour and wavelength model
        n_total_subchannels = len(std_channels_to_include) * n_subdivisions
        channel_def = LVFContourChannelDefinition(
            subdivided_contour,
            wavelength_model=wl_model,
            arrays=[array_id],
            channels=list(range(1, n_total_subchannels + 1))
        )
        logging.info(f"Created {n_total_subchannels} subchannels")

        # Find subchannel nearest to target wavelength to get R
        all_subchannels = channel_def.get_all_channels()

        def get_subchannel_wl(ch):
            info = channel_def.get_channel_info(*ch)
            wl = info.wavelength.to(u.um).value
            return wl if not np.isnan(wl) else float('inf')

        nearest_subch = min(
            all_subchannels,
            key=lambda ch: abs(get_subchannel_wl(ch) - target_wavelength)
        )
        nearest_info = channel_def.get_channel_info(*nearest_subch)
        R = nearest_info.R
        if R is None or np.isnan(R):
            # Fall back to standard channel R if subchannel R not available
            def get_std_wl(ch):
                return std_channel_def.get_channel_info(*ch).wavelength.to(u.um).value
            nearest_std = min(
                target_array_channels,
                key=lambda ch: abs(get_std_wl(ch) - target_wavelength)
            )
            R = std_channel_def.get_channel_info(*nearest_std).R
            logging.warning(
                f"Subchannel R not available, using standard channel R = {R:.1f}"
            )

        # Calculate interpolation range using subchannel R
        delta_wl = target_wavelength / R
        wl_min = target_wavelength - n_interp * delta_wl
        wl_max = target_wavelength + n_interp * delta_wl

        logging.info(f"Interpolation range: [{wl_min:.4f}, {wl_max:.4f}] μm")
        logging.info(f"Using {n_interp} resolution widths, R = {R:.1f}, Δλ = {delta_wl:.4f} μm")

        # Filter subchannels to interpolation range
        channels_to_process = []
        for array, subchannel in all_subchannels:
            wl = get_subchannel_wl((array, subchannel))
            if not np.isinf(wl) and wl_min <= wl <= wl_max:
                channels_to_process.append((array, subchannel))
    else:
        # Use standard channels without subdivision
        channel_def = std_channel_def

        # Find channel nearest to target wavelength to get representative R
        def get_channel_wl(ch):
            return std_channel_def.get_channel_info(*ch).wavelength.to(u.um).value

        nearest_ch = min(
            target_array_channels,
            key=lambda ch: abs(get_channel_wl(ch) - target_wavelength)
        )
        R = std_channel_def.get_channel_info(*nearest_ch).R

        delta_wl = target_wavelength / R
        wl_min = target_wavelength - n_interp * delta_wl
        wl_max = target_wavelength + n_interp * delta_wl

        logging.info(f"Interpolation range: [{wl_min:.4f}, {wl_max:.4f}] μm")
        logging.info(f"Using {n_interp} resolution widths, R = {R:.1f}, Δλ = {delta_wl:.4f} μm")

        # Filter standard channels to interpolation range
        channels_to_process = []
        for array, subchannel in target_array_channels:
            wl = get_channel_wl((array, subchannel))
            if wl_min <= wl <= wl_max:
                channels_to_process.append((array, subchannel))

    n_channels = len(channels_to_process)
    if n_channels < 2:
        raise ValueError(
            f"Insufficient channels for interpolation at {target_wavelength} μm. "
            f"Found {n_channels} channels in range [{wl_min:.4f}, {wl_max:.4f}] μm, need at least 2."
        )
    logging.info(f"Building {n_channels} channel mosaics for interpolation...")

    # Create mosaic configuration
    config = MosaicConfig(
        ra_center=center[0],
        dec_center=center[1],
        width_deg=size[0],
        height_deg=size[1],
        pixel_size_arcsec=pixel_scale,
        rotation_deg=rotation if rotation != 0 else None,
        projection='TAN',
        bad_pixel_flags=BAD_PIX_FLAGS,
        propagated_flags=PROPAGATED_FLAGS
    )

    # Create builder
    builder = MosaicBuilder(channel_def, config)
    logging.info(
        f"Mosaic grid: {builder.shape_mosaic[0]} x {builder.shape_mosaic[1]} pixels"
    )

    # Build channel mosaics
    def progress_callback(file_index: int, total_files_count: int, message: str):
        logging.info(f"[{file_index}/{total_files_count}] {message}")

    # Filter l2_files_by_array to only include target array
    l2_files_filtered = {array_id: l2_files_by_array.get(array_id, [])}

    channel_mosaics = {}
    for mosaic in builder.build_all(
        l2_files_filtered,
        channels=channels_to_process,
        progress_callback=progress_callback
    ):
        ch_id = (mosaic.channel_info.detector, mosaic.channel_info.subchannel)
        channel_mosaics[ch_id] = mosaic
        wl = mosaic.channel_info.wavelength.to(u.um).value
        logging.info(f"  Built mosaic for channel {ch_id}: λ={wl:.4f} μm")

    if len(channel_mosaics) < 2:
        raise ValueError(
            f"Insufficient channel mosaics built. Got {len(channel_mosaics)}, need at least 2."
        )

    # Perform interpolation
    logging.info(f"Interpolating to target wavelength {target_wavelength} μm...")
    interp_config = InterpConfig(n_interp=n_interp, n_req=n_req)
    interpolator = SpectralInterpolator(channel_mosaics, interp_config)

    result_mosaic = interpolator.interpolate(target_wavelength, array_id)

    # Save all mosaics (contributing channels + interpolated) and create cube
    os.makedirs(output_dir, exist_ok=True)

    # Save individual channel mosaics
    mosaic_paths = []
    for ch_id, mosaic in channel_mosaics.items():
        array, subchan = ch_id
        if n_subdivisions > 1:
            filename = f'mosaic_array{array}_subch{subchan:03d}.fits'
        else:
            filename = f'mosaic_array{array}_ch{subchan:02d}.fits'
        filepath = os.path.join(output_dir, filename)
        mosaic.write(filepath)
        mosaic_paths.append(filepath)
        logging.debug(f"Saved channel mosaic: {filename}")

    # Save interpolated mosaic
    interp_filename = f'mosaic_interp_{target_wavelength:.3f}um.fits'
    interp_filepath = os.path.join(output_dir, interp_filename)
    result_mosaic.write(interp_filepath)
    mosaic_paths.append(interp_filepath)
    logging.info(f"Interpolated mosaic saved to: {interp_filepath}")

    # Log statistics
    valid_fraction = result_mosaic.valid_fraction * 100
    logging.info(f"Valid pixels: {valid_fraction:.1f}%")

    # Create combined cube with all channel mosaics + interpolated result
    cube_filename = f'mosaic_interp_cube_{target_wavelength:.3f}um.fits'
    cube_filepath = os.path.join(output_dir, cube_filename)
    logging.info(f"Writing combined cube with {len(mosaic_paths)} planes...")
    cube_filepath = write_mosaic_cube(mosaic_paths, cube_filepath, compress=compress)
    logging.info(f"Combined cube saved to: {cube_filepath}")

    log_resource_usage("Interpolated mosaic complete")

    return cube_filepath
