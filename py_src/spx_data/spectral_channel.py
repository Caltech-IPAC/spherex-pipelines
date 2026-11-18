"""Spectral channel definitions for SPHEREx detector arrays.

This module provides abstract and concrete implementations of spectral channel
definitions that map detector pixels to discrete wavelength channels.
"""

__all__ = [
    "ChannelDefinition",
    "ChannelInfo",
    "LogSpacedChannelDefinition",
    "LinSpacedChannelDefinition",
    "LVFContourChannelDefinition",
    "LVFChannelDefinitionV2",
    "get_lvf_contour_path",
]

import copy
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterator, Optional, Self

import astropy.units as u
import numba
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.time import Time

from .wavelength_model import WavelengthModel


@dataclass
class ChannelInfo:
    """Spectral statistics for a single channel.

    Attributes
    ----------
    detector : int
        Detector array number (1-6).
    subchannel : int
        Subchannel number within the array.
    wavelength : astropy.units.Quantity
        Effective (median) wavelength with units.
    wavelength_min : astropy.units.Quantity, optional
        Minimum per-pixel wavelength with units.
    wavelength_max : astropy.units.Quantity, optional
        Maximum per-pixel wavelength with units.
    R : float, optional
        Median resolving power R = λ/Δλ.
    R_std : float, optional
        Standard deviation of per-pixel resolving power.
    bandwidth : astropy.units.Quantity, optional
        Pre-calculated effective bandwidth. If not provided,
        use `effective_bandwidth` property to compute from wavelength/R.

    Examples
    --------
    >>> info = channel_def.get_channel_info(array_id=1, channel=8)
    >>> print(f"Wavelength: {info.wavelength:.3f}")
    >>> print(f"Bandwidth: {info.effective_bandwidth:.4f}")

    """

    detector: int
    subchannel: int
    wavelength: u.Quantity
    wavelength_min: u.Quantity = None
    wavelength_max: u.Quantity = None
    R: Optional[float] = None
    R_std: Optional[float] = None
    bandwidth: Optional[u.Quantity] = None

    @property
    def effective_bandwidth(self) -> Optional[u.Quantity]:
        """Return bandwidth if available, otherwise compute as wavelength/R."""
        if self.bandwidth is not None:
            return self.bandwidth
        if self.R is not None and self.R > 0:
            return self.wavelength / self.R
        return None


def get_lvf_contour_path() -> str:
    """Get path to the bundled LVF contour FITS file.

    Returns
    -------
    path : str
        Path to LVF_contour_17Channels.fits.gz

    Examples
    --------
    >>> path = get_lvf_contour_path()

    """
    fname = "LVF_contour_17Channels.fits.gz"
    return str(files("spx_data.data").joinpath(fname))


class ChannelDefinition(ABC):
    """Abstract base class for spectral channel definitions.

    A channel definition maps detector pixels to spectral channels.
    Subclasses implement different strategies for this mapping:

    - `LogSpacedChannelDefinition`: Log-spaced wavelength bins (default N=17)
    - `LinSpacedChannelDefinition`: Linear-spaced wavelength bins (arbitrary N)
    - `LVFContourChannelDefinition`: Pre-defined standard 17-channel from FITS
    """

    @abstractmethod
    def get_all_channels(self) -> list[tuple[int, int]]:
        """Get list of all (array_id, channel) tuples.

        Returns
        -------
        channels : list[tuple[int, int]]
            List of (array_id, channel) tuples defining all channels.
            For 6 arrays with 17 channels each, this returns 102 tuples.

        Examples
        --------
        >>> channels = channel_def.get_all_channels()
        >>> print(channels[:3])
        [(1, 0), (1, 1), (1, 2)]

        """
        pass

    @abstractmethod
    def get_channel_mask(
        self,
        array_id: int,
        channel: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get boolean mask for pixels belonging to a channel.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).
        channel : int
            Channel index within the array.
        shape : tuple[int, int]
            Shape of the data array as (ny, nx).

        Returns
        -------
        mask : numpy.ndarray
            Boolean mask with shape `shape`. True where pixel belongs
            to the specified channel.

        Notes
        -----
        The shape parameter allows handling data that may have different
        sizes (e.g., if blind pixels were removed from L2 data).

        """
        pass

    @abstractmethod
    def get_channel_info(
        self,
        array_id: int,
        channel: int,
    ) -> ChannelInfo:
        """Get all spectral statistics for a channel.

        This method returns a ChannelInfo dataclass containing all spectral
        metadata for the specified channel. It is the preferred way to get
        channel information as it returns all statistics in a single call.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).
        channel : int
            Channel index within the array.

        Returns
        -------
        info : ChannelInfo
            Dataclass containing wavelength, wavelength range, resolving power,
            and bandwidth information.

        Examples
        --------
        >>> info = channel_def.get_channel_info(array_id=1, channel=8)
        >>> print(f"Wavelength: {info.wavelength:.3f}")
        >>> print(f"Bandwidth: {info.effective_bandwidth:.4f}")

        """
        pass

    def get_n_channels(self) -> int:
        """Return total number of channels across all arrays."""
        return len(self.get_all_channels())

    def get_channels_for_array(self, array_id: int) -> list[tuple[int, int]]:
        """Get all channels for a specific array."""
        return [(a, c) for a, c in self.get_all_channels() if a == array_id]

    def iter_channels(
        self,
        array_id: int,
        shape: tuple[int, int],
        channels: Optional[list[int]] = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over (channel, mask) pairs for an array.

        This provides an iterator interface similar to RegionMaskIterable
        for backward compatibility.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).
        shape : tuple[int, int]
            Shape of the data array as (ny, nx).
        channels : list[int], optional
            Specific channels to iterate. If None, iterates all channels.

        Yields
        ------
        channel_id : int
            Channel index.
        mask : numpy.ndarray
            Boolean mask for pixels in this channel.

        Examples
        --------
        >>> for channel_id, mask in channel_def.iter_channels(1, (2040, 2040)):
        ...     data_in_channel = image[mask]

        """
        array_channels = [c for a, c in self.get_all_channels() if a == array_id]
        if channels is not None:
            array_channels = [c for c in array_channels if c in channels]

        for channel in array_channels:
            mask = self.get_channel_mask(array_id, channel, shape)
            yield channel, mask


class LogSpacedChannelDefinition(ChannelDefinition):
    """Channel definition using log-spaced wavelength bins.

    Each array's wavelength range is divided into `n_channels` bins with
    equal width in log(wavelength) space. This is the standard SPHEREx
    channel definition.

    Parameters
    ----------
    wavelength_model : WavelengthModel
        Model providing per-pixel wavelengths for each array.
    n_channels : int, optional
        Number of channels per array. Default is 17.
    arrays : list[int], optional
        Array numbers to include. Default is [1, 2, 3, 4, 5, 6].

    Examples
    --------
    >>> from spx_data import TabularWavelengthModel, LogSpacedChannelDefinition
    >>> wl_model = TabularWavelengthModel()
    >>> channel_def = LogSpacedChannelDefinition(wl_model, n_channels=17)
    >>> print(channel_def.get_n_channels())
    102
    >>> mask = channel_def.get_channel_mask(array_id=1, channel=8, shape=(2040, 2040))

    """

    def __init__(
        self,
        wavelength_model: WavelengthModel,
        n_channels: int = 17,
        arrays: Optional[list[int]] = None,
    ):
        self.wavelength_model = wavelength_model
        self.n_channels = n_channels
        self.arrays = arrays if arrays is not None else [1, 2, 3, 4, 5, 6]
        self._edge_cache: dict[int, np.ndarray] = {}
        self._cropped_wl_cache: dict[tuple[int, tuple[int, int]], np.ndarray] = {}

    def get_all_channels(self) -> list[tuple[int, int]]:
        """Return all (array_id, channel) combinations."""
        return [
            (array_id, channel)
            for array_id in self.arrays
            for channel in range(self.n_channels)
        ]

    def _get_wavelength_edges(self, array_id: int) -> np.ndarray:
        """Get log-spaced wavelength bin edges for an array.

        Returns array of n_channels + 1 edge values in microns.
        """
        if array_id not in self._edge_cache:
            wl_min, wl_max = self.wavelength_model.get_wavelength_range(array_id)

            # Small tolerance to ensure edge pixels are included
            tol = 1e-5
            wl_min_adj = (1 - tol) * wl_min
            wl_max_adj = (1 + tol) * wl_max

            edges = np.logspace(
                np.log10(wl_min_adj),
                np.log10(wl_max_adj),
                num=self.n_channels + 1,
            )
            self._edge_cache[array_id] = edges

        return self._edge_cache[array_id]

    def _get_wavelength_array_for_shape(
        self,
        array_id: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get wavelength array for the given shape, with caching."""
        cache_key = (array_id, shape)
        if cache_key not in self._cropped_wl_cache:
            wl_array = self.wavelength_model.get_wavelength_array(array_id)
            if wl_array.shape != shape:
                wl_array = _crop_to_shape(wl_array, shape)
            self._cropped_wl_cache[cache_key] = wl_array
        return self._cropped_wl_cache[cache_key]

    def get_channel_mask(
        self,
        array_id: int,
        channel: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get boolean mask for pixels in the specified channel."""
        wl_array = self._get_wavelength_array_for_shape(array_id, shape)
        edges = self._get_wavelength_edges(array_id)

        wl_min = edges[channel]
        wl_max = edges[channel + 1]

        return (wl_array >= wl_min) & (wl_array < wl_max)

    def get_channel_info(
        self,
        array_id: int,
        channel: int,
    ) -> ChannelInfo:
        """Get all spectral statistics for a channel."""
        wl_array = self.wavelength_model.get_wavelength_array(array_id)
        edges = self._get_wavelength_edges(array_id)

        edge_min = edges[channel]
        edge_max = edges[channel + 1]

        mask = (wl_array >= edge_min) & (wl_array < edge_max)

        # Wavelength: median of pixels in channel, or geometric mean of edges
        if np.any(mask):
            wavelength = np.median(wl_array[mask]) * u.um
        else:
            wavelength = np.sqrt(edge_min * edge_max) * u.um

        # Wavelength range from bin edges
        wl_min = edge_min * u.um
        wl_max = edge_max * u.um

        # Resolving power from per-pixel bandwidth if available
        R = None
        R_std = None
        if self.wavelength_model.has_bandwidth() and np.any(mask):
            bw_array = self.wavelength_model.get_bandwidth_array(array_id)
            if bw_array is not None:
                wl_values = wl_array[mask]
                bw_values = bw_array[mask]
                r_values = wl_values / bw_values
                R = float(np.median(r_values))
                R_std = float(np.std(r_values))

        return ChannelInfo(
            detector=array_id,
            subchannel=channel,
            wavelength=wavelength,
            wavelength_min=wl_min,
            wavelength_max=wl_max,
            R=R,
            R_std=R_std,
            bandwidth=None,  # Computed from wavelength/R via effective_bandwidth
        )


class LinSpacedChannelDefinition(ChannelDefinition):
    """Channel definition using linear-spaced wavelength bins.

    Each array's wavelength range is divided into `n_channels` bins with
    equal width in linear wavelength space. This is useful for calibration
    and analysis tasks that require finer wavelength resolution.

    Parameters
    ----------
    wavelength_model : WavelengthModel
        Model providing per-pixel wavelengths for each array.
    n_channels : int
        Number of channels per array (e.g., 64 or 256).
    arrays : list[int], optional
        Array numbers to include. Default is [1, 2, 3, 4, 5, 6].
    channel_list : list[int], optional
        Specific channels to include. If None, all channels are included.

    Examples
    --------
    >>> from spx_data import TabularWavelengthModel, LinSpacedChannelDefinition
    >>> wl_model = TabularWavelengthModel()
    >>> # 64 linear bins for gain matrix estimation
    >>> channel_def = LinSpacedChannelDefinition(wl_model, n_channels=64)
    >>> for channel, mask in channel_def.iter_channels(1, (2040, 2040)):
    ...     print(f"Channel {channel}: {mask.sum()} pixels")

    """

    def __init__(
        self,
        wavelength_model: WavelengthModel,
        n_channels: int,
        arrays: Optional[list[int]] = None,
        channel_list: Optional[list[int]] = None,
    ):
        self.wavelength_model = wavelength_model
        self.n_channels = n_channels
        self.arrays = arrays if arrays is not None else [1, 2, 3, 4, 5, 6]
        self.channel_list = set(channel_list) if channel_list else None
        self._edge_cache: dict[int, np.ndarray] = {}
        self._cropped_wl_cache: dict[tuple[int, tuple[int, int]], np.ndarray] = {}

    def get_all_channels(self) -> list[tuple[int, int]]:
        """Return all (array_id, channel) combinations."""
        channels = []
        for array_id in self.arrays:
            for channel in range(self.n_channels):
                if self.channel_list is None or channel in self.channel_list:
                    channels.append((array_id, channel))
        return channels

    def _get_wavelength_edges(self, array_id: int) -> np.ndarray:
        """Get linear-spaced wavelength bin edges for an array.

        Returns array of n_channels + 1 edge values in microns.
        """
        if array_id not in self._edge_cache:
            wl_min, wl_max = self.wavelength_model.get_wavelength_range(array_id)

            # Small tolerance to ensure edge pixels are included
            tol = 1e-5
            wl_min_adj = (1 - tol) * wl_min
            wl_max_adj = (1 + tol) * wl_max

            edges = np.linspace(wl_min_adj, wl_max_adj, num=self.n_channels + 1)
            self._edge_cache[array_id] = edges

        return self._edge_cache[array_id]

    def _get_wavelength_array_for_shape(
        self,
        array_id: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get wavelength array for the given shape, with caching."""
        cache_key = (array_id, shape)
        if cache_key not in self._cropped_wl_cache:
            wl_array = self.wavelength_model.get_wavelength_array(array_id)
            if wl_array.shape != shape:
                wl_array = _crop_to_shape(wl_array, shape)
            self._cropped_wl_cache[cache_key] = wl_array
        return self._cropped_wl_cache[cache_key]

    def get_channel_mask(
        self,
        array_id: int,
        channel: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get boolean mask for pixels in the specified channel."""
        wl_array = self._get_wavelength_array_for_shape(array_id, shape)
        edges = self._get_wavelength_edges(array_id)

        wl_min = edges[channel]
        wl_max = edges[channel + 1]

        return (wl_array >= wl_min) & (wl_array < wl_max)

    def get_channel_info(
        self,
        array_id: int,
        channel: int,
    ) -> ChannelInfo:
        """Get all spectral statistics for a channel."""
        wl_array = self.wavelength_model.get_wavelength_array(array_id)
        edges = self._get_wavelength_edges(array_id)

        edge_min = edges[channel]
        edge_max = edges[channel + 1]

        mask = (wl_array >= edge_min) & (wl_array < edge_max)

        # Wavelength: median of pixels in channel, or midpoint of edges
        if np.any(mask):
            wavelength = np.median(wl_array[mask]) * u.um
        else:
            wavelength = (edge_min + edge_max) / 2 * u.um

        # Wavelength range from bin edges
        wl_min = edge_min * u.um
        wl_max = edge_max * u.um

        # Resolving power from per-pixel bandwidth if available
        R = None
        R_std = None
        if self.wavelength_model.has_bandwidth() and np.any(mask):
            bw_array = self.wavelength_model.get_bandwidth_array(array_id)
            if bw_array is not None:
                wl_values = wl_array[mask]
                bw_values = bw_array[mask]
                r_values = wl_values / bw_values
                R = float(np.median(r_values))
                R_std = float(np.std(r_values))

        return ChannelInfo(
            detector=array_id,
            subchannel=channel,
            wavelength=wavelength,
            wavelength_min=wl_min,
            wavelength_max=wl_max,
            R=R,
            R_std=R_std,
            bandwidth=None,  # Computed from wavelength/R via effective_bandwidth
        )


class LVFContourChannelDefinition(ChannelDefinition):
    """Channel definition using LVF contour maps from FITS files.

    Uses pre-defined contour boundaries that follow the LVF geometry.
    The contour map defines which pixels belong to which channel:

    - Positive values: channel number (channel index)
    - Zero: pixel does not belong to any channel (ignored)

    The same contour map is used for all arrays. The array_id parameter is
    only used for wavelength calculations (each array has different wavelengths).

    Spectral statistics (wavelength, resolving power) can be either:
    - Computed from a wavelength model during initialization
    - Loaded from a pre-computed SPECTRAL_CHANNELS table

    Parameters
    ----------
    contour_map : numpy.ndarray
        2D contour array (2040x2040). Pixel values: positive int = channel
        number, 0 = no channel.
    wavelength_model : WavelengthModel, optional
        Model providing per-pixel wavelengths for computing channel statistics.
        Required if spectral_stats is not provided.
    arrays : list[int], optional
        Array numbers to include. Default: [1, 2, 3, 4, 5, 6].
    spectral_stats : astropy.table.Table, optional
        Pre-computed spectral statistics table with columns: DETECTOR, SUBCHAN,
        WAVELENGTH, WL_MIN, WL_MAX, R, R_STD, and optionally BANDWIDTH.
        If provided, wavelength_model is not needed and channels are determined
        from the table.
    channels : list[int], optional
        Channel numbers to include. Only used when spectral_stats is not
        provided. If None, all channels found in contour_map are included.

    Examples
    --------
    Load from spectral_channels calibration file (contains SPECTRAL_CHANNELS
    table with pre-computed statistics, no wavelength model needed):

    >>> from spx_data import LVFContourChannelDefinition
    >>> channel_def = LVFContourChannelDefinition.from_fits(
    ...     'spectral_channels.fits'
    ... )
    >>> mask = channel_def.get_channel_mask(array_id=1, channel=5, shape=(2040, 2040))
    >>> info = channel_def.get_channel_info(array_id=1, channel=5)

    Load from bundled template (requires wavelength model):

    >>> from spx_data import ArrayWavelengthModel, LVFContourChannelDefinition
    >>> wl_model = ArrayWavelengthModel.from_fits('wavelength_{array_id}.fits')
    >>> channel_def = LVFContourChannelDefinition.from_bundled(wl_model)

    """

    def __init__(
        self,
        contour_map: np.ndarray,
        wavelength_model: Optional[WavelengthModel] = None,
        arrays: Optional[list[int]] = None,
        spectral_stats: Optional[Table] = None,
        channels: Optional[list[int]] = None,
    ):
        if not isinstance(contour_map, np.ndarray):
            contour_map = np.asarray(contour_map)
        if contour_map.ndim != 2:
            raise ValueError(
                f"Contour map must be 2D, got shape {contour_map.shape}"
            )

        self._contour_map = contour_map
        self.arrays = arrays if arrays is not None else [1, 2, 3, 4, 5, 6]

        # Get spectral statistics either from pre-computed table or by computing
        if spectral_stats is not None:
            self._spectral_stats = spectral_stats
            # Extract unique channel numbers from stats table
            self._channels = sorted(set(int(v) for v in spectral_stats['SUBCHAN']))
        elif wavelength_model is not None:
            # Extract unique channel numbers from contour map (positive values only)
            unique_values = np.unique(contour_map)
            all_channels = sorted([int(v) for v in unique_values if v > 0])
            # Filter by channels parameter if provided
            if channels is not None:
                channel_set = set(channels)
                self._channels = [c for c in all_channels if c in channel_set]
            else:
                self._channels = all_channels
            self._spectral_stats = self._compute_stats_table(
                contour_map, wavelength_model
            )
        else:
            # Extract unique channel numbers from contour map (positive values only)
            unique_values = np.unique(contour_map)
            all_channels = sorted([int(v) for v in unique_values if v > 0])
            # Filter by channels parameter if provided
            if channels is not None:
                channel_set = set(channels)
                self._channels = [c for c in all_channels if c in channel_set]
            else:
                self._channels = all_channels
            self._spectral_stats = None

        # Build index for fast lookup: (array_id, channel) -> row index
        self._stats_index: dict[tuple[int, int], int] = {}
        if self._spectral_stats is not None:
            for i, row in enumerate(self._spectral_stats):
                self._stats_index[(int(row['DETECTOR']), int(row['SUBCHAN']))] = i

        # Cache for cropped contour maps
        self._cropped_contour_cache: dict[tuple[int, int], np.ndarray] = {}

    @classmethod
    def from_fits(
        cls,
        filepath: str,
        wavelength_model: Optional[WavelengthModel] = None,
        arrays: Optional[list[int]] = None,
        channels: Optional[list[int]] = None,
    ) -> 'LVFContourChannelDefinition':
        """Create channel definition from a FITS file.

        Supports two FITS file formats:

        1. Bundled format: Image data in primary HDU (HDU 0)
        2. Calibration format: Primary HDU may be empty, image data in first
           IMAGE extension, with SPECTRAL_CHANNELS binary table.

        If the FITS file contains a SPECTRAL_CHANNELS extension, pre-computed
        statistics from that table are used and wavelength_model is ignored.

        Parameters
        ----------
        filepath : str
            Path to the LVF contour FITS file.
        wavelength_model : WavelengthModel, optional
            Model providing per-pixel wavelengths. Only used if the FITS file
            does not contain a SPECTRAL_CHANNELS extension.
        arrays : list[int], optional
            Arrays to include. Default: [1, 2, 3, 4, 5, 6].
        channels : list[int], optional
            Channel numbers to include. Only used when spectral_stats is not
            available. If None, all channels found in contour_map are included.

        Returns
        -------
        channel_def : LVFContourChannelDefinition
            Channel definition with contour map loaded from FITS file.

        Raises
        ------
        ValueError
            If no image data found in FITS file, or if neither wavelength_model
            nor SPECTRAL_CHANNELS extension is available.

        """
        with fits.open(filepath) as hdul:
            # Find first HDU with image data
            contour_map = None
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    contour_map = hdu.data
                    break

            if contour_map is None:
                raise ValueError(
                    f"No 2D image data found in FITS file: {filepath}"
                )

            # Check for pre-computed stats - takes precedence over wavelength_model
            spectral_stats = None
            if 'SPECTRAL_CHANNELS' in hdul:
                spectral_stats = Table(hdul['SPECTRAL_CHANNELS'].data)

        if spectral_stats is not None:
            return cls(contour_map, arrays=arrays, spectral_stats=spectral_stats)
        elif wavelength_model is not None:
            return cls(
                contour_map, wavelength_model, arrays=arrays, channels=channels
            )
        else:
            raise ValueError(
                "Either wavelength_model must be provided or FITS file "
                "must contain SPECTRAL_CHANNELS extension"
            )

    @classmethod
    def from_bundled(
        cls,
        wavelength_model: Optional[WavelengthModel] = None,
        arrays: Optional[list[int]] = None,
        channels: Optional[list[int]] = None,
    ) -> 'LVFContourChannelDefinition':
        """Create channel definition using the bundled LVF contour file.

        This uses the default 17-channel LVF contour map included with
        the spx_data package.

        If the bundled file contains a SPECTRAL_CHANNELS extension and no
        wavelength_model is provided, uses pre-computed statistics.

        Parameters
        ----------
        wavelength_model : WavelengthModel, optional
            Model providing per-pixel wavelengths. If provided, statistics
            are computed from the model. If not provided, the bundled file
            must contain a SPECTRAL_CHANNELS extension.
        arrays : list[int], optional
            Arrays to include. Default: [1, 2, 3, 4, 5, 6].
        channels : list[int], optional
            Channel numbers to include. Only used when spectral_stats is not
            available. Default: [1, 2, ..., 17] (excludes channel 18 - lower
            corners).

        Returns
        -------
        channel_def : LVFContourChannelDefinition
            Channel definition with bundled contour map.

        Examples
        --------
        >>> from spx_data import ArrayWavelengthModel, LVFContourChannelDefinition
        >>> wl_model = ArrayWavelengthModel.from_fits('wavelength_{array_id}.fits')
        >>> channel_def = LVFContourChannelDefinition.from_bundled(wl_model)

        """
        # Default to channels 1-17 (exclude channel 18 from bundled contour map)
        if channels is None:
            channels = list(range(1, 18))
        return cls.from_fits(
            get_lvf_contour_path(), wavelength_model, arrays, channels
        )

    @classmethod
    def from_hdul(
        cls,
        hdul: fits.HDUList,
        arrays: Optional[list[int]] = None,
    ) -> 'LVFContourChannelDefinition':
        """Create channel definition from an open HDUList.

        Expects the calibration file format with CHANNEL_MAP image extension
        and SPECTRAL_CHANNELS binary table extension.

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            Open FITS HDU list containing CHANNEL_MAP and SPECTRAL_CHANNELS
            extensions.
        arrays : list[int], optional
            Arrays to include. Default: [1, 2, 3, 4, 5, 6].

        Returns
        -------
        channel_def : LVFContourChannelDefinition
            Channel definition with contour map and spectral statistics.

        Raises
        ------
        ValueError
            If CHANNEL_MAP or SPECTRAL_CHANNELS extension is missing.

        """
        if 'CHANNEL_MAP' not in hdul:
            raise ValueError("HDUList must contain CHANNEL_MAP extension")
        if 'SPECTRAL_CHANNELS' not in hdul:
            raise ValueError("HDUList must contain SPECTRAL_CHANNELS extension")

        contour_map = hdul['CHANNEL_MAP'].data
        spectral_stats = Table(hdul['SPECTRAL_CHANNELS'].data)

        return cls(contour_map, arrays=arrays, spectral_stats=spectral_stats)

    def to_hdul(self) -> fits.HDUList:
        """Convert channel definition to an HDUList.

        Creates a FITS structure with:
        - Primary HDU: empty (no data)
        - CHANNEL_MAP extension: 2D contour map image
        - SPECTRAL_CHANNELS extension: binary table with spectral statistics

        Returns
        -------
        hdul : astropy.io.fits.HDUList
            HDU list ready to be written to a FITS file.

        Raises
        ------
        ValueError
            If spectral statistics are not available.

        """
        if self._spectral_stats is None:
            raise ValueError(
                "Cannot create HDUList: spectral statistics not available. "
                "Initialize with wavelength_model or spectral_stats."
            )

        # Primary HDU with no data
        primary = fits.PrimaryHDU()

        # CHANNEL_MAP image extension
        channel_map_hdu = fits.ImageHDU(
            data=self._contour_map,
            name='CHANNEL_MAP'
        )
        channel_map_hdu.header['EXTNAME'] = 'CHANNEL_MAP'
        channel_map_hdu.header['BUNIT'] = ''
        channel_map_hdu.header.comments['BUNIT'] = 'Channel number (unitless)'

        # SPECTRAL_CHANNELS binary table extension
        spectral_hdu = fits.BinTableHDU(self._spectral_stats, name='SPECTRAL_CHANNELS')

        return fits.HDUList([primary, channel_map_hdu, spectral_hdu])

    @staticmethod
    def subdivide_contour_map(
        contour_map: np.ndarray,
        channels: Optional[list[int]] = None,
        n_subdivisions: int = 4,
        poly_degree: int = 2,
        output_dir: Optional[str] = None,
    ) -> np.ndarray:
        """Create a new contour map with N subdivisions per channel.

        Fits a degree-`poly_degree` polynomial through each channel's
        `y_min(x)` and `y_max(x)` boundary rows across all columns, then
        divides that smoothed (sub-pixel) boundary curve into
        `n_subdivisions` equal parts per column, floor-rounded to a pixel
        row. Pixels outside the requested channels are set to 0.

        A channel whose true geometric boundary would extend past the
        image edge is clipped there in the rasterized input, its raw
        `y_min`/`y_max` is pinned at row 0 or `ny - 1` for those columns
        rather than following the real curve. Those clipped columns are
        excluded when fitting (falling back to fitting all columns if too
        few unclipped ones remain to support `poly_degree`); the fitted
        curve is still evaluated at every column, including the clipped
        ones, where it legitimately extrapolates outside `[0, ny - 1]`,
        fine, since it's only used to compute the fractional position of
        pixels that actually exist in the mask.

        Parameters
        ----------
        contour_map : np.ndarray
            Original 2D contour map where pixel values indicate channel numbers
            (e.g., 1-17 for standard channels, 0 for excluded regions).
        channels : list[int], optional
            Channels to include in subdivision. Default is [1, 2, ..., 17].
            Channels are processed in sorted order for consistent numbering.
        n_subdivisions : int
            Number of subdivisions per channel (1-7). Default is 4.
            With 17 channels and n_subdivisions=4, creates 68 subchannels.
        poly_degree : int, optional
            Degree of the polynomial fit through each channel's `y_min(x)`/
            `y_max(x)` boundary rows. Default 2 (quadratic) -- degree 1 (a
            straight line) underfits the LVF's actual curvature (~70% match
            in testing); degrees above 3 gave no further improvement and
            risk Runge-phenomenon wiggle near the column edges.
        output_dir : str, optional
            If provided, saves the new contour map as a FITS file in this
            directory with name LVF_contour_{max_subchannel}Subchannels.fits.gz

        Returns
        -------
        np.ndarray
            New contour map (same shape as input) with subdivided channels
            numbered sequentially from 1. Pixels outside the requested
            channels are set to 0.

        Raises
        ------
        ValueError
            If n_subdivisions is not between 1 and 7, or if a requested
            channel spans too few columns to fit `poly_degree`.

        Examples
        --------
        >>> # Create 68-subchannel map from standard 17-channel map
        >>> from spx_data import get_lvf_contour_path
        >>> from astropy.io import fits
        >>> with fits.open(get_lvf_contour_path()) as hdul:
        ...     contour_map = hdul[0].data
        >>> new_map = LVFContourChannelDefinition.subdivide_contour_map(
        ...     contour_map, n_subdivisions=4, output_dir='/tmp'
        ... )
        >>> print(f"Max subchannel: {new_map.max()}")
        Max subchannel: 68

        >>> # Subdivide only channels 5-10
        >>> new_map = LVFContourChannelDefinition.subdivide_contour_map(
        ...     contour_map, channels=[5, 6, 7, 8, 9, 10], n_subdivisions=4
        ... )
        >>> print(f"Max subchannel: {new_map.max()}")
        Max subchannel: 24

        """
        import os

        if n_subdivisions < 1 or n_subdivisions > 7:
            raise ValueError(
                f"n_subdivisions must be between 1 and 7, got {n_subdivisions}"
            )

        if channels is None:
            channels = list(range(1, 18))

        # Sort channels to ensure consistent ordering
        channels = sorted(channels)

        # Create mapping from original channel to subchannel offset
        # channel_to_offset[c] gives the base subchannel number (0-indexed)
        channel_to_offset = {ch: i * n_subdivisions for i, ch in enumerate(channels)}

        ny, nx = contour_map.shape
        new_contour = np.zeros((ny, nx), dtype=np.uint8)
        all_x = np.arange(nx)
        row_idx = np.arange(ny)[:, None]  # column vector, broadcasts against (nx,) curves

        for channel in channels:
            mask = contour_map == channel
            cols = np.where(mask.any(axis=0))[0]
            if len(cols) == 0:
                continue
            if len(cols) <= poly_degree:
                raise ValueError(
                    f"Channel {channel} spans only {len(cols)} column(s), too few "
                    f"to fit a degree-{poly_degree} polynomial boundary curve."
                )

            # Vectorized per-column first/last True row: argmax on a
            # boolean array returns the index of the first True, and
            # reversing the array first turns "first" into "last" --
            # avoids a np.where(...).min()/.max() call per column.
            y_min_raw = np.argmax(mask, axis=0)[cols].astype(float)
            y_max_raw = (ny - 1 - np.argmax(mask[::-1, :], axis=0))[cols].astype(float)

            # Exclude image-edge-clipped columns from the fit -- their raw
            # y_min/y_max is an artifact of the frame boundary, not the
            # channel's true curve. Fall back to fitting everything if that
            # leaves too few points for poly_degree (e.g. a channel that's
            # clipped almost everywhere it exists).
            unclipped = (y_max_raw < ny - 1) & (y_min_raw > 0)
            if unclipped.sum() <= poly_degree:
                fit_cols, fit_y_max, fit_y_min = cols, y_max_raw, y_min_raw
            else:
                fit_cols = cols[unclipped]
                fit_y_max = y_max_raw[unclipped]
                fit_y_min = y_min_raw[unclipped]

            y_max_curve = np.poly1d(np.polyfit(fit_cols, fit_y_max, poly_degree))
            y_min_curve = np.poly1d(np.polyfit(fit_cols, fit_y_min, poly_degree))

            # Evaluate both curves once per column and broadcast against
            # every row to get the whole image's fractional position in a
            # handful of array ops, instead of looping per column.
            #
            # Use (y_max - y) so that smaller y values get higher subdivision
            # numbers, matching the wavelength direction (wavelength increases
            # with y, so subchannel 1 should be at highest y within channel)
            y_max_at_col = y_max_curve(all_x)
            y_range_at_col = y_max_at_col - y_min_curve(all_x)
            relative_pos = y_max_at_col[None, :] - row_idx
            subdivision = np.clip(
                (relative_pos * n_subdivisions / y_range_at_col[None, :]).astype(np.int64),
                0, n_subdivisions - 1,
            )

            offset = channel_to_offset[channel]
            new_contour[mask] = offset + subdivision[mask] + 1

        # Save if output_dir provided
        if output_dir is not None:
            max_subchannel = len(channels) * n_subdivisions
            filename = f"LVF_contour_{max_subchannel}Subchannels.fits.gz"
            filepath = os.path.join(output_dir, filename)

            hdu = fits.PrimaryHDU(new_contour)
            hdu.header['NSUBDIV'] = (n_subdivisions, 'Number of subdivisions per channel')
            hdu.header['NCHAN'] = (len(channels), 'Number of original channels included')
            hdu.header['MAXSUB'] = (max_subchannel, 'Highest subchannel number')
            hdu.header['CHAN_MIN'] = (min(channels), 'Minimum original channel number')
            hdu.header['CHAN_MAX'] = (max(channels), 'Maximum original channel number')
            hdu.header['POLYDEG'] = (poly_degree, 'Per-column boundary polynomial fit degree')
            hdu.writeto(filepath, overwrite=True)

        return new_contour

    @staticmethod
    def subdivide_bundled_contour_map(n_subdivisions: int = 4, poly_degree: int = 2) -> np.ndarray:
        """Create a subdivided contour map from the bundled 17-channel LVF
        contour template.

        Convenience wrapper combining `get_lvf_contour_path()` and
        `subdivide_contour_map()`, so callers don't need to load the
        bundled file manually. Always uses channels 1-17 (excludes channel
        18, the lower corners) from the bundled template.

        Parameters
        ----------
        n_subdivisions : int, optional
            Number of subdivisions per channel (1-7). Default is 4.
        poly_degree : int, optional
            Passed through to `subdivide_contour_map()`. Default 2.

        Returns
        -------
        numpy.ndarray
            Subdivided contour map, same shape as the bundled template.

        Examples
        --------
        >>> half_map = LVFContourChannelDefinition.subdivide_bundled_contour_map(
        ...     n_subdivisions=2
        ... )

        """
        with fits.open(get_lvf_contour_path()) as hdul:
            base_contour_map = hdul[0].data
        return LVFContourChannelDefinition.subdivide_contour_map(
            base_contour_map, channels=list(range(1, 18)), n_subdivisions=n_subdivisions,
            poly_degree=poly_degree,
        )

    @staticmethod
    def channel_numbers_from_map(
        contour_map: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        dtype: np.dtype = np.int32,
    ) -> np.ndarray:
        """Vectorized channel-number lookup for a subdivided contour map.

        For a single contour map shared across all detector arrays (e.g.
        produced by `subdivide_contour_map`), pixel values already ARE the
        channel numbers at that map's own resolution. This is a plain
        vectorized 2D lookup: round each (x, y) to its containing pixel,
        clip to the map's bounds, and read off the value.

        Parameters
        ----------
        contour_map : numpy.ndarray
            2D contour map (e.g. from `subdivide_contour_map`), shared
            across all detector arrays. Pixel values are channel numbers;
            0 means no channel.
        x : numpy.ndarray of float or int
            X pixel coordinate (column, 0-based) for each measurement.
            Mapped to the containing integer pixel via ceil(v - 0.5).
        y : numpy.ndarray of float or int
            Y pixel coordinate (row, 0-based) for each measurement.
            Mapped to the containing integer pixel via ceil(v - 0.5).
        dtype : numpy.dtype, optional
            Integer dtype of the returned array. Default `numpy.int32`;
            pass a narrower dtype (e.g. `numpy.uint8`) if your channel
            numbers are known to fit, to reduce output memory.

        Returns
        -------
        channels : numpy.ndarray
            Channel number for each measurement, as `dtype`. 0 where the
            (clipped) pixel does not belong to any channel.

        Examples
        --------
        >>> half_map = LVFContourChannelDefinition.subdivide_contour_map(
        ...     base_contour_map, n_subdivisions=2
        ... )
        >>> channels = LVFContourChannelDefinition.channel_numbers_from_map(
        ...     half_map, x=x_pix, y=y_pix
        ... )

        """
        x_idx = np.ceil(np.asarray(x) - 0.5).astype(np.int32)
        y_idx = np.ceil(np.asarray(y) - 0.5).astype(np.int32)

        ny, nx = contour_map.shape
        # Valid pixel-center coordinates span the inclusive range
        # [-0.5, dim - 0.5]; clip into the nearest valid pixel index rather
        # than raising or wrapping negative indices.
        np.clip(x_idx, 0, nx - 1, out=x_idx)
        np.clip(y_idx, 0, ny - 1, out=y_idx)

        return contour_map[y_idx, x_idx].astype(dtype, copy=False)

    @staticmethod
    def global_channel_numbers(
        array_ids: np.ndarray,
        channels: np.ndarray,
        n_channels_per_array: int,
        dtype: np.dtype = np.int32,
    ) -> np.ndarray:
        """Combine per-detector channel numbers into global channel numbers.

        Maps (array_id, channel) to a single number unique across all
        detectors: detector 1's channels occupy
        `[1, n_channels_per_array]`, detector 2's occupy
        `[n_channels_per_array + 1, 2 * n_channels_per_array]`, and so on.

        Parameters
        ----------
        array_ids : numpy.ndarray of int
            Detector array number for each measurement.
        channels : numpy.ndarray of int
            Per-detector channel number for each measurement. 0 means no
            channel and is preserved as 0 in the output -- not silently
            folded into a spurious nonzero global number.
        n_channels_per_array : int
            Number of channels per detector at the relevant resolution.
        dtype : numpy.dtype, optional
            Integer dtype of the returned array. Default `numpy.int32`.

        Returns
        -------
        global_channels : numpy.ndarray
            Global channel number for each measurement, as `dtype`. 0
            where the input channel was 0 (no channel).

        Examples
        --------
        >>> channels = LVFContourChannelDefinition.channel_numbers_from_map(
        ...     half_map, x=x_pix, y=y_pix
        ... )
        >>> global_channels = LVFContourChannelDefinition.global_channel_numbers(
        ...     array_ids=detector, channels=channels, n_channels_per_array=34
        ... )

        """
        array_ids = np.asarray(array_ids)
        channels = np.asarray(channels)
        valid = channels > 0
        global_channels = np.where(
            valid, (array_ids - 1) * n_channels_per_array + channels, 0
        )
        return global_channels.astype(dtype, copy=False)

    def get_all_channels(self) -> list[tuple[int, int]]:
        """Return all (array_id, channel) combinations."""
        if self._stats_index:
            return sorted(self._stats_index.keys())
        # Fall back to generating from arrays and channels
        return [
            (array_id, channel)
            for array_id in self.arrays
            for channel in self._channels
        ]

    @property
    def spectral_stats(self) -> Optional[Table]:
        """Return the spectral statistics table, if available."""
        return self._spectral_stats

    def _compute_stats_table(
        self,
        contour_map: np.ndarray,
        wavelength_model: WavelengthModel,
    ) -> Table:
        """Compute spectral statistics table for all (array_id, channel) combinations.

        This is called once during initialization to pre-compute all stats,
        allowing the wavelength model to be discarded afterward.

        Returns
        -------
        table : astropy.table.Table
            Table with columns: DETECTOR, SUBCHAN, WAVELENGTH, WL_MIN, WL_MAX,
            R, R_STD, BANDWIDTH. Values are NaN for channels with no valid pixels.

        """
        rows = []
        has_bandwidth = wavelength_model.has_bandwidth()

        for array_id in self.arrays:
            if array_id not in wavelength_model.array_ids:
                continue
            wl_array = wavelength_model.get_wavelength_array(array_id)
            bw_array = (
                wavelength_model.get_bandwidth_array(array_id)
                if has_bandwidth else None
            )

            # Handle shape mismatch
            if contour_map.shape != wl_array.shape:
                target_shape = (
                    min(contour_map.shape[0], wl_array.shape[0]),
                    min(contour_map.shape[1], wl_array.shape[1]),
                )
                contour_cropped = _crop_to_shape(contour_map, target_shape)
                wl_cropped = _crop_to_shape(wl_array, target_shape)
                bw_cropped = (
                    _crop_to_shape(bw_array, target_shape)
                    if bw_array is not None else None
                )
            else:
                contour_cropped = contour_map
                wl_cropped = wl_array
                bw_cropped = bw_array

            for channel in self._channels:
                mask = contour_cropped == channel
                if not np.any(mask):
                    row = {
                        'DETECTOR': array_id,
                        'SUBCHAN': channel,
                        'WAVELENGTH': np.nan,
                        'WL_MIN': np.nan,
                        'WL_MAX': np.nan,
                        'R': np.nan,
                        'R_STD': np.nan,
                        'BANDWIDTH': np.nan,
                    }
                else:
                    wl_values = wl_cropped[mask]
                    r_median = np.nan
                    r_std = np.nan
                    bandwidth = np.nan

                    if bw_cropped is not None:
                        bw_values = bw_cropped[mask]
                        # Per-pixel resolving power R = λ/Δλ
                        r_values = wl_values / bw_values
                        r_median = np.nanmedian(r_values)
                        r_std = np.nanstd(r_values)
                        # Effective bandwidth = wavelength / R
                        wl_median = np.nanmedian(wl_values)
                        if r_median > 0 and not np.isnan(r_median):
                            bandwidth = wl_median / r_median

                    # Use sigma clipping to exclude outliers for min/max
                    wl_clipped = sigma_clip(wl_values, sigma=3, masked=True)
                    wl_good = wl_clipped.compressed()

                    row = {
                        'DETECTOR': array_id,
                        'SUBCHAN': channel,
                        'WAVELENGTH': np.nanmedian(wl_values),
                        'WL_MIN': np.min(wl_good) if len(wl_good) > 0 else np.nan,
                        'WL_MAX': np.max(wl_good) if len(wl_good) > 0 else np.nan,
                        'R': r_median,
                        'R_STD': r_std,
                        'BANDWIDTH': bandwidth,
                    }
                rows.append(row)

        # Create table with explicit dtypes
        table = Table(rows)
        table['DETECTOR'] = table['DETECTOR'].astype(np.int32)
        table['SUBCHAN'] = table['SUBCHAN'].astype(np.int32)
        for col in ['WAVELENGTH', 'WL_MIN', 'WL_MAX', 'R', 'R_STD', 'BANDWIDTH']:
            table[col] = table[col].astype(np.float32)

        # Add units
        table['WAVELENGTH'].unit = 'um'
        table['WL_MIN'].unit = 'um'
        table['WL_MAX'].unit = 'um'
        table['BANDWIDTH'].unit = 'um'

        # Add descriptions
        table['DETECTOR'].description = 'Detector array number (1-6)'
        table['SUBCHAN'].description = 'Subchannel number (1-17)'
        table['WAVELENGTH'].description = 'Effective (median) wavelength'
        table['WL_MIN'].description = 'Min per-pixel wavelength'
        table['WL_MAX'].description = 'Max per-pixel wavelength'
        table['R'].description = 'Median resolving power'
        table['R_STD'].description = 'Std dev of resolving power'
        table['BANDWIDTH'].description = 'Effective bandwidth'

        return table

    def _get_contour_map_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        """Get contour map cropped to match data shape, with caching."""
        if shape not in self._cropped_contour_cache:
            contour_map = self._contour_map
            if contour_map.shape != shape:
                contour_map = _crop_to_shape(contour_map, shape)
            self._cropped_contour_cache[shape] = contour_map
        return self._cropped_contour_cache[shape]

    def get_channel_mask(
        self,
        array_id: int,
        channel: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get boolean mask for pixels in the specified channel.

        Note: The array_id parameter is ignored for mask generation since all
        arrays use the same contour map. It is only used for wavelength
        calculations.
        """
        contour_map = self._get_contour_map_for_shape(shape)
        return contour_map == channel

    def get_channel_info(
        self,
        array_id: int,
        channel: int,
    ) -> ChannelInfo:
        """Get all spectral statistics for a channel.

        If the spectral_stats table contains a BANDWIDTH column, that value
        is used. Otherwise, bandwidth can be computed via the effective_bandwidth
        property (wavelength/R).
        """
        idx = self._stats_index.get((array_id, channel))
        if idx is None or self._spectral_stats is None:
            return ChannelInfo(
                detector=array_id,
                subchannel=channel,
                wavelength=np.nan * u.um,
                wavelength_min=np.nan * u.um,
                wavelength_max=np.nan * u.um,
                R=None,
                R_std=None,
                bandwidth=None,
            )

        row = self._spectral_stats[idx]

        # Wavelength
        wl = row['WAVELENGTH']
        wavelength = wl * u.um if not np.isnan(wl) else np.nan * u.um

        # Wavelength range
        wl_min = row['WL_MIN']
        wl_max = row['WL_MAX']
        wavelength_min = wl_min * u.um if not np.isnan(wl_min) else np.nan * u.um
        wavelength_max = wl_max * u.um if not np.isnan(wl_max) else np.nan * u.um

        # Resolving power
        r_median = row['R']
        r_std = row['R_STD']
        R = float(r_median) if not np.isnan(r_median) else None
        R_std_val = float(r_std) if not np.isnan(r_std) else None

        # Bandwidth (pre-computed if available in table)
        bandwidth = None
        if 'BANDWIDTH' in self._spectral_stats.colnames:
            bw_value = row['BANDWIDTH']
            if not np.isnan(bw_value):
                bandwidth = bw_value * u.um

        return ChannelInfo(
            detector=array_id,
            subchannel=channel,
            wavelength=wavelength,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            R=R,
            R_std=R_std_val,
            bandwidth=bandwidth,
        )


def _divisors(n: int) -> list[int]:
    """Return all positive divisors of n, descending (n itself first)."""
    return sorted((d for d in range(1, n + 1) if n % d == 0), reverse=True)


@numba.njit(cache=True, fastmath=True, parallel=True)
def _global_channel_numbers_kernel(
    array_ids: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    contour_stack: np.ndarray,
    ny: int,
    nx: int,
    n_subchannels: int,
    resolution: int,
    group_size: int,
    max_fine: int,
    n_channels_per_array: int,
    out: np.ndarray,
) -> None:
    """Per-measurement kernel for `LVFChannelDefinitionV2.get_global_channel_numbers`.

    A single compiled pass avoids the numpy temporaries (coordinate casts,
    clip, fancy-index result, divmod intermediates, `np.where` output) the
    equivalent vectorized numpy code would allocate, cutting both peak
    memory and wall time at the thousands-to-tens-of-thousands scale this
    is called at.
    """
    for i in numba.prange(array_ids.shape[0]):
        # ceil(v - 0.5) rather than round(): gives the containing pixel
        # index and (unlike numpy's round-half-to-even) is consistent at
        # exact half-integer pixel-boundary values.
        xi = np.int32(np.ceil(x[i] - 0.5))
        if xi < 0:
            xi = 0
        elif xi > nx - 1:
            xi = nx - 1
        yi = np.int32(np.ceil(y[i] - 0.5))
        if yi < 0:
            yi = 0
        elif yi > ny - 1:
            yi = ny - 1

        array_id = array_ids[i]
        # Stack is indexed directly by array_id,
        # no array_ids - 1 shift is needed here.
        fine = contour_stack[array_id, yi, xi]
        # A fine value is only a genuine channel if it's in [1, max_fine].
        if fine <= 0 or fine > max_fine:
            out[i] = 0
            continue

        if group_size == 1:
            # Active resolution is the finest stored resolution: the fine
            # value already *is* the channel number.
            channel = np.int32(fine)
        else:
            shifted = np.int32(fine) - 1
            base_idx0 = shifted // n_subchannels
            bucket0 = (shifted - base_idx0 * n_subchannels) // group_size
            channel = base_idx0 * resolution + bucket0 + 1
        out[i] = (array_id - 1) * n_channels_per_array + channel


def _compute_stats_row(
    detector: int,
    subchannel: int,
    wl_array: np.ndarray,
    bw_array: Optional[np.ndarray],
    mask: np.ndarray,
) -> dict:
    """Compute one spectral-statistics row from a channel pixel mask."""
    if not np.any(mask):
        return {
            'DETECTOR': detector,
            'SUBCHAN': subchannel,
            'WAVELENGTH': np.nan,
            'WL_MIN': np.nan,
            'WL_MAX': np.nan,
            'R': np.nan,
            'R_STD': np.nan,
            'BANDWIDTH': np.nan,
        }

    wl_values = wl_array[mask]
    r_median = np.nan
    r_std = np.nan
    bandwidth = np.nan

    if bw_array is not None:
        bw_values = bw_array[mask]
        # Per-pixel resolving power R = λ/Δλ
        r_values = wl_values / bw_values
        r_median = np.median(r_values)
        r_std = np.std(r_values)
        wl_median = np.nanmedian(wl_values)
        if r_median > 0 and not np.isnan(r_median):
            bandwidth = wl_median / r_median

    # Use sigma clipping to exclude outliers for min/max
    wl_clipped = sigma_clip(wl_values, sigma=3, masked=True)
    wl_good = wl_clipped.compressed()

    return {
        'DETECTOR': detector,
        'SUBCHAN': subchannel,
        'WAVELENGTH': np.nanmedian(wl_values),
        'WL_MIN': np.min(wl_good) if len(wl_good) > 0 else np.nan,
        'WL_MAX': np.max(wl_good) if len(wl_good) > 0 else np.nan,
        'R': r_median,
        'R_STD': r_std,
        'BANDWIDTH': bandwidth,
    }


def _make_stats_table(rows: list[dict]) -> Table:
    """Assemble a spectral-statistics Table with consistent dtypes/units."""
    table = Table(rows)
    # DETECTOR is always 1-6, always safe as uint8. SUBCHAN is normally
    # <= 17*7=119 (subdivide_contour_map's own cap) but an
    # already_subdivided=True caller could in principle exceed uint8's
    # range with a custom template, so fall back to int32 when needed.
    table['DETECTOR'] = table['DETECTOR'].astype(np.uint8)
    subchan_dtype = np.uint8 if table['SUBCHAN'].max() <= 255 else np.int32
    table['SUBCHAN'] = table['SUBCHAN'].astype(subchan_dtype)
    for col in ['WAVELENGTH', 'WL_MIN', 'WL_MAX', 'R', 'R_STD', 'BANDWIDTH']:
        table[col] = table[col].astype(np.float32)

    table['WAVELENGTH'].unit = 'um'
    table['WL_MIN'].unit = 'um'
    table['WL_MAX'].unit = 'um'
    table['BANDWIDTH'].unit = 'um'

    table['DETECTOR'].description = 'Detector array number (1-6)'
    table['SUBCHAN'].description = 'Subchannel number at this resolution'
    table['WAVELENGTH'].description = 'Effective (median) wavelength'
    table['WL_MIN'].description = 'Min per-pixel wavelength'
    table['WL_MAX'].description = 'Max per-pixel wavelength'
    table['R'].description = 'Median resolving power'
    table['R_STD'].description = 'Std dev of resolving power'
    table['BANDWIDTH'].description = 'Effective bandwidth'

    for col in ['WAVELENGTH', 'WL_MIN', 'WL_MAX', 'BANDWIDTH']:
        table[col].format = '10.3f'
    table['R'].format = '10.0f'
    table['R_STD'].format = '10.1f'

    return table


class LVFChannelDefinitionV2(ChannelDefinition):
    """Multi-resolution LVF contour channel definition with per-detector templates.

    This is the next-generation `spectral_channels` calibration product. A
    single calibration file stores one fine-grained contour map per template
    (a default template shared by all detectors, plus optional per-detector
    overrides) subdivided into `n_subchannels` equally-sized sub-channels per
    standard channel. Pre-computed spectral statistics are stored for every
    *resolution* that evenly divides `n_subchannels` (e.g. with
    `n_subchannels=4`: quarter channels at resolution 4, half channels at
    resolution 2, and standard channels at resolution 1), so quarter, half,
    and standard channels are all derived from the same geometric template
    without re-reading pixel data.

    An instance is bound to one active resolution at a time (`self.resolution`,
    a divisor of `n_subchannels`). Use `at_resolution()` to get a view of the
    same loaded data bound to a different resolution — this is O(1) and does
    not re-parse the FITS file or recompute statistics.

    Per-detector templates
    -----------------------
    Any subset of detectors (1-6) may have their own contour map,
    distinguished by a `DETECTOR` header keyword and `CHANNEL_MAP_D{n}`
    extension name. A detector without its own template falls back to the
    default template (`CHANNEL_MAP`, no `DETECTOR` header). The override only
    affects which template is used to derive channel masks and statistics
    for that detector — the resulting SPECTRAL_CHANNELS tables already have
    the override baked in and are not themselves split per detector.

    Parameters
    ----------
    contour_maps : dict[Optional[int], numpy.ndarray]
        Fine-resolution (finest, `n_subchannels`-per-channel) contour maps.
        Key `None` is the default template shared by detectors with no
        override; integer keys (1-6) are detector-specific overrides.
    stats : dict[int, astropy.table.Table]
        Pre-computed spectral statistics keyed by resolution (a divisor of
        `n_subchannels`). Each table has columns DETECTOR, SUBCHAN,
        WAVELENGTH, WL_MIN, WL_MAX, R, R_STD, BANDWIDTH and includes rows for
        every detector, whether covered by the default or an override
        template.
    n_subchannels : int
        Finest resolution: number of sub-channels per standard channel.
    n_base_channels : int
        Number of standard channels (e.g. 17).
    arrays : list[int], optional
        Array numbers to include. Default: [1, 2, 3, 4, 5, 6].
    resolution : int, optional
        Initial active resolution (must be a key of `stats`). Default: 1
        (standard channels) if available, otherwise `n_subchannels`.

    Examples
    --------
    Build from an existing standard contour template and wavelength model,
    with quarter-channel resolution as the finest level:

    >>> from spx_data import get_lvf_contour_path, ArrayWavelengthModel
    >>> from spx_data import LVFChannelDefinitionV2
    >>> from astropy.io import fits
    >>> with fits.open(get_lvf_contour_path()) as hdul:
    ...     base_map = hdul[0].data
    >>> wl_model = ArrayWavelengthModel.from_fits('wavelength_{array_id}.fits')
    >>> channel_def = LVFChannelDefinitionV2.build(
    ...     base_map, wl_model, n_subchannels=4
    ... )
    >>> channel_def.to_hdul().writeto('spectral_channels_v2.fits')

    Load a calibration file and switch resolutions without re-parsing:

    >>> channel_def = LVFChannelDefinitionV2.from_fits('spectral_channels_v2.fits')
    >>> standard = channel_def  # resolution=1 by default
    >>> quarter = channel_def.at_resolution(4)
    >>> mask = quarter.get_channel_mask(array_id=1, channel=30, shape=(2040, 2040))

    """

    def __init__(
        self,
        contour_maps: dict,
        stats: dict,
        n_subchannels: int,
        n_base_channels: int,
        arrays: Optional[list[int]] = None,
        resolution: Optional[int] = None,
    ):
        if None not in contour_maps:
            missing = sorted(set(range(1, 7)) - set(contour_maps))
            if missing:
                raise ValueError(
                    "No default CHANNEL_MAP and no detector-specific "
                    f"template for detector(s) {missing}; every array needs "
                    "either an override or a default template."
                )
        for det in contour_maps:
            if det is not None and det not in (1, 2, 3, 4, 5, 6):
                raise ValueError(f"Detector ids must be 1-6, got {det}")

        self._contour_maps = contour_maps
        self._stats = stats
        self._n_subchannels = n_subchannels
        self._n_base_channels = n_base_channels
        self.arrays = arrays if arrays is not None else [1, 2, 3, 4, 5, 6]
        self._cropped_contour_cache: dict = {}
        self._contour_stack_cache: dict = {}

        self._set_resolution(
            resolution if resolution is not None
            else (1 if 1 in stats else n_subchannels)
        )

    @classmethod
    def build(
        cls,
        base_contour_map: np.ndarray,
        wavelength_model: WavelengthModel,
        n_subchannels: int = 4,
        channels: Optional[list[int]] = None,
        detector_contour_maps: Optional[dict] = None,
        arrays: Optional[list[int]] = None,
        already_subdivided: bool = False,
        stats_resolutions: Optional[list[int]] = None,
    ) -> Self:
        """Build a new multi-resolution channel definition from base templates.

        Subdivides `base_contour_map` (and any per-detector override maps in
        `detector_contour_maps`) into `n_subchannels` equally-sized fine
        sub-channels per standard channel. The full geometry (masks,
        `get_global_channel_numbers`) is available at every resolution that evenly
        divides `n_subchannels` regardless of `stats_resolutions`; only the
        precomputed statistics tables are scoped to `stats_resolutions`.

        Parameters
        ----------
        base_contour_map : numpy.ndarray
            Standard (undivided) contour map, e.g. the bundled 17-channel LVF
            contour template, or (if `already_subdivided` is True) a map
            already at the finest (`n_subchannels`-per-channel) resolution.
            Used for every detector without an entry in
            `detector_contour_maps`.
        wavelength_model : WavelengthModel
            Model providing per-pixel wavelengths (and optionally bandwidth)
            for every detector array being built.
        n_subchannels : int, optional
            Number of equal-sized fine sub-channels per standard channel.
            Must be between 1 and 7 (the range supported by
            `LVFContourChannelDefinition.subdivide_contour_map`). Default is
            4 (quarter channels).
        channels : list[int], optional
            Standard channel numbers to include. Default is [1, ..., 17].
            Ignored if `already_subdivided` is True.
        detector_contour_maps : dict[int, numpy.ndarray], optional
            Per-detector override contour maps, keyed by detector array
            number (1-6). Any detector not present here uses
            `base_contour_map`. Default: no overrides (all detectors share
            the default template).
        arrays : list[int], optional
            Detector arrays to compute statistics for. Default: [1..6].
        already_subdivided : bool, optional
            If True, `base_contour_map` and any `detector_contour_maps` are
            already at the finest (`n_subchannels`-per-channel) resolution
            and are used as-is, skipping the internal subdivision step.
            `channels` is ignored in this case; the number of standard
            channels is inferred from the maximum contour value. Default is
            False (inputs are standard, undivided 1-17 maps that this
            method subdivides internally).

        stats_resolutions : list[int], optional
            Which resolutions (divisors of `n_subchannels`) to precompute
            statistics tables for. Default: `[1]` (standard channels only)
            -- geometry at finer resolutions remains fully usable via
            `at_resolution()` for masks/`get_global_channel_numbers` even with no
            stats computed there. Pass e.g. `[4, 2, 1]` for full
            multi-resolution stats (the previous default behavior).

        Returns
        -------
        channel_def : LVFChannelDefinitionV2
            New instance bound to the standard-channel resolution (1), with
            statistics precomputed for `stats_resolutions`.

        """
        arrays = arrays if arrays is not None else [1, 2, 3, 4, 5, 6]
        detector_contour_maps = detector_contour_maps or {}
        for det in detector_contour_maps:
            if det not in (1, 2, 3, 4, 5, 6):
                raise ValueError(f"Detector ids must be 1-6, got {det}")

        divisors = _divisors(n_subchannels)

        if stats_resolutions is None:
            stats_resolutions = [1]
        invalid = set(stats_resolutions) - set(divisors)
        if invalid:
            raise ValueError(
                f"stats_resolutions {sorted(invalid)} are not divisors of "
                f"n_subchannels={n_subchannels} (valid: {divisors})"
            )

        if already_subdivided:
            contour_maps = {None: base_contour_map, **detector_contour_maps}
            n_base_channels = int(np.max(base_contour_map)) // n_subchannels
        else:
            if channels is None:
                channels = list(range(1, 18))
            channels = sorted(channels)

            contour_maps = {
                None: LVFContourChannelDefinition.subdivide_contour_map(
                    base_contour_map, channels=channels, n_subdivisions=n_subchannels
                )
            }
            for det, det_map in detector_contour_maps.items():
                contour_maps[det] = LVFContourChannelDefinition.subdivide_contour_map(
                    det_map, channels=channels, n_subdivisions=n_subchannels
                )

            n_base_channels = len(channels)

        has_bandwidth = wavelength_model.has_bandwidth()

        stats = {}
        for divisor in stats_resolutions:
            group_size = n_subchannels // divisor
            rows = []
            for array_id in arrays:
                if array_id not in wavelength_model.array_ids:
                    continue
                fine_map = contour_maps.get(array_id, contour_maps[None])
                wl_array = wavelength_model.get_wavelength_array(array_id)
                bw_array = (
                    wavelength_model.get_bandwidth_array(array_id)
                    if has_bandwidth else None
                )

                if fine_map.shape != wl_array.shape:
                    target_shape = (
                        min(fine_map.shape[0], wl_array.shape[0]),
                        min(fine_map.shape[1], wl_array.shape[1]),
                    )
                    fine_cropped = _crop_to_shape(fine_map, target_shape)
                    wl_cropped = _crop_to_shape(wl_array, target_shape)
                    bw_cropped = (
                        _crop_to_shape(bw_array, target_shape)
                        if bw_array is not None else None
                    )
                else:
                    fine_cropped = fine_map
                    wl_cropped = wl_array
                    bw_cropped = bw_array

                for base_idx in range(n_base_channels):
                    for bucket in range(divisor):
                        coarse_channel = base_idx * divisor + bucket + 1
                        fine_start = (
                            base_idx * n_subchannels + bucket * group_size + 1
                        )
                        mask = (
                            (fine_cropped >= fine_start)
                            & (fine_cropped < fine_start + group_size)
                        )
                        rows.append(
                            _compute_stats_row(
                                array_id, coarse_channel, wl_cropped, bw_cropped, mask
                            )
                        )
            stats[divisor] = _make_stats_table(rows)

        return cls(
            contour_maps=contour_maps,
            stats=stats,
            n_subchannels=n_subchannels,
            n_base_channels=n_base_channels,
            arrays=arrays,
            resolution=1,
        )

    @classmethod
    def from_hdul(
        cls,
        hdul: fits.HDUList,
        arrays: Optional[list[int]] = None,
    ) -> Self:
        """Create a channel definition from an open HDUList.

        Expects the multi-resolution calibration format: a default
        `CHANNEL_MAP` image extension and/or per-detector `CHANNEL_MAP_D{n}`
        overrides, plus one `SPECTRAL_CHANNELS` (standard resolution) or
        `SPECTRAL_CHANNELS_{n}` table extension per pre-computed resolution.
        Each statistics table covers every detector — any per-detector
        `CHANNEL_MAP_D{n}` override is already baked into its rows, so there
        are no separate per-detector statistics extensions.

        Also backward compatible with two other formats: the plain
        `CHANNEL_MAP` + `SPECTRAL_CHANNELS` calibration format (loads as a
        degenerate `n_subchannels=1` case), and the bundled format (image
        data directly in the primary HDU, no extension name at all -- used
        if no named `CHANNEL_MAP`/`CHANNEL_MAP_D{n}` extension is found,
        mirroring `LVFContourChannelDefinition.from_fits()`'s detection).

        Parameters
        ----------
        hdul : astropy.io.fits.HDUList
            Open FITS HDU list in the LVFChannelDefinitionV2 calibration
            format.
        arrays : list[int], optional
            Arrays to include. Default: [1, 2, 3, 4, 5, 6].

        Returns
        -------
        channel_def : LVFChannelDefinitionV2
            Channel definition bound to the standard-channel resolution (1)
            if available, otherwise the finest resolution. Use
            `at_resolution()` to switch resolutions.

        Raises
        ------
        ValueError
            If no CHANNEL_MAP/CHANNEL_MAP_D{n} or no SPECTRAL_CHANNELS*
            extensions are found.

        """
        det_map_re = re.compile(r'^CHANNEL_MAP_D(\d+)$')
        stats_re = re.compile(r'^SPECTRAL_CHANNELS(?:_(\d+))?$')

        contour_maps: dict = {}
        stats: dict = {}

        for hdu in hdul:
            name = (hdu.name or '').upper()
            if name == 'CHANNEL_MAP':
                contour_maps[None] = hdu.data
                continue
            m = det_map_re.match(name)
            if m:
                contour_maps[int(m.group(1))] = hdu.data
                continue
            m = stats_re.match(name)
            if m:
                divisor = int(m.group(1)) if m.group(1) else 1
                stats[divisor] = Table(hdu.data)
                continue

        if not contour_maps:
            # Bundled format: no named CHANNEL_MAP/CHANNEL_MAP_D{n}
            # extension -- fall back to the first HDU with 2D image data,
            # regardless of name (this may be the primary HDU itself).
            for hdu in hdul:
                if hdu.data is not None and hdu.data.ndim == 2:
                    contour_maps[None] = hdu.data
                    break

        if not contour_maps:
            raise ValueError(
                "HDUList must contain a CHANNEL_MAP or CHANNEL_MAP_D{n} "
                "extension, or a 2D image in some HDU (bundled format)"
            )
        if not stats:
            raise ValueError(
                "HDUList must contain at least one SPECTRAL_CHANNELS "
                "extension"
            )

        # NSUBCHN/NCHAN (written by to_hdul()) are authoritative when
        # present. Files with non-subdivided channels fall back to
        # inferring NCHAN from the stats table, NSUBCHN will be 1.
        primary_header = hdul[0].header
        n_subchannels = (
            int(primary_header['NSUBCHN']) if 'NSUBCHN' in primary_header
            else max(stats)
        )
        n_base_channels = (
            int(primary_header['NCHAN']) if 'NCHAN' in primary_header
            else (int(np.max(stats[1]['SUBCHAN'])) if 1 in stats else n_subchannels)
        )

        return cls(
            contour_maps=contour_maps,
            stats=stats,
            n_subchannels=n_subchannels,
            n_base_channels=n_base_channels,
            arrays=arrays,
            resolution=1 if 1 in stats else n_subchannels,
        )

    @classmethod
    def from_fits(
        cls,
        filepath: str,
        arrays: Optional[list[int]] = None,
    ) -> Self:
        """Create a channel definition from a calibration FITS file.

        See `from_hdul` for the expected extension layout.

        Parameters
        ----------
        filepath : str
            Path to the calibration FITS file.
        arrays : list[int], optional
            Arrays to include. Default: [1, 2, 3, 4, 5, 6].

        Returns
        -------
        channel_def : LVFChannelDefinitionV2
            Channel definition loaded from the FITS file.

        """
        with fits.open(filepath) as hdul:
            return cls.from_hdul(hdul, arrays=arrays)

    @classmethod
    def from_reference(
        cls,
        n_subchannels: int = 1,
        arrays: Optional[list[int]] = None,
    ) -> Self:
        """Build from the bundled standard-channel template and reference table.

        Uses the built-in LVF contour template (`get_lvf_contour_path()`)
        and the hand-curated reference statistics
        (`spectral_channels_reference.make_reference_table()`) -- no
        wavelength model or calibration file needed. Statistics are only
        ever available at standard-channel resolution (1), since the
        reference table has no finer-resolution data; `n_subchannels > 1`
        subdivides the geometry only, for masks/`get_global_channel_numbers` at
        finer resolutions.

        Parameters
        ----------
        n_subchannels : int, optional
            Finest geometric resolution to subdivide the template into.
            Default is 1 (standard channels only, template used as-is).
        arrays : list[int], optional
            Array numbers to include. Default: [1, 2, 3, 4, 5, 6].

        Returns
        -------
        channel_def : LVFChannelDefinitionV2
            Channel definition bound to standard-channel resolution (1),
            with statistics from the reference table.

        """
        # Local import: spectral_channels_reference imports from this
        # module, so importing it at module scope would be circular.
        from .spectral_channels_reference import make_reference_table

        with fits.open(get_lvf_contour_path()) as hdul:
            base_map = hdul[0].data

        channels = list(range(1, 18))
        if n_subchannels > 1:
            contour_map = LVFContourChannelDefinition.subdivide_contour_map(
                base_map, channels=channels, n_subdivisions=n_subchannels
            )
        else:
            contour_map = base_map

        return cls(
            contour_maps={None: contour_map},
            stats={1: make_reference_table()},
            n_subchannels=n_subchannels,
            n_base_channels=len(channels),
            arrays=arrays,
            resolution=1,
        )

    def to_hdul(self) -> fits.HDUList:
        """Convert this channel definition to an HDUList.

        Writes every loaded template (default plus any per-detector
        overrides) and one statistics table extension per pre-computed
        resolution. Each statistics table already reflects any per-detector
        `CHANNEL_MAP_D{n}` override baked in during `build()`, so it is
        written as a single extension covering all detectors.

        Returns
        -------
        hdul : astropy.io.fits.HDUList
            HDU list ready to be written to a FITS file.

        """
        primary = fits.PrimaryHDU()
        primary.header['DATE'] = Time.now().to_value(format='fits')
        primary.header.comments['DATE'] = 'HDU creation time (UTC)'
        if self._n_subchannels > 1:
            primary.header['NSUBCHN'] = (
                self._n_subchannels, 'Sub-channels per standard channel (finest)'
            )
        primary.header['NCHAN'] = (self._n_base_channels, 'Number of standard channels per detector')
        hdus = [primary]

        default_map = self._contour_maps.get(None)
        if default_map is not None:
            hdu = fits.ImageHDU(data=default_map, name='CHANNEL_MAP')
            hdu.header['BUNIT'] = ''
            hdu.header.comments['BUNIT'] = 'Channel number (unitless)'
            hdus.append(hdu)

        for det in sorted(d for d in self._contour_maps if d is not None):
            hdu = fits.ImageHDU(data=self._contour_maps[det], name=f'CHANNEL_MAP_D{det}')
            hdu.header['DETECTOR'] = (det, 'Detector array of the override')
            hdu.header['BUNIT'] = ''
            hdu.header.comments['BUNIT'] = 'Channel number (unitless)'
            hdus.append(hdu)

        for divisor in sorted(self._stats, reverse=True):
            ext_name = 'SPECTRAL_CHANNELS' if divisor == 1 else f'SPECTRAL_CHANNELS_{divisor}'
            hdus.append(fits.BinTableHDU(self._stats[divisor], name=ext_name))

        return fits.HDUList(hdus)

    def _set_resolution(self, resolution: int) -> None:
        """Bind this instance to `resolution` sub-channels per standard channel.

        `resolution` only needs to be a divisor of `n_subchannels` (a
        geometrically valid resolution); it does not need precomputed
        statistics. `get_channel_mask`/`get_global_channel_numbers` work at any
        such resolution. `get_channel_info`/`get_all_channels` fall back to
        NaN-filled/geometric results when this resolution has no entry in
        `stats_resolutions`.
        """
        if resolution not in _divisors(self._n_subchannels):
            raise ValueError(
                f"{resolution} is not a valid resolution for "
                f"n_subchannels={self._n_subchannels} (must be one of "
                f"{_divisors(self._n_subchannels)})"
            )
        self._resolution = resolution
        self._group_size = self._n_subchannels // resolution
        stats_table = self._stats.get(resolution)
        self._stats_index = (
            {
                (int(row['DETECTOR']), int(row['SUBCHAN'])): i
                for i, row in enumerate(stats_table)
            }
            if stats_table is not None else {}
        )

    def at_resolution(self, n_subchannels: int) -> Self:
        """Return a view of this channel definition at a different resolution.

        If `n_subchannels` is already one of `self.available_resolutions`,
        this is an O(1) view switch: shares the already-loaded contour maps,
        pre-computed statistics tables, and cropped-contour cache with
        `self` — no FITS re-parsing or recomputation occurs.

        If `n_subchannels` is *not* already available but `self` is at the
        full-width resolution (`self.available_resolutions == [1]` — e.g.
        loaded from a calibration file that only ever stores standard
        channels), this instead subdivides the stored contour map(s) into
        `n_subchannels` finer sub-channels per standard channel and returns
        a genuinely new instance bound to that resolution. Unlike the O(1)
        case, this recomputes the contour map over every loaded pixel array
        (see `LVFContourChannelDefinition.subdivide_contour_map`). The
        returned instance has no statistics at the new resolution — there
        is no wavelength model available here to derive them, only the
        standard-channel statistics table carried over unchanged — so
        `stats_resolutions` remains `[1]`; `get_channel_info`/
        `spectral_stats` fall back to NaN/`None` at the new resolution
        until `at_resolution(1)` is called again.

        Parameters
        ----------
        n_subchannels : int
            Number of sub-channels per standard channel for the returned
            view.

        Returns
        -------
        view : LVFChannelDefinitionV2
            New instance bound to the requested resolution.

        Raises
        ------
        ValueError
            If `n_subchannels` is not in `self.available_resolutions` and
            `self` is already subdivided (`available_resolutions != [1]`),
            so it cannot be subdivided further.

        Examples
        --------
        >>> quarter = channel_def.at_resolution(4)
        >>> half = channel_def.at_resolution(2)

        Deriving quarter channels from a full-width-only calibration file:

        >>> full = LVFChannelDefinitionV2.from_fits('spectral_channels_v2.fits')
        >>> quarter = full.at_resolution(4)  # subdivides (full.n_subchannels == 1)
        >>> standard = quarter.at_resolution(1)  # O(1) view switch back

        """
        if n_subchannels in _divisors(self._n_subchannels):
            view = copy.copy(self)
            view._set_resolution(n_subchannels)
            return view
        if self._n_subchannels == 1:
            return self._subdivided(n_subchannels)
        raise ValueError(
            f"{n_subchannels} is not a valid resolution for "
            f"n_subchannels={self._n_subchannels} (must be one of "
            f"{_divisors(self._n_subchannels)}), and this instance is "
            "already subdivided (n_subchannels != 1) so it cannot be "
            "subdivided further."
        )

    def _subdivided(self, n_subchannels: int) -> Self:
        """Derive a new, finer-resolution instance from a full-width source.

        Only called from `at_resolution()` when `self._n_subchannels == 1`
        (a genuinely full-width source, e.g. loaded from a calibration file
        that only stores standard channels). Subdivides every loaded
        contour map (default plus any per-detector overrides) via
        `LVFContourChannelDefinition.subdivide_contour_map`, carrying the
        existing (resolution-1) statistics over unchanged.
        """
        channels = list(range(1, self._n_base_channels + 1))
        contour_maps = {
            det: LVFContourChannelDefinition.subdivide_contour_map(
                contour_map, channels=channels, n_subdivisions=n_subchannels
            )
            for det, contour_map in self._contour_maps.items()
        }
        return type(self)(
            contour_maps=contour_maps,
            stats=self._stats,
            n_subchannels=n_subchannels,
            n_base_channels=self._n_base_channels,
            arrays=self.arrays,
            resolution=n_subchannels,
        )

    @property
    def resolution(self) -> int:
        """Active number of sub-channels per standard channel."""
        return self._resolution

    @property
    def available_resolutions(self) -> list[int]:
        """All resolutions the geometry supports (divisors of `n_subchannels`).

        Usable with `at_resolution()`/`get_channel_mask()`/
        `get_global_channel_numbers()` regardless of whether statistics were
        precomputed for them. See `stats_resolutions` for the subset with
        precomputed statistics.
        """
        return _divisors(self._n_subchannels)

    @property
    def stats_resolutions(self) -> list[int]:
        """Resolutions with precomputed statistics.

        A subset of `available_resolutions`.
        """
        return sorted(self._stats.keys(), reverse=True)

    @property
    def n_base_channels(self) -> int:
        """Number of standard channels (e.g. 17), independent of resolution."""
        return self._n_base_channels

    @property
    def n_channels_per_array(self) -> int:
        """Number of channels per detector array at the active resolution.

        Equal to `n_base_channels * resolution` (e.g. 34 for half-channel
        resolution with 17 standard channels).
        """
        return self._n_base_channels * self._resolution

    @property
    def spectral_stats(self) -> Optional[Table]:
        """Spectral statistics table for the active resolution, if precomputed."""
        return self._stats.get(self._resolution)

    def get_all_channels(self) -> list[tuple[int, int]]:
        """Return all (array_id, channel) combinations at the active resolution.

        Derived from precomputed statistics when available for this
        resolution; otherwise enumerated geometrically from `self.arrays`
        and `n_channels_per_array`.
        """
        if self._stats_index:
            return sorted(self._stats_index.keys())
        return [
            (array_id, channel)
            for array_id in self.arrays
            for channel in range(1, self.n_channels_per_array + 1)
        ]

    def _get_contour_map_for_shape(
        self,
        array_id: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get the fine-resolution contour map for `array_id`, cropped to `shape`.

        Uses the detector-specific template if one was loaded for this
        array, otherwise falls back to the default template.
        """
        template_key = array_id if array_id in self._contour_maps else None
        cache_key = (template_key, shape)
        if cache_key not in self._cropped_contour_cache:
            contour_map = self._contour_maps.get(template_key)
            if contour_map is None:
                raise ValueError(
                    f"No channel map available for array {array_id}: no "
                    "detector-specific template and no default template loaded."
                )
            if contour_map.shape != shape:
                contour_map = _crop_to_shape(contour_map, shape)
            self._cropped_contour_cache[cache_key] = contour_map
        return self._cropped_contour_cache[cache_key]

    def _get_contour_stack_for_shape(self, shape: tuple[int, int]) -> np.ndarray:
        """Stack of every configured array's contour map, cached per shape.

        Indexed directly as `stack[array_id]` (index 0 is unused padding,
        or -- when there are no per-detector overrides -- just another
        alias of the single shared map) so callers don't need an
        `array_id - 1` shift/allocation at lookup time. Built once per shape
        (covering `self.arrays`, independent of resolution) so
        `get_global_channel_numbers` can resolve any number of (array_id, x, y)
        measurements with a single compiled-kernel call instead of a
        Python loop over detector arrays.

        When no array has a detector-specific override, every array shares
        the exact same underlying map, so the stack is a read-only
        `numpy.broadcast_to` view (stride 0 on the leading axis) rather than
        `max(self.arrays) + 1` physical copies of identical data. Fancy
        indexing into a broadcast axis is fully supported for reads: with
        stride 0, every `array_id` contributes zero to the computed memory
        offset, so any index along that axis correctly resolves to the one
        shared map. This is only valid because we never write into
        `contour_stack` (`broadcast_to` marks it non-writeable for exactly
        that reason). Per-detector overrides break this -- a single axis
        can't have different real data at different positions without
        actually storing it -- so that case falls back to a materialized
        stack.
        """
        if shape not in self._contour_stack_cache:
            override_ids = [d for d in self._contour_maps if d is not None]
            max_id = max(self.arrays)

            if not override_ids:
                default_map = self._get_contour_map_for_shape(self.arrays[0], shape)
                stack = np.broadcast_to(default_map, (max_id + 1,) + shape)
            else:
                maps = {
                    array_id: self._get_contour_map_for_shape(array_id, shape)
                    for array_id in self.arrays
                }
                dtype = next(iter(maps.values())).dtype
                stack = np.zeros((max_id + 1,) + shape, dtype=dtype)
                for array_id, contour_map in maps.items():
                    stack[array_id] = contour_map

            self._contour_stack_cache[shape] = stack
        return self._contour_stack_cache[shape]

    def get_channel_mask(
        self,
        array_id: int,
        channel: int,
        shape: tuple[int, int],
    ) -> np.ndarray:
        """Get boolean mask for pixels in the specified channel.

        `channel` is interpreted at the active resolution (`self.resolution`).
        Note: the same fine-resolution contour map underlies every
        resolution, so switching resolution via `at_resolution()` does not
        invalidate the cropped-contour cache.
        """
        contour_map = self._get_contour_map_for_shape(array_id, shape)
        base_idx0, bucket0 = divmod(channel - 1, self._resolution)
        fine_start = base_idx0 * self._n_subchannels + bucket0 * self._group_size + 1
        if self._group_size == 1:
            return contour_map == fine_start
        return (contour_map >= fine_start) & (contour_map < fine_start + self._group_size)

    def get_global_channel_numbers(
        self,
        array_ids: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        shape: tuple[int, int] = (2040, 2040),
        *,
        dtype: np.dtype = np.int32,
    ) -> np.ndarray:
        """Global (cross-detector) channel-number lookup.

        Answers "which global channel does this pixel belong to" for many
        (array_id, x, y) measurements at once, at the active resolution.
        Every measurement is resolved by a single call to a compiled,
        parallelized Numba kernel (`_global_channel_numbers_kernel`) over a
        per-shape cached stack of all configured arrays' contour maps (see
        `_get_contour_stack_for_shape`). Maps (array_id, channel)
        to a number unique across all detectors: detector 1's channels
        occupy `[1, n_channels_per_array]`, detector 2's occupy
        `[n_channels_per_array + 1, 2 * n_channels_per_array]`, and so on.

        Parameters
        ----------
        array_ids : numpy.ndarray of int
            Detector array number for each measurement.
        x : numpy.ndarray of float or int
            X pixel coordinate (column, 0-based) for each measurement.
            Mapped to the containing integer pixel via ceil(v - 0.5).
        y : numpy.ndarray of float or int
            Y pixel coordinate (row, 0-based) for each measurement.
            Mapped to the containing integer pixel via ceil(v - 0.5).
        shape : tuple[int, int]
            Shape of the data array as (ny, nx), used to crop the contour
            map consistently with `get_channel_mask`.
        dtype : numpy.dtype, optional
            Integer dtype of the returned array. Default `numpy.int32`,
            which safely covers the global range (up to
            `6 * n_channels_per_array`).

        Returns
        -------
        global_channels : numpy.ndarray
            Global channel number for each measurement, as `dtype`. 0
            where the (clipped) pixel does not belong to any channel.

        Examples
        --------
        >>> global_channels = channel_def.get_global_channel_numbers(
        ...     array_ids=detector, x=x_pix, y=y_pix
        ... )

        """
        # Pinning to one canonical dtype keeps the kernel compiled/cached for a
        # single signature regardless of what dtype a given caller passes.
        array_ids = np.asarray(array_ids).astype(np.uint8, copy=False)
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        ny, nx = shape
        contour_stack = self._get_contour_stack_for_shape(shape)
        max_fine = self._n_base_channels * self._n_subchannels

        global_channels = np.empty(array_ids.shape[0], dtype=np.int32)
        _global_channel_numbers_kernel(
            array_ids, x, y, contour_stack, ny, nx,
            self._n_subchannels, self._resolution, self._group_size,
            max_fine, self.n_channels_per_array, global_channels,
        )
        return global_channels.astype(dtype, copy=False)

    def get_channel_info(
        self,
        array_id: int,
        channel: int,
    ) -> ChannelInfo:
        """Get all spectral statistics for a channel at the active resolution."""
        idx = self._stats_index.get((array_id, channel))
        if idx is None:
            return ChannelInfo(
                detector=array_id,
                subchannel=channel,
                wavelength=np.nan * u.um,
                wavelength_min=np.nan * u.um,
                wavelength_max=np.nan * u.um,
                R=None,
                R_std=None,
                bandwidth=None,
            )

        row = self._stats[self._resolution][idx]

        wl = row['WAVELENGTH']
        wavelength = wl * u.um if not np.isnan(wl) else np.nan * u.um

        wl_min = row['WL_MIN']
        wl_max = row['WL_MAX']
        wavelength_min = wl_min * u.um if not np.isnan(wl_min) else np.nan * u.um
        wavelength_max = wl_max * u.um if not np.isnan(wl_max) else np.nan * u.um

        r_median = row['R']
        r_std = row['R_STD']
        R = float(r_median) if not np.isnan(r_median) else None
        R_std_val = float(r_std) if not np.isnan(r_std) else None

        bandwidth = None
        bw_value = row['BANDWIDTH']
        if not np.isnan(bw_value):
            bandwidth = bw_value * u.um

        return ChannelInfo(
            detector=array_id,
            subchannel=channel,
            wavelength=wavelength,
            wavelength_min=wavelength_min,
            wavelength_max=wavelength_max,
            R=R,
            R_std=R_std_val,
            bandwidth=bandwidth,
        )


def _crop_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop array to target shape.

    Reconciles a wavelength model or contour map whose shape doesn't match
    the data being masked (e.g. built at a different active-area size, or
    against an older/newer calibration convention). Current calibration
    templates and wavelength models are 2040x2040, matching L2 data
    directly, so this is normally a no-op; it remains as a defensive
    fallback for shape mismatches rather than a routine code path.

    Parameters
    ----------
    arr : numpy.ndarray
        Array to crop.
    target_shape : tuple[int, int]
        Target shape as (ny, nx).

    Returns
    -------
    cropped : numpy.ndarray
        Center-cropped array with shape `target_shape`.

    """
    ny_full, nx_full = arr.shape
    ny_target, nx_target = target_shape

    if ny_target > ny_full or nx_target > nx_full:
        raise ValueError(
            f"Target shape {target_shape} larger than array shape {arr.shape}"
        )

    if (ny_target, nx_target) == (ny_full, nx_full):
        return arr

    y_start = (ny_full - ny_target) // 2
    x_start = (nx_full - nx_target) // 2

    return arr[y_start:y_start + ny_target, x_start:x_start + nx_target]
