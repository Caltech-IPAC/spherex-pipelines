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
    "get_lvf_contour_path",
]

from abc import ABC, abstractmethod
from dataclasses import dataclass
from importlib.resources import files
from typing import Iterator, Optional

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.table import Table

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
        channel_map_hdu.header['BUNIT'] = 'um'
        channel_map_hdu.header.comments['BUNIT'] = 'Channel number (unitless)'

        # SPECTRAL_CHANNELS binary table extension
        spectral_hdu = fits.BinTableHDU(self._spectral_stats, name='SPECTRAL_CHANNELS')

        return fits.HDUList([primary, channel_map_hdu, spectral_hdu])

    @staticmethod
    def subdivide_contour_map(
        contour_map: np.ndarray,
        channels: Optional[list[int]] = None,
        n_subdivisions: int = 4,
        output_dir: Optional[str] = None,
    ) -> np.ndarray:
        """Create a new contour map with N subdivisions per channel.

        For each column, divides each channel's y-range into N equal parts
        and assigns sequential subchannel numbers. Pixels outside the
        requested channels are set to 0.

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
            If n_subdivisions is not between 1 and 7.

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

        # For each column, subdivide each channel's y-range
        for x in range(nx):
            col = contour_map[:, x]

            for channel in channels:
                # Find y indices where this channel exists in this column
                y_indices = np.where(col == channel)[0]

                if len(y_indices) == 0:
                    continue

                y_min, y_max = y_indices.min(), y_indices.max()
                y_range = y_max - y_min + 1

                # Assign subdivision based on relative position within y-range
                # Use (y_max - y) so that smaller y values get higher subdivision
                # numbers, matching the wavelength direction (wavelength increases
                # with y, so subchannel 1 should be at highest y within channel)
                for y in y_indices:
                    relative_pos = y_max - y
                    # subdivision index from 0 to n_subdivisions-1
                    subdivision = int(relative_pos * n_subdivisions / y_range)
                    subdivision = min(subdivision, n_subdivisions - 1)

                    # Subchannel numbers start at 1
                    new_subchannel = channel_to_offset[channel] + subdivision + 1
                    new_contour[y, x] = new_subchannel

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
            hdu.writeto(filepath, overwrite=True)

        return new_contour

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
                        r_median = np.median(r_values)
                        r_std = np.std(r_values)
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


def _crop_to_shape(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Center-crop array to target shape.

    Handles the case where the wavelength model or contour map is for the full
    2048x2048 array but the data has blind pixels removed.

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
