__all__ = ["DetectorSpectralInfo", "RegionMaskIterable"]

import numpy as np
import astropy.units as u
from functools import lru_cache
from typing import Iterator, Optional
from numpy.typing import ArrayLike
from .lvf import get_wavelength_mesh
from .spx_param import get_lvf_values


class DetectorSpectralInfo:
    """Per-pixel spectral information for a SPHEREx detector."""

    def __init__(self, detector: int,
                 central_wl: Optional[np.ndarray] = None, central_bw: Optional[np.ndarray] = None,
                 wl_unit=u.micron, bp_unit=u.micron):
        """Per-pixel spectral info for a detector.

        Parameters
        ----------
        detector : `int`
            Detector id 1-6.
        central_wl : `numpy.ndarray`, optional
            Central wavelength for each active pixel of the detector.
            If `None`, central wavelength is based on simulator's Tabular
            Bandpass class.
        central_bw : `numpy.ndarray`, optional
            Central bandwidth for each active pixel of the detector.
            If `None` central bandwidth is derived from central wavelength as
            central wavelength divided by per-detector spectral resolution.
        wl_unit : `str`
            Wavelength unit, default - micron.
        bp_unit : `str`
            Bandpass unit, default - micron.

        """
        self.array_id = detector

        if central_wl is None:
            # use wavelength model based on simulator's Tabular Bandpass
            self.central_wl = get_wavelength_mesh(detector)
        else:
            self.central_wl = central_wl
        self.wl_unit = wl_unit

        lvf = get_lvf_values(detector)
        self.spectral_r = lvf["R"]
        if central_bw is None:
            self.central_bw = None
        else:
            self.central_bw = central_bw
        self.bp_unit = bp_unit

    @classmethod
    @lru_cache(maxsize=6)
    def create(cls, detector: int):
        """Create and cache DetectorSpectralInfo instances.

        Parameters
        ----------
        detector : `int`
            Detector ID (1-6).

        Returns
        -------
        DetectorSpectralInfo
            A cached or newly created instance of DetectorSpectralInfo
            for the given detector.

        """
        return cls(detector)

    def get_wl_mid_fwhm(self, x: ArrayLike, y: ArrayLike) -> tuple[ArrayLike, ArrayLike]:
        """Get wavelength and bandwidth at the center of given pixels.

        Parameters
        ----------
        x : `numpy.ndarray` or `list` [`int`]
            Zero-based indexes in x or columns direction.
        y : `numpy.ndarray` or `list` [`int`]
            Zero-based indexes in y or rows direction.

        Returns
        -------
        result : `tuple` [`~numpy.typing.ArrayLike`, `~numpy.typing.ArrayLike`]
            Central wavelength and bandwidth.

        """
        wl = self.central_wl[y, x]
        if self.central_bw is None:
            # approximate bandwidth using spectral resolution for each array
            bw = wl / self.spectral_r
        else:
            bw = self.central_bw[y, x]
        return wl, bw


class RegionMaskIterable:
    """Iterable that yields (region_id, mask) pairs for wavelength-based regions.

    Divides a detector into N regions based on linear wavelength spacing.
    Each region is defined by wavelength bin edges computed as:
    ``np.linspace(wl_min, wl_max, num_regions + 1)``

    Parameters
    ----------
    central_wl : numpy.ndarray
        2D array of per-pixel central wavelengths (e.g., from DetectorSpectralInfo
        or WavelengthModel).
    num_regions : int
        Number of wavelength regions to divide the detector into.
    region_list : list[int], optional
        If provided, only iterate over these region indices.
        All elements must be in ``range(num_regions)``.

    Attributes
    ----------
    central_wl : numpy.ndarray
        The wavelength array used for region computation.
    num_regions : int
        Number of wavelength regions.
    edges : numpy.ndarray
        Array of wavelength bin edges (length num_regions + 1).

    Examples
    --------
    Using with DetectorSpectralInfo:

    >>> from spx_data import RegionMaskIterable
    >>> region_iter = RegionMaskIterable(
    ...     central_wl=spectral_wcs["CWAVE"].data,
    ...     num_regions=64
    ... )
    >>> for region_id, mask in region_iter:
    ...     flux_in_region = data[mask]

    """

    def __init__(
        self,
        central_wl: np.ndarray,
        num_regions: int,
        region_list: Optional[list[int]] = None,
    ):
        if central_wl.ndim != 2:
            raise ValueError(f"central_wl must be 2D, got shape {central_wl.shape}")

        self.central_wl = central_wl
        self.num_regions = num_regions
        self.region_list = set(region_list) if region_list else None

        # Validate region_list elements are in valid range
        if self.region_list is not None:
            invalid = [r for r in self.region_list if r < 0 or r >= num_regions]
            if invalid:
                raise ValueError(
                    f"region_list contains invalid region IDs {invalid}. "
                    f"All elements must be in range(0, {num_regions})."
                )

        # Compute wavelength bin edges once
        wl_min, wl_max = np.nanmin(central_wl), np.nanmax(central_wl)
        self.edges = np.linspace(wl_min, wl_max, num_regions + 1)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the wavelength array (ny, nx)."""
        return self.central_wl.shape

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over (region_id, mask) pairs.

        Yields
        ------
        region_id : int
            Index of the wavelength region (0 to num_regions-1).
        mask : numpy.ndarray
            Boolean mask where True indicates pixels in this region.

        """
        wl = self.central_wl
        edges = self.edges
        region_list = self.region_list

        for region_id in range(self.num_regions):
            if region_list is not None and region_id not in region_list:
                continue
            # Include upper edge for last region to ensure all pixels are assigned
            if region_id == self.num_regions - 1:
                mask = (wl >= edges[region_id]) & (wl <= edges[region_id + 1])
            else:
                mask = (wl >= edges[region_id]) & (wl < edges[region_id + 1])
            yield region_id, mask

    def __len__(self) -> int:
        """Return the number of regions that will be yielded."""
        if self.region_list is not None:
            return len(self.region_list)
        return self.num_regions
