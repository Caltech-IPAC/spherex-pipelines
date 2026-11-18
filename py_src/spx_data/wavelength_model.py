"""Wavelength models for SPHEREx detector arrays.

This module provides abstract and concrete implementations of wavelength models
that map detector pixels to their central wavelengths.
"""

__all__ = [
    "WavelengthModel",
    "TabularWavelengthModel",
    "ArrayWavelengthModel",
]

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from astropy.io import fits

from . import spx_param


class WavelengthModel(ABC):
    """Abstract base class for wavelength models.

    A wavelength model provides per-pixel wavelength values for SPHEREx
    detector arrays. Different implementations support various data sources:

    - `TabularWavelengthModel`: Uses SPHEREx_InstrumentSimulator Tabular_Bandpass
    - `ArrayWavelengthModel`: Uses wavelength arrays from FITS files

    Optionally, implementations may also provide per-pixel bandwidth values
    via the `get_bandwidth_array()` method.

    See Also
    --------
    TabularWavelengthModel : Uses SPHEREx_InstrumentSimulator.
    ArrayWavelengthModel : Uses wavelength arrays from FITS files.

    """

    @property
    @abstractmethod
    def array_ids(self) -> list[int]:
        """List of array IDs for which wavelength data is available."""
        pass

    @abstractmethod
    def get_wavelength_array(self, array_id: int) -> np.ndarray:
        """Get per-pixel wavelength array for a detector array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        wavelength_array : numpy.ndarray
            2D array of wavelengths in microns with shape (ny, nx).
            The array is in sky coordinates (consistent across all arrays).

        """
        pass

    @abstractmethod
    def get_wavelength_range(self, array_id: int) -> tuple[float, float]:
        """Get wavelength range for an array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        wl_range : tuple[float, float]
            (wavelength_min, wavelength_max) in microns.

        """
        pass

    def get_shape(self, array_id: int) -> tuple[int, int]:
        """Get the shape of the wavelength array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        shape : tuple[int, int]
            Shape as (ny, nx).

        """
        return self.get_wavelength_array(array_id).shape

    def get_bandwidth_array(self, array_id: int) -> Optional[np.ndarray]:
        """Get per-pixel bandwidth array for a detector array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        bandwidth_array : numpy.ndarray or None
            2D array of bandwidths in microns with shape (ny, nx).
            Returns None if bandwidth data is not available.

        """
        return None

    def has_bandwidth(self) -> bool:
        """Check if bandwidth data is available.

        Returns
        -------
        bool
            True if bandwidth arrays are available, False otherwise.

        """
        return False


class TabularWavelengthModel(WavelengthModel):
    """Wavelength model using SPHEREx Tabular_Bandpass.

    Uses the Tabular_Bandpass class from SPHEREx_InstrumentSimulator
    to compute per-pixel wavelengths based on the Linear Variable Filter
    (LVF) calibration.

    This class is implemented as a singleton - multiple calls to the
    constructor return the same instance. This avoids redundant loading
    of the Tabular_Bandpass calibration data.

    Notes
    -----
    Requires SPHEREx_InstrumentSimulator package to be installed.

    For MWIR detectors (arrays 4-6), the x-axis is flipped to convert from
    device coordinates to sky coordinates, ensuring consistent orientation
    across all arrays.

    Examples
    --------
    >>> wl_model1 = TabularWavelengthModel()
    >>> wl_model2 = TabularWavelengthModel()
    >>> wl_model1 is wl_model2  # Same instance
    True
    >>> wl_array = wl_model1.get_wavelength_array(array_id=1)
    >>> print(wl_array.shape)
    (2040, 2040)

    """

    _instance: Optional['TabularWavelengthModel'] = None

    def __new__(cls) -> 'TabularWavelengthModel':
        """Return the singleton instance, creating it if necessary."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        # Only initialize once
        if self._initialized:
            return
        self._initialized = True

        self._npix_x = spx_param.get_npix_x_active()
        self._npix_y = spx_param.get_npix_y_active()
        self._lvf_bandpass: Optional[object] = None
        self._cache: dict[int, np.ndarray] = {}

    @property
    def array_ids(self) -> list[int]:
        """List of array IDs for which wavelength data is available."""
        return [1, 2, 3, 4, 5, 6]

    def _get_lvf(self):
        """Lazy initialization of Tabular_Bandpass."""
        if self._lvf_bandpass is None:
            try:
                from SPHEREx_InstrumentSimulator import Tabular_Bandpass
                self._lvf_bandpass = Tabular_Bandpass()
            except ImportError as e:
                raise ImportError(
                    "TabularWavelengthModel requires SPHEREx_InstrumentSimulator."
                ) from e
        return self._lvf_bandpass

    def get_wavelength_array(self, array_id: int) -> np.ndarray:
        """Get per-pixel wavelength array for a detector array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        wavelength_array : numpy.ndarray
            2D array of wavelengths in microns with shape (ny, nx).
            Array is in sky coordinates.

        """
        if array_id not in self._cache:
            lvf = self._get_lvf()

            # Create pixel index grids
            xi_mesh, yi_mesh = np.meshgrid(
                np.arange(self._npix_x),
                np.arange(self._npix_y)
            )

            # Tabular_Bandpass expects pixel indices offset by blind pixels
            ref_pix = spx_param.get_npix_blind()
            wl_array = lvf(
                xi_mesh + ref_pix,
                yi_mesh + ref_pix,
                array=array_id,
                central_bandpass_only=True
            )[0].astype(np.float32)

            # Flip x-axis for MWIR detectors to convert to sky coordinates
            if array_id > 3:
                wl_array = wl_array[:, ::-1]

            self._cache[array_id] = wl_array

        return self._cache[array_id]

    def get_wavelength_range(self, array_id: int) -> tuple[float, float]:
        """Get wavelength range for an array in microns."""
        wl = self.get_wavelength_array(array_id)
        return float(np.nanmin(wl)), float(np.nanmax(wl))

    def clear_cache(self):
        """Clear cached wavelength arrays to free memory."""
        self._cache.clear()


class ArrayWavelengthModel(WavelengthModel):
    """Wavelength model using wavelength arrays from FITS files.

    This allows using wavelength calibration products delivered as FITS files.
    Handles the DETCOORD convention for coordinate orientation:

    - DETCOORD='sky': No transformation needed
    - DETCOORD='device' or absent: X-axis flip for MWIR detectors (arrays 4-6)

    Optionally supports per-pixel bandwidth arrays from the same calibration
    files (CBAND extension).

    Parameters
    ----------
    wavelength_arrays : dict[int, numpy.ndarray]
        Dictionary mapping array_id (1-6) to 2D wavelength array in microns.
    bandwidth_arrays : dict[int, numpy.ndarray], optional
        Dictionary mapping array_id (1-6) to 2D bandwidth array in microns.
        If not provided, bandwidth methods return None.
    detcoord : dict[int, str], optional
        Dictionary mapping array_id to DETCOORD value ('sky' or 'device').
        If not provided, assumes 'device' (flip needed for MWIR).

    Examples
    --------
    >>> # Load from FITS files
    >>> wl_model = ArrayWavelengthModel.from_fits(
    ...     'calibration/wavelength_array_{array_id}.fits'
    ... )
    >>> wl_array = wl_model.get_wavelength_array(1)

    >>> # Provide arrays directly
    >>> wl_arrays = {1: wl_array_1, 2: wl_array_2, ...}
    >>> wl_model = ArrayWavelengthModel(wl_arrays)

    """

    def __init__(
        self,
        wavelength_arrays: dict[int, np.ndarray],
        bandwidth_arrays: Optional[dict[int, np.ndarray]] = None,
        detcoord: Optional[dict[int, str]] = None,
    ):
        self._arrays: dict[int, np.ndarray] = {}
        self._bandwidth_arrays: dict[int, np.ndarray] = {}
        self._detcoord = detcoord or {}

        for array_id, wl_array in wavelength_arrays.items():
            if not isinstance(wl_array, np.ndarray):
                wl_array = np.asarray(wl_array)
            if wl_array.ndim != 2:
                raise ValueError(
                    f"Wavelength array for array {array_id} must be 2D, "
                    f"got shape {wl_array.shape}"
                )

            # Apply x-flip for MWIR if needed based on DETCOORD
            coord = self._detcoord.get(array_id, 'device')
            if array_id > 3 and coord != 'sky':
                wl_array = wl_array[:, ::-1]

            self._arrays[array_id] = wl_array.astype(np.float32)

        # Process bandwidth arrays with same coordinate transform
        if bandwidth_arrays:
            for array_id, bw_array in bandwidth_arrays.items():
                if not isinstance(bw_array, np.ndarray):
                    bw_array = np.asarray(bw_array)
                if bw_array.ndim != 2:
                    raise ValueError(
                        f"Bandwidth array for array {array_id} must be 2D, "
                        f"got shape {bw_array.shape}"
                    )

                # Apply same x-flip for MWIR as wavelength arrays
                coord = self._detcoord.get(array_id, 'device')
                if array_id > 3 and coord != 'sky':
                    bw_array = bw_array[:, ::-1]

                self._bandwidth_arrays[array_id] = bw_array.astype(np.float32)

    @property
    def array_ids(self) -> list[int]:
        """List of array IDs for which wavelength data is available."""
        return sorted(self._arrays.keys())

    def get_wavelength_array(self, array_id: int) -> np.ndarray:
        """Get per-pixel wavelength array for a detector array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        wavelength_array : numpy.ndarray
            2D array of wavelengths in microns. Array is in sky coordinates.

        Raises
        ------
        KeyError
            If wavelength array for the specified array was not provided.

        """
        if array_id not in self._arrays:
            available = sorted(self._arrays.keys())
            raise KeyError(
                f"No wavelength array provided for array {array_id}. "
                f"Available arrays: {available}"
            )
        return self._arrays[array_id]

    def get_wavelength_range(self, array_id: int) -> tuple[float, float]:
        """Get wavelength range for an array in microns."""
        wl = self.get_wavelength_array(array_id)
        return float(np.nanmin(wl)), float(np.nanmax(wl))

    def get_bandwidth_array(self, array_id: int) -> Optional[np.ndarray]:
        """Get per-pixel bandwidth array for a detector array.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        bandwidth_array : numpy.ndarray or None
            2D array of bandwidths in microns. Array is in sky coordinates.
            Returns None if bandwidth was not provided for this array.

        """
        return self._bandwidth_arrays.get(array_id)

    def has_bandwidth(self) -> bool:
        """Check if bandwidth data is available."""
        return bool(self._bandwidth_arrays)

    def get_bandwidth_range(self, array_id: int) -> Optional[tuple[float, float]]:
        """Get bandwidth range for an array in microns.

        Parameters
        ----------
        array_id : int
            Detector array number (1-6).

        Returns
        -------
        bw_range : tuple[float, float] or None
            (bandwidth_min, bandwidth_max) in microns.
            Returns None if bandwidth not available for this array.

        """
        bw = self._bandwidth_arrays.get(array_id)
        if bw is None:
            return None
        return float(np.nanmin(bw)), float(np.nanmax(bw))

    @classmethod
    def from_fits(
        cls,
        file_pattern: str,
        arrays: Optional[list[int]] = None,
    ) -> 'ArrayWavelengthModel':
        """Create wavelength model from FITS files.

        Parameters
        ----------
        file_pattern : str
            File pattern with {array_id} placeholder.
            Example: 'wavelength_array_{array_id}.fits'
        arrays : list[int], optional
            Array numbers to load. Default is [1, 2, 3, 4, 5, 6].

        Returns
        -------
        model : ArrayWavelengthModel
            Wavelength model with arrays loaded from FITS files.

        Notes
        -----
        The DETCOORD keyword is read from each FITS header to determine
        whether x-axis flipping is needed for MWIR detectors.

        If the FITS file contains a CBAND extension (pixel-center bandwidth),
        it will be loaded alongside the wavelength data.

        Examples
        --------
        >>> wl_model = ArrayWavelengthModel.from_fits(
        ...     'calibration/wl_array_{array_id}.fits',
        ...     arrays=[1, 2, 3, 4, 5, 6]
        ... )

        """
        if arrays is None:
            arrays = [1, 2, 3, 4, 5, 6]

        wl_arrays = {}
        bw_arrays = {}
        detcoord = {}

        print(f"Loading Wavelength for {file_pattern}")
        for array_id in arrays:
            filepath = file_pattern.format(array_id=array_id)
            with fits.open(filepath) as hdul:
                if len(hdul) > 1:
                    ext = "CWAVE"  # wavelength extension in spectral_wcs
                else:
                    # fallback to the primary HDU
                    ext = 0
                wl_arrays[array_id] = hdul[ext].data

                # Read DETCOORD from header, default to 'device' if absent
                detcoord[array_id] = hdul[ext].header.get('DETCOORD', 'device')

                # Load bandwidth if available
                if "CBAND" in hdul:
                    bw_arrays[array_id] = hdul["CBAND"].data

        return cls(
            wl_arrays,
            bandwidth_arrays=bw_arrays if bw_arrays else None,
            detcoord=detcoord
        )
