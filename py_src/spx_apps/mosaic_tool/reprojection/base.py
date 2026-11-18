"""Abstract base class for reprojection methods."""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from astropy.wcs import WCS


def reproject_flags_histogram(
    flags: np.ndarray,
    wcs_in: WCS,
    wcs_out: WCS,
    shape_out: tuple[int, int],
    origin: tuple[int, int] = (0, 0)
) -> np.ndarray:
    """Reproject flags using histogram-style mapping with bitwise OR.

    This is a standalone implementation for flag reprojection that uses
    nearest-neighbor pixel mapping and bitwise OR accumulation. It is used
    by reprojection methods that rely on external libraries (reproject_exact,
    reproject_interp) which don't support bitwise flag accumulation.

    Parameters
    ----------
    flags : numpy.ndarray
        Input 2D flag array with shape (ny, nx), dtype int32.
    wcs_in : astropy.wcs.WCS
        Input WCS.
    wcs_out : astropy.wcs.WCS
        Output WCS.
    shape_out : tuple of int
        Output shape as (ny_out, nx_out).
    origin : tuple of int
        Origin offset (y_offset, x_offset).

    Returns
    -------
    out_flags : numpy.ndarray
        Reprojected flags with shape `shape_out`, dtype int32.

    """
    flags = np.asarray(flags, dtype=np.int32)
    ny_in, nx_in = flags.shape
    ny_out, nx_out = shape_out
    n_pixels = ny_out * nx_out

    # Create input pixel coordinate grids with origin offset
    y_offset, x_offset = origin
    iy_in, ix_in = np.indices((ny_in, nx_in), dtype=np.float32)
    iy_in += y_offset
    ix_in += x_offset

    # Transform to sky coordinates then to output pixel coordinates
    ra, dec = wcs_in.all_pix2world(ix_in, iy_in, 0)
    del iy_in, ix_in
    ix_out, iy_out = wcs_out.all_world2pix(ra, dec, 0)
    del ra, dec

    # Round to nearest integer pixel
    ix_idx = np.floor(ix_out + 0.5).astype(np.int32).ravel()
    iy_idx = np.floor(iy_out + 0.5).astype(np.int32).ravel()
    del ix_out, iy_out

    # Filter to pixels within output bounds
    bounds_mask = (
        (ix_idx >= 0) & (ix_idx < nx_out)
        & (iy_idx >= 0) & (iy_idx < ny_out)
    )

    if not bounds_mask.any():
        return np.zeros(shape_out, dtype=np.int32)

    flags_flat = flags.ravel()
    ix_valid = ix_idx[bounds_mask]
    iy_valid = iy_idx[bounds_mask]
    flags_valid = flags_flat[bounds_mask]
    del ix_idx, iy_idx, flags_flat, bounds_mask

    # Convert to flat indices
    flat_indices = np.ravel_multi_index(
        (iy_valid, ix_valid),
        (ny_out, nx_out)
    )
    del ix_valid, iy_valid

    # Accumulate using bitwise OR
    out_flags = np.zeros(n_pixels, dtype=np.int32)
    np.bitwise_or.at(out_flags, flat_indices, flags_valid)

    return out_flags.reshape(shape_out)


class ReprojectionMethod(ABC):
    """Abstract base class for reprojection methods.

    Subclasses must implement the `reproject` method to transform
    image data from one WCS grid to another.

    See Also
    --------
    ExactReprojection : Flux-conserving spherical polygon intersection.
    HistogramReprojection : Fast histogram-based reprojection.
    InterpReprojection : Interpolation-based reprojection.

    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this method."""
        pass

    @property
    def supports_origin(self) -> bool:
        """Whether this method correctly handles the origin parameter.

        Methods that return False will receive the full (uncropped) input
        data when bounding box optimization is enabled.
        """
        return False

    @property
    @abstractmethod
    def nhits_dtype(self) -> np.dtype:
        """Numpy dtype for the nhits array returned by this method."""
        pass

    @abstractmethod
    def reproject(
        self,
        data: np.ndarray,
        wcs_in: WCS,
        wcs_out: WCS,
        shape_out: tuple[int, int],
        origin: tuple[int, int] = (0, 0),
        flags: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Reproject image data to output grid.

        Parameters
        ----------
        data : numpy.ndarray
            Input 2D data array with shape (ny, nx). May have astropy units.
        wcs_in : astropy.wcs.WCS
            Input WCS defining the coordinate system of `data`.
        wcs_out : astropy.wcs.WCS
            Output WCS defining the target coordinate system.
        shape_out : tuple of int
            Output shape as (ny_out, nx_out).
        origin : tuple of int, optional
            Origin offset (y_offset, x_offset) for the input data within the
            full frame. Used when `data` is a crop of a larger image. The pixel
            at data[0, 0] corresponds to pixel (y_offset, x_offset) in the full
            frame. This is important for WCS with SIP distortion coefficients.
            Default is (0, 0) meaning data starts at the origin.
        flags : numpy.ndarray, optional
            Input 2D flag array with shape (ny, nx), dtype int32. If provided,
            flags are reprojected using bitwise OR accumulation and returned
            as the third element of the result tuple.

        Returns
        -------
        reprojected : numpy.ndarray
            Reprojected data with shape `shape_out`. Same units as input
            if input had units. Contains NaN where no input data contributed.
        nhits : numpy.ndarray
            Weight/hit map with shape `shape_out`. For histogram-based methods,
            this is an integer count. For exact methods, this may be fractional
            representing overlap area.
        reprojected_flags : numpy.ndarray or None
            Reprojected flags with shape `shape_out`, dtype int32. Contains
            bitwise OR of all input flags that contributed to each output pixel.
            None if `flags` parameter was not provided.

        """
        pass
