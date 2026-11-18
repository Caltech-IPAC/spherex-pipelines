"""Interpolation-based reprojection using reproject_interp."""

from typing import Optional
import numpy as np
from astropy.wcs import WCS

from .base import ReprojectionMethod, reproject_flags_histogram


class InterpReprojection(ReprojectionMethod):
    """Interpolation-based reprojection.

    Uses `reproject.reproject_interp` from the reproject package.
    This method samples input pixels at output pixel locations using
    interpolation.

    Parameters
    ----------
    order : str, optional
        Interpolation order. One of 'nearest-neighbor', 'bilinear',
        'biquadratic', or 'bicubic'. Default is 'nearest-neighbor'.

    Properties
    ----------
    - Fast: Simple interpolation, no geometric calculations
    - Configurable: Multiple interpolation orders available
    - No flux conservation: Values are interpolated, not redistributed
    - Smooth output: Higher orders produce smoother results

    Notes
    -----
    This method does not conserve flux. For flux-conserving reprojection,
    use ExactReprojection instead.

    """

    def __init__(self, order: str = 'nearest-neighbor'):
        """Initialize interpolation reprojection.

        Parameters
        ----------
        order : str, optional
            Interpolation order. One of 'nearest-neighbor', 'bilinear',
            'biquadratic', or 'bicubic'. Default is 'nearest-neighbor'.

        """
        valid_orders = ('nearest-neighbor', 'bilinear', 'biquadratic', 'bicubic')
        if order not in valid_orders:
            raise ValueError(
                f"Invalid order '{order}'. Must be one of: {', '.join(valid_orders)}"
            )
        self._order = order

    @property
    def name(self) -> str:
        return "reproject_interp"

    @property
    def order(self) -> str:
        """Return the interpolation order."""
        return self._order

    @property
    def supports_origin(self) -> bool:
        """Interp method supports origin via CRPIX adjustment."""
        return True

    @property
    def nhits_dtype(self) -> np.dtype:
        """Return nhits dtype: int32 for nearest-neighbor, float32 for others."""
        return np.int32 if self._order == 'nearest-neighbor' else np.float32

    def reproject(
        self,
        data: np.ndarray,
        wcs_in: WCS,
        wcs_out: WCS,
        shape_out: tuple[int, int],
        origin: tuple[int, int] = (0, 0),
        flags: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Reproject using interpolation.

        See base class for parameter documentation.

        Notes
        -----
        For cropped data with origin != (0, 0), astropy's WCS slicing is used
        to adjust the coordinate system. This properly handles CRPIX and SIP
        distortion reference points.

        This implementation handles NaN propagation during interpolation:
        higher-order interpolation (bilinear, biquadratic, bicubic) can cause
        NaN values to "bleed" into neighboring pixels because the interpolation
        kernel samples multiple input pixels. We work around this by:
        1. Replacing NaN with 0 before reprojection
        2. Reprojecting a validity mask (1 where valid, 0 where NaN)
        3. Dividing the reprojected data by the reprojected validity mask

        Flags are reprojected using histogram-style nearest-neighbor mapping
        with bitwise OR accumulation, since interpolation doesn't apply to
        discrete bit flags.

        """
        from reproject import reproject_interp

        # Handle units
        unit = getattr(data, 'unit', None)
        data_values = data.value if unit else np.asarray(data)

        # Handle origin offset using astropy's WCS slicing
        # This properly updates CRPIX and SIP distortion reference points
        y_offset, x_offset = origin
        wcs_in_adjusted = wcs_in
        if x_offset != 0 or y_offset != 0:
            wcs_in_adjusted = wcs_in[y_offset:, x_offset:]

        # Step 1: Replace NaN with 0 to prevent NaN bleeding during interpolation
        nan_mask = np.isnan(data_values)
        data_clean = np.where(nan_mask, 0.0, data_values)

        # Step 2: Create validity mask (1 where valid, 0 where NaN)
        validity_mask = np.where(nan_mask, 0.0, 1.0)

        # Step 3: reproject the cleaned data and validity mask
        combined_input = np.stack([data_clean, validity_mask])
        combined_reprojected, footprint = reproject_interp(
            (combined_input, wcs_in_adjusted),
            wcs_out,
            shape_out=shape_out,
            order=self._order,
            return_footprint=True
        )

        reprojected = combined_reprojected[0]
        footprint = footprint[0]
        reprojected_validity = combined_reprojected[1]
        np.nan_to_num(reprojected_validity, nan=0.0, copy=False)

        # Step 4: Combine footprint with validity to get effective nhits
        # nhits = footprint * reprojected_validity represents the effective
        # coverage accounting for both geometric footprint and valid input pixels
        nhits = footprint * reprojected_validity

        # Convert to appropriate dtype
        if self._order == 'nearest-neighbor':
            nhits = nhits.astype(np.int32)
        else:
            nhits = nhits.astype(np.float32)

        # Step 5: Renormalize by dividing by reprojected validity
        # This corrects for the contribution of NaN (now 0) pixels
        with np.errstate(divide='ignore', invalid='ignore'):
            reprojected = np.where(
                reprojected_validity > 0,
                reprojected / reprojected_validity,
                np.nan
            )

        if unit:
            reprojected = reprojected * unit

        # Handle flags using histogram-style nearest-neighbor mapping
        out_flags = None
        if flags is not None:
            out_flags = reproject_flags_histogram(
                flags, wcs_in, wcs_out, shape_out, origin
            )

        return reprojected, nhits, out_flags
