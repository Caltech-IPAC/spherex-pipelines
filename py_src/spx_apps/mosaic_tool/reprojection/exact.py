"""Exact (flux-conserving) reprojection using spherical polygon intersection."""

from typing import Optional
import numpy as np
from astropy.wcs import WCS

from .base import ReprojectionMethod, reproject_flags_histogram


class ExactReprojection(ReprojectionMethod):
    """Flux-conserving spherical polygon intersection reprojection.

    Uses `reproject.reproject_exact` from the reproject package.
    This method computes exact geometric overlap between input and
    output pixels on the celestial sphere.

    Properties
    ----------
    - Flux conservation: Total flux is preserved
    - Fractional nhits: Hit counts can be non-integer (fractional overlap)
    - Handles distortion: Properly accounts for WCS distortions
    - Slow: O(n*m) comparisons for n input x m output pixels

    Notes
    -----
    This implementation handles the NaN propagation quirk of reproject_exact:
    if any input pixel contributing to an output pixel is NaN, the output
    becomes NaN. We work around this by:
    1. Replacing NaN with 0 before reprojection
    2. Reprojecting a footprint (1 where valid, 0 where NaN)
    3. Dividing the reprojected data by the reprojected footprint

    """

    @property
    def name(self) -> str:
        return "reproject_exact"

    @property
    def supports_origin(self) -> bool:
        """Exact method supports origin via CRPIX adjustment."""
        return True

    @property
    def nhits_dtype(self) -> np.dtype:
        """Exact method returns float32 nhits (fractional overlap)."""
        return np.float32

    def reproject(
        self,
        data: np.ndarray,
        wcs_in: WCS,
        wcs_out: WCS,
        shape_out: tuple[int, int],
        origin: tuple[int, int] = (0, 0),
        flags: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Reproject using flux-conserving spherical polygon intersection.

        See base class for parameter documentation.

        Notes
        -----
        For cropped data with origin != (0, 0), astropy's WCS slicing is used
        to adjust the coordinate system. This properly handles CRPIX and SIP
        distortion reference points.

        Flags are reprojected using histogram-style nearest-neighbor mapping
        with bitwise OR accumulation, since fractional pixel overlap doesn't
        apply to discrete bit flags.

        """
        from reproject import reproject_exact

        # Handle units
        unit = getattr(data, 'unit', None)
        data_values = data.value if unit else np.asarray(data)

        # Handle origin offset using astropy's WCS slicing
        # This properly updates CRPIX and SIP distortion reference points
        y_offset, x_offset = origin
        wcs_in_adjusted = wcs_in
        if x_offset != 0 or y_offset != 0:
            wcs_in_adjusted = wcs_in[y_offset:, x_offset:]

        # Step 1: Replace NaN with 0
        nan_mask = np.isnan(data_values)
        data_clean = np.where(nan_mask, 0.0, data_values)

        # Step 2: Reproject data
        reprojected, _ = reproject_exact(
            (data_clean, wcs_in_adjusted),
            wcs_out,
            shape_out=shape_out,
            return_footprint=True
        )

        # Step 3: Create footprint (1 where valid, 0 where NaN)
        footprint_in = np.where(nan_mask, 0.0, 1.0)

        # Step 4: Reproject the footprint to get nhits
        nhits, _ = reproject_exact(
            (footprint_in, wcs_in_adjusted),
            wcs_out,
            shape_out=shape_out,
            return_footprint=True
        )

        # Step 5: Normalize - set NaN in nhits to 0, then divide
        nhits = np.where(np.isnan(nhits), 0.0, nhits).astype(np.float32)

        with np.errstate(divide='ignore', invalid='ignore'):
            reprojected = np.where(nhits > 0, reprojected / nhits, np.nan)

        if unit:
            reprojected = reprojected * unit

        # Handle flags using histogram-style nearest-neighbor mapping
        out_flags = None
        if flags is not None:
            out_flags = reproject_flags_histogram(
                flags, wcs_in, wcs_out, shape_out, origin
            )

        return reprojected, nhits, out_flags
