"""Histogram-based (pointillist) reprojection."""

from typing import Optional
import numpy as np
from astropy.wcs import WCS

from .base import ReprojectionMethod


class HistogramReprojection(ReprojectionMethod):
    """Histogram-based (pointillist) reprojection.

    Maps each input pixel center to the output grid using WCS transformations,
    then bins values using 2D histogram. This is equivalent to nearest-neighbor
    assignment from input to output.

    Properties
    ----------
    - Fast: Simple histogram, no geometric calculations
    - Integer nhits: Hit counts are always integers (pixel counts)
    - No smoothing: Each input pixel goes to exactly one output pixel
    - No noise correlation: Independent pixel assignments
    - Gaps possible: Some output pixels may have no input data

    Notes
    -----
    This is a standalone implementation matching the algorithm in
    SPHEREx_L4GF_map_maker.map_maker_tools.reproject_tools.reproject_hist.
    It has no dependency on the map_maker package.

    Algorithm
    ---------
    1. Get input pixel centers as (x, y) coordinates
    2. Transform to sky coordinates (RA, Dec) via input WCS
    3. Transform sky to output pixel coordinates via output WCS
    4. Bin values using histogram2d with edges at pixel boundaries (-0.5 offset)
    5. Return average = sum / count

    """

    @property
    def name(self) -> str:
        return "reproject_hist"

    @property
    def supports_origin(self) -> bool:
        """Histogram method fully supports origin parameter."""
        return True

    @property
    def nhits_dtype(self) -> np.dtype:
        """Histogram method returns int32 nhits (integer counts)."""
        return np.int32

    def reproject(
        self,
        data: np.ndarray,
        wcs_in: WCS,
        wcs_out: WCS,
        shape_out: tuple[int, int],
        origin: tuple[int, int] = (0, 0),
        flags: Optional[np.ndarray] = None
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Reproject using histogram binning.

        See base class for parameter documentation.

        Notes
        -----
        This implementation exactly matches reproject_tools.reproject_hist:
        - Uses histogram2d with bin edges at -0.5, 0.5, 1.5, ... (pixel boundaries)
        - histogram2d returns shape (nx, ny), so we transpose the result
        - NaN pixels in input are excluded before histogramming

        Flags are reprojected using the same pixel mapping but accumulated
        using bitwise OR instead of sum/average.

        """
        # Handle units
        unit = getattr(data, 'unit', None)
        data_values = data.value if unit else np.asarray(data)

        # Shape follows numpy convention: (y, x) = (rows, cols)
        ny_in, nx_in = data_values.shape
        ny_out, nx_out = shape_out
        n_pixels = ny_out * nx_out

        # Create input pixel coordinate grids with origin offset applied
        # Using float32 to reduce memory (pixel coords don't need float64 precision)
        y_offset, x_offset = origin
        iy_in, ix_in = np.indices((ny_in, nx_in), dtype=np.float32)
        iy_in += y_offset
        ix_in += x_offset

        # Transform input pixel coordinates to sky coordinates
        # WCS uses (x, y) order
        ra, dec = wcs_in.all_pix2world(ix_in, iy_in, 0)
        del iy_in, ix_in  # Free coordinate arrays

        # Transform sky coordinates to output pixel coordinates
        # Reuse ra, dec arrays for output coordinates
        ix_out, iy_out = wcs_out.all_world2pix(ra, dec, 0)
        del ra, dec  # Free sky coordinate arrays

        # Round to nearest integer pixel (bin index) - on full arrays
        ix_idx = np.floor(ix_out + 0.5).astype(np.int32).ravel()
        iy_idx = np.floor(iy_out + 0.5).astype(np.int32).ravel()
        del ix_out, iy_out

        # Create bounds mask for pixels within output grid
        bounds_mask = (
            (ix_idx >= 0) & (ix_idx < nx_out)
            & (iy_idx >= 0) & (iy_idx < ny_out)
        )

        # Create combined mask: valid (non-NaN) AND within bounds
        data_flat = data_values.ravel()
        data_mask = bounds_mask & ~np.isnan(data_flat)

        # Handle empty result case
        if not data_mask.any():
            reprojected = np.full(shape_out, np.nan, dtype=np.float32)
            if unit:
                reprojected = reprojected * unit
            out_flags = np.zeros(shape_out, dtype=np.int32) if flags is not None else None
            return reprojected, np.zeros(shape_out, dtype=self.nhits_dtype), out_flags

        # Extract valid data pixels
        ix_data = ix_idx[data_mask]
        iy_data = iy_idx[data_mask]
        data_valid = data_flat[data_mask]
        del data_flat

        # Convert 2D coordinates to 1D flat indices for data
        flat_indices_data = np.ravel_multi_index(
            (iy_data, ix_data),
            (ny_out, nx_out)
        )
        del ix_data, iy_data

        # Bin counts (nhits)
        nhits = np.bincount(flat_indices_data, minlength=n_pixels).astype(np.int32)
        nhits = nhits.reshape(shape_out)

        # Bin weighted values (accumulator)
        acc = np.bincount(flat_indices_data, weights=data_valid, minlength=n_pixels)
        acc = acc.astype(np.float32).reshape(shape_out)
        del flat_indices_data, data_valid

        # Compute average in-place (reuse acc array for result)
        with np.errstate(divide='ignore', invalid='ignore'):
            acc /= nhits  # In-place division

        if unit:
            acc = acc * unit

        # Handle flags if provided
        out_flags = None
        if flags is not None:
            flags_flat = np.asarray(flags, dtype=np.int32).ravel()

            # Extract pixels within bounds (flags don't use NaN masking)
            ix_flags = ix_idx[bounds_mask]
            iy_flags = iy_idx[bounds_mask]
            flags_valid = flags_flat[bounds_mask]
            del flags_flat

            # Convert to flat indices
            flat_indices_flags = np.ravel_multi_index(
                (iy_flags, ix_flags),
                (ny_out, nx_out)
            )
            del ix_flags, iy_flags

            # Accumulate flags using bitwise OR
            out_flags = np.zeros(n_pixels, dtype=np.int32)
            np.bitwise_or.at(out_flags, flat_indices_flags, flags_valid)
            out_flags = out_flags.reshape(shape_out)
            del flat_indices_flags, flags_valid

        del ix_idx, iy_idx, bounds_mask, data_mask

        return acc, nhits, out_flags
