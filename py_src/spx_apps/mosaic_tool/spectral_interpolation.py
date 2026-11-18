"""Spectral interpolation for stripe reduction in mosaics.

Extended sources with strong spectral features (e.g., emission lines from nebulae)
create stripe-like artifacts in mosaics when images from multiple roll angles are
coadded. This module provides spectral interpolation to reduce these artifacts
while preserving the emission signal.

The key insight is that stripe artifacts are *incoherent* across channels (different
pattern in each), while the true emission signal is *coherent* (smooth spectral trend).
Interpolation preserves coherent signals and averages out incoherent artifacts.
"""

__all__ = ["InterpConfig", "SpectralInterpolator"]

import logging
import numpy as np
from scipy.interpolate import PchipInterpolator
from typing import Optional
from dataclasses import dataclass
import astropy.units as u

from spx_data import ChannelInfo
from .results import ChannelMosaic


@dataclass
class InterpConfig:
    """Configuration for spectral interpolation.

    Parameters
    ----------
    n_interp : float
        Number of resolution widths to include in interpolation.
        Channels within ±n_interp×Δλ of target are used. Default 2.5.
    n_req : float
        Minimum distance (in resolution widths) within which at least
        one channel must contribute. Default 0.55.
    interp_method : str
        Interpolation method. Currently only 'pchip' supported.

    """

    n_interp: float = 2.5
    n_req: float = 0.55
    interp_method: str = "pchip"


class SpectralInterpolator:
    """Interpolate across spectral channels to reduce stripe artifacts.

    This class combines multiple channel mosaics via spectral interpolation
    to produce a mosaic at a specific target wavelength. By averaging across
    channels (which have different stripe patterns), stripe artifacts are
    reduced while preserving the emission signal.

    The stripe artifacts occur because the LVF maps wavelength to detector
    position. For an extended source with emission at a specific wavelength,
    images at different roll angles sample the emission differently. When
    coadded in a single channel, this creates stripe patterns. By interpolating
    across channels, the different stripe patterns are averaged out.

    Parameters
    ----------
    channel_mosaics : dict
        Mapping of channel_id to ChannelMosaic objects.
        Keys are (array_id, subchannel) tuples.
    config : InterpConfig
        Interpolation configuration.

    Examples
    --------
    >>> from spx_apps.mosaic_tool.spectral_interpolation import (
    >>>     SpectralInterpolator, InterpConfig)
    >>> config = InterpConfig(n_interp=2.5, n_req=0.55)
    >>> interpolator = SpectralInterpolator(channel_mosaics, config)
    >>> result = interpolator.interpolate(target_wl=4.80, array_id=6)

    """

    def __init__(
        self, channel_mosaics: dict[tuple[int, int], ChannelMosaic], config: Optional[InterpConfig] = None
    ):
        self.channel_mosaics = channel_mosaics
        self.config = config or InterpConfig()

        # Build sorted list of (wavelength, channel_id) pairs
        self._wl_channel_pairs = sorted(
            [(self._get_wavelength(ch_id), ch_id) for ch_id in channel_mosaics.keys()], key=lambda x: x[0]
        )

    def _get_wavelength(self, channel_id: tuple[int, int]) -> float:
        """Get wavelength in microns for a channel."""
        mosaic = self.channel_mosaics[channel_id]
        return mosaic.channel_info.wavelength.to(u.um).value

    def get_spectral_resolution(self, channel_id: tuple[int, int]) -> float:
        """Get spectral resolution R for a channel from ChannelInfo."""
        mosaic = self.channel_mosaics[channel_id]
        return mosaic.channel_info.R

    def _select_channels(self, target_wl: float, array_id: int) -> list[tuple[float, tuple[int, int]]]:
        """Select channels within interpolation range of target wavelength.

        Only includes channels from the same array (same spectral resolution).
        Uses actual R from ChannelInfo for each channel to compute the
        interpolation range.

        Parameters
        ----------
        target_wl : float
            Target wavelength in microns.
        array_id : int
            Array ID (1-6) for selecting channels.

        Returns
        -------
        list of (wavelength, channel_id)
            Selected channels sorted by wavelength.

        """
        selected = []
        for wl, ch_id in self._wl_channel_pairs:
            ch_array, _ = ch_id
            if ch_array != array_id:
                continue

            # Get actual R for this channel from ChannelInfo
            R = self.get_spectral_resolution(ch_id)
            delta_wl = target_wl / R
            wl_range = self.config.n_interp * delta_wl

            if abs(wl - target_wl) <= wl_range:
                selected.append((wl, ch_id))

        return selected

    def _validate_spectrum(
        self, wavelengths: np.ndarray, values: np.ndarray, target_wl: float, delta_wl: float
    ) -> bool:
        """Check if spectrum meets interpolation requirements.

        Requirements:
        1. At least 2 valid (non-NaN) data points
        2. At least one point within n_req resolution widths of target
        3. At least one point below and one above target (for interpolation)

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength values for each channel.
        values : np.ndarray
            Flux values for each channel.
        target_wl : float
            Target wavelength in microns.
        delta_wl : float
            Resolution width at target wavelength.

        Returns
        -------
        bool
            True if spectrum is valid for interpolation.

        """
        valid_mask = ~np.isnan(values)
        n_valid = np.sum(valid_mask)

        if n_valid < 2:
            return False

        valid_wls = wavelengths[valid_mask]

        # Check: at least one point within n_req of target
        n_req = self.config.n_req
        near_target = np.any(np.abs(valid_wls - target_wl) <= n_req * delta_wl)
        if not near_target:
            return False

        # Check: at least one point below and one above target
        has_below = np.any(valid_wls < target_wl)
        has_above = np.any(valid_wls > target_wl)
        if not (has_below and has_above):
            return False

        return True

    def interpolate_pixel(
        self, wavelengths: np.ndarray, values: np.ndarray, target_wl: float, delta_wl: float
    ) -> float:
        """Interpolate spectrum at a single pixel to target wavelength.

        Uses PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
        interpolation which is monotonic and avoids overshoots.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength values for each channel.
        values : np.ndarray
            Flux values for each channel.
        target_wl : float
            Target wavelength in microns.
        delta_wl : float
            Resolution width at target wavelength.

        Returns
        -------
        float
            Interpolated value at target wavelength, or NaN if invalid.

        """
        if not self._validate_spectrum(wavelengths, values, target_wl, delta_wl):
            return np.nan

        # Remove NaN values
        valid_mask = ~np.isnan(values)
        wl_valid = wavelengths[valid_mask]
        val_valid = values[valid_mask]

        # Sort by wavelength
        sort_idx = np.argsort(wl_valid)
        wl_sorted = wl_valid[sort_idx]
        val_sorted = val_valid[sort_idx]

        # PCHIP interpolation
        try:
            interpolator = PchipInterpolator(wl_sorted, val_sorted)
            return float(interpolator(target_wl))
        except Exception:
            return np.nan

    def interpolate(self, target_wl: float, array_id: int) -> ChannelMosaic:
        """Create interpolated mosaic at target wavelength.

        This method:
        1. Selects channels within the interpolation range
        2. Builds a data cube from the selected channel mosaics
        3. For each pixel, extracts the spectrum and interpolates to target
        4. Returns a new ChannelMosaic with the interpolated data

        Parameters
        ----------
        target_wl : float
            Target wavelength in microns.
        array_id : int
            Array ID (1-6) for selecting channels and spectral resolution.

        Returns
        -------
        ChannelMosaic
            Interpolated mosaic at target wavelength.

        Raises
        ------
        ValueError
            If insufficient channels are available for interpolation.

        """
        # Select contributing channels
        selected = self._select_channels(target_wl, array_id)
        if len(selected) < 2:
            raise ValueError(
                f"Insufficient channels for interpolation at {target_wl} μm. "
                f"Found {len(selected)} channels, need at least 2."
            )

        wavelengths = np.array([wl for wl, _ in selected])
        channel_ids = [ch_id for _, ch_id in selected]

        # Log contributing channels
        logging.info(f"Interpolating with {len(channel_ids)} channels:")
        for wl, ch_id in selected:
            R_ch = self.get_spectral_resolution(ch_id)
            logging.info(f"  Channel {ch_id}: λ={wl:.4f} μm, R={R_ch:.1f}")

        # Get reference mosaic for shape and WCS
        ref_mosaic = self.channel_mosaics[channel_ids[0]]
        data_values = ref_mosaic.data.value if hasattr(ref_mosaic.data, "value") else ref_mosaic.data
        shape = data_values.shape
        wcs = ref_mosaic.wcs

        # Stack data and flags into cubes
        n_channels = len(channel_ids)
        data_cube = np.zeros((n_channels,) + shape, dtype=np.float32)
        flags_cube = np.zeros((n_channels,) + shape, dtype=np.int32)
        propagated_flags = 0

        for i, ch_id in enumerate(channel_ids):
            mosaic = self.channel_mosaics[ch_id]
            mosaic_data = mosaic.data.value if hasattr(mosaic.data, "value") else mosaic.data
            data_cube[i] = mosaic_data
            flags_cube[i] = mosaic.flags
            if mosaic.propagated_flags is not None:
                propagated_flags |= mosaic.propagated_flags

        # Calculate output channel parameters from contributing channels
        R_values = np.array([self.get_spectral_resolution(ch_id) for ch_id in channel_ids])
        R_input_mean = float(np.mean(R_values))
        R_input_std = float(np.std(R_values)) if len(R_values) > 1 else 0.0
        wl_min = float(np.min(wavelengths))
        wl_max = float(np.max(wavelengths))

        # Effective R after interpolation is degraded by the interpolation window
        R_eff = R_input_mean / (2 * self.config.n_interp)

        # Bandwidth = wavelength/R, consistent with standard channels
        bandwidth = target_wl / R_eff

        # For pixel-level interpolation, use mean input R
        delta_wl = target_wl / R_input_mean

        logging.info(f"Target: {target_wl:.4f} μm")
        logging.info(f"Wavelength range: [{wl_min:.4f}, {wl_max:.4f}] μm")
        logging.info(f"Effective bandwidth: {bandwidth:.4f} μm")
        logging.info(f"Input R: mean={R_input_mean:.1f}, std={R_input_std:.1f}")
        logging.info(f"Effective R (after interpolation): {R_eff:.1f}")

        # Perform interpolation for each pixel
        result = np.zeros(shape, dtype=np.float32)
        nhits = np.zeros(shape, dtype=np.int32)
        combined_flags = np.zeros(shape, dtype=np.int32)

        for y in range(shape[0]):
            for x in range(shape[1]):
                spectrum = data_cube[:, y, x]
                interp_val = self.interpolate_pixel(wavelengths, spectrum, target_wl, delta_wl)
                result[y, x] = interp_val
                if not np.isnan(interp_val):
                    # Count how many channels contributed
                    valid_mask = ~np.isnan(spectrum)
                    nhits[y, x] = np.sum(valid_mask)
                    # Combine flags from contributing channels using bitwise OR
                    for i in range(n_channels):
                        if valid_mask[i]:
                            combined_flags[y, x] |= flags_cube[i, y, x]

        # Create output channel info with calculated parameters
        channel_info = ChannelInfo(
            detector=array_id,
            subchannel=-1,  # Interpolated, not a native channel
            wavelength=target_wl * u.um,
            wavelength_min=wl_min * u.um,
            wavelength_max=wl_max * u.um,
            bandwidth=bandwidth * u.um,
            R=R_eff,
            R_std=R_input_std / (2 * self.config.n_interp),  # Scale by same factor
        )

        return ChannelMosaic(
            data=result * (u.MJy / u.sr),
            nhits=nhits,
            flags=combined_flags,
            wcs=wcs,
            channel_info=channel_info,
            propagated_flags=propagated_flags if propagated_flags else None,
        )


def wavelength_to_array(wavelength_um: float) -> int:
    """Determine SPHEREx array from wavelength.

    Parameters
    ----------
    wavelength_um : float
        Wavelength in microns.

    Returns
    -------
    int
        Array ID (1-6).

    Raises
    ------
    ValueError
        If wavelength is outside SPHEREx range.

    """
    # Approximate wavelength ranges per array (μm)
    array_ranges = {
        1: (0.75, 1.11),
        2: (1.11, 1.64),
        3: (1.64, 2.42),
        4: (2.42, 3.82),
        5: (3.82, 4.42),
        6: (4.42, 5.00),
    }
    for array_id, (wl_min, wl_max) in array_ranges.items():
        if wl_min <= wavelength_um <= wl_max:
            return array_id
    raise ValueError(f"Wavelength {wavelength_um} μm outside SPHEREx range [0.75, 5.0]")
