"""LVF wavelength mesh utilities.

This module provides backward-compatible access to wavelength meshes
that use Simulator Wavelength Model.
New code should use `TabularWavelengthModel` directly.
"""

__all__ = ["get_wavelength_mesh"]

import numpy as np

from . import spx_param
from .wavelength_model import TabularWavelengthModel


def get_wavelength_mesh(
    array_id: int,
    npix_x: int = None,
    npix_y: int = None,
) -> np.ndarray:
    """Get the 2D wavelength mesh for a detector array.

    This function provides backward compatibility. New code should use
    `TabularWavelengthModel` directly::

        from spx_data import TabularWavelengthModel
        wl_model = TabularWavelengthModel()
        wavelength_mesh = wl_model.get_wavelength_array(array_id)

    Parameters
    ----------
    array_id : int
        The array ID (1-6).
    npix_x : int, optional
        Number of pixels in x direction.
        X direction is along similar wavelength.
        Default is active area size.
    npix_y : int, optional
        Number of pixels in y direction.
        Y direction is along wavelength gradient.
        Default is active area size.

    Returns
    -------
    wavelength_mesh : numpy.ndarray
        2D wavelength mesh with central wavelength at each pixel position.
        Array is in sky coordinates (MWIR x-flip applied).

    """
    npix_x_active = spx_param.get_npix_x_active()
    npix_y_active = spx_param.get_npix_y_active()

    if npix_x is None:
        npix_x = npix_x_active
    if npix_y is None:
        npix_y = npix_y_active

    if npix_x > npix_x_active or npix_y > npix_y_active:
        raise ValueError(
            f"Invalid wavelength mesh parameters ({npix_x}, {npix_y}). "
            f"Maximum is ({npix_x_active}, {npix_y_active})."
        )

    # TabularWavelengthModel is a singleton - no need for module-level caching
    wl_model = TabularWavelengthModel()
    full_mesh = wl_model.get_wavelength_array(array_id)

    # Crop if requested size is smaller than active area
    if npix_x < npix_x_active or npix_y < npix_y_active:
        full_mesh = full_mesh[:npix_y, :npix_x]

    return full_mesh
