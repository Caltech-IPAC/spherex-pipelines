__all__ = ["get_wavelength_mesh"]

# Standard library imports
import os
from functools import lru_cache

# Related third party imports
from astropy.io import fits
import numpy as np
from pathlib import Path

# Local application/library specific imports
from spx_data import spx_param
# from spx_utils.spec_cal_response import speccal_calculate_response

# Simulator dependency
try:
    from SPHEREx_InstrumentSimulator import Tabular_Bandpass
    lvf_bandpass = Tabular_Bandpass()
except ImportError:
    Tabular_Bandpass = None
    lvf_bandpass = None


def get_wavelength_mesh(array_id: int,
                        npix_x: int = spx_param.get_npix_x_active(),
                        npix_y: int = spx_param.get_npix_y_active(),
                        use_inst_cal: bool = False,
                        base_dir: str = None) -> np.ndarray:
    """Gets the two-dimensional expression for the central wavelength at pixel
    center.

    Parameters
    ----------
    array_id : int
        The array ID to use
    npix_x : int, optional
        The number of pixels in the wavelength mesh along similar wavelength
        direction. The default is the number of active area pixels from ``spx_param``.
    npix_y : int, optional
        The number of pixels in the wavelength mesh along wavelength gradient
        direction. The default is the number of active area pixels from ``spx_param``.
    use_inst_cal : `bool`
        Use instrument calibration delivery for center wavelengths.
        Default is False; use simulator model.
    base_dir : `str`
        Option to give base directory path instead of using environment
        variable, default is None.

    Returns
    -------
    wavelength_mesh: `numpy.ndarray`
        2-D wavelength mesh with central wavelength at any position.
    """
    return _get_wavelength_mesh(array_id, npix_x, npix_y, use_inst_cal, base_dir)


@lru_cache(maxsize=1)
def _get_wavelength_mesh(array_id: int,
                         npix_x: int,
                         npix_y: int,
                         use_inst_cal: bool,
                         base_dir: str) -> np.ndarray:
    """A helper function whose signature contains only positional arguments.
    It guarantees that calls that resolve to the same numbers hit the same
    cache entry, no matter how the public wrapper was invoked.

    Parameters
    ----------
    array_id : int
        The array ID to use
    npix_x : int
        The number of pixels in the wavelength mesh along similar wavelength
        direction.
    npix_y : int
        The number of pixels in the wavelength mesh along wavelength gradient
        direction.
    use_inst_cal : `bool`
        If ``True``, use instrument calibration delivery for center wavelengths,
        otherwise use ``Tabular_Bandpass`` from the Simulator.
    base_dir : `str` or `None`
        Base directory to be used to locate center wavelength calibration product
        delivered by the Instrument Team. If ``None``, use
        ``SSDC_CENTER_WAVELENGTH_DIR`` environment variable.

    Returns
    -------
    wavelength_mesh: `numpy.ndarray`
        2-D wavelength mesh with central wavelength at any position.
    """
    if npix_x > spx_param.get_npix_x_active() or npix_y > spx_param.get_npix_y_active():
        raise ValueError(f"Invalid wavelength mesh parameters {npix_x}, {npix_y}")

    mwir_detector: bool = array_id > 3
    if use_inst_cal:
        # Path to SSDC center wavelength FITS file
        if base_dir is None:
            base_dir = os.environ.get("SSDC_CENTER_WAVELENGTH_DIR")
            if base_dir is None:
                raise ValueError("SSDC_CENTER_WAVELENGTH_DIR environment variable is not set,"
                                 " and no base_dir was provided.")

        center_path = Path(f"{base_dir}/20250202_SSDC_CENTER_Band{array_id}.FITS")

        if not center_path.exists():
            raise FileNotFoundError(f"Central wavelength FITS file not found: {center_path}")

        with fits.open(center_path) as hdul:
            full_mesh = hdul[0].data.astype(np.float32)

        # Optionally crop the mesh if npix_x or npix_y is set
        if npix_x < spx_param.get_npix_x_active() or npix_y < spx_param.get_npix_y_active():
            full_mesh = full_mesh[:npix_y, :npix_x]

        if mwir_detector:
            full_mesh = full_mesh[:, ::-1]

        return full_mesh

    else:
        # Create a grid
        xi_mesh, yi_mesh = np.meshgrid(np.arange(npix_x), np.arange(npix_y))
        # use Simulator's Tabular Bandpass
        if lvf_bandpass is None:
            raise ImportError("SPHEREx Simulator is not available")
        ref_pix = spx_param.get_npix_blind()
        wavelength_mesh = (lvf_bandpass(xi_mesh + ref_pix, yi_mesh + ref_pix, array=array_id,
                                        central_bandpass_only=True)[0]).astype(np.float32)

        # Flip x-axis MWIR
        if mwir_detector:
            wavelength_mesh = wavelength_mesh[:, ::-1]

        return wavelength_mesh
