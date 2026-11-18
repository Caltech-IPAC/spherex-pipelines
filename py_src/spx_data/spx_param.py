__all__ = ["get_sample_time", "get_npix_x", "get_npix_y",
           "get_num_samples", "get_nonlinear_flag_threshold", "get_lvf_values",
           "format_unit", "get_pix_scale", "readout_params"]

from types import SimpleNamespace
from typing import Union
import astropy.units as u

try:
    from spherex_parameters import as_namespace, load_spherex_parameters, get_array_values

    _param_dict: dict = load_spherex_parameters(append_quantity=True)

    readout_params: SimpleNamespace = as_namespace(_param_dict["readout"])

    # dictionary with one key per array (1-6)
    lvf_params: dict = {i: get_array_values(_param_dict, array_id=i, group="filter") for i in range(1, 7)}

except ImportError:

    # Fallback values when spherex-parameters are not installed
    #
    # This module is a single point of access for the values from
    # spherex-parameters repo. While this repo is useful for simulations,
    # SSDC pipelines should be using the values from the headers
    # and migrate away from using the parameters in this module.
    _param_dict = {
        'nominal_camera_model': {
            'pix_scale': {
                'name': 'Pixel Scale',
                'quantity': 6.15 << u.arcsec
            },
            'npix_x': {
                'name': 'Number of Array Pixels (Spatial Direction)',
                'value': 2048,
                'unit': None,
            },
            'npix_y': {
                'name': 'Number of Array Pixels (Cross Channel Direction)',
                'value': 2048,
                'unit': None,
            }
        },
        'readout': {
            't_samp': {
                'name': 'Array Sample time',
                'quantity': 1.5349 << u.s
            },
            'n_samp': {
                'name': 'Number of ramp samples per exposure',
                'value': 73,
                'unit': None,
            },
            'nonlinear_flag_threshold': {
                'name': 'Readout threshold for nonlinear flag',
                'quantity': 30000. << u.electron
            }
        },
        'detector': {
            'read_noise_cds': {
                'name': 'CDS Read Noise',
                'value': [17.2, 16.5, 17.2, 10.8, 13.9, 11.6],
                'unit': 'electron',
            },
        },
    }

    # Readout parameters
    readout_params = SimpleNamespace(
        t_samp=1.5349 << u.s,
        n_samp=73,
        n_readout_channels=32,
        Nvisit_row=4,
        Nvisit_col=1,
        Mphan=4,
        wchannel=64,
        jref=8,
        kref=8,
        Jactive=2040,
        Kactive=2040,
        Nskip=32)

    # LVF model parameters
    lvf_params = {
        1: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 0.75,
            'lambda_end': 1.11,
            'lambda_min': 0.7455,
            'delta_lambda_min': 0.0026,
            'lambda_max': 1.1167,
            'delta_lambda_max': 0.0036,
            'R': 41.2,
            'deltaR': 0.1},
        2: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 1.11,
            'lambda_end': 1.64,
            'lambda_min': 1.1033,
            'delta_lambda_min': 0.0038,
            'lambda_max': 1.6499,
            'delta_lambda_max': 0.0053,
            'R': 41.7,
            'deltaR': 0.1},
        3: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 1.64,
            'lambda_end': 2.42,
            'lambda_min': 1.6301,
            'delta_lambda_min': 0.0102,
            'lambda_max': 2.4346,
            'delta_lambda_max': 0.0011,
            'R': 41.1,
            'deltaR': 0.1},
        4: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 2.42,
            'lambda_end': 3.82,
            'lambda_min': 2.4027,
            'delta_lambda_min': 0.0013,
            'lambda_max': 3.8473,
            'delta_lambda_max': 0.0268,
            'R': 34.6,
            'deltaR': 0.1},
        5: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 3.82,
            'lambda_end': 4.42,
            'lambda_min': 3.8113,
            'delta_lambda_min': 0.0056,
            'lambda_max': 4.43,
            'delta_lambda_max': 0.0066,
            'R': 117.1,
            'deltaR': 0.1},
        6: {'X_C': -136.98,
            'X_smile': 1.225,
            'X_oversize': 0.0,
            'A_x': 36.72,
            'A_y': 36.72,
            'lambda_str': 4.42,
            'lambda_end': 5.0,
            'lambda_min': 4.4115,
            'delta_lambda_min': 0.0058,
            'lambda_max': 5.0096,
            'delta_lambda_max': 0.0062,
            'R': 132.8,
            'deltaR': 0.07}
    }


def get_sample_time() -> u.Quantity:
    """Return the detector ramp sample readout cadence.

    Returns
    -------
    sample_time : `astropy.units.Quantity`
        The sample time in seconds.

    """
    return _param_dict["readout"]["t_samp"]["quantity"]


def get_pix_scale() -> u.Quantity:
    """Return the pixel scale.

    Returns
    -------
    pixel_scale : `astropy.units.Quantity`
        The pixel scale in arcseconds.

    """
    return _param_dict["nominal_camera_model"]["pix_scale"]["quantity"]


def get_npix_x() -> int:
    """Return number of pixels along similar wavelength direction of the detector."""
    return _param_dict["nominal_camera_model"]["npix_x"]["value"]


def get_npix_y() -> int:
    """Return number of pixels along wavelength gradient direction of the detector."""
    return _param_dict["nominal_camera_model"]["npix_y"]["value"]


def get_npix_blind() -> int:
    """Return number of non-photosensitive pixels on each edge of the detector."""
    return 4


def get_npix_x_active() -> int:
    """Return number of photosensitive pixels along similar wavelength
    direction of the detector.
    """
    return get_npix_x() - 2 * get_npix_blind()


def get_npix_y_active() -> int:
    """Return number of photosensitive pixels along wavelength gradient
    direction of the detector.
    """
    return get_npix_y() - 2 * get_npix_blind()


def get_num_samples() -> int:
    """Return the number of ramp samples in each exposure.
    Number of samples multiplied by sample duration gives exposure time.

    Returns
    -------
    num_samples: int
        Number of samples taken during the exposure integration.

    """
    return _param_dict["readout"]["n_samp"]["value"]


def get_nonlinear_flag_threshold():
    """Return the number of accumulated electrons that triggers the nonlinearity flag.

    Returns
    -------
    nonlinear_flag_threshold : Quantity
        The electron count threshold for nonlinearity flagging.

    """
    return _param_dict["readout"]["nonlinear_flag_threshold"]["quantity"]


def get_cds_read_noise(detector: int):
    """Return array-mean H2RG/amplifier noise in CDS mode.

    Parameters
    ----------
    detector : int
        Detector ID (1-6).

    Returns
    -------
    result : float
        CDS-read noise for a given detector in electrons.

    """
    return _param_dict["detector"]["read_noise_cds"]["value"][detector-1]


def get_lvf_values(array_id):
    """Return LVF values for specific array."""
    return lvf_params[array_id]


def format_unit(unit: Union[u.Unit, str]):
    """Standardize units appearance in strings.

    Parameters
    ----------
    unit : `~astropy.units.Unit` or `str`
        The unit to format.

    Returns
    -------
    formatted_str : `str`
        The formatted unit string.

    """
    # use generic formatter to have more control over how unit appears
    return u.format.Generic.to_string(u.Unit(unit))
