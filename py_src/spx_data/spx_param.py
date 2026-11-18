# all calls should go through spx_data.spx_param
__all__ = ["format_unit", "get_aperture", "get_cds_read_noise", "get_detector_efficiency",
           "get_filter_resolving_power", "get_lvf_values", "get_npix_blind", "get_npix_x",
           "get_npix_x_active", "get_npix_y", "get_npix_y_active", "get_num_samples",
           "get_pix_scale", "get_sample_time", "readout_params"]

from types import SimpleNamespace
from typing import Union
import astropy.units as u


def _as_namespace(param_dict: dict) -> SimpleNamespace:
    """Local copy of spherex_parameters.as_namespace()'s algorithm.

    Used only in the fallback branch below, where spherex_parameters (and
    its as_namespace()) is by definition unavailable, so that fallback
    namespaces (e.g. readout_params) are derived from the fallback
    _param_dict the same way the live branch derives them.
    """
    if "quantity" in param_dict:
        return param_dict["quantity"]
    elif "value" in param_dict:
        return param_dict["value"]
    else:
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in param_dict.items()})


def _get_array_values(param_dict: dict, array_id: int, group: str) -> dict:
    """Local copy of spherex_parameters.get_array_values()'s algorithm.

    Used only in the fallback branch below (see _as_namespace() above for
    why), so lvf_params is derived from the fallback _param_dict the same
    way the live branch derives it.
    """
    if group not in param_dict:
        raise ValueError(f"Group {group} does not exist.")
    if not 1 <= array_id <= 6:
        raise ValueError(f"Array ID {array_id} is out of range 1 to 6.")
    array_dict = {}
    for param, item in param_dict[group].items():
        value = item["value"]
        if not isinstance(value, list) or len(value) != 6:
            continue
        array_dict[param] = value[array_id - 1]
    return array_dict


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
    # spherex-parameters repo.
    #
    # SSDC pipelines should be using the values from the headers
    # and migrate away from using the parameters in this module.
    # fmt: off
    _param_dict = {
        'nominal_camera_model': {
            # get_pix_scale() -- measure_psf.py, compute_solar_system_objects.py,
            # psf.py (PSF model WCS), psf_fwhm.py, tractor_utils.py (photometry WCS)
            'pix_scale': {
                'name': 'Pixel Scale',
                'quantity': 6.15 << u.arcsec
            },
            # get_npix_x() -- fake_products.py, sample_data.py (test/fake data
            # shape); also the basis for get_npix_x_active() (perform_forced_
            # photometry.py, lvf.py, wavelength_model.py, spectral_lookup.py,
            # fake_products.py)
            'npix_x': {
                'name': 'Number of Array Pixels (Spatial Direction)',
                'value': 2048,
                'unit': None,
            },
            # get_npix_y() -- fake_products.py, sample_data.py (test/fake data
            # shape); also the basis for get_npix_y_active() (same consumers
            # as get_npix_x_active() above)
            'npix_y': {
                'name': 'Number of Array Pixels (Cross Channel Direction)',
                'value': 2048,
                'unit': None,
            }
        },
        'nominal_instrument_models': {
            # get_aperture() -- update_photometric_calibration.py (effective
            # aperture area in the MJy/sr -> electron/s flux conversion)
            'aperture': {
                'name': 'Aperture',
                'quantity': 20 << u.cm
            },
        },
        # get_sample_time() -- computing observation time margins in ingest code
        # get_num_samples() -- fake_products.py (synthetic ramp generation)
        # derived readout_parameters below -- used in
        # - fake_products.py
        # - gain_factors.py (n_readout_channels only)
        # - QRC (number of active pixels only)
        'readout': {
            't_samp': {
                'name': 'Array Sample time',
                'quantity': 1.5349 << u.s
            },
            'n_samp': {
                'name': 'Number of ramp samples per exposure',
                'value': 75,
                'unit': None,
            },
            'n_readout_channels': {
                'name': 'Number of video channels that reads out each H2RG',
                'value': 32,
                'unit': None,
            },
            'Nvisit_row': {
                'name': 'Number of visits to each reference row.',
                'value': 4,
                'unit': None,
            },
            'Nvisit_col': {
                'name': 'Number of visits to each reference column',
                'value': 1,
                'unit': None,
            },
            'Mphan': {
                'name': 'Number of phantom pixel reads',
                'value': 4,
                'unit': None,
            },
            'wchannel': {
                'name': 'Number of columns of physical pixels per readout channel.',
                'value': 64,
                'unit': None,
            },
            'jref': {
                'name': 'Number of reference rows on the H2RG',
                'value': 8,
                'unit': None,
            },
            'kref': {
                'name': 'Number of reference columns on the H2RG',
                'value': 8,
                'unit': None,
            },
            'Jactive': {
                'name': 'Number of active pixel rows in the detector',
                'value': 2040,
                'unit': None,
            },
            'Kactive': {
                'name': 'Number of active pixel columns in the detector',
                'value': 2040,
                'unit': None,
            },
            'Nskip': {
                'name': 'Number of rows to skip',
                'value': 32,
                'unit': None,
            },
        },
        'detector': {
            # get_cds_read_noise() -- fake_products.py (synthetic noise injection)
            'read_noise_cds': {
                'name': 'CDS Read Noise',
                'value': [17.2, 16.5, 17.2, 10.8, 13.9, 11.6],
                'unit': 'electron',
            },
            # get_detector_efficiency() -- update_photometric_calibration.py
            # (end-to-end throughput term in the MJy/sr -> electron/s flux
            # conversion)
            'eta_total': {
                'name': 'Array-mean end-to-end efficiency',
                'value': [0.81, 0.83, 0.77, 0.81, 0.57, 0.53],
                'unit': None,
            },
        },
        'filter': {
            # get_filter_resolving_power() -- update_photometric_calibration.py
            # (resolving power term in the MJy/sr -> electron/s flux conversion)
            #
            # derived lvf_params below -- used
            #     when populating detector dimension table in Butler
            #     in fake spectral wcs lookup data (R only)
            'R': {
                'name': 'Spectral Resolution',
                'value': [41.2, 41.7, 41.1, 34.6, 117.1, 132.8],
                'unit': None,
            },
            'lambda_min': {
                'name': 'Lowest active area wavelength',
                'value': [0.7455, 1.1033, 1.6301, 2.4027, 3.8113, 4.4115]
            },
            'lambda_max': {
                'name': 'Highest active area wavelength',
                'value': [1.1167, 1.6499, 2.4346, 3.8473, 4.4300, 5.0096]
            }
        },
    }
    # fmt: on

    # Readout parameters, derived from _param_dict["readout"] above (single
    # source of truth -- see _as_namespace()).
    readout_params = _as_namespace(_param_dict["readout"])

    # LVF model parameters, derived from _param_dict["filter"] above (single
    # source of truth -- see _get_array_values()).
    lvf_params = {i: _get_array_values(_param_dict, array_id=i, group="filter") for i in range(1, 7)}


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


def get_aperture() -> u.Quantity:
    """Return the nominal instrument aperture diameter.

    Used by update_photometric_calibration.py for the effective aperture
    area in the MJy/sr -> electron/s flux conversion.

    Returns
    -------
    aperture : `astropy.units.Quantity`
        The aperture diameter.

    """
    return _param_dict["nominal_instrument_models"]["aperture"]["quantity"]


def get_detector_efficiency() -> list:
    """Return array-mean end-to-end efficiency for each detector array.

    Used by update_photometric_calibration.py for the throughput term in
    the MJy/sr -> electron/s flux conversion.

    Returns
    -------
    eta_total : list of float
        Efficiency (mirror x dichroic x LVF x detector QE), one value per
        detector array, ordered 1-6.

    """
    return _param_dict["detector"]["eta_total"]["value"]


def get_filter_resolving_power() -> list:
    """Return LVF spectral resolving power for each detector array.

    Used by update_photometric_calibration.py for the resolving-power term
    in the MJy/sr -> electron/s flux conversion (same values as
    get_lvf_values(array_id)['R'], as a flat per-array list instead of
    per-array dicts).

    Returns
    -------
    resolving_power : list of float
        Resolving power (lambda / delta lambda), one value per detector
        array, ordered 1-6.

    """
    return _param_dict["filter"]["R"]["value"]


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
