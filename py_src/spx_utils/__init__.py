"""
spx_utils - SPHEREx Pipeline Utilities

This module uses lazy loading to avoid importing all utilities at package initialization.
Utilities are only imported when explicitly accessed.

Usage:
    from spx_utils import get_initial_linear_wcs
    from spx_utils import AggregateGDS
"""
import importlib
from typing import TYPE_CHECKING

# Used by tools (IDE), see PEP-484
if TYPE_CHECKING:
    from .aggregate_gds import *
    from .alt_wcs import set_active_pix_wcs
    from .bsh_flagger import *
    from .coarse_astro_utils import *
    from .idqr_utils import *
    from .tempfile_utils import *
    from .template_utils import *
    from .initial_distortion_model import *
    from .dqa_utils import *
    from .image_catalog import *
    from .l2bwcs_utils import *
    from .l1_utils import *
    from .odir_utils import *
    from .utils import *
    from .rss_utils import *
    from .solar_elongation import *
    from .solid_angle_utils import *
    from .spectral_lookup import *
    from .update_distortion_utils import *
    from .update_photo_calib_utils import *


# Mapping of module names to their exported attributes
_SUBMODULE_ATTRS = {
    "aggregate_gds": ["AggregateGDS"],
    "alt_wcs": ["set_active_pix_wcs"],
    "bsh_flagger": ["BSHConfig", "BSHFlagger", "apply_bsh_flags_to_spx"],
    "coarse_astro_utils": [
        "get_ra_dec_lon_lat_from_quat",
        "pc_to_cd",
        "cd_to_pc",
        "get_initial_linear_wcs",
        "get_initial_wcs_with_sip",
        "position_angle_from_cd",
        "calculate_new_crval",
        "calculate_detector_pointings_with_pa",
        "get_corrected_linear_wcs",
        "compute_side_pas",
        "cd_from_wcs",
        "north_aligned_pa_from_wcs",
        "build_wcs_triplet_from_boresight",
    ],
    "idqr_utils": ["get_idqr"],
    "tempfile_utils": ["get_ssdc_tempfiledir", "get_temp_path"],
    "template_utils": [
        "create_masks",
        "fit_airglow_he",
        "subtract_airglow_he",
        "attach_airglow_fit_to_header",
        "read_airglow_fit_from_header",
        "reconstruct_p_fit_from_results",
        "correlate_image",
        "template_correlate",
    ],
    "initial_distortion_model": ["get_initial_distortion_model"],
    "dqa_utils": ["flags_summary", "log_threshold_test", "nan_pixels_flagged", "log_dqa_result"],
    "image_catalog": ["SourcePipelineClass", "ImageCatalogIterable", "preprocess_catalog"],
    "l1_utils": ["active_row_col", "active_area_range", "with_extra_refs"],
    "l2bwcs_utils": [
        "DET_TO_OPPOSITE",
        "DET_TO_DICHROICPAIR",
        "opposite_detector",
        "build_l2bwcs_payload",
        "payload_to_wcs",
    ],
    "odir_utils": ["get_odir"],
    "utils": [
        "add_detcoord",
        "is_device_detcoord",
        "get_data_file",
        "get_overflow_threshold",
        "get_refcat_provenance",
    ],
    "rss_utils": ["get_current_rss"],
    "solar_elongation": ["compute_tilt_function", "tilt_correct", "construct_tilt_template"],
    "solid_angle_utils": ["calculate_solid_angle", "calculate_median_std_solid_angle"],
    "spectral_lookup": ["spectral_wcs_lookup_data", "spectral_wcs_tab_header"],
    "update_distortion_utils": [
        "sip_coefficients",
        "initialize_polynomial_from_sip",
        "calculate_sip_coefficients",
        "update_wcs_with_sip",
    ],
    "update_photo_calib_utils": ["CalspecPredict"],
}

# Build reverse lookup: attribute name -> module name
_ATTR_TO_MODULE = {}
for _module, _attrs in _SUBMODULE_ATTRS.items():
    for _attr in _attrs:
        _ATTR_TO_MODULE[_attr] = _module

# Build __all__ from all exported attributes
__all__ = list(_ATTR_TO_MODULE.keys())


def __getattr__(name):
    """Lazily import attributes on first access."""
    if name in _ATTR_TO_MODULE:
        module_name = _ATTR_TO_MODULE[name]
        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """List all available attributes for IDE autocompletion."""
    return __all__