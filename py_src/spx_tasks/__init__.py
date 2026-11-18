"""
spx_tasks - SPHEREx Pipeline Tasks

This module uses lazy loading to avoid importing all tasks at package initialization.
Tasks are only imported when explicitly accessed.

Usage:
    from spx_tasks import L2aTask, L2aConfig
    from spx_tasks import add_history
"""
import importlib
from typing import TYPE_CHECKING

# Used by tools (IDE), see PEP-484
if TYPE_CHECKING:
    from ._provenance import *
    from .aggregate_idqr import *
    from .aggregate_odir import *
    from .apply_bad_pixel_mask import *
    from .apply_dark_flat_field_and_photo_calib import *
    from .combine_catalog_tiles import *
    from .compute_solar_system_objects import *
    from .convert_to_engineering_units import *
    from .correct_dark_current import *
    from .correct_non_linearity import *
    from .correct_preamplifier import *
    from .create_all_sky_catalog import *
    from .create_data_cube import *
    from .create_high_reliability_cat import *
    from .create_image_data_files import *
    from .create_source_mask import *
    from .create_spectral_image import *
    from .create_variance_map import *
    from .detect_outlier_pixels import *
    from .dqa_l1 import *
    from .dqa_l2 import *
    from .dqa_l3 import *
    from .estimate_coarse_astrometry import *
    from .estimate_diffuse_background import *
    from .estimate_fine_astrometry import *
    from .estimate_abs_gain_matrix import *
    from .estimate_psf import *
    from .estimate_source_variability import *
    from .estimate_subpixel_gain import *
    from .filter_by_mask import *
    from .gds_image_association import *
    from .get_zodi_model import *
    from .l1 import *
    from .l1_split import *
    from .l2a import *
    from .l2b import *
    from .l3a import *
    from .measure_psf import *
    from .perform_forced_photometry import *
    from .query_custom_catalog import *
    from .query_reference_catalog import *
    from .remove_reference_pixels import *
    from .source_bloom_and_cosmic_ray_halo_flagging import *
    from .update_field_distortion import *
    from .update_persistence import *
    from .update_photometric_calibration import *
    from .validate_sur import *
    from .estimate_fine_astrometry_retry import *
    from .l2b_retry import *

# Mapping of module names to their exported attributes
_SUBMODULE_ATTRS = {
    "_provenance": [
        "add_calibration_provenance",
        "add_history",
        "get_calibration_inputs",
        "module_provenance",
        "add_psf_history",
    ],
    "aggregate_idqr": ["AggregateIdqrTaskConfig", "AggregateIdqrTask"],
    "aggregate_odir": ["AggregateOdirTaskConfig", "AggregateOdirTask"],
    "apply_bad_pixel_mask": ["ApplyBadPixelMaskConfig", "ApplyBadPixelMaskTask"],
    "apply_dark_flat_field_and_photo_calib": [
        "ApplyDarkFlatFieldAndPhotoCalibConfig",
        "ApplyDarkFlatFieldAndPhotoCalibTask",
    ],
    "combine_catalog_tiles": ["CombineCatalogTilesConfig", "CombineCatalogTilesTask"],
    "compute_solar_system_objects": ["SolarSystemObjectsConfig", "SolarSystemObjectsTask"],
    "convert_to_engineering_units": [
        "ConverttoEngineeringUnitsConfig",
        "ConverttoEngineeringUnitsTask",
    ],
    "correct_dark_current": ["CorrectDarkCurrentConfig", "CorrectDarkCurrentTask"],
    "correct_non_linearity": ["CorrectNonLinearityConfig", "CorrectNonLinearityTask"],
    "correct_preamplifier": ["CorrectPreamplifierConfig", "CorrectPreamplifierTask"],
    "create_all_sky_catalog": ["CreateAllSkyCatalogConfig", "CreateAllSkyCatalogTask"],
    "create_data_cube": ["CreateDataCubeConfig", "CreateDataCubeTask"],
    "create_high_reliability_cat": ["CreateHighReliabilityCatConfig", "CreateHighReliabilityCatTask"],
    "create_image_data_files": ["CreateImageDataFilesConfig", "CreateImageDataFilesTask"],
    "create_source_mask": ["CreateSourceMaskConfig", "CreateSourceMaskTask"],
    "create_spectral_image": ["CreateSpectralImageConfig", "CreateSpectralImageTask"],
    "create_variance_map": [
        "CreateVarianceMapConfig",
        "CreateVarianceMapTask",
        "calculate_poisson_term",
    ],
    "detect_outlier_pixels": ["DetectOutlierPixelsConfig", "DetectOutlierPixelsTask"],
    "dqa_l1": ["DQAL1Config", "DQAL1Task", "check_l1_dqa"],
    "dqa_l2": ["DQAL2Config", "DQAL2Task"],
    "dqa_l3": ["DQAL3Config", "DQAL3Task"],
    "estimate_coarse_astrometry": [
        "EstimateCoarseAstrometryConfig",
        "EstimateCoarseAstrometryTask",
    ],
    "estimate_diffuse_background": [
        "EstimateDiffuseBackgroundConfig",
        "EstimateDiffuseBackgroundTask",
    ],
    "estimate_fine_astrometry": ["EstimateFineAstrometryConfig", "EstimateFineAstrometryTask"],
    "estimate_abs_gain_matrix": ["EstimateAbsGainMatrixConfig", "EstimateAbsGainMatrixTask"],
    "estimate_psf": ["EstimatePSFConnections", "EstimatePSFConfig", "EstimatePSFTask"],
    "estimate_source_variability": [
        "EstimateSourceVariabilityConfig",
        "EstimateSourceVariabilityTask",
    ],
    "estimate_subpixel_gain": ["EstimateSubpixelGainConfig", "EstimateSubpixelGainTask"],
    "filter_by_mask": ["FilterByMaskTask", "FilterByMaskConfig"],
    "gds_image_association": ["GdsImageAssociationConfig", "GdsImageAssociationTask"],
    "get_zodi_model": ["GetZodiModelConfig", "GetZodiModelTask", "hasSPHERExSimulator"],
    "l1": ["L1Config", "L1Task"],
    "l2a": ["L2aConfig", "L2aTask"],
    "l2b": ["L2bConfig", "L2bTask"],
    "l3a": ["L3aConfig", "L3aTask"],
    "measure_psf": ["MeasurePSFConfig", "MeasurePSFTask"],
    "perform_forced_photometry": ["PerformForcedPhotometryConfig", "PerformForcedPhotometryTask"],
    "query_custom_catalog": ["QueryCustomCatalogConfig", "QueryCustomCatalogTask"],
    "query_reference_catalog": ["QueryReferenceCatalogConfig", "QueryReferenceCatalogTask"],
    "remove_reference_pixels": ["RemoveReferencePixelsConfig", "RemoveReferencePixelsTask"],
    "source_bloom_and_cosmic_ray_halo_flagging": [
        "SourceBloomAndCosmicRayHaloFlaggingConfig",
        "SourceBloomAndCosmicRayHaloFlaggingTask",
    ],
    "update_field_distortion": [
        "UpdateFieldDistortionConnections",
        "UpdateFieldDistortionConfig",
        "UpdateFieldDistortionTask",
    ],
    "update_persistence": [
        "UpdatePersistenceConfig",
        "UpdatePersistenceTask",
        "PersistenceModel",
        "PersistenceModelState",
    ],
    "update_photometric_calibration": [
        "UpdatePhotometricCalibrationConfig",
        "UpdatePhotometricCalibrationTask",
    ],
    "validate_sur": ["ValidateSURConfig", "ValidateSURTask"],
    "estimate_fine_astrometry_retry": [
        "EstimateFineAstrometryRetryConfig",
        "EstimateFineAstrometryRetryTask",
    ],
    "l2b_retry": [
        "L2bRetryConfig",
        "L2bRetryTask",
    ],
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
