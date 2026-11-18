import numpy as np
from astropy.table import Table
from .spectral_channel import LVFChannelDefinitionV2

# Reference spectral channel data from SPHEREx instrument specification
# Source: SPHEREx SpectralBandDef 202601, revision 2/24
# The bandwidth is median channel wavelength divided by median channel R
# fmt: off
REFERENCE_SPECTRAL_CHANNELS_HISTORY = [
    "SSDC R7 spectral-calibration delivery 7/22, Table 2",
    "Derived from SpecCal_202607",
    "No changes to detectors 1-4 since SpecCal_202509",
]
REFERENCE_SPECTRAL_CHANNELS_COMMENTS = [
    "WAVELENGTH is median per-pixel wavelength [um]",
    "R is median per-pixel spectral resolution",
    "BANDWIDTH is calculated as WAVELENGTH/R",
]
REFERENCE_SPECTRAL_CHANNELS = {
    # (DETECTOR, SUBCHAN): (WAVELENGTH, WL_MIN, WL_MAX, R, R_STD, BANDWIDTH)
    # Band 1
    (1, 1): (0.753, 0.743, 0.762, 42, 1.1, 0.018),
    (1, 2): (0.771, 0.761, 0.781, 42, 1.0, 0.019),
    (1, 3): (0.790, 0.780, 0.800, 41, 0.9, 0.019),
    (1, 4): (0.809, 0.798, 0.819, 40, 0.9, 0.020),
    (1, 5): (0.828, 0.817, 0.839, 40, 0.8, 0.020),
    (1, 6): (0.848, 0.837, 0.859, 40, 0.8, 0.021),
    (1, 7): (0.868, 0.857, 0.879, 40, 0.7, 0.022),
    (1, 8): (0.889, 0.877, 0.900, 39, 0.7, 0.023),
    (1, 9): (0.910, 0.898, 0.922, 39, 0.6, 0.023),
    (1, 10): (0.932, 0.920, 0.944, 39, 0.6, 0.024),
    (1, 11): (0.955, 0.942, 0.967, 39, 0.5, 0.025),
    (1, 12): (0.978, 0.964, 0.990, 38, 0.5, 0.026),
    (1, 13): (1.001, 0.988, 1.014, 38, 0.5, 0.026),
    (1, 14): (1.025, 1.012, 1.039, 38, 0.5, 0.027),
    (1, 15): (1.050, 1.036, 1.064, 38, 0.5, 0.028),
    (1, 16): (1.075, 1.061, 1.090, 37, 0.4, 0.029),
    (1, 17): (1.102, 1.086, 1.116, 37, 0.4, 0.030),
    # Band 2
    (2, 1): (1.113, 1.099, 1.126, 43, 0.7, 0.026),
    (2, 2): (1.139, 1.125, 1.153, 42, 0.6, 0.027),
    (2, 3): (1.167, 1.152, 1.182, 42, 0.5, 0.028),
    (2, 4): (1.195, 1.180, 1.210, 41, 0.5, 0.029),
    (2, 5): (1.224, 1.209, 1.240, 41, 0.5, 0.030),
    (2, 6): (1.254, 1.238, 1.269, 41, 0.5, 0.030),
    (2, 7): (1.284, 1.267, 1.300, 42, 0.4, 0.031),
    (2, 8): (1.315, 1.298, 1.331, 42, 0.4, 0.032),
    (2, 9): (1.346, 1.329, 1.363, 42, 0.4, 0.032),
    (2, 10): (1.378, 1.360, 1.396, 42, 0.4, 0.033),
    (2, 11): (1.412, 1.393, 1.430, 41, 0.4, 0.034),
    (2, 12): (1.446, 1.426, 1.464, 41, 0.3, 0.035),
    (2, 13): (1.481, 1.461, 1.500, 41, 0.3, 0.036),
    (2, 14): (1.517, 1.496, 1.537, 41, 0.3, 0.037),
    (2, 15): (1.554, 1.533, 1.575, 41, 0.3, 0.038),
    (2, 16): (1.592, 1.570, 1.613, 41, 0.3, 0.039),
    (2, 17): (1.631, 1.608, 1.651, 41, 0.3, 0.039),
    # Band 3
    (3, 1): (1.656, 1.636, 1.676, 42, 0.3, 0.039),
    (3, 2): (1.695, 1.675, 1.715, 42, 0.3, 0.041),
    (3, 3): (1.735, 1.714, 1.756, 41, 0.3, 0.042),
    (3, 4): (1.775, 1.754, 1.797, 41, 0.3, 0.043),
    (3, 5): (1.818, 1.795, 1.840, 41, 0.3, 0.045),
    (3, 6): (1.861, 1.838, 1.884, 41, 0.3, 0.046),
    (3, 7): (1.905, 1.881, 1.928, 41, 0.3, 0.047),
    (3, 8): (1.949, 1.925, 1.973, 41, 0.3, 0.048),
    (3, 9): (1.995, 1.970, 2.019, 41, 0.3, 0.049),
    (3, 10): (2.041, 2.015, 2.067, 40, 0.4, 0.051),
    (3, 11): (2.089, 2.062, 2.115, 40, 0.3, 0.052),
    (3, 12): (2.139, 2.111, 2.166, 40, 0.3, 0.054),
    (3, 13): (2.189, 2.160, 2.216, 40, 0.3, 0.055),
    (3, 14): (2.240, 2.211, 2.268, 40, 0.3, 0.056),
    (3, 15): (2.292, 2.262, 2.321, 40, 0.4, 0.057),
    (3, 16): (2.345, 2.314, 2.375, 40, 0.3, 0.058),
    (3, 17): (2.399, 2.367, 2.440, 41, 4.0, 0.059),
    # Band 4
    (4, 1): (2.445, 2.422, 2.474, 42, 7.6, 0.058),
    (4, 2): (2.506, 2.470, 2.542, 34, 0.4, 0.073),
    (4, 3): (2.575, 2.538, 2.612, 35, 0.5, 0.074),
    (4, 4): (2.646, 2.607, 2.684, 35, 0.5, 0.076),
    (4, 5): (2.719, 2.678, 2.758, 35, 0.6, 0.078),
    (4, 6): (2.794, 2.752, 2.835, 35, 0.8, 0.079),
    (4, 7): (2.871, 2.828, 2.914, 34, 0.6, 0.084),
    (4, 8): (2.951, 2.906, 2.995, 34, 0.6, 0.086),
    (4, 9): (3.033, 2.985, 3.079, 34, 0.6, 0.089),
    (4, 10): (3.118, 3.069, 3.164, 34, 0.6, 0.091),
    (4, 11): (3.204, 3.155, 3.252, 35, 0.5, 0.093),
    (4, 12): (3.293, 3.242, 3.342, 35, 0.5, 0.094),
    (4, 13): (3.383, 3.330, 3.434, 35, 0.6, 0.096),
    (4, 14): (3.476, 3.422, 3.528, 36, 0.6, 0.098),
    (4, 15): (3.570, 3.514, 3.624, 36, 0.6, 0.099),
    (4, 16): (3.667, 3.610, 3.723, 36, 0.8, 0.102),
    (4, 17): (3.768, 3.707, 3.827, 36, 0.9, 0.103),
    # Band 5
    (5, 1): (3.833, 3.8144, 3.8514, 109, 0.4, 0.035),
    (5, 2): (3.867, 3.8494, 3.8854, 109, 0.6, 0.036),
    (5, 3): (3.901, 3.8824, 3.9194, 109, 0.4, 0.036),
    (5, 4): (3.935, 3.9164, 3.9534, 108, 0.5, 0.036),
    (5, 5): (3.970, 3.9504, 3.9884, 109, 0.5, 0.036),
    (5, 6): (4.005, 3.9854, 4.0244, 110, 0.6, 0.036),
    (5, 7): (4.040, 4.0204, 4.0594, 110, 0.6, 0.037),
    (5, 8): (4.076, 4.0554, 4.0954, 111, 0.7, 0.037),
    (5, 9): (4.112, 4.0914, 4.1314, 112, 0.6, 0.037),
    (5, 10): (4.148, 4.1264, 4.1674, 113, 0.6, 0.037),
    (5, 11): (4.184, 4.1624, 4.2034, 113, 0.6, 0.037),
    (5, 12): (4.220, 4.1994, 4.2404, 115, 1.1, 0.037),
    (5, 13): (4.257, 4.2354, 4.2774, 114, 0.9, 0.037),
    (5, 14): (4.294, 4.2714, 4.3134, 115, 0.8, 0.037),
    (5, 15): (4.331, 4.3084, 4.3514, 116, 0.9, 0.037),
    (5, 16): (4.368, 4.3454, 4.3884, 116, 1.0, 0.038),
    (5, 17): (4.405, 4.3824, 4.4264, 117, 1.2, 0.038),
    # Band 6
    (6, 1): (4.435, 4.4184, 4.4524, 128, 1.6, 0.035),
    (6, 2): (4.467, 4.4504, 4.4844, 125, 1.1, 0.036),
    (6, 3): (4.500, 4.4824, 4.5174, 127, 1.2, 0.035),
    (6, 4): (4.533, 4.5154, 4.5504, 126, 1.2, 0.036),
    (6, 5): (4.566, 4.5484, 4.5844, 127, 1.3, 0.036),
    (6, 6): (4.600, 4.5824, 4.6184, 125, 1.4, 0.037),
    (6, 7): (4.635, 4.6154, 4.6534, 127, 1.4, 0.036),
    (6, 8): (4.669, 4.6494, 4.6884, 126, 1.7, 0.037),
    (6, 9): (4.704, 4.6844, 4.7234, 128, 1.9, 0.037),
    (6, 10): (4.739, 4.7194, 4.7584, 130, 3.4, 0.037),
    (6, 11): (4.773, 4.7534, 4.7924, 131, 2.4, 0.036),
    (6, 12): (4.808, 4.7884, 4.8274, 132, 2.6, 0.036),
    (6, 13): (4.843, 4.8224, 4.8624, 131, 2.3, 0.037),
    (6, 14): (4.879, 4.8574, 4.8994, 130, 2.5, 0.038),
    (6, 15): (4.915, 4.8934, 4.9364, 130, 2.8, 0.038),
    (6, 16): (4.951, 4.9304, 4.9724, 132, 3.0, 0.037),
    (6, 17): (4.988, 4.9664, 5.0094, 135, 4.2, 0.037),
}
# fmt: on


def make_reference_table(include_bandwidth: bool = True) -> Table:
    """Create SPECTRAL_CHANNELS table from reference data.

    Parameters
    ----------
    include_bandwidth : bool, optional
        If True, include BANDWIDTH column (default True).

    Returns
    -------
    Table
        Astropy table with spectral metadata for all channels.

    """
    rows = []
    for (detector, subchan), (wl, wl_min, wl_max, r, r_std, bandwidth) in REFERENCE_SPECTRAL_CHANNELS.items():
        row = {
            "DETECTOR": detector,
            "SUBCHAN": subchan,
            "WAVELENGTH": wl,
            "WL_MIN": wl_min,
            "WL_MAX": wl_max,
            "R": r,
            "R_STD": r_std,
        }
        if include_bandwidth:
            row["BANDWIDTH"] = bandwidth
        rows.append(row)

    table = Table(rows)
    table.sort(["DETECTOR", "SUBCHAN"])

    # Set dtypes
    table["DETECTOR"] = table["DETECTOR"].astype(np.uint8)
    table["SUBCHAN"] = table["SUBCHAN"].astype(np.uint8)
    for col in ["WAVELENGTH", "WL_MIN", "WL_MAX", "R", "R_STD"]:
        table[col] = table[col].astype(np.float32)
    if include_bandwidth:
        table["BANDWIDTH"] = table["BANDWIDTH"].astype(np.float32)

    # Add units
    table["WAVELENGTH"].unit = "um"
    table["WL_MIN"].unit = "um"
    table["WL_MAX"].unit = "um"
    if include_bandwidth:
        table["BANDWIDTH"].unit = "um"

    # Add descriptions
    table["DETECTOR"].description = "Detector array number (1-6)"
    table["SUBCHAN"].description = "Subchannel number (1-17)"
    table["WAVELENGTH"].description = "Effective (median) wavelength"
    table["WL_MIN"].description = "Min per-pixel wavelength"
    table["WL_MAX"].description = "Max per-pixel wavelength"
    table["R"].description = "Median resolving power"
    table["R_STD"].description = "Std dev of resolving power"
    if include_bandwidth:
        table["BANDWIDTH"].description = "Effective bandwidth"

    # Add format for floating point columns
    wl_cols = ["WAVELENGTH", "WL_MIN", "WL_MAX"]
    if include_bandwidth:
        wl_cols.append("BANDWIDTH")
    for col in wl_cols:
        table[col].format = "10.3f"

    table["R"].format = "10.0f"
    table["R_STD"].format = "10.1f"

    return table


def get_reference_spectral_channel_defs() -> LVFChannelDefinitionV2:
    """Get spectral channel definitions for standard channels.

    Built from the bundled LVF contour template and this module's
    hand-curated reference statistics (see `REFERENCE_SPECTRAL_CHANNELS_HISTORY`
    for provenance).
    """
    return LVFChannelDefinitionV2.from_reference()
