from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
from spx_data import ObservationId


DET_TO_OPPOSITE = {
    1: 4, 4: 1,
    2: 5, 5: 2,
    3: 6, 6: 3,
}

DET_TO_DICHROICPAIR = {
    1: 1, 4: 1,
    2: 2, 5: 2,
    3: 3, 6: 3,
}


def opposite_detector(detector: int) -> int:
    """Get dichroic pair detector."""
    try:
        return DET_TO_OPPOSITE[int(detector)]
    except KeyError as e:
        raise ValueError(f"Unsupported detector {detector}") from e


def _to_builtin(value):
    """Convert numpy/astropy scalar values to YAML-safe builtin Python types."""
    if isinstance(value, np.generic):
        return value.item()
    return value


def build_l2bwcs_payload(header, *, source_collection: str) -> dict:
    """Serialize just enough WCS info for retry seeding."""
    obsid_str = str(header["OBSID"])
    obs = ObservationId(obsid_str)
    detector = int(header["DETECTOR"])
    wcs = WCS(header)
    hdr = wcs.to_header(relax=True)

    payload = {
        "observation": int(obs.pointing_id),
        "obsid": obsid_str,
        "detector": detector,
        "dichroicpair": int(DET_TO_DICHROICPAIR[detector]),
        "finast": int(header.get("FINAST", -999)),
        "source_collection": str(source_collection),
        "large_slew_counter": int(obs.large_slew_counter),
        "small_slew_counter": int(obs.small_slew_counter),
        "slewgroup_id": int(obs.slewgroup_id),
        "slewgroup_counter_id": int(obs.slewgroup_counter_id),
        "plan_name": str(obs.plan_name),
        "sps_id": str(obs.sps_id),
        "wcs_header": {str(k): _to_builtin(hdr[k]) for k in hdr.keys()},
    }
    return payload


def payload_to_wcs(payload: dict) -> WCS:
    """Convert payload to WCS."""
    hdr = fits.Header()
    for k, v in payload["wcs_header"].items():
        hdr[k] = v
    return WCS(hdr)
