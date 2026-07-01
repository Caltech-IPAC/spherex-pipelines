__all__ = ["to_dichroicpair", "from_dichroicpair", "ensure_registered_instrument"]
from lsst.daf.butler import Registry
from lsst.pipe.base import Instrument


def to_dichroicpair(detector: int) -> int:
    """Calculate the dichroic pair ID from the detector ID.

    Parameters
    ----------
    detector : int
        Detector ID.

    Returns
    -------
    dichroicpair : int
        The matching dichroic pair ID.
    """
    return detector if detector < 4 else detector - 3


def from_dichroicpair(dichroicpair: int) -> tuple[int, int]:
    """Get IDs of the detectors from the dichroic pair ID.

    Parameters
    ----------
    dichroicpair : int
        Dichroic pair ID.

    Returns
    -------
    result : tuple[int, int]
        A tuple with matching SWIR and MWIR detector IDs.
    """
    # the SWIR detector ID is the same as dichroic pair id,
    # the MWIR detector ID is SWIR detector ID plus 3
    if dichroicpair < 1 or dichroicpair > 3:
        raise ValueError(f"Invalid dichroic pair ID value {dichroicpair}")
    return dichroicpair, dichroicpair + 3


def ensure_registered_instrument(instrument: Instrument, registry: Registry) -> None:
    """Register the instrument if it's not already registered.

    This ensures basic dimension tables such as instrument, detector, dichroicpair,
    and slewgroup_counter are populated.

    Parameters
    ----------
    instrument :
        Instrument to register
    registry :
        Butler registry
    """
    try:
        _ = Instrument.fromName(instrument.getName(), registry)
    except LookupError:
        instrument.register(registry)
