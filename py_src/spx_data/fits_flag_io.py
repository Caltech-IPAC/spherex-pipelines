"""FITS header I/O for flag definitions.

This module provides functions to write and read flag definitions to/from
FITS headers. It supports two formats:

1. MP_ format (current): MP_<FLAG_NAME> = bit_position
2. MSK format (alternative): MSKNnnnn/MSKMnnnn/MSKDnnnn triplets

The format can be selected per-call or globally via DEFAULT_FLAG_FORMAT.

Examples
--------
Write specific flags from a bitmask::

    >>> from spx_data import Flag
    >>> write_flag_headers(header, Flag.HOT | Flag.COLD)

Write using MSK format::

    >>> write_flag_headers(header, Flag.HOT | Flag.COLD, format=FlagHeaderFormat.MSK)

Read flag definitions::

    >>> flag_defs = read_flag_headers(header)  # auto-detects format
    >>> print(flag_defs)
    {'HOT': 10, 'COLD': 11}

Read as bitmask::

    >>> bitmask = read_flag_headers_as_bitmask(header)
    >>> bool(bitmask & Flag.HOT)
    True
"""

__all__ = [
    'FlagHeaderFormat',
    'DEFAULT_FLAG_FORMAT',
    'write_flag_headers',
    'read_flag_headers',
    'read_flag_headers_as_bitmask',
]

from enum import Enum
from math import log2
from typing import Dict, Iterable, Optional, Union

from astropy.io import fits

from .spx_flags import Flag, flag2bit, flag_description


class FlagHeaderFormat(Enum):
    """Flag header format options.

    Attributes
    ----------
    MP : `str`
        MP_<FLAG_NAME> = bit_position format (current standard).
        Uses HIERARCH prefix for keywords longer than 8 characters.
    MSK : `str`
        MSKNnnnn/MSKMnnnn/MSKDnnnn triplet format.
        Each flag uses three keywords: name, mask value, and description.
    """

    MP = "mp"
    MSK = "msk"


# Default format for writing (can be changed globally)
DEFAULT_FLAG_FORMAT = FlagHeaderFormat.MSK


def write_flag_headers(
    header: fits.Header,
    flags: Union[int, Dict[str, int], Iterable[Flag], None] = None,
    *,
    format: Optional[FlagHeaderFormat] = None,
    set_exttype: bool = True
) -> None:
    """Write flag definitions to FITS header.

    Follows the Firefly convention: EXTTYPE=MASK and flag definitions
    in standardized keyword format.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header to update (modified in place).
    flags : `int`, `dict`, `Iterable[Flag]`, or `None`, optional
        Flags to write. Accepts multiple input types:

        - ``int``: Bitwise OR of Flag values. Only flags present in
          the bitmask are written. A value of 0 writes nothing.
        - ``dict``: Mapping of {flag_name: bit_position}.
        - ``Iterable[Flag]``: Sequence of Flag enum members.
        - ``None``: Writes nothing (default). Use this when there are
          no flags to propagate.

    format : `FlagHeaderFormat`, optional
        Header format to use. If None, uses DEFAULT_FLAG_FORMAT.
    set_exttype : `bool`, optional
        If True (default), sets EXTTYPE='MASK' header keyword.
        This is the convention for Firefly mask overlay support.

    Examples
    --------
    Write from bitmask::

        >>> write_flag_headers(header, Flag.HOT | Flag.COLD | Flag.OVERFLOW)

    Write from dict::

        >>> write_flag_headers(header, {'HOT': 10, 'COLD': 11})

    Write specific flags::

        >>> write_flag_headers(header, [Flag.HOT, Flag.COLD])

    Write using MSK format::

        >>> flags = Flag.HOT | Flag.COLD
        >>> write_flag_headers(header, flags, format=FlagHeaderFormat.MSK)

    Skip EXTTYPE keyword::

        >>> write_flag_headers(header, Flag.HOT, set_exttype=False)
    """
    if format is None:
        format = DEFAULT_FLAG_FORMAT

    # Normalize input to dict[str, int]
    flag_defs = _normalize_flags_input(flags)

    if not flag_defs:
        return

    if format == FlagHeaderFormat.MP:
        _write_mp_format(header, flag_defs, set_exttype)
    elif format == FlagHeaderFormat.MSK:
        _write_msk_format(header, flag_defs, set_exttype)
    else:
        raise ValueError(f"Unknown format: {format}")


def read_flag_headers(
    header: fits.Header,
    format: Optional[FlagHeaderFormat] = None,
    *,
    remove_from_header: bool = False
) -> Dict[str, int]:
    """Read flag definitions from FITS header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header containing flag definitions.
    format : `FlagHeaderFormat`, optional
        Header format to read. If None, auto-detects based on
        header content (looks for MSKN0001 for MSK format,
        otherwise assumes MP_ format).
    remove_from_header : `bool`, optional
        If True, removes the flag definition keywords from the header
        after reading. This is useful when the header will be stored
        separately from the flag definitions to avoid duplication.
        Default is False.

    Returns
    -------
    flag_defs : `dict`
        Dictionary mapping flag names to bit positions.
        Returns empty dict if no flag definitions found.

    Examples
    --------
    Read with auto-detection::

        >>> flag_defs = read_flag_headers(header)
        >>> print(flag_defs)
        {'HOT': 10, 'COLD': 11}

    Read specific format::

        >>> flag_defs = read_flag_headers(header, format=FlagHeaderFormat.MP)

    Read and remove keywords from header::

        >>> flag_defs = read_flag_headers(header, remove_from_header=True)
    """
    if format is None:
        format = _detect_format(header)

    if format == FlagHeaderFormat.MP:
        flag_defs = _read_mp_format(header)
    elif format == FlagHeaderFormat.MSK:
        flag_defs = _read_msk_format(header)
    else:
        raise ValueError(f"Unknown format: {format}")

    if remove_from_header:
        _remove_flag_keywords(header, format)

    return flag_defs


def read_flag_headers_as_bitmask(
    header: fits.Header,
    format: Optional[FlagHeaderFormat] = None
) -> int:
    """Read flag definitions from FITS header and return as bitmask.

    Parses flag definitions and reconstructs the bitwise OR of all
    Flag enum values found in the header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        FITS header containing flag definitions.
    format : `FlagHeaderFormat`, optional
        Header format to read. If None, auto-detects.

    Returns
    -------
    bitmask : `int`
        Bitwise OR of Flag values found in header.
        Returns 0 if no flag definitions found.

    Examples
    --------
    ::

        >>> bitmask = read_flag_headers_as_bitmask(header)
        >>> bool(bitmask & Flag.HOT)
        True
        >>> bool(bitmask & Flag.OVERFLOW)
        False
    """
    flag_defs = read_flag_headers(header, format)

    bitmask = 0
    for flag_name in flag_defs:
        flag_enum = getattr(Flag, flag_name, None)
        if flag_enum is not None:
            bitmask |= int(flag_enum)

    return bitmask


# ---------------------------------------------------------------------------
# Internal helper functions
# ---------------------------------------------------------------------------

def _normalize_flags_input(
    flags: Union[int, Dict[str, int], Iterable[Flag], None]
) -> Dict[str, int]:
    """Normalize various input types to dict[str, int].

    Parameters
    ----------
    flags : `int`, `dict`, `Iterable[Flag]`, or `None`
        Input flags in various formats.

    Returns
    -------
    flag_defs : `dict`
        Dictionary mapping flag names to bit positions.
    """
    if flags is None:
        # None means no flags to write (consistent with propagated_flags=None)
        return {}

    if isinstance(flags, int):
        # Bitmask: extract present flags
        if flags == 0:
            return {}
        return {f.name: flag2bit(f) for f in Flag if flags & f}

    if isinstance(flags, dict):
        # Already a dict - validate and return
        return dict(flags)

    # Iterable of Flag enums
    return {f.name: flag2bit(f) for f in flags}


def _write_mp_format(
    header: fits.Header,
    flag_defs: Dict[str, int],
    set_exttype: bool
) -> None:
    """Write MP_<FLAG_NAME> format.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to update.
    flag_defs : `dict`
        Mapping of flag names to bit positions.
    set_exttype : `bool`
        Whether to set EXTTYPE keyword.
    """
    if set_exttype:
        header["EXTTYPE"] = ("MASK", "Extension can be used as a mask overlay")

    # Sort by bit position for readability
    for name, bit_pos in sorted(flag_defs.items(), key=lambda x: x[1]):
        keyword = f"MP_{name}"
        # FITS standard: keywords > 8 chars need HIERARCH
        if len(keyword) > 8:
            keyword = f"HIERARCH {keyword}"

        # Get description from Flag enum if available
        flag_enum = getattr(Flag, name, None)
        description = flag_description(flag_enum) if flag_enum else ""

        header[keyword] = (bit_pos, description)


def _write_msk_format(
    header: fits.Header,
    flag_defs: Dict[str, int],
    set_exttype: bool
) -> None:
    """Write MSKNnnnn/MSKMnnnn/MSKDnnnn format.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to update.
    flag_defs : `dict`
        Mapping of flag names to bit positions.
    set_exttype : `bool`
        Whether to set EXTTYPE keyword.
    """
    if set_exttype:
        header["EXTTYPE"] = ("MASK", "Extension can be used as a mask overlay")

    # Sort by bit position and write triplets
    # MSK keyword number matches bit position (starting from 0)
    for name, bit_pos in sorted(flag_defs.items(), key=lambda x: x[1]):
        suffix = f"{bit_pos:04d}"
        mask_value = 1 << bit_pos  # Convert bit position to mask value

        # Get description from Flag enum if available
        flag_enum = getattr(Flag, name, None)
        description = flag_description(flag_enum) if flag_enum else ""

        header[f"MSKN{suffix}"] = (name, f"Name for mask plane {bit_pos}")
        header[f"MSKM{suffix}"] = (mask_value, f"Bitmask for plane n={bit_pos}; always 1<<n")
        header[f"MSKD{suffix}"] = description


def _read_mp_format(header: fits.Header) -> Dict[str, int]:
    """Read MP_<FLAG_NAME> keywords.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to read from.

    Returns
    -------
    flag_defs : `dict`
        Mapping of flag names to bit positions.
    """
    flag_defs = {}

    for key, val in header.items():
        if key.startswith("MP_"):
            try:
                intval = int(val)
                if intval > 63:
                    # Allow 64-bit flags (PhotometryFlag uses bits 32-33)
                    raise ValueError(
                        f"bit value {intval} for flag {key} is too large"
                    )
                flag_name = key[3:]  # Remove "MP_" prefix
                flag_defs[flag_name] = intval
            except (ValueError, TypeError):
                # Skip non-integer values
                pass

    return flag_defs


def _read_msk_format(header: fits.Header) -> Dict[str, int]:
    """Read MSKNnnnn/MSKMnnnn keywords.

    MSK keyword numbers match bit positions (starting from 0), so there
    may be gaps in the numbering sequence.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to read from.

    Returns
    -------
    flag_defs : `dict`
        Mapping of flag names to bit positions.
    """
    flag_defs = {}

    # Scan header for all MSKN* keywords (handles gaps in numbering)
    for key in header.keys():
        if key.startswith("MSKN") and len(key) == 8:
            suffix = key[4:]  # e.g., "0003"
            mask_key = f"MSKM{suffix}"

            if mask_key not in header:
                raise ValueError(f"Missing {mask_key} for {key}")

            name = str(header[key]).strip()
            mask_val = int(header[mask_key])

            # Convert mask value to bit position
            bit_pos = int(log2(mask_val))
            flag_defs[name] = bit_pos

    return flag_defs


def _detect_format(header: fits.Header) -> FlagHeaderFormat:
    """Auto-detect format from header content.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to analyze.

    Returns
    -------
    format : `FlagHeaderFormat`
        Detected format.
    """
    # Check for MSK format first (more specific)
    # Look for any MSKN keyword since numbering matches bit position
    for key in header.keys():
        if key.startswith("MSKN") and len(key) == 8:
            return FlagHeaderFormat.MSK

    # Default to MP_ format
    return FlagHeaderFormat.MP


def _remove_flag_keywords(
    header: fits.Header,
    format: Optional[FlagHeaderFormat] = None
) -> None:
    """Remove flag definition keywords from header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        Header to clean (modified in place).
    format : `FlagHeaderFormat`, optional
        If specified, only removes keywords of this format.
        If None, removes all flag keywords (both MP_ and MSK formats).
    """
    # Remove MP_ keywords
    if format is None or format == FlagHeaderFormat.MP:
        mp_keys = [key for key in header.keys() if key.startswith("MP_")]
        for key in mp_keys:
            header.remove(key, ignore_missing=True, remove_all=True)

    # Remove MSK format keywords (handles gaps in numbering)
    if format is None or format == FlagHeaderFormat.MSK:
        msk_keys = [
            key for key in header.keys()
            if key.startswith(("MSKN", "MSKM", "MSKD")) and len(key) == 8
        ]
        for key in msk_keys:
            header.remove(key, ignore_missing=True)
