__all__ = ["ImageMetadata", "ImageMetadataStats", "set_metadata_provenance"]
import importlib.metadata
from datetime import datetime, UTC
from typing import Optional, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, field_serializer, field_validator

ssdc_pipelines_version = importlib.metadata.version("spherex_pipelines")

_history = [
    "[SSDC] Spec: SPHEREx_ICE_FSW_KA_Telemetry_Database",
    f"[SSDC] SSDC pipelines version: {ssdc_pipelines_version}"
]

_meta2_history = ["[SSDC] Image Metadata 2 block content"] + _history
_meta3_history = ["[SSDC] Image Metadata 3 block content"] + _history


class Provenance(BaseModel):
    """Attributes
    ----------
    date_created : `str`
        UTC date/time file was created.
    observation : `int`
        SSDC Observation ID.
    detector: `int`

    """

    date_created: str
    observation: int
    detector: int


class ImageMetadata(BaseModel):
    """Metadata necessary to interpret the content of compressed image blocks.

    Attributes
    ----------
    start_time_days : `int`
        Observation start time days since 1/1/1958 (CDS Timestamp format).
    start_time_ms : `int`
        Observation start time milliseconds of day (since midnight).
    start_time_us : `int`
        Observation start time microseconds within millisecond.
    rob_temp_start : `np.ndarray`
        Averaged ROB temperatures at the beginning of observation.
    fpa_temp_start : `np.ndarray`
        Averaged FPA temperatures at the beginning of observation.
    ceb_temp_start : `np.ndarray`
        Averaged CEB temperatures at the beginning of observation.
    aux_temp_start : `np.ndarray`
        Averaged AUX temperatures at the beginning of observation.
    rob_temp_end : `np.ndarray`
        Averaged ROB temperatures at the end of observation.
    fpa_temp_end : `np.ndarray`
        Averaged FPA temperatures at the end of observation.
    ceb_temp_end : `np.ndarray`
        Averaged CEB temperatures at the end of observation.
    aux_temp_end : `np.ndarray`
        Averaged AUX temperatures at the end of observation.
    regpix_binwidth : `int`
        Regular pixel compression bin width.
    refpix_binwidth : `int`
        Reference pixel compression bin width.
    phanpix_binwidth : `int`
        Phantom pixel compression bin width.
    avg_phanpix_slopes : `np.ndarray`
        Average phantom pixel slopes.
    surlim : `int`
        Number of SUR frames.
    nskip : `int`
        Row skip parameter.
    nref : `int`
        Number of times the top/bottom reference rows are read.
    ken : `np.ndarray`
        Transient detection white noise threshold parms.
    ksn : `np.ndarray`
        Transient detection shot noise threshold parms.
    kovr : `np.ndarray`
        Overflow detection limits (in ADU).
    rstdly_surcnt : `int`
        SURCNT value below which sur_math is in reset mode with SETY0 enabled.
    surdly_surcnt : `int`
        SURCNT value at which SUR sums accumulation begins.
    trandly_surcnt : `int`
        SURCNT value at which transient detection begins.
    ovrdly_surcnt : `int`
        SURCNT value below which overflow sets error flag
    tcut_surcnt : `int`
        SURCNT value below which transient detection sets error flag
    raw_row_start : `int`
        Starting row number (5-2037) for raw samples, by default 5.
    raw_col_section : `int`
        Column section number (0-63) for raw samples, by default 0.
    framet : `int`
        Length of the H2RG frame in units of 10 usec.
    block_size: `int`
        Image Metadata 2 block size in words (235 in revN)
    assumed_framet : `int`
        Frame time in seconds assumed in image processing.
    provenance : `list[~spx_data.ProvenanceItem]`, optional
        Provenance information.
    history : `list[str]`, optional
        History information.

    """

    start_time_days: int
    start_time_ms: int
    start_time_us: int
    rob_temp_start: np.ndarray
    fpa_temp_start: np.ndarray
    ceb_temp_start: np.ndarray
    aux_temp_start: np.ndarray
    rob_temp_end: np.ndarray
    fpa_temp_end: np.ndarray
    ceb_temp_end: np.ndarray
    aux_temp_end: np.ndarray
    regpix_binwidth: int
    refpix_binwidth: int
    phanpix_binwidth: int
    avg_phanpix_slopes: np.ndarray
    surlim: int
    nskip: int
    nref: int
    ken: np.ndarray
    ksn: np.ndarray
    kovr: np.ndarray
    rstdly_surcnt: int
    surdly_surcnt: int
    trandly_surcnt: int
    ovrdly_surcnt: Optional[int] = None  # added in revN
    tcut_surcnt: Optional[int] = None  # added in revN
    framet: Optional[int] = None  # added in revN
    raw_row_start: int = 5
    raw_col_section: int = 0
    assumed_framet: Optional[float] = None
    block_size: int
    provenance: Optional[Provenance] = None
    history: list[str] = _meta2_history

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("rob_temp_start", "fpa_temp_start", "ceb_temp_start", "aux_temp_start",
                     "rob_temp_end", "fpa_temp_end", "ceb_temp_end", "aux_temp_end",
                     "avg_phanpix_slopes", "ken", "ksn", "kovr",
                     mode="before")
    @classmethod
    def convert_to_array(cls, v):
        if isinstance(v, (list, np.ndarray)):
            return np.array(v)
        raise ValueError(f"Expected list or ndarray, got {type(v)}")

    @field_serializer("rob_temp_start", "fpa_temp_start", "ceb_temp_start", "aux_temp_start",
                      "rob_temp_end", "fpa_temp_end", "ceb_temp_end", "aux_temp_end",
                      "avg_phanpix_slopes", "ken", "ksn", "kovr")
    def serialize_array(self, value: np.ndarray):
        return value.tolist()

    def get_binwidth(self, pix_type: Literal[0, 1, 2]) -> int:
        """Get the compression bin width for pixel type.

        Parameter
        ---------
        pix_type : Literal[0, 1, 2]
            Pixel type: 0 = normal, 1 = reference, 2 = phantom.

        Returns
        -------
        bin_width : `int`
            Compression bin width.

        """
        match pix_type:
            case 0:
                bin_width = self.regpix_binwidth
            case 1:
                bin_width = self.refpix_binwidth
            case 2:
                bin_width = self.phanpix_binwidth
            case _:
                raise ValueError(f"Invalid pix_type {pix_type}")
        return bin_width


class ImageMetadataStats(BaseModel):
    """Metadata needed to compare on-board statistics to observed statistics.

    Attributes
    ----------
    start_time_days : `int`
        Observation start time days since 1/1/1958 (CDS Timestamp fomat).
    start_time_ms : `int`
        Observation start time milliseconds of day (since midnight).
    start_time_us : `int`
        Observation start time microseconds within millisecond.
    regpix_flagstats : `np.ndarray`
        Regular pixel flag statistics.
    refpix_flagstats : `np.ndarray`
        Reference pixel flag statistics.
    phanpix_flagstats : `np.ndarray`
        Phantom pixel flag statistics.
    nbits_stats : `np.ndarray`
        nbits statistic.
    blocklen_stats : `np.ndarray`
        Block length statistics.
    obswords_stats : `int`
        Number of 3-byte words of compressed data for the observation.
    msecs_stats : `int`
        Elapsed time to compress/packetize/transmit the observation data.
    block_size : `int`
        Image Metadata 3 block size in words; 46 in revN
    provenance : `list[~spx_data.ProvenanceItem]`, optional
        Provenance information.
    history : `list[str]`, optional
        History information.

    Notes
    -----
    For flag statistics pixel_flag = {sur_error, overflow, transient}.
    8 items are counts[pixel_flag], for pixel_flag = 0, 1, ..., 7.

    """

    start_time_days: int
    start_time_ms: int
    start_time_us: int
    regpix_flagstats: np.ndarray
    refpix_flagstats: np.ndarray
    phanpix_flagstats: np.ndarray
    nbits_stats: np.ndarray
    blocklen_stats: np.ndarray
    obswords_stats: int
    msecs_stats: int
    block_size: int
    provenance: Optional[Provenance] = None
    history: list[str] = _meta3_history

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("regpix_flagstats", "refpix_flagstats", "phanpix_flagstats",
                     "nbits_stats", "blocklen_stats",
                     mode="before")
    @classmethod
    def convert_to_array(cls, v):
        if isinstance(v, (list, np.ndarray)):
            return np.array(v)
        raise ValueError(f"Expected list or ndarray, got {type(v)}")

    @field_serializer("regpix_flagstats", "refpix_flagstats", "phanpix_flagstats",
                      "nbits_stats", "blocklen_stats")
    def serialize_array(self, value: np.ndarray):
        return value.tolist()


def set_metadata_provenance(meta2: ImageMetadata | None,
                            meta3: ImageMetadataStats | None,
                            observation: int,
                            detector: int):
    """Set provenance information on image metadata objects."""
    if meta2 is None and meta3 is None:
        return

    provenance = Provenance(
        date_created=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S"),
        observation=observation,
        detector=detector
    )

    if meta2 is not None:
        meta2.provenance = provenance
    if meta3 is not None:
        meta3.provenance = provenance
