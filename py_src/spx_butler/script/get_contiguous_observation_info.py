from lsst.daf.butler import Butler
from astropy.time import Time
from pydantic import BaseModel
from typing import List, Dict, Optional


class ContiguityChunk(BaseModel):
    """Represents a chunk of consecutive observations.

    Attributes
    ----------
    type : `str`
        The type of the chunk — either "contiguous" (available data)
        or "missing" (commanded but not acquired).
    count : `int`
        Number of observations in the chunk.
    first_id : `int`
        Observation ID of the first observation in the chunk.
    last_id : `int`
        Observation ID of the last observation in the chunk.
    """

    type: str  # "contiguous" or "missing"
    count: int
    first_id: int
    last_id: int


class ContiguityReport(BaseModel):
    """Summary report of commanded vs available observations for a given detector.

    Attributes
    ----------
    contig_chunks : `List[ContiguityChunk]`
        List of contiguous and missing observation blocks, in order.
    total_commanded : `int`
        Total number of commanded observations in the time range.
    total_available : `int`
        Number of commanded observations with available level1 data.
    total_missing : `int`
        Number of commanded observations without corresponding data.
    commanded_first_id : `int`
        ID of the first commanded observation in the overlap window.
    commanded_last_id : `int`
        ID of the last commanded observation in the overlap window.
    commanded_first_time : `str`
        Start time (ISO format) of the first commanded observation.
    commanded_last_time : `str`
        End time (ISO format) of the last commanded observation.
    available_first_id : `Optional[int]`
        ID of the first available observation.
    available_last_id : `Optional[int]`
        ID of the last available observation.
    available_first_time :`Optional[str]`
        Start time (ISO format) of the first available observation.
    available_last_time : `Optional[str]`
        End time (ISO format) of the last available observation.
    spsplans : `List[int]`
        List of SPS Plan IDs (`spsplan`) associated with the available level1 data.
    """

    contig_chunks: List[ContiguityChunk]
    total_commanded: int
    total_available: int
    total_missing: int
    commanded_first_id: int
    commanded_last_id: int
    commanded_first_time: str
    commanded_last_time: str
    available_first_id: Optional[int]
    available_last_id: Optional[int]
    available_first_time: Optional[str]
    available_last_time: Optional[str]
    spsplans: List[int]


def get_contiguous_observation_info(repo: str,
                                    collections: List[str],
                                    detector: int,
                                    instrument: str = 'spx') -> Dict[int, ContiguityReport]:
    """Generate a continuity report for level1 observations from a given detector.

    This function compares available level1 datasets (acquired observations)
    against the set of commanded observations for the same detector and instrument,
    identifying which commanded observations are missing in the data and organizing
    them into contiguous and missing observation chunks.

    The report includes the total number of commanded and available observations,
    the time span of each set, and all identified SPS plan IDs (`spsplan`) present
    in the available datasets.

    The time range of commanded observations is filtered to only those that overlap
    with the start and end times of the available data (based on exposure times).

    Parameters
    ----------
    repo : `str`
        Path to the Butler repository (or an initialized Butler instance).
    collections : `List[str]`
        One or more input Butler collections to search for level1 datasets.
    detector : `int`
        Detector ID to filter observations (e.g., values 1–6 for SPHEREx).
    instrument : `str`, optional
        Instrument name, by default "spx".

    Returns
    -------
    `Dict[int, ContiguityReport]`
        A dictionary keyed by detector ID (only one in this case), containing
        a `ContiguityReport` object that summarizes the observation continuity
        and availability across the specified time window.

    Raises
    ------
    RuntimeError
        If no level1 datasets are found, or if no commanded observations overlap
        with the available data time range.

    Example Usage
    -------------
    ```
    butler report-contiguous-obs \
    --repo ./butler \
    --collections level1 \
    --detector 1 \
    --output /tmp/contig_report_1.json
    ```

    Returns :
    {
    "1": {
        "contig_chunks": [
        {
            "type": "contiguous",
            "count": 101,
            "first_id": 202552223056710,
            "last_id": 202552223061910
        }
        ],
        "total_commanded": 101,
        "total_available": 101,
        "total_missing": 0,
        "commanded_first_id": 202552223056710,
        "commanded_last_id": 202552223061910,
        "commanded_first_time": "2025-12-27T15:39:01.957500",
        "commanded_last_time": "2025-12-27T20:08:56.565900",
        "available_first_id": 202552223056710,
        "available_last_id": 202552223061910,
        "available_first_time": "2025-12-27T15:39:01.957500",
        "available_last_time": "2025-12-27T20:06:59.913500",
        "spsplans": [
        202552223
        ]
    }
    }
    Contiguity report saved to /tmp/contig_report_1.json

    ---

    To use the output json file to get the first contiguous block's observations:
    ```
    python -c "\
    import json; \
    d = json.load(open('/tmp/contig_report_1.json')); \
    chunks = d['1']['contig_chunks']; \
    c = next(x for x in chunks if x['type'] == 'contiguous'); \
    print(c['first_id'], c['last_id'])"
    ```

    will print in this example: 202552223056710 202552223061910
    }
    """
    if isinstance(repo, Butler):
        butler = repo
    else:
        butler = Butler(repo)

    # Get level1 observation IDs for this detector and collection
    level1_refs = butler.query_datasets("level1",
                                        collections=collections,
                                        where=f"instrument='{instrument}' AND detector={detector}",
                                        order_by="observation")
    if not level1_refs:
        raise RuntimeError("No available level1 observations found for this detector.")

    spsplans = {ref.dataId["spsplan"] for ref in level1_refs if "spsplan" in ref.dataId}
    spsplans_str = ",".join(str(plan) for plan in spsplans)

    level1_obs = [ref.dataId["observation"] for ref in level1_refs]
    level1_obs_set = set(level1_obs)

    # Determine start and end time from available observations
    start_mjds = []
    end_mjds = []
    for ref in level1_refs:
        record = butler.get("level1", ref.dataId, collections=collections)
        start_mjds.append(record.getExposureStartTime())
        end_mjds.append(record.getExposureEndTime())

    time_start = min(start_mjds)
    time_end = max(end_mjds)

    # Get commanded observation records overlapping with available time range
    where_clause = (
        f"instrument='{instrument}' AND detector={detector} AND spsplan IN ({spsplans_str})"
    )
    commanded_all = list(butler.query_dimension_records(
        "observation",
        where=where_clause,
        order_by="observation"
    ))

    # Filter to overlap with available data
    commanded_obs = [
        obs for obs in commanded_all
        if obs.timespan.begin < time_end and obs.timespan.end > time_start
    ]

    if not commanded_obs:
        raise RuntimeError("No commanded observations found overlapping available data for this detector.")

    # Compare availability
    contig_chunks = []
    current_chunk = None

    available_obs_ids = []
    available_obs_times = {}

    for obs in commanded_obs:
        obs_id = obs.id
        obs_time = Time(obs.timespan.begin, format="mjd").isot
        in_available = obs_id in level1_obs_set

        if in_available:
            available_obs_ids.append(obs_id)
            available_obs_times[obs_id] = obs_time

        chunk_type = "contiguous" if in_available else "missing"

        if current_chunk is None:
            current_chunk = {
                "type": chunk_type,
                "first_id": obs_id,
                "last_id": obs_id,
                "count": 1,
            }
        elif current_chunk["type"] == chunk_type:
            current_chunk["last_id"] = obs_id
            current_chunk["count"] += 1
        else:
            contig_chunks.append(ContiguityChunk(**current_chunk))
            current_chunk = {
                "type": chunk_type,
                "first_id": obs_id,
                "last_id": obs_id,
                "count": 1,
            }

    # Add the last chunk
    if current_chunk is not None:
        contig_chunks.append(ContiguityChunk(**current_chunk))

    # Summary fields
    total_commanded = len(commanded_obs)
    total_available = len(available_obs_ids)
    total_missing = total_commanded - total_available

    commanded_first = commanded_obs[0]
    commanded_last = commanded_obs[-1]

    available_first_id = available_obs_ids[0] if available_obs_ids else None
    available_last_id = available_obs_ids[-1] if available_obs_ids else None
    available_first_time = available_obs_times.get(available_first_id)
    available_last_time = available_obs_times.get(available_last_id)

    return {
        detector: ContiguityReport(
            contig_chunks=contig_chunks,
            total_commanded=total_commanded,
            total_available=total_available,
            total_missing=total_missing,
            commanded_first_id=commanded_first.id,
            commanded_last_id=commanded_last.id,
            commanded_first_time=Time(commanded_first.timespan.begin, format="mjd").isot,
            commanded_last_time=Time(commanded_last.timespan.end, format="mjd").isot,
            available_first_id=available_first_id,
            available_last_id=available_last_id,
            available_first_time=available_first_time,
            available_last_time=available_last_time,
            spsplans=spsplans
        )
    }
