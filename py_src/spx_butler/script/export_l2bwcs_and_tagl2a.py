import re
import logging

from astropy.io import fits
from lsst.daf.butler import Butler, CollectionType, DatasetType

from spx_utils.l2bwcs_utils import build_l2bwcs_payload
from spx_data import ObservationId
from ..instrument import SPHERExInstrument

logger = logging.getLogger(__name__)

SPS_PLAN_PATTERN = re.compile(r"(^\d\d\d\dW\d\d_\d[A-Z]$)")


def ensure_dataset_type(butler: Butler, name: str = "l2bwcs") -> None:
    """Ensure dataset type handling."""
    registry = butler.registry
    try:
        registry.getDatasetType(name)
        return
    except Exception:
        pass

    dt = DatasetType(
        name=name,
        dimensions=("instrument", "observation"),
        storageClass="StructuredDataDict",
        universe=butler.dimensions,
    )
    registry.registerDatasetType(dt)


def ensure_collection(butler: Butler, name: str, collection_type) -> None:
    """Ensure collection handling."""
    registry = butler.registry
    try:
        registry.getCollectionType(name)
        return
    except Exception:
        registry.registerCollection(name, type=collection_type)


def _make_slewgroup_key(observation: int) -> tuple:
    obs = ObservationId(observation)
    return (
        obs.year,
        obs.week,
        obs.plan_number,
        obs.plan_version,
        obs.large_slew_counter,
    )


def _detector_payload_with_obs_meta(ref, header, input_run: str) -> dict:
    observation = int(ref.dataId["observation"])
    detector = int(ref.dataId["detector"])
    obs = ObservationId(observation)

    payload = build_l2bwcs_payload(
        header,
        source_collection=input_run,
    )

    payload["instrument"] = str(ref.dataId["instrument"])
    payload["observation"] = observation
    payload["detector"] = detector
    payload["finast"] = int(header.get("FINAST", -999))
    payload["small_slew_counter"] = int(obs.small_slew_counter)
    payload["large_slew_counter"] = str(obs.large_slew_counter)
    payload["slewgroup_id"] = int(obs.slewgroup_id)
    payload["slewgroup_counter_id"] = int(obs.slewgroup_counter_id)
    payload["plan_name"] = obs.plan_name
    payload["sps_id"] = obs.sps_id
    return payload


def export_l2bwcs_and_tagl2a(
    repo: str,
    collections: str,
    output_run: str,
    output_tag: str,
    in_type: str = "level2",
    out_type: str = "l2bwcs",
    sps_plan: str = None,
    instrument: str = SPHERExInstrument.getName(),
    progress_every: int = 500,
):
    """Export L2b WCS and make a tagged collection of L2a corresponding
    to L2b finast failures.

    Parameters
    ----------
    repo : `str`
        URI to the repository.
    collections : `str`
        Input collections searched for both level2 and level2a datasets.
    output_run : `str`
        The path to the location, the run, where l2bwcs datasets should be put.
    output_tag : `str`
        Output tag collection name for level2a finast failures.
    in_type : `str`, optional
        Input images data type.
    out_type : `str`, optional
        Output l2bwcs data type.
    sps_plan : `str`, optional
        SPS plan.
    instrument : `str`, optional
        Instrument dimension name.
    progress_every : `int`, optional
        Outputs progress every N exposures. Default is 500.
    """
    butler = Butler(repo, writeable=True)

    ensure_dataset_type(butler, out_type)
    ensure_collection(butler, output_run, CollectionType.RUN)
    ensure_collection(butler, output_tag, CollectionType.TAGGED)

    collections = list(collections)

    where = f"instrument='{instrument}'"

    if sps_plan:
        match = SPS_PLAN_PATTERN.search(sps_plan)
        if not match:
            raise ValueError(f"Invalid SPS Plan name: {sps_plan}")
        spsplan_int = ObservationId(sps_plan + "_0001_1").plan_id
        where += f" AND spsplan={spsplan_int}"

    level2_refs = list(
        butler.query_datasets(
            in_type,
            collections=collections,
            where=where,
            find_first=True,
        )
    )

    slewgroups = {}
    retry_data_ids = []
    retry_keys = set()

    n_total = 0
    n_good = 0
    n_bad = 0

    for ref in level2_refs:
        n_total += 1
        uri = butler.getURI(ref)
        header = fits.getheader(uri.ospath, ext=1)

        sps_flags = str(header.get("SPS_FLAGS", "")).lower()
        if "saa" in [flag.strip() for flag in sps_flags.split(",")]:
            continue

        observation = int(ref.dataId["observation"])
        instrument_name = str(ref.dataId["instrument"])
        detector = int(ref.dataId["detector"])
        finast = int(header.get("FINAST", -999))
        obs = ObservationId(observation)
        key = _make_slewgroup_key(observation)

        if key not in slewgroups:
            slewgroups[key] = {
                "instrument": instrument_name,
                "source_collection": collections,
                "plan_name": obs.plan_name,
                "large_slew_counter": str(obs.large_slew_counter),
                "observations_in_group": set(),
                "records": [],
                "has_bad_finast": False,
            }

        payload = _detector_payload_with_obs_meta(ref, header, collections)

        slewgroups[key]["observations_in_group"].add(observation)
        slewgroups[key]["records"].append(
            {
                "ref": ref,
                "data_id": dict(ref.dataId.mapping),
                "observation": observation,
                "detector": detector,
                "finast": finast,
                "payload": payload,
            }
        )

        if finast == 0:
            n_good += 1
        else:
            n_bad += 1
            slewgroups[key]["has_bad_finast"] = True

            retry_key = tuple(sorted(dict(ref.dataId.mapping).items()))
            if retry_key not in retry_keys:
                retry_keys.add(retry_key)
                retry_data_ids.append(dict(ref.dataId.mapping))

        if progress_every > 0 and n_total % progress_every == 0:
            logger.info(f"scanned={n_total} good={n_good} bad={n_bad}")

    # Write l2bwcs payloads only for observations with bad finast
    observations_to_retry = {d["observation"] for d in retry_data_ids}

    n_written = 0
    n_groups_written = 0

    for key, group in slewgroups.items():
        if not group["has_bad_finast"]:
            continue

        observations_sorted = sorted(group["observations_in_group"])
        solutions_sorted = sorted(
            [rec["payload"] for rec in group["records"]],
            key=lambda s: (
                int(s["detector"]),
                int(s["small_slew_counter"]),
                int(s["observation"]),
            ),
        )

        for observation in observations_sorted:
            if observation not in observations_to_retry:
                continue
            payload = {
                "instrument": group["instrument"],
                "observation": observation,
                "source_collection": group["source_collection"],
                "seed_strategy_version": "same_detector_adjacent_small_slew_v1",
                "slewgroup": {
                    "plan_name": group["plan_name"],
                    "large_slew_counter": group["large_slew_counter"],
                    "observations_in_group": observations_sorted,
                },
                "slewgroup_solutions": solutions_sorted,
            }

            butler.put(
                payload,
                out_type,
                dataId={
                    "instrument": group["instrument"],
                    "observation": observation,
                },
                run=output_run,
            )
            n_written += 1

        n_groups_written += 1

    # Tag matching level2a refs
    to_associate = []
    missing_l2a = 0

    for data_id in retry_data_ids:
        l2a_refs = list(
            butler.query_datasets(
                "level2a",
                collections=collections,
                data_id=data_id,
                find_first=True,
            )
        )
        if not l2a_refs:
            logger.warning(f"no level2a found for {data_id}")
            missing_l2a += 1
            continue

        to_associate.append(l2a_refs[0])

    if to_associate:
        butler.registry.associate(output_tag, to_associate)

    logger.info(
        f"total_level2={n_total} "
        f"good_finast={n_good} "
        f"bad_finast={n_bad} "
        f"written_groups={n_groups_written} "
        f"written_observation_level_l2bwcs={n_written} "
        f"tagged_level2a={len(to_associate)} "
        f"missing_level2a={missing_l2a} "
        f"tag={output_tag}"
    )
