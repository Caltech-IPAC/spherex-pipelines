__all__ = ["recomputeZodi"]

# Standard library imports
import logging
import re
from typing import Optional
from uuid import UUID

# Related third party imports
from alive_progress import alive_bar
from lsst.daf.butler import (
    Butler,
    CollectionType,
    DatasetRef,
    DatasetType,
    MissingCollectionError,
    MissingDatasetTypeError,
)

# Local application/library specific imports
from spx_data import ObservationId, SPHERExImage

from ..defaults import DatasetTypes
from ..instrument import SPHERExInstrument
from .utils import apply_config_overrides

logger = logging.getLogger(__name__)

SPS_PLAN_PATTERN = re.compile(r"(^\d\d\d\dW\d\d_\d[A-Z]$)")

# Marker of the zodi_scaling calibration provenance line written by
# `~spx_tasks._provenance.add_calibration_provenance`.
ZODI_SCALING_PROVENANCE = "[CALIB] zodi_scaling"


def _drop_last_zodi_scaling_provenance(spximg: SPHERExImage) -> Optional[str]:
    """Remove the last ``[CALIB] zodi_scaling`` HISTORY line from an image.

    The recomputed image gets a fresh zodi_scaling provenance line, so the
    one left by the run that produced the input image is dropped: keeping
    both would make it ambiguous which calibration the zodi layer in the
    output was actually built from.

    Parameters
    ----------
    spximg :
        Image whose header is modified in place.

    Returns
    -------
    removed :
        The HISTORY line that was removed, or `None` if the image carried no
        zodi_scaling provenance.

    """
    header = spximg.header
    for index in reversed(range(len(header.cards))):
        card = header.cards[index]
        if card.keyword == "HISTORY" and ZODI_SCALING_PROVENANCE in str(card.value):
            removed = str(card.value)
            del header[index]
            return removed
    return None


class _CalibrationCache:
    """Load calibration datasets, keeping the most recent one of each type.

    The calibrations are per-detector and can be validity-ranged, so the registry
    lookup has to be repeated for every exposure, but the (expensive)
    deserialization is skipped while consecutive exposures resolve to the
    same calibration dataset.

    Only the last dataset of each type is kept: ``spectral_wcs`` is ~33 MB per
    detector, so caching all six would hold ~200 MB for no gain. This relies
    on the exposures being processed grouped by detector; in any other order
    the calibration is simply deserialized again.
    """

    def __init__(self, butler: Butler, collections: list[str]):
        self._butler = butler
        self._collections = collections
        self._cache: dict[str, tuple[UUID, object]] = {}

    def get(self, dataset_type_name: str, dataid) -> tuple[object, DatasetRef]:
        """Return the calibration object and the reference it was loaded from."""
        deferred = self._butler.getDeferred(dataset_type_name,
                                            dataId=dataid,
                                            collections=self._collections)
        ref = deferred.ref
        cached = self._cache.get(dataset_type_name)
        if cached is None or cached[0] != ref.id:
            self._cache[dataset_type_name] = (ref.id, deferred.get())
        return self._cache[dataset_type_name][1], ref


def _existing_data_ids(butler: Butler, dataset_type: str, output_run: str) -> set:
    """Return the data IDs already present as ``dataset_type`` in ``output_run``.

    Used to resume an interrupted run: those exposures have been written
    already and the Butler would reject writing them a second time.

    Parameters
    ----------
    butler :
        Butler pointing at the repository.
    dataset_type :
        Output dataset type name.
    output_run :
        Run collection the outputs are written to.

    Returns
    -------
    data_ids :
        Data IDs already written; empty if the run holds no such datasets yet.

    """
    try:
        # limit=None means "no limit"; a limit of 0 would return nothing
        return {ref.dataId for ref in butler.query_datasets(
            dataset_type, [output_run], find_first=True, explain=False, limit=None)}
    except (MissingCollectionError, MissingDatasetTypeError):
        return set()


def recomputeZodi(repo, collections, output_run,
                  in_type="level2", out_type="level2",
                  calib_collections=None,
                  instrument=SPHERExInstrument.getName(),
                  detector=None, sps_plan=None,
                  where=None, limit=None,
                  config_file=None, dump_config=False,
                  skip_errors=False, skip_existing=True):
    """Recompute the zodi layer of level 2 images with Get Zodi Model.

    Rubin/LSST ``PipelineTask`` cannot have the same dataset type as both
    input and output, which makes it impossible to run Get Zodi Model
    in-place on ``level2`` images through a pipeline.  This script reads
    ``in_type`` images from ``collections``, replaces their ZODI layer with
    a freshly computed zodi model and writes the images as ``out_type``
    into ``output_run``.

    Only the ZODI layer is replaced; all other layers are copied unchanged
    from the input image.  Products derived from the zodi model by later
    modules (for example the spectral image layers) are *not* recomputed.

    Parameters
    ----------
    repo : `str`
        URI to the repository.
    collections : `str` or `~collections.abc.Iterable` [`str`]
        Input collections searched for the input images.
    output_run : `str`
        The run collection where the output images should be put.
    in_type : `str`, optional
        Input image dataset type.
    out_type : `str`, optional
        Output image dataset type.  May be the same as ``in_type``, in which
        case ``output_run`` must not be one of the input ``collections``.
    calib_collections : `str` or `~collections.abc.Iterable` [`str`], optional
        Collections searched for the ``zodi_scaling`` and ``spectral_wcs``
        calibrations.  Defaults to the input ``collections``.
    instrument : `str`, optional
        Instrument dimension name.
    detector : `int`, optional
        Restrict processing to a single detector.
    sps_plan : `str`, optional
        SPS plan, for example ``2025W18_1B``.
    where : `str`, optional
        Additional SQL WHERE-like expression, ANDed with the other
        constraints.
    limit : `int`, optional
        Maximum number of input datasets to process.
    config_file : `str`, optional
        The path to configuration override file for the task.
    dump_config : `bool`, optional
        If True, dump the task configuration to standard output and exit.
    skip_errors : `bool`, optional
        If True, log and skip exposures that fail instead of aborting.
    skip_existing : `bool`, optional
        If True (default), skip the exposures already present as ``out_type``
        in ``output_run``, which makes a re-run resume where the previous one
        stopped. If False, such exposures are reprocessed, which the Butler
        rejects with a `~lsst.daf.butler.registry.ConflictingDefinitionError`.

    Raises
    ------
    ValueError
        Raised if the requested input and output would collide, or if the
        SPS plan name is invalid.
    ImportError
        Raised if the SPHEREx Simulator, required by Get Zodi Model,
        is not available.

    """
    from spx_tasks import GetZodiModelConfig, GetZodiModelTask, hasSPHERExSimulator, _provenance

    collections = [collections] if isinstance(collections, str) else list(collections)
    if calib_collections is None:
        calib_collections = collections
    elif isinstance(calib_collections, str):
        calib_collections = [calib_collections]
    else:
        calib_collections = list(calib_collections)

    if out_type == in_type and output_run in collections:
        raise ValueError(
            f"Output run {output_run!r} is one of the input collections and "
            f"the output dataset type is the same as the input one ({out_type!r}). "
            "Use a separate output run or a different --out-type."
        )

    # set up the task
    config = GetZodiModelConfig()
    apply_config_overrides(config, config_file, dump_config)

    if not hasSPHERExSimulator:
        raise ImportError("SPHEREx Simulator is not available, "
                          "it is required to compute the zodi model.")

    task: GetZodiModelTask = GetZodiModelTask(config=config)

    dataset_types = DatasetTypes()

    butler = Butler(repo, writeable=True)
    dimension_universe = butler.dimensions

    # dataset type for the output image; registration is idempotent
    out_dataset_type = DatasetType(out_type,
                                   dimension_universe.conform(("instrument", "exposure")),
                                   "SPHERExImage",
                                   universe=dimension_universe)
    butler.registry.registerDatasetType(out_dataset_type)

    # idempotent output run registration
    butler.registry.registerCollection(output_run, type=CollectionType.RUN)

    # construct where clause
    constraints = [f"instrument='{instrument}'"]
    if detector is not None:
        constraints.append(f"detector={detector}")
    if sps_plan is not None:
        # verify that SPS Plan name follows the pattern
        if not SPS_PLAN_PATTERN.search(sps_plan):
            raise ValueError(f"Invalid SPS Plan name: {sps_plan}")
        spsplan_int = ObservationId(sps_plan + "_0001_1").plan_id
        constraints.append(f"spsplan={spsplan_int}")

    if where:
        constraints.append(f"({where})")

    where_clause = " AND ".join(constraints)
    print(f"query-datasets search constraints: --where: {where_clause}")

    query_kwargs = {} if limit is None else {"limit": limit}
    input_refs: list[DatasetRef] = butler.query_datasets(
        in_type,
        collections,
        where=where_clause,
        # Group by detector: the calibrations are per-detector, so this lets
        # _CalibrationCache hold a single detector's calibrations instead of
        # re-reading them as the detector cycles from one exposure to the next.
        order_by=("detector", "exposure"),
        find_first=True,
        explain=False,
        # Fetch the dimension records in this one batched query. Without them
        # every calibration lookup in the loop re-expands the exposure's
        # dimension chain, which costs extra registry queries per exposure.
        with_dimension_records=True,
        **query_kwargs,
    )

    if len(input_refs) == 0:
        logger.error("No %s datasets to process.", in_type)
        return

    n_found = len(input_refs)
    n_skipped = 0
    if skip_existing:
        already_done = _existing_data_ids(butler, out_type, output_run)
        if already_done:
            input_refs = [ref for ref in input_refs if ref.dataId not in already_done]
            n_skipped = n_found - len(input_refs)
            if n_skipped:
                print(f"SKIPPING {n_skipped} exposure(s) already in {output_run}")

    if len(input_refs) == 0:
        print(f"Nothing to do: all {n_found} exposure(s) are already in {output_run}")
        return

    calibrations = _CalibrationCache(butler, calib_collections)

    n_processed = 0
    n_failed = 0

    # The caching context keeps the collection records and summaries of the
    # searched collections in memory, which halves the registry queries the
    # per-exposure calibration lookups would otherwise issue.
    with butler.registry.caching_context(), \
            alive_bar(len(input_refs),
                      title=f"Recomputing zodi for {in_type} images: ") as bar:
        for ref in input_refs:
            bar()
            dataid = ref.dataId
            try:
                spximg = butler.get(ref)

                # Drop the provenance of the calibration used by the run that
                # produced the input, so the output records only the
                # zodi_scaling its own zodi layer was built from.
                stale = _drop_last_zodi_scaling_provenance(spximg)
                if stale is not None:
                    logger.debug("%s: removed stale provenance %r", dataid, stale)

                zodi_scaling, zodi_scaling_ref = calibrations.get(
                    dataset_types.zodi_scaling_v2, dataid)
                spectral_wcs, spectral_wcs_ref = calibrations.get(
                    dataset_types.spectral_wcs, dataid)

                result = task.run(spximg=spximg,
                                  zodi_scaling_v2=zodi_scaling,
                                  spectral_wcs=spectral_wcs)

                # do not include spectral_wcs_ref, it should be the same as before
                calib_items = [(r.datasetType.name, r.id)
                               for r in (zodi_scaling_ref,)]
                _provenance.add_calibration_provenance(result.spximg_out, calib_items)

                butler.put(result.spximg_out, out_dataset_type, dataid, run=output_run)
                n_processed += 1
            except Exception:
                n_failed += 1
                if not skip_errors:
                    raise
                logger.exception("Failed to recompute zodi for %s", dataid)

    print(f"PROCESSED {n_processed} DATASETS")
    if n_skipped:
        print(f"SKIPPED {n_skipped} DATASETS already present in {output_run}")
    if n_failed:
        print(f"FAILED {n_failed} DATASETS")
