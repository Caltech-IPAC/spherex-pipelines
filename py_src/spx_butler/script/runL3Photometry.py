from alive_progress import alive_bar
from astropy.time import Time
from lsst.daf.butler import (
    Butler,
    CollectionType,
    DatasetType,
    Timespan
)
from spx_data import ObservationId
import spx_tasks

from ..defaults import DatasetTypes
from ..instrument import SimulatorInstrument, to_dichroicpair
from .utils import apply_config_overrides


def runL3Photometry(repo, collections, output_run,
                    in_type="level2", out_type="phot_meas",
                    detector=2, iso_start=None, iso_end=None,
                    config_file=None, dump_config=False):
    """Run L3 photometry pipeline on exposures
    overlapping with the given time interval.

    Parameters
    ----------
    repo : `str`
        URI to the repository.
    collections : `str`
        The collections to search for input datasets, including prior
        persistence state.
    output_run : `str`
        The path to the location, the run, where datasets should be put.
    in_type : `str`
        Input images data type.
    out_type : `str`
        Output image (est. flat) data type.
    detector : `int`
        Detector id
    iso_start : `str`
        Selection time interval start in ISO-8601 date (TAI) format.
    iso_end : `str`
        Selection time interval end in ISO-8601 date (TAI) format.
    config_file : `str`
        The path to configuration override file for the task.
    dump_config : `bool`
        If True, dump the task configuration to standard output and exit.

    Raises
    ------
    Exception
        Raised if operations on configuration object fail.

    Notes
    -----
    This method runs Estimate Flat Field task
    """

    # set up the task
    config = spx_tasks.PerformForcedPhotometryConfig()
    apply_config_overrides(config, config_file, dump_config)
    config.combinedOutput = False
    task: spx_tasks.PerformForcedPhotometryTask = spx_tasks.PerformForcedPhotometryTask(config=config)

    butler = Butler(repo, writeable=True)
    dimension_universe = butler.registry.dimensions

    stime = None
    if iso_start is not None:
        stime = Time(iso_start, scale="utc", format="iso")
    etime = None
    if iso_end is not None:
        etime = Time(iso_end, scale="utc", format="iso")
    where = f"instrument='{SimulatorInstrument.getName()}' AND " \
            f"detector={detector} AND " \
            "observation.timespan OVERLAPS ts"
    bind = {"ts": Timespan(begin=stime, end=etime)}
    recs = butler.query_dimension_records("observation", where=where, bind=bind)
    obs_ids = {record.id for record in recs}

    # dataset types used by L2 pipelines
    dataset_types = DatasetTypes()
    dataset_types.level2 = in_type

    commonSkyPix = butler.registry.dimensions.commonSkyPix
    dimensions = ("instrument", "exposure", commonSkyPix.name)
    datasetType = DatasetType(out_type,
                              dimension_universe.conform(dimensions),
                              "AstropyParquet",
                              universe=dimension_universe)
    # idempotent output dataset type registration
    butler.registry.registerDatasetType(datasetType)

    # idempotent collection registration
    run = "all" if (output_run is None) else output_run
    butler.registry.registerCollection(run, type=CollectionType.RUN)

    n_missing = 0
    n_catmissing = 0
    with butler.transaction():
        with alive_bar(
                len(obs_ids),
                title=f"Running photometry on detector {detector} images: ") as bar:
            for obs_id in obs_ids:
                bar()
                dataid = {"instrument": SimulatorInstrument.getName(),
                          "detector": detector, "observation": obs_id}

                try:
                    img = butler.get(dataset_types.level2, dataId=dataid, collections=collections)
                except LookupError:
                    n_missing += 1
                    continue

                dataid_cat = {"instrument": SimulatorInstrument.getName(),
                              "slewgroup": ObservationId(obs_id).slewgroup_id,
                              "dichroicpair": to_dichroicpair(detector),
                              "spsplan": ObservationId(obs_id).plan_id}
                try:
                    srccat = butler.get(dataset_types.srccat, dataId=dataid_cat, collections=collections)
                except LookupError:
                    n_catmissing += 1
                    continue

                result = task.run(img, srccat)

                if result.phot_meas is not None:
                    for hips_idx, table in result.phot_meas.items():
                        out_dataid = dataid.copy()
                        out_dataid[f"{commonSkyPix.name}"] = hips_idx
                        butler.put(table, datasetType, out_dataid, run=run)

    print(f"PROCESSED {len(obs_ids)-n_missing} LVFs")
    if n_missing > 0:
        print(f"- missing {in_type} images for {n_missing} observations")
    if n_catmissing > 0:
        print(f"- missing catalogs for {n_catmissing} observations")
