import logging
import re
from dataclasses import dataclass
from alive_progress import alive_bar
from astropy.table import Table
from astropy.time import Time
from lsst.daf.butler import (
    Butler,
    CollectionType,
    DatasetType,
    DatasetRef,
    EmptyQueryResultError,
    Timespan
)
from typing import Optional
from lsst.daf.butler.registry import ConflictingDefinitionError
from spx_data import (SPHERExImage, ObservationId,
                      GainFactors, NonLinearityParameters,
                      ReadNoiseParameters, PersistenceState,)

from ..defaults import DatasetTypes, Collections
from ..instrument import SimulatorInstrument
from .utils import apply_config_overrides

logger = logging.getLogger(__name__)

SPS_PLAN_PATTERN = re.compile(r"(^\d\d\d\dW\d\d_\d[A-Z]$)")


@dataclass
class PipeState:
    """Pipeline state holding calibration data and image for L2a processing."""

    # will be updated by pipeline
    spximg: SPHERExImage
    gain_factors: GainFactors
    nonlinear_pars: NonLinearityParameters
    readnoise_pars: ReadNoiseParameters
    persistence_state_in: PersistenceState
    sgt: Optional[Table] = None
    spk: Optional[Table] = None


def get_initial_state(butler: Butler,
                      dataid: dict,
                      persistence_prior,
                      dataset_types: DatasetTypes,
                      collections, config) -> tuple[PipeState, list[DatasetRef]]:
    """Retrieve initial pipeline state and calibration references from butler."""
    img = butler.get(dataset_types.level1, dataId=dataid, collections=collections)
    img.ensure_layers([])  # make sure no layers are in the image

    # Walkaround for missing OBSID in the simulated images
    if "OBSID" not in img.header:
        obsid = dataid["exposure"]
        img.header["OBSID"] = ObservationId(obsid).sps_id
    if config.gds_image_association.useSpk:
        spk = butler.get(dataset_types.spk, dataId=dataid, collections=collections)
    else:
        spk = None
    if config.gds_image_association.useSgt:
        sgt = butler.get(dataset_types.sgt, dataId=dataid, collections=collections)
    else:
        sgt = None
    calibration_refs: list[DatasetRef] = []

    # Gain factors - needed for Convert To Engineering Units
    deferred = butler.getDeferred(dataset_types.gain_factors,
                                  dataid,
                                  collections=collections)
    calibration_refs.append(deferred.ref)
    gain_factors = deferred.get()

    # Nonlinear and transient parameters, and nonlinear thresholds.
    deferred = butler.getDeferred(dataset_types.nonlinear_pars,
                                  dataId=dataid,
                                  collections=collections)
    calibration_refs.append(deferred.ref)
    nonlinear_pars = deferred.get()

    # Read noise maps, full and single - needed for Create Variance Map.
    deferred = butler.getDeferred(dataset_types.readnoise_pars,
                                  dataid,
                                  collections=collections)
    calibration_refs.append(deferred.ref)
    readnoise_pars = deferred.get()

    return PipeState(spximg=img,
                     gain_factors=gain_factors,
                     nonlinear_pars=nonlinear_pars,
                     readnoise_pars=readnoise_pars,
                     persistence_state_in=persistence_prior,
                     spk=spk,
                     sgt=sgt), calibration_refs


def runL2a(repo, collections, output_run,
           in_type="level1", out_type="level2a",
           sps_plan=None, detector=0, instrument=SimulatorInstrument.getName(),
           iso_start=None, iso_end=None,
           config_file=None, dump_config=False,
           delayed_model_save=True, prior_required=True,
           persistence_state_collection=None):
    """Run L2a pipeline for exposures in given collections.

    The images are processed in the order of the increasing observation id.

    The persistence state prior should be in the ordered ``collections``,
    in which case the processing will start with the first level1
    image that has observation id greater than the observation id
    of the last persistence state prior.

    If persistence state prior is not found, the first processed image
    will have no persistence state prior.

    Parameters
    ----------
    repo : `str`
        URI to the repository.
    collections : `str`
        Input collections.
    output_run : `str`
        The path to the location, the run, where datasets should be put.
    in_type : `str`
        Input images data type.
    out_type : `str`
        Output image (est. flat) data type.
    sps_plan : `str`
        SPS plan
    detector : `int`
        Detector id
    instrument : `str`
        Instrument dimension name.
    iso_start : `str`
        Selection time interval start in ISO-8601 date (TAI) format.
    iso_end : `str`
        Selection time interval end in ISO-8601 date (TAI) format.
    config_file : `str`
        The path to configuration override file for the task.
    dump_config : `bool`
        If True, dump the task configuration to standard output and exit.
    delayed_model_save : `bool`
        If True, save only the last persistence model available.
    prior_required : `bool`
        If True and prior is not found, raise an exception.
    persistence_state_collection : `str`
        The name of the persistence state collection, default `None`.

    Raises
    ------
    Exception
        Raised if operations on configuration object fail.

    Notes
    -----
    This method runs Estimate Flat Field task

    """
    instrument_name = instrument
    persistence_prior = None

    from spx_tasks import L2aConfig, L2aTask, _provenance

    # set up the task
    config = L2aConfig()
    apply_config_overrides(config, config_file, dump_config)
    task: L2aTask = L2aTask(config=config)

    # dataset types used by L2a pipeline
    dataset_types = DatasetTypes()
    dataset_types.level1 = in_type

    butler = Butler(repo, writeable=True)
    registry = butler.registry
    dimension_universe = registry.dimensions

    # dataset type for output image
    datasetType = DatasetType(out_type,
                              dimension_universe.conform(("instrument", "exposure")),
                              "SPHERExImage",
                              universe=dimension_universe)
    # idempotent output dataset type registration
    butler.registry.registerDatasetType(datasetType)

    datasetTypeDiag = DatasetType(
        "bsh_diag_img",
        dimension_universe.conform(("instrument", "exposure")),
        "SPHERExImage",
        universe=dimension_universe,
    )
    butler.registry.registerDatasetType(datasetTypeDiag)

    # dataset type for persistence state
    datasetTypePersistenceState = \
        DatasetType(dataset_types.persistence_state,
                    dimension_universe.conform(("instrument", "exposure")),
                    "PersistenceState",
                    universe=dimension_universe)
    # mutable persistence state dataset type registration
    butler.registry.registerDatasetType(datasetTypePersistenceState)

    # dataset type for persistence image
    datasetTypePersistenceImage = \
        DatasetType(dataset_types.persistence_image,
                    dimension_universe.conform(("instrument", "exposure")),
                    "PersistenceImage",
                    universe=dimension_universe)
    # idempotent persistence signal image dataset type registration
    butler.registry.registerDatasetType(datasetTypePersistenceImage)

    # construct where clause
    where = f"instrument='{instrument_name}' AND " \
            f"detector={detector}"

    # find the latest observation with persistence state
    try:
        persist_datasets = butler.query_datasets(
            dataset_types.persistence_state,
            collections,
            where=where,
            order_by="-observation",
            explain=prior_required
        )
        persist_data_ids = [dataset_ref.dataId for dataset_ref in persist_datasets]

        if len(persist_data_ids) > 0:
            prior_obs_id = persist_data_ids[0]["observation"]
            print(f"Prior observation id: {prior_obs_id}")
        else:
            prior_obs_id = 0
            print("No prior found.")
    except EmptyQueryResultError as e:
        logger.error("Persistence State prior is not found", e)
        raise e

    # find observation ids to be processed
    where_l1 = where + f" AND observation > {prior_obs_id}"

    # more observation constraints
    if sps_plan is not None:
        # verify that SPS Plan name follows the pattern
        match = SPS_PLAN_PATTERN.search(sps_plan)
        if not match:
            raise ValueError(f"Invalid SPS Plan name: {sps_plan}")
        spsplan_int = ObservationId(sps_plan + "_0001_1").plan_id
        where_l1 = where_l1 + f" AND spsplan={spsplan_int}"

    bind = None

    # add time constraints
    if iso_start is not None or iso_end is not None:
        stime = None
        if iso_start is not None:
            stime = Time(iso_start, scale="utc", format="iso")
        etime = None
        if iso_end is not None:
            etime = Time(iso_end, scale="utc", format="iso")
        where_l1 += " AND observation.timespan OVERLAPS ts"
        bind = {"ts": Timespan(begin=stime, end=etime)}

    print(f"query-dataset search constraints (l1): --where: {where_l1}, --bind: {bind}")

    l1_datasets = butler.query_datasets(
        dataset_types.level1,
        collections,
        where=where_l1,
        bind=bind,
        order_by="observation",
        explain=False
    )
    l1_data_ids = [dataset_ref.dataId for dataset_ref in l1_datasets]

    # find prior persistence state
    obs_ids = [record["observation"] for record in l1_data_ids]

    if len(obs_ids) == 0:
        logger.error("No observations to process.")
        return

    # get prior persistence
    if prior_obs_id > 0:
        persistence_prior_dataid = {
            "instrument": instrument_name,
            "exposure": prior_obs_id + detector,
        }
        persistence_prior = butler.get(dataset_types.persistence_state,
                                       persistence_prior_dataid,
                                       collections=collections)

    # idempotent collection registration

    # output run
    run = "all" if (output_run is None) else output_run
    butler.registry.registerCollection(run, type=CollectionType.RUN)

    # persistence state tagged collection
    if persistence_state_collection:
        c_pstate = Collections.persistence_state(persistence_state_collection)
        registry.registerCollection(c_pstate.name,
                                    c_pstate.collection_type,
                                    c_pstate.description)

    last_persistence_state = None
    last_persistence_state_dataid = None

    def save_persistence_state(persistence_state: PersistenceState, dataid_pstate: dict):
        ref = butler.put(persistence_state,
                         dataset_types.persistence_state,
                         dataid_pstate,
                         run=run)
        if bool(persistence_state_collection) and (ref is not None):
            try:
                butler.registry.associate(c_pstate.name, [ref])
            except ConflictingDefinitionError as cde:
                # this error is not fatal:
                # persistent state is saved in the run collection
                logger.error("Failed to associate %s with %s.\nDetails: %s",
                             ref, c_pstate.name, cde, exc_info=True)
    try:
        with alive_bar(
                len(obs_ids),
                title=f"Running l2a on detector-{detector} images: ") as bar:
            for obs_id in obs_ids:
                bar()
                dataid = {
                    "instrument": instrument_name,
                    "exposure": obs_id + detector,
                    "observation": obs_id,
                }

                state, calibration_refs = get_initial_state(butler, dataid, persistence_prior,
                                                            dataset_types, collections, config)

                result = task.run(**state.__dict__)

                if result.spximg_out is not None:
                    calib_items = [(ref.datasetType.name, ref.id) for ref in calibration_refs]
                    _provenance.add_calibration_provenance(result.spximg_out, calib_items)

                persistence_prior = result.persistence_state_out

                with butler.transaction():
                    if result.spximg_out is not None:
                        butler.put(result.spximg_out, datasetType, dataid, run=run)

                    if result.bsh_diag_img is not None:
                        butler.put(result.bsh_diag_img, datasetTypeDiag, dataid, run=run)

                    if result.persistence_image is not None:
                        butler.put(result.persistence_image, dataset_types.persistence_image, dataid, run=run)
                    if result.persistence_state_out is not None:
                        if delayed_model_save and obs_id != obs_ids[-1]:
                            # save only the last persistence state available
                            last_persistence_state = result.persistence_state_out
                            last_persistence_state_dataid = dataid
                        else:
                            # saving persistence state for every exposure and
                            # last exposure in delayed_model_save mode
                            save_persistence_state(
                                result.persistence_state_out,
                                dataid
                            )
                            last_persistence_state = None
                            last_persistence_state_dataid = None
        print(f"PROCESSED {len(obs_ids)} DATASETS")
    finally:
        if last_persistence_state is not None and last_persistence_state_dataid is not None:
            # this should be only executed if the run did not finish successfully
            with butler.transaction():
                save_persistence_state(
                    last_persistence_state,
                    last_persistence_state_dataid
                )
