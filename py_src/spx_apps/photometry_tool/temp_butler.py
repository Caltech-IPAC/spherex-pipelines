import os
import datetime
import logging
import tempfile
from typing import Optional, Literal
from astropy.table import Table
from astropy.time import Time
from lsst.daf.butler import (
    Butler, CollectionType, DatasetRef, DatasetType, Timespan
)
from lsst.resources import ResourcePath
from spx_butler import ra_dec_to_unit_vector
from spx_butler.defaults import DatasetTypes as dsTypes
from spx_butler.instrument import SPHERExInstrument, SimulatorInstrument
from spx_data import SPSFlag
from spx_utils.tempfile_utils import get_temp_path
from .exceptions import EmptyLVFQueryResultError

BUTLER_CONFIG = "resource://spx_butler/configs"  # URI to configuration directory
BUTLER_YAML = "butler.yaml"  # YAML file for butler configuration
DIMENSION_YAML = "dimensions.yaml"  # YAML file for dimension configuration


def get_skypix_ids(butler: Butler, target_table: Table) -> str:
    """Convert RA/DEC to sky pixel IDs and return them as data query constraint.

    For SPHEREx, this corresponds to HEALPix nested pixel indices at level 6.

    Parameters
    ----------
    butler :
        Butler instance
    target_table :
        Astropy table with "ra" and "dec" columns.

    Returns
    -------
    `str`
        Query constraint on sky pixel (hips6) ids that correspond to the
        sources in the target table.

    """
    commonSkyPix = butler.registry.dimensions.commonSkyPix
    pixelization = commonSkyPix.pixelization

    skypix_ids = set()
    for (ra, dec) in zip(target_table["ra"], target_table["dec"]):
        unit_vector = ra_dec_to_unit_vector(ra, dec)  # Convert to UnitVector3d
        healpix_id = int(pixelization.index(unit_vector))  # Convert to HEALPix6 ID
        skypix_ids.add(healpix_id)

    skypix_ids_str = ", ".join(str(i) for i in list(skypix_ids))
    skypix_ids_constraint = f"{commonSkyPix.name} in ({skypix_ids_str})"
    return skypix_ids_constraint


def lvf_lookup(
        butler: Butler,
        target_table: Table,
        time_interval: Optional[tuple[Time, Time]] = None,
        detectors: Optional[list[int]] = None,
        instrument="spx",
        level2_dataset_type="level2",
        collections="*",
        verbose=False,
        use_deep=False
) -> tuple[Table, list[list[DatasetRef]]]:
    """Get per-source coverage table and dataset references.

    Parameters
    ----------
    butler :
        LSST Butler instance.
    target_table :
        Astropy table with "ra" and "dec" columns.
    time_interval :
        Time interval, with ``None`` used for open end; default is ``None``.
    detectors :
        Detector IDs, default is ``None``.
    instrument :
        Instrument name. Default is "spx".
    level2_dataset_type :
        Level 2 dataset type, default is "level2".
    collections :
        Collections expression. Default is "*".
    verbose :
        If ``True``, log extra info for debugging. Default is False.

    Returns
    -------
    result : tuple[~astropy.table.Table, list[list[~lsst.daf.butler.DatasetRef]]]
        Coverage table and lists of DatasetRef objects, one list per target.

    """
    # construct where clause
    base_where = f"instrument='{instrument}'"
    bind = None

    # Exclude deep field observations
    if not use_deep:
        # Select all-sky observations only
        flgval = SPSFlag.ALL_SKY
        base_where += f" AND observation.sps_flag % {2*flgval} >= {flgval}"

    # add time constraints
    if time_interval is not None:
        stime: Time = time_interval[0]
        etime: Time = time_interval[1]
        if stime is not None or etime is not None:
            base_where += " AND observation.timespan OVERLAPS ts"
            bind = {"ts": Timespan(begin=stime, end=etime)}

    # add detector constraints
    if detectors is not None:
        detectors_str = ", ".join(str(d) for d in detectors)
        base_where += f" AND detector in ({detectors_str})"

    dataset_refs = []  # the values are lists of DatasetRef objects, one list per target

    for idx, (ra, dec) in enumerate(zip(target_table["ra"], target_table["dec"])):
        if verbose:
            logging.info(f"Row {idx}: ra={ra}, dec={dec}")
        where = base_where + f" AND exposure.region OVERLAPS POINT({ra}, {dec})"

        find_first = "*" not in collections
        level2_datasets = butler.query_datasets(
            level2_dataset_type,
            collections.split(","),
            where=where,
            bind=bind,
            find_first=find_first,
            explain=False  # to prevent EmptyQueryResultError
        )
        refs = [r for r in level2_datasets]

        if verbose:
            logging.info("Matching LVF Images:")
            for ref in refs:
                uri = butler.getURI(ref)
                logging.info(ref.id, uri.ospath)

        dataset_refs.append(refs)

    coverage_table = Table(target_table.columns)
    coverage_table["n_img"] = [len(refs) for refs in dataset_refs]

    return coverage_table, dataset_refs


def get_unique_dataset_refs(dataset_refs: list[list[DatasetRef]]) -> list[DatasetRef]:
    """Combine multiple lists of dataset references into a single list of
    dataset references with unique IDs.

    Parameters
    ----------
    dataset_refs :
        Lists of dataset references, one list per category (e.g., target)

    Returns
    -------
        List of unique dataset references.

    """
    unique_dataset_refs_dict = {
        ref.id: ref
        for sublist in dataset_refs
        for ref in sublist
    }
    return list(unique_dataset_refs_dict.values())


def get_unique_calibration_refs(butler: Butler,
                                collections: str,
                                calibration_types: list[str],
                                detectors: Optional[list[int]] = None,
                                instrument: Optional[str] = "spx") -> list[DatasetRef]:
    """Get unique dataset references for calibration products.

    Parameters
    ----------
    butler :
        Instance of Butler, which contains all available LVF products and calibrations.
    collections :
        Calibration collection expr.
    calibration_types :
        List of calibration types to search.
    detectors :
        Detector IDs, default is ``None``.
    instrument :
        Instrument name. Default is "spx".

    Returns
    -------
    result :
        De-duplicated list of the calibration product dataset references.

    """
    calibration_refs: list[DatasetRef] = []

    where = ""
    if detectors is not None:
        where += f"instrument='{instrument}'"
        detectors_str = ", ".join(str(d) for d in detectors)
        where += f" AND detector in ({detectors_str})"

    # for each calibration dataset type, pull *all* matching refs in one go
    for dtype in calibration_types:
        try:
            refs = butler.query_datasets(
                dataset_type=dtype,
                collections=collections.split(","),
                find_first=True,  # stop at first collection that wins
                where=where
            )
            calibration_refs.extend(refs)  # refs are already resolved
        except Exception:
            logging.warning(f"Error while querying {dtype}, skipping")

    return get_unique_dataset_refs([calibration_refs])


def export_datasets(butler: Butler,
                    dataset_refs: list[DatasetRef],
                    export_file: str,
                    export_directory: str,
                    transfer: Literal["symlink", "direct"] = "symlink") -> None:
    """Export unique dataset references for LVFs and relevant calibrations.

    Parameters
    ----------
    butler :
        Butler with all available LVFs and calibrations.
    dataset_refs :
        Dataset references of LVFs and calibrations to be exported.
    export_file :
        Name of export file that will store export information.
    export_directory :
        Path to export directory, which should be the temp butler datastore root.
    transfer :
        Transfer method. Acceptable values are "symlink" and "direct", which
        do not create physical copies of the files. Default is "symlink".

    """
    export_directory_exists_and_empty = os.path.isdir(export_directory) and not os.listdir(export_directory)
    if not export_directory_exists_and_empty:
        raise ValueError(f"Export directory {export_directory} must exist and be empty.")

    with butler.export(
        directory=export_directory,
        filename=export_file,
        transfer=transfer
    ) as export:
        export.saveDatasets(dataset_refs)


def create_temp_butler_registry(butler_root: str,
                                instrument_name: str = "spx") -> Butler:
    """Create temporary Butler, which will only contain the inputs and outputs
    for the current spectrophotometry job.

    Parameters
    ----------
    butler_root :
        Path to the temporary butler root, the same as temporary burler datastore root,
        the same as the export directory, where we exported the LFVs and calibrations
        for the current job.

    instrument_name :
        Instrument name. Default is "spx".

    Returns
    -------
    result :
        Butler instance for this job temporary Butler.

    """
    butler_yaml = os.path.join(butler_root, BUTLER_YAML)
    if not os.path.exists(butler_yaml):
        # create butler repo
        config_uri = ResourcePath(BUTLER_CONFIG, forceDirectory=True)
        butler_config = config_uri.join(BUTLER_YAML)
        dimension_config = config_uri.join(DIMENSION_YAML)

        Butler.makeRepo(root=butler_root,
                        config=butler_config,
                        dimensionConfig=dimension_config)

        # make sure instrument and detector dimensions are populated
        instrument = SPHERExInstrument()
        if instrument.getName() != instrument_name:
            instrument = SimulatorInstrument()
        butler = Butler(butler_root, writeable=True)
        instrument.register(butler.registry)
        return butler
    else:
        raise ValueError(f"Butler registry already exists under {butler_root}")


def get_unique_srccat_dataids(unique_lvf_dataset_refs: list[DatasetRef]) -> list[dict]:
    """Get unique combinations of instrument, spsplan, dichroicpair, and slewgroup
    that define the data IDs of source catalogs associated with a group of LVF images.

    Parameters
    ----------
    unique_lvf_dataset_refs :
        a list of unique dataset references corresponding to the input LVF images

    Returns
    -------
    result :
        A list of unique data ids corresponding the in input source catalogs

    """
    srccat_dataids = []
    keys_to_extract = ["instrument", "dichroicpair", "spsplan", "slewgroup"]
    for ref in unique_lvf_dataset_refs:
        lvf_dataid = ref.dataId
        dataid = {k: lvf_dataid[k] for k in keys_to_extract}
        if dataid not in srccat_dataids:
            srccat_dataids.append(dataid)
    return srccat_dataids


def ingest_srccat(temp_butler: Butler,
                  coverage_table: Table,
                  srccat_dataids: list[dict],
                  srccat_run="tgt"
                  ) -> Table:
    """Ingest source catalogs into the temporary butler."""
    temp_butler.collections.register(
        srccat_run,
        CollectionType.RUN,
        doc="Source Catalogs: Targets to Photometer"
    )

    # register srccat dataset type
    dimension_universe = temp_butler.registry.dimensions
    dimensions = dimension_universe.conform(["instrument", "dichroicpair", "spsplan", "slewgroup"])
    datasetType = DatasetType(name=dsTypes.srccat,
                              dimensions=dimensions,
                              storageClass="AstropyParquet",
                              universe=dimension_universe,
                              isCalibration=False)
    temp_butler.registry.registerDatasetType(datasetType)

    # filter coverage table by n_img column
    src_tbl = coverage_table[coverage_table["n_img"] > 0]
    with temp_butler.transaction():
        for dataid in srccat_dataids:
            temp_butler.put(src_tbl, dsTypes.srccat, dataId=dataid, run=srccat_run)

    return src_tbl


def collect_inputs_for_temp_butler(
        butler_repo: str,
        target_table: Table,
        tmp_dir: str,
        time_interval: Optional[tuple[Time, Time]] = None,
        detectors: Optional[list[int]] = None,
        use_deep: bool= False
) -> None:
    """Find and export all necessary inputs from a source Butler to run
    spectrophotometry.

    Runs in an isolated process and places the path to the temporary Butler,
    export file path, coverage table and srccat ids into the provided queue
    with a status result.

    Parameters
    ----------
    butler_repo : str
        Path to the source Butler repository to query.
    target_table :
        Astropy table with "ra" and "dec" columns.
    tmp_dir :
        Temporary directory where to create temporary butler.
    time_interval :
        Time interval, with ``None`` used for open end; default is ``None``.
    detectors :
        Detector IDs to filter on, default is ``None``.
    use_deep : bool
        Flag on whether to include or exclude deep field images.

    Raises
    ------
    lsst.daf.butler.MissingDatasetTypeError
        Raised if the requested dataset type has not been registered.

    """
    instrument = os.getenv("SPXPHOTO_INSTRUMENT", "spx")
    if instrument not in ["spx", "simu"]:
        raise ValueError("Unsupported instrument: {instrument}")
    lvf_dataset_type_name = os.getenv("SPXPHOTO_L2_DATASET_TYPE", "level2")
    butler = Butler(butler_repo)
    # verify that dataset type is registered
    # lsst.daf.butler.MissingDatasetTypeError is raised if not registered
    _ = butler.get_dataset_type(lvf_dataset_type_name)

    # Find LVF images that contain ra and dec from the target table
    # Coverage table tells how many images are found per source
    # if the value in "n_img" column is 0, no images are found.
    input_coll = os.getenv("SPXPHOTO_INPUT_COLL", "IRSA-qr2-visible")
    coverage_table, dataset_refs = lvf_lookup(butler,
                                              target_table,
                                              time_interval=time_interval,
                                              detectors=detectors,
                                              instrument=instrument,
                                              level2_dataset_type=lvf_dataset_type_name,
                                              collections=input_coll,
                                              use_deep=use_deep
                                              )
    # Input sources with no LVF coverage
    src_tbl_no_matches = coverage_table[coverage_table["n_img"] < 1]
    n_src_no_matches = len(src_tbl_no_matches)
    if n_src_no_matches > 0:
        logging.warning(f"{n_src_no_matches} input sources without matching LVF images.")
        logging.warning("Sources without matching LVF:\n" + repr(src_tbl_no_matches))

    # Unique LVF image references
    unique_dataset_refs = get_unique_dataset_refs(dataset_refs)
    if len(unique_dataset_refs) < 1:
        raise EmptyLVFQueryResultError(n_src_no_matches,
                                       instrument=instrument,
                                       dataset_type=lvf_dataset_type_name)

# Unique calibration product references
    calibration_collections = os.getenv("SPXPHOTO_CALIBRATION_COLL", "IRSA-qr2-tools")

    # Per-detector calibrations consumed alongside the level2 images; safe to
    # restrict to the detectors the user asked for.
    calibration_types = [
        dsTypes.solid_angle_pixel_map,
        dsTypes.spectral_wcs,
        dsTypes.epsf,          # perform_forced_photometry.override_epsf: true
        dsTypes.he_template,   # estimate_diffuse_background.est_he_airglow: true
    ]
    unique_calibration_refs = get_unique_calibration_refs(butler,
                                                          calibration_collections,
                                                          calibration_types,
                                                          detectors=detectors)

    # CreateAllSkyCatalog declares l3_flux_corrections with minimum=6, i.e. it
    # needs the full detector set even when the job is restricted to a subset.
    # Query without the detector constraint.
    unique_calibration_refs += get_unique_calibration_refs(butler,
                                                           calibration_collections,
                                                           [dsTypes.l3_flux_corrections],
                                                           detectors=None)
    unique_calibration_refs = get_unique_dataset_refs([unique_calibration_refs])

    # Create temporary butler directory, which will also act as export directory
    timestamp = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    temp_butler_root = tempfile.mkdtemp(prefix=f"{timestamp}_butler_", dir=tmp_dir)
    export_file = get_temp_path(prefix=f"{timestamp}_export_", suffix=".yaml", dir=tmp_dir)

    # Export LVF Images and related calibrations
    export_datasets(butler,
                    dataset_refs=(unique_dataset_refs + unique_calibration_refs),
                    export_file=export_file,
                    export_directory=str(temp_butler_root))

    # Get unique source catalog data ids for the input LVF images
    srccat_dataids = get_unique_srccat_dataids(unique_dataset_refs)

    results = {
        "temp_butler_root": str(temp_butler_root),
        "export_file": str(export_file),
        "coverage_table": coverage_table,
        "srccat_dataids": srccat_dataids
    }
    return results


def populate_temp_butler_from_export(
        temp_butler_root: str,
        export_file: str,
        coverage_table: Table,
        srccat_dataids: list[dict] = None
) -> None:
    """Create and populate a new temporary Butler from exported data.

    Runs in an isolated process and places the path to the temporary Butler
    into the provided queue with a status result.

    Parameters
    ----------
    temp_butler_root : str
        Path to the directory where the new temporary Butler will be created.
    export_file : str
        Path to the ``export.yaml`` file generated by the collection step.
    coverage_table : astropy.table.Table
        Astropy table with coverage information needed to ingest source catalogs.
    srccat_dataids : list[dict]
        A list of data IDs for the source catalogs to be ingested.
        Defaults to None to be compatible with mosaic tool.

    """
    instrument = os.getenv("SPXPHOTO_INSTRUMENT", "spx")

    # Create temporary Butler
    temp_butler = create_temp_butler_registry(temp_butler_root, instrument_name=instrument)

    # Import LVF images and related calibrations
    temp_butler.import_(filename=export_file, transfer=None)

    # Ingest source catalogs into temporary butler
    # We are using the same source catalog for all data ids:
    # the Perform Forced Photometry module will filter out
    # the sources outside the image.
    # skip this step for mosaic tool (where srccat_dataids is None.)
    if srccat_dataids:
        ingest_srccat(temp_butler, coverage_table, srccat_dataids)

    temp_butler_root = None

    for datastore_root in temp_butler.get_datastore_roots().values():
        temp_butler_root = datastore_root
        logging.info(f"Temporary Butler Datastore Root URI: {datastore_root}")

    results = {
        "temp_butler_root": str(temp_butler_root.ospath)
    }

    return results
