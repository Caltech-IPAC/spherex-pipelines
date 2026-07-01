import re
import logging
from collections.abc import Iterator

import astropy.units as u
from astropy.time import TimeDelta, Time
from astropy.io import fits
from astropy.wcs import WCS

from lsst.daf.butler import (
    Butler,
    CollectionType,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    FileDataset,
    Timespan
)
from lsst.resources import ResourcePath
from spx_data import ObservationId, SPSFlag

from ..instrument import SimulatorInstrument, SPHERExInstrument
from ..formatters import SPHERExImageFormatter


def nbitToInt(n: int, nbit: int, bitmask: int):
    """Convert binary to int by shifting bits"""
    return (n >> nbit) & bitmask


def getDimensionIds(file: str):
    """
    Gets the dimensions needed for butler from file name.

    Parameters
    ----------
    file : `str`
        The name of the file.

    Returns
    -------
    result: tuple(ObservationId, int)
        Tuple with observation id object, and detector id.
    """
    # Pattern matches either the short or long SPS observation ID
    # Short: *_2024W25_1A_0094_3_detector_2_*.fits
    # <year>W<week>_<planning-period>_<large-slew>_<step>...
    # Long: *_2024W25_1A_0094_3_4_detector_2_*.fits
    # <year>W<week>_<planning-period>_<large-slew>_<step>_<num_steps>...
    # there are also two variants for detector number: detector_3 or D3
    pattern = re.compile(r"(\d{4}W\d{2}_\d\w_+\d{4}_\d(?:_\d|))(?:_detector_|D)(\d)_.+\.(?:fits|gz)")

    # Parse human-readable Observation ID and detector ID from file name
    m = pattern.search(file)
    if m is None:
        raise ValueError(f"{file} does not match human-readable Observation ID pattern")
    g = m.groups()
    if len(g) != 2:
        raise ValueError("Unable to get human-readable Observation ID and "
                         f"detector from file name: {file}")

    detector_id = int(g[1])

    return ObservationId(g[0]), detector_id


def getObservationExposureRecords(file: str,
                                  instrument: str,
                                  obsid: ObservationId,
                                  detector_id: int):
    # This code knows the format of the file and can extract information from it

    # SPS observation id from file name - short form
    sps_id = obsid.sps_id

    with fits.open(file) as hdul:
        hdr = hdul[1].header
        if "SFLAG" in hdr:
            # one-year simulated images
            sps_flag = SPSFlag.from_string(hdr["SFLAG"])
        elif "SPS_FLAGS" in hdr:
            # SSDC level1 and level2 images
            sps_flag = SPSFlag.from_string(hdr["SPS_FLAGS"])
        else:
            # assume all_sky
            sps_flag = SPSFlag.ALL_SKY
        if "DGL_QNTL" in hdr:
            # sky coordinates must be available
            dgl_qntl = hdr["DGL_QNTL"]
            # Add galactic lat/lon
            wcs = WCS(hdr)
            coord = wcs.pixel_to_world(wcs.wcs.crpix[0], wcs.wcs.crpix[1])
            galactic_coord = coord.transform_to("galactic")
            galactic_lat = galactic_coord.b.value
            galactic_lon = galactic_coord.l.value
        else:
            # no sky coordinates
            dgl_qntl = None
            galactic_lat = None
            galactic_lon = None
        if "MJD-END" in hdr and "TELAPSE" in hdr:
            end = Time(hdr["MJD-END"], format="mjd", scale="utc")
            begin = end - TimeDelta(hdr["TELAPSE"] * u.s)
        else:
            begin = None
            end = None

        if "DETECTOR" in hdr:
            det_id = hdr["DETECTOR"]
        else:
            det_id = hdr["ARRAY"]
        if "OBSID" in hdr:
            obs_id = hdr["OBSID"]  # observation id from Survey Planning Software
            # Survey Planning Software is using longer form of observation id
            # with total number of small slews in a group at the end
            if not obs_id.startswith(sps_id):
                raise ValueError(f"Inconsistent observation id: {obs_id} in header, {sps_id} in file name")

        if det_id != detector_id:
            raise ValueError(f"Inconsistent detector id: {det_id} in header, {detector_id} in file name")

    observation_id = obsid.pointing_id
    plan_id = obsid.plan_id
    plan_name = obsid.plan_name
    slewgroup_id = obsid.slewgroup_id  # large slew counter
    slewgroup_counter_id = obsid.slewgroup_counter_id

    plan_record = {
        "instrument": instrument,
        "id": plan_id,
        "name": plan_name
    }
    slewgroup_record = {
        "instrument": instrument,
        "id": slewgroup_id
    }
    obs_record = {
        "instrument": instrument,
        "id": observation_id,
        "spsplan": plan_id,
        "canceled": False,
        "slewgroup": slewgroup_id,
        "slewgroup_counter": slewgroup_counter_id,
        "sps_name": sps_id,
        "sps_flag": sps_flag,
        "in_sband": False,
        "timespan": Timespan(begin=begin, end=end)
    }
    exp_record = {
        "instrument": instrument,
        "id": obsid.to_exposure_id(det_id),
        "observation": observation_id,
        "detector": det_id,
        "dgl_qntl": dgl_qntl,
        "galactic_lat": galactic_lat,
        "galactic_lon": galactic_lon
    }
    return plan_record, slewgroup_record, obs_record, exp_record


def is_within_range(obsid, obsid_min=None, obsid_max=None):
    """Function to safely convert to integer and compare for OBSID bounds.

    Parameters:
        obsid : `str`
            Observation ID
        obsid_min : `str`, default is None.
            Minimum observation ID.
        obsid_max : `str`
            Maximum observation ID.

    Returns: `bool`
        True if observation within boundary.
    """
    try:
        obsid_num = int(obsid)

        # Convert obsid_min and obsid_max only if they are not None
        if obsid_min is not None:
            obsid_min_num = int(obsid_min)
        if obsid_max is not None:
            obsid_max_num = int(obsid_max)

        # Perform comparison based on which bounds are provided
        if obsid_min is not None and obsid_max is not None:
            return obsid_min_num <= obsid_num <= obsid_max_num
        elif obsid_min is not None:
            return obsid_min_num <= obsid_num
        elif obsid_max is not None:
            return obsid_num <= obsid_max_num
        else:
            return True  # If both bounds are None, all obsid values are considered

    except ValueError:
        # Handle the error if conversion fails
        print("Invalid input: not a numeric observation ID")
        return False


def ingestSimulated(repo,
                    locations,
                    regex,
                    output_run,
                    transfer="auto",
                    ingest_type="raw",
                    obsid_min=None,
                    obsid_max=None,
                    instrument=None):
    """Ingests raw frames into the butler registry

    Parameters
    ----------
    repo : `str`
        URI to the repository.
    locations : `list` [`str`]
        Files to ingest and directories to search for files that match
        ``regex`` to ingest.
    regex : `str`
        Regex string used to find files in directories listed in locations.
    output_run : `str`
        The path to the location, the run, where datasets should be put.
    transfer : `str` or None
        The external data transfer type, by default "auto".
    ingest_type : `str`
        ingest product data type.
    obsid_min : `str`, Optional. Default is `None`
        Minimum observation id.
    obsid_max : `str`, Optional. Default is `None`
        Maximum observation id.
    instrument : `str`
        Instrument dimension value.

    Raises
    ------
    Exception
        Raised if operations on configuration object fail.

    Notes
    -----
    This method inserts all datasets for an observation within a transaction,
    guaranteeing that partial observations are never ingested.  The observation
    dimension record is inserted with `Registry.syncDimensionData` first
    (in its own transaction), which inserts only if a record with the same
    primary key does not already exist.  This allows different files within
    the same observation to be ingested in different runs.
    """

    butler = Butler(repo, writeable=True)

    # use Simulator instrument by default
    sim_instr = SimulatorInstrument()
    sim_instr_name = sim_instr.getName()
    if instrument is None or instrument == sim_instr_name:
        instr = sim_instr
        instrument = sim_instr_name
    else:
        instr = SPHERExInstrument()
        if instrument != instr.getName():
            raise ValueError(f"Invalid instrument {instrument}")
    # make sure instrument and detector dimensions are populated
    instr.register(butler.registry, update=True)

    dimension_universe = butler.registry.dimensions
    datasetType = DatasetType(ingest_type,
                              dimension_universe.conform(("instrument", "exposure")),
                              "SPHERExImage",
                              universe=dimension_universe)
    # idempotent dataset type registration
    butler.registry.registerDatasetType(datasetType)

    # idempotent collection registration
    run = "all" if (output_run is None) else output_run
    butler.registry.registerCollection(run, type=CollectionType.RUN)

    n_failed = 0
    fileURIs: Iterator[ResourcePath] = ResourcePath.findFileResources(locations, regex)

    # do we want to group observations?
    # grp = datetime.date.today().strftime("%Y%m%d")

    datasets = []
    for fileURI in fileURIs:
        file: str = fileURI.path

        try:
            obsid, detector_id = getDimensionIds(file)
            plan_record, slewgroup_record, obs_record, exp_record = \
                getObservationExposureRecords(file, instrument,
                                              obsid, detector_id)
            # idempotent insertion of individual dimension rows
            # dimension_universe["exposure"].RecordClass(exp_record)

            if obsid_min is not None or obsid_max is not None:
                if not is_within_range(obsid.spx_id, obsid_min, obsid_max):
                    continue

            butler.registry.syncDimensionData("spsplan", plan_record)
            butler.registry.syncDimensionData("slewgroup", slewgroup_record)
            butler.registry.syncDimensionData("observation", obs_record, update=True)
            butler.registry.syncDimensionData("exposure", exp_record)
        except Exception as e:
            n_failed += 1
            logging.error(f"Unable to insert dimension records for file {file}: {e}")
            continue

        dataId = DataCoordinate.standardize(
            instrument=instrument,
            exposure=obsid.to_exposure_id(detector_id),
            universe=dimension_universe)
        ref = DatasetRef(datasetType, dataId=dataId, run=run)
        datasets.append(FileDataset(refs=ref, path=file, formatter=SPHERExImageFormatter))

    with butler.transaction():
        butler.ingest(*datasets, transfer=transfer)
    print(f"INGESTED {len(datasets)} DATASETS")
