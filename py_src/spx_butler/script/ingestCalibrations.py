from collections.abc import Iterator
from lsst.daf.butler import (
    Butler,
    CollectionType,
    DataCoordinate,
    DatasetType,
    DatasetRef,
    FileDataset
)
from lsst.resources import ResourcePath

from ..defaults import DatasetTypes, fake_products
from ..instrument import ensure_registered_instrument, SimulatorInstrument, SPHERExInstrument


def ingestCalibrations(repo,
                       locations,
                       regex,
                       output_run,
                       transfer="auto",
                       ingest_type="fake",
                       detector=0,
                       instrument=SimulatorInstrument.getName()):
    """Ingests calibrations into the butler registry.

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
        Ingest product dataset type.
    detector : `int`
        Detector id.
    instrument : `str`
        Instrument dimension value.

    Raises
    ------
    Exception
        Raised if operations on configuration object fail.
    """
    butler = Butler(repo, writeable=True)

    # make sure instrument and detector dimensions are populated
    instr = SimulatorInstrument() if instrument == SimulatorInstrument.getName() else SPHERExInstrument()
    ensure_registered_instrument(instr, butler.registry)

    dimension_universe = butler.registry.dimensions

    # register calibration dataset types
    dt = DatasetTypes()
    datasetType = None
    # Create default dataid
    dataid_kwargs = {"instrument": instrument, "universe": dimension_universe}
    dimensions = ["instrument"]

    if detector == 0:
        dataset_and_storage_class = (
            (dt.gain_factors, "GainFactors"),
            (dt.initial_distortion_table, "InitDistTable"),
            (dt.sband_mnemonics, "SbandMnemonics")
        )

    else:
        dataset_and_storage_class = (
            (dt.nonlinear_pars, "NonLinearityParameters"),
            (dt.readnoise_pars, "ReadNoiseParameters"),
            (dt.nonfunc_map, "SPHERExImage"),
            # (dt.nonfunc_map_v2, "NonfuncData"),
            (dt.dichroic_cut_map, "SPHERExImage"),
            (dt.ff_calib, "SPHERExImage"),
            (dt.dc_calib, "SPHERExImage"),
            (dt.abs_gain_matrix, "SPHERExImage"),
            (dt.average_psf, "PSF"),
            (dt.sip, "AstropySIP"),
            (dt.spectral_wcs, "AstropyImage"),
            (dt.photo_calib, "SpxPhotoCalib"),
            (dt.solid_angle_pixel_map, "SPHERExImage"),
            (dt.xtalk_coupling_matrix, "SPHERExImage")
        )
        # Add additional dimensions
        dataid_kwargs["detector"] = detector
        dimensions.append("detector")

    dimensions = dimension_universe.conform(dimensions)
    dataid = DataCoordinate.standardize(**dataid_kwargs)

    for name, sclass in dataset_and_storage_class:
        if ingest_type in ("fake", name):
            datasetType = DatasetType(name=name,
                                      dimensions=dimensions,
                                      storageClass=sclass,
                                      universe=dimension_universe,
                                      isCalibration=True)
            # idempotent dataset type registration
            butler.registry.registerDatasetType(datasetType)

    # we need to know the storage class of the datasetType to ingest files
    if datasetType is None:
        raise ValueError(f"Unknown dataset type {ingest_type}")

    # idempotent collection registration
    run = "all" if (output_run is None) else output_run
    butler.registry.registerCollection(run, type=CollectionType.RUN)

    if ingest_type == "fake" and detector == 0:
        with butler.transaction():
            # Gain factors - needed for Convert To Engineering Units
            gain_factors = fake_products.fake_gain_factors()
            butler.put(gain_factors, dt.gain_factors, dataId=dataid, run=run)

            # Initial distortion table - needed for initial SIP derivation
            initial_distortion_table = fake_products.initial_distortion()
            butler.put(initial_distortion_table, dt.initial_distortion_table, dataId=dataid, run=run)

            # Test Sband telemetry list
            sband_mnemonics = fake_products.sband_subset_table()
            butler.put(sband_mnemonics, dt.sband_mnemonics, dataId=dataid, run=run)

            print("Ingested fake calibrations")

    elif ingest_type == "fake":
        with butler.transaction():
            # Nonlinear and transient parameters, and nonlinear thresholds.
            nonlinear_pars = fake_products.fake_nonlin_inputs(detector)
            butler.put(nonlinear_pars, dt.nonlinear_pars, dataId=dataid, run=run)

            # Read noise parameters.
            readnoise_pars = fake_products.fake_read_noise_map(detector)
            butler.put(readnoise_pars, dt.readnoise_pars, dataId=dataid, run=run)

            # Binned calibration data.
            photo_calib = fake_products.fake_binned_calib_data(detector)
            butler.put(photo_calib, dt.photo_calib, dataId=dataid, run=run)

            # non-functional pixel map - set as a pixel flag
            nonfunc_map = fake_products.fake_nonfunc_map(detector)
            # nonfunc_map_v2 = fake_products.fake_nonfunc_v2_map(detector)
            butler.put(nonfunc_map, dt.nonfunc_map, dataId=dataid, run=run)
            # butler.put(nonfunc_map_v2, dt.nonfunc_map_v2, dataId=dataid, run=run)

            # dichroic cut map - set as a pixel flag
            dichroic_cut_map = fake_products.fake_dichroic_cut_map(detector)
            butler.put(dichroic_cut_map, dt.dichroic_cut_map, dataId=dataid, run=run)

            # spectral wcs
            spectral_wcs = fake_products.fake_spectral_wcs(detector)
            butler.put(spectral_wcs, dt.spectral_wcs, dataId=dataid, run=run)

            # dark current calibration - needed for Dark Current Field Correction
            dc_calib = fake_products.fake_dc_calib(detector)
            butler.put(dc_calib, dt.dc_calib, dataId=dataid, run=run)

            # flat field calibration - from instrument or faked
            ff_calib = fake_products.fake_ff_calib(detector)
            butler.put(ff_calib, dt.ff_calib, dataId=dataid, run=run)

            # absolute gain matrix
            # needed for Apply Flat Field Correction and Photometric Calibration
            abs_gain_matrix = fake_products.fake_abs_gain_matrix(detector)
            butler.put(abs_gain_matrix, dt.abs_gain_matrix, dataId=dataid, run=run)

            # distortion / SIP coefficients - needed for Estimate Fine Astrometry
            sip = fake_products.fake_sip(detector)
            butler.put(sip, dt.sip, dataId=dataid, run=run)

            # exposure averaged PSF - needed for Measure Per-Exposure PSF
            average_psf = fake_products.fake_PSF(detector)
            butler.put(average_psf, dt.average_psf, dataId=dataid, run=run)

            # solid angle pixel map
            solid_angle_pixel_map = fake_products.fake_solid_angle_pixel_map(detector)
            butler.put(solid_angle_pixel_map, dt.solid_angle_pixel_map, dataId=dataid, run=run)

            # Crosstalk coupling matrix
            xtalk_coupling_matrix = fake_products.fake_xtalk_coupling(detector)
            butler.put(xtalk_coupling_matrix, dt.xtalk_coupling_matrix, dataId=dataid, run=run)

            print(f"Ingested fake calibrations for detector {detector}")
    else:
        fileURIs: Iterator[ResourcePath] = ResourcePath.findFileResources(locations, regex)
        datasets = []
        for fileURI in fileURIs:
            file: str = fileURI.path
            ref = DatasetRef(datasetType, dataId=dataid, run=run)
            datasets.append(FileDataset(refs=ref, path=file))
        if len(datasets) > 1:
            raise ValueError("Only one calibration with the given dataset type and"
                             " detector id can be ingested into a given collection.")
        elif len(datasets) == 0:
            raise ValueError(f"No matching files are found in {locations}.")
        else:
            with butler.transaction():
                butler.ingest(*datasets, transfer=transfer)
            print(f"Ingested {ingest_type} calibration for detector {detector} into collection "
                  f"{run} from {datasets[0].path}")
