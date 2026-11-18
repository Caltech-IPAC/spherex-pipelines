import click

from lsst.daf.butler.cli.opt import (
    repo_argument,
    locations_argument,
    collections_argument,
    config_file_option,
    regex_option,
    run_option,
    transfer_option,
    where_option, limit_option
)
from lsst.daf.butler.cli.utils import (ButlerCommand)
from ... import script
from ..opt import (
    in_type_option,
    out_type_option,
    instrument_option,
    detector_option,
    iso_start_option,
    iso_end_option,
    sps_plan_option,
    dump_config_option
)
import json

# default regular expression to find sumulated files
# Selects all files that end with fits or gz
rawexp_re = r"^.+\.(?:fits|gz)"


@click.command(short_help="Ingest simulated images.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@locations_argument(help="LOCATIONS specifies files to ingest and/or locations to search for files.",
                    required=True)
@regex_option(default=rawexp_re,
              help="Regex string used to find files in directories listed in LOCATIONS. "
                   "Searches for fits files by default.")
@run_option(required=False)
@transfer_option()
@click.option("--ingest-type", default="raw", help="Raw image dataset type.")
@click.option("--obsid_min", default=None, help="Minimum observation id.")
@click.option("--obsid_max", default=None, help="Maximum observation id.")
@instrument_option()
def ingest_simulated(*args, **kwargs):
    """Ingest raw frames from a directory into the butler registry."""
    script.ingestSimulated(*args, **kwargs)


@click.command(short_help="Ingest calibration product with.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@locations_argument(help="LOCATIONS specifies files to ingest and/or locations to search for files.",
                    required=False)  # no location for fake calibrations
@regex_option(default=rawexp_re,
              help="Regex string used to find files in directories listed in LOCATIONS. "
                   "Searches for fits files by default.")
@run_option(required=False)
@transfer_option()
@click.option("--ingest-type", default="fake",
              help="Calibration dataset type.")
@detector_option(default=0)
@instrument_option()
def ingest_calibrations(*args, **kwargs):
    """Ingest raw frames from a directory into the butler registry."""
    script.ingestCalibrations(*args, **kwargs)


@click.command(short_help="Run estimate flat field pipeline.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@run_option(required=False)
@in_type_option(default="raw")
@out_type_option(default="flat")
@detector_option(default=2)
@iso_start_option(required=False)
@iso_end_option(required=False)
@config_file_option(required=False,
                    help="Path to a config file overrides file.")
@dump_config_option(required=False)
def estimate_flat_field(*args, **kwargs):
    """Run flat field estimation."""
    script.estimateFlatField(*args, **kwargs)


@click.command(short_help="Run L2a pipeline.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@collections_argument(required=True,
                      help="COLLECTIONS is one or more expressions that identify "
                           "the collections to search for input datasets, "
                           "including prior persistence state.")
@run_option(required=False)
@in_type_option(default="level1")
@out_type_option(default="level2a")
@sps_plan_option(default=None)
@detector_option(default=2)
@iso_start_option(required=False)
@iso_end_option(required=False)
@config_file_option(required=False,
                    help="Path to a config file overrides file.")
@dump_config_option(required=False)
@instrument_option()
@click.option("--delayed-model-save/--no-delayed-model-save", default=True,
              help="Save persistence model once at the end of the run (default: enabled)")
@click.option("--prior-required/--no-prior-required", default=True,
              help="Error if persistence prior is not found. (default: enabled)")
@click.option("--persistence-state-collection", required=False, default=None,
              help="The name of tagged collection for persistence state datasets.")
def run_l2a(*args, **kwargs):
    """Run l2a modules."""
    script.runL2a(*args, **kwargs)


@click.command(short_help="Recompute the zodi layer of level 2 images.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository.")
@collections_argument(required=True,
                      help="COLLECTIONS is one or more expressions that identify "
                           "the collections to search for the input images.")
@run_option(required=True)
@in_type_option(default="level2")
@out_type_option(default="level2")
@click.option("--calib-collections", required=False, default=None,
              help="Comma-separated collections to search for the zodi_scaling and "
                   "spectral_wcs calibrations (default: the input COLLECTIONS).")
@sps_plan_option(default=None)
@detector_option(default=None)
@where_option()
@limit_option()
@config_file_option(required=False,
                    help="Path to a config file overrides file.")
@dump_config_option(required=False)
@instrument_option()
@click.option("--skip-errors/--no-skip-errors", default=False,
              help="Log and skip exposures that fail instead of aborting "
                   "(default: disabled).")
@click.option("--skip-existing/--no-skip-existing", default=True,
              help="Skip exposures already present in the output run, so a "
                   "re-run resumes where the previous one stopped "
                   "(default: enabled).")
def recompute_zodi(repo, collections, output_run, calib_collections, **kwargs):
    """Recompute the zodi layer with Get Zodi Model."""
    if calib_collections is not None:
        calib_collections = [c.strip() for c in calib_collections.split(",") if c.strip()]
    script.recomputeZodi(repo, collections, output_run,
                         calib_collections=calib_collections, **kwargs)


@click.command(short_help="Update exposure region.", cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@collections_argument(required=True,
                      help="COLLECTIONS is one or more expressions that identify "
                           "the collections to search for input datasets, "
                           "including prior persistence state.")
@in_type_option(default="level2b_metadata")
@where_option()
@limit_option()
def update_exposure_region(*args, **kwargs):
    """Update region field in exposure dimension table."""
    script.updateExposureRegion(*args, **kwargs)


@click.command(short_help="Create a tagged collection of datasets containing primary calibrators.",
               cls=ButlerCommand)
@repo_argument(required=True, help="Path to the Butler repository.")
@collections_argument(required=True, help="List of input collections to search for datasets.")
@click.option("--output-collection", required=True, help="Name of the output tagged collection.")
@click.option("--dataset-types", required=True, help="List of dataset types, separated by commas.")
@click.option("--list-tiles", is_flag=True,
              help="Print the list of primary calibrator HEALPixel IDs and exit.")
def create_tagged_primary_calibrators(repo, collections, output_collection, dataset_types, list_tiles):
    """Create a tagged collection of primary calibrators."""
    dataset_types = dataset_types.split(",")
    collections = list(collections)
    script.createCalibratorTaggedCollection(
        repo=repo,
        collections=collections,
        output_collection=output_collection,
        dataset_types=dataset_types,
        list_tiles=list_tiles
    )


@click.command(short_help="Query for contiguous observations in a planning period.",
               cls=ButlerCommand)
@click.option("--repo", required=True, help="Path to the Butler repo.")
@click.option("--collections", "collections", required=True,
              help="Comma-separated input collections (e.g. col1,col2)")
@click.option("--detector", required=True, type=int, help="Detector ID")
@click.option("--instrument", default="spx", help="Instrument name (default: spx)")
@click.option("--output", required=True, type=click.Path(writable=True),
              help="Output JSON file to save the report.")
def report_contiguous_obs(repo, collections, detector, instrument, output):
    """Generate a report of contiguous observations for a detector."""
    # Convert comma-separated string to list
    collection_list = [c.strip() for c in collections.split(",") if c.strip()]
    # Call function
    result = script.get_contiguous_observation_info(repo, collection_list, detector, instrument)
    print(json.dumps({k: v.dict() for k, v in result.items()}, indent=2))
    # Write JSON to file
    with open(output, "w") as f:
        json.dump({k: v.dict() for k, v in result.items()}, f, indent=2)
        click.echo(f"Contiguity report saved to {output}")


@click.command(short_help="Export L2b WCS and make a tagged collection of L2a "
                          "corresponding to L2b finast failures.",
               cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@collections_argument(required=True,
                      help="COLLECTIONS is one or more collection expressions containing the "
                           "level2 and level2a dataset types.")
@click.option("--output-tag", required=True,
              help="Name of output tagged collection.")
@run_option(required=True)
@in_type_option(default="level2")
@out_type_option(default="l2bwcs")
@sps_plan_option(default=None)
@instrument_option()
@click.option("--progress-every", required=False, type=int, default=500,
              help="Show progress of export every N exposure(s) (default: 500).")
def export_l2bwcs_and_tagl2a(*args, **kwargs):
    """
    Export L2b WCS and make a tagged collection of L2a corresponding
    to L2b finast failures.
    """
    script.export_l2bwcs_and_tagl2a(*args, **kwargs)


@click.command(short_help="Add photometry exceptions as phot_meas datasets.",
               cls=ButlerCommand)
@repo_argument(required=True,
               help="REPO is the URI or path to the gen3 repository. "
                    "Will be created if it does not already exist.")
@collections_argument(required=True,
                      help="COLLECTIONS is one or more expressions that identify "
                           "the collections to search for input datasets.")
@in_type_option(default="phot_exceptions")
@where_option()
@limit_option()
def add_photometry_exceptions(repo, collections, in_type, where, limit):
    """Add photometry exceptions to the butler as `phot_meas` datasets."""
    script.addPhotometryExceptions(repo, collections, in_type, where, limit)
