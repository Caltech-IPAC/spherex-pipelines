import yaml
__all__ = ["perform_photometry"]

# Standard library imports
import atexit
import logging
import os
import pathlib
import re
import shutil
import signal
import sys
import time
import traceback
from datetime import datetime
from importlib import resources
from collections import Counter
# Related 3rd-party imports
import click
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from lsst.daf.butler import Butler, MissingDatasetTypeError

# Local application/library-specific imports
from ..temp_butler import collect_inputs_in_temp_butler, get_skypix_ids
from ..photometry import run_photometry
from ..exceptions import EmptyLVFQueryResultError, PipetaskError, VOTableSearchError
from ..votable import make_votable
from spx_apps import photometry_tool
from spx_apps.common_cli import runid_option, jobid_option, result_dir_option, \
    err_dir_option, tmp_dir_option, log_dir_option, make_runid_default

from spx_tasks import PerformForcedPhotometryConfig

# Global variables for timing
_start_time = None
_script_name = "photometry_tool"


def log_start_time():
    """Log the script start time"""
    global _start_time
    _start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"=== {_script_name} STARTED at {start_datetime} ===")
    return _start_time


def log_end_time(exit_code=None):
    """Log the script end time and duration"""
    end_time = time.time()
    end_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if _start_time:
        duration = end_time - _start_time
        duration_str = f"{duration:.2f} seconds"
        if duration > 60:
            minutes = int(duration // 60)
            seconds = duration % 60
            duration_str = f"{minutes}m {seconds:.2f}s"
    else:
        duration_str = "unknown (start time not recorded)"

    exit_info = f" (exit code: {exit_code})" if exit_code is not None else ""
    # Make it crystal clear this is the final log entry from the main process
    logging.info("=" * 80)
    logging.info(f"FINAL LOG ENTRY - {_script_name} MAIN PROCESS ENDED")
    logging.info(f"End time: {end_datetime}")
    logging.info(f"Duration: {duration_str}{exit_info}")
    logging.info("NOTE: Subprocess logs may appear after this message due to buffering")
    logging.info("=" * 80)


def setup_exit_handlers():
    """Setup handlers to ensure end time is logged regardless of how script exits"""
    atexit.register(log_end_time)

    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logging.warning(f"Received signal {signal_name} ({signum}), shutting down...")
        log_end_time(exit_code=128 + signum)
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    for sig in [signal.SIGINT, signal.SIGTERM, signal.SIGHUP]:
        try:
            signal.signal(sig, signal_handler)
        except (OSError, ValueError):
            pass


def safe_exit(code):
    """Exit with logging"""
    atexit.unregister(log_end_time)  # Prevent duplicate call
    log_end_time(exit_code=code)
    sys.exit(code)


@click.group()
@click.option('--version', is_flag=True, is_eager=True,
              help='Show the version and exit.')
@click.pass_context
def cli(ctx, version):
    """Photometry tool"""
    if version:

        click.echo(f"Photometry Tool Version: {photometry_tool.__version__}")
        ctx.exit()


def parse_point(ctx, param, value):
    """
    Parse multiple --point options into a list of SkyCoord and name tuples.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    param : click.Parameter
        Click parameter object.
    value : tuple of str
        Tuple of strings formatted as 'ra,dec[,name]'.

    Returns
    -------
    list of tuple
        List of tuples with (SkyCoord, name) for each point.

    Raises
    ------
    click.BadParameter
        If the input format is invalid or contains duplicate names.
    """
    points = []
    names_seen = ctx.ensure_object(dict).setdefault("names_seen", set())

    for v in value:
        parts = v.split(",")
        if len(parts) < 2 or len(parts) > 3:
            raise click.BadParameter(f"{v}. Use ra,dec[,name]",
                                     param_hint="--point")
        try:
            ra, dec = float(parts[0]), float(parts[1])
            name = parts[2] if len(parts) == 3 else f"{ra}_{dec}"
        except Exception:
            raise click.BadParameter(f"{v}. Use ra,dec[,name]",
                                     param_hint="--point")

        if name in names_seen:
            raise click.BadParameter(f"Duplicate point name: {name}",
                                     param_hint="--point")
        names_seen.add(name)
        try:
            coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
            points.append((coord, name))
        except Exception:
            raise click.BadParameter(f"Invalid point: {v}",
                                     param_hint="--point")

    return points


def parse_time(value) -> tuple[Time, Time] | None:
    """
    Parse --time option into a tuple of astropy.time.Time objects using MJD.

    Parameters
    ----------
    value : `str`
        Time range as 'min,max' using MJD values.
        Use '-Inf' or '+Inf' for open-ended bounds.

    Returns
    -------
    `tuple` of `astropy.time.Time` or ``None``
        Tuple of (min_time, max_time).

    Raises
    ------
    click.BadParameter
        If the input format is invalid.
    """
    try:
        if value is None:
            return value

        parts = value.split(",")
        if len(parts) != 2:
            raise ValueError

        parsed_parts = []
        if parts[0] == "-Inf" or parts[0] == "":
            # '2024-01-01T00:00:00'
            parsed_parts.append(Time(60310.0, format="mjd", scale="utc"))
        else:
            parsed_parts.append(Time(float(parts[0]), format="mjd", scale="utc"))
        if parts[1] == "+Inf" or parts[1] == "":
            # '2050-01-01T00:00:00'
            parsed_parts.append(Time(69807.0, format="mjd", scale="utc"))
        else:
            parsed_parts.append(Time(float(parts[1]), format="mjd", scale="utc"))
        return tuple(parsed_parts)
    except Exception:
        raise click.BadParameter(f"{value}. Use 'min,max' with MJD values.",
                                 param_hint="--time")


def parse_elliptical(ctx, param, value):
    """
    Parse multiple --elliptical options into a list of elliptical source definitions.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    param : click.Parameter
        Click parameter object.
    value : tuple of str
        Tuple of strings formatted as 'ra,dec,sersic,ratio,pa,effrad[,name]'.

    Returns
    -------
    list of tuple
        List of tuples with (SkyCoord, sersic, ratio, pa, effrad, name).

    Raises
    ------
    click.BadParameter
        If the input format is invalid or contains duplicate names.
    """
    ellipses = []
    names_seen = ctx.meta.setdefault("names_seen", set())

    for v in value:
        parts = v.split(",")
        if len(parts) < 6 or len(parts) > 7:
            raise click.BadParameter(
                f"{v}; expected: ra,dec,sersic,ratio,pa,effrad[,name].",
                param_hint="--elliptical")

        ra, dec, sersic, ratio, pa, effrad = map(float, parts[:6])
        name = parts[6] if len(parts) == 7 else f"{ra}_{dec}"

        if name in names_seen:
            raise click.BadParameter(f"Duplicate name: {name}",
                                     param_hint="--elliptical")
        names_seen.add(name)

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        ellipses.append((coord, sersic, ratio, pa, effrad, name))

    return ellipses


def parse_detectors(value) -> list[int] | None:
    """
    Parse multiple --detector options into a list of detector IDs.

    Parameters
    ----------
    value : `tuple` of `str`
        Tuple of detector labels, e.g. (``D3``, ``D4``)

    Returns
    -------
    `list` of `int` or `None`
        List of integer detector ids from 1 to 6,
        or `None` if no detectors were specified.

    Raises
    ------
    click.BadParameter
        If the input value contains duplicate or invalid detector label.
    """
    if not value:
        return None
    else:
        counts = Counter(value)
        duplicates = [item for item, count in counts.items() if count > 1]
        if len(duplicates) > 0:
            raise click.BadParameter(f"duplicate label: {duplicates}.",
                                     param_hint="--detector")

        pattern = re.compile(r"^D(\d+)$")
        detectors = []
        for det_str in value:
            match = pattern.fullmatch(det_str)
            if not match:
                raise click.BadParameter(f"{det_str}; unexpected detector label format.",
                                         param_hint="--detector")
            det = int(match.group(1))
            if det < 1 or det > 6:
                raise click.BadParameter(f"D{det}; detector id must be an integer from 1 to 6.",
                                         param_hint="--detector")
            detectors.append(det)
        return detectors


def validate_bkg_region(value: int) -> None:
    """
    Parameters
    ----------
    value :
        Background region value

    Raises
    ------
    click.BadParameter
        If the input value is inconsistent with other photometry parameters.
    """
    try:
        _ = PerformForcedPhotometryConfig(bkg_side_length=value)
    except ValueError as ve:
        raise click.BadParameter(f"{value}. {ve}", param_hint="--bkg-region")


def generate_input_table(point, elliptical) -> Table:
    """
    Save input parameters to a consolidated input table, which
    can be used by the Forced Photometry module.

    Parameters
    ----------
    point : list of tuple
        List of point sources as (SkyCoord, name).
    elliptical : list of tuple
        List of elliptical sources as (SkyCoord, sersic, ratio, pa, effrad, name).

    Returns
    -------
    result : astropy.table.Table
    """
    records = []

    # Point sources
    # -9999 is used in ref catalog for null values
    for coord, name in point or []:
        records.append({
            "input_type": "point",
            "ra": coord.ra.deg,
            "dec": coord.dec.deg,
            "name": name,
            "sersic": -9999.0,
            "ab_ratio": -9999.0,
            "phi": -9999.0,
            "shape_r": -9999.0,
        })

    # Elliptical sources
    for coord, sersic, ratio, pa, effrad, name in elliptical or []:
        records.append({
            "input_type": "elliptical",
            "ra": coord.ra.deg,
            "dec": coord.dec.deg,
            "name": name,
            "sersic": sersic,
            "ab_ratio": ratio,
            "phi": pa,
            "shape_r": effrad,
        })

    if not records:
        raise ValueError("No input sources to record.")

    table = Table(rows=records, names=[
        "input_type", "ra", "dec", "name", "sersic", "ab_ratio", "phi", "shape_r"
    ])

    # Add SPHERExRefID column (mimicking df.index)
    table["SPHERExRefID"] = np.arange(len(table))

    # Cast float columns to float32
    for name in ("sersic", "ab_ratio", "phi", "shape_r"):
        table[name] = table[name].astype(np.float32)

    # units requires by source catalog validator
    required_units = [("ra", u.deg), ("dec", u.deg), ("shape_r", u.arcsec)]
    for col, unit in required_units:
        table[col].unit = unit

    return table


def generate_error_summary(message="", error_dir=None, write_to_file=False, error_type="fatal"):
    """
    Generate a JSON-style error summary based on an exit code.

    Args:
        message (str, optional): A descriptive error message.
        Defaults to an empty string.
        error_dir (str, optional): Directory to write
        the errorSummary.yaml file. Defaults to None.
        write_to_file (bool, optional): Whether to write
        the error summary to a file. Defaults to False.
        error_type (str, optional): Type of error
        (e.g., "fatal", "warning"). Defaults to "fatal".

    Returns:
        dict: The error summary structure.
    """
    # Construct error summary structure
    error_summary = {
        "errorType": error_type,
        "message": str(message)
    }

    # Optionally write to file
    if write_to_file and error_dir:
        os.makedirs(error_dir, exist_ok=True)
        error_path = os.path.join(error_dir, "errorSummary.yaml")
        with open(error_path, 'w') as f:
            yaml.dump(error_summary, f)

    return error_summary


@click.command()
@result_dir_option()
@err_dir_option()
@runid_option("photo")
@jobid_option()
@tmp_dir_option()
@log_dir_option()
@click.option("--point", multiple=True, callback=parse_point,
              help="Point-like target in format ra,dec[,name].\n"
              "[One of  --point OR --elliptical is required.]")
@click.option("--elliptical", multiple=True, callback=parse_elliptical,
              help="Elliptical target ra,dec,sersic,ratio,pa,effrad[,name].\n\n"
              "[One of  --point OR --elliptical is required.]")
@click.option("--time", help="Time range as min,max (MJD). [optional]", default=None)
@click.option("--detector", multiple=True, type=str,
              help="Detector label: D1, D2, D3, D4, D5, or D6\n[optional]")
@click.option("--bkg-region", default=15, type=int,
              help="Background region in pixels (default: 15)\n[optional]")
@click.option("--fit-positions/--no-fit-positions", default=False,
              help="Allow positions to float in Tractor fit. (Not implemented.)\n"
              "[optional flag]")
@click.option(
    '--fake',
    type=click.Choice(['normal', 'error-short', 'error-detail', 'error-signal']),
    default=None,
    help='Run a simulated job. Modes: '
         '"normal" = simulate successful run, '
         '"error-short" = create errorSummary.yaml, '
         '"error-detail" = create errorSummary.yaml and symlink "error" to actual error file, '
         '"error-signal" = exit with SIGINT.\n'
         '[optional flag]'
)
@click.option(
    '--verbose/--no-verbose',
    default=False,
    show_default=True,
    help=(
        "Turns on additional verbosity in a job's logging; should not affect the output visible to the user. "
        "Defaults to --no-verbose. Translated within the science tools to logging control options."
    )
)
def perform_photometry(result_dir, tmp_dir, point, elliptical, time, detector, bkg_region, fit_positions,
                       err_dir: str, verbose: bool, log_dir: str, fake: str, runid: str, jobid: str):
    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
    error_symlink = pathlib.Path(os.path.join(err_dir, "error"))

    if fake == "normal":
        src_path = resources.files("spx_apps") / "photometry_tool" / "data" / "ff-result.vot"
        destination_dir = pathlib.Path(result_dir)
        dest_file = destination_dir / src_path.name
        with resources.as_file(src_path) as src_real_path:
            shutil.copy(src_real_path, dest_file)
    elif fake is None:
        time = parse_time(time)
        detectors = parse_detectors(detector)
        # Real run
        logging.info(f"Result directory: {result_dir}")

        for coord, name in point:
            logging.info(f"Target '{name}': RA = {coord.ra.deg:.6f}, DEC = {coord.dec.deg:.6f} (ICRS)")

        for coord, sersic, ratio, pa, effrad, name in elliptical:
            logging.info(f"Elliptical '{name}': RA = {coord.ra.deg:.6f}, DEC = {coord.dec.deg:.6f}, "
                         f"sersic={sersic}, ratio={ratio}, pa={pa}, effrad={effrad}")

        if time:
            logging.info(f"Time filter: MJD range = {time}")
        if detectors:
            logging.info(f"Wavelength filter: detectors={detectors}")

        logging.info(f"Background region size: {bkg_region} pixels")
        validate_bkg_region(bkg_region)
        logging.info(f"Fit positions: {'enabled' if fit_positions else 'disabled'}")
        if fit_positions:
            raise click.BadParameter("position fitting is not implemented.",
                                     param_hint="--fit-positions")

        # === Butler Repo and Data Checks ===
        butler_repo = os.getenv("SPXTOOL_BUTLER")
        try:
            butler = Butler(butler_repo)

        except Exception as e:
            error_description = f"Butler ({butler_repo}) does not exist or is not a valid Butler path."
            logging.error(e)
            logging.error(error_description)

            generate_error_summary(message=error_description, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(11)
        target_table = generate_input_table(point, elliptical)
        logging.info("Target Table:\n" + repr(target_table))

        temp_butler_root = None
        try:
            temp_butler = collect_inputs_in_temp_butler(
                butler=butler,
                target_table=target_table,
                tmp_dir=tmp_dir,
                time_interval=time,
                detectors=detectors
            )
            for datastore_root in temp_butler.get_datastore_roots().values():
                temp_butler_root = datastore_root
                logging.info(f"Temporary Butler Datastore Root URI: {datastore_root}")
        except MissingDatasetTypeError as mdte:
            logging.error(f"Error: {mdte}")
            exit_code = 12
            generate_error_summary(message=mdte, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
        except EmptyLVFQueryResultError as elqe:
            logging.error(f"Error: {elqe}")
            exit_code = 13
            generate_error_summary(message=elqe, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)

        # Run L3a and create primary catalog
        try:
            skypix_ids_constraint: str = get_skypix_ids(butler, target_table)
            run_photometry(temp_butler_root.ospath, skypix_ids_constraint, bkg_region,
                           log_file=log_file, verbose=verbose)
        except PipetaskError as pte:
            logging.error(f"{pte.short_desc}. Temp butler: {temp_butler_root}")
            if pte.err.stderr:
                logging.error("\n"+pte.err.stderr)
            exit_code = 14
            generate_error_summary(message="Subprocess executing one of the pipelines failed",
                                   error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            safe_exit(exit_code)
        except BaseException:
            msg = traceback.format_exc()
            logging.error(msg)
            exit_code = 15
            msg = "Unexpected error when running photometry."
            generate_error_summary(message=msg, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)

        # Find combined catalog file
        combined_cat_glob = pathlib.Path(f"{temp_butler_root.ospath}/allsky_cat_combined").rglob(
            pattern="*.parq")
        combined_cat_path = next(combined_cat_glob, None)
        if combined_cat_path is None:
            err_msg = "Error: No combined all sky catalog found."
            logging.error(err_msg)
            generate_error_summary(message=err_msg, error_dir=err_dir, write_to_file=True,
                                   error_type="fatal")
            exit_code = 16
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)

        # Set up arguments/options for VOTable creation
        destination_dir = pathlib.Path(result_dir)
        dest_file = destination_dir / "phot-tool-result.vot"
        extra_resource = os.getenv("SPXPHOTO_EXTRA_RESOURCE", default=None)

        # Make VOTable
        try:
            make_votable(catalog_path=combined_cat_path, votab_path=dest_file, schema="tool",
                         extra_resource=extra_resource)
        except VOTableSearchError as vtse:
            logging.error(vtse)
            exit_code = 17
            generate_error_summary(message="Failed to add service descriptor to VOTable XML content.",
                                   error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
        except BaseException:
            msg = traceback.format_exc()
            logging.error(msg)
            exit_code = 18
            msg = "Unexpected failure during VOTable file creation or modification:\n"
            generate_error_summary(message=msg, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
    elif fake == "error-short":
        exit_code = 13
        try:
            raise EmptyLVFQueryResultError(n_src=1,
                                           instrument="spx",
                                           dataset_type="level2")
        except EmptyLVFQueryResultError as mdte:
            logging.error(mdte)
            generate_error_summary(message="No level2 image matches for the provided sources.",
                                   error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
    elif fake == "error-signal":
        # Kill with SIGINT
        os.kill(os.getpid(), signal.SIGINT)
    elif fake == "error-detail":
        exit_code = 13
        try:
            raise EmptyLVFQueryResultError(n_src=1,
                                           instrument="spx",
                                           dataset_type="level2")
        except EmptyLVFQueryResultError as mdte:
            text = "Sample text corresponding to detailed errors"
            # Write mode ('w') — overwrites if the file exists
            error_path = os.path.join(err_dir, "output.txt")
            with open(error_path, "w") as file:
                file.write(text)
            generate_error_summary(message=mdte, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            error_symlink = pathlib.Path(os.path.join(err_dir, "error"))
            error_symlink.symlink_to(error_path)
            safe_exit(exit_code)

    # === Create symlink to output VOTable ===
    symlink_path = destination_dir / "result"
    if symlink_path.exists() or symlink_path.is_symlink():
        symlink_path.unlink()

    symlink_path.symlink_to(dest_file)
    logging.info(f"Copied VOTable to {dest_file} and symlinked as {symlink_path}")


def extract_err_dir_from_argv(default=os.getcwd()):
    argv = sys.argv
    if '--err-dir' in argv:
        try:
            return argv[argv.index('--err-dir') + 1]
        except IndexError:
            return default
    for arg in argv:
        if arg.startswith('--err-dir='):
            return arg.split('=', 1)[1]
    return default


def extract_log_config_and_ids_from_argv(default_runid=None, default_jobid=None):
    argv = sys.argv

    # Defaults
    log_dir = os.getcwd()
    verbose = False
    runid = default_runid
    jobid = default_jobid

    def extract_arg(key):
        if key in argv:
            try:
                return argv[argv.index(key) + 1]
            except IndexError:
                return None
        for arg in argv:
            if arg.startswith(f'{key}='):
                return arg.split('=', 1)[1]
        return None

    # Parse arguments
    log_dir = extract_arg('--log-dir') or log_dir
    runid = extract_arg('--runid') or runid
    jobid = extract_arg('--jobid') or jobid

    if '--verbose' in argv:
        verbose = True
    if '--no-verbose' in argv:
        verbose = False  # Overrides --verbose if both are present

    return log_dir, verbose, runid, jobid


def force_override_runid(new_value):
    # Remove all existing --runid or --runid=XYZ entries
    new_argv = []
    skip_next = False
    for i, arg in enumerate(sys.argv):
        if skip_next:
            skip_next = False
            continue
        if arg == "--runid":
            skip_next = True  # skip the next value
            continue
        elif arg.startswith("--runid="):
            continue
        else:
            new_argv.append(arg)

    # Append our forced --runid value
    new_argv += ["--runid", new_value]
    sys.argv = new_argv


cli.add_command(perform_photometry)


def main():
    # Setup exit handlers first, before any other code
    setup_exit_handlers()

    try:
        # Extract log dir and associated flags.
        log_dir, verbose, runid, jobid = extract_log_config_and_ids_from_argv()
        error_symlink = None
        if runid is None:
            runid = make_runid_default("photo")
            force_override_runid(new_value=runid)
        err_dir = extract_err_dir_from_argv()
        log_level = logging.DEBUG if verbose else logging.INFO
        if log_dir is None:
            logging.basicConfig(
                level=log_level,
                stream=sys.stdout,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            log_file = os.path.join(log_dir, f"{runid}_JOB_{jobid}_log.out")

            logging.basicConfig(
                filename=log_file,
                filemode='w',
                format='%(asctime)s %(levelname)s: %(message)s',
                level=log_level,
                force=True
            )
            error_symlink = pathlib.Path(os.path.join(err_dir, "error"))

        # Log start time after logging is configured
        log_start_time()

        # Use runid here as foo and pass it to photometry tool
        cli(standalone_mode=False, obj={})
    except (click.BadParameter, click.MissingParameter) as e:
        param_name = e.param_hint
        message = f"Command line validation error for {param_name}"
        generate_error_summary(
            message=message,
            error_dir=err_dir,
            write_to_file=True,
            error_type="fatal"
        )
        logging.error(e)
        logging.error(message)
        if log_dir is not None and error_symlink is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(19)

    except click.UsageError as e:
        message = "Command line validation error"
        generate_error_summary(
            message=message,
            error_dir=err_dir,
            write_to_file=True,
            error_type="fatal"
        )
        logging.error(e)
        logging.error(message)
        if log_dir is not None and error_symlink is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(19)

    except (Exception) as e:
        logging.error(e)
        logging.error(traceback.format_exc())
        generate_error_summary(
            message="Unknown error",
            error_dir=err_dir,
            write_to_file=True,
            error_type="fatal"
        )
        if log_dir is not None and error_symlink is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(20)
