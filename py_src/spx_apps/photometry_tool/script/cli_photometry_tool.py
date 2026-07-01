__all__ = ["perform_photometry"]

# Standard library imports
import logging
import os
import pathlib
import shutil
import signal
import sys
import traceback
from importlib import resources
import multiprocessing
# Related 3rd-party imports
import click
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from lsst.daf.butler import Butler, MissingDatasetTypeError

# Local application/library-specific imports
from ..temp_butler import collect_inputs_for_temp_butler, populate_temp_butler_from_export, \
    get_skypix_ids
from ..photometry import run_photometry
from ..exceptions import EmptyLVFQueryResultError, PipetaskError, VOTableSearchError
from ..votable import make_votable
from spx_apps import photometry_tool

from spx_tasks import PerformForcedPhotometryConfig


from spx_apps.common_cli import runid_option, jobid_option, result_dir_option, \
    err_dir_option, tmp_dir_option, log_dir_option, make_runid_default, \
    setup_exit_handlers, extract_log_config_and_ids_from_argv, force_override_runid, \
    extract_err_dir_from_argv, log_start_time, safe_exit, generate_error_summary, \
    parse_detectors, parse_time, fake_option

# Global variables for timing
_start_time = None
_script_name = "photometry_tool"

# Starts a fresh python interpreter process
multiprocessing.set_start_method("spawn", force=True)


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
    Parse multiple --point options into a list of dictionaries, preserving order.
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
    List[`dict`]
        List of dictionaries with point attributes and argument order.
    Raises
    ------
    click.BadParameter
        If the input format is invalid.
    """
    points = []
    for idx, v in enumerate(value):
        parts = v.split(",")
        if len(parts) < 2 or len(parts) > 3:
            raise click.BadParameter(f"{v}. Use ra,dec[,name]", param_hint="--point")
        try:
            ra, dec = float(parts[0]), float(parts[1])
            name = parts[2] if len(parts) == 3 else None
        except Exception:
            raise click.BadParameter(f"{v}. Use ra,dec[,name]", param_hint="--point")

        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
        points.append({
            "type": "point",
            "coord": coord,
            "name": name,
            "order": ctx.obj.get("_arg_order", []).pop(0) if ctx.obj else idx
        })
    return points


def parse_elliptical(ctx, param, value):
    """
     Parse multiple --elliptical options into a list of dictionaries, preserving order.

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
    for idx, v in enumerate(value):
        parts = v.split(",")
        if len(parts) < 6 or len(parts) > 7:
            raise click.BadParameter(f"{v}; expected: ra,dec,sersic,ratio,pa,effrad[,name].",
                                     param_hint="--elliptical")

        ra, dec, sersic, ratio, pa, effrad = map(float, parts[:6])
        name = parts[6] if len(parts) == 7 else None
        coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")

        ellipses.append({
            "type": "elliptical",
            "coord": coord,
            "name": name,
            "sersic": sersic,
            "ab_ratio": ratio,
            "phi": pa,
            "shape_r": effrad,
            "order": ctx.obj.get("_arg_order", []).pop(0) if ctx.obj else idx
        })
    return ellipses


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
    all_sources = (point or []) + (elliptical or [])
    if not all_sources:
        raise ValueError("No input sources to record.")

    # Sort by original input order
    all_sources = sorted(all_sources, key=lambda s: s["order"])

    records = []
    counter = 1

    for src in all_sources:
        name = src.get("name")
        if name is None:
            name = f"Target_{counter:04d}"

        if src["type"] == "point":
            records.append({
                "input_type": "point",
                "ra": src["coord"].ra.deg,
                "dec": src["coord"].dec.deg,
                "name": name,
                "sersic": -9999,
                "ab_ratio": -9999,
                "phi": -9999,
                "shape_r": -9999,
            })
        else:
            records.append({
                "input_type": "elliptical",
                "ra": src["coord"].ra.deg,
                "dec": src["coord"].dec.deg,
                "name": name,
                "sersic": src["sersic"],
                "ab_ratio": src["ab_ratio"],
                "phi": src["phi"],
                "shape_r": src["shape_r"],
            })

        counter += 1

    table = Table(rows=records, names=[
        "input_type", "ra", "dec", "name", "sersic", "ab_ratio", "phi", "shape_r"
    ])
    table["SPHERExRefID"] = np.arange(1, len(table)+1)

    for name in ("sersic", "ab_ratio", "phi", "shape_r"):
        table[name] = table[name].astype(np.float32)

    required_units = [("ra", u.deg), ("dec", u.deg), ("shape_r", u.arcsec)]
    for col, unit in required_units:
        table[col].unit = unit
    logging.info(table)

    return table


@click.command()
@result_dir_option()
@err_dir_option()
@runid_option("photo")
@jobid_option()
@tmp_dir_option()
@log_dir_option()
@fake_option()
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
    '--verbose/--no-verbose',
    default=False,
    show_default=True,
    help=(
        "Turns on additional verbosity in a job's logging; should not affect the output visible to the user. "
        "Defaults to --no-verbose. Translated within the science tools to logging control options."
    )
)
@click.option(
    '--use-simulator-bandpass/--no-use-simulator-bandpass',
    default=False,
    show_default=True,
    help=(
        "If True, simulator bandpass model is used to populate measurement wavelength"
        " and bandwidth. Otherwise, spectral_wcs calibration product is used."
    )
)
@click.option(
    '--add-image-ids/--no-add-image-ids',
    default=False,
    show_default=True,
    help=(
        "If True, image ids are included in the resulting VOTable. Defaults to False. "
    )
)
def perform_photometry(result_dir, tmp_dir, point, elliptical, time, detector, bkg_region, fit_positions,
                       err_dir: str, verbose: bool, log_dir: str, fake: str, use_simulator_bandpass: bool,
                       runid: str, jobid: str, add_image_ids: bool):
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
        logging.info(point)
        logging.info(elliptical)
        for p in point:
            logging.info(
                f"Target '{p['name']}': RA = {p['coord'].ra.deg:.6f}, DEC = {p['coord'].dec.deg:.6f} (ICRS)")

        for e in elliptical:
            logging.info(f"Elliptical '{e['name']}': \
                         RA = {e['coord'].ra.deg:.6f}, DEC = {e['coord'].dec.deg:.6f}, "
                         f"sersic={e['sersic']}, ratio={e['ab_ratio']}, pa={e['phi']}, effrad={e['shape_r']}")

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
            # Create pool for collect process
            with multiprocessing.Pool(processes=1) as pool:
                collect_result = pool.apply_async(
                    collect_inputs_for_temp_butler,
                    args=(butler_repo, target_table, tmp_dir, time, detectors, use_simulator_bandpass)
                )
                # This will automatically propagate exceptions from the child process
                collect_results = collect_result.get()

            # Create pool for populate process
            with multiprocessing.Pool(processes=1) as pool:
                populate_result = pool.apply_async(
                    populate_temp_butler_from_export,
                    args=(
                        collect_results["temp_butler_root"],
                        collect_results["export_file"],
                        collect_results["coverage_table"],
                        collect_results["srccat_dataids"]
                    )
                )
                # This will also automatically propagate exceptions
                populate_results = populate_result.get()

            temp_butler_root = populate_results["temp_butler_root"]

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
                                   write_to_file=True, error_type="fatal", user_error=True)
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)

        # Run L3a and create primary catalog
        try:
            skypix_ids_constraint: str = get_skypix_ids(butler, target_table)
            run_photometry(temp_butler_root, skypix_ids_constraint, bkg_region,
                           log_file=log_file, verbose=verbose,
                           use_simulator_bandpass=use_simulator_bandpass)
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
        combined_cat_glob = pathlib.Path(f"{temp_butler_root}/allsky_cat_combined").rglob(
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
            make_votable(catalog_path=combined_cat_path, votab_path=dest_file,
                         schema="tool", extra_resource=extra_resource,
                         add_publisher_did=add_image_ids, target_table=target_table)
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
                                   error_dir=err_dir, write_to_file=True,
                                   error_type="fatal", user_error=True)
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


cli.add_command(perform_photometry)


def main():
    # Setup exit handlers first, before any other code
    setup_exit_handlers(_script_name)

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
            error_type="fatal",
            user_error=True
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
            error_type="fatal",
            user_error=True
        )
        logging.error(e)
        logging.error(message)
        if log_dir is not None and error_symlink is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(19)

    except (Exception) as e:
        logging.error(e)
        message = str(traceback.format_exc())
        logging.error(message)
        generate_error_summary(
            message="Unknown error " + message,
            error_dir=err_dir,
            write_to_file=True,
            error_type="fatal",
        )
        if log_dir is not None and error_symlink is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(20)
