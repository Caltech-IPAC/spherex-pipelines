__all__ = ["create_mosaic"]

# Standard library imports
import logging
import os
import pathlib
import sys
import traceback
import signal
import multiprocessing
import shutil
# Related 3rd-party imports
import click
from spx_apps import mosaic_tool

from spx_apps.common_cli import runid_option, jobid_option, result_dir_option, \
    err_dir_option, tmp_dir_option, log_dir_option, make_runid_default, \
    setup_exit_handlers, extract_log_config_and_ids_from_argv, force_override_runid, \
    extract_err_dir_from_argv, log_start_time, safe_exit, generate_error_summary, \
    parse_detectors, parse_time, fake_option
from spx_apps.photometry_tool.exceptions import EmptyLVFQueryResultError


# Global variables for timing
_start_time = None
_script_name = "mosaic_tool"

# Starts a fresh python interpreter process
multiprocessing.set_start_method("spawn", force=True)


@click.group()
@click.option('--version', is_flag=True, is_eager=True,
              help='Show the version and exit.')
@click.pass_context
def cli(ctx, version):
    """Mosaic tool"""
    if version:

        logging.info(f"Mosaic Tool Version: {mosaic_tool.__version__}")
        ctx.exit()


def validate_center(ctx, param, value):
    if not value:
        raise click.BadParameter("--center is required (format: ra,dec)")
    try:
        ra_str, dec_str = value.split(",")
        ra, dec = float(ra_str), float(dec_str)
    except ValueError:
        raise click.BadParameter(f"Invalid --center format: '{value}', expected ra,dec (floats)")

    if not (0 <= ra < 360):
        raise click.BadParameter(f"Invalid RA {ra}: must be in [0, 360)")
    if not (-90 <= dec <= 90):
        raise click.BadParameter(f"Invalid Dec {dec}: must be in [-90, 90]")

    return (ra, dec)


def validate_size(ctx, param, value):
    if not value:
        raise click.BadParameter("--size is required (format: xsize,ysize)")
    try:
        x_str, y_str = value.split(",")
        xsize, ysize = float(x_str), float(y_str)
    except ValueError:
        raise click.BadParameter(f"Invalid --size format: '{value}', expected xsize,ysize (floats)")

    if xsize <= 0 or ysize <= 0:
        raise click.BadParameter(f"Invalid size values: {xsize},{ysize}. Both must be positive.")
    return (xsize, ysize)


def validate_rotation(ctx, param, value):
    if not (-360 <= value <= 360):
        raise click.BadParameter(f"Invalid rotation {value}: must be within [-360, 360].")
    return value


def validate_pixel_scale(ctx, param, value):
    if value < 6.15:
        raise click.BadParameter(f"Invalid pixel scale {value}: minimum is 6.15 arcsec.")
    return value


def validate_channel_range(ctx, param, value):
    if value is None:
        return None
    try:
        parts = [float(x) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter(f"Invalid --channel-range format: '{value}', expected min,max floats.")

    if len(parts) != 2:
        raise click.BadParameter(f"--channel-range must have exactly 2 comma-separated values\
                                    (min,max). {parts}")
    if not (0.75 <= float(value.split(",")[0]) <= 1 and 0.75 <= float(value.split(",")[1]) <= 1):
        raise click.BadParameter(
            "Invalid --channel-range format: '{value}', all values must be between 0.75 and 1.")

    min_ch, max_ch = parts
    if min_ch <= 0 or max_ch <= 0:
        raise click.BadParameter("Invalid channel-range values: must be positive.")
    if min_ch >= max_ch:
        raise click.BadParameter("Invalid channel-range: min must be less than max.")
    return (min_ch, max_ch)


def validate_user_channels(ctx, param, value):
    if value is None:
        return None
    try:
        vals = [float(x) for x in value.split(",")]
    except ValueError:
        raise click.BadParameter(
            f"Invalid --user-channels format: '{value}', expected comma-separated floats.")

    if len(vals) % 2 != 0:
        raise click.BadParameter("--user-channels: must specify an even number of values (pairs of min,max).")

    vals = [float(x) for x in value.split(",")]
    for val in vals:
        if not (0.75 <= val <= 1):
            raise click.BadParameter(f"--user-channels: {vals}  all values must be between 0.75 and 1.")
    # Validate ordering inside each pair
    for i in range(0, len(vals), 2):
        if vals[i] >= vals[i + 1]:
            raise click.BadParameter(
                f"--user-channels: invalid pair ({vals[i]}, {vals[i+1]}). Each min must be < max."
            )

    return list(zip(vals[::2], vals[1::2]))  # list of (min, max) tuples


def validate_channel_exclusivity(ctx, param, value):
    # Defer exclusivity check to after all parameters parsed
    channel_range = ctx.params.get("channel_range")
    user_channels = ctx.params.get("user_channels")

    if channel_range and user_channels:
        raise click.BadParameter("Options --channel-range and --user-channels are mutually exclusive.")
    return value


@click.command()
@result_dir_option()
@err_dir_option()
@runid_option("mosaic")
@jobid_option()
@tmp_dir_option()
@log_dir_option()
@fake_option()
@click.option("--center", required=True, callback=validate_center,
              help="""Center of a rectangular mosaic in TAN projection.
                    Values are in ICRS degrees, with the usual con
                    straints: ra in [0, 360) and dec in [-90, 90]. The
                    center is placed in the center of a pixel, so the
                    mosaic will always have an odd number of rows and
                    columns. Example: --center=6.2001,-14.8314."""
              )
@click.option("--size", required=True, callback=validate_size,
              help="""
                Sizes of the row and column “diameters” of the
                mosaic. Values are in relative degrees in the rotated
                frame controlled by the --rotation option.
                The precise definition, which matters more near
                the poles, and for larger sizes, is that the limits of
                the image are determined as follows: go out along
                the central diameters of the mosaic in the row/x (or
                column/y) directions from the requested center, in
                the TAN projection, one pixel at a time, until the
                outer edge of a pixel is more than half of the requested
                size away from the center. This does not in
                general yield the same result as “divide the relative
                size by the pixel size”. The specified values are, in
                general, not equal to the angular path lengths of
                the sides of the image, nor the great-circle distance
                between adjacent corners. No default. Provisionally:
                maximum size is 5 degrees. Format: xsize,ysize. Example: --size=0.05,0.1.""")
@click.option("--rotation", type=float, default=0.0, show_default=True, callback=validate_rotation,
              help="""Position angle of the mosaic relative to equatorial
                    north. Values are in degrees, with positive values
                    corresponding to a rotation east of north. Specifically,
                    the position angle controls the angle from
                    the north vector at the center of the mosaic to the
                    vector along the column direction from that point
                    in the direction of increasing row number. Default
                    is 0. Allowed range is [-360,360]. Example: --rotation=-45""")
@click.option("--pixel-scale", type=float, default=6.15, show_default=True, callback=validate_pixel_scale,
              help="""Pixel scale of the produced mosaic at its center,
                    reflected directly in CDELT1/2 in the result. Values
                    are in arcseconds. Default is 6.15 arcseconds.
                    Allowed range is [6.15,TBD]. (Super-resolution is
                    not an option for at least the initial release, if ever;
                    maximum scale will be set once performance can
                    be tested.)""")
@click.option("--channel-range", callback=validate_channel_range, default=None,
              help="""Range of central wavelengths of standard
                    SPHEREx spectral channels to include in
                    the result. Values are in microns and must be in range [0.75,1]
                    . Default is the 102 standard channels. Incompatible with the
                    --user-channels option. V1 of the tool will not
                    provide access to the “partial channels” at the
                    short and long wavelength ends of the detectors.
                    (This might be enabled in future versions for
                    the two partial channels at the extrema of the
                    whole SPHEREx bandpass, and/or for the partial
                    channels for a single detector if only one detector
                    is chosen via --detector; or this capability may
                    never be exposed in the IRSA UI, but only
                    programmatically.)""")
@click.option("--user-channels", callback=validate_user_channels, default=None,
              help=""" 'Top-hat' spectral channels to be used to construct
                        the planes of the mosaic result. Values
                        are in microns. Pairs of consecutive values in
                        the list define channels. No default; must be
                        specified explicitly.Values must be in range [0.75,1].
                        Specifying just a single channel
                        is allowed; this still produces cube extensions
                        (NAXIS=3), just with NAXIS3=1. Incompatible with
                        the --channel-range option; overrides the normal
                        “all-standard-channels” default that applies when
                        --channel-range is not specified. TBD whether
                        channels will be allowed to overlap. There is a
                        good science case to do so, and it does not make
                        the SSDC implementation harder, we think. V1
                        of the tool will not provide access to the “partial
                        channels” at the short and long wavelength ends
                        of the detectors (TBC). (This might be enabled in
                        future versions for the two partial channels at the
                        extrema of the whole SPHEREx bandpass, and/or
                        for the partial channels for a single detector if only
                        one detector is chosen via --detector; or this capability
                        may never be exposed in the IRSA UI, but
                        only programmatically.)""")
@click.option("--time", help="""Time restriction. Values are in days, as MJD values,
                            following the IVOA DALI and SIAv2 standards.
                            Restricts the selection of Level 2 LVF images
                            used as input to the mosaic. Note that application
                            of a time restriction may well interfere
                            with obtaining the full requested spectral coverage.
                            The tool will not warn about this before execution.
                            The min value may be given as -Inf and the max
                            value may be given as +Inf in order to specify the
                            absence of a lower or upper bound on the time interval,
                            respectively. Defaults to no restriction.""", default=None)
@click.option("--detector", multiple=True, type=str,
              help="""Detector label: D1, D2, D3, D4, D5, or D6. Restricts
                    inputs to come from the named detector(s).
                    Not intended to be available through the IRSA UI;
                    mainly useful in certain testing situations. Will in
                    general interfere with obtaining the full requested
                    spectral coverage. Defaults to no restriction. Selecting
                    no detectors is treated as equivalent to selecting
                    all 6.""")
@click.option(
    '--verbose/--no-verbose',
    default=False,
    show_default=True,
    help=(
        "Turns on additional verbosity in a job's logging; should not affect the output visible to the user. "
        "Defaults to --no-verbose. Translated within the science tools to logging control options."
    )
)
def create_mosaic(
    result_dir,
    err_dir: str,
    runid: str,
    jobid: str,
    tmp_dir: str,
    log_dir: str,
    fake: str,
    verbose: bool,
    center,
    size,
    rotation,
    pixel_scale,
    channel_range,
    user_channels,
    time,
    detector,
):
    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
    error_symlink = pathlib.Path(os.path.join(err_dir, "error"))
    time = parse_time(time)
    detectors = parse_detectors(detector)
    x_size, y_size = size[0], size[1]
    size_scale_text = f"Size (x,y): {size}, Pixel scale: {pixel_scale}"
    if not (x_size/pixel_scale > 1):
        raise click.BadParameter(f"{size_scale_text}, x_size / --pixel-scale should be greater than 1")
    if not (y_size/pixel_scale > 1):
        raise click.BadParameter(f"{size_scale_text}, y_size / --pixel-scale should be greater than 1")

    logging.info("Mosaic configuration:")
    logging.info(f"  Center (ra,dec): {center}")
    logging.info(f"  Size (x,y): {size}")
    logging.info(f"  Rotation: {rotation}")
    logging.info(f"  Pixel scale: {pixel_scale}")
    logging.info(f"  Channel range: {channel_range}")
    logging.info(f"  User channels: {user_channels}")
    logging.info(f"  Time: {time}")
    logging.info(
        f"  Detector(s): {detectors}.(Not intended for to be exposed via UI, None indicates all detectors)")
    logging.info(f"  Verbose: {verbose}")
    logging.info(f"  Fake mode: {fake}")
    logging.info(f"  Run ID: {runid}, Job ID: {jobid}")
    logging.info(f"  Result dir: {result_dir}")
    logging.info(f"  Temp dir: {tmp_dir}, Error dir: {err_dir}, Log dir: {log_dir}")

    if fake == "normal":
        src_path = pathlib.Path(os.getenv("SPXMOSAIC_SAMPLE_RESULT",
                                          "~/tmp-result/mosaic-tool-result.fits")).expanduser()
        destination_dir = pathlib.Path(result_dir)
        dest_file = destination_dir / src_path.name
        # Ensure the destination directory exists
        destination_dir.mkdir(parents=True, exist_ok=True)

        # Destination file path
        dest_file = destination_dir / src_path.name
        # Copy the file
        shutil.copy(src_path, dest_file)
        # === Create symlink to output VOTable ===
        symlink_path = destination_dir / "result"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        symlink_path.symlink_to(dest_file)
        logging.info(f"Copied VOTable to {dest_file} and symlinked as {symlink_path}")
    elif fake in ["error-short", "error-detail"]:
        exit_code = 13
        try:
            raise EmptyLVFQueryResultError(n_src=1,
                                           instrument="spx",
                                           dataset_type="level2")
        except EmptyLVFQueryResultError as mdte:
            logging.error(mdte)
            generate_error_summary(message="No level2 image matches for the requested mosaic volume.",
                                   error_dir=err_dir, write_to_file=True,
                                   error_type="fatal", user_error=True)
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
    elif fake == "error-signal":
        # Kill with SIGINT
        os.kill(os.getpid(), signal.SIGINT)


cli.add_command(create_mosaic)


def main():
    # Setup exit handlers first, before any other code
    setup_exit_handlers(_script_name)

    try:
        # Extract log dir and associated flags.
        log_dir, verbose, runid, jobid = extract_log_config_and_ids_from_argv()
        error_symlink = None
        if runid is None:
            runid = make_runid_default("mosaic")
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

        # Use runid here as foo and pass it to mosaic tool
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
