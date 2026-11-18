__all__ = ["create_mosaic"]

# Standard library imports
import logging
import os
import pathlib
import sys
import traceback
import signal
import multiprocessing
# Related 3rd-party imports
import click
from astropy.io import fits
from spx_apps import mosaic_tool
from spx_apps.mosaic_tool.mosaic_builder import (
    build_mosaics_from_temp_butler,
    build_interpolated_mosaic,
)
from pathlib import Path
import tempfile
from spx_apps.common_cli import runid_option, jobid_option, result_dir_option, \
    err_dir_option, tmp_dir_option, log_dir_option, make_runid_default, \
    setup_exit_handlers, extract_log_config_and_ids_from_argv, force_override_runid, \
    extract_err_dir_from_argv, log_start_time, safe_exit, generate_error_summary, \
    parse_detectors, parse_time, fake_option, use_deep_option, validate_writable
from spx_apps.photometry_tool.temp_butler import populate_temp_butler_from_export
from ..exceptions import EmptyMosaicQueryError
from ..hips_builder import (
    HIPS_ORDER_MIN,
    HIPS_ORDER_MAX,
    build_hips_tile_from_temp_butler,
)
from ..hips_grid import hips_pixel_scale_arcsec, npix_for_coordinate
from ..temp_butler import (
    collect_inputs_for_temp_butler,
    collect_inputs_for_hips_tile,
)
from lsst.daf.butler import Butler, MissingDatasetTypeError
# Global variables for timing
_start_time = None
_script_name = "mosaic_tool"
CHANNEL_LIMITS = (0.75, 5)

# Starts a fresh python interpreter process
multiprocessing.set_start_method("spawn", force=True)


@click.group()
@click.option('--version', is_flag=True, is_eager=True,
              help='Show the version and exit.')
@click.pass_context
def cli(ctx, version):
    """Mosaic tool."""
    if version:

        click.echo(f"Mosaic Tool Version: {mosaic_tool.__version__}")
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

    return ra, dec


def validate_size(ctx, param, value):
    if not value:
        raise click.BadParameter("--size is required (format: xsize,ysize)")
    try:
        x_str, y_str = value.split(",")
        xsize, ysize = float(x_str), float(y_str)
    except ValueError:
        raise click.BadParameter(f"Invalid --size format: '{value}', expected xsize,ysize (floats)")
    if xsize > 5 or ysize > 5:
        raise click.BadParameter(f"Invalid size values: {xsize},{ysize}. Maximum size is 5 degrees.")
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
        raise click.BadParameter(f"Invalid --channel-range format: {value}, expected min,max floats.")

    if len(parts) != 2:
        raise click.BadParameter(f"--channel-range must have exactly 2 comma-separated values\
                                    (min,max). {parts}")
    val_1, val_2 = [float(x) for x in value.split(",")[:2]]
    lower_check = CHANNEL_LIMITS[0] <= val_1 <= CHANNEL_LIMITS[1]
    upper_check = CHANNEL_LIMITS[0] <= val_2 <= CHANNEL_LIMITS[1]
    if not (lower_check and upper_check):
        err_str = f"Invalid --channel-range format: {value} , all values must be between"
        err_str += f" {CHANNEL_LIMITS[0]} microns and {CHANNEL_LIMITS[1]} microns."
        raise click.BadParameter(err_str)

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
    vals_tuples = list(zip(vals[::2], vals[1::2]))

    if len(set(vals_tuples)) != len(vals_tuples):
        raise click.BadParameter(f"--user-channels: {vals_tuples}, must not be duplicate ranges")

    for val in vals:
        if not (CHANNEL_LIMITS[0] <= val <= CHANNEL_LIMITS[1]):
            err_str = f"--user-channels: {vals}  all values must be between"
            err_str += f" {CHANNEL_LIMITS[0]} microns and {CHANNEL_LIMITS[1]} microns."
            raise click.BadParameter(err_str)
    # Validate ordering inside each pair
    for i in range(0, len(vals), 2):
        if vals[i] >= vals[i + 1]:
            raise click.BadParameter(
                f"--user-channels: invalid pair ({vals[i]}, {vals[i+1]}). Each min must be < max."
            )

    return vals_tuples  # list of (min, max) tuples


def validate_channel_exclusivity(ctx, param, value):
    # Defer exclusivity check to after all parameters parsed
    channel_range = ctx.params.get("channel_range")
    user_channels = ctx.params.get("user_channels")

    if channel_range and user_channels:
        raise click.BadParameter("Options --channel-range and --user-channels are mutually exclusive.")
    return value


def validate_interpolate_wavelength(ctx, param, value):
    if value is None:
        return None
    if not (CHANNEL_LIMITS[0] <= value <= CHANNEL_LIMITS[1]):
        raise click.BadParameter(
            f"Invalid --interpolate-wavelength: {value} μm. "
            f"Must be between {CHANNEL_LIMITS[0]} and {CHANNEL_LIMITS[1]} μm."
        )
    return value


def validate_interp_n_width(ctx, param, value):
    if value <= 0:
        raise click.BadParameter("--interp-n-width must be positive.")
    if value > 10:
        raise click.BadParameter("--interp-n-width exceeds maximum value of 10.")
    return value


def validate_interp_n_req(ctx, param, value):
    if value <= 0:
        raise click.BadParameter("--interp-n-req must be positive.")
    if value > 5:
        raise click.BadParameter("--interp-n-req exceeds maximum value of 5.")
    return value


def validate_interp_n_subdivisions(ctx, param, value):
    if value < 1 or value > 7:
        raise click.BadParameter("--interp-n-subdivisions must be between 1 and 7.")
    return value


@click.command()
@result_dir_option()
@err_dir_option()
@runid_option("mosaic")
@jobid_option()
@tmp_dir_option()
@log_dir_option()
@fake_option()
@use_deep_option()
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
                    the result. Values are in microns and must be in range [0.75,5]
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
                        specified explicitly.Values must be in range [0.75,5].
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
@click.option(
    '--compress',
    is_flag=True,
    default=True,
    show_default=True,
    help="Compress output cube with gzip (produces .fits.gz instead of .fits)."
)
@click.option(
    '--interpolate-wavelength',
    type=float,
    default=None,
    callback=validate_interpolate_wavelength,
    help="""Target wavelength (μm) for spectral interpolation.
            Creates a single mosaic by interpolating across neighboring
            spectral channels to reduce stripe artifacts from extended
            sources with strong spectral features. This option is
            incompatible with --channel-range and --user-channels.
            The output is a single mosaic file, not a cube."""
)
@click.option(
    '--interp-n-width',
    type=float,
    default=1.0,
    show_default=True,
    callback=validate_interp_n_width,
    help="""Interpolation range in resolution widths (default 1.0).
            Channels within ±n×Δλ of the target wavelength are used.
            Larger values use more channels but increase spectral
            smoothing. Only used with --interpolate-wavelength."""
)
@click.option(
    '--interp-n-req',
    type=float,
    default=0.55,
    show_default=True,
    callback=validate_interp_n_req,
    help="""Required contribution distance in resolution widths (default 0.55).
            At least one channel must be within this distance of target
            for a pixel to be valid. Only used with --interpolate-wavelength."""
)
@click.option(
    '--interp-n-subdivisions',
    type=int,
    default=4,
    show_default=True,
    callback=validate_interp_n_subdivisions,
    help="""Number of subdivisions per standard channel (1-7, default 4).
            When > 1, creates finer spectral sampling by dividing each
            standard channel into n subchannels. Requires wavelength
            calibration files in the butler. Only used with
            --interpolate-wavelength."""
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
    compress: bool,
    interpolate_wavelength,
    interp_n_width: float,
    interp_n_req: float,
    interp_n_subdivisions: int,
    center,
    size,
    rotation,
    pixel_scale,
    channel_range,
    user_channels,
    time,
    detector,
    use_deep: bool
):
    """Create a mosaic image from SPHEREx data."""
    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
    error_symlink = pathlib.Path(os.path.join(err_dir, "error"))
    time = parse_time(time)
    detectors = parse_detectors(detector)
    x_size, y_size = size[0], size[1]
    size_scale_text = f"Size (x,y): {size} degrees, Pixel scale: {pixel_scale} arcsec"
    pixel_scale_deg = pixel_scale / 3600
    if not (x_size/pixel_scale_deg > 1):
        raise click.BadParameter(f"{size_scale_text}, x_size / --pixel-scale should be greater than 1")
    if not (y_size/pixel_scale_deg > 1):
        raise click.BadParameter(f"{size_scale_text}, y_size / --pixel-scale should be greater than 1")

    # Validate mutual exclusivity of interpolation mode with channel options
    if interpolate_wavelength is not None:
        if channel_range is not None:
            raise click.BadParameter(
                "Options --interpolate-wavelength and --channel-range are mutually exclusive."
            )
        if user_channels is not None:
            raise click.BadParameter(
                "Options --interpolate-wavelength and --user-channels are mutually exclusive."
            )

    logging.info("Mosaic configuration:")
    logging.info(f"  Center (ra,dec): {center}")
    logging.info(f"  Size (x,y): {size} degrees")
    logging.info(f"  Rotation: {rotation} degrees")
    logging.info(f"  Pixel scale: {pixel_scale} arcsec")
    if interpolate_wavelength is not None:
        logging.info(f"  Interpolate wavelength: {interpolate_wavelength} μm")
        logging.info(f"  Interpolation n_width: {interp_n_width}")
        logging.info(f"  Interpolation n_req: {interp_n_req}")
        logging.info(f"  Interpolation n_subdivisions: {interp_n_subdivisions}")
    else:
        logging.info(f"  Channel range: {channel_range}")
        logging.info(f"  User channels: {user_channels}")
    logging.info(f"  Time: {time}")
    logging.info(
        f"  Detector(s): {detectors}.(Not intended for to be exposed via UI, None indicates all detectors)")
    logging.info(f"  Verbose: {verbose}")
    logging.info(f"  Compress: {compress}")
    logging.info(f"  Fake mode: {fake}")
    logging.info(f"  Run ID: {runid}, Job ID: {jobid}")
    logging.info(f"  Result dir: {result_dir}")
    logging.info(f"  Temp dir: {tmp_dir}, Error dir: {err_dir}, Log dir: {log_dir}")

    if fake == "normal":
        src_path = pathlib.Path(os.getenv("SPXMOSAIC_SAMPLE_RESULT",
                                          "~/tmp-result/mosaic-tool-result.fits")).expanduser()
        hdul = fits.open(src_path)
        # Insert CRVAL1 and CRVAL2 into headers
        for i in range(1, len(hdul)-1):
            hdul[i].header["CRVAL1"] = center[0]
            hdul[i].header["CRVAL2"] = center[1]
        destination_dir = pathlib.Path(result_dir)
        dest_file = destination_dir / "mosaic-tool-result.fits"
        # Ensure the destination directory exists
        destination_dir.mkdir(parents=True, exist_ok=True)
        hdul.writeto(dest_file, overwrite=True)
        # === Create symlink to output VOTable ===
        symlink_path = destination_dir / "result"
        if symlink_path.exists() or symlink_path.is_symlink():
            symlink_path.unlink()

        symlink_path.symlink_to(dest_file)
        logging.info(f"Copied VOTable to {dest_file} and symlinked as {symlink_path}")
    elif fake in ["error-short", "error-detail"]:
        sample_polygon = "POLYGON 322.1172900 -2.4604856 325.6680542 -5.9804809"
        sample_polygon += " 322.1172900 -9.5233144 318.5665258 -5.9804809"
        exit_code = 13
        try:
            raise EmptyMosaicQueryError(mosaic_region=sample_polygon)
        except EmptyMosaicQueryError as mdte:
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
    elif fake is None:
        butler_repo = os.getenv("SPXTOOL_BUTLER")
        try:
            butler = Butler(butler_repo)
            logging.info(f"Butler {butler} exists")
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

        try:
            # Create pool for collect process
            with multiprocessing.Pool(processes=1) as pool:
                collect_result = pool.apply_async(
                    collect_inputs_for_temp_butler,
                    args=(butler_repo, tmp_dir, center, size,
                          rotation, time, detectors, pixel_scale_deg, use_deep)
                )
                # This will automatically propagate exceptions from the child process
                collect_results = collect_result.get()
            logging.info(f"Temp butler root: {collect_results['temp_butler_root']}")
            logging.info(f"export_file: {collect_results['export_file']}")
            logging.info(f"coverage_table:\n{collect_results['coverage_table']}")
            # Create pool for populate process
            with multiprocessing.Pool(processes=1) as pool:
                populate_result = pool.apply_async(
                    populate_temp_butler_from_export,
                    args=(
                        collect_results["temp_butler_root"],
                        collect_results["export_file"],
                        collect_results["coverage_table"]
                    )
                )
                # This will also automatically propagate exceptions
                populate_results = populate_result.get()

            logging.info(f'Temp butler root: {populate_results["temp_butler_root"]}')

            # ================================================================
            # MOSAIC BUILDING
            # ================================================================
            if interpolate_wavelength is not None:
                # Spectral interpolation mode - creates a single interpolated mosaic
                logging.info("Using spectral interpolation mode...")
                output_filepath = build_interpolated_mosaic(
                    temp_butler_root=populate_results["temp_butler_root"],
                    center=center,
                    size=size,
                    pixel_scale=pixel_scale,
                    rotation=rotation,
                    output_dir=result_dir,
                    target_wavelength=interpolate_wavelength,
                    n_interp=interp_n_width,
                    n_req=interp_n_req,
                    n_subdivisions=interp_n_subdivisions,
                    compress=compress,
                )
            else:
                # Standard mosaic building mode
                if user_channels is not None:
                    # TODO: user_channels not implemented in this version
                    logging.warning("--user-channels not implemented yet, using all standard channels")

                with tempfile.TemporaryDirectory(
                    dir=tmp_dir, prefix="channel_mosaics_"
                ) as channel_mosaics_dir:
                    output_filepath = build_mosaics_from_temp_butler(
                        temp_butler_root=populate_results["temp_butler_root"],
                        center=center,
                        size=size,
                        pixel_scale=pixel_scale,
                        rotation=rotation,
                        output_dir=result_dir,
                        channel_mosaics_dir=channel_mosaics_dir,
                        channel_range=channel_range,
                        compress=compress,
                        user_channels=user_channels,
                    )

            # Symlink creation happens after the context manager exits
            result_symlink = pathlib.Path(result_dir) / "result"
            if result_symlink.exists() or result_symlink.is_symlink():
                result_symlink.unlink()
            result_symlink.symlink_to(os.path.basename(output_filepath))
            logging.info(f"Result symlink created: {result_symlink}")
            logging.info("Mosaic building complete!")
        except MissingDatasetTypeError as mdte:
            logging.error(f"Error: {mdte}")
            exit_code = 12
            generate_error_summary(message=mdte, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal")
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)
        except EmptyMosaicQueryError as elqe:
            logging.error(f"Error: {elqe}")
            exit_code = 13
            generate_error_summary(message=elqe, error_dir=err_dir,
                                   write_to_file=True, error_type="fatal", user_error=True)
            if log_dir is not None:
                error_symlink.unlink(missing_ok=True)
                error_symlink.symlink_to(log_file)
            safe_exit(exit_code)


def validate_hips_tile(ctx, param, value):
    if value is None:
        return None
    try:
        order_str, npix_str = value.split(",")
        order, npix = int(order_str), int(npix_str)
    except ValueError:
        raise click.BadParameter(
            f"Invalid --hips-tile format: '{value}', expected order,npix (integers)"
        )
    if not (HIPS_ORDER_MIN <= order <= HIPS_ORDER_MAX):
        raise click.BadParameter(
            f"Invalid HiPS order {order}: must be in [{HIPS_ORDER_MIN}, {HIPS_ORDER_MAX}]."
        )
    npix_max = 12 * (1 << (2 * order))
    if not (0 <= npix < npix_max):
        raise click.BadParameter(
            f"Invalid HiPS npix {npix}: must be in [0, {npix_max}) for order {order}."
        )
    return (order, npix)


def validate_hips_order(ctx, param, value):
    if value is None:
        return None
    if not (HIPS_ORDER_MIN <= value <= HIPS_ORDER_MAX):
        raise click.BadParameter(
            f"Invalid HiPS order {value}: must be in [{HIPS_ORDER_MIN}, {HIPS_ORDER_MAX}]."
        )
    return value


def validate_optional_center(ctx, param, value):
    if not value:
        return None
    return validate_center(ctx, param, value)


def _default_hips_jobid(argv):
    """Derive the default HiPS job identifier, Norder{order}Npix{npix},
    from the tile-selection arguments in `argv`.

    Returns None if the tile cannot be determined from the arguments;
    proper validation errors are raised later by the click callbacks.
    """
    def extract(key):
        if key in argv:
            try:
                return argv[argv.index(key) + 1]
            except IndexError:
                return None
        for arg in argv:
            if arg.startswith(f"{key}="):
                return arg.split("=", 1)[1]
        return None

    try:
        tile = extract("--hips-tile")
        if tile is not None:
            order, npix = (int(v) for v in tile.split(","))
            return f"Norder{order}Npix{npix}"
        order = extract("--hips-order")
        center = extract("--center")
        if order is not None and center is not None:
            ra, dec = (float(v) for v in center.split(","))
            return f"Norder{int(order)}Npix{npix_for_coordinate(int(order), ra, dec)}"
    except ValueError:
        return None
    return None


@click.command()
@result_dir_option()
@err_dir_option()
@runid_option("hips")
@click.option(
    '--jobid',
    default=None,
    help=(
        "Allows the UWS system to pass in the opaque UWS 'Job Identifier' by which the job "
        "is referenceable in the REST API. Optional for HiPS tile jobs; defaults to "
        "Norder{order}Npix{npix} derived from the tile selection. This is not the same as "
        "the UWS 'Run ID' which is set by the user and passed to the tool via the --runid option."
    )
)
@tmp_dir_option()
@log_dir_option()
@use_deep_option()
@click.option("--hips-tile", callback=validate_hips_tile, default=None,
              help=f"""HiPS tile to build, as ORDER,NPIX. ORDER is the HiPS
                    tile order (HEALPix NSIDE = 2^ORDER), allowed range
                    [{HIPS_ORDER_MIN},{HIPS_ORDER_MAX}]; NPIX is the HEALPix
                    NESTED pixel index of the tile at that order. The tile
                    fully determines the output grid: 512x512 pixels in the
                    HPX projection, matching the HiPS 1.0 standard tiling
                    (~12.88 arcsec pixels at order 5). Incompatible with
                    --hips-order. Example: --hips-tile=5,1000.""")
@click.option("--hips-order", type=int, callback=validate_hips_order, default=None,
              help=f"""HiPS tile order, allowed range
                    [{HIPS_ORDER_MIN},{HIPS_ORDER_MAX}]. Must be combined with
                    --center: the tile containing that coordinate is built.
                    Incompatible with --hips-tile. Example: --hips-order=5.""")
@click.option("--center", callback=validate_optional_center, default=None,
              help="""Sky coordinate selecting the tile, used with
                    --hips-order. Values are in ICRS degrees, with ra in
                    [0, 360) and dec in [-90, 90]. Note that unlike
                    create-mosaic, the coordinate does not become the mosaic
                    center; it only identifies the HiPS tile containing it.
                    Example: --center=6.2001,-14.8314.""")
@click.option("--channel-range", callback=validate_channel_range, default=None,
              help="""Range of central wavelengths of standard SPHEREx
                    spectral channels to include in the result. Values are
                    in microns and must be in range [0.75,5]. Default is
                    the 102 standard channels.""")
@click.option("--time", default=None,
              help="""Time restriction. Values are in days, as MJD values,
                    following the IVOA DALI and SIAv2 standards. Restricts
                    the selection of Level 2 LVF images used as input.
                    The min value may be given as -Inf and the max value as
                    +Inf. Defaults to no restriction.""")
@click.option("--detector", multiple=True, type=str,
              help="""Detector label: D1, D2, D3, D4, D5, or D6. Restricts
                    inputs to come from the named detector(s). Defaults to
                    no restriction.""")
@click.option('--verbose/--no-verbose', default=False, show_default=True,
              help="Turns on additional verbosity in a job's logging.")
@click.option('--compress', is_flag=True, default=True, show_default=True,
              help="Compress output cube with gzip (produces .fits.gz instead of .fits).")
def create_hips_tile(
    result_dir,
    err_dir: str,
    runid: str,
    jobid: str,
    tmp_dir: str,
    log_dir: str,
    use_deep: bool,
    hips_tile,
    hips_order,
    center,
    channel_range,
    time,
    detector,
    verbose: bool,
    compress: bool
):
    """Create a HiPS tile spectral cube from SPHEREx data.

    Builds a mosaic cube whose spatial grid is one HiPS tile (HiPS 1.0
    standard: 512x512 pixels, HPX projection, NESTED tile numbering).
    The output cube is written under the HiPS directory convention
    Norder{order}/Dir{D}/Npix{npix}_cube.fits inside the result directory.
    """
    log_file = None
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            log_file = handler.baseFilename
            break
    error_symlink = pathlib.Path(os.path.join(err_dir, "error"))
    time = parse_time(time)
    detectors = parse_detectors(detector)

    # Resolve the tile selection
    if hips_tile is not None and hips_order is not None:
        raise click.BadParameter(
            "Options --hips-tile and --hips-order are mutually exclusive."
        )
    if hips_tile is None:
        if hips_order is None:
            raise click.BadParameter(
                "Either --hips-tile or --hips-order with --center is required."
            )
        if center is None:
            raise click.BadParameter(
                "--hips-order requires --center to select the tile."
            )
        hips_tile = (hips_order, npix_for_coordinate(hips_order, center[0], center[1]))
    elif center is not None:
        raise click.BadParameter("--center can only be used with --hips-order.")
    order, npix = hips_tile

    if jobid is None:
        # Must match the derivation in _default_hips_jobid(), which names
        # the log file before click parses the command line.
        jobid = f"Norder{order}Npix{npix}"

    logging.info("HiPS tile configuration:")
    logging.info(f"  Order: {order}, Npix: {npix}")
    logging.info(f"  Pixel scale: {hips_pixel_scale_arcsec(order):.2f} arcsec")
    logging.info(f"  Channel range: {channel_range}")
    logging.info(f"  Time: {time}")
    logging.info(f"  Detector(s): {detectors}")
    logging.info(f"  Verbose: {verbose}")
    logging.info(f"  Compress: {compress}")
    logging.info(f"  Run ID: {runid}, Job ID: {jobid}")
    logging.info(f"  Result dir: {result_dir}")
    logging.info(f"  Temp dir: {tmp_dir}, Error dir: {err_dir}, Log dir: {log_dir}")

    butler_repo = os.getenv("SPXTOOL_BUTLER")
    try:
        butler = Butler(butler_repo)
        logging.info(f"Butler {butler} exists")
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

    try:
        # Create pool for collect process
        with multiprocessing.Pool(processes=1) as pool:
            collect_result = pool.apply_async(
                collect_inputs_for_hips_tile,
                args=(butler_repo, tmp_dir, order, npix,
                      time, detectors, use_deep)
            )
            # This will automatically propagate exceptions from the child process
            collect_results = collect_result.get()
        logging.info(f"Temp butler root: {collect_results['temp_butler_root']}")
        logging.info(f"export_file: {collect_results['export_file']}")
        logging.info(f"coverage_table:\n{collect_results['coverage_table']}")
        # Create pool for populate process
        with multiprocessing.Pool(processes=1) as pool:
            populate_result = pool.apply_async(
                populate_temp_butler_from_export,
                args=(
                    collect_results["temp_butler_root"],
                    collect_results["export_file"],
                    collect_results["coverage_table"]
                )
            )
            # This will also automatically propagate exceptions
            populate_results = populate_result.get()

        logging.info(f'Temp butler root: {populate_results["temp_butler_root"]}')

        # ================================================================
        # HIPS TILE BUILDING
        # ================================================================
        with tempfile.TemporaryDirectory(
            dir=tmp_dir, prefix="channel_mosaics_"
        ) as channel_mosaics_dir:
            output_filepath = build_hips_tile_from_temp_butler(
                temp_butler_root=populate_results["temp_butler_root"],
                hips_order=order,
                hips_npix=npix,
                output_dir=result_dir,
                channel_mosaics_dir=channel_mosaics_dir,
                channel_range=channel_range,
                compress=compress,
            )

        # Symlink creation happens after the context manager exits.
        # The cube lives in the Norder/Dir hierarchy, so the symlink
        # target is a relative path, not just a basename.
        result_symlink = pathlib.Path(result_dir) / "result"
        if result_symlink.exists() or result_symlink.is_symlink():
            result_symlink.unlink()
        result_symlink.symlink_to(os.path.relpath(output_filepath, result_dir))
        logging.info(f"Result symlink created: {result_symlink}")
        logging.info("HiPS tile building complete!")
    except MissingDatasetTypeError as mdte:
        logging.error(f"Error: {mdte}")
        exit_code = 12
        generate_error_summary(message=mdte, error_dir=err_dir,
                               write_to_file=True, error_type="fatal")
        if log_dir is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(exit_code)
    except EmptyMosaicQueryError as elqe:
        logging.error(f"Error: {elqe}")
        exit_code = 13
        generate_error_summary(message=elqe, error_dir=err_dir,
                               write_to_file=True, error_type="fatal", user_error=True)
        if log_dir is not None:
            error_symlink.unlink(missing_ok=True)
            error_symlink.symlink_to(log_file)
        safe_exit(exit_code)


cli.add_command(create_mosaic)
cli.add_command(create_hips_tile)


def main():
    if "--help" in sys.argv or "--version" in sys.argv:
        cli(standalone_mode=False, obj={})
    else:
        # Setup exit handlers first, before any other code
        # except help or version calls
        setup_exit_handlers(_script_name)

        try:
            # Extract log dir and associated flags.
            log_dir, verbose, runid, jobid = extract_log_config_and_ids_from_argv()
            error_symlink = None
            if runid is None:
                runid = make_runid_default("mosaic")
                force_override_runid(new_value=runid)
            if jobid is None and "create-hips-tile" in sys.argv:
                # --jobid is optional for HiPS tile jobs; derive the same
                # default the command itself will use, so the log file
                # name carries the tile identification.
                jobid = _default_hips_jobid(sys.argv)
            err_dir = extract_err_dir_from_argv()
            log_level = logging.DEBUG if verbose else logging.INFO
            if log_dir is None:
                logging.basicConfig(
                    level=log_level,
                    stream=sys.stdout,
                    format='%(asctime)s %(levelname)s: %(message)s'
                )
            else:
                log_dir = validate_writable(None, None, Path(log_dir))
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

            message = f"Command line validation error: {e.message}"
            if e.param_hint is not None:
                message += f" (parameter: {e.param_hint})"
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
