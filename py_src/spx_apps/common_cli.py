from pathlib import Path
import os
from collections import Counter
import sys
import atexit
import logging
import signal
import re
import time
import yaml

from astropy.time import Time
from datetime import datetime
import click
from .resource_usage import log_resource_usage
# Function for runid handling when not provided.

_start_time = None


def make_runid_default(tool_name):
    """Generate a default run ID using the tool name and current timestamp."""
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "").replace("-", "")
    return f"{tool_name}-{timestamp}"


def use_deep_option():
    """Create a click option for toggling inclusion of deep-field data."""
    return click.option(
        '--use-deep/--no-use-deep',
        default=False,
        help=(
            "Controls whether deep-field data are included in tool functions. "
            "Use --use-deep to enable inclusion of deep-field data, or --no-use-deep to exclude it. "
            "--no-use-deep is equivalent to selecting only all sky data."
            "Disabled by default; deep-field data are excluded unless explicitly requested."
        )
    )

def fake_option():
    """Create a click option for running simulated jobs."""
    return click.option(
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

# Reusable runid option decorator with parameterized tool name


def runid_option(tool_name):
    """Create a click option for specifying a UWS run ID."""
    return click.option(
        '--runid',
        default=make_runid_default(tool_name),
        show_default=False,
        help=(
            "Allows specifying a UWS 'runid', essentially a job name, which may be used in "
            "log messages or other metadata. Defaults to a short string indicating the specific "
            f"science tool being run (e.g., '{tool_name}') followed by an ISO-8601 timestamp, "
            f"e.g., '{tool_name}-20250307T172132'. This is not the same as the UWS 'Job Identifier' "
            "passed via --jobid.\n [optional]"
        )
    )

# Reusable --jobid option decorator (required, no default)


def jobid_option():
    """Create a click option for specifying the UWS job identifier."""
    return click.option(
        '--jobid',
        required=True,
        help=(
            "Allows the UWS system to pass in the opaque UWS 'Job Identifier' by which the job "
            "is referenceable in the REST API. Required; no default is supported. This is not the same as "
            "the UWS 'Run ID' which is set by the user and passed to the tool via the --runid option."
        )
    )


def validate_writable(ctx, param, value):
    """Validate that a directory path exists and is writable."""
    if value is not None:
        # Ensure the path exists and is a directory
        if not value.exists():
            raise click.BadParameter(f"Directory '{value}' does not exist.")
        if not value.is_dir():
            raise click.BadParameter(f"'{value}' is not a directory.")
        # Check if directory is writable
        if not os.access(value, os.W_OK):
            raise click.BadParameter(f"Directory '{value}' is not writable.")
    return value

# Reusable --result-dir option with writable check


def result_dir_option():
    """Create a click option for specifying the result directory."""
    return click.option(
        '--result-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory in which all the intended UWS results of the job will be deposited. "
            "Defaults to '.', the current working directory. The tool will create a symlink "
            "'result' in that directory pointing to the 'principal' result"
        )
    )


# Reusable --err-dir option with writable check
def err_dir_option():
    """Create a click option for specifying the error directory."""
    return click.option(
        '--err-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory in which an error document for the job, if any, will "
            "be deposited by the tool. Defaults to “.”, the current working "
            "directory."
        )
    )

# Reusable --tmp-dir option with writable check


def tmp_dir_option():
    """Create a click option for specifying the temporary directory."""
    return click.option(
        '--tmp-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory available for use as temporary scratch space for the "
            "job. Defaults to “.”, the current working directory. When "
            "executing in the UWS system, tmp-dir should never be the same "
            "as result-dir, unless it's deliberately intended for temporary "
            "files to be returned to the user as results."
        )
    )


def log_dir_option():
    """Create a click option for specifying the log directory."""
    return click.option(
        '--log-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory into which any log files from the job will be deposited. "
            "Defaults to “.”, the current working directory. When "
            "executing in the UWS system, log-dir should never be the same as "
            "result-dir, unless it's deliberately intended for any logs to be "
            "returned to the user as results."
        )
    )


def parse_time(value) -> tuple[Time, Time] | None:
    """Parse --time option into a tuple of astropy.time.Time objects using MJD.

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


def parse_detectors(value) -> list[int] | None:
    """Parse multiple --detector options into a list of detector IDs.

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


def log_start_time():
    """Log the script start time."""
    global _start_time
    _start_time = time.time()
    start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    logging.info(f"=== {_script} STARTED at {start_datetime} ===")
    return _start_time


def log_end_time(exit_code=None):
    """Log the script end time and duration."""
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
    logging.info(f"FINAL LOG ENTRY - {_script} MAIN PROCESS ENDED")
    logging.info(f"End time: {end_datetime}")
    logging.info(f"Duration: {duration_str}{exit_info}")
    log_resource_usage()
    logging.info("NOTE: Subprocess logs may appear after this message due to buffering")
    logging.info("=" * 80)


def setup_exit_handlers(script_name):
    """Set up handlers to ensure end time is logged regardless of how script exits."""
    atexit.register(log_end_time)

    global _script
    _script = script_name

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
    """Exit with logging."""
    atexit.unregister(log_end_time)  # Prevent duplicate call
    log_end_time(exit_code=code)
    sys.exit(code)


def extract_log_config_and_ids_from_argv(default_runid=None, default_jobid=None):
    """Extract logging configuration and IDs from command-line arguments."""
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
    """Override the runid in sys.argv with a new value."""
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


def extract_err_dir_from_argv(default=os.getcwd()):
    """Extract the error directory path from command-line arguments."""
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


def generate_error_summary(
        message="", error_dir=None, write_to_file=False, error_type="fatal", user_error=False):
    """Generate a JSON-style error summary based on an exit code.

    Parameters
    ----------
    message : str, optional
        A descriptive error message. Defaults to an empty string.
    error_dir : str, optional
        Directory to write the errorSummary.yaml file. Defaults to None.
    write_to_file : bool, optional
        Whether to write the error summary to a file. Defaults to False.
    error_type : str, optional
        Type of error (e.g., "fatal", "warning"). Defaults to "fatal".
    user_error : bool, optional
        Whether the error is a user error. Defaults to False.

    Returns
    -------
    dict
        The error summary structure.

    """
    if user_error:
        final_message = str(message)
    else:
        final_message = f"Internal error; contact IRSA user support ({(message)})"
    # Construct error summary structure
    error_summary = {
        "errorType": error_type,
        "message": final_message
    }

    # Optionally write to file
    if write_to_file and error_dir:
        os.makedirs(error_dir, exist_ok=True)
        error_path = os.path.join(error_dir, "errorSummary.yaml")
        with open(error_path, 'w') as f:
            yaml.dump(error_summary, f)

    return error_summary


