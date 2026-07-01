from datetime import datetime
import click
from pathlib import Path
import os
# Function for runid handling when not provided.


def make_runid_default(tool_name):
    timestamp = datetime.now().isoformat(timespec="seconds").replace(":", "").replace("-", "")
    return f"{tool_name}-{timestamp}"

# Reusable runid option decorator with parameterized tool name


def runid_option(tool_name):
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
    return click.option(
        '--err-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory in which an error document for the job, if any, will"
            "be deposited by the tool. Defaults to “.”, the current working"
            "directory."
        )
    )

# Reusable --tmp-dir option with writable check


def tmp_dir_option():
    return click.option(
        '--tmp-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=Path("."),
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory available for use as temporary scratch space for the"
            "job. Defaults to “.”, the current working directory. When exe-"
            "cuting in the UWS system, tmp-dir should never be the same"
            "as result-dir, unless it's deliberately intended for temporary"
            "files to be returned to the user as results."
        )
    )


def log_dir_option():
    return click.option(
        '--log-dir',
        type=click.Path(file_okay=False, path_type=Path),
        default=None,
        show_default=True,
        callback=validate_writable,
        help=(
            "Directory into which any log files from the job will be deposited."
            "Defaults to “.”, the current working directory. When execut-"
            "ing in the UWS system, log-dir should never be the same as"
            "result-dir, unless it's deliberately intended for any logs to be"
            "returned to the user as results."
        )
    )
