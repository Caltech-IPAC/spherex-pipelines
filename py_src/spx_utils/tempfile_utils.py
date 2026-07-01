__all__ = ["get_ssdc_tempfiledir", "get_temp_path"]
import os
import tempfile


def get_ssdc_tempfiledir(env_var_name="SSDC_TEMPDIR") -> str:
    """Get a directory for SSDC temporary files.

    The location of the temporary files is defined by an environment variable.
    If the environment variable is not set, the default temporary directory is used.

    Parameters
    ----------
    env_var_name : `str`

    Returns
    -------
    result : `str`
        The absolute path of the temporary files directory.
    """
    # Check if the environment variable is set
    temp_dir = os.getenv(env_var_name)

    if temp_dir:
        # Check if the specified path is a directory and is writable
        if os.path.isdir(temp_dir) and os.access(temp_dir, os.W_OK):
            return temp_dir
        else:
            raise ValueError(
                f"Environment variable '{env_var_name}' is set to '{temp_dir}',"
                f" but it is not a valid writable directory.")
    else:
        # Fall back to the default system temporary directory
        return tempfile.gettempdir()


ssdc_temp_dir = get_ssdc_tempfiledir()


def get_temp_path(prefix: str, suffix: str, dir=ssdc_temp_dir) -> str:
    """Create a temporary file in the directory designated for SSDC temporary files.

    Parameters
    ----------
    prefix : `str`
        Temporary file prefix.
    suffix : `str`
        Temporary file suffix.
    dir : `str`
        Directory in which to create the temporary file. Defaults to ``ssdc_temp_dir``,
        which resolves to the system temporary directory unless overridden by the
        ``SSDC_TEMPDIR`` environment variable.
        This parameter should not be modified unless explicitly required, as improper
        use may interfere with file management behavior.

    Returns
    -------
    result : `str`
        The absolute path of the created file.
    """
    with tempfile.NamedTemporaryFile(dir=dir, delete=False,
                                     prefix=prefix, suffix=suffix) as temp_file:
        return temp_file.name
