import os
import pytest
import tempfile
from unittest.mock import patch

from spx_utils.tempfile_utils import get_ssdc_tempfiledir, get_temp_path


def test_get_ssdc_tempfiledir_env_var_set_valid_dir():
    # Create a temporary directory and make it writable
    with tempfile.TemporaryDirectory() as temp_dir:
        with patch.dict("os.environ", {"SSDC_TEMPDIR": temp_dir}):
            assert get_ssdc_tempfiledir() == temp_dir


def test_get_ssdc_tempfiledir_env_var_set_invalid_dir():
    # Set an invalid directory path
    invalid_dir = "/invalid/path/to/dir"
    with patch.dict("os.environ", {"SSDC_TEMPDIR": invalid_dir}):
        with pytest.raises(ValueError):
            get_ssdc_tempfiledir()


def test_get_ssdc_tempfiledir_env_var_not_set():
    with patch("tempfile.gettempdir") as mock_gettempdir:
        mock_gettempdir.return_value = "/default/temp"
        # Test when environment variable is not set
        with patch.dict("os.environ", {}, clear=True):
            assert get_ssdc_tempfiledir() == mock_gettempdir.return_value


def test_get_temp_path_creates_file():
    prefix = "test_prefix_"
    suffix = "_test_suffix.txt"

    # Create a temp file and check its path
    temp_path = get_temp_path(prefix=prefix, suffix=suffix)
    assert os.path.isfile(temp_path)
    assert os.path.basename(temp_path).startswith(prefix)
    assert temp_path.endswith(suffix)

    # Clean up created temp file
    os.remove(temp_path)
