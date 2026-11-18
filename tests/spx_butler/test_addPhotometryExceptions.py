# Standard library imports
from unittest.mock import MagicMock, patch

# Related third-party imports
from astropy.table import Table
from click.testing import CliRunner
from lsst.daf.butler import Butler, EmptyQueryResultError
import pytest

# Local application/library specific imports
from spx_butler.cli.cmd.commands import add_photometry_exceptions
from spx_butler.script import addPhotometryExceptions

# Helper functions ============================================================
def mock_exc_ref(exposure: int):
    """Create fake HiPS tile dataset references."""
    ref = MagicMock()
    ref.dataId = MagicMock()
    ref.dataId.mapping = {"instrument": "spx", "exposure": exposure}
    ref.run = "test_run"
    return ref

def make_exc_tab():
    return Table({
        "hips6": [123, 123, 123, 456],
        "SPHERExRefID": [1, 2, 3, 4],
    })

# Fixtures ====================================================================
@pytest.fixture()
def patched_butler():
    mock_butler = MagicMock(spec=Butler)
    with patch("spx_butler.script.addPhotometryExceptions.Butler", return_value=mock_butler):
        yield mock_butler

# Test cases ==================================================================
@patch("spx_butler.script.addPhotometryExceptions")
@pytest.mark.parametrize("expected_args, use_defaults", [
    (["test_type", "test_where", 1], False),  # Use specific values
    (["phot_exceptions", "", -20000], True)  # Use defaults
])
def test_add_photometry_exceptions(mock_add_exceptions: MagicMock, expected_args: list, use_defaults: bool):
    """Test the add photometry exceptions CLI command."""
    runner = CliRunner()
    runner_args = ["test_repo", "test_collection"]
    if not use_defaults:
        runner_args.extend(["--in-type", expected_args[0], "--where", expected_args[1], "--limit",
                            expected_args[2]])

    result = runner.invoke(add_photometry_exceptions, runner_args)
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    mock_add_exceptions.assert_called_once_with(
        "test_repo",
        ("test_collection",),
        expected_args[0],
        expected_args[1],
        expected_args[2]
    )

@pytest.mark.parametrize("query_case", ["empty", None, "non-empty"])
def test_addPhotometryExceptions(query_case: str, patched_butler):
    """Test that datasets are queried and exceptions are added."""
    # Mock return values for different cases
    if query_case == "empty":
        patched_butler.query_datasets.side_effect = EmptyQueryResultError("Empty query case")
    elif query_case == "non-empty" or query_case is None:
        patched_butler.query_datasets.return_value = [mock_exc_ref(202517101000126)]
        sample_exception = make_exc_tab() if query_case == "non-empty" else None
        patched_butler.get.return_value = sample_exception

    # Verify expected calls
    addPhotometryExceptions("test_repo", "test_collection")
    if query_case == "non-empty":
        put_calls = patched_butler.put.call_args_list
        tiles = {c.kwargs['dataId']['hips6'] for c in put_calls}
        assert len(put_calls) == 2
        assert tiles == {123, 456}
        assert all(c.kwargs['run'] == "test_run" for c in put_calls)
    else:
        patched_butler.put.assert_not_called()  # No `phot_meas` datasets added
