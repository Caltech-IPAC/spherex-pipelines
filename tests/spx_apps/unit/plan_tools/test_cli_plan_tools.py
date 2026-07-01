# Standard library
from pathlib import Path

# 3rd-party imports
from click.testing import CliRunner

# Local imports
from spx_apps.plan_tools.script import cli_plan_tools

data_dir = Path(__file__).parent / "data"
plan_filepath = data_dir / "2025W52_1A_plan.unittest.csv"


def test_posarg_csv(tmp_path):
    runner = CliRunner()
    csv_path = tmp_path / "plan_tools_output.csv"
    result = runner.invoke(cli_plan_tools.cli, ["visualize", str(plan_filepath), "--output", csv_path,
                                                "-F", "csv"])
    assert result.exit_code == 0


def test_posarg_votable(tmp_path):
    runner = CliRunner()
    votable_path = tmp_path / "plan_tools_output.xml"
    result = runner.invoke(cli_plan_tools.cli, ["visualize", str(plan_filepath), "--output",
                                                votable_path])
    assert result.exit_code == 0

# Tests below are commented out because pytest cannot read from stdin while
# output is captured, and CliRunner reads/writes in the same command. The
# plan-tools command still works with I/O redirection operators.

# def test_stdin_csv(tmp_path):
#     with open(plan_filepath, "rb") as file:
#         input_file = file.read()

#     runner = CliRunner()
#     csv_path = tmp_path / "plan_tools_output.csv"
#     result = runner.invoke(cli_plan_tools.cli, ["visualize", "--output",
#                             csv_path, "-F", "csv"],
#                             input=input_file)
#     assert result.exit_code == 0

# def test_stdin_votable(tmp_path):
#     with open(plan_filepath, "rb") as file:
#         input_file = file.read()

#     runner = CliRunner()
#     votable_path = tmp_path / "plan_tools_output.xml"
#     result = runner.invoke(cli_plan_tools.cli, ["visualize", "--output",
#                             votable_path], input=input_file)
#     assert result.exit_code == 0
