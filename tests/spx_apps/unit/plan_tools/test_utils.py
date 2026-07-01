# Standard library
from pathlib import Path

# 3rd-party
from astropy.table import Table

# Local
from spx_apps.plan_tools import utils

data_dir = Path(__file__).parent / "data"
plan_filepath = data_dir / "2025W52_1A_plan.unittest.csv"


def test_plan_table():
    with open(plan_filepath, "r") as file:
        plan_table = utils.make_plan_table(file, plan_format="csv")

    assert isinstance(plan_table, Table)
