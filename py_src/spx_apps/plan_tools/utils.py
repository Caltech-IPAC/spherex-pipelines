__all__ = ["make_plan_table"]

# Imports
# 3rd-party
from astropy.table import Table
from click import File


# Functions
def make_plan_table(plan_file: File, plan_format: str) -> Table:
    """Makes an astropy table from a plan file.

    Parameters
    ----------
    plan_file: click.File
        Input plan file
    plan_format: str
        astropy.io.ascii-supported format of input plan file

    Returns
    -------
    plan_tab: astropy.table.Table
        Table of plan file observations
    """
    return Table.read(plan_file.read(), format=plan_format)
