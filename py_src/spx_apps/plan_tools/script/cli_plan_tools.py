__all__ = ["cli"]

# Imports
# Standard library
import sys
# 3rd-party
import click
# Local
from ..obscore import central_coords, spatial_coverage
from ..utils import make_plan_table
from lsst.daf.butler.cli.butler import LoaderCLI
from lsst.daf.butler.cli.opt import log_file_option, \
    log_label_option, log_level_option, log_tty_option, \
    long_log_option


class PlantoolsCLI(LoaderCLI):
    localCmdPkg = "spx_apps.plan_tools.script"


@click.command(cls=PlantoolsCLI, context_settings=dict(help_option_names=["-h", "--help"]))
@log_level_option()
@long_log_option()
@log_file_option()
@log_tty_option()
@log_label_option()
def cli(log_level: str, long_log: bool, log_file: str, log_tty: bool, log_label: str) -> None:
    """Plan tools."""
    pass


@click.command()
@click.argument("input", type=click.File("r"), default=sys.stdin)
@click.option("-o", "--output", required=True, type=str, help="Output filename")
@click.option("-F", "--format", default="votable", type=click.Choice(["votable", "csv"]),
              help="Output file format", show_default=True)
def visualize(input, output, format):
    """Add IVOA ObsCore/ObsLocTAP columns to table to facilitate visualization.

    INPUT: Input filepath
    """
    # Read planning file; select relevant records
    input_tab = make_plan_table(input, "ascii.csv")
    output_tab = input_tab[input_tab["EngFlag"] == "arrive_to_sci"]

    # Add columns for central coordinates
    eq_coords = central_coords(output_tab)
    output_tab.add_column(eq_coords.ra, name="s_ra", index=0)
    output_tab.add_column(eq_coords.dec, name="s_dec", index=1)

    # Add column for spatial coverage
    poly_str = spatial_coverage(output_tab, eq_coords)
    output_tab.add_column(poly_str, name="s_region")

    # Generate output file in appropriate format
    output_tab.write(output, format=format)

    return
