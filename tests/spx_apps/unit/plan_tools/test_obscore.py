# Standard library
from pathlib import Path

# 3rd-party imports
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

# Local imports
from spx_apps.plan_tools import obscore

data_dir = Path(__file__).parent / "data"
sci_tab_filepath = data_dir / "2025W52_1A_plan-arrive_to_sci.unittest.dat"
sci_tab = Table.read(sci_tab_filepath, format="ascii")

coords_tab_filepath = data_dir / "2025W52_1A_plan-arrive_to_sci-coords.unittest.dat"
coords_tab = Table.read(coords_tab_filepath, format="ascii")
eq_coords = SkyCoord.guess_from_table(coords_tab, unit=u.degree)


def test_central_coords():
    test_coords = obscore.central_coords(sci_tab)
    assert isinstance(test_coords, SkyCoord)


def test_spatial_coverage():
    test_poly_str = obscore.spatial_coverage(sci_tab, eq_coords)
    assert isinstance(test_poly_str, list)
