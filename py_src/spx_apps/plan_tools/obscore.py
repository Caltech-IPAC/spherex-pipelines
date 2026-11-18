__all__ = ["central_coords", "spatial_coverage"]

# Imports
# 3rd-party
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.time import Time
from healpy import Rotator
import astropy.units as u
import datetime
import numpy as np
# Local
from .params import FocalPlane


def central_coords(plan_tab: Table) -> SkyCoord:
    """Calculate central coordinates for plan file observations.

    Parameters
    ----------
    plan_tab : `~astropy.table.Table`
        Table of plan file observations

    Returns
    -------
    equatorial_coords: `~astropy.coordinates.SkyCoord`
        Equatorial right ascension & declination (degrees)

    """
    # Get times
    # (adapted from SPHEREx-Sky-Simulator/py/SPHEREx_ObsSimulator/tools.py)
    obs_times = []
    for time in plan_tab["time"]:
        obs_times.append(datetime.datetime.strptime(time, "%Y-%jT%H:%M:%S.%f"))
    obs_times = Time(obs_times, format="datetime", scale="utc")

    # Convert lat, lon, & PA from ecliptic to equatorial coordinates
    # (adapted from SPHEREx-Sky-Simulator/py/SPHEREx_ObsSimulator/tools.py)
    ecliptic_coords = SkyCoord(lon=plan_tab["TargetLon"]*u.degree,
                               lat=plan_tab["TargetLat"]*u.degree,
                               frame="geocentrictrueecliptic",
                               obstime=obs_times)
    equatorial_coords = ecliptic_coords.icrs

    return equatorial_coords


def spatial_coverage(plan_tab: Table, equatorial_coords: SkyCoord) -> list:
    """Calculate polygon string for plan file observations.

    Parameters
    ----------
    plan_tab : `~astropy.table.Table`
        Table of plan file observations.
    equatorial_coords : `~astropy.coordinates.SkyCoord`
        Equatorial coordinates for each observation.

    Returns
    -------
    polygon_str: `list`
        IVOA Polygon string for each observation

    """
    # Convert position angles
    ecliptic_pa = plan_tab['PositionAngle']
    r = Rotator(coord=["E", "C"], deg=True)  # ecliptic to celestial/equatorial
    equatorial_pa = ((-1 * ecliptic_pa) - np.rad2deg(r.angle_ref(plan_tab["TargetLon"], plan_tab["TargetLat"],
                                                                 lonlat=True, deg=True))) * u.deg

    polygon_str = []
    # Calculate vertices for each pointing
    for ptg_coords, ptg_pa in zip(equatorial_coords, equatorial_pa):
        # Go CW (upper right, lower right, lower left, upper left vertices)
        vtx_1 = ptg_coords.directional_offset_by((ptg_pa + FocalPlane.upper_right_pa), FocalPlane.separation)
        vtx_2 = ptg_coords.directional_offset_by((ptg_pa + FocalPlane.lower_right_pa), FocalPlane.separation)
        vtx_3 = ptg_coords.directional_offset_by((ptg_pa + FocalPlane.lower_left_pa), FocalPlane.separation)
        vtx_4 = ptg_coords.directional_offset_by((ptg_pa + FocalPlane.upper_left_pa), FocalPlane.separation)
        polygon_str.append(f"POLYGON {vtx_1.ra.value:.5f} {vtx_1.dec.value:.5f} "
                           + f"{vtx_2.ra.value:.5f} {vtx_2.dec.value:.5f} "
                           + f"{vtx_3.ra.value:.5f} {vtx_3.dec.value:.5f} "
                           + f"{vtx_4.ra.value:.5f} {vtx_4.dec.value:.5f}")

    return polygon_str
