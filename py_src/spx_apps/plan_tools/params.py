# Imports
# Standard library
from dataclasses import dataclass
# 3rd-party
from astropy.coordinates import Angle, SkyCoord
from astropy.units import Quantity
import astropy.units as u

# Calculate coordinates of the rectangle bounding the focal plane
# FoV from https://spherex.caltech.edu/page/instrument (03/22/2024)
# (FUTURE MOD: use values from system parameters "database" repo)
fov_w = 11.3*u.deg
fov_h = 3.5*u.deg
# For a pointing at (0 deg, 0 deg) in eq. coords
origin = SkyCoord(ra=0*u.deg, dec=0*u.deg, frame="icrs")
upper_right = origin.spherical_offsets_by(d_lon=fov_w/2.0, d_lat=fov_h/2.0)
lower_right = origin.spherical_offsets_by(d_lon=fov_w/2.0, d_lat=-fov_h/2.0)
lower_left = origin.spherical_offsets_by(d_lon=-fov_w/2.0, d_lat=-fov_h/2.0)
upper_left = origin.spherical_offsets_by(d_lon=-fov_w/2.0, d_lat=fov_h/2.0)


@dataclass(frozen=True)
class FocalPlane:
    """Stores position angles and separations to the focal plane vertices.

    Attributes (class)
    ------------------
    upper_right_pa: Angle | Quantity
        Position angle to upper right vertex, from a [0,0] ICRS origin (deg)
    lower_right_pa: Angle | Quantity
        Position angle to lower right vertex, from a [0,0] ICRS origin (deg)
    lower_left_pa: Angle | Quantity
        Position angle to lower left vertex, from a [0,0] ICRS origin (deg)
    upper_left_pa: Angle | Quantity
        Position angle to upper left vertex, from a [0,0] ICRS origin (deg)
    separation: Angle | Quantity
        Angular separation to each vertex (deg)
    """
    upper_right_pa: Angle | Quantity = origin.position_angle(upper_right).to(u.deg)
    lower_right_pa: Angle | Quantity = origin.position_angle(lower_right).to(u.deg)
    lower_left_pa: Angle | Quantity = origin.position_angle(lower_left).to(u.deg)
    upper_left_pa: Angle | Quantity = origin.position_angle(upper_left).to(u.deg)
    separation: Angle | Quantity = origin.separation(upper_left).to(u.deg)
