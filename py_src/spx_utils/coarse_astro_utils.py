__all__ = ["get_ra_dec_lon_lat_from_quat", "pc_to_cd", "cd_to_pc",
           "get_initial_linear_wcs", "get_initial_wcs_with_sip",
           "position_angle_from_cd", "calculate_new_crval",
           "calculate_detector_pointings_with_pa", "get_corrected_linear_wcs",
           "compute_side_pas", "cd_from_wcs", "north_aligned_pa_from_wcs",
           "build_wcs_triplet_from_boresight"]

# Standard library imports
import logging
import copy

# Related third party imports
from astropy import coordinates as coord
from astropy.coordinates import CartesianRepresentation, SkyCoord, SkyOffsetFrame, Angle
from astropy.modeling import models
from astropy import wcs
from astropy.wcs import WCS
from astropy.table import Table
import gwcs
import numpy as np
from scipy.spatial.transform import Rotation
from astropy import units as u
from scipy.linalg import solve

# Local application/library specific imports
from spx_utils.initial_distortion_model import get_initial_distortion_model
from spx_data.observation_id import ObservationId
from spx_data import PlannedObservation


def get_ra_dec_lon_lat_from_quat(Q: np.array) -> tuple[float, float, float]:
    """Calculate RA, Dec, and PA from a quaternion."""
    xPLD = np.array([1, 0, 0])
    zPLD = np.array([0, 0, 1])
    rotation = Rotation.from_quat(Q)
    zLOS_boresight = rotation.apply(zPLD)
    skycoord = SkyCoord(CartesianRepresentation(x=zLOS_boresight[0]*u.km,
                                                y=zLOS_boresight[1]*u.km,
                                                z=zLOS_boresight[2]*u.km))
    ra_eq, dec_eq = skycoord.icrs.ra.deg, skycoord.icrs.dec.deg
    xLOS_boresight = rotation.apply(xPLD)
    xLOS_eq_pa0 = np.array([-np.sin(np.radians(ra_eq)), np.cos(np.radians(ra_eq)), 0.0])
    cos_PA_eq = np.clip(np.dot(xLOS_eq_pa0, xLOS_boresight) / (np.linalg.norm(xLOS_eq_pa0)
                                                               * np.linalg.norm(xLOS_boresight)), -1, 1)
    unsigned_PA_eq = np.arccos(cos_PA_eq)
    cross_prod_eq = np.cross(xLOS_eq_pa0, xLOS_boresight)
    sign_of_PA_eq = np.sign(np.dot(zLOS_boresight, cross_prod_eq))
    pa_eq = sign_of_PA_eq * np.rad2deg(unsigned_PA_eq)
    return ra_eq, dec_eq, (360. - pa_eq)


def get_initial_linear_wcs(ra_boresight, dec_boresight, array, roll=None, nx=2040, ny=2040):
    """Generate a WCS for a detector based on the main boresight pointing."""
    w = wcs.WCS(naxis=2)
    w.wcs.crval = [ra_boresight, dec_boresight]
    gap_size_pixels = (22.4 * 60.0 / 6.15 + 2 * 4)
    if array in [2, 5]:
        w.wcs.crpix = [nx / 2 + 0.5, ny / 2 + 0.5]
    elif array in [3, 6]:
        w.wcs.crpix = [3 * nx / 2 + gap_size_pixels + 0.5, ny / 2 + 0.5]
    elif array in [1, 4]:
        w.wcs.crpix = [-nx / 2 - gap_size_pixels + 0.5, ny / 2 + 0.5]
    w.wcs.cdelt = [6.15 / 3600.0, 6.15 / 3600.0]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    if roll is not None:
        theta = np.deg2rad(roll)
        w.wcs.pc = [[-np.cos(theta), np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    else:
        w.wcs.pc = [[-1.0, 0.0], [0.0, 1.0]]
    return w


def calculate_new_crval(original_wcs, dx_pixel, dy_pixel):
    """Calculate a new CRVAL based on a pixel offset from the CRPIX."""
    start_pixel = original_wcs.wcs.crpix
    offset_pixel_location = [start_pixel[0] + dx_pixel, start_pixel[1] + dy_pixel]
    new_crval_coord = original_wcs.pixel_to_world(*offset_pixel_location)
    return (new_crval_coord.ra.degree, new_crval_coord.dec.degree)


def calculate_detector_pointings_with_pa(pointing_d2, pointing_d5):
    """Calculate geometric pointings for side detectors
    using corrected center detectors.
    """
    ra_d2, dec_d2, pa_d2 = pointing_d2
    ra_d5, dec_d5, pa_d5 = pointing_d5
    offsets = {1: (3.87, 0.04), 3: (-3.8671964, 0.065),
               4: (3.871964, 0.042), 6: (-3.871964, 0.07)}

    def _calculate_offset(ra_c, dec_c, pa_c, dx, dy):
        ra_c_rad, dec_c_rad, pa_c_rad, dx_rad, dy_rad = np.deg2rad([ra_c, dec_c, pa_c, dx, dy])

        # Central pointing unit vector
        v_c = np.array([np.cos(ra_c_rad)*np.cos(dec_c_rad),
                        np.sin(ra_c_rad)*np.cos(dec_c_rad),
                        np.sin(dec_c_rad)])

        # Local north and east at central location
        v_n_c = np.array([-np.sin(dec_c_rad)*np.cos(ra_c_rad),
                          -np.sin(dec_c_rad)*np.sin(ra_c_rad),
                          np.cos(dec_c_rad)])
        v_e_c = np.array([-np.sin(ra_c_rad), np.cos(ra_c_rad), 0])

        # Pixel axis directions at central detector
        v_y_c = v_n_c * np.cos(pa_c_rad) + v_e_c * np.sin(pa_c_rad)
        v_x_c = np.cross(v_y_c, v_c)

        # Apply offset
        v_s = v_c + dx_rad * v_x_c + dy_rad * v_y_c
        v_s /= np.linalg.norm(v_s)

        # Convert to RA/Dec
        dec_s_rad = np.arcsin(v_s[2])
        ra_s_rad = np.arctan2(v_s[1], v_s[0])

        # compute correct PA for side detector
        # compute local north/east at side detector sky position
        ra_s, dec_s = np.rad2deg(ra_s_rad) % 360.0, np.rad2deg(dec_s_rad)
        ra_s_r, dec_s_r = ra_s_rad, dec_s_rad

        v_n_s = np.array([-np.sin(dec_s_r)*np.cos(ra_s_r),
                          -np.sin(dec_s_r)*np.sin(ra_s_r),
                          np.cos(dec_s_r)])
        v_e_s = np.array([-np.sin(ra_s_r), np.cos(ra_s_r), 0])

        # express the central +Y axis in the side detector’s tangent basis
        pa_s = np.arctan2(np.dot(v_y_c, v_e_s),
                          np.dot(v_y_c, v_n_s))

        pa_s_deg = np.rad2deg(pa_s) % 360.0

        return (ra_s, dec_s, pa_s_deg)

    return {
        1: _calculate_offset(ra_d2, dec_d2, pa_d2, *offsets[1]),
        2: pointing_d2,
        3: _calculate_offset(ra_d2, dec_d2, pa_d2, *offsets[3]),
        4: _calculate_offset(ra_d5, dec_d5, pa_d5, *offsets[4]),
        5: pointing_d5,
        6: _calculate_offset(ra_d5, dec_d5, pa_d5, *offsets[6]),
    }


def build_detector_wcs_with_pa(pointing_d2, pointing_d5,
                               crpix=(1020.5, 1020.5)):
    """Build a WCS for each of the 6 detectors from corrected D2/D5 pointings.

    Sets CRVAL from calculate_detector_pointings_with_pa, PA encoded in CD/PC
    via get_initial_linear_wcs, and CRPIX fixed at `crpix`.
    """
    detector_pointings = calculate_detector_pointings_with_pa(pointing_d2, pointing_d5)
    wcs_by_det = {}

    for det_num, (ra, dec, pa) in detector_pointings.items():
        w = get_initial_linear_wcs(
            ra_boresight=ra,
            dec_boresight=dec,
            array=det_num,
            roll=pa
        )
        # Lock the reference pixel at the detector center
        w.wcs.crpix = list(crpix)
        # (Optional) enforce CRVAL numerically
        # w.wcs.crval = [ra, dec]

        wcs_by_det[det_num] = w

    return wcs_by_det


def get_corrected_linear_wcs(planned_obs: PlannedObservation,
                             obs_from_sps: ObservationId,
                             array_id: int,
                             crpix=(1020.5, 1020.5)):
    """Construct a corrected linear WCS for a given array using planned obs."""
    roll_correction = 0.76195

    center_shifts_d2_dx = {1: 23.4740, 2: 21.8969, 3: 21.0275, 4: 20.4142}
    center_shifts_d2_dy = {1: -1.4856, 2: -1.1970, 3: -0.9019, 4: -0.5564}
    center_shifts_d5_dx = {1: 21.3121, 2: 19.7378, 3: 18.8781, 4: 18.2573}
    center_shifts_d5_dy = {1: 8.5014, 2: 8.7837, 3: 9.1026, 4: 9.4051}

    # Get boresight planned pointing from quaternion
    quat_dict = planned_obs.quaternions
    quat = np.array([
        quat_dict["q1"],
        quat_dict["q2"],
        quat_dict["q3"],
        quat_dict["q4"]
    ], dtype=np.float64)
    planned_ra_d2, planned_dec_d2, planned_pa_d2 = get_ra_dec_lon_lat_from_quat(quat)

    # Get Slew Group Key (small slew counter)
    slew_group_key = int(obs_from_sps.small_slew_counter)

    # Apply universal roll correction
    corrected_pa = (planned_pa_d2 + roll_correction) % 360

    if slew_group_key in center_shifts_d2_dx:
        # Get shifts for center detectors from the four new configs
        dx_d2 = center_shifts_d2_dx[slew_group_key]
        dy_d2 = center_shifts_d2_dy[slew_group_key]
        dx_d5 = center_shifts_d5_dx[slew_group_key]
        dy_d5 = center_shifts_d5_dy[slew_group_key]
    else:
        # Slew groups beyond 1-4 arise only in IOC planning periods, whose data
        # serves solely for persistence bootstrapping. As these corrections have no
        # bearing on persistence, a zero shift is applied.
        logging.warning(f"No hard-coded shifts found for Slew {slew_group_key}. Setting all to 0.0")
        dx_d2 = dy_d2 = dx_d5 = dy_d5 = 0.0

    # Create initial WCS templates for D2 and D5
    # (Using the planned RA/Dec and the *corrected* PA)
    wcs_d2_template = get_initial_linear_wcs(planned_ra_d2, planned_dec_d2, 2, corrected_pa)
    wcs_d5_template = get_initial_linear_wcs(planned_ra_d2, planned_dec_d2, 5, corrected_pa)

    # Calculate new CRVALs for D2 and D5 by applying pixel shifts
    crval_d2 = calculate_new_crval(wcs_d2_template, dx_d2, dy_d2)
    crval_d5 = calculate_new_crval(wcs_d5_template, dx_d5, dy_d5)

    # Create the final corrected pointings for D2 and D5
    corrected_pointing_d2 = (*crval_d2, corrected_pa)
    corrected_pointing_d5 = (*crval_d5, corrected_pa)

    # Derive pointings for ALL 6 detectors from the corrected D2/D5
    all_6_pointings = build_detector_wcs_with_pa(
        corrected_pointing_d2, corrected_pointing_d5, crpix
    )

    # Get the final, correct pointing for the *current* array_id
    final_corrected_wcs = all_6_pointings[array_id]

    return final_corrected_wcs


def get_initial_wcs_with_sip(w: WCS,
                             initial_distortion_table: Table,
                             array: int,
                             nx: int = 2040, ny: int = 2040,
                             oversample: int = 1,
                             edge_pixels: int = 4,
                             projection="TAN"):
    """Determine an initial WCS with SIP distortion correction.

    Parameters
    ----------
    w : `~astropy.wcs.WCS`
        Initial WCS object without SIP distortion correction.
    initial_distortion_table : `~astropy.table.Table`
        Table with initial distortion model data.
    array : `int`
        Array/detector ID (1-6).
    nx : `int`, optional
        Number of pixels in x direction. Default is 2040.
    ny : `int`, optional
        Number of pixels in y direction. Default is 2040.
    oversample : `int`, optional
        Oversampling factor. Default is 1.
    edge_pixels : `int`, optional
        Number of edge pixels to exclude. Default is 4.
    projection : `str`, optional
        WCS projection type. Default is "TAN".

    Returns
    -------
    `~astropy.wcs.WCS`
        WCS object with SIP distortion correction applied.

    Notes
    -----
    This function is a partial replication of `get_spherex_wcs`:
    https://github.com/SPHEREx/SPHEREx-Sky-Simulator/blob/main/py/SPHEREx_InstrumentSimulator/tools.py
    Where the initial distortion model is taken from:
    https://github.com/SPHEREx/SPHEREx-Pipeline/blob/master/doc/sip/Initial_SIP.ipynb

    - This function does not change the linear part of the WCS;
      rather, it adds the SIP (Simple Imaging Polynomial) distortion model
      on top of the existing WCS.

    """
    # Poly_Transform_Forward, Poly_Transform_Inverse = get_initial_distortion_model()
    Array_Distortion_Model = get_initial_distortion_model(initial_distortion_table)

    # Shift input detector pixel coordinates by `CRPIX`
    # Subtract 1 from `CRPIX` values for the compensation of of difference
    # in the pixel-coordinate origin convention between the FITS WCS and
    # gwcs
    shift_by_crpix = (models.Shift(-(w.wcs.crpix[0] - 1))
                      & models.Shift(-(w.wcs.crpix[1] - 1)))

    # Rotation and pixel-to-degree conversion using CD matrix
    if w.wcs.has_cd():
        cd = np.array(w.wcs.cd, dtype=float)
        original_pc = cd / np.sqrt((cd**2).sum(axis=0))  # not used; only if you need a PC back
        original_cdelt = np.array([np.hypot(cd[0, 0], cd[1, 0]),
                                   np.hypot(cd[0, 1], cd[1, 1])], dtype=float)
    else:
        original_pc = copy.deepcopy(w.wcs.get_pc())
        original_cdelt = copy.deepcopy(w.wcs.get_cdelt())
        cd = np.diag(original_cdelt) @ original_pc
        logging.info(f"PC = {original_pc}")
        logging.info(f"CDELT = {original_cdelt}")
        logging.info(f"CD = {cd}")

    rotation = models.AffineTransformation2D(cd)
    rotation.inverse = models.AffineTransformation2D(np.linalg.inv(cd))

    # Spherical projection
    if projection == "SIN":
        pix2sky = models.Pix2Sky_SIN()
    elif projection == "TAN":
        pix2sky = models.Pix2Sky_TAN()

    # Rotation on the sky
    celestial_rotation = models.RotateNative2Celestial(
        w.wcs.crval[0], w.wcs.crval[1], 180.0
    )

    # Combine the above transformations into a linear transform
    det2icrs = shift_by_crpix | rotation | pix2sky | celestial_rotation
    det2icrs.name = "linear_transform"

    # Distortion-correcting (undistorting) transform
    dist_poly_x = Array_Distortion_Model[array]["x"]
    dist_poly_y = Array_Distortion_Model[array]["y"]
    # distortion offset
    dist_xo = Array_Distortion_Model[array]["xo"]
    dist_yo = Array_Distortion_Model[array]["yo"]
    # undistortion polynomials
    undist_poly_x = Array_Distortion_Model[array]["undist-x"]
    undist_poly_y = Array_Distortion_Model[array]["undist-y"]

    # shift of pixel coordinates taking into account the edge pixels cut out
    shift_by_edge = (
        models.Shift(oversample * edge_pixels)
        & models.Shift(oversample * edge_pixels)
    )
    unshift_by_edge = (
        models.Shift(-oversample * edge_pixels)
        & models.Shift(-oversample * edge_pixels)
    )

    # downsampling and oversampling conversions of pixel coordinates
    # following the 0-based FITS convention
    if oversample == 1:
        downsampling = oversampling = models.Identity(2)
    else:
        downsampling = (
            (models.Shift(0.5) & models.Shift(0.5))
            | (models.Scale(1.0 / oversample) & models.Scale(1.0 / oversample))
            | (models.Shift(-0.5) & models.Shift(-0.5))
        )
        oversampling = (
            (models.Shift(0.5) & models.Shift(0.5))
            | (models.Scale(oversample) & models.Scale(oversample))
            | (models.Shift(-0.5) & models.Shift(-0.5))
        )
    # distortion-correcting (undistorting) transformation taking into
    # account the oversampling and the edge pixels cut out
    undistortion = (
        shift_by_edge
        | downsampling
        | (models.Shift(dist_xo) & models.Shift(dist_yo))
        | models.Mapping((0, 1, 0, 1))
        | (undist_poly_x & undist_poly_y)
        | oversampling
        | unshift_by_edge
    )
    undistortion.inverse = (
        shift_by_edge
        | downsampling
        | models.Mapping((0, 1, 0, 1))
        | (dist_poly_x & dist_poly_y)
        | (models.Shift(-dist_xo) & models.Shift(-dist_yo))
        | oversampling
        | unshift_by_edge
    )
    undistortion.name = "undistortion"

    # Define coordinate frames
    # detector coordinate frame
    detector_frame = gwcs.coordinate_frames.Frame2D(
        name="detector", axes_names=("x", "y"), unit=(u.pix, u.pix)
    )

    # distortion-corrected (undistorted) pixel coordinates frame
    undistorted_frame = gwcs.coordinate_frames.Frame2D(
        name="undistorted_frame",
        axes_names=("undist_x", "undist_y"),
        unit=(u.pix, u.pix),
    )

    # ICRS sky frame
    icrs_frame = gwcs.coordinate_frames.CelestialFrame(
        reference_frame=coord.ICRS(), name="icrs", unit=(u.deg, u.deg)
    )

    # Build a gwcs WCS object from the pipeline of transformations
    pipeline = [
        (detector_frame, undistortion),
        (undistorted_frame, det2icrs),
        (icrs_frame, None),
    ]
    wcsobj_gwcs = gwcs.wcs.WCS(pipeline)

    # Get an astropy WCS object through FITS header
    # Fit the SIP coefficients over an area slightly wider than the given
    # pixel numbers to avoid boundary effect
    x0 = -0.5 - edge_pixels*oversample
    x1 = -0.5 + (nx + edge_pixels)*oversample
    y0 = -0.5 - edge_pixels*oversample
    y1 = -0.5 + (ny + edge_pixels)*oversample
    fitshdr_sip = wcsobj_gwcs.to_fits_sip(
        bounding_box=((x0, x1), (y0, y1)),
        max_pix_error=0.1,
        max_inv_pix_error=0.1,
        projection=projection,
    )
    w.wcs.ctype = [f"RA---{projection}-SIP", f"DEC--{projection}-SIP"]
    w_sip = wcs.WCS(header=fitshdr_sip)
    w.sip = w_sip.sip
    return w


def calculate_pa_from_pc(wcs: WCS) -> Angle:
    """Calculate the position angle image using the PC matrix.

    Parameters
    ----------
    wcs : `astropy.wcs.WCS`
        The WCS object for the image.

    Returns
    -------
    position_angle : `astropy.coordinates.Angle`
        The position angle in degrees.

    """
    # Extract the PC matrix (or use CD matrix if present)
    if wcs.wcs.has_cd():
        pc_matrix = wcs.wcs.cd
    else:
        pc_matrix = wcs.wcs.pc

    # Extract elements of the PC matrix
    pc00, pc10 = pc_matrix[0, 0], pc_matrix[1, 0]

    # Calculate the PA in radians
    pa_radians = np.arctan2(pc10, pc00)

    # Convert to degrees
    pa_degrees = np.degrees(pa_radians)

    return pa_degrees


def pc_to_cd(pc: np.ndarray, cdelt: np.ndarray) -> np.ndarray:
    """Convert PC matrix and CDELT to CD matrix."""
    diag_cdelt = np.diag(cdelt)
    return diag_cdelt @ pc


def cd_to_pc(cd: np.ndarray, cdelt: np.ndarray) -> np.ndarray:
    """Convert CD matrix and CDELT to PC matrix."""
    diag_cdelt = np.diag(cdelt)
    return solve(diag_cdelt, cd)


def position_angle_from_cd(cd_matrix):
    """Calculate the position angle in degrees from a CD matrix."""
    CD1_1, CD1_2 = cd_matrix[0]
    CD2_1, CD2_2 = cd_matrix[1]

    # Compute PA in radians
    pa_rad = np.arctan2(-CD1_2, CD2_2)

    # Convert to degrees, normalize to [0, 360)
    pa_deg = np.degrees(pa_rad) % 360

    return pa_deg


def compute_side_pas(boresight_ra: float,
                     boresight_dec: float,
                     boresight_pa: float,
                     fov_width_deg: float = 11.3,
                     wcs_builder=None,  # function(ra,dec,roll)->center-array linear WCS
                     nx: int = 2040,
                     ny: int = 2040,
                     return_roll: bool = False):
    """Compute the position angles (PAs) for the left, center, and right arrays
    of a three-array focal plane, given a boresight and roll.

    Parameters
    ----------
    boresight_ra : `float`
        Boresight right ascension in degrees (ICRS).
    boresight_dec : float
        Boresight declination in degrees (ICRS).
    boresight_pa : `float`
        Boresight roll/position angle in degrees. Interpreted as the rotation
        of the focal plane around the boresight; passed to ``wcs_builder``.
    fov_width_deg : `float`, optional
        Full on-sky width of the three-array focal plane in degrees.
        The side arrays are placed at offsets of +/- fov_width_deg/3 from
        the boresight along the boresight-aligned tangent plane. Default is
        11.3.
    wcs_builder : callable or `None`, optional
        Function with signature ``wcs_builder(ra, dec, roll) -> astropy.wcs.WCS``
        that returns a center-array linear WCS at the given sky location and
        roll. If ``None``, a default builder is used:
        ``get_initial_linear_wcs(ra_boresight=..., dec_boresight=..., array=2,
        roll=..., nx=nx, ny=ny, edge_pixels=0)``.
    nx, ny : `int`, optional
        Pixel dimensions to pass to the default ``wcs_builder`` when a custom
        builder is not provided. They do not otherwise affect PA computation.
        Defaults are 2040 x 2040.
    return_roll : `bool`, optional
        If ``False`` (default), return north-aligned PAs (east of north,
        degrees) for (left, center, right). If ``True``, return spacecraft
        rolls for (left, center, right), computed as ``(180 deg - PA) mod 360``.

    Returns
    -------
    pa_left : `float`
        Left array PA (degrees), north-aligned (if ``return_roll=False``) or
        spacecraft roll (if ``return_roll=True``).
    pa_center : `float`
        Center array PA (degrees), north-aligned or roll as above.
    pa_right : `float`
        Right array PA (degrees), north-aligned or roll as above.

    Notes
    -----
    This routine constructs a center linear WCS at the boresight, then places
    two side pointings at +/-(fov_width_deg/3) along the boresight-aligned
    tangent plane. It determines which side is "image-left" by comparing the
    pixel-x coordinates of the two side sky points in the center WCS. For each
    of the three sky locations (left/center/right), it estimates the PA of the
    image +y direction, then converts that to a north-aligned PA (east of
    north, 0 - 360 deg). Optionally, the north-aligned PAs can be converted back to
    spacecraft rolls (i.e., 180deg - PA).

    PA is defined as the angle measured east of celestial north,
    (0 deg = north, 90 deg = east).

    Spacecraft roll is the rotation of the entire spacecraft/focal plane
    around its boresight, expressed in the spacecraft's own frame.

    """
    ra0 = boresight_ra * u.deg
    dec0 = boresight_dec * u.deg
    roll = boresight_pa * u.deg

    boresight = SkyCoord(ra=ra0, dec=dec0, frame="icrs")
    off = SkyOffsetFrame(origin=boresight, rotation=roll)

    dx = fov_width_deg / 3.0
    sky_plus = SkyCoord(lon=+dx*u.deg, lat=0*u.deg, frame=off).icrs
    sky_minus = SkyCoord(lon=-dx*u.deg, lat=0*u.deg, frame=off).icrs

    # Build linear WCS for the *center* array
    if wcs_builder is None:
        def wcs_builder(ra_c, dec_c, roll_c):
            return get_initial_linear_wcs(ra_boresight=ra_c, dec_boresight=dec_c,
                                          array=2, roll=roll_c, nx=nx, ny=ny)
    w_center = wcs_builder(boresight_ra, boresight_dec, boresight_pa)

    # Decide which side is image-left by pixel-x parity in the center WCS
    x_plus, _ = w_center.world_to_pixel(sky_plus)
    x_minus, _ = w_center.world_to_pixel(sky_minus)
    plus_is_left = (x_plus < x_minus)

    # Helper: PA aligned to the image +y direction, flipped to be "north-aligned"
    def north_aligned_pa_at(world: SkyCoord) -> float:
        x, y = w_center.world_to_pixel(world)
        up_world = w_center.pixel_to_world(x, y + 1e-3)   # tiny +y pixel step
        pa = world.position_angle(up_world).to(u.deg).value
        # If +y points toward decreasing Dec (i.e., south), flip by 180 deg
        if (up_world.dec - world.dec).to_value(u.deg) < 0:
            pa = (pa + 180.0) % 360.0
        else:
            pa = (pa - 180.0) % 360.0
        return pa

    # Assign to left/right using the parity test
    if plus_is_left:
        pa_center = north_aligned_pa_at(boresight)
        pa_plus = north_aligned_pa_at(sky_plus)
        pa_minus = north_aligned_pa_at(sky_minus)
        pa_left, pa_right = pa_plus, pa_minus
    else:
        pa_center = north_aligned_pa_at(boresight)
        pa_plus = north_aligned_pa_at(sky_minus)
        pa_minus = north_aligned_pa_at(sky_plus)
        pa_left, pa_right = pa_minus, pa_plus

    if not return_roll:
        # Return North-aligned PAs
        return pa_left, pa_center, pa_right
    else:
        # Return spacecraft rolls
        return (180. - pa_left) % 360., (180. - pa_center) % 360., (180. - pa_right) % 360.


def cd_from_wcs(w: WCS) -> np.ndarray:
    """Return CD matrix (scale + rotation) for a WCS.

    Parameters
    ----------
    w : `astropy.wcs.WCS`
        Input WCS.

    Returns
    -------
    `numpy.ndarray`
        A (2, 2) array containing the CD matrix as floats.

    Notes
    -----
    If a CD matrix is present, it is returned directly.
    Otherwise, the CD matrix is synthesized from
    ``PC`` and ``CDELT`` as: ``CD = diag(CDELT) @ PC``

    """
    # unify scale+rotation into a CD matrix
    if w.wcs.has_cd():
        return np.asarray(w.wcs.cd, dtype=float)
    pc = np.asarray(w.wcs.pc, dtype=float)
    cdelt = np.asarray(w.wcs.cdelt, dtype=float)
    return np.diag(cdelt) @ pc


def north_aligned_pa_from_wcs(w: WCS) -> float:
    """Compute the north-aligned position angle (PA) of the image +y axis
    for a given WCS, measured east of celestial north.

    Parameters
    ----------
    w : `astropy.wcs.WCS`
        Input WCS.

    Returns
    -------
    `float`
        Position angle in degrees on [0, 360), representing the direction of
        the image +y axis measured east of north.

    Notes
    -----
    The PA is first derived from the CD matrix via
    :func:`position_angle_from_cd`. If the CD elements indicate that image +y
    points south (i.e., north component of +y is negative), the angle is
    flipped by 180° to yield a conventional north-aligned PA.

    """
    cd = cd_from_wcs(w)
    pa = position_angle_from_cd(cd)   # PA of +y (east of north)
    if cd[1, 1] < 0:                  # if north component of +y is negative -> +y points south
        pa = (pa + 180.0) % 360.0
    return pa


def build_wcs_triplet_from_boresight(boresight_ra: float,
                                     boresight_dec: float,
                                     boresight_pa: float,
                                     fov_width_deg: float = 11.3,
                                     wcs_builder=None,
                                     nx: int = 2040,
                                     ny: int = 2040,
                                     put_scale_in_pc: bool = True,
                                     negative_det: bool = True):
    """Construct three linear WCS objects (left, center, right) for a
    three-array focal plane centered on a given boresight and roll.

    Parameters
    ----------
    boresight_ra : `float`
        Boresight right ascension in degrees (ICRS).
    boresight_dec : `float`
        Boresight declination in degrees (ICRS).
    boresight_pa : `float`
        Boresight roll/position angle in degrees.
        Passed to the WCS builder for each array.
    fov_width_deg : `float`, optional
        Total field-of-view width (deg) across the three arrays. Side centers
        are placed at +/-(fov_width_deg/3). Default is 11.3.
    wcs_builder : callable or `None`, optional
        Function with signature ``wcs_builder(ra, dec, roll) -> astropy.wcs.WCS``
        that returns a center-array linear WCS at the given sky location and
        roll. If ``None``, defaults to
        ``get_initial_linear_wcs(ra_boresight=..., dec_boresight=..., array=2,
        roll=..., nx=nx, ny=ny, edge_pixels=0)``.
    nx, ny : `int`, optional
        Pixel dimensions supplied to the default ``wcs_builder`` when a custom
        builder is not provided. Defaults are 2040 x 2040.
    put_scale_in_pc : `bool`, optional
        If ``True`` (default), multiply the PC columns by CDELT and then set
        CDELT to 1.0. This produces a scale-in-PC representation (no CDELT).
    negative_det : `bool`, optional
        If ``True`` (default) and ``put_scale_in_pc`` is enabled, flips the
        sign of the PC y-row to enforce a negative determinant (conventional
        “image y up” behavior as defined here).

    Returns
    -------
    w_left : `astropy.wcs.WCS`
        Linear WCS for the left array.
    w_center : `astropy.wcs.WCS`
        Linear WCS for the center array at the boresight.
    w_right : `astropy.wcs.WCS`
        Linear WCS for the right array.

    """
    # PAs (north-aligned, consistent with detector +y)
    pa_left, pa_center, pa_right = compute_side_pas(
        boresight_ra, boresight_dec, boresight_pa,
        fov_width_deg=fov_width_deg, wcs_builder=wcs_builder, nx=nx, ny=ny
    )

    # default builder
    if wcs_builder is None:
        def wcs_builder(ra_c, dec_c, roll_c):
            return get_initial_linear_wcs(ra_boresight=ra_c, dec_boresight=dec_c,
                                          array=2, roll=roll_c, nx=nx, ny=ny)

    # SkyOffset geometry: side centers at (±dx, 0) in boresight-aligned tangent plane
    boresight = SkyCoord(boresight_ra*u.deg, boresight_dec*u.deg, frame="icrs")
    off = SkyOffsetFrame(origin=boresight, rotation=pa_center*u.deg)
    dx = fov_width_deg / 3.0
    dy = 0
    sky_plus = SkyCoord(lon=+dx*u.deg, lat=dy*u.deg, frame=off).icrs
    sky_minus = SkyCoord(lon=-dx*u.deg, lat=dy*u.deg, frame=off).icrs

    # Use center WCS only to decide which sign is image-left
    w_center_tmp = wcs_builder(boresight_ra, boresight_dec, pa_center)
    x_plus, _ = w_center_tmp.world_to_pixel(sky_plus)
    x_minus, _ = w_center_tmp.world_to_pixel(sky_minus)
    plus_is_left = (x_plus > x_minus)

    # Assign CRVALs directly from SkyOffset
    if plus_is_left:
        pa_right_tmp = pa_right.copy()
        pa_left_tmp = pa_left.copy()
        pa_left = pa_right_tmp
        pa_right = pa_left_tmp

    sky_left, sky_right = sky_minus, sky_plus

    # Build WCSes at those centers with their rolls
    w_left = wcs_builder(sky_left.ra.to_value(u.deg), sky_left.dec.to_value(u.deg), pa_left)
    w_center = wcs_builder(boresight_ra, boresight_dec, pa_center)
    w_right = wcs_builder(sky_right.ra.to_value(u.deg), sky_right.dec.to_value(u.deg), pa_right)

    if put_scale_in_pc:
        for w in (w_left, w_center, w_right):
            pc = w.wcs.pc.copy()
            cdelt = w.wcs.cdelt.copy()
            pc[:, 0] *= cdelt[0]
            pc[:, 1] *= cdelt[1]
            w.wcs.cdelt[:] = 1.0
            if negative_det:
                pc[1, :] *= -1.0   # flip y to get negative determinant
            w.wcs.pc = pc

    # w_left._naxis = [2040, 2040]
    # w_center._naxis = [2040, 2040]
    # w_right._naxis = [2040, 2040]

    return w_left, w_center, w_right
