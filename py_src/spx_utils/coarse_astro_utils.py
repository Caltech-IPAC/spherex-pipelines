__all__ = ["get_ra_dec_lon_lat_from_quat", "pc_to_cd", "cd_to_pc",
           "get_initial_linear_wcs", "get_initial_wcs_with_sip",
           "position_angle_from_cd", "compute_side_pas"]

# Standard library imports
import logging
import copy

# Related third party imports
from astropy import coordinates as coord
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.coordinates import SkyOffsetFrame, Angle
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


def get_ra_dec_lon_lat_from_quat(Q: np.array,
                                 debug: bool = False) -> tuple[float, float, float, float, float, float]:
    """Calculates the latitude and longitude of the boresight vector
    after applying the quaternion rotation.

    Parameters
    ----------
    Q : `np.array`
        Input quaternion for a pointing.
    debug : `bool`
        If True, log coordinates and PA in both ICRS and Ecliptic coordinates

    Returns
    -------
    `tuple[float, float, float, float, float, float]`
        ra_eq, dec_eq, pa_eq, lon_ecl, lat_ecl, pa_ecl

    Notes
    -----
    Quaternions from C-kernel file are in LOS frame. (Bret Naylor)
    """
    # --- Step 1: Define initial PLD axes ---
    xPLD = np.array([1, 0, 0])
    zPLD = np.array([0, 0, 1])

    # --- Step 2: Apply cant rotation: rotate PLD -> LOS ---
    cant_angle_deg = 8.0  # needs to be verified
    cant_rotation = Rotation.from_euler('y', cant_angle_deg, degrees=True)

    xLOS_init = cant_rotation.apply(xPLD)
    zLOS_init = cant_rotation.apply(zPLD)
    # Apply the quaternion rotation
    # to get the boresight vector in equatorial coordinates
    # NB: scipy expects quaternions in scalar-last format
    rotation = Rotation.from_quat(Q)
    zLOS_boresight = rotation.apply(zLOS_init)  # Equatorial

    # Get RA and Dec in Equatorial
    skycoord = SkyCoord(CartesianRepresentation(x=zLOS_boresight[0] * u.km,
                                                y=zLOS_boresight[1] * u.km,
                                                z=zLOS_boresight[2] * u.km))
    ra_eq = skycoord.icrs.ra.deg
    dec_eq = skycoord.icrs.dec.deg

    # Obliquity of the ecliptic: SkySimulator as reference `simulated_adcs`.
    obliq_deg = 23.439291

    # Define the rotation matrix to convert between equatorial and ecliptic coordinates
    R_eq2eclip = Rotation.from_euler('x', -obliq_deg, degrees=True).as_matrix()
    # Convert to ecliptic coordinates
    zLOS_boresight_eclip = R_eq2eclip @ zLOS_boresight  # Ecliptic

    # Calculate latitude and longitude
    dnorm = np.linalg.norm(zLOS_boresight_eclip)  # norm of boresight vector
    theta = np.arccos(zLOS_boresight_eclip[2] / dnorm)  # ecliptic latitude
    phi = np.arctan2(zLOS_boresight_eclip[1], zLOS_boresight_eclip[0])  # ecliptic longitude

    lon_ecl = np.degrees(phi) % 360.0
    lat_ecl = 90.0 - np.degrees(theta)

    # Calculate the x-axis direction vector after quaternion rotation
    xLOS_boresight = rotation.apply(xLOS_init)

    # Calculate the position angle (PA)
    # Define the reference vector for PA=0 in ecliptic coordinates
    xLOS_eclip_pa0 = np.array([-np.sin(np.radians(lon_ecl)),
                               np.cos(np.radians(lon_ecl)),
                               0.0])

    # Rotate this reference vector to equatorial coordinates
    xLOS_eq_pa0 = R_eq2eclip.T @ xLOS_eclip_pa0

    # Calculate the dot product to get the cosine of the angle
    cos_PA = (np.dot(xLOS_eq_pa0, xLOS_boresight)
              / (np.linalg.norm(xLOS_eq_pa0) * np.linalg.norm(xLOS_boresight)))
    cos_PA = np.clip(cos_PA, -1, 1)  # Clip for numerical stability

    # Calculate the unsigned PA
    unsigned_PA = np.arccos(cos_PA)

    # Determine the sign of PA using the cross product
    cross_prod = np.cross(xLOS_eq_pa0, xLOS_boresight)
    sign_of_PA = np.sign(np.dot(zLOS_boresight, cross_prod))

    # Compute the signed PA
    pa_ecl = np.degrees(sign_of_PA * unsigned_PA) % 360.0
    if np.round(pa_ecl) == 0.0:
        pa_ecl = 0.0
    elif np.round(pa_ecl) == 180.0:
        pa_ecl = 180.0
    elif np.round(pa_ecl) == 360.0:
        pa_ecl = 0.0

    pa_ecl = np.degrees(sign_of_PA * unsigned_PA)  # ecliptic

    # Calculate PA in equatorial coordinates
    xLOS_eq_pa0 = np.array([-np.sin(np.radians(ra_eq)),
                            np.cos(np.radians(ra_eq)),
                            0.0])
    cos_PA_eq = (np.dot(xLOS_eq_pa0, xLOS_boresight)
                 / (np.linalg.norm(xLOS_eq_pa0) * np.linalg.norm(xLOS_boresight)))
    cos_PA_eq = np.clip(cos_PA_eq, -1, 1)
    unsigned_PA_eq = np.arccos(cos_PA_eq)
    cross_prod_eq = np.cross(xLOS_eq_pa0, xLOS_boresight)
    sign_of_PA_eq = np.sign(np.dot(zLOS_boresight, cross_prod_eq))
    pa_eq = sign_of_PA_eq * unsigned_PA_eq * 180.0/np.pi

    if debug:
        logging.info(f"Right Ascension (deg): {ra_eq}")
        logging.info(f"Declination (deg): {dec_eq}")
        logging.info(f"Position Angle EQ (deg)= {pa_eq}")
        logging.info(f"Longitude (deg): {lon_ecl}")
        logging.info(f"Latitude (deg): {lat_ecl}")
        logging.info(f"Position Angle ECL (deg): {pa_ecl}")

    return ra_eq, dec_eq, pa_eq, lon_ecl, lat_ecl, pa_ecl


def get_initial_linear_wcs(ra_boresight,
                           dec_boresight,
                           array,
                           roll=None,
                           oversample=1,
                           pixel_size=6.15,
                           nx=2040,
                           ny=2040,
                           edge_pixels=4,
                           projection="TAN"):
    """A routine to generate the World Coordinate System corresponding to a
       SPHEREx array given a particular SPHEREx boresight pointing.

    Parameters
    ----------
    ra_boresight : `float`
        The boresight right ascencion in degrees.
    dec_boresight : `float`
        The boresight declination in degrees.
    roll : `float`, optional
        A roll angle around the telescope line of sight in degrees.
        The default is None.
    oversample : `int`, optional
        If oversample=1, assume the native SPHEREx array.
        Otherwise, specify a high resolution oversampled imgage by a factor
        specified with this keyword.
        The default is 1.
    array : `int`, optional
        SPHEREx array: 1, 2, 3, 4, 5, 6. The default is 1.
        The values 1, 2, 3, 4, 5 or 6 return a WCS for the indvidual arrays.
    pixel_size : `float`, optional
        Pixel size in arcseconds. The default is 6.15.
    nx, ny : `int`, optional
        Pixel counts in the x- and y-direction of target image. The default is 2040.
    edge_pixels : `int`, optional
        Width of edge pixels cut out from the target image.  The default is 4.
        Use this parameter when your target image has the edge pixels cut out (e.g.,
        Level-2-format SPHEREx data saved with `save_to_file()` method of
        `SPHEREx_Image` class.
    projection : string, optional
        The WCS projection to use.  Currently "SIN" and "TAN" are supported.
        The default is "SIN".

    Returns
    -------
    w : WCS
        An astropy WCS (world-coordinate system) object.

    Notes
    -----
    This function is a partial replication of `get_spherex_wcs`
    https://github.com/SPHEREx/SPHEREx-Sky-Simulator/blob/main/py/SPHEREx_InstrumentSimulator/tools.py

    """
    assert projection in (
        "SIN",
        "TAN",
    ), "Unsupported projection.  Projection must be either 'SIN' or 'TAN'"

    # set up the wcs
    w = wcs.WCS(naxis=2)
    w.wcs.crval = [ra_boresight, dec_boresight]  # boresight pointing

    # what is the space between the array in units of pixels
    # taking into account the edge pixels
    gap_size_pixels = (22.4 * 60.0 / pixel_size + 2 * edge_pixels) * oversample

    # Reference pixel is the center of the middle array
    # Note that pixel coordinates in the FITS standard are 1-based and
    # integer coordinates refer to the centers of pixels.
    if array in [3, 6]:
        w.wcs.crpix = [3 * nx * oversample / 2 + gap_size_pixels + 0.5, ny * oversample / 2 + 0.5]
    if array in [2, 5]:
        w.wcs.crpix = [nx * oversample / 2 + 0.5, ny * oversample / 2 + 0.5]
    if array in [1, 4]:
        w.wcs.crpix = [-nx * oversample / 2 - gap_size_pixels + 0.5, ny * oversample / 2 + 0.5]

    w.wcs.cdelt = [pixel_size / 3600.0, pixel_size / 3600.0]
    w.wcs.ctype = [f"RA---{projection}", f"DEC--{projection}"]

    if roll is not None:
        theta = np.deg2rad(roll)
        w.wcs.pc = [[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]]

    else:
        print("No roll angle present, returning identity.")
        w.wcs.pc = [[1.0, 0.0], [0.0, 1.0]]
    # Remove CROTA2 to avoid conflicts
    if hasattr(w.wcs, "crota"):
        del w.wcs.crota

    return w


def get_initial_wcs_with_sip(w: WCS,
                             initial_distortion_table: Table,
                             array: int,
                             nx: int = 2040, ny: int = 2040,
                             oversample: int = 1,
                             edge_pixels: int = 4,
                             projection="TAN"):
    """Determines an initial WCS with SIP distortion correction.

    Parameters
    ----------
    w : `astropy.wcs.WCS`
        Initial WCS object without SIP distortion correction.

    Returns
    -------
    `astropy.wcs.WCS`
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
        cd = w.wcs.cd
        raise Warning("Please use PC formalism. CD is not supported.")
    else:
        pc = w.wcs.get_pc()
        original_pc = copy.deepcopy(w.wcs.get_pc())
        cdelt = w.wcs.get_cdelt()
        original_cdelt = copy.deepcopy(w.wcs.get_cdelt())
        cd = np.array([[cdelt[0], 0.0], [0.0, cdelt[1]]]) @ pc
        _ = copy.deepcopy(cd)
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
        w.wcs.crval[0], w.wcs.crval[1], w.wcs.lonpole
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
    fitshdr_sip = wcsobj_gwcs.to_fits_sip(
        # bounding_box=(
        #     (-0.5 - 100 * oversample, -0.5 + (nx + 100) * oversample),
        #     (-0.5 - 100 * oversample, -0.5 + (ny + 100) * oversample),
        # ),
        bounding_box=(
            (-0.5, nx - 0.5),
            (-0.5, ny - 0.5),
        ),
        max_pix_error=0.1,
        max_inv_pix_error=0.1,
        projection=projection,
    )
    print(fitshdr_sip)
    w = wcs.WCS(header=fitshdr_sip)
    # Ensure CTYPE includes the -SIP suffix for SIP distortion
    w.wcs.ctype = [f"RA---{projection}-SIP", f"DEC--{projection}-SIP"]
    cdelt = [6.15/3600., 6.15/3600]
    pc = cd_to_pc(w.wcs.cd, cdelt)
    w.wcs.pc = pc
    del w.wcs.cd
    w.wcs.cdelt = original_cdelt

    if array in [4, 5, 6]:
        # flipping X means flipping the first axis before scaling.
        # Negating the first column of the PC matrix does this.
        print("MWIR doing x-axis flip (RA direction) using PC matrix.")
        # Flip X by negating the first column of the PC matrix
        w.wcs.pc[0, 0] *= -1  # PC1_1
        w.wcs.pc[1, 0] *= -1  # PC2_1

    w.wcs.pc[0][0] *= -1
    w.wcs.pc[1][0] *= -1

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
    CD1_1, CD1_2 = cd_matrix[0]
    CD2_1, CD2_2 = cd_matrix[1]

    # Compute PA in radians
    pa_rad = np.arctan2(-CD1_2, CD2_2)

    # Convert to degrees, normalize to [0, 360)
    pa_deg = np.degrees(pa_rad) % 360

    return pa_deg


def compute_side_pas(boresight_ra: float, boresight_dec: float, boresight_pa: float,
                     fov_width_deg: float = 11.3) -> tuple[u.Quantity, u.Quantity, u.Quantity]:
    """
    Compute the sky-aligned position angles (PA) at the left, right
    detectors of the SPHEREx focal plane.

    Parameters
    ----------
    boresight_ra : `float`
        Right Ascension of telescope boresight (degrees).
    boresight_dec : `float`
        Declination of telescope boresight (degrees).
    boresight_pa : `float`
        Position angle (roll) at the boresight (degrees, east of north).
    fov_width_deg : `float`, optional
        Total focal plane width in degrees. Default is 11.3.

    Returns
    -------
    `tuple[Quantity, Quantity, Quantity]`
        (left PA, center PA, right PA), each as astropy Quantity in degrees.
    """
    # Convert inputs
    ra = boresight_ra * u.deg
    dec = boresight_dec * u.deg
    pa_center = boresight_pa * u.deg

    # Create boresight coordinate and offset frame
    boresight = SkyCoord(ra=ra, dec=dec, frame='icrs')
    offset_frame = SkyOffsetFrame(origin=boresight, rotation=pa_center)

    # Offset in degrees from center to side detectors
    dx = fov_width_deg / 3.0
    dy = 0.0

    def pa_at_offset(dx_val):
        # Step 1: detector center
        detector = SkyCoord(lon=dx_val*u.deg, lat=dy*u.deg, frame=offset_frame).transform_to('icrs')
        # Step 2: small +Y offset
        unit_up = SkyCoord(lon=dx_val*u.deg, lat=(dy + 1e-4)*u.deg, frame=offset_frame).transform_to('icrs')
        # Step 3: local PA of "up"
        return detector.position_angle(unit_up).to(u.deg)

    pa_left = pa_at_offset(-dx).to_value()
    pa_right = pa_at_offset(+dx).to_value()

    if boresight_pa > 360.:
        pa_left += 360
        pa_right += 360

    return pa_left, pa_center, pa_right
