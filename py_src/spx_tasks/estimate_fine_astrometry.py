__all__ = ["EstimateFineAstrometryConfig", "EstimateFineAstrometryTask"]

# Standard library imports
import copy
import logging
import os
import glob
from pathlib import Path
import warnings
import subprocess

# Related third party imports
from astropy import coordinates as coord
from astropy.io import fits
from astropy.coordinates import SkyCoord, GCRS
from astropy.table import Table
import astropy.units as u
from astropy.stats import SigmaClip
from astropy.wcs import Sip, WCS
from lsst.pex.config import Field, DictField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from photutils.background import Background2D, MedianBackground
from astropy.io.fits.verify import VerifyWarning

# Local application/library specific imports
from spx_utils.image_catalog import preprocess_catalog, SourcePipelineClass
from spx_data import SPHERExImage, add_exposure_boundary_info
from spx_utils import get_temp_path, get_ssdc_tempfiledir
from spx_utils.astrometry.fine_astrometry_utils import se_run, scamp_run, astronet_run
from spx_utils.solid_angle_utils import calculate_solid_angle, calculate_median_std_solid_angle
from spx_tasks import add_history
from spx_utils.coarse_astro_utils import pc_to_cd, cd_to_pc
# position_angle_from_cd  # calculate_pa_from_pc
# ==============================================================================

root_dir = Path(__file__).parent.parent
logger = logging.getLogger(__name__)


# Suppress Astropy FITS VerifyWarnings
warnings.simplefilter('ignore', category=VerifyWarning)

# Also suppress generic Astropy warnings if needed
warnings.filterwarnings('ignore', category=UserWarning, module='astropy')


def convert_hdu_to_ldac(hdu) -> tuple[fits.BinTableHDU, fits.BinTableHDU]:
    """Convert an hdu table to a fits_ldac table (format used by astromatic suite).

    Parameters
    ----------
    hdu: `astropy.io.fits.BinTableHDU` or `astropy.io.fits.TableHDU`
        HDUList to convert to fits_ldac HDUList

    Returns
    -------
    tbl1: `astropy.io.fits.BinTableHDU`
        Header info for fits table (LDAC_IMHEAD)
    tbl2: `astropy.io.fits.BinTableHDU`
        Data table (LDAC_OBJECTS)
    """
    tblhdr = np.array([hdu.header.tostring(',')])
    col1 = fits.Column(name='Field Header Card', array=tblhdr, format='13200A')
    cols = fits.ColDefs([col1])
    tbl1 = fits.BinTableHDU.from_columns(cols)
    tbl1.header['TDIM1'] = '(80, {0})'.format(len(hdu.header))
    tbl1.header['EXTNAME'] = 'LDAC_IMHEAD'
    tbl2 = fits.BinTableHDU(hdu.data)
    tbl2.header['EXTNAME'] = 'LDAC_OBJECTS'
    return tbl1, tbl2


def catalog2ldac(catalog, ldac_filepath) -> None:
    """Write ldac file.

    Converts an astropy table to a fits_ldac.

    Parameters
    ----------
    catalog: `astropy.table.Table`
        Source catalog.
    ldac_filepath:
        Path to ldac file.

    Returns
    -------
    `None`
    """
    hdu = fits.BinTableHDU(catalog)
    tbl1, tbl2 = convert_hdu_to_ldac(hdu)
    new_hdulist = [fits.PrimaryHDU(), tbl1, tbl2]
    new_hdulist = fits.HDUList(new_hdulist)
    new_hdulist.writeto(ldac_filepath, overwrite=True)


def get_matched_pairs(scamp_out) -> tuple[np.ndarray, np.ndarray, SkyCoord]:
    """Retrieve matched pairs from scamp.

    Parameters
    ----------
    scamp_out:

    Returns
    -------
    `tuple[~numpy.ndarray, ~numpy.ndarray, ~astropy.coordinates.SkyCoord]`
    Source point loctions of matched pairs.
    """
    import warnings
    # retrieves matched pairs from the last run
    tbl_path = scamp_out[:-5]+'_1.fits'
    # Set the default action for UserWarning to 'ignore'
    warnings.simplefilter('ignore', category=UserWarning)
    T = Table.read(tbl_path, format='fits', hdu=2)
    science = T[T["CATALOG_NUMBER"] == 1]
    reference = T[T["CATALOG_NUMBER"] == 0]
    _, iscience, ireference = np.intersect1d(science["SOURCE_NUMBER"].data,
                                             reference["SOURCE_NUMBER"].data,
                                             assume_unique=True, return_indices=True)
    science = science[iscience]
    reference = reference[ireference]

    c = SkyCoord(reference["ALPHA_J2000"], reference["DELTA_J2000"], unit='deg', frame='icrs')
    return science["X_IMAGE"].data, science["Y_IMAGE"].data, c


def wcs_from_head(filepath) -> WCS:
    """Gets WCS from output scamp output head file.

    Parameters
    ----------
    filepath: `string`

    Returns
    -------
    `WCS`: Solved WCS.
    """
    H = fits.Header()
    fp = open(filepath, 'r')
    for line in fp:
        s = line.split()
        if len(s) > 1:
            if s[1] == '=':
                key = s[0]
                val = s[2]
                if val[-1] == "'":
                    val = val[:-1]
                if val[0] == "'":
                    val = val[1:]
                else:
                    val = float(val)

                H[key] = val
    fp.close()
    logging.info(f'wcs from head : {WCS(H)}')
    return WCS(H)


def linear_err_function(p: np.array,
                        lon: np.array,
                        lat: np.array,
                        x: np.array,
                        y: np.array,
                        w_obj: WCS) -> np.ndarray:
    """Computes residuals between reference celestial and WCS pixel coordinates.

    Parameters:
    p : `np.array`
        Array of parameters defining the WCS transformation.
        The first four elements define the CD matrix,
        and the next two elements define the CRPIX values.
    lon : `np.array`
        Array of reference longitude (Right Ascension) coordinates in degrees.
    lat : `np.array`
        Array of reference latitude (Declination) coordinates in degrees.
    x : `np.array`
        Array of pixel x coordinates.
    y : `np.array`
        Array of pixel y coordinates.
    w_obj : `WCS`
        An astropy WCS object that will be updated with the WCS parameters from `p`.

    Returns:
    resids : ~numpy.ndarray
        The concatenated residuals of longitude and latitude coordinates
        after applying the WCS transformation and comparing to the reference
        coordinates.
    """
    cd = p[0:4]
    crpix = p[4:6]
    w_obj.wcs.cd = ((cd[0], cd[1]), (cd[2], cd[3]))
    w_obj.wcs.crpix = crpix
    lon2, lat2 = w_obj.wcs_pix2world(x, y, 0)
    lat_resids = lat - lat2
    lon_resids = lon - lon2
    lon_resids = np.mod(lon_resids - 180.0, 360.0) - 180.0
    resids = np.concatenate((lon_resids * np.cos(np.radians(lat)), lat_resids))
    return resids


def linear_fit_wcs_nocrpix(x, y, refcoords, guess_wcs):
    """
    Fits WCS parameters to reference celestial coordinates given pixel coordinates,
    keeping the CRPIX values constant.

    Parameters:
    - x: np.array of x pixel coordinates
    - y: np.array of y pixel coordinates
    - refcoords: SkyCoord object with reference celestial coordinates
    - guess_wcs: initial WCS object to be optimized

    Returns:
    - wcs: Optimized WCS object
    """
    wcs = guess_wcs.deepcopy()
    lon, lat = refcoords.ra.deg, refcoords.dec.deg

    # Check if the WCS uses PC, and convert it to CD
    if wcs.wcs.has_pc():
        wcs.wcs.cd = wcs.wcs.pc
        wcs.wcs.cd[0, 0] *= wcs.wcs.cdelt[0]
        wcs.wcs.cd[0, 1] *= wcs.wcs.cdelt[0]
        wcs.wcs.cd[1, 0] *= wcs.wcs.cdelt[1]
        wcs.wcs.cd[1, 1] *= wcs.wcs.cdelt[1]
        wcs.wcs.cdelt = (1., 1.)  # Normalize CDELT after conversion
        del wcs.wcs.pc  # Remove the PC matrix to use CD only

    # Flatten the CD matrix to use as the initial guess for the optimizer
    pguess = wcs.wcs.cd.flatten()

    # Define the error function for optimization
    def linear_err_function(params, lon, lat, x, y, wcs):
        # Update the WCS object's CD matrix with new parameters
        wcs.wcs.cd = params.reshape((2, 2))
        # Calculate the pixel coordinates from WCS using updated CD
        calc_x, calc_y = wcs.wcs_world2pix(lon, lat, 0)
        # Return the residuals (differences) for both coordinates
        return np.concatenate([(calc_x - x), (calc_y - y)])

    # Perform the least squares optimization
    po, cov, infodict, errmsg, success = optimize.leastsq(
        linear_err_function, pguess, args=(lon, lat, x, y, wcs), full_output=True
    )

    # Check if optimization was successful
    if success in [1, 2, 3, 4]:  # Successful optimization codes from leastsq
        wcs.wcs.cd = np.array(po).reshape((2, 2))
        print("Optimization successful.")
    else:
        print(f"Optimization failed: {errmsg}")
        return guess_wcs  # Return the initial guess if optimization fails

    return wcs


def linear_fit_wcs(x: np.array,
                   y: np.array,
                   refcoords: SkyCoord,
                   guess_wcs: WCS,
                   crpix_true: np.array) -> WCS:
    """Fits WCS parameters to reference celestial coordinates given pixel coordinates.

    Parameters are as previously defined.

    Returns:
    wcs : `WCS`
        The optimized WCS object or returns the
        initial guess WCS if optimization was not possible.
    """
    logging.info('Linear fit WCS function:')
    logging.info(f'guess wcs = {guess_wcs}')
    # Initial check for empty inputs
    if x.size == 0 or y.size == 0 or len(refcoords) == 0:
        print("Warning: One or more input arrays are empty. Skipping optimization.")
        return guess_wcs

    # Debugging: print the sizes of x, y, and refcoords
    print(f"Debug Info: x size={x.size}, y size={y.size}, refcoords size={len(refcoords)}")

    # Ensure there are matched pairs for optimization
    if len(refcoords) < 3:  # Require a minimum number of points to perform a meaningful fit
        print("Warning: Not enough matched pairs for meaningful WCS optimization."
              " Returning initial guess WCS.")
        return guess_wcs

    # Copy the initial guess WCS to modify
    wcs = guess_wcs.deepcopy()
    lon, lat = refcoords.ra.deg, refcoords.dec.deg

    try:
        # Extract Right Ascension and Declination from refcoords

        # Optimize the WCS parameters
        po, cov, infodict, errmsg, success = optimize.leastsq(
            lambda p: linear_err_function(np.concatenate([p, crpix_true]), lon, lat, x, y, wcs),
            wcs.wcs.cd.flatten(),
            full_output=True
        )

        if success in [1, 2, 3, 4]:  # scipy.optimize.leastsq success codes
            # Update the WCS with the optimized parameters
            wcs.wcs.cd = np.array(po[0:4].reshape((2, 2)))
            print(f"Optimization successful with {len(x)} matched pairs.")
        else:
            print(f"Optimization failed: {errmsg}")
            return guess_wcs

    except Exception as e:
        print(f"Optimization failed with an exception: {e}")
        return guess_wcs
    logging.info(f'wcs = {wcs}')
    return wcs, len(x)


def tweak_wcs(input_wcs: WCS,
              shift_deg: np.ndarray = None,
              shift_pix: np.ndarray = None,
              rotate: float = None,
              scale: float = None) -> WCS:
    """Modifies linear WCS with shift and rotation;
       used to assess recovery of astrometry with known error.

    Parameters
    ----------
    input_wcs : `astropy.wcs.WCS`
        A WCS (World Coordinate System) object to tweak.
    shift_deg : `array-like`, optional
        amount by which to shift the CRVAL keywords (in degrees).
        The default is None.
    shift_pix : `array-like`, optional
        amount by which to shift the CRPIX keywords (in pixels).
        The default is None.
    rotate : `float`, optional
        Angle (in degrees) by which to rotate the WCS.
        uses the PC matrix formalism in FITS.
        The default is None.
    scale : `float`, optional
        factor by which to scale the CDELT keyword.
        The default is None.

    Returns
    -------
    output_wcs : `astropy.wcs.WCS`
        The tweaked WCS.

    """
    output_wcs = input_wcs.deepcopy()

    if shift_deg is not None:
        output_wcs.wcs.crval += shift_deg
    if shift_pix is not None:
        output_wcs.wcs.crpix += shift_pix
    if rotate is not None:
        cosr = np.cos(rotate*np.pi/180.0)
        sinr = np.sin(rotate*np.pi/180.0)
        R = np.array([[cosr, -sinr], [sinr, cosr]])
        output_wcs.wcs.pc = np.matmul(output_wcs.wcs.pc, R)
    if scale is not None:
        output_wcs.wcs.cdelt *= scale
    return output_wcs


def plot_errors(true_wcs, solved_wcs, naxis_1, naxis_2, obsid: int,
                topstring=None, plot: bool = False, plot_path: str = ""):
    xx, yy = np.meshgrid(np.arange(naxis_1), np.arange(naxis_2))
    c1 = true_wcs.pixel_to_world(xx, yy)
    c2 = solved_wcs.pixel_to_world(xx, yy)
    err = (c1.separation(c2)).to(u.arcsec)
    dra, ddec = c1.spherical_offsets_to(c2)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.add_subplot(1, 4, 1)
    im = ax.imshow(err.to(u.arcsecond).value, origin='lower')
    ax.set_title('Total Error')
    # ax.set_axis_off()
    fig.colorbar(im, label='arcsec', ax=ax, shrink=0.5)

    ax3 = fig.add_subplot(1, 4, 2)
    im3 = ax3.imshow(dra.to(u.arcsecond).value, origin='lower')
    ax3.set_title('RA Error')
    # ax3.set_axis_off()
    fig.colorbar(im3, label='arcsec', ax=ax3, shrink=0.5)

    ax4 = fig.add_subplot(1, 4, 3)
    im4 = ax4.imshow(ddec.to(u.arcsecond).value, origin='lower')
    ax4.set_title('Dec Error')
    # ax4.set_axis_off()
    fig.colorbar(im4, label='arcsec', ax=ax4, shrink=0.5)

    yh, bins = np.histogram(err.flatten(), bins=100)
    ax2 = fig.add_subplot(1, 4, 4)
    ax2.plot(0.5*(bins[1:]+bins[:-1]), yh, label='linear fit')
    mm = np.median(err.to(u.arcsecond).value)
    ax2.axvline(x=mm, ls=':', label='Median')
    ax2.set_title(f'position error rms ={np.mean(err):0.3f} ')
    ax2.set_xlabel('offset [arcsec]')
    ax2.legend()

    if topstring is not None:
        fig.suptitle(topstring)
    if plot is True:
        plt.show()
        plt.savefig(f"{plot_path}error_hist_{obsid}.png")
    else:
        plt.close(fig)

    astr_errlow, astr_errup = np.nanpercentile(err.to(u.arcsecond).value.flatten(), q=(16, 84))

    return np.mean(err.value), astr_errlow, astr_errup


def remove_sip_from_wcs(spximg):
    # Remove SIP from image header for SCAMP
    wcs_mod = spximg.wcs
    wcs_mod_header = wcs_mod.to_header(relax=True)
    logger.info("\nInput image header:")
    logger.info(repr(wcs_mod_header))
    # Modifying SIP in header
    wcs_mod_header["CTYPE1"] = "RA---TAN"
    wcs_mod_header["CTYPE2"] = "DEC--TAN"
    # Remove SIP coefficients by deleting SIP-related keys.
    # It's important to remove all these keys, and
    # not just set A_ORDER and B_ORDER to zero.
    sip_keys = ["A_ORDER", "B_ORDER", "A_0_2", "A_1_1", "A_2_0", "B_0_2", "B_1_1", "B_2_0",
                "AP_ORDER", "BP_ORDER", "AP_0_2", "AP_1_1", "AP_2_0", "BP_0_2", "BP_1_1", "BP_2_0"]
    for key in sip_keys:
        if key in wcs_mod_header:
            del wcs_mod_header[key]
    # Removing keys for higher order terms as well.
    for key in list(wcs_mod_header.keys()):
        if key.startswith('A_') or key.startswith('B_') or key.startswith('AP_') or key.startswith('BP_'):
            del wcs_mod_header[key]
    # Create a new WCS object
    logger.info(repr(wcs_mod_header))
    spximg.wcs = WCS(wcs_mod_header)
    logger.info("\nModified Header:")
    logger.info(repr(spximg.wcs.to_header(relax=True)))
    return spximg


def write_temp_img(spximg, obsid, array_id):
    img = SPHERExImage(data=spximg.data, wcs=spximg.wcs, unit=spximg.unit)
    img_path = get_temp_path(prefix=f"{obsid}_{array_id}_spximg_", suffix=".fits")
    img.write(img_path, overwrite=True)
    return img_path


def apply_aberration_correction(spximg: SPHERExImage, ref_cat: Table, verbose=False) -> Table:
    """Apply aberration correction to reference catalog
       using spacecraft position and velocity.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Image object containing spacecraft metadata and exposure time.
    ref_cat : `astropy.table.Table`
        Reference catalog table with 'ra' and 'dec' columns.
    verbose : `bool`
        Whether to print verbose output, default is False.

    Returns
    -------
    ref_cat : `astropy.table.Table`
        Reference catalog with aberration-corrected coordinates.
    """
    try:
        # S/C position and velocity info for aberration.
        posvec_x = spximg.header["X_SC"] * u.kilometer
        posvec_y = spximg.header["Y_SC"] * u.kilometer
        posvec_z = spximg.header["Z_SC"] * u.kilometer
        velvec_x = spximg.header["VX_SC"] * u.kilometer / u.second
        velvec_y = spximg.header["VY_SC"] * u.kilometer / u.second
        velvec_z = spximg.header["VZ_SC"] * u.kilometer / u.second
        if verbose:
            print("Used Position and Velocity from Image Header.")
    except KeyError:
        print("Position and Velocity not found, continuing without aberration correction...")
        return ref_cat

    # Construct ICRS coordinates from reference catalog
    # Unit handling (legacy in working with sim vs data)
    if ref_cat["ra"].unit is None:
        ref_coords = SkyCoord(ref_cat["ra"] * u.deg, ref_cat["dec"] * u.deg, frame="icrs")
    elif ref_cat["ra"].unit == u.deg:
        ref_coords = SkyCoord(ref_cat["ra"], ref_cat["dec"], frame="icrs")

    # Position vector in km, geocentric
    spx_posvec = u.Quantity([posvec_x, posvec_y, posvec_z])
    # Velocity vector in km/s, geocentric
    spx_velvec = u.Quantity([velvec_x, velvec_y, velvec_z])
    # SPHEREx observer GCRS frame
    gcrs_frame = GCRS(obstime=spximg.getExposureStartTime(),
                      obsgeoloc=spx_posvec,
                      obsgeovel=spx_velvec)
    # Define ICRS
    icrs = coord.ICRS()

    # Transform the reference catalog ICRS coordinates to the SPHEREx-GCRS.
    ref_cat_spx = ref_coords.transform_to(gcrs_frame)

    # ICRS -> GCRS -> ICRS round trip (sanity check)
    ref_cat_icrs = ref_cat_spx.transform_to(icrs)
    # Assert we were able to transform back to original
    assert np.allclose(ref_cat_icrs.ra.deg, ref_coords.ra.deg)
    assert np.allclose(ref_cat_icrs.dec.deg, ref_coords.dec.deg)

    # Store aberration-corrected coordinates
    ref_cat["ra_aberr"] = ref_cat_spx.icrs.ra.deg
    ref_cat["dec_aberr"] = ref_cat_spx.icrs.dec.deg

    return ref_cat


class EstimateFineAstrometryConnections(pipeBase.PipelineTaskConnections,
                                        dimensions=("instrument", "exposure"),
                                        defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalogs.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    sip = cT.PrerequisiteInput(
        name="sip",
        doc="Input distortion as SIP (Simple Imaging Polynomial).",
        storageClass="AstropySIP",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after fine astrometry estimation.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class EstimateFineAstrometryConfig(pipeBase.PipelineTaskConfig,
                                   pipelineConnections=EstimateFineAstrometryConnections):
    min_ncalibrators = Field(
        doc="The minimum number of astrometric calibrators the module needs."
            " If the number of calibrators is below this threshold,"
            " the module will raise an ValueError.",
        dtype=int,
        default=10)

    detect_thresh_hib = DictField(
        doc="SExtractor detect_thresh parameter, high |b|.",
        keytype=int,
        itemtype=str,
        default={1: "200", 2: "200", 3: "200", 4: "200", 5: "100", 6: "15"})

    cross_radius_hib = DictField(
        doc="SCAMP cross_radius parameter, high |b|.",
        keytype=int,
        itemtype=str,
        default={1: "20.0", 2: "20.0", 3: "20.0", 4: "20.0", 5: "20.0", 6: "20.0"})

    detect_thresh_lowb = DictField(
        doc="SExtractor detect_thresh parameter, low |b|.",
        keytype=int,
        itemtype=str,
        default={1: "200", 2: "200", 3: "200", 4: "200", 5: "100", 6: "35"})

    cross_radius_lowb = DictField(
        doc="SCAMP cross_radius parameter, low |b|.",
        keytype=int,
        itemtype=str,
        default={1: "20.0", 2: "20.0", 3: "20.0", 4: "20.0", 5: "20.0", 6: "20.0"})

    detect_minarea = DictField(
        doc="SExtractor detect_minarea parameter.",
        keytype=int,
        itemtype=str,
        default={1: "2.0", 2: "2.0", 3: "2.0", 4: "2.0", 5: "2.0", 6: "2.0"})

    detect_maxarea = DictField(
        doc="SExtractor detect_maxarea parameter.",
        keytype=int,
        itemtype=str,
        default={1: "5.0", 2: "5.0", 3: "5.0", 4: "5.0", 5: "5.0", 6: "5.0"})

    position_maxerr = Field(
        doc="SCAMP position_maxerr parameter",
        dtype=str,
        default="70.0")  # 70, 3

    posangle_maxerr = Field(
        doc="SCAMP posangle_maxerr parameter",
        dtype=str,
        default="10")  # 10, 3

    pixscale_maxerr = Field(
        doc="SCAMP pixscale_maxerr parameter",
        dtype=str,
        default="1.2")  # 1.2

    astrclip_nsigma = Field(
        doc="Astrom. clipping threshold in sigmas",
        dtype=str,
        default="3.0")

    plot_path = Field(
        doc="The location to save plots.",
        dtype=str,
        default="")

    pix_scale = Field(
        doc="SPHEREx pixel scale.",
        dtype=float,
        default=6.15)

    distort_degrees = Field(
        doc="Distortion solution in SCAMP.",
        dtype=str,
        default="3")

    pv_to_sip = Field(
        doc="Convert PV to SIP",
        dtype=bool,
        default=True)

    pc_formalism = Field(
        doc="PC with CDELT matrix formalism",
        dtype=bool,
        default=False)  # NOTE needs validation!

    do_aberr_corr = Field(
        doc="Do aberration correction.",
        dtype=bool,
        default=False
    )


class EstimateFineAstrometryTask(pipeBase.PipelineTask):
    """Implementation of L2 Estimate Fine Astrometry.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-016

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "estimate-fine-astrometry"
    ConfigClass = EstimateFineAstrometryConfig
    full_name = "Estimate Fine Astrometry"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-016 v2.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            srccat: Table,
            index_dir: str = None,
            sip: Sip | None = None,
            max_sep: float = 1.00,
            naxis_1: int = 2040,
            naxis_2: int = 2040,
            verbose: bool = True,
            plot: bool = True) -> pipeBase.Struct:
        """

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            SPHERExImage with the necessary meta and catalog
        srccat : `~astropy.table.Table`
            Source catalog
        sip : `~astropy.wcs.Sip`
            Simple Image Polynomials, representing distortion
        max_sep : float
            Max separation in arcseconds.
        naxis_1 : `int`
            NAXIS1.
        naxis_2 : `int`
            NAXIS2.
        verbose : `bool`
            Request verbose feedback

        Returns
        -------
        output : `~lsst.pipe.base.Struct`
            Output with `~spx_data.SPHERExImage` with the adjusted WCS

        Notes
        -----
        - Module receives initial WCS and distortion parameters
          (from Estimate coarse astrometry)
        - ICRS = International Celestial Reference System
        - GCRS = Geocentric Celestial Reference System
        - See: https://spherex-docs.ipac.caltech.edu/ssdc-tr-030/v/v1.5/
        """
        print(self.config.pv_to_sip)
        print(self.config.pc_formalism)
        if index_dir is None:
            index_dir = os.getenv("ASTRONET_INDEX_DIR")
            if index_dir is None:
                raise RuntimeError(
                    "ASTRONET_INDEX_DIR environment variable is not set and no index_dir was provided. "
                    "Please set ASTRONET_INDEX_DIR or pass index_dir explicitly.")

        obsid = spximg.header["OBSID"]
        array_id = spximg.getArrayId()
        if len(srccat) == 0:
            raise ValueError("Bad srccat input.")
        # Check astrometric calibrators exist
        ncalibrators = len(srccat[srccat["spherex_class"] & SourcePipelineClass.ASTROMETRIC > 0])
        print("---------------------------------------------------------")
        print(f"Number of astrometric calibrators found: {ncalibrators}")
        print("---------------------------------------------------------")
        nmin = self.config.min_ncalibrators
        if ncalibrators < nmin:
            self.log.error(f"Insufficient astrometric calibrators found: {ncalibrators}"
                           f" of {nmin} required. Returning image and wcs.")
            spximg.header["FINAST"] = 4
            spximg.header.comments['FINAST'] = "Insufficient calibrators"
            return pipeBase.Struct(spximg_out=spximg)

        # Modified to rerun fine astrometry twice with error handling for timeout.
        try:
            result = self._run_astrometry(spximg, srccat, index_dir, sip, max_sep,
                                          naxis_1, naxis_2, verbose, plot)
            return result
        except Exception as e:
            print(f"FINE ASTROMETRY FAILED {e} SPHEREx OBSID {spximg.header['OBSID']}")
            spximg.header["FINAST"] = 3
            spximg.header.comments['FINAST'] = "failed fine-astrometric solution"
            return pipeBase.Struct(spximg_out=spximg)
        finally:
            clean_all_tempfiles(obsid, array_id)

    def _run_astrometry(self, spximg, srccat, index_dir, sip, max_sep, naxis_1, naxis_2,
                        verbose, plot, iter_2: bool = True) -> pipeBase.Struct:
        array_id = spximg.getArrayId()
        obsid = spximg.header['OBSID']
        # Make copy of input SPHEREx image.
        spximg_input = copy.deepcopy(spximg)

        # Pre-processing for Estimate Fine Astrometry:
        self.background_sub(spximg)

        # Determine |b| to select appropriate config parameters
        r, d = spximg.wcs.wcs.crval[0], spximg.wcs.wcs.crval[1]
        sky_coords = SkyCoord(ra=r*u.deg, dec=d*u.deg, frame='icrs')
        galactic_coords = sky_coords.galactic
        beta = galactic_coords.b.deg

        if spximg.wcs.wcs.has_pc():
            _pc = spximg.wcs.wcs.pc
            _cdelt = spximg.wcs.wcs.cdelt
            _cd = pc_to_cd(_pc, _cdelt)
            logging.info(f"cd initial = {_cd}")
            logging.info(f"pc = {_pc}")
            logging.info(f"cdelt =  {_cdelt}")
            del spximg.wcs.wcs.pc
            spximg.wcs.wcs.cdelt = [1.0, 1.0]
            spximg.wcs.wcs.cd = _cd

        logging.info(f"cd initial = {spximg.wcs.wcs.cd}")
        logging.info(f"spximg.wcs : {spximg.wcs.to_header(relax=True)}")
        # Input image should have initial SIP or updated SIP from calibration.
        if '-SIP' not in spximg.wcs.wcs.ctype[0] or '-SIP' not in spximg.wcs.wcs.ctype[1]:
            raise ValueError("SIP is not available in WCS: not acceptable.")

        # variables for temp file absolute paths
        img_path1 = write_temp_img(spximg, obsid, array_id)
        img_ldac = None
        refcat_ldac = None
        scamp_out = None

        # Define output paths for astronet
        img_stripped = get_temp_path(prefix=f"{obsid}_{array_id}_stripped_img_", suffix=".fits")
        wcs_output = get_temp_path(prefix=f"{obsid}_{array_id}_tmp_wcs_", suffix=".fits")

        # Get path to astrometry.net index files
        if not os.path.exists(index_dir):
            raise ValueError("Missing index files")

        ###############################################################
        # Run Astrometry.net before se_run
        ###############################################################
        astronet_run_pass = False
        try:
            print(f"Running astrometry.net on SPXOBSID {spximg.header['OBSID']}")
            wcs_output, stripped_path = astronet_run(
                input_image=img_path1,
                stripped_image=img_stripped,
                wcs_output=wcs_output,
                index_dir=index_dir,
                crpix=(spximg.wcs.wcs.crpix[0], spximg.wcs.wcs.crpix[1]),
                pixel_scale_bounds=(6.0, 6.3),
                tweak_order=3,
                pixel_error=2,
                downsample=2,
                extension=0,
                verbose=verbose,
                array_id=array_id)
            print(f"wcs_output = {wcs_output}")
            print(f"stripped_path = {stripped_path}")

            # Load updated WCS from solve-field and assign to spximg
            with fits.open(wcs_output) as wcs_hdul:
                solved_wcs = WCS(wcs_hdul[0].header)
                print(solved_wcs)
                spximg.wcs = solved_wcs
                astronet_run_pass = True
        except Exception as e:
            print(f"[WARNING] astrometry.net failed on OBSID {spximg.header['OBSID']}: {e}")
            print("[INFO] Using original image WCS for fine astrometry.")
            # fall back to original WCS
            spximg.wcs = spximg_input.wcs
        finally:
            clean(img_path1, wcs_output, img_stripped)

        # Now write the updated SPHERExImage to file
        img_path2 = write_temp_img(spximg, obsid, array_id)

        ###############################################################
        # Transform Reference Catalog:
        # from barycenter-centric to SPHEREx-centric coordinate system.
        ###############################################################
        # Reference catalog.
        ref_cat = preprocess_catalog(srccat,
                                     spximg,
                                     srcPipeFlag=SourcePipelineClass.ASTROMETRIC,
                                     addPixelCoords=True,
                                     pixelCoordsWithDistortion=True,
                                     filterSourcesOutsideImage=False)

        self.log.info(f"len(ref_cat) = {len(ref_cat)}")

        #  Aberration corection
        if self.config.do_aberr_corr:
            ref_cat = apply_aberration_correction(spximg, ref_cat, verbose=verbose)
            print("Aberration correction applied. Example RA before/after:")
            print(ref_cat["ra"][:3], ref_cat["ra_aberr"][:3])
            print(ref_cat["dec"][:3], ref_cat["dec_aberr"][:3])

        ###############################################################
        # Reference catalog in LDAC format w/ needed headers for SCAMP.
        ###############################################################
        # See: https://faun.rc.fas.harvard.edu/ameisner/scamp.pdf
        # "X_WORLD" and "Y_WORLD" are the default ref cat
        # column names expected by SCAMP (pg 6):
        # ASTREFCENT_KEYS* X_WORLD, Y_WORLD strings Names
        # of the columns, in the local astrometric
        # reference catalogue(s), that contain the centroid
        # coordinates in degrees. Active only if
        # ASTREF CATALOG is set to FILE.
        if self.config.do_aberr_corr:
            ref_cat['X_WORLD'] = ref_cat['ra_aberr']
            ref_cat['Y_WORLD'] = ref_cat['dec_aberr']
        else:
            ref_cat['X_WORLD'] = ref_cat['ra']
            ref_cat['Y_WORLD'] = ref_cat['dec']
        # See SP-648 and SP-455:
        # two magnitude columns for the corresponding detectors are present
        # in the output of Query Reference Catalog module.
        # Should only use MAG from reference catalog,
        # Estimations have already been done.
        det_id = str(spximg.getArrayId())
        ref_cat['MAG'] = ref_cat['mag_det'+det_id]
        ref_cat['MAGERR'] = 0.1  # Need in output of query refcat.
        size = len(ref_cat)
        dum_erra = dum_errb = np.ones((size,))*0.005/3600  # This is in degrees, same unit as X/Y_WORLD.
        dum_errtheta = np.ones((size,))
        ref_cat['ERRA_WORLD'] = dum_erra
        ref_cat['ERRB_WORLD'] = dum_errb
        ref_cat['ERRTHETA_WORLD'] = dum_errtheta
        refcat_ldac = get_temp_path(prefix=f"{obsid}_{array_id}_refcat_", suffix=".ldac")

        # SELECT SOURCES FROM REFCAT AND SAVE TO LDAC
        # NOTE change mag based on number of calibrators
        # such that number or sources < 8000
        # 8000 is max for SCAMP
        # sel_calibrators =
        # np.where((ref_cat["MAG"] > 10) & (ref_cat["MAG"] < 15) # 13.5
        # & (ref_cat["dist_nearest"] > 0))[0]  # select isolated sources
        # ref_cat_use = ref_cat[sel_calibrators]
        # # ref_cat_all = ref_cat.copy()
        # ref_cat = ref_cat_use.copy()
        # logger.info("{} sources selected out of"
        # "{} in the reference catalog".format(len(ref_cat_use), len(ref_cat)))
        # num_refcat = len(ref_cat)

        min_sources = 2000
        max_sources = 10000
        mag_lower = 10.0
        mag_upper = 15.0  # start with a reasonable default
        mag_step = 0.25   # change this for finer or coarser steps
        max_iter = 20     # prevent infinite loop

        for i in range(max_iter):
            sel_calibrators = np.where((ref_cat["MAG"] > mag_lower)
                                       & (ref_cat["MAG"] < mag_upper)
                                       & (ref_cat["dist_nearest"] > 0))[0]
            ref_cat_use = ref_cat[sel_calibrators]
            n = len(ref_cat_use)

            logger.info(f"Iter {i}: {n} sources selected with MAG < {mag_upper:.2f}")

            if min_sources <= n <= max_sources:
                break
            elif n < min_sources:
                mag_upper += mag_step  # widen magnitude range to get more stars
            else:
                mag_upper -= mag_step  # narrow magnitude range to get fewer stars

        # Final selection
        ref_cat = ref_cat_use.copy()
        print(f"Final selection: {len(ref_cat)} sources with MAG < {mag_upper:.2f}")
        catalog2ldac(ref_cat, refcat_ldac)

        contrast = 0.0
        MAX_ITER = 20
        iteration = 0
        se_thresh = 0
        daxis1 = np.inf
        daxis2 = np.inf

        # Initial detect threshold for SExtractor
        if np.abs(beta) > 10:
            detect_thresh = self.config.detect_thresh_hib[array_id]
            print(f".detect_thresh_hib used: {detect_thresh}")
        else:
            detect_thresh = self.config.detect_thresh_lowb[array_id]
            print(f".detect_thresh_lowb used: {detect_thresh}")

        while (daxis1 > 1.5 or daxis2 > 1.5) and iteration < MAX_ITER:
            print(f"Iteration [{iteration}]: using detect_thresh = {detect_thresh}")
            # Step 1: Write temp image with current WCS
            img_path = write_temp_img(spximg, obsid, array_id)

            # Step 2: Run SExtractor with current image WCS
            img_ldac = get_temp_path(prefix=f"{obsid}_{array_id}_imgcat_{iteration}_", suffix=".ldac")
            if iteration == 0:
                override_loop = False
            else:
                override_loop = True
            try:
                se_thresh = se_run(img_path,
                                   img_ldac,
                                   num_ref_cat=len(ref_cat),
                                   detect_thresh=detect_thresh,
                                   override_loop=override_loop,
                                   detect_minarea=self.config.detect_minarea[array_id],
                                   detect_maxarea=self.config.detect_maxarea[array_id],
                                   verbose=True)
            except Exception as e:
                raise ValueError(f"SExtractor failed {e} SPHEREx {spximg_input.header['OBSID']}")

            # update detect_thresh
            detect_thresh = se_thresh

            # Step 3: Run SCAMP
            scamp_out = get_temp_path(prefix=f"{obsid}_{array_id}_fulloutcat_{iteration}_", suffix=".fits")
            if np.abs(beta) > 10:
                cross_radius = self.config.cross_radius_hib[array_id]
                print(f"hi |b|>10 cross radius used: {cross_radius}")
            else:
                cross_radius = self.config.cross_radius_lowb[array_id]
                print(f"hi |b|<=10 cross radius used: {cross_radius}")
            (contrast,
             detections_loaded,
             standards_loaded,
             ra_shift,
             dec_shift,
             daxis1,
             daxis2,
             nstars) = scamp_run(img_ldac,
                                 refcat_ldac,
                                 scamp_out,
                                 verbose=True,
                                 position_maxerr=self.config.position_maxerr,
                                 posangle_maxerr=self.config.posangle_maxerr,
                                 pixscale_maxerr=self.config.pixscale_maxerr,
                                 cross_radius=cross_radius,
                                 astrclip_nsigma=self.config.astrclip_nsigma,
                                 distort_degrees=self.config.distort_degrees)

            print(f"[{iteration}] SCAMP contrast = {contrast}")

            # Step 5: Update WCS for next iteration
            head_path = img_ldac[:-4] + "head"
            new_wcs = wcs_from_head(head_path)
            spximg.wcs = new_wcs
            print(f">>>>>> dAXIS1 = {daxis1}\", dAXIS2 = {daxis2}\"")
            if daxis1 < 1.0 and daxis2 < 1.0:
                break
            elif daxis1 < 1.5 and daxis2 < 1.5:
                print(f"[{iteration}] Accepting early SCAMP solution with contrast={contrast:.2f} "
                      f"and shifts RA={ra_shift:.2f}\", DEC={dec_shift:.2f}\""
                      f" dAXIS1 = {daxis1}\", dAXIS2 = {daxis2}\"")
                break

            print(f"ITERATION {iteration}")
            iteration += 1

        if contrast < 1.0 and (abs(ra_shift) > 1.0 or abs(dec_shift) > 1.0):
            raise ValueError(
                f"WARNING: SCAMP contrast ({contrast}) did not converge after max iterations.\n"
                f"SPHEREx image {spximg_input.header['OBSID']}\n"
                f"CRVAL1 = {spximg_input.wcs.wcs.crval[0]} CRVAL2 = {spximg_input.wcs.wcs.crval[1]}\n"
                f"SCAMP solution has shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"\n"
                f"detections_loaded = {detections_loaded}\n"
                f"standards loaded = {standards_loaded}\n"
                f"SExtractor detect thresh = {detect_thresh}\n"
                f"nstars matched = {nstars}\n")
        elif contrast < 2.0 and contrast > 1.0 and (abs(ra_shift) > 1.0 or abs(dec_shift) > 1.0):
            raise ValueError(
                f"WARNING: SCAMP contrast ({contrast}) did not converge after max iterations.\n"
                f"SPHEREx image {spximg_input.header['OBSID']}\n"
                f"CRVAL1 = {spximg_input.wcs.wcs.crval[0]} CRVAL2 = {spximg_input.wcs.wcs.crval[1]}\n"
                f"SCAMP solution has shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"\n"
                f"detections_loaded = {detections_loaded}\n"
                f"standards loaded = {standards_loaded}\n"
                f"SExtractor detect thresh = {detect_thresh}\n"
                f"nstars matched = {nstars}\n")
        elif contrast < 1.0 and abs(ra_shift) < 1.0 and abs(dec_shift) < 1.0:
            raise ValueError(
                f"WARNING: SCAMP contrast ({contrast}) did not converge after max iterations.\n"
                f"SPHEREx image {spximg_input.header['OBSID']}\n"
                f"CRVAL1 = {spximg_input.wcs.wcs.crval[0]} CRVAL2 = {spximg_input.wcs.wcs.crval[1]}\n"
                f"SCAMP solution has shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"\n"
                f"detections_loaded = {detections_loaded}\n"
                f"standards loaded = {standards_loaded}\n"
                f"SExtractor detect thresh = {detect_thresh}\n"
                f"nstars matched = {nstars}\n")
        print(
            f"1SCAMP contrast ({contrast}).\n"
            f"SPHEREx image {spximg_input.header['OBSID']}\n"
            f"CRVAL1 = {spximg_input.wcs.wcs.crval[0]} CRVAL2 = {spximg_input.wcs.wcs.crval[1]}\n"
            f"SCAMP solution has shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"\n"
            f"detections_loaded = {detections_loaded}\n"
            f"standards loaded = {standards_loaded}\n"
            f"SExtractor detect thresh = {detect_thresh}\n"
            f"nstars matched = {nstars}\n")

        # ###############################################################
        # # Select bright sources on given exposure w/ SExtracotr.
        # ###############################################################
        # # Define the output catalog name
        # img_ldac = get_temp_path(prefix="imgcat_", suffix=".ldac")
        # if np.abs(beta) > 10:
        #     detect_thresh = self.config.detect_thresh_hib[array_id]
        #     print(f".detect_thresh_hib used: {detect_thresh}")
        # else:
        #     detect_thresh = self.config.detect_thresh_lowb[array_id]
        #     print(f".detect_thresh_lowb used: {detect_thresh}")
        # try:
        #     se_run(img_path2,
        #            img_ldac,
        #            num_ref_cat=len(ref_cat),
        #            detect_thresh=detect_thresh,
        #            detect_minarea=self.config.detect_minarea[array_id],
        #            detect_maxarea=self.config.detect_maxarea[array_id],
        #            verbose=True)

        # except Exception as e:
        #     clean(img_path2, img_ldac, refcat_ldac, scamp_out)
        #     self.log.error(f"SExtractor run failed: {e}")
        #     raise ValueError("SE failed")

        # # Read catalog
        # # Only use the XWIN_IMAGE and YWIN_IMAGE. Note that the
        # # ALPHA_J2000 and DELTA_J2000 are using the image WCS.
        # sextracted_sources_count = len(Table.read(img_ldac, hdu=2))
        # print("Number of extracted sources: {}".format(sextracted_sources_count))

        # ###############################################################
        # # Initial matching between img_cat and ref_cat with SCAMP.
        # ###############################################################
        # scamp_out = get_temp_path(prefix="fulloutcat_", suffix=".fits")
        # if np.abs(beta) > 10:
        #     cross_radius = self.config.cross_radius_hib[array_id]
        #     print(f"hi |b|>10 cross radius used: {cross_radius}")
        # else:
        #     cross_radius = self.config.cross_radius_lowb[array_id]
        #     print(f"hi |b|<=10 cross radius used: {cross_radius}")
        # try:
        #     print(">>>>> FIRST SCAMP <<<<<")
        #     scamp_run(img_ldac,
        #               refcat_ldac,
        #               scamp_out,
        #               verbose=True,
        #               position_maxerr=self.config.position_maxerr,
        #               posangle_maxerr=self.config.posangle_maxerr,
        #               pixscale_maxerr=self.config.pixscale_maxerr,
        #               cross_radius=cross_radius,
        #               astrclip_nsigma=self.config.astrclip_nsigma,
        #               distort_degrees=self.config.distort_degrees)
        # except subprocess.CalledProcessError as e:
        #     clean(scamp_out, None, None, None)

        #     self.log.error(f"SCAMP error {e.stderr}.")
        #     raise ValueError(f"SCAMP error {e.stderr}.")

        spximg_temp = copy.copy(spximg_input)
        print(spximg_temp.wcs.to_header(relax=True))
        # Get SCAMP WCS (aka solved wcs)
        head_path = img_ldac[:-4] + "head"
        scamp_wcs = wcs_from_head(head_path)
        print(">>>> SCAMP_WCS <<<")
        print(scamp_wcs)
        scamp_hdr = scamp_wcs.to_header(relax=True)
        spximg_temp.wcs = scamp_wcs
        spximg_temp.header = scamp_hdr
        print(spximg_temp.wcs.to_header(relax=True))

        # variables for temp file absolute paths
        img_path3 = write_temp_img(spximg_temp, obsid, array_id)
        ###############################################################
        # Select bright sources on given exposure w/ SExtracotr.
        ###############################################################
        # Define the output catalog name
        img_ldac = get_temp_path(prefix="imgcat2_", suffix=".ldac")
        detect_thresh = se_thresh/1.5
        _ = se_run(img_path3,
                   img_ldac,
                   num_ref_cat=len(ref_cat),
                   detect_thresh=str(se_thresh/1.5),
                   detect_minarea=self.config.detect_minarea[array_id],
                   detect_maxarea=self.config.detect_maxarea[array_id],
                   verbose=True,
                   override_loop=True,
                   )

        sextracted_sources_count = len(Table.read(img_ldac, hdu=2))
        logger.info("Number of extracted sources:{}".format(sextracted_sources_count))

        # RUN SCAMP AGAIN!
        print('RUN SCAMP AGAIN!')
        scamp_out = get_temp_path(prefix="fulloutcat2_", suffix=".fits")
        if np.abs(beta) > 10:
            cross_radius = self.config.cross_radius_hib[array_id]
            self.log.info(f"hi |b|>10 cross radius used: {cross_radius}")
        else:
            cross_radius = self.config.cross_radius_lowb[array_id]
            self.log.info(f"hi |b|<=10 cross radius used: {cross_radius}")
        try:
            print(f"astrclip_nsigma = {self.config.astrclip_nsigma}")
            print(">>>>> SECOND SCAMP <<<<<")
            (contrast,
             detections_loaded,
             standards_loaded,
             ra_shift,
             dec_shift,
             daxis1,
             daxis2,
             nstars) = scamp_run(img_ldac,
                                 refcat_ldac,
                                 scamp_out,
                                 verbose=True,
                                 position_maxerr=self.config.position_maxerr,
                                 posangle_maxerr=self.config.posangle_maxerr,
                                 pixscale_maxerr=self.config.pixscale_maxerr,
                                 cross_radius=cross_radius,
                                 astrclip_nsigma=self.config.astrclip_nsigma,
                                 distort_degrees=self.config.distort_degrees)
        except subprocess.CalledProcessError as e:
            self.log.error(f"SCAMP error {e.stderr}.")
            raise ValueError(f"SCAMP error {e.stderr}.")
        finally:
            clean(img_path3, img_ldac, refcat_ldac, scamp_out)

        print(
            f"2SCAMP contrast ({contrast}).\n"
            f"SPHEREx image {spximg_input.header['OBSID']}\n"
            f"CRVAL1 = {spximg_input.wcs.wcs.crval[0]} CRVAL2 = {spximg_input.wcs.wcs.crval[1]}\n"
            f"SCAMP solution has shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"\n"
            f"detections_loaded = {detections_loaded}\n"
            f"standards loaded = {standards_loaded}\n"
            f"SExtractor detect thresh = {detect_thresh}\n"
            f"nstars matched = {nstars}\n")

        spximg_out = spximg_input
        # Get SCAMP WCS (aka solved wcs)
        head_path = img_ldac[:-4] + "head"
        scamp_wcs = wcs_from_head(head_path)
        scamp_hdr = scamp_wcs.to_header(relax=True)

        print(">>>> SCAMP_WCS <<<")
        print(scamp_wcs)

        # Create New Image
        hdr_new = copy.deepcopy(spximg_temp.header)
        hdr_new.update(scamp_hdr)

        # # TPV WCS
        # spximg_out.wcs = scamp_wcs
        # print(spximg_out.wcs.to_header(relax=True))
        # final_wcs = scamp_wcs

        # wcs_pv = copy.deepcopy(scamp_wcs)
        # wcs_sip = copy.deepcopy(scamp_wcs)
        # if not hasattr(wcs_pv.wcs, 'pc'):
        #     # it's CD format, change to PC
        #     cdelt = np.array([self.config.pix_scale/3600, self.config.pix_scale/3600])
        #     pc = cd_to_pc(wcs_pv.wcs.cd, cdelt)
        #     wcs_sip.wcs.pc = pc
        #     del wcs_sip.wcs.cd
        #     wcs_sip.wcs.cdelt = cdelt

        # final_wcs = wcs_sip  # still have PV values here
        # print(">>>> FINAL_WCS <<<")
        # print(final_wcs)
        # # Now, calculate PA from PC matrix
        # pa_eq = calculate_pa_from_pc(final_wcs)

        # Converstion back to SIP WCS
        if self.config.pv_to_sip is True:
            head_file = None
            sip_wcs_file_output = None
            try:
                # Create temporary file paths
                head_file = get_temp_path(prefix="img_tpv_", suffix=".fits")  # input to wcs-pv2sip
                sip_wcs_file_output = get_temp_path(prefix="img_sip_", suffix=".fits")  # output header
                # Save TPV WCS-only FITS file (no data needed)
                # hdr = wcs_sip.to_header(relax=True)
                # hdr = spximg_out.wcs.to_header(relax=True)
                # hdr = spximg_out.header.copy()
                # wcs_header = spximg_out.wcs.to_header(relax=True)
                # hdr.update(wcs_header)
                dummy_data = np.zeros((2040, 2040), dtype=np.float32)
                hdu = fits.PrimaryHDU(data=dummy_data, header=hdr_new)
                hdu.writeto(head_file, overwrite=True)
                # Convert TPV -> SIP
                pv_cmd = f"wcs-pv2sip -o {self.config.distort_degrees} -e 0 {head_file} {sip_wcs_file_output}"
                subprocess.run(pv_cmd, shell=True, check=True)

                # Get SIP WCS with header
                with fits.open(sip_wcs_file_output) as hdul:
                    final_wcs = WCS(hdul[0].header)

                spximg_out.wcs = final_wcs  # SIP
                print(spximg_out.wcs.to_header(relax=True))
            except subprocess.CalledProcessError as e:
                self.log.error(f"wcs-pv2sip error {e.stderr}.")
                raise ValueError(f"wcs-pv2sip error {e.stderr}.")
            finally:
                clean(head_file, sip_wcs_file_output)
        else:
            spximg_out.wcs = scamp_wcs

        cdelt = np.array([self.config.pix_scale/3600, self.config.pix_scale/3600])
        if self.config.pc_formalism:
            if not hasattr(spximg_out.wcs, 'pc'):  # CD format
                # it's CD format, change to PC
                pc = cd_to_pc(spximg_out.wcs.wcs.cd, cdelt)
                spximg_out.wcs.pc = pc
                del spximg_out.wcs.wcs.cd
                spximg_out.wcs.wcs.cdelt = cdelt

        print(">>>> FINAL_WCS <<<")
        final_wcs = spximg_out.wcs
        print(final_wcs)

        spximg_out.wcs.array_shape = spximg_out.data.shape
        # Calculate solid angles with linear WCS solution and current SIPs
        solid_angle_pixel_map = calculate_solid_angle(final_wcs, array_id, plot=False)
        # Calculate median and std of solid angle map
        solid_angle_median, solid_angle_stddev = calculate_median_std_solid_angle(solid_angle_pixel_map)
        print(f"solid angle = {solid_angle_median}, {solid_angle_stddev}")

        # Add solid angle median and std dev to L2 image header
        try:
            spximg_out.header['HIERARCH OMEGA_MEDIAN'] = solid_angle_median
            spximg_out.header.comments['HIERARCH OMEGA_MEDIAN'] = " [arcsec**2] solid angle pixel map median"
            spximg_out.header['HIERARCH OMEGA_STDDEV'] = solid_angle_stddev
            spximg_out.header.comments['HIERARCH OMEGA_STDDEV'] = " [arcsec**2] solid angle pixel map std"
        except Exception as e:
            spximg_out.header['HIERARCH OMEGA_MEDIAN'] = 0
            spximg_out.header.comments['HIERARCH OMEGA_MEDIAN'] = "error, nan encounterd"
            spximg_out.header['HIERARCH OMEGA_STDDEV'] = 0
            spximg_out.header.comments['HIERARCH OMEGA_STDDEV'] = "error, nan encounterd"
            print(f"Exceptions {e}")

        if daxis1 < 1.5 and daxis2 < 1.5:
            spximg_out.header["FINAST"] = 0
            spximg_out.header.comments['FINAST'] = "good fine-astrometric solution"
        elif (daxis1 >= 1.5 and daxis1 < 5.0) or (daxis2 >= 1.5 and daxis2 < 5.0):
            spximg_out.header["FINAST"] = 1
            spximg_out.header.comments['FINAST'] = "poor fine-astrometric solution"
        elif daxis1 >= 5.0 or daxis2 >= 5.0:
            spximg_out.header["FINAST"] = 2
            spximg_out.header.comments['FINAST'] = "bad fine-astrometric solution"

        # If the fine astrometry solution is not good (FINAST > 0),
        # return the original WCS from coarse-astrometry which is in the
        # input image that was saved at the start of the function.
        if spximg_out.header["FINAST"] > 1:
            self.log.warning(f"Fine astrometry was not successful (FINAST={spximg_out.header['FINAST']}). "
                             "Returning original input image.")
            # Revert the WCS of the output object to the original WCS.
            spximg_out.wcs = spximg_input.wcs
            final_wcs = spximg_input.wcs

        # if spximg_out.header["FINAST"] == 0:
        #     # Need to calculate PA from CD matrix (not PC)
        #     # only change if finast is zero.
        #     pa_eq = position_angle_from_cd(final_wcs.wcs.cd)
        #     print(f"pa_eq = {pa_eq}")
        #     # Update PA after fine-astro linear wcs calculation
        #     spximg_out.header['PA_EQ'] = pa_eq
        #     spximg_out.header.comments['PA_EQ'] =
        # "PA_EQ: calculated from fine astrometry wcs (FINAST=0)"

        # add exposure center and boundary points to the task metadata
        spximg_out.wcs.array_shape = (naxis_2, naxis_1)
        add_exposure_boundary_info(self.metadata, spximg_out.wcs)
        self.log.info(f'spximg_out.wcs = {spximg_out.wcs}')

        # UPDATE SPHERExIMAGE META
        spximg_out.wcs.wcs.radesys = "ICRS"
        if hasattr(spximg_out.wcs.wcs, "equinox"):
            delattr(spximg_out.wcs.wcs, "equinox")
        spximg_out.meta.update(spximg_out.wcs.to_header())
        # Add comment to RADESYS keyword
        spximg_out.header.set("RADESYS", "ICRS", "Equatorial coordinates referenced to Gaia DR3")

        # Remove temporary output files from subprocesses.
        if astronet_run_pass:
            clean(img_path1, img_path2, img_ldac, refcat_ldac,
                  scamp_out, wcs_output, stripped_path,
                  img_ldac[:-4] + "head", scamp_out[:-5]+'_1.fits')
        else:
            clean(img_path1, img_path2, img_ldac, refcat_ldac,
                  scamp_out, wcs_output,
                  img_ldac[:-4] + "head", scamp_out[:-5]+'_1.fits')

        return pipeBase.Struct(spximg_out=spximg_out,
                               scamp_wcs=scamp_wcs)

    def background_sub(self, spximg):
        # Pre-processing for Estimate Fine Astrometry:
        # Background subtraction
        sigma_clip = SigmaClip(sigma=7.0)
        bkg_estimator = MedianBackground()
        bkg = Background2D(spximg.data, (50, 50), filter_size=(3, 3),
                           sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
        bkg_std = np.nanstd(bkg.background)
        if bkg_std > 0.05:
            spximg.data -= bkg.background
            self.log.info("Background subtraction performed")
        else:
            self.log.info("No background subtraction performed")


def clean(*file_paths):
    for file in file_paths:
        if file is None:
            continue
        try:
            os.remove(file)
        except FileNotFoundError:
            logger.info(f"File {file} not found, skipping.")
        except Exception as e:
            logger.warning(f"Failed to delete {file}: {e}")


def clean_all_tempfiles(obsid, array_id, prefixes=["spximg_", "imgcat_", "refcat_", "tmp_wcs_",
                                                   "stripped_img_", "fulloutcat_", "fulloutcat2_"]):
    temp_dir = get_ssdc_tempfiledir()
    for prefix in prefixes:
        prefix = f"{obsid}_{array_id}_"+prefix
        pattern = os.path.join(temp_dir, prefix + "*")
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {path}: {e}")
