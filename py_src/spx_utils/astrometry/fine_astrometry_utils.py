__all__ = ["astronet_run", "se_run", "scamp_run"]

# Standard library imports
import subprocess
from pathlib import Path
import os
import re

# Related third party imports
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.convolution import convolve, Gaussian2DKernel
import sep

# Local application/library specific imports
from spx_utils.utils import get_data_file

working_dir = Path(__file__).parent

SUBPROCESS_TIMEOUT = 300  # seconds


def astronet_run(input_image: str | Path,
                 stripped_image: str | Path,
                 wcs_output: str | Path,
                 index_dir: str,
                 crpix: tuple[float, float] = (1020.5, 1020.5),
                 pixel_scale_bounds: tuple[float, float] = (6.0, 6.3),
                 tweak_order: int = 3,
                 pixel_error: int = 2,
                 downsample: int = 2,
                 extension: int = 0,
                 radius: float = 10.0,
                 sigma_extraction: float = 100,
                 verbose: bool = True,
                 array_id: int = 1
                 ) -> None:
    """Run astrometry.net `solve-field` on a stripped source image.

    Parameters
    ----------
    input_image : str or Path
        Path to the FITS image with initial CRVALs.
    stripped_image : str or Path
        Output path for the SEP + Gaussian-convolved fake source image.
    wcs_output : str or Path
        Output path for the solved WCS header file.
    index_dir : str
        Path to the directory containing astrometry.net index files.
    crpix : tuple
        CRPIX values used in the solve-field header.
    ...
    """
    print(f"Detector = {array_id}")

    # Load image and header
    with fits.open(input_image) as hdul:
        img = hdul[1].data
        hdr = hdul[1].header
    print(f" hdr{hdr}")

    # Get RA/Dec
    ra, dec = hdr["CRVAL1"], hdr["CRVAL2"]
    if verbose:
        print(f"RA/Dec = {ra} / {dec}")

    nsigma = 3
    # Run SEP
    if array_id in [4, 5, 6]:
        nsigma = 3
        pixel_error = 5
        radius = 40
        sigma_extraction = 5

    mean, median, stddev = sigma_clipped_stats(img, sigma=3)
    # objects = sep.extract(img - median, sigma_extraction, err=stddev)
    # objects = objects[objects["tnpix"] > 3]
    print(f"mean, median, stddev : {mean}, {median}, {stddev}")

    min_sources = 1000
    max_sources = 2000
    sigma_extraction_current = sigma_extraction
    converged = False

    for i in range(10):
        objects = sep.extract(img - median, sigma_extraction_current, err=stddev)
        objects = objects[objects["tnpix"] > 3]
        n_objects = len(objects)

        if verbose:
            print(f"SEP iteration {i+1}: sigma_extraction = {sigma_extraction_current}, N = {n_objects}")

        if min_sources <= n_objects <= max_sources:
            converged = True
            break
        elif n_objects > max_sources:
            sigma_extraction_current *= 1.5  # fewer detections → raise threshold
        else:
            sigma_extraction_current *= 0.7  # more detections → lower threshold

    if not converged:
        print("Warning: Could not converge SEP extraction after 10 iterations. "
              f"Using N={n_objects} at sigma={sigma_extraction_current}")

    if verbose:
        print(f"Number of SEP detections: {len(objects)}")

    # Create fake image
    img2 = np.zeros_like(img)
    for obj in objects:
        y, x = int(round(obj["y"])), int(round(obj["x"]))
        # y, x = obj["y"], obj["x"]
        if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
            img2[y, x] = obj["flux"]

    kernel = Gaussian2DKernel(x_stddev=3/6.15)
    img2 = convolve(img2, kernel)
    img2 += np.random.normal(loc=0, scale=stddev/50, size=img2.shape)  # 50

    # Write stripped image
    hdu2 = fits.PrimaryHDU(data=img2)
    hdu2.writeto(stripped_image, overwrite=True)

    if stripped_image in (None, "none") or not os.path.exists(stripped_image):
        raise ValueError(f"Invalid stripped_image path: {stripped_image}")

    # Build astrometry.net command
    # output_flags = "--no-plots --new-fits none --index-xyls none
    # --axy none --match none --rdls none --scamp-ref none --corr none --pnm none"
    output_flags = "--no-plots --new-fits none --corr none"
    cmd = f"""
    solve-field {output_flags}
    --scale-low {pixel_scale_bounds[0]} --scale-high {pixel_scale_bounds[1]} --scale-units arcsecperpix
    --fits-image --source-extractor-path sex
    --ra {ra} --dec {dec} --radius {radius}
    --index-dir {index_dir}
    --downsample {downsample} --nsigma {nsigma}
    --wcs {wcs_output} --guess-scale
    --crpix-x {crpix[0]} --crpix-y {crpix[1]}
    --tweak-order {tweak_order} --pixel-error {pixel_error}
    --extension {extension} --overwrite
    {stripped_image}
    """.strip().replace("\n", " ")

    if verbose:
        print(f"Running astrometry.net with:\n{cmd} on SPHEREx OBSID")

    # Run subprocess
    try:
        result = subprocess.run(
            cmd, shell=True, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300, text=True
        )
        if verbose:
            print("solve-field completed successfully.")
            print(result.stdout)
            print(result.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"solve-field failed: {e.stderr}")
    return wcs_output, stripped_image


def se_run(img_path: str | Path,
           img_ldac: str | Path,
           num_ref_cat: int,
           verbose: bool = True,
           detect_thresh: str = '12',
           override_loop: bool = False,
           **kwargs) -> None:

    def run_sextractor(detect_thresh_local: str, img_ldac_local: str):
        sextractor_config_file = get_data_file("astrometry/default.sex", package="spx_utils")
        param_file = get_data_file("astrometry/default.param", package="spx_utils")
        cmd = [
            'sex', img_path,
            '-c', sextractor_config_file,
            '-CATALOG_NAME', img_ldac_local,
            '-PARAMETERS_NAME', param_file,
            '-DETECT_THRESH', detect_thresh_local,
            '-ANALYSIS_THRESH', kwargs.get('analysis_thresh', '2'),
            '-DETECT_MINAREA', kwargs.get('detect_minarea', '2'),
            '-DETECT_MAXAREA', kwargs.get('detect_maxarea', '5'),
            '-CATALOG_TYPE', kwargs.get('catalog_type', 'FITS_LDAC'),
            '-WRITE_XML', kwargs.get('write_xml', 'N'),
            '-DETECT_TYPE', kwargs.get('detect_type', 'CCD'),
            '-THRESH_TYPE', kwargs.get('thresh_type', 'RELATIVE'),
            '-FILTER', kwargs.get('filter', 'N'),
            '-DEBLEND_NTHRESH', kwargs.get('deblend_nthresh', '32'),
            '-DEBLEND_MINCONT', kwargs.get('deblend_mincont', '0.1'),
            '-CLEAN', kwargs.get('clean', 'Y'),
            '-CLEAN_PARAM', kwargs.get('clean_param', '1.0'),
            '-MASK_TYPE', kwargs.get('mask_type', 'CORRECT'),
            '-WEIGHT_TYPE', kwargs.get('weight_type', 'NONE'),
            '-FLAG_IMAGE', kwargs.get('flag_image', 'NONE'),
            '-FLAG_TYPE', kwargs.get('flag_type', 'OR'),
            '-PHOT_APERTURES', kwargs.get('phot_aperatures', '30'),
            '-PHOT_AUTOPARAMS', kwargs.get('phot_autoparams', '2.5,3.5'),
            '-PHOT_PETROPARAMS', kwargs.get('phot_petroparams', '2.0,3.5'),
            '-PHOT_AUTOAPERS', kwargs.get('phot_autoapers', '0.0,0.0'),
            '-PHOT_FLUXFRAC', kwargs.get('phot_fluxfrac', '0.5'),
            '-SATUR_LEVEL', kwargs.get('satur_level', '50000.0'),
            '-MAG_ZEROPOINT', kwargs.get('mag_zeropoint', '0.0'),
            '-MAG_GAMMA', kwargs.get('mag_gamma', '4.0'),
            '-GAIN', kwargs.get('gain', '0.0'),
            '-PIXEL_SCALE', kwargs.get('pixel_scale', '6.15'),
            '-SEEING_FWHM', kwargs.get('seeing_fwhm', '0.1'),
            '-BACK_TYPE', kwargs.get('back_type', 'AUTO'),
            '-BACK_VALUE', kwargs.get('back_value', '0.0'),
            '-BACK_SIZE', kwargs.get('back_size', '64'),
            '-BACK_FILTERSIZE', kwargs.get('back_filtersize', '3'),
            '-BACKPHOTO_TYPE', kwargs.get('backphoto_type', 'GLOBAL'),
            '-BACKPHOTO_THICK', kwargs.get('backphoto_thick', '24'),
            '-BACK_FILTTHRESH', kwargs.get('back_filtthresh', '0.0'),
            '-CHECKIMAGE_TYPE', kwargs.get('checkimage_type', 'NONE'),
            '-NTHREADS', kwargs.get('nthreads', '1')
        ]
        result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=SUBPROCESS_TIMEOUT, text=True)
        print(f"SExtractor run with DETECT_THRESH = {detect_thresh_local}")
        # print(result.stdout)
        print(result.stderr)
        # Extract 'Objects: detected <N>' using regex
        match = re.search(r'Objects:\s+detected\s+(\d+)', result.stderr)
        n_detected = int(match.group(1)) if match else -1
        print(f">>> N detected = {n_detected}")
        return n_detected, result.stderr

    # Adaptive threshold loop
    min_sources, max_sources = num_ref_cat*0.8, num_ref_cat*1.2
    new_thresh = float(detect_thresh)
    converged = False
    max_iterations = 20
    if override_loop is True:
        max_iterations = 1

    for i in range(max_iterations):  # limit to 20 iterations
        print(f">>>>>>> THRESHOLD USED : {new_thresh}")
        n_detected, stderr = run_sextractor(str(new_thresh), img_ldac)
        try:
            with fits.open(img_ldac) as hdul:
                _ = len(hdul[2].data)
        except Exception as e:

            raise RuntimeError(f"Failed to read number of sources from {img_ldac}: {e}"
                               f"stderr = {stderr}")

        if verbose:
            print(f"Iteration {i+1}: Detected in log = {n_detected}, ")
        if override_loop is True:
            break
        if min_sources <= n_detected <= max_sources:
            converged = True
            break
        elif n_detected > max_sources:
            new_thresh *= 1.5
        else:
            new_thresh *= 0.7
    if not converged and verbose:
        print(f"Could not converge to desired source count within {max_iterations} iterations.")

    return new_thresh


def scamp_run(img_ldac,
              refcat_ldac,
              output,
              verbose: bool = True,
              astref_catalog: str = 'FILE',
              solve_astrometry: str = 'Y',
              projection_type: str = 'TAN',
              position_maxerr: str = '20',
              posangle_maxerr: str = '5',
              pixscale_maxerr: str = '1.2',
              cross_radius: str = '30',
              match_flipped: str = 'N',
              mosaic_type: str = 'UNCHANGED',
              distort_degrees: str = '3',
              # FOCDISTORT_DEGREE: str = "3",
              solve_photom: str = 'N',
              astrclip_nsigma: str = "3.0",
              match: str = 'Y',
              fulloutcat_type: str = 'FITS_LDAC',
              checkplot_dev: str = 'NULL',
              write_xml: str = 'N'
              ) -> None:

    # Define the SCAMP command with its options
    # https://faun.rc.fas.harvard.edu/ameisner/scamp.pdf
    scamp_config_file = get_data_file("astrometry/scamp.conf", package="spx_utils")
    scamp_cmd = [
        'scamp',  # The SCAMP executable
        img_ldac,  # Input image
        '-c', scamp_config_file,
        '-ASTREF_CATALOG', astref_catalog,
        '-ASTREFCAT_NAME', refcat_ldac,
        '-SOLVE_ASTROM', solve_astrometry,
        '-ASTR_FLAG', '',
        '-PROJECTION_TYPE', projection_type,
        '-PIXSCALE_MAXERR', pixscale_maxerr,  # Max scale-factor uncertainty
        '-POSANGLE_MAXERR', posangle_maxerr,  # Max position-angle uncertainty (deg)
        '-POSITION_MAXERR', position_maxerr,  # Max positional uncertainty (arcmin)
        '-CROSSID_RADIUS', cross_radius,  # Matching radius in arcseconds
        '-MATCH_FLIPPED', match_flipped,  # Allow matching with flipped axes?
        '-MOSAIC_TYPE', mosaic_type,  # UNCHANGED, SAME_CRVAL, SHARE_PROJAXIS,
        '-DISTORT_DEGREES', distort_degrees,  # Polynom degree for each group
        # '-FOCDISTORT_DEGREE', FOCDISTORT_DEGREE,
        '-ASTRCLIP_NSIGMA', astrclip_nsigma,
        # '-DISTORT_GROUPS', '1,2,3,4,5,6'
        '-SOLVE_PHOTOM', solve_photom,
        '-MATCH', match,
        '-FULLOUTCAT_TYPE', fulloutcat_type,  # FITS_LDAC, ASCII_HEAD
        '-FULLOUTCAT_NAME', output,
        '-CHECKPLOT_DEV', checkplot_dev,
        '-WRITE_XML', write_xml,
    ]
    print(f"distort_degrees = {distort_degrees}")
    print(f"position_maxerr = {position_maxerr}")
    print(f"posangle_maxerr = {posangle_maxerr}")
    print(f"pixscale_maxerr = {pixscale_maxerr}")
    print(f"cross_radius = {cross_radius}")
    print(f"projection_type = {projection_type}")
    print(f"astrclip_nsigma = {astrclip_nsigma}")

    # Run SCAMP
    result = subprocess.run(scamp_cmd,
                            check=True,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=SUBPROCESS_TIMEOUT,
                            text=True)
    if verbose is True:
        print("SCAMP completed successfully.")
        print(result.stdout)
        print(result.stderr)
        print(f"Executed on the LDAC image: {img_ldac}")
        print(f"Output scamp astrometry saved as: {output}")

    ra_shift, dec_shift = parse_scamp_shift(result.stderr)
    contrast = parse_scamp_contrast(result.stderr)
    detections_loaded, standards_loaded = parse_scamp_counts(result.stderr)
    daxis1, daxis2 = parse_scamp_daxis(result.stderr)
    nstars_external = parse_scamp_nstars_external(result.stderr)
    print(f">>>>>NSTARS = {nstars_external}")

    print(f"SCAMP contrast = {contrast}")
    print(f"dAXIS1, dAXIS2 = {daxis1}, {daxis2}")

    if detections_loaded is not None and standards_loaded is not None:
        print(f"SCAMP used {detections_loaded} detections and {standards_loaded} reference stars.")

    if check_scamp_matched_warning(result.stderr):
        print(">>>>>SCAMP may have failed due to too few matched sources.<<<<<<")

    if ra_shift is not None and (abs(ra_shift) > 100 or abs(dec_shift) > 100):
        print(f"WARNING: SCAMP solution has large shift: RA = {ra_shift:.1f}\"  DEC = {dec_shift:.1f}\"")
        print(f"SCAMP poor match: large astrometric shift detected."
              f"detections_loaded = {detections_loaded}"
              f"standards loaded = {standards_loaded}")
    return contrast, detections_loaded, standards_loaded, ra_shift, dec_shift, daxis1, daxis2, nstars_external


def parse_scamp_shift(stderr: str) -> tuple[float, float]:
    """Parse SCAMP stderr and extract RA and Dec shift."""
    # Match pattern like: -1.7e+03"+9.4e+02"
    shift_match = re.search(r"([-+.\deE]+)\"\s*([-+.\deE]+)\"\s+\d+\.\d+", stderr)
    if shift_match:
        ra_shift = float(shift_match.group(1))
        dec_shift = float(shift_match.group(2))
        print(f">>>>>> ra_shift, dec_shift : {ra_shift}, {dec_shift}")
        return ra_shift, dec_shift
    return None, None


def parse_scamp_counts(stderr: str) -> tuple[int, int]:
    """
    Parse SCAMP stderr and extract:
    - Number of detections loaded from the input image LDAC.
    - Number of standards loaded from the reference LDAC.

    Returns
    -------
    detections_loaded : int
        Number of detections loaded from the image.
    standards_loaded : int
        Number of standards loaded from the reference catalog.
    """
    detections_loaded = None
    standards_loaded = None

    # Match pattern like: ----- 3175 detections loaded
    det_match = re.search(r"[-]+ (\d+) detections loaded", stderr)
    if det_match:
        detections_loaded = int(det_match.group(1))
        print(f">>>>> detections_loaded = {detections_loaded}")

    # Match pattern like: Group  1:       30 standards in file (band 1)
    std_match = re.search(r"Group\s+\d+:\s+(\d+)\s+standards in file", stderr)
    if std_match:
        standards_loaded = int(std_match.group(1))
        print(f">>>>> standards_loaded = {standards_loaded}")

    return detections_loaded, standards_loaded


def check_scamp_matched_warning(stderr: str) -> bool:
    """
    Check if SCAMP output contains the 'not enough matched detections' warning.

    Parameters
    ----------
    stderr : `str`
        The stderr output from SCAMP.

    Returns
    -------
    bool
        True if the warning is found, False otherwise.
    """
    pattern = r"> WARNING: Not enough matched detections in instrument A1"
    found = re.search(pattern, stderr)
    if found:
        return True
    return False


def parse_scamp_contrast(stderr: str) -> float | None:
    """
    Extract the final contrast value (last number on the line) from SCAMP output.

    Returns
    -------
    contrast : float or None
        The final contrast value (typically around 1.0), or None if not found.
    """
    print(">>> TRY GET CONT.")
    for line in stderr.splitlines():
        if " A1 " in line and " P1 " in line and '"' in line:
            # Strip everything before the last quote
            try:
                last_quote_idx = line.rindex('"')
                remainder = line[last_quote_idx+1:]
                # Now grab float from remainder
                match = re.search(r'([-+]?\d*\.\d+(?:[eE][-+]?\d+)?|\d+)', remainder)
                if match:
                    contrast = float(match.group(1))
                    print(f">>>>>> contrast : {contrast}")
                    return contrast
                else:
                    print(f"[WARN] No number found after last quote in: {remainder}")
            except Exception as e:
                print(f"[ERROR] Parsing contrast failed: {e}")
    return None


def parse_scamp_daxis(stderr: str) -> tuple[float, float] | None:
    """Parse SCAMP stderr and extract external dAXIS1 and dAXIS2
       from astrometric stats."""
    # Look for the external astrometric stats block
    pattern = (
        r"----- Astrometric stats \(external\):.*?"
        r"Group\s+1:\s+([-+.\deE]+)\"\s+([-+.\deE]+)\"",  # external dAXIS1 and dAXIS2
    )
    match = re.search(pattern[0], stderr, re.DOTALL)
    if match:
        try:
            daxis1 = float(match.group(1))
            daxis2 = float(match.group(2))
            print(f">>>>>> dAXIS1 = {daxis1}\", dAXIS2 = {daxis2}\"")
            return daxis1, daxis2
        except ValueError:
            print("[ERROR] Failed to convert dAXIS values to float.")
    else:
        print("[WARN] Could not find external dAXIS1/dAXIS2 in SCAMP output.")
    return None


def parse_scamp_nstars_external(stderr: str) -> int | None:
    """
    Extract the number of external matched stars (nstars) from SCAMP output.

    Parameters
    ----------
    stderr : str
        Standard error output from SCAMP.

    Returns
    -------
    nstars : int or None
        Number of matched external stars, or None if not found.
    """
    # Match: Group  1:   1.68"   2.54" 8.3e+02    2296
    match = re.search(r"Group\s+\d+:\s+[-+.\deE]+\"\s+[-+.\deE]+\"\s+[-+.\deE]+\s+(\d+)", stderr)
    if match:
        nstars = int(match.group(1))
        print(f">>>>> External matched nstars = {nstars}")
        return nstars
    print("[WARN] Could not extract external matched nstars.")
    return None
