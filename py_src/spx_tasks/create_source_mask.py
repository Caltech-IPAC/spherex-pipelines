__all__ = ["CreateSourceMaskConfig", "CreateSourceMaskTask"]

# Standard library imports
import copy
from typing import Union

# Related third party imports
from lsst.pex.config import Field, ListField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import regions
# from regions import PixCoord, CirclePixelRegion
import matplotlib.pyplot as plt
import scipy.linalg
from astropy import stats as astrostats
import numpy as np
from astropy.table.table import Table
from scipy.ndimage import gaussian_filter

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_tasks import add_history
from spx_data import ReadoutParameters
from spx_utils import log_dqa_result
from spx_utils import refcat_utils


def blended_exponential_radius(m, a1, b1, a2, b2, m0, w) -> np.ndarray:
    """Compute blended exponential mask-radius model.

    Parameters
    ----------
    m : `np.ndarray`
        Magnitude(s).
    a1, b1, a2, b2 : `float`
        Linear-in-magnitude coefficients for the two exponential regimes.
    m0 : `float`
        Transition magnitude (midpoint of the blend).
    w : `float`
        Transition width (controls how sharp/smooth the blend is).

    Returns
    -------
    `np.ndarray`
        Radius value(s).

    Notes
    -----
    Radius(m) = exp( [1 - s(m)](a1 + b1*m) + s(m)(a2 + b2*m) )
    where
    s(m) = 1 / (1 + exp(-(m - m0)/w))
    https://rigel.caltech.edu/calibration_analysis_logbook/20251205_BrightStarMasking/

    """
    m = np.asarray(m, dtype=float)

    if w == 0:  # Divide by zero catch
        raise ValueError("Divide by Zero, should no occur in parameters.")
    else:
        s = 1. / (1. + np.exp(-(m - m0) / w))

    R = np.exp((1.0 - s) * (a1 + b1 * m) + s * (a2 + b2 * m))

    return R


def mask_sources(spximg: SPHERExImage,
                 cat: Table,
                 blended_params_dict: dict,
                 plot_this: bool = True):
    """Mask sources based on their magnitudes and positions.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image object.
    cat : `Table`
        Source catalog.
    blended_params_dict : `dict`
        Blended model params dict keyed by band:
        {band: [a1, b1, a2, b2, m0, w]}
    plot_this: `bool`, optional
        Whether to plot the source mask.

    Returns
    -------
    `np.ndarray`: The source mask.

    """
    band = spximg.getArrayId()

    # Get readout parameters from image
    rp = ReadoutParameters.from_metadata(spximg.meta)
    nx = rp.Jactive
    ny = rp.Kactive
    xoff = yoff = nx // 2

    # Pull out the needed parameters from the source catalog
    ABmag = np.array(cat[f'mag_det{band}'])

    ra_values = cat['ra']
    dec_values = cat['dec']

    wcs_obj = spximg.wcs
    ix, iy = wcs_obj.world_to_pixel_values(ra_values, dec_values)

    # Filter magnitudes
    if band in [1, 2, 3]:
        mag_mask = cat['do_mask_swir'].astype(bool)
    elif band in [4, 5, 6]:
        mag_mask = cat['do_mask_mwir'].astype(bool)

    xpix = ix[mag_mask]
    ypix = iy[mag_mask]
    abmag = ABmag[mag_mask]

    image_mask = np.zeros([2 * nx, 2 * ny])

    # Pre-load blended parameters (per band)
    if band not in blended_params_dict:
        raise ValueError(f"No blended-exponential params provided for band={band}")
    a1, b1, a2, b2, m0, w_blend = blended_params_dict[band]

    # Loop through sources applying the mask
    for xval, yval, ab in zip(xpix, ypix, abmag):
        try:
            radius = blended_exponential_radius(ab, a1, b1, a2, b2, m0, w_blend)
        except Exception as e:
            print(f"DEBUG: Exception {e} occured")
            continue

        if (not np.isfinite(xval) or not np.isfinite(yval) or not np.isfinite(radius) or radius <= 0):
            continue

        pixcoord = regions.PixCoord(x=xoff + xval, y=yoff + yval)
        aperture = regions.CirclePixelRegion(pixcoord, radius)
        mask = aperture.to_mask(mode='exact')
        bbox = aperture.bounding_box

        iymin, iymax = bbox.iymin, bbox.iymax
        ixmin, ixmax = bbox.ixmin, bbox.ixmax

        # Check if bbox is valid and has nonzero area
        if iymax <= iymin or ixmax <= ixmin:
            continue  # skip degenerate mask

        # Clip bbox to image bounds
        iymin_clip = max(iymin, 0)
        iymax_clip = min(iymax, image_mask.shape[0])
        ixmin_clip = max(ixmin, 0)
        ixmax_clip = min(ixmax, image_mask.shape[1])

        # Skip if nothing would be added
        if iymin_clip >= iymax_clip or ixmin_clip >= ixmax_clip:
            continue

        # Slice the mask accordingly
        mask_yslice = slice(iymin_clip - iymin, iymax_clip - iymin)
        mask_xslice = slice(ixmin_clip - ixmin, ixmax_clip - ixmin)

        mask_data = mask.data
        if mask_data is not None:
            image_mask[iymin_clip:iymax_clip, ixmin_clip:ixmax_clip] += mask_data[mask_yslice, mask_xslice]

    whpl_type1mask = image_mask[xoff: xoff + nx, yoff: yoff + ny] > 0.2

    print(f"Type 1 mask fraction: {np.sum(whpl_type1mask)/(nx*ny)}")
    return whpl_type1mask


def compute_planet_mask_radius(x: Union[int, float],
                               y: Union[int, float],
                               detector: int,
                               planet: str,
                               planet_mask_radii: dict) -> float:
    """Return a masking radius (in pixels) for a planet on a SPHEREx detector.


    Parameters
    ----------
    x, y : `int` or `float`
        Pixel coordinates of the planet on the detector. These are currently
        unused by the function, but are included to allow future extensions
        where the mask radius may depend on position.
    detector : `int`
        SPHEREx detector ID.
    planet : `str`
        Name or identifier of the planet. This value is normalized internally
        (lowercased and stripped) before lookup.
    planet_mask_radii : `dict`
        Dictionary mapping normalized planet names to a list of
        masking radii (in pixels), one for each detector.

    Returns
    -------
    radius : `float`
        Masking radius in pixels for the given planet on the specified detector.

    Raises
    ------
    ValueError
        If the detector index is not in 1-6, if the planet name is not present
        in ``planet_mask_radii``, or if the corresponding radii entry is not
        a length-6 sequence.

    """
    if detector not in {1, 2, 3, 4, 5, 6}:
        raise ValueError("detector index must be an integer in 1..6")

    planet_key = str(planet).strip().lower()
    if planet_key not in planet_mask_radii:
        raise ValueError(f"unknown planet '{planet}'")

    radii = planet_mask_radii[planet_key]
    if not hasattr(radii, "__len__") or len(radii) != 6:
        raise ValueError(
            f"planet_mask_radii['{planet_key}'] must be a length-6 list (detectors 1..6)"
        )

    radius = float(radii[detector - 1])
    return radius


def mask_planets(spximg: SPHERExImage,
                 ssocat: Table,
                 planet_mask_radii: dict,
                 name_col: str = "desig",
                 ra_col: str = "RA",
                 dec_col: str = "DEC") -> np.ndarray:
    """Build a boolean mask for bright planets/SSOs using fixed radii.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Input SPHEREx image.
    ssocat : `Table`
        Table of solar system objects. Must contain RA/DEC columns and a
        name/identifier column.
    planet_mask_radii : `dict`
        Dictionary mapping normalized planet names to a list of
        masking radii (in pixels), one for each detector.
    name_col : `str`, optional
        Column holding the SSO name (e.g., "Mars", "Jupiter").
    ra_col, dec_col : `str`, optional
        RA/Dec column names.

    Returns
    -------
    mask : `np.ndarray`
        Boolean mask of shape (Jactive, Kactive) where True means "mask it".

    """
    rp = ReadoutParameters.from_metadata(spximg.meta)
    nx = rp.Jactive  # rows (y)
    ny = rp.Kactive  # cols (x)

    planet_mask = np.zeros((nx, ny), dtype=bool)
    if ssocat is None or len(ssocat) == 0:
        return planet_mask

    det_id = spximg.getArrayId()

    for col in (ra_col, dec_col):
        if col not in ssocat.colnames:
            raise ValueError(f"ssocat is missing required column '{col}'")
    if name_col not in ssocat.colnames:
        raise ValueError(f"ssocat is missing required column '{name_col}'")

    ra = np.asarray(ssocat[ra_col], dtype=float)
    dec = np.asarray(ssocat[dec_col], dtype=float)
    ix, iy = spximg.wcs.world_to_pixel_values(ra, dec)
    ix = np.asarray(ix, dtype=float)
    iy = np.asarray(iy, dtype=float)

    for xval, yval, pname in zip(ix, iy, ssocat[name_col]):
        if not np.isfinite(xval) or not np.isfinite(yval):
            continue

        pname_norm = str(pname).strip().lower()

        try:
            radius = compute_planet_mask_radius(xval,
                                                yval,
                                                det_id,
                                                pname_norm,
                                                planet_mask_radii=planet_mask_radii)
        except ValueError:
            continue

        if not np.isfinite(radius) or radius <= 0:
            continue

        # fully-off-detector quick reject (x against ny, y against nx)
        if (xval + radius < 0) or (xval - radius >= ny) or (yval + radius < 0) or (yval - radius >= nx):
            continue

        aperture = regions.CirclePixelRegion(regions.PixCoord(x=xval, y=yval), radius)

        try:
            m = aperture.to_mask(mode="center")
        except regions.NoOverlap:
            continue

        if m.data is None:
            continue

        bbox = aperture.bounding_box
        iymin, iymax = bbox.iymin, bbox.iymax
        ixmin, ixmax = bbox.ixmin, bbox.ixmax

        # clip to detector bounds (rows=y against nx, cols=x against ny)
        y0 = max(iymin, 0)
        y1 = min(iymax, nx)
        x0 = max(ixmin, 0)
        x1 = min(ixmax, ny)

        if y0 >= y1 or x0 >= x1:
            continue

        mask_yslice = slice(y0 - iymin, y1 - iymin)
        mask_xslice = slice(x0 - ixmin, x1 - ixmin)
        stamp = m.data[mask_yslice, mask_xslice]

        # For mode="center", stamp is 0/1
        planet_mask[y0:y1, x0:x1] |= (stamp == 1)

    return planet_mask


def ssocat_has_any_planets(ssocat: Table, name_col: str, planet_keys: list) -> bool:
    """Check whether a solar-system object catalog contains any known planets.

    Parameters
    ----------
    ssocat : `astropy.table.Table`
        Table of solar-system objects. Must contain a column with object names
        or identifiers.
    name_col : `str`
        Name of the column in ``ssocat`` that holds the object identifiers
        (e.g., planet names).
    planet_keys : `list`
        Iterable list of planet identifiers to search for. Values are normalized
        internally by stripping whitespace and converting to lowercase.

    Returns
    -------
    has_planets : `bool`
        True if any of the planet identifiers appear in the catalog;
        False otherwise.

    """
    if ssocat is None or len(ssocat) == 0 or name_col not in ssocat.colnames:
        return False

    keys = {str(k).strip().lower() for k in planet_keys}
    ssokeys = {str(v).strip().lower() for v in ssocat[name_col]}

    return not keys.isdisjoint(ssokeys)  # True if any overlap


def _prepare_xtalk_full_frame(spximg: SPHERExImage) -> tuple[np.ndarray, tuple[slice, slice]]:
    rp = ReadoutParameters.from_metadata(spximg.meta)

    BAD_MASK = (
        Flag.HOT
        | Flag.COLD
        | Flag.OVERFLOW
        | Flag.TRANSIENT
        | Flag.SNOWBALL
        | Flag.BLOOM
        | Flag.SUR_ERROR
        | Flag.MISSING_DATA
        | Flag.SUR_MISMATCH
        | Flag.NONLINEAR
        | Flag.NONFUNC
    )

    # Start from active science image
    active_img = np.asarray(spximg.data, dtype=float).copy()

    # Build bad-pixel mask on the active image
    bad = ~np.isfinite(active_img)

    if hasattr(spximg, "flags") and spximg.flags is not None:
        bad |= (spximg.flags & BAD_MASK) != 0

    good = ~bad
    if np.any(good):
        fill_value = np.nanmedian(active_img[good])
    else:
        fill_value = 0.0

    active_img[bad] = fill_value

    # Full detector shape including refs
    full_shape = (rp.Jactive + rp.jref, rp.Kactive + rp.kref)
    full_img = np.full(full_shape, fill_value, dtype=float)

    # Embed active image in the middle of the full detector frame
    nrows, ncols = active_img.shape
    row0 = (full_shape[0] - nrows) // 2
    col0 = (full_shape[1] - ncols) // 2
    row1 = row0 + nrows
    col1 = col0 + ncols

    if row1 > full_shape[0] or col1 > full_shape[1]:
        raise ValueError(
            f"Active image shape {active_img.shape} does not fit in full detector shape {full_shape}"
        )

    row_range = slice(row0, row1)
    col_range = slice(col0, col1)

    full_img[row_range, col_range] = active_img

    return full_img, (row_range, col_range)


def class1_table_to_array(tbl: Table) -> np.ndarray:
    channels = np.asarray(tbl["Channel"], dtype=int)
    amps = np.asarray(tbl["Coupling Amplitude"], dtype=float)

    if len(tbl) != 32:
        raise ValueError("xtalk_class1 table must have 32 rows")

    out = np.zeros(32, dtype=float)
    for ch, amp in zip(channels, amps):
        if ch < 1 or ch > 32:
            raise ValueError(f"Invalid Class 1 channel {ch}; expected 1..32")
        out[ch - 1] = amp

    return out


def class2_table_to_matrix(tbl: Table) -> np.ndarray:
    colnames = list(tbl.colnames)

    if len(colnames) != 32:
        raise ValueError(f"xtalk_class2 table must have 32 columns; got {len(colnames)}")

    matrix = np.column_stack([np.asarray(tbl[col], dtype=float) for col in colnames])

    if matrix.shape != (32, 32):
        raise ValueError(f"xtalk_class2 table must be shape (32, 32); got {matrix.shape}")

    return matrix


def mask_crosstalk(
    spximg: SPHERExImage,
    xtalk_class2: Table,
    xtalk_class1: Table | None = None,
    xtalk_threshold: float = 0.3,
    gaussian_sigma: float = 1.5,
    class1_swir_detectors: tuple[int, ...] = (1, 2, 3),
) -> np.ndarray:
    """Build CROSSTALK mask following SSDC-MS-019 v3.5."""
    if xtalk_class2 is None:
        raise ValueError("xtalk_class2 is required when mask_xtalk=True")

    class2_coupling_matrix = class2_table_to_matrix(xtalk_class2)
    class1_amplitudes = None
    if xtalk_class1 is not None:
        class1_amplitudes = class1_table_to_array(xtalk_class1)

    full_img, (row_range, col_range) = _prepare_xtalk_full_frame(spximg)
    filtered = gaussian_filter(full_img, sigma=gaussian_sigma)

    ny, nx = filtered.shape
    n_channels = 32

    if nx % n_channels != 0:
        raise ValueError(f"Image width {nx} is not divisible by {n_channels} channels.")

    channel_width = nx // n_channels

    if class2_coupling_matrix.shape != (n_channels, n_channels):
        raise ValueError("class2_coupling_matrix must have shape (32, 32).")

    class2_coupling_matrix = np.nan_to_num(class2_coupling_matrix, nan=0.0)

    xtalk_mask = np.zeros((ny, nx), dtype=bool)

    # Class 2
    for i in range(n_channels):
        x0_i = i * channel_width
        x1_i = x0_i + channel_width
        src_sub = filtered[:, x0_i:x1_i]

        for j in range(n_channels):
            coupling_ij = class2_coupling_matrix[i, j]
            if coupling_ij == 0:
                continue

            xtalk_img = src_sub * coupling_ij
            mask_ij = np.abs(xtalk_img) > xtalk_threshold
            if not np.any(mask_ij):
                continue

            x0_j = j * channel_width
            x1_j = x0_j + channel_width
            xtalk_mask[:, x0_j:x1_j] |= mask_ij

    # Class 1
    det_id = spximg.getArrayId()
    if det_id in class1_swir_detectors and class1_amplitudes is not None:
        class1_amplitudes = np.asarray(class1_amplitudes, dtype=float)
        if class1_amplitudes.shape != (n_channels,):
            raise ValueError("class1_amplitudes must have shape (32,)")

        class1_amplitudes = np.nan_to_num(class1_amplitudes, nan=0.0)
        class1_mask_unshifted = np.zeros((ny, nx), dtype=bool)

        for i in range(n_channels):
            x0 = i * channel_width
            x1 = x0 + channel_width
            sub = filtered[:, x0:x1]

            amp = abs(class1_amplitudes[i])
            if amp == 0:
                continue

            xtalk_img = sub * amp
            class1_mask_unshifted[:, x0:x1] |= (np.abs(xtalk_img) > xtalk_threshold)

        shift = 32
        class1_mask_shifted = np.zeros_like(class1_mask_unshifted)
        if shift < ny:
            class1_mask_shifted[shift:, :] = class1_mask_unshifted[:-shift, :]
        class1_mask_shifted[:shift, :] = False

        xtalk_mask |= class1_mask_shifted

    return xtalk_mask[row_range, col_range]


def mask_dbs_ghosts(spximg: SPHERExImage,
                    cat: Table,
                    params_edge: dict,
                    params_main: dict,
                    roi_edge: dict,
                    roi_main: dict,
                    bleed_flux_thresh: float,
                    bleed_roi: list,
                    verbose: bool = False) -> np.ndarray:
    """Masks DBS ghosts: main field and edge field.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image object (to get band, shape, and WCS).
    cat : `Table`
        Source catalog for the corresponding dichroic pair.
    params_edge : `dict`
        Dictionary of edge ghost model parameters, keyed by band.
    params_main : `dict`
        Dictionary of main field ghost model parameters, keyed by band.
    roi_edge : `dict`
        Dictionary of edge ghost source ROIs, keyed by band.
    roi_main : `dict`
        Dictionary of main field ghost source ROIs, keyed by band.
    bleed_flux_thresh : `float`
        Flux threshold (Jy) for Band 4 optical bleeding.
    bleed_roi : `list`
        Band 4 optical bleeding mask region [x_min, x_max, y_min, y_max].
    verbose : `bool
        Print verbose debug statements if True, default is False.

    Returns
    -------
    `np.ndarray`
        Integer flag mask (dtype=np.int32) of the same shape as
        spximg.data, containing the ghost flags.

    Notes
    -----
    - This function implements the geometric transformation model for
    main field (Bands 4, 5, 6) and edge region (Bands 3, 4) ghosts.
    It also handles the special optical bleeding effect for bright
    edge-region sources in Band 4.
    - Specific H-band flux values from the refcat are not propagated
    to the srccats. Thus, a proxy for the H-band flux is derived
    from other bands based on the implementation:
      - For Band 3: Uses flux from `mag_det3`.
      - For Band 4: Uses the average flux from `mag_det1` and `mag_det4`.
      - For Band 5: Uses flux from `mag_det2`.
      - For Band 6: Uses flux from `mag_det3`.

    """
    band = spximg.getArrayId()
    ny, nx = spximg.data.shape  # Get detector bounds
    ghost_mask = np.zeros((ny, nx), dtype=np.int32)

    # Bands 1 and 2 do not have DBS ghosts
    if band in [1, 2]:
        return ghost_mask

    # Pre-calculate pixel coordinates and flux
    cat_proc = cat.copy()

    # Use WCS to get source pixel coordinates (xs, ys)
    try:
        ix, iy = spximg.wcs.world_to_pixel_values(cat_proc['ra'], cat_proc['dec'])
        cat_proc['__ix'] = ix
        cat_proc['__iy'] = iy
    except Exception as e:
        print(f"Error during WCS transformation: {e}. Skipping DBS masking.")
        return ghost_mask

    # Calculate H-band proxy flux (B) in Janskys
    try:
        if band == 3:
            flux_jy = refcat_utils.mag_to_flux(cat_proc['mag_det3'])
        elif band == 4:
            # Average flux from Band 1 and Band 4
            flux_b1 = refcat_utils.mag_to_flux(cat_proc['mag_det1'])
            flux_b4 = refcat_utils.mag_to_flux(cat_proc['mag_det4'])
            fluxes = np.array([flux_b1, flux_b4])
            # Compute nan-safe mean along axis 0 (per-source average)
            # This handles sources present in only one of the bands.
            flux_jy = np.nanmean(fluxes, axis=0)
        elif band == 5:
            flux_jy = refcat_utils.mag_to_flux(cat_proc['mag_det2'])  # Using Band 2 as proxy
        else:  # band == 6
            flux_jy = refcat_utils.mag_to_flux(cat_proc['mag_det3'])  # Using Band 3 as proxy

    except KeyError as e:
        print(f"Error accessing mag column for H-band proxy flux: {e}")
        flux_jy = np.full(len(cat_proc), np.nan)  # Fallback to NaN

    # First, dynamically find the absolute minimum flux threshold
    # for the current band and filter by that.
    cat_proc['__h_flux_jy'] = flux_jy

    # Pre-filter catalog
    # Find the minimum flux threshold (B_lim or bleed_flux_thresh)
    # that could possibly trigger any ghost for this band.
    thresholds = []
    if band in params_edge:
        thresholds.append(params_edge[band][0])  # B_lim is the first param
    if band in params_main:
        thresholds.append(params_main[band][0])  # B_lim is the first param
    if band == 4:
        thresholds.append(bleed_flux_thresh)

    # If no ghost parameters are defined, min_thresh will be 0.0,
    # and no sources will be filtered by flux (which is correct).
    min_flux_thresh = np.min(thresholds) if thresholds else 0.0

    # Create a boolean mask to filter the catalog.
    # This selects sources that are bright enough.
    # It also implicitly filters out sources with NaN flux,
    # since (np.nan > min_flux_thresh) is False.
    bright_enough_mask = cat_proc['__h_flux_jy'] > min_flux_thresh
    cat_proc_filtered = cat_proc[bright_enough_mask]

    band4_bleed_triggered = False
    # Flag from the first source that triggers bleed
    triggering_bleed_flag = 0

    # Iterate over the filtered table
    for src in cat_proc_filtered:
        try:
            # Get pre-computed values
            xs, ys = src['__ix'], src['__iy']
            flux_h = src['__h_flux_jy']  # This is the H-band proxy flux in Jy

        except (KeyError, TypeError):
            continue  # Skip sources with missing data

        # We know flux_h is finite, just check pixel coords
        if not np.isfinite(xs) or not np.isfinite(ys):
            continue

        # Derive ghost flag based on source pixel position
        # (0 <= xs < nx) and (0 <= ys < ny) means the source is on this detector.
        # Otherwise, it's external. GHOST_FPA is ignored.
        if (0 <= xs < nx) and (0 <= ys < ny):
            flag_to_set = Flag.GHOST
            flag_name = "GHOST"
        else:
            flag_to_set = Flag.GHOST_EXT
            flag_name = "GHOST_EXT"

        if verbose:
            # This prints debug info for *all* bright sources
            print(f"--- Potential {flag_name} Source (Band {band}) ---")
            print(f"  RA, Dec: ({src['ra']:.6f}, {src['dec']:.6f})")
            print(f"  Pix Coords (xs, ys): ({xs:.2f}, {ys:.2f})")
            print(f"  H-band Proxy Flux (Jy): {flux_h:.4e}")
            # Try to print the original AB mags that created the proxy
            mag_info = []
            if band == 3 and 'mag_det3' in src.colnames:
                mag_info.append(f"mag_det3: {src['mag_det3']:.4f}")
            elif band == 4:
                if 'mag_det1' in src.colnames:
                    mag_info.append(f"mag_det1: {src['mag_det1']:.4f}")
                if 'mag_det4' in src.colnames:
                    mag_info.append(f"mag_det4: {src['mag_det4']:.4f}")
            elif band == 5 and 'mag_det2' in src.colnames:
                mag_info.append(f"mag_det2: {src['mag_det2']:.4f}")
            elif band == 6 and 'mag_det3' in src.colnames:
                mag_info.append(f"mag_det3: {src['mag_det3']:.4f}")

            if mag_info:
                print(f"  Source AB Mag(s): {', '.join(mag_info)}")
            print("-------------------------------------------------")

        # List to store (params, flag) tuples for ghosts from this one source
        ghosts_to_create = []

        # Check for Main Field Ghosts
        if band in params_main:
            B_lim, axx, axy, aox, ayx, ayy, aoy, r = params_main[band]
            x_min, x_max, y_min, y_max = roi_main[band]
            if (x_min < xs < x_max) and (y_min < ys < y_max) and (flux_h > B_lim):
                ghosts_to_create.append((axx, axy, aox, ayx, ayy, aoy, r, flag_to_set))

        # Check for Edge Region Ghosts
        if band in params_edge:
            B_lim, axx, axy, aox, ayx, ayy, aoy, r = params_edge[band]
            x_min, x_max, y_min, y_max = roi_edge[band]
            if (x_min < xs < x_max) and (y_min < ys < y_max) and (flux_h > B_lim):
                ghosts_to_create.append((axx, axy, aox, ayx, ayy, aoy, r, flag_to_set))

        # Check for Band 4 Optical Bleeding
        if band == 4 and not band4_bleed_triggered:
            x_min, x_max, y_min, y_max = roi_edge[4]  # Edge ROI
            if (x_min < xs < x_max) and (y_min < ys < y_max) and (flux_h > bleed_flux_thresh):
                band4_bleed_triggered = True
                triggering_bleed_flag = flag_to_set  # Store the flag

        # Apply Ghost Masks (for this source)
        for params in ghosts_to_create:
            axx, axy, aox, ayx, ayy, aoy, r, flag = params

            # Geometric transformation (Eqs. 2 & 3)
            xg = (1 + axx) * xs + axy * ys + aox
            yg = (1 + ayy) * ys + ayx * xs + aoy

            if not np.isfinite(xg) or not np.isfinite(yg) or r <= 0:
                continue

            # Create circular mask region
            pixcoord = regions.PixCoord(x=xg, y=yg)
            aperture = regions.CirclePixelRegion(pixcoord, r)

            try:
                mask = aperture.to_mask(mode='exact')
            except regions.NoOverlap:
                print("No Overlap when trying to make circle pixel region. Continuing...")
                continue

            bbox = aperture.bounding_box
            iymin, iymax = bbox.iymin, bbox.iymax
            ixmin, ixmax = bbox.ixmin, bbox.ixmax

            # Clip bounding box to image dimensions
            iymin_clip = max(iymin, 0)
            iymax_clip = min(iymax, ny)
            ixmin_clip = max(ixmin, 0)
            ixmax_clip = min(ixmax, nx)

            if iymin_clip >= iymax_clip or ixmin_clip >= ixmax_clip:
                continue

            mask_yslice = slice(iymin_clip - iymin, iymax_clip - iymin)
            mask_xslice = slice(ixmin_clip - ixmin, ixmax_clip - ixmin)
            target_slice = (slice(iymin_clip, iymax_clip), slice(ixmin_clip, ixmax_clip))

            if mask.data is None:
                continue

            mask_data = mask.data[mask_yslice, mask_xslice]
            new_mask_pixels = (mask_data > 0)

            # Use bitwise_or to add the flag
            ghost_mask[target_slice][new_mask_pixels] = np.bitwise_or(
                ghost_mask[target_slice][new_mask_pixels],
                flag
            )

    # Apply Band 4 Optical Bleeding Mask (if triggered)
    if band4_bleed_triggered:
        # Mask region: –52 < x < 2092, y > 1920
        _x_min, _x_max, _y_min, _y_max = bleed_roi
        x_min_bleed = max(0, _x_min)
        x_max_bleed = min(nx, _x_max)
        y_min_bleed = max(0, _y_min)
        y_max_bleed = min(ny, _y_max)

        if y_min_bleed < y_max_bleed and x_min_bleed < x_max_bleed:
            bleed_slice = (slice(y_min_bleed, y_max_bleed), slice(x_min_bleed, x_max_bleed))

            # Use the flag from the first source that triggered the bleed
            ghost_mask[bleed_slice] = np.bitwise_or(
                ghost_mask[bleed_slice],
                triggering_bleed_flag
            )

    return ghost_mask


def dqa_check_dbs_ghost_ext_corners(spximg: SPHERExImage,
                                    srccat: Table,
                                    roi_edge_band: list,
                                    corner_box_size: float = 20.0) -> dict:
    """Check that external sources exist near the corners of the DBS
    GHOST_EXT edge ROI (used for det 3 and 4).

    Parameters
    ----------
    spximg : `SPHERExImage`
        Image whose WCS defines the pixel coordinate system.
    srccat : `Table`
        Source catalog (no magnitude cut applied).
    roi_edge_band : `list`
        Edge ROI [x_min, x_max, y_min, y_max] for this band.
    corner_box_size : `float`, optional
        Size (in pixels) of the small corner boxes along each corner
        of the ROI where we require at least one external source.

    Returns
    -------
    `dict`
        DQA result dict with "result" and "description".

    """
    band = spximg.getArrayId()
    ny, nx = spximg.data.shape

    if roi_edge_band is None:
        result = {
            "result": "FAIL",
            "description": (
                f"No DBS edge ROI defined for Band {band}. "
                "Cannot perform GHOST_EXT corner check."
            )
        }
        log_dqa_result("dbs ghost_ext corner dqa", result)
        return result

    x_min, x_max, y_min, y_max = roi_edge_band

    # Compute pixel coordinates for all catalog sources
    try:
        ix, iy = spximg.wcs.world_to_pixel_values(srccat["ra"], srccat["dec"])
    except Exception as e:
        result = {
            "result": "FAIL",
            "description": (
                f"Could not compute pixel coords for DBS GHOST_EXT corner check "
                f"for Band {band}: {e}"
            )
        }
        log_dqa_result("dbs ghost_ext corner dqa", result)
        return result

    ix = np.asarray(ix)
    iy = np.asarray(iy)

    # External sources (would produce GHOST_EXT)
    external = (ix < 0) | (ix >= nx) | (iy < 0) | (iy >= ny)

    # Define two small corner boxes at opposite corners of the ROI
    # Corner 1: near (x_min, y_min)
    c1 = (ix >= x_min) & (ix <= x_min + corner_box_size) & \
         (iy >= y_min) & (iy <= y_min + corner_box_size) & external

    # Corner 2: near (x_max, y_max)
    c2 = (ix >= x_max - corner_box_size) & (ix <= x_max) & \
         (iy >= y_max - corner_box_size) & (iy <= y_max) & external

    n_c1 = int(np.count_nonzero(c1))
    n_c2 = int(np.count_nonzero(c2))

    if (n_c1 > 0) and (n_c2 > 0):
        result = {
            "result": "PASS",
            "description": (
                f"DBS GHOST_EXT corner check (Band {band}): "
                f"found {n_c1} external sources near one ROI corner and "
                f"{n_c2} near the opposite corner."
            )
        }
    else:
        missing_parts = []
        if n_c1 == 0:
            missing_parts.append("first corner box")
        if n_c2 == 0:
            missing_parts.append("second corner box")
        missing_str = " and ".join(missing_parts) if missing_parts else "corners"

        result = {
            "result": "FAIL",
            "description": (
                f"DBS GHOST_EXT corner check (Band {band}): "
                f"no external sources found in {missing_str} of the edge ROI "
                f"[x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}]. "
                "This may indicate insufficient srccat coverage near the ROI corners."
            )
        }

    log_dqa_result("dbs ghost_ext corner dqa", result)
    return result


def dqa_check_masked_sources(spximg: SPHERExImage, srccat: Table, threshold: float = 3.) -> dict:
    """Check that the number of SOURCE flagged pixels
       is within a factor of 3 of the number of sources.

    Parameters
    ----------
    spximg : `SPHERExImage`
        The SPHEREx image containing flags.
    srccat : `Table`
        The source catalog.
    threshold : `float`
        Multiplicative range (from N/threshold to N*threshold) for the
        number of flagged pixels as a function of the number N of sources.

    Returns
    -------
    `dict`
        Dictionary with "result" and "description" for Level2DQA.

    """
    mask = (spximg.flags & Flag.SOURCE) > 0
    num_masked_pixels = np.sum(mask)
    num_sources = len(srccat)
    n_total = spximg.data.size
    frac = 100 * num_masked_pixels / n_total

    if num_sources == 0:
        result = {
            "result": "FAIL",
            "description": (
                f"No sources in the catalog. {num_masked_pixels} pixels flagged as SOURCE "
                f"out of {n_total} total ({frac:.2f}%). Cannot validate source mask."
            )
        }
    else:
        lower_bound = num_sources / threshold
        upper_bound = num_sources * threshold

        if lower_bound <= num_masked_pixels <= upper_bound:
            result = {
                "result": "PASS",
                "description": (
                    f"{num_masked_pixels} SOURCE flagged pixels out of {n_total} "
                    f"({frac:.2f}%). Source count: {num_sources}. "
                    f"Masked count is within expected range "
                    f"({lower_bound:.1f}–{upper_bound:.1f})."
                )
            }
        else:
            result = {
                "result": "FAIL",
                "description": (
                    f"{num_masked_pixels} SOURCE flagged pixels out of {n_total} "
                    f"({frac:.2f}%). Source count: {num_sources}. "
                    f"Masked count is outside expected range "
                    f"({lower_bound:.1f}–{upper_bound:.1f})."
                )
            }

    log_dqa_result("source mask dqa", result)
    return result


def dqa_check_ghost_flux(spximg: SPHERExImage, threshold: float = 1000.0) -> dict:
    """Check that the flagged GHOST pixels have flux greater than the
    threshold.
    """
    ghost_mask = (spximg.flags & Flag.GHOST) > 0
    ghost_mask_ext = (spximg.flags & Flag.GHOST_EXT) > 0
    ghost_pix_flux = spximg.data[ghost_mask | ghost_mask_ext]
    n_flagged = ghost_pix_flux.size
    n_total = spximg.data.size
    frac = 100 * n_flagged / n_total

    if n_flagged == 0:
        result = {
            "result": "FAIL",
            "description": (
                f"No GHOST flagged pixels found. Total pixels: {n_total}."
            )
        }
    elif np.any(ghost_pix_flux < threshold):
        n_fail = np.sum(ghost_pix_flux < threshold)
        result = {
            "result": "FAIL",
            "description": (
                f"{n_flagged} GHOST pixels flagged out of {n_total} "
                f"({frac:.2f}%). {n_fail} have flux < {threshold:.1f} e-/s."
            )
        }
    else:
        result = {
            "result": "PASS",
            "description": (
                f"{n_flagged} GHOST pixels flagged out of {n_total} "
                f"({frac:.2f}%). All have flux > {threshold:.1f} e-/s."
            )
        }

    log_dqa_result("ghost dqa", result)
    return result


def dqa_check_crosstalk_flux(spximg: SPHERExImage, threshold: float = 0.1) -> dict:
    """Check that CROSSTALK flagged pixels are less than threshold percent
    of the full image.
    """
    xtalk_mask = (spximg.flags & Flag.CROSSTALK) > 0
    n_flagged = int(np.sum(xtalk_mask))
    n_total = int(spximg.data.size)
    frac_percent = 100.0 * n_flagged / n_total

    if frac_percent < threshold:
        result = {
            "result": "PASS",
            "description": (
                f"{n_flagged} CROSSTALK flagged pixels out of {n_total} "
                f"({frac_percent:.6f}%). This is below the allowed "
                f"{threshold:.3f}% threshold."
            )
        }
    else:
        result = {
            "result": "FAIL",
            "description": (
                f"{n_flagged} CROSSTALK flagged pixels out of {n_total} "
                f"({frac_percent:.6f}%). This exceeds the allowed "
                f"{threshold:.3f}% threshold."
            )
        }

    log_dqa_result("crosstalk dqa", result)
    return result
class CreateSourceMaskConnections(pipeBase.PipelineTaskConnections,
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
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    ssocat = cT.Input(
        name="ssocat",
        doc="Input solar system object catalog.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "observation"],
    )
    xtalk_class1 = cT.PrerequisiteInput(
        name="xtalk_class1",
        doc="Class 1 crosstalk amplitudes table.",
        storageClass="Xtalk",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )

    xtalk_class2 = cT.PrerequisiteInput(
        name="xtalk_class2",
        doc="Class 2 crosstalk coupling table.",
        storageClass="Xtalk",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    abs_gain_matrix = cT.PrerequisiteInput(
        name="abs_gain_matrix",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after source mask created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CreateSourceMaskConfig(pipeBase.PipelineTaskConfig,
                             pipelineConnections=CreateSourceMaskConnections):
    """Configuration for CreateSourceMask task."""

    sr = Field(
        "High resolution sampling rate.",
        dtype=int,
        default=5)

    brightness_cuts = ListField(
        doc="Bin edges for defining brightness intervals and associated in masking radii.",
        dtype=float,
        default=[5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.7, 18.5, 19.5, 20.5])

    masking_radii_det1 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 1.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 3.5, 3.5, 3.5, 2.5, 2.0, 1.5, 1.5, 1.5, 1.5, 1.0, 0.708])

    masking_radii_det2 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 2.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 5.0, 4.5, 4.0, 2.5, 2.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.708])

    masking_radii_det3 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 3.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.5, 3.0, 2.0, 2.0, 1.5, 1.5, 1.5, 1.0, 0.708])

    masking_radii_det4 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 4.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 3.5, 3.0, 3.0, 2.0, 2.0, 1.5, 1.5, 1.0, 0.708])

    masking_radii_det5 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 5.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 4.0, 4.0, 4.0, 2.5, 2.5, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])

    masking_radii_det6 = ListField(
        doc="Masking radius of each brightness interval in pixel units, for detector 6.",
        dtype=float,
        default=[8.0, 7.0, 6.0, 5.0, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.5, 1.5, 1.0, 1.0, 1.0])

    masking_th = Field(
        doc="Minimum fraction of (sr*sr) high resolution pixels to mark the native pixel as masked.",
        dtype=float,
        default=.24)

    save_srcmask_percent = Field(
        doc="Save the percentage of pixels with the source mask in PXSRCMSK header.",
        dtype=bool,
        default=True)

    mask_planets = Field(
        doc="Do masking for bright planets/SSOs using compute_planet_mask_radius.",
        dtype=bool,
        default=True,
    )

    planet_name_keys = ListField(
        doc=("Normalized (lowercase) planet/SSO names to mask. "),
        dtype=str,
        default=["mars", "jupiter", "saturn barycenter", "uranus", "neptune", "phoebe"],
    )

    planet_radii_jupiter = ListField(
        doc="Planet mask radius (pixels) for Jupiter, det1..det6.",
        dtype=float,
        default=[750., 500., 400., 400., 400., 400.],
    )
    planet_radii_saturn_barycenter = ListField(
        doc="Planet mask radius (pixels) for Saturn barycenter, det1..det6.",
        dtype=float,
        default=[600., 400., 200., 150., 150., 150.],
    )
    planet_radii_mars = ListField(
        doc="Planet mask radius (pixels) for Mars, det1..det6.",
        dtype=float,
        default=[600., 400., 200., 150., 150., 150.],
    )
    planet_radii_uranus = ListField(
        doc="Planet mask radius (pixels) for Uranus, det1..det6.",
        dtype=float,
        default=[50., 28., 15., 15., 15., 15.],
    )
    planet_radii_neptune = ListField(
        doc="Planet mask radius (pixels) for Neptune, det1..det6.",
        dtype=float,
        default=[30., 10., 5., 5., 5., 5.],
    )
    planet_radii_phoebe = ListField(
        doc="Planet mask radius (pixels) for Phoebe, det1..det6.",
        dtype=float,
        default=[3., 3., 3., 3., 3., 3.],
    )
    planet_name_col = Field(
        doc="Column name in ssocat containing the object name (e.g. 'Mars').",
        dtype=str,
        default="desig",
    )

    planet_flag_name = Field(
        doc=(
            "Which Flag bit to set for planet masking. "
            "If this flag name does not exist on Flag, falls back to SOURCE."
        ),
        dtype=str,
        default="SOURCE",
    )

    xtalk_amplitude_thresh = Field(
        doc="Crosstalk coupling amplitude threshold",
        dtype=float,
        default=1e-4
    )

    ghost_radius = Field(
        doc="Ghost masking radius, pixels. (Used by old mask_ghosts function/tests)",
        dtype=int,
        default=52
    )

    dbs_params_3 = ListField(
        doc="DBS ghost masking parameters, detector 3. (Used by old mask_ghosts function/tests)",
        dtype=float,
        default=[-0.0051, 0.0013, -7.4, 0.0, 255.0, 0.0]
    )

    dbs_params_4 = ListField(
        doc="DBS ghost masking parameters, detector 4. (Used by old mask_ghosts function/tests)",
        dtype=float,
        default=[-0.0040, 0.0013, -8.9, 0.0, 255.0, 0.0]
    )

    dbs_params_5 = ListField(
        doc="DBS ghost masking parameters, detector 5. (Used by old mask_ghosts function/tests)",
        dtype=float,
        default=[-0.0042, 0.0005, 0.1, 0.0, 255.0, 0.0]
    )

    dbs_params_6 = ListField(
        doc="DBS ghost masking parameters, detector 6. (Used by old mask_ghosts function/tests)",
        dtype=float,
        default=[-0.0043, 0.0013, 9.4, 0.0, 225.0, 0.0]
    )

    edge_mag_limit = Field(
        doc="Edge ghost mask magnitude limit. (Used by old mask_ghosts function/tests)",
        dtype=float,
        default=11.0
    )

    edge_offset_min = Field(
        doc="Virtual pixel edge offset minimum.",
        dtype=float,
        default=10.0
    )

    edge_offset_max = Field(
        doc="Virtual pixel edge offset maximum.",
        dtype=float,
        default=45.0
    )

    blended_radius_params_det1 = ListField(
        doc="Blended-exponential radius params for det1: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[6.020, -0.2364, 1.623, -0.048, 10.175, 2.112]
    )

    blended_radius_params_det2 = ListField(
        doc="Blended-exponential radius params for det2: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[6.097, -0.3016, 4.539, -0.212, 8.190, 1.552]
    )

    blended_radius_params_det3 = ListField(
        doc="Blended-exponential radius params for det3: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[5.845, -0.3020, 2.245, -0.075, 12.364, 2.066]
    )

    blended_radius_params_det4 = ListField(
        doc="Blended-exponential radius params for det4: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[11.193, -0.1271, 0.566, -0.001, 0.998, 6.278]
    )

    blended_radius_params_det5 = ListField(
        doc="Blended-exponential radius params for det5: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[6.005, -0.2561, 2.729, -0.096, 9.314, 1.979]
    )

    blended_radius_params_det6 = ListField(
        doc="Blended-exponential radius params for det6: [a1, b1, a2, b2, m0, w].",
        dtype=float,
        default=[5.631, -0.2801, 3.174, -0.124, 10.950, 2.007]
    )

    dbs_ghost_params_edge_3 = ListField(
        doc="DBS ghost model parameters for Band 3 (Edge): [B_lim, axx, axy, aox, ayx, ayy, aoy, r]",
        dtype=float,
        default=[100.0, -0.0051, 0.0013, -5.0, 0.0, 0.0, 225.0, 52]
    )

    dbs_ghost_params_edge_4 = ListField(
        doc="DBS ghost model parameters for Band 4 (Edge): [B_lim, axx, axy, aox, ayx, ayy, aoy, r]",
        dtype=float,
        default=[100.0, 0.0, 0.0, 10.0, 0.0, 0.0, 221.0, 52]
    )

    dbs_ghost_params_main_4 = ListField(
        doc="DBS ghost model parameters for Band 4 (Main): [B_lim, axx, axy, aox, ayx, ayy, aoy, r]",
        dtype=float,
        default=[3000.0, 0.0, 0.0, 5.0, 0.0, 0.0, 221.0, 52]
    )

    dbs_ghost_params_main_5 = ListField(
        doc="DBS ghost model parameters for Band 5 (Main): [B_lim, axx, axy, aox, ayx, ayy, aoy, r]",
        dtype=float,
        default=[10000.0, -0.0042, 0.0005, 0.1, 0.0, 0.0, 225.0, 52]
    )

    dbs_ghost_params_main_6 = ListField(
        doc="DBS ghost model parameters for Band 6 (Main): [B_lim, axx, axy, aox, ayx, ayy, aoy, r]",
        dtype=float,
        default=[10000.0, -0.0043, 0.0013, 9.4, 0.0, 0.0, 225.0, 52]
    )

    dbs_ghost_roi_edge_3 = ListField(
        doc="DBS ghost source ROI for Band 3 (Edge): [x_min, x_max, y_min, y_max]",
        dtype=int,
        default=[-52, 2092, -250, 175]
    )

    dbs_ghost_roi_edge_4 = ListField(
        doc="DBS ghost source ROI for Band 4 (Edge): [x_min, x_max, y_min, y_max]",
        dtype=int,
        default=[-52, 2092, 1535, 2040]
    )

    dbs_ghost_roi_main_4 = ListField(
        doc="DBS ghost source ROI for Band 4 (Main): [x_min, x_max, y_min, y_max]",
        dtype=int,
        default=[-52, 2092, 52, 2261]
    )

    dbs_ghost_roi_main_5 = ListField(
        doc="DBS ghost source ROI for Band 5 (Main): [x_min, x_max, y_min, y_max]",
        dtype=int,
        default=[-52, 2092, 52, 2265]
    )

    dbs_ghost_roi_main_6 = ListField(
        doc="DBS ghost source ROI for Band 6 (Main): [x_min, x_max, y_min, y_max]",
        dtype=int,
        default=[-52, 2092, 52, 2265]
    )

    band4_bleed_flux_thresh = Field(
        doc="Flux threshold (Jy) for Band 4 optical bleeding.",
        dtype=float,
        default=3000.0
    )

    band4_bleed_roi = ListField(
        doc="Band 4 optical bleeding mask region [x_min, x_max, y_min, y_max].",
        dtype=int,
        default=[-52, 2092, 1920, 2048]  # Default y_max is clipped to ny.
    )

    mask_ghosts = Field(
        doc="Do ghost masking.",
        dtype=bool,
        default=True
    )

    mask_xtalk = Field(
        doc="Do crosstalk masking.",
        dtype=bool,
        default=False
    )

    xtalk_threshold = Field(
        doc="Threshold in e-/s for Class 1 and Class 2 crosstalk images.",
        dtype=float,
        default=0.3,
    )

    xtalk_gaussian_sigma = Field(
        doc="Gaussian sigma in pixels for crosstalk pre-filtering.",
        dtype=float,
        default=1.5,
    )

    class1_swir_detectors = ListField(
        doc="Detector IDs on which to apply Class 1 crosstalk masking.",
        dtype=int,
        default=[1, 2, 3],
    )


class CreateSourceMaskTask(pipeBase.PipelineTask):
    """Implementation of L2 Create Source Mask.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-019

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "create-source-mask"
    ConfigClass = CreateSourceMaskConfig
    full_name = "Create Source Mask"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-019 v3.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            srccat: Table,
            ssocat: Table = None,
            xtalk_class2: Table | None = None,
            xtalk_class1: Table | None = None,
            abs_gain_matrix: SPHERExImage | None = None,
            plot: bool = False,
            verbose: bool = False) -> pipeBase.Struct:
        """Update SOURCE flag to mask pixels containing signal from known
        sources.

        Parameters
        ----------
        spximg : `SPHERExImage`
            Input image.
        srccat : `Table`
            Catalog of the sources within or close to the image
        ssocat : `Table`
            Table with the positions of SSO objects in image
        xtalk_coupling : `SPHERExImage`
            Crosstalk coupling calibration values, per-detector.
        abs_gain_matrix : `SPHERExImage`
            Absolute gain matrix.
        plot : `bool`, optional
            Flag for plotting analysis, default is False.
        verbose : `bool`, optional
            Flag to printout masking radius and catalog info,
            default is False.

        Returns
        -------
        output: `Output`
            An `Output` object containing the corrected ``image``
            as an attribute of type `~spx_data.SPHERExImage`.

        """
        if len(srccat) == 0:
            # If the catalog is empty, return the input image unchanged.
            spximg_out = copy.copy(spximg)
            return pipeBase.Struct(spximg_out=spximg_out)

        catalog = srccat.copy()
        det_id = spximg.getArrayId()
        rp = ReadoutParameters.from_metadata(spximg.meta)
        nx = rp.Jactive
        ny = rp.Kactive
        rind = rp.jref // 2

        sky_image = np.zeros([nx, ny])
        # sky_image[rind: nx-rind, rind: ny-rind] = spximg.data
        sky_image[rind: nx-rind, rind: ny-rind] = spximg.data[rind:-rind, rind:-rind]

        # Ensure brightness column exists in the catalog.
        b_col = f"mag_det{det_id}"
        assert b_col in catalog.colnames, f"Catalog is missing column '{b_col}' for source brightness."

        blended_params_dict = {1: self.config.blended_radius_params_det1,
                               2: self.config.blended_radius_params_det2,
                               3: self.config.blended_radius_params_det3,
                               4: self.config.blended_radius_params_det4,
                               5: self.config.blended_radius_params_det5,
                               6: self.config.blended_radius_params_det6,
                               }

        planet_mask_radii = {
            "jupiter": list(self.config.planet_radii_jupiter),
            "saturn barycenter": list(self.config.planet_radii_saturn_barycenter),
            "mars": list(self.config.planet_radii_mars),
            "uranus": list(self.config.planet_radii_uranus),
            "neptune": list(self.config.planet_radii_neptune),
            "phoebe": list(self.config.planet_radii_phoebe),
        }

        # Mask construction (SOURCE / CROSSTALK / GHOST) is WCS-dependent.
        # If WCS blows up with InvalidCoordinateError and FINAST!=0, skip masks.
        try:

            source_mask = mask_sources(spximg=spximg,
                                       cat=catalog,
                                       blended_params_dict=blended_params_dict,)

            planet_mask = np.zeros_like(source_mask, dtype=bool)
            if (self.config.mask_planets and ssocat_has_any_planets(ssocat,
                                                                    self.config.planet_name_col,
                                                                    self.config.planet_name_keys)):
                planet_mask = mask_planets(
                    spximg=spximg,
                    ssocat=ssocat,
                    planet_mask_radii=planet_mask_radii,
                    name_col=self.config.planet_name_col,
                    ra_col="RA",
                    dec_col="DEC")

            # Crosstalk masking
            xtalk_mask = np.zeros_like(source_mask, dtype=bool)
            if self.config.mask_xtalk:
                if xtalk_class2 is None:
                    self.log.warning(
                        "mask_xtalk=True but no xtalk_class2 was provided; "
                        "skipping crosstalk masking for this run."
                    )
                elif abs_gain_matrix is None:
                    self.log.warning(
                        "mask_xtalk=True but no abs_gain_matrix was provided; "
                        "skipping crosstalk masking for this run."
                    )
                else:
                    # Convert calibrated MJy/sr image back to dark-subtracted e-/s
                    # for crosstalk masking only.
                    spximg_xtalk = copy.copy(spximg)
                    gain = np.asarray(abs_gain_matrix.data, dtype=float)
                    data = np.asarray(spximg.data, dtype=float)

                    spximg_xtalk.data = np.full_like(data, np.nan, dtype=float)
                    good = np.isfinite(gain) & (gain != 0) & np.isfinite(data)
                    spximg_xtalk.data[good] = data[good] / gain[good]

                    xtalk_mask = mask_crosstalk(
                        spximg=spximg_xtalk,
                        xtalk_class2=xtalk_class2,
                        xtalk_class1=xtalk_class1,
                        xtalk_threshold=self.config.xtalk_threshold,
                        gaussian_sigma=self.config.xtalk_gaussian_sigma,
                        class1_swir_detectors=tuple(self.config.class1_swir_detectors),
                    )

            # Assemble DBS ghost parameter dicts from config
            dbs_params_edge = {
                3: self.config.dbs_ghost_params_edge_3,
                4: self.config.dbs_ghost_params_edge_4,
            }
            dbs_params_main = {
                4: self.config.dbs_ghost_params_main_4,
                5: self.config.dbs_ghost_params_main_5,
                6: self.config.dbs_ghost_params_main_6,
            }
            dbs_roi_edge = {
                3: self.config.dbs_ghost_roi_edge_3,
                4: self.config.dbs_ghost_roi_edge_4,
            }
            dbs_roi_main = {
                4: self.config.dbs_ghost_roi_main_4,
                5: self.config.dbs_ghost_roi_main_5,
                6: self.config.dbs_ghost_roi_main_6,
            }

            ghost_mask = np.zeros_like(sky_image, dtype=np.int32)
            if self.config.mask_ghosts:
                # Transpose (.T) the result (Kactive, Jactive) to (Jactive, Kactive)
                # to match the shape of source_mask, xtalk_mask, and spximg.flags
                ghost_mask = mask_dbs_ghosts(
                    spximg=spximg,
                    cat=catalog,
                    params_edge=dbs_params_edge,
                    params_main=dbs_params_main,
                    roi_edge=dbs_roi_edge,
                    roi_main=dbs_roi_main,
                    bleed_flux_thresh=self.config.band4_bleed_flux_thresh,
                    bleed_roi=self.config.band4_bleed_roi,
                    verbose=verbose
                )
        except Exception as e:
            finast = int(spximg.header.get("FINAST", 0))
            if finast != 0:
                self.log.warning(
                    "CreateSourceMask: WCS-related failure with FINAST=%s; "
                    "skipping SOURCE/CROSSTALK/GHOST masking. Error: %s",
                    finast, str(e)
                )
                spximg_out = copy.copy(spximg)
                if self.config.save_srcmask_percent:
                    spximg_out.header["PXSRCMSK"] = (0.0, "Percentage of pixels with SOURCE flag")
                spximg_out.header["HIERARCH L2 N_SOURCE"] = (0, "number of source flagged pixels")
                if self.config.mask_xtalk:
                    spximg_out.header["HIERARCH L2 N_XTALK"] = (0, "number of crosstalk flagged pixels")
                if self.config.mask_ghosts:
                    spximg_out.header["HIERARCH L2 N_GHOST"] = (0, "number of ghost flagged pixels")
                    spximg_out.header["HIERARCH L2 N_GHOST_FPA"] = (0, "number of ghost_fpa flagged pixels")
                    spximg_out.header["HIERARCH L2 N_GHOST_EXT"] = (0, "number of ghost_ext flagged pixels")

                return pipeBase.Struct(spximg_out=spximg_out)

            # if FINAST==0, then this is a real bug.
            raise

        # Combine source and crosstalk masks
        # combined_mask = np.logical_or(source_mask, xtalk_mask)
        # Align xtalk_mask to same shape as source_mask before logical_or
        rp = ReadoutParameters.from_metadata(spximg.meta)

        # Masks are already shape (Jactive, Kactive)
        if self.config.mask_xtalk:
            combined_mask = np.logical_or(source_mask, xtalk_mask)
        else:
            combined_mask = source_mask

        # Include planet mask in the combined mask
        # (so PXSRCMSK / filtering treat it as masked)
        if self.config.mask_planets and planet_mask is not None:
            combined_mask = np.logical_or(combined_mask, planet_mask)

        whpl_compmask = combined_mask > 0.1

        # Create compmask of same shape directly
        compmask = np.ones_like(combined_mask, dtype=int)
        compmask[whpl_compmask] = 0

        # Masks
        whpl_type1mask = source_mask > 0.1  # SOURCE
        # Also set SOURCE for planets
        if self.config.mask_planets and planet_mask is not None:
            whpl_type1mask |= planet_mask
        if self.config.mask_xtalk:
            whpl_type2mask = xtalk_mask > 0.1  # CROSSTALK

        # Plot the masked image with better visualization
        if plot:
            fig, ax = plt.subplots()
            # Display the original image
            ax.imshow(sky_image, cmap='gray', vmin=0, vmax=2000)
            # Overlay the mask with transparency
            ax.imshow(compmask, cmap='Reds', alpha=0.5, origin="lower")
            plt.title('Masked data with highlighted masked regions')
            plt.show()
        if plot:
            fig, ax = plt.subplots()
            ax.imshow(sky_image, cmap='gray', vmin=0, vmax=2000, origin="lower")
            ax.imshow(whpl_type1mask, cmap='Blues', alpha=0.5, origin="lower")
            plt.title("Source Mask Overlay")
            plt.show()
        # if plot:
        #     fig, ax = plt.subplots()
        #     # Display the original image
        #     ax.imshow(sky_image, cmap='gray', vmin=0, vmax=2000, origin="lower")
        #     # Overlay mask as contours
        #     ax.contour(whpl_type2mask, levels=[0.5], colors='red', linewidths=1)
        #     plt.title("Crosstalk Mask Overlay")
        #     plt.show()
        ########################################################################
        # filter out plane so we can inspect the distribution of remaining values
        # this may seem odd, but in fact the gradient in source-free pixels
        # complicates interpretation and our lives are easier if it's subtracted.
        # plus, I guess this is a free check that ZL gradient filtering will work

        # first, make a copy of the sky before we do anything to it
        unfiltered_data = sky_image

        # now, if less than 90% of the pixels are masked we might hope to do something
        if (1. - compmask.sum() / (nx * ny)) < 0.9:

            # find a logical representation as compmask is int
            # and python doesn't want to index against it
            whpl_compmask = compmask == 1

            # now fit and subtract a gradient from the unmasked pixels
            # make a pixel grid
            X1, X2 = np.mgrid[:nx, :ny]

            # pull out the pixels and data that are unmasked
            XX1 = X1[whpl_compmask]
            XX2 = X2[whpl_compmask]
            Y = unfiltered_data[whpl_compmask]
            nsamps = whpl_compmask.sum()

            # make a design matrix for a linear 2-d fit
            A = np.c_[np.ones(nsamps), XX1, XX2, XX1*XX2, XX1**2, XX2**2]
            # do the fit
            valid = np.isfinite(Y)
            A_valid = A[valid]
            Y_valid = Y[valid]
            C, _, _, _ = scipy.linalg.lstsq(A_valid, Y_valid)

            # evaluate it on a grid
            plane = np.dot(np.c_[np.ones(X1.shape).flatten(),
                                 X1.flatten(), X2.flatten(),
                                 X1.flatten()*X2.flatten(),
                                 X1.flatten()**2, X2.flatten()**2], C).reshape(X1.shape)

            # subtract the modeled gradient from the image
            filtered_data = unfiltered_data - plane

            # compute the histogram of the filtered pixels
            stats_histout, stats_histbins = np.histogram(filtered_data[whpl_compmask],
                                                         bins=100, range=(-200, 200))
            hist_centers = (stats_histbins[:-1] + stats_histbins[1:]) / 2

            # do the same thing for just the xtalking pixels
            # to demonstrate what's being removed
            # whpl_xtalk = xtalk_mask > 0.1
            # stats_histxtk, _ =
            # np.histogram(filtered_data[whpl_xtalk], bins=100, range=(-200, 200))

            # and nan out the filtered pixels for further calculation
            filtered_data[whpl_compmask == 0] = np.nan

            # plot the filtered data
            if plot:
                plt.imshow(filtered_data, vmin=-50, vmax=50, origin="lower")
                plt.title("Spatially filtered data")
                plt.show(block=False)

            # plot the histograms
            if plot:
                plt.semilogy(hist_centers, stats_histout, label="Unmasked pixels")
                # plt.semilogy(hist_centers, stats_histxtk, label="X-Talking pixels")
                plt.title("Histogram of remaining pixels.")
                plt.legend()
                plt.show(block=False)

            # this is how many pixels we lost
            if verbose:
                print(f"Final mask fraction is: {(1-whpl_compmask.sum()/(nx*ny))}")
                # tell me STD of the remainder
                print(f"STD of remaining pixels is: {np.std(filtered_data[whpl_compmask])}")
                # tell me the MAD of the remainder
                print(f"MAD of remaining pixels is: {astrostats.mad_std(filtered_data[whpl_compmask])}")
                # tell me the maximum remaining pixel value
                print(f"Maximum remaining pixel value is: {filtered_data[whpl_compmask].max()}")
                # tell me the maximum remaining pixel value
                print(f"Minimum remaining pixel value is: {filtered_data[whpl_compmask].min()}")

        else:
            # if we lost most of the data, just remove an offset and move on
            filtered_data = unfiltered_data - np.median(unfiltered_data)

        # Set flags based on the masks
        spximg_out = copy.copy(spximg)

        # Set source mask flag
        spx_source = whpl_type1mask  # whpl_type1mask[rp.jref//2: -rp.jref//2, rp.kref//2: -rp.kref//2]
        spximg_out.setFlagsFromMask(spx_source, Flag.SOURCE)

        # Set crosstalk mask flag
        if self.config.mask_xtalk:
            spx_xtalk = whpl_type2mask  # whpl_type2mask[rp.jref//2: -rp.jref//2, rp.kref//2: -rp.kref//2]
            spximg_out.setFlagsFromMask(spx_xtalk, Flag.CROSSTALK)

        # Set ghost mask flags
        if self.config.mask_ghosts:
            # ghost_mask is an integer mask (Jactive, Kactive)
            # with flag bits already set.
            # Create boolean masks for each flag type and apply them.
            ghost_bool = (ghost_mask & Flag.GHOST) > 0
            ghost_fpa_bool = (ghost_mask & Flag.GHOST_FPA) > 0
            ghost_ext_bool = (ghost_mask & Flag.GHOST_EXT) > 0

            spximg_out.setFlagsFromMask(ghost_bool, Flag.GHOST)
            spximg_out.setFlagsFromMask(ghost_fpa_bool, Flag.GHOST_FPA)
            spximg_out.setFlagsFromMask(ghost_ext_bool, Flag.GHOST_EXT)

        # save percentage of the masked pixels
        if self.config.save_srcmask_percent:
            srcmask_percent = round(100.0 * np.count_nonzero(whpl_type1mask) / (nx * ny), 2)
            spximg_out.header["PXSRCMSK"] = (
                srcmask_percent,
                "Percentage of pixels with SOURCE flag"
            )

        # Perform validation checks
        _ = dqa_check_masked_sources(spximg_out, srccat)
        _ = dqa_check_ghost_flux(spximg_out)

        # Additional DBS GHOST_EXT coverage check for Bands 3 and 4
        if self.config.mask_ghosts and det_id in (3, 4):
            roi_edge_band = dbs_roi_edge.get(det_id)
            _ = dqa_check_dbs_ghost_ext_corners(spximg=spximg_out,
                                                srccat=srccat,
                                                roi_edge_band=roi_edge_band,
                                                corner_box_size=20.0,  # pix
                                                )

        # Add DQA info and stats to header
        numpix_flagged_source = len(spximg_out.data[(spximg_out.flags & Flag.SOURCE) > 0])
        # Number of pixels source flagged
        spximg_out.header["HIERARCH L2 N_SOURCE"] = (numpix_flagged_source,
                                                     "number of source flagged pixels")
        # Number of pixels xtalk flagged
        if self.config.mask_xtalk:
            numpix_flagged_xtalk = np.count_nonzero((spximg_out.flags & Flag.CROSSTALK))
            spximg_out.header["HIERARCH L2 N_XTALK"] = (numpix_flagged_xtalk,
                                                        "number of crosstalk flagged pixels")
        # Number of pixels ghost flagged
        if self.config.mask_ghosts:
            numpix_flagged_ghost = np.count_nonzero((spximg_out.flags & Flag.GHOST))
            numpix_flagged_ghost_fpa = np.count_nonzero((spximg_out.flags & Flag.GHOST_FPA))
            numpix_flagged_ghost_ext = np.count_nonzero((spximg_out.flags & Flag.GHOST_EXT))

            spximg_out.header["HIERARCH L2 N_GHOST"] = (numpix_flagged_ghost,
                                                        "number of ghost flagged pixels")
            spximg_out.header["HIERARCH L2 N_GHOST_FPA"] = (numpix_flagged_ghost_fpa,
                                                            "number of ghost_fpa flagged pixels")
            spximg_out.header["HIERARCH L2 N_GHOST_EXT"] = (numpix_flagged_ghost_ext,
                                                            "number of ghost_ext flagged pixels")

        # Calculate BLOOM/SNOW pixels that are not flagged as SOURCE
        is_source = (spximg_out.flags & Flag.SOURCE) > 0
        is_bloom = (spximg_out.flags & Flag.BLOOM) > 0
        is_snow = (spximg_out.flags & Flag.SNOWBALL) > 0

        # Count BLOOM pixels that are not SOURCE
        numpix_bloom_not_source = np.count_nonzero(is_bloom & ~is_source)

        # Count SNOW pixels that are not SOURCE
        numpix_snow_not_source = np.count_nonzero(is_snow & ~is_source)

        # Add counts to the header
        spximg_out.header["HIERARCH L2 N_BLOOM_NS"] = (numpix_bloom_not_source,
                                                       "number of BLOOM flagged pixels not flagged SOURCE")
        spximg_out.header["HIERARCH L2 N_SNOW_NS"] = (numpix_snow_not_source,
                                                      "number of SNOW flagged pixels not flagged SOURCE")

        return pipeBase.Struct(spximg_out=spximg_out)
