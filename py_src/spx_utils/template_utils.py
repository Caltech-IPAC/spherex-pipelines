__all__ = ["create_masks", "fit_airglow_he", "subtract_airglow_he",
           "attach_airglow_fit_to_header", "read_airglow_fit_from_header",
           "reconstruct_p_fit_from_results", "correlate_image",
           "template_correlate", "correlate_image", "HE_FIT_BKG_KWS"]

# Standard library imports

# Related third party imports
from astropy.coordinates import SkyCoord
from astropy.io.fits import HDUList, Header
from astropy.modeling import models, fitting
from astropy.modeling.core import Fittable2DModel
from astropy.modeling.models import Polynomial2D
from astropy.stats import sigma_clip, sigma_clipped_stats
import numpy as np
from typing import Tuple
from skimage.measure import block_reduce

# Local application/library specific imports
from spx_data import SPHERExImage, Flag

# ===================================================================
HE_TEMP_PAR_KWS = {
    "ag_wavelength": "HE_WLAIR",  # airglow wavelength
    "ag_halfwidth": "HE_HWAIR",  # airglow half-width (wavelength direction)
    "ag_halfwidth_bkg": "HE_HWBKG",  # half-width of background region (in terms of wavelength)
    "wl_top_1": "HIERARCH HE_WLTOP1",  # start wavelength of the profile plateau (90% of the peak)
    "wl_top_2": "HIERARCH HE_WLTOP2"  # end wavelength of the profile plateau (90% of the peak)
}

HE_FIT_BKG_KWS = {
    "amplitude": "AMPL",
    "slope": "XSLOPE",
    "chi2red": "CHI2R",
    "rms_peak": "RMS_PK",
    "bkg_median": "BKG_MED",
    "bkg_degree": "BKG_DEG",
    "bkg_coeff": "BKG_COEFF"
}


def _safe_float(val, default=0.0):
    try:
        val = float(val)
        if not np.isfinite(val):
            return default
        return val
    except Exception:
        return default


def create_masks(wmap: np.ndarray,
                 flag: np.ndarray,
                 mask_source: bool = False,
                 **kwargs) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create Boolean masks for using wavelength map and quality flags.

    Parameters
    ----------
    wmap : `np.ndarray`
        The 2D map of wavelengths or wavelength-like values.
    flag : `np.ndarray`
        The 2D array of quality/flag bits corresponding to the data.
    mask_source : `bool`, optional
        If True, only pixels with flag == 0 are considered 'good'.
        If False, pixels with flag == 0 OR flag == 2**21 (Source mask bit)
        are considered 'good', allowing source-contaminated pixels
        to be masked separately. Default is `False`.
    **kwargs : `dict`
        A dictionary containing the following required parameters:
        WL_AIR : `float`
            The central wavelength of the airglow.
        HW_AIR : `float`
            The half-width (in wavelength units) of the airglow.
        HW_BKG : `float`
            The half-width defining the outer boundary for the
            background region.

    Returns
    -------
    `tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]`
        mask_airglow : (bool) Mask including only airglow region
        mask_all     : (bool) airglow + background
        mask_good    : (bool) mask_airglow & good_pixel
        mask_bkg     : (bool) background & good_pixel

    """
    WL_AIR = kwargs['WL_AIR']
    HW_AIR = kwargs['HW_AIR']
    HW_BKG = kwargs['HW_BKG']

    # source mask bit = 21
    mask_airglow = np.abs(wmap - WL_AIR) < HW_AIR
    mask_all = np.abs(wmap - WL_AIR) < HW_BKG

    if flag is None:
        flag = np.zeros_like(wmap, dtype=np.bool)

    if mask_source:
        mask_flag = (flag == 0)
    else:
        mask_flag = (flag == 0) | (flag == 2**21)

    mask_good = mask_airglow & mask_flag
    mask_bkg = ((np.abs(wmap - WL_AIR) > HW_AIR)
                & (np.abs(wmap - WL_AIR) < HW_BKG)
                & mask_flag)

    return mask_airglow, mask_good, mask_bkg, mask_all


def fit_background(data: np.ndarray,
                   xx: np.ndarray,
                   yy: np.ndarray,
                   mask_bkg: np.ndarray) -> Tuple[Fittable2DModel, np.ndarray]:
    """Fit a 2D second-order polynomial background model with sigma clipping
    for outlier rejection. The fit is performed only on pixels selected by
    `mask_bkg`.

    Parameters
    ----------
    data : `np.ndarray`
        2D array containing the image data values to be modeled
        (e.g., flux or surface brightness).

    xx : `np.ndarray`
        2D array of x-coordinates corresponding to `data`. Typically represents
        the spatial direction.

    yy : `np.ndarray`
        2D array of y-coordinates corresponding to `data`. Typically represents
        the wavelength or vertical direction.

    mask_bkg : `np.ndarray` of `bool`
        Boolean mask selecting background pixels to be used in the fit.
        Must have the same shape as `data`.

    Returns
    -------
    model : `astropy.modeling.core.Fittable2DModel`
        The fitted 2D polynomial background model.

    filtered_data : `np.ndarray`
        Array of the input `z` values after sigma-clipping, corresponding to
        the points used in the final fit. This is returned by
        `FittingWithOutlierRemoval`.

    Raises
    ------
    ValueError
        If `mask_bkg` has fewer pixels than the model has free parameters.

    Notes
    -----
    - The background uncertainty is estimated using the standard deviation
      from sigma-clipped statistics of the background pixels.
    - All background points are assigned equal uncertainty based on this
      estimate.
    - The polynomial model is of degree 2 in both x and y.

    """
    # Select background pixels
    x = xx[mask_bkg]
    y = yy[mask_bkg]
    z = data[mask_bkg]

    # Exclude non-finite pixels. Leaving them in is fatal: `z * 0` below is
    # NaN (not 0) wherever z is NaN, so even a single bad pixel poisons the
    # per-pixel weights passed to the linear fitter and crashes it.
    valid = np.isfinite(z)
    if not np.all(valid):
        x, y, z = x[valid], y[valid], z[valid]

    # Define initial polynomial model
    p_init = models.Polynomial2D(degree=2)

    # An underfilled mask_bkg (when too much of it flagged out) leaves
    # too few points to solve for all model parameters. astropy's fitter
    # doesn't fail cleanly in that case, check explicitly and fail with
    # a clear message instead.
    n_params = len(p_init.parameters)
    if z.size < n_params:
        raise ValueError(
            f"{z.size} valid pixel(s) for background fit."
        )

    # Estimate background uncertainty from sigma-clipped statistics
    zunc = z * 0 + sigma_clipped_stats(z)[2]

    # Set up fitter with sigma-clipping
    fitter = fitting.FittingWithOutlierRemoval(
        fitting.LinearLSQFitter(),
        sigma_clip,
        niter=10,
        sigma=3,
    )

    return fitter(p_init, x, y, z, weights=1.0 / zunc**2)


def compute_proportional_coefficient(imageA: np.ndarray,
                                     imageB: np.ndarray,
                                     variance: np.ndarray = None,
                                     mask: np.ndarray = None,
                                     lsigma: int = 5,
                                     hsigma: int = 3,
                                     max_iter: int = 10,
                                     return_mask: bool = True,
                                     verbose: bool = False) -> tuple:
    """Compute the proportional coefficient 'k' that minimizes the chi-square
    difference between imageB and k * imageA.

    Parameters
    ----------
    imageA : `np.ndarray`
        The reference 2D image. (template)
    imageB : `np.ndarray`
        The target 2D image to fit. (data)
    variance : `np.ndarray`, optional
        The variance 2D image of imageB
    mask : `np.ndarray`, optional
        A boolean mask to specify which pixels
        to include in the fit.
    lsigma : `int`
        Lower sigma threshold for clipping residuals. Default is 5.
        (keeps pixels with residual > -lsigma*rms).
    hsigma : `int`
        Upper sigma threshold for clipping residuals. Default is 3.
        (keeps pixels with residual < hsigma*rms).
    max_iter : `int`
        Number of sigma-clipping iterations. Default is 10.
    return_mask : `bool`
        If True, return the final clipped mask. Default is `True`.
    verbose : `bool`
        If True, print per-iteration diagnostics. Default is `True`.

    Returns
    -------
    `tuple`
    k : `float`
        The best-fit proportional coefficient.
    chi2 : `float`
        The minimized chi-square value.
        if variance == None, sqrt(chi2/n_pixel) = rms
    rms : `float`
        rms deviation of (data - model)

    """
    if mask is None:
        mask = np.ones_like(imageB, dtype=np.bool)

    if variance is None:
        varianceB = np.ones_like(imageB, dtype=np.float32)
    else:
        varianceB = variance

    # Pre-calculate the inverse variance (weight)
    weight = 1.0 / varianceB

    for i in range(max_iter):
        # Numerator: Sum (A * B / VarB)
        # Denominator: Sum (A^2 / VarB)
        numerator = np.sum(imageA[mask] * imageB[mask] * weight[mask])
        denominator = np.sum(imageA[mask] * imageA[mask] * weight[mask])
        k = numerator / denominator

        # Compute chi-square & rms
        residual = imageB - k * imageA
        chi2 = np.sum(residual[mask] ** 2 * weight[mask])
        rms = np.sqrt(np.sum(residual[mask] ** 2) / mask.sum())

        # Update mask
        mask_clipped = ((imageB - k*imageA) > -lsigma*rms) & ((imageB - k*imageA) < hsigma*rms)
        mask = mask & mask_clipped

        if verbose:
            if i == 0:
                print(f"{'iter':>4s}  {'ampl':>10s}  {'chi2':>10s}  {'rms':>8s}")
            print(f"{i:4d}  {k:10.4f}  {chi2:10.4e}  {rms:8.4f}")

    if return_mask:
        return k, chi2, rms, mask
    else:
        return k, chi2, rms


def compute_linear_scaling_coefficient(imageA, imageB, variance=None, mask=None,
                                       lsigma=5, hsigma=3, max_iter=10,
                                       return_mask=True, verbose=False) -> tuple:
    """Compute the linear scaling coefficients k0 and k1 that minimize the
    chi-square difference between imageB and (k0 + k1*(X-X0)) * imageA,
    where X is the 2D array of column indices (x-coordinate).

    The model is: imageB ~ [k0 + k1*(X-X0)] * imageA

    Parameters
    ----------
    imageA : `np.ndarray`
        The reference 2D image (template).
    imageB : `np.ndarray`
        The target 2D image to fit (data).
    variance : `np.ndarray`, optional
        The variance 2D image of imageB.
    mask : `np.ndarray`, optional
        A boolean mask for good pixels [good = True]
    lsigma : `int`
        Lower sigma threshold for clipping residuals. Default is 5.
    hsigma : `int`
        Upper sigma threshold for clipping residuals. Default is 3.
    max_iter : `int`
        Number of sigma-clipping iterations. Default is 10.
    return_mask : `bool`
        If True, return the final clipped mask. Default is `True`.
    verbose : `bool`
        If True, print per-iteration diagnostics. Default is `False`.

    Returns
    -------
    `tuple`
    k0 : `float`
        The constant scaling coefficient.
    k1 : `float`
        The x-direction slope coefficient.
    chi2 : `float`
        The minimized chi-square value.
    rms : `float`
        RMS deviation of (data - model).
    mask : `np.ndarray`, optional
        The final mask if return_mask is True.

    """
    if mask is None:
        mask = np.ones_like(imageB, dtype=bool)

    NY, NX = imageB.shape
    xv, yv = np.arange(NX)-NX/2, np.arange(NY)
    xx, _ = np.meshgrid(xv, yv)

    if variance is None:
        varianceB = np.ones_like(imageB, dtype=np.float32)
    else:
        varianceB = variance

    weight = 1.0 / varianceB

    for i in range(max_iter):
        # Setup the Linear System: y = k * M
        # M is the design matrix (N_pixels x 2)
        # M = [A, (X-X0)*A]
        # k = [k0, k1]

        # Apply mask to all relevant 2D arrays and flatten them
        w = weight[mask]
        a = imageA[mask]
        b = imageB[mask]
        x_coords = xx[mask]

        # Construct the design matrix M_masked
        # Column 1: A
        # Column 2: X * A
        M = np.vstack([a, x_coords * a]).T

        # Apply the weights: M_w = W * M, y_w = W * b
        # where W is the diagonal matrix of sqrt(w)
        sqrt_w = np.sqrt(w)
        M_w = M * sqrt_w[:, np.newaxis]
        y_w = b * sqrt_w

        # Solve the Weighted Least Squares problem
        # The solution k_vec = [k0, k1] minimizes || M_w @ k_vec - y_w ||^2
        try:
            k_vec, residuals, rank, s = np.linalg.lstsq(M_w, y_w, rcond=None)
            k0, k1 = k_vec
        except np.linalg.LinAlgError as e:
            if i == 0:
                # No k0/k1/chi2/rms from a prior iteration to fall back on
                # (e.g. all-zero/non-finite variance made the weighted
                # design matrix degenerate); raise instead of letting the
                # return statement below crash with an UnboundLocalError.
                raise ValueError(
                    "Linear scaling fit failed on the first iteration "
                    "(check for zero or non-finite variance values)"
                ) from e
            # Handle singular matrix case; keep the previous iteration's fit.
            print("Warning: Least squares solution failed due to LinAlgError.")
            break

        model = (k0 + k1 * xx) * imageA
        residual = imageB - model

        chi2 = np.sum(residual[mask]**2 * weight[mask])
        rms = np.sqrt(np.sum(residual[mask]**2) / mask.sum())
        normalized_residual = residual / np.sqrt(varianceB)

        mask_clipped = (normalized_residual > -lsigma) & (normalized_residual < hsigma)
        mask = mask & mask_clipped

        if verbose:
            if i == 0:
                print(f"{'iter':>4s} {'k0':>10s} {'k1':>10s} {'chi2':>10s} {'rms':>8s}")
            print(f"{i:4d} {k0:10.4f} {k1:10.4e} {chi2:10.4e} {rms:8.4f}")

    if return_mask:
        return k0, k1, chi2, rms, mask
    else:
        return k0, k1, chi2, rms


def fit_airglow_he(spximg: SPHERExImage,
                   band: int,
                   he_template_img: SPHERExImage,
                   spectral_wcs: HDUList,
                   params_airglow_helium: dict,
                   fit_back: bool = True,
                   use_variance: bool = False,
                   fit_xslope: bool = False,
                   verbose: bool = False):
    """Fit He airglow template parameters and return results needed for
    subtraction. Does NOT return a subtracted image.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Imnput SPHEREx image.
    band : `int`
        Detector ID.
    he_template_img : `SPHERExImage`
        He I Airglor template.
    spectral_wcs : `HDUList`
        Wavelength map array.
    params_airglow_helium : `dict`
        Airglow helium parameters.
    fit_back : `bool`
        If True, subtract the fitted background model within the airglow region
        before fitting the template scaling, and report ``AG_BACK`` as the
        median background in that region. If False, the template fit is done on
        the original data and ``AG_BACK=0``. Default is True.
    use_variance : `bool`
        If True, uses ``spximg.uncertainty.array`` as the variance image in the
        coefficient fit. If False, uses unit weights. Default is False.
    fit_xslope : `bool`
        If True, fits a model ``k(x) = k0 + k1*(x - NX/2)`` for the template
        scaling. If False, fits a single proportional scale ``k0``. Default is
        False.
    verbose : `bool`
        If True, prints diagnostics from the underlying coefficient fitting.
        Default is False.

    Returns
    -------
    restuls : `dict`
    AG_AMPL : `float`
        fitted template amplitude (k0)
    AG_XSLP : `float`
        fitted x-slope term (k1; 0.0 if ``fit_xslope=False``)
    AG_CHI2M : `float`
        chi2 divided by number of used (clipped) pixels
    AG_RMS : `float`
        RMS of post-template-subtraction residuals in WL_TOP window
    AG_BACK : `float`
        median fitted background in the airglow region (0.0 if not fit_back)
    AG_BKG : `dict`
        background model serialization:
        {"degree": int, "params": {param_name: float, ...}}
    _NX : `int`
        image NX used for x-centering

    Notes
    -----
    The returned background serialization ``AG_BKG`` is stored in header
    and later reconstructed into a ``Polynomial2D`` model for subtraction.

    """
    data = spximg.data
    flag = spximg.flags
    wmap = spectral_wcs["CWAVE"].data
    wl_top = params_airglow_helium["WL_TOP"]
    he_template = he_template_img.data

    variance = np.array(spximg.uncertainty.array) if use_variance else None

    NY, NX = data.shape
    xx, _ = np.meshgrid(np.arange(NX), np.arange(NY))
    yy = wmap

    mask_airglow, mask_good, mask_bkg, mask_all = create_masks(wmap, flag, **params_airglow_helium)
    p_fit, _ = fit_background(data, xx, yy, mask_bkg)

    if fit_back:
        data_for_fit = data.copy()
        data_for_fit[mask_airglow] = data_for_fit[mask_airglow] - p_fit(xx[mask_airglow], yy[mask_airglow])
        bkg_med = np.median(p_fit(xx[mask_airglow], yy[mask_airglow]))
    else:
        data_for_fit = data
        bkg_med = 0.0

    if fit_xslope:
        k0, k1, chi2, rms, mask_clipped = compute_linear_scaling_coefficient(
            he_template, data_for_fit, variance=variance, mask=mask_good,
            return_mask=True, verbose=verbose
        )
        airglow_ampl, airglow_xslope = k0, k1
    else:
        k0, chi2, rms, mask_clipped = compute_proportional_coefficient(
            he_template, data_for_fit, variance=variance, mask=mask_good,
            return_mask=True, verbose=verbose
        )
        airglow_ampl, airglow_xslope = k0, 0.0

    n_clipped = int(mask_clipped.sum())
    if n_clipped > 0:
        chi2mean = chi2 / n_clipped
    else:
        chi2mean = 0.0

    mask_top = (wmap > wl_top[0]) & (wmap < wl_top[1]) & mask_clipped

    data_tmp = data_for_fit.copy()
    data_tmp[mask_airglow] = data_tmp[mask_airglow] - he_template[mask_airglow] * (
        airglow_ampl + airglow_xslope * (xx[mask_airglow] - NX/2)
    )

    n_top = int(mask_top.sum())
    if n_top > 0:
        rms_top = np.sqrt(np.sum(data_tmp[mask_top]**2) / n_top)
    else:
        rms_top = 0.0

    # Background serialization
    # degree + param_name -> value dict (fitted/free params only)
    bkg = {
        "degree": int(getattr(p_fit, "degree", 2)),
        "params": {name: float(getattr(p_fit, name).value) for name in p_fit.param_names},
    }

    results = {
        'AG_AMPL': airglow_ampl,
        'AG_XSLP': airglow_xslope,
        'AG_CHI2M': chi2mean,
        'AG_RMS': rms_top,
        'AG_BACK': bkg_med,
        'AG_BKG': bkg,
        '_NX': NX,
    }

    return results


def subtract_airglow_he(spximg: SPHERExImage,
                        he_template_img: SPHERExImage,
                        spectral_wcs: HDUList,
                        params_airglow_helium: dict,
                        results: dict,
                        return_model: bool = False) -> np.ndarray:
    """Apply airglow subtraction using pre-fit results.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Input SPHEREx image.
    he_template_img : `SPHERExImage`
        He I Airglow template.
    spectral_wcs : `HDUList`
        Wavelength map array.
    params_airglow_helium : `dict`
        Airglow helium parameters.
    results : `dict`
        Dictionary containing airglow fit results.
    return_model : `bool`
        If True, returns the He I airglow background model without subtracting
        it. If False, returns the airglow-subtracted data. Default is False.

    Returns
    -------
    `np.ndarray`
        If ``return_model`` is False, contains the airglow-subtracted image
        data. If ``return_model`` is True, contains the He I airglow background
        model. The input ``spximg.data`` is unchanged.

    Raises
    ------
    ValueError
        If ``results`` does not contain a reconstructed background model
        (``_p_fit``).
    """
    data = spximg.data
    flag = spximg.flags
    NY, NX = data.shape
    wmap = spectral_wcs["CWAVE"].data
    he_template = he_template_img.data

    # pull what we need
    airglow_ampl = results['AG_AMPL']
    airglow_xslope = results['AG_XSLP']
    p_fit = results.get('_p_fit', None)
    if p_fit is None:
        raise ValueError("results missing reconstructed _p_fit (read from header).")

    # recompute mask (don’t store it)
    mask_airglow, mask_good, mask_bkg, mask_all = create_masks(wmap, flag, **params_airglow_helium)

    xv = np.arange(NX)
    yv = np.arange(NY)
    xx, _ = np.meshgrid(xv, yv)
    yy = wmap

    if return_model:
        he_ag_bkgd = np.zeros_like(data)
        he_ag_bkgd[mask_airglow] = (he_template[mask_airglow]
                                    * (airglow_ampl + airglow_xslope * (xx[mask_airglow] - NX/2)))

        return he_ag_bkgd
    else:
        data_sub = data.copy()

        # subtract background first (same procedure as above)
        data_sub[mask_airglow] -= p_fit(xx[mask_airglow], yy[mask_airglow])

        # subtract fitted template
        data_sub[mask_airglow] -= (he_template[mask_airglow]
                                   * (airglow_ampl + airglow_xslope * (xx[mask_airglow] - NX/2)))

        # add background back
        data_sub[mask_airglow] += p_fit(xx[mask_airglow], yy[mask_airglow])

        return data_sub


def _safe_header_float(x, default=0.0, ndigits=6):
    try:
        x = float(x)
    except Exception:
        return default
    if not np.isfinite(x):
        return default
    return round(x, ndigits)


def attach_airglow_fit_to_header(header: Header, results: dict, prefix: str = "HE_AG_") -> None:
    """Store airglow fit outputs into header.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        SPHEREx image header object.
    results : `dict`
        Dictionary produced by ``fit_airglow_he``. Must contain:
        ``AG_AMPL``, ``AG_XSLP``, ``AG_CHI2M``, ``AG_RMS``, ``AG_BACK``
        ``AG_BKG`` : dict with keys ``"degree"`` (int) and
        ``"params"`` (mapping of parameter name to float value).
    prefix : `str`
        Prefix used for all airglow-related header keywords (default:
        ``"AG_"``).

    Returns
    -------
    `None`
        The header is modified in place.

    Notes
    -----
    This function writes scalar fit diagnostics and a serialized background
    model into ``header`` using standard FITS keywords and ``HIERARCH``
    keywords where needed. The background model is stored in full numeric
    precision using one keyword per fitted coefficient.

    The header entries written are:

      Scalars
        - ``HE_AG_AMPL``   : fitted He template amplitude
        - ``HE_AG_XSLOPE`` : fitted x-slope term
        - ``HE_AG_CHI2R``  : reduced chi-square (mean)
        - ``HE_AG_RMS_PK`` : RMS in the peak
        - ``AG_BKG_MED``   : median fitted background in airglow region

      Background model
        - ``HE_AG_BKG_DEG``     : Polynomial2D degree
        - ``HE_AG_BKG_COEFF``  : comma-separated list of parameter names
        - ``HE_AG_BKG_<param>`` : numeric value for each fitted parameter

    """
    # He-I airglow fitting model
    header[f"HIERARCH {prefix}MODL"] = (
        "[k0 + k1*(X-X0)] * template",
        "He-I airglow fitting model"
    )

    # Best-fit template parameters
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['amplitude']}"] = (
        _safe_header_float(results["AG_AMPL"]),
        "He-I airglow best-fit model amplitude (k0)"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['slope']}"] = (
        _safe_header_float(results["AG_XSLP"]),
        "He-I airglow best-fit model slope (k1)"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['chi2red']}"] = (
        _safe_header_float(results["AG_CHI2M"]),
        "He-I airglow best-fit reduced chi^2"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['rms_peak']}"] = (
        _safe_header_float(results["AG_RMS"]),
        "Peak r.m.s of He-I airglow"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_median']}"] = (
        _safe_header_float(results["AG_BACK"]),
        "Median bkg value for He-I airglow"
    )

    # Background model metadata
    bkg = results["AG_BKG"]
    deg = int(bkg["degree"])
    params = bkg["params"]
    names = sorted(params.keys())

    header[f"HIERARCH {prefix}BKG_MOD"] = (
        "n-order 2d-polynomial",
        "Bkg model for He-I airglow"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_degree']}"] = (
        deg,
        "n-order of bkg poly mod for He-I airglow"
    )
    header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_coeff']}"] = (
        ",".join(names),
        "He-I bkg mod coeff"
    )

    for name in names:
        key = f"HIERARCH {prefix}BKG_{name.upper()}"
        header[key] = (
            _safe_header_float(params[name]),
            f"He-I best-fit bkg model {name}"
        )

    # header.add_history("Airglow fit stored:
    # scalars + per-coefficient numeric keywords (full precision).")


def read_airglow_fit_from_header(header: Header, prefix: str = "HE_AG_") -> dict:
    """Reconstruct the He airglow fit results from metadata.

    Parameters
    ----------
    header : `astropy.io.fits.Header`
        SPHEREx image header object.
    prefix : `str`
        Prefix used for all airglow-related header keywords,
        (default: ``"HE_AG_"``).

    Returns
    -------
    results : `dict`
    AG_AMPL : `float`
        Fitted He template amplitude
    AG_XSLP : `float`
        Fitted x-slope term
    AG_CHI2M : `float`
        Reduced chi-square (mean)
    AG_RMS : `float`
        RMS in the WL_TOP window
    AG_BACK : `float`
        Median fitted background in airglow region
    AG_BKG : `dict`
        Background model serialization with keys:
        "degree" (int) and "params" (dict mapping parameter
        names to float values)

    Notes
    -----
    The returned dictionary is made for passing to
    ``reconstruct_p_fit_from_results`` to rebuild the background model
    (e.g. a ``Polynomial2D``) for subtraction.

    """
    out = {
        "AG_AMPL": float(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['amplitude']}"]),
        "AG_XSLP": float(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['slope']}"]),
        "AG_CHI2M": float(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['chi2red']}"]),
        "AG_RMS": float(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['rms_peak']}"]),
        "AG_BACK": float(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_median']}"]),
    }

    deg = int(header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_degree']}"])
    names = header[f"HIERARCH {prefix}{HE_FIT_BKG_KWS['bkg_coeff']}"].split(",")

    params = {}
    for name in names:
        key = f"HIERARCH {prefix}BKG_{name.upper()}"
        params[name] = float(header[key])

    out["AG_BKG"] = {"degree": deg, "params": params}
    return out


def reconstruct_p_fit_from_results(results: dict) -> Polynomial2D:
    """Reconstruct a background polynomial model from airglow fit results.

    Parameters
    ----------
    results : `dict`
        Dictionary containing an ``"AG_BKG"`` entry with:
          - ``"degree"`` (int): polynomial degree
          - ``"params"`` (dict): mapping of Polynomial2D parameter names
            (e.g. ``"c0_0"``, ``"c1_0"``) to float values.

    Returns
    -------
    p_fit : `Polynomial2D`
        Reconstructed ``Polynomial2D`` background model with parameters set
        to the stored values.

    Notes
    -----
    The returned model is callable and can be used directly as
    ``p_fit(x, y)`` during airglow subtraction.

    """
    bkg = results["AG_BKG"]
    p_fit = Polynomial2D(int(bkg["degree"]))
    for name, val in bkg["params"].items():
        getattr(p_fit, name).value = float(val)
    return p_fit


def template_correlate(data, template, maxnanfraction=0.6, remove_median=True):
    """Compute the correlation coefficient between data and template,
    used for Aurora and Shuttle-glow template with SPHEREx data.

    Parameters
    ----------
    data : `np.ndarray`
        2D array of image values to correlate. May contain NaNs.
    template : `numpy.ndarray`
        Array with the same shape as `data` representing the template pattern.
        Only pixels with `template > 0` are used in the correlation.
    maxnanfraction : `float`, optional
        Maximum allowed fraction of NaN pixels in `data`. If the fraction of
        NaNs in `data` exceeds this value, the function returns 0.0.
        Default is 0.6.

    Returns
    -------
    coeff : `float`
        Normalized correlation coefficient.

    Notes
    -----
    - This is a normalized dot product:
        coeff = (z · t) / (||z|| * ||t||)
      computed over the subset of pixels passing the selection mask.
    - This function computes the normalized dot product between
      the flattened `data` array and the flattened `template` array,
      using only pixels that satisfy:
        1) the data pixel is finite (not NaN), and
        2) the template pixel is strictly positive (template > 0).
    - If too many pixels in `data` are NaN (as defined by `maxnanfraction`),
      the function returns 0.0 to avoid unstable correlations.

    """
    z = data.flatten().astype(float)
    t = template.flatten().astype(float)

    # Bail out before computing nanmedian below: subtracting a scalar
    # doesn't change which elements are NaN, so checking first is
    # equivalent and avoids an all-NaN nanmedian (and its RuntimeWarning)
    # when too much of the data is unusable.
    if np.count_nonzero(np.isnan(z)) > maxnanfraction * z.size:
        return 0.0

    if remove_median:
        z = z - np.nanmedian(z)
        t = t - np.nanmedian(t)

    igood = np.isfinite(z) & np.isfinite(t) & (t > 0.0)
    if not np.any(igood):
        return 0.0

    denom_z = np.sqrt(np.sum(z[igood] * z[igood]))
    denom_t = np.sqrt(np.sum(t[igood] * t[igood]))
    if denom_z == 0.0 or denom_t == 0.0:
        return 0.0

    coeff = np.sum(z[igood] * t[igood]) / denom_z / denom_t
    return float(coeff)


def correlate_image(
    spximg: SPHERExImage,
    template: SPHERExImage | None,
) -> float:
    """Correlate SPHEREx image with templates.

    Parameters
    ----------
    spximg : `SPHERExImage`
        Input SPHEREx image.
    template : `SPHERExImage | None`
        Aurora or Shuttle-glow template.

    Returns
    -------
    `float`
        Correlation coefficient.
    """
    if spximg.zodi is None:
        raise ValueError("spximg.zodi is missing")

    downsamp = 20

    C = SkyCoord(
        spximg.wcs.wcs.crval[0],
        spximg.wcs.wcs.crval[1],
        unit="deg",
        frame="icrs",
    )
    CG = C.transform_to("galactic")
    b = abs(CG.b.deg)

    if template is None:
        return 0.0

    if b < 15.0:
        return np.nan

    dat = spximg.data.copy()

    hasflag = (spximg.flags & (Flag.TRANSIENT | Flag.OVERFLOW | Flag.SUR_ERROR)) > 0
    dat[hasflag] = np.nan

    src_mask = (spximg.flags & Flag.SOURCE) > 0
    dat[src_mask] = np.nan

    nonfunc_mask = (spximg.flags & Flag.NONFUNC) > 0
    dat[nonfunc_mask] = np.nan

    # spximg.zodi already contains the corrected zodi model
    dat -= spximg.zodi.zodi_data

    # If flag masking left no valid pixels at all, avoid an all-NaN
    # nanmedian block-reduce (and its RuntimeWarning) below; the result
    # would be rejected by template_correlate's maxnanfraction check anyway.
    if not np.any(np.isfinite(dat)):
        return 0.0

    data = block_reduce(dat, block_size=downsamp, func=np.nanmedian)

    return template_correlate(data, template.data, remove_median=True)
