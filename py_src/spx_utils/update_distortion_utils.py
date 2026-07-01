__all__ = ["sip_coefficients",
           "initialize_polynomial_from_sip",
           "calculate_sip_coefficients",
           "update_wcs_with_sip"]

# Standard library imports
import logging

# Related third party imports
from astropy.wcs import WCS
import numpy as np
from astropy.modeling import models, fitting
from astropy.wcs import Sip

# Local application/library specific imports


def sip_coefficients(wcs_init: WCS, poly_order: int):
    """Extract SIP coefficients from initial WCS.

    Parameters
    ----------
    wcs_init : `WCS`
        Initial WCS.
    poly_order : `int`
        Polynomial order.

    Returns
    --------
    a_coeffs, b_coeffs, ap_coeffs, bp_coeffs : `tuple`
        Initial coefficients for SIP fitting procedure.
    """
    # Initialize arrays to hold coefficients, all set to zero
    a_coeffs = np.zeros((poly_order + 1, poly_order + 1))
    b_coeffs = np.zeros((poly_order + 1, poly_order + 1))
    ap_coeffs = np.zeros((poly_order + 1, poly_order + 1))
    bp_coeffs = np.zeros((poly_order + 1, poly_order + 1))

    sip = wcs_init.sip
    if sip:
        # Safely assign coefficients to arrays, handling different polynomial orders
        a_max_order = min(sip.a_order, poly_order)
        b_max_order = min(sip.b_order, poly_order)
        ap_max_order = min(sip.ap_order, poly_order)
        bp_max_order = min(sip.bp_order, poly_order)
        a_coeffs[:a_max_order + 1, :a_max_order + 1] = sip.a[:a_max_order + 1, :a_max_order + 1]
        b_coeffs[:b_max_order + 1, :b_max_order + 1] = sip.b[:b_max_order + 1, :b_max_order + 1]
        ap_coeffs[:ap_max_order + 1, :ap_max_order + 1] = sip.ap[:ap_max_order + 1, :ap_max_order + 1]
        bp_coeffs[:bp_max_order + 1, :bp_max_order + 1] = sip.bp[:bp_max_order + 1, :bp_max_order + 1]

    return a_coeffs, b_coeffs, ap_coeffs, bp_coeffs


def initialize_polynomial_from_sip(wcs_init: WCS, poly_order=3):
    """Initialize a polynomial model from SIP coefficients.

    Parameters
    ----------
    wcs_init : `WCS`
        Initial WCS.
    poly_order : `int`
        Polynomial order.

    Returns
    -------
    p_init_x, p_init_y, p_init_xp, p_init_yp : `tuple`
        Initializing polynomial for SIP fitting procedure.

    """
    a_coeffs, b_coeffs, ap_coeffs, bp_coeffs = sip_coefficients(wcs_init, poly_order)

    p_init_x = models.Polynomial2D(degree=poly_order)
    p_init_y = models.Polynomial2D(degree=poly_order)
    p_init_xp = models.Polynomial2D(degree=poly_order)
    p_init_yp = models.Polynomial2D(degree=poly_order)

    # Assign coefficients to the polynomial models
    for i in range(poly_order + 1):
        for j in range(poly_order + 1):
            setattr(p_init_x, f'c{i}_{j}', a_coeffs[i, j])
            setattr(p_init_y, f'c{i}_{j}', b_coeffs[i, j])
            setattr(p_init_xp, f'c{i}_{j}', ap_coeffs[i, j])
            setattr(p_init_yp, f'c{i}_{j}', bp_coeffs[i, j])

    return p_init_x, p_init_y, p_init_xp, p_init_yp


def calculate_sip_coefficients(cat_xs_nosip: list,
                               cat_ys_nosip: list,
                               det_xs: list,
                               det_ys: list,
                               wcs_init: WCS):
    """Calculate SIP coefficients, initializing from an initial SIP.

    Parameters
    ----------
    cat_xs_no_sip : `list`
        Catalog sources with no SIP, x-position.
    cat_ys_no_sip : `list`
        Catalog sources with no SIP, y-position.
    det_xs : `list`
        Distorted detected sources, x-position.
    det_ys : `list`
        Distorted deteced sources, y-position.
    wcs_init : `WCS`
        Initial WCS.

    Returns
    -------
    poly_x, poly_y, poly_xp, poly_yp : `tuple`
        Fitted polynomial solution.
    """
    # p_init_x, p_init_y, p_init_xp, p_init_yp =
    # initialize_polynomial_from_sip(wcs_init, poly_order=3)
    p_init_x = p_init_y = p_init_xp = p_init_yp = models.Polynomial2D(degree=3)

    # Fit polynomial models
    fit_p = fitting.LinearLSQFitter()

    # Forward transform: predict distorted positions
    poly_x = fit_p(p_init_x, cat_xs_nosip, cat_ys_nosip, det_xs)
    poly_y = fit_p(p_init_y, cat_xs_nosip, cat_ys_nosip, det_ys)

    # Backward transform: takes the distorted positions and
    # estimates where those points would be in undistorted space
    poly_xp = fit_p(p_init_xp, det_xs, det_ys, cat_xs_nosip)
    poly_yp = fit_p(p_init_yp, det_xs, det_ys, cat_ys_nosip)

    # Calculate the residuals and the reduced chi-squared for each fit
    def calculate_reduced_chi_squared(fitted_model, x, y, observed):
        expected = fitted_model(x, y)
        residuals = observed - expected
        chi_squared = np.sum((residuals / np.std(residuals))**2)
        dof = len(x) - len(fitted_model.parameters)
        reduced_chi_squared = chi_squared / dof
        return reduced_chi_squared

    # Calculate chi-squared values
    chi2_x = calculate_reduced_chi_squared(poly_x, cat_xs_nosip, cat_ys_nosip, det_xs)
    chi2_y = calculate_reduced_chi_squared(poly_y, cat_xs_nosip, cat_ys_nosip, det_ys)
    chi2_xp = calculate_reduced_chi_squared(poly_xp, det_xs, det_ys, cat_xs_nosip)
    chi2_yp = calculate_reduced_chi_squared(poly_yp, det_xs, det_ys, cat_ys_nosip)

    return poly_x, poly_y, poly_xp, poly_yp, chi2_x, chi2_y, chi2_xp, chi2_yp


def update_wcs_with_sip(poly_x: models.Polynomial2D,
                        poly_y: models.Polynomial2D,
                        poly_xp: models.Polynomial2D,
                        poly_yp: models.Polynomial2D,
                        wcs: WCS):
    """Updates a given WCS by setting its SIP (Simple Imaging Polynomial)
        coefficients based on provided polynomial models for both the
        forward and inverse transformations.

    Parameters
    ----------
    poly_x : `Polynomial2D`
        The polynomial model for forward transformation in the x-direction.
    poly_y : `Polynomial2D`
        The polynomial model for forward transformation in the y-direction.
    poly_xp : `Polynomial2D`
        The polynomial model for inverse transformation in the x-direction.
    poly_yp : `Polynomial2D`
        The polynomial model for inverse transformation in the y-direction.
    wcs : `WCS`
        The World Coordinate System object that will be updated with the new
        SIP coefficients.

    Returns:
    --------
    `WCS`
        The updated WCS object with new SIP coefficients configured. The SIP
        attributes of the WCS object are modified to include the calculated
        coefficients from the module.

    Notes:
    ------
    - The SIP coefficients are determined for orders up to 3 (configurable by
      the `a_order`, `b_order`, `ap_order`, `bp_order` variables).
    - The coefficients are directly extracted from the attributes of the Polynomial2D
      models and are set for the respective indices in the SIP coefficient matrices.
    - The function handles the creation and configuration of both forward (a, b)
      and inverse (ap, bp) distortion matrices.
    """
    import pprint
    pprint.pprint(wcs.to_header(relax=True))
    a_order = b_order = ap_order = bp_order = 3

    # Initialize the coefficient matrices
    sip_coeff_a = np.zeros((a_order + 1, b_order + 1))
    sip_coeff_b = np.zeros((a_order + 1, b_order + 1))
    sip_coeff_ap = np.zeros((ap_order + 1, bp_order + 1))
    sip_coeff_bp = np.zeros((bp_order + 1, bp_order + 1))

    c1_0 = poly_x.param_names.index('c1_0')  # x
    c2_0 = poly_x.param_names.index('c2_0')  # x^2
    c3_0 = poly_x.param_names.index('c3_0')  # x^3
    c0_1 = poly_x.param_names.index('c0_1')  # y
    c0_2 = poly_x.param_names.index('c0_2')  # y^2
    c0_3 = poly_x.param_names.index('c0_3')  # y^3
    c1_1 = poly_x.param_names.index('c1_1')  # xy
    c1_2 = poly_x.param_names.index('c1_2')  # xy^2
    c2_1 = poly_x.param_names.index('c2_1')  # x^2y

    forward_indices = {
        (0, 2): poly_x.parameters[c0_2],
        (0, 3): poly_x.parameters[c0_3],
        (1, 1): poly_x.parameters[c1_1],
        (1, 2): poly_x.parameters[c1_2],
        (2, 0): poly_x.parameters[c2_0],
        (2, 1): poly_x.parameters[c2_1],
        (3, 0): poly_x.parameters[c3_0],
    }

    backward_indices = {
        (0, 2): poly_y.parameters[c0_2],
        (0, 3): poly_y.parameters[c0_3],
        (1, 1): poly_y.parameters[c1_1],
        (1, 2): poly_y.parameters[c1_2],
        (2, 0): poly_y.parameters[c2_0],
        (2, 1): poly_y.parameters[c2_1],
        (3, 0): poly_y.parameters[c3_0],
    }

    # Fill the forward coefficient matrix
    for index, value in forward_indices.items():
        i, j = index
        sip_coeff_a[i, j] = value

    # Fill the backward coefficient matrix
    for index, value in backward_indices.items():
        i, j = index
        sip_coeff_b[i, j] = value

    # Inverse transformation
    inverse_forward_indices = {
        (0, 1): poly_xp.parameters[c0_1],
        (0, 2): poly_xp.parameters[c0_2],
        (0, 3): poly_xp.parameters[c0_3],
        (1, 0): poly_xp.parameters[c1_0],
        (1, 1): poly_xp.parameters[c1_1],
        (1, 2): poly_xp.parameters[c1_2],
        (2, 0): poly_xp.parameters[c2_0],
        (2, 1): poly_xp.parameters[c2_1],
        (3, 0): poly_xp.parameters[c3_0],
    }

    inverse_backward_indices = {
        (0, 1): poly_yp.parameters[c0_1],
        (0, 2): poly_yp.parameters[c0_2],
        (0, 3): poly_yp.parameters[c0_3],
        (1, 0): poly_yp.parameters[c1_0],
        (1, 1): poly_yp.parameters[c1_1],
        (1, 2): poly_yp.parameters[c1_2],
        (2, 0): poly_yp.parameters[c2_0],
        (2, 1): poly_yp.parameters[c2_1],
        (3, 0): poly_yp.parameters[c3_0],
    }

    # Fill the forward inverse coefficient matrix
    for index, value in inverse_forward_indices.items():
        i, j = index
        sip_coeff_ap[i, j] = value

    # Fill the backward inverse coefficient matrix
    for index, value in inverse_backward_indices.items():
        i, j = index
        sip_coeff_bp[i, j] = value

    # Create a SIP object using these matrices
    sip = Sip(sip_coeff_a, sip_coeff_b, sip_coeff_ap, sip_coeff_bp, wcs.wcs.crpix)

    wcs.sip = sip
    logging.info(wcs.to_header(relax=True))

    return wcs
