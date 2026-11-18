__all__ = ["get_initial_distortion_model"]

# Standard library imports
import logging

# Related third party imports
from astropy.table import Table
import numpy as np
from astropy.modeling import models, fitting

# Local application/library specific imports


def get_initial_distortion_model(T: Table, poly_order=3):
    """Calculate the optical field distortion derived from pre-launch
       ray trace model (August 2023) of the telescope field distortion.
       This creates an initial pre-launch representation of the
       field distortion of the SPHEREx telescope for use prior to
       computing the distortion from flight data.

    Parameters
    ----------
    T : `~astropy.table.Table`
        Tabular data of ray-tracing for deriving initial distortion model.
    poly_order : `int`, optional
        Order of polynomial for distortion model. Default is 3.

    Notes
    -----
    This function is a replication of Optical_Distortion_Model
    https://github.com/SPHEREx/SPHEREx-Sky-Simulator/blob/main/py/
    SPHEREx_InstrumentSimulator/optical_field_distorter.py
    Also refers to notebook:
    https://github.com/SPHEREx/SPHEREx-Pipeline/blob/master/doc/sip/Initial_SIP.ipynb

    """
    Array_Distortion_Model = {}
    for array in [1, 2, 3, 4, 5, 6]:
        fit_p = fitting.LinearLSQFitter()
        Array_Distortion_Model[array] = {}

        T2 = T[T["Band"] == array]
        # lets shift this to pixel coordinates for each array
        xp = T2["xp"] + 3
        yp = T2["yp"] + 3
        xp_dist = T2["xp_dist"] + 3
        yp_dist = T2["yp_dist"] + 3

        p_init = models.Polynomial2D(poly_order)

        poly_xp = fit_p(p_init, xp.value, yp.value, xp_dist.value)
        poly_yp = fit_p(p_init, xp.value, yp.value, yp_dist.value)

        Array_Distortion_Model[array]["x"] = poly_xp
        Array_Distortion_Model[array]["y"] = poly_yp

        Array_Distortion_Model[array]["xo"] = poly_xp(1023.5, 1023.5) - 1023.5
        Array_Distortion_Model[array]["yo"] = poly_yp(1023.5, 1023.5) - 1023.5

        residual_x = poly_xp(xp.value, yp.value) - xp_dist
        residual_y = poly_yp(xp.value, yp.value) - yp_dist

        res = np.sqrt(residual_x**2 + residual_y**2)
        rms = np.std(res)

        logging.info(
            f"array {array} rms residual distortion after {poly_order}th "
            f"order polynomial fit: {rms:0.3f} pixels"
        )

        # Derive the array *un*-distortion polynomials
        # We will derive them from model-calculated relation between
        # distorted and undistorted pixel coordinates for better inversion.
        xp_grid, yp_grid = np.meshgrid(
            np.linspace(0, 2048, 100), np.linspace(0, 2048, 100)
        )
        xp_grid = xp_grid.ravel()
        yp_grid = yp_grid.ravel()

        xp_grid_dist = poly_xp(xp_grid, yp_grid)
        yp_grid_dist = poly_yp(xp_grid, yp_grid)

        undist_poly_xp = fit_p(p_init, xp_grid_dist, yp_grid_dist, xp_grid)
        undist_poly_yp = fit_p(p_init, xp_grid_dist, yp_grid_dist, yp_grid)

        Array_Distortion_Model[array]["undist-x"] = undist_poly_xp
        Array_Distortion_Model[array]["undist-y"] = undist_poly_yp

    return Array_Distortion_Model
