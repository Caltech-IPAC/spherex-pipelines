# Standard library imports
import logging
import unittest
from typing import Literal

# Related third party imports
import numpy as np
from astropy.table import QTable
from photutils.datasets import make_gaussian_sources_image, make_noise_image

# Local application/library specific imports
from spx_data import SPHERExImage, Flag
from spx_tasks import (EstimateDiffuseBackgroundConfig,
                       EstimateDiffuseBackgroundTask)
from spx_tests import add_gradient, get_sample_spx_img


def make_sources(crowded: bool) -> QTable:
    """Create Gaussian objects to simulate sky.

    Parameters
    ----------
    crowded : `bool`
        If True, creates a source table simulating a dense region. If False,
        creates a table with only four sources.

    Returns
    -------
    table : `astropy.table.QTable`
        Quantity table containing fake gaussian objects to use in image.
    """
    np.random.seed(42)  # set seed for reproducibility
    if crowded:
        table = QTable()
        table['amplitude'] = np.random.uniform(150, 200, 4000)
        table['x_mean'] = np.random.randint(0, 250, 4000)
        table['y_mean'] = np.random.randint(0, 250, 4000)
        table['x_stddev'] = np.random.uniform(1., 2., 4000)
        table['y_stddev'] = np.random.uniform(1., 2., 4000)
        table['theta'] = np.radians(np.random.uniform(0., 180., 4000))
    else:
        table = QTable()
        table['amplitude'] = [50, 70, 150, 210]
        table['x_mean'] = [160, 25, 150, 90]
        table['y_mean'] = [70, 40, 25, 60]
        table['x_stddev'] = [15.2, 5.1, 3., 8.1]
        table['y_stddev'] = [2.6, 2.5, 3., 4.7]
        table['theta'] = np.radians(np.array([145., 20., 0., 60.]))
    return table


def make_spximg(case: Literal["spec_fail", "diff_fail", "pass_estimates"]) -> SPHERExImage:
    """Create a sample SPHEREx image object.

    Parameters
    ----------
    case : {'spec_fail', 'diff_fail', 'pass_estimates'}
        If 'spec_fail', creates a crowded image with almost all pixels flagged.
        If 'diff_fail', creates a crowded image with most pixels flagged. If
        'pass_estimates', creates an uncrowded image with source position
        pixels flagged.

    Returns
    -------
    spximg : `spx_data.SPHERExImage`
        Sample image, including WCS and flags.
    """
    # Fake required source density
    if (case == "spec_fail") or (case == "diff_fail"):
        # Make high-density image
        source_table = make_sources(crowded=True)
        img = make_gaussian_sources_image((250, 250), source_table)
        img = img + make_noise_image(img.shape, distribution="gaussian",
                                     mean=30., stddev=5.)
        img = add_gradient(img)
        spximg = get_sample_spx_img(data=img, shape=img.shape)
    elif case == "pass_estimates":
        # Make low-density image
        source_table = make_sources(crowded=False)
        img = make_gaussian_sources_image((250, 250), source_table)
        img = img + make_noise_image(img.shape, distribution="gaussian",
                                     mean=10., stddev=5.)
        img = add_gradient(img)
        spximg = get_sample_spx_img(data=img, shape=img.shape)

    # Fake flagging
    if case == "spec_fail":
        # Flag almost every pixel
        mask = np.full(shape=img.shape, fill_value=True)
        mask[(0, 124, 249), (0, 124, 249)] = False
    elif case == "diff_fail":
        # Flag most of the pixels
        mask = np.full(shape=img.shape, fill_value=True)
        mask[list(source_table["y_mean"]), list(source_table["x_mean"])] = False
    elif case == "pass_estimates":
        # Only flag directly at source locations
        mask = np.full(shape=img.shape, fill_value=False)
        mask[list(source_table["y_mean"]), list(source_table["x_mean"])] = True

    spximg.setFlagsFromMask(mask, Flag.SOURCE)
    spximg.header["DETECTOR"] = 1
    return spximg


class TestEstimateDiffuseBackground(unittest.TestCase):
    """Test Subtract Diffuse Background Task."""
    config = EstimateDiffuseBackgroundConfig(box_size=(5, 5), sc_median_thresh=0.1)
    task = EstimateDiffuseBackgroundTask(config=config)
    spec_fail_img = make_spximg(case="spec_fail")
    diff_fail_img = make_spximg(case="diff_fail")
    pass_estimates_img = make_spximg(case="pass_estimates")
    shape_error = "SPHERExImage and global background estimate do not have same shape."
    nan_error = "Output array elements are not all NaN for failing global background estimate case."

    def test_spec_fail_case(self):
        """Test failing spectral component estimation."""
        with self.assertLogs(self.task.log.name, level=logging.INFO) as captured:
            output = self.task.run(self.spec_fail_img)

        # Check log, all NaN values, and shape
        self.assertEqual(captured.records[0].getMessage(), "Spectral background estimation failed; passing "
                         "null global background to photometry.")
        assert np.all(np.isnan(output.global_bkgd)), self.nan_error
        assert self.spec_fail_img.data.shape == output.global_bkgd.shape, self.shape_error

    def test_diff_fail_case(self):
        """Test failing diffuse component estimation."""
        with self.assertLogs(self.task.log.name, level=logging.INFO) as captured:
            output = self.task.run(self.diff_fail_img)

        # Check log, all NaN values, and shape
        self.assertEqual(captured.records[0].getMessage(), "Background2D estimate failed; passing null "
                         "global background to photometry.")
        assert np.all(np.isnan(output.global_bkgd)), self.nan_error
        assert self.diff_fail_img.data.shape == output.global_bkgd.shape, self.shape_error

    def test_med_fail_case(self):
        """Test passing all BG component estimates but exceeding the
        sigma-clipped median threshold."""
        config = EstimateDiffuseBackgroundConfig(box_size=(5, 5), sc_median_thresh=1e-6)
        task = EstimateDiffuseBackgroundTask(config=config)

        with self.assertLogs(task.log.name, level=logging.INFO) as captured:
            output = task.run(self.pass_estimates_img)

        # Check log, all NaN values, and shape
        self.assertEqual(captured.records[0].getMessage(),
                         f"Sigma-clipped median of unmasked pixels in subtracted image >= "
                         f"{config.sc_median_thresh}; passing null global background to photometry.")
        assert np.all(np.isnan(output.global_bkgd)), self.nan_error
        assert self.pass_estimates_img.data.shape == output.global_bkgd.shape, self.shape_error

    def test_global_pass_case(self):
        """Test passing all BG component estimates and remaining below the
        sigma-clipped median threshold."""
        with self.assertLogs(self.task.log.name, level=logging.INFO) as captured:
            output = self.task.run(self.pass_estimates_img)

        # Check log, all finite values, and shape
        self.assertEqual(captured.records[0].getMessage(), "Global background estimation succeeded; passing "
                         "sum of spectral and diffuse components to photometry.")
        nonfinite_error = ("Output array includes non-finite/NaN values for passing global background "
                           "estimate case.")
        assert np.all(np.isfinite(output.global_bkgd)), nonfinite_error
        assert self.pass_estimates_img.data.shape == output.global_bkgd.shape, self.shape_error
