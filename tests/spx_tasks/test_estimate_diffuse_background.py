# Standard library imports
import logging
import pytest

# Related third party imports
import numpy as np
from astropy.nddata import VarianceUncertainty
from astropy.modeling.models import Gaussian2D
from astropy.table import QTable
from photutils.datasets import make_model_params, make_model_image, make_noise_image

# Local application/library specific imports
from spx_butler.defaults.fake_products import fake_he_template, fake_spectral_wcs
from spx_data import SPHERExImage, Flag
from spx_tasks import EstimateDiffuseBackgroundConfig, EstimateDiffuseBackgroundTask
from spx_tasks.estimate_diffuse_background import add_he_airglow
from spx_tests import add_gradient, get_sample_spx_img
from spx_utils import fit_airglow_he, attach_airglow_fit_to_header

# Needs SPHEREx_InstrumentSimulator, via TabularWavelengthModel.
pytestmark = pytest.mark.optional_deps


def make_sources(crowded: bool, shape: tuple[int, int]) -> QTable:
    """Create Gaussian objects to simulate sky.

    Parameters
    ----------
    crowded : `bool`
        If True, creates a source table simulating a dense region. If False,
        creates a table with only four sources.
    shape : `tuple` of `int`
        Shape of the image in which the sources will be placed.

    Returns
    -------
    table : `astropy.table.QTable`
        Quantity table containing fake gaussian objects to use in image.

    """
    if crowded:
        table = make_model_params(
            shape, n_sources=4000, seed=42, amplitude=(150, 200), x_mean=(1, shape[1]-1),
            y_mean=(1, shape[0]-1), x_stddev=(1., 2.), y_stddev=(1., 2.), theta=(0., 180.),
            x_name='x_mean', y_name='y_mean')
    else:
        table = make_model_params(
            shape, n_sources=4, seed=42, amplitude=(50, 210), x_mean=(1, shape[1]-2), y_mean=(1, shape[0]-2),
            x_stddev=(3., 15.2), y_stddev=(2.5, 8.1), theta=(0., 145.),
            x_name='x_mean', y_name='y_mean')
    return table


def make_spximg(spec_pass: bool, he_airglow: bool, missing_kws: bool = False) -> SPHERExImage:
    """Create a sample SPHEREx image object.

    Parameters
    ----------
    spec_pass : `bool`
        If True, creates an uncrowded image with only source position pixels
        flagged that passes spectral background estimation. If False, creates a
        crowded image with almost all pixels flagged that fails spectral
        background estimation.
    he_airglow : `bool`
        If True, sets detector to 1. If False, sets detector to 3.
    missing_kws : `bool`
        If True, does not add He I airglow header keywords. If False, adds
        the keywords.

    Returns
    -------
    spximg : `spx_data.SPHERExImage`
        Sample image, including WCS and flags.

    """
    # Set up model for sources
    model = Gaussian2D()

    # Create image with desired source density and noise properties
    if spec_pass:
        # high-density image
        crowded = False
        mean = 10.
        stddev = 5.
    else:
        # low-density image
        crowded = True
        mean = 30.
        stddev = 5.
    source_table = make_sources(crowded=crowded, shape=(2040, 2040))
    img = make_model_image((2040, 2040), model=model, params_table=source_table,
                           x_name='x_mean', y_name='y_mean')
    img = img + make_noise_image(img.shape, distribution="gaussian",
                                 mean=mean, stddev=stddev, seed=42)
    img = add_gradient(img)
    spximg = get_sample_spx_img(data=img, shape=img.shape)
    spximg.uncertainty = VarianceUncertainty(np.ones_like(img))

    # Fake flagging
    if spec_pass:
        # Only flag directly at source locations
        mask = np.full(shape=img.shape, fill_value=False)
        mask[np.round(source_table["y_mean"]).astype(int),
             np.round(source_table["x_mean"]).astype(int)] = True
    else:
        # Flag almost every pixel
        mask = np.full(shape=img.shape, fill_value=True)
        mask[(0, 124, 249), (0, 124, 249)] = False
    spximg.setFlagsFromMask(mask, Flag.SOURCE)

    # Fake He I airglow detector and header keywords
    if he_airglow:
        spximg.header["DETECTOR"] = 1

        if not missing_kws:
            spectral_wcs = fake_spectral_wcs(detector_id=1)
            params_airglow_helium = {
                'WL_AIR': 1.083,            # airglow wavelength
                'HW_AIR': 0.027+0.007,      # airglow half-width (wavelength direction)
                'HW_BKG': 0.035+0.010,      # half-width of background region (in terms of wavelength)
                'WL_TOP': [1.0737, 1.0915]  # Range for the plateau of the profile (90% of the peak)
            }
            fit_params = fit_airglow_he(spximg, 1, fake_he_template(), spectral_wcs, params_airglow_helium,
                                        use_variance=False, fit_xslope=True, verbose=False)
            attach_airglow_fit_to_header(spximg.header, fit_params)
    else:
        spximg.header["DETECTOR"] = 3

    return spximg


class TestEstimateDiffuseBackground():
    """Test Estimate Diffuse Background Task."""

    @pytest.mark.parametrize(
        "spec_pass, he_airglow, missing_kws, max_bkgd, expected_msgs",
        ([True, True, False, 834.,  # spectral + He I airglow
          ["Reading He I airglow fit parameters from image header...",
           "He I airglow component added to global background estimate.",
           "Spectral component added to global background estimate."]],
         [True, False, False, 833.,  # spectral only
          ["Spectral component added to global background estimate."]],
         [False, True, True, 3.,  # He I airglow only
          ["He I airglow fit parameters not found in image header. Calculating from He I template...",
           "He I airglow component added to global background estimate.",
           "Spectral background estimation failed."]],
         [False, False, False, np.nan,  # no global components
          ["Spectral background estimation failed."]])
    )
    def test_edb(self, spec_pass: bool, he_airglow: bool, missing_kws: bool, max_bkgd: float,
                 expected_msgs: list[str], caplog: pytest.LogCaptureFixture):
        """Test that logs and outputs are as expected for different
        combinations of components.
        """
        # Set up input data and task
        spximg = make_spximg(spec_pass=spec_pass, he_airglow=he_airglow,
                             missing_kws=missing_kws)
        spectral_wcs = fake_spectral_wcs(detector_id=spximg.getArrayId())
        he_template = fake_he_template() if he_airglow else None
        config = EstimateDiffuseBackgroundConfig(est_he_airglow=he_airglow)
        task = EstimateDiffuseBackgroundTask(config=config)

        # Run task with logs captured
        with caplog.at_level(logging.INFO, logger=task.log.name):
            output = task.run(spximg, spectral_wcs, he_template)

        # Check max expected background value
        assert np.array_equal(np.round(np.nanmax(output.global_bkgd)), max_bkgd, equal_nan=True), \
            (f"Max background estimate {np.round(np.nanmax(output.global_bkgd))} does not match expected"
             f" {max_bkgd} for spec_pass={spec_pass}, he_airglow={he_airglow} case.")

        # Check expected log messages
        for msg in expected_msgs:
            assert msg in caplog.messages, f"Missing expected log message: {msg!r}"

    @pytest.mark.parametrize(
        "fit_successful, has_components, max_bkgd, expected_msg",
        ([True, True, 100., "He I airglow component added to global background estimate."],  # successful fit
         [False, False, np.nan, "He I airglow fit failed."])  # unsuccessful fit
    )
    def test_add_he_airglow(self, fit_successful: bool, has_components: bool, max_bkgd: float,
                            expected_msg: str, caplog: pytest.LogCaptureFixture):
        """Test that the He I airglow component is added to the background
        estimate when the fit is successful and not added when the fit fails.
        """
        spximg = make_spximg(spec_pass=False, he_airglow=True, missing_kws=False)
        global_bkgd = np.full(spximg.data.shape, np.nan, dtype=spximg.data.dtype)
        logger = logging.getLogger("test_logger")

        # Fake compute function
        def dummy_compute_fn(good_he_fit: bool) -> np.ndarray:
            if good_he_fit:
                return np.full(spximg.data.shape, 100., dtype=spximg.data.dtype)
            else:
                return np.full(spximg.data.shape, np.nan, dtype=spximg.data.dtype)

        with caplog.at_level(logging.INFO, logger=logger.name):
            result = add_he_airglow(global_bkgd, logger, lambda: dummy_compute_fn(good_he_fit=fit_successful))

        # Check expected background value and registry of component addition
        assert result == has_components, \
            (f"Expected add_he_airglow to return {has_components} but got {result} for"
             f" fit_successful={fit_successful} case.")
        assert np.array_equal(np.nanmax(global_bkgd), max_bkgd, equal_nan=True), \
            (f"Max background value {np.nanmax(global_bkgd)} does not match expected {max_bkgd} for"
             f" fit_successful={fit_successful} case.")

        # Check expected log messages
        msgs = caplog.messages
        assert expected_msg in msgs, f"Missing expected log message: {expected_msg!r}"
