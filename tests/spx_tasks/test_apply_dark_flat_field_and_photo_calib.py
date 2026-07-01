# Standard library imports
from pathlib import Path
import numpy as np

# Related third party imports

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import ApplyDarkFlatFieldAndPhotoCalibConfig
from spx_tasks import ApplyDarkFlatFieldAndPhotoCalibTask
from spx_tests import sample_data
import astropy.units as u

test_root = Path(__file__).parent.parent

# Bulk conversion factor for e/s to MJy/sr. This value is intentionally off.
cf = .1


def generate_dark(spx_img: SPHERExImage) -> SPHERExImage:
    """Creates dark current image to use for testing.

    Parameters
    ----------
    spx_img : SPHERExImage
        Science image to generate dark current size.

    Returns
    -------
    dc_img : SPHERExImage
        The generated dark current image with data and uncertainty.

    Notes
    -----
    Tests and validates the generated data for:
    - ``dc_img`` has uncertainty populated.
    """
    # Generate sample data
    size = spx_img.data.shape
    dc_data = np.random.normal(size=size, loc=1.0, scale=0.1)
    dc_img = SPHERExImage(dc_data, unit="electron / s")

    return dc_img


def test_ApplyDarkFlatFieldAndPhotoCalib():
    config = ApplyDarkFlatFieldAndPhotoCalibConfig()
    task = ApplyDarkFlatFieldAndPhotoCalibTask(config=config)
    spximg = sample_data.get_sample_l2_img()

    # Use a calibration matrix with some variation to avoid sigma=0
    np.random.seed(42)  # Ensure reproducibility
    calib_data = np.full(spximg.data.shape, cf) + np.random.normal(0, 1., spximg.data.shape)
    calib = SPHERExImage(calib_data, unit="MJy s / (electron sr)")
    # Fake dark
    dark = generate_dark(spximg)

    # Run task.
    output = task.run(spximg, calib, dark)
    assert output.spximg_out.uncertainty is not None

    # Shape
    assert output.spximg_out.data.shape == spximg.data.shape, (f"Inconsistent input {spximg.data.shape} and"
                                                               f" output image {output.spximg_out.data.shape}"
                                                               " shapes.")
    # Image values, calib.data = 0.1.
    expected_data = (spximg.data - dark.data) * calib_data  # Apply element-wise multiplication
    assert np.allclose(output.spximg_out.data, expected_data, equal_nan=True), (
        "Wrong image values after calibration applied."
    )
    # Image unit
    assert output.spximg_out.unit == u.Unit("MJy / sr"), (f"Wrong image unit: {output.spximg_out.unit = }")
    # Variance unit
    variance = output.spximg_out.uncertainty
    assert variance.unit == u.Unit("MJy / sr")**2, (f"Wrong variance unit: {variance.unit = }")
    # Variance values, calib.data**2 = 0.01
    expected_variance = spximg.uncertainty.array * calib_data**2  # Correct variance propagation
    assert np.allclose(variance.array, expected_variance, equal_nan=True), (
        "Wrong variance values after propagation."
    )
