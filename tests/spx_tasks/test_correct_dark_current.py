# Standard library imports
from pathlib import Path
import numpy as np
from astropy.nddata.nduncertainty import VarianceUncertainty

# Related third party imports

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import CorrectDarkCurrentConfig, CorrectDarkCurrentTask

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return SPHERExImage.read(sample_file)


def generate_dark(spx_img: SPHERExImage) -> SPHERExImage:
    """Create dark current image to use for testing.

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


def test_CorrectDarkCurrent():
    """Runs and validates task - correct dark current.

    Notes
    -----
    Test the task results for:
    - ``spx_img`` and ``output`` have same shape.
    - `numpy.subtract` and `ccddata.subtract` have same result.
    - ``output`` does have uncertainty after subtraction.
    - ``output`` and ``spx_img`` have same uncertainty after subtraction.

    """
    # Setup and run the correct dark current task
    config = CorrectDarkCurrentConfig()
    task = CorrectDarkCurrentTask(config=config)

    # Get SPHEREx image and add uncertainty
    spx_img = get_sample_img()
    spx_uncertainty = VarianceUncertainty(0.1 * spx_img.data)
    spx_img.uncertainty = spx_uncertainty

    dc_img = generate_dark(spx_img)
    output = task.run(spx_img, dc_img)

    # Perform same calculation using numpy subtract to validate results
    manual_subtract = np.subtract(spx_img.data, dc_img.data)

    # Test the Output
    assert spx_img.data.shape == output.spximg_out.data.shape, (
           "Data shapes are not the same")
    assert np.allclose(output.spximg_out.data, manual_subtract), (
           "Manual subtraction and CCDData subtraction did not match")
    assert output.spximg_out.uncertainty, "New image doesn't have uncertainty"
    assert np.array_equal(spx_img.uncertainty.array,
           output.spximg_out.uncertainty.array), (
           "Uncertainties are not the same after subtraction.")
