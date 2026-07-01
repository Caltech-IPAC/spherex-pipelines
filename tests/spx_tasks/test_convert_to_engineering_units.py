# Standard library imports
import astropy.units as u

# Related third party imports

# Local application/library specific imports
from spx_tasks import ConverttoEngineeringUnitsConfig, ConverttoEngineeringUnitsTask
from spx_tests import sample_data

eps = u.electron / u.s


def test_ConverttoEngineeringUnits():
    config = ConverttoEngineeringUnitsConfig()
    task = ConverttoEngineeringUnitsTask(config=config)
    spximg = sample_data.get_sample_level1_img()
    gain_factors = sample_data.get_sample_gain_factors(spximg)
    output = task.run(spximg, gain_factors)
    assert spximg.data.shape == output.spximg_out.data.shape
    assert spximg.unit == eps
