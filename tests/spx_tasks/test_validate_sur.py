# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, spherex_image_reader, SUR, Flag, ReadoutParameters
from spx_tasks import ValidateSURConfig, ValidateSURTask


test_root = Path(__file__).parent.parent

np.random.seed(111)
num_readouts = 74
pixel_rows, pixel_cols = 8, 34  # SUR window dimenstions.
image_size = (102, 102)
initial_mean = 100.  # Initial mean signal
mean_increment = 2.  # Increment in mean signal per readout
noise_std_dev = 1.  # Standard deviation of the Gaussian noise

# Simulate 74 successive readouts
readouts = np.zeros((num_readouts, pixel_rows, pixel_cols))
for i in range(num_readouts):
    current_mean = initial_mean + i * mean_increment
    readouts[i, :, :] = np.random.normal(current_mean, noise_std_dev, (pixel_rows, pixel_cols))

sur_sigma = 1.0
# Model full exposure as the mean slope (where unit time is 1s).
full_image = np.random.normal(mean_increment, sur_sigma, image_size)


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


def test_ValidateSUR():
    config = ValidateSURConfig()
    task = ValidateSURTask(config=config)
    # Setup unit test image and raw data.
    flg = np.full(full_image.shape, 0)
    flg[:, :] = Flag.FULLSAMPLE
    spximg = SPHERExImage(data=full_image, flags=flg, unit="adu")
    rp = ReadoutParameters(Mphan=4, Jactive=192, wchannel=64, nchannel=3, jref=0, kref=0, nref=0, Kactive=192)
    hdr = fits.Header(rp.to_metadata())
    hdr["MJD-BEG"] = 60312.80288850694  # Exposure start time in MJD.
    hdr["MJD-END"] = 60312.80420312037  # Exposure stop time in MJD.
    hdr["ROWSTART"] = 5
    hdr["COLSECTN"] = 1
    spximg.meta = hdr
    sur = SUR(data=readouts, row_start=1, col_sectn=1)
    spximg.sur = sur
    # Run task
    output = task.run(spximg)
    # Output diagnostics
    diag_image = output.sur_diag.data
    diag_ratio = output.sur_diag.diagnostic_ratio
    # Diagnostic image L0' should be all close to zero.
    assert 0.0 < np.mean(diag_image) < 1.0
    # Diagnostic ratio should be all close to one.
    assert 0.0 < np.mean(diag_ratio) < 1.0
