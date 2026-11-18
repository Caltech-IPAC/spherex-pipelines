from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
import numpy as np
from scipy.ndimage import gaussian_filter
import unittest
from spx_data import SPHERExImage
import copy
from spx_tasks import EstimateAbsGainMatrixTask, EstimateAbsGainMatrixConfig
from spx_tests import MockDeferredImageHandle
from astropy.io import fits
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

np.random.seed(111)
image_size = (100, 100)
num_images = 100
base_signal_level = 100
signal_variation = 0.1
outlier_variation = 20
num_outlier_images = 7
num_outliers_per_image = int(0.10 * np.prod(image_size))

# Create fake data for 64 rows
num_rows = 64
data = {
    'wl_min': np.random.uniform(0.73, 0.8, num_rows),
    'wl_max': np.random.uniform(0.73, 0.8, num_rows),
    'a_bin': np.random.uniform(0.09, 0.12, num_rows),
    'z_bin': np.random.uniform(0.99, 1.01, num_rows),
    'omega_bin': np.random.uniform(8.9e-10, 9.0e-10, num_rows)
}

# Sort wl_min and wl_max so that wl_min < wl_max for each row
data['wl_min'], data['wl_max'] = (np.minimum(data['wl_min'], data['wl_max']),
                                  np.maximum(data['wl_min'], data['wl_max']))

# Create a sample fake binned calibration data table.
spx_binned_cal = Table(data)


def prepare_mock_lvf_fits(array_id=1, shape=(2040, 2040)):
    mock_data = np.random.uniform(0.75, 4.6, shape).astype(np.float32)
    tmpdir = tempfile.TemporaryDirectory()
    fits_path = Path(tmpdir.name) / f"20250202_SSDC_CENTER_Band{array_id}.FITS"
    fits.writeto(fits_path, mock_data, overwrite=True)
    return tmpdir, fits_path


def mock_spectral_wcs(shape=(100, 100), wl_min=0.75, wl_max=1.0):
    """Create a lightweight mock spectral_wcs HDUList for testing.

    Parameters
    ----------
    shape : tuple
        Shape of the wavelength array (ny, nx).
    wl_min : float
        Minimum wavelength in microns.
    wl_max : float
        Maximum wavelength in microns.

    Returns
    -------
    mock_hdul : MagicMock
        Mock HDUList with CWAVE extension containing wavelength data.

    """
    # Create a simple wavelength gradient across the image
    # Wavelength varies along y-axis (row direction) like real LVF
    ny, nx = shape
    wl_gradient = np.linspace(wl_min, wl_max, ny)
    wl_array = np.tile(wl_gradient[:, np.newaxis], (1, nx)).astype(np.float32)

    # Create mock HDU for CWAVE extension
    mock_cwave_hdu = MagicMock()
    mock_cwave_hdu.data = wl_array

    # Create mock HDUList that supports indexing by extension name
    mock_hdul = MagicMock()
    mock_hdul.__getitem__ = lambda self, key: mock_cwave_hdu if key == "CWAVE" else None

    return mock_hdul


class MockSPHERExImage:
    def __init__(self, data, galactic_latitude, latitude, longitude):
        self.data = data
        self.flags = np.zeros_like(data, dtype=np.uint32)
        self.wcs = self.MockWCS(latitude, longitude)  # Correctly pass latitude and longitude here
        self.header = {
            'ARRAY': 5,
            'LAT_ECL': latitude,
            'LON_ECL': longitude,
            'FINAST': 0,
            'GAL_LAT': galactic_latitude,
            'MJD-BEG': Time("2024-01-01T00:00:00", format="isot", scale="utc").mjd}
        self.exposure_start_time = Time("2024-01-01T00:00:00", format="isot", scale="utc")

    class MockWCS(WCS):
        """Mock WCS for testing with configurable coordinates."""

        def __init__(self, latitude, longitude):
            super().__init__(naxis=2)
            self.latitude = latitude
            self.longitude = longitude
            self.wcs.crpix = [50, 50]  # Reference pixel at the center
            self.wcs.cdelt = np.array([-0.1, 0.1])  # Pixel scale in degrees
            self.wcs.crval = [longitude, latitude]  # Reference coordinates
            self.wcs.ctype = ["RA---TAN", "DEC--TAN"]  # TAN projection
            self.sip = None

        def __copy__(self):
            return MockSPHERExImage.MockWCS(self.latitude, self.longitude)

        def __deepcopy__(self, memo):
            # Add deepcopy handler to support copy.deepcopy(self)
            return MockSPHERExImage.MockWCS(copy.deepcopy(self.latitude, memo),
                                            copy.deepcopy(self.longitude, memo))

        def pixel_to_world(self, x, y):
            world_coords = self.all_pix2world(x, y, 0)
            return SkyCoord(ra=world_coords[0] * u.deg, dec=world_coords[1] * u.deg, frame='icrs')

    def getArrayId(self) -> int:
        return self.header["ARRAY"]

    def get(self):
        return self

    def getFlagsAsMask(self, bad_flags):
        return np.zeros_like(self.data, dtype=bool)

    def getExposureStartTime(self):
        return self.exposure_start_time


def generate_flat_field_solution(image_size=(100, 100), smoothness=1):
    x, y = np.meshgrid(np.linspace(-1, 1, image_size[1]), np.linspace(-1, 1, image_size[0]))
    flat_field = x * y
    flat_field_smooth = gaussian_filter(flat_field, sigma=smoothness)

    flat_field_normalized = ((flat_field_smooth - np.min(flat_field_smooth))
                             / (np.max(flat_field_smooth) - np.min(flat_field_smooth)))
    flat_field_variation = 0.01 * flat_field_normalized
    flat_field_final = 1 + flat_field_variation
    return flat_field_final


def apply_flat_field_correction(image, flat_field):
    corrected_image = image.data / flat_field
    return corrected_image


def generate_images(image_size, base_signal_level, signal_variation, flat_field):
    imgs = []
    for _ in range(num_images):
        image = np.random.normal(base_signal_level, signal_variation, image_size)
        if flat_field is not None:
            imgs.append(image * flat_field)
        else:
            imgs.append(image)

    outlier_images = np.random.choice(range(num_images), num_outlier_images, replace=False)
    for index in outlier_images:
        num_outlier_pixels = 600
        for _ in range(num_outlier_pixels):
            x = np.random.randint(0, 100)
            y = np.random.randint(0, 100)
            imgs[index][x, y] = copy.deepcopy(imgs[index][x, y] * 2)

    return imgs, outlier_images


# Generate images
imgs, outlier_image_indices = generate_images(image_size, base_signal_level,
                                              signal_variation, None)
abs_cal = generate_flat_field_solution(image_size, smoothness=20)

# Mock SPHERExImages.
mock_images = []
for i in range(num_images):
    latitude = np.random.uniform(-90, 90)
    longitude = np.random.uniform(-180, 180)
    mock_images.append(MockSPHERExImage(data=imgs[i] * abs_cal,
                                        galactic_latitude=np.random.uniform(-90, 90),
                                        latitude=latitude,
                                        longitude=longitude))
abs_cal = SPHERExImage(abs_cal, unit=u.dimensionless_unscaled)


class TestEstFF(unittest.TestCase):

    def mockDataId(self, i: int):
        return {"exposure": i}

    def test_estimate_flat_field(self):
        tmpdir, _ = prepare_mock_lvf_fits()
        with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
            image_handles =\
                [MockDeferredImageHandle(image=mock_images[i],
                                         dataid=self.mockDataId(i)) for i in range(len(mock_images))]

            config = EstimateAbsGainMatrixConfig()
            # Use smaller region counts appropriate for 100x100 test images
            config.num_regions = 10        # 100 rows / 10 = ~10 pixels per region
            config.num_regions_fine = 20   # 100 rows / 20 = ~5 pixels per region
            task = EstimateAbsGainMatrixTask(config=config)
            spectral_wcs = mock_spectral_wcs(shape=image_size)
            output = task.run(dataset=image_handles,
                              abs_cal=abs_cal,
                              photo_calib=spx_binned_cal,
                              spectral_wcs=spectral_wcs)
            assert output is not None
            assert output.array_id == 5
            assert output.abs_gain_matrix.shape == output.ff_est.shape

            # Verify history is added to the SPHERExImage header of
            # ff_est and abs_gain_matrix
            assert "HISTORY" in output.ff_est.header, "No HISTORY entry found in the SPHERExImage header."
            history_lines = output.ff_est.header.get("HISTORY", None)
            assert history_lines is not None, "HISTORY lines are missing in the SPHERExImage header."
            header = output.abs_gain_matrix.header
            assert "HISTORY" in header, "No HISTORY entry found in the SPHERExImage header."
            history_lines = output.abs_gain_matrix.header.get("HISTORY", None)
            assert history_lines is not None, "HISTORY lines are missing in the SPHERExImage header."


if __name__ == '__main__':
    unittest.main()
