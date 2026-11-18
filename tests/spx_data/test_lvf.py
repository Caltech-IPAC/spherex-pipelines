from spx_data import get_wavelength_mesh
import tempfile
import numpy as np
import pytest
from astropy.io import fits
import os

# get_wavelength_mesh() builds a TabularWavelengthModel, which needs
# SPHEREx_InstrumentSimulator (spx_data/wavelength_model.py:_get_lvf).
pytestmark = pytest.mark.optional_deps


def test_wavelength_mesh():
    # Create a mock 2D wavelength array
    mock_data = np.random.uniform(0.75, 4.6, (2040, 2040)).astype(np.float32)

    # Write mock FITS file
    with tempfile.TemporaryDirectory() as tmpdir:
        fits_path = os.path.join(tmpdir, "20250202_SSDC_CENTER_Band1.FITS")
        fits.writeto(fits_path, mock_data, overwrite=True)

        # Patch the environment variable temporarily
        os.environ["SSDC_CENTER_WAVELENGTH_DIR"] = tmpdir

        # Run the function
        mesh = get_wavelength_mesh(array_id=1)

        # Validate shape and approximate range
        assert mesh.shape == (2040, 2040)
        assert 0.7 <= mesh.min() <= 0.8  # microns
        assert 1.0 <= mesh.max() <= 1.2  # microns

        mesh_sm = get_wavelength_mesh(array_id=1, npix_x=5, npix_y=5)
        assert np.array_equal(mesh_sm, mesh[:5, :5])
