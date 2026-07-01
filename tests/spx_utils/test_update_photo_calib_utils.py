# Standard library imports
import tempfile
import logging

# Related third party imports
import pytest
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

# Local application/library specific imports
from spx_utils.update_photo_calib_utils import CalspecPredict


@pytest.fixture
def calspec():
    """Fixture to create a CalspecPredict instance."""
    return CalspecPredict(Logger=logging.getLogger())


def test_initialize(calspec):
    """Test initialization of CalspecPredict."""
    assert isinstance(calspec, CalspecPredict)
    assert calspec.Names == []
    assert calspec.Coordinates == {}
    assert calspec.Spectra == {}
    assert calspec.interp == {}
    assert calspec.flux_unit == {}
    assert calspec.nsources == 0


def test_load_single_source(calspec):
    """Test loading a single source spectrum from a FITS file."""
    # Create a temporary FITS file with a simple spectrum
    spectrum = Table(data=[[0.5, 1.0, 1.5], [10.0, 20.0, 30.0]],
                     names=["lambda", "FLUX"],
                     dtype=[float, float])
    spectrum["lambda"].unit = u.um
    spectrum["FLUX"].unit = u.mJy

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
        spectrum.write(temp_file.name, overwrite=True)
        temp_path = temp_file.name

    # Load the single source
    calspec.load_single("TestSource", 150.0 * u.deg, -30.0 * u.deg, temp_path)

    # Verify source was added correctly
    assert "TestSource" in calspec.Names
    assert "TestSource" in calspec.Spectra
    assert "TestSource" in calspec.interp
    assert calspec.flux_unit["TestSource"] == u.mJy

    # Check coordinate
    coord = calspec.Coordinates["TestSource"]
    assert isinstance(coord, SkyCoord)
    assert coord.ra.deg == 150.0
    assert coord.dec.deg == -30.0

    # Test interpolation function
    interpolated_flux = calspec.interp["TestSource"](1.25)  # Between 1.0 and 1.5
    expected_flux = 25.0  # Linear interpolation
    assert np.isclose(interpolated_flux, expected_flux, atol=1e-5)


def test_load_catalog(calspec):
    """Test loading a catalog of sources from a FITS file."""
    # Create mock band data
    band_data = Table(data=[["BandA", "BandB", "BandC"], [0.5, 1.0, 1.5]],
                      names=["band", "wl"],
                      dtype=[str, float])

    # Create a mock catalog
    catalog = Table(data=[[1, 2], [150.0, 151.0], [-30.0, -31.0], [10.0, np.nan], [20.0, 40.0]],
                    names=["source_id", "ra", "dec", "BandA", "BandB"],
                    dtype=[int, float, float, float, float])

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as band_file, \
         tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as cat_file:

        band_data.write(band_file.name, overwrite=True)
        catalog.write(cat_file.name, overwrite=True)

        # Load the catalog
        calspec.load_catalog(cat_file.name, band_data_file=band_file.name)

    # Verify two sources were added
    assert len(calspec.Names) == 2
    assert "1" in calspec.Names
    assert "2" in calspec.Names

    # Verify coordinates
    assert calspec.Coordinates["1"].ra.deg == 150.0
    assert calspec.Coordinates["1"].dec.deg == -30.0
    assert calspec.Coordinates["2"].ra.deg == 151.0
    assert calspec.Coordinates["2"].dec.deg == -31.0

    # Check spectrum data
    assert "1" in calspec.Spectra
    assert len(calspec.Spectra["1"]) == 2  # Both BandA and BandB should be included
    assert calspec.Spectra["1"]["lambda"][0] == 0.5
    assert calspec.Spectra["1"]["FLUX"][0] == 10.0 / 1e3  # Converted to mJy

    assert len(calspec.Spectra["2"]) == 2  # Both BandA and BandB are included
    # Verify that BandB has the correct value
    assert calspec.Spectra["2"]["lambda"][1] == 1.0
    assert calspec.Spectra["2"]["FLUX"][1] == 40.0 / 1e3  # Converted to mJy


def test_interpolation_out_of_bounds(calspec):
    """Test interpolation behavior for out-of-bounds wavelengths."""
    spectrum = Table(data=[[0.5, 1.0, 1.5], [10.0, 20.0, 30.0]],
                     names=["lambda", "FLUX"],
                     dtype=[float, float])
    spectrum["lambda"].unit = u.um
    spectrum["FLUX"].unit = u.mJy

    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as temp_file:
        spectrum.write(temp_file.name, overwrite=True)
        temp_path = temp_file.name

    calspec.load_single("TestSource", 150.0 * u.deg, -30.0 * u.deg, temp_path)

    # Test interpolation at an out-of-bounds wavelength
    assert calspec.interp["TestSource"](2.0) == 0.0  # Out of bounds should return fill_value


if __name__ == "__main__":
    pytest.main()
