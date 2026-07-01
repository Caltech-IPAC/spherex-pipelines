import pytest
import numpy as np
from astropy.io import fits
from tempfile import NamedTemporaryFile
from spx_data import NonLinearityParameters


@pytest.fixture
def invalid_fits_fixture():
    # Create a temporary fits file missing Q_nl
    with NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
        fits_filename = temp_file.name
        fits_content = [
            fits.PrimaryHDU(header=fits.Header({"DETECTOR": 1})),
            # fits.ImageHDU with a missing required parameter Q_nl
            fits.ImageHDU(
                data=np.array([[9, 10], [11, 12]], dtype=np.float32),
                header=fits.Header({'EXTNAME': 'Qmax'})),
            fits.ImageHDU(
                data=np.array([[25, 26], [27, 28]], dtype=np.float32), header=fits.Header({'EXTNAME': 'b1'})),
            fits.ImageHDU(
                data=np.array([[29, 30], [31, 32]], dtype=np.float32), header=fits.Header({'EXTNAME': 'b2'})),
            fits.ImageHDU(
                data=np.array([[33, 34], [35, 36]], dtype=np.float32), header=fits.Header({'EXTNAME': 'b3'})),
        ]

        fits.HDUList(fits_content).writeto(fits_filename, overwrite=True)

    yield fits_filename

    # Clean up: Delete the temporary fits file after the test
    import os
    os.remove(fits_filename)


@pytest.fixture
def fits_file_fixture():
    # Create a temporary fits file for testing
    with NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
        fits_filename = temp_file.name
        fits_content = [
            fits.PrimaryHDU(header=fits.Header({"DETECTOR": 1})),
            fits.ImageHDU(data=np.array([[13, 14], [15, 16]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'Q_nl'})),
            fits.ImageHDU(data=np.array([[13, 14], [15, 16]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'Qmax'})),
            fits.ImageHDU(data=np.array([[25, 26], [27, 28]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'b1'})),
            fits.ImageHDU(data=np.array([[29, 30], [31, 32]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'b2'})),
            fits.ImageHDU(data=np.array([[33, 34], [35, 36]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'b3'})),
        ]

        fits.HDUList(fits_content).writeto(fits_filename, overwrite=True)

    yield fits_filename

    # Clean up: Delete the temporary fits file after the test
    import os
    os.remove(fits_filename)


@pytest.fixture
def fits_file_with_holes():
    # Create a temporary fits file for testing
    with NamedTemporaryFile(suffix='.fits', delete=False) as temp_file:
        fits_filename = temp_file.name
        fits_content = [
            fits.PrimaryHDU(header=fits.Header({"DETECTOR": 1})),
            fits.ImageHDU(data=np.array([[1, 2], [3, 4]], dtype=np.float32),
                          header=fits.Header({'EXTNAME': 'm1'})),
            fits.ImageHDU(data=np.array([[13, 14], [15, 16]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'Q_nl'})),
            fits.ImageHDU(data=np.array([[13, 14], [15, 16]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'Qmax'})),
            fits.ImageHDU(data=np.array([[25, 26], [27, 28]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'b1'})),
            # b2 is missing
            fits.ImageHDU(data=np.array([[33, 34], [35, 36]],
                                        dtype=np.float32), header=fits.Header({'EXTNAME': 'b3'})),
        ]

        fits.HDUList(fits_content).writeto(fits_filename, overwrite=True)

    yield fits_filename

    # Clean up: Delete the temporary fits file after the test
    import os
    os.remove(fits_filename)


def test_change_in_hdulist(fits_file_fixture):
    fits_filename = fits_file_fixture

    nonlinearity_params = NonLinearityParameters(fits.open(fits_filename))
    # Changed hdulist and validating error.
    nonlinearity_params.hdulist = nonlinearity_params.hdulist[3:]
    with pytest.raises(ValueError):
        nonlinearity_params.validate()


def test_extraction_of_data_by_parameter(fits_file_fixture):
    fits_filename = fits_file_fixture

    nonlinearity_params = NonLinearityParameters(fits.open(fits_filename))
    # Test extracting data for existing parameter
    extracted_data_m2 = nonlinearity_params.get_extension_data('Q_nl')
    assert extracted_data_m2 is not None
    with pytest.raises(KeyError, match="Extension 'invalid_param' not found."):
        # Test extracting data for non-existing parameter
        nonlinearity_params.get_extension_data('invalid_param')


def test_extraction_of_data_by_parameter_missing_required_param(invalid_fits_fixture):
    fits_filename = invalid_fits_fixture
    with pytest.raises(ValueError, match="Required extensions are missing: {'Q_nl'}"):
        NonLinearityParameters(fits.open(fits_filename))


def test_transient_holes(fits_file_with_holes):
    fits_filename = fits_file_with_holes
    with pytest.raises(ValueError):
        NonLinearityParameters(fits.open(fits_filename))
