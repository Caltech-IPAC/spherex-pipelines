# Standard library imports
from pathlib import Path
import astropy.units as u
import numpy as np
import copy

# Related third party imports
from astropy.io import fits
from astropy.units import Unit
from astropy.wcs import WCS
from astropy.nddata import VarianceUncertainty

# Local application/library specific imports
from spx_data import SPHERExImage, Layer, SpectralWCS
from spx_tasks import CreateSpectralImageConfig, CreateSpectralImageTask
from spx_tests import sample_data, get_5x5sample_spectral_wcs

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small_with_cat.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file)
    # Fake variance for this image
    mu, sigma = 0, 0.1  # Fake mean and standard deviation.
    variance = np.random.normal(mu, sigma, size=spximg.shape)
    spximg_unit = Unit(spximg.unit)
    spximg.uncertainty = VarianceUncertainty(variance, unit=spximg_unit**2)
    # Covert from e/s to MJy / sr
    calib = SPHERExImage(np.full(spximg.data.shape, .1),
                         unit="MJy s / (electron sr)")
    # Convert image data unit from e/s to MJy/sr
    spximg.data *= calib.data
    spximg.unit *= calib.unit
    # Convert variance map unit from (e/s)**2 to (MJy/sr)**2
    variance_esp = spximg.uncertainty.array
    spximg.uncertainty = VarianceUncertainty(
        variance_esp * calib.data**2, unit=spximg.uncertainty.unit * calib.unit**2)

    # Add zodi.
    zodi = sample_data.get_sample_zodi(spximg.data.shape)
    spximg.zodi = zodi

    # Add sample catalog.
    cat = sample_data.get_sample_catalog()
    spximg.catalog = cat

    return spximg


def test_CreateSpectralImage():
    spximg_in = get_sample_img()
    (spectral_wcs_data, spectral_coord_vals) = get_5x5sample_spectral_wcs()
    # TODO: Add Layer.PSF after PSF module is implemented
    config1 = CreateSpectralImageConfig(layers=[Layer.CATALOG, Layer.ZODI])
    task1 = CreateSpectralImageTask(config=config1)

    output = task1.run(copy.deepcopy(spximg_in), spectral_wcs=spectral_wcs_data)
    spximg = output.spximg_out

    # > Check basic products
    # Image data
    assert spximg_in.data.shape == spximg.data.shape
    # Variance map
    assert spximg.uncertainty is not None
    # Scientific units
    assert spximg.unit is not None
    # Image has scientific units
    assert output.spximg_out.unit == u.Unit("MJy / sr"), (
        f'Wrong image unit: {output.image.unit = }')
    # Variance map has scientific units
    variance = output.spximg_out.uncertainty
    assert variance.unit == u.Unit("MJy / sr")**2, (
        f'Wrong variance unit: {variance.unit = }')
    # WCS
    assert spximg.wcs is not None
    # Flag bitmap
    assert spximg.flags is not None

    # > Check for presence of layers
    assert spximg.catalog is not None
    # TODO: assert spximg.psf is not None
    assert spximg.zodi is not None

    # spectral WCS
    assert isinstance(spximg.spectral_wcs, SpectralWCS)
    spectral_wcs = spximg.spectral_wcs
    hdul = fits.HDUList([fits.PrimaryHDU(), spectral_wcs.tab])
    hdr = spectral_wcs.hdr
    wcs = WCS(hdr, hdul, key=spectral_wcs.WCS_KEY)
    x, y = np.meshgrid(np.arange(5), np.arange(5))
    coord_vals = np.stack(
        (wcs.pixel_to_world_values(x, y)[0].astype(np.int32),
         wcs.pixel_to_world_values(x, y)[1].astype(np.int32),
         ), axis=-1
    )
    np.array_equal(coord_vals, spectral_coord_vals)

    # > Check for absence of layers
    config2 = CreateSpectralImageConfig(layers=[])
    task2 = CreateSpectralImageTask(config=config2)
    output = task2.run(copy.deepcopy(spximg_in))
    spximg = output.spximg_out
    assert spximg.catalog is None
    # TODO: assert spximg.psf is None
    assert spximg.zodi is None

    # > the presence of the required metadata will be checked in DQA module
