# Standard library imports
import copy
from pathlib import Path

# Related third party imports
import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.wcs import WCS

# Local application/library specific imports
from spx_data import Flag, ReadoutParameters, SPHERExImage, spherex_image_reader, Layer, SpectralWCS
from spx_tasks import CreateImageDataFilesConfig, CreateImageDataFilesTask
from spx_tests import get_sample_level1_img, get_6x6sample_spectral_wcs
from spx_utils import active_area_range

test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small-l1.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


def test_CreateImageDataFiles():
    spximg_in = get_sample_level1_img()

    config1 = CreateImageDataFilesConfig(layers=[Layer.RAWDATA])
    task1 = CreateImageDataFilesTask(config=config1)
    output = task1.run(copy.deepcopy(spximg_in))
    assert output.spximg_out.sur is not None

    config2 = CreateImageDataFilesConfig(layers=[])
    task2 = CreateImageDataFilesTask(config=config2)
    output = task2.run(copy.deepcopy(spximg_in))
    assert output.spximg_out.sur is None

    # nonfunctional pixel map - the diagonal of active area
    rp = ReadoutParameters.from_metadata(spximg_in.meta)
    nonfunc_arr = np.zeros((rp.Jactive + rp.jref, rp.Kactive + rp.kref), dtype=np.uint8)
    np.fill_diagonal(nonfunc_arr, 1)
    nonfunc_map = SPHERExImage(nonfunc_arr, unit="")

    #  nonfunc pixels
    output = task2.run(copy.deepcopy(spximg_in), nonfunc_map)
    (ix, iy) = np.where(output.spximg_out.flags & Flag.NONFUNC == Flag.NONFUNC)
    # nonfunc flag should be along the diagonal of active pixel area
    assert len(np.unique(ix-iy)) == 1 + rp.jref // 2 * rp.nref
    assert len(ix) == rp.Jactive + rp.jref * rp.nref

    # spectral wcs
    (spectral_wcs_data, actv_spectral_coord_vals) = get_6x6sample_spectral_wcs()
    # create an image with 6x6 active pixels
    rp = ReadoutParameters(nref=1, Mphan=1, nchannel=2, wchannel=5,
                           kref=4, jref=4, Jactive=6, Kactive=6)
    spximg_in = get_sample_level1_img(rp)
    output = task2.run(spximg_in, spectral_wcs=spectral_wcs_data)
    spximg = output.spximg_out

    # check spectral wcs
    assert isinstance(spximg.spectral_wcs, SpectralWCS)
    spectral_wcs = spximg.spectral_wcs
    hdr = spectral_wcs.hdr
    imghdu = fits.ImageHDU(spximg.data)
    spectral_wcs.extendImageHeader(imghdu.header)
    hdul = fits.HDUList([fits.PrimaryHDU(), spectral_wcs.tab])
    wcs = WCS(hdr, hdul, key=spectral_wcs.WCS_KEY)
    (actv_rows, actv_cols) = active_area_range(spximg)

    # verify that active pixel spectral coords match the expected values
    x, y = np.meshgrid(np.arange(actv_cols.start, actv_cols.stop),
                       np.arange(actv_rows.start, actv_rows.stop))
    coord_vals = np.stack(
        (wcs.pixel_to_world_values(x, y)[0].astype(np.int32),
         wcs.pixel_to_world_values(x, y)[1].astype(np.int32),
         ), axis=-1
    )
    np.array_equal(coord_vals, actv_spectral_coord_vals)

    # verify that non-active pixels have value 0
    y, x = np.where(spximg.flags & (Flag.PHANTOM | Flag.REFERENCE) > 0)
    wl_vals = wcs.pixel_to_world_values(x, y)[0].astype(np.int32)
    bp_vals = wcs.pixel_to_world_values(x, y)[1].astype(np.int32)
    assert np.all(wl_vals == 0)
    assert np.all(bp_vals == 0)

    # Image data
    assert spximg_in.data.shape == spximg.data.shape
    # Image data in engineering units (e/s)
    assert output.spximg_out.unit == (u.electron / u.s), (
        f'Wrong image unit: {output.spximg_out.unit = }')
    # WCS
    assert spximg.wcs is not None
    # Flag bitmap
    assert spximg.flags is not None
