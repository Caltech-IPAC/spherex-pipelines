# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage, PSFZoneData
from spx_tasks import MeasurePSFConfig, MeasurePSFTask
from spx_data.psf import spx_psf_reader


test_root = Path(__file__).parent.parent


def get_sim_img() -> tuple[SPHERExImage, Table]:
    """
    Gets input simulated SPHEREx level 2 image.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.
    """
    sample_file = test_root / "data/simplified_psf.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file, hdu_layers=[])

    cat_file = test_root / "data/simplified_cat.fits"
    with fits.open(cat_file) as hdul_cat:
        cat = Table(hdul_cat[1].data)
        cat.rename_column('ABmag', 'mag_det1')

    return spximg, cat


def get_psf_lib() -> PSFZoneData:
    sample_file = test_root / "data/psf_lib.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    psf_lib = spx_psf_reader(sample_file)
    print(psf_lib.data.shape)
    return psf_lib


def test_MeasurePSF():
    config = MeasurePSFConfig()
    config.run_parallel = False
    config.do_jitter = True
    config.llx = [340.5, 1020.5, 1700.5]
    config.lly = [340.5, 1020.5, 1700.5]
    config.bins = 9
    config.bins_x = 3
    config.bins_y = 3
    config.trim_pixels = 1
    task = MeasurePSFTask(config=config)
    spximg, srccat = get_sim_img()
    psf_avg = get_psf_lib()  # exposure-averaged PSF

    # Run task.
    output = task.run(spximg, srccat, psf_avg)
    psf_per_exp = output.spximg_out.psfcube  # per-exposure PSF
    assert spximg.data.shape == output.spximg_out.data.shape
    assert psf_per_exp is not None
    assert spximg.header["PSF_FWHM"] is not None
