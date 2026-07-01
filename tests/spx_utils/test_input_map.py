# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils.psf.input_map import Sources
from spx_tasks.measure_psf import MeasurePSFConfig

test_root = Path(__file__).parent.parent


def get_sim_img() -> SPHERExImage:
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

    spximg.catalog = cat
    return spximg


def test_input_map():
    spximg = get_sim_img()
    config = MeasurePSFConfig()
    sources = Sources(config=config, spximg=spximg, cat=spximg.catalog, det_id=1,
                      x_min=0., x_max=680., y_min=0., y_max=680.)
    assert sources.zone_mask_x.shape == sources.starmsk.shape
    assert sources.zone_mask_x.shape == sources.distmsk.shape
    assert sources.zone_mask_x.shape == sources.zone_mask_y.shape

    # Downsampled catalog sources.
    cat = sources.cat_coords()
    assert cat.shape[0] == sources.good_coords.shape[0]
    abmag = sources.vmag()
    assert abmag.shape[0] == sources.good_coords.shape[0]
    good_mags = sources.magnitude(abmag=abmag)
    assert good_mags[0].shape[0] < abmag.shape[0]
