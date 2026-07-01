# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils.psf.input_map import Sources
from spx_utils.psf.superresolution_psf import SuperresolutionPSF
from spx_tasks.measure_psf import MeasurePSFConfig

test_root = Path(__file__).parent.parent


def get_inputs() -> SPHERExImage:
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

    config = MeasurePSFConfig()
    sources = Sources(config=config, spximg=spximg, cat=cat, det_id=1,
                      x_min=0., x_max=680., y_min=0., y_max=680.)

    cat_down = sources.cat_coords()
    abmag = sources.vmag()
    good_mags = sources.magnitude(abmag=abmag)

    return cat_down, abmag, good_mags


def test_superresolution_psf():
    cat_down, abmag, good_mags = get_inputs()
    super_psf = SuperresolutionPSF(source_coord_down=cat_down, abmag=abmag, good_mags=good_mags)
    assert super_psf.sourceCoordRoundDown_stacker.shape == super_psf.sourceCoordRoundUp_stacker.shape
