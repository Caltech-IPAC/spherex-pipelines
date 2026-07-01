# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table
import numpy as np


# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils.psf.input_map import Sources
from spx_utils.psf.superresolution_psf import SuperresolutionPSF
from spx_utils.psf.stack import StackerVirtual
from spx_utils.psf.regrid import psf_grid
from spx_utils.psf.deconvolution import TurboRLD
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
    grid_down = spximg.data
    grid_down[spximg.flags >= 1] = np.nan

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

    starmask = np.zeros(spximg.flags.shape, dtype=np.bool_)
    statmask = np.zeros(spximg.flags.shape, dtype=np.bool_)

    return grid_down, cat_down, abmag, good_mags, starmask, statmask


def test_deconvolution():
    grid_down, cat_down, abmag, good_mags, starmask, statmask = get_inputs()
    s = SuperresolutionPSF(source_coord_down=cat_down, abmag=abmag, good_mags=good_mags)
    stack, _, stack_delete, resnumsource, select_idx, _, _, _, _ = StackerVirtual(grid_down,
                                                                                  cat_down,
                                                                                  abmag,
                                                                                  starmask,
                                                                                  statmask,
                                                                                  s,
                                                                                  False)
    psf_pix, _ = psf_grid(10)
    damper = np.ones((80, 80))*5
    s.FLG_basicRLD = 1
    s.FLG_norm = 2
    s.stackMemLimit = 1
    # s.halfStampSizeUp_tru == s.halfStampSizeUp

    TurboRLD(stack[s.halfStampSizeUp - s.halfStampSizeUp_tru: s.halfStampSizeUp
                   + s.halfStampSizeUp_tru, s.halfStampSizeUp
                   - s.halfStampSizeUp_tru: s.halfStampSizeUp
                   + s.halfStampSizeUp_tru],
             psf_pix, 60, damper, False, weight=0, readout=0)
