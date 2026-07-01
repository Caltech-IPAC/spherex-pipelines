# Standard library imports
from pathlib import Path

# Related third party imports
from astropy.io import fits
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import UpdateFieldDistortionConfig, UpdateFieldDistortionTask
from spx_tests.mock_dataset import MockDeferredImageHandle, MockDeferredCatalogHandle

test_root = Path(__file__).parent.parent


def get_sim_img() -> SPHERExImage:
    """
    Gets input simulated SPHEREx level 2 image.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.
    """
    sample_file = test_root / "data/distortion_cutout.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    spximg = SPHERExImage.read(sample_file, unit="electron / s")
    spximg.header["ARRAY"] = 2
    return spximg


def get_cat() -> Table:
    """Gets input catalog.

    Returns
    -------
    `SPHERExImage`
        Simulated L2 image includes: data, flags, catalog.
    """
    cat_file = test_root / "data/distortion_cat.fits"
    with fits.open(cat_file) as hdul:
        cat = Table(hdul[1].data)
    return cat


def test_UpdateFieldDistortion(dataset=None):
    config = UpdateFieldDistortionConfig(cutout_hw=3, n_edge_pix=0, min_n_samples=1)
    task = UpdateFieldDistortionTask(config=config)
    # create a dataset with a few differed dataset handlers:
    spximg = get_sim_img()
    cat = get_cat()
    spximg.catalog = cat

    def mockDataId(i: int):
        return {"spsplan": 111, "slewgroup": i, "dichroicpair": 0}

    image_handles = [MockDeferredImageHandle(image=spximg, dataid=mockDataId(i)) for i in range(1)]
    catalog_handles = [MockDeferredCatalogHandle(h) for h in image_handles]
    output = task.run(image_handles, catalog_handles)
    assert output.sip is not None
    # Verify history is added to the SPHERExImage header of solid_angle_pixel_map
    header = output.solid_angle_pixel_map.header
    assert "HISTORY" in header, "No HISTORY entry found in the SPHERExImage header."
    history_lines = output.solid_angle_pixel_map.header.get("HISTORY", None)
    assert history_lines is not None, "HISTORY lines are missing in the SPHERExImage header."
