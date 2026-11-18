# Standard library imports
from pathlib import Path
import tempfile

# Related third party imports
import numpy as np
import pyarrow as pa
from astropy.table import Table, vstack
from astropy.io import fits
from unittest.mock import patch

# Local application/library specific imports
from spx_data import SPHERExImage

# Local application/library specific imports
from spx_tasks import (
    PerformForcedPhotometryConfig, PerformForcedPhotometryTask,
    CreateAllSkyCatalogConfig, CreateAllSkyCatalogTask,
    CombineCatalogTilesConfig, CombineCatalogTilesTask)
from spx_tests import get_sample_global_bkgd, get_sample_pix_map

test_root = Path(__file__).parent.parent


# Little hack to give a way to pass in generated allsky_cat without filepath
class MockDeferredDatasetHandle:
    def __init__(self, table: Table | pa.Table):
        self.table = table

    def get(self) -> Table | pa.Table:
        return self.table


def prepare_mock_lvf_fits(array_id=1, shape=(2040, 2040)):
    mock_data = np.random.uniform(0.75, 4.6, shape).astype(np.float32)
    tmpdir = tempfile.TemporaryDirectory()
    fits_path = Path(tmpdir.name) / f"20250202_SSDC_CENTER_Band{array_id}.FITS"
    fits.writeto(fits_path, mock_data, overwrite=True)
    return tmpdir, fits_path


def get_sample_file(type: str) -> SPHERExImage:
    if type == "point source":
        sample_file = test_root / "data/photometry_point_sources.fits"  # NOTE: image shape is 15x15
    elif type == "galaxy":
        sample_file = test_root / "data/photometry_galaxy.fits"   # NOTE: image shape is 10x10
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return str(sample_file)


def get_sample_spectral_wcs(data: np.ndarray) -> fits.HDUList:
    primary_hdu = fits.PrimaryHDU()
    cwave_hdu = fits.ImageHDU(data=np.full(data.shape, 0.123), name="CWAVE")
    cband_hdu = fits.ImageHDU(data=np.full(data.shape, 0.001), name="CBAND")
    return fits.HDUList([primary_hdu, cwave_hdu, cband_hdu])


def run_pfp(config, fits_files, runs=2):
    """Run the ``PerformForcedPhotometryTask`` ``runs`` amount of times.

    Aggregate results into one dictionary with an entry for each healpix. All
    source measurements per healpix are combined into its respective healpix
    index.

    Parameters
    ----------
    config : `spx_tasks.PerformForcedPhotometryConfig`
        Configuration for task.
    fits_files : `list`
        List of paths to the fits files to use in PFP.
    runs : `int`, Optional
        The amount of times to generate results from PFP per fits file

    Returns
    -------
    healpixes : `dict`
        Dictionary of healpixes with lists for each source measurements.

    """
    pfp_task = PerformForcedPhotometryTask(config=config)

    healpixes = {}
    # Generate data for the same images twice to simulate having multiple obs
    for _ in range(runs):
        for img in fits_files:
            spx_img = SPHERExImage.read(img)
            bkgd = get_sample_global_bkgd(spx_img.data)
            pix_map = get_sample_pix_map(spx_img.data)
            spectral_wcs = get_sample_spectral_wcs(spx_img.data)
            output = pfp_task.run(spximg=spx_img, srccat=spx_img.catalog, global_bkgd=bkgd,
                                  solid_angle_pixel_map=pix_map, spectral_wcs=spectral_wcs)

            # Want to combine it all together so combining healpixes
            for healpix, table in output.phot_meas.items():
                if healpix not in healpixes.keys():
                    healpixes[healpix] = Table()
                healpixes[healpix] = vstack([healpixes[healpix], table])

    return healpixes


def run_casc(deferred_meas_table_list: list[MockDeferredDatasetHandle]) -> pa.Table:
    """Run the ``Create All Sky Catalog`` task.

    Parameters
    ----------
    deferred_meas_table_list : `list`[`MockDeferredDatasetHandle`]
        List of per-HEALPix photometry measurement tables from multiple images.

    Returns
    -------
    allsky_cat : `pyarrow.Table`
        A table of all-sky catalog measurements for a single HEALpix.

    """
    # No filtering; no pass through columns
    config = CreateAllSkyCatalogConfig(sort_on="lambda", merge_refcat=False, pass_thru_cols=["name"],
                                       filter_spherex_class=0)
    task = CreateAllSkyCatalogTask(config=config)
    output = task.run(deferred_meas_table_list)
    return output.allsky_cat


def test_combine_catalog_tiles():
    # Run Perform Forced Photometry and Create All Sky Catalog for distinct
    # images/HEALPix tiles
    tmpdir, _ = prepare_mock_lvf_fits()
    with patch.dict("os.environ", {"SSDC_CENTER_WAVELENGTH_DIR": tmpdir.name}):
        # Point source sample (15x15 image cutout; 6 sources; HEALPix 19644)
        ps_file = get_sample_file("point source")
        ps_config = PerformForcedPhotometryConfig(hips_order=6, combinedOutput=False, merge_srccat=True,
                                                  pass_thru_cols=["name"], fit_sso=False)
        ps_healpixes = run_pfp(ps_config, [ps_file], runs=2)
        ps_id = next(iter(ps_healpixes))
        ps_allsky = run_casc([MockDeferredDatasetHandle(ps_healpixes[ps_id])])

        # Galaxy sample (10x10 image cutout; 1 source; HEALPix 27259)
        gal_file = get_sample_file("galaxy")
        gal_config = PerformForcedPhotometryConfig(hips_order=6, combinedOutput=False, merge_srccat=True,
                                                   pass_thru_cols=["name"], cut_side_length=2, frame_width=1,
                                                   fit_sso=False)
        gal_healpixes = run_pfp(gal_config, [gal_file], runs=5)
        gal_id = next(iter(gal_healpixes))
        gal_allsky = run_casc([MockDeferredDatasetHandle(gal_healpixes[gal_id])])

    # Run Combine Catalog Tiles
    cct_config = CombineCatalogTilesConfig()
    cct_task = CombineCatalogTilesTask(config=cct_config)
    output = cct_task.run([MockDeferredDatasetHandle(ps_allsky), MockDeferredDatasetHandle(gal_allsky)])
    allsky_cat_combined = output.allsky_cat_combined

    # Check for expected number of sources
    n_rows_msg = "The number of rows in the combined all sky catalog is not equal to the sum of sources"\
                 "in the constituent all sky catalogs."
    assert allsky_cat_combined.num_rows == (ps_allsky.num_rows + gal_allsky.num_rows), n_rows_msg

    # Check number of chunks is 1
    n_chunks_msg = "1 or more columns in the combined all sky catalog have more than 1 chunks."
    assert all(col.num_chunks == 1 for col in allsky_cat_combined.columns), n_chunks_msg
