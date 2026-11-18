# Standard library imports
import pytest

# Related third party imports
import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS

# Local application/library specific imports
from spx_tasks import QueryCustomCatalogTask, QueryCustomCatalogConfig
from spx_butler.defaults.fake_products import fake_selection_mask
from spx_utils.refcat_utils import EmptySourceCatalogError

# Needs SPHEREx_InstrumentSimulator, via TabularWavelengthModel.
pytestmark = pytest.mark.optional_deps


# Helper functions ============================================================
def make_simple_tan_wcs(ra_deg: float, dec_deg: float) -> WCS:
    """Create a small TAN WCS centered at (ra, dec)."""
    w = WCS(naxis=2)
    # center a ~2040x2040 pixel image at CRPIX ~ (1020.5, 1020.5)
    w.wcs.crpix = [1020.5, 1020.5]
    w.wcs.crval = [ra_deg, dec_deg]
    # ~1 arcsec/pixel scale and no rotation
    scale_deg = (1.0 / 3600.0)  # 1 arcsec in degrees
    w.wcs.cd = np.array([[scale_deg, 0.0],
                         [0.0, scale_deg]])
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


class MockDeferredDatasetHandle():
    """Hack for faking DeferredDatasetHandle."""

    def __init__(self, *args, **kwargs):
        pass

    def get(self, **kwargs):
        """Retrieve the dataset pointed to by this handle."""
        # Three near the center (276.5, -4.6) and one far away
        ra = np.array([276.5, 276.51, 276.49, 280.0]) * u.deg
        dec = np.array([-4.6, -4.61, -4.59, -5.0]) * u.deg
        ra_error = [0.001, 0.005, 0.003, 0.004] * u.deg
        dec_error = [0.002, 0.003, 0.001, 0.006] * u.deg
        spx_ref_id = [12, 34, 56, 78]
        PS1_z = [11, 12, -1, 14]
        WISE_W1 = [9, 10, 11, 12]
        WISE_W2 = [10, 11, 12, 13]
        Gaia_RP = [8, 9, 10, 11]
        spherex_class = [0, 1, 129, 1]
        gaia_dr3_source_id = [4295806720, -9999, -9999, -9999]
        legacy_survey_uid = [-9999, -9999, 1419692849758322, -9999]
        ps1_dr1_stackobject_objid = [-9999, -9999, -9999, -9999]
        catwise_source_id = [b"N/A", b"0453p030_b0-060944", b"N/A", b"N/A"]  # byte col in refcat
        allwise_designation = [b"N/A", b"N/A", b"N/A", b"J030028.75+000620.5"]  # byte col in refcat
        twomass_designation = [b"N/A", b"N/A", b"N/A", b"N/A"]  # byte col in refcat
        coord_src = [1, 3, 6, 4]
        ls_z = [-9999, -9999, -9999, -9999]
        shape_r = [0.0, -9999.0, 0.8984056, 1.2922444]
        shape_e1 = [0.0, -9999.0, 0.13974826, 0.47856075]
        shape_e2 = [0.0, -9999.0, 0.10788058, 0.13009189]
        sersic = [0.0, -9999.0, 4.0, 1.0]
        c1 = [1, 2, 3, 4]
        c2 = [11, 22, 33, 44]
        pmra = [0.5, -0.3, 1.2, -1.0] * u.mas/u.year
        pmdec = [-0.8, 0.4, -1.1, 0.9] * u.mas/u.year
        parallax = [10.0, 5.5, 7.8, 3.2] * u.mas
        ref_epoch = [-9999.0, 2016.0, -9999.0, 2016.0] * u.year
        snr_max = [15.0, 20.0, 5.0, 25.0]

        return Table({'ra': ra, 'dec': dec, 'ra_error': ra_error, 'dec_error': dec_error,
                      'SPHERExRefID': spx_ref_id, 'PS1_z': PS1_z, 'WISE_W1': WISE_W1,
                      'WISE_W2': WISE_W2, 'Gaia_RP': Gaia_RP, 'c1': c1, 'c2': c2,
                      'spherex_class': spherex_class, 'Gaia_DR3_source_id': gaia_dr3_source_id, 'LS_z': ls_z,
                      'LegacySurvey_uid': legacy_survey_uid,
                      'PS1_DR1_StackObject_objID': ps1_dr1_stackobject_objid,
                      'CatWISE_source_id': catwise_source_id, 'AllWISE_designation': allwise_designation,
                      '2MASS_designation': twomass_designation, 'coord_src': coord_src, 'pmra': pmra,
                      'pmdec': pmdec, 'parallax': parallax, 'shape_r': shape_r, 'shape_e1': shape_e1,
                      'shape_e2': shape_e2, 'sersic': sersic, 'ref_epoch': ref_epoch, 'SNR_MAX': snr_max})


# Fixtures ===================================================================
@pytest.fixture()
def wcs_list(request):
    # Determine WCS CRVAL based on requested param
    mode = request.param
    if mode == "in_refcat_area":
        # Centered on the mock refcat sources
        return [make_simple_tan_wcs(276.5, -4.6)]
    elif mode == "out_refcat_area":
        # Outside refcat footprint
        return [make_simple_tan_wcs(0.0, 0.0)]


# Test cases ==================================================================
class TestQueryCustomCatalog():
    """Test Query Custom Catalog."""

    # Make a small 'planned_obs'-like row purely to define time bounds
    t0 = Time("2025-01-01T00:00:00", format="isot", scale="utc")
    t1 = t0 + 0.01 * u.day
    start_mjd = t0.mjd
    stop_mjd = t1.mjd
    slewgroup_mjd_avg = 0.5 * (start_mjd + stop_mjd)

    # Set up other inputs
    refcat_allsky = MockDeferredDatasetHandle()
    dichroic_pair = 2
    refcat_run_name = "refcat_v073"  # Fake run/collection name
    cosmo_hdul = fake_selection_mask("cosmology", QueryCustomCatalogConfig().mask_hips_order)
    hi_density_hdul = fake_selection_mask("high_density", QueryCustomCatalogConfig().mask_hips_order)

    @pytest.mark.parametrize("selection_cuts, wcs_list",
                             [(True, "in_refcat_area"), (False, "in_refcat_area")], indirect=["wcs_list"])
    def test_filtering(self, selection_cuts, wcs_list):
        """Test that enabling/disabling selection cuts works as expected."""
        config = QueryCustomCatalogConfig(
            selection_cuts=selection_cuts,
            extra_cols=['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP', 'c2', 'spherex_class',
                        'ref_epoch', 'pmra', 'pmdec', 'parallax'],
            # Use wide magnitude range to avoid any selection cuts
            minmag=-100,
            maxmag=100)
        task = QueryCustomCatalogTask(config=config)
        output = task.run(wcs_list=wcs_list, refcat_allsky=self.refcat_allsky,
                          dichroicpair=self.dichroic_pair, slewgroup_mjd_avg=self.slewgroup_mjd_avg,
                          refcat_run_name=self.refcat_run_name, cosmo_hdul=self.cosmo_hdul,
                          hi_density_hdul=self.hi_density_hdul)

        catalog = output.catalog
        if not selection_cuts:
            # Expect 3/4 sources in bounding box, all of which pass cuts
            assert np.count_nonzero(catalog["do_phot"] == 1) == 3
        else:
            # Expect 3/4 sources in bounding box, but only 2 pass cuts
            assert np.count_nonzero(catalog["do_phot"] == 1) == 2

    @pytest.mark.parametrize("wcs_list, allow_empty",
                             [("out_refcat_area", True), ("out_refcat_area", False)], indirect=["wcs_list"])
    def test_allow_empty(self, wcs_list, allow_empty):
        """Test that reference catalog queries returning no sources are
        handled correctly based on allow_empty config.
        """
        config = QueryCustomCatalogConfig(
            allow_empty=allow_empty,
            extra_cols=['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP', 'c2', 'spherex_class',
                        'ref_epoch', 'pmra', 'pmdec', 'parallax', 'do_phot'])
        task = QueryCustomCatalogTask(config=config)

        if allow_empty:
            # Should complete without error and return None catalog
            output = task.run(wcs_list=wcs_list, refcat_allsky=self.refcat_allsky,
                              dichroicpair=self.dichroic_pair, slewgroup_mjd_avg=self.slewgroup_mjd_avg,
                              refcat_run_name=self.refcat_run_name, cosmo_hdul=self.cosmo_hdul,
                              hi_density_hdul=self.hi_density_hdul)
            assert output.catalog is None
        else:
            with pytest.raises(EmptySourceCatalogError):
                task.run(wcs_list=wcs_list, refcat_allsky=self.refcat_allsky,
                         dichroicpair=self.dichroic_pair, slewgroup_mjd_avg=self.slewgroup_mjd_avg,
                         refcat_run_name=self.refcat_run_name, cosmo_hdul=self.cosmo_hdul,
                         hi_density_hdul=self.hi_density_hdul)
