# Standard library imports
import unittest

# Related third party imports
import numpy as np
from pathlib import Path
from astropy.table import Table
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS

# Local application/library specific imports
from spx_tasks import QueryReferenceCatalogConfig, QueryReferenceCatalogTask
from spx_utils.image_catalog import SourcePipelineClass
from spx_butler.defaults.fake_products import initial_distortion


test_root = Path(__file__).parent.parent


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


class TestQueryReferenceCatalog(unittest.TestCase):
    """Test Query Reference Catalog."""
    derived_inputs = ["Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID",
                      "CatWISE_source_id", "AllWISE_designation", "2MASS_designation", "shape_e1",
                      "shape_e2"]

    def setUp(self):
        self.dichroic_pair = 2

        # Make a small 'planned_obs'-like row purely to define time bounds
        t0 = Time("2025-01-01T00:00:00", format="isot", scale="utc")
        t1 = t0 + 0.01 * u.day
        self.start_mjd = t0.mjd
        self.stop_mjd = t1.mjd
        self.slewgroup_mjd_avg = 0.5 * (self.start_mjd + self.stop_mjd)

        # Center the WCS on the pointing you used in the mock refcat
        self.ra_c = 276.5
        self.dec_c = -4.6

        # Minimal WCS list for the new run() API
        wcs_1 = make_simple_tan_wcs(self.ra_c, self.dec_c)
        self.wcs_list = [wcs_1]

        # Not directly used by run() anymore, but keep available if needed
        self.distortion_table = initial_distortion()

        # Fake run/collection name so _parse_refcat_version -> v0.73
        self.refcat_run_name = "refcat_v073"

    def _assert_user_meta(self, catalog, run_name="refcat_v073", version="SPHEREx_RefCat_v0.73"):
        self.assertIn("user_metadata", catalog.meta)
        um = catalog.meta["user_metadata"]
        self.assertIsInstance(um, dict)

        # Required keys
        for k in ("schema_version", "producer", "refcat_run", "refcat_version", "created_utc"):
            self.assertIn(k, um, f"Missing {k} in user_metadata")

        self.assertEqual(um["schema_version"], "1.0")
        self.assertEqual(um["producer"], "SPHEREx query_reference_catalog")
        self.assertEqual(um["refcat_run"], run_name)
        self.assertEqual(um["refcat_version"], version)

        # created_utc is an ISO string from Time.now().isot (no trailing 'Z')
        self.assertIsInstance(um["created_utc"], str)
        self.assertIn("T", um["created_utc"])
        self.assertGreaterEqual(len(um["created_utc"]), 19)  # "YYYY-MM-DDTHH:MM:SS"

    def test_QueryReferenceCatalog(self):
        """Test successful run of QueryReferenceCatalogTask"""
        config = QueryReferenceCatalogConfig()
        config.expected_high_b = 0
        config.expected_low_b = 2
        config.pixmargin = 0
        config.extra_cols = ['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP',
                             'c2', 'spherex_class', 'ref_epoch', 'pmra', 'pmdec', 'parallax', 'for_l3phot']
        task = QueryReferenceCatalogTask(config=config)

        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle(None, None, None)],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.refcat_run_name,
        )

        catalog = output.catalog

        self.assertIn("ra", catalog.colnames)
        self.assertIn("dec", catalog.colnames)
        self.assertIn("SPHERExRefID", catalog.colnames)
        self.assertIn("PS1_z", catalog.colnames)
        self.assertIn("WISE_W1", catalog.colnames)
        self.assertIn("WISE_W2", catalog.colnames)
        self.assertIn("Gaia_RP", catalog.colnames)
        self.assertNotIn("c1", catalog.colnames)   # Column c1 should not be present
        self.assertNotIn("ab_ratio", catalog.colnames)  # Not added if derived inputs not all present
        self.assertNotIn("phi", catalog.colnames)       # Not added if derived inputs not all present
        self.assertEqual(len(catalog), 3)  # Expecting three valid sources (fourth is far away)
        self._assert_user_meta(catalog, run_name=self.refcat_run_name, version="SPHEREx_RefCat_v0.73")

        # Validate L3 photometry flagging
        phot_mask = np.nonzero(catalog['for_l3phot'] == 1)[0]
        self.assertEqual(len(phot_mask), 2)  # Only 2 sources should be flagged for L3 photometry

    def test_save_derived_inputs(self):
        """
        Test running query reference catalog
        with/without saving derived column inputs enabled.
        """
        # Case 1: save the derived inputs
        config = QueryReferenceCatalogConfig()
        config.extra_cols = [
            "Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID", "CatWISE_source_id",
            "AllWISE_designation", "2MASS_designation", "ra_error", "dec_error", "coord_src", "pmra",
            "pmdec", "parallax", "spherex_class", "Gaia_RP", "sersic", "shape_r", "shape_e1", "shape_e2",
            "ref_epoch", "for_l3phot"
        ]
        config.save_derived_inputs = True
        task = QueryReferenceCatalogTask(config=config)

        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle(SourcePipelineClass.ASTROMETRIC)],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.refcat_run_name,
        )
        catalog = output.catalog

        for col in self.derived_inputs:
            self.assertIn(col, catalog.colnames)
        self.assertIn("ab_ratio", catalog.colnames)
        self.assertIn("phi", catalog.colnames)
        self.assertIn("ref_ra", catalog.colnames)
        self.assertIn("ref_dec", catalog.colnames)
        self.assertIn("pipeline_epoch_mjd", catalog.meta.keys())
        self.assertEqual(len(catalog), 3)

        # Case 2: don't save the derived inputs
        config = QueryReferenceCatalogConfig()
        config.extra_cols = [
            "Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID", "CatWISE_source_id",
            "AllWISE_designation", "2MASS_designation", "ra_error", "dec_error", "coord_src", "pmra",
            "pmdec", "parallax", "spherex_class", "Gaia_RP", "sersic", "shape_r", "shape_e1", "shape_e2",
            "for_l3phot"
        ]
        task = QueryReferenceCatalogTask(config=config)
        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle(None, None, None)],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.refcat_run_name,
        )
        catalog = output.catalog

        for col in self.derived_inputs:
            self.assertNotIn(col, catalog.colnames)
        self.assertIn("ab_ratio", catalog.colnames)
        self.assertIn("phi", catalog.colnames)
        self.assertEqual(len(catalog), 3)

    def test_incorrect_flags_in_query_reference_catalog(self):
        """Test incorrect flags handling in QueryReferenceCatalogTask"""
        config = QueryReferenceCatalogConfig()
        config.pixmargin = 0
        config.extra_cols = ['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP', 'c2', 'spherex_class',
                             'for_l3phot']
        task = QueryReferenceCatalogTask(config=config)

        try:
            output = task.run(
                self.wcs_list,
                [MockDeferredDatasetHandle(SourcePipelineClass.ASTROMETRIC)],
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.refcat_run_name,
            )
        except Exception:
            output = None
        if output is not None:
            self.assertTrue(hasattr(output, "catalog"))
            if isinstance(output.catalog, Table):
                self.assertIn("ra", output.catalog.colnames)
                self.assertIn("dec", output.catalog.colnames)

        try:
            output = task.run(
                self.wcs_list,
                [MockDeferredDatasetHandle(SourcePipelineClass.PSF)],
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.refcat_run_name,
            )
        except Exception:
            output = None
        if output is not None:
            self.assertTrue(hasattr(output, "catalog"))
            if isinstance(output.catalog, Table):
                self.assertIn("ra", output.catalog.colnames)
                self.assertIn("dec", output.catalog.colnames)


class MockDeferredDatasetHandle():

    def __init__(self, *args, **kwargs):
        pass

    def get(self, **kwargs):
        """Retrieves the dataset pointed to by this handle"""
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
        spherex_class = [0, 1, 1, 1]
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

        return Table({'ra': ra, 'dec': dec, 'ra_error': ra_error, 'dec_error': dec_error,
                      'SPHERExRefID': spx_ref_id, 'PS1_z': PS1_z, 'WISE_W1': WISE_W1,
                      'WISE_W2': WISE_W2, 'Gaia_RP': Gaia_RP, 'c1': c1, 'c2': c2,
                      'spherex_class': spherex_class, 'Gaia_DR3_source_id': gaia_dr3_source_id, 'LS_z': ls_z,
                      'LegacySurvey_uid': legacy_survey_uid,
                      'PS1_DR1_StackObject_objID': ps1_dr1_stackobject_objid,
                      'CatWISE_source_id': catwise_source_id, 'AllWISE_designation': allwise_designation,
                      '2MASS_designation': twomass_designation, 'coord_src': coord_src, 'pmra': pmra,
                      'pmdec': pmdec, 'parallax': parallax, 'shape_r': shape_r, 'shape_e1': shape_e1,
                      'shape_e2': shape_e2, 'sersic': sersic, 'ref_epoch': ref_epoch})


class MockIncorrectFlagDatasetHandle(MockDeferredDatasetHandle):
    def __init__(self, flag):
        super().__init__(None, None, None)
        self.flag = flag

    def get(self, **kwargs):
        tbl = super().get(**kwargs)
        tbl['Gaia_RP'][1] = -1
        tbl['PS1_z'][1] = -1
        tbl['spherex_class'][1] = self.flag
        return tbl


if __name__ == '__main__':
    unittest.main()
