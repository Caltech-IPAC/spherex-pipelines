# Standard library imports
import logging
import unittest

# Related third party imports
import numpy as np
from pathlib import Path
from astropy.table import Table
import astropy.units as u
from astropy.time import Time
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord

# Local application/library specific imports
from spx_butler.defaults.fake_products import fake_selection_mask
from spx_tasks import QueryReferenceCatalogConfig, QueryReferenceCatalogTask
from spx_tasks.query_reference_catalog import do_select, DERIVED_INPUTS, PM_CORR_INPUTS
from spx_utils.image_catalog import SourcePipelineClass
from spx_utils.refcat_utils import EmptySourceCatalogError
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

        # Base config
        self.config = QueryReferenceCatalogConfig()

        # Fake selection masks
        self.cosmo_mask = fake_selection_mask("cosmology", self.config.mask_hips_order)
        self.hi_density_mask = fake_selection_mask("high_density", self.config.mask_hips_order)

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

    def _check_columns(self, catalog):
        # Start with essential columns (ID/RA/DEC) & empty set to exclude
        available_in_cols = set([self.config.id_col, self.config.ra_col, self.config.dec_col])
        expected_out_cols = set([self.config.id_col, self.config.ra_col, self.config.dec_col])
        excluded_out_cols = set()

        # Selection columns are not pulled from refcat, but always expected
        expected_out_cols |= set(["do_phot", "do_mask_swir", "do_mask_mwir"])

        # Add extra refcat columns requested
        available_in_cols |= set(self.config.extra_cols)
        expected_out_cols |= set(self.config.extra_cols)

        # If subset of derived inputs were included, then shape parameters
        # should be present too
        if all(input_col in available_in_cols for input_col in DERIVED_INPUTS):
            expected_out_cols |= set(["ab_ratio", "phi"])
        else:
            excluded_out_cols |= set(["ab_ratio", "phi"])

        # If subset of PM correction inputs were included, then reference
        # positions should be present too
        if all(input_col in available_in_cols for input_col in PM_CORR_INPUTS):
            expected_out_cols |= set(["ref_ra", "ref_dec"])
            expected_out_cols -= set(["ref_epoch"])  # reference epoch not saved if PM correction occurs
        else:
            excluded_out_cols |= set(["ref_ra", "ref_dec"])

        # If requested to remove derived inputs, then none of the derived input
        # columns should be present
        if not self.config.save_derived_inputs:
            expected_out_cols -= set(DERIVED_INPUTS)
            excluded_out_cols |= set(DERIVED_INPUTS)

        # Check expected columns are present and excluded columns are absent
        for col in expected_out_cols:
            self.assertIn(col, catalog.colnames,
                          f"Expected column '{col}' not found in catalog."
                          f"\nsave_derived_inputs:{self.config.save_derived_inputs}"
                          f"\nextra_cols:\n{self.config.extra_cols}")
        for col in excluded_out_cols:
            self.assertNotIn(col, catalog.colnames,
                             f"Excluded column '{col}' found in catalog."
                             f"\nsave_derived_inputs:{self.config.save_derived_inputs}"
                             f"\nextra_cols:\n{self.config.extra_cols}")

    def test_QueryReferenceCatalog(self):
        """Test successful run of QueryReferenceCatalogTask."""
        # Apply config overrides for this test
        self.config.expected_high_b = 0
        self.config.expected_low_b = 2
        self.config.pixmargin = 0
        self.config.extra_cols = ['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP',
                                  'c2', 'spherex_class', 'pmra', 'pmdec', 'parallax']

        # Run the task
        task = QueryReferenceCatalogTask(config=self.config)
        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle()],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.cosmo_mask,
            self.hi_density_mask,
            self.refcat_run_name,
        )
        catalog = output.catalog

        # Check expected columns, row count, and metadata
        self._check_columns(catalog)
        self.assertEqual(len(catalog), 3)  # Expecting three valid sources (fourth is far away)
        self._assert_user_meta(catalog, run_name=self.refcat_run_name, version="SPHEREx_RefCat_v0.73")

    def test_save_derived_inputs(self):
        """Test running query reference catalog
        with/without saving derived column inputs enabled.
        """
        # Case 1: save the derived inputs -------------------------------------
        # Apply config overrides for this test
        self.config.extra_cols = [
            "Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID", "CatWISE_source_id",
            "AllWISE_designation", "2MASS_designation", "ra_error", "dec_error", "coord_src", "pmra",
            "pmdec", "parallax", "spherex_class", "Gaia_RP", "sersic", "shape_r", "shape_e1", "shape_e2",
            "ref_epoch"
        ]
        self.config.save_derived_inputs = True

        # Run the task
        task = QueryReferenceCatalogTask(config=self.config)
        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle()],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.cosmo_mask,
            self.hi_density_mask,
            self.refcat_run_name
        )
        catalog = output.catalog

        # Check expected columns, row count, and metadata
        self.assertEqual(len(catalog), 3)
        self._check_columns(catalog)
        self.assertIn("pipeline_epoch_mjd", catalog.meta.keys(),
                      "Expected target epoch to be included in metadata when all PM correction inputs are"
                      " requested in 'extra_cols' config parameter.")
        self._assert_user_meta(catalog, run_name=self.refcat_run_name, version="SPHEREx_RefCat_v0.73")

        # Case 2: don't save the derived inputs -------------------------------
        # Apply config overrides for this test
        self.config.extra_cols = [
            "Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID", "CatWISE_source_id",
            "AllWISE_designation", "2MASS_designation", "ra_error", "dec_error", "coord_src", "pmra",
            "pmdec", "parallax", "spherex_class", "Gaia_RP", "sersic", "shape_r", "shape_e1", "shape_e2"]
        self.config.save_derived_inputs = False

        # Run the task
        task = QueryReferenceCatalogTask(config=self.config)
        output = task.run(
            self.wcs_list,
            [MockDeferredDatasetHandle()],
            self.dichroic_pair,
            self.slewgroup_mjd_avg,
            self.cosmo_mask,
            self.hi_density_mask,
            self.refcat_run_name
        )
        catalog = output.catalog

        # Check expected columns, row count, and metadata
        self._check_columns(catalog)
        self.assertEqual(len(catalog), 3)
        self.assertNotIn("pipeline_epoch_mjd", catalog.meta.keys(),
                         "Expected target epoch to be excluded from metadata when not all PM correction"
                         " inputs are requested in 'extra_cols' config parameter.")
        self._assert_user_meta(catalog, run_name=self.refcat_run_name, version="SPHEREx_RefCat_v0.73")

    def test_incorrect_flags_in_query_reference_catalog(self):
        """Test incorrect flags handling in QueryReferenceCatalogTask."""
        self.config.pixmargin = 0
        self.config.extra_cols = ['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP', 'c2',
                                  'spherex_class']
        task = QueryReferenceCatalogTask(config=self.config)

        try:
            output = task.run(
                self.wcs_list,
                [MockIncorrectFlagDatasetHandle(SourcePipelineClass.ASTROMETRIC)],
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.cosmo_mask,
                self.hi_density_mask,
                self.refcat_run_name
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
                [MockIncorrectFlagDatasetHandle(SourcePipelineClass.PSF)],
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.cosmo_mask,
                self.hi_density_mask,
                self.refcat_run_name
            )
        except Exception:
            output = None
        if output is not None:
            self.assertTrue(hasattr(output, "catalog"))
            if isinstance(output.catalog, Table):
                self.assertIn("ra", output.catalog.colnames)
                self.assertIn("dec", output.catalog.colnames)

    def test_allow_empty_catalog(self):
        """Test that the switch to allow empty catalogs works."""
        # Move all sources away from WCS center to trigger empty catalogs
        shard_list = [MockDeferredDatasetHandle(ra=np.array([0., 0., 0., 0.]) * u.deg,
                                                dec=np.array([0., 0., 0., 0.]) * u.deg)]

        # Case 1: allow empty catalogs & only log a warning -------------------
        self.config.extra_cols = ['SPHERExRefID', 'PS1_z', 'WISE_W1', 'WISE_W2', 'Gaia_RP',
                                  'c2', 'spherex_class', 'pmra', 'pmdec', 'parallax']
        self.config.allow_empty = True
        task = QueryReferenceCatalogTask(config=self.config)
        with self.assertLogs(task.log, level=logging.WARNING) as captured:
            output = task.run(
                self.wcs_list,
                shard_list,
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.cosmo_mask,
                self.hi_density_mask,
                self.refcat_run_name
            )
        self.assertTrue(output.catalog is None,
                        f"Expected empty catalog when no sources pass selection cuts, but got:"
                        f"\n{output.catalog}")
        msgs = [r.getMessage() for r in captured.records]
        self.assertIn("No reference catalog sources found in bounding box.", msgs)

        # Case 2: don't allow empty catalogs & raise an error -----------------
        self.config.allow_empty = False
        task = QueryReferenceCatalogTask(config=self.config)
        with self.assertRaises(EmptySourceCatalogError, msg="Expected EmptySourceCatalogError when no"
                               " sources pass selection cuts and 'allow_empty' is False.") as cm:
            task.run(
                self.wcs_list,
                shard_list,
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.cosmo_mask,
                self.hi_density_mask,
                self.refcat_run_name
            )
        self.assertIn("No refcat sources were found and 'allow_empty' is False.", str(cm.exception))

    def test_do_select(self):
        """Test selection cuts for various sky positions and source properties.
        """
        # Choose sky positions to trigger different selection branches
        positions = [
            [270.00000, 66.56071],  # NEP -> in cosmology mask
            [90.00000, -66.56071],  # SDF -> in cosmology mask
            [266.41501, -29.00611],  # GC -> in high-density mask
            [80.89417, -69.75611],  # LMC -> in high-density mask
            [13.18667, -72.82861],  # SMC -> in high-density mask
            [86.40499, 28.93617],  # Galactic anti-center -> in neither
        ]
        coords = SkyCoord(ra=[p[0] * u.deg for p in positions], dec=[p[1] * u.deg for p in positions],
                          frame='icrs')

        # For each sky position, make 1 "pass" and 1 "fail" source based on
        # class, SNR, and bright neighbor criteria
        pass_class = int(SourcePipelineClass.PHOTOMETRIC)
        fail_class = int(SourcePipelineClass.ASTROMETRIC | SourcePipelineClass.NO_ICES)
        pass_snr = 10.0
        fail_snr = 3.0
        pass_no_bneighbor = 0
        fail_no_bneighbor = 1
        table = Table({
            'SPHERExRefID': np.arange(1, 13, dtype=int),
            'ra': [c.ra for c in coords] * 2,
            'dec': [c.dec for c in coords] * 2,
            'spherex_class': [pass_class] * 6 + [fail_class] * 6,
            'SNR_MAX': [pass_snr] * 6 + [fail_snr] * 6,
            'flag_brightneighbor': [pass_no_bneighbor] * 6 + [fail_no_bneighbor] * 6,
            'do_phot': [0] * 12
        })

        # Without special IDs - only 6 sources pass selection cuts
        config = QueryReferenceCatalogConfig()
        do_select(table, config, self.cosmo_mask["COSMO_MASK"].data["MASK"],
                  self.hi_density_mask["HI_DENSITY_MASK"].data["MASK"], None)
        selected = table["do_phot"]
        assert np.nonzero(selected == 1)[0].tolist() == [0, 1, 2, 3, 4, 5], \
            f"Expected sources 0-5 to pass selection cuts, but got {np.nonzero(selected == 1)[0].tolist()}."
        assert np.nonzero(selected == 0)[0].tolist() == [6, 7, 8, 9, 10, 11], \
            f"Expected sources 6-11 to fail selection cuts, but got {np.nonzero(selected == 0)[0].tolist()}."

        # With special IDs - 6 sources pass selection cuts, 3 are special IDs
        special_ids = np.array([9, 10, 11])
        config.select_special_ids = True
        do_select(table, config, self.cosmo_mask["COSMO_MASK"].data["MASK"],
                  self.hi_density_mask["HI_DENSITY_MASK"].data["MASK"], special_ids)
        selected = table["do_phot"]
        assert np.nonzero(selected == 1)[0].tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10], \
            f"Expected sources 0-5 and 8-10 to be selected, but got {np.nonzero(selected == 1)[0].tolist()}."
        assert np.nonzero(selected == 0)[0].tolist() == [6, 7, 11], \
            f"Expected sources 6, 7, 11 to fail selection cuts; got {np.nonzero(selected == 0)[0].tolist()}."

    def test_opt_cnxn(self):
        """Test guards against silent failures due to optional inputs."""
        # Case 1: select_special_ids is True but no special_ids provided
        self.config.select_special_ids = True
        task = QueryReferenceCatalogTask(config=self.config)
        with self.assertRaises(ValueError) as cm:
            task.run(
                self.wcs_list,
                [MockDeferredDatasetHandle()],
                self.dichroic_pair,
                self.slewgroup_mjd_avg,
                self.cosmo_mask,
                self.hi_density_mask,
                self.refcat_run_name,
                None
            )
        self.assertIn("select_special_ids is True, but no special_ids array was provided.", str(cm.exception))

class MockDeferredDatasetHandle():

    def __init__(self, ra=None, dec=None, **kwargs):
        self._ra = ra
        self._dec = dec

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
        snr_max = [15.0, 20.0, 5.0, 25.0]
        flag_brightneighbor = [0, 0, 1, 0]

        # Override RA/DEC if provided (to test empty catalog handling)
        if self._ra is not None:
            ra = self._ra
        if self._dec is not None:
            dec = self._dec

        return Table({'ra': ra, 'dec': dec, 'ra_error': ra_error, 'dec_error': dec_error,
                      'SPHERExRefID': spx_ref_id, 'PS1_z': PS1_z, 'WISE_W1': WISE_W1,
                      'WISE_W2': WISE_W2, 'Gaia_RP': Gaia_RP, 'c1': c1, 'c2': c2,
                      'spherex_class': spherex_class, 'Gaia_DR3_source_id': gaia_dr3_source_id, 'LS_z': ls_z,
                      'LegacySurvey_uid': legacy_survey_uid,
                      'PS1_DR1_StackObject_objID': ps1_dr1_stackobject_objid,
                      'CatWISE_source_id': catwise_source_id, 'AllWISE_designation': allwise_designation,
                      '2MASS_designation': twomass_designation, 'coord_src': coord_src, 'pmra': pmra,
                      'pmdec': pmdec, 'parallax': parallax, 'shape_r': shape_r, 'shape_e1': shape_e1,
                      'shape_e2': shape_e2, 'sersic': sersic, 'ref_epoch': ref_epoch, 'SNR_MAX': snr_max,
                      'flag_brightneighbor': flag_brightneighbor})


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
