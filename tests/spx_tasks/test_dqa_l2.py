# Standard library imports
import unittest
from pathlib import Path

# Related third party imports
import numpy as np
from astropy.io.fits import Header
from astropy.nddata import VarianceUncertainty
from astropy.table import Table
from astropy import units as u
from astropy.units import Unit

# Local application/library specific imports
from spx_data import SPHERExImage, spherex_image_reader, Flag
from spx_tasks import DQAL2Config, DQAL2Task


test_root = Path(__file__).parent.parent


def get_sample_img() -> SPHERExImage:
    sample_file = test_root / "data/small.fits"
    assert sample_file.exists(), f"Sample file {sample_file} does not exist."
    return spherex_image_reader(sample_file)


def get_mock_srccat() -> Table:
    t = Table({
        "ra": [150.0, 151.0] * u.deg,
        "dec": [2.0, 2.5] * u.deg,
        "spherex_class": [1, 2],
        "WISE_W1": [1e5, 2e5],
        "LS_z": [1e5, 1e5],
        "Gaia_DR3_source_id": [1234567890, 9876543210],
        "mag_det1": [10.0, 12.0],
        "detector": [1, 1],
        "ix": [100, 200],
        "iy": [150, 250],
        "dQ": [1000.0, 1100.0],
    })

    # Provide the footer metadata the task expects
    t.meta["user_metadata"] = {
        "schema_version": "1.0",
        "producer": "spherex-pipelines",
        "refcat_run": "refcat_v073",
        "refcat_version": "SPHEREx_RefCat_v0.73",
        "created_utc": "2025-08-27T12:34:56Z",
    }
    return t


def get_mock_srccat_pass(spximg: SPHERExImage, pad: int = 300) -> Table:
    """
    Create a mock srccat whose sky positions cover the image
    with expected number of pad pixels on all sides.
    """
    ny, nx = spximg.data.shape
    w = spximg.wcs

    # pixel-frame coverage points
    px = np.array([
        -pad,            # left
        nx - 1 + pad,    # right
        nx // 2,         # center-x
        nx // 2
    ])
    py = np.array([
        ny // 2,         # center-y
        ny // 2,
        -pad,            # bottom
        ny - 1 + pad     # top
    ])

    # convert to sky coords
    world = w.pixel_to_world(px, py)

    ra = world.ra.degree
    dec = world.dec.degree

    t = Table({
        "ra": ra * u.deg,
        "dec": dec * u.deg,
        "spherex_class": [1, 1, 1, 1],
        "WISE_W1": [1e5, 1e5, 1e5, 1e5],
        "LS_z": [1e5, 1e5, 1e5, 1e5],
        "Gaia_DR3_source_id": [1, 2, 3, 4],
        "mag_det1": [10.0, 11.0, 12.0, 13.0],
        "detector": [1, 1, 1, 1],
        "ix": px,
        "iy": py,
        "dQ": [1000.0, 1000.0, 1000.0, 1000.0]
    })

    t.meta["user_metadata"] = {
        "schema_version": "1.0",
        "producer": "spherex-pipelines",
        "refcat_run": "refcat_v073",
        "refcat_version": "SPHEREx_RefCat_v0.73",
        "created_utc": "2025-08-27T12:34:56Z",
    }

    return t


def check_flag_counts(img_hdr: Header, flg_hdr: Header, keywords: list[str]):
    """Check flag counts are reported in header content.

    Parameters
    ----------
    img_hdr : `astropy.io.fits.Header`
        Image layer header.
    flg_hdr : `astropy.io.fits.Header`
    keywords : `list` of `str`
        List of flag count keywords.
    """
    for kw in keywords:
        assert kw in list(img_hdr.keys()), f"Keyword {kw} missing from image header."
        assert kw in list(flg_hdr.keys()), f"Keyword {kw} missing from flag header."
        ct_msg = f"{kw} in image header ({img_hdr[kw]}) is not equal to {kw} in flag header ({flg_hdr[kw]})."
        assert img_hdr[kw] == flg_hdr[kw], ct_msg


class TestL2DQA(unittest.TestCase):
    """Test L2 DQA Task."""
    # NOTE: The max_transient_pix config is no longer used by this task.
    config = DQAL2Config()
    task = DQAL2Task(config=config)
    spximg = get_sample_img()
    # Ensure required layers exist for DQA

    flag_ct_kw = ["L2 N_TRANSIENT", "L2 N_OVERFLOW", "L2 N_SUR_ERROR",
                  "L2 N_NONFUNC", "L2 N_DICHROIC", "L2 N_MISSING", "L2 N_FULLSAMPLE", "L2 N_PHANMISS"]

    def test_pass_case(self):
        self.spximg.header["ARRAY"] = 1
        self.spximg.header["HIERARCH OMEGA_MEDIAN"] = 37.70
        self.spximg.header["HIERARCH OMEGA_STDDEV"] = 0.23
        self.spximg.header["OBSID"] = "202552222057310"
        self.spximg.header["PSF_FWHM"] = 1.0
        self.spximg.header["HIERARCH SPS_TARGET"] = 518
        self.spximg.header["HIERARCH SPS_FLAGS"] = "survey"
        self.spximg.header["HIERARCH.PSF_OMEGA"] = 47.0
        self.spximg.header["PA_ECL"] = 0.0
        self.spximg.header["corlim"] = 0.5
        self.spximg.header["medres"] = 0.01
        self.spximg.header["n_corr"] = 2040*2040
        self.spximg.header['NL_DQA'] = "PASS"
        self.spximg.unit = Unit("MJy/sr")

        # Add variance array with median sigma within tolerance
        variance = np.full(self.spximg.data.shape, (1e-3)**2)
        self.spximg.uncertainty = VarianceUncertainty(variance)
        self.spximg.uncertainty.unit = Unit("MJy/sr")**2
        # Mock catalog
        srccat = get_mock_srccat_pass(self.spximg)
        # Fake IDQR parameters
        self.spximg.header["SPS_FLAGS"] = "deep_north"
        self.spximg.flags = np.zeros(self.spximg.data.shape, dtype=np.int32)

        output = self.task.run(self.spximg, srccat)
        assert output.spximg_out.header["L2DQAFLG"] == "Pass"
        check_flag_counts(output.spximg_out.header, output.spximg_out.flag_header, self.flag_ct_kw)
        obs = output.idqr_obs.observations["2025W52_2V_0573_1"]
        assert obs.survey_plan_indicator  # check True for no re-observe flag
        assert obs.images[1].status == "good"  # check below transient threshold

        self.assertIn("HIERARCH REFCAT.VERSION", output.spximg_out.header)
        self.assertEqual(output.spximg_out.header["HIERARCH REFCAT.VERSION"], "SPHEREx_RefCat_v0.73")
        # srccat coverage check
        self.assertIn("srccat_coverage_check", output.dqa_l2.model_dump())
        self.assertEqual(output.dqa_l2.srccat_coverage_check["result"],
                         "PASS",
                         "srccat_coverage_check should PASS in test_pass_case")

    def test_fail_case(self):
        self.spximg.header["ARRAY"] = 1
        self.spximg.header["HIERARCH OMEGA_MEDIAN"] = 39.0
        self.spximg.header["HIERARCH OMEGA_STDDEV"] = 0.5
        self.spximg.header["OBSID"] = "202552222057310"
        self.spximg.header["PSF_FWHM"] = 9.0
        self.spximg.header["HIERARCH SPS_TARGET"] = 519
        self.spximg.header["HIERARCH SPS_FLAGS"] = "deep_north,all_sky"
        self.spximg.header["PA_ECL"] = 180.0
        self.spximg.header["HIERARCH.PSF_OMEGA"] = 60.0
        self.spximg.header["FINAST"] = 0  # Ensure we test the >50% fail criteria
        self.spximg.unit = Unit("MJy/sr")
        self.spximg.header["corlim"] = 0.5
        self.spximg.header["medres"] = 0.6
        self.spximg.header["ovrcor"] = 100
        self.spximg.header['maxcor'] = 0.9
        self.spximg.header['medcor'] = 0.6
        self.spximg.header['NL_DQA'] = "FAIL"

        # Add variance array with median sigma outside tolerance
        variance = np.full(self.spximg.data.shape, (5e-3)**2)
        self.spximg.uncertainty = VarianceUncertainty(variance)
        self.spximg.uncertainty.unit = Unit("MJy/sr")**2

        shape = self.spximg.data.shape
        self.spximg.data = np.ones(shape)*np.nan
        self.spximg.uncertainty = VarianceUncertainty(np.full(shape, np.nan))

        # Mock catalog
        srccat = get_mock_srccat()

        # Add enough transient flags to trigger one DQA FAIL
        flags = np.zeros(self.spximg.data.shape, dtype=np.int32)
        flags[0, 0] = flags[2, 2] = Flag.TRANSIENT
        mask_transient = (flags & Flag.TRANSIENT) > 0
        self.spximg.setFlagsFromMask(mask_transient, Flag.TRANSIENT)
        self.spximg.flags = flags

        # Run DQA
        output = self.task.run(self.spximg, srccat)

        # Check for FAIL due to >50% of DQA checks failing (with FINAST == 0)
        assert output.spximg_out.header["L2DQAFLG"] == "Fail"
        check_flag_counts(output.spximg_out.header, output.spximg_out.flag_header, self.flag_ct_kw)

        obs = output.idqr_obs.observations["2025W52_2V_0573_1"]
        # The IDQR generated by this task is no longer affected by the transient count.
        # Its status should be "good" and the plan indicator should be True.
        # The re-observe logic is now handled by the ODIR workflow.
        self.assertTrue(obs.survey_plan_indicator)
        self.assertEqual(obs.images[1].status, "good")

        self.assertIn("HIERARCH REFCAT.VERSION", output.spximg_out.header)
        self.assertEqual(output.spximg_out.header["HIERARCH REFCAT.VERSION"], "SPHEREx_RefCat_v0.73")

        # srccat coverage check
        self.assertIn("srccat_coverage_check", output.dqa_l2.model_dump())
        self.assertEqual(output.dqa_l2.srccat_coverage_check["result"],
                         "FAIL",
                         "srccat_coverage_check should FAIL in test_fail_case")


if __name__ == "__main__":
    unittest.main()
