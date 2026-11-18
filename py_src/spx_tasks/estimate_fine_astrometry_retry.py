__all__ = ["EstimateFineAstrometryRetryConfig", "EstimateFineAstrometryRetryTask"]

from astropy.table import Table
import lsst.pipe.base as pipeBase
import copy
import logging
from astropy.wcs import Sip, WCS
from lsst.pex.config import Field

from .estimate_fine_astrometry import (
    EstimateFineAstrometryConfig,
    EstimateFineAstrometryTask,
)
from spx_tasks import add_history
from spx_utils.l2bwcs_utils import payload_to_wcs

logger = logging.getLogger(__name__)


class EstimateFineAstrometryRetryConfig(EstimateFineAstrometryConfig):
    """Estimate Fine Astrometry Retry Config"""

    keep_target_crpix = Field(
        doc="If True, preserve the target detector CRPIX while adopting donor WCS terms.",
        dtype=bool,
        default=False,
    )

    keep_target_crval = Field(
        doc="If True, preserve the target pointing CRVAL while adopting donor WCS terms.",
        dtype=bool,
        default=True,
    )

    dichroic_pair_crpix2_offset = Field(
        doc="Detector-frame CRPIX2 offset in pixels between dichroic-paired detectors.",
        dtype=float,
        default=11.0,
    )

    always_use_astrometric_refcat_flag = Field(
        doc=("If True, require SourcePipelineClass.ASTROMETRIC in the retry reference "
            "catalog. If False, allow retry to use preprocess_catalog without the "
            "ASTROMETRIC source flag as an additional search for solution."),
        dtype=bool,
        default=False,
    )


class EstimateFineAstrometryRetryTask(EstimateFineAstrometryTask):
    """Retry fine astrometry using a donor-seeded WCS, then the normal SCAMP path.

    This task intentionally mirrors the regular EstimateFineAstrometryTask SCAMP flow,
    with only two differences:
      1. before running fine astrometry,
      the input WCS is replaced by a donor-seeded WCS;
      2. the astrometry.net solve-field call is omitted.
    """

    ConfigClass = EstimateFineAstrometryRetryConfig
    _DefaultName = "estimate-fine-astrometry-retry"

    @staticmethod
    def _get_retry_pair_crpix2_shift(target_array_id: int, donor_array_id: int, offset_pix: float) -> float:
        valid_pairs = {(1, 4), (4, 1), (2, 5), (5, 2), (3, 6), (6, 3)}
        if (donor_array_id, target_array_id) not in valid_pairs:
            raise ValueError(
                f"Invalid retry donor/target pair: donor={donor_array_id}, target={target_array_id}"
            )

        if donor_array_id in (1, 2, 3) and target_array_id in (4, 5, 6):
            return -float(offset_pix)
        if donor_array_id in (4, 5, 6) and target_array_id in (1, 2, 3):
            return +float(offset_pix)

        raise ValueError(
            f"Could not determine CRPIX2 shift for donor={donor_array_id}, target={target_array_id}"
        )

    def _print_wcs_summary(self, label: str, wcs_obj: WCS) -> None:
        try:
            print(f"\n===== {label} =====")
            print(f"CTYPE = {wcs_obj.wcs.ctype}")
            print(f"CRVAL = {wcs_obj.wcs.crval}")
            print(f"CRPIX = {wcs_obj.wcs.crpix}")
            if wcs_obj.wcs.has_cd():
                print("CD =")
                print(wcs_obj.wcs.cd)
            elif wcs_obj.wcs.has_pc():
                print("PC =")
                print(wcs_obj.wcs.pc)
                print(f"CDELT = {wcs_obj.wcs.cdelt}")
            print(f"SIP present = {wcs_obj.sip is not None}")
        except Exception as e:
            print(f"[WCS DEBUG ERROR] {label}: {e}")

    def _seed_from_donor_wcs(self, spximg, donor_payload: dict, sip: Sip | None):
        donor_wcs = payload_to_wcs(donor_payload)
        target_array_id = int(spximg.getArrayId())
        donor_array_id = int(donor_payload["detector"])

        self._print_wcs_summary("TARGET INPUT WCS BEFORE RETRY SEED", spximg.wcs)
        self._print_wcs_summary("DONOR WCS FROM L2BWCS PAYLOAD", donor_wcs)

        seeded = copy.deepcopy(donor_wcs)
        crpix2_shift = 0.0

        target_wcs = spximg.wcs.deepcopy()

        if donor_array_id == target_array_id:
            retry_mode = "adjacent_small_slew_same_detector"

            shift_x = float(donor_payload.get("retry_small_slew_shift_x", 0.0))
            shift_y = float(donor_payload.get("retry_small_slew_shift_y", 0.0))

            seeded.wcs.crpix[0] += shift_x
            seeded.wcs.crpix[1] += shift_y


        elif (donor_array_id, target_array_id) in {(1, 4), (4, 1), (2, 5), (5, 2), (3, 6), (6, 3)}:
            retry_mode = "same_small_slew_dichroic"

            # only dichroic pair uses target pointing/linear terms
            seeded.wcs.crval = copy.deepcopy(target_wcs.wcs.crval)
            seeded.wcs.crpix = copy.deepcopy(target_wcs.wcs.crpix)

            if target_wcs.wcs.has_pc():
                seeded.wcs.pc = copy.deepcopy(target_wcs.wcs.pc)
                seeded.wcs.cdelt = copy.deepcopy(target_wcs.wcs.cdelt)

            if seeded.wcs.has_cd():
                del seeded.wcs.cd

            crpix2_shift = self._get_retry_pair_crpix2_shift(
                target_array_id=target_array_id,
                donor_array_id=donor_array_id,
                offset_pix=self.config.dichroic_pair_crpix2_offset,
            )
            seeded.wcs.crpix[1] += crpix2_shift

        elif (
            (target_array_id in (1, 2, 3) and donor_array_id in (1, 2, 3))
            or (target_array_id in (4, 5, 6) and donor_array_id in (4, 5, 6))
        ):
            retry_mode = "same_small_slew_same_row"
            if self.config.keep_target_crval:
                seeded.wcs.crval = copy.deepcopy(spximg.wcs.wcs.crval)
            seeded.wcs.crpix = copy.deepcopy(spximg.wcs.wcs.crpix)

        else:
            raise ValueError(
                f"Unsupported retry donor/target combination: donor={donor_array_id}, "
                f"target={target_array_id}"
            )

        # Retry always uses calibrated target-detector SIP input
        if sip is None:
            raise ValueError("fine-astrometry-retry requires input sip dataset, but sip=None")

        seeded.sip = sip

        ctype = list(seeded.wcs.ctype)
        for i in (0, 1):
            if "SIP" not in ctype[i]:
                ctype[i] = ctype[i].replace("TAN", "TAN-SIP")
        seeded.wcs.ctype = ctype

        spximg.wcs = seeded
        slew_dx = float(donor_payload.get("retry_small_slew_shift_x", 0.0))
        slew_dy = float(donor_payload.get("retry_small_slew_shift_y", 0.0))

        # spximg.header["HIERARCH WCSRETRY"] = True
        spximg.header["HIERARCH RETRY_DONOR"] = donor_array_id
        spximg.header["HIERARCH RETRY_DONOROBS"] = int(donor_payload.get("observation", -1))
        spximg.header["HIERARCH RETRY_DONORFIN"] = int(donor_payload.get("finast", -999))
        spximg.header["HIERARCH RETRY_WCSMODE"] = retry_mode
        spximg.header["HIERARCH RETRY_SLEWDX"] = slew_dx
        spximg.header["HIERARCH RETRY_SLEWDY"] = slew_dy
        spximg.header["HIERARCH RETRY_DICHDY"] = float(crpix2_shift)
        spximg.header["HIERARCH ASTRONETSOLVED"] = False
        spximg.header["HIERARCH RETRY_SIPSRC"] = "input_sip"

        print("\n================ RETRY SEED DEBUG ================")
        print(f"target_array_id = {target_array_id}")
        print(f"donor_array_id  = {donor_array_id}")
        print(f"donor obsid     = {donor_payload.get('obsid', 'UNKNOWN')}")
        print(f"donor finast    = {donor_payload.get('finast', 'UNKNOWN')}")
        print(f"crpix2_shift    = {crpix2_shift}")
        print(f"retry_mode      = {retry_mode}")
        self._print_wcs_summary("SEEDED RETRY WCS", seeded)

        return spximg

    def _clean_retry_release_headers(self, spximg):
        """Remove internal retry-processing header keywords before downstream release.

        Keep only the public retry provenance keywords.
        """
        retry_keys_to_remove = [
            "RETRY_WCSMODE",
            "RETRY_DONOR",
            "RETRY_DONOROBS",
            "RETRY_DONORFIN",
            "RETRY_SLEWDX",
            "RETRY_SLEWDY",
            "RETRY_DICHDY",
            "ASTRONETSOLVED",
            "RETRY_SIPSRC",
            "RETRY_RUNASTRO",
            "RETRY_DCRVAL1",
            "RETRY_DCRVAL2",
            "RETRY_NCRVMAT",
            "RETRY_REFCAT"
        ]

        for key in retry_keys_to_remove:
            if key in spximg.header:
                del spximg.header[key]

        return spximg

    @add_history()
    def run(
        self,
        spximg,
        srccat: Table,
        l2bwcs: dict,
        index_dir: str = None,
        sip: Sip | None = None,
        donor_spximg=None,
        max_sep: float = 1.00,
        naxis_1: int = 2040,
        naxis_2: int = 2040,
        verbose: bool = True,
        plot: bool = True,
    ) -> pipeBase.Struct:
        """Run retry fine astrometry:
        seed WCS, then regular SCAMP-only fine astrometry.
        """
        if l2bwcs is not None:
            spximg = self._seed_from_donor_wcs(spximg, l2bwcs, sip)
            retry_mode = spximg.header.get("HIERARCH RETRY_WCSMODE", "")
            spximg.header["HIERARCH RETRY_RUNASTRO"] = (
                retry_mode == "same_small_slew_dichroic"
            )
        else:
            spximg.header["HIERARCH WCSRETRY"] = True
            spximg.header["HIERARCH RETRY_WCSMODE"] = "no_donor_run_astronet"
            spximg.header["HIERARCH RETRY_RUNASTRO"] = True


        input_retry_mode = spximg.header.get("RETRY_WCSMODE", "no_donor_run_astronet")

        result = super().run(
            spximg=spximg,
            srccat=srccat,
            index_dir=index_dir,
            sip=sip,
            donor_spximg=donor_spximg,
            max_sep=max_sep,
            naxis_1=naxis_1,
            naxis_2=naxis_2,
            verbose=verbose,
            plot=plot,
        )

        result.spximg_out.header["HIERARCH RETRY_WCSMODE"] = input_retry_mode
        result.spximg_out = self._clean_retry_release_headers(result.spximg_out)

        return result
