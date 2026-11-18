__all__ = ["L2bRetryConfig", "L2bRetryTask"]

from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.wcs import Sip
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from spx_data import PSFZoneData, SPHERExImage
from ._provenance import add_calibration_provenance, get_calibration_inputs
from .remove_reference_pixels import RemoveReferencePixelsTask
from .apply_bad_pixel_mask import ApplyBadPixelMaskTask
from .apply_dark_flat_field_and_photo_calib import ApplyDarkFlatFieldAndPhotoCalibTask
from .estimate_fine_astrometry_retry import EstimateFineAstrometryRetryTask
from .create_source_mask import CreateSourceMaskTask
from .detect_outlier_pixels import DetectOutlierPixelsTask
from .get_zodi_model import GetZodiModelTask
from .measure_psf import MeasurePSFTask
from .create_spectral_image import CreateSpectralImageTask
from .dqa_l1 import check_l1_dqa
from .dqa_l2 import DQAL2Task
from spx_utils.l2bwcs_utils import opposite_detector
from spx_data import ObservationId


def _get_small_slew_shift(target_observation, donor_observation) -> tuple[float, float]:
    """Return CRPIX shift to apply to donor WCS so it seeds the target small slew."""
    target_obs = ObservationId(target_observation)
    donor_obs = ObservationId(donor_observation)

    target_small = int(target_obs.small_slew_counter)
    donor_small = int(donor_obs.small_slew_counter)

    # Measured shift from small slew N to N+1.
    shift_x_per_slew = 13.7 - 17.0
    shift_y_per_slew = 154.9 - 270.0

    delta_slew = target_small - donor_small

    return (
        delta_slew * shift_x_per_slew,
        delta_slew * shift_y_per_slew,
    )


def _get_target_finast(
    l2bwcs_payload: dict,
    target_observation,
    target_detector: int,
    default: int = 999,
) -> int:
    target_obs = ObservationId(target_observation)
    target_pointing_id = int(target_obs.pointing_id)

    solutions = l2bwcs_payload.get(
        "slewgroup_solutions",
        l2bwcs_payload.get("solutions", []),
    )

    for sol in solutions:
        if int(sol.get("detector", -1)) != int(target_detector):
            continue

        cand_obs_like = sol.get("obsid", sol.get("observation"))
        cand_obs = ObservationId(cand_obs_like)

        if int(cand_obs.pointing_id) == target_pointing_id:
            return int(sol.get("finast", default))

    return default


def _detector_row(detector: int) -> tuple[int, ...]:
    if detector in (1, 2, 3):
        return (1, 2, 3)
    if detector in (4, 5, 6):
        return (4, 5, 6)
    raise ValueError(f"Unexpected detector: {detector}")


def _select_same_small_slew_same_row_solution(
    l2bwcs_payload: dict,
    target_observation,
    target_detector: int,
    max_finast: int,
):
    """Choose a FINAST=0 donor from the same small slew and same detector row.

    Preference:
      1. same row as target
      2. exclude target detector itself
      3. same small slew
      4. smallest |detector - target_detector|
    """
    target_obs = ObservationId(target_observation)
    target_small = int(target_obs.small_slew_counter)
    row = set(_detector_row(target_detector))

    candidates = []
    solutions = l2bwcs_payload.get("slewgroup_solutions", l2bwcs_payload.get("solutions", []))
    for sol in solutions:
        det = int(sol.get("detector", -1))
        if det == int(target_detector):
            continue
        if det not in row:
            continue
        # if int(sol.get("finast", -999)) != 0:
        #     continue
        cand_finast = int(sol.get("finast", 999))
        if cand_finast not in (0, 1):
            continue
        if int(sol.get("small_slew_counter", -999)) != target_small:
            continue

        det_delta = abs(det - int(target_detector))
        candidates.append((det_delta, det, sol))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def _select_adjacent_same_detector_solution(
    l2bwcs_payload: dict,
    target_observation,
    target_detector: int,
    max_finast: int,
):
    """Choose same-detector adjacent slew donor, preferring FINAST=0
    then FINAST=1.
    """
    target_obs = ObservationId(target_observation)
    target_small = int(target_obs.small_slew_counter)
    target_pointing_id = int(target_obs.pointing_id)

    candidates = []
    solutions = l2bwcs_payload.get("slewgroup_solutions", l2bwcs_payload.get("solutions", []))

    for sol in solutions:
        if int(sol.get("detector", -1)) != int(target_detector):
            continue

        cand_finast = int(sol.get("finast", 999))

        if cand_finast not in (0, 1):
            continue

        cand_obs_like = sol.get("obsid", sol.get("observation"))
        cand_obs = ObservationId(cand_obs_like)

        if int(cand_obs.pointing_id) == target_pointing_id:
            continue

        cand_small = int(cand_obs.small_slew_counter)
        delta = abs(cand_small - target_small)

        direction_rank = 0 if cand_small < target_small else 1

        shift_x, shift_y = _get_small_slew_shift(
            target_observation=target_observation,
            donor_observation=cand_obs_like,
        )

        sol = dict(sol)
        sol["retry_small_slew_shift_x"] = float(shift_x)
        sol["retry_small_slew_shift_y"] = float(shift_y)

        candidates.append((cand_finast, delta, direction_rank, cand_small, sol))

    if not candidates:
        return None
    # donor preference best FINAST first, then nearest small slew
    # This is the preference:
    # 1. adjacent same-detector FINAST=0
    # 2. adjacent same-detector FINAST=1
    # 3. nearest slew within each FINAST group
    # 4. previous slew before future slew
    # 5. dichroic pair only if no adjacent FINAST 0/1 donor exists
    candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return candidates[0][4]


def _select_same_small_slew_dichroic_solution(
    l2bwcs_payload: dict,
    target_observation: int,
    target_detector: int,
    max_finast: int,
):
    """Choose a FINAST <= target FINAST donor from the opposite detector
    in the same small slew.
    """
    target_obs = ObservationId(target_observation)
    target_small = int(target_obs.small_slew_counter)
    donor_det = opposite_detector(target_detector)

    candidates = []
    solutions = l2bwcs_payload.get("slewgroup_solutions", l2bwcs_payload.get("solutions", []))
    for sol in solutions:
        if int(sol.get("detector", -1)) != int(donor_det):
            continue
        cand_finast = int(sol.get("finast", 999))
        if cand_finast not in (0, 1):
            continue
        if int(sol.get("small_slew_counter", -999)) != target_small:
            continue

        candidates.append(sol)

    if not candidates:
        return None

    # usually there should be at most one, but return the first if multiple exist
    return candidates[0]


class L2bRetryConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("instrument", "exposure"),
    defaultTemplates={},
):
    spximg = cT.Input(
        name="level2a",
        doc="Input level2a image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    l2bwcs = cT.Input(
        name="l2bwcs",
        doc="Successful donor WCS exported from prior L2b pass.",
        storageClass="StructuredDataDict",
        dimensions=("instrument", "observation"),
    )
    abs_gain_matrix = cT.PrerequisiteInput(
        name="abs_gain_matrix",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    dark = cT.PrerequisiteInput(
        name="dark",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    sip = cT.PrerequisiteInput(
        name="sip",
        doc="Input distortion as SIP.",
        storageClass="AstropySIP",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    ssocat = cT.Input(
        name="ssocat",
        doc="Input solar system object catalog.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "observation"],
        minimum=0,
    )
    psf_avg = cT.PrerequisiteInput(
        name="average_psf",
        doc="Input psf library.",
        storageClass="PSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0
    )
    epsf = cT.PrerequisiteInput(
        name="epsf",
        doc="Input EPSF product used as template for EPSF binary-table output.",
        storageClass="EPSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral WCS with lookup table.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    he_template = cT.PrerequisiteInput(
        name="he_template",
        doc="He I Airglow template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    aurora_template = cT.PrerequisiteInput(
        name="aurora_template",
        doc="Aurora template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    shuttle_glow_template = cT.PrerequisiteInput(
        name="shuttle_glow_template",
        doc="Shuttle-glow template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    zodi_scaling_v2 = cT.PrerequisiteInput(
        name="zodi_scaling_v2",
        doc="Detector-dependent 1D zodi scaling table in ECSV format.",
        storageClass="AstropyTable",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    xtalk_class1 = cT.PrerequisiteInput(
        name="xtalk_class1",
        doc="Class 1 crosstalk amplitudes table.",
        storageClass="Xtalk",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    xtalk_class2 = cT.PrerequisiteInput(
        name="xtalk_class2",
        doc="Class 2 crosstalk coupling table.",
        storageClass="Xtalk",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0,
    )
    spximg_out = cT.Output(
        name="level2",
        doc="Retried level2 output.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    idqr_obs = cT.Output(
        name="idqr_obs",
        doc="IDQR per observation.",
        storageClass="IdqrObs",
        dimensions=("instrument", "exposure"),
    )
    dqa_l2 = cT.Output(
        name="dqa_l2",
        doc="Level2 DQA record.",
        storageClass="Level2DQA",
        dimensions=("instrument", "exposure"),
    )
    astr_diag = cT.Output(
        name="astr_diag",
        doc="Diagnostic JSON file with WCS solution details.",
        storageClass="AstrDiag",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.estimate_fine_astrometry.do_write_diagnostics:
            del self.astr_diag


class L2bRetryConfig(pipeBase.PipelineTaskConfig, pipelineConnections=L2bRetryConnections):
    """Config for L2b Retry"""

    remove_reference_pixels = pexConfig.ConfigurableField(
        doc="Remove Reference Pixels", target=RemoveReferencePixelsTask
    )
    apply_bad_pixel_mask = pexConfig.ConfigurableField(
        doc="Apply Bad Pixel Mask", target=ApplyBadPixelMaskTask
    )
    apply_dark_flat_field_and_photo_calib = pexConfig.ConfigurableField(
        doc="Apply dark/flat/photo calib", target=ApplyDarkFlatFieldAndPhotoCalibTask
    )
    estimate_fine_astrometry = pexConfig.ConfigurableField(
        doc="Retry fine astrometry", target=EstimateFineAstrometryRetryTask
    )
    create_source_mask = pexConfig.ConfigurableField(
        doc="Create source mask", target=CreateSourceMaskTask
    )
    detect_outlier_pixels = pexConfig.ConfigurableField(
        doc="Detect outlier pixels", target=DetectOutlierPixelsTask
    )
    get_zodi_model = pexConfig.ConfigurableField(
        doc="Get zodi model", target=GetZodiModelTask
    )
    measure_psf = pexConfig.ConfigurableField(
        doc="Measure PSF", target=MeasurePSFTask
    )
    create_spectral_image = pexConfig.ConfigurableField(
        doc="Create spectral image", target=CreateSpectralImageTask
    )
    dqa_l2 = pexConfig.ConfigurableField(
        doc="Perform DQA", target=DQAL2Task
    )


class L2bRetryTask(pipeBase.PipelineTask):
    """Retry task for L2b"""

    _DefaultName = "l2b-retry-pipeline"
    ConfigClass = L2bRetryConfig
    full_name = "SPHEREx L2b Retry Pipeline"
    ssdc_pipe_level = 2

    def __init__(self, **keyArgs):
        super().__init__(**keyArgs)
        self.makeSubtask("remove_reference_pixels")
        self.makeSubtask("apply_bad_pixel_mask")
        self.makeSubtask("apply_dark_flat_field_and_photo_calib")
        self.makeSubtask("estimate_fine_astrometry")
        self.makeSubtask("create_source_mask")
        self.makeSubtask("detect_outlier_pixels")
        self.makeSubtask("get_zodi_model")
        self.makeSubtask("measure_psf")
        self.makeSubtask("create_spectral_image")
        self.makeSubtask("dqa_l2")

    def _select_donor_payload(self, spximg, l2bwcs_payload: dict) -> tuple[dict, str]:
        """Select donor payload for retry.

        Preference:
        1. nearest adjacent small slew on same detector with strictly better FINAST
        2. opposite detector in same small slew
        3. nearest detector in same row within same small slew
        """
        target_observation = spximg.header["OBSID"]
        target_detector = int(spximg.getArrayId())

        target_finast = _get_target_finast(
            l2bwcs_payload=l2bwcs_payload,
            target_observation=target_observation,
            target_detector=target_detector,
            default=999,
        )

        donor = _select_adjacent_same_detector_solution(
            l2bwcs_payload=l2bwcs_payload,
            target_observation=target_observation,
            target_detector=target_detector,
            max_finast=target_finast,
        )
        if donor is not None:
            return donor, "adjacent_small_slew_same_detector"

        donor = _select_same_small_slew_dichroic_solution(
            l2bwcs_payload=l2bwcs_payload,
            target_observation=target_observation,
            target_detector=target_detector,
            max_finast=target_finast,
        )
        if donor is not None:
            return donor, "same_small_slew_dichroic"

        # donor = _select_same_small_slew_same_row_solution(
        #     l2bwcs_payload=l2bwcs_payload,
        #     target_observation=target_observation,
        #     target_detector=target_detector,
        #     max_finast=target_finast,
        # )
        # if donor is not None:
        #     return donor, "same_small_slew_same_row"

        # in _select_donor_payload, instead of raise RuntimeError
        return None, "no_donor_run_astronet"

    def runQuantum(
        self,
        butlerQC,
        inputRefs: InputQuantizedConnection,
        outputRefs: OutputQuantizedConnection,
    ) -> None:
        inputs = butlerQC.get(inputRefs)
        # preserve observation-level payload before replacing l2bwcs with donor payload
        l2bwcs_payload = inputs["l2bwcs"]
        prior_finast = _get_target_finast(
            l2bwcs_payload=l2bwcs_payload,
            target_observation=inputs["spximg"].header["OBSID"],
            target_detector=int(inputs["spximg"].getArrayId()),
            default=999,
        )

        # retrieve donor payload
        donor_payload, donor_strategy = self._select_donor_payload(
            inputs["spximg"], inputs["l2bwcs"]
        )
        if donor_payload is not None:
            inputs["l2bwcs"] = donor_payload
        else:
            inputs["l2bwcs"] = None

        if donor_payload is not None:
            print("========== DONOR PAYLOAD ==========")
            for k, v in donor_payload.items():
                print(f"{k}: {v}")
            print(f"donor observation = {donor_payload.get('observation')}")
            print(f"donor obsid       = {donor_payload.get('obsid')}")
            print("===================================")

        donor_spximg = None

        if donor_payload is not None:

            donor_collection = donor_payload.get("source_collection")

            try:

                print("========== DONOR IMAGE DEBUG ==========")
                print(f"type(butlerQC) = {type(butlerQC)}")
                print(f"hasattr(butlerQC, 'butler') = {hasattr(butlerQC, 'butler')}")
                print(f"dir(butlerQC) = {dir(butlerQC)}")
                print("=======================================")

                qc_butler = butlerQC._QuantumContext__butler

                refs = list(
                    qc_butler.registry.queryDatasets(
                        "level2",
                        collections=[donor_collection],
                        where=(
                            f"instrument='spx' "
                            f"AND observation={int(donor_payload['observation'])} "
                            f"AND detector={int(donor_payload['detector'])}"
                        ),
                    )
                )

                if len(refs) == 1:
                    donor_spximg = qc_butler.get(refs[0])
                else:
                    raise RuntimeError(
                        f"Expected 1 donor level2 dataset, found {len(refs)}"
                    )

                self.log.info(
                    "Retrieved donor level2 image for CRVAL refinement: obs=%s det=%s collection=%s",
                    donor_payload["observation"],
                    donor_payload["detector"],
                    donor_collection,
                )
            except Exception as e:
                self.log.warning(
                    "Could not retrieve donor level2 image for CRVAL refinement: %s",
                    e,
                )
                donor_spximg = None

        # If donor_payload is None: no donor WCS seed, runs astrometry.net from scratch
        # If donor_spximg is None (but donor_payload exists): uses donor-seeded WCS
        #   but skips CRVAL refinement via source matching
        outputs = self.run(**inputs, donor_spximg=donor_spximg)
        input_ids = get_calibration_inputs(inputRefs)
        add_calibration_provenance(outputs.spximg_out, input_ids)
        # write info to header
        outputs.spximg_out.header["HIERARCH WCSRETRY"] = True
        outputs.spximg_out.header["HIERARCH RETRY_WCSMODE"] = donor_strategy

        if donor_payload is not None:
            outputs.spximg_out.header["HIERARCH WCSSEED"] = int(donor_payload["detector"])
            outputs.spximg_out.header["HIERARCH WCSSEEDOBS"] = int(donor_payload["observation"])
            outputs.spximg_out.header["HIERARCH WCSSEEDFIN"] = int(donor_payload.get("finast", -999))
            donor_det = donor_payload.get("detector")
            donor_obs = donor_payload.get("observation")
        else:
            outputs.spximg_out.header["HIERARCH WCSSEED"] = "NONE"
            outputs.spximg_out.header["HIERARCH WCSSEEDOBS"] = "NONE"
            outputs.spximg_out.header["HIERARCH WCSSEEDFIN"] = "NONE"
            donor_det = None
            donor_obs = None

        new_finast = int(outputs.spximg_out.header.get("FINAST", -999))

        self.log.info(
            "L2bRetry FINAST comparison: prior=%s new=%s "
            "target_obs=%s target_det=%s donor_strategy=%s donor_det=%s donor_obs=%s",
            prior_finast,
            new_finast,
            inputs["spximg"].header.get("OBSID"),
            inputs["spximg"].getArrayId(),
            donor_strategy,
            donor_det,
            donor_obs,
        )

        # Write FINAST retry solution
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        spximg: SPHERExImage,
        l2bwcs: dict,
        abs_gain_matrix: SPHERExImage,
        dark: SPHERExImage,
        sip: Sip,
        srccat: Table,
        zodi_scaling_v2: Table,
        spectral_wcs: HDUList,
        psf_avg: PSFZoneData | None = None,
        epsf: SPHERExImage | None = None,
        ssocat: Table | None = None,
        xtalk_class2: Table | None = None,
        xtalk_class1: Table | None = None,
        he_template: SPHERExImage | None = None,
        aurora_template: SPHERExImage | None = None,
        shuttle_glow_template: SPHERExImage | None = None,
        donor_spximg: SPHERExImage | None = None,
    ) -> pipeBase.Struct:

        check_l1_dqa(spximg)

        result = self.remove_reference_pixels.run(spximg)
        spximg = result.spximg_out

        result = self.apply_bad_pixel_mask.run(spximg)
        spximg = result.spximg_out

        result = self.apply_dark_flat_field_and_photo_calib.run(spximg, abs_gain_matrix, dark)
        spximg = result.spximg_out

        result = self.estimate_fine_astrometry.run(spximg, srccat, l2bwcs=l2bwcs, sip=sip,
                                                   donor_spximg=donor_spximg,)
        spximg = result.spximg_out
        astr_diag = result.astrometry_diagnostics

        result = self.create_source_mask.run(
            spximg=spximg,
            srccat=srccat,
            ssocat=ssocat,
            xtalk_class2=xtalk_class2,
            xtalk_class1=xtalk_class1,
            abs_gain_matrix=abs_gain_matrix
        )
        spximg = result.spximg_out

        result = self.detect_outlier_pixels.run(spximg)
        spximg = result.spximg_out

        result = self.get_zodi_model.run(spximg, zodi_scaling_v2, spectral_wcs)
        spximg = result.spximg_out

        result = self.measure_psf.run(spximg=spximg, srccat=srccat, psf_avg=psf_avg, epsf=epsf)
        spximg = result.spximg_out

        if he_template is not None and spximg.getArrayId() in (1, 2):
            result = self.create_spectral_image.run(
                spximg,
                spectral_wcs,
                he_template,
                aurora_template=aurora_template,
                shuttle_glow_template=None,
            )
        elif spximg.getArrayId() == 5:
            result = self.create_spectral_image.run(
                spximg,
                spectral_wcs,
                he_template=None,
                aurora_template=None,
                shuttle_glow_template=shuttle_glow_template,
            )
        else:
            result = self.create_spectral_image.run(
                spximg,
                spectral_wcs,
                he_template=None,
                aurora_template=None,
                shuttle_glow_template=None,
            )
        spximg = result.spximg_out

        result = self.dqa_l2.run(spximg, srccat)
        spximg = result.spximg_out
        dqa_l2 = result.dqa_l2
        idqr_obs = result.idqr_obs

        return pipeBase.Struct(
            spximg_out=spximg,
            dqa_l2=dqa_l2,
            idqr_obs=idqr_obs,
            astr_diag=astr_diag,
        )
