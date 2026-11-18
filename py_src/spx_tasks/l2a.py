__all__ = ["L2aConfig", "L2aTask"]

# Standard library imports

# Related third party imports
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

# Local application/library specific imports
from spx_data import (SPHERExImage, GainFactors, NonLinearityParameters, ObservationId,
                      ReadNoiseParameters, PersistenceState)
from spx_tasks.gds_image_association import GdsImageAssociationTask
from spx_tasks.source_bloom_and_cosmic_ray_halo_flagging import SourceBloomAndCosmicRayHaloFlaggingTask
from typing import Optional
from astropy.table import Table
from astropy.time import Time
from ._provenance import add_calibration_provenance, get_calibration_inputs
from .correct_non_linearity import CorrectNonLinearityTask
from .create_variance_map import CreateVarianceMapTask
from .update_persistence import UpdatePersistenceTask
from .dqa_l1 import check_l1_dqa
# ==============================================================================


class L2aConnections(pipeBase.PipelineTaskConnections,
                     dimensions=("instrument", "exposure"),
                     defaultTemplates={}):
    spximg = cT.Input(
        name="level1",  # Input dataset type for beginning Level 2.
        doc="Input level 1 image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    persistence_state_in = cT.Input(
        name="persistence_state",
        doc="Persistence model state from the previous exposure",
        storageClass="PersistenceState",
        dimensions=("instrument", "exposure"),
        deferGraphConstraint=True,
        # option to defer connection binding to custom QA builder
        # deferBinding=True
    )
    nonlinear_pars = cT.PrerequisiteInput(
        name="nonlinear_pars",
        storageClass="NonLinearityParameters",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    gain_factors = cT.PrerequisiteInput(
        name="gain_factors",
        doc="Gain factors to apply to image.",
        storageClass="GainFactors",
        dimensions=["instrument"],
        isCalibration=True
    )
    readnoise_pars = cT.PrerequisiteInput(
        name="readnoise_pars",
        doc="Read noise map.",
        storageClass="ReadNoiseParameters",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="level2a",  # default dataset type for output
        doc="Output image after nonlinearity correction.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    persistence_state_out = cT.Output(
        name="persistence_state",
        doc="Persistence model state.",
        storageClass="PersistenceState",
        dimensions=("instrument", "exposure"),
    )
    persistence_image = cT.Output(
        name="persistence_image",
        doc="Persistence image.",
        storageClass="PersistenceImage",
        dimensions=("instrument", "exposure"),
    )
    spk = cT.Input(
        name="spk",
        doc="Pointing ephemeris data",
        storageClass="GdsData",
        dimensions=["instrument", "observation"],
        isCalibration=False,
    )

    sgt = cT.Input(
        name="sgt",
        doc="Reconstructed ground track data",
        storageClass="AstropyTable",
        dimensions=["instrument", "observation"],
        isCalibration=False,
    )

    sband_mnemonics = cT.Input(
        name="sband_mnemonics",
        doc="List of sband data to aggregate",
        storageClass="SbandMnemonics",
        dimensions=["instrument"],
        isCalibration=True,
    )
    sband = cT.Input(
        name="sband",
        doc="S-band data",
        storageClass="GdsData",
        dimensions=["instrument", "observation"],
        isCalibration=False,
    )
    bsh_diag_img = cT.Output(
        name="bsh_diag_img",
        doc="Diagnostic image containing BSH diagnostic layers (BLM_DIAG, SNW_DIAG, HLO_DIAG).",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.gds_image_association.useSband:
            del self.sband
            del self.sband_mnemonics
        if not config.gds_image_association.useSpk:
            del self.spk
        if not config.gds_image_association.useSgt:
            del self.sgt
        if not config.persistence_prior:
            self.inputs.discard("persistence_state_in")
        if not config.source_bloom_and_cosmic_ray_halo_flagging.write_diagnostics:
            del self.bsh_diag_img


class L2aConfig(pipeBase.PipelineTaskConfig,
                pipelineConnections=L2aConnections):
    """Configuration for L2a pipeline task."""

    gds_image_association = pexConfig.ConfigurableField(
        doc="GDS Image Association",
        target=GdsImageAssociationTask
    )
    source_bloom_and_cosmic_ray_halo_flagging = pexConfig.ConfigurableField(
        doc="Source Blooming and Transient Halo Flagging",
        target=SourceBloomAndCosmicRayHaloFlaggingTask
    )
    correct_non_linearity = pexConfig.ConfigurableField(
        doc="Correct Nonlinearity",
        target=CorrectNonLinearityTask
    )
    create_variance_map = pexConfig.ConfigurableField(
        doc="Create Variance map",
        target=CreateVarianceMapTask
    )
    update_persistence = pexConfig.ConfigurableField(
        doc="Update Persistence Mask",
        target=UpdatePersistenceTask
    )
    persistence_prior = pexConfig.Field(
        doc="True, if pipeline should expect persistence prior.",
        dtype=bool,
        default=False)

    fail_on_bad_l1_dqa = pexConfig.Field(
        doc="True, if quanta should fail on bad L1 DQA.",
        dtype=bool,
        default=False)

    def setDefaults(self):
        super().setDefaults()


class L2aTask(pipeBase.PipelineTask):
    """Implementation of Level2a data processing pipeline."""

    _DefaultName = "l2a-pipeline"
    ConfigClass = L2aConfig
    full_name = "SPHEREx L2a Pipeline v2024.01-1"
    ssdc_pipe_level = 2

    def __init__(self, **keyArgs):
        super().__init__(**keyArgs)
        self.makeSubtask("gds_image_association")
        self.makeSubtask("source_bloom_and_cosmic_ray_halo_flagging")
        self.makeSubtask("correct_non_linearity")
        self.makeSubtask("create_variance_map")
        self.makeSubtask("update_persistence")

    def runQuantum(self, butlerQC,
                   inputRefs: InputQuantizedConnection,
                   outputRefs: OutputQuantizedConnection) -> None:
        inputs = butlerQC.get(inputRefs)

        # Walkaround for missing OBSID in the simulated images
        hdr = inputs["spximg"].header
        if "OBSID" not in hdr:
            obsid = inputRefs.spximg.dataId["exposure"]
            hdr["OBSID"] = ObservationId(obsid).sps_id

        outputs = self.run(**inputs)
        # add provenance info with the IDs of all input calibrations
        input_ids = get_calibration_inputs(inputRefs)
        add_calibration_provenance(outputs.spximg_out, input_ids)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            spximg: SPHERExImage,
            gain_factors: GainFactors,
            nonlinear_pars: NonLinearityParameters,
            readnoise_pars: ReadNoiseParameters,
            persistence_state_in: PersistenceState = None,
            spk: Optional[Table] = None,
            sgt: Optional[Table] = None,
            sband: Optional[Table] = None,
            sband_mnemonics: Optional[Table] = None,
            ) -> pipeBase.Struct:
        # Raise error if L1 DQA was not passed
        if self.config.fail_on_bad_l1_dqa:
            check_l1_dqa(spximg)

        task: GdsImageAssociationTask = getattr(self, "gds_image_association")
        # SPK and SGT
        result = task.run(spximg,
                          None,  # Planned observation ingest
                          sgt,
                          sband,
                          spk,
                          sband_mnemonics)
        spximg = result.spximg_out

        # Source bloom and transient halo flagging
        task: SourceBloomAndCosmicRayHaloFlaggingTask =\
            getattr(self, "source_bloom_and_cosmic_ray_halo_flagging")
        result = task.run(spximg)
        spximg = result.spximg_out
        bsh_diag_img = result.bsh_diag_img

        # Correct Non-linearity
        task: CorrectNonLinearityTask = getattr(self, "correct_non_linearity")
        result = task.run(spximg, nonlinear_pars, gain_factors)
        spximg = result.spximg_out

        # Create Variance Map: adds variance layer.
        task: CreateVarianceMapTask = getattr(self, "create_variance_map")
        result = task.run(spximg, gain_factors, readnoise_pars)
        spximg = result.spximg_out

        # Update Persistence Mask
        task: UpdatePersistenceTask = getattr(self, "update_persistence")
        result = task.run(spximg, persistence_state_in)
        spximg = result.spximg_out
        # Updated persistence state needed to be preserved.
        persistence_state_out = result.persistence_state_out
        persistence_image = result.persistence_image

        # reset HDU creation time
        spximg.header["DATE"] = Time.now().to_value(format="fits")
        spximg.header.comments["DATE"] = "HDU creation time (UTC)"

        return pipeBase.Struct(spximg_out=spximg, bsh_diag_img=bsh_diag_img,
                               persistence_state_out=persistence_state_out,
                               persistence_image=persistence_image)
