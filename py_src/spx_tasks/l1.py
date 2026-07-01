__all__ = ["L1Config", "L1Task"]

# Standard library imports

# Related third party imports
from astropy.table import Table
from lsst.daf.butler import DeferredDatasetHandle
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from astropy.io.fits import HDUList
from typing import Optional

# Local application/library specific imports
from spx_butler.defaults import SPHERExImageConfig
from spx_data import SPHERExImage, GainFactors, PlannedObservation
from ._provenance import add_calibration_provenance, get_calibration_inputs
from spx_tasks.gds_image_association import GdsImageAssociationTask
from spx_tasks.estimate_coarse_astrometry import EstimateCoarseAstrometryTask
from spx_tasks.convert_to_engineering_units import ConverttoEngineeringUnitsTask
from spx_tasks.validate_sur import ValidateSURTask
from spx_tasks.correct_preamplifier import CorrectPreamplifierTask
from spx_tasks.create_image_data_files import CreateImageDataFilesTask
from spx_tasks.dqa_l1 import DQAL1Task


class L1Connections(pipeBase.PipelineTaskConnections,
                    dimensions=("instrument", "exposure"),
                    defaultTemplates={}):
    spximg = cT.Input(
        name="raw",
        doc="Raw images to pass through L1 pipeline.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "exposure"],
        multiple=False,
        deferLoad=True
    )
    planned_obs = cT.Input(
        name="planned_obs",
        doc="Planned observation data",
        storageClass="PredictedObservation",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )
    sband_mnemonics = cT.PrerequisiteInput(
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
    sgt = cT.Input(
        name="sgt",
        doc="Reconstructed ground track data",
        storageClass="AstropyTable",
        dimensions=["instrument", "observation"],
        isCalibration=False,
    )
    spk = cT.Input(
        name="spk",
        doc="Pointing ephemeris data",
        storageClass="GdsData",
        dimensions=["instrument", "observation"],
        isCalibration=False,
    )
    gain_factors = cT.PrerequisiteInput(
        name="gain_factors",
        doc="Gain factors to apply to image.",
        storageClass="GainFactors",
        dimensions=["instrument"],
        isCalibration=True
    )
    nonfunc_map = cT.PrerequisiteInput(
        name="nonfunc",
        doc="Permanently unusable.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )

    dichroic_cut_map = cT.PrerequisiteInput(
        name="dichroic",
        doc="Low efficiency due to dichroic.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral coordinates lookup data.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    initial_distortion_table = cT.PrerequisiteInput(
        name="initial_distortion_table",
        doc="Initial data to fit distortion from ray-tracing.",
        storageClass="InitDistTable",
        dimensions=["instrument"],
        isCalibration=True
    )
    level1_img = cT.Output(
        name="level1",
        doc="The output of the level 1 pipeline.",
        storageClass="SPHERExL1Image",
        dimensions=("instrument", "exposure")
    )
    dqa_l1 = cT.Output(
        name="dqa_l1",
        doc="Level 1 DQA record.",
        storageClass="Level1DQA",
        dimensions=("instrument", "exposure"),
    )
    sur_diag = cT.Output(
        name="sur_diag",
        doc="The SUR diagnostics.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure")
    )
    odir_obs = cT.Output(
        name="odir_obs",
        doc="ODIR per observation.",
        storageClass="OdirObs",
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


class SPHERExL1ImageConfig(SPHERExImageConfig):
    def setDefaults(self):
        super().setDefaults()
        # Override the set values.
        self.dtype_flags = "int16"


class L1Config(pipeBase.PipelineTaskConfig, pipelineConnections=L1Connections):

    gds_image_association = pexConfig.ConfigurableField(
        doc="GDS Image Association",
        target=GdsImageAssociationTask
    )
    estimate_coarse_astrometry = pexConfig.ConfigurableField(
        doc="Estimate Coarse Astrometry",
        target=EstimateCoarseAstrometryTask
    )
    validate_sur = pexConfig.ConfigurableField(
        doc="Validate SUR",
        target=ValidateSURTask
    )
    convert_to_engineering_units = pexConfig.ConfigurableField(
        doc="Convert To Engineering Units",
        target=ConverttoEngineeringUnitsTask
    )
    correct_preamplifier = pexConfig.ConfigurableField(
        doc="Correct Preamplifier",
        target=CorrectPreamplifierTask
    )
    create_image_data_files = pexConfig.ConfigurableField(
        doc="Create Image Data Files",
        target=CreateImageDataFilesTask
    )
    dqa_l1 = pexConfig.ConfigurableField(
        doc="Perform Data Quality Assessment",
        target=DQAL1Task
    )
    l1_image = pexConfig.ConfigField(
        doc="Configuration for storage class `SPHERExImageL1` butler parameters.",
        dtype=SPHERExL1ImageConfig
    )

    def setDefaults(self):
        super().setDefaults()


class L1Task(pipeBase.PipelineTask):
    """Implementation of Level1 data processing pipeline."""
    _DefaultName = "l1-pipeline"
    ConfigClass = L1Config
    full_name = "SPHEREx L1 Pipeline v2024.01-1"
    ssds_pipe_level = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("gds_image_association")
        self.makeSubtask("estimate_coarse_astrometry")
        self.makeSubtask("validate_sur")
        self.makeSubtask("convert_to_engineering_units")
        self.makeSubtask("correct_preamplifier")
        self.makeSubtask("create_image_data_files")
        self.makeSubtask("dqa_l1")

    def runQuantum(self, butlerQC,
                   inputRefs: InputQuantizedConnection,
                   outputRefs: OutputQuantizedConnection) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        # add provenance info with the IDs of all input calibrations
        input_ids = get_calibration_inputs(inputRefs)
        add_calibration_provenance(outputs.level1_img, input_ids)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            spximg: DeferredDatasetHandle,
            planned_obs: PlannedObservation,
            gain_factors: GainFactors,
            nonfunc_map: SPHERExImage,
            dichroic_cut_map: SPHERExImage,
            spectral_wcs: HDUList,
            initial_distortion_table: Table,
            sgt: Optional[Table] = None,
            spk: Optional[Table] = None,
            sband: Optional[Table] = None,
            sband_mnemonics: Optional[Table] = None
            ) -> pipeBase.Struct:
        """Runs the L1 pipeline.

        Parameters
        ----------
        spximg
            The raw input image to process.
        sgt : `astropy.table.Table`
            Table of reconstructed ground track data.
        gain_factors
            The gain factors to use to convert to engineering units.
        nonfunc_map
            The image of nonfunctional pixels.
        dichroic_cut_map
            Low efficiency pixels due to dichroic.
        spectral_wcs
            Spectral coordinates lookup data.
        initial_distortion_table
            Ray-tracing for initial distortion model.

        Returns
        -------
        `~lsst.pipe.base.Struct`
            Output of L1 pipeline.

        NOTES
        -----
        TODO need to ingest GDS for ck, spk, sband, planned_obs,
        in order to run gds_image_association task.
        """
        # With deferred objects, can pass parameters to get.
        spximg = spximg.get(parameters=self.config.l1_image.toDict())

        # NB: Ingesting, decompressing, and assembling images are handled elsewhere

        # Associate images with GDS data.
        if sgt and self.config.gds_image_association.useSgt:
            print("SGT WORKS")
        if spk and self.config.gds_image_association.useSpk:
            print("SPK WORKS")
        else:
            print("SPK DOES NOT WORK")
        task: GdsImageAssociationTask = getattr(self, "gds_image_association")
        result = task.run(spximg,
                          planned_obs,
                          sgt,
                          sband,
                          spk,
                          sband_mnemonics)
        spximg = result.spximg_out

        # Estimate coarse astrometry
        task: EstimateCoarseAstrometryTask = getattr(self, "estimate_coarse_astrometry")
        result = task.run(spximg, planned_obs, initial_distortion_table)
        spximg = result.spximg_out

        # Construct SUR is done in SPHEREx ingest.

        # Validate SUR
        task: ValidateSURTask = getattr(self, "validate_sur")
        result = task.run(spximg)
        sur_diag = result.sur_diag

        # Convert to engineering units
        task: ConverttoEngineeringUnitsTask = getattr(self, "convert_to_engineering_units")
        result = task.run(spximg, gain_factors)
        spximg = result.spximg_out

        # Correct preamplifier
        task: CorrectPreamplifierTask = getattr(self, "correct_preamplifier")
        result = task.run(spximg)
        spximg = result.spximg_out
        even_refs_masked = result.even_refs_masked
        odd_refs_masked = result.odd_refs_masked

        # Create image data files
        task: CreateImageDataFilesTask = getattr(self, "create_image_data_files")
        # nonfunc map includes all physical pixels, active and refs,
        # but dichroic map includes only active pixels.
        result = task.run(spximg, nonfunc_map, dichroic_cut_map, spectral_wcs)
        spximg = result.spximg_out

        # Perform data quality assessment
        task: DQAL1Task = getattr(self, "dqa_l1")
        result = task.run(spximg)
        spximg = result.spximg_out
        dqa_l1 = result.dqa_l1
        odir_obs = result.odir_obs

        # Metadata are saved in butler repository
        self.metadata.add("even_refs_masked", even_refs_masked)
        self.metadata.add("odd_refs_masked", odd_refs_masked)

        output = pipeBase.Struct(
            level1_img=spximg,
            dqa_l1=dqa_l1,
            sur_diag=sur_diag,
            odir_obs=odir_obs
        )

        return output
