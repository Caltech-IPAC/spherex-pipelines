__all__ = ["L1OdirOnlyConfig", "L1OdirOnlyTask", "L1WithSbandNoOdirConfig", "L1WithSbandNoOdirTask"]

from astropy.io.fits import HDUList
from astropy.table import Table
from typing import Optional

from lsst.daf.butler import DeferredDatasetHandle
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection

from spx_data import SPHERExImage, GainFactors, PlannedObservation, NonfuncData
from ._provenance import add_calibration_provenance, get_calibration_inputs
from .l1 import L1Task, L1Config


class L1OdirOnlyConnections(pipeBase.PipelineTaskConnections,
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
    nonfunc_map_v2 = cT.PrerequisiteInput(
        name="nonfunc",
        doc="Permanently unusable.",
        storageClass="NonfuncData",
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


class L1OdirOnlyConfig(L1Config, pipelineConnections=L1OdirOnlyConnections):
    """Config for L1 and output only ODIR task."""

    def setDefaults(self):
        super().setDefaults()
        self.gds_image_association.useSband = False
        self.create_image_data_files.layers = ["RAWDATA", "CUBEFLGS"]


class L1OdirOnlyTask(L1Task):
    """Run full L1 task, disables S-band by default,
    and only outputs `odir_obs` plus optional diagnostics.
    """

    _DefaultName = "l1-odir-only"
    ConfigClass = L1OdirOnlyConfig
    full_name = "SPHEREx L1 Pipeline (ODIR-only persist)"

    def runQuantum(self, butlerQC,
                   inputRefs: InputQuantizedConnection,
                   outputRefs: OutputQuantizedConnection) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            spximg: DeferredDatasetHandle,
            planned_obs: PlannedObservation,
            gain_factors: GainFactors,
            nonfunc_map_v2: NonfuncData,
            dichroic_cut_map: SPHERExImage,
            spectral_wcs: HDUList,
            initial_distortion_table: Table,
            sgt: Optional[Table] = None,
            spk: Optional[Table] = None,
            sband: Optional[Table] = None,
            sband_mnemonics: Optional[Table] = None
            ) -> pipeBase.Struct:
        outputs = super().run(
            spximg=spximg,
            planned_obs=planned_obs,
            gain_factors=gain_factors,
            nonfunc_map_v2=nonfunc_map_v2,
            dichroic_cut_map=dichroic_cut_map,
            spectral_wcs=spectral_wcs,
            initial_distortion_table=initial_distortion_table,
            sgt=sgt,
            spk=spk,
            sband=sband,
            sband_mnemonics=sband_mnemonics,
        )

        return pipeBase.Struct(
            dqa_l1=outputs.dqa_l1,
            sur_diag=outputs.sur_diag,
            odir_obs=outputs.odir_obs,
        )


class L1WithSbandNoOdirConnections(pipeBase.PipelineTaskConnections,
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
    nonfunc_map_v2 = cT.PrerequisiteInput(
        name="nonfunc",
        doc="Permanently unusable.",
        storageClass="NonfuncData",
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

    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.gds_image_association.useSband:
            del self.sband
            del self.sband_mnemonics
        if not config.gds_image_association.useSpk:
            del self.spk
        if not config.gds_image_association.useSgt:
            del self.sgt


class L1WithSbandNoOdirConfig(L1Config, pipelineConnections=L1WithSbandNoOdirConnections):
    """Config for L1 with S-band and no ODIR output task."""

    def setDefaults(self):
        super().setDefaults()
        self.gds_image_association.useSband = True
        self.create_image_data_files.layers = ["RAWDATA", "CUBEFLGS", "GDSDATA"]


class L1WithSbandNoOdirTask(L1Task):
    """Run full L1 task with S-band by default,
    and only outputs L1 image plus optional diagnostics.
    Does not output `odir_obs`.
    """

    _DefaultName = "l1-with-sband-no-odir"
    ConfigClass = L1WithSbandNoOdirConfig
    full_name = "SPHEREx L1 Pipeline (with S-band, no ODIR persist)"

    def runQuantum(self, butlerQC,
                   inputRefs: InputQuantizedConnection,
                   outputRefs: OutputQuantizedConnection) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)

        input_ids = get_calibration_inputs(inputRefs)
        add_calibration_provenance(outputs.level1_img, input_ids)

        butlerQC.put(outputs, outputRefs)

    def run(self,
            spximg: DeferredDatasetHandle,
            planned_obs: PlannedObservation,
            gain_factors: GainFactors,
            nonfunc_map_v2: SPHERExImage,
            dichroic_cut_map: SPHERExImage,
            spectral_wcs: HDUList,
            initial_distortion_table: Table,
            sgt: Optional[Table] = None,
            spk: Optional[Table] = None,
            sband: Optional[Table] = None,
            sband_mnemonics: Optional[Table] = None
            ) -> pipeBase.Struct:
        outputs = super().run(
            spximg=spximg,
            planned_obs=planned_obs,
            gain_factors=gain_factors,
            nonfunc_map_v2=nonfunc_map_v2,
            dichroic_cut_map=dichroic_cut_map,
            spectral_wcs=spectral_wcs,
            initial_distortion_table=initial_distortion_table,
            sgt=sgt,
            spk=spk,
            sband=sband,
            sband_mnemonics=sband_mnemonics,
        )

        return pipeBase.Struct(
            level1_img=outputs.level1_img,
            dqa_l1=outputs.dqa_l1,
            sur_diag=outputs.sur_diag,
        )

