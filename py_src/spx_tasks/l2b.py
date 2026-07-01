__all__ = ["L2bConfig", "L2bTask"]

# Standard library imports

# Related third party imports
from astropy.io.fits import HDUList
from astropy.table import Table
# from astropy.wcs import Sip
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.connections import InputQuantizedConnection, OutputQuantizedConnection
from astropy.wcs import Sip

# Local application/library specific imports
from spx_data import (PSFZoneData, SPHERExImage)
from ._provenance import add_calibration_provenance, get_calibration_inputs
from .remove_reference_pixels import RemoveReferencePixelsTask
from .apply_bad_pixel_mask import ApplyBadPixelMaskTask
from .apply_dark_flat_field_and_photo_calib import ApplyDarkFlatFieldAndPhotoCalibTask
from .estimate_fine_astrometry import EstimateFineAstrometryTask
from .create_source_mask import CreateSourceMaskTask
from .detect_outlier_pixels import DetectOutlierPixelsTask
from .get_zodi_model import GetZodiModelTask
from .measure_psf import MeasurePSFTask
from .create_spectral_image import CreateSpectralImageTask
from .dqa_l1 import check_l1_dqa
from .dqa_l2 import DQAL2Task

# ==============================================================================


class L2bConnections(pipeBase.PipelineTaskConnections,
                     dimensions=("instrument", "exposure"),
                     defaultTemplates={}):
    spximg = cT.Input(
        name="level2a",  # Input dataset type for beginning Level 2.
        doc="Input raw (should be level 1) image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    abs_gain_matrix = cT.PrerequisiteInput(
        name="abs_gain_matrix",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    dark = cT.PrerequisiteInput(
        name="dark",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    sip = cT.PrerequisiteInput(
        name="sip",
        doc="Input distortion as SIP (Simple Imaging Polynomial).",
        storageClass="AstropySIP",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    # ssocat = cT.Input(
    #     name="ssocat",
    #     doc="Input solar system object catalog.",
    #     storageClass="AstropyParquet",
    #     dimensions=["instrument", "exposure"],
    # )
    psf_avg = cT.PrerequisiteInput(
        name="average_psf",
        doc="Input psf library.",
        storageClass="PSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral WCS with lookup table.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    # xtalk_coupling_matrix = cT.PrerequisiteInput(
    #    name="xtalk_coupling_matrix",
    #    doc="Crosstalk coupling matrix",
    #    storageClass="SPHERExImage",
    #    dimensions=["instrument", "detector"],
    #    isCalibration=True,
    # )
    spximg_out = cT.Output(
        name="level2",  # default dataset type for output
        doc="Output spectral image.",
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
        doc="Level 2 DQA record.",
        storageClass="Level2DQA",
        dimensions=("instrument", "exposure"),
    )
    astr_diag = cT.Output(
        name="astr_diag",
        doc="Diagnostic JSON file with WCS solution details.",
        storageClass="AstrDiag",
        dimensions=("instrument", "exposure"),
    )

    # Deactivate the astr diagnostics output connection if not desired
    def __init__(self, *, config=None):
        super().__init__(config=config)
        if not config.estimate_fine_astrometry.do_write_diagnostics:
            del self.astr_diag


class L2bConfig(pipeBase.PipelineTaskConfig,
                pipelineConnections=L2bConnections):
    remove_reference_pixels = pexConfig.ConfigurableField(
        doc="Remove Reference Pixels",
        target=RemoveReferencePixelsTask
    )
    apply_bad_pixel_mask = pexConfig.ConfigurableField(
        doc="Apply Bad Pixel Mask",
        target=ApplyBadPixelMaskTask
    )
    apply_dark_flat_field_and_photo_calib = pexConfig.ConfigurableField(
        doc="Apply Diffuse Flat-Field Correction and Photometric Calibration",
        target=ApplyDarkFlatFieldAndPhotoCalibTask
    )
    estimate_fine_astrometry = pexConfig.ConfigurableField(
        doc="Estimate Fine Astrometry",
        target=EstimateFineAstrometryTask
    )
    create_source_mask = pexConfig.ConfigurableField(
        doc="Create Source Mask",
        target=CreateSourceMaskTask
    )
    detect_outlier_pixels = pexConfig.ConfigurableField(
        doc="Detect Outlier Pixels",
        target=DetectOutlierPixelsTask
    )
    get_zodi_model = pexConfig.ConfigurableField(
        doc="Get Zodi Model",
        target=GetZodiModelTask
    )
    measure_psf = pexConfig.ConfigurableField(
        doc="Measure Per-Exposure PSF",
        target=MeasurePSFTask
    )
    create_spectral_image = pexConfig.ConfigurableField(
        doc="Create Spectral Image",
        target=CreateSpectralImageTask
    )
    dqa_l2 = pexConfig.ConfigurableField(
        doc="Perform Data Quality Assessment",
        target=DQAL2Task
    )

    def setDefaults(self):
        super().setDefaults()


class L2bTask(pipeBase.PipelineTask):
    """Implementation of Level2b data processing pipeline"""

    _DefaultName = "l2b-pipeline"
    ConfigClass = L2bConfig
    full_name = "SPHEREx L2b Pipeline v2024.01-1"
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

    def runQuantum(self, butlerQC,
                   inputRefs: InputQuantizedConnection,
                   outputRefs: OutputQuantizedConnection) -> None:
        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        # add provenance info with the IDs of all input calibrations
        input_ids = get_calibration_inputs(inputRefs)
        add_calibration_provenance(outputs.spximg_out, input_ids)
        butlerQC.put(outputs, outputRefs)

    def run(self,
            spximg: SPHERExImage,
            abs_gain_matrix: SPHERExImage,
            dark: SPHERExImage,
            sip: Sip,
            srccat: Table,
            # ssocat: Table,
            # xtalk_coupling_matrix: SPHERExImage,
            psf_avg: PSFZoneData,
            spectral_wcs: HDUList
            ) -> pipeBase.Struct:
        # Raise error if L1 DQA was not passed
        check_l1_dqa(spximg)

        # Remove Reference Pixels
        task: RemoveReferencePixelsTask = getattr(self, "remove_reference_pixels")
        result = task.run(spximg)
        spximg = result.spximg_out

        # Apply Bad Pixel Mask
        task: ApplyBadPixelMaskTask = getattr(self, "apply_bad_pixel_mask")
        result = task.run(spximg)
        spximg = result.spximg_out

        # Apply Diffuse Flat Field Correction and Photometric Calibration
        task: ApplyDarkFlatFieldAndPhotoCalibTask = getattr(self, "apply_dark_flat_field_and_photo_calib")
        result = task.run(spximg, abs_gain_matrix, dark)
        spximg = result.spximg_out

        # Estimate Fine Astrometry
        task: EstimateFineAstrometryTask = getattr(self, "estimate_fine_astrometry")
        result = task.run(spximg, srccat, sip=sip)
        spximg = result.spximg_out
        astr_diag = result.astrometry_diagnostics

        # Create Source Mask
        task: CreateSourceMaskTask = getattr(self, "create_source_mask")
        # TODO: handle ssocat
        result = task.run(spximg, srccat,
                          # xtalk_coupling_matrix=xtalk_coupling_matrix,
                          abs_gain_matrix=abs_gain_matrix, ssocat=None)
        spximg = result.spximg_out

        # Detect Outlier Pixels
        task: DetectOutlierPixelsTask = getattr(self, "detect_outlier_pixels")
        result = task.run(spximg)
        spximg = result.spximg_out

        # Get Zodi Model
        # Adds zodi layer
        task: GetZodiModelTask = getattr(self, "get_zodi_model")
        result = task.run(spximg)
        spximg = result.spximg_out

        # Measure Per-Exposure PSF
        # Adds psf layer
        task: MeasurePSFTask = getattr(self, "measure_psf")
        result = task.run(spximg, srccat, psf_avg)
        spximg = result.spximg_out
        # print(spximg.header["PSF_FWHM"])

        # Create Spectral Image
        task: CreateSpectralImageTask = getattr(self, "create_spectral_image")
        result = task.run(spximg, spectral_wcs)
        spximg = result.spximg_out

        # Perform Data Quality Assessment
        task: DQAL2Task = getattr(self, "dqa_l2")
        result = task.run(spximg, srccat)
        spximg = result.spximg_out
        dqa_l2 = result.dqa_l2
        idqr_obs = result.idqr_obs

        return pipeBase.Struct(spximg_out=spximg, dqa_l2=dqa_l2, idqr_obs=idqr_obs, astr_diag=astr_diag)
