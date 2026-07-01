__all__ = ["L3aConfig", "L3aTask"]

# Standard library imports

# Related third party imports
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy import units as u
from astropy.io.fits import HDUList
from astropy.table import Table

# Local application/library specific imports
from spx_data import SPHERExImage
from .estimate_diffuse_background import EstimateDiffuseBackgroundTask
from .perform_forced_photometry import PerformForcedPhotometryTask


def validate_table(srccat: Table):
    """Validates input table for the level 3 photometry pipeline.

    Parameters
    ----------
    srccat : `astropy.table.Table`
        Query Reference Catalog-like table of sources to photometer.

    Raises
    ------
    ValueError
        If expected column is absent from list of input table column names.
    ValueError
        If expected column has incorrect data format.
    ValueError
        If expected column has incorrect units.
    """
    expected_cols = np.array([
        ("SPHERExRefID", "int64", u.Unit()),
        ("ra", "float64", u.deg),
        ("dec", "float64", u.deg),
        ("shape_r", "float32", u.arcsecond),
        ("ab_ratio", "float32", u.Unit()),
        ("phi", "float32", u.Unit()),
        ("sersic", "float32", u.Unit())
    ], dtype=[("name", "U20"), ("format", "U10"), ("unit", "U20")]
    )

    for col in expected_cols:
        if col["name"] in srccat.colnames:
            if srccat[col["name"]].dtype.name != col["format"]:
                raise ValueError(f"Column {col['name']} does not have expected type {col['format']}.")
            actual_unit = (u.Unit(srccat[col["name"]].unit) if srccat[col["name"]].unit is not None
                           else u.dimensionless_unscaled)
            if actual_unit != col["unit"]:
                raise ValueError(f"Column {col['name']} does not have expected units {col['unit']}")
        else:
            raise ValueError(f"Expected column {col['name']} is missing in input photometry catalog.")

    return


class L3aConnections(pipeBase.PipelineTaskConnections,
                     dimensions=("instrument", "exposure"),
                     defaultTemplates={}):
    spximg = cT.Input(
        name="level2",
        doc="Input level 2 image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    srccat = cT.Input(
        name="srccat",
        doc="Input catalog.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "dichroicpair", "spsplan", "slewgroup"),
    )
    solid_angle_pixel_map = cT.PrerequisiteInput(
        name="solid_angle_pixel_map",
        doc="Output solid angle pixel map.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral WCS with lookup table.",
        storageClass="AstropyImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
    )
    phot_meas = cT.Output(
        name="phot_meas",
        doc="Output astropy table per HEALpix",
        storageClass="AstropyParquet",
        # dimensions should include a table with spatial dimension (exposure)
        # skypix dimension should match commonSkyPix in dimensions.yaml
        dimensions=("instrument", "exposure", "hips6"),
        multiple=True,
    )
    phot_meas_combined = cT.Output(
        name="phot_meas_combined",
        doc="Output photometry measurement table (full image).",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.perform_forced_photometry.combinedOutput:
            self.outputs.discard("phot_meas")
        else:
            self.outputs.discard("phot_meas_combined")

        use_simulator_bandpass_in_edb = config.estimate_diffuse_background.use_simulator_bandpass
        use_simulator_bandpass_in_phot = config.perform_forced_photometry.use_simulator_bandpass
        if use_simulator_bandpass_in_edb and use_simulator_bandpass_in_phot:
            self.inputs.discard("spectral_wcs")


class L3aConfig(pipeBase.PipelineTaskConfig,
                pipelineConnections=L3aConnections):
    estimate_diffuse_background = pexConfig.ConfigurableField(
        doc="Estimate Diffuse Background",
        target=EstimateDiffuseBackgroundTask
    )
    perform_forced_photometry = pexConfig.ConfigurableField(
        doc="Perform Forced Photometry",
        target=PerformForcedPhotometryTask
    )

    def setDefaults(self):
        super().setDefaults()


class L3aTask(pipeBase.PipelineTask):
    """Implementation of Level 3a data processing pipeline."""

    _DefaultName = "l3a-pipeline"
    ConfigClass = L3aConfig
    full_name = "SPHEREx L3a Pipeline"
    ssdc_pipe_level = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.makeSubtask("estimate_diffuse_background")
        self.makeSubtask("perform_forced_photometry")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        if getattr(self, "perform_forced_photometry").config.combinedOutput:
            super().runQuantum(butlerQC, inputRefs, outputRefs)
        else:
            inputs = butlerQC.get(inputRefs)
            result = self.run(**inputs)
            if result.phot_meas is not None:
                healpix_IDs = result.phot_meas.keys()
                for name, ref_or_lst in outputRefs:
                    if name == "phot_meas":
                        expected_healpix_ids = {ref.dataId.get("hips6") for ref in ref_or_lst}
                        unexpected_ids = set(healpix_IDs).difference(expected_healpix_ids)
                        if len(unexpected_ids) > 0:
                            self.log.error(f"Unexpected healpix IDs: {sorted(unexpected_ids)}")
                        for ref in ref_or_lst:
                            healpix_id = ref.dataId.get("hips6")
                            if healpix_id in healpix_IDs:
                                table = result.phot_meas[healpix_id]
                                butlerQC.put(table, ref)
                    else:
                        # TODO handle other types of output
                        print(f"ERROR: output for connection {name} is not saved")

    def run(self, spximg: SPHERExImage, srccat: Table,
            solid_angle_pixel_map: SPHERExImage,
            spectral_wcs: HDUList = None) -> pipeBase.Struct:
        # Validate minimal columns required for fitting
        validate_table(srccat)

        # Filter out images that failed fine astrometry
        if ("FINAST" not in spximg.header) or (spximg.header["FINAST"] != 0):
            self.log.warning("Input level 2 image failed fine astrometry; skipping level 3 processing.")
            return pipeBase.Struct(phot_meas=None)

        # Estimate Diffuse Background
        task: EstimateDiffuseBackgroundTask = getattr(self, "estimate_diffuse_background")
        if self.config.estimate_diffuse_background.use_simulator_bandpass:
            spectral_wcs_edb = None
        else:
            spectral_wcs_edb = spectral_wcs
        result = task.run(spximg, spectral_wcs=spectral_wcs_edb)
        global_bkgd = result.global_bkgd

        # Perform Forced Photometry
        task: PerformForcedPhotometryTask = getattr(self, "perform_forced_photometry")
        if self.config.perform_forced_photometry.use_simulator_bandpass:
            spectral_wcs_phot = None
        else:
            spectral_wcs_phot = spectral_wcs
        result = task.run(spximg, srccat, global_bkgd, solid_angle_pixel_map, spectral_wcs=spectral_wcs_phot)

        return result  # PFP task returns obj with correct output connection
