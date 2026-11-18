__all__ = ["L3aConfig", "L3aTask"]

# Related third party imports
import lsst.pex.config as pexConfig
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy import units as u
from astropy.io.fits import HDUList
from astropy.table import Table, vstack
from lsst.daf.butler import DeferredDatasetHandle

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils import get_current_rss
from .estimate_diffuse_background import EstimateDiffuseBackgroundTask
from .perform_forced_photometry import PerformForcedPhotometryTask


def validate_table(srccat: Table):
    """Validate input table for the level 3 photometry pipeline.

    Parameters
    ----------
    srccat : `astropy.table.Table`
        Query Reference Catalog-like table of sources to photometer.

    Raises
    ------
    ValueError
        If expected column has incorrect data format.
    ValueError
        If expected column has incorrect units.
    ValueError
        If negative SPHERExRefIDs are found in the input catalog.
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
        if srccat[col["name"]].dtype.name != col["format"]:
            raise ValueError(f"Column '{col['name']}' does not have expected type {col['format']}.")
        actual_unit = (u.Unit(srccat[col["name"]].unit) if srccat[col["name"]].unit is not None
                        else u.dimensionless_unscaled)
        if actual_unit != col["unit"]:
            raise ValueError(f"Column '{col['name']}' does not have expected units {col['unit']}")

    if not np.all(srccat["SPHERExRefID"] >= 0):
        raise ValueError("Negative SPHERExRefIDs found in srccat.")

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
        deferLoad=True
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
    he_template = cT.PrerequisiteInput(
        name="he_template",
        doc="He I Airglow Template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0
    )
    special_ids = cT.PrerequisiteInput(
        name="special_ids",
        doc="List of filtered source IDs to photometer at the window level.",
        storageClass="NumpyArray",
        dimensions=["instrument"],
        isCalibration=False
    )
    epsf = cT.PrerequisiteInput(
        name="epsf",
        doc="Input average EPSF product used as a template for EPSF binary-table output.",
        storageClass="EPSF",
        dimensions=("instrument", "detector"),
        isCalibration=True
    )
    ssocat = cT.Input(
        name="ssocat",
        doc="Input Solar System Object catalog with sources found in SPHEREx FOV.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "observation"),
        deferLoad=True
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
    phot_exceptions = cT.Output(
        name="phot_exceptions",
        doc="Output photometry measurement table for sources with HEALPix-"
            "related exceptions.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
    )
    phot_meas_combined = cT.Output(
        name="phot_meas_combined",
        doc="Output photometry measurement table (full image).",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
        multiple=False
    )
    phot_meas_sso = cT.Output(
        name="phot_meas_sso",
        doc="Output photometry measurement table for Solar System Objects.",
        storageClass="AstropyParquet",
        dimensions=("instrument", "exposure"),
        multiple=False
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.estimate_diffuse_background.est_he_airglow:
            del self.he_template

        if config.perform_forced_photometry.combinedOutput:
            del self.phot_meas
            del self.phot_exceptions
        else:
            del self.phot_meas_combined

        if not config.perform_forced_photometry.filter_by_ids:
            del self.special_ids

        if not config.perform_forced_photometry.override_epsf:
            del self.epsf

        if not config.perform_forced_photometry.fit_sso:
            del self.ssocat
            del self.phot_meas_sso


class L3aConfig(pipeBase.PipelineTaskConfig,
                pipelineConnections=L3aConnections):
    """Configuration for L3a pipeline task."""

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
        self.log.debug(get_current_rss("L3a runQuantum start"))
        if getattr(self, "perform_forced_photometry").config.combinedOutput:
            super().runQuantum(butlerQC, inputRefs, outputRefs)
        else:
            inputs = butlerQC.get(inputRefs)
            output_ref_map = {name: getattr(outputRefs, name) for name in vars(outputRefs)
                              if name in ["phot_meas", "phot_exceptions", "phot_meas_sso"]}
            result = self.run(**inputs)

            if result.phot_meas is not None:
                ref_or_lst = output_ref_map["phot_meas"]
                out_healpix_ids = result.phot_meas.keys()
                expected_healpix_ids = {ref.dataId.get("hips6") for ref in ref_or_lst}
                unexpected_ids = set(out_healpix_ids).difference(expected_healpix_ids)

                # Per-HEALPixel measurement tables predicted by level 2 WCS/
                # exposure region
                for ref in ref_or_lst:
                    healpix_id = ref.dataId.get("hips6")
                    if healpix_id in out_healpix_ids:
                        butlerQC.put(result.phot_meas[healpix_id], ref)
                    else:
                        self.log.warning(f"No sources found for expected healpix ID {healpix_id}.")

                # Per-exposure measurement table for high proper motion sources
                # whose corrected positions shifted the output HEALPix ID
                if len(unexpected_ids) > 0:
                    unexpected_tabs = []
                    for uid in unexpected_ids:
                        self.log.warning(f"Adding sources from unexpected healpix ID {uid} to"
                                         " exceptions table.")
                        unexpected_tabs.append(result.phot_meas[uid])
                    exceptions = vstack(unexpected_tabs)
                    if len(exceptions) > 0:
                        butlerQC.put(exceptions, output_ref_map["phot_exceptions"])

            if getattr(self, "perform_forced_photometry").config.fit_sso \
                and result.phot_meas_sso is not None \
                and len(result.phot_meas_sso) > 0:
                butlerQC.put(result.phot_meas_sso, output_ref_map["phot_meas_sso"])

            supported = {"phot_meas", "phot_exceptions", "phot_meas_sso"}
            for name in output_ref_map:
                if name not in supported:
                    # TODO handle other types of output
                    self.log.warning(f"ERROR: output for connection {name} is not saved")

    def run(self, spximg: SPHERExImage, srccat: DeferredDatasetHandle,
            solid_angle_pixel_map: SPHERExImage, spectral_wcs: HDUList,
            he_template: SPHERExImage = None, special_ids: np.ndarray = None,
            epsf: SPHERExImage = None, ssocat: DeferredDatasetHandle = None) -> pipeBase.Struct:
        """Level 3a pipeline execution method.

        Parameters
        ----------
        spximg : `spx_data.SPHERExImage`
            LVF image with catalog of sources to be measured
        srccat : `lsst.daf.butler.DeferredDatasetHandle`
            Catalog of sources within or close to the image.
        solid_angle_pixel_map : `spx_data.SPHERExImage`
            Solid angle pixel map derived by Update Field Distortion module.
        spectral_wcs : `~astropy.io.fits.HDUList`
            Spectral WCS calibration with central wavelength and bandwidth.
        he_template : `spx_data.SPHERExImage`, optional
            He I airglow template. If not specified, no airglow component will
            be estimated.
        special_ids : `numpy.ndarray`, optional
            Array of filtered source IDs to photometer at the window level. If
            not specified, all sources flagged for photometry in the input
            catalog are photometered at the window level.
        epsf : `spx_data.SPHERExImage`, optional
            EPSF calibration to override the extension stored in the image.
        ssocat : `lsst.daf.butler.DeferredDatasetHandle`, optional
            Deferred `astropy.table.Table` of Solar System Objects to
            photometer.

        Returns
        -------
        result : `pipeBase.Struct`
            Container for output connection fields. By default, only includes
            `phot_meas` (`dict` of `astropy.table.Table` grouped by HEALPix). If
            `self.config.perform_forced_photometry.return_fit` is True, includes
            `tractor_fits` (`dict` of `astropy.fits.io.HDUList`).
            If `self.config.perform_forced_photometry.combinedOutput` is True,
            includes `phot_meas_combined` (`astropy.table.Table`). If
            `self.config.perform_forced_photometry.combinedOutput` is True,
            includes `phot_meas_sso` (`astropy.table.Table`).

        Raises
        ------
        ValueError
            If input source catalog is missing requested column or has negative
            SPHERExRefIDs.
        ValueError
            If input Solar System Object catalog is missing requested column.
        """
        # Only read necessary columns from input source catalog
        if self.config.perform_forced_photometry.merge_srccat:
            read_cols = list(self.config.perform_forced_photometry.cat_phot_cols) + \
                list(self.config.perform_forced_photometry.pass_thru_cols)
        else:
            read_cols = list(self.config.perform_forced_photometry.cat_phot_cols)

        try:
            srccat = srccat.get(
                parameters={
                    "columns": read_cols
                }
            )
        except KeyError as e:
            raise ValueError(f"Requested column {e} is not in the input catalog.")
        else:
            validate_table(srccat)

        # If fitting SSOs, read catalog
        if self.config.perform_forced_photometry.fit_sso and ssocat is not None:
            try:
                ssocat = ssocat.get(
                    parameters={
                        "columns": ["desig", "RA", "DEC"]  # only need identifier and positions
                    }
                )
            except KeyError as e:
                raise ValueError(f"Requested column {e} is not in the solar system object catalog.")

        # Filter out images that failed fine astrometry
        if ("FINAST" not in spximg.header) or (spximg.header["FINAST"] != 0):
            self.log.warning("Input level 2 image failed fine astrometry; skipping level 3 processing.")
            sso_kwargs = {"phot_meas_sso": None} if self.config.perform_forced_photometry.fit_sso else {}
            if self.config.perform_forced_photometry.combinedOutput:
                return pipeBase.Struct(phot_meas_combined=None, **sso_kwargs)
            else:
                return pipeBase.Struct(phot_meas=None, **sso_kwargs)

        # Estimate Diffuse Background
        task: EstimateDiffuseBackgroundTask = getattr(self, "estimate_diffuse_background")
        result = task.run(spximg=spximg, spectral_wcs=spectral_wcs, he_template=he_template)
        self.log.debug(get_current_rss("EDB task end"))
        global_bkgd = result.global_bkgd

        # Perform Forced Photometry
        task: PerformForcedPhotometryTask = getattr(self, "perform_forced_photometry")
        result = task.run(spximg=spximg, srccat=srccat, global_bkgd=global_bkgd,
                          solid_angle_pixel_map=solid_angle_pixel_map, spectral_wcs=spectral_wcs,
                          special_ids=special_ids, epsf=epsf, ssocat=ssocat)
        self.log.debug(get_current_rss("PFP task end"))

        return result  # PFP task returns obj with correct output connection
