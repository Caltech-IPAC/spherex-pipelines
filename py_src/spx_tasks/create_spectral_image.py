__all__ = ["CreateSpectralImageConfig", "CreateSpectralImageTask"]

# Standard library imports
import copy

# Related third party imports
from astropy.io.fits import HDUList
from astropy.time import Time
from lsst.pex.config import Field, ListField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, SpectralWCS, Layer
from spx_tasks import add_history
from spx_utils import fit_airglow_he, attach_airglow_fit_to_header, correlate_image
from spx_utils.template_utils import HE_TEMP_PAR_KWS
# ==============================================================================


class CreateSpectralImageConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
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
    aurora_template = cT.PrerequisiteInput(
        name="aurora_template",
        doc="Aurora Template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0
    )
    shuttle_glow_template = cT.PrerequisiteInput(
        name="shuttle_glow_template",
        doc="Shuttle-glow Template.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        minimum=0
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output after spectral image created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CreateSpectralImageConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CreateSpectralImageConnections):
    """Configuration for CreateSpectralImage task."""

    layers = ListField(
        doc="Additional image layers (apart from data and flags) to preserve.",
        dtype=str,
        default=[Layer.EPSF, Layer.ZODI])

    threshold = Field(
        doc="Threshold above which aurora/shuttleglow contamination is flagged True.",
        dtype=float,
        default=0.2,
    )


class CreateSpectralImageTask(pipeBase.PipelineTask):
    """Implementation of L2 Create Spectral Image.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-024

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "create-spectral-image"
    ConfigClass = CreateSpectralImageConfig
    full_name = "Create Spectral Image"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-024 v2.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage,
            spectral_wcs: HDUList | None,
            he_template: SPHERExImage | None,
            aurora_template: SPHERExImage | None,
            shuttle_glow_template: SPHERExImage | None) -> pipeBase.Struct:
        """Generate an spectral image containing Level 2 data products.
        In implementation, this method will remove any unwanted information or
        data from the input image.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.
        spectral_wcs : `~astropy.io.fits.HDUList`, optional
            Spectral WCS data.
        he_template : `SPHERExImage` or `None`
            Input He I Airglow template.
        aurora_template : `SPHERExImage` or `None`
            Input Aurora template.
        shuttle_glow_template : `SPHERExImage` or `None`
            Input Shuttle-glow template.

        Returns
        -------
        output: `Output`
            An `Output` object containing the ``image`` with all Level 2
            data products as an attribute of type `~spx_data.SPHERExImage`.

        """
        spximg_out = copy.copy(spximg)

        # Remove unwanted layers/information
        # DQA: ensure flux, flags, zodi, PSF layers exist in L2 image
        spximg_out.ensure_layers(self.config.layers)

        # Check that variance layer is in L2 image
        if spximg_out.uncertainty is None:
            raise ValueError("SPHEREx image variance is not set.")

        # set spectral WCS data
        if spectral_wcs is not None:
            spximg_out.spectral_wcs = SpectralWCS(spectral_wcs, key=" ")

        # reset HDU creation time
        spximg_out.header["DATE"] = Time.now().to_value(format="fits")
        spximg_out.header.comments["DATE"] = "HDU creation time (UTC)"

        # explicitly remove `TINT`
        spximg_out.header.remove("TINT", ignore_missing=True)

        # Remove unwanted diagnostic header keywords from earlier modules
        keywords_to_remove = [
            "SNWCIRCL", "HIGHTRNS", "MXTRNSFR", "XTENDRAD", "MINOVRFL", "WTAVGCTR",
            "MAXOVRLP", "MAXSNWLP", "SATHALOM", "SATHALOF", "BRDRMIND", "MAXBLMLP",
            "TRANSFRC", "NSNOWPIX", "NBLMPIX", "NHALOPIX"
        ]

        for key in keywords_to_remove:
            spximg_out.header.remove(key, ignore_missing=True)

        # Force FITS-friendly string representation for headers
        spximg_out.header["BUNIT"] = ("MJy/sr", "Unit of pixel value")
        # Force FITS-friendly string representation for headers
        # spximg_out.uncertainty.unit = "MJy2/sr2"

        band = spximg_out.getArrayId()
        # Fit He Aurora Template on det 1 and 2
        if he_template is not None:
            params_airglow_helium = {
                "WL_AIR": he_template.header[HE_TEMP_PAR_KWS["ag_wavelength"]],
                "HW_AIR": he_template.header[HE_TEMP_PAR_KWS["ag_halfwidth"]],
                "HW_BKG": he_template.header[HE_TEMP_PAR_KWS["ag_halfwidth_bkg"]],
                "WL_TOP": [
                    he_template.header[HE_TEMP_PAR_KWS["wl_top_1"]],
                    he_template.header[HE_TEMP_PAR_KWS["wl_top_2"]],
                ],
            }
            # Get best-fit parameters
            fit_params = fit_airglow_he(spximg_out, band, he_template, spectral_wcs, params_airglow_helium,
                                        use_variance=True, fit_xslope=True, verbose=False)
            # Attach fit pars to image metadata
            attach_airglow_fit_to_header(spximg_out.header, fit_params)

        # Defaults
        aurora_corr = 0.0
        shuttle_glow_corr = 0.0

        airglow_cont = band in (1, 2)
        aurora_cont = False
        shuttle_glow_cont = False

        # Disallow both templates at once
        if aurora_template is not None and shuttle_glow_template is not None:
            raise ValueError("Provide only one of aurora_template or shuttle_glow_template (not both).")

        def _finite_or_zero(x) -> float:
            return 0.0 if not np.isfinite(x) else float(x)

        if aurora_template is not None:
            if band != 1:
                raise ValueError("Aurora template only applies to detector 1.")
            aurora_corr = _finite_or_zero(correlate_image(spximg, aurora_template))
            aurora_cont = aurora_corr > self.config.threshold

        elif shuttle_glow_template is not None:
            if band != 5:
                raise ValueError("Shuttle glow template only applies to detector 5.")
            shuttle_glow_corr = _finite_or_zero(correlate_image(spximg, shuttle_glow_template))
            shuttle_glow_cont = shuttle_glow_corr > self.config.threshold

        spximg_out.header["HIERARCH AURORA_CORR"] = (
            round(aurora_corr, 3), "Correl coeff with aurora template"
        )
        spximg_out.header["HIERARCH SHUTTLE_GLOW_CORR"] = (
            round(shuttle_glow_corr, 3), "Correl coeff with shuttle-glow template"
        )
        spximg_out.header["HIERARCH HE_AIRGLOW_CONT"] = (
            airglow_cont, "True for detectors 1 and 2, False otherwise"
        )
        spximg_out.header["HIERARCH SHUTTLE_GLOW_CONT"] = (
            shuttle_glow_cont, f"True if SHUTTLE_GLOW_CORR > {self.config.threshold}"
        )
        spximg_out.header["HIERARCH AURORA_CONT"] = (
            aurora_cont, f"True if AURORA_CORR > {self.config.threshold}"
        )

        return pipeBase.Struct(spximg_out=spximg_out)
