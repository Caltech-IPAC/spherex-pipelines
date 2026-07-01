__all__ = ["CreateSpectralImageConfig", "CreateSpectralImageTask"]

# Standard library imports
import copy
from typing import Optional

# Related third party imports
from astropy.io.fits import HDUList
from astropy.time import Time
from lsst.pex.config import ListField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import SPHERExImage, SpectralWCS, Layer
from spx_tasks import add_history

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
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output after spectral image created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CreateSpectralImageConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CreateSpectralImageConnections):
    layers = ListField(
        doc="Additional image layers (apart from data and flags) to preserve.",
        dtype=str,
        default=[Layer.PSF, Layer.ZODI])


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
    def run(self, spximg: SPHERExImage, spectral_wcs: Optional[HDUList] = None) -> pipeBase.Struct:
        """Generate an spectral image containing Level 2 data products.
        In implementation, this method will remove any unwanted information or
        data from the input image.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.
        spectral_wcs : `~astropy.io.fits.HDUList`, optional
            Spectral WCS data.

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

        # Force FITS-friendly string representation for headers
        spximg_out.header["BUNIT"] = ("MJy/sr", "Unit of pixel value")
        # Force FITS-friendly string representation for headers
        # spximg_out.uncertainty.unit = "MJy2/sr2"

        return pipeBase.Struct(spximg_out=spximg_out)
