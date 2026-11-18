__all__ = ["SourceBloomAndCosmicRayHaloFlaggingConfig", "SourceBloomAndCosmicRayHaloFlaggingTask"]

# Standard library imports
import copy

# Related third party imports
from astropy.io import fits
from lsst.pex.config import ConfigField, Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import add_history
from spx_utils import BSHConfig, BSHFlagger, apply_bsh_flags_to_spx
# ==============================================================================


class SourceBloomAndCosmicRayHaloFlaggingConnections(pipeBase.PipelineTaskConnections,
                                                     dimensions=("instrument", "exposure"),
                                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after variance map created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    bsh_diag_img = cT.Output(
        name="bsh_diag_img",
        doc="Diagnostic image containing BSH diagnostic layers (BLM_DIAG, SNW_DIAG, HLO_DIAG).",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )

    # Deactivate the bsh diagnostics output connection if not desired
    def __init__(self, *, config=None):
        super().__init__(config=config)
        if config and not config.write_diagnostics:
            del self.bsh_diag_img


class SourceBloomAndCosmicRayHaloFlaggingConfig(
    pipeBase.PipelineTaskConfig,
    pipelineConnections=SourceBloomAndCosmicRayHaloFlaggingConnections,
):
    """Configuration for SourceBloomAndCosmicRayHaloFlagging task."""

    bsh = ConfigField(
        doc="Configuration for Bloom, Snowball, and Halo (BSH) flagging.",
        dtype=BSHConfig,
        default=BSHConfig(),
    )
    write_diagnostics = Field(
        doc="Write BLM_DIAG / SNW_DIAG / HLO_DIAG layers.",
        dtype=bool,
        default=False
    )


class SourceBloomAndCosmicRayHaloFlaggingTask(pipeBase.PipelineTask):
    """Implementation of source blooming and cosmic ray halo and snowball flagging.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-007

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "source-blooming-and-cosmic-ray-halo-flagging"
    ConfigClass = SourceBloomAndCosmicRayHaloFlaggingConfig
    full_name = "Source blooming and cosmic ray halo flagging"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-0011 v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:
        """Identify and flag artifacts from source blooming,
           and cosmic ray halo and snowball flagging.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.

        Returns
        -------
        output: `Output`
            An `Output` object containing the ``image`` with
            source bloom, and transient snowball and halo
            flagging.

        """
        spximg_out = copy.copy(spximg)

        if spximg_out.flags is not None and spximg_out.flags.dtype != np.int32:
            self.log.info(f"Converting flags from {spximg_out.flags.dtype} to int32 for BSH.")
            spximg_out.flags = spximg_out.flags.astype(np.int32, copy=False)

        # BSH flagging
        self.log.info("Starting BSH (Bloom, Snowball, Halo) flagging.")
        bsh_diag_image = None  # Initialize diagnostic image as None

        # Instantiate the flagger
        bsh = BSHFlagger(spximg_out, cfg=self.config.bsh, verbose=False)

        # Run the detection methods
        bsh.start_bloom_flag()
        bsh.get_snowball_flag()
        bsh.get_nonoverflow_bloom_flag()
        bsh.mark_transient_halo()

        # Apply the BSH flags back to the spximg_out's flag plane
        apply_bsh_flags_to_spx(spximg_out, bsh)

        # Add BSH counts to header for diagnostics
        n_bloom = np.count_nonzero(bsh.bloom_flag > 0)
        n_snow = np.count_nonzero(bsh.snowball_flag > 0)
        n_halo = np.count_nonzero(bsh.halo_flag > 0)

        self.log.info(f"BSH flagging complete. Found: "
                      f"{n_bloom} bloom, {n_snow} snowball, {n_halo} halo pixels.")

        bsh_diag_image = None
        # Create and populate diagnostic image if requested
        if self.config.write_diagnostics:
            self.log.info("Creating BSH diagnostic image.")

            # Create diagnostic image from the *original* input spximg.
            # The SPHERExImage writer will automatically save spximg.data as the
            # primary 'IMAGE' HDU and spximg.flags as the 'FLAGS' extension.
            bsh_diag_image = SPHERExImage(
                data=spximg_out.data,
                flags=spximg_out.flags,
                meta=copy.deepcopy(spximg_out.meta),
                unit=spximg_out.unit,
                flag_defs=spximg_out.flag_defs
            )

            # Add the BSH diagnostic layers
            bsh_diag_image._layers["BLOOM_FLAG"] = fits.ImageHDU(
                data=bsh.bloom_flag.astype(np.int16),
                name="BLOOM_FLAG"
            )
            bsh_diag_image._layers["SNOW_FLAG"] = fits.ImageHDU(
                data=bsh.snowball_flag.astype(np.int16),
                name="SNOW_FLAG"
            )
            bsh_diag_image._layers["HALO_FLAG"] = fits.ImageHDU(
                data=bsh.halo_flag.astype(np.uint8),
                name="HALO_FLAG"
            )

            # Add the border info table if requested
            if self.config.bsh.save_borderinfo:
                self.log.info("Adding BLMBDRIN layer to diagnostic image.")
                bsh_diag_image._layers["BLMBDRIN"] = fits.ImageHDU(
                    data=bsh.border_diff_info.astype(np.float32),
                    name="BLMBDRIN"
                )

        return pipeBase.Struct(spximg_out=spximg_out, bsh_diag_img=bsh_diag_image)
