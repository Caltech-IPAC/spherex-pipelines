__all__ = ["RemoveReferencePixelsConfig", "RemoveReferencePixelsTask"]

# Standard library imports
import copy

# Related third party imports
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_utils import l1_utils, set_active_pix_wcs
from spx_utils.utils import add_detcoord

# ==============================================================================


class RemoveReferencePixelsConnections(pipeBase.PipelineTaskConnections,
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
        doc="Output image after reference pixels removed.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class RemoveReferencePixelsConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=RemoveReferencePixelsConnections):
    """Configuration for RemoveReferencePixels task."""

    exclude_l1_flag_headers = Field(
        doc="Remove L1 flag headers from the output image.",
        dtype=bool,
        default=True,
    )
    exclude_uncertainty_flip = Field(
        doc="Exclude flipping uncertainty layer in SPHERExImage.",
        dtype=bool,
        default=False,
    )
    num_ref_pixels = Field(
        doc="Number of pixels along the edges to be removed.",
        dtype=int,
        default=4)


def remove_l1_flag_headers(header):
    """Remove specific L1 flag-related header keywords if present."""
    keys_to_remove = [
        "L1 N_TRANSIENT",
        "L1 N_OVERFLOW",
        "L1 N_SUR_ERROR",
        "L1 N_PHANTOM",
        "L1 N_REFERENCE",
        "L1 N_NONFUNC",
        "L1 N_DICHROIC",
        "L1 N_MISSING",
        "L1 N_FULLSAMPLE",
        "L1 N_PHANMISS",
    ]

    for key in keys_to_remove:
        # Remove if it exists
        try:
            header.pop(key)
        except KeyError:
            pass


def remove_phantom_reference_flag_defs(spximg):
    """Remove PHANTOM and REFERENCE from flag_defs since those pixels are removed.

    Parameters
    ----------
    spximg : `~spx_data.SPHERExImage`
        Image with flag_defs to modify.

    """
    if spximg.flag_defs:
        for flag_name in ["PHANTOM", "REFERENCE"]:
            spximg.flag_defs.pop(flag_name, None)


class RemoveReferencePixelsTask(pipeBase.PipelineTask):
    """Removes reference and phantom pixels from L1 image.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-002

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "remove-reference-pixels"
    ConfigClass = RemoveReferencePixelsConfig
    full_name = "Remove Reference Pixels"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "None"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:
        """Remove reference and phantom pixels from L1 image.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.

        Returns
        -------
        output: `~lsst.pipe.base.Struct`
            An object containing the ``spximg_out`` of type `~spx_data.SPHERExImage`
            with photosensitive pixels only.

        Note
        ----
        Refer to instrument delivery in alfresco for details.

        """
        actv_area = l1_utils.active_area_range(spximg)

        # Removing reference and phantom pixels from l1 image extensions.
        spximg_out = spximg[actv_area]

        # Set or update 0-based active pixel coordinates
        set_active_pix_wcs(spximg_out.header)

        # Remove L1 flag count header entries
        if self.config.exclude_l1_flag_headers:
            remove_l1_flag_headers(spximg_out.header)

        # For backward compatibility
        # Prior to R7, L1 flag count headers were also set in FLAGS header
        remove_l1_flag_headers(spximg_out.flag_header)

        # Remove PHANTOM and REFERENCE from flag_defs (no longer relevant)
        remove_phantom_reference_flag_defs(spximg_out)

        if spximg_out.getArrayId() in [4, 5, 6]:
            spximg = copy.deepcopy(spximg_out)
            # Flip Data
            spximg_out.data = spximg.data[:, ::-1]
            if not self.config.exclude_uncertainty_flip:
                spximg_out.uncertainty = spximg.uncertainty[:, ::-1]
            spximg_out.flags = spximg.flags[:, ::-1]

        # add detector coordinate system
        add_detcoord(spximg_out)

        return pipeBase.Struct(spximg_out=spximg_out)
