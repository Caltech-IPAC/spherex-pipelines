__all__ = ["CorrectDarkCurrentConfig", "CorrectDarkCurrentTask"]

# Standard library imports

# Related third party imports
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_tasks import add_history

# ==============================================================================


class CorrectDarkCurrentConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", "exposure"),
                                    defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # default dataset type for input
        doc="Input level 2 image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    dc_calib = cT.PrerequisiteInput(
        name="dark",
        doc="Input dark current image.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after dark current correction.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure")
    )


class CorrectDarkCurrentConfig(pipeBase.PipelineTaskConfig,
                               pipelineConnections=CorrectDarkCurrentConnections):
    pass


class CorrectDarkCurrentTask(pipeBase.PipelineTask):
    """Implementation of L3 Correct Dark Current.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-031

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "correct-dark-current"
    ConfigClass = CorrectDarkCurrentConfig
    full_name = "Correct Dark Current"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "SSDC-MS-031 v1.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage, dc_calib: SPHERExImage) -> pipeBase.Struct:
        """Returns the dark current corrected image by subtracting the input
        image by the provided dark current template.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            A 2-D science image in astrophysical units (MJy/sr).
        dc_calib : `~spx_data.SPHERExImage`
            A 2-D dark current image to be subtracted from the spximg. The
            detector-specific image should be a measurement of the relative
            dark current in each pixel calibrated in astrophysical units
            (MJy/sr).

        Returns
        -------
        output : `Output`
            Contains the corrected ``image`` as an attribute of type
            `~spx_data.SPHERExImage`.

        Notes
        -----
        ``spximg`` and ``dc_calib`` must be the same size and shape. Variance
        for dark current is ignored. Copy image variance to output.
        """
        # Copies original image flags, flag_defs, meta, and layers
        spx_reduced = spximg.subtract(operand=dc_calib,
                                      propagate_uncertainties=None,
                                      handle_flags="ff", handle_meta="ff",
                                      handle_layers="ff")
        spx_reduced.uncertainty = spximg.uncertainty

        return pipeBase.Struct(spximg_out=spx_reduced)
