__all__ = ["ConverttoEngineeringUnitsConfig", "ConverttoEngineeringUnitsTask"]

# Standard library imports
import astropy.units as u

# Related third party imports
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import SPHERExImage
from spx_data import GainFactors, ReadoutParameters, l1_readout_channels
from spx_tasks import add_history

# ==============================================================================


class ConverttoEngineeringUnitsConnections(pipeBase.PipelineTaskConnections,
                                           dimensions=("instrument", "exposure"),
                                           defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    gain_factors = cT.PrerequisiteInput(
        name="gain_factors",
        doc="Gain factors to apply to image.",
        storageClass="GainFactors",
        dimensions=["instrument"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image SUR cube constructed.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class ConverttoEngineeringUnitsConfig(pipeBase.PipelineTaskConfig,
                                      pipelineConnections=ConverttoEngineeringUnitsConnections):
    """Configuration for ConverttoEngineeringUnits task."""

    pass


class ConverttoEngineeringUnitsTask(pipeBase.PipelineTask):
    """Implementation of L1 Convert to Engineering Units.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-005

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "convert-to-engineering-units"
    ConfigClass = ConverttoEngineeringUnitsConfig
    full_name = "Convert to Engineering Units"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-005 v3.2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def adu_to_eps(self, spximg: SPHERExImage, g: GainFactors) -> SPHERExImage:
        """Convert adu to electron/s for a SPHERExImage.

        Input
        -----
        spximg : `~spx_data.SPHERExImage`
            Decompressed image in units of ADU.

        rp : `~spx_data.ReadoutParameters`
            Input instrument readout parameters.

        Output
        ------
        spximg : `~spx_data.SPHERExImage`
            Decompressed image in units of electrons/s.

        Notes
        -----
        array_id : `int`
            Detector array id number, starting from 1.

        spximg.data : `SPHERExImage.data`
            Decompressed slopes [ADU] from image.

        """
        rp = ReadoutParameters.from_metadata(spximg.meta)

        array_id = spximg.getArrayId()

        # Get respective gain factor for array id.
        g_k = g.get_gains_for_detector(array_id)

        ro_channels = l1_readout_channels(rp)

        for i, ch in enumerate(ro_channels):
            phan_istart = ch.phan_cols.start  # start phantom column index for this readout channel
            phan_istop = ch.phan_cols.stop  # stop phantom column for this readout channel, exclusive
            spximg[:, phan_istart:phan_istop].data *= g_k[i]

            istart = ch.det_cols.start  # start data column index for this readout channel
            istop = ch.det_cols.stop  # stop data column index for  this readout channel, exclusive
            # Multiplying by gain factors for respective channels in active/h2rg pixels.
            spximg[:, istart:istop].data *= g_k[i]

        eps: float = u.electron / u.s  # unit
        spximg.unit = eps

        return spximg

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            gain_factors: GainFactors) -> pipeBase.Struct:
        """Convert image from ADU to e/s.

        Parameters
        ----------
        spximg
            Input image.
        gain_factors
            Input gain factors.

        Returns
        -------
        output: `~lsst.pipe.base.Struct`
            An object containing the ``image`` in units of
            electron per second as an attribute of type `~spx_data.SPHERExImage`.

        Notes
        -----
        Input gain factors are in units of e-/ADU.

        """
        spximg_out = self.adu_to_eps(spximg, gain_factors)

        return pipeBase.Struct(spximg_out=spximg_out)
