__all__ = ["CreateImageDataFilesConfig", "CreateImageDataFilesTask"]

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
from spx_data import SPHERExImage, SpectralWCS, Flag, ReadoutParameters, NonfuncData
from spx_utils import active_area_range, with_extra_refs
from spx_utils.utils import add_detcoord, is_device_detcoord
from spx_tasks import add_history
# ==============================================================================


class CreateImageDataFilesConnections(pipeBase.PipelineTaskConnections,
                                      dimensions=("instrument", "exposure"),
                                      defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
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
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after variance map created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CreateImageDataFilesConfig(pipeBase.PipelineTaskConfig,
                                 pipelineConnections=CreateImageDataFilesConnections):
    """Configuration for CreateImageDataFiles task."""

    layers = ListField(
        doc="Additional image layers (apart from data and flags) to preserve.",
        dtype=str,
        default=["RAWDATA", "CUBEFLGS", "GDSDATA"])


class CreateImageDataFilesTask(pipeBase.PipelineTask):
    """Implementation of L1 Create Image Data Files.

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

    _DefaultName = "create-image-data-files"
    ConfigClass = CreateImageDataFilesConfig
    full_name = "Create Image Data Files"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-007 v1.2.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            nonfunc_map: Optional[NonfuncData] = None,
            dichroic_cut_map: Optional[SPHERExImage] = None,
            spectral_wcs: Optional[HDUList] = None) -> pipeBase.Struct:
        """Generate a spectral image containing Level 1 data products.
        In implementation, this method will remove any unwanted information or
        data from the input image.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input image.
        nonfunc_map : `NonfuncData`
            Non-functional pixel map.
        dichroic_cut_map : `~spx_data.SPHERExImage`
            Map of the pixels affected by dichroic beam splitter.
        spectral_wcs : `~astropy.io.fits.HDUList`, optional
            Spectral WCS data.

        Returns
        -------
        output: `Output`
            An `Output` object containing the ``image`` with all Level 1
            data products as an attribute of type `~spx_data.SPHERExImage`.

        """
        spximg_out = copy.copy(spximg)
        rp = ReadoutParameters.from_metadata(spximg.meta)

        # Active pixel area.
        active_area = active_area_range(spximg_out)
        # Active and reference, with extra ref reads, pixel area.
        actv_ref_area = active_area_range(spximg_out, include_refs=True)

        # Apply NONFUNC flags to active and reference (with extra reads) pixels.
        if nonfunc_map is not None:
            if isinstance(nonfunc_map, SPHERExImage):
                nonfunc_map_nrefs = with_extra_refs(rp, nonfunc_map.data)
            elif isinstance(nonfunc_map, NonfuncData):
                # COMBINED plane (1=bad, 0=good) from NonfuncData
                nonfunc_mask = nonfunc_map.get_combined_data()
                nonfunc_map_nrefs = with_extra_refs(rp, nonfunc_mask)
            else:
                raise TypeError(f"Unsupported type for nonfunc_map: {type(nonfunc_map)}"
                                f"Expected SPHERExImage or NonfuncData.")
            spximg_out.setFlagsFromMask(nonfunc_map_nrefs, Flag.NONFUNC, area=actv_ref_area)

        # Apply DICHROIC flags to active pixels.
        if dichroic_cut_map is not None:
            spximg_out.setFlagsFromMask(dichroic_cut_map.data, Flag.DICHROIC, area=active_area)

        # Remove unwanted layers/information.
        spximg_out.ensure_layers(self.config.layers)

        # Set spectral WCS data.
        # The following error encountered in pipeline from spectral_wcs?
        # ValueError: ERROR 5 in wcsset() at line 2764 of file cextern/wcslib/C/wcs.c:
        # Invalid parameter value.
        # ERROR 3 in tabset() at line 808 of file cextern/wcslib/C/tab.c:
        # Invalid tabular parameters: Each element of K must be positive, got 0.
        if spectral_wcs is not None:
            device_detccord = is_device_detcoord(spectral_wcs)
            mwir_detector = spximg_out.getArrayId() > 3
            xflip_needed = mwir_detector and not device_detccord
            spximg_out.spectral_wcs = SpectralWCS(spectral_wcs, key=" ",
                                                  imgshape=spximg.data.shape,
                                                  actv_area=active_area,
                                                  xflip=xflip_needed)
        # reset HDU creation time
        spximg_out.header["DATE"] = Time.now().to_value(format="fits")
        spximg_out.header.comments["DATE"] = "HDU creation time (UTC)"
        # explicitly set DETCOORD
        add_detcoord(spximg_out, device_coord=True)

        return pipeBase.Struct(spximg_out=spximg_out)
