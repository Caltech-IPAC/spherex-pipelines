__all__ = ["GetZodiModelConfig", "GetZodiModelTask", "hasSPHERExSimulator"]

# Standard library imports
import copy
import functools

# Related third party imports
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
from scipy.interpolate import RectBivariateSpline

# Simulator dependencies
try:
    import SPHEREx_SkySimulator as SPsky
    import SPHEREx_InstrumentSimulator as SPinst
    hasSPHERExSimulator = True
except ImportError:
    hasSPHERExSimulator = False

# Local application/library specific imports
from spx_data import spx_param, SPHERExImage, Zodi
from spx_tasks import add_history

# ==============================================================================


@functools.lru_cache(maxsize=1)
def _get_zodi_model():
    """Return the shared zodi model.

    Constructing the model takes ~0.2 s and the result does not depend on the
    exposure, so it is built once per process.
    """
    return SPsky.zodicalc.ModifiedKelsallModelWithHPFT()


@functools.lru_cache(maxsize=1)
def _get_instrument():
    """Return the shared instrument model.

    Constructing the tabular bandpass takes ~1.3 s and does not depend on the
    exposure, so it is built once per process.
    """
    return SPinst.Instrument(lvf_model=SPinst.Tabular_Bandpass())


def _build_zodi_correction_maps(zodi_scaling: Table, spectral_wcs: HDUList) -> np.ndarray:
    """Interpolate 1D zodi scaling to 2D map using spectral WCS."""
    wl = np.asarray(zodi_scaling["wavelength_um"], dtype=float)
    scale_1d = np.asarray(zodi_scaling["ZodiScale"], dtype=float)
    cwave_um = np.asarray(spectral_wcs["CWAVE"].data, dtype=float)

    scale_2d = np.interp(cwave_um, wl, scale_1d)

    return scale_2d


class GetZodiModelConnections(pipeBase.PipelineTaskConnections,
                              dimensions=("instrument", "exposure"),
                              defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    zodi_scaling_v2 = cT.PrerequisiteInput(
        name="zodi_scaling_v2",
        doc="Detector-dependent 1D zodi scaling table in ECSV format.",
        storageClass="AstropyTable",
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    spectral_wcs = cT.PrerequisiteInput(
        name="spectral_wcs",
        doc="Spectral calibration product containing CWAVE image.",
        storageClass="AstropyImage",  # replace with your repo's actual storageClass if different
        dimensions=("instrument", "detector"),
        isCalibration=True,
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after detected outlier pixels.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class GetZodiModelConfig(pipeBase.PipelineTaskConfig,
                         pipelineConnections=GetZodiModelConnections):
    """Configuration for GetZodiModel task."""

    zodi_subsample = Field(
        "Pixel grid spacing for sampling Zodi model.",
        dtype=int,
        default=15)

    sparse_bandpass = Field(
        "If True, fewer bandpass points will be used to improve performance "
        "at some expense in accuracy.",
        dtype=bool,
        default=True)


class GetZodiModelTask(pipeBase.PipelineTask):
    """Implementation of L2 Get Zodi Model.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-022

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "get-zodi-model"
    ConfigClass = GetZodiModelConfig
    full_name = "Get Zodi Model"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-022 v2.4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(
        self,
        spximg: SPHERExImage,
        zodi_scaling_v2: Table,
        spectral_wcs: HDUList,
        ) -> pipeBase.Struct:
        """Return zodi model based on exposure meta-data.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            Input SPHEREx image. Observation time, observation coordinates, and
            array ID will be used to generate an image of the zodi model,
            appended to the input `spx_image` object with Zodi model appended.
        zodi_scaling_v2 : `astropy.table.Table`
            Wavelength-dependent zodi scale calibration.
        spectral_wcs : `astropy.io.fits.HDUList`
            Spectral WCS calibration containing central wavelength.

        Returns
        -------
        output : `Output`
            A dataclass with the returned SPHEREx Image as ``image`` attribute.
            This image contains the input image with a ``zodi`` layer
            from the diffused zodiacal model with the same shape as the image
            layer.

        """
        # Algorithm Implementation
        # > Extract relevant meta-data
        array_id = spximg.getArrayId()
        shape = spximg.data.shape
        jdate = spximg.getExposureStartTime()  # Julian Date
        wcs = spximg.wcs

        if not hasSPHERExSimulator:
            raise ImportError("SPHEREx Simulator is not available")

        # > Get zodi model from SPHEREx Sky Simulator
        zodi_model = _get_zodi_model()
        subsample = self.config.zodi_subsample
        sparse_bandpass = self.config.sparse_bandpass

        # > Main Calculations
        # get the bandpass info and coordinates
        nx, ny = shape

        # for best results, subsample size should be chosen so that
        # the image is covered by an integer number of subsamples
        # the implementation below does not quite work otherwise
        if nx % subsample != 0 or ny % subsample != 0:
            raise ValueError(f"Bad subsample: {nx} / {subsample} is not an integer.")

        nx_lr = nx // subsample + 1
        ny_lr = ny // subsample + 1

        # get ra, dec at subsampled pixels and make SkyCoord object
        iy_grid_ss, ix_grid_ss = np.indices((nx_lr, ny_lr)) * subsample
        ra_, dec_ = wcs.all_pix2world(ix_grid_ss, iy_grid_ss, 0)
        cc = SkyCoord(ra_, dec_, frame='icrs', unit='deg')

        SPHEREx_Inst = _get_instrument()

        # get bandpass functions at subsampled pixels
        npix_blind = spx_param.get_npix_blind()
        wls, t = SPHEREx_Inst.pixel_to_band(
            ix_grid_ss + npix_blind, iy_grid_ss + npix_blind,
            array=array_id, sparse=sparse_bandpass)

        # calculate ZL brightnesses at subsampled pixels
        obstime = Time(jdate, format='jd', scale='utc')
        r = zodi_model(obstime, cc, wls)

        # apply the bandpass functions
        # we are interested in the true Zodi background value at a given pixel position,
        # therefore we have to normalize out the transmission
        zmap_lowres = np.sum(r * t, axis=-1) / np.sum(t, axis=-1)

        # see https://scipy.github.io/devdocs/notebooks/interp_transition_guide.html
        # the degrees of bivariate spline are kx=3 and ky=3 by default
        rect_bivariate_spline = RectBivariateSpline(
            ix_grid_ss[0, :], iy_grid_ss[:, 0],
            zmap_lowres.T)

        def zodi_interpolator(x, y):
            return rect_bivariate_spline(x, y).T

        zl_map = zodi_interpolator(np.arange(nx), np.arange(ny))

        zodi_model_name = "KEL98_WAVESCL_V1"

        # Get zodi correction
        scale_2d = _build_zodi_correction_maps(zodi_scaling_v2, spectral_wcs)
        zl_map_corr = zl_map * scale_2d

        # Create an Output object
        # Use shallow copy when feasible
        spximg_out = copy.copy(spximg)

        # Attach zodi layer
        spximg_out.zodi = Zodi(zl_map_corr, zodi_model_name) # default unit is "MJy / sr"

        if spximg_out.zodi is None:
            raise ValueError("Zodi layer was not added to L2 image.")

        return pipeBase.Struct(spximg_out=spximg_out)
