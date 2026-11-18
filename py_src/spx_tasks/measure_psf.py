__all__ = ["MeasurePSFConfig", "MeasurePSFTask"]

# Standard library imports
import random
import string

# Related third party imports
from astropy.table import Table
import astropy.units as u
from lsst.pex.config import Field, ListField
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, PSFZoneData, Layer
from spx_tasks import add_history
from spx_utils.psf.psf_fwhm import calculate_fwhm, plot_fwhm_histogram, process_psf_zone_data
from spx_utils.psf.psf_plots import plot_psf
from spx_data.spx_param import get_pix_scale
# ==============================================================================

pixel_scale_arcsec = get_pix_scale()

def _get_psf_zone_centers(psf_avg: PSFZoneData, config):
    """Get zone centers for output PSFZoneData.

    Prefer the input PSF library centers; fall back to config lists.
    """
    x_centers = getattr(psf_avg, "x_centers", None)
    y_centers = getattr(psf_avg, "y_centers", None)

    if x_centers is None or len(x_centers) == 0:
        x_centers = config.llx
    if y_centers is None or len(y_centers) == 0:
        y_centers = config.lly

    return x_centers, y_centers

def _make_psf_from_avg(psf_avg: PSFZoneData, config) -> PSFZoneData:
    """Create output PSFZoneData using the average PSF input."""
    if psf_avg is None:
        raise RuntimeError("average_psf input is required for PSF-cube output")

    x_centers, y_centers = _get_psf_zone_centers(psf_avg, config)

    return PSFZoneData(
        data=np.asarray(psf_avg.data, dtype=np.float64),
        x_centers=x_centers,
        y_centers=y_centers,
    )

def calculate_psf_solid_angle(psf: np.ndarray) -> float:
    """Calculate the PSF solid angle."""
    maximum_flux = np.nanmax(psf)
    if np.nansum(psf) == 0 or maximum_flux == 0:
        return np.nan * u.arcsec**2

    normalized_psf = psf / maximum_flux
    integrated_psf = np.nansum(normalized_psf)

    pixel_solid_angle = (pixel_scale_arcsec**2).to(u.arcsec**2)
    psf_solid_angle = integrated_psf * pixel_solid_angle

    return psf_solid_angle

def fwhm_neff(neff: float) -> float:
    """Calculate the 2D Gaussian FWHM as a function of the effective number
    of pixels.

    Parameters
    ----------
    neff : float
        Effective number of pixels.

    Notes
    -----
    Derived assuming:
    - N_eff = 4 * pi * sigma^2
    - FWHM = 2 * sqrt(2 ln 2) * sigma
    """
    fwhm = np.sqrt((2 * np.log(2)) / np.pi) * np.sqrt(neff) * pixel_scale_arcsec
    return fwhm

class MeasurePSFConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("instrument", "exposure"),
                            defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",
        doc="Input image is output from previous module.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    psf_avg = cT.PrerequisiteInput(
        name="average_psf",
        doc="Input PSF cube library.",
        storageClass="PSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    epsf = cT.PrerequisiteInput(
        name="epsf",
        doc="Input average EPSF product used as a template for EPSF binary-table output.",
        storageClass="EPSF",
        dimensions=("instrument", "detector"),
        isCalibration=True,
        minimum=0,
    )
    srccat = cT.Input(
        name="catalog",
        doc="Input catalogs.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "dichroicpair", "spsplan", "slewgroup"],
    )
    spximg_out = cT.Output(
        name="spximg_out",
        doc="Output after measure per-exposure PSF.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if config.write_epsf_bintable:
            del self.psf_avg
        else:
            del self.epsf


class MeasurePSFConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=MeasurePSFConnections):
    """Configuration parameters for Measure PSF task."""

    llx = ListField(
        doc="X offsets of the effective PSF bins.",
        dtype=float,
        default=[340.5, 1020.5, 1700.5])

    lly = ListField(
        doc="Y offsets of the effective PSF bins.",
        dtype=float,
        default=[340.5, 1020.5, 1700.5])

    plot_path = Field(
        doc="Path to save plots.",
        dtype=str,
        default=""
    )

    do_psf_omega = Field(
        doc="Add PSF solid angle to header.",
        dtype=bool,
        default=False
    )

    write_epsf_bintable = Field(
        doc="If True, attach EPSF binary table to output image instead of PSF cube.",
        dtype=bool,
        default=True
    )

    def update(self, **kw):
        """Override default update method to validate upon field changes.
        """
        super().update(**kw)
        self.__validate__()

    def __validate__(self):
        """Override default validate method to include inter-field checks.
        """
        super().validate()

        if self.write_epsf_bintable and self.plot_path != "":
            raise NotImplementedError("FWHM plotting is not yet supported for effective PSF.")

class MeasurePSFTask(pipeBase.PipelineTask):
    """Implementation of L2 Measure PSF.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-023

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "measure-psf"
    ConfigClass = MeasurePSFConfig
    full_name = "Measure Per-Exposure PSF"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-023 v3.3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            srccat: Table,
            psf_avg: PSFZoneData = None,
            epsf: SPHERExImage = None,
            plt: bool = False,
            verbose: bool = False) -> pipeBase.Struct:
        c = self.config

        # Prevent silent failure due to mismatch between config and run args
        if c.write_epsf_bintable and epsf is None:
            raise ValueError("write_epsf_bintable is True but epsf input is None")
        if not c.write_epsf_bintable and psf_avg is None:
            raise ValueError("write_epsf_bintable is False but psf_avg input is None")

        try:
            obsid = spximg.header["OBSID"]
        except KeyError:
            char_set = string.ascii_letters + string.digits
            obsid = ''.join(random.choice(char_set) for _ in range(8))

        # Choose PSF product
        if c.write_epsf_bintable:
            spximg.epsf_table = epsf.epsf_table
            spximg.layers.pop(Layer.PSF, None)
            fwhms = u.Quantity([fwhm_neff(row["NEFF_MEAN"]) for row in epsf.epsf_table.data])
        else:
            psf = _make_psf_from_avg(psf_avg, c)
            spximg.psfcube = psf
            spximg.layers.pop(Layer.EPSF, None)
            fwhms = u.Quantity(calculate_fwhm(psf)["fwhms_arcsec"])

        # Add header products
        fwhm_median = np.nanmedian(fwhms)
        if np.isnan(fwhm_median):
            fwhm_median = 0.0 * fwhms.unit
        spximg.header["PSF_FWHM"] = fwhm_median.to_value(u.arcsec)
        spximg.header.comments["PSF_FWHM"] = " [arcsec] PSF FWHM Median"

        if c.do_psf_omega:
            if c.write_epsf_bintable:
                psf_omega = (np.pi / (4 * np.log(2))) * fwhm_median**2
            else:
                psf_omega = calculate_psf_solid_angle(psf.data)
            if np.isnan(psf_omega):
                psf_omega = 0.0 * u.arcsec**2
            spximg.header["HIERARCH PSF_OMEGA"] = (
                psf_omega.to_value(u.arcsec**2),
                "[arcsec^2] PSF solid angle"
            )

        if plt and c.plot_path and not c.write_epsf_bintable:
            process_psf_zone_data(psf, output_dir=c.plot_path, obsid=obsid)
            plot_fwhm_histogram(fwhms.to_value(u.arcsec), 1, output_dir=c.plot_path, obsid=obsid)
            plot_psf(psf.data, output_dir=c.plot_path, obsid=obsid)

        return pipeBase.Struct(
            spximg_out=spximg,
            jitter_psf=None,
            deconvolved_stack_bins=None,
            fwhm_median=fwhm_median,
        )
