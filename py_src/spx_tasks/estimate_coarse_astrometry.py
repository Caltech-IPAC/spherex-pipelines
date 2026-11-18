__all__ = ["EstimateCoarseAstrometryConfig", "EstimateCoarseAstrometryTask"]

# Standard library imports
import copy
import logging

# Related third party imports
from astropy.table import Table
from astropy.wcs import WCS
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np

# Local application/library specific imports
from spx_data import SPHERExImage, PlannedObservation
from spx_utils import (get_ra_dec_lon_lat_from_quat, get_initial_linear_wcs,
                       get_corrected_linear_wcs)
from spx_data import ReadoutParameters
from spx_data.observation_id import ObservationId
from spx_tasks import add_history


class EstimateCoarseAstrometryConnections(pipeBase.PipelineTaskConnections,
                                          dimensions=("instrument", "exposure"),
                                          defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # default dataset type for input
        doc="Input Level_1 image.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "exposure"],
    )
    planned_obs = cT.Input(
        name="planned_obs",
        doc="Pointing orientation data from PEF",
        storageClass="PredictedObservation",
        dimensions=["instrument", "observation"],
        isCalibration=False
    )
    initial_distortion_table = cT.Input(
        name="initial_distortion_table",
        doc="Initial data to fit distortion from ray-tracing.",
        storageClass="InitDistTable",
        dimensions=["instrument"],
        isCalibration=True
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after Estimate Coarse Astrometry.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "exposure"],
    )


class EstimateCoarseAstrometryConfig(pipeBase.PipelineTaskConfig,
                                     pipelineConnections=EstimateCoarseAstrometryConnections):
    """Configuration for EstimateCoarseAstrometry task."""

    naif_id = Field(
        doc="The NAIF ID of the instrument or spacecraft.",
        dtype=int,
        default=-163000)  # WISE NAIF ID

    spacecraft_clock_id = Field(
        doc="The spacecraft clock ID.",
        dtype=int,
        default=-163)  # WISE spacecraft clock ID

    tolerance = Field(
        "The time tolerance in seconds.",
        dtype=float,
        default=1)

    reference_frame = Field(
        doc="The reference frame for pointing and velocity.",
        dtype=str,
        default="J2000")

    apply_cant_angle = Field(
        doc="Apply the cant angle correction to quaternions.",
        dtype=bool,
        default=False)

    t_step = Field(
        doc="Query interval in seconds for pointing information from C-kernel.",
        dtype=int,
        default=1)

    minimum_pointings_threshold = Field(
        doc="The minimum number of pointings required to estimate astrometry.",
        dtype=int,
        default=1)

    pix_scale = Field(
        doc="SPHEREx pixel scale.",
        dtype=float,
        default=6.15)

    only_ecl_in_header = Field(
        doc="Only output ECL values from survey plan in the image header.",
        dtype=bool,
        default=True)

    use_coarse_correction = Field(
        doc="If True, apply a hard-coded pointing correction (roll + shift) ",
        dtype=bool,
        default=True
    )


class EstimateCoarseAstrometryTask(pipeBase.PipelineTask):
    """Implementation of L1 Estimate Coarse Astrometry.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-003

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "estimate-coarse-astrometry"
    ConfigClass = EstimateCoarseAstrometryConfig
    full_name = "Estimate Coarse Astrometry"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-003 v2.4"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self,
            spximg: SPHERExImage,
            planned_obs: PlannedObservation,
            initial_distortion_table: Table,
            override_wcs: WCS = None) -> pipeBase.Struct:
        """Create WCS and jitter table from PEF pointings.

        Parameters
        ----------
        spximg : `~spx_data.SPHERExImage`
            The image used to build WCS and add jitter metadata.
        planned_obs : `PlannedObservation`
            Pointing information from PEF.
        initial_distortion_table : `~astropy.table.Table`
            Initial distortion table from ray-tracing.
        override_wcs : `~astropy.wcs.WCS`
            Overrides the output WCS in the image with the given WCS.
            This is for handling with astrometry.net solution.

        Returns
        -------
        `Output`
            Object containing image, wcs header, and jitter table.

        Raises
        ------
        ValueError
            Raised if no pointings are found within time range.

        """
        spximg_out = copy.copy(spximg)
        array_id = spximg.getArrayId()
        rp = ReadoutParameters.from_metadata(spximg.meta)

        if override_wcs is not None:
            # The wcs may be overriden by solution from
            # astrometry.net if needed.
            spximg_out.wcs = override_wcs
            wcs_header = override_wcs.to_header(relax=True)
            wcs_initial_with_sip = override_wcs.wcs.sip
            return pipeBase.Struct(spximg_out=spximg_out,
                                   wcs_header=wcs_header,
                                   wcs_initial_with_sip=wcs_initial_with_sip,
                                   jitter_table=None)

        if self.config.use_coarse_correction:
            self.log.info(f"Attempting on-the-fly coarse correction for Det {array_id}.")

            # Get Slew Group Key (small slew counter)
            sps_id = spximg.header["OBSID"]
            obs_from_sps = ObservationId(sps_id)

            # Build the final corrected WCS for this detector
            crpix = (rp.Jactive + 1)/2, (rp.Kactive + 1)/2,
            linear_wcs = get_corrected_linear_wcs(planned_obs, obs_from_sps, array_id, crpix)

        else:
            self.log.info(f"No systematic correction appplied to linear WCS for Det {array_id}.")
            # Get boresight planned pointing from quaternion
            quat_dict = planned_obs.quaternions
            quat = np.array([
                quat_dict["q1"],
                quat_dict["q2"],
                quat_dict["q3"],
                quat_dict["q4"]
            ], dtype=np.float64)
            planned_ra, planned_dec, planned_pa = get_ra_dec_lon_lat_from_quat(quat)
            linear_wcs = get_initial_linear_wcs(planned_ra, planned_dec, array_id, planned_pa)

        # Obtain header info from planned observation
        ra_eq = planned_obs.ra.value
        dec_eq = planned_obs.dec.value
        pa_eq = roll = planned_obs.pa_eq.value
        print(f"ra = {ra_eq}, dec = {dec_eq}, roll = {roll}")
        lon_ecl = planned_obs.lon_ecl
        lat_ecl = planned_obs.lat_ecl
        pa_ecl = planned_obs.pa_ecl

        logging.info(linear_wcs.to_header(relax=True))
        wcs_header = linear_wcs.to_header(relax=True)

        crpix_x = linear_wcs.wcs.crpix[0]
        crpix_y = linear_wcs.wcs.crpix[1]
        center_x = (rp.Jactive + 1) / 2
        center_y = (rp.Kactive + 1) / 2

        if crpix_x != center_x and crpix_y != center_y:
            raise ValueError(
                "WCS CRPIX is not in center of array. "
                f"center:{center_x} {center_y}"
                f"CRPIX:{crpix_x} {crpix_y}"
            )

        # Initial coarse linear WCS
        spximg_out.wcs = copy.deepcopy(linear_wcs)

        # Shift WCS to Level-1 format
        spximg_out.wcs.wcs.crpix[0] += (rp.Mphan * rp.nchannel + rp.jref/2)
        spximg_out.wcs.wcs.crpix[1] += (rp.kref/2)

        # Lat, Lon, PA in header.
        spximg_out.setFreeFormatHeaderCard(
            "SPS_ELON", lon_ecl, 6,
            "[deg] Ecliptic longitude of planned telescope boresight")
        spximg_out.setFreeFormatHeaderCard(
            "SPS_ELAT", lat_ecl, 6,
            "[deg] Ecliptic latitude of planned telescope boresight")
        pa_ecl_std = pa_ecl % 360
        spximg_out.setFreeFormatHeaderCard(
            "SPS_EPA", pa_ecl_std, 6,
            "[deg] PA of FP y axis at planned boresight, E of ecl N")

        # Equatorial Ra, Dec, PA in header
        if not self.config.only_ecl_in_header:
            spximg_out.setFreeFormatHeaderCard(
                "SPS_RA", ra_eq, 6,
                "[deg] Equatorial longitude of planned telescope boresight")
            spximg_out.setFreeFormatHeaderCard(
                "SPS_DEC", dec_eq, 6,
                "[deg] Ecliptic latitude of planned telescope boresight")
            spximg_out.header.comments['DEC_EQ'] = "DEC_EQ: from survey plan"
            spximg_out.setFreeFormatHeaderCard(
                "SPS_EQPA", pa_eq, 6,
                "[deg] PA of FP y axis at planned boresight, E of equ N"
            )

        spximg_out.header["NAXIS1"] = rp.Kframe  # 2176
        spximg_out.header["NAXIS2"] = rp.Jframe  # 2072

        return pipeBase.Struct(spximg_out=spximg_out,
                               wcs_header=wcs_header)
