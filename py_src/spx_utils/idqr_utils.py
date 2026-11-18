__all__ = ["get_idqr"]

# Standard library imports
import logging

# Related third party imports
from astropy.coordinates import Angle
import astropy.units as u
import numpy as np

# Local application/library specific imports
from spx_data import ObservationId, SPHERExImage, SPSFlag
from spx_data.idqr import IDQR, IdqrObservation, IdqrImage, PointingSolution, DataQualityIndicators, Timestamp
from spx_utils.coarse_astro_utils import calculate_pa_from_pc


def get_idqr(spximgs: SPHERExImage | list[SPHERExImage]) -> IDQR:
    """Create an IDQR from one or more SPHEREx images.

    Parameters
    ----------
    spximgs : `SPHERExImage` or `list[SPHERExImage]`
        A SPHERExImage or a list of SPHERExImages
        containing WCS and header data.

    Returns
    -------
    idqr : `IDQR`
        The IDQR with observations, pointing, and data quality details.

    """
    # Ensure input is always a list for uniform processing
    if not isinstance(spximgs, list):
        spximgs = [spximgs]

    # Dictionary to hold all observations for the IDQR instance
    observations = {}

    # Process each image
    for spximg in spximgs:
        # Extract RA, DEC, and Position Angle from the image
        ra = spximg.wcs.wcs.crval[0]
        dec = spximg.wcs.wcs.crval[1]
        try:
            pa_degrees = spximg.header['PA_ECL']
        except KeyError:
            pa_degrees = calculate_pa_from_pc(spximg.wcs)

        position_angle = Angle(pa_degrees, unit=u.deg)

        # Create the pointing solution object
        pointing_solution = PointingSolution(
            right_ascension=ra,
            declination=dec,
            position_angle=position_angle
        )

        median_psf_size = spximg.header.get("PSF_FWHM", 0.0)
        if median_psf_size == 0.0:
            logging.warning("Median PSF FWHM not found, setting to zero.")
        # Create data quality indicators
        data_quality_indicators = DataQualityIndicators(
            median_psf_size=median_psf_size
        )

        # The transient check has been moved to the ODIR generation.
        # The status here is defaulted to "good".
        # Do we want to replace this with some other status check?
        status_flag = "good"

        # Create an Image instance
        image = IdqrImage(
            status=status_flag,
            pointing_solution=pointing_solution,
            data_quality_indicators=data_quality_indicators
        )

        # Extract observation start and end times
        start = spximg.getExposureStartTime()
        end = spximg.getExposureEndTime()

        # Define the observation timestamp
        timestamp = Timestamp.from_time(start, end)

        # Get detector ID
        detector = spximg.getArrayId()

        # Generate the observation ID
        spx_obsid = spximg.header["OBSID"]
        obs_id = ObservationId(spx_obsid)
        sps_id = obs_id.sps_id

        # SPS_TARGET - get integer target ID for IDQR report
        # First try SPS_TARGET_PEF (new format),
        # then fall back to SPS_TARGET if integer (old format)
        target_id = spximg.header.get("SPS_TARGET_PEF")
        if target_id is None:
            sps_target = spximg.header.get("SPS_TARGET")
            if isinstance(sps_target, int):
                target_id = sps_target
            else:
                target_id = 0
        if target_id == 0:
            logging.warning("No SPS target ID found. This is probably a simulated level1 image.")

        # SPS_FLAGS
        sps_flag = spximg.header.get("SPS_FLAGS", "")
        if not sps_flag:
            logging.warning("No SPS flags found. This is probably a simulated level1 image.")
        sps_flag_int = SPSFlag.from_string(sps_flag)

        # Create an Observation instance
        if (sps_flag_int & SPSFlag.ALL_SKY) and not (SPSFlag.is_saa(sps_flag_int)) and (status_flag == "bad"):
            plan_indicator = False
        else:
            plan_indicator = True
        observation = IdqrObservation(survey_plan_indicator=plan_indicator,
                                      target_id=target_id,
                                      sps_flag=sps_flag,
                                      timestamp=timestamp,
                                      images={detector: image})

        # Generate the observation ID and add it to the observations dictionary
        spx_obsid = spximg.header["OBSID"]
        obs_id = ObservationId(spx_obsid)
        sps_id = obs_id.sps_id
        if sps_id not in observations:
            # Add observation to dict if it does not exist
            observations[sps_id] = observation
        else:
            # Merge the images dictionary for existing observations
            observations[sps_id].images.update(observation.images)

            # If observation is all-sky and 1+ images exceeded the transient
            # threshold, mark for re-observation
            imgs = observations[sps_id].images
            bad_img_msk = [(imgs[key].status == "bad") for key in list(imgs.keys())]
            if (
                (sps_flag_int & SPSFlag.ALL_SKY)
                and not (SPSFlag.is_saa(sps_flag_int))
                and np.any(bad_img_msk)
            ):
                observations[sps_id].survey_plan_indicator = False

    # Create a single IDQR instance with all the observations
    idqr = IDQR(observations=observations)

    return idqr
