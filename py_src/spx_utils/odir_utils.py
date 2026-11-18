__all__ = ["get_odir"]

# Standard library imports
import logging

# Related third party imports
import numpy as np

# Local application/library specific imports
from spx_data import Flag, ObservationId, SPHERExImage, SPSFlag
from spx_data.odir import ODIR, OdirObservation, OdirImage
from spx_data.idqr import Timestamp


def get_odir(spximgs: SPHERExImage | list[SPHERExImage], transient_threshold: int) -> ODIR:
    """Create an ODIR from one or more SPHEREx images.

    Parameters
    ----------
    spximgs : `SPHERExImage` or `list[SPHERExImage]`
        A SPHERExImage or a list of SPHERExImages.
    transient_threshold : `int`
        Maximum number of transient pixels allowed in all-sky images before
        triggering ODIR re-observe flag.

    Returns
    -------
    odir : `ODIR`
        The ODIR with observations and status details.

    Notes
    -----
    This function checks for the number of transient pixels and sets a "bad"
    status if a given threshold is exceeded, which can flag an observation
    for re-observation.

    """
    # Ensure input is always a list for uniform processing
    if not isinstance(spximgs, list):
        spximgs = [spximgs]

    # Dictionary to hold all observations for the ODIR instance
    observations = {}

    # Process each image
    for spximg in spximgs:
        # Check if the number of transient pixels is above the threshold
        n_transients = np.sum((spximg.flags & Flag.TRANSIENT) > 0)
        if n_transients > transient_threshold:
            status_flag = "bad"
        else:
            status_flag = "good"

        # Create a simplified OdirImage instance with only the status
        image = OdirImage(status=status_flag)

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

        # SPS_TARGET - get integer target ID for ODIR report
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

        # Determine if the observation counts toward the survey plan.
        # This is set to False if it's an all-sky observation, not in the SAA,
        # and has a "bad" status due to transients.
        if (sps_flag_int & SPSFlag.ALL_SKY) and not (SPSFlag.is_saa(sps_flag_int)) and (status_flag == "bad"):
            plan_indicator = False
        else:
            plan_indicator = True

        # Create an OdirObservation instance using the simplified models
        observation = OdirObservation(survey_plan_indicator=plan_indicator,
                                      target_id=target_id,
                                      sps_flag=sps_flag,
                                      timestamp=timestamp,
                                      images={detector: image})

        # Add or merge the observation into the main dictionary
        if sps_id not in observations:
            observations[sps_id] = observation
        else:
            # Merge the images dictionary for existing observations
            observations[sps_id].images.update(observation.images)

            # Re-evaluate the survey plan indicator for the merged observation.
            # If any image in an all-sky observation is bad, the whole thing is bad.
            imgs = observations[sps_id].images
            bad_img_msk = [(imgs[key].status == "bad") for key in list(imgs.keys())]
            if (
                (sps_flag_int & SPSFlag.ALL_SKY)
                and not (SPSFlag.is_saa(sps_flag_int))
                and np.any(bad_img_msk)
            ):
                observations[sps_id].survey_plan_indicator = False

    # Create the final ODIR instance
    odir = ODIR(observations=observations)

    return odir
