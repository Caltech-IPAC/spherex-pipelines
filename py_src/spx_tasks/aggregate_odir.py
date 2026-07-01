__all__ = ["AggregateOdirTaskConfig", "AggregateOdirTask"]

# Standard library imports

# Related third party imports
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.daf.butler import DeferredDatasetHandle

# Local application/library specific imports
from spx_data import SPSFlag
from spx_data.odir import ODIR


def merge_odirs(odir_list):
    """Merge multiple ODIR objects into a single ODIR."""
    merged_observations = {}

    for odir in odir_list:
        for obs_id, observation in odir.observations.items():
            sps_flag_int = SPSFlag.from_string(observation.sps_flag)
            if SPSFlag.is_saa(sps_flag_int):
                continue  # drop any SAA observations
            else:
                if obs_id not in merged_observations:
                    # Add the observation if it doesn't exist
                    merged_observations[obs_id] = observation
                else:
                    # Merge the images dictionary for existing observations
                    merged_observations[obs_id].images.update(observation.images)

                    # If observation is all-sky and 1+ images exceeded the
                    # transient threshold, mark for re-observation
                    imgs = merged_observations[obs_id].images
                    bad_img_msk = [(imgs[key].status == "bad") for key in list(imgs.keys())]
                    if (sps_flag_int & SPSFlag.ALL_SKY) and np.any(bad_img_msk):
                        merged_observations[obs_id].survey_plan_indicator = False

    return ODIR(observations=merged_observations)


class AggregateOdirConnections(pipeBase.PipelineTaskConnections,
                               dimensions=["instrument"],
                               defaultTemplates={}):
    odir_obs = cT.Input(
        name="odir_obs",
        doc="Per-observation ODIRs to aggregate.",
        storageClass="OdirObs",
        dimensions=["instrument", "exposure"],
        deferLoad=True,
        multiple=True
    )
    odir_agg = cT.Output(
        name="odir_agg",
        doc="ODIR for L1 images.",
        storageClass="OdirAgg",
        dimensions=["instrument"],
    )


class AggregateOdirTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=AggregateOdirConnections):
    pass


class AggregateOdirTask(pipeBase.PipelineTask):
    _DefaultName = "ODIR"
    ConfigClass = AggregateOdirTaskConfig
    full_name = "ODIR (light): ntrans reobservations"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "None"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, odir_obs: list[DeferredDatasetHandle]) -> pipeBase.Struct:
        odir_list = [handle.get() for handle in odir_obs]
        # Combine results to create one ODIR object
        odir = merge_odirs(odir_list)
        return pipeBase.Struct(odir_agg=odir)
