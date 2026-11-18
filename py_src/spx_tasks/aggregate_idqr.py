__all__ = ["AggregateIdqrTaskConfig", "AggregateIdqrTask"]

# Standard library imports

# Related third party imports
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from lsst.daf.butler import DeferredDatasetHandle

# Local application/library specific imports
from spx_data import SPSFlag
from spx_data.idqr import IDQR


def merge_idqrs(idqr_list):
    """Merge multiple IDQR objects into a single IDQR."""
    merged_observations = {}

    for idqr in idqr_list:
        for obs_id, observation in idqr.observations.items():
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

    return IDQR(observations=merged_observations)


class AggregateIdqrConnections(pipeBase.PipelineTaskConnections,
                               dimensions=["instrument"],
                               defaultTemplates={}):
    idqr_obs = cT.Input(
        name="idqr_obs",  # This matches L2bTask's output
        doc="Per-observation IDQRs to aggregate.",
        storageClass="IdqrObs",
        dimensions=["instrument", "exposure"],
        deferLoad=True,
        multiple=True
    )
    idqr_agg = cT.Output(
        name="idqr_agg",
        doc="IDQR for L2 images.",
        storageClass="IdqrAgg",
        dimensions=["instrument"],
    )


class AggregateIdqrTaskConfig(pipeBase.PipelineTaskConfig,
                              pipelineConnections=AggregateIdqrConnections):
    """Configuration for AggregateIdqrTask."""

    pass


class AggregateIdqrTask(pipeBase.PipelineTask):
    """Task to aggregate IDQR data across multiple observations."""

    _DefaultName = "pointing-solution"
    ConfigClass = AggregateIdqrTaskConfig
    full_name = "Pointing Solution"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "None"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, idqr_obs: list[DeferredDatasetHandle]) -> pipeBase.Struct:
        idqr_list = [handle.get() for handle in idqr_obs]
        # Combine results to create one IDQR object
        idqr = merge_idqrs(idqr_list)
        return pipeBase.Struct(idqr_agg=idqr)
