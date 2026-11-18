__all__ = ["QueryCustomCatalogConfig", "QueryCustomCatalogTask"]

# Standard library imports
from dataclasses import dataclass
import logging

# Related third party imports
from astropy.io.fits import HDUList
from astropy.table import Table
from astropy.time import Time
from astropy.wcs import WCS
from lsst.daf.butler import DeferredDatasetHandle
import lsst.pipe.base.connectionTypes as cT
import lsst.pipe.base as pipeBase

# Local application/library specific imports
from spx_data.observation_id import ObservationId
from spx_tasks import QueryReferenceCatalogConfig, QueryReferenceCatalogTask


class QueryCustomCatalogConnections(pipeBase.PipelineTaskConnections,
                                    dimensions=("instrument", "dichroicpair", "spsplan", "slewgroup"),
                                    defaultTemplates={}):
    planned_obs = cT.Input(
        name="planned_obs",
        doc="Planned observation metadata table. (corresponding to dichroic pair 2)",
        storageClass="PredictedObservation",
        dimensions=("instrument", "slewgroup", "slewgroup_counter", "spsplan", "observation"),
        deferLoad=True,
        multiple=True,
    )
    refcat_allsky = cT.PrerequisiteInput(
        name="refcat_allsky",
        doc="Custom input reference catalog.",
        storageClass="AstropyTable",
        dimensions=["instrument"],
        deferLoad=True,
        # lookupFunction=refcatLookup
    )
    initial_distortion_table = cT.PrerequisiteInput(
        name="initial_distortion_table",
        doc="Initial data to fit distortion from ray-tracing.",
        storageClass="InitDistTable",
        dimensions=["instrument"],
        isCalibration=True
    )
    cosmo_mask = cT.PrerequisiteInput(
        name="cosmo_mask",
        doc="List of HiPS8 tiles in cosmology selection mask.",
        storageClass="AstropyImage",
        dimensions=["instrument"],
        isCalibration=True
    )
    hi_density_mask = cT.PrerequisiteInput(
        name="hi_density_mask",
        doc="List of HiPS8 tiles in high-density selection mask.",
        storageClass="AstropyImage",
        dimensions=["instrument"],
        isCalibration=True
    )
    catalog = cT.Output(
        name="srccat",  # default dataset type for output
        doc="Output catalog with the sources overlapping the footprints "
            "of the images for the given slew group and detector pair",
        storageClass="AstropyParquet",
        dimensions=("instrument", "dichroicpair", "spsplan", "slewgroup")
    )


class QueryCustomCatalogConfig(QueryReferenceCatalogConfig,
                               pipelineConnections=QueryCustomCatalogConnections):
    """Configuration for QueryCustomCatalog task."""

    pass


@dataclass
class Output:
    catalog: Table


class QueryCustomCatalogTask(QueryReferenceCatalogTask):
    """Task to query a custom input reference catalog.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.

    """

    ConfigClass = QueryCustomCatalogConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self._DefaultName)

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """- Handles I/O with the Butler.
        - Fetches inputs, calls the `run` method for the core algorithm,
          and saves the outputs.
        """
        self.log.info("runQuantum: Fetching inputs from the Butler.")

        slewgroup_id = inputRefs.planned_obs[0].dataId.get("slewgroup")
        self.log.info(f"Running QueryCustomCatalogTask for slewgroup={slewgroup_id}")

        # Filter to unique observation IDs
        planned_obs_filtered = []
        seen_obsids = set()
        for ref in inputRefs.planned_obs:
            obsid = ref.dataId.get("observation")
            if obsid not in seen_obsids:
                seen_obsids.add(obsid)
                planned_obs_filtered.append(ref)

        # Initial distortion model for SIPs
        # butlerQC.get(inputRefs.initial_distortion_table)

        # Single output dataset ref (one catalog per quantum)
        catalog_ref = outputRefs.catalog
        if catalog_ref is None:
            raise RuntimeError("Expected an output 'catalog' ref, got None")
        # Pull dichroicpair directly
        dichroicpair = catalog_ref.dataId.get("dichroicpair")

        # Prepare accumulators used while processing planned_obs_filtered
        wcs_list: list[WCS] = []
        mid_sum = 0.0
        mid_count = 0

        # Now load each PlannedObservation
        for ref in planned_obs_filtered:
            po = butlerQC.get(ref).get()
            if not hasattr(po, "obs_id"):
                self.log.error(f"Expected PlannedObservation but got {type(po)} from {ref}")
                raise TypeError(f"planned_obs must be PlannedObservation, got {type(po)}")

            self.log.info(f"Planned obs in slewgroup: {po.obs_id}")
            # Get Slew Group Key (small slew counter)
            obs_from_sps = ObservationId(po.obs_id)
            self.log.info(f"Observation ID: {obs_from_sps}")

            # mid_time accumulation (no array needed)
            start_mjd = Time(po.timespan.begin).mjd
            stop_mjd = Time(po.timespan.end).mjd
            mid_sum += 0.5 * (start_mjd + stop_mjd)
            mid_count += 1

            # Build WCS and accumulate HEALPix coverage
            wcs_sip = self._get_detector_wcs(po, obs_from_sps, dichroicpair)
            wcs_list.append(wcs_sip)

        # Take the run/collection name from the input custom reference catalog
        refcat_allsky_handle = butlerQC.get(inputRefs.refcat_allsky)
        dsref = getattr(refcat_allsky_handle, "ref", None) \
            or getattr(refcat_allsky_handle, "datasetRef", None)
        if dsref is None:
            raise RuntimeError("Cannot access DatasetRef from DeferredDatasetHandle.")
        refcat_run_name = getattr(dsref.run, "name", str(dsref.run))
        self.log.info("run: %s", refcat_run_name)

        if not wcs_list:
            self.log.warning(f"No WCS generated for dichroic pair {dichroicpair}. Skipping.")

        # MJD average
        if mid_count == 0:
            raise RuntimeError("Cannot compute slewgroup_mjd_avg: no planned observations after filtering.")
        slewgroup_mjd_avg = mid_sum / mid_count

        # Selection HDU lists
        cosmo_hdul = butlerQC.get(inputRefs.cosmo_mask)
        hi_density_hdul = butlerQC.get(inputRefs.hi_density_mask)

        # Call the run method with in-memory data
        output: Output = self.run(wcs_list,
                                  refcat_allsky_handle,
                                  dichroicpair,
                                  slewgroup_mjd_avg,
                                  cosmo_hdul,
                                  hi_density_hdul,
                                  refcat_run_name)

        # Save the outputs back to the Butler (N to 3)
        self.log.info("runQuantum: Saving outputs to the Butler.")
        if output.catalog:
            butlerQC.put(output.catalog, outputRefs.catalog)
            self.log.info(f"Successfully saved catalog for dichroic pair {dichroicpair}.")
        else:
            self.log.warning(f"No catalog was generated for dichroic pair {dichroicpair}. Skipping put.")

    def run(self, wcs_list: list[WCS],
            refcat_allsky: DeferredDatasetHandle,
            dichroicpair: int,
            slewgroup_mjd_avg: float,
            cosmo_hdul: HDUList,
            hi_density_hdul: HDUList,
            refcat_run_name: str | None = None) -> Output:
        """Run the custom catalog query."""
        result = super().run(wcs_list=wcs_list, refcatShards=[refcat_allsky], dichroicpair=dichroicpair,
                             slewgroup_mjd_avg=slewgroup_mjd_avg, cosmo_hdul=cosmo_hdul,
                             hi_density_hdul=hi_density_hdul, refcat_run_name=refcat_run_name)
        return result
