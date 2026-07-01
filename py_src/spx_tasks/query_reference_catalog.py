__all__ = ["QueryReferenceCatalogConfig", "QueryReferenceCatalogTask"]

# Standard library imports
from dataclasses import dataclass
import logging
import re

# Related third party imports
import numpy as np
import astropy.units as u
from astropy.table import Table, Column, vstack
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS, Sip
from lsst.daf.butler import DeferredDatasetHandle
from lsst.utils.timer import timeMethod
from lsst.pex.config import Field, ListField
import lsst.pipe.base.connectionTypes as cT
import lsst.pipe.base as pipeBase

# Local application/library specific imports
from spx_butler import getSkyPixIds, instrument, REFCAT_SKYPIX_NAME
from spx_data import SPHERExImage, ReadoutParameters
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_utils import SourcePipelineClass as c
from spx_utils.refcat_utils import get_magnitudes, EmptySourceCatalogError
from spx_utils import log_dqa_result, get_corrected_linear_wcs, north_aligned_pa_from_wcs
from spx_data.observation_id import ObservationId
from spx_data import PlannedObservation
# ==============================================================================


# Columns that must remain float64 in the srccat output.
# Times are currently stored in metadata.
FLOAT64_PRESERVE_COLS = {
    "ra",      # final positions
    "dec",
    "ref_ra",  # reference-epoch positions
    "ref_dec",
}


def _downgrade_float_columns(tbl: Table,
                             keep_double: set[str] = FLOAT64_PRESERVE_COLS) -> None:
    """Downcast float64 columns to float32 except for a specified keep list.

    Parameters
    ----------
    tbl : `astropy.table.Table`
        Table to be modified in place.
    keep_double : `set[str]`
        Column names that must remain float64.
    """
    for name in tbl.colnames:
        col = tbl[name]
        # Only touch floating-point columns
        if not np.issubdtype(col.dtype, np.floating):
            continue

        # Skip columns we want to keep at double precision
        if name in keep_double:
            continue

        # Downcast to float32 in place
        tbl[name] = col.astype(np.float32)


def add_external_ids(input_tab: Table):
    """Construct external source ID strings and add them to the input table.

    Parameters
    ----------
    input_tab : `astropy.table.Table`
        Table of reference catalog sources, including columns for the source
        of merged coordinates and external survey IDs/designations.

    Returns
    -------
    None
    """
    coord_src = input_tab["coord_src"]

    # Set default value to represent LC stellar input catalog or SPLICES source
    ext_src_id = np.full(len(input_tab), "N/A", dtype="<U64")

    # Create masks and assign values for external surveys
    mask = (coord_src == 1)  # Gaia_DR3
    ext_src_id[mask] = ["Gaia-DR3 " + str(v) for v in input_tab["Gaia_DR3_source_id"][mask]]

    mask = (coord_src == 2)  # PS1_DR1_Stack
    ext_src_id[mask] = ["PS1-DR1 " + str(v) for v in input_tab["PS1_DR1_StackObject_objID"][mask]]

    mask = (coord_src == 3)  # CatWISE
    ext_src_id[mask] = ["CatWISE2020 " + str(v) for v in input_tab["CatWISE_source_id"][mask]]

    mask = (coord_src == 4)  # AllWISE
    ext_src_id[mask] = ["AllWISE-PSC " + str(v) for v in input_tab["AllWISE_designation"][mask]]

    mask = (coord_src == 5)  # 2MASS
    ext_src_id[mask] = ["2MASS-PSC " + str(v) for v in input_tab["2MASS_designation"][mask]]

    mask = (coord_src == 6)  # LegacySurvey
    ext_src_id[mask] = ["DESI-LS-DR9_10 " + str(v) for v in input_tab["LegacySurvey_uid"][mask]]

    input_tab["external_source_id"] = ext_src_id
    return


def add_shape_params(input_tab: Table):
    """Calculate a/b axis ratio and East of North position angle phi using DESI
    Legacy Survey ellipticity components and add them to the input table.

    Parameters
    ----------
    input_tab : `astropy.table.Table`
        Table of reference catalog sources, including columns for galaxy model
        ellipticity components.

    Returns
    -------
    None
    """
    # Set default value to represent LS point sources
    ab_ratio = np.full(len(input_tab), 0.0, dtype="float32")
    phi = np.full(len(input_tab), 0.0, dtype="float32")

    # Create masks and assign values
    mask = (input_tab["shape_r"] > 0.0)  # LS galaxies
    e = np.hypot(input_tab["shape_e1"][mask], input_tab["shape_e2"][mask])
    ab_ratio[mask] = (1 + e) / (1 - e)
    phi[mask] = np.rad2deg(np.arctan2(input_tab["shape_e2"][mask], input_tab["shape_e1"][mask]) / 2.)

    mask = (input_tab["shape_r"] == -9999.0)  # sources outside LS footprint
    ab_ratio[mask] = -9999.0
    phi[mask] = -9999.0

    input_tab["ab_ratio"] = ab_ratio
    input_tab["phi"] = phi
    return


class QueryReferenceCatalogConnections(pipeBase.PipelineTaskConnections,
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
    refcat = cT.PrerequisiteInput(
        name="refcat",
        doc="Reference catalog.",
        storageClass="AstropyTable",
        dimensions=(REFCAT_SKYPIX_NAME,),
        deferLoad=True,
        multiple=True,
        minimum=1,
        # lookupFunction=refcatLookup
    )
    initial_distortion_table = cT.PrerequisiteInput(
        name="initial_distortion_table",
        doc="Initial data to fit distortion from ray-tracing.",
        storageClass="InitDistTable",
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


class QueryReferenceCatalogConfig(pipeBase.PipelineTaskConfig,
                                  pipelineConnections=QueryReferenceCatalogConnections):
    pixmargin = Field(
        doc="Margin in pixels."
        "The sources will be added to the catalog table if they fall "
        "within the image with this margin.",
        dtype=float,
        default=300.0)

    id_col = Field(
        doc="Name of the Reference Catalog ID column.",
        dtype=str,
        default="SPHERExRefID"
    )

    sel_mag_thres = Field(
        doc="LS_z magnitude threshold for including sources into the output catalog."
            " The sources with this or higher magnitudes will be excluded.",
        dtype=float,
        default=23)

    sel_color_thres = Field(
        doc="z - W1 color lower threshold for including sources into the output catalog."
            " The sources with tis or lower color will be excluded.",
        dtype=float,
        default=1.5)

    ra_col = Field(
        doc="Name of the Right Ascension column.",
        dtype=str,
        default="ra")

    dec_col = Field(
        doc="Name of the Declination column.",
        dtype=str,
        default="dec")

    # "Gaia_DR3_source_id" and "LS_z" are only needed to verify photometry
    # selection
    extra_cols = ListField(
        doc="Names of additional columns to include in the output.",
        dtype=str,
        default=("Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID", "CatWISE_source_id",
                 "AllWISE_designation", "2MASS_designation", "ra_error", "dec_error", "coord_src", "pmra",
                 "pmdec", "parallax", "flag_brightneighbor", "dist_nearest", "spherex_class",
                 # "Gaia_DR3_source_id", "LS_z",
                 "PS1_y", "PS1_z", "WISE_W1", "WISE_W2", "Gaia_RP",
                 "PS1_y_error", "PS1_z_error", "WISE_W1_error", "WISE_W2_error", "Gaia_RP_error",
                 "sersic", "shape_r", "shape_e1", "shape_e2", "ref_epoch", "for_l3phot"))

    save_derived_inputs = Field(
        doc="Save columns used to calculate or construct derived columns. "
            "Default is False.",
        dtype=bool,
        default=False)

    expected_high_b = Field(
        doc="Expected number of high galactic latitude sources",
        dtype=float,
        default=0.5e6)  # 0.5 Million for |b| > 5 deg

    expected_low_b = Field(
        doc="Expected number of low galactic latitude sources",
        dtype=float,
        default=3e6)  # 3 Million for |b| < 5 deg

    tolerance_factor = Field(
        doc="Allowed tolerance for number of sources in low (high) galactic latitude",
        dtype=float,
        default=3.)

    debug = Field(
        doc="Printouts for debugging",
        dtype=bool,
        default=False)

    img_nx = Field(
        doc="Dummy L2 image nx.",
        dtype=int,
        default=2040)

    img_ny = Field(
        doc="Dummy L2 image ny.",
        dtype=int,
        default=2040)

    selection_cuts = Field(
        doc="If True, performs magnitude and color selection on the output catalog.",
        dtype=bool,
        default=True)

    allow_empty = Field(
        doc="If True, allows output catalog to be empty without raising an error.",
        dtype=bool,
        default=False)


@dataclass
class Output:
    catalog: Table


class QueryReferenceCatalogTask(pipeBase.PipelineTask):
    """Implementation of L2 Query Reference Catalog.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-018

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "query-reference-catalog"
    ConfigClass = QueryReferenceCatalogConfig
    full_name = "Query Reference Catalog"
    ssdc_pipe_level = 2
    ssdc_doc_tag = "SSDC-MS-018 v2.1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.log = logging.getLogger(self._DefaultName)

    @timeMethod
    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        """
        - Handles I/O with the Butler.
        - Fetches inputs, calls the `run` method for the core algorithm,
          and saves the outputs.
        """
        self.log.info("runQuantum: Fetching inputs from the Butler.")

        slewgroup_id = inputRefs.planned_obs[0].dataId.get("slewgroup")
        self.log.info(f"Running QueryReferenceCatalogTask for slewgroup={slewgroup_id}")

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
        healpixIDs: set[int] = set()
        wcs_list: list[WCS] = []
        dummy_img = SPHERExImage(
            data=np.zeros((self.config.img_ny, self.config.img_nx)),  # (ny, nx)
            wcs=None, unit=""
        )
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

            dummy_img.wcs = wcs_sip
            dummy_img.wcs.array_shape = (self.config.img_ny, self.config.img_nx)
            healpixIDs |= set(getSkyPixIds(dummy_img, npixMargin=int(self.config.pixmargin)))

        self.log.info(f"Identified {len(healpixIDs)} HEALPix IDs for the full focal plane.")

        refcat_shards_handles = []
        for ref in inputRefs.refcat:
            if ref.dataId.get(REFCAT_SKYPIX_NAME) in healpixIDs:
                handle = butlerQC.get(ref)
                refcat_shards_handles.append(handle)

        self.log.info(f"Loaded {len(refcat_shards_handles)} reference catalog shards handles into memory.")

        # Take the run/collection name from the first refcat shard (they must all match)
        first_handle = refcat_shards_handles[0]
        dsref = getattr(first_handle, "ref", None) or getattr(first_handle, "datasetRef", None)
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

        # Call the run method with in-memory data
        output: Output = self.run(wcs_list,
                                  refcat_shards_handles,
                                  dichroicpair,
                                  slewgroup_mjd_avg,
                                  refcat_run_name)

        # Save the outputs back to the Butler (N to 3)
        self.log.info("runQuantum: Saving outputs to the Butler.")
        if output.catalog:
            butlerQC.put(output.catalog, outputRefs.catalog)
            self.log.info(f"Successfully saved catalog for dichroic pair {dichroicpair}.")
        else:
            self.log.warning(f"No catalog was generated for dichroic pair {dichroicpair}. Skipping put.")

    def run(self,
            wcs_list: list[WCS],
            refcatShards: list[DeferredDatasetHandle],
            dichroicpair: int,
            slewgroup_mjd_avg: float,
            refcat_run_name: str | None = None) -> Output:
        """

        Parameters
        ----------
        wcs_list: `list` [`~astropy.wcs WCS`]
            List of WCS for quantum.
        refcatShards: `list` [`lsst.daf.butler.DeferredDatasetHandle`]
            List of the deferred dataset handles to the catalog shards.
            Table representing the shard can be obtained from the
            deferred dataset handle, using get method.

        Returns
        -------
        output: Output
            Output that contains `~spx_data.SPHERExImage` with the catalog
            layer.
        """

        # Pixel margin
        pixmargin = self.config.pixmargin
        rp = ReadoutParameters()

        # Get bounding box for all WCSes
        xyBox = xyBoundingBox(wcs_list, pixmargin, rp, dichroicpair)

        matchedtbl = None
        for shard in refcatShards:
            try:
                tbl = sourcesInXYBox(xyBox, shard, self.config, dichroicpair)
                if matchedtbl is None:
                    matchedtbl = tbl
                else:
                    matchedtbl = vstack([matchedtbl, tbl], join_type='exact')
            except ValueError as e:
                # Log error but continue processing
                self.log.error(f"Error processing reference catalog shard: {e}")
                continue

        if len(matchedtbl) == 0:
            if self.config.allow_empty:
                self.log.warning("No reference catalog sources found in bounding box.")
                return Output(catalog=None)
            else:
                raise EmptySourceCatalogError("No refcat sources were found and 'allow_empty' is False.")

        # Construct "derived" columns for photometry
        derived_inputs = ["Gaia_DR3_source_id", "LegacySurvey_uid", "PS1_DR1_StackObject_objID",
                          "CatWISE_source_id", "AllWISE_designation", "2MASS_designation", "shape_e1",
                          "shape_e2"]
        if all(col in matchedtbl.colnames for col in derived_inputs):
            add_external_ids(matchedtbl)
            add_shape_params(matchedtbl)
            if not self.config.save_derived_inputs:
                matchedtbl.remove_columns(derived_inputs)

        if all(col in matchedtbl.colnames for col in ["ra", "dec", "pmra", "pmdec", "parallax", "ref_epoch"]):
            # Check for stars with GAIA proper motions
            matchedtbl.rename_column("ra", "ref_ra")
            matchedtbl.rename_column("dec", "ref_dec")

            valid_pm_mask = ((matchedtbl["ref_epoch"] != -9999.0)
                             & (matchedtbl["pmra"] != -9999.0)
                             & (matchedtbl["pmdec"] != -9999.0))
            gaia_pm_idx = np.nonzero(valid_pm_mask)

            # If 1+ are present, compute updated positions
            if len(gaia_pm_idx[0]) > 0:
                ref_epochs = Time(matchedtbl["ref_epoch"][gaia_pm_idx], format="jyear")
                target_epoch = Time(slewgroup_mjd_avg, format="mjd")
                gaia_star_coords = SkyCoord(
                    ra=matchedtbl["ref_ra"][gaia_pm_idx],
                    dec=matchedtbl["ref_dec"][gaia_pm_idx],
                    pm_ra_cosdec=matchedtbl["pmra"][gaia_pm_idx],
                    pm_dec=matchedtbl["pmdec"][gaia_pm_idx],
                    frame="icrs",
                    obstime=ref_epochs,
                )
                updated_coords = gaia_star_coords.apply_space_motion(new_obstime=target_epoch)

                if self.config.debug is True:
                    self.log.info("\nApplying Proper Motion Corrections")
                    # Check for a source ID column; use row idx if not found
                    id_col_name = "Gaia_DR3_source_id" if "Gaia_DR3_source_id" in matchedtbl.colnames \
                        else None

                    # Loop through each corrected source and print its details
                    for i in range(len(updated_coords)):
                        # Get the absolute idx of the source in the main table
                        source_table_idx = gaia_pm_idx[0][i]

                        source_id = (matchedtbl[id_col_name][source_table_idx]
                                     if id_col_name else f"Row {source_table_idx}")

                        # Original and new positions in degrees
                        orig_ra, orig_dec = gaia_star_coords.ra[i].deg, gaia_star_coords.dec[i].deg
                        new_ra, new_dec = updated_coords.ra[i].deg, updated_coords.dec[i].deg

                        # Calculate correction in arcseconds for printing
                        correction_ra_arcsec = (new_ra - orig_ra) * 3600.0
                        correction_dec_arcsec = (new_dec - orig_dec) * 3600.0

                        self.log.info(f"Source: {source_id}")
                        self.log.info(f"  Original Pos: RA={orig_ra:10.6f}, Dec={orig_dec:10.6f}")
                        self.log.info(f"  Updated Pos:  RA={new_ra:10.6f}, Dec={new_dec:10.6f}")
                        self.log.info(f"  Correction:   "
                                      f"dRA={correction_ra_arcsec:+.4f}\", "
                                      f"dDec={correction_dec_arcsec:+.4f}\"")

                    self.log.info(f"Total sources corrected: {len(updated_coords)}")
            else:
                self.log.info("No sources with valid Gaia proper motions found for correction.")

            # Update coordinates, report target epoch, and remove ref. epoch
            matchedtbl["ra"] = matchedtbl["ref_ra"].copy()
            matchedtbl["dec"] = matchedtbl["ref_dec"].copy()
            if len(gaia_pm_idx[0]) > 0:
                matchedtbl["ra"][gaia_pm_idx] = updated_coords.ra
                matchedtbl["dec"][gaia_pm_idx] = updated_coords.dec
            matchedtbl.meta["pipeline_epoch_mjd"] = slewgroup_mjd_avg.item()
            matchedtbl.remove_column("ref_epoch")

        # DQA perform source count consistency check
        _ = dqa_check_source_count(matchedtbl)
        self.metadata.add("numSources", len(matchedtbl))

        # Downcast float columns to float32 where safe to do so
        # call ``_downgrade_float_columns`` after all proper-motion corrections,
        # derived columns, and DQA checks, so that calculations use higher precision
        # DEBUG: print column dtypes before downcast
        if self.config.debug:
            col_info_before = {name: str(matchedtbl[name].dtype) for name in matchedtbl.colnames}
            self.log.info(f"srccat BEFORE downcast: {col_info_before}")

        _downgrade_float_columns(matchedtbl)

        # DEBUG: print column dtypes after downcast
        if self.config.debug:
            col_info_after = {name: str(matchedtbl[name].dtype) for name in matchedtbl.colnames}
            self.log.info(f"srccat AFTER downcast: {col_info_after}")

        # Add file-level Parquet metadata (to be written by the formatter)
        v = _parse_refcat_version(refcat_run_name) if refcat_run_name else None
        user_meta = {
            "schema_version": "1.0",
            "producer": "SPHEREx query_reference_catalog",
            "refcat_run": refcat_run_name or "",
            "refcat_version": f"SPHEREx_RefCat_{v}" if v else "",
            "version": f"spherex-pipelines {SSDC_PIPELINES_VERSION}",
            "created_utc": Time.now().isot,
        }
        # Put into Table.meta so our formatter can pick it up
        matchedtbl.meta["user_metadata"] = user_meta
        self.log.info("Will embed Parquet footer user_metadata: %s", user_meta)

        return Output(catalog=matchedtbl)

    def _get_detector_wcs(self, planned_obs: PlannedObservation,
                          obs_from_sps: ObservationId,
                          array_id: int,
                          sip: Sip = None) -> WCS:
        """
        Build a local WCS (CRPIX centered) for a dichroic detector,
        and then add SIP distortion.
        Dichroicpairs are 1=left, 2=center, 3=right.
        """
        wcs = get_corrected_linear_wcs(planned_obs, obs_from_sps, array_id)
        wcs.sip = sip
        return wcs


def _parse_refcat_version(run_name: str) -> str | None:
    """Parse refcat version tag from a run/collection name.

    Parameters
    ----------
    run_name : `str`
        Butler run/collection name (or any string) that may contain a
        reference-catalog version token.

    Returns
    -------
    `str` or `None`
        Normalized version string beginning with ``"v"`` (e.g., ``"v0.73"``),
        or ``None`` if no version token is recognized.
    """
    # Match refcat_v073 -> v0.73
    m = re.search(r'refcat_v(\d{3})', run_name)
    if m:
        return f"v{int(m.group(1))/100:.2f}"

    # Or refcat_v0.73 / refcat_v1.02 -> v0.73 / v1.02
    m = re.search(r'refcat_(v\d+(?:\.\d+)?)', run_name)
    if m:
        return m.group(1)

    return None


def dqa_check_source_count(catalog: Table) -> dict:
    """Check if the number of sources is consistent with expected galactic
    latitudes.

    Parameters
    ----------
    catalog : `astropy.table.Table`
        The matched reference catalog table.

    Returns
    -------
    result_qa : `dict`
        Dictionary containing DQA result and description.
    """
    from spx_tasks.query_reference_catalog import QueryReferenceCatalogConfig
    config = QueryReferenceCatalogConfig()

    coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], unit='deg', frame='icrs')
    gal_b = np.abs(coords.galactic.b.degree)

    b_min = np.min(gal_b)
    b_max = np.max(gal_b)
    n_sources = len(catalog)
    tolerance = config.tolerance_factor

    if b_min > 5:
        # All sources are in high-|b|
        expected = config.expected_high_b
        label = "high-|b|"
    elif b_max <= 5:
        # All sources are in low-|b|
        expected = config.expected_low_b
        label = "low-|b|"
    else:
        # Mixed case: spans both low and high-|b|
        result = "FAIL"
        description = (
            f"Galactic latitude spans both regimes (|b| range: {b_min:.2f}–{b_max:.2f} deg). "
            f"Cannot reliably assess source count expectation. Total sources: {n_sources}."
            f"gal_b = {gal_b}"
        )
        result_qa = {"result": result, "description": description}
        log_dqa_result("qrc dqa", result_qa)
        return result_qa

    # Proceed with source count check
    lower = expected / tolerance
    upper = expected * tolerance
    passed = lower <= n_sources <= upper

    result = "PASS" if passed else "FAIL"
    description = (
        f"{label.capitalize()} field (|b| range: {b_min:.2f}–{b_max:.2f} deg): "
        f"{n_sources} sources found. Expected range: [{lower:.0f}, {upper:.0f}]."
        f"gal_b = {gal_b}"
    )
    result_qa = {"result": result, "description": description}
    log_dqa_result("qrc dqa", result_qa)
    return result_qa


def _mean_sky_center(coords: SkyCoord) -> SkyCoord:
    """
    Compute a robust mean sky position from many RA/Dec points, avoiding RA-wrap.

    Parameters
    ----------
    coords : `astropy.coordinates.SkyCoord`
        Array-like sky coordinates (ICRS) from which to compute the mean.

    Returns
    -------
    `SkyCoord`
        The mean sky position in ICRS.

    Notes
    -----
    The mean is taken in 3D by converting each input coordinate to a unit vector
    on the celestial sphere, averaging those vectors, and converting the result
    back to ICRS RA/Dec. This prevents errors near the 0°/360° RA boundary.
    """
    # robust mean on unit vectors to avoid RA wrap issues
    x = np.cos(coords.ra.radian) * np.cos(coords.dec.radian)
    y = np.sin(coords.ra.radian) * np.cos(coords.dec.radian)
    z = np.sin(coords.dec.radian)
    vx = np.mean(x)
    vy = np.mean(y)
    vz = np.mean(z)
    r = np.hypot(vx, vy)
    ra = np.arctan2(vy, vx) * u.rad
    dec = np.arctan2(vz, r) * u.rad
    return SkyCoord(ra=ra.to(u.deg), dec=dec.to(u.deg), frame="icrs")


def _circ_mean_deg(angs) -> float:
    """Circular mean of angles given in degrees.

    Parameters
    ----------
    angs : array-like
        Angles in degrees.

    Returns
    -------
    `float`
        Circular mean angle in degrees. The returned value lies in the
        interval (-180°, 180]; apply ``% 360`` if you prefer [0°, 360).

    Notes
    -----
    Computes the mean direction on the circle by averaging sine/cosine
    components and converting back with ``atan2``.
    """
    a = np.radians(angs)
    return np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))


def _footprint_world(w, nx=2040, ny=2040, nedge=16) -> SkyCoord:
    """Sample the detector border and return its footprint as sky coordinates.

    Parameters
    ----------
    w : `astropy.wcs.WCS`
        WCS describing the detector/image.
    nx : `int`, optional
        Number of pixels in the x (column) dimension. Default is 2040.
    ny : `int`, optional
        Number of pixels in the y (row) dimension. Default is 2040.
    nedge : `int`, optional
        Number of samples per edge. Default is 16.

    Returns
    -------
    `astropy.coordinates.SkyCoord`
        Sky coordinates (ICRS) of the sampled border points.

    Notes
    -----
    The border is sampled more densely than just the four corners:
    ``nedge`` points are taken along each edge (bottom, top, left, right)
    using the Astropy WCS low-level 0-based pixel convention
    (pixel centers at *.5).
    """
    # sample the border more finely than 4 corners
    xs = np.linspace(-0.5, nx-0.5, nedge)
    ys = np.linspace(-0.5, ny-0.5, nedge)
    border = []
    for x in xs:  # bottom & top
        border.append((x, -0.5))
        border.append((x, ny-0.5))
    for y in ys:  # left & right
        border.append((-0.5, y))
        border.append((nx-0.5, y))
    px, py = np.array(border).T
    ra, dec = w.all_pix2world(px, py, 0)
    return SkyCoord(ra*u.deg, dec*u.deg, frame="icrs")


def _bbox_world_for_detector(w_det: WCS, rp, pixmargin: float) -> tuple[SkyCoord, float]:
    """Build a symmetric, margin-inflated bounding box for a single detector and
       return it as a sky polygon, along with the detector's north-aligned PA.

    Parameters
    ----------
    w_det : `astropy.wcs.WCS`
        WCS for this detector. It is treated as the reference plane in which
        the symmetric bounding box is computed.
    rp : `ReadoutParameters`
        Instrument readout geometry. ``rp.Kactive`` and ``rp.Jactive`` define
        the detector's active pixel dimensions and are used when sampling the
        footprint border.
    pixmargin : `float`
        Margin (in pixels) added to the symmetric bounding box in
        the detector's pixel frame.

    Returns
    -------
    poly_sky : `astropy.coordinates.SkyCoord`
        A 5-vertex, closed sky polygon (RA/Dec, ICRS) corresponding to the
        detector's margin-inflated symmetric bounding box.
    pa_north_aligned : `float`
        The detector's north-aligned position angle (degrees, 0-360),
        i.e., the direction of the image +y axis measured east of celestial
        north at the detector's center.
    """
    # Use the detector’s WCS as the plane
    w_ref = w_det

    world = _footprint_world(w_det, nx=rp.Kactive, ny=rp.Jactive, nedge=24)
    ix, iy = w_ref.all_world2pix(world.ra.deg, world.dec.deg, 0)

    cx, cy = np.median(ix), np.median(iy)
    rx = np.max(np.abs(ix - cx))
    ry = np.max(np.abs(iy - cy))
    xmin, xmax = cx - rx - pixmargin, cx + rx + pixmargin
    ymin, ymax = cy - ry - pixmargin, cy + ry + pixmargin

    # Get a sky polygon back
    px = np.array([xmin, xmax, xmax, xmin, xmin])
    py = np.array([ymin, ymin, ymax, ymax, ymin])
    ra_cnr, dec_cnr = w_ref.all_pix2world(px, py, 0)
    return SkyCoord(ra_cnr*u.deg, dec_cnr*u.deg, frame="icrs"), north_aligned_pa_from_wcs(w_det)


def xyBoundingBox(wcs_list: list[WCS],
                  pixmargin: float,
                  rp: ReadoutParameters,
                  dichroicpair: int) -> tuple[WCS, float, float, float, float]:
    """Compute a rectangular pixel bounding box in a single *representative*
    WCS that encloses all detectors in the slewgroup / dichroic pair.

    We choose one WCS from `wcs_list` as the reference plane (currently the
    middle one) and project the footprints of all other WCSes into that plane.
    The returned xmin/xmax/ymin/ymax are defined in this reference WCS.
    """
    if not wcs_list:
        raise ValueError("xyBoundingBox: empty wcs_list")

    # 1. Pick a representative WCS – here we just take the middle one.
    ref_idx = len(wcs_list) // 2
    w_ref = wcs_list[ref_idx]

    # 2. Collect all sky-footprint points from every WCS
    ra_all = []
    dec_all = []
    for w in wcs_list:
        # Sample the detector border in that WCS
        world = _footprint_world(w, nx=rp.Kactive, ny=rp.Jactive, nedge=24)
        ra_all.append(world.ra.deg)
        dec_all.append(world.dec.deg)

    ra_all = np.concatenate(ra_all)
    dec_all = np.concatenate(dec_all)
    all_world = SkyCoord(ra_all * u.deg, dec_all * u.deg, frame="icrs")

    # 3. Project all footprints into the reference WCS
    ix, iy = w_ref.all_world2pix(all_world.ra.deg, all_world.dec.deg, 0)

    # 4. Take min/max and expand by pixel margin
    xmin = np.min(ix) - pixmargin
    xmax = np.max(ix) + pixmargin
    ymin = np.min(iy) - pixmargin
    ymax = np.max(iy) + pixmargin

    return w_ref, xmin, xmax, ymin, ymax


def sourcesInXYBox(xyBox,
                   refcatShard: DeferredDatasetHandle,
                   config: QueryReferenceCatalogConfig,
                   dichroicpair: int) -> Table:
    """Get refcat sources within the bounding box.

    Parameters
    ----------
    xyBox: tuple[WCS, float, float, float, float]
        WCS and zero-based boundaries: xmin, xmax, ymin, ymax
    refcatShard: `~lsst.daf.butler.DeferredDatasetHandle`
    config: `QueryReferenceCatalogConfig`
        Configuration parameters.
    dichroicpair: `int`
        Dichroic pair ID.

    Returns
    -------
    table: astropy table
    """

    tbl = refcatShard.get()

    id_col = config.id_col
    ra_col = config.ra_col
    dec_col = config.dec_col

    ra = tbl[ra_col].data
    dec = tbl[dec_col].data
    keep = np.full((len(ra),), False)

    # find which sources fall into bounding box
    if xyBox is not None:
        wcs, xmin, xmax, ymin, ymax = xyBox
        # image coordinates of the catalog sources
        coord = SkyCoord(ra, dec, frame="icrs", unit="deg")
        (ix, iy) = wcs.all_world2pix(coord.ra.deg, coord.dec.deg, 0)
        keep |= (ix >= xmin) & (ix < xmax) & \
                (iy >= ymin) & (iy < ymax)

    # limit the sources to those that we want to photometer
    tbl.add_column(
        Column(data=np.full(shape=len(tbl), fill_value=0 if config.selection_cuts else 1),
               name="for_l3phot",
               dtype=np.uint8,
               description="Flag indicating if a source should be photometered during L3"))
    if config.selection_cuts:
        selected = do_select(tbl, config)
        keep &= selected

    c1 = Column(tbl[id_col].data[keep], name=id_col, dtype=tbl[id_col].dtype, unit=tbl[id_col].unit)
    c2 = Column(ra[keep], name='ra', dtype=tbl[ra_col].dtype, unit=tbl[ra_col].unit)
    c3 = Column(dec[keep], name='dec', dtype=tbl[dec_col].dtype, unit=tbl[dec_col].unit)

    cols = [c1, c2, c3]

    # Retrieve magnitudes
    mag_swir, mag_mwir = get_magnitudes(dichroicpair, tbl, keep)

    # Get detector IDs in this dichroic pair
    det1, det2 = instrument.from_dichroicpair(dichroicpair)

    # Rename magnitude columns appropriately; store as float32 to reduce size
    mag_swir = Column(mag_swir.value.astype(np.float32),
                      name=f"mag_det{det1}", dtype=np.float32)
    mag_mwir = Column(mag_mwir.value.astype(np.float32),
                      name=f"mag_det{det2}", dtype=np.float32)

    cols.append(mag_swir)
    cols.append(mag_mwir)

    # since catalog is shared between images in the group
    # pixel coordinates of the sources must be calculated where they are needed
    # c3 = Column(ix[keep], name='ix', dtype='float32')
    # c4 = Column(iy[keep], name='iy', dtype='float32')
    # cols = [c1, c2, c3, c4]

    # discard duplicates, preserve order
    extra_col_names = list(dict.fromkeys(config.extra_cols).keys())
    if id_col in extra_col_names:
        extra_col_names.remove(id_col)
    if ra_col in extra_col_names:
        extra_col_names.remove(ra_col)
    if dec_col in extra_col_names:
        extra_col_names.remove(dec_col)
    # preserve the order of extra columns
    for cn in extra_col_names:
        col = Column(tbl[cn].data[keep], name=cn, dtype=tbl[cn].dtype, unit=tbl[cn].unit)
        cols.append(col)

    return Table(cols)


def do_select(tbl: Table, config: QueryReferenceCatalogConfig) -> np.array:
    """Return numpy array of booleans indicating whether a table row should be
    selected.

    Eventually, we might want to have a flag that tells us which sources
    to photometer. For now, we will use this function filter the ref cat table.

    A source should be selected if it meets the following criteria:
    (ASTROMETRIC OR PSF OR PHOTOMETRIC OR CALSPEC OR BRIGHT OR COSMOLOGY OR
       ICES OR SECONDARY OR STELLAR OR CLUSTER
    OR IN_GAIA
    OR LS_z_mag < 23 (AB)
    OR ((z-W1) > 1.5)) AND ~(NO_ICES)

    Note the selection on bits (which may or may not be set), LS_z_mag
    brightness, and one color, if it can be determined. Also note the "IN_GAIA"
    condition. Any source with a valid GAIA ID should be included.

    Details on the mag / color cuts:
    LS_z_mag = -2.5*log10(LS_z)+23.9 (If LS_z exists)
    z - W1 = -2.5 * log10(LS_z / WISE_W1) (If both LS_z and WISE_W1 exist)

    Parameters
    ----------
    tbl : `~astropy.table.Table
        Input table with one row per source.
    config : `QueryReferenceCatalogConfig`
        Configuration parameters.

    Returns
    -------
    result : `~numpy.ndarray`
        1-d numpy array with the length matching the number of rows. ``True``
        value indicates that the row should be included into selection.
    """
    spherex_class_bitmask = (c.ASTROMETRIC | c.PSF | c.PHOTOMETRIC | c.CALSPEC | c.BRIGHT
                             | c.COSMOLOGY | c.ICES | c.SECONDARY | c.STELLAR | c.CLUSTER)

    def to_mag(v):
        result = np.full(v.shape, np.nan)
        positive_mask = v > 0
        result[positive_mask] = -2.5 * np.log10(v[positive_mask]) + 23.9
        return result

    def to_color(v1, v2):
        result = np.full(v1.shape, np.nan)
        positive_mask = (v1 > 0) & (v2 > 0)
        result[positive_mask] = -2.5 * np.log10(v1[positive_mask] / v2[positive_mask])
        return result

    def is_selected_by_mag_or_color(v1, v2):
        return (to_mag(v1) < config.sel_mag_thres) | (to_color(v1, v2) > config.sel_color_thres)

    selected = (tbl["Gaia_DR3_source_id"] >= 0) | (tbl['spherex_class'] & spherex_class_bitmask > 0)

    ls_z_missing = (tbl["LS_z"] < 0).all()
    if not ls_z_missing:
        selected |= is_selected_by_mag_or_color(tbl["LS_z"], tbl["WISE_W1"])

    not_no_ices = (tbl["spherex_class"] & c.NO_ICES == 0)
    selected &= not_no_ices

    # Set an initial flag to photometer all selected sources
    tbl["for_l3phot"][selected] = 1

    # Exclude galactic plane sources that are not in 2MASS or WISE
    g_lat = SkyCoord(tbl["ra"], tbl["dec"], unit="deg", frame="icrs").galactic.b.deg
    no_phot = ((np.abs(g_lat) < 12.)
               & (tbl["CatWISE_source_id"] == b"N/A")
               & (tbl["2MASS_designation"] == b"N/A")
               & ~ (tbl["spherex_class"] & spherex_class_bitmask > 0))
    tbl["for_l3phot"][no_phot] = 0
    return selected
