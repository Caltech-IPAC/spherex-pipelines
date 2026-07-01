__all__ = ["QueryReferenceCatalogConfig", "QueryReferenceCatalogTask"]

# Standard library imports
from dataclasses import dataclass
import logging

# Related third party imports
import numpy as np
from astropy.table import Table, Column, vstack
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.wcs import WCS
from lsst.daf.butler import (
    DeferredDatasetHandle,
    # inputs needed by refcatLookup
    # CollectionSearch,
    # DataCoordinate,
    # DatasetRef,
    # DatasetType,
    # Registry
)
from lsst.utils.timer import timeMethod
from lsst.pex.config import Field, ListField
import lsst.pipe.base.connectionTypes as cT
import lsst.pipe.base as pipeBase

# Local application/library specific imports
from spx_butler import getSkyPixIds, instrument, REFCAT_SKYPIX_NAME
from spx_data import SPHERExImage
from spx_utils import active_area_range, refcat_utils
from spx_utils import SourcePipelineClass as c
from spx_utils import log_dqa_result

# ==============================================================================


# class PrerequisiteLookupFunctionException(Exception):
#     """An exception class to clearly identifies when an unexpected situation
#     has occurred in the prerequisite input lookup functions.
#     """
#     pass


# def refcatLookup(datasetType: DatasetType,
#                  registry: Registry,
#                  quantumDataId: DataCoordinate,
#                  collections: CollectionSearch) -> Iterable[DatasetRef]:
#     """Return datasetRefs for reference catalog tiles overlapping with
#     exposures represented by quantumDataId.
#     """
#     pass


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
    images = cT.Input(
        name="level1",  # default dataset type for input
        doc="Input level 2images.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
        deferLoad=True,
        multiple=True
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
        default=250.0)

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
                 "sersic", "shape_r", "shape_e1", "shape_e2", "ref_epoch"))

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
        default=False
    )


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

        # quantum unit is (slewgroup, dichroicpair)
        # Both detectors in dichroic pair look at the same sky scene,
        # hence there will be two images per sky scene.
        # Keep only one image per dichroic pair.
        imageRefsFiltered = []
        obsids = []
        for ref in inputRefs.images:
            dataId = ref.dataId
            obsid = (dataId.get("observation"))
            if obsid not in obsids:
                obsids.append(obsid)
                imageRefsFiltered.append(ref)

        # collect healpix indexes for these images
        imageList = []
        healpixIDs = set()
        for ref in imageRefsFiltered:
            imageHandle: DeferredDatasetHandle = butlerQC.get(ref)
            image = imageHandle.get()
            imageList.append(image)
            healpixIDs |= set(getSkyPixIds(image, self.config.pixmargin))

        # get all healpix ids that cover images with margin
        # print("ALL HEALPix IDs", healpixIDs)

        # filter the input refcat references based on healpix id
        refcatShardsRefs = []
        for ref in inputRefs.refcat:
            if ref.dataId.get(REFCAT_SKYPIX_NAME) in healpixIDs:
                refcatShardsRefs.append(ref)

        refcatShards = [butlerQC.get(ref) for ref in refcatShardsRefs]
        output: Output = self.run(imageList, refcatShards)
        butlerQC.put(output.catalog, outputRefs.catalog)

    def run(self,
            imageList: list[SPHERExImage],
            refcatShards: list[DeferredDatasetHandle]) -> Output:
        """

        Parameters
        ----------
        imageList: `list` [`~spx_data.SPHERExImage`]
            List of SPHERExImages in L1 format.
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

        # Validate pixmargin
        pixmargin = self.config.pixmargin
        first_spximg = imageList[0]
        npixX, npixY = first_spximg.data.shape
        if abs(pixmargin) >= npixX / 2 or abs(pixmargin) >= npixY / 2:
            self.log.info(f"Pixel margin {pixmargin} exceeds half-image pixel size")

        # Get bounding box for all images
        xyBox = xyBoundingBox(imageList, pixmargin)

        # Confirm that every image in the imageList is from the dichroic pair.
        # Get dichroic pair ID.
        dichroicpair = instrument.to_dichroicpair(first_spximg.getArrayId())
        dichroicpair_ids = instrument.from_dichroicpair(dichroicpair)
        for i in range(1, len(imageList)):
            if imageList[i].getArrayId() not in dichroicpair_ids:
                raise ValueError(f"Image {i} in ImageList is not in the expected dichroic pair.")

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
            # Compute updated positions for stars with GAIA proper motions
            matchedtbl.rename_column("ra", "ref_ra")
            matchedtbl.rename_column("dec", "ref_dec")

            valid_pm_mask = ((matchedtbl["ref_epoch"] != -9999.0)
                             & (matchedtbl["pmra"] != -9999.0)
                             & (matchedtbl["pmdec"] != -9999.0))
            gaia_pm_idx = np.nonzero(valid_pm_mask)

            ref_epochs = Time(matchedtbl["ref_epoch"][gaia_pm_idx], format="jyear")
            slewgroup_mjd_avg = np.average([spximg.header["MJD-AVG"] for spximg in imageList])
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
                print("\nApplying Proper Motion Corrections")
                # Check for a source ID column, fall back to row index if not found
                id_col_name = "Gaia_DR3_source_id" if "Gaia_DR3_source_id" in matchedtbl.colnames else None

                # Loop through each corrected source and print its details
                for i in range(len(updated_coords)):
                    # Get the absolute index of the source in the main table
                    source_table_idx = gaia_pm_idx[0][i]

                    source_id = (matchedtbl[id_col_name][source_table_idx]
                                 if id_col_name else f"Row {source_table_idx}")

                    # Original and new positions in degrees
                    orig_ra, orig_dec = gaia_star_coords.ra[i].deg, gaia_star_coords.dec[i].deg
                    new_ra, new_dec = updated_coords.ra[i].deg, updated_coords.dec[i].deg

                    # Calculate correction in arcseconds for printing
                    correction_ra_arcsec = (new_ra - orig_ra) * 3600.0
                    correction_dec_arcsec = (new_dec - orig_dec) * 3600.0

                    print(f"Source: {source_id}")
                    print(f"  Original Pos: RA={orig_ra:10.6f}, Dec={orig_dec:10.6f}")
                    print(f"  Updated Pos:  RA={new_ra:10.6f}, Dec={new_dec:10.6f}")
                    print(f"  Correction:   "
                          f"dRA={correction_ra_arcsec:+.4f}\", dDec={correction_dec_arcsec:+.4f}\"")

                print(f"Total sources corrected: {len(updated_coords)}")

            # Update coordinates, report target epoch, and remove ref. epoch
            matchedtbl["ra"] = matchedtbl["ref_ra"].copy()
            matchedtbl["dec"] = matchedtbl["ref_dec"].copy()
            matchedtbl["ra"][gaia_pm_idx] = updated_coords.ra
            matchedtbl["dec"][gaia_pm_idx] = updated_coords.dec
            matchedtbl.meta["pipeline_epoch_mjd"] = slewgroup_mjd_avg.item()
            matchedtbl.remove_column("ref_epoch")

        # DQA perform source count consistency check
        _ = dqa_check_source_count(matchedtbl)
        self.metadata.add("numSources", len(matchedtbl))

        return Output(catalog=matchedtbl)


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

    coords = SkyCoord(ra=catalog['ra'], dec=catalog['dec'], frame='icrs')
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


def xyBoundingBox(imageList: list[SPHERExImage], pixmargin) -> tuple[WCS, float, float, float, float]:
    """Get bounding box of all the images in pixel coordinates of the first
    image.

    Parameters
    ----------
    imageList
        List of images separated by short slews
    pixmargin
        Margin in pixels to apply on the edge

    Returns
    -------
    result : tuple[WCS, float, float, float, float]
        WCS and zero-based boundaries: xmin, xmax, ymin, ymax
    """
    # get ra and dec of active area corners for each image
    actv_ra = np.array([])
    actv_dec = np.array([])
    for spximg in imageList:
        actv_rows, actv_cols = active_area_range(spximg)
        # high level API use 0-base image pixels
        # https://docs.astropy.org/en/stable/wcs/index.html#pixel-conventions-and-definitions # noqa
        xmin = actv_cols.start - 0.5 - pixmargin
        xmax = actv_cols.stop - 0.5 + pixmargin
        ymin = actv_rows.start - 0.5 - pixmargin
        ymax = actv_rows.stop - 0.5 + pixmargin

        x, y = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]).T
        ra, dec = spximg.wcs.all_pix2world(x, y, 0)
        actv_ra = np.concatenate((actv_ra, ra))
        actv_dec = np.concatenate((actv_dec, dec))

    if len(imageList) > 0:
        wcs_fst = imageList[0].wcs
        coord_actv_corners = SkyCoord(actv_ra, actv_dec, frame='icrs', unit='deg')
        (ixc, iyc) = wcs_fst.all_world2pix(coord_actv_corners.ra.deg,
                                           coord_actv_corners.dec.deg, 0)
        xmin = np.min(ixc)
        xmax = np.max(ixc)
        ymin = np.min(iyc)
        ymax = np.max(iyc)
        return wcs_fst, xmin, xmax, ymin, ymax


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
    selected = do_select(tbl, config)
    keep &= selected

    c1 = Column(tbl[id_col].data[keep], name=id_col, dtype=tbl[id_col].dtype, unit=tbl[id_col].unit)
    c2 = Column(ra[keep], name='ra', dtype=tbl[ra_col].dtype, unit=tbl[ra_col].unit)
    c3 = Column(dec[keep], name='dec', dtype=tbl[dec_col].dtype, unit=tbl[dec_col].unit)

    cols = [c1, c2, c3]

    # Retrieve magnitudes
    mag_swir, mag_mwir = refcat_utils.get_magnitudes(dichroicpair, tbl, keep)

    # Get detector IDs in this dichroic pair
    det1, det2 = instrument.from_dichroicpair(dichroicpair)

    # Rename magnitude columns appropriately
    mag_swir = Column(mag_swir.value, name=f"mag_det{det1}", dtype=np.float64)
    mag_mwir = Column(mag_mwir.value, name=f"mag_det{det2}", dtype=np.float64)

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
    return selected
