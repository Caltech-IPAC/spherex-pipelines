__all__ = ["updateExposureRegion"]

import json
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from lsst.pipe.base import TaskMetadata
from lsst.daf.butler import (
    Butler,
    DatasetRef,
)
from spx_data import ExposureBoundary, get_exposure_boundary_info
from typing import Optional, Tuple


def updateExposureRegion(
    repo,
    collections,
    in_type="level2b_metadata",
    where="",
    limit=-30000,
) -> None:
    """Update region fields based on the results from Fine Astrometry.

    Currently, the fields necessary to calculate exposure region are stored
    in Level 2b metadata.

    Parameters
    ----------
    repo : `str`
        URI to the Butler repository.
    collections : `str`
        Input collections.
    in_type : `str`
        Dataset type of Level 2b metadata or Level 2 images.
    where : `str`, optional
        A string expression similar to a SQL WHERE clause.  May involve any
        column of a dimension table or (as a shortcut for the primary key
        column of a dimension table) dimension name.
    limit : `int`, optional
        Maximum number of datasets to query from the Butler.
    """
    butler = Butler(repo, writeable=True)

    # Query datasets in the specified collection
    datasets = butler.query_datasets(
        dataset_type=in_type,
        collections=collections,
        find_first=True,
        where=where,
        limit=limit,
    )

    def boundary_from_metadata(ref: DatasetRef) -> tuple[float, float, list, list]:
        uri = butler.getURI(ref)
        task_metadata = TaskMetadata(**json.load(uri))
        ra = task_metadata.getScalar("level2b:estimate_fine_astrometry.center_ra")
        dec = task_metadata.getScalar("level2b:estimate_fine_astrometry.center_dec")
        ra_vals = task_metadata.getArray("level2b:estimate_fine_astrometry.boundaries_ra")
        dec_vals = task_metadata.getArray("level2b:estimate_fine_astrometry.boundaries_dec")
        return ra, dec, ra_vals, dec_vals

    def boundary_from_image(
            ref: DatasetRef,
    ) -> Optional[Tuple[float, float, list, list]]:
        uri = butler.getURI(ref)
        header = fits.getheader(uri.ospath, ext=1)
        fin_ast_key = 'FINAST'

        if fin_ast_key in header and header[fin_ast_key] == 0:
            wcs = WCS(header)
            return get_exposure_boundary_info(wcs)
        else:
            return None

    # Set function to use
    if in_type.endswith("metadata"):
        # faster way to get points on the boundary of an image from metadata
        # the points are precalculated by Fine Astrometry module
        # TODO: Metadata does not working properly.
        get_exp_boundary = boundary_from_metadata
    else:
        # slower way to get points on the boundary of an image from image WCS
        get_exp_boundary = boundary_from_image

    exp_boundaries_map = {}  # maps exposure id to ExposureBoundary
    # Retrieve exposure boundaries from level 2 metadata
    for dataset_ref in datasets:
        # Access the dataId which contains the dimension values
        data_id = dataset_ref.dataId
        result = get_exp_boundary(dataset_ref)

        if result:
            ra_c, dec_c, ra_lst, dec_lst = result
            expid = data_id["exposure"]
            exp_boundaries_map[expid] = ExposureBoundary(
                expid=expid,
                center=(ra_c, dec_c),
                bounds=zip(ra_lst, dec_lst),
            )

    recs = butler.query_dimension_records(
        "exposure",
        where=where,
        order_by="observation",
        limit=limit,
    )

    new_exp_records = []
    center_ra_lst = []
    center_dec_lst = []
    # Construct updated exposure records
    for rec in recs:
        new_rec = rec.toDict()
        exp_boundary = exp_boundaries_map.get(rec.id, None)

        if exp_boundary:
            try:
                new_rec["region"] = exp_boundary.to_sphgeom_region()
                center_ra_lst.append(exp_boundary.center[0])
                center_dec_lst.append(exp_boundary.center[1])
                new_exp_records.append(new_rec)
            except Exception as e:
                print(f"Error updating exposure record for exposure ID {rec.id}: {e}")

    # Populate galactic longitude and latitude of the image center
    equatorial_coords = SkyCoord(
        ra=center_ra_lst * u.deg,
        dec=center_dec_lst * u.deg,
        frame="icrs",
    )
    galactic_coords = equatorial_coords.galactic
    center_l_lst = galactic_coords.l.value
    center_b_lst = galactic_coords.b.value
    for i, new_rec in enumerate(new_exp_records):
        new_rec["galactic_lon"] = float(center_l_lst[i])
        new_rec["galactic_lat"] = float(center_b_lst[i])

    # TODO: populate dgl_qntl column of exposure table

    with butler.transaction():
        for exp_rec in new_exp_records:
            butler.registry.syncDimensionData("exposure", exp_rec, update=True)

    n_updated = len(new_exp_records)
    print(f"Updated {n_updated} exposure{'s' if n_updated != 1 else ''}.")
