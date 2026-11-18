__all__ = ["addPhotometryExceptions"]

# Standard library imports
from typing import Iterable

# Related third-party imports
from lsst.daf.butler import Butler, EmptyQueryResultError

# Local application/library specific imports
from ..defaults import DatasetTypes

def addPhotometryExceptions(repo: str, collections: str | Iterable[str], in_type: str = "phot_exceptions",
                            where: str = "", limit: int = -100000):
    """Add photometry exceptions to the butler as `phot_meas` datasets.

    Parameters
    ----------
    repo : `str`
        URI to the Butler repository.
    collections : `str` or `Iterable[str]`
        Collection name(s) to search.
    in_type: `str`, optional
        Dataset type for photometry exceptions. Default is "phot_exceptions".
    where : `str`, optional
        SQL WHERE-like expression.
    limit : `int`, optional
        Maximum number of returned datasets. Default is -100000 (warning issued
        if results exceed 100000).
    """
    from spx_tasks.perform_forced_photometry import PerformForcedPhotometryConfig  # circular import guard

    butler = Butler(repo, writeable=True)

    # Query exceptions datasets in the specified collection
    try:
        datasets = butler.query_datasets(
            dataset_type=in_type,
            collections=collections,
            find_first=True,
            where=where,
            limit=limit,
        )
    except EmptyQueryResultError:
        datasets = []
        print(f"No {in_type} datasets found in collections {collections} with where='{where}' and"
              f" limit={limit}.")

    # Write phot_meas datasets for each tile in the exceptions dataset
    hips_col_name = f"hips{PerformForcedPhotometryConfig().hips_order}"
    dt = DatasetTypes()
    if len(datasets) > 0:
        for ref in datasets:
            data_id = dict(ref.dataId.mapping)  # includes instrument and exposure dimensions
            exceptions = butler.get(ref)
            if exceptions is not None:
                exceptions_by_tile = exceptions.group_by(keys=hips_col_name)
                for tile, group in zip(exceptions_by_tile.groups.keys, exceptions_by_tile.groups):
                    tile_data_id = {**data_id, hips_col_name: tile[0]}  # use idx to select val, not col
                    butler.put(group, dt.phot_meas, dataId=tile_data_id, run=ref.run)
                    print(f"Added phot_meas for tile {tile[0]} (dataId: {tile_data_id}) to run {ref.run}.")
