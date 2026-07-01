__all__ = ["AstropyParquetFormatter"]

# Standard library imports
import json
import logging

# Related third party imports
from astropy.table import Table
import pyarrow as pa
from lsst.daf.butler.formatters.parquet import (
    _apply_astropy_metadata, arrow_to_numpy_dict, astropy_to_arrow, ParquetFormatter)

# Local application/library specific imports
from spx_utils import get_current_rss

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


_META_KEY = b"user_metadata"
_ALL_META_KEY = b"spherex:astropy_meta"
ASTROPY_PANDAS_INDEX_KEY = "lsst::arrow::astropy_pandas_index"


# NOTE: This function is adapted from
#       lsst.daf.butler.formatters.parquet.arrow_to_astropy.
def shallow_arrow_to_astropy(arrow_table: pa.Table) -> Table:
    """Convert a pyarrow table to an `astropy.Table` using a shallow copy.

    Parameters
    ----------
    arrow_table : `pyarrow.Table`
        Input arrow table to convert. If the table has astropy unit
        metadata in the schema it will be used in the construction
        of the ``astropy.Table``.

    Returns
    -------
    table : `astropy.Table`
        Converted astropy table.
    """
    from astropy.table import Table

    astropy_table = Table(arrow_to_numpy_dict(arrow_table), copy=False)

    _apply_astropy_metadata(astropy_table, arrow_table.schema)

    if (key := ASTROPY_PANDAS_INDEX_KEY) in astropy_table.meta:
        if astropy_table.meta[key] not in astropy_table.columns:
            astropy_table.meta.pop(key)

    return astropy_table


class AstropyParquetFormatter(ParquetFormatter):
    """Formatter to read/write astropy table to/from parquet file."""
    extension = ".parq"
    supportExtensions = frozenset({".parquet", ".parq"})

    def read_from_local_file(self, path: str, component: str | None = None, expected_size: int = -1) -> Table:
        """Read some or all of the parquet file into a `pyarrow.Table`.

        Parameters
        ----------
        component : Optional[str], optional
            Currently not used, by default None.

        Returns
        -------
        table : `astropy.table.Table`
            An astropy table.

        Raises
        ------
        NotImplementedError
            Raised if a ``component`` is passed.
        ValueError
            Raised if a column name is not found in the table schema.
        """
        if component is not None:
            raise NotImplementedError("'component' not implemented.")

        # Read parquet file and convert to astropy Table object
        logger.debug(get_current_rss("Before reading parquet file"))
        pa_table: pa.Table = super().read_from_local_file(path, component=component,
                                                          expected_size=expected_size)
        logger.debug(get_current_rss("After reading parquet file"))
        tbl = shallow_arrow_to_astropy(pa_table)
        logger.debug(get_current_rss("After astropy table conversion"))

        # Recover astropy meta if present
        md = pa_table.schema.metadata or {}
        if _META_KEY in md:
            try:
                tbl.meta.update(json.loads(md[_META_KEY].decode("utf-8")))
            except Exception:
                pass
        return tbl

    def write_to_local_file(self, in_memory_dataset: Table, path: str) -> None:
        # Convert to Arrow
        pa_table = astropy_to_arrow(in_memory_dataset)

        # Start with any existing schema metadata
        meta = dict(pa_table.schema.metadata or {})

        # Drop the compact user metadata dict (if present)
        user_meta = in_memory_dataset.meta.get("user_metadata")
        if user_meta is not None:
            try:
                meta[_META_KEY] = json.dumps(user_meta, default=str).encode("utf-8")
            except Exception:
                # be defensive
                meta[_META_KEY] = json.dumps(
                    {k: str(v) for k, v in user_meta.items()}
                ).encode("utf-8")

        # Snapshot the entire Table.meta for completeness
        try:
            meta[_ALL_META_KEY] = json.dumps(
                in_memory_dataset.meta, default=str
            ).encode("utf-8")
        except Exception:
            meta[_ALL_META_KEY] = json.dumps(
                {k: str(v) for k, v in in_memory_dataset.meta.items()}
            ).encode("utf-8")

        # Attach metadata to schema
        pa_table = pa_table.replace_schema_metadata(meta)

        # Delegate actual writing
        super().write_to_local_file(pa_table, path)
