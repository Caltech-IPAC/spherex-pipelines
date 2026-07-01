__all__ = ["AstropyParquetFormatter"]

from lsst.daf.butler.formatters.parquet import arrow_to_astropy, ParquetFormatter
from astropy.table import Table
import pyarrow as pa


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

        pa_table: pa.Table = super().read_from_local_file(path, component=component,
                                                          expected_size=expected_size)

        if pa_table is not None:
            return arrow_to_astropy(pa_table)
