__all__ = ["PyarrowTableFormatter"]

from typing import Optional

from lsst.daf.butler.formatters.file import Formatter
from lsst.utils.introspection import get_full_type_name
import pyarrow as pa
import pyarrow.parquet as pq


class PyarrowTableFormatter(Formatter):
    """Formatter to read/write pyarrow table to/from parquet file."""
    extension = ".parq"
    supportExtensions = frozenset({".parquet", ".parq"})

    def read(self, component: Optional[str] = None) -> pa.Table:
        """Read some or all of the parquet file into a `pyarrow.Table`.

        Parameters
        ----------
        component : Optional[str], optional
            Currently not used, by default None.

        Returns
        -------
        table : `pyarrow.Table`
            A `pyarrow.Table` with all or some of the columns and rows.

        Raises
        ------
        NotImplementedError
            Raised if a ``component`` is passed.
        ValueError
            Raised if a column name is not found in the table schema.
        """
        if component is not None:
            raise NotImplementedError("'component' not implemented.")

        path = self.fileDescriptor.location.path
        # Parameter validation done by StorageClass
        parameters = self.fileDescriptor.parameters

        parquet_file = pq.ParquetFile(path)

        # Read in entire parquet file if no parameters
        if not parameters:
            try:
                table = parquet_file.read()

            except FileNotFoundError:
                table = None

            return table

        # Handle parameters and read in table
        columns = parameters["columns"]

        if isinstance(columns, str):
            columns = [columns]

        # Get table columns to compare to
        parquet_columns = parquet_file.schema.names
        for column in columns:
            if column not in parquet_columns:
                raise ValueError(f"Column '{column}' not found in parquet file.")

        try:
            table = parquet_file.read(columns=columns)

        except FileNotFoundError:
            table = None

        return table

    def write(self, inMemoryDataset: pa.Table) -> None:
        """Write in memory dataset to file on disk

        Parameters
        ----------
        inMemoryDataset : `pyarrow.Table`
            Pyarrow table to write to parquet file.

        Raises
        ------
        NotImplementedError
            Raised if ``inMemoryDataset`` is not `pyarrow.Table`
        """
        if not isinstance(inMemoryDataset, pa.Table):
            raise NotImplementedError(
                f"Unable to write {get_full_type_name(inMemoryDataset)} to"
                " a parquet file, requires pyarrow.Table"
            )
        location = self.makeUpdatedLocation(self.fileDescriptor.location)
        path = location.path

        pq.write_table(inMemoryDataset, path)

        return
