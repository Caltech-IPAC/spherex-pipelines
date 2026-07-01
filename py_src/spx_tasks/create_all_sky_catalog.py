__all__ = ["CreateAllSkyCatalogConfig", "CreateAllSkyCatalogTask"]

# Standard library imports
from typing import Literal

# Related third party imports
import numpy as np
import pyarrow as pa
from astropy.table import meta, Column
from lsst.pex.config import ListField, ChoiceField, Field
from lsst.daf.butler import DeferredDatasetHandle
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spherex_parameters import load_spherex_parameters
from spx_data.spx_version import SSDC_PIPELINES_VERSION

params = load_spherex_parameters()
# ==============================================================================

META_YAML_TYPE_TAG = "meta: !!omap"  # ordered mapping for YAML parsers


def compute_nyquist_metric(wavelengths, params) -> float:
    """Computes the Nyquist completeness metric based on the number
       of spectral channels containing at least two measurements.

    Parameters
    ----------
    wavelengths : `list[float]`
        The list of observed wavelengths for a given source.
    params : `dict`
        Dictionary containing spectral channel parameters.

    Returns
    -------
    `float`
        Nyquist sampling metric in the range (0,1),
        where 1 indicates complete sampling.
    """
    if len(wavelengths) < 2:
        return 0.0  # Not enough data points to define a metric

    num_channels_covered = 0

    # Define channel properties
    channel_sizes = (params["spectral_channel"]["lmean"]["quantity"].value
                     / params["spectral_channel"]["R"]["quantity"])
    channel_maxes = params["spectral_channel"]["lmax"]["quantity"].value

    # Assign wavelengths to channels
    channel_assignment = np.digitize(wavelengths, channel_maxes)

    # Iterate over all channels
    for channel_idx in range(len(channel_maxes)):
        meas_in_channel = wavelengths[np.where(channel_assignment == channel_idx)[0]]
        if len(meas_in_channel) > 1:
            spread_in_channel = np.max(meas_in_channel) - np.min(meas_in_channel)
            if spread_in_channel > 0.33 * channel_sizes[channel_idx]:
                num_channels_covered += 1

    # Normalize metric by total number of channels (assumed 102)
    print(f"n channels = {len(channel_maxes)}")
    nyquist_metric = num_channels_covered / len(channel_maxes)

    return max(0, min(1, nyquist_metric))


def add_pass_thru(col_name: str, output_fields: list):
    """Add a pass-through column to the list of pyarrow fields for the output
    parquet table.

    Parameters
    ----------
    col_name : `str`
        Column name in input source catalog.
    input_cat : `astropy.table.Table`
        Input source catalog.
    output_tab : `astropy.table.Table`
        Output photometry measurement table.
    """
    pass_thru_lookup = {
        "external_source_id": {
            "final_idx": 2,
            "pa_type": pa.string()
        },
        "external_survey": {
            "final_idx": 1,
            "pa_type": pa.uint8()
        },
        "name": {
            "final_idx": 1,
            "pa_type": pa.string()
        },
        "parallax": {
            "final_idx": 8,
            "pa_type": pa.float32()
        },
        "pm_dec": {
            "final_idx": 7,
            "pa_type": pa.float32()
        },
        "pm_ra": {
            "final_idx": 6,
            "pa_type": pa.float32()
        },
        "spherex_class": {
            "final_idx": 3,
            "pa_type": pa.int32()
        },
    }

    output_fields.insert(pass_thru_lookup[col_name]["final_idx"],
                         (col_name, pass_thru_lookup[col_name]["pa_type"]))


def add_meta_yaml(meta_yaml: list, mode: Literal["field", "version"], field_dict: dict | None = None):
    """Add field or version metadata to a list of text lines with YAML header
    content.

    Parameters
    ----------
    meta_yaml : `list` of `str`
        List of strings representing YAML header content.
    mode : {'field', 'version'}
        Selects which type of metadata to add.
    field_dict : `dict`, optional
        Dictionary of field metadata, including keys (values): 'name' (string),
        'datatype' (string), 'description' (string), and optionally 'unit'
        (`astropy.units.Quantity`). Default is None.

    Returns
    -------
    None
    """
    if mode == "field" and field_dict is not None:
        if "unit" in field_dict.keys():
            new_meta = (f"- {{name: {field_dict['name']}, unit: {field_dict['unit']},"
                        f" datatype: {field_dict['datatype']}, description: {field_dict['description']}}}")
        else:
            new_meta = (f"- {{name: {field_dict['name']}, datatype: {field_dict['datatype']},"
                        f" description: {field_dict['description']}}}")
    elif mode == "version":
        new_meta = f"- {{version_allsky_cat: {SSDC_PIPELINES_VERSION}}}"

    if META_YAML_TYPE_TAG in meta_yaml:
        if mode == "field":
            # Insert fields at end of "datatype" list, before LSST metadata
            idx = meta_yaml.index(META_YAML_TYPE_TAG)
        elif mode == "version":
            # Insert version after LSST metadata
            idx = meta_yaml.index(META_YAML_TYPE_TAG) + 1
        meta_yaml.insert(idx, new_meta)
    else:
        # Add metadata at general level
        meta_yaml.append(new_meta)
    return


class CreateAllSkyCatalogConnections(pipeBase.PipelineTaskConnections,
                                     dimensions={"instrument", "hips6"},
                                     defaultTemplates={}):
    meas_table_list = cT.Input(
        name="phot_meas",
        doc="Input astropy table per HEALPix.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "exposure", "hips6"],
        multiple=True,
        deferLoad=True)
    allsky_cat = cT.Output(
        name="allsky_cat",
        doc="Filtered all sky catalog per HEALPix.",
        storageClass="PyarrowTable",
        dimensions=["instrument", "hips6"])

    def __init__(self, *, config=None):
        super().__init__(config=config)


class CreateAllSkyCatalogConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CreateAllSkyCatalogConnections):
    list_items = ListField(
        doc="Measurements for sources to be recorded in a list format. This is to"
        " be able to store multiple measurements for a single source in"
        " parquet.",
        dtype=str,
        default=("x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err", "flux_bkg",
                 "local_bkg_flg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id"))

    sort_on = ChoiceField(
        doc="Column to use to sort additional measurements stored as arrays.",
        dtype=str,
        default="mjd",
        allowed={
            "source_id": "Unique SPHEREx Reference Catalog ID",
            "mjd": "Timestamp (up to ms precision) of midpoint of exposure",
            "lambda": "Native wavelengths of observations"})

    filter_spherex_class = Field(
        doc="Bitmask to filter SPHEREx class values. If 0, no filtering is applied.",
        dtype=int,
        default=0  # Default means no filtering is applied
    )

    pass_thru_cols = ListField(
        doc="List of columns to pass through to the output catalog, in order of relative placement in the"
            " output catalog.",
        dtype=str,
        default=["external_survey", "external_source_id", "spherex_class", "pm_ra", "pm_dec", "parallax"]
    )


class CreateAllSkyCatalogTask(pipeBase.PipelineTask):
    """Implementation of L3 Create All-Sky Catalog
       with optional filtering for calibrators.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-035

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    """

    _DefaultName = "create-all-sky-catalog"
    ConfigClass = CreateAllSkyCatalogConfig
    full_name = "Create All-Sky Catalog"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "SSDC-MS-035 v3.0"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, meas_table_list: list[DeferredDatasetHandle]) -> pipeBase.Struct:
        """Returns a condensed table of all source measurements for one
        HEALPix.

        Parameters
        ----------
        meas_table_list
            List of deferred `~astropy.table.Table` with source measurements
            for a HEALPix.

        Returns
        -------
        output: `lsst.pipe.base.Struct`
            Output with component:

            - ``allsky_cat`` : `pyarrow.Table`
                Contains a condensed `pyarrow.Table` with one row per source
                with array columns for multiple measurements.
            - ``primary_calibrators`` : `pyarrow.Table`
                Contains a filtered catalog of primary calibrator sources.

        Notes
        -----
        Reference catalog flags
        - PHOTOMETRIC_FLAG = 1 << 2  # Bit 2 = 4  Primary calibrators
        - CALSPEC_FLAG = 1 << 3      # Bit 3 = 8  Calspec sources
        - SECONDARY_FLAG = 1 << 8    # Bit 8 = 256  Secondary calibrators
        """
        num_tables = len(meas_table_list)
        self.log.debug(f"Number of input tables: {num_tables}")

        # Store source IDs and index so can quickly append measurements
        done = {}
        # Only doing this task for one HEALPix at a time
        rows = []

        meas_table = None
        for deferred_meas_table in meas_table_list:
            # Get the data and colnames
            meas_table = deferred_meas_table.get()
            colnames = meas_table.colnames

            # Apply filtering efficiently using NumPy bitwise operations
            if self.config.filter_spherex_class > 0:
                meas_table = meas_table[meas_table["spherex_class"] & self.config.filter_spherex_class > 0]

            # If no sources remain after filtering, continue to next table
            if len(meas_table) == 0:
                continue

            for meas in meas_table:
                source_id = meas["source_id"]

                # Check to see if source has already been recorded
                index = done.get(source_id, None)
                if index is not None:
                    # Append measurements to list items
                    for item in self.config.list_items:
                        if meas[item] != np.ma.core.MaskedConstant:
                            rows[index][item].append(meas[item])
                    # Move on to next measurement
                    continue

                # Convert specific items to a list for storage if first time
                row = {}
                for index, item in enumerate(meas):
                    colname = colnames[index]
                    if colname in self.config.list_items:
                        row[colname] = []
                        if item != np.ma.core.MaskedConstant:
                            row[colname].append(item)
                    else:
                        row[colname] = item

                # Compute Nyquist completeness
                row["nyquist_comp"] = compute_nyquist_metric(row["lambda"], params)

                # Append the meas and store the index for quick updating
                rows.append(row)
                done[source_id] = len(rows) - 1

                # Append metadata

        # If no sources were selected, return an empty table
        if len(rows) == 0:
            self.log.warning("No sources matched the filter criteria. Writing an empty table.")

        # columns metadata (units, descriptions) in format understood by
        # astropy
        metadata = {}
        astropy_cols = {}
        if meas_table is not None:
            meta_yaml = meta.get_yaml_from_table(meas_table)
            add_meta_yaml(meta_yaml, mode="field", field_dict={"name": "nyquist_comp",
                          "datatype": "float32", "description": "Nyquist flag"})
            add_meta_yaml(meta_yaml, mode="version")
            meta_yaml_str = "\n".join(meta_yaml)
            metadata = {"table_meta_yaml": meta_yaml_str}
            astropy_cols = meas_table.columns

        # Sorting measurements based on configured criteria
        self.log.debug(f"Sorting array columns by '{self.config.sort_on}'")
        for source in rows:
            # Get the order to sort all list_items by increasing `sort_on`
            order = np.argsort(source[self.config.sort_on])
            # Apply to every item in list_items
            for item in self.config.list_items:
                # Can only index like this with np.array, return list
                if len(source[item]) > 0:
                    source[item] = np.array(source[item])[order].tolist()

        # Define lists of types for parquet library
        lfloat64 = pa.list_(pa.float64())
        lfloat32 = pa.list_(pa.float32())
        lbool = pa.list_(pa.bool_())
        lint64 = pa.list_(pa.int64())
        luint8 = pa.list_(pa.uint8())

        # ex. field with metadata:
        # pa.field("flux", lfloat32,
        #           metadata={"unit": "uJy", "description": "Measured Flux"}),
        pa_flds = [
            ("source_id", pa.int64()),
            ("ra", pa.float64()),
            ("dec", pa.float64()),
            ("x_image", lfloat32),
            ("y_image", lfloat32),
            ("mjd", lfloat64),
            ("lambda", lfloat32),
            ("lambda_width", lfloat32),
            ("flux", lfloat32),
            ("flux_err", lfloat32),
            ("flux_bkg", lfloat32),
            ("local_bkg_flg", lbool),
            ("ext_flg", pa.bool_()),
            ("flags", lint64),
            ("fit_ql", lfloat32),
            ("deep_flg", lbool),
            ("lvf_id", lint64),
            ("det_id", luint8),
            ("nyquist_comp", pa.float32()),
        ]

        # Add pass through columns
        if self.config.pass_thru_cols:
            for col_name in self.config.pass_thru_cols:
                add_pass_thru(col_name, pa_flds)
                if col_name == "external_source_id":
                    # Add numpy string length to metadata
                    metadata[f"table::len::{col_name}"] = str(astropy_cols.get(col_name).dtype.itemsize // 4)

        # function to create field with metadata
        def create_pa_fld(name, pa_type):
            if name == "nyquist_comp":
                fld_metadata = {"description": "Nyquist flag"}
                return pa.field(name, pa_type, metadata=fld_metadata)
            else:
                # Pull other field metadata from input measurement table
                acol = astropy_cols.get(name, Column(name=name, description=name))
                unit = acol.unit
                descr = acol.description
                # all columns should have description
                fld_metadata = {"description": descr}
                if acol.unit is not None:
                    fld_metadata["unit"] = str(unit)
                return pa.field(name, pa_type, metadata=fld_metadata)

        schema = pa.schema([create_pa_fld(t[0], t[1]) for t in pa_flds], metadata=metadata)

        # Create pyarrow table with schema and rows
        allsky_cat = pa.Table.from_pylist(rows, schema=schema)

        # Metadata are saved in Butler repository
        num_sources = len(allsky_cat)
        self.log.debug(f"Number of sources in final table: {num_sources}")
        self.metadata.add("num_sources", num_sources)
        self.metadata.add("num_tables", num_tables)

        return pipeBase.Struct(allsky_cat=allsky_cat)
