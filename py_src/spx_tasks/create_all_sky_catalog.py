__all__ = ["CreateAllSkyCatalogConfig", "CreateAllSkyCatalogTask"]

# Standard library imports
from collections.abc import Mapping
from typing import Literal

# Related third party imports
import numpy as np
import pyarrow as pa
from astropy.table import meta, Column
from lsst.pex.config import ListField, ChoiceField, Field
from lsst.daf.butler import DataCoordinate, DeferredDatasetHandle
import astropy.units as u
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_butler import REFCAT_SKYPIX_NAME
from spherex_parameters import load_spherex_parameters
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_utils.utils import get_refcat_provenance

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


def hips6_to_hips5(hips6: int) -> int:
    """Convert child HiPS 6 ID to parent HiPS 5 ID by bit shifting.

    Parameters
    ----------
    hips6 : `int`
        HiPS 6 ID.

    Returns
    -------
    `int`
        Corresponding parent HiPS 5 ID.
    """
    parent_order = 5
    child_order = 6
    shift = 2 * (child_order - parent_order)
    return hips6 >> shift


class CreateAllSkyCatalogConnections(pipeBase.PipelineTaskConnections,
                                     dimensions={"instrument", "hips6"},
                                     defaultTemplates={}):
    meas_table_list = cT.Input(
        name="phot_meas",
        doc="List of input photometry measurement tables per HEALPix.",
        storageClass="AstropyParquet",
        dimensions=["instrument", "exposure", "hips6"],
        multiple=True,
        deferLoad=True)
    refcat = cT.PrerequisiteInput(
        name="refcat",
        doc="Reference catalog.",
        storageClass="AstropyTable",
        dimensions=[REFCAT_SKYPIX_NAME],
        deferLoad=True,
        multiple=True,
        minimum=1)
    allsky_cat = cT.Output(
        name="allsky_cat",
        doc="Filtered all sky catalog per HEALPix.",
        storageClass="PyarrowTable",
        dimensions=["instrument", "hips6"])

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.merge_refcat:
            del self.refcat

    def adjustQuantum(self, inputs: dict, outputs: Mapping, label: str, data_id: DataCoordinate):
        """Filters out input reference catalog shards if they do not overlap
        with the corresponding input photometry measurement tables.
        """
        # Only need to adjust if using reference catalog
        if "refcat" not in inputs:
            return super().adjustQuantum(inputs, outputs, label, data_id)

        # Determine HiPS6 and HiPS5 IDs for this quantum
        hips6 = data_id["hips6"]
        hips5 = hips6_to_hips5(hips6)

        # Filter reference catalog dataset references and validate only 1 match
        connection, old_refs = inputs["refcat"]
        new_refs = [ref for ref in old_refs if ref.dataId.get(REFCAT_SKYPIX_NAME) == hips5]
        assert len(new_refs) == 1, (
            f"Expected 1 'refcat' dataset reference for {REFCAT_SKYPIX_NAME}={hips5} found {len(new_refs)}.")

        # Update inputs
        adjusted_inputs = {"refcat": (connection, new_refs)}
        inputs.update(adjusted_inputs)

        # Ignore empty outputs from super
        super().adjustQuantum(inputs, outputs, label, data_id)

        # Only return modified connections
        return adjusted_inputs, {}


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
        default="lambda",
        allowed={
            "mjd": "Timestamp (up to ms precision) of midpoint of exposure",
            "lambda": "Native wavelengths of observations"})

    filter_spherex_class = Field(
        doc="Bitmask to filter SPHEREx class values. If 0, no filtering is applied.",
        dtype=int,
        default=0)  # Default means no filtering is applied

    merge_refcat = Field(
        doc="If True, adds 'pass_thru_cols' from the reference catalog to the output all sky catalog."
            " Default is True.",
        dtype=bool,
        default=True)

    pass_thru_cols = ListField(
        doc="List of columns to pass through to the output catalog, in order of relative placement in the"
            " output catalog.",
        dtype=str,
        default=["coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec"])

    def update(self, **kw):
        """Override default update method to validate upon field changes.
        """
        super().update(**kw)
        self.validate()

    def validate(self):
        """Override default validate method to include inter-field checks.
        """
        super().validate()

        # If merging refcat data,  1+ pass-through columns must be specified
        if self.merge_refcat and (len(self.pass_thru_cols) == 0):
            raise ValueError("To merge data from the refcat, 1+ pass-through columns must be specified.")


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

    def run(self, meas_table_list: list[DeferredDatasetHandle],
            refcat: DeferredDatasetHandle = None) -> pipeBase.Struct:
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

        # If using reference catalog, get the data and prepare indexing
        if refcat is not None:
            # Fetch reference catalog data and validate
            refcat = refcat[0].get()  # TODO: can reading this be filtered in QR2 butler?
            uniq_ref_ids, counts = np.unique(refcat["SPHERExRefID"].data, return_counts=True)
            if len(uniq_ref_ids) != len(refcat):
                dup_ids = uniq_ref_ids[counts > 1]
                raise ValueError(f"Duplicate 'SPHERExRefID' values found in reference catalog: {dup_ids}")

            # Indexing
            refcat_ids = refcat["SPHERExRefID"].data
            sorted_idx = np.argsort(refcat_ids)
            refcat_sorted = refcat[sorted_idx]

        # If pass-through columns requested, prepare lookup dictionary
        if len(self.config.pass_thru_cols) > 0:
            # Lookup for adding columns to astropy table and/or pyarrow schema
            pass_thru_lookup = {
                "coord_src": {
                    "final_name": "external_survey",
                    "description": "Identifier of external survey",
                    "dtype": np.uint8,
                    "unit": None,
                    "final_idx": 1,
                    "pa_type": pa.uint8(),
                },
                "dec": {
                    "final_name": "dec",
                    "description": "DEC of the source in the reference catalog",
                    "dtype": np.float64,
                    "unit": u.degree,
                    # NOTE: no final_idx/pa_type since replacing existing col
                },
                "name": {
                    # NOTE: no description/dtype/unit since only need to add
                    #       to pyarrow schema
                    "final_name": "name",
                    "final_idx": 1,
                    "pa_type": pa.string(),
                },
                "parallax": {
                    "final_name": "parallax",
                    "description": "Parallax (from reference catalog)",
                    "dtype": np.float32,
                    "unit": u.mas,
                    "final_idx": 7,
                    "pa_type": pa.float32(),
                },
                "pmdec": {
                    "final_name": "pm_dec",
                    "description": "Proper motion in Decl. (from reference catalog)",
                    "dtype": np.float32,
                    "unit": u.mas/u.year,
                    "final_idx": 6,
                    "pa_type": pa.float32(),
                },
                "pmra": {
                    "final_name": "pm_ra",
                    "description": "Proper motion in R.A. (from reference catalog)",
                    "dtype": np.float32,
                    "unit": u.mas/u.year,
                    "final_idx": 5,
                    "pa_type": pa.float32(),
                },
                "ra": {
                    "final_name": "ra",
                    "description": "R.A. of the source in the reference catalog",
                    "dtype": np.float64,
                    "unit": u.degree,
                    # NOTE: no final_idx/pa_type since replacing existing col
                },
                "spherex_class": {
                    "final_name": "spherex_class",
                    "description": "Source origin in reference catalog",
                    "dtype": np.int32,
                    "unit": None,
                    "final_idx": 2,
                    "pa_type": pa.int32(),
                },
            }

        # Fetch data from measurement tables
        done = {}  # Store source IDs and index so can quickly append measurements
        rows = []  # Only doing this task for one HEALPix at a time
        ref_ver = None  # Keep a local copy for reuse when writing parquet metadata
        meas_table = None  # Keep a copy for getting field-level metadata
        for deferred_meas_table in meas_table_list:
            # Get the data and colnames
            meas_table = deferred_meas_table.get()
            colnames = meas_table.colnames

            # Extract refcat provenance (version) once from the 1st table.
            # Stash it and later add it into the output Parquet metadata.
            if ref_ver is None:
                try:
                    _, _ref_ver = get_refcat_provenance(meas_table)
                except Exception:
                    _ref_ver = None

                # Fallbacks: top-level meta or nested user_metadata
                if not _ref_ver:
                    try:
                        _ref_ver = (
                            meas_table.meta.get("refcat_version")
                            or (meas_table.meta.get("user_metadata") or {}).get("refcat_version")
                        )
                    except Exception:
                        _ref_ver = None

                if _ref_ver:
                    ref_ver = str(_ref_ver)  # keep it locally
                    self.metadata.add("refcat_version", ref_ver)  # and in task metadata

            # If requested, add pass-through columns from refcat
            if self.config.merge_refcat:
                # Double check all source IDs have a match
                if not np.all(np.isin(meas_table["source_id"], refcat_ids)):
                    missing_in_refcat = list(set(meas_table["source_id"]) - set(refcat_ids))
                    raise ValueError(f"1+ photometered sources were not found in the reference catalog:"
                                     f" {missing_in_refcat}")

                # Find matches and add columns
                matched_idx = np.searchsorted(refcat_sorted["SPHERExRefID"], meas_table["source_id"])
                # NOTE: Double check "spherex_class", which can be pass-through
                #       in PFP or CASC.
                if (
                    ("spherex_class" in self.config.pass_thru_cols)
                    and ("spherex_class" in meas_table.colnames)
                ):
                    missing_cols = [col for col in self.config.pass_thru_cols if col != "spherex_class"]
                else:
                    missing_cols = self.config.pass_thru_cols
                for col in missing_cols:
                    if col == "ra" or col == "dec":
                        meas_table.replace_column(
                            name=col,
                            col=Column(
                                data=refcat[col][matched_idx],
                                name=col,
                                unit=pass_thru_lookup[col]["unit"],
                                description=pass_thru_lookup[col]["description"],
                                dtype=pass_thru_lookup[col]["dtype"]))
                    else:
                        meas_table.add_column(
                            Column(
                                data=refcat[col][matched_idx],
                                name=pass_thru_lookup[col]["final_name"],
                                unit=pass_thru_lookup[col]["unit"],
                                description=pass_thru_lookup[col]["description"],
                                dtype=pass_thru_lookup[col]["dtype"]),
                            index=pass_thru_lookup[col]["final_idx"])
                        if col == "spherex_class":
                            meas_table["spherex_class"].format = None

            # Fetch colnames after possible addition of pass-through columns
            colnames = meas_table.colnames

            # Apply filtering efficiently using NumPy bitwise operations
            if self.config.filter_spherex_class > 0:
                meas_table = meas_table[meas_table["spherex_class"] & self.config.filter_spherex_class > 0]

            # If no sources remain after filtering, continue to next table
            if len(meas_table) == 0:
                continue

            # Loop over photometry measurements
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
            if ref_ver:
                metadata["refcat_version"] = ref_ver

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

        # If requested, add pass-through columns to fields for pyarrow schema
        if self.config.pass_thru_cols:
            for col_name in self.config.pass_thru_cols:
                if col_name not in ("ra", "dec"):  # 'ra', 'dec' replace existing columns
                    pa_flds.insert(
                        pass_thru_lookup[col_name]["final_idx"],
                        (pass_thru_lookup[col_name]["final_name"], pass_thru_lookup[col_name]["pa_type"]))
                    if col_name in ("external_source_id", "name"):
                        # Add numpy string length to metadata
                        metadata[f"table::len::{col_name}"] = str(
                            astropy_cols.get(col_name).dtype.itemsize // 4)

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
