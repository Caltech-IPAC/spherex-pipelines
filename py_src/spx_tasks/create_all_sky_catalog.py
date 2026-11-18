__all__ = ["CreateAllSkyCatalogConfig", "CreateAllSkyCatalogTask"]

# Standard library imports
from collections.abc import Mapping
from typing import Literal

# Related third party imports
import numpy as np
import pandas as pd
import pyarrow as pa
from astropy.table import meta, Column
from lsst.pex.config import ListField, ChoiceField, Field
from lsst.daf.butler import DataCoordinate, DeferredDatasetHandle
from numba import njit
import astropy.units as u
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import FluxCorrectionsCube
from spx_butler import REFCAT_SKYPIX_NAME
from spx_data.spx_version import SSDC_PIPELINES_VERSION
from spx_utils.utils import get_refcat_provenance
from spx_data.spectral_channel import LVFChannelDefinitionV2

# ==============================================================================

META_YAML_TYPE_TAG = "meta: !!omap"  # ordered mapping for YAML parsers
N_DETECTORS = 6


@njit(cache=True)
def compute_nyquist_metric(meas_half_channels: np.ndarray, seen: np.ndarray) -> float:
    """Compute Nyquist completeness as fraction of spectral half-channels
    containing at least one measurement.

    Parameters
    ----------
    meas_half_channels : `numpy.ndarray`
        Per-measurement array of spectral half-channel numbers.
    seen : `numpy.ndarray`
        Boolean array of length equal to the total number of spectral
        half-channels across all detectors. Tracks which half-channels contain
        at least one measurement.

    Returns
    -------
    `float`
        Nyquist sampling metric in the range (0,1), where 1 indicates complete
        sampling.

    Notes
    -----
    This is the production metric used for nyquist_computation.
    """
    if len(meas_half_channels) < 1:  # no measurements for this source
        return 0.0

    seen[:] = False  # reset seen array for this source
    n_seen = 0
    n_half_chan_glob = len(seen)
    for k in range(meas_half_channels.shape[0]):
        c = meas_half_channels[k]
        if 1 <= c <= n_half_chan_glob:
            idx = c - 1  # convert channel # to 0-based idx
            if not seen[idx]:
                seen[idx] = True
                n_seen += 1

    if n_seen == 0:  # no valid active resolution channels for this source
        return 0.0

    return n_seen / n_half_chan_glob


@njit(cache=True)
def _nyquist_diagnostics_core(wavelengths: np.ndarray, meas_full_channels: np.ndarray,
                              n_full_chan_global: int, channel_sizes: np.ndarray, counts: np.ndarray,
                              spread: np.ndarray, spread_fraction: np.ndarray, max_w: np.ndarray,
                              min_w: np.ndarray) -> tuple:
    """JIT-optimized core function that fills counts/spread/spread_fraction in
    place and computes the six summary fractions.

    Returns
    -------
    `tuple`
        A tuple of six float values representing the fraction of spectral
        channels with the following properties: 1+ measurements,
        2+ measurements, spread >= 0.05, spread >= 0.15, spread >= 0.33, and
        spread >= 0.50.
    """
    n = wavelengths.shape[0]

    # Reset all pre-allocated buffer arrays for this source
    for c in range(n_full_chan_global):
        counts[c] = 0
        spread[c] = np.nan
        spread_fraction[c] = np.nan
        max_w[c] = -np.inf
        min_w[c] = np.inf

    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    # Compute per-channel counts and min/max wavelengths
    for k in range(n):
        chan = meas_full_channels[k]
        if chan < 1 or chan > n_full_chan_global:
            continue
        idx = chan - 1  # channel number -> 0-based index
        wl = wavelengths[k]
        counts[idx] += 1
        if wl > max_w[idx]:
            max_w[idx] = wl
        if wl < min_w[idx]:
            min_w[idx] = wl

    # Compute summary fractions
    n_1plus = 0
    n_2plus = 0
    n_005 = 0
    n_015 = 0
    n_033 = 0
    n_050 = 0
    for c in range(n_full_chan_global):
        if counts[c] >= 1:
            n_1plus += 1
        if counts[c] >= 2:
            n_2plus += 1
            sf = (max_w[c] - min_w[c]) / channel_sizes[c]
            spread[c] = max_w[c] - min_w[c]
            spread_fraction[c] = sf
            if sf >= 0.05:
                n_005 += 1
            if sf >= 0.15:
                n_015 += 1
            if sf >= 0.33:
                n_033 += 1
            if sf >= 0.50:
                n_050 += 1

    return (
        n_1plus / n_full_chan_global,
        n_2plus / n_full_chan_global,
        n_005 / n_full_chan_global,
        n_015 / n_full_chan_global,
        n_033 / n_full_chan_global,
        n_050 / n_full_chan_global,
    )


def compute_nyquist_diagnostics(base_chan_def: LVFChannelDefinitionV2, wavelengths: np.ndarray,
                                meas_full_channels: np.ndarray, n_full_chan_global: int,
                                channel_sizes: np.ndarray, counts: np.ndarray, spread: np.ndarray,
                                spread_fraction: np.ndarray, max_w: np.ndarray, min_w: np.ndarray) -> dict:
    """Compute diagnostic information for testing Nyquist spread behavior.

    Parameters
    ----------
    base_chan_def : `spx_data.spectral_channel.LVFChannelDefinitionV2`
        Base channel definition for the LVF.
    wavelengths: `numpy.ndarray`
        Array of wavelengths per measurement.
    meas_full_channels : `numpy.ndarray`
        Array of full-resolution spectral channel numbers per measurement.
    n_full_chan_global : `int`
        Total number of active, full-resolution spectral channels across all
        detectors.
    channel_sizes : `numpy.ndarray`
        Array of spectral channel sizes per spectral channel.
    counts, spread, spread_fraction, max_w, min_w : `numpy.ndarray`
        Pre-allocated buffer arrays of length ``n_full_chan_global`` to store
        per-channel counts, spreads, spread fractions, and maximum/minimum
        wavelengths.

    Returns
    -------
    dict
        Dictionary containing:
        - counts_per_channel
        - spread_per_channel
        - spread_fraction_per_channel
        - frac_bins_with_1plus
        - frac_bins_with_2plus
        - frac_bins_with_spread_005
        - frac_bins_with_spread_015
        - frac_bins_with_spread_033
        - frac_bins_with_spread_050

    Raises
    ------
    ValueError
        If ``n_full_chan_global`` is not equal to product of the number of base
        channels and the number of detectors. Although ``n_full_chan_global``
        is invariant across sources, this guards against silent errors from
        using this function with unsupported resolutions.

    Notes
    -----
    ``counts_per_channel``, ``spread_per_channel``, and
    ``spread_fraction_per_channel`` are the same buffer arrays passed in as
    arguments, NOT copies. Since they are reset and overwritten in place on
    each call, they only reflect the current source's values until the next
    call to this function. If you need to retain the current source's values,
    copy and store them before the next call.
    """
    if n_full_chan_global != base_chan_def.n_base_channels * N_DETECTORS:
        raise ValueError(f"Nyquist diagnostics are currently only supported for full-resolution global"
                         f" channels (1-{base_chan_def.n_base_channels * N_DETECTORS}). Found"
                         f" n_full_chan_global={n_full_chan_global}.")

    frac_1plus, frac_2plus, frac_005, frac_015, frac_033, frac_050 = _nyquist_diagnostics_core(
        wavelengths, meas_full_channels, n_full_chan_global, channel_sizes, counts, spread, spread_fraction,
        max_w, min_w)

    return {
        "counts_per_channel": counts,
        "spread_per_channel": spread,
        "spread_fraction_per_channel": spread_fraction,
        "frac_bins_with_1plus": float(frac_1plus),
        "frac_bins_with_2plus": float(frac_2plus),
        "frac_bins_with_spread_005": float(frac_005),
        "frac_bins_with_spread_015": float(frac_015),
        "frac_bins_with_spread_033": float(frac_033),
        "frac_bins_with_spread_050": float(frac_050),
    }


def _channel_dtype(n_glob: int) -> np.dtype:
    """Determine the smallest unsigned integer dtype that can hold the given
    number of channels.
    """
    return np.uint8 if n_glob <= np.iinfo(np.uint8).max else np.uint16


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
        if mode == "version":
            # No table-level meta section yet; create it
            meta_yaml.append(META_YAML_TYPE_TAG)
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
    l3_flux_corrections = cT.PrerequisiteInput(
        name="l3_flux_corrections",
        doc="Gain maps to apply for flux correction.",
        storageClass="SPHERExImage",
        dimensions=["instrument", "detector"],
        isCalibration=True,
        deferLoad=True,
        multiple=True,
        minimum=6)
    special_ids = cT.PrerequisiteInput(
        name="special_ids",
        doc="List of source IDs to include in the output catalog.",
        storageClass="NumpyArray",
        dimensions=["instrument"],
        isCalibration=False
    )
    allsky_cat = cT.Output(
        name="allsky_cat",
        doc="Filtered all sky catalog per HEALPix.",
        storageClass="PyarrowTable",
        dimensions=["instrument", "hips6"])

    def __init__(self, *, config=None):
        super().__init__(config=config)

        if not config.merge_refcat:
            del self.refcat

        if not config.apply_flux_corrections:
            del self.l3_flux_corrections

        if not config.filter_by_ids:
            del self.special_ids

    def adjustQuantum(self, inputs: dict, outputs: Mapping, label: str, data_id: DataCoordinate):
        """Filter out input reference catalog shards if they do not overlap
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
    """Configuration for CreateAllSkyCatalog task."""

    list_items = ListField(
        doc="Measurements for sources to be recorded in a list format. This is to"
        " be able to store multiple measurements for a single source in"
        " parquet.",
        dtype=str,
        default=("x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",
                 "flux_err_tractor", "flux_bkg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id"))

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

    apply_flux_corrections = Field(
        doc="If True, applies gain ratios to 'flux' and 'flux_err' measurements. Default is False.",
        dtype=bool,
        default=False)

    apply_variance_boost = Field(
        doc="If True, reports the raw Tractor error and a fraction of the flux added in quadrature. Default"
            " is True.",
        dtype=bool,
        default=True)

    var_boost_frac = Field(
        doc="The fraction of the flux to add in quadrature to the raw Tractor error when"
            " `apply_variance_boost` is True. Default is 0.03.",
        dtype=float,
        default=0.03)

    phot_tool_mode = Field(
        doc="If true, this piepline is being run as part of the photometry tool.",
        dtype=bool,
        default=False)

    enable_nyquist_diagnostics = Field(
        doc="If True, compute additional Nyquist diagnostic columns; Default is False.",
        dtype=bool,
        default=False)

    filter_by_ids = Field(
        doc="If True, only write table if 1 or more IDs are in the `special_ids` input list."
            "If False, do not apply filtering by IDs. Default is False.",
        dtype=bool,
        default=False)

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

        if self.apply_variance_boost and (self.var_boost_frac <= 0):
            raise ValueError("To apply the variance boost, the boosted flux fraction must be > 0.")

        if (
            (self.apply_variance_boost and ("flux_err_tractor" not in self.list_items))
            or (not self.apply_variance_boost and ("flux_err_tractor" in self.list_items))
        ):
            raise ValueError("To report both the raw and boosted flux errors, 'flux_err_tractor' must be"
                             " included in 'list_items' and 'apply_variance_boost' must be True.")

        valid_pass_thru = {"coord_src", "spherex_class", "pmra", "pmdec", "parallax", "ra", "dec", "name"}
        for c in self.pass_thru_cols:
            if c not in valid_pass_thru:
                raise ValueError(f"Unsupported pass-through column '{c}' requested. Supported columns are:"
                                 f" {valid_pass_thru}.")


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

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        if not self.config.apply_flux_corrections:
            flux_corr_run = None
            l3_flux_corrections = None
        else:
            # Fetch flux corrections calibration run name for cache key
            flux_corr_handle = butlerQC.get(inputRefs.l3_flux_corrections[0])
            dsref = getattr(flux_corr_handle, "ref", None)
            if dsref is None:
                raise RuntimeError("Could not access DatasetRef from DeferredDatasetHandle.")
            flux_corr_run = getattr(dsref.run, "name", str(dsref.run))
            l3_flux_corrections = butlerQC.get(inputRefs.l3_flux_corrections)

        # Collate data in measurement tables and save output to butler
        result = self.run(
            meas_table_list=butlerQC.get(inputRefs.meas_table_list),
            l3_flux_corrections=l3_flux_corrections,
            flux_corr_run=flux_corr_run,
            refcat=butlerQC.get(inputRefs.refcat) if self.config.merge_refcat else None,
            special_ids=butlerQC.get(inputRefs.special_ids) if self.config.filter_by_ids else None
        )
        if result.allsky_cat is not None:
            butlerQC.put(result.allsky_cat, outputRefs.allsky_cat)

    def run(self, meas_table_list: list[DeferredDatasetHandle],
            l3_flux_corrections: list[DeferredDatasetHandle] = None, flux_corr_run: str = None,
            refcat: DeferredDatasetHandle = None, special_ids: np.ndarray = None) -> pipeBase.Struct:
        """Return a condensed table of all source measurements for one
        HEALPix.

        Parameters
        ----------
        meas_table_list : `list` of `~lsst.daf.butler.DeferredDatasetHandle`
            List of deferred `~astropy.table.Table` with source measurements
            for a HEALPix.
        l3_flux_corrections : `list` of
        `~lsst.daf.butler.DeferredDatasetHandle`, optional
            List of deferred `spx_data.SPHERExImage` gain maps to apply for
            flux correction. Default is None.
        flux_corr_run : `str`, optional
            The run name for the flux corrections. Default is None.
        refcat : `~lsst.daf.butler.DeferredDatasetHandle`, optional
            Reference catalog for cross-matching. Default is None.
        special_ids : `numpy.ndarray`, optional
            Array of source IDs to filter the output catalog. Default is None.

        Returns
        -------
        output: `lsst.pipe.base.Struct`
            Output with component:
            - ``allsky_cat`` : `pyarrow.Table`
                Contains a condensed `pyarrow.Table` with one row per source
                with array columns for multiple measurements.

        Notes
        -----
        Reference catalog flags
        - PHOTOMETRIC_FLAG = 1 << 2  # Bit 2 = 4  Primary calibrators
        - CALSPEC_FLAG = 1 << 3      # Bit 3 = 8  Calspec sources
        - SECONDARY_FLAG = 1 << 8    # Bit 8 = 256  Secondary calibrators

        """
        num_tables = len(meas_table_list)
        self.log.debug(f"Number of input tables: {num_tables}")

        # Prevent silent failure due to mismatch between config and run args
        if self.config.apply_flux_corrections and l3_flux_corrections is None:
            raise ValueError("`apply_flux_corrections` is True but `l3_flux_corrections` input was not"
                             " provided.")
        if self.config.merge_refcat and refcat is None:
            raise ValueError("`merge_refcat` is True but `refcat` input was not provided.")
        if self.config.filter_by_ids and special_ids is None:
            raise ValueError("`filter_by_ids` is True but `special_ids` input was not provided.")

        # If using reference catalog, get the data and prepare indexing
        if self.config.merge_refcat and refcat is not None:
            # Fetch reference catalog data and validate
            refcat = refcat[0].get()
            refcat.keep_columns(
                ["SPHERExRefID"] + list(self.config.pass_thru_cols))  # discard excess cols to save memory
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

        # If filtering by source IDs, pre-sort for efficient lookups in loop
        if self.config.filter_by_ids and special_ids is not None:
            special_ids_sorted = np.sort(special_ids)

        # Fetch data from measurement tables.
        # we collect processed tables as
        # pandas DataFrames and aggregate them all at once after the loop.
        processed_tables = []
        ref_ver = None  # Keep a local copy for reuse when writing parquet metadata
        meas_table = None  # Keep a copy for getting field-level metadata
        colnames = None  # Keep track of column names after possible addition of pass-through columns
        for deferred_meas_table in meas_table_list:
            meas_table = deferred_meas_table.get()

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
            if self.config.merge_refcat and refcat is not None:
                # Find matches
                matched_idx = np.searchsorted(refcat_sorted["SPHERExRefID"], meas_table["source_id"])

                # Double check all matches are valid
                out_of_bounds = (matched_idx >= len(refcat_sorted))  # indices pointing past end of array
                safe_idx = np.clip(matched_idx, 0, len(refcat_sorted) - 1)  # valid range for next check
                mismatched = refcat_sorted["SPHERExRefID"][safe_idx] != meas_table["source_id"]
                if np.any(out_of_bounds | mismatched):
                    missing_in_refcat = list(set(meas_table["source_id"]) - set(refcat_ids))
                    raise ValueError(f"1+ photometered sources were not found in the reference catalog:"
                                     f" {missing_in_refcat}")

                # Add columns
                # NOTE: Double check "spherex_class", which can be pass-through
                #       in Perform Forced Photometry or Create All-Sky Catalog.
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
                                data=refcat_sorted[col][matched_idx],
                                name=col,
                                unit=pass_thru_lookup[col]["unit"],
                                description=pass_thru_lookup[col]["description"],
                                dtype=pass_thru_lookup[col]["dtype"]))
                    else:
                        meas_table.add_column(
                            Column(
                                data=refcat_sorted[col][matched_idx],
                                name=pass_thru_lookup[col]["final_name"],
                                unit=pass_thru_lookup[col]["unit"],
                                description=pass_thru_lookup[col]["description"],
                                dtype=pass_thru_lookup[col]["dtype"]),
                            index=pass_thru_lookup[col]["final_idx"])
                        if col == "spherex_class":
                            meas_table["spherex_class"].format = None

            # Apply filtering efficiently using NumPy bitwise operations
            if self.config.filter_spherex_class > 0:
                meas_table = meas_table[(meas_table["spherex_class"] & self.config.filter_spherex_class) > 0]

            # Apply source ID filtering
            if self.config.filter_by_ids and special_ids is not None:
                ids = meas_table["source_id"].data
                idx = np.searchsorted(special_ids_sorted, ids)
                idx = np.clip(idx, 0, len(special_ids_sorted) - 1)  # prevent out-of-bounds indices
                mask = special_ids_sorted[idx] == ids
                meas_table = meas_table[mask]

            # If no sources remain after filtering, continue to next table
            if len(meas_table) == 0:
                continue

            # Fetch colnames after possible addition of pass-through columns
            # and variance-boosted flux error
            if colnames is None:
                colnames = meas_table.colnames

            # Collect the per-exposure frame. Before converting to pandas,
            # stash the astropy mask for each list column as a companion
            # boolean column so we can replicate main-branch masking semantics
            # exactly: drop masked values, keep unmasked NaNs.
            for item in self.config.list_items:
                col_name = "flux_err" if item == "flux_err_tractor" else item
                col_data = meas_table[col_name].data
                if np.ma.is_masked(col_data):
                    mask_vals = np.ma.getmaskarray(col_data)
                else:
                    mask_vals = np.zeros(len(meas_table), dtype=bool)
                meas_table[f"__mask__{item}"] = mask_vals
            processed_tables.append(meas_table.to_pandas())

        # Free refcat objects immediately after the loop — they are no longer
        # needed
        if self.config.merge_refcat and refcat is not None:
            del refcat, refcat_sorted, refcat_ids, sorted_idx

        # If applying variance boost, update sample astropy table metadata and
        # column names
        if self.config.apply_variance_boost:
            meas_table = meas_table.copy()  # avoid mutating converted pandas DataFrame
            meas_table.rename_column("flux_err", "flux_err_tractor")
            meas_table["flux_err_tractor"].description = "SPHEREx flux error reported by Tractor"
            meas_table.add_column(Column(
                name="flux_err",
                data=np.ones(len(meas_table), dtype=np.float32),  # real values computed on combined DF below
                description="SPHEREx flux error best estimate",
                dtype=meas_table["flux_err_tractor"].dtype,
                unit=meas_table["flux_err_tractor"].unit,
                format=meas_table["flux_err_tractor"].format
            ), index=meas_table.colnames.index("flux_err_tractor") - 1)
            colnames = [c for c in meas_table.colnames if not c.startswith("__mask__")]

        # We build pyarrow ListArrays directly from the dense flat column
        # buffers plus per-source offsets arrays.
        # If `combined` is sorted by [source_id, sort_on],
        # each source's rows are contiguous in the buffer.
        if processed_tables:
            combined = pd.concat(processed_tables, ignore_index=True)
            del processed_tables  # free the per-table frames as soon as we've concatenated

            # Vectorized application of flux corrections and variance boost
            if self.config.apply_flux_corrections and l3_flux_corrections is not None:
                flux_ratios = FluxCorrectionsCube(l3_flux_corrections, run_cache_key=flux_corr_run).cube
                idx_ax0 = combined["det_id"] - 1  # detector
                idx_ax1 = np.ceil(combined["y_image"]-0.5).astype(np.int32)  # y pixel
                idx_ax2 = np.ceil(combined["x_image"]-0.5).astype(np.int32)  # x pixel
                factors = flux_ratios[idx_ax0, idx_ax1, idx_ax2]
                # In photometry tool mode the provided flux corrections are
                # QR2-only (IRSA prod butler ingests QR2 flux corrs only), so
                # restrict corrections to QR2 measurements and leave QR3 (and
                # any other collection) measurements unchanged (factor = 1).
                if self.config.phot_tool_mode:
                    is_qr2 = (
                        combined["data_collection"].astype(str)
                        .str.contains("qr2")
                        .to_numpy()
                    )
                    factors = np.where(is_qr2, factors, 1.0)
                combined["flux"] *= factors
                combined["flux_err"] *= factors
                combined["flux_bkg"] *= factors
            if self.config.apply_variance_boost:
                combined.drop(columns=["__mask__flux_err"], inplace=True)  # avoid orphan mask col
                combined.rename(columns={"flux_err": "flux_err_tractor"}, inplace=True)
                boost = self.config.var_boost_frac * combined["flux"]
                combined.insert(loc=combined.columns.get_loc("flux_err_tractor") - 1, column="flux_err",
                                value=np.sqrt(combined["flux_err_tractor"]**2 + boost**2))
                combined["__mask__flux_err"] = False  # always valid; fxn of valid flux and flux_err_tractor

            # Sort by source_id PRIMARY (so each source's rows are contiguous
            # in the buffer) and by `sort_on` SECONDARY (so within a source
            # the rows are ordered as required). Stable sort preserves the
            # within-source order on equal `sort_on` values.
            combined.sort_values(["source_id", self.config.sort_on], inplace=True, kind="stable")
            combined.reset_index(drop=True, inplace=True)

            list_items = list(self.config.list_items)
            mask_col_names = {item: f"__mask__{item}" for item in list_items}
            scalar_cols_no_id = [c for c in colnames
                                 if c not in self.config.list_items and c != "source_id"]

            # Compute per-source row offsets directly from the sorted
            # source_id column. `offsets_np` has shape (n_sources + 1,) with
            # offsets_np[0] == 0 and offsets_np[-1] == len(combined). Each
            # source occupies rows [offsets_np[i] : offsets_np[i+1]].
            sid_arr = combined["source_id"].to_numpy()
            change_points = np.flatnonzero(np.diff(sid_arr)) + 1
            offsets_np = np.empty(len(change_points) + 2, dtype=np.int32)
            offsets_np[0] = 0
            offsets_np[1:-1] = change_points
            offsets_np[-1] = len(sid_arr)
            n_sources = len(offsets_np) - 1

            # First row of each source -- gives unique source_ids in order
            # AND the value to take for each per-source scalar (refcat-derived
            # columns are constant within a source).
            first_row_idx = offsets_np[:-1]
            uniq_source_ids = sid_arr[first_row_idx]

            # Precompute global channel numbers per detector/pixel up front
            # so the per-source loop below can slice them out (zero-copy numpy
            # views) at effectively no extra allocation cost.
            x_data = combined["x_image"].to_numpy()
            y_data = combined["y_image"].to_numpy()
            det_data = combined["det_id"].to_numpy()

            base_chan_def = LVFChannelDefinitionV2.from_reference(n_subchannels=2)
            half_chan_def = base_chan_def.at_resolution(2)
            n_half_chan_glob = half_chan_def.n_channels_per_array * N_DETECTORS
            combined_half_chan_nums = half_chan_def.get_global_channel_numbers(
                array_ids=det_data, x=x_data, y=y_data, dtype=_channel_dtype(n_half_chan_glob))

            seen = np.full(n_half_chan_glob, False, dtype=bool)
            nyquist_vals = np.empty(n_sources, dtype=np.float32)

            if self.config.enable_nyquist_diagnostics:
                lambda_data = combined["lambda"].to_numpy()
                spec_stats = base_chan_def.spectral_stats
                channel_sizes = spec_stats["BANDWIDTH"].value

                full_chan_def = base_chan_def.at_resolution(1)
                n_full_chan_glob = full_chan_def.n_channels_per_array * N_DETECTORS
                combined_full_chan_nums = full_chan_def.get_global_channel_numbers(
                    array_ids=det_data, x=x_data, y=y_data, dtype=_channel_dtype(n_full_chan_glob))

                # Pre-allocate buffer arrays for Nyquist diagnostics
                counts = np.zeros(n_full_chan_glob, dtype=np.int32)
                spread = np.full(n_full_chan_glob, np.nan, dtype=np.float32)
                spread_fraction = np.full(n_full_chan_glob, np.nan, dtype=np.float32)
                max_w = np.full(n_full_chan_glob, -np.inf, dtype=np.float32)
                min_w = np.full(n_full_chan_glob, np.inf, dtype=np.float32)

                diag_frac_1plus = np.empty(n_sources, dtype=np.float32)
                diag_frac_2plus = np.empty(n_sources, dtype=np.float32)
                diag_spread_005 = np.empty(n_sources, dtype=np.float32)
                diag_spread_015 = np.empty(n_sources, dtype=np.float32)
                diag_spread_033 = np.empty(n_sources, dtype=np.float32)
                diag_spread_050 = np.empty(n_sources, dtype=np.float32)

            for i in range(n_sources):
                meas_half_chan_glob = combined_half_chan_nums[offsets_np[i]:offsets_np[i + 1]]
                nyquist_vals[i] = compute_nyquist_metric(meas_half_chan_glob, seen)
                if self.config.enable_nyquist_diagnostics:
                    wls = lambda_data[offsets_np[i]:offsets_np[i + 1]]
                    meas_full_chan_glob = combined_full_chan_nums[offsets_np[i]:offsets_np[i + 1]]
                    diag = compute_nyquist_diagnostics(base_chan_def, wls, meas_full_chan_glob,
                                                       n_full_chan_glob, channel_sizes, counts, spread,
                                                       spread_fraction, max_w, min_w)
                    diag_frac_1plus[i] = diag["frac_bins_with_1plus"]
                    diag_frac_2plus[i] = diag["frac_bins_with_2plus"]
                    diag_spread_005[i] = diag["frac_bins_with_spread_005"]
                    diag_spread_015[i] = diag["frac_bins_with_spread_015"]
                    diag_spread_033[i] = diag["frac_bins_with_spread_033"]
                    diag_spread_050[i] = diag["frac_bins_with_spread_050"]

            del combined_half_chan_nums, seen

            # Build pyarrow arrays directly, column by column.
            arrays_by_name = {
                "source_id": pa.array(uniq_source_ids, type=pa.int64()),
                "nyquist_comp": pa.array(nyquist_vals, type=pa.float32()),
            }

            if self.config.enable_nyquist_diagnostics:
                arrays_by_name.update({
                    "nyquist_frac_1plus": pa.array(diag_frac_1plus, type=pa.float32()),
                    "nyquist_frac_2plus": pa.array(diag_frac_2plus, type=pa.float32()),
                    "nyquist_spread_005": pa.array(diag_spread_005, type=pa.float32()),
                    "nyquist_spread_015": pa.array(diag_spread_015, type=pa.float32()),
                    "nyquist_spread_033": pa.array(diag_spread_033, type=pa.float32()),
                    "nyquist_spread_050": pa.array(diag_spread_050, type=pa.float32()),
                })

            del nyquist_vals

            if self.config.enable_nyquist_diagnostics:
                del (
                    lambda_data,
                    channel_sizes,
                    combined_full_chan_nums,
                    counts,
                    spread,
                    spread_fraction,
                    max_w,
                    min_w,
                    diag_frac_1plus,
                    diag_frac_2plus,
                    diag_spread_005,
                    diag_spread_015,
                    diag_spread_033,
                    diag_spread_050,
                )

            # Per-source scalar columns: index the flat buffer at the first
            # row of each group. Drop each column from combined immediately
            # after use to reduce peak memory while arrays_by_name grows.
            for col in scalar_cols_no_id:
                col_data = combined[col].to_numpy()
                arrays_by_name[col] = pa.array(col_data[first_row_idx])
                combined.drop(columns=[col], inplace=True)

            # Per-measurement list columns: drop only astropy-masked values,
            # matching main-branch behaviour exactly.
            flux_valid_counts = None  # used below to drop no-flux sources
            for col in list_items:
                col_series = combined[col]
                valid_mask = ~combined[mask_col_names[col]].to_numpy().astype(bool)

                valid_counts = np.add.reduceat(
                    valid_mask.astype(np.int32), offsets_np[:-1]
                )
                if col == "flux":
                    flux_valid_counts = valid_counts

                col_offsets = np.empty(n_sources + 1, dtype=np.int32)
                col_offsets[0] = 0
                col_offsets[1:] = np.cumsum(valid_counts)

                flat_valid = pa.Array.from_pandas(col_series.to_numpy()[valid_mask])
                arrays_by_name[col] = pa.ListArray.from_arrays(
                    pa.array(col_offsets, type=pa.int32()), flat_valid
                )
                combined.drop(columns=[col, mask_col_names[col]], inplace=True)

            # Drop sources that have no valid flux measurements. Such sources
            # carry no science value and would produce confusing rows with
            # empty flux/flux_err lists alongside non-empty lambda/mjd lists.
            keep = flux_valid_counts > 0
            if not np.all(keep):
                n_dropped = int((~keep).sum())
                self.log.debug(f"Dropping {n_dropped} source(s) with no valid flux measurements.")
                keep_indices = pa.array(np.flatnonzero(keep), type=pa.int64())
                arrays_by_name = {
                    name: arr.take(keep_indices)
                    for name, arr in arrays_by_name.items()
                }
                n_sources = int(keep.sum())

            del combined, sid_arr, change_points
        else:
            arrays_by_name = None
            n_sources = 0

        # If no sources were selected, log a warning. The empty-table path
        # is handled below in pyarrow construction.
        if n_sources == 0:
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
        # We pre-sort `combined` by [source_id, sort_on](stable sort) before computing
        # offsets, so the aggregated lists are already in the desired order.
        self.log.debug(f"Array columns are pre-sorted by '{self.config.sort_on}' via offset buckets")

        # Define lists of types for parquet library
        lfloat64 = pa.list_(pa.float64())
        lfloat32 = pa.list_(pa.float32())
        lbool = pa.list_(pa.bool_())
        lint64 = pa.list_(pa.int64())
        luint8 = pa.list_(pa.uint8())

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
            ("ext_flg", pa.bool_()),
            ("flags", lint64),
            ("fit_ql", lfloat32),
            ("deep_flg", lbool),
            ("lvf_id", lint64),
            ("det_id", luint8),
            ("nyquist_comp", pa.float32())
        ]

        if self.config.enable_nyquist_diagnostics:
            pa_flds.extend([
                ("nyquist_frac_1plus", pa.float32()),
                ("nyquist_frac_2plus", pa.float32()),
                ("nyquist_spread_005", pa.float32()),
                ("nyquist_spread_015", pa.float32()),
                ("nyquist_spread_033", pa.float32()),
                ("nyquist_spread_050", pa.float32()),
            ])

        if self.config.phot_tool_mode:
            pa_flds.append(("data_collection", pa.list_(pa.string())))
            # Record numpy string width so the parquet round-trip restores the
            # correct fixed-width dtype.
            metadata["table::len::data_collection"] = str(
                astropy_cols.get("data_collection").dtype.itemsize // 4)

        # If requested, add pass-through columns to fields for pyarrow schema.
        # Guard on colnames so we only insert columns that were actually added
        # to the measurement table (e.g. skip refcat-derived columns when
        # refcat=None was passed to run()).
        if self.config.pass_thru_cols and colnames is not None:
            for col_name in self.config.pass_thru_cols:
                if col_name not in ("ra", "dec"):  # 'ra', 'dec' replace existing columns
                    final_name = pass_thru_lookup[col_name]["final_name"]
                    if final_name not in colnames:
                        continue
                    pa_flds.insert(
                        pass_thru_lookup[col_name]["final_idx"],
                        (final_name, pass_thru_lookup[col_name]["pa_type"]))
                    if col_name in ("external_source_id", "name"):
                        # Add numpy string length to metadata
                        metadata[f"table::len::{col_name}"] = str(
                            astropy_cols.get(col_name).dtype.itemsize // 4)

        # If applying variance boost, add Tractor flux error to pyarrow schema
        if self.config.apply_variance_boost:
            pa_flds.insert(
                pa_flds.index(("flux_err", lfloat32)) + 1,
                ("flux_err_tractor", lfloat32))

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

        # Build the output table from pyarrow arrays
        # directly. Each column array is cast to its declared schema
        # type if needed; thecast is vectorized and effectively free
        # when types already match.
        if arrays_by_name is not None:
            columns = []
            for field in schema:
                arr = arrays_by_name[field.name]
                if arr.type != field.type:
                    arr = arr.cast(field.type)
                columns.append(arr)
            allsky_cat = pa.Table.from_arrays(columns, schema=schema)
            del arrays_by_name, columns
            num_sources = len(allsky_cat)
        else:
            # Empty table path -- no sources matched the filter
            self.log.warning(f"No sources matched the class filter bit value"
                             f" {self.config.filter_spherex_class}.")
            allsky_cat = None
            num_sources = 0

        # Metadata are saved in Butler repository
        self.log.debug(f"Number of sources in final table: {num_sources}")
        self.metadata.add("num_sources", num_sources)
        self.metadata.add("num_tables", num_tables)

        return pipeBase.Struct(allsky_cat=allsky_cat)
