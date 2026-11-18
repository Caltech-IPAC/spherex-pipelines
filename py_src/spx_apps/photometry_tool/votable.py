# Standard library imports
import logging
import pathlib
from typing import Literal, Optional
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import io

# Related third party imports
import numpy as np
from astropy.io import votable
from astropy.io.votable.tree import Group, FieldRef, TableElement
from astropy.table import Table, Column
from astropy.table import join
import click
from astropy.io.votable import tree


# Local application/library-specific imports
from .exceptions import VOTableSearchError

from spx_data import ObservationId


# Set up logging
logger = logging.getLogger("votable.py")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def add_measurement_count_col(tbl: Table):
    """Add a column giving the number of measurements per source.

    Parameters
    ----------
    tbl : `astropy.table.Table`
        Input table with array-valued columns (e.g., flux, lambda).

    """
    if "flux" not in tbl.colnames:
        raise ValueError("Expected 'flux' column in table to count measurements.")

    counts = [len(row["flux"]) for row in tbl]
    tbl.add_column(
        Column(
            data=counts,
            name="n_meas",
            description="Number of measurements returned for this source",
            dtype="int32"
        ),
        copy=False,
    )


def _ns_of(root):
    m = re.match(r"\{(.*)\}", root.tag)
    return m.group(1) if m else None


def patch_field_xml_bytes(xml_bytes, id_="obs_publisher_did"):
    """Patch the FIELD by auto-detecting width from its
    TD column and return new XML bytes.
    """
    tree = ET.ElementTree(ET.fromstring(xml_bytes))
    root = tree.getroot()
    ns = _ns_of(root)

    # find all FIELDS
    xpath_field = f".//{{{ns}}}FIELD" if ns else ".//FIELD"
    fields = root.findall(xpath_field)

    # locate column index of the target FIELD
    col_index = None
    for i, fld in enumerate(fields):
        if fld.get("ID") == id_:
            col_index = i
            target_field = fld
            break

    if col_index is None:
        raise ValueError(f"FIELD with ID={id_} not found.")

    # find all TR rows under TABLEDATA
    xpath_tr = f".//{{{ns}}}TR" if ns else ".//TR"
    width = None
    fallback_width = None
    for tr in root.findall(xpath_tr):
        tds = tr.findall(f"{{{ns}}}TD") if ns else tr.findall("TD")
        if col_index < len(tds):
            text = (tds[col_index].text or "")
            matches = [m.start() for m in re.finditer(r"ivo://", text)]
            if len(matches) >= 2:
                width = matches[1] - matches[0]
                break
            elif len(matches) == 1 and fallback_width is None:
                # single entry: the whole (unstripped) cell is one stride
                fallback_width = len(text)
    if width is None:
        width = fallback_width

    if width is None:
        raise ValueError(f"Could not determine width from <TD> values in column {col_index}.")

    # patch the target FIELD
    target_field.set("arraysize", f"{width}x*")
    target_field.set("datatype", "char")
    target_field.set("name", id_)

    if ns:
        ET.register_namespace("", ns)

    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue()

def write_and_patch_votable(votab,
                            votab_path: Path, ids: list[str] = ["obs_publisher_did", "data_collection"]):
    """Write a VOTable to file, patching the specified field's arraysize."""
    # Step 1: Write VOTable to memory buffer
    buf = io.BytesIO()
    votable.writeto(votab, buf, tabledata_format="tabledata")

    # Step 2: Patch XML and get new bytes
    # For array valued columns
    patched_xml = buf.getvalue()
    for id_ in ids:
        patched_xml = patch_field_xml_bytes(patched_xml, id_=id_)

    # Step 3: Write patched XML to the actual file path
    with open(votab_path, "wb") as f:
        f.write(patched_xml)


def sort_by_wavelength(input_cat: Table):
    """Sorts multiple measurements in array-valued columns in order of
    increasing wavelength.

    Parameters
    ----------
    input_cat : `astropy.table.Table`
        Table of combined all sky catalog measurements.

    """
    array_cols = ["x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err",  "flux_bkg",
                  "flux_err_tractor", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id"]
    for idx, row in enumerate(input_cat):
        # Make table for individual spectrum/source
        spec_tab = Table()
        for col in array_cols:
            spec_tab[col] = row[col]
        # Sort all columns by wavelength
        spec_tab.sort(keys="lambda")
        # Populate array-valued columns in larger catalog with sorted values
        for col in array_cols:
            input_cat[col][idx] = spec_tab[col].data


def add_field_metadata(votable, info_str):
    """Add metadata attributes such as precision, ucd, utype, and width to
    existing FIELD elements and an source coverage status INFO header.

    Parameters
    ----------
    votable : `astropy.io.votable.VOTableFile`
        Multi-spectrum VOTable.
    info_str : `str`
        String containing information regarding source coverage.

    """
    field_overrides = {
        "nyquist_comp": {
            "ucd": "meta.code.qual"
        },
        "ext_flg": {
            "ucd": "src.morph"
        },
        "x_image": {"ucd": "pos;instr.pixel"},
        "y_image": {"ucd": "pos;instr.pixel"},
        "mjd": {"ucd": "time.epoch"},
        "flux_bkg": {"ucd": "instr.skyLevel;phot.flux.sb"},
        "fit_ql": {"ucd": "meta.code.qual"},
        "lvf_id": {"ucd": "meta.id;obs"},
        "det_id": {"ucd": "meta.id;instr.det"},
        "sersic": {
            "description": "Sersic parameter for radial profile",
            "ucd": "src.morph.param"
        },
        "obs_publisher_did": {"ucd": "meta.ref.ivoid"},
        "data_collection": {
            "ucd": "meta.version;obs",
            "description": "Abbreviated form of the obs_collection containing the data for the measurement",
        },
        "ab_ratio": {
            "description": "Major/minor axis ratio",
            "ucd": "src.morph.param;phys.size.axisRatio"

        },
        "source_id": {
            "ucd": "meta.id;meta.main",
            "description": "Unique id of the target in this job",
        },
        "name": {
            "ID": "label",
            "name": "label",
            "description": "User-specified label of the target ('Target_NNNN' if no name provided)",
            "ucd": "meta.id"
        },
        "external_source_id": {
            "description": "Unique id of the target in this job",
            "ucd": "meta.id",
        },
        "ra": {
            "description": "R.A. of the target",
            "precision": "7",
            "ucd": "pos.eq.ra;meta.main",
            "width": "11",
        },
        "dec": {
            "description": "DEC of the target",
            "precision": "7",
            "ucd": "pos.eq.dec;meta.main",
            "width": "11",
        },
        "lambda": {
            "ucd": "em.wl",
            "utype": "spec:Spectrum.Data.SpectralAxis.Value",
        },
        "lambda_width": {
            "ucd": "em.wl;instr.bandwidth",
            "utype": "spec:Spectrum.Data.SpectralAxis.Accuracy.BinSize",
        },
        "flux": {
            "ucd": "phot.flux;meta.main",
            "utype": "spec:Spectrum.Data.FluxAxis.Value",
        },
        "flux_err": {
            "ucd": "stat.error;phot.flux;meta.main",
            "utype": "spec:Spectrum.Data.FluxAxis.Accuracy.StatError",
        },
        "flux_err_tractor": {
            "ucd": "stat.error;phot.flux",
            "utype": "spec:Spectrum.Data.FluxAxis.Accuracy.StatError",
        },
        "flux_interp": {
            "ucd": "phot.flux",
            "utype": "spec:Spectrum.Data.FluxAxis.Value",
        },
        "flux_interp_err": {
            "ucd": "stat.error;phot.flux",
            "utype": "spec:Spectrum.Data.FluxAxis.Accuracy.StatError",
        },
        "flags": {
            "ucd": "meta.code.qual",
            "utype": "spec:Spectrum.Data.FluxAxis.Quality",
        },
        "n_meas": {
            "ucd": "meta.number;obs",
        },
        "phi": {
            "unit": "deg",
            "description": "Position angle",
            "ucd": "src.morph.param;pos.posAng"
        },
        "shape_r": {
            "unit": "arcsec",
            "description": "Effective radius",
            "ucd": "src.morph.param;phys.size.axisRadius"
        }
    }

    for resource in votable.resources:
        for table in resource.tables:
            for field in table.fields:
                current_id = field.ID
                if current_id in field_overrides:
                    overrides = field_overrides[current_id]
                    for attr, value in overrides.items():
                        if hasattr(field, attr):
                            setattr(field, attr, value)
                        else:
                            logging.warning(f"Warning: field '{current_id}' has no attribute '{attr}'")
                    # Handle ID change explicitly
                    if "ID" in overrides and overrides["ID"] != current_id:
                        field.ID = overrides["ID"]

    if votable.resources:
        info = tree.Info(ID="source_coverage", name="SourceCoverage", value=info_str)
        votable.resources[0].infos.append(info)
    return votable


def spec_arr_data_group(table: TableElement, ID: str, spec_ref: str, flux_ref: str, spec_acc_ref: str = None,
                        flux_acc_ref: str = None, flux_tractor_acc_ref: str = None) -> Group:
    """Create an XML GROUP element with unified content descriptor type
    '"ipac:Spectrum.ArrayData"'.

    Parameters
    ----------
    table : `astropy.io.votable.tree.TableElement`
        Parent XML TABLE element with unified content descriptor type
        '"ipac:MultiSpectrum"'.
    ID : `str`
        ID attribute value.
    spec_ref : `str`
        ID to reference an `astropy.io.votable.tree.Field` element representing
        spectral measurement data in the table.
    flux_ref : `str`
        ID to reference an `astropy.io.votable.tree.Field` element representing
        flux measurement data in the table.
    spec_acc_ref : `str` (default is None)
        ID to reference an `astropy.io.votable.tree.Field` element representing
        spectral measurement accuracy in the table.
    flux_acc_ref : `str` (default is None)
        ID to reference an `astropy.io.votable.tree.Field` element representing
        flux measurement accuracy in the table, boosted to account for an
        empirically-determined additional flux-dependent source of variance.
    flux_tractor_acc_ref : `str` (default is None)
        ID to reference an `astropy.io.votable.tree.Field` element representing
        the original Tractor-computed flux error variance in the table.

    Returns
    -------
    arr_data_group : `astropy.io.votable.tree.Group`
        XML GROUP element.

    """
    arr_data_group = Group(table=table, ID=ID, utype="ipac:Spectrum.ArrayData")

    # Append the spectral axis GROUP before the flux axis GROUP so
    # the photometry GROUP presents the SpectralAxis GROUP first, then the
    # FluxAxis GROUP.
    # Add child spectral axis GROUP element
    spec_group = Group(table=table, utype="spec:Spectrum.Data.SpectralAxis")
    spec_group.entries.append(FieldRef(table=table, ref=spec_ref))

    if spec_acc_ref is not None:
        spec_acc_group = Group(table=table, utype="spec:Spectrum.Data.SpectralAxis.Accuracy")
        spec_acc_group.entries.append(FieldRef(table=table, ref=spec_acc_ref))
        spec_group.entries.append(spec_acc_group)

    arr_data_group.entries.append(spec_group)

    # Add child flux axis GROUP element
    flux_group = Group(table=table, utype="spec:Spectrum.Data.FluxAxis")
    flux_group.entries.append(FieldRef(table=table, ref=flux_ref))

    if flux_acc_ref is not None:
        flux_acc_group = Group(table=table, utype="spec:Spectrum.Data.FluxAxis.Accuracy")
        flux_acc_group.entries.append(FieldRef(table=table, ref=flux_acc_ref))
        flux_group.entries.append(flux_acc_group)

    if flux_acc_ref is not None:
        flux_acc_group = Group(table=table, utype="spec:Spectrum.Data.FluxAxis.Accuracy")
        flux_acc_group.entries.append(FieldRef(table=table, ref=flux_tractor_acc_ref))
        flux_group.entries.append(flux_acc_group)

    arr_data_group.entries.append(flux_group)

    return arr_data_group


def add_publisher_did_col(tbl: Table):
    """Add a publisher DID column to the table based on LVF IDs."""
    def publisher_did(v: np.int64) -> str:
        lvf_id_int = int(v)
        d = lvf_id_int % 10
        return ObservationId(lvf_id_int).sps_id + "/D" + str(d)

    publisher_did_vec = np.vectorize(publisher_did)

    lvf_id_col = tbl["lvf_id"]
    deep_flag_col = tbl["deep_flg"]
    data_collection_col = tbl["data_collection"]

    obs_publisher_col = []
    for i in range(len(lvf_id_col)):
        lvf_id_strs = publisher_did_vec(lvf_id_col[i])  # array of strings

        n_entries = len(lvf_id_strs)

        # per-measurement prefix: ivo://irsa.ipac/{data_collection}
        prefix = np.char.add("ivo://irsa.ipac/", np.asarray(data_collection_col[i]).astype(str))

        # pre-suffix array: "?" or "_deep?"
        pre_suffix = np.full(n_entries, "?", dtype="U10")
        pre_suffix[deep_flag_col[i]] = "_deep?"

        # padding suffix array
        suffix = np.full(n_entries, " " * (1 + len("_deep")), dtype="U10")
        suffix[deep_flag_col[i]] = " "  # deep gets only one space

        # combine in vectorized manner
        dids = np.char.add(prefix, pre_suffix)
        dids = np.char.add(dids, lvf_id_strs)
        dids = np.char.add(dids, suffix)

        obs_publisher_col.append("".join(dids).encode("ascii"))

    tbl.add_column(
        Column(
            data=obs_publisher_col,
            name="obs_publisher_did",
            description="Publisher dataset identifier in IRSA archive",
            unit=None,
        ),
        copy=False,
    )


def add_data_collection_col(tbl: Table):
    """Pack the data_collection list column into a fixed-width byte string
    per source, mirroring the obs_publisher_did encoding.
    """
    packed_col = []
    for row in tbl["data_collection"]:
        packed = "".join(v + " " for v in row)  # "6.4+psffix11 6.4+psffix11 .."
        packed_col.append(packed.encode("ascii"))

    tbl.remove_column("data_collection")
    tbl.add_column(
        Column(
            data=packed_col,
            name="data_collection",
            description="Abbreviated form of the obs_collection containing the data for the measurement",
            unit=None,
        ),
        copy=False,
    )



def convert_to_votable(catalog_path: pathlib.Path, votab_path: pathlib.Path,
                       schema: Literal["tool", "allsky"],
                       add_publisher_did: bool = True,
                       target_table=None):
    """Convert the data in a combined all-sky catalog to VOTable format and
    write it to an XML file.

    Parameters
    ----------
    catalog_path : `pathlib.Path`
        Path to combined all-sky catalog parquet file.
    votab_path : `pathlib.Path`
        Path to output VOTable containing photometry data.
    schema : {'tool', 'allsky'}
        Specifies which output format to use based on presence of
        initial columns. If 'tool', initial columns include 'source_id'
        and 'name'. If 'allsky', initial columns include 'source_id',
        'external_survey', 'external_source_id', and 'spherex_class'.
    add_publisher_did : `bool`
        If True, adds 'obs_publisher_did' field, corresponding to each lvf id.
    target_table : `astropy.table.Table`, optional
        Target table with elliptical source parameters to join.

    """
    # === Read catalog and convert to VOTable format ===
    # Read parquet file

    allsky_cat_combined = Table.read(catalog_path, format="parquet")

    # Order array-valued columns by increasing wavelength
    sort_by_wavelength(allsky_cat_combined)

    add_measurement_count_col(allsky_cat_combined)
    # print(allsky_cat_combined.meta["data_collection"])
    if target_table is not None:
        # Columns to bring from target_table
        cols_to_add = ["sersic", "ab_ratio", "phi", "shape_r"]
        join_cols = ["name"] + cols_to_add
        # Perform an inner join on 'name'
        allsky_cat_combined_joined = join(
            allsky_cat_combined,
            target_table[join_cols],
            keys="name",
            join_type="inner"  # only keep matching names
        )
        unmatched_names = list(set(target_table["name"]) - set(allsky_cat_combined_joined["name"]))

        for col in cols_to_add:
            # Replace -9999 with np.nan for elliptical source parameters
            allsky_cat_combined_joined[col] = \
                [np.nan if v == -9999 else v for v in allsky_cat_combined_joined[col]]
    else:
        # If we are not doing matching, leave as is.
        allsky_cat_combined_joined = allsky_cat_combined
    # Select which columns from input to include in final VOTable
    # and make types compatible with astropy.io.votable native mapping
    #     S<N> (byte string)  ->  char
    # Basic format/columns

    # Non-array (scalar) columns come first, then the array-valued
    # columns, controlling FIELD declaration order in the VOTable. The schema-
    # specific scalar ID columns (e.g. source_id, name) are folded into the
    # scalar ordering rather than appended separately. The array columns are
    # ordered to match the photometry GROUP presentation order (SpectralAxis
    # fields, then FluxAxis fields, then the remainder) so the FIELD
    # declarations and the GROUP read in the same order.
    array_cols = ["lambda", "lambda_width",
                  "flux", "flux_err", "flux_err_tractor",
                  "mjd", "data_collection", "deep_flg", "det_id", "flags",
                  "fit_ql", "flux_bkg", "x_image", "y_image", "lvf_id"]
    # All-sky catalog
    if schema == "allsky":
        # Scalar columns first, in the requested order.
        scalar_cols = ["source_id", "external_survey", "external_source_id",
                       "spherex_class", "ra", "dec", "n_meas", "nyquist_comp",
                       "ext_flg", "shape_r", "sersic", "ab_ratio", "phi"]
        include_cols = scalar_cols + array_cols

        allsky_cat_combined_joined["external_source_id"] = \
            allsky_cat_combined_joined["external_source_id"].astype(
            "S")
    # Spectrophotometry tool
    elif schema == "tool":
        # Scalar columns first, in the requested order
        # (source_id, name(->label), ra, dec, n_meas, nyquist_comp, ext_flg,
        # shape_r, sersic, ab_ratio, phi).
        scalar_cols = ["source_id", "name", "ra", "dec", "n_meas",
                       "nyquist_comp", "ext_flg", "shape_r", "sersic",
                       "ab_ratio", "phi"]
        include_cols = scalar_cols + array_cols

        allsky_cat_combined_joined["name"] = allsky_cat_combined_joined["name"].astype("S")
    allsky_cat_combined_joined = allsky_cat_combined_joined[include_cols]

    if add_publisher_did:
        add_publisher_did_col(allsky_cat_combined_joined)


    add_data_collection_col(allsky_cat_combined_joined)

    # Reorder the table after add_publisher_did_col/add_data_collection_col
    # run so that all scalar columns (in their requested order) come first and
    # the array-valued columns (including obs_publisher_did) follow.
    array_order = [c for c in array_cols] + (["obs_publisher_did"] if add_publisher_did else [])
    scalar_order = [c for c in allsky_cat_combined_joined.colnames
                    if c not in array_order]
    allsky_cat_combined_joined = allsky_cat_combined_joined[scalar_order + array_order]

    votab = votable.from_table(allsky_cat_combined_joined)
    # === Refine and populate XML elements ===
    # Add in VOTable-specific field metadata

    if target_table is not None and len(unmatched_names) > 0:
        info_str = f"No matching SPHEREx coverage was found for \
            {len(unmatched_names)} further target(s) in this request: " + ",".join(
            unmatched_names)
    else:
        info_str = "All user provided sources have SPHEREx coverage."
    add_field_metadata(votab, info_str)

    multi_spec_tab = votab.get_first_table()

    # Add results RESOURCE element and TABLE element attributes
    results_rsrc = votab.resources[0]
    results_rsrc.type = "results"
    results_rsrc.utype = "ipac:MultiSpectrum"
    multi_spec_tab.utype = "ipac:MultiSpectrum"

    # Add photometry GROUP element
    phot_group = spec_arr_data_group(table=multi_spec_tab, ID="photometry", spec_ref="lambda",
                                     flux_ref="flux", spec_acc_ref="lambda_width",
                                     flux_acc_ref="flux_err", flux_tractor_acc_ref="flux_err_tractor")

    # Add the remaining FIELDref elements for the photometry GROUP in
    # the requested presentation order (drives Firefly column order). These
    # follow the SpectralAxis and FluxAxis sub-GROUPs.
    array_fields = ["mjd", "data_collection", "deep_flg", "det_id", "flags",
                    "fit_ql", "flux_bkg", "x_image", "y_image", "ra", "dec",
                    "lvf_id"]
    # obs_publisher_did goes last, only when present
    if add_publisher_did:
        array_fields = array_fields + ["obs_publisher_did"]
    for idx, col in enumerate(array_fields):
        phot_group.entries.append(FieldRef(table=multi_spec_tab, ref=col))
    multi_spec_tab.groups.append(phot_group)

    patch_ids = ["data_collection"]
    # Write file with obs_publisher_did
    if add_publisher_did:
        patch_ids.append("obs_publisher_did")
    write_and_patch_votable(votab, Path(str(votab_path)), patch_ids)
    logger.info(f"VOTable file created at {str(votab_path)}.")


def add_extra_resource(votab_path: pathlib.Path, extra_resource: pathlib.Path,
                       fake_search_err: Optional[Literal["votab", "closing"]] = None):
    """Insert IRSA-provided XML content after the first 'VOTABLE' element
    in the photometry tool output VOTable file.

    Parameters
    ----------
    votab_path : `pathlib.Path`
        Path to VOTable file containing photometry data.
    extra_resource : `pathlib.Path`
        Path to IRSA-provided file containing service descriptor.
    fake_search_err : {'votab', 'closing'}, optional
        Fakes a search error to test exception handling.

    """
    with open(votab_path, "r", encoding="utf-8") as file:
        photometry_xml = file.read()

    with open(extra_resource, "r", encoding="utf-8") as file:
        meta_resource = file.read()

    if fake_search_err == "votab":
        photometry_xml = photometry_xml.replace("VOTAB", "RESOURCE")
    elif fake_search_err == "closing":
        photometry_xml = photometry_xml.replace(">", "*")

    # Find first "<VOTAB" occurrence
    opening_tag_idx = photometry_xml.find("<VOTAB")
    if opening_tag_idx != -1:
        # Find the closing tag associated with VOTAB element
        closing_tag_idx = photometry_xml.find(">", opening_tag_idx)
        if closing_tag_idx != -1:
            updated_xml = (photometry_xml[:closing_tag_idx+1] + "\n" + meta_resource
                           + photometry_xml[closing_tag_idx+1:])
            with open(votab_path, "w") as file:
                file.write(updated_xml)
            logger.info(f"Service descriptor added to VOTable file {votab_path}.")
        else:
            raise VOTableSearchError(substring=">", start_idx=opening_tag_idx)
    else:
        raise VOTableSearchError(substring="<VOTAB", start_idx=0)


def make_votable(catalog_path: pathlib.Path, votab_path: pathlib.Path,
                 schema: Literal["tool", "allsky"] = "tool", extra_resource: pathlib.Path = None,
                 add_publisher_did: bool = True,
                 fake_search_err: Optional[Literal["votab", "closing"]] = None, target_table=None):
    """Make a VOTable converted from combined all sky catalog data.

    Parameters
    ----------
    catalog_path : `pathlib.Path`
        Path to combined all-sky catalog parquet file.
    votab_path : `pathlib.Path`
        Path to output VOTable containing photometry data.
    schema : {'tool', 'allsky'}, optional
        Specifies which output format to use based on presence of initial
        columns. If 'tool', initial columns include 'source_id'
        and 'name'. If 'allsky', initial columns include 'source_id',
        'external_survey', 'external_source_id', and 'spherex_class'. Default
        is 'tool'.
    extra_resource : `pathlib.Path`, optional
        Path to IRSA-provided file containing service descriptor. Default is
        None.
    add_publisher_did : `bool`, optional
        If True, adds 'obs_publisher_did' field. Default is True.
    fake_search_err : {'votab', 'closing'}, optional
        Fakes a search error to test exception handling. Default is None.
    target_table : `astropy.table.Table`, optional
        Target table with elliptical source parameters to join.

    """
    convert_to_votable(catalog_path, votab_path, schema,
                       add_publisher_did=add_publisher_did, target_table=target_table)
    if extra_resource is not None:
        add_extra_resource(votab_path, extra_resource, fake_search_err)


@click.command()
@click.option("--catalog_path", help="Path to combined all sky catalog file", required=True,
              type=click.Path(exists=True, dir_okay=False, readable=True,
                              path_type=pathlib.Path))
@click.option("--votab_path", help="Path to output VOTable file", required=True,
              type=click.Path(path_type=pathlib.Path))
@click.option("--schema", help="Specifies which output format to use based on presence of initial columns",
              default="tool", type=click.Choice(["tool", "allsky"]))
@click.option("--extra_resource", help="Path to IRSA-provided file containing service descriptor",
              default=None, type=click.Path(exists=True, dir_okay=False, readable=True,
                                            path_type=pathlib.Path))
@click.option("--fake_search_err", help="Fake a search error when incorporating the service descriptor "
              "(for testing exception handling only)", default=None, type=click.Choice(["votab", "closing"]))
def votable_cli(catalog_path, votab_path, schema, extra_resource, fake_search_err):
    """Run the VOTable conversion CLI command."""
    make_votable(catalog_path, votab_path, schema, extra_resource, fake_search_err)


if __name__ == "__main__":
    votable_cli()
