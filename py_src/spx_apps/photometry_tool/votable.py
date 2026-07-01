# Standard library imports
import logging
import pathlib
from typing import Literal, Optional
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import io
import os

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
    TD column and return new XML bytes."""
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
    for tr in root.findall(xpath_tr):
        tds = tr.findall(f"{{{ns}}}TD") if ns else tr.findall("TD")
        if col_index < len(tds):
            text = (tds[col_index].text or "").strip()
            # find only the first two ivo://
            matches = []
            for m in re.finditer(r"ivo://", text):
                matches.append(m.start())
                if len(matches) == 2:
                    break
            if len(matches) == 2:
                # Get the width between two ivo://
                width = matches[1] - matches[0]
                break

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


def write_and_patch_votable(votab, votab_path: Path, id_="obs_publisher_did"):
    # Step 1: Write VOTable to memory buffer
    buf = io.BytesIO()
    votable.writeto(votab, buf, tabledata_format="tabledata")

    # Step 2: Patch XML and get new bytes
    patched_xml = patch_field_xml_bytes(buf.getvalue(), id_=id_)

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
    array_cols = ["x_image", "y_image", "mjd", "lambda", "lambda_width", "flux", "flux_err", "flux_bkg",
                  "local_bkg_flg", "flags", "fit_ql", "deep_flg", "lvf_id", "det_id"]
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
    table : `astropy.io.votable.VOTableFile`
        Multi-spectrum VOTable.
    info_str: `str`
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
        "local_bkg_flg": {"ucd": "meta.code"},
        "fit_ql": {"ucd": "meta.code.qual"},
        "lvf_id": {"ucd": "meta.id;obs"},
        "det_id": {"ucd": "meta.id;instr.det"},
        "sersic": {
            "description": "Sersic parameter for radial profile",
            "ucd": "src.morph.param"
        },
        "obs_publisher_did": {"ucd": "meta.ref.ivoid"},
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
            "ucd": "phot.flux",
            "utype": "spec:Spectrum.Data.FluxAxis.Value",
        },
        "flux_err": {
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
                        flux_acc_ref: str = None) -> Group:
    """Creates an XML GROUP element with unified content descriptor type
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
        flux measurement accuracy in the table.

    Returns
    -------
    arr_data_group : `astropy.io.votable.tree.Group`
        XML GROUP element.

    """
    arr_data_group = Group(table=table, ID=ID, utype="ipac:Spectrum.ArrayData")

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

    arr_data_group.entries.append(flux_group)

    return arr_data_group


def add_publisher_did_col(tbl: Table, collection_name: str):
    def publisher_did(v: np.int64) -> str:
        lvf_id_int = int(v)
        d = lvf_id_int % 10
        return ObservationId(lvf_id_int).sps_id + "/D" + str(d)

    publisher_did_vec = np.vectorize(publisher_did)

    lvf_id_col = tbl["lvf_id"]
    deep_flag_col = tbl["deep_flg"]
    prefix = f"ivo://irsa.ipac/{collection_name}"

    obs_publisher_col = []
    for i in range(len(lvf_id_col)):
        lvf_id_strs = publisher_did_vec(lvf_id_col[i])  # array of strings

        n_entries = len(lvf_id_strs)

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


def convert_to_votable(catalog_path: pathlib.Path, votab_path: pathlib.Path,
                       schema: Literal["tool", "allsky"],
                       add_publisher_did: bool = True,
                       publisher_collection: str =
                       "spherex_" + os.getenv("SPXPHOTO_INPUT_COLL",
                                              "IRSA-qr2-visible").split("-")[1],
                       target_table=None):
    """Converts the data in a combined all-sky catalog to VOTable format and
    writes it to an XML file.

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
    publisher_collection : `str`
        Collection name for allsky measurements in 'obs_publisher_did'.
        Collection name for deep field measurements is constructed by adding
        '_deep' to `publisher_collection`.

    """
    # === Read catalog and convert to VOTable format ===
    # Read parquet file

    allsky_cat_combined = Table.read(catalog_path, format="parquet")

    # Order array-valued columns by increasing wavelength
    sort_by_wavelength(allsky_cat_combined)

    add_measurement_count_col(allsky_cat_combined)

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

    include_cols = ["ra", "dec", "n_meas", "nyquist_comp", "ext_flg", "x_image", "y_image", "mjd", "lambda",
                    "lambda_width", "flux", "flux_err", "flux_bkg",
                    "local_bkg_flg", "flags", "fit_ql", "deep_flg",
                    "lvf_id", "det_id", "sersic", "ab_ratio", "phi", "shape_r"]
    # All-sky catalog
    if schema == "allsky":
        tool_cols = ["source_id", "external_survey", "external_source_id",
                     "spherex_class"]
        include_cols = tool_cols + include_cols[:]

        allsky_cat_combined_joined["external_source_id"] = \
            allsky_cat_combined_joined["external_source_id"].astype(
            "S")
    # Spectrophotometry tool
    elif schema == "tool":
        allsky_cols = ["source_id", "name"]
        include_cols = allsky_cols + include_cols[:]

        allsky_cat_combined_joined["name"] = allsky_cat_combined_joined["name"].astype("S")

    allsky_cat_combined_joined = allsky_cat_combined_joined[include_cols]

    if add_publisher_did:
        add_publisher_did_col(allsky_cat_combined_joined, publisher_collection)
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
                                     flux_acc_ref="flux_err")

    # Add FIELDref elements for photometry GROUP
    array_fields = ["ra", "dec", "x_image", "y_image", "mjd", "flux_bkg", "local_bkg_flg",
                    "flags", "fit_ql", "deep_flg",
                    "det_id", "lvf_id"]
    # if add_publisher_did:
    #     array_fields.append("obs_publisher_did")
    for idx, col in enumerate(array_fields):
        phot_group.entries.append(FieldRef(table=multi_spec_tab, ref=col))
    # Add obs_publisher_did last if needed
    if add_publisher_did:
        phot_group.entries.append(FieldRef(table=multi_spec_tab, ref="obs_publisher_did"))

    multi_spec_tab.groups.append(phot_group)

    # Write file with obs_publisher_did
    if add_publisher_did:
        write_and_patch_votable(votab, Path(str(votab_path)))
    else:
        votable.writeto(table=votab, file=str(votab_path), tabledata_format="tabledata")
    logger.info(f"VOTable file created at {str(votab_path)}.")


def add_extra_resource(votab_path: pathlib.Path, extra_resource: pathlib.Path,
                       fake_search_err: Optional[Literal["votab", "closing"]] = None):
    """Inserts IRSA-provided XML content after the first 'VOTABLE' element
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
    """Makes a VOTable converted from combined all sky catalog data.

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
    fake_search_err : {'votab', 'closing'}, optional
        Fakes a search error to test exception handling. Default is None.
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
    make_votable(catalog_path, votab_path, schema, extra_resource, fake_search_err)


if __name__ == "__main__":
    votable_cli()
