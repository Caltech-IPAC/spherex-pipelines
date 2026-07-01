# Standard library imports
import logging
import pathlib
from typing import Literal, Optional

# Related third party imports
from astropy.io import votable
from astropy.io.votable.tree import Group, FieldRef, TableElement
from astropy.table import Table
import click

# Local application/library-specific imports
from .exceptions import VOTableSearchError


# Set up logging
logger = logging.getLogger("votable.py")
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter("%(name)s - %(message)s")
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


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


def add_field_metadata(table: TableElement):
    """Add metadata attributes such as precision, ucd, utype, and width to
    existing FIELD elements.

    Parameters
    ----------
    table : `astropy.io.votable.tree.TableElement`
        Multi-spectrum TABLE element with child FIELD elements populated for
        each column.
    """
    meta = {
        "source_id": {
            "ucd": "meta.id",
        },
        "name": {
            "ucd": "meta.id",
        },
        "external_source_id": {
            "ucd": "meta.id",
        },
        "ra": {
            "precision": "7",
            "ucd": "pos.eq.ra;meta.main",
            "width": "11",
        },
        "dec": {
            "precision": "7",
            "ucd": "pos.eq.dec;meta.main",
            "width": "11",
        },
        "lambda": {
            "ucd": "em.wavenumber",
            "utype": "spec:Spectrum.Data.SpectralAxis.Value",
        },
        "lambda_width": {
            "utype": "spec:Spectrum.Data.SpectralAxis.Accuracy.BinSize",
        },
        "flux": {
            "utype": "spec:Spectrum.Data.FluxAxis.Value",
        },
        "flux_err": {
            "utype": "spec:Spectrum.Data.FluxAxis.Accuracy.StatError",
        },
        "flux_interp": {
            "utype": "spec:Spectrum.Data.FluxAxis.Value",
        },
        "flux_interp_err": {
            "utype": "spec:Spectrum.Data.FluxAxis.Accuracy.StatError",
        },
        "flags": {
            "utype": "spec:Spectrum.Data.FluxAxis.Quality",
        }
    }

    # For every potential field, add the relevant metadata attributes
    for field_name, field_meta in zip(list(meta.keys()), list(meta.values())):
        try:
            field = table.get_field_by_id(field_name)
            if "precision" in list(field_meta.keys()):
                field.precision = field_meta["precision"]
            if "ucd" in list(field_meta.keys()):
                field.ucd = field_meta["ucd"]
            if "utype" in list(field_meta.keys()):
                field.utype = field_meta["utype"]
            if "width" in list(field_meta.keys()):
                field.width = field_meta["width"]
        except KeyError:
            # Skip to next field
            continue

    return


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


def convert_to_votable(catalog_path: pathlib.Path, votab_path: pathlib.Path,
                       schema: Literal["tool", "allsky"]):
    """Converts the data in a combined all-sky catalog to VOTable format and
    writes it to an XML file.

    Parameters
    ----------
    catalog_path : `pathlib.Path`
        Path to combined all-sky catalog parquet file.
    votab_path : `pathlib.Path`
        Path to output VOTable containing photometry data.
    schema : {'tool', 'allsky'}
        Specifies which output format to use based on presence of initial
        initial columns. If 'tool', initial columns include 'source_id'
        and 'name'. If 'allsky', initial columns include 'source_id',
        'external_survey', 'external_source_id', and 'spherex_class'.
    """
    # === Read catalog and convert to VOTable format ===
    # Read parquet file
    allsky_cat_combined = Table.read(catalog_path, format="parquet")

    # Order array-valued columns by increasing wavelength
    sort_by_wavelength(allsky_cat_combined)

    # Select which columns from input to include in final VOTable
    # and make types compatible with astropy.io.votable native mapping
    #     S<N> (byte string)  ->  char
    # Basic format/columns
    include_cols = ["ra", "dec", "x_image", "y_image", "mjd", "lambda",
                    "lambda_width", "flux", "flux_err", "flux_bkg",
                    "local_bkg_flg", "ext_flg", "flags", "fit_ql", "deep_flg",
                    "lvf_id", "det_id", "nyquist_comp"]
    # All-sky catalog
    if schema == "allsky":
        tool_cols = ["source_id", "external_survey", "external_source_id",
                     "spherex_class"]
        include_cols = tool_cols + include_cols[:]

        allsky_cat_combined["external_source_id"] = allsky_cat_combined["external_source_id"].astype("S")
    # Spectrophotometry tool
    elif schema == "tool":
        allsky_cols = ["source_id", "name"]
        include_cols = allsky_cols + include_cols[:]

        allsky_cat_combined["name"] = allsky_cat_combined["name"].astype("S")

    allsky_cat_combined = allsky_cat_combined[include_cols]
    votab = votable.from_table(allsky_cat_combined)

    # === Refine and populate XML elements ===
    # Add in VOTable-specific field metadata
    multi_spec_tab = votab.get_first_table()
    add_field_metadata(multi_spec_tab)

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
    array_fields = ["ra", "dec", "x_image", "y_image", "mjd", "flux_bkg", "flags", "fit_ql", "deep_flg",
                    "lvf_id", "det_id"]
    for idx, col in enumerate(array_fields):
        phot_group.entries.append(FieldRef(table=multi_spec_tab, ref=col))

    multi_spec_tab.groups.append(phot_group)

    # Write file
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
                 fake_search_err: Optional[Literal["votab", "closing"]] = None):
    """Makes a VOTable converted from combined all sky catalog data.

    Parameters
    ----------
    catalog_path : `pathlib.Path`
        Path to combined all-sky catalog parquet file.
    votab_path : `pathlib.Path`
        Path to output VOTable containing photometry data.
    schema : {'tool', 'allsky'}, optional
        Specifies which output format to use based on presence of initial
        initial columns. If 'tool', initial columns include 'source_id'
        and 'name'. If 'allsky', initial columns include 'source_id',
        'external_survey', 'external_source_id', and 'spherex_class'. Default
        is 'tool'.
    extra_resource : `pathlib.Path`, optional
        Path to IRSA-provided file containing service descriptor. Default is
        None.
    fake_search_err : {'votab', 'closing'}, optional
        Fakes a search error to test exception handling. Default is None.
    """
    convert_to_votable(catalog_path, votab_path, schema)
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
