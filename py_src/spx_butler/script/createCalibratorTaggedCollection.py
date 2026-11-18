from lsst.daf.butler import Butler, CollectionType
from lsst.sphgeom import PixelizationABC
from spx_utils.primary_calibrator_info import primary_calibrator_names_and_positions
from spx_butler import ra_dec_to_unit_vector


def get_healpix6_ids(primary_calibrators: dict,
                     pixelization: PixelizationABC,
                     list_tiles: bool = False) -> list:
    """Convert RA/DEC positions of primary calibrators to HEALPix level 6 IDs.

    Parameters
    ----------
    primary_calibrators : `dict`
        Dictionary of calibrator names with (RA, Dec) tuples.
    pixelization : `~lsst.sphgeom.PixelizationABC`
        HEALPix level.
    list_tiles : `bool`
        Flag to print list of healpix tiles.

    Returns
    -------
    `list`
        List of unique HEALPix level 6 IDs.

    """
    healpix_ids = set()
    for name, (ra, dec) in primary_calibrators.items():
        unit_vector = ra_dec_to_unit_vector(ra, dec)  # Convert to UnitVector3d
        healpix_id = int(pixelization.index(unit_vector))  # Convert to HEALPix6 ID
        healpix_ids.add(healpix_id)
        if list_tiles:
            print(f"Calibrator {name}: HEALPix6 ID = {healpix_id}")

    return list(healpix_ids)


def validate_dataset_types(butler, dataset_types):
    """Ensure all dataset types have the hips6 dimension."""
    for ds_type in dataset_types:
        dataset_type = butler.registry.getDatasetType(ds_type)
        if "hips6" not in dataset_type.dimensions:
            raise ValueError(f"Dataset type '{ds_type}' does not have hips6 dimension.")


def createCalibratorTaggedCollection(repo, collections, output_collection, dataset_types, list_tiles=False):
    """Create a tagged collection of datasets of primary calibrator HEALPixel IDs."""
    butler = repo if isinstance(repo, Butler) else Butler(repo, writeable=True)
    commonSkyPix = butler.registry.dimensions.commonSkyPix

    pixelization = commonSkyPix.pixelization  # HealpixPixelization(6)
    name = commonSkyPix.name  # "hips6"

    hips6_tiles = get_healpix6_ids(primary_calibrator_names_and_positions, pixelization, list_tiles)

    validate_dataset_types(butler, dataset_types)

    refs = []
    for ds_type in dataset_types:
        query_results = butler.query_datasets(dataset_type=ds_type,
                                              collections=collections,
                                              data_id={name: hips6_tiles}
                                              )
        refs.extend(query_results)

    if not refs:
        print("No matching datasets found.")
        return

    print(f"Tagging {len(refs)} datasets into collection: {output_collection}")

    butler.registry.registerCollection(output_collection, CollectionType.TAGGED)
    butler.registry.associate(output_collection, refs)
