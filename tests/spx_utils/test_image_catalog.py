from unittest.mock import MagicMock
import numpy as np
from astropy.table import Table
from spx_data import SPHERExImage
from spx_utils import ImageCatalogIterable, SourcePipelineClass


def test_image_catalog():
    def make_dataId(spsplan, slewgroup, pair, counter=None):
        d = {"spsplan": spsplan, "slewgroup": slewgroup, "dichroicpair": pair}
        if counter is not None:
            d["slewgroup_counter"] = counter
        return d

    # -----------------------
    # Create 2 images sharing the same catalog
    # -----------------------
    shared_data_id = (100, 10, 1)
    image_handles = []
    for c in [0, 1]:  # Two images with different slewgroup_counter
        img = MagicMock(spec=SPHERExImage)
        img.wcs.world_to_pixel_values.return_value = (np.array([42.0]), np.array([24.0]))
        img.shape = (64, 64)
        img_handle = MagicMock()
        img_handle.dataId = make_dataId(*shared_data_id, counter=c)
        img_handle.get.return_value = img
        image_handles.append(img_handle)

    # -----------------------
    # Create matching catalog
    # -----------------------
    matching_catalog = Table({
        "ra": [0.0], "dec": [0.0],
        "spherex_class": [int(SourcePipelineClass.UNDEFINED)]
    })
    matching_cat_handle = MagicMock()
    matching_cat_handle.dataId = make_dataId(*shared_data_id)
    matching_cat_handle.get.return_value = matching_catalog

    # -----------------------
    # Create extra non-matching catalogs with lower IDs
    # -----------------------
    unmatched_handles = []
    unmatched_ids = [
        (1, 10, 1),   # spsplan mismatch
        (100, 1, 1),  # slewgroup mismatch
        (100, 10, 0)  # dichroicpair mismatch
    ]
    for i, bad_id in enumerate(unmatched_ids):
        bad_cat = Table({
            "ra": [1.0 + i], "dec": [2.0 + i],
            "spherex_class": [0]
        })
        bad_handle = MagicMock()
        bad_handle.dataId = make_dataId(*bad_id)
        bad_handle.get.return_value = bad_cat
        unmatched_handles.append(bad_handle)

    # Unordered: unmatched first, then matching
    catalog_handles = unmatched_handles + [matching_cat_handle]

    # -----------------------
    # Run ImageCatalogIterable
    # -----------------------
    iterable = ImageCatalogIterable(
        imageHandles=image_handles,
        catalogHandles=catalog_handles,
        srcPipeFlag=SourcePipelineClass.UNDEFINED,
        addPixelCoords=True,
        pixelCoordsWithDistortion=False,
        filterSourcesOutsideImage=True
    )

    results = list(iterable)

    # -----------------------
    # Assertions
    # -----------------------
    assert len(results) == 2  # One per image
    for img, cat in results:
        assert isinstance(cat, Table)
        assert len(cat) == 1
        assert "ix" in cat.colnames
        assert "iy" in cat.colnames
        assert cat["ix"][0] == 42.0
        assert cat["iy"][0] == 24.0


def test_all_definition():
    """Tests that the manually defined `ALL` member includes all previously
    defined members.
    """
    for name, member in SourcePipelineClass.__members__.items():
        if name != "ALL":
            msg = f"Member {member} is missing from the manually-defined `ALL` member of "\
                  "`SourcePipelineClass`."
            # use == member instead of > 0 since UNDEFINED is 0
            assert (SourcePipelineClass.ALL & member == member), msg


def test_flag_names():
    """Tests fetching list of flag names for a `SourcePipelineClass` object."""
    combo_flg = SourcePipelineClass.ICES | SourcePipelineClass.STELLAR
    combo_flg_names = combo_flg.get_names_as_list()
    assert combo_flg_names == ["ICES", "STELLAR"]

    solo_flg = SourcePipelineClass.ICES
    solo_flg_names = solo_flg.get_names_as_list()
    assert solo_flg_names == ["ICES"]

    no_flg = SourcePipelineClass.UNDEFINED
    no_flg_names = no_flg.get_names_as_list()
    assert no_flg_names == ["UNDEFINED"]
