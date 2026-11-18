import pytest
from unittest.mock import MagicMock, ANY
from lsst.daf.butler import CollectionType, Butler
from click.testing import CliRunner
from unittest.mock import patch
from lsst.sphgeom import HealpixPixelization
from spx_butler.cli.cmd.commands import create_tagged_primary_calibrators
from spx_butler.script.createCalibratorTaggedCollection import (
    get_healpix6_ids,
    validate_dataset_types,
    createCalibratorTaggedCollection
)


@pytest.fixture
def runner():
    """Provide a Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_primary_calibrators():
    """Fixture providing sample primary calibrators with RA/Dec values."""
    return {
        "test_star_1": (10.0, 20.0),
        "test_star_2": (150.0, -10.0),
        "test_star_3": (300.0, 60.0),
    }


def test_get_healpix6_ids(sample_primary_calibrators):
    """Test HEALPix6 ID computation from RA/Dec."""
    healpix6_ids = get_healpix6_ids(sample_primary_calibrators, HealpixPixelization(6))

    # Ensure a list of IDs is returned
    assert isinstance(healpix6_ids, list)
    assert len(healpix6_ids) == len(sample_primary_calibrators)

    # Ensure each ID is an integer
    assert all(isinstance(hid, int) for hid in healpix6_ids)


def test_validate_dataset_types():
    """Test dataset type validation in Butler."""
    mock_butler = MagicMock()
    mock_dataset_type = MagicMock()
    mock_dataset_type.dimensions = {"instrument", "hips6"}  # Mock dataset with hips6
    mock_butler.registry.getDatasetType.return_value = mock_dataset_type

    validate_dataset_types(mock_butler, ["test_dataset"])  # Should not raise error

    # Now test with a dataset type missing "hips6"
    mock_dataset_type.dimensions = {"instrument"}
    mock_butler.registry.getDatasetType.return_value = mock_dataset_type

    with pytest.raises(ValueError, match="does not have hips6 dimension"):
        validate_dataset_types(mock_butler, ["test_dataset"])


def test_createCalibratorTaggedCollection():
    """Test that datasets are queried and tagged collection is created."""
    mock_butler = MagicMock(spec=Butler)  # Create a proper Butler mock
    mock_dataset_type = MagicMock()
    mock_dataset_type.dimensions = {"instrument", "hips6"}
    mock_butler.registry.getDatasetType.return_value = mock_dataset_type

    # Mock dataset references returned from query_datasets
    mock_query_results = ["dataset1", "dataset2"]
    mock_butler.query_datasets.return_value = mock_query_results  # Mock new query system

    # Mock `commonSkyPix` retrieval
    mock_commonSkyPix = MagicMock()
    mock_commonSkyPix.pixelization = MagicMock()
    mock_commonSkyPix.name = "hips6"
    mock_commonSkyPix.level = 6
    mock_butler.registry.dimensions.commonSkyPix = mock_commonSkyPix  # Mock the HEALPix config

    dataset_types = ["test_type"]
    input_collections = ["test_collection"]
    output_collection = "tagged_collection"

    # Call function with a real mock instance, not a path
    createCalibratorTaggedCollection(mock_butler, input_collections, output_collection, dataset_types)

    # Ensure datasets were queried with the correct method
    mock_butler.query_datasets.assert_called_with(
        dataset_type="test_type",
        collections=input_collections,
        data_id=ANY  # We expect hips6 filtering but don't assert exact values
    )

    # Ensure the tagged collection is registered
    mock_butler.registry.registerCollection.assert_called_with(output_collection, CollectionType.TAGGED)

    # Ensure datasets were added to the collection
    mock_butler.registry.associate.assert_called_with(output_collection, mock_query_results)

    mock_butler.registry.query_collections.return_value = iter([output_collection])

    # Verify that `query-collections` now contains the new collection
    queried_collections = list(mock_butler.registry.query_collections())
    assert output_collection in queried_collections, "Tagged collection was not created successfully!"


@patch("spx_butler.script.createCalibratorTaggedCollection")
def test_create_tagged_primary_calibrators(mock_create_collection, runner):
    """Test the create_tagged_primary_calibrators CLI command."""
    result = runner.invoke(create_tagged_primary_calibrators, [
        "test_repo",
        "test_collection",
        "--output-collection", "test_output_collection",
        "--dataset-types", "test_type1,test_type2",
        "--list-tiles"
    ])

    # Ensure the command executed without errors
    assert result.exit_code == 0, f"CLI failed: {result.output}"

    # Verify the function was called with the correct parameters
    mock_create_collection.assert_called_once_with(
        repo="test_repo",
        collections=["test_collection"],
        output_collection="test_output_collection",
        dataset_types=["test_type1", "test_type2"],
        list_tiles=True
    )
