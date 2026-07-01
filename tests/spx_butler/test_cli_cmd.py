from spx_butler.cli.cmd import (
    ingest_simulated,
    ingest_calibrations,
    create_tagged_primary_calibrators,
    report_contiguous_obs
)


def test_imports():
    assert ingest_simulated is not None
    assert ingest_calibrations is not None
    assert create_tagged_primary_calibrators is not None
    assert report_contiguous_obs is not None
