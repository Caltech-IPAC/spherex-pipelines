import numpy as np
from lsst.pex.config import Config
from lsst.pipe.base import Task, Struct
from lsst.pipe.base.connections import InputQuantizedConnection
from spx_data import SPHERExImage, PSFZoneData
from spx_tasks import add_history, add_psf_history, get_calibration_inputs, add_calibration_provenance

from lsst.daf.butler import (
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionUniverse,
    StorageClass,
)


class TstTaskConfig(Config):
    pass


class TstTask(Task):
    _DefaultName = "tst-task"
    ConfigClass = TstTaskConfig
    full_name = "Test Task"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-123"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history(fld_name="spximg")
    @add_psf_history(field_name="psf")
    def run(self) -> Struct:
        spximg = SPHERExImage(data=np.zeros((5, 5), dtype=np.float32), unit="electron")
        psf = PSFZoneData(
            data=np.zeros((9, 3, 3), dtype=np.float32),
            x_centers=[340.5, 1020.5, 1700.5],
            y_centers=[340.5, 1020.5, 1700.5],
            oversampling=10
        )
        return Struct(spximg=spximg, psf=psf)


def test_add_history():
    config = TstTaskConfig()
    task = TstTask(config=config)
    result = task.run()
    assert result.spximg.header["HISTORY"] is not None
    first_history_comment = result.spximg.header["HISTORY"][0]
    assert first_history_comment.startswith(task.full_name)
    assert first_history_comment.endswith(task.ssdc_doc_tag)


def test_calibration_provenance():
    universe = DimensionUniverse()
    dimensions = universe.conform(("instrument", "detector"))
    # test with one calibration input with dataset type name "dp2"
    ds_types = [
        ("dp1", StorageClass("Cal1"), False),
        ("dp2", StorageClass("Cal2"), True)]
    datasetTypeLst = [
        DatasetType(name, dimensions, storage_class, isCalibration=cal)
        for name, storage_class, cal in ds_types
    ]
    data_id = DataCoordinate.standardize({"instrument": "instr", "detector": 2},
                                         universe=universe)
    run = "arun"
    refs = [DatasetRef(datasetType, data_id, run=run) for datasetType in datasetTypeLst]
    inputs = InputQuantizedConnection()
    for ref in refs:
        setattr(inputs, ref.datasetType.name, ref)
    calib_items = get_calibration_inputs(inputs)
    spximg = SPHERExImage(data=np.zeros((5, 5), dtype=np.float32), unit="electron")
    add_calibration_provenance(spximg, calib_items)
    assert spximg.header["HISTORY"] is not None
    first_history_comment = spximg.header["HISTORY"][0]
    assert first_history_comment.startswith("[CALIB] dp2")
    assert first_history_comment.endswith(str(refs[-1].id))


def test_add_psf_history():
    config = TstTaskConfig()
    task = TstTask(config=config)
    result = task.run()

    # Test PSFZoneData history
    psf_header = result.psf.to_hdu().header
    assert "HISTORY" in psf_header
    history_comments = psf_header["HISTORY"]
    assert history_comments is not None
    assert any(task.full_name in line for line in history_comments)
    assert any(task.ssdc_doc_tag in line for line in history_comments)
