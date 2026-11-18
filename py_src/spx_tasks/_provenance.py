__all__ = ["add_calibration_provenance", "add_history", "get_calibration_inputs",
           "module_provenance", "add_psf_history"]
import logging
import functools
from typing import Callable
from uuid import UUID
from lsst.pipe.base import Task, Struct
from lsst.pipe.base.connections import InputQuantizedConnection

from spx_data import SPHERExImage
from spx_data import PSFZoneData


def get_calibration_inputs(input_refs: InputQuantizedConnection) -> list[tuple[str, UUID]]:
    """Get dataset type name and dataset UUID for each calibration input.

    The result can be as an input to the `add_calibration_provenance` function.

    Parameters
    ----------
    input_refs :
        A Namespace to map input connection names to the associated
        `lsst.daf.butler.DatasetRef` objects.

    Returns
    -------
    input_items :
        List of tuples with dataset type name and dataset UUID for each input.

    """
    calibrations = [(ref.datasetType.name, ref.id)
                    for _, ref in input_refs if ref.datasetType.isCalibration()]
    # Sort by calibration dataset type name
    return sorted(calibrations, key=lambda x: x[0])


def add_calibration_provenance(spx_image: SPHERExImage,
                               input_items: list[tuple[str, UUID]]) -> None:
    """Add HISTORY lines with the IDs of all calibration inputs.

    Parameters
    ----------
    spx_image :
        SPHEREx image to which we are adding provenance information.
    input_items :
        List of tuples with dataset type name and dataset UUID for each input.

    """
    for name, item_id in input_items:
        history_line = f"[CALIB] {name} {item_id}"
        spx_image.header.add_history(history_line)


def module_provenance(spxtask: Task) -> str:
    """Return a summary line about the task for provenance info.

    Parameters
    ----------
    spxtask : `~lsst.pipe.base.Task`
        Task implementing the module.

    Returns
    -------
    result : `str`
        Module name and version summary.

    """
    full_module_name = getattr(spxtask, "full_name", spxtask.getFullName())
    ssdc_doc_tag = getattr(spxtask, "ssdc_doc_tag", "SSDC-MS-XXX")
    return f"{full_module_name}, {ssdc_doc_tag}"


def add_history(history_line_getter: Callable = module_provenance, fld_name: str = "spximg_out"):
    """Decorate a method to add HISTORY comment to a SPHERExImage header.

    Parameters
    ----------
    history_line_getter : `~typing.Callable`, optional
        A callable that takes Task instance and returns a string to be added
        to the HISTORY field in the header. Defaults to `~module_provenance`.
    fld_name : `str`, optional
        The name of the output SPHEREx Image field in the output.
        Default is ``spximg_out``.

    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(spxtask: Task, *args, **kwargs):

            if method.__name__ != 'run':
                raise ValueError(f"add_history decorator can only be applied to 'run', "
                                 f"not '{method.__name__}'")

            if not isinstance(spxtask, Task):
                raise TypeError("add_history decorator can only be added to a Task method")

            # call the original run method
            result = method(spxtask, *args, **kwargs)

            # ensure the result includes SPHERExImage
            if isinstance(result, Struct) and hasattr(result, fld_name):
                spximg = getattr(result, fld_name)

                # generate the HISTORY line from the task attributes
                history_line = history_line_getter(spxtask)

                # add HISTORY to the header
                spximg.header.add_history(history_line)
            else:
                logging.warning(f"Unable to populate history for {spxtask.getFullName()}")

            return result

        return wrapper

    return decorator


def add_psf_history(history_line_getter: Callable = module_provenance, field_name: str = "psf"):
    """Decorate a method to add HISTORY comments to a PSFZoneData FITS header.

    Parameters
    ----------
    history_line_getter : `~typing.Callable`, optional
        A callable that takes Task instance and returns a string to be added
        to the HISTORY field in the header. Defaults to `~module_provenance`.
    field_name : `str`, optional
        The name of the output PSFZoneData field in the output object. Default is "psf".

    """
    def decorator(method):
        @functools.wraps(method)
        def wrapper(task, *args, **kwargs):
            # Ensure the decorated method is 'run'
            if method.__name__ != 'run':
                raise ValueError(f"add_psf_history decorator can only be applied to 'run', "
                                 f"not '{method.__name__}'.")

            # Call the original method
            result = method(task, *args, **kwargs)

            # Check if the result contains PSFZoneData
            if hasattr(result, field_name):
                psf_data = getattr(result, field_name)
                if not isinstance(psf_data, PSFZoneData):
                    raise TypeError(f"The field '{field_name}' does not contain a PSFZoneData object.")

                history_line = history_line_getter(task)
                psf_data.header.add_history(history_line)  # Add history directly to the header

                logging.info(f"Added HISTORY to PSFZoneData for task: {task.getFullName()}.")
            else:
                logging.warning(f"Field '{field_name}' not found in task output.")

            return result

        return wrapper

    return decorator
