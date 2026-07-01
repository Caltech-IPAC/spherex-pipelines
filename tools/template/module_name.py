__all__ = ["ModuleNameConfig", "ModuleNameTask"]

# Standard library imports
from dataclasses import dataclass
import copy

# Related third party imports
from lsst.pex.config import Config, Field, DictField
from lsst.pipe.base import Task

# Local application/library specific imports
from spx_data import SPHERExImage

# ==============================================================================


class ModuleNameConfig(Config):
    param_int = Field(
        doc="An example of int configuration parameter."
        "An example of a two-line doc.",
        dtype=int,
        default=42)

    param_float = Field(
        "An example of float configuration parameter.",
        dtype=float,
        default=4.2)

    param_dict = DictField(
        doc="An example of dictionary configuration parameter.",
        keytype=int,
        itemtype=float,
        default={1: 12.5, 2: 12.5, 3: 12.5, 4: 10.5, 5: 10.5, 6: 10.5})


@dataclass
class Output:
    image: SPHERExImage


class ModuleNameTask(Task):
    """Implementation of Lx Full Module-Name.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-xxx

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `dataclass` object.
    """

    _DefaultName = "module-name"
    ConfigClass = ModuleNameConfig
    full_name = "Full Module-Name"
    ssdc_pipe_level = 0
    ssdc_doc_tag = "SSDC-MS-xxx"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, spximg: SPHERExImage) -> Output:
        # Algorithm Implementation
        pass

        # Create an Output object and
        # pass all module outputs during its instantiation
        # Example:
        # Use shallow copy when feasible
        spximg_out = copy.copy(spximg)
        output = Output(image=spximg_out)

        return output
