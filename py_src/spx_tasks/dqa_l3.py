__all__ = ["DQAL3Config", "DQAL3Task"]

# Standard library imports
from dataclasses import dataclass
import copy

# Related third party imports
from lsst.pex.config import Config, Field, DictField
from lsst.pipe.base import Task

# Local application/library specific imports
from spx_data import SPHERExImage

# ==============================================================================


class DQAL3Config(Config):
    """Configuration for DQAL3 (Level 3 Data Quality Assessment) task."""

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


class DQAL3Task(Task):
    """Implementation of L3 Perform Data Quality Assessment for Level 3.

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
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-036

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.

    """

    _DefaultName = "dqa-l3"
    ConfigClass = DQAL3Config
    full_name = "Perform Data Quality Assessment for Level 3"
    ssdc_pipe_level = 3
    ssdc_doc_tag = "SSDC-MS-036 v1.0"

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
