__all__ = ["SequenceFlag"]

from enum import IntFlag


class SequenceFlag(IntFlag):
    """Sequence flags for timed data associated with an observation.

    These flags indicate the overlap between the time range of per-observation
    data snippets and the time range of an observation. They specify how much
    of the observation time range is covered by a data fragment.

    There are four types of overlaps:

    - **Middle overlap**: The observation's start and end are outside
      the snippet's time range.
    - **Start overlap**: The observation starts within the snippet's time range,
      but its end is outside.
    - **End overlap**: The observation ends within the snippet's time renge,
      but its start is outside.
    - **Complete overlap**: The snippet's time range fully contains the observation.

    Flag values:

    - ``0b00`` (0) : Data fragment corresponds to the middle of the observation.
    - ``0b01`` (1) : Data fragment corresponds to the beginning of the observation.
    - ``0b10`` (2) : Data fragment corresponds to the end of the observation.
    - ``0b11`` (3) : Data fragment corresponds to the complete observation.
    """

    MIDDLE = 0b00  # data fragment without beginning and end of the observation
    START = 0b01  # data fragment that contains the beginning of the observation
    END = 0b10  # data fragment that contains the end of the observation
    COMPLETE = 0b11  # data fragment that contains complete observation
