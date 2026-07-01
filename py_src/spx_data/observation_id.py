__all__ = ['ObservationId']

from typing import Union

# 100 UTF-8 chars for 2 digits to a character mapping,
# includes Latin, Cyrillic, and Greek characters.
# Used to convert random 2 digits to fake plan version;
# could be removed when this capability is no longer needed.
# Characters with indices 1-26 should map to A-Z, valid plan versions.
PLAN_VERSION_CHARS = "яABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" \
                     "ΔΛΞΣΨΩαβδεζηθλμξπρστφψωБГДЖЗИЛПФЦЧШЩЫЭЮЯбвгджзи"


def _char_to_position(char: str) -> str:
    """Converts the plan version letter value to the numeric value, base 1."""
    # valid plan versions A-Z are converted to two digits using
    # the position of a character in the Unicode map relative to 'A'
    position = ord(char) - ord("A") + 1
    if position < 1 or position > 26:
        # Fake plan versions can appear when we convert a unique combination
        # of digits, such as timestamp to an observation id.
        # Replace withan exception when this capability is no longer needed.
        position = PLAN_VERSION_CHARS.index(char)
    return f"{position:02d}"


def _position_to_char(position: int) -> str:
    """Converts the plan version numeric value, base 1, to the letter value."""
    if position < 1 or position > 26:
        # Fake plan versions can appear when we convert a unique combination
        # of digits, such as timestamp to an observation id.
        # Replace withan exception when this capability is no longer needed.
        return PLAN_VERSION_CHARS[position]
    else:
        # valid plan versions are A to Z
        return chr(position + ord("A") - 1)


def _format_spx(dict_obs_id: dict[str, str]) -> str:
    """Returns the formatted SPHEREx observation ID."""
    date = f"{dict_obs_id['year']}{dict_obs_id['week']}"
    plan = f"{dict_obs_id['plan_number']}{dict_obs_id['plan_version']}"
    slew = (f"{dict_obs_id['large_slew_counter']}"
            f"{dict_obs_id['small_slew_counter']}")
    return f"{date}{plan}{slew}0"


def _format_sps(dict_obs_id: dict[str, str]) -> str:
    """Returns the formatted SPS observation ID."""
    date = f"{dict_obs_id['year']}W{dict_obs_id['week']}"
    slew = (f"{dict_obs_id['large_slew_counter']}_"
            f"{dict_obs_id['small_slew_counter']}")
    return f"{date}_{dict_obs_id['plan']}_{slew}"


def _breakdown(obs_id: Union[str, int]) -> dict[str, str]:
    """Takes in either SPHEREx or SPS observation ID and extracts information.

    Parameters
    ----------
    obs_id : `str` or `int`
        The observation ID to parse.

    Returns
    -------
    dict_obs_id : `dict`
        Dictionary of parsed information from observation ID.

    Raises
    ------
    ValueError
        Raised if the observation ID does not match SPS or SPHEREx.
    """
    # Need in string format to slice
    if isinstance(obs_id, int):
        obs_id = str(obs_id)

    # Handle SPHEREx observation ID
    if len(obs_id) == 15 and obs_id.isdigit():
        obs_id = str(obs_id)
        year = obs_id[:4]
        week = obs_id[4:6]
        plan_number = obs_id[6]
        plan_version = obs_id[7:9]
        plan = f"{plan_number}{_position_to_char(int(plan_version))}"
        large_slew_counter = obs_id[9:13]
        small_slew_counter = obs_id[13]
        # 15th digit is either 0 or detector id
        detector = int(obs_id[14])
        if detector > 6:
            raise ValueError(f"Invalid observation ID: {obs_id}, "
                             "Last digit should be either 0 or detector ID.")
        original_format = "SPHEREx"

    # Handle SPS short observation ID, length 17
    # Handle SPS long observation ID, length 19
    elif "_" in obs_id and len(obs_id) in [17, 19]:
        year = obs_id[:4]
        week = obs_id[5:7]
        plan_number = obs_id[8]
        plan_version = _char_to_position(obs_id[9])
        plan = obs_id[8:10]
        large_slew_counter = obs_id[11:15]
        small_slew_counter = obs_id[16]
        original_format = "SPS"

        if len(obs_id) == 19:
            # Ignoring the total number of exposures per pointing
            pass

    else:
        raise ValueError(f"Incorrect format for obs ID: {obs_id}")

    dict_obs_id = {"year": year, "week": week, "plan_number": plan_number,
                   "plan_version": plan_version, "plan": plan,
                   "small_slew_counter": small_slew_counter,
                   "original_format": original_format,
                   "large_slew_counter": large_slew_counter}

    return dict_obs_id


class ObservationId:
    """Class that parses an SPS or SPHEREx observation ID.

    Attributes
    ----------
    sps_id: `str`
        The SPS Observation ID without total number of small slews (17 chars)
    spx_id: `int`
        Unique ID of a SPHEREx Observation

    Parameters
    ----------
    obs_id : `str` or `int`
        The observation ID, either SPHEREx or SPS, to parse.

    Notes
    -----
    Class can accept either of these formats below:

    - SPS short observation ID: 2025W37_2A_0486_2

      - <year>W<week>_<plan number><plan version>_<large slew counter>
        _<small slew counter>
      - year: 2025
      - week: 37
      - plan number: 2
      - plan version: A
      - large slew counter: 0486
      - small slew counter: 2

    - SPS long observation ID: 2025W37_2A_0486_2_4

      - <year>W<week>_<plan number><plan version>_<large slew counter>
        _<small slew counter>_<total number of pointings per small slew>
      - year: 2025
      - week: 37
      - plan number: 2
      - plan version: A
      - large slew counter: 0486
      - small slew counter: 2
      - total number of pointings in small slew group: 4

    - SPHEREx pointing ID: 202537201048620 (plan, slew group, small slew counter, 0)
    - SPHEREx detector image ID: 202537201048625 (pointing ID followed by detector id)

    Documentation about the observation ID located here:
    https://confluence.ipac.caltech.edu/pages/viewpage.action?spaceKey=SPHEREx&title=Binary+form+of+the+observation+ID

    """
    def __init__(self, obs_id: Union[str, int]):
        dict_obs_id = _breakdown(obs_id)
        self.spx_id = _format_spx(dict_obs_id)
        self.sps_id = _format_sps(dict_obs_id)
        self.year = dict_obs_id["year"]
        self.week = dict_obs_id["week"]
        self.plan_number = dict_obs_id["plan_number"]
        self.plan_version = dict_obs_id["plan_version"]
        self.large_slew_counter = dict_obs_id["large_slew_counter"]
        self.small_slew_counter = dict_obs_id["small_slew_counter"]

    @property
    def pointing_id(self) -> int:
        return int(self.spx_id)

    @property
    def plan_id(self) -> int:
        return int(self.spx_id[:-6])

    @property
    def plan_name(self) -> str:
        return self.sps_id[:10]

    @property
    def slewgroup_id(self) -> int:
        return int(self.large_slew_counter)

    @property
    def slewgroup_counter_id(self) -> int:
        return int(self.small_slew_counter)

    def to_exposure_id(self, detector_id: int):
        return self.pointing_id + detector_id

    def __repr__(self):
        return f"ObservationId({self.sps_id!r})"

    def __str__(self):
        return f"ObservationId: {self.spx_id}"

    def __eq__(self, other):
        return isinstance(other, ObservationId) and \
            self.spx_id == other.spx_id and \
            self.sps_id == other.sps_id

    @staticmethod
    def pack(sps_id: str) -> str:
        """Packs SPS observation ID and returns SPHEREx observation ID.

        Parameters
        ----------
        sps_id : `str`
            The SPS observation ID to pack.

        Returns
        -------
        spx_id : `str`
            The SPHEREx observation ID.

        Raises
        ------
        ValueError
            Raised when users provide a SPHEREx observation ID to this method
            which expects a SPS observation ID.

        """
        dict_obs_id = _breakdown(sps_id)
        if dict_obs_id["original_format"] == "SPHEREx":
            raise ValueError("Provided SPHEREx observation ID to the pack "
                             "method when it expects a SPS observation ID.")
        return _format_spx(dict_obs_id)

    @staticmethod
    def unpack(spx_id: Union[int, str]) -> str:
        """Unpacks SPHEREx observation ID and returns SPS observation ID.

        Parameters
        ----------
        spx_id : `int` or `str`
            The SPHEREx observation ID to unpack.

        Returns
        -------
        sps_id : `str`
            The SPS observation ID.

        Raises
        ------
        ValueError
            Raised when users provide an SPS observation ID to this method which
            expects a SPHEREx observation ID.

        """
        dict_obs_id = _breakdown(spx_id)
        if dict_obs_id["original_format"] == "SPS":
            raise ValueError("Provided SPS observation ID to the unpack method "
                             "when it expects a SPHEREx observation ID.")
        return _format_sps(dict_obs_id)
