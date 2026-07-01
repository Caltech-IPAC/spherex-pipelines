__all__ = ["SPSFlag"]

from enum import IntFlag


class SPSFlag(IntFlag):
    """Bit fields corresponding to Survey Planning Software flags
    for a single planned observation."""
    DEEP_SOUTH = 1 << 0  # deep field survey observation: south zone
    DEEP_NORTH = 1 << 1  # deep field survey observation: north zone
    ALL_SKY = 1 << 2  # all-sky survey observation
    DOWNLINK = 1 << 3  # downlink contingency: observation occurs if downlink is cancelled
    SAA = 1 << 4  # observation from South Atlantic Anomaly zone
    EARTH_STRAYLIGHT = 1 << 5  # observation is a part of Earth Stray Light test
    MOON_STRAYLIGHT = 1 << 6  # observation is a part of Moon Stray Light test
    SPECIAL = 1 << 7  # Targeted observations of comet 3I/Atlas

    @classmethod
    def from_string(cls, sps_flag: str) -> int:
        """Return integer representation of SPS bit flags.

        Parameters
        ----------
        sps_flag: `str`
            PEF observation item's SPS flag, comma separated flags
            from the survey planning.

        Returns
        -------
        sps_flag_int: `int`
            Integer representation of SPS bit flags.

        Raises
        ------
        ValueError
            Raised if encounters an invalid flag.

        Notes
        -----
        The flag logic will be updated as the PEF item information evolves.
        SPS flags: deep_south, deep_north, all_sky, downlink, saa,
        earth_straylight, moon_straylight, special.
        """
        flag_list = [x.strip() for x in sps_flag.split(',')]
        flag_dict = {
            'deep_south': SPSFlag.DEEP_SOUTH,
            'deep_north': SPSFlag.DEEP_NORTH,
            'all_sky': SPSFlag.ALL_SKY,
            'downlink': SPSFlag.DOWNLINK,
            'saa': SPSFlag.SAA,
            'earth_straylight': SPSFlag.EARTH_STRAYLIGHT,
            'moon_straylight': SPSFlag.MOON_STRAYLIGHT,
            'special': SPSFlag.SPECIAL
        }
        sps_flag_int = 0
        for flag in flag_list:
            if flag not in flag_dict:
                raise ValueError(f'Not a valid metadata flag: {flag}')
            sps_flag_int |= flag_dict[flag]

        return int(sps_flag_int)

    @classmethod
    def to_string(cls, sps_flags_int: int) -> str:
        """Return comma-separated flag names in lower case.

        Parameters
        ----------
        sps_flags_int : `int`

        Returns
        -------
        result : `str`
            Comma-separated flag names in lower case.
        """
        active_flags = [flag.name.lower() for flag in cls if sps_flags_int & flag.value]
        return ",".join(active_flags)

    @classmethod
    def is_deep_field(cls, sps_flags_int: int) -> bool:
        """Return ``True`` for Deep Field observations.

        Parameters
        ----------
        sps_flags_int : `int`

        Returns
        -------
        result : `bool`
            ``True`` for Deep Field observations, ``False`` otherwise.
        """
        return sps_flags_int & (cls.DEEP_SOUTH | cls.DEEP_NORTH) > 0

    @classmethod
    def is_saa(cls, sps_flags_int: int) -> bool:
        """Return ``True`` for SAA observations.

        Parameters
        ----------
        sps_flags_int : `int`

        Returns
        -------
        `bool`
            ``True`` for SAA observations, ``False`` otherwise.
        """
        return sps_flags_int & cls.SAA > 0
