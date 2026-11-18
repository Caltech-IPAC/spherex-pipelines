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
    GO_BACK_S12 = 1 << 8  # # Go back observations for Surveys 1/2,
    # to be executed in Surveys 3/4 (or later)

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
        earth_straylight, moon_straylight, special, go_back_s12.

        According to current policy, the PEF "go-back"
        mechanism will only be used on all-sky targets
        from previous surveys, so we add the all_sky flag
        to our copy of the SPS flags to make it clear
        that go-backs are part of the all-sky survey.

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
            'special': SPSFlag.SPECIAL,
            'go_back_s12': SPSFlag.GO_BACK_S12
        }
        go_back_dict = {'go_back_s12': SPSFlag.GO_BACK_S12}
        sps_flag_int = 0
        for flag in flag_list:
            if flag not in flag_dict:
                raise ValueError(f'Not a valid metadata flag: {flag}')
            if flag in go_back_dict:
                # Add all_sky flag when go_back flag occurs.
                sps_flag_int |= SPSFlag.ALL_SKY
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
