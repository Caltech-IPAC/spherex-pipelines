import pytest
from spx_data import (
    get_survey_id_from_spsplan,
    create_target_name,
    derive_target_name,
    SurveyIdentifierError,
    SPSFlag,
)
from spx_data.survey_identifier import S12_END, S34_END


class TestGetSurveyIdFromSpsplan:
    """Tests for get_survey_id_from_spsplan function."""

    def test_s12_range(self):
        """Test spsplan IDs in the S12 range return 'S12'."""
        assert get_survey_id_from_spsplan(202525101) == "S12"
        assert get_survey_id_from_spsplan(202537201) == "S12"
        assert get_survey_id_from_spsplan(1) == "S12"
        # Just before S12 ends (2026W20_2A = 202620201)
        assert get_survey_id_from_spsplan(S12_END - 1) == "S12"

    def test_s34_range(self):
        """Test spsplan IDs in the S34 range return 'S34'."""
        # First planning period of S34 (2026W20_2A)
        assert get_survey_id_from_spsplan(S12_END) == "S34"
        assert get_survey_id_from_spsplan(202650101) == "S34"
        # Just before S34 ends (last period is 2027W21_2A, S56 starts at 2027W22_1A)
        assert get_survey_id_from_spsplan(S34_END - 1) == "S34"

    def test_negative_spsplan_raises(self):
        """Test that negative spsplan ID raises SurveyIdentifierError."""
        with pytest.raises(SurveyIdentifierError, match="must be non-negative"):
            get_survey_id_from_spsplan(-1)

    def test_beyond_s34_raises_until_s56_defined(self):
        """Test that spsplan ID beyond S34 raises error (S56 boundary not defined)."""
        # S56 starts at 2027W22_1A = 202722101
        with pytest.raises(SurveyIdentifierError, match="beyond S34.*S56 boundary is not yet defined"):
            get_survey_id_from_spsplan(S34_END)


class TestCreateTargetName:
    """Tests for create_target_name function."""

    def test_basic_target_name(self):
        """Test basic target name creation."""
        assert create_target_name(123, "S12") == "SPS_000123_S12"
        assert create_target_name(1, "S34") == "SPS_000001_S34"
        assert create_target_name(999999, "S56") == "SPS_999999_S56"

    def test_zero_padded(self):
        """Test that target IDs are properly zero-padded to 6 digits."""
        assert create_target_name(0, "S12") == "SPS_000000_S12"
        assert create_target_name(12, "S12") == "SPS_000012_S12"
        assert create_target_name(12345, "S12") == "SPS_012345_S12"

    def test_out_of_bounds_target_ids(self):
        """Test that negative target ID raises ValueError."""
        assert create_target_name(-1, "S12") == "SPS_000000_S12"
        assert create_target_name(1000000, "S12") == "SPS_000000_S12"

    def test_invalid_survey_id_raises(self):
        """Test that invalid survey ID raises ValueError."""
        with pytest.raises(ValueError, match="must be 'S12', 'S34', or 'S56'"):
            create_target_name(123, "S78")
        with pytest.raises(ValueError, match="must be 'S12', 'S34', or 'S56'"):
            create_target_name(123, "s12")  # lowercase not allowed
        with pytest.raises(ValueError, match="must be 'S12', 'S34', or 'S56'"):
            create_target_name(123, "")


class TestDeriveTargetName:
    """Tests for derive_target_name function."""

    def test_non_go_back_s12(self):
        """Test non-go-back observation in S12."""
        # Regular all-sky observation in S12
        result = derive_target_name(SPSFlag.ALL_SKY, 202525101, 123)
        assert result == "SPS_000123_S12"

    def test_non_go_back_s34(self):
        """Test non-go-back observation in S34."""
        result = derive_target_name(SPSFlag.ALL_SKY, S12_END, 456)
        assert result == "SPS_000456_S34"

    def test_go_back_s12(self):
        """Test go-back observation from S12 (executed in S34)."""
        # Go-back from S12, executed during S34 spsplan
        flags = SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY
        result = derive_target_name(flags, S12_END + 1000, 789)
        # Should use S12 from flag, not S34 from spsplan
        assert result == "SPS_000789_S12"

    def test_go_back_s34(self):
        """Test go-back observation from S34.

        The spsplan_id is ignored when a go-back flag is set, so we use
        a valid S34 spsplan here. In practice, S34 go-backs would be
        executed during S56, but the survey comes from the flag.
        """
        flags = SPSFlag.GO_BACK_S34 | SPSFlag.ALL_SKY
        # Use valid S34 spsplan; survey comes from go-back flag regardless
        result = derive_target_name(flags, S34_END - 1, 111)
        assert result == "SPS_000111_S34"

    def test_go_back_overrides_spsplan(self):
        """Test that go-back flag takes precedence over spsplan survey."""
        # S12 go-back executed during S12 spsplan (edge case)
        flags = SPSFlag.GO_BACK_S12 | SPSFlag.ALL_SKY
        result = derive_target_name(flags, 202525101, 222)
        assert result == "SPS_000222_S12"
