__all__ = ["Level1DQA", "Level2DQA"]

from pydantic import BaseModel
from pydantic.functional_validators import field_validator


class Level1DQA(BaseModel):
    """Holds level 1 data quality assessment results.

    Attributes
    ----------
    metadata_summary : `dict`
        Key-value pairs represent metadata/instrument parameters and their
        measured values, respectively.
    image_statistics : `dict`
        Key-value pairs represent image statistics and their measured values,
        respectively.
    threshold_tests : `list`[`dict`]
        List of threshold test records. Each test record dictionary includes
        results (pass/fail) and a description.

    """

    metadata_summary: dict
    image_statistics: dict
    threshold_tests: list[dict]

    @field_validator("image_statistics")
    @classmethod
    def check_image_statistics(cls, v: dict, field: str):
        """Validate the image statistics dictionary values.

        Parameters
        ----------
        v : `dict`
            Image statistics dictionary.
        field : `str`
            Name of class field.

        Returns
        -------
        v : `dict`
            The same value if it is valid.

        Raises
        ------
        ValueError
            Raised if image statistics dictionary is missing a required
            statistic.

        """
        required_stats = ["MEAN", "MEDIAN", "RMS"]
        for stat in required_stats:
            if stat not in list(v.keys()):
                raise ValueError(f"{field} keys do not include required statistic '{stat}'.")
        return v


class Level2DQA(BaseModel):
    """Level 2 data quality assessment record.

    Attributes
    ----------
    update_field_distortion : `dict`
        Test result and description for L2.18.

    """

    nonlinear_flag_flux_check: dict
    dqa_variance: dict
    persistence_flag_flux_check: dict
    badpix_flag_fraction_check: dict
    # flat_field_check: dict
    source_count_check: dict
    source_mask_check: dict
    ghost_flag_flux_check: dict
    crosstalk_flag_flux_check: dict
    outlier_flag_check: dict
    zodi_layer_check: dict
    layer_presence_check: dict
    unflagged_nan_check: dict
    unflagged_nan_variance_check: dict
    update_field_distortion: dict
    # psf_solid_angle_check: dict
    finast_check: dict
    srccat_coverage_check: dict
    overall_result: str = "UNKNOWN"

    @field_validator("*", mode="before")
    @classmethod
    def check_record_dict(cls, v: dict, field: str):
        """Validate the dictionary values.

        Parameters
        ----------
        v : `dict`
            Module DQA test record.
        field : `str`
            Name of the module field.

        Returns
        -------
        v : `dict`
            The same value if it is valid.

        Raises
        ------
        ValueError
            Raised if test record result is not Pass or Fail. Raised if test
            record description is not a string.

        """
        if v["result"] not in ["PASS", "FAIL"]:
            raise ValueError(f"{field} result {v['result']} is not Pass or Fail.")
        if not isinstance(v["description"], str):
            raise ValueError(f"{field} description is not a string.")
        return v
