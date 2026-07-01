__all__ = ["GainFactors"]

# Standard library imports
from datetime import datetime
from typing import List
from pydantic import model_validator, ConfigDict, BaseModel

# Related third party imports

# Local application/library specific imports
from .readout_parameters import ReadoutParameters


class ProvenanceItem(BaseModel):
    name: str
    value: str
    description: str

    def dict(self, *args, **kwargs):
        # Get the dictionary representation of the model
        data = super().dict(*args, **kwargs)
        # Return an OrderedDict to maintain the order of fields
        return {k: data[k] for k in self.__annotations__.keys()}


class Provenance(BaseModel):
    general: List[ProvenanceItem]
    gains_1: List[ProvenanceItem]
    gains_2: List[ProvenanceItem]
    gains_3: List[ProvenanceItem]
    gains_4: List[ProvenanceItem]
    gains_5: List[ProvenanceItem]
    gains_6: List[ProvenanceItem]
    history: List[str]


def get_default_provenance() -> Provenance:
    general_provenance = [
        ProvenanceItem(name="DATE",
                       value=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                       description="UTC date/time file was created"),
        ProvenanceItem(name="ORIGIN",
                       value="SPHEREx Instrument Team",
                       description="Fixed string"),
        ProvenanceItem(name="AUTHOR",
                       value="SSDC",
                       description="Name of the contact person for content"),
        ProvenanceItem(name="TELESCOP",
                       value="SPHEREx",
                       description="Fixed string"),
        ProvenanceItem(name="INSTRUME",
                       value="SPHEREx",
                       description="Fixed string"),
        ProvenanceItem(name="BUNIT",
                       value="electron/adu",
                       description="Gain factor unit, where adu is ice_adu/2**12"),
    ]

    detector_provenance = [
        ProvenanceItem(name="DATE-OBS",
                       value="YYYY-MM-DDThh:mm:ss",
                       description="UTC date/time detector data were acquired (if applicable)"),
        ProvenanceItem(name="DETSER",
                       value="ddddd",
                       description="Detector serial number"),
        ProvenanceItem(name="ROBSER",
                       value="ddddd",
                       description="Readout board serial number"),
        ProvenanceItem(name="TESTENV",
                       value="xx",
                       description="Test environment"),
    ]

    history = [
        "[INST] Gains are recorded per amplifier channel",
        "[INST] Version 0.0.1 - Format example"
    ]

    return Provenance(general=general_provenance,
                      gains_1=detector_provenance,
                      gains_2=detector_provenance,
                      gains_3=detector_provenance,
                      gains_4=detector_provenance,
                      gains_5=detector_provenance,
                      gains_6=detector_provenance,
                      history=history)


class GainFactors(BaseModel):
    """Instrument model that converts ADU to e/s for each detector array.

    Attributes
    ----------
    provenance : `Provenance`
        Provenance information for electronic gains.
    gains_1 : `list` [`float`]
        Gain factors per channel for detector 1.
    gains_2 : `list` [`float`]
        Gain factors per channel for detector 2.
    gains_3 : `list` [`float`]
        Gain factors per channel for detector 3.
    gains_4 : `list` [`float`]
        Gain factors per channel for detector 4.
    gains_5 : `list` [`float`]
        Gain factors per channel for detector 5.
    gains_6 : `list` [`float`]
        Gain factors per channel for detector 6.
    num_channels : `int` | `None`
        The number of channels for which gain factors are expected, by default
        :attr:`~spx_data.ReadoutParameters.nchannel`.

    Notes
    -----
    - Gain factors are expected to be one per readout channel (32)
      for each detector array (6).
    - Gain factors will be obtained on the ground.
    - Input image needs to be level 1 format.
    """

    provenance: Provenance | None = get_default_provenance()
    gains_1: list[float]
    gains_2: list[float]
    gains_3: list[float]
    gains_4: list[float]
    gains_5: list[float]
    gains_6: list[float]
    num_channels: int | None = ReadoutParameters.nchannel
    model_config = ConfigDict()

    @model_validator(mode="before")
    @classmethod
    def check_gains_length(cls, values: dict) -> dict:
        """
        Validates the length of each gain factors list. The length of each gain
        factors list must match the value of `num_channels`.

        Parameters
        ----------
        values : `dict`
            The dictionary containing the attribute values parsed so far.

        Returns
        -------
        values : `dict`
            The validated attribute values.

        Raises
        ------
        ValueError
            If the length of the gain factors list does not match `num_channels`.
        """
        num_channels = values.get("num_channels", ReadoutParameters.nchannel)
        for gain_name in ["gains_1", "gains_2", "gains_3", "gains_4", "gains_5", "gains_6"]:
            gains = values.get(gain_name)
            if gains is not None and len(gains) != num_channels:
                raise ValueError(f"{gain_name} must have {num_channels} items")

        return values

    def get_gains_for_detector(self, detector: int) -> list[float]:
        """Returns the gains for a specific detector.

        Parameters
        ----------
        detector
            The detector ID (1-6).

        Returns
        -------
        result : `list` [`float`]
            The gains for the specific detector.
        """
        return getattr(self, f"gains_{detector}")
