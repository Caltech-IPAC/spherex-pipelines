__all__ = ["SPHERExInstrument"]

from typing import Type

from lsst.daf.butler import DataId, Formatter, Registry
from lsst.pipe.base import Instrument
from lsst.utils.introspection import get_full_type_name

from spx_data import spx_param
from .utils import to_dichroicpair


class SPHERExInstrument(Instrument):
    """Instrument class for the real instrument.

    It needs to be used in class_name attribute of instrument dimension record.
    """

    def register(self, registry: Registry, *, update: bool = False) -> None:
        """Insert instrument, and other relevant records into `Registry`.

        Parameters
        ----------
        registry : `lsst.daf.butler.Registry`
            Registry client for the data repository to modify.
        update : `bool`, optional
            If `True` (`False` is default), update existing records if they
            differ from the new ones.
        Raises
        ------
        lsst.daf.butler.registry.ConflictingDefinitionError
            Raised if any existing record has the same key but a different
            definition as one being registered.
        """
        instr_name: str = self.getName()
        # make sure instrument and detector dimensions are populated
        with registry.transaction():
            instrument_record = {
                "name": instr_name,
                "obs_max": 600000,
                "detector_max": 6,
                "class_name": get_full_type_name(self)
            }
            registry.syncDimensionData("instrument", instrument_record, update=update)
            # dichroic pair is a pair of detectors that see the same sky scene
            for idx in range(1, 4):
                dichroicpair_record = {
                    "instrument": instr_name,
                    "id": idx,
                    "full_name": f"arrays_{idx}_and_{idx + 3}"
                }
                registry.syncDimensionData("dichroicpair", dichroicpair_record, update=update)
            for idx in range(1, 7):
                dichroicpair_id = to_dichroicpair(idx)
                lvf = spx_param.get_lvf_values(idx)
                detector_record = {
                    "instrument": instr_name,
                    "id": idx,
                    "dichroicpair": dichroicpair_id,
                    "full_name": f"array{idx}",
                    "lmin": lvf["lambda_min"],
                    "lmax": lvf["lambda_max"],
                    "r": lvf["R"]
                }
                registry.syncDimensionData("detector", detector_record, update=update)
            for idx in range(0, 10):
                # in survey, slewgroup_counter can take values 1, 2, 3, 4,
                # but when we are creating fake observation IDs it could be 0
                # in IOC sequences, it can be more than 4
                slewgroup_counter_record = {
                    "instrument": instr_name,
                    "id": idx
                }
                registry.insertDimensionData("slewgroup_counter",
                                             slewgroup_counter_record,
                                             skip_existing=True)
            for idx in range(0, 4):
                # in survey, sequenceflg can take values 0, 1, 2, 3,
                sequenceflg_record = {
                    "id": idx
                }
                registry.insertDimensionData("sequenceflg",
                                             sequenceflg_record,
                                             skip_existing=True)

    def getRawFormatter(self, dataId: DataId) -> Type[Formatter]:
        """Return the Formatter class that should be used to read a particular
        raw file.

        Parameters
        ----------
        dataId : `DataId`
            Dimension-based ID for the raw file or files being ingested.

        Returns
        -------
        formatter : `lsst.daf.butler.Formatter` class
            Class to be used that reads the file into the correct
            Python object for the raw data.
        """
        pass

    @classmethod
    def getName(cls):
        """Return the short (dimension) name for this instrument.

        """
        return "spx"
