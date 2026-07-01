__all__ = ["SimulatorInstrument"]

from ._spherex_instrument import SPHERExInstrument


class SimulatorInstrument(SPHERExInstrument):
    """Instrument class for the simulator.

    It needs to be used in class_name attribute of instrument dimension record.
    """

    @classmethod
    def getName(cls):
        """Return the short (dimension) name for this instrument.

        """
        return "simu"
