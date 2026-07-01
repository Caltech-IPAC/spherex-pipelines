import unittest
from unittest.mock import Mock, patch
from spx_butler.instrument import ensure_registered_instrument


class TestEnsureRegisteredInstrument(unittest.TestCase):
    def setUp(self):
        self.instrument = Mock()
        self.instrument.getName.return_value = "tst_instr"
        self.registry = Mock()

    @patch("lsst.pipe.base.Instrument.fromName")
    def test_already_registered(self, mock_fromName):
        # Simulate instrument already registered
        mock_fromName.return_value = self.instrument

        ensure_registered_instrument(self.instrument, self.registry)

        mock_fromName.assert_called_once_with("tst_instr", self.registry)
        self.instrument.register.assert_not_called()

    @patch("lsst.pipe.base.Instrument.fromName")
    def test_not_registered(self, mock_fromName):
        # Simulate LookupError meaning instrument not registered
        mock_fromName.side_effect = LookupError

        ensure_registered_instrument(self.instrument, self.registry)

        mock_fromName.assert_called_once_with("tst_instr", self.registry)
        self.instrument.register.assert_called_once_with(self.registry)


if __name__ == "__main__":
    unittest.main()
