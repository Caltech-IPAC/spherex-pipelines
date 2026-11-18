__all__ = ["L1ReadoutChannel", "l1_readout_channels"]

from collections.abc import Iterator
from .readout_parameters import ReadoutParameters


class L1ReadoutChannel:
    """Definition of a readout channel in L1 image.

    Attributes
    ----------
    ichan : `int`
        Channel index from 0 to 31
    phan_cols : `range`
        Range of the phantom column indexes in this channel.
    det_cols : `range`
        Range of the H2RG column indexes in this channel.

    """

    def __init__(self, i: int, ro_params: ReadoutParameters):
        self.ichan: int = i
        self.phan_cols = range(i * ro_params.Mphan, (i + 1) * ro_params.Mphan)
        # detector columns offset is the number of phantom columns
        self.det_cols = range(ro_params.Kphan + i * ro_params.wchannel,
                              ro_params.Kphan + (i + 1) * ro_params.wchannel)

    def __repr__(self):
        return f"L1ReadoutChannel ichan={self.ichan}: " \
               f"phan_cols={self.phan_cols}, " \
               f"det_cols={self.det_cols}); "


def l1_readout_channels(ro_params: ReadoutParameters) -> Iterator[L1ReadoutChannel]:
    """Generate iterator through readout channels.

    Parameters
    ----------
    ro_params : `~spx_data.ReadoutParameters`
        Readout parameters

    Returns
    -------
    readout_channel : `Iterator[L1ReadoutChannel]`
        Iterator through readout channels (the information about image columns
        that belong to a readout channel)

    """
    for i in range(ro_params.nchannel):
        yield L1ReadoutChannel(i, ro_params)
