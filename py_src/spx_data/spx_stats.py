import numpy as np

from spx_data.spx_flags import Flag
from .spx_image import SPHERExImage


class Stats:
    """Image and flag statistics."""

    def __init__(self, spximg: SPHERExImage):
        # > Flag
        self.flag_count = {}
        # names of the flags that were set in the image
        applicable_flag_names = spximg.flag_defs.keys()
        for flg in Flag:
            if flg.name in applicable_flag_names:
                self.flag_count[flg.name] = spximg.getFlagsAsMask(flg).sum()
        # > Data
        self.v_mean = np.mean(spximg.data)
        self.v_median = np.median(spximg.data)
        self.v_rms = np.sqrt(np.mean(spximg.data ** 2))
        self.q95 = np.quantile(spximg.data, .95)
        self.q99 = np.quantile(spximg.data, .99)
