__all__ = ["CorrectPreamplifierConfig", "CorrectPreamplifierTask"]

# Standard library imports
import copy

# Related third party imports
import numpy as np
from astropy.stats import sigma_clip
from lsst.pex.config import Field
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT

# Local application/library specific imports
from spx_data import SPHERExImage, Flag, ReadoutParameters, l1_readout_channels
from spx_utils import active_area_range
from spx_tasks import add_history

# ==============================================================================


class CorrectPreamplifierConnections(pipeBase.PipelineTaskConnections,
                                     dimensions=("instrument", "exposure"),
                                     defaultTemplates={}):
    spximg = cT.Input(
        name="spximg",  # input is output from previous module
        doc="Input image.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )
    spximg_out = cT.Output(
        name="spximg_out",  # default dataset type for output
        doc="Output image after variance map created.",
        storageClass="SPHERExImage",
        dimensions=("instrument", "exposure"),
    )


class CorrectPreamplifierConfig(pipeBase.PipelineTaskConfig,
                                pipelineConnections=CorrectPreamplifierConnections):
    sigma = Field(
        doc="Sigma for sigma-clipping.",
        dtype=int,
        default=5)

    no_ref_correction = Field(
        doc="Phantoms correction only.",
        dtype=bool,
        default=False)


class CorrectPreamplifierTask(pipeBase.PipelineTask):
    """Implementation of L1 Correct Preamplifier.

    Attributes
    ----------
    ConfigClass : `lsst.pex.config.Config`
        Task configuration class.
    full_name : `str`
        Full module name as it appears in SSDC document.
    ssdc_pipe_level: `int`
        Module pipeline level.
    ssdc_doc_tag: `str`
        SPHEREx Science Data Center (SSDC) document ID.
        Document GitHub page: https://github.com/SPHEREx/SSDC-MS-006

    Notes
    -----
    - Use config to customize algorithm parameters.
    - Pass module inputs as arguments of the ``run`` method.
    - Module outputs are returned by the ``run`` method as
      attributes of a `~dataclasses.dataclass` object.
    """

    _DefaultName = "correct-preamplifier"
    ConfigClass = CorrectPreamplifierConfig
    full_name = "Correct Preamplifier"
    ssdc_pipe_level = 1
    ssdc_doc_tag = "SSDC-MS-006 v1.5"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @add_history()
    def run(self, spximg: SPHERExImage) -> pipeBase.Struct:

        sigma = self.config.sigma
        no_ref_correction = self.config.no_ref_correction
        rp = ReadoutParameters.from_metadata(spximg.meta)

        data = np.copy(spximg.data)
        flags = spximg.flags
        fmask = np.full(shape=data.shape, fill_value=False)  # mask showing pixels that can not be corrected

        # Mask and set to NaN invalid pixels per error logic
        is_error = (flags & Flag.SUR_ERROR) > 0  # 0b100
        is_transient = (flags & Flag.TRANSIENT) > 0  # 0b001
        is_overflow = (flags & Flag.OVERFLOW) > 0  # 0b010

        # Early transient: bit pattern 0b101 (SUR_ERROR + TRANSIENT)
        is_early_transient = is_error & is_transient & ~is_overflow

        # Very early overflow: bit pattern 0b110 (SUR_ERROR + OVERFLOW)
        is_early_overflow = is_error & is_overflow & ~is_transient

        # Early transient + overflow: bit pattern 0b111
        is_early_transient_overflow = is_error & is_transient & is_overflow

        # Construct mask
        nan_mask_from_zeroed = (
            # Case 1: early transient-type => only zero values are NaN'd
            # Case 2: very early overflow or early transient with overflow
            #         or just SUR_ERROR => always NaN
            (is_early_transient & (data == 0))
            | (is_early_transient_overflow & (data == 0))
            | (is_early_overflow | (is_error & ~is_transient & ~is_overflow))
        )

        # Apply NaN
        data[nan_mask_from_zeroed] = np.nan

        # Extra check and log for unexpected zero pixels
        zero_remaining_mask = (data == 0) & ~nan_mask_from_zeroed

        if np.any(zero_remaining_mask):
            remaining_flags = flags[zero_remaining_mask]
            unique_flags, counts = np.unique(remaining_flags, return_counts=True)
            self.log.info("Remaining unexpected zero-value pixels:")
            for uf, count in zip(unique_flags, counts):
                try:
                    desc = Flag(uf).name
                except ValueError:
                    desc = f"UNKNOWN (value={uf})"
                self.log.info(f"  Flag={uf:08b} ({desc}): {count} pixel(s)")

        # Phantom pixel correction
        # phantom pixels
        phan = spximg.data[:, :rp.nchannel * rp.Mphan]
        phan_masked = sigma_clip(phan, sigma=sigma)

        ro_channels = l1_readout_channels(rp)
        for ch in ro_channels:
            phan_istart = ch.phan_cols.start  # start phantom column index for this readout channel
            phan_istop = ch.phan_cols.stop  # stop phantom column for this readout channel, exclusive

            # phantom even columns
            phan_even_masked = phan_masked[:, phan_istart:phan_istop:2]
            phan_even_masked_avg = np.ma.mean(phan_even_masked, axis=-1)
            phan_even_masked_avg.data[phan_even_masked_avg.mask] = np.nan

            # phantom odd columns
            phan_odd_masked = phan_masked[:, phan_istart+1:phan_istop:2]
            phan_odd_masked_avg = np.ma.mean(phan_odd_masked, axis=-1)
            phan_odd_masked_avg.data[phan_odd_masked_avg.mask] = np.nan

            istart = ch.det_cols.start  # start data column index for this readout channel
            istop = ch.det_cols.stop  # stop data column index for  this readout channel, exclusive

            # even columns - data and mask
            data[:, istart:istop:2] -= phan_even_masked_avg.data[:, np.newaxis]
            even_masked = np.ma.count_masked(phan_even_masked_avg)
            if even_masked > 0:
                fmask[:, istart:istop:2] = phan_even_masked_avg.mask[:, np.newaxis]

            # odd columns - data and mask
            data[:, istart + 1:istop:2] -= phan_odd_masked_avg.data[:, np.newaxis]
            odd_masked = np.ma.count_masked(phan_odd_masked_avg)
            if odd_masked > 0:
                fmask[:, istart + 1:istop:2] = phan_odd_masked_avg.mask[:, np.newaxis]

        if no_ref_correction:
            # Apply only phantom pixel correction
            spximg_out = copy.copy(spximg)
            spximg_out.data = data
            spximg_out.setFlagsFromMask(fmask, Flag.PHANMISS)
            return pipeBase.Struct(spximg_out=spximg_out)

        # Reference pixel correction
        # reference row indexes are outside active pixel row indexes
        # reference column indexes are outside active pixel column indexes
        irows, icols = active_area_range(spximg)  # active pixel range

        # refs excluding the 4 corners where the row/column references overlap
        refs1 = data[:irows.start, icols.start:icols.stop]
        refs2 = data[irows.stop:, icols.start:icols.stop]
        refs = np.vstack((refs1, refs2))

        # ref flags
        ref_flgs1 = flags[:irows.start, icols.start:icols.stop]
        ref_flgs2 = flags[irows.stop:, icols.start:icols.stop]
        ref_flgs = np.vstack((ref_flgs1, ref_flgs2))

        # Verify that all reference row pixels are flagged as reference
        # If not true, our reference rows must be either not flagged or
        # picked incorrectly.
        if not np.array_equal(ref_flgs & Flag.REFERENCE,
                              np.full(shape=refs.shape, fill_value=Flag.REFERENCE)):
            raise ValueError("Ref pixels must be flagged")

        ro_channels = l1_readout_channels(rp)
        even_refs_masked = 0
        odd_refs_masked = 0
        for ch in ro_channels:
            icol_refs_start = ch.ichan * rp.wchannel - rp.kref // 2
            icol_refs_stop = icol_refs_start + rp.wchannel
            if icol_refs_start < 0:
                icol_refs_start = 0
            if icol_refs_stop > rp.Kactive:
                icol_refs_stop = rp.Kactive

            # refs even columns
            refs_even = refs[:, icol_refs_start:icol_refs_stop:2]
            refs_even_masked = sigma_clip(refs_even, sigma=sigma)
            even_refs_masked = np.ma.count_masked(refs_even_masked)
            refs_even_masked_median = np.ma.median(refs_even_masked)

            # refs odd columns
            refs_odd = refs[:, icol_refs_start + 1:icol_refs_stop:2]
            refs_odd_masked = sigma_clip(refs_odd, sigma=sigma)
            odd_refs_masked = np.ma.count_masked(refs_odd_masked)
            refs_odd_masked_median = np.ma.median(refs_odd_masked)

            istart = icols.start + icol_refs_start
            iend = icols.start + icol_refs_stop

            # even columns - data and mask
            data[irows.start:irows.stop, istart:iend:2] -= refs_even_masked_median

            # odd columns - data and mask
            data[irows.start:irows.stop, istart + 1:iend:2] -= refs_odd_masked_median

        # Create an Output object and
        # pass all module outputs during its instantiation
        spximg_out = copy.copy(spximg)
        spximg_out.data = data

        spximg_out.setFlagsFromMask(fmask, Flag.PHANMISS)

        return pipeBase.Struct(spximg_out=spximg_out,
                               even_refs_masked=even_refs_masked,
                               odd_refs_masked=odd_refs_masked)
