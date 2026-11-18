import numpy as np
import pytest
from types import SimpleNamespace
from spx_data import Flag
from spx_utils import BSHFlagger, BSHConfig, apply_bsh_flags_to_spx


class MockSPHERExImage:
    """Minimal SPHERExImage stand-in for BSHFlagger tests."""

    def __init__(self, data, flags=None, array_id=2, telapse=1.0):
        self.data = np.asarray(data, dtype=float)  # e/s
        if flags is None:
            flags = np.zeros_like(self.data, dtype=np.int32)
        self.flags = flags.astype(np.int32, copy=False)
        self._array_id = int(array_id)
        self._telapse = float(telapse)
        self.header = {}

    def getFlagsAsMask(self, flag: Flag) -> np.ndarray:
        # return boolean mask where the bit is set
        return (self.flags & int(flag)) != 0

    def getExposureDuration(self):
        # matches interface used in BSHFlagger.__init__
        return SimpleNamespace(sec=self._telapse)

    def getArrayId(self) -> int:
        return self._array_id

    def setFlagsFromMask(self, mask: np.ndarray, flag: Flag):
        """Mock method to set flags from a boolean mask."""
        self.flags[mask] |= int(flag)


@pytest.fixture
def make_spx():
    def _make(
        shape=(32, 32),
        array_id=2,
        telapse=10.0,
        fill_rate=100.0,
        seed=None,
    ):
        data = np.full(shape, fill_rate, dtype=float)   # counts/s
        flags = np.zeros(shape, dtype=np.int32)
        return MockSPHERExImage(data=data, flags=flags, array_id=array_id, telapse=telapse)
    return _make


def set_bit(mask: np.ndarray, bit: int, rows, cols):
    mask[rows, cols] |= bit


def test_ctor_sets_detector_thresholds_and_header(make_spx):
    """Verify init chooses the right min_overflow by detector id
    and writes header keys.
    """
    # Detector 5 should pick d456 threshold
    spx = make_spx(array_id=5, telapse=5.0)
    cfg = BSHConfig()
    flagger = BSHFlagger(spx, cfg)

    assert flagger.min_overflow == cfg.min_overflow_d456
    assert flagger.header["TRANSFRC"][0] == 0.0  # no transients yet
    assert flagger.header["MXTRNSFR"][0] == cfg.max_trans_frac
    assert "MINOVRFL" in flagger.header


def test_mark_transient_halo_convolution_kernel(make_spx):
    """A single transient pixel should mark itself and
    4-neighbors in halo via the 4-connected kernel.
    """
    spx = make_spx(shape=(9, 9))
    flags = spx.flags

    r, c = 4, 4
    set_bit(flags, int(Flag.TRANSIENT), np.array([r]), np.array([c]))
    spx.flags = flags

    flagger = BSHFlagger(spx, BSHConfig())
    # No snowball/bloom yet
    assert np.count_nonzero(flagger.snowball_flag) == 0
    assert np.count_nonzero(flagger.bloom_flag) == 0

    flagger.mark_transient_halo()
    # halo_flag uses 4-neighborhood + center
    expected = {(4, 4), (3, 4), (5, 4), (4, 3), (4, 5)}
    got = set(map(tuple, np.column_stack(np.where(flagger.halo_flag > 0))))
    assert expected.issubset(got)


def test_apply_bsh_flags_sets_bits_and_satellite_halo(make_spx):
    """Projection function should set BLOOM/SNOWBALL/HALO and SATELLITE_HALO bits."""
    spx = make_spx(shape=(16, 16))
    flagger = BSHFlagger(spx, BSHConfig())

    # Seed internal per-feature masks
    flagger.bloom_flag[2, 2] = 1
    flagger.snowball_flag[3, 3] = 2
    flagger.halo_flag[4, 4] = 1

    # Pretend satellite path was detected by eccentric branch
    flagger.halo_rows = np.array([8, 8, 8])
    flagger.halo_cols = np.array([5, 6, 7])
    # Indicate at least one satellite streak was detected
    flagger.header["SNWSTRK"] = (1, "Saturating streaks detected by BSH algorithm")

    # Apply into spx.flags
    apply_bsh_flags_to_spx(spx, flagger)

    assert (spx.flags[2, 2] & int(Flag.BLOOM)) != 0
    assert (spx.flags[3, 3] & int(Flag.SNOWBALL)) != 0
    assert (spx.flags[4, 4] & int(Flag.HALO)) != 0
    # All track pixels receive SATELLITE_HALO
    for cc in (5, 6, 7):
        assert (spx.flags[8, cc] & int(Flag.SATELLITE_HALO)) != 0


def test_apply_halo_flag_eccentric_border_expand_and_project(make_spx):
    """Force the eccentric-track branch (r_max > r_) and
    verify SNWSTRK and SATELLITE_HALO pixels.
    """
    spx = make_spx(shape=(32, 32), telapse=10.0, fill_rate=200.0)
    cfg = BSHConfig(satellite_halomethod="border_expand")  # avoid skimage dependency in test
    flagger = BSHFlagger(spx, cfg)

    # Create a long horizontal "track" (simulate snowball center collection)
    rr = np.array([16] * 15, dtype=int)
    cc = np.arange(8, 23, dtype=int)
    flagger.bloom_rows = rr.copy()
    flagger.bloom_cols = cc.copy()

    # Mark the center pixels as "OVERFLOW" so _apply_halo_flag can find the centroid
    over_mask = spx.flags
    over_mask[rr, cc] |= int(Flag.OVERFLOW)
    spx.flags = over_mask
    flagger.over = spx.getFlagsAsMask(Flag.OVERFLOW)

    # Fluence already high from data*telapse; call the method
    flagger._apply_halo_flag()

    # We should be in the eccentric path (track longer than circular radius),
    # which increments SNWSTRK and fills halo_rows/cols.
    sat_str = flagger.header.get("SNWSTRK", (0,))[0] if isinstance(
        flagger.header.get("SNWSTRK"), tuple
    ) else flagger.header.get("SNWSTRK", 0)
    assert sat_str > 0
    assert hasattr(flagger, "halo_rows") and flagger.halo_rows.size > 0

    # Now project bits into spx.flags, verify SATELLITE_HALO is set on halo rows/cols
    apply_bsh_flags_to_spx(spx, flagger)
    idx = (flagger.halo_rows, flagger.halo_cols)
    assert np.all((spx.flags[idx] & int(Flag.SATELLITE_HALO)) != 0)


def test_get_nonoverflow_bloom_flag_sets_flag5(make_spx):
    """Craft a small scenario that triggers the
    non-overflow bloom path, setting bloom_flag=5.
    """
    shape = (21, 21)
    spx = make_spx(shape=shape, telapse=10.0, fill_rate=500.0)  # fluence=5000e-, comfortably above thresholds
    cfg = BSHConfig()
    # Make min_overflow small so thresholds are easy to satisfy
    cfg.min_overflow_d123 = 100.0
    cfg.min_overflow_d456 = 100.0

    flagger = BSHFlagger(spx, cfg)
    r0, c0 = 10, 10

    # Center candidates: TRANSIENT + EARLY (SUR_ERROR), no OVERFLOW
    spx.flags[r0, c0] |= int(Flag.TRANSIENT) | int(Flag.SUR_ERROR)

    # Build a small ring of late transients (TRANSIENT & ~EARLY) near center,
    # still with no overflow.
    ring = [(r0, c0 + 1), (r0, c0 - 1), (r0 + 1, c0), (r0 - 1, c0)]
    for r, c in ring:
        spx.flags[r, c] |= int(Flag.TRANSIENT)  # no SUR_ERROR, so these are "late"

    # Refresh the masks inside the flagger to see the updated flags
    flagger.transient = spx.getFlagsAsMask(Flag.TRANSIENT)
    flagger.early = spx.getFlagsAsMask(Flag.SUR_ERROR)
    flagger.over = spx.getFlagsAsMask(Flag.OVERFLOW)

    # Run the non-overflow bloom detection
    flagger.get_nonoverflow_bloom_flag()

    # Expect bloom_flag=5 somewhere in the small collection (center+ring)
    sel = (flagger.bloom_flag > 0)
    assert np.count_nonzero(sel) > 0
    # specifically the algorithm assigns "5" when both early(5) and late(1) are present
    assert np.any(flagger.bloom_flag[sel] == 5)


def test_remove_overlapping_indices_helper():
    """Direct test of the static helper that removes overlaps preserving order."""
    A = (np.array([0, 1, 2, 3]), np.array([5, 5, 6, 7]))
    B = (np.array([1, 2]), np.array([5, 6]))
    rows, cols = BSHFlagger._remove_overlapping_indices_BfromA(A, B)
    kept = set(map(tuple, np.column_stack([rows, cols])))
    # (0,5) and (3,7) remain; (1,5) and (2,6) removed
    assert kept == {(0, 5), (3, 7)}
