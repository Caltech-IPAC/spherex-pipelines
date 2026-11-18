__all__ = ["FluxCorrectionsCube"]

# Related third-party imports
from lsst.daf.butler import DeferredDatasetHandle
import numpy as np

_FLUX_CORRECTIONS_CACHE = {}  # module-level cache


class FluxCorrectionsCube():
    """Holds a 3D cube of flux corrections for all detectors."""

    def __init__(self, deferred_maps: list[DeferredDatasetHandle],
                 run_cache_key: str):
        """Initialize the flux corrections cube, using a cache to avoid
        redundant computations.

        Parameters
        ----------
        deferred_maps : `list` of `lsst.daf.butler.DeferredDatasetHandle`
            A list of deferred `spx_data.SPHERExImage` 2D ratio maps for each
            detector.
        run_cache_key : `str`
            The L3 flux corrections run name.
        """
        if run_cache_key in _FLUX_CORRECTIONS_CACHE:
            self.cube = _FLUX_CORRECTIONS_CACHE[run_cache_key]
        else:
            self.cube = self._build_cube(deferred_maps)
            _FLUX_CORRECTIONS_CACHE[run_cache_key] = self.cube

    def _build_cube(self, deferred_maps: list[DeferredDatasetHandle]) -> np.ndarray:
        """Build a 3D cube of flux corrections from a list of deferred maps."""
        # Fetch all maps once
        maps = [m.get() for m in deferred_maps]

        # Create 3D cube with shape (detectors, y pixels, x pixels)
        cube = np.empty(
            (len(maps), ) + maps[0].shape,
            dtype=maps[0].dtype)
        for i, m in enumerate(maps):
            detector = int(m.getArrayId())
            cube[detector - 1] = m.data
            maps[i] = None  # release SPHERExImage to free memory once data is copied

        return cube
