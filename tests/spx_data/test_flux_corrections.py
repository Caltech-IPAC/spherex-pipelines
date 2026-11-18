# Standard library imports
from typing import Literal

# Related third party imports
import pytest
from unittest.mock import MagicMock

# Local application imports
from spx_butler.defaults import fake_products
from spx_data import FluxCorrectionsCube


class TestFluxCorrectionsCube():
    """Test Flux Corrections Cube class."""

    @pytest.fixture(autouse=True)
    def clear_flux_cache(self):
        """Clear the flux corrections cache before each test for isolation."""
        from spx_data import flux_corrections
        yield
        flux_corrections._FLUX_CORRECTIONS_CACHE = {}

    @pytest.fixture()
    def flux_handles(self, request) -> list[list[MagicMock]]:
        """Create sets of 6 detector handles with flux factors.

        Parameters
        ----------
        request : `pytest.FixtureRequest`
            Contains test request context via `param` attribute with `list` of
            `float` correction factors.

        Returns
        -------
        all_handles : `list` of `list` of `unittest.mock.MagicMock`
            A nested list of 6 mocked, deferred `spx_data.SPHERExImage`. The
            first element corresponds to the first requested correction
            factor, and so on.
        """
        factors = request.param
        all_handles = []
        for f in factors:
            handles = []
            for i in range(6):
                h = MagicMock()
                h.get.return_value = fake_products.fake_l3_flux_corrections(i + 1, factor=f)
                handles.append(h)
            all_handles.append(handles)
        return all_handles

    @pytest.mark.parametrize(
        "flux_handles, case, keys",
        [([1, 1], "hit", ["run_key1", "run_key1"]),  # cache hit on second call with same key
         ([1, 2], "rebuild", ["run_key1", "run_key2"])],  # different keys; cache not used on second call
        indirect=["flux_handles"])
    def test_caching(self, flux_handles: list[list[MagicMock]], case: Literal["hit", "rebuild"],
                     keys: list[str]):
        """Test caching behavior.

        Parameters
        ----------
        flux_handles : `list` of `list` of `unittest.mock.MagicMock`
            Nested list of mocked deferred handles for flux corrections.
        case : {"hit", "rebuild"}
            Description of the test case used for validation. Either "hit" to
            check for cache miss and hit or "rebuild" to check for cache miss
            and rebuild.
        keys : `list` of `str`
            List of cache keys to use for each set of handles. Identical keys
            should lead to a cache hit on the second call, while different keys
            should lead to a cache miss and rebuild.
        """
        cubes = []  # save cube instances for comparison
        for handle_set, k in zip(flux_handles, keys):
            cube = FluxCorrectionsCube(handle_set, run_cache_key=k)
            cubes.append(cube)

        # Check cache behavior during initialization
        if case == "hit":
            for h in flux_handles[0]:
                h.get.assert_called_once()  # First set of handles are called once
            for h in flux_handles[1]:
                h.get.assert_not_called()  # Cache hit -> second set of handles are not called
        elif case == "rebuild":
            for handle_set in flux_handles:
                for h in handle_set:
                    h.get.assert_called_once()  # Different keys -> both sets of handles are called once

        # Check object identity
        if len(cubes) == 2:
            if case == "hit":
                assert cubes[0].cube is cubes[1].cube  # Expect same 3D cube for cache hit
            elif case == "rebuild":
                assert cubes[0].cube is not cubes[1].cube  # Expect different 3D cubes for distinct keys
