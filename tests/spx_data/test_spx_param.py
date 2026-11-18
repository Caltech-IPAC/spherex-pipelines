"""Tests that spherex-parameters is genuinely optional.

``spx_data.spx_param`` loads instrument parameters from the private
``spherex-parameters`` package when it is installed, and falls back to a
hand-authored ``_param_dict`` when it is not.  The fallback only makes the
dependency optional if the module's public API returns the same values
either way, so that is what this test compares: the public API is
evaluated twice in the same process -- once against the real package, once
with the import forced to fail -- and the two results must agree.

The test only runs where ``spherex-parameters`` is installed, since that is
the only place both branches are reachable.

When it fails, the fix is normally to update ``_param_dict`` in
``py_src/spx_data/spx_param.py`` to match the new upstream value.
"""

import importlib
import importlib.util
import sys
import unittest

import pytest

pytestmark = pytest.mark.optional_deps

HAS_SPHEREX_PARAMETERS = importlib.util.find_spec("spherex_parameters") is not None


def _public_values(mod):
    """Return everything ``spx_param`` exposes, as a {label: value} mapping.

    ``format_unit`` is not included: it formats an astropy unit and never
    reads the parameters, so it cannot differ between the two branches.

    ``lvf_params`` and ``readout_params`` are containers rather than single
    values, and the fallback deliberately holds only the subset of their
    parameters the pipelines actually read, so they are compared key by key
    (see `SpxParamFallbackTestCase.test_fallback_containers_match_package`)
    rather than as a whole.
    """
    return {
        "get_aperture()": mod.get_aperture(),
        "get_cds_read_noise()": [mod.get_cds_read_noise(d) for d in range(1, 7)],
        "get_detector_efficiency()": mod.get_detector_efficiency(),
        "get_filter_resolving_power()": mod.get_filter_resolving_power(),
        "get_num_samples()": mod.get_num_samples(),
        "get_pix_scale()": mod.get_pix_scale(),
        "get_sample_time()": mod.get_sample_time(),
        "lvf_params": mod.lvf_params,
        "readout_params": vars(mod.readout_params),
    }


def _evaluate_both_branches():
    """Evaluate the public API with and without ``spherex_parameters``.

    The fallback branch is reached by planting ``None`` in ``sys.modules``,
    which makes ``import spherex_parameters`` raise `ImportError`, and then
    reloading the module.  The live state is restored afterwards so that
    other tests in the same session are unaffected.
    """
    import spx_data.spx_param as mod

    live = _public_values(mod)

    saved = sys.modules.pop("spherex_parameters")
    sys.modules["spherex_parameters"] = None
    try:
        fallback = _public_values(importlib.reload(mod))
    finally:
        sys.modules["spherex_parameters"] = saved
        importlib.reload(mod)

    return live, fallback


@unittest.skipUnless(HAS_SPHEREX_PARAMETERS, "spherex-parameters is not installed")
class SpxParamFallbackTestCase(unittest.TestCase):
    """The public API must behave identically with and without the package."""

    # compared key by key below, not as a whole
    CONTAINERS = ("lvf_params", "readout_params")

    @classmethod
    def setUpClass(cls):
        cls.live, cls.fallback = _evaluate_both_branches()

    def test_fallback_returns_same_values_as_package(self):
        for label, live in self.live.items():
            if label in self.CONTAINERS:
                continue
            with self.subTest(call=label):
                self.assertEqual(self.fallback[label], live)

    def test_fallback_containers_match_package(self):
        """Every parameter the fallback defines must match upstream.

        The fallback holds fewer parameters than ``spherex-parameters``
        does, so only the keys it defines are compared; add a parameter to
        ``_param_dict`` when the pipelines start needing it.
        """
        for key, value in self.fallback["readout_params"].items():
            with self.subTest(param=f"readout_params.{key}"):
                self.assertEqual(value, self.live["readout_params"][key])

        for array_id, params in self.fallback["lvf_params"].items():
            for key, value in params.items():
                with self.subTest(param=f"lvf_params[{array_id}][{key!r}]"):
                    self.assertEqual(value, self.live["lvf_params"][array_id][key])


if __name__ == "__main__":
    unittest.main()
