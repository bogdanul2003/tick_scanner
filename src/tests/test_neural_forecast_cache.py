"""
Tests for CoreMLForecaster's model-metadata cache.

The module-level `_model_cache` in models/neural_forecast.py lets a second
CoreMLForecaster for the same .mlpackage skip re-parsing the file. These tests
pin the invariant that makes that safe: a cache-hit instance must be
indistinguishable from a freshly parsed one.

Regression covered: the cache-hit path once copied out only a subset of the
fields it stored, so `feature_names` and `target_size` were missing on every
instance after the first, and the second forecast raised AttributeError.

No .mlpackage, coremltools install, or NPU required — coremltools is stubbed.
Run from src/:
    python -m unittest tests.test_neural_forecast_cache
"""
import os
import sys
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import neural_forecast as nf


# Metadata of a multi-feature residual model — the configuration that exposed
# the original bug (3 inputs, 2 targets, non-default everything).
MULTI_FEATURE_METADATA = {
    "mean": "[0.1, 0.02, 1.0]",
    "std": "[0.5, 0.1, 0.3]",
    "seq_length": "30",
    "forecast_horizon": "5",
    "normalization_type": "global",
    "include_delta": "True",
    "residual_target": "True",
    "feature_names": "macd,delta,volume",
    "input_size": "3",
    "target_size": "2",
    "hidden_size": "32",
    "num_layers": "2",
    "batch_size": "64",
}

# Every attribute _read_metadata is responsible for populating.
METADATA_FIELDS = (
    "seq_length", "forecast_horizon", "normalization_type", "include_delta",
    "residual_target", "feature_names", "input_size", "target_size",
    "hidden_size", "num_layers", "batch_size",
)

_real_exists = os.path.exists


def _fake_exists(path):
    """Pretend any .mlpackage path exists; defer to the real check otherwise."""
    return True if str(path).endswith(".mlpackage") else _real_exists(path)


class _FakeMLModel:
    """Stand-in for ct.models.MLModel — carries metadata, predicts nothing."""

    def __init__(self, path, metadata=None):
        self.path = path
        self.user_defined_metadata = dict(metadata or MULTI_FEATURE_METADATA)


class CoreMLForecasterCacheTest(unittest.TestCase):

    def setUp(self):
        nf._model_cache.clear()
        self.addCleanup(nf._model_cache.clear)

        self._install_coremltools(lambda path: _FakeMLModel(path))

        patcher = mock.patch("os.path.exists", _fake_exists)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _install_coremltools(self, model_factory):
        """Put a stub coremltools in sys.modules for the duration of one test."""
        stub = types.ModuleType("coremltools")
        stub.models = types.SimpleNamespace(MLModel=model_factory)
        patcher = mock.patch.dict(sys.modules, {"coremltools": stub})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_cache_hit_matches_cold_load(self):
        """A cached instance carries the same metadata as a freshly parsed one."""
        path = "models/fake_multi.mlpackage"

        cold = nf.CoreMLForecaster(path)          # parses the metadata
        warm = nf.CoreMLForecaster(path)          # cache hit

        for field in METADATA_FIELDS:
            self.assertEqual(
                getattr(cold, field), getattr(warm, field),
                msg=f"{field} differs between cold and cached load",
            )
        np.testing.assert_array_equal(cold.mean, warm.mean)
        np.testing.assert_array_equal(cold.std, warm.std)
        self.assertIs(cold.model, warm.model, "cached instance re-loaded the model")
        self.assertTrue(warm.is_available)

    def test_cache_hit_restores_multi_feature_fields(self):
        """The specific fields the old cache-hit path dropped."""
        path = "models/fake_multi.mlpackage"
        nf.CoreMLForecaster(path)
        warm = nf.CoreMLForecaster(path)

        # Reading these is what used to raise AttributeError.
        self.assertEqual(warm.feature_names, ["macd", "delta", "volume"])
        self.assertEqual(warm.input_size, 3)
        self.assertEqual(warm.target_size, 2)
        self.assertTrue(warm.residual_target)

    def test_cache_entry_covers_every_attribute_it_claims(self):
        """
        The cache entry is the attribute set, so each stored key must land on the
        instance verbatim. This is the invariant that keeps a newly added
        metadata field from being silently dropped on the cached path.
        """
        path = "models/fake_multi.mlpackage"
        forecaster = nf.CoreMLForecaster(path)

        for key, value in nf._model_cache[path].items():
            actual = getattr(forecaster, key, mock.sentinel.missing)
            self.assertIsNot(actual, mock.sentinel.missing,
                             msg=f"cached key {key!r} is not an instance attribute")
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual, value)
            else:
                self.assertEqual(actual, value, msg=f"cached {key!r} not applied")

    def test_per_path_isolation(self):
        """Two different models don't share a cache entry."""
        univariate = dict(MULTI_FEATURE_METADATA,
                          include_delta="False", residual_target="False",
                          feature_names="macd", input_size="1", target_size="1",
                          mean="0.05", std="0.4")

        def factory(path):
            meta = univariate if "uni" in path else MULTI_FEATURE_METADATA
            return _FakeMLModel(path, meta)

        self._install_coremltools(factory)

        multi = nf.CoreMLForecaster("models/fake_multi.mlpackage")
        uni = nf.CoreMLForecaster("models/fake_uni.mlpackage")
        multi_again = nf.CoreMLForecaster("models/fake_multi.mlpackage")

        self.assertEqual(uni.feature_names, ["macd"])
        self.assertEqual(uni.target_size, 1)
        self.assertEqual(uni.mean, 0.05)
        self.assertEqual(multi_again.feature_names, multi.feature_names)
        self.assertEqual(multi_again.target_size, 2)

    def test_missing_model_file_keeps_usable_defaults(self):
        """No model on disk: unavailable, but every attribute still present."""
        with mock.patch("os.path.exists", lambda p: False):
            forecaster = nf.CoreMLForecaster("models/absent.mlpackage")

        self.assertFalse(forecaster.is_available)
        self.assertEqual(forecaster.feature_names, ["macd"])
        self.assertEqual(forecaster.input_size, 1)
        self.assertEqual(forecaster.target_size, 1)

    def test_failed_parse_leaves_forecaster_unavailable(self):
        """
        A model whose metadata cannot be parsed must not report itself available
        — predicting with default normalization stats would return silent
        garbage. It must not be cached either.
        """
        def exploding_factory(path):
            raise ValueError("corrupt metadata")

        self._install_coremltools(exploding_factory)

        path = "models/broken.mlpackage"
        forecaster = nf.CoreMLForecaster(path)

        self.assertFalse(forecaster.is_available)
        self.assertIsNone(forecaster.model)
        self.assertEqual(forecaster.feature_names, ["macd"])
        self.assertEqual(forecaster.target_size, 1)
        self.assertNotIn(path, nf._model_cache, "a failed load was cached")

    def test_missing_coremltools_is_not_fatal(self):
        """Without coremltools the forecaster degrades instead of raising."""
        with mock.patch.dict(sys.modules, {"coremltools": None}):
            forecaster = nf.CoreMLForecaster("models/fake_multi.mlpackage")

        self.assertFalse(forecaster.is_available)
        self.assertEqual(forecaster.target_size, 1)


if __name__ == "__main__":
    unittest.main()
