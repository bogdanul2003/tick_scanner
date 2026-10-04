"""
Tests for the model_name/version path resolution added to
models/lstm_forecaster.py: get_latest_model_version, get_model_path, and
get_pytorch_model_path.

Pins the invariant from docs/TRAINING_CONFIG_JSON_PLAN.md's version-resolution
fix: a training run's .pt and .mlpackage must resolve to the SAME version
number, which only holds if the caller resolves the version once and passes
it explicitly to both path functions, rather than letting each one re-derive
"latest + 1" independently (the second call would see the first call's
already-written file and skip a version).

No DB, Core ML, or training run required — only filesystem scans against a
temp directory.
Run from src/:
    python -m unittest tests.test_model_versioning
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_forecaster import get_latest_model_version, get_model_path, get_pytorch_model_path


class GetLatestModelVersionTest(unittest.TestCase):
    def test_empty_dir_returns_zero(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(get_latest_model_version(d, "gru_v1"), 0)

    def test_missing_dir_returns_zero(self):
        self.assertEqual(get_latest_model_version("/no/such/dir", "gru_v1"), 0)

    def test_finds_highest_version_across_both_extensions(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["gru_v1_1.pt", "gru_v1_1.mlpackage", "gru_v1_2.pt", "gru_v1_3.mlpackage"]:
                open(os.path.join(d, name), "w").close()
            self.assertEqual(get_latest_model_version(d, "gru_v1"), 3)

    def test_ignores_other_model_names(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["gru_v1_5.pt", "gru_v2_9.pt"]:
                open(os.path.join(d, name), "w").close()
            self.assertEqual(get_latest_model_version(d, "gru_v1"), 5)

    def test_ignores_non_matching_files(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ["gru_v1_forecaster.pt", "gru_v1.pt", "notes.txt"]:
                open(os.path.join(d, name), "w").close()
            self.assertEqual(get_latest_model_version(d, "gru_v1"), 0)

    def test_special_regex_characters_in_model_name_are_escaped(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "gru.v1_1.pt"), "w").close()
            open(os.path.join(d, "gruXv1_9.pt"), "w").close()  # would match unescaped "." wildcard
            self.assertEqual(get_latest_model_version(d, "gru.v1"), 1)


class ModelPathModelNameVersionTest(unittest.TestCase):
    def test_explicit_version_is_used_verbatim(self):
        pt = get_pytorch_model_path("macd", "bidirectional_gru", model_name="gru_v1", version=7)
        ml = get_model_path("macd", "bidirectional_gru", model_name="gru_v1", version=7)
        self.assertTrue(pt.endswith("gru_v1_7.pt"))
        self.assertTrue(ml.endswith("gru_v1_7.mlpackage"))

    def test_omitted_version_resolves_to_latest_not_next(self):
        # get_model_path/get_pytorch_model_path resolve a None version to the
        # latest EXISTING version (for loading), not latest + 1 (for saving) —
        # callers that are about to save must compute "+1" themselves via
        # get_latest_model_version and pass it explicitly.
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models"
        )
        latest = get_latest_model_version(models_dir, "__unit_test_probe__")
        self.assertEqual(latest, 0)
        pt = get_pytorch_model_path("macd", "bidirectional_gru", model_name="__unit_test_probe__")
        self.assertTrue(pt.endswith("__unit_test_probe___0.pt"))

    def test_no_model_name_keeps_legacy_naming(self):
        pt = get_pytorch_model_path("macd", "bidirectional_gru")
        ml = get_model_path("macd", "bidirectional_gru")
        self.assertTrue(pt.endswith("macd_bidirectional_gru_forecaster.pt"))
        self.assertTrue(ml.endswith("macd_bidirectional_gru_forecaster.mlpackage"))

    def test_pt_and_mlpackage_share_one_version_when_resolved_once(self):
        """The fix this test pins: resolve version ONCE, pass to both calls."""
        with tempfile.TemporaryDirectory() as models_dir:
            open(os.path.join(models_dir, "gru_v1_1.pt"), "w").close()
            open(os.path.join(models_dir, "gru_v1_1.mlpackage"), "w").close()

            # Simulate what train_forecast_model.py's main() now does: resolve
            # the next version once, before either save call.
            version = get_latest_model_version(models_dir, "gru_v1") + 1
            pt_filename = f"gru_v1_{version}.pt"
            ml_filename = f"gru_v1_{version}.mlpackage"

            self.assertEqual(pt_filename, "gru_v1_2.pt")
            self.assertEqual(ml_filename, "gru_v1_2.mlpackage")

    def test_resolving_version_independently_per_call_would_mismatch(self):
        """Documents the bug the fix avoids: calling get_latest_model_version
        fresh before EACH save (rather than once, shared) skews the second
        call after the first save lands on disk."""
        with tempfile.TemporaryDirectory() as models_dir:
            # Nothing saved yet for this training run.
            pt_version = get_latest_model_version(models_dir, "gru_v1") + 1
            self.assertEqual(pt_version, 1)
            # Simulate trainer.save() writing the .pt before export_to_coreml() runs.
            open(os.path.join(models_dir, f"gru_v1_{pt_version}.pt"), "w").close()

            # A naive independent re-resolution for the mlpackage now sees the
            # just-written .pt and skips ahead to version 2 instead of 1.
            ml_version_if_resolved_independently = get_latest_model_version(models_dir, "gru_v1") + 1
            self.assertEqual(ml_version_if_resolved_independently, 2)
            self.assertNotEqual(pt_version, ml_version_if_resolved_independently)


if __name__ == "__main__":
    unittest.main()
