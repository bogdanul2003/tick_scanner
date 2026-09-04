"""
Tests for scripts/config_utils.py — the CLI/JSON/default merge that backs
train_forecast_model.py's --config and evaluate_forecast_model.py's --config.

No DB, Core ML, or training run required.
Run from src/:
    python -m unittest tests.test_config_utils
"""
import json
import os
import sys
import tempfile
import unittest

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, "scripts"))

import config_utils


def _all_none(keys):
    return {k: None for k in keys}


class ResolveConfigPrecedenceTest(unittest.TestCase):
    """CLI (non-None) > JSON > DEFAULT_CONFIG, key by key."""

    def test_no_cli_no_json_returns_defaults(self):
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        resolved = config_utils.resolve_config(cli, json_config=None)
        self.assertEqual(resolved, config_utils.DEFAULT_CONFIG)

    def test_json_value_overrides_default(self):
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        resolved = config_utils.resolve_config(cli, json_config={"epochs": 300})
        self.assertEqual(resolved["epochs"], 300)
        # Untouched keys still fall through to defaults.
        self.assertEqual(resolved["batch_size"], config_utils.DEFAULT_CONFIG["batch_size"])

    def test_explicit_cli_value_overrides_json(self):
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        cli["epochs"] = 500
        resolved = config_utils.resolve_config(cli, json_config={"epochs": 300})
        self.assertEqual(resolved["epochs"], 500)

    def test_explicit_cli_value_overrides_default_with_no_json(self):
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        cli["hidden_size"] = 128
        resolved = config_utils.resolve_config(cli, json_config=None)
        self.assertEqual(resolved["hidden_size"], 128)

    def test_cli_boolean_false_is_not_distinguishable_from_unset(self):
        # store_true flags surface as None (unset) or True (passed) from argparse,
        # never False — so a JSON boolean is the only way to get an explicit False.
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        resolved = config_utils.resolve_config(cli, json_config={"include_delta": True})
        self.assertTrue(resolved["include_delta"])

    def test_result_covers_every_default_key_and_no_extras(self):
        cli = _all_none(config_utils.DEFAULT_CONFIG)
        resolved = config_utils.resolve_config(cli, json_config={"epochs": 1})
        self.assertEqual(set(resolved.keys()), set(config_utils.DEFAULT_CONFIG.keys()))


class LoadJsonConfigTest(unittest.TestCase):
    def test_round_trips_a_config_file(self):
        payload = {"model_name": "test_model", "epochs": 42}
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "cfg.json")
            with open(path, "w") as f:
                json.dump(payload, f)
            loaded = config_utils.load_json_config(path)
        self.assertEqual(loaded, payload)


class GetModelNameTest(unittest.TestCase):
    def test_returns_model_name_when_present(self):
        self.assertEqual(config_utils.get_model_name({"model_name": "foo"}), "foo")

    def test_raises_when_missing(self):
        with self.assertRaises(ValueError):
            config_utils.get_model_name({})

    def test_raises_when_empty_string(self):
        with self.assertRaises(ValueError):
            config_utils.get_model_name({"model_name": ""})


if __name__ == "__main__":
    unittest.main()
