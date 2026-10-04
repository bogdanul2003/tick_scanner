"""
Tests for seed-ensemble forecasting: EnsembleForecaster, the ensemble registry, and
the bulk forecast that replaces ARIMA behind "Show Bullish Forecast".

What these pin, because a silent wrong answer here looks like a right one:
  - the ensemble is the plain mean of its members and refuses mismatched members
  - a retrain's state overrides the registry's versions, and an ensemble with a
    missing model file is reported unavailable rather than served short-handed
  - forecast_symbols returns the ARIMA endpoint's shape, so the UI and the
    combined-forecast cache treat both engines alike

The DB, macd_utils and Core ML are all stubbed — no Postgres, no .mlpackage, no NPU.
Run from src/:
    python -m unittest tests.test_ensemble_service
"""
import json
import os
import sys
import tempfile
import types
import unittest
from datetime import date
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.neural_forecast import EnsembleForecaster
from services import ensemble_service as es


class _ConstantForecaster:
    """A member that forecasts `last + offset * (k+1)` for the macd column."""

    def __init__(self, offset, seq_length=5, horizon=5, feature_names=("macd", "delta"), available=True):
        self.offset = offset
        self.seq_length = seq_length
        self.forecast_horizon = horizon
        self.include_delta = True
        self.feature_names = list(feature_names)
        self.input_size = len(feature_names)
        self.is_available = available

    def predict(self, sequence, prev_value=None):
        seq = np.asarray(sequence, dtype=np.float32)
        last = float(seq[-1, 0] if seq.ndim == 2 else seq[-1])
        steps = np.arange(1, self.forecast_horizon + 1, dtype=np.float32)
        out = np.zeros((self.forecast_horizon, 2), dtype=np.float32)
        out[:, 0] = last + self.offset * steps
        out[:, 1] = self.offset
        return out


class EnsembleForecasterTest(unittest.TestCase):

    def test_predict_is_the_mean_of_the_members(self):
        ensemble = EnsembleForecaster([_ConstantForecaster(1.0), _ConstantForecaster(2.0), _ConstantForecaster(6.0)])
        forecast = ensemble.predict(np.array([0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32))
        np.testing.assert_allclose(forecast[:, 0], 10.0 + 3.0 * np.arange(1, 6))
        np.testing.assert_allclose(forecast[:, 1], 3.0)

    def test_exposes_what_members_share(self):
        ensemble = EnsembleForecaster([_ConstantForecaster(1.0, seq_length=30), _ConstantForecaster(2.0, seq_length=30)])
        self.assertEqual(ensemble.seq_length, 30)
        self.assertEqual(ensemble.feature_names, ["macd", "delta"])
        self.assertTrue(ensemble.is_available)

    def test_members_with_different_inputs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "seq_length"):
            EnsembleForecaster([_ConstantForecaster(1.0, seq_length=30), _ConstantForecaster(1.0, seq_length=60)])
        with self.assertRaisesRegex(ValueError, "feature_names"):
            EnsembleForecaster([_ConstantForecaster(1.0),
                                _ConstantForecaster(1.0, feature_names=("macd", "delta", "open-close"))])

    def test_empty_ensemble_is_refused(self):
        with self.assertRaises(ValueError):
            EnsembleForecaster([])

    def test_one_unloaded_member_makes_the_ensemble_unavailable(self):
        ensemble = EnsembleForecaster([_ConstantForecaster(1.0), _ConstantForecaster(1.0, available=False)])
        self.assertFalse(ensemble.is_available)


class _RegistryFixture(unittest.TestCase):
    """A temp configs dir + models dir with one or two registered ensembles."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.configs_dir = os.path.join(tmp.name, "configs")
        self.models_dir = os.path.join(tmp.name, "models")
        os.makedirs(self.configs_dir)
        os.makedirs(self.models_dir)
        self._write_config("a.json", {"model_name": "model_a", "signal_type": "macd", "watchlist": "sp500"})
        self._write_config("b.json", {"model_name": "model_b", "watchlist": "sp500"})
        self.registry_path = os.path.join(self.configs_dir, "ensembles.json")

    def _write_config(self, name, body):
        with open(os.path.join(self.configs_dir, name), "w") as f:
            json.dump(body, f)

    def _write_registry(self, body):
        with open(self.registry_path, "w") as f:
            json.dump(body, f)

    def _load(self):
        return es.load_registry(self.registry_path, self.configs_dir)

    def _add_models(self, model_name, versions):
        for v in versions:
            os.makedirs(os.path.join(self.models_dir, f"{model_name}_{v}.mlpackage"))

    def _valid(self):
        return {
            "default": "a",
            "ensembles": [
                {"id": "a", "label": "A", "config": "a.json", "versions": [1, 2, 3], "metrics": {"mae": 0.9}},
                {"id": "b", "config": "b.json", "versions": [2, 3, 4]},
            ],
        }


class LoadRegistryTest(_RegistryFixture):

    def test_entries_carry_their_configs_model_name(self):
        self._write_registry(self._valid())
        registry = self._load()
        self.assertEqual(registry["default"], "a")
        self.assertEqual([e["model_name"] for e in registry["ensembles"]], ["model_a", "model_b"])
        self.assertEqual(registry["ensembles"][1]["label"], "b")  # label falls back to the id
        self.assertEqual(registry["seeds"], es.DEFAULT_SEEDS)

    def test_duplicate_ids_are_refused(self):
        body = self._valid()
        body["ensembles"][1]["id"] = "a"
        self._write_registry(body)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self._load()

    def test_missing_config_is_refused(self):
        body = self._valid()
        body["ensembles"][0]["config"] = "nope.json"
        self._write_registry(body)
        with self.assertRaisesRegex(ValueError, "missing config"):
            self._load()

    def test_empty_or_non_integer_versions_are_refused(self):
        for bad in ([], [1, "2"], [0]):
            body = self._valid()
            body["ensembles"][0]["versions"] = bad
            self._write_registry(body)
            with self.assertRaisesRegex(ValueError, "versions"):
                self._load()

    def test_default_must_be_listed(self):
        body = self._valid()
        body["default"] = "zzz"
        self._write_registry(body)
        with self.assertRaisesRegex(ValueError, "default"):
            self._load()

    def test_signal_line_models_are_refused(self):
        self._write_config("a.json", {"model_name": "model_a", "signal_type": "signal_line"})
        self._write_registry(self._valid())
        with self.assertRaisesRegex(ValueError, "only MACD"):
            self._load()


class ResolveEnsembleTest(_RegistryFixture):

    def setUp(self):
        super().setUp()
        self._write_registry(self._valid())
        self.registry = self._load()

    def _resolve(self, ensemble_id=None, state=None):
        return es.resolve_ensemble(ensemble_id, self.registry, state or {}, self.models_dir)

    def test_none_means_the_default(self):
        self.assertEqual(self._resolve()["id"], "a")
        self.assertTrue(self._resolve()["is_default"])
        self.assertFalse(self._resolve("b")["is_default"])

    def test_unknown_id_raises(self):
        with self.assertRaises(es.EnsembleNotFoundError):
            self._resolve("zzz")

    def test_available_only_when_every_version_is_on_disk(self):
        self._add_models("model_a", [1, 2])
        resolved = self._resolve("a")
        self.assertFalse(resolved["available"])
        self.assertEqual(resolved["missing_versions"], [3])

        self._add_models("model_a", [3])
        self.assertTrue(self._resolve("a")["available"])

    def test_retrain_state_overrides_versions_and_metrics(self):
        self._add_models("model_a", [1, 2, 3, 7, 8, 9])
        state = {"a": {"versions": [7, 8, 9], "trained_at": "2026-10-05T10:00:00", "metrics": {"mae": 0.8}}}
        resolved = self._resolve("a", state)
        self.assertEqual(resolved["versions"], [7, 8, 9])
        self.assertEqual(resolved["metrics"], {"mae": 0.8})
        from datetime import datetime
        # recorded without an offset, reported as that local time with one
        self.assertEqual(datetime.fromisoformat(resolved["trained_at"]).replace(tzinfo=None),
                         datetime(2026, 10, 5, 10, 0, 0))
        self.assertIsNotNone(datetime.fromisoformat(resolved["trained_at"]).tzinfo)
        # an ensemble the state says nothing about keeps the registry's values
        self.assertEqual(self._resolve("b", state)["versions"], [2, 3, 4])

    def test_trained_at_falls_back_to_the_checkpoint_files_time(self):
        self.assertIsNone(self._resolve("a")["trained_at"])  # nothing on disk yet
        self._add_models("model_a", [1, 2, 3])
        from datetime import datetime
        trained = {1: 1_780_000_000, 2: 1_790_000_000, 3: 1_785_000_000}
        for v, when in trained.items():
            checkpoint = os.path.join(self.models_dir, f"model_a_{v}.pt")
            open(checkpoint, "w").close()
            os.utime(checkpoint, (when, when))
        # Loading a Core ML model rewrites its package, so the package time is later
        # than training and must not be what gets reported.
        self.assertEqual(
            datetime.fromisoformat(self._resolve("a")["trained_at"]).timestamp(), 1_790_000_000)

    def test_trained_at_uses_the_package_data_dir_when_the_checkpoint_is_gone(self):
        self._add_models("model_a", [1, 2, 3])
        from datetime import datetime
        for v in (1, 2, 3):
            data_dir = os.path.join(self.models_dir, f"model_a_{v}.mlpackage", "Data")
            os.makedirs(data_dir)
            os.utime(data_dir, (1_770_000_000, 1_770_000_000))
        self.assertEqual(
            datetime.fromisoformat(self._resolve("a")["trained_at"]).timestamp(), 1_770_000_000)

    def test_unavailable_ensemble_is_not_loaded(self):
        with self.assertRaises(es.EnsembleUnavailableError):
            es.load_forecaster(self._resolve("a"), self.models_dir)


class StateFileTest(unittest.TestCase):

    def test_round_trip_and_missing_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "state.json")
            self.assertEqual(es.load_state(path), {})
            es.save_state({"a": {"versions": [4, 5, 6]}}, path)
            self.assertEqual(es.load_state(path), {"a": {"versions": [4, 5, 6]}})
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_corrupt_state_is_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "state.json")
            with open(path, "w") as f:
                f.write("{not json")
            self.assertEqual(es.load_state(path), {})


def _rows(last_macd, count=40):
    """Cache rows shaped like get_macd_for_range_bulk's, MACD rising by 1 to `last_macd`."""
    return [
        {"symbol": "X", "date": f"d{i}", "open": 10.0, "close": 10.5, "volume": 1000,
         "macd": last_macd - (count - 1 - i), "signal_line": 0.0}
        for i in range(count)
    ]


class ForecastSymbolsTest(unittest.TestCase):
    """forecast_symbols against stubbed data, DB and models."""

    END_DATE = date(2026, 10, 2)  # a Friday

    def setUp(self):
        self.cached = []
        self.recorded = []
        self.rows = {
            "RISING": _rows(last_macd=-1.0),    # negative now, forecast crosses zero
            "FALLING": _rows(last_macd=5.0),    # positive now
            "SHORT": _rows(last_macd=1.0, count=4),
        }

        fake_macd_utils = types.ModuleType("macd_utils")
        fake_macd_utils.get_latest_market_date = lambda: self.END_DATE
        fake_macd_utils.get_macd_for_range_bulk = lambda symbols, start, end: {s: self.rows.get(s, []) for s in symbols}
        fake_macd_utils.get_macd_for_range = mock.Mock(side_effect=AssertionError("must use the bulk loader"))
        fake_macd_utils.get_macd_for_date = mock.Mock()

        fake_db_utils = types.ModuleType("db_utils")
        fake_db_utils.cache_macd_positive_forecast = lambda symbol, d, flag: self.cached.append((symbol, d, flag))

        def fake_save(model, as_of_date, rows, **meta):
            self.recorded.append({"model": model, "as_of_date": as_of_date, "rows": rows, **meta})
            return len(rows)
        fake_db_utils.save_forecast_predictions = fake_save

        # forecast_utils binds macd_utils names at import, so it must be re-imported
        # against the stub rather than reused from an earlier test module.
        sys.modules.pop("forecast_utils", None)
        patcher = mock.patch.dict(sys.modules, {"macd_utils": fake_macd_utils, "db_utils": fake_db_utils})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: sys.modules.pop("forecast_utils", None))

        ensemble = {"id": "a", "model_name": "model_a", "signal_type": "macd", "versions": [1, 2, 3],
                    "available": True, "trained_at": "2026-10-04T09:43:00"}
        self.forecaster = EnsembleForecaster([_ConstantForecaster(0.5), _ConstantForecaster(1.0), _ConstantForecaster(1.5)])
        for target, value in (("resolve_ensemble", lambda ensemble_id=None: ensemble),
                              ("load_forecaster", lambda e: self.forecaster)):
            p = mock.patch.object(es, target, value)
            p.start()
            self.addCleanup(p.stop)

    def test_result_has_the_arima_endpoints_shape(self):
        result = es.forecast_symbols(["rising"], "a")["RISING"]
        self.assertEqual(set(result), {"will_become_positive", "forecasted_macd", "forecasted_dates", "details"})
        # the five weekdays after Friday 2026-10-02
        self.assertEqual(result["forecasted_dates"],
                         ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"])
        self.assertEqual(list(result["forecasted_macd"]), result["forecasted_dates"])
        # mean member offset is 1.0/day from a last MACD of -1.0
        np.testing.assert_allclose(list(result["forecasted_macd"].values()), [0.0, 1.0, 2.0, 3.0, 4.0], atol=1e-5)
        self.assertEqual(result["details"]["last_macd"], -1.0)
        self.assertEqual(result["details"]["ensemble"], "a")
        self.assertEqual(result["details"]["versions"], [1, 2, 3])

    def test_will_become_positive_and_its_cache_write(self):
        results = es.forecast_symbols(["RISING", "FALLING"], "a")
        self.assertTrue(results["RISING"]["will_become_positive"])
        self.assertFalse(results["FALLING"]["will_become_positive"])
        self.assertEqual(sorted(self.cached),
                         [("FALLING", self.END_DATE, False), ("RISING", self.END_DATE, True)])

    def test_symbol_without_enough_data_fails_alone_and_is_not_cached(self):
        results = es.forecast_symbols(["SHORT", "RISING", "UNKNOWN"], "a")
        for symbol in ("SHORT", "UNKNOWN"):
            self.assertFalse(results[symbol]["will_become_positive"])
            self.assertEqual(results[symbol]["forecasted_macd"], [])
            self.assertIn("error", results[symbol]["details"])
        self.assertNotIn("error", results["RISING"]["details"])
        self.assertEqual([c[0] for c in self.cached], ["RISING"])

    def test_the_run_is_recorded_with_the_model_that_made_it(self):
        es.forecast_symbols(["RISING", "SHORT"], "a")
        self.assertEqual(len(self.recorded), 1)
        run = self.recorded[0]
        self.assertEqual((run["model"], run["as_of_date"]), ("a", self.END_DATE))
        self.assertEqual((run["model_name"], run["model_versions"], run["model_trained_at"]),
                         ("model_a", [1, 2, 3], "2026-10-04T09:43:00"))
        # five horizon days for the symbol that forecast; nothing for the one that failed
        self.assertEqual([(r[0], r[1], r[2]) for r in run["rows"]],
                         [("RISING", day, date) for day, date in enumerate(
                             ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09"], start=1)])
        symbol, _, _, predicted, last_macd, flag = run["rows"][0]
        self.assertAlmostEqual(predicted, 0.0, places=5)
        self.assertEqual((last_macd, flag), (-1.0, True))

    def test_a_failure_to_record_does_not_fail_the_forecast(self):
        def boom(*args, **kwargs):
            raise RuntimeError("db down")
        sys.modules["db_utils"].save_forecast_predictions = boom
        results = es.forecast_symbols(["RISING"], "a")
        self.assertTrue(results["RISING"]["will_become_positive"])

    def test_forecast_days_is_capped_at_the_model_horizon(self):
        self.assertEqual(len(es.forecast_symbols(["RISING"], "a", forecast_days=3)["RISING"]["forecasted_dates"]), 3)
        self.assertEqual(len(es.forecast_symbols(["RISING"], "a", forecast_days=30)["RISING"]["forecasted_dates"]), 5)


class ForecastHelpersTest(unittest.TestCase):

    def setUp(self):
        fake_macd_utils = types.ModuleType("macd_utils")
        for name in ("get_latest_market_date", "get_macd_for_date", "get_macd_for_range"):
            setattr(fake_macd_utils, name, mock.Mock())
        sys.modules.pop("forecast_utils", None)
        patcher = mock.patch.dict(sys.modules, {"macd_utils": fake_macd_utils})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(lambda: sys.modules.pop("forecast_utils", None))
        import forecast_utils
        self.fu = forecast_utils

    def test_will_become_positive_rule(self):
        rule = self.fu.macd_will_become_positive
        self.assertTrue(rule(-1.0, [-0.5, 0.2]))        # negative now, turns positive
        self.assertTrue(rule(0.5, [-0.1, 0.3]))         # dips negative first, then positive
        self.assertFalse(rule(-1.0, [-0.9, -0.8]))      # stays negative
        self.assertFalse(rule(1.0, [1.2, 1.5]))         # already positive throughout
        self.assertFalse(rule(1.0, [-0.2, -0.4]))       # turns negative and stays there
        self.assertFalse(rule(-1.0, []))

    def test_prediction_rows_skip_failures_and_missing_values(self):
        rows = self.fu.forecast_prediction_rows({
            "OK": {"will_become_positive": True, "forecasted_macd": {"2026-10-05": 0.5, "2026-10-06": None, "2026-10-07": 1.5},
                   "details": {"last_macd": -0.2}},
            "FAILED": {"will_become_positive": False, "forecasted_macd": [], "details": {"error": "Not enough MACD data"}},
            "BROKEN": {"error": "worker crashed"},
        })
        self.assertEqual(rows, [("OK", 1, "2026-10-05", 0.5, -0.2, True),
                                ("OK", 3, "2026-10-07", 1.5, -0.2, True)])

    def test_next_weekday_dates_skip_weekends(self):
        self.assertEqual(self.fu.next_weekday_dates(date(2026, 10, 1), 3),   # a Thursday
                         ["2026-10-02", "2026-10-05", "2026-10-06"])
        self.assertEqual(self.fu.next_weekday_dates(date(2026, 10, 2), 0), [])


if __name__ == "__main__":
    unittest.main()
