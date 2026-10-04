"""
Tests for the ARIMA comparison path in scripts/evaluate_forecast_model.py.

The bug pinned here: a failed ARIMA fit is recorded as [None] * horizon, and a plain
truthiness check treats that as a real forecast. The Nones then reach _compute_metrics,
which raises TypeError — and run_watchlist_evaluation did not catch exceptions, so one
illiquid symbol could discard a ~30-minute watchlist pass.

Pure logic — no DB, no statsmodels, no Core ML.
Run from src/:
    python -m unittest tests.test_arima_comparison
"""
import os
import sys
import unittest

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, "scripts"))

from evaluate_forecast_model import _usable_arima, _compute_metrics


class UsableArimaTest(unittest.TestCase):
    def test_complete_forecast_is_returned_trimmed_to_horizon(self):
        self.assertEqual(_usable_arima([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 5),
                         [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_all_none_forecast_is_rejected(self):
        """The exact shape the worker loop records on a failed fit."""
        self.assertIsNone(_usable_arima([None] * 5, 5))

    def test_partially_none_forecast_is_rejected(self):
        self.assertIsNone(_usable_arima([1.0, 2.0, None, 4.0, 5.0], 5))

    def test_none_after_the_horizon_does_not_reject(self):
        """Only the values actually scored matter."""
        self.assertEqual(_usable_arima([1.0, 2.0, None], 2), [1.0, 2.0])

    def test_empty_and_none_inputs_are_rejected(self):
        self.assertIsNone(_usable_arima([], 5))
        self.assertIsNone(_usable_arima(None, 5))

    def test_shorter_than_horizon_is_kept_if_complete(self):
        self.assertEqual(_usable_arima([1.0, 2.0], 5), [1.0, 2.0])

    def test_zero_values_are_not_mistaken_for_missing(self):
        """0.0 is falsy — a naive check would drop a legitimate flat forecast."""
        self.assertEqual(_usable_arima([0.0, 0.0], 2), [0.0, 0.0])


class RegressionTest(unittest.TestCase):
    def test_the_crash_this_guard_prevents(self):
        """Without the guard, [None]*h reached _compute_metrics and raised."""
        with self.assertRaises(TypeError):
            _compute_metrics([[None] * 3], [[1.0, 2.0, 3.0]], [0.0])
        # With the guard the row is dropped before it ever gets there.
        self.assertIsNone(_usable_arima([None] * 3, 3))


if __name__ == "__main__":
    unittest.main()
