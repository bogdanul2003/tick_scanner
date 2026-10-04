"""
Tests for the multi-feature input matrix NeuralForecastService.forecast_macd
hands to the model.

Regression covered: the macd/signal_line branch read the column named by the
service's primary signal instead of the column named by the feature, so a MACD
model trained with signal_line as an extra feature was fed MACD twice.

The DB and Core ML are both stubbed — no Postgres, no .mlpackage, no NPU.
Run from src/:
    python -m unittest tests.test_neural_forecast_features
"""
import os
import sys
import types
import unittest
from datetime import datetime
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import neural_forecast as nf


class _RecordingForecaster:
    """Captures the feature matrix instead of running inference on it."""

    def __init__(self, feature_names, include_delta=True, seq_length=5, horizon=3):
        self.feature_names = feature_names
        self.include_delta = include_delta
        self.seq_length = seq_length
        self.forecast_horizon = horizon
        self.input_size = len(feature_names)
        self.target_size = 2 if include_delta else 1
        self.is_available = True
        self.seen = None

    def predict(self, sequence, prev_value=None):
        self.seen = np.asarray(sequence)
        return np.zeros((self.forecast_horizon, self.target_size), dtype=np.float32)


def _rows(count=40):
    """Cache rows shaped like get_macd_for_range's output, all columns distinct."""
    return [
        {
            "symbol": "TEST",
            "date": f"2026-01-{i+1:02d}",
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 1_000_000 + i,
            "macd": 1.0 + i,          # column values are deliberately far apart
            "signal_line": -50.0 - i,  # so a mix-up cannot look like a match
            "ma20": 200.0 + i,
            "ma50": 300.0 + i,
        }
        for i in range(count)
    ]


class MultiFeatureInputTest(unittest.TestCase):

    def setUp(self):
        self.rows = _rows()
        fake_macd_utils = types.ModuleType("macd_utils")
        fake_macd_utils.get_macd_for_range = lambda symbol, start, end: self.rows
        fake_macd_utils.get_latest_market_date = lambda: datetime(2026, 2, 10)
        patcher = mock.patch.dict(sys.modules, {"macd_utils": fake_macd_utils})
        patcher.start()
        self.addCleanup(patcher.stop)

    def _service(self, feature_names, signal_type="macd", include_delta=True):
        """A service wired to a recording forecaster, bypassing Core ML loading."""
        service = object.__new__(nf.NeuralForecastService)
        service.fallback_to_arima = False
        service.signal_type = signal_type
        service.forecaster = _RecordingForecaster(feature_names, include_delta)
        return service

    def test_signal_line_feature_reads_the_signal_line_column(self):
        """signal_line as an extra feature must not be a second copy of MACD."""
        service = self._service(["macd", "delta", "signal_line"])
        service.forecast_macd("TEST")

        matrix = service.forecaster.seen
        self.assertEqual(matrix.shape[1], 3)

        expected_macd = np.array([r["macd"] for r in self.rows], dtype=np.float32)
        expected_signal = np.array([r["signal_line"] for r in self.rows], dtype=np.float32)

        np.testing.assert_allclose(matrix[:, 0], expected_macd)
        np.testing.assert_allclose(matrix[:, 2], expected_signal)
        # The bug made these two identical.
        self.assertFalse(np.allclose(matrix[:, 0], matrix[:, 2]),
                         "signal_line column duplicates the MACD column")

    def test_macd_feature_on_a_signal_line_model(self):
        """The mirror case: a signal_line model asking for macd as an extra."""
        service = self._service(["signal_line", "delta", "macd"],
                                signal_type="signal_line")
        service.forecast_macd("TEST")

        matrix = service.forecaster.seen
        expected_signal = np.array([r["signal_line"] for r in self.rows], dtype=np.float32)
        expected_macd = np.array([r["macd"] for r in self.rows], dtype=np.float32)

        np.testing.assert_allclose(matrix[:, 0], expected_signal)
        np.testing.assert_allclose(matrix[:, 2], expected_macd)

    def test_primary_column_unchanged_when_it_is_the_only_signal_feature(self):
        """The common case still reads the primary signal, and delta follows it."""
        service = self._service(["macd", "delta", "ma20"])
        service.forecast_macd("TEST")

        matrix = service.forecaster.seen
        expected_macd = np.array([r["macd"] for r in self.rows], dtype=np.float32)
        expected_ma20 = np.array([r["ma20"] for r in self.rows], dtype=np.float32)

        np.testing.assert_allclose(matrix[:, 0], expected_macd)
        np.testing.assert_allclose(matrix[1:, 1], np.diff(expected_macd))
        np.testing.assert_allclose(matrix[:, 2], expected_ma20)

    def test_null_non_primary_signal_becomes_zero_without_dropping_rows(self):
        """
        A NULL in the non-primary signal column must not shorten that column —
        np.column_stack would raise if the columns disagreed in length.
        """
        self.rows[3]["signal_line"] = None
        service = self._service(["macd", "delta", "signal_line"])
        service.forecast_macd("TEST")

        matrix = service.forecaster.seen
        self.assertEqual(len(matrix), len(self.rows))
        self.assertEqual(matrix[3, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
