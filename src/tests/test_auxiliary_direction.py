"""
Tests for A1 (docs/FORECAST_MODEL_IMPROVEMENTS.md) — the auxiliary directional
(BCE) loss head added to the forecaster architectures.

Pure forward-pass shape checks (tiny random tensors, no .train() calls) plus a
pure-numpy check of the direction-label derivation used in
MACDForecasterTrainer.prepare_sequences. No DB, Core ML, or training run
required.
Run from src/:
    python -m unittest tests.test_auxiliary_direction
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_forecaster import create_model, ARCHITECTURES


class AuxiliaryDirectionHeadShapeTest(unittest.TestCase):
    """create_model(..., auxiliary_direction=True) must add a second output of
    shape (batch, forecast_horizon) without disturbing the magnitude head."""

    def _check_architecture(self, architecture):
        batch, seq_len, input_size, horizon, target_size = 3, 20, 2, 7, 2
        x = torch.randn(batch, seq_len, input_size)

        plain = create_model(architecture, input_size=input_size, target_size=target_size,
                              hidden_size=8, num_layers=1, forecast_horizon=horizon,
                              auxiliary_direction=False)
        out = plain(x)
        self.assertNotIsInstance(out, tuple, f"{architecture}: plain model must return a single tensor")
        self.assertEqual(out.shape, (batch, horizon * target_size))

        aux = create_model(architecture, input_size=input_size, target_size=target_size,
                            hidden_size=8, num_layers=1, forecast_horizon=horizon,
                            auxiliary_direction=True)
        out2 = aux(x)
        self.assertIsInstance(out2, tuple, f"{architecture}: auxiliary model must return a tuple")
        forecast, direction_logits = out2
        self.assertEqual(forecast.shape, (batch, horizon * target_size))
        self.assertEqual(direction_logits.shape, (batch, horizon))

    def test_bidirectional_gru(self):
        self._check_architecture("bidirectional_gru")

    def test_stacked_lstm(self):
        self._check_architecture("stacked_lstm")

    def test_stacked_gru(self):
        self._check_architecture("stacked_gru")

    def test_standard_lstm(self):
        self._check_architecture("standard_lstm")

    def test_standard_gru(self):
        self._check_architecture("gru")

    def test_covers_every_supported_architecture(self):
        # Guards against a new architecture being added to ARCHITECTURES without
        # a corresponding auxiliary_direction wiring being exercised here.
        covered = {"stacked_lstm", "bidirectional_gru", "stacked_gru", "standard_lstm", "gru"}
        self.assertEqual(set(ARCHITECTURES), covered)


class DirectionLabelDerivationTest(unittest.TestCase):
    """Mirrors the label logic in MACDForecasterTrainer.prepare_sequences:
    day k's label is sign(y_raw[k] - previous_value), independent of include_delta."""

    def _labels(self, last_input_value, future_values):
        y_raw_col0 = np.array(future_values, dtype=np.float32)
        prev = np.concatenate([[last_input_value], y_raw_col0[:-1]])
        return (y_raw_col0 - prev > 0).astype(np.float32)

    def test_all_increasing(self):
        labels = self._labels(last_input_value=1.0, future_values=[1.5, 2.0, 3.0])
        np.testing.assert_array_equal(labels, [1.0, 1.0, 1.0])

    def test_all_decreasing(self):
        labels = self._labels(last_input_value=5.0, future_values=[4.0, 3.0, 1.0])
        np.testing.assert_array_equal(labels, [0.0, 0.0, 0.0])

    def test_mixed_direction(self):
        labels = self._labels(last_input_value=2.0, future_values=[3.0, 1.0, 1.0])
        # day1: 3.0 > 2.0 -> up; day2: 1.0 > 3.0 -> down; day3: 1.0 > 1.0 -> not up (flat = down)
        np.testing.assert_array_equal(labels, [1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
