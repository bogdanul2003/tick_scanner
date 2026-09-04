"""
Tests for A2 (docs/FORECAST_MODEL_IMPROVEMENTS.md) — reconstructing the primary
signal from a delta-only network output via
MACDForecasterTrainer._reconstruct_deltas_only, plus the guardrails at
MACDForecasterTrainer.__init__ that restrict predict_deltas_only to configurations
the reconstruction math has actually been worked out for.

No DB, Core ML, or training run required — the reconstruction is exercised
directly with hand-picked mean/std/anchor values, not through prepare_sequences
or train().
Run from src/:
    python -m unittest tests.test_delta_reconstruction
"""
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_forecaster import MACDForecasterTrainer


def _make_trainer(**overrides):
    kwargs = dict(
        seq_length=5, forecast_horizon=4, hidden_size=4, num_layers=1,
        architecture="gru", include_delta=True, predict_deltas_only=True,
    )
    kwargs.update(overrides)
    return MACDForecasterTrainer(**kwargs)


class DeltaReconstructionMathTest(unittest.TestCase):
    def test_reconstructs_macd_from_normalized_deltas(self):
        t = _make_trainer()
        # mean/std: [macd_mean, delta_mean], [macd_std, delta_std]
        t.mean = np.array([10.0, 0.0], dtype=np.float32)
        t.std = np.array([2.0, 1.0], dtype=np.float32)

        # Two windows, forecast_horizon=4. Normalized delta predictions are all
        # zero, so raw deltas equal delta_mean (0.0) -> macd_pred stays at anchor.
        delta_pred_norm = torch.zeros(2, 4)
        anchor = torch.tensor([100.0, -5.0])

        combined = t._reconstruct_deltas_only(delta_pred_norm, anchor)
        self.assertEqual(combined.shape, (2, 4 * 2))

        combined_np = combined.detach().numpy().reshape(2, 4, 2)
        macd_pred = combined_np[:, :, 0]
        delta_pred = combined_np[:, :, 1]

        # Raw macd stays at the anchor for every day (zero deltas), normalized by macd's own mean/std.
        expected_macd_raw = np.array([[100.0] * 4, [-5.0] * 4])
        expected_macd_norm = (expected_macd_raw - 10.0) / 2.0
        np.testing.assert_allclose(macd_pred, expected_macd_norm, atol=1e-5)
        # Delta predictions pass through unchanged (normalized delta was 0 -> normalized delta stays 0).
        np.testing.assert_allclose(delta_pred, np.zeros((2, 4)), atol=1e-5)

    def test_nonzero_deltas_compound_via_cumsum(self):
        t = _make_trainer()
        t.mean = np.array([0.0, 0.0], dtype=np.float32)
        t.std = np.array([1.0, 1.0], dtype=np.float32)  # identity normalization

        # Raw deltas == normalized deltas here (mean 0, std 1): [1, 1, 1, 1] per day.
        delta_pred_norm = torch.ones(1, 4)
        anchor = torch.tensor([50.0])

        combined = t._reconstruct_deltas_only(delta_pred_norm, anchor)
        combined_np = combined.detach().numpy().reshape(1, 4, 2)
        macd_pred = combined_np[0, :, 0]

        # anchor + cumsum([1,1,1,1]) = [51, 52, 53, 54]
        np.testing.assert_allclose(macd_pred, [51.0, 52.0, 53.0, 54.0], atol=1e-5)

    def test_reconstruction_is_differentiable_back_to_delta_predictions(self):
        t = _make_trainer()
        t.mean = np.array([0.0, 0.0], dtype=np.float32)
        t.std = np.array([1.0, 1.0], dtype=np.float32)

        delta_pred_norm = torch.ones(1, 4, requires_grad=True)
        anchor = torch.tensor([0.0])
        combined = t._reconstruct_deltas_only(delta_pred_norm, anchor)
        combined.sum().backward()
        self.assertIsNotNone(delta_pred_norm.grad)
        self.assertTrue(torch.all(delta_pred_norm.grad != 0))


class PredictDeltasOnlyValidationTest(unittest.TestCase):
    def test_requires_include_delta(self):
        with self.assertRaises(ValueError):
            _make_trainer(include_delta=False)

    def test_incompatible_with_residual_target(self):
        with self.assertRaises(ValueError):
            _make_trainer(residual_target=True)

    def test_requires_global_normalization(self):
        with self.assertRaises(ValueError):
            _make_trainer(normalization_type="internal")

    def test_valid_configuration_constructs_with_network_target_size_one(self):
        t = _make_trainer()
        self.assertEqual(t.network_target_size, 1)
        self.assertEqual(t.target_size, 2)  # loss/metrics still see the macd+delta pair

    def test_off_by_default_does_not_raise_on_incompatible_flags(self):
        # predict_deltas_only=False (default) must never trigger A2's guardrails,
        # even with residual_target/internal normalization.
        MACDForecasterTrainer(seq_length=5, forecast_horizon=4, hidden_size=4,
                               architecture="gru", include_delta=False,
                               residual_target=True, normalization_type="internal")


if __name__ == "__main__":
    unittest.main()
