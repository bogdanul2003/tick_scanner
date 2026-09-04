"""
Tests for A3 (docs/FORECAST_MODEL_IMPROVEMENTS.md) — independent per-target
horizon-decay weighting via MACDForecasterTrainer._build_loss_weights /
loss_decay_gamma_delta.

Pure math against a constructed trainer's self.loss_weights — no training run,
no DB, no Core ML required.
Run from src/:
    python -m unittest tests.test_loss_decay_weights
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_forecaster import MACDForecasterTrainer


def _make_trainer(**overrides):
    kwargs = dict(seq_length=5, forecast_horizon=5, hidden_size=4, num_layers=1, architecture="gru")
    kwargs.update(overrides)
    return MACDForecasterTrainer(**kwargs)


class BackwardCompatTest(unittest.TestCase):
    """A config that only ever set loss_decay_gamma must behave identically to
    before loss_decay_gamma_delta existed (the old np.repeat(day_weights, target_size) formula)."""

    def test_single_gamma_matches_legacy_repeat_formula(self):
        t = _make_trainer(include_delta=True, loss_decay_gamma=0.8)
        day_weights = np.array([0.8 ** k for k in range(5)], dtype=np.float32)
        legacy = np.repeat(day_weights, 2)
        legacy = legacy / legacy.mean()
        np.testing.assert_allclose(t.loss_weights.cpu().numpy(), legacy, atol=1e-6)

    def test_single_gamma_target_size_one_matches_legacy(self):
        t = _make_trainer(include_delta=False, loss_decay_gamma=0.5)
        day_weights = np.array([0.5 ** k for k in range(5)], dtype=np.float32)
        legacy = day_weights / day_weights.mean()
        np.testing.assert_allclose(t.loss_weights.cpu().numpy(), legacy, atol=1e-6)

    def test_gamma_none_disables_weighting(self):
        t = _make_trainer(include_delta=True, loss_decay_gamma=None)
        self.assertIsNone(t.loss_weights)

    def test_gamma_delta_alone_without_macd_gamma_stays_off(self):
        # Preserves the exact legacy gate: weighting is keyed off loss_decay_gamma, not
        # loss_decay_gamma_delta alone.
        t = _make_trainer(include_delta=True, loss_decay_gamma=None, loss_decay_gamma_delta=0.5)
        self.assertIsNone(t.loss_weights)


class IndependentDecayTest(unittest.TestCase):
    def test_macd_and_delta_decay_at_their_own_rates(self):
        t = _make_trainer(include_delta=True, loss_decay_gamma=0.8, loss_decay_gamma_delta=0.95)
        weights = t.loss_weights.cpu().numpy().reshape(t.forecast_horizon, t.target_size)
        macd_col, delta_col = weights[:, 0], weights[:, 1]

        # Day-5/day-1 ratio within each column must match gamma**4, independent of normalization.
        np.testing.assert_allclose(macd_col[-1] / macd_col[0], 0.8 ** 4, atol=1e-5)
        np.testing.assert_allclose(delta_col[-1] / delta_col[0], 0.95 ** 4, atol=1e-5)
        # Delta must decay slower than macd (it's the whole point of A3).
        self.assertGreater(delta_col[-1] / delta_col[0], macd_col[-1] / macd_col[0])

    def test_each_column_gets_an_equal_share_of_the_loss_budget(self):
        """The regression this file previously missed: normalizing the flattened matrix by one
        shared mean let the slower-decaying column steal gradient share from the faster one.
        With gamma_macd=0.5/gamma_delta=0.9 that pushed macd from 50% of the budget down to 32%,
        which starves the macd head everywhere rather than only at the later horizon days.

        Unlike the day5/day1 ratio checks above, this compares columns in *absolute* terms, so it
        is sensitive to exactly the global-scalar error the ratio tests are invariant to."""
        t = _make_trainer(include_delta=True, loss_decay_gamma=0.5, loss_decay_gamma_delta=0.9)
        weights = t.loss_weights.cpu().numpy().reshape(t.forecast_horizon, t.target_size)
        macd_col, delta_col = weights[:, 0], weights[:, 1]

        np.testing.assert_allclose(macd_col.sum(), delta_col.sum(), rtol=1e-5)
        np.testing.assert_allclose(macd_col.mean(), 1.0, rtol=1e-5)
        np.testing.assert_allclose(delta_col.mean(), 1.0, rtol=1e-5)

    def test_column_weights_are_independent_of_the_other_columns_gamma(self):
        """A column's weights must be a function of its own gamma alone. Under the shared-mean
        bug, changing only gamma_delta moved every macd weight too."""
        a = _make_trainer(include_delta=True, loss_decay_gamma=0.5, loss_decay_gamma_delta=0.9)
        b = _make_trainer(include_delta=True, loss_decay_gamma=0.5, loss_decay_gamma_delta=0.6)

        macd_a = a.loss_weights.cpu().numpy().reshape(a.forecast_horizon, a.target_size)[:, 0]
        macd_b = b.loss_weights.cpu().numpy().reshape(b.forecast_horizon, b.target_size)[:, 0]
        np.testing.assert_allclose(macd_a, macd_b, atol=1e-6)

    def test_two_gamma_macd_column_matches_the_single_gamma_column(self):
        """Setting gamma_delta must not change the macd column at all: the macd column under
        (0.5, 0.9) has to equal the macd column a plain gamma=0.5 config produces."""
        two = _make_trainer(include_delta=True, loss_decay_gamma=0.5, loss_decay_gamma_delta=0.9)
        one = _make_trainer(include_delta=True, loss_decay_gamma=0.5)

        macd_two = two.loss_weights.cpu().numpy().reshape(two.forecast_horizon, two.target_size)[:, 0]
        macd_one = one.loss_weights.cpu().numpy().reshape(one.forecast_horizon, one.target_size)[:, 0]
        np.testing.assert_allclose(macd_two, macd_one, atol=1e-6)

    def test_overall_loss_scale_is_preserved(self):
        """Mean weight of 1.0 keeps the weighted MSE on the same scale as an unweighted one, so
        turning decay on doesn't quietly rescale the effective learning rate."""
        for kwargs in (
            dict(include_delta=True, loss_decay_gamma=0.5, loss_decay_gamma_delta=0.9),
            dict(include_delta=True, loss_decay_gamma=0.8),
            dict(include_delta=False, loss_decay_gamma=0.5),
        ):
            with self.subTest(**kwargs):
                t = _make_trainer(**kwargs)
                np.testing.assert_allclose(t.loss_weights.cpu().numpy().mean(), 1.0, rtol=1e-5)

    def test_flattened_order_matches_row_major_horizon_target_layout(self):
        t = _make_trainer(include_delta=True, loss_decay_gamma=0.8, loss_decay_gamma_delta=0.95)
        flat = t.loss_weights.cpu().numpy()
        reshaped = flat.reshape(t.forecast_horizon, t.target_size)
        # Day 0's macd/delta weights come before day 1's, matching y_norm.flatten()'s layout.
        self.assertEqual(flat[0], reshaped[0, 0])
        self.assertEqual(flat[1], reshaped[0, 1])
        self.assertEqual(flat[2], reshaped[1, 0])


class TargetSizeOneIgnoresDeltaGammaTest(unittest.TestCase):
    def test_warns_and_ignores_without_include_delta(self):
        t = _make_trainer(include_delta=False, loss_decay_gamma=0.8, loss_decay_gamma_delta=0.5)
        day_weights = np.array([0.8 ** k for k in range(5)], dtype=np.float32)
        expected = day_weights / day_weights.mean()
        np.testing.assert_allclose(t.loss_weights.cpu().numpy(), expected, atol=1e-6)


class InvalidGammaTest(unittest.TestCase):
    def test_macd_gamma_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _make_trainer(include_delta=True, loss_decay_gamma=1.5)

    def test_delta_gamma_zero_raises(self):
        with self.assertRaises(ValueError):
            _make_trainer(include_delta=True, loss_decay_gamma=0.8, loss_decay_gamma_delta=0.0)

    def test_delta_gamma_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            _make_trainer(include_delta=True, loss_decay_gamma=0.8, loss_decay_gamma_delta=2.0)


class SaveLoadRoundTripTest(unittest.TestCase):
    def test_reloaded_trainer_produces_identical_loss_weights(self):
        t = _make_trainer(include_delta=True, loss_decay_gamma=0.7, loss_decay_gamma_delta=0.9)
        original_weights = t.loss_weights.clone()

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.pt")
            t.save(path)

            reloaded = _make_trainer(include_delta=True)
            reloaded.load(path)

        self.assertEqual(reloaded.loss_decay_gamma, 0.7)
        self.assertEqual(reloaded.loss_decay_gamma_delta, 0.9)
        np.testing.assert_allclose(reloaded.loss_weights.cpu().numpy(), original_weights.cpu().numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
