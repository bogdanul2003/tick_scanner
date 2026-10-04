"""
Tests for the recursive (iterated) multi-step forecast path —
evaluate_forecast_model.py::_rollout_predict, exposed as --rollout.

The whole point of the rollout is that step k+1's input contains step k's *output*, so the
bugs that matter are all bookkeeping ones: feeding back the wrong column, letting the window
grow or drift, or silently recomputing the filler from already-synthesized rows. A fake model
with known arithmetic pins all of that without a DB, Core ML, or a trained checkpoint.

Run from src/:
    python -m unittest tests.test_rollout_predict
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

from evaluate_forecast_model import _rollout_predict


class RecordingModel:
    """Returns a fixed (horizon, target_size) block and records every window it was handed."""

    def __init__(self, forecast, horizon=5):
        self.forecast = np.asarray(forecast, dtype=np.float32)
        self.horizon = horizon
        self.seen = []

    def predict(self, seq, prev_value=None):
        self.seen.append(np.array(seq, dtype=np.float32, copy=True))
        return self.forecast


class IncrementModel:
    """macd_pred = last macd in the window + 1; delta_pred = a constant 7.0.

    Because it reads the window it is given, its outputs are a direct probe of whether the
    feedback loop is wired up: a broken rollout that keeps re-predicting from the original
    window would emit the same value every step.
    """

    def __init__(self, delta_value=7.0):
        self.delta_value = delta_value
        self.seen = []

    def predict(self, seq, prev_value=None):
        self.seen.append(np.array(seq, dtype=np.float32, copy=True))
        last = float(np.asarray(seq)[-1, 0])
        return np.array([[last + 1.0, self.delta_value]] * 5, dtype=np.float32)


def _window(rows=6, n_feat=3):
    """Distinct values per cell so a misindexed column shows up immediately."""
    return np.arange(rows * n_feat, dtype=np.float32).reshape(rows, n_feat)


class FeedbackLoopTest(unittest.TestCase):
    def test_each_step_sees_the_previous_steps_prediction(self):
        model = IncrementModel()
        w = _window()
        last_macd = float(w[-1, 0])

        macd, _ = _rollout_predict(model, w, None, 4, inc_delta=True)

        # +1 per step, compounding — proves step k+1 read step k's output, not the original window.
        self.assertEqual(macd, [last_macd + 1, last_macd + 2, last_macd + 3, last_macd + 4])

    def test_returns_exactly_steps_values(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        macd, delta = _rollout_predict(model, _window(), None, 3, inc_delta=True)
        self.assertEqual(len(macd), 3)
        self.assertEqual(len(delta), 3)
        self.assertEqual(len(model.seen), 3)

    def test_only_day_one_of_each_prediction_is_used(self):
        # Rows 2..5 are decoys: if the rollout ever reads past row 0 they leak into the output.
        model = RecordingModel([[1.0, 2.0], [99.0, 99.0], [99.0, 99.0], [99.0, 99.0], [99.0, 99.0]])
        macd, delta = _rollout_predict(model, _window(), None, 3, inc_delta=True)
        self.assertEqual(macd, [1.0, 1.0, 1.0])
        self.assertEqual(delta, [2.0, 2.0, 2.0])

    def test_window_length_is_preserved_across_steps(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window(rows=6)
        _rollout_predict(model, w, None, 4, inc_delta=True)
        for seen in model.seen:
            self.assertEqual(seen.shape, w.shape)

    def test_oldest_row_is_dropped_not_overwritten(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window(rows=6)
        _rollout_predict(model, w, None, 2, inc_delta=True)
        # After one step the window must be the original rows 1.. plus one synthesized row.
        np.testing.assert_allclose(model.seen[1][:-1], w[1:])

    def test_the_caller_s_window_is_not_mutated(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window()
        original = w.copy()
        _rollout_predict(model, w, None, 3, inc_delta=True)
        np.testing.assert_array_equal(w, original)


class DeltaModeTest(unittest.TestCase):
    def test_predicted_mode_uses_the_models_delta_head(self):
        model = IncrementModel(delta_value=7.0)
        _, delta = _rollout_predict(model, _window(), None, 3, inc_delta=True, delta_mode="predicted")
        self.assertEqual(delta, [7.0, 7.0, 7.0])

    def test_diff_mode_derives_delta_from_the_primary_series(self):
        # IncrementModel advances macd by exactly 1.0 per step, so the differenced delta is 1.0 —
        # and must ignore the head's 7.0 entirely.
        model = IncrementModel(delta_value=7.0)
        _, delta = _rollout_predict(model, _window(), None, 3, inc_delta=True, delta_mode="diff")
        self.assertEqual(delta, [1.0, 1.0, 1.0])

    def test_delta_is_written_into_column_one_of_the_fed_back_row(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        _rollout_predict(model, _window(), None, 2, inc_delta=True, delta_mode="predicted")
        synthesized = model.seen[1][-1]
        self.assertEqual(float(synthesized[0]), 1.0)   # macd
        self.assertEqual(float(synthesized[1]), 2.0)   # delta

    def test_without_include_delta_no_delta_series_is_returned(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        macd, delta = _rollout_predict(model, _window(), None, 3, inc_delta=False)
        self.assertEqual(len(macd), 3)
        self.assertIsNone(delta)

    def test_single_feature_model_rolls_forward_on_macd_alone(self):
        model = RecordingModel(np.array([[1.5]] * 5, dtype=np.float32))
        w = np.arange(6, dtype=np.float32).reshape(6, 1)
        macd, delta = _rollout_predict(model, w, None, 3, inc_delta=False)
        self.assertEqual(macd, [1.5, 1.5, 1.5])
        self.assertIsNone(delta)
        self.assertEqual(float(model.seen[1][-1, 0]), 1.5)


class FillPolicyTest(unittest.TestCase):
    """Columns past macd/delta cannot be predicted; --rollout-fill decides what goes there."""

    def test_hold_carries_the_last_observed_value(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window()
        expected = float(w[-1, 2])
        _rollout_predict(model, w, None, 3, inc_delta=True, fill="hold")
        for seen in model.seen[1:]:
            self.assertEqual(float(seen[-1, 2]), expected)

    def test_zero_fills_with_zero(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        _rollout_predict(model, _window(), None, 3, inc_delta=True, fill="zero")
        for seen in model.seen[1:]:
            self.assertEqual(float(seen[-1, 2]), 0.0)

    def test_mean_uses_the_original_windows_column_mean(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window()
        expected = float(w[:, 2].mean())
        _rollout_predict(model, w, None, 3, inc_delta=True, fill="mean")
        for seen in model.seen[1:]:
            np.testing.assert_allclose(float(seen[-1, 2]), expected, rtol=1e-6)

    def test_filler_does_not_drift_as_synthesized_rows_enter_the_window(self):
        """The filler is computed once from the real window. Recomputing it each step would let
        the synthesized rows feed back into it and drift the value away from the data."""
        model = RecordingModel([[1.0, 2.0]] * 5)
        w = _window(rows=6)
        _rollout_predict(model, w, None, 5, inc_delta=True, fill="mean")
        fills = [float(seen[-1, 2]) for seen in model.seen[1:]]
        self.assertEqual(len(set(fills)), 1, f"filler drifted across steps: {fills}")

    def test_fill_policy_never_touches_the_macd_or_delta_columns(self):
        model = RecordingModel([[1.0, 2.0]] * 5)
        for fill in ("mean", "hold", "zero"):
            with self.subTest(fill=fill):
                m = RecordingModel([[1.0, 2.0]] * 5)
                _rollout_predict(m, _window(), None, 2, inc_delta=True, fill=fill)
                self.assertEqual(float(m.seen[1][-1, 0]), 1.0)
                self.assertEqual(float(m.seen[1][-1, 1]), 2.0)


class EndToEndTrajectoryTest(unittest.TestCase):
    def test_full_five_step_rollout_matches_hand_computed_values(self):
        """One explicit trajectory, computed by hand, so a refactor that quietly changes the
        recursion (off-by-one in the window, wrong feedback column) fails loudly."""
        model = IncrementModel(delta_value=0.5)
        w = np.array([[10.0, 1.0, 0.2],
                      [11.0, 1.0, 0.4],
                      [12.0, 1.0, 0.6]], dtype=np.float32)

        macd, delta = _rollout_predict(model, w, None, 5, inc_delta=True,
                                       fill="hold", delta_mode="predicted")

        self.assertEqual(macd, [13.0, 14.0, 15.0, 16.0, 17.0])
        self.assertEqual(delta, [0.5] * 5)
        # The window handed to the final (5th) call: every original row has been pushed out,
        # so the model is predicting entirely from its own earlier output — the exposure-bias
        # regime this whole path lives in.
        final = model.seen[-1]
        np.testing.assert_allclose(final[:, 0], [14.0, 15.0, 16.0])
        np.testing.assert_allclose(final[:, 1], [0.5, 0.5, 0.5])
        np.testing.assert_allclose(final[:, 2], [0.6, 0.6, 0.6])   # held from the original last row


if __name__ == "__main__":
    unittest.main()
