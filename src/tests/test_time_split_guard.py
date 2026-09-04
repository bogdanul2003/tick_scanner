"""
Tests for TS-009 (docs/plans/FORECAST_FIXES_PLAN.md) — the `--split-strategy time`
guard in scripts/train_forecast_model.py.

The bug this pins: a symbol whose split leaves either side shorter than
seq_length + forecast_horizon goes entirely into training. That is the correct
handling, but it used to be silent, so a run could train and save a model with an
empty test set and print a normal-looking summary.

Pure list/array logic — no DB, no training run.
Run from src/:
    python -m unittest tests.test_time_split_guard
"""
import os
import sys
import unittest

import numpy as np

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_SRC_DIR, "scripts"))

from train_forecast_model import split_symbols_by_time, check_test_set_usable


def _sym(n):
    return np.zeros((n, 2), dtype=np.float32)


class SplitSymbolsByTimeTest(unittest.TestCase):
    def test_long_symbol_is_split_into_both_sides(self):
        train, test, skipped = split_symbols_by_time([_sym(300)], 0.15, 30, 5)
        self.assertEqual([len(t) for t in train], [255])
        self.assertEqual([len(t) for t in test], [45])
        self.assertEqual(skipped, [])

    def test_short_test_side_sends_the_whole_symbol_to_training(self):
        """100 points at test_split=0.15 leaves 15 test points, under the 35 needed."""
        train, test, skipped = split_symbols_by_time([_sym(100)], 0.15, 30, 5)
        self.assertEqual([len(t) for t in train], [100])
        self.assertEqual(test, [])
        self.assertEqual(skipped, [(0, 100, 85)])

    def test_short_train_side_also_skips(self):
        train, test, skipped = split_symbols_by_time([_sym(40)], 0.15, 30, 5)
        self.assertEqual([len(t) for t in train], [40])
        self.assertEqual(test, [])
        self.assertEqual(len(skipped), 1)

    def test_mixed_batch_reports_only_the_skipped_ones(self):
        train, test, skipped = split_symbols_by_time([_sym(300), _sym(40), _sym(400)],
                                                     0.15, 30, 5)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(test), 2)
        self.assertEqual([i for i, _, _ in skipped], [1])

    def test_train_side_always_receives_every_symbol(self):
        """Skipping must never drop data from training, only from the test set."""
        data = [_sym(300), _sym(40), _sym(90)]
        train, _, _ = split_symbols_by_time(data, 0.15, 30, 5)
        self.assertEqual(len(train), len(data))

    def test_the_seq60_days600_case_that_motivated_the_ticket(self):
        """Sizing the seq_length sweep: at days=600, seq_length=60 empties the test
        set outright — the failure that used to be reported as a normal split."""
        symbols = [_sym(410) for _ in range(20)]     # ~600 calendar days of trading data
        _, test, skipped = split_symbols_by_time(symbols, 0.15, 60, 5)
        self.assertEqual(test, [])
        self.assertEqual(len(skipped), 20)


class CheckTestSetUsableTest(unittest.TestCase):
    def test_usable_test_set_returns_no_problems(self):
        self.assertEqual(
            check_test_set_usable([_sym(45)], 30, 5, 0, 1, "time", 0.15, False), [])

    def test_empty_test_set_is_an_error(self):
        problems = check_test_set_usable([], 30, 5, 3, 3, "time", 0.15, False)
        self.assertTrue(problems)
        self.assertTrue(problems[0].startswith("Error:"))

    def test_test_symbols_that_are_all_too_short_is_an_error(self):
        """Non-empty but unusable — evaluate() needs MORE than seq+horizon points."""
        problems = check_test_set_usable([_sym(35)], 30, 5, 0, 1, "time", 0.15, False)
        self.assertTrue(problems)
        self.assertTrue(problems[0].startswith("Error:"))

    def test_message_names_the_knobs_that_fix_it(self):
        text = " ".join(check_test_set_usable([], 30, 5, 3, 3, "time", 0.15, False))
        for knob in ("--days", "--seq-length", "--test-split", "--allow-empty-test-set"):
            self.assertIn(knob, text)

    def test_skipped_count_is_surfaced(self):
        text = " ".join(check_test_set_usable([], 30, 5, 7, 20, "time", 0.15, False))
        self.assertIn("7 of 20 symbols", text)

    def test_override_downgrades_the_error_to_a_warning(self):
        problems = check_test_set_usable([], 30, 5, 3, 3, "time", 0.15, True)
        self.assertTrue(problems)
        self.assertTrue(problems[0].startswith("Warning:"))
        self.assertFalse(any(p.startswith("Error:") for p in problems))

    def test_symbol_strategy_with_no_test_data_is_also_caught(self):
        """test_split=0 under the 'symbol' strategy sets test_data to None."""
        problems = check_test_set_usable(None, 30, 5, 0, 10, "symbol", 0.0, False)
        self.assertTrue(problems)
        self.assertTrue(problems[0].startswith("Error:"))


if __name__ == "__main__":
    unittest.main()
