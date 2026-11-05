"""
Tests for the freshness backtest's planning and statistics
(services/freshness_backtest.py). Both are pure: no training, no DB, no Core ML.

What matters: the evaluation origins come after every copy's training cutoff, and the
verdict says "retrain" only when older copies are clearly and consistently worse,
"not needed" only when they clearly are not.
Run from src/:
    python -m unittest tests.test_freshness_backtest
"""
import os
import sys
import unittest
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import freshness_backtest as fb

AGES = (0, 4, 8, 13)


def _weekdays(start: date, count: int):
    days, d = [], start
    while len(days) < count:
        if d.weekday() < 5:
            days.append(d)
        d += timedelta(days=1)
    return days


def _series(err_pct_by_origin, horizon=5, n=100, dir_share=0.5):
    """A series whose error (% of price) on origin o is err_pct_by_origin[o] on every day."""
    return {
        "n": [[n] * horizon for _ in err_pct_by_origin],
        "err": [[e * n for _ in range(horizon)] for e in err_pct_by_origin],
        "err_pct": [[e * n for _ in range(horizon)] for e in err_pct_by_origin],
        "dir": [[dir_share * n] * horizon for _ in err_pct_by_origin],
    }


def _raw(level_by_age, day_noise=0.0, seed_spread=0.0, origins=20, served=None, drift=None):
    """Scoring output where the copy of age a has error level_by_age[a] on every day,
    plus a shared day-to-day swing, and seeds spread by +-seed_spread."""
    swing = [1 + day_noise * ((i * 7) % 5 - 2) for i in range(origins)]
    series = {}
    for age, level in level_by_age.items():
        series[f"age{age}"] = _series([level * s for s in swing])
        for k, offset in enumerate((-seed_spread, 0.0, seed_spread)):
            series[f"age{age}#{k}"] = _series([(level + offset) * s for s in swing])
    series["drift"] = _series([(drift or 1.0) * s for s in swing])
    series["flat"] = _series([1.5 * s for s in swing])
    if served is not None:
        series["served"] = _series([served * s for s in swing])
    return {"origins": [f"2026-09-{i + 1:02d}" for i in range(origins)], "horizon": 5,
            "symbols_scored": 100, "series": series}


class PlanWindowTest(unittest.TestCase):
    def test_origins_have_closed_days_and_follow_every_cutoff(self):
        calendar = _weekdays(date(2026, 1, 5), 200)
        window = fb.plan_window(calendar, origins=20, horizon=5, ages_weeks=AGES)

        self.assertEqual(len(window["origins"]), 20)
        # The last origin's five forecast days are the last five calendar days.
        self.assertEqual(window["last_origin"], calendar[-6])
        first = window["first_origin"]
        self.assertEqual(window["cutoffs"][0], calendar[calendar.index(first) - 1])
        for age in AGES:
            self.assertLess(window["cutoffs"][age], first)
            self.assertIn(window["cutoffs"][age], calendar)
        self.assertEqual((window["cutoffs"][0] - window["cutoffs"][13]).days, 91)

    def test_too_short_a_calendar_is_refused(self):
        with self.assertRaises(ValueError):
            fb.plan_window(_weekdays(date(2026, 1, 5), 20), origins=20, horizon=5, ages_weeks=AGES)


class DataPeriodsTest(unittest.TestCase):
    def test_time_split_reports_weights_checkpoint_and_held_out_parts(self):
        calendar = _weekdays(date(2025, 1, 1), 500)
        cutoff = calendar[-1]
        periods = fb.data_periods(calendar, cutoff, {"days": 600, "test_split": 0.15,
                                                     "split_strategy": "time", "forecast_horizon": 5})
        self.assertLess(periods["weights_to"], periods["checkpoint_from"])
        self.assertLess(periods["checkpoint_to"], periods["held_out_from"])
        self.assertEqual(periods["held_out_to"], cutoff.isoformat())


class VerdictTest(unittest.TestCase):
    def test_clear_decay_with_age_says_retrain(self):
        raw = _raw({0: 1.00, 4: 1.08, 8: 1.16, 13: 1.26}, day_noise=0.05, seed_spread=0.005,
                   served=1.04)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=6)

        s = result["staleness"]
        self.assertAlmostEqual(s["per_week_pct"], 2.0, delta=0.05)
        self.assertAlmostEqual(s["cost_now_pct"], 12.0, delta=0.3)
        self.assertGreater(s["cost_now_ci"][0], 0)
        self.assertEqual(s["fresh_beats_oldest_days"], 20)
        self.assertEqual(result["verdict"]["code"], "retrain")

    def test_no_change_with_age_says_not_needed(self):
        raw = _raw({0: 1.00, 4: 1.00, 8: 1.00, 13: 1.00}, day_noise=0.05, seed_spread=0.005, served=0.9)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=2)

        self.assertAlmostEqual(result["staleness"]["per_week_pct"], 0.0, places=6)
        self.assertEqual(result["verdict"]["code"], "not_needed")

    def test_decay_smaller_than_seed_luck_is_not_a_reason_to_retrain(self):
        # 0.3% a week is real here, but seeds alone move an ensemble by several %.
        raw = _raw({0: 1.00, 4: 1.012, 8: 1.024, 13: 1.039}, seed_spread=0.05, served=0.9)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=2)

        self.assertGreater(result["staleness"]["seed_luck_pct"], 3)
        self.assertNotEqual(result["verdict"]["code"], "retrain")

    def test_a_copy_better_by_seed_luck_is_not_significant(self):
        # The 4-week copy is 4% better on every day, but seeds spread by ~10%:
        # resampling days alone would call that significant.
        raw = _raw({0: 1.00, 4: 0.96, 8: 1.00, 13: 1.00}, day_noise=0.05, seed_spread=0.10, served=0.9)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=2)
        ci = next(a for a in result["arms"] if a["age_weeks"] == 4)["vs_fresh_ci"]
        self.assertLess(ci[0], 0)
        self.assertGreater(ci[1], 0)

    def test_unknown_served_age_is_inconclusive(self):
        raw = _raw({0: 1.00, 4: 1.10, 8: 1.20, 13: 1.30})
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=None)
        self.assertEqual(result["verdict"]["code"], "inconclusive")

    def test_reports_when_the_served_model_does_not_beat_drift(self):
        raw = _raw({0: 1.00, 4: 1.00, 8: 1.00, 13: 1.00}, seed_spread=0.005, served=1.05, drift=1.0)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=2)

        self.assertAlmostEqual(result["served_scores"]["vs_drift_err_pct"], 1.05)
        self.assertTrue(any("does not beat drift" in r for r in result["verdict"]["reasons"]))

    def test_per_day_and_relative_numbers(self):
        raw = _raw({0: 1.0, 4: 1.1, 8: 1.2, 13: 1.3}, served=1.0)
        result = fb.summarize_backtest(raw, AGES, served_age_weeks=2,
                                       cutoffs={0: "2026-08-29", 4: "2026-08-01", 8: "2026-07-03", 13: "2026-05-29"})
        arms = {a["age_weeks"]: a for a in result["arms"]}
        self.assertAlmostEqual(arms[4]["vs_fresh_pct"], 10.0)
        self.assertEqual(arms[13]["cutoff"], "2026-05-29")
        self.assertEqual(len(arms[0]["per_day"]), 5)
        self.assertAlmostEqual(arms[0]["per_day"][2]["err_pct"], 1.0)
        self.assertAlmostEqual(arms[0]["direction_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
