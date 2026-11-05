"""
Tests for the forecast track record: how a model's stored forecasts compared with
the actual MACD on each of the last trading days.

summarize_track_record is pure; get_track_record is run with the DB and the market
calendar stubbed in sys.modules — no Postgres, no Core ML required.
Run from src/:
    python -m unittest tests.test_forecast_track_record
"""
import os
import sys
import types
import unittest
from datetime import date
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import forecast_track_record as tr

MON, TUE, WED = date(2026, 10, 5), date(2026, 10, 6), date(2026, 10, 7)


def _row(symbol, predicted, actual, origin=0.0, origin_prev=0.0, prev_actual=None,
         target=MON, lead=1):
    return {
        "symbol": symbol, "target_date": target, "lead_days": lead,
        "predicted": predicted, "actual": actual,
        "prev_actual": origin if prev_actual is None else prev_actual,
        "origin": origin, "origin_prev": origin_prev,
    }


class DayScoreTest(unittest.TestCase):
    def test_direction_error_and_drift(self):
        rows = [
            # Up from 1.0 to 2.0; forecast 1.5 (right, off by 0.5); drift 1.0+0.4=1.4 (right, off by 0.6).
            _row("AAA", predicted=1.5, actual=2.0, origin=1.0, origin_prev=0.6),
            # Down from 1.0 to 0.0; forecast 1.2 (wrong, off by 1.2); drift 1.0-0.5=0.5 (right, off by 0.5).
            _row("BBB", predicted=1.2, actual=0.0, origin=1.0, origin_prev=1.5),
        ]
        (day,) = tr.summarize_track_record([MON], rows)

        self.assertTrue(day["available"])
        self.assertEqual(day["date"], "2026-10-05")
        self.assertEqual(day["symbols"], 2)
        self.assertEqual(day["lead_days"], [1])
        self.assertAlmostEqual(day["direction_accuracy"], 0.5)
        self.assertAlmostEqual(day["drift_direction_accuracy"], 1.0)
        self.assertAlmostEqual(day["mae"], 0.85)
        self.assertAlmostEqual(day["median_error"], 0.85)
        self.assertAlmostEqual(day["drift_mae"], 0.55)
        self.assertEqual(day["beats_drift"], 1)

    def test_direction_is_judged_against_the_previous_days_actual(self):
        # Three days ahead: MACD went 0 -> ... -> 5 (day before) -> 4. The forecast, 4.5,
        # is above the starting value but below the previous day: it called a fall, correctly.
        rows = [_row("AAA", predicted=4.5, actual=4.0, origin=0.0, origin_prev=0.0,
                     prev_actual=5.0, lead=3)]
        (day,) = tr.summarize_track_record([MON], rows)
        self.assertAlmostEqual(day["direction_accuracy"], 1.0)

    def test_drift_extrapolates_one_step_per_lead_day(self):
        # Origin 1.0 after a +0.5 day; two days ahead drift says 2.0.
        rows = [_row("AAA", predicted=0.0, actual=2.0, origin=1.0, origin_prev=0.5,
                     prev_actual=1.5, lead=2)]
        (day,) = tr.summarize_track_record([MON], rows)
        self.assertAlmostEqual(day["drift_mae"], 0.0)

    def test_a_forecast_without_the_day_before_its_start_has_no_drift(self):
        rows = [_row("AAA", predicted=1.5, actual=2.0, origin=1.0, origin_prev=None)]
        (day,) = tr.summarize_track_record([MON], rows)
        self.assertAlmostEqual(day["direction_accuracy"], 1.0)
        self.assertIsNone(day["drift_mae"])
        self.assertIsNone(day["drift_direction_accuracy"])
        self.assertEqual(day["beats_drift"], 0)
        self.assertIsNone(day["top"]["symbols"][0]["beats_drift"])


class DaySelectionTest(unittest.TestCase):
    def test_a_day_without_forecasts_is_unavailable(self):
        days = tr.summarize_track_record([MON, TUE], [_row("AAA", 1.5, 2.0, origin=1.0, target=TUE)])

        self.assertEqual([d["date"] for d in days], ["2026-10-05", "2026-10-06"])
        self.assertFalse(days[0]["available"])
        self.assertEqual(days[0]["symbols"], 0)
        for key in ("direction_accuracy", "drift_direction_accuracy", "mae", "median_error",
                    "drift_mae", "beats_drift", "top"):
            self.assertIsNone(days[0][key], key)
        self.assertTrue(days[1]["available"])

    def test_no_rows_at_all(self):
        days = tr.summarize_track_record([MON, TUE, WED], [])
        self.assertEqual([d["available"] for d in days], [False, False, False])
        self.assertEqual(tr.summarize_track_record([], []), [])

    def test_the_most_recent_forecast_for_a_day_wins(self):
        rows = [
            _row("AAA", predicted=9.0, actual=2.0, origin=1.0, lead=3),
            _row("AAA", predicted=2.0, actual=2.0, origin=1.0, lead=1),
            _row("AAA", predicted=5.0, actual=2.0, origin=1.0, lead=2),
            _row("BBB", predicted=3.0, actual=3.0, origin=1.0, lead=4),
        ]
        (day,) = tr.summarize_track_record([MON], rows)
        self.assertEqual(day["symbols"], 2)
        self.assertEqual(day["lead_days"], [1, 4])
        self.assertAlmostEqual(day["mae"], 0.0)

    def test_rows_with_missing_or_non_finite_values_are_ignored(self):
        rows = [
            _row("AAA", predicted=float("nan"), actual=2.0, origin=1.0),
            _row("BBB", predicted=1.0, actual=None, origin=1.0),
            _row("CCC", predicted=1.5, actual=2.0, origin=1.0),
        ]
        (day,) = tr.summarize_track_record([MON], rows)
        self.assertEqual(day["symbols"], 1)


class TopMatchTest(unittest.TestCase):
    def _day(self, errors_pct, top_n):
        # Every symbol moves from 0 to 10, so an error of e is e*10 % of the move.
        rows = [_row(f"S{i:03d}", predicted=10.0 - pct / 10.0, actual=10.0, origin=0.0)
                for i, pct in enumerate(errors_pct)]
        (day,) = tr.summarize_track_record([MON], rows, top_n=top_n)
        return day

    def test_ranked_best_first_and_cut_at_top_n(self):
        day = self._day([50, 5, 20, 80, 10, 30], top_n=3)
        top = day["top"]
        self.assertEqual([s["symbol"] for s in top["symbols"]], ["S001", "S004", "S002"])
        self.assertEqual((top["requested"], top["size"], top["ranked"]), (3, 3, 6))
        self.assertAlmostEqual(top["cutoff_pct_of_move"], 20.0)
        self.assertAlmostEqual(top["symbols"][0]["error_pct_of_move"], 5.0)

    def test_fewer_symbols_than_top_n_uses_the_best_half(self):
        top = self._day([50, 5, 20, 80, 10], top_n=100)["top"]
        self.assertEqual(top["size"], 2)
        self.assertEqual([s["symbol"] for s in top["symbols"]], ["S001", "S004"])
        self.assertEqual(self._day([40], top_n=100)["top"]["size"], 1)

    def test_a_symbol_whose_macd_did_not_move_is_scored_but_not_ranked(self):
        rows = [
            _row("FLAT", predicted=1.0, actual=1.0, origin=1.0, origin_prev=1.0),
            _row("AAA", predicted=1.5, actual=2.0, origin=1.0),
            _row("BBB", predicted=1.9, actual=2.0, origin=1.0),
        ]
        (day,) = tr.summarize_track_record([MON], rows, top_n=100)
        self.assertEqual(day["symbols"], 3)
        self.assertEqual(day["top"]["ranked"], 2)
        self.assertEqual([s["symbol"] for s in day["top"]["symbols"]], ["BBB"])

    def test_beat_drift_is_counted_within_the_list(self):
        rows = [
            # Drift exact (origin 0 after +10/day would be 10): the model loses.
            _row("AAA", predicted=9.9, actual=10.0, origin=0.0, origin_prev=-10.0),
            # Drift says 0: the model wins.
            _row("BBB", predicted=9.8, actual=10.0, origin=0.0, origin_prev=0.0),
            _row("CCC", predicted=5.0, actual=10.0, origin=0.0, origin_prev=0.0),
            _row("DDD", predicted=4.0, actual=10.0, origin=0.0, origin_prev=0.0),
        ]
        (day,) = tr.summarize_track_record([MON], rows, top_n=2)
        self.assertEqual([s["symbol"] for s in day["top"]["symbols"]], ["AAA", "BBB"])
        self.assertEqual(day["top"]["beats_drift"], 1)
        self.assertEqual(day["beats_drift"], 3)


class GetTrackRecordTest(unittest.TestCase):
    def setUp(self):
        self.calls = []

        def fake_rows(model, symbols, latest_date, days=5, model_versions=None):
            self.calls.append({"model": model, "symbols": symbols, "latest_date": latest_date,
                               "days": days, "model_versions": model_versions})
            return [MON, TUE], [_row("AAA", 1.5, 2.0, origin=1.0, target=TUE)]

        fake_db_utils = types.ModuleType("db_utils")
        fake_db_utils.get_forecast_track_rows = fake_rows
        fake_macd_utils = types.ModuleType("macd_utils")
        fake_macd_utils.get_latest_market_date = lambda: TUE
        fake_ensembles = types.ModuleType("services.ensemble_service")

        def resolve_ensemble(ensemble_id=None):
            if ensemble_id not in (None, "macdonly_warm5", "hidden16"):
                raise ValueError(f"Unknown ensemble: {ensemble_id}")
            known = {None: ("macdonly_warm5", [1, 2, 3]), "macdonly_warm5": ("macdonly_warm5", [1, 2, 3]),
                     "hidden16": ("hidden16", [4, 5, 6])}
            name, versions = known[ensemble_id]
            return {"id": name, "versions": versions}

        fake_ensembles.resolve_ensemble = resolve_ensemble
        patcher = mock.patch.dict(sys.modules, {
            "db_utils": fake_db_utils, "macd_utils": fake_macd_utils,
            "services.ensemble_service": fake_ensembles,
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_an_ensemble_is_scored_on_the_versions_it_serves(self):
        result = tr.get_track_record(["AAA"], "hidden16", days=2, top_n=10)

        self.assertEqual(self.calls[0], {"model": "hidden16", "symbols": ["AAA"], "latest_date": TUE,
                                         "days": 2, "model_versions": [4, 5, 6]})
        self.assertEqual((result["model"], result["versions"]), ("hidden16", [4, 5, 6]))
        self.assertEqual([d["available"] for d in result["days"]], [False, True])
        self.assertEqual(result["days"][1]["top"]["requested"], 10)

    def test_no_model_means_the_default_ensemble(self):
        result = tr.get_track_record(["AAA"])
        self.assertEqual(result["model"], "macdonly_warm5")
        self.assertEqual(self.calls[0]["model_versions"], [1, 2, 3])

    def test_arima_has_no_versions(self):
        result = tr.get_track_record(["AAA"], "arima")
        self.assertEqual((result["model"], result["versions"]), ("arima", None))
        self.assertIsNone(self.calls[0]["model_versions"])

    def test_an_unknown_ensemble_is_an_error(self):
        with self.assertRaises(ValueError):
            tr.get_track_record(["AAA"], "nope")


if __name__ == "__main__":
    unittest.main()
