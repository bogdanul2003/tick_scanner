"""
Track record of a forecast model: how its stored forecasts compared with the actual
MACD on each of the last trading days.

Only forecasts recorded before the outcome count (forecast_predictions table); nothing
is recomputed after the fact. See docs/FORECAST_TRACK_RECORD_PLAN.md.
"""
import math
from typing import Any, Dict, Iterable, List, Optional

ARIMA_MODEL = "arima"


def _mean(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _median(values: List[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    return ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2


def _usable(*values) -> bool:
    return all(v is not None and math.isfinite(v) for v in values)


def _score_forecast(row: Dict[str, Any]) -> Dict[str, Any]:
    """One stored forecast against its outcome."""
    predicted, actual = float(row["predicted"]), float(row["actual"])
    prev_actual, origin = float(row["prev_actual"]), float(row["origin"])
    lead = int(row["lead_days"])
    error = abs(predicted - actual)
    scored = {
        "symbol": row["symbol"],
        "lead_days": lead,
        "origin": origin,
        "predicted": predicted,
        "actual": actual,
        "move": actual - origin,
        "error": error,
        # Same rule as evaluate_forecast_model.py: forecast and actual on the same
        # side of the previous day's actual value.
        "direction_correct": (predicted > prev_actual) == (actual > prev_actual),
        "drift_error": None,
        "drift_direction_correct": None,
        "beats_drift": None,
        "error_pct_of_move": None,
    }
    origin_prev = row.get("origin_prev")
    if origin_prev is not None and math.isfinite(origin_prev):
        drift = origin + lead * (origin - float(origin_prev))
        scored["drift_error"] = abs(drift - actual)
        scored["drift_direction_correct"] = (drift > prev_actual) == (actual > prev_actual)
        scored["beats_drift"] = error < scored["drift_error"]
    if scored["move"] != 0:
        scored["error_pct_of_move"] = 100.0 * error / abs(scored["move"])
    return scored


def _top_size(rankable: int, top_n: int) -> int:
    """`top_n`, or the best half when there are not that many symbols to rank."""
    if rankable >= top_n:
        return top_n
    return max(1, rankable // 2) if rankable else 0


def _empty_day(day) -> Dict[str, Any]:
    return {
        "date": day.isoformat(), "available": False, "symbols": 0, "lead_days": [],
        "direction_accuracy": None, "drift_direction_accuracy": None,
        "mae": None, "median_error": None, "drift_mae": None, "beats_drift": None,
        "top": None,
    }


def summarize_track_record(trading_days: Iterable, rows: Iterable[Dict[str, Any]],
                           top_n: int = 100) -> List[Dict[str, Any]]:
    """
    Per-day track record from the rows db_utils.get_forecast_track_rows returns.

    For each symbol and day the most recent forecast made before that day is used
    (smallest lead_days). A day no forecast targets comes back with available=False
    and null scores. Days are returned in the order given.

    `top` ranks the day's symbols by error as a share of how far MACD moved since the
    forecast's starting day, best first, and keeps the first top_n (the best half
    when fewer than top_n can be ranked). A symbol whose MACD did not move has no
    such ratio and is left out of the ranking.
    """
    latest: Dict[Any, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        if not _usable(row.get("predicted"), row.get("actual"), row.get("prev_actual"), row.get("origin")):
            continue
        by_symbol = latest.setdefault(row["target_date"], {})
        kept = by_symbol.get(row["symbol"])
        if kept is None or row["lead_days"] < kept["lead_days"]:
            by_symbol[row["symbol"]] = row

    summary = []
    for day in trading_days:
        scored = [_score_forecast(r) for r in latest.get(day, {}).values()]
        if not scored:
            summary.append(_empty_day(day))
            continue

        with_drift = [s for s in scored if s["drift_error"] is not None]
        ranked = sorted((s for s in scored if s["error_pct_of_move"] is not None),
                        key=lambda s: (s["error_pct_of_move"], s["symbol"]))
        best = ranked[:_top_size(len(ranked), top_n)]

        summary.append({
            "date": day.isoformat(),
            "available": True,
            "symbols": len(scored),
            "lead_days": sorted({s["lead_days"] for s in scored}),
            "direction_accuracy": _mean([float(s["direction_correct"]) for s in scored]),
            "drift_direction_accuracy": _mean([float(s["drift_direction_correct"]) for s in with_drift]),
            "mae": _mean([s["error"] for s in scored]),
            "median_error": _median([s["error"] for s in scored]),
            "drift_mae": _mean([s["drift_error"] for s in with_drift]),
            "beats_drift": sum(1 for s in with_drift if s["beats_drift"]),
            "top": {
                "requested": top_n,
                "size": len(best),
                "ranked": len(ranked),
                "cutoff_pct_of_move": best[-1]["error_pct_of_move"] if best else None,
                "beats_drift": sum(1 for s in best if s["beats_drift"]),
                "symbols": best,
            },
        })
    return summary


def get_track_record(symbols: List[str], model: Optional[str] = None, days: int = 5,
                     top_n: int = 100) -> Dict[str, Any]:
    """
    Track record of `model` (an ensemble id, 'arima', or None for the default
    ensemble) over the last `days` trading days, for `symbols`.

    An ensemble is scored on forecasts made by the versions it currently serves, so
    the record describes the models a forecast run now would use.
    """
    from db_utils import get_forecast_track_rows
    from macd_utils import get_latest_market_date

    versions = None
    if model != ARIMA_MODEL:
        from services.ensemble_service import resolve_ensemble
        ensemble = resolve_ensemble(model)
        model, versions = ensemble["id"], ensemble["versions"]

    trading_days, rows = get_forecast_track_rows(
        model, symbols, get_latest_market_date(), days=days, model_versions=versions
    )
    return {
        "model": model,
        "versions": versions,
        "days": summarize_track_record(trading_days, rows, top_n=top_n),
    }
