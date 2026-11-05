#!/usr/bin/env python3
"""
Score several groups of forecast models on the same recent forecast origins, for the
freshness backtest (services/freshness_backtest.py).

Reads a plan JSON:
    {"symbols": [...], "origins": ["YYYY-MM-DD", ...], "horizon": 5,
     "calendar": ["YYYY-MM-DD", ...],          # trading days, covering origins + horizon
     "arms": [{"id": "age0", "members": ["/path/model_1.pt", ...]}, ...]}

For every arm it scores the averaged forecast of its members (as the served ensemble
does) and each member on its own, plus the drift and flat baselines, against the
actual MACD. The output keeps per-origin, per-horizon sums rather than one average, so
the caller can resample origin days for confidence intervals:

    {"origins": [...], "horizon": 5, "symbols_scored": n,
     "series": {"<arm>": {...}, "<arm>#<member index>": {...}, "drift": {...}, "flat": {...}}}

with each series {"n", "err", "err_pct", "dir"} as [origin][horizon] lists: forecasts
scored, summed absolute error, summed error as % of the origin day's close, and
forecasts that called the direction right. Direction uses the rule of
evaluate_forecast_model.py and the live track record: forecast and actual on the same
side of the previous day's actual MACD.

Every model is loaded from its PyTorch checkpoint, so backtest models need no Core ML
export and every arm runs through the same code.

Usage (from src/):
    python scripts/evaluate_freshness.py --plan plan.json --json-out result.json
"""
import argparse
import json
import os
import sys
from datetime import date, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate_forecast_model as efm


def load_member(path: str):
    """A trained model from its .pt checkpoint, the way evaluate_forecast_model.load_model does."""
    import torch
    from models.lstm_forecaster import MACDForecasterTrainer

    cp = torch.load(path, map_location="cpu", weights_only=False)
    feature_names = cp.get("feature_names", ["macd", "delta"] if cp.get("include_delta", False) else ["macd"])
    trainer = MACDForecasterTrainer(
        seq_length=cp.get("seq_length", 30),
        forecast_horizon=cp.get("forecast_horizon", 5),
        hidden_size=cp.get("hidden_size", 64),
        num_layers=cp.get("num_layers", 2),
        architecture=cp.get("architecture", "bidirectional_gru"),
        feature_names=feature_names,
        device="cpu",
    )
    trainer.load(path)
    return trainer


def _empty_series(origins: int, horizon: int):
    return {key: [[0.0] * horizon for _ in range(origins)] for key in ("n", "err", "err_pct", "dir")}


def _add(series, o: int, h: int, predicted: float, actual: float, prev_actual: float, close: float):
    error = abs(predicted - actual)
    series["n"][o][h] += 1
    series["err"][o][h] += error
    series["err_pct"][o][h] += 100.0 * error / close
    series["dir"][o][h] += float((predicted > prev_actual) == (actual > prev_actual))


def _symbol_data(symbol: str, feature_names, first_needed: date, last: date):
    """Feature matrix, dates and closes for one symbol, rows aligned."""
    from db_utils import fetch_bulk_from_cache

    days_back = int((last - first_needed).days / 1.6) + 10
    matrix, dates = efm.get_historical_data_for_features(symbol, feature_names, "macd", last, days_back)
    if len(matrix) == 0:
        return None
    df = fetch_bulk_from_cache([symbol], dates[0], last).get(symbol)
    if df is None or "Close" not in df.columns:
        return None
    close_by_date = {idx.date(): float(c) for idx, c in zip(df.index, df["Close"].values)}
    day_list = [d.date() for d in dates]
    closes = [close_by_date.get(d) for d in day_list]
    return matrix, day_list, closes


def score(plan):
    origins = [date.fromisoformat(d) for d in plan["origins"]]
    calendar = [date.fromisoformat(d) for d in plan["calendar"]]
    horizon = int(plan.get("horizon", 5))
    position = {d: i for i, d in enumerate(calendar)}

    arms = []
    for arm in plan["arms"]:
        members = [load_member(p) for p in arm["members"]]
        arms.append((arm["id"], members))
        print(f"Loaded arm {arm['id']}: {len(members)} member(s)", flush=True)

    reference = arms[0][1][0]
    seq_length = reference.seq_length
    feature_names = list(reference.feature_names)
    include_delta = bool(getattr(reference, "include_delta", "delta" in feature_names))
    for arm_id, members in arms:
        for m in members:
            if m.seq_length != seq_length or list(m.feature_names) != feature_names:
                raise ValueError(f"Arm {arm_id} has a member with different inputs; arms must share one config")

    series = {"drift": _empty_series(len(origins), horizon), "flat": _empty_series(len(origins), horizon)}
    for arm_id, members in arms:
        series[arm_id] = _empty_series(len(origins), horizon)
        for k in range(len(members)):
            series[f"{arm_id}#{k}"] = _empty_series(len(origins), horizon)

    # Calendar days before the first origin, padded for the input window and holidays.
    first_needed = origins[0] - timedelta(days=int((seq_length + 5) * 1.6) + 10)
    last = calendar[-1]
    scored_symbols = 0
    for count, symbol in enumerate(plan["symbols"], start=1):
        data = _symbol_data(symbol, feature_names, first_needed, last)
        if data is None:
            continue
        matrix, days, closes = data
        row_of = {d: i for i, d in enumerate(days)}
        any_scored = False
        for o, origin in enumerate(origins):
            idx = row_of.get(origin)
            start = position[origin]
            if idx is None or idx - seq_length < 0 or idx + horizon >= len(days):
                continue
            # Only score a symbol whose next rows are the next trading days: a missing
            # bar would otherwise shift every later horizon.
            if any(days[idx + h] != calendar[start + h] for h in range(1, horizon + 1)):
                continue
            close = closes[idx]
            if not close:
                continue
            window = matrix[idx - seq_length + 1: idx + 1]
            prev_value = float(matrix[idx - seq_length, 0]) if include_delta else None
            origin_value = float(matrix[idx, 0])
            step = origin_value - float(matrix[idx - 1, 0])
            actual = [float(matrix[idx + h, 0]) for h in range(1, horizon + 1)]
            prev_actual = [origin_value] + actual[:-1]

            for arm_id, members in arms:
                outputs = [m.predict(window, prev_value=prev_value)[:horizon, 0] for m in members]
                for k, out in enumerate(outputs):
                    for h in range(horizon):
                        _add(series[f"{arm_id}#{k}"], o, h, float(out[h]), actual[h], prev_actual[h], close)
                mean = np.mean(outputs, axis=0)
                for h in range(horizon):
                    _add(series[arm_id], o, h, float(mean[h]), actual[h], prev_actual[h], close)
            for h in range(horizon):
                _add(series["drift"], o, h, origin_value + (h + 1) * step, actual[h], prev_actual[h], close)
                _add(series["flat"], o, h, origin_value, actual[h], prev_actual[h], close)
            any_scored = True
        scored_symbols += any_scored
        if count % 50 == 0:
            print(f"  {count}/{len(plan['symbols'])} symbols", flush=True)

    return {"origins": plan["origins"], "horizon": horizon, "symbols_scored": scored_symbols, "series": series}


def main():
    parser = argparse.ArgumentParser(description="Score model groups on the same recent forecast origins")
    parser.add_argument("--plan", required=True, help="Plan JSON (see module docstring)")
    parser.add_argument("--json-out", required=True, help="Where to write the per-origin sums")
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    result = score(plan)
    tmp = f"{args.json_out}.tmp"
    with open(tmp, "w") as f:
        json.dump(result, f)
    os.replace(tmp, args.json_out)
    print(f"Scored {result['symbols_scored']} symbols on {len(result['origins'])} origins")


if __name__ == "__main__":
    main()
