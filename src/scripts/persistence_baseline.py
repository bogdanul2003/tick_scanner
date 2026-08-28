#!/usr/bin/env python3
"""
Persistence baselines for MACD / Signal Line forecasting.

Answers the question a model's metrics cannot answer on their own: how much of
the reported skill is free? MACD is EMA12 - EMA26 -- a heavily filtered, strongly
autocorrelated series -- so naive rules can score high directional accuracy
without predicting anything.

Two baselines, no model and no training involved:

  flat   macd[t+k] = macd[t]                  (predict no change)
         delta[t+k] = 0
  drift  macd[t+k] = macd[t] + (k+1)*delta[t] (linear extrapolation)
         delta[t+k] = delta[t]                (momentum persists)

`drift` is the one that matters for direction: its delta prediction carries the
sign of the last observed change, so its Day-1 delta accuracy IS the momentum
persistence rate of the series. Compare a model's Day-1 delta DA against it --
if the model isn't clearly above, it has learned nothing beyond "MACD keeps
moving the way it was moving".

Sampling, metrics and aggregation are taken from evaluate_forecast_model.py, so
the numbers are directly comparable to a model evaluation run with the same
--samples / --seq-length / --forecast-horizon / --watchlist.

Usage (from src/):
    python scripts/persistence_baseline.py --watchlist sp500 --samples 20 \
        --seq-length 30 --forecast-horizon 5
"""
import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate_forecast_model import (  # noqa: E402
    get_historical_data,
    _compute_metrics,
    _compute_per_day_metrics,
)

BASELINES = ("flat", "drift")


def _predict(name: str, in_seq: np.ndarray, horizon: int):
    """Return (macd_predictions, delta_predictions) for one baseline."""
    last = float(in_seq[-1])
    last_delta = float(in_seq[-1] - in_seq[-2])

    if name == "flat":
        return [last] * horizon, [0.0] * horizon
    if name == "drift":
        return (
            [last + last_delta * (k + 1) for k in range(horizon)],
            [last_delta] * horizon,
        )
    raise ValueError(f"Unknown baseline: {name}")


def run_symbol(symbol, signal_type, seq_len, horizon, num_samples):
    """
    Build the same sample windows evaluate_forecast_model.py uses, then score
    every baseline on them. Returns {baseline: {...metrics}} or None.
    """
    from macd_utils import get_latest_market_date

    # +1 so a real delta exists at the window's first position, matching the
    # evaluate script's accounting.
    total_needed = seq_len + num_samples + horizon + 1
    all_vals, _dates = get_historical_data(
        symbol, signal_type, get_latest_market_date(), total_needed + 10
    )
    if len(all_vals) < total_needed:
        return None

    samples = []
    for i in range(num_samples):
        s_idx = len(all_vals) - num_samples - horizon + i - seq_len + 1
        e_idx = s_idx + seq_len
        if s_idx < 1:  # need index s_idx-1 for the leading delta
            continue
        in_seq = np.array(all_vals[s_idx:e_idx])
        act_v = all_vals[e_idx:min(e_idx + horizon, len(all_vals))]
        if not act_v:
            continue
        act_d = [
            act_v[j] - (in_seq[-1] if j == 0 else act_v[j - 1])
            for j in range(len(act_v))
        ]
        samples.append((in_seq, act_v, act_d))

    if not samples:
        return None

    out = {}
    for name in BASELINES:
        p_macd, p_delta, a_macd, a_delta, bases = [], [], [], [], []
        for in_seq, act_v, act_d in samples:
            pm, pd = _predict(name, in_seq, horizon)
            n = min(len(pm), len(act_v))
            p_macd.append(pm[:n])
            p_delta.append(pd[:n])
            a_macd.append(act_v[:n])
            a_delta.append(act_d[:n])
            bases.append(float(in_seq[-1]))

        out[name] = {
            "macd": _compute_metrics(p_macd, a_macd, bases),
            "delta": _compute_metrics(p_delta, a_delta, [0.0] * len(p_delta)),
            "macd_per_day": _compute_per_day_metrics(p_macd, a_macd, bases),
            "delta_per_day": _compute_per_day_metrics(p_delta, a_delta, [0.0] * len(p_delta)),
        }
    return out


def _avg(results, name, channel, key):
    return float(np.mean([r[name][channel][key] for r in results]))


def _avg_day(results, name, channel, day, key):
    vals = [
        r[name][f"{channel}_per_day"][day][key]
        for r in results
        if day in r[name][f"{channel}_per_day"]
    ]
    return float(np.mean(vals)) if vals else float("nan")


def report(results, symbols_attempted, horizon, num_samples, label):
    width = 90
    print("\n" + "=" * width)
    print(f"PERSISTENCE BASELINES: {label}")
    print("=" * width)
    print(f"Symbols:       {len(results)} / {symbols_attempted} "
          f"(Samples/Symbol: {num_samples})")
    print(f"Horizons:      Seq=n/a, Forecast={horizon}, Eval={horizon}")
    print("-" * width)

    for name in BASELINES:
        macd_mae = _avg(results, name, "macd", "mae")
        macd_rmse = _avg(results, name, "macd", "rmse")
        macd_da = _avg(results, name, "macd", "directional_accuracy")
        delta_mae = _avg(results, name, "delta", "mae")
        delta_rmse = _avg(results, name, "delta", "rmse")
        delta_da = _avg(results, name, "delta", "directional_accuracy")
        print(f"\n[{name.upper()}]")
        print(f"  MACD:  MAE={macd_mae:.6f}, RMSE={macd_rmse:.6f}, DA={macd_da:.2%}")
        print(f"  DELTA: MAE={delta_mae:.6f}, RMSE={delta_rmse:.6f}, DA={delta_da:.2%}")

    print("\n" + "-" * width)
    print("PER-DAY DIRECTIONAL ACCURACY (compare against your model's table)")
    header = f"  {'Day':<5}" + "".join(f"{n + ' MACD':>14}{n + ' DELTA':>15}"
                                       for n in BASELINES)
    print(header)
    for day in range(1, horizon + 1):
        row = f"  {day:<5}"
        for name in BASELINES:
            row += f"{_avg_day(results, name, 'macd', day, 'directional_accuracy'):>13.1%} "
            row += f"{_avg_day(results, name, 'delta', day, 'directional_accuracy'):>14.1%} "
        print(row)

    print("\n" + "-" * width)
    print("PER-DAY MAE")
    print(f"  {'Day':<5}" + "".join(f"{n + ' MACD':>14}{n + ' DELTA':>15}"
                                    for n in BASELINES))
    for day in range(1, horizon + 1):
        row = f"  {day:<5}"
        for name in BASELINES:
            row += f"{_avg_day(results, name, 'macd', day, 'mae'):>13.4f} "
            row += f"{_avg_day(results, name, 'delta', day, 'mae'):>14.4f} "
        print(row)

    print("\n" + "-" * width)
    drift_d1 = _avg_day(results, "drift", "delta", 1, "directional_accuracy")
    print(f"HEADLINE: drift Day-1 DELTA directional accuracy = {drift_d1:.2%}")
    print("A model's Day-1 delta DA must be clearly above this to have learned")
    print("anything beyond 'MACD keeps moving the way it was moving'.")
    print("=" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Persistence baselines for MACD forecasting"
    )
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--watchlist", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--signal-type", type=str,
                        choices=["macd", "signal_line"], default="macd")
    parser.add_argument("--seq-length", type=int, default=30,
                        help="Input window length; match the model being compared")
    parser.add_argument("--forecast-horizon", type=int, default=5,
                        help="Forecast horizon; match the model being compared")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.symbol and not args.watchlist:
        args.symbol = "MSFT"

    if args.watchlist:
        from db_utils import get_watchlist_symbols
        symbols = sorted(get_watchlist_symbols(args.watchlist))
        if not symbols:
            print(f"Error: watchlist '{args.watchlist}' not found or empty")
            sys.exit(1)
        label = args.watchlist
    else:
        symbols = [args.symbol.upper()]
        label = symbols[0]

    if args.exclude:
        excluded = {s.strip().upper() for s in args.exclude.split(",")}
        symbols = [s for s in symbols if s.upper() not in excluded]

    results = []
    for i, sym in enumerate(symbols):
        if len(symbols) > 1:
            print(f"[{i + 1}/{len(symbols)}] {sym:<8}...", end="", flush=True)
        try:
            r = run_symbol(sym, args.signal_type, args.seq_length,
                           args.forecast_horizon, args.samples)
        except Exception as e:
            r = None
            if len(symbols) > 1:
                print(f" FAILED: {e}")
        if r is None:
            if len(symbols) > 1:
                print(" SKIPPED (not enough data)")
            continue
        results.append(r)
        if len(symbols) > 1:
            print(f" drift D1 delta DA="
                  f"{r['drift']['delta_per_day'][1]['directional_accuracy']:.1%}")

    if not results:
        print("Error: no symbols could be evaluated")
        sys.exit(1)

    report(results, len(symbols), args.forecast_horizon, args.samples, label)


if __name__ == "__main__":
    main()
