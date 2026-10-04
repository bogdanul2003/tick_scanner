#!/usr/bin/env python3
"""
Evaluate a seed ensemble: several versions of one model, predictions averaged.

Loads models/{model_name}_{version}.* for each requested version, averages their
forecasts per input window, and scores the average with evaluate_forecast_model.py's
own watchlist loop and metric functions — so the numbers are directly comparable to a
single-model run of that script at the same --samples.

Motivation: seeds of one config agree on day 1 but spread by 10+ points of MACD DA on
days 3-5, so any single checkpoint is partly a draw from that spread.

Usage (from src/):
    python scripts/evaluate_ensemble.py --model-name gru_v1_residual_macdonly_warm5 \
        --versions 1,2,3 --watchlist sp500 --samples 20 --breakdown-by-day --lag-test
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import evaluate_forecast_model as efm
from models.neural_forecast import EnsembleForecaster


def load_ensemble(model_name: str, versions, signal_type: str, architecture: str):
    """Load every version and return a load_model()-shaped tuple for the ensemble."""
    loaded = []
    for v in versions:
        engine, model, normalization, details = efm.load_model(architecture, signal_type, model_name, v)
        if not model:
            raise FileNotFoundError(f"No model found for {model_name}_{v}")
        loaded.append((engine, model, normalization, details))

    engines = sorted({e for e, _, _, _ in loaded})
    ensemble = EnsembleForecaster([m for _, m, _, _ in loaded])
    return ("+".join(engines), ensemble, loaded[0][2], loaded[0][3])


def main():
    parser = argparse.ArgumentParser(description="Evaluate the averaged forecast of several model versions")
    parser.add_argument("--model-name", type=str, required=True,
                        help="Model name shared by the members: models/{model_name}_{version}.*")
    parser.add_argument("--versions", type=str, required=True,
                        help="Comma-separated versions to average, e.g. 1,2,3")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--watchlist", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--signal-type", type=str, choices=["macd", "signal_line"], default="macd")
    parser.add_argument("--architecture", type=str, default="bidirectional_gru")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--inference-forcast-horizon", type=int)
    parser.add_argument("--breakdown-by-day", action="store_true")
    parser.add_argument("--lag-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json-out", type=str, default=None,
                        help="With --watchlist: also write the headline metrics "
                             "(avg_mae, avg_da) to this path as JSON, for callers "
                             "that need the numbers rather than the printed summary.")
    args = parser.parse_args()

    versions = [int(v) for v in args.versions.split(",") if v.strip()]
    cached = load_ensemble(args.model_name, versions, args.signal_type, args.architecture)
    label = f"{args.model_name} ensemble of v{','.join(str(v) for v in versions)}"
    print(f"Ensemble: {len(versions)} members averaged ({label})")

    # run_watchlist_evaluation loads its model through load_model; hand it the ensemble
    # instead so the sampling, metrics and summary are the stock script's, unchanged.
    efm.load_model = lambda *a, **k: cached

    if not args.symbol and not args.watchlist:
        args.symbol = "MSFT"
    ex_list = [s.strip().upper() for s in args.exclude.split(",")] if args.exclude else []

    if args.watchlist:
        summary = efm.run_watchlist_evaluation(
            args.watchlist, args.signal_type, args.architecture, 30, args.samples, None,
            args.inference_forcast_horizon, False, False, ex_list, args.breakdown_by_day,
            args.lag_test, args.verbose, label, None
        )
        if "error" in summary:
            print(f"Evaluation failed: {summary['error']}")
            sys.exit(1)
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump({"watchlist": args.watchlist, "samples": args.samples,
                           "mae": float(summary["avg_mae"]),
                           "directional_accuracy": float(summary["avg_da"])}, f)
    else:
        efm.run_evaluation(
            args.symbol.upper(), args.signal_type, args.architecture, 30, args.samples, None,
            args.inference_forcast_horizon, False, False, True, cached, args.breakdown_by_day,
            args.lag_test, label, None
        )


if __name__ == "__main__":
    main()
