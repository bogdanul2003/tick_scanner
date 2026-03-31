#!/usr/bin/env python3
"""
Evaluation script for MACD/Signal Line Forecaster models.

This script loads a trained model and runs inference on historical data,
comparing predictions to actual values stored in the database.

Supports side-by-side comparison of Neural (LSTM/GRU) vs ARIMA predictions
against actual values using the --compare flag.
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def get_historical_data(
    symbol: str, 
    signal_type: str, 
    end_date: datetime, 
    days_back: int = 60
) -> Tuple[List[float], List[datetime]]:
    """Get historical MACD or Signal Line data from database."""
    from macd_utils import get_macd_for_range
    calendar_days = int(days_back * 1.6)
    start_date = end_date - timedelta(days=calendar_days)
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    field_name = "macd" if signal_type == "macd" else "signal_line"
    values, dates = [], []
    for d in macd_data:
        if field_name in d and d[field_name] is not None:
            values.append(float(d[field_name]))
            dt = d["date"]
            dates.append(datetime.strptime(dt, "%Y-%m-%d") if isinstance(dt, str) else dt)
    return values, dates


def _run_arima_forecast_worker(symbol: str, signal_type: str, end_date_str: str, forecast_horizon: int, days_past: int = 100) -> np.ndarray:
    """Worker function for parallel ARIMA forecasting."""
    from datetime import datetime
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return run_arima_forecast(symbol, signal_type, end_date, forecast_horizon, days_past)


def run_arima_forecast(symbol: str, signal_type: str, end_date: datetime, forecast_horizon: int, days_past: int = 100) -> np.ndarray:
    """Run ARIMA forecast using production flow."""
    from forecast_utils import arima_macd_positive_forecast
    result = arima_macd_positive_forecast(symbol, days_past=days_past, forecast_days=forecast_horizon, end_date=end_date, skip_cache=True, verbose=False)
    if "error" in result.get("details", {}): raise RuntimeError(result["details"]["error"])
    forecasted = result.get("forecasted_macd", {})
    values = list(forecasted.values()) if isinstance(forecasted, dict) else list(forecasted)
    if not values: raise RuntimeError("ARIMA returned no forecast values")
    return np.array(values[:forecast_horizon])


def load_model(architecture: str, signal_type: str):
    """Load the trained model (CoreML or PyTorch)."""
    from models.lstm_forecaster import get_model_path, get_pytorch_model_path
    coreml_path = get_model_path(signal_type, architecture)
    pytorch_path = get_pytorch_model_path(signal_type, architecture)
    if os.path.exists(coreml_path):
        try:
            from models.neural_forecast import CoreMLForecaster
            f = CoreMLForecaster(coreml_path, signal_type)
            if f.is_available: return ("coreml", f, f.normalization_type)
        except Exception as e: print(f"Warning: CoreML load failed: {e}")
    if os.path.exists(pytorch_path):
        try:
            import torch
            from models.lstm_forecaster import MACDForecasterTrainer
            cp = torch.load(pytorch_path, map_location="cpu")
            trainer = MACDForecasterTrainer(seq_length=cp.get("seq_length", 30), forecast_horizon=cp.get("forecast_horizon", 5), hidden_size=cp.get("hidden_size", 64), architecture=cp.get("architecture", architecture))
            trainer.load(pytorch_path)
            return ("pytorch", trainer, cp.get("normalization_type", "global"))
        except Exception as e: print(f"Warning: PyTorch load failed: {e}")
    return (None, None, "unknown")


def _compute_metrics(predictions: List[List[float]], actuals: List[List[float]], baselines: List[float]) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, WAPE and Directional Accuracy."""
    p_flat = np.array([p for seq in predictions for p in seq])
    a_flat = np.array([a for seq in actuals for a in seq])
    if len(p_flat) == 0: return {"mae": 0, "rmse": 0, "mape": 0, "wape": 0, "directional_accuracy": 0}
    mae = float(np.mean(np.abs(p_flat - a_flat)))
    rmse = float(np.sqrt(np.mean((p_flat - a_flat) ** 2)))
    mape = float(np.mean(np.abs((p_flat - a_flat) / (a_flat + 1e-8))) * 100)
    wape = float(np.sum(np.abs(a_flat - p_flat)) / (np.sum(np.abs(a_flat)) + 1e-8) * 100)
    correct, total = 0, 0
    for i, (p_seq, a_seq) in enumerate(zip(predictions, actuals)):
        base = baselines[i]
        for p, a in zip(p_seq, a_seq):
            if (p > base) == (a > base): correct += 1
            total += 1
    return {"mae": mae, "rmse": rmse, "mape": mape, "wape": wape, "directional_accuracy": correct/total if total > 0 else 0, "correct_directions": correct, "total_directions": total}


def _compute_per_day_metrics(predictions: List[List[float]], actuals: List[List[float]], baselines: List[float]) -> Dict[int, Dict[str, float]]:
    """Compute metrics broken down by forecast day."""
    res = {}
    if not predictions: return res
    horizon = len(predictions[0])
    for day_idx in range(horizon):
        d_preds, d_actuals, correct, total = [], [], 0, 0
        for i, (p_seq, a_seq) in enumerate(zip(predictions, actuals)):
            if day_idx < len(p_seq):
                d_preds.append(p_seq[day_idx]); d_actuals.append(a_seq[day_idx])
                if (p_seq[day_idx] > baselines[i]) == (a_seq[day_idx] > baselines[i]): correct += 1
                total += 1
        if d_preds:
            dp, da = np.array(d_preds), np.array(d_actuals)
            res[day_idx+1] = {"mae": float(np.mean(np.abs(dp - da))), "rmse": float(np.sqrt(np.mean((dp - da)**2))), "directional_accuracy": correct/total if total > 0 else 0}
    return res


def _compute_lag_metrics(predictions: List[List[float]], actuals: List[List[float]], baselines: List[float]) -> Dict[str, float]:
    """Compute metrics against 'lagged' actual values."""
    p_all, l_all = [], []
    for i, (p_seq, a_seq) in enumerate(zip(predictions, actuals)):
        lagged = [baselines[i]] + a_seq[:-1]
        mlen = min(len(p_seq), len(lagged))
        p_all.extend(p_seq[:mlen]); l_all.extend(lagged[:mlen])
    p_all, l_all = np.array(p_all), np.array(l_all)
    if len(p_all) == 0: return {"mae": 0, "rmse": 0}
    return {"mae": float(np.mean(np.abs(p_all - l_all))), "rmse": float(np.sqrt(np.mean((p_all - l_all)**2)))}


def run_evaluation(symbol: str, signal_type: str, architecture: str, input_days: int, num_samples: int, forecast_horizon: int = None, inference_forcast_horizon: int = None, compare_arima: bool = False, arima_only: bool = False, verbose: bool = True, cached_model: Any = None, breakdown_by_day: bool = False, lag_test: bool = False) -> Dict[str, Any]:
    """Run model evaluation for a single symbol."""
    from macd_utils import get_latest_market_date
    if arima_only: engine_type, model, normalization_type = "statsmodels", None, "N/A"
    elif cached_model: engine_type, model, normalization_type = cached_model
    else: engine_type, model, normalization_type = load_model(architecture, signal_type)
    if not arima_only and not model: return {"error": f"No model found for {signal_type}_{architecture}"}
    
    inc_delta = getattr(model, "include_delta", False) if model else False
    in_size = getattr(model, "input_size", 1) if model else 1
    seq_len = input_days if arima_only else model.seq_length
    fh = forecast_horizon if forecast_horizon else (model.forecast_horizon if model else 5)
    efh = min(inference_forcast_horizon if inference_forcast_horizon else fh, fh)
    
    model_info = {
        "architecture": "ARIMA" if arima_only else architecture,
        "engine": "statsmodels" if arima_only else engine_type.upper(),
        "normalization": normalization_type,
        "features": "MACD" if arima_only else ("MACD + Delta" if inc_delta else "MACD"),
        "seq_len": seq_len,
        "forecast_horizon": fh,
        "eval_horizon": efh
    }

    total_needed = seq_len + num_samples + fh + (1 if inc_delta else 0)
    all_vals, all_dates = get_historical_data(symbol, signal_type, get_latest_market_date(), total_needed)
    if len(all_vals) < total_needed: return {"error": f"Not enough data: {len(all_vals)} < {total_needed}"}

    sample_data = []
    for i in range(num_samples):
        s_idx = len(all_vals) - num_samples - fh + i - seq_len
        e_idx = s_idx + seq_len
        if s_idx < 0: continue
        in_seq = np.array(all_vals[s_idx:e_idx])
        dt = all_dates[e_idx-1]
        act_v = all_vals[e_idx:e_idx+fh]
        act_d = [act_v[j] - (in_seq[-1] if j==0 else act_v[j-1]) for j in range(len(act_v))]
        sample_data.append((i, in_seq, dt, act_v, act_d, all_dates[e_idx:e_idx+fh]))

    neural_macd, neural_delta = [], []
    for i, seq, dt, av, ad, ads in sample_data:
        if not arima_only:
            p = model.predict(seq)
            if inc_delta and in_size > 1: neural_macd.append(p[:efh, 0].tolist()); neural_delta.append(p[:efh, 1].tolist())
            else: neural_macd.append(p[:efh].tolist()); neural_delta.append(None)
        else: neural_macd.append(None); neural_delta.append(None)

    arima_res = {}
    if compare_arima or arima_only:
        with ProcessPoolExecutor(max_workers=4) as ex:
            futs = {ex.submit(_run_arima_forecast_worker, symbol, signal_type, dt.strftime("%Y-%m-%d"), fh): i for i, _, dt, _, _, _ in sample_data}
            for f in as_completed(futs):
                try: arima_res[futs[f]] = f.result().tolist()
                except: arima_res[futs[f]] = [None]*fh

    p_preds, d_preds, a_preds, actuals, d_actuals, bases = [], [], [], [], [], []
    for i, (_, seq, _, av, ad, _) in enumerate(sample_data):
        ap = arima_res.get(i)[:efh] if (compare_arima or arima_only) else None
        pp = ap if arima_only else neural_macd[i]
        if pp is not None:
            mlen = min(len(pp), len(av[:efh]))
            p_preds.append(pp[:mlen]); actuals.append(av[:mlen]); bases.append(float(seq[-1]))
            if neural_delta[i]: d_preds.append(neural_delta[i][:mlen]); d_actuals.append(ad[:mlen])
            if ap: a_preds.append(ap[:mlen])
            
            if verbose:
                print(f"Sample {len(p_preds)}: Ends {sample_data[i][2].strftime('%Y-%m-%d')} | Last MACD: {seq[-1]:.4f}")
                for j in range(mlen):
                    line = f"  Day {j+1}: Pred={pp[j]:8.4f}, Act={av[j]:8.4f}, Err={pp[j]-av[j]:+8.4f}"
                    if neural_delta[i]: line += f" | Delta: P={neural_delta[i][j]:.4f}, A={ad[j]:.4f}"
                    print(line)
                print()

    if not p_preds: return {"error": "No predictions generated"}
    
    m_macd = _compute_metrics(p_preds, actuals, bases)
    m_delta = _compute_metrics(d_preds, d_actuals, [0.0]*len(d_preds)) if d_preds else None
    m_arima = _compute_metrics(a_preds, actuals, bases) if a_preds else None
    
    res = {"symbol": symbol, "primary_metrics": m_macd, "delta_metrics": m_delta, "arima_metrics": m_arima, "model_info": model_info}
    
    if lag_test:
        res["primary_lag"] = _compute_lag_metrics(p_preds, actuals, bases)
        if d_preds:
            d_bases = [sample_data[idx][1][-1] - sample_data[idx][1][-2] for idx in range(len(d_preds))]
            res["delta_lag"] = _compute_lag_metrics(d_preds, d_actuals, d_bases)
            
    if breakdown_by_day:
        res["primary_per_day"] = _compute_per_day_metrics(p_preds, actuals, bases)
        if d_preds:
            res["delta_per_day"] = _compute_per_day_metrics(d_preds, d_actuals, [0.0]*len(d_preds))

    if verbose:
        print("="*90 + "\nEVALUATION SUMMARY\n" + "="*90)
        print(f"Symbol:        {symbol}")
        print(f"Model:         {model_info['architecture']} ({model_info['engine']})")
        print(f"Features:      {model_info['features']}")
        print(f"Normalization: {model_info['normalization']}")
        print(f"Horizons:      Seq={model_info['seq_len']}, Forecast={model_info['forecast_horizon']}, Eval={model_info['eval_horizon']}")
        print(f"Samples:       {num_samples}")
        print("-" * 90)
        print(f"MACD:  MAE={m_macd['mae']:.6f}, RMSE={m_macd['rmse']:.6f}, DirAcc={m_macd['directional_accuracy']:.2%}")
        if m_delta: print(f"DELTA: MAE={m_delta['mae']:.6f}, RMSE={m_delta['rmse']:.6f}, DirAcc={m_delta['directional_accuracy']:.2%}")
        
        if lag_test:
            lm = res["primary_lag"]
            print(f"\nLAG ANALYSIS (MACD):  Lagged MAE={lm['mae']:.6f} (Ratio={lm['mae']/m_macd['mae']:.2f}x)")
            if m_delta:
                ldm = res["delta_lag"]
                print(f"LAG ANALYSIS (DELTA): Lagged MAE={ldm['mae']:.6f} (Ratio={ldm['mae']/m_delta['mae']:.2f}x)")
        
        if breakdown_by_day:
            print("\nPER-DAY (MACD):")
            for d, m in sorted(res["primary_per_day"].items()): print(f"  Day {d}: DA={m['directional_accuracy']:.1%}, MAE={m['mae']:.6f}")
            if m_delta:
                print("\nPER-DAY (DELTA):")
                for d, m in sorted(res["delta_per_day"].items()): print(f"  Day {d}: DA={m['directional_accuracy']:.1%}, MAE={m['mae']:.6f}")
        print("="*90)

    return res


def run_watchlist_evaluation(watchlist_name: str, signal_type: str, architecture: str, input_days: int, num_samples: int, forecast_horizon: int = None, inference_forcast_horizon: int = None, compare_arima: bool = False, arima_only: bool = False, exclude_list: List[str] = None, breakdown_by_day: bool = False, lag_test: bool = False, verbose: bool = False) -> Dict[str, Any]:
    from db_utils import get_watchlist_symbols
    try: symbols = sorted(get_watchlist_symbols(watchlist_name))
    except Exception as e: return {"error": str(e)}
    if not symbols: return {"error": "Empty watchlist"}
    if exclude_list: symbols = [s for s in symbols if s.upper() not in exclude_list]
    
    cm = load_model(architecture, signal_type) if not arima_only else None
    print(f"\nWATCHLIST: {watchlist_name} | Samples: {num_samples}\n" + "="*90)
    
    all_res = []
    for i, sym in enumerate(symbols):
        if not verbose: print(f"[{i+1}/{len(symbols)}] {sym:<8}...", end="", flush=True)
        r = run_evaluation(sym, signal_type, architecture, input_days, num_samples, forecast_horizon, inference_forcast_horizon, compare_arima, arima_only, verbose, cm, breakdown_by_day, lag_test)
        if "error" not in r:
            all_res.append(r)
            if not verbose: print(f" DONE (DA: {r['primary_metrics']['directional_accuracy']:.1%}, MAE: {r['primary_metrics']['mae']:.4f})")
        else:
            if not verbose: print(f" FAILED: {r['error']}")
    
    if not all_res: return {"error": "No symbols evaluated"}
    
    # AGGREGATE
    def avg(key, subkey): return np.mean([r[key][subkey] for r in all_res if r.get(key)])
    
    p_mae, p_rmse, p_da = avg("primary_metrics", "mae"), avg("primary_metrics", "rmse"), avg("primary_metrics", "directional_accuracy")
    m_info = all_res[0]["model_info"]

    print("\n" + "="*90 + f"\nWATCHLIST SUMMARY: {watchlist_name}\n" + "="*90)
    print(f"Model:         {m_info['architecture']} ({m_info['engine']})")
    print(f"Features:      {m_info['features']}")
    print(f"Normalization: {m_info['normalization']}")
    print(f"Horizons:      Seq={m_info['seq_len']}, Forecast={m_info['forecast_horizon']}, Eval={m_info['eval_horizon']}")
    print(f"Symbols:       {len(all_res)} / {len(symbols)} (Samples/Symbol: {num_samples})")
    print("-" * 90)
    print(f"MACD:  MAE={p_mae:.6f}, RMSE={p_rmse:.6f}, DA={p_da:.2%}")
    
    if any(r.get("delta_metrics") for r in all_res):
        d_mae, d_rmse, d_da = avg("delta_metrics", "mae"), avg("delta_metrics", "rmse"), avg("delta_metrics", "directional_accuracy")
        print(f"DELTA: MAE={d_mae:.6f}, RMSE={d_rmse:.6f}, DA={d_da:.2%}")
        
    if lag_test:
        pl_mae = avg("primary_lag", "mae")
        print(f"\nLAG (MACD):  MAE={pl_mae:.6f} (Ratio={pl_mae/p_mae:.2f}x)")
        if any(r.get("delta_lag") for r in all_res):
            dl_mae = avg("delta_lag", "mae")
            print(f"LAG (DELTA): MAE={dl_mae:.6f} (Ratio={dl_mae/d_mae:.2f}x)")
            
    if compare_arima and any(r.get("arima_metrics") for r in all_res):
        a_mae, a_da = avg("arima_metrics", "mae"), avg("arima_metrics", "directional_accuracy")
        print(f"\nARIMA: MAE={a_mae:.6f}, DA={a_da:.2%}")

    if breakdown_by_day:
        print("\nPER-DAY (MACD):")
        horizon = len(all_res[0]["primary_per_day"])
        for d in range(1, horizon + 1):
            d_da = np.mean([r["primary_per_day"][d]["directional_accuracy"] for r in all_res if r.get("primary_per_day") and d in r["primary_per_day"]])
            d_mae = np.mean([r["primary_per_day"][d]["mae"] for r in all_res if r.get("primary_per_day") and d in r["primary_per_day"]])
            print(f"  Day {d}: DA={d_da:.1%}, MAE={d_mae:.6f}")
            
        if any(r.get("delta_per_day") for r in all_res):
            print("\nPER-DAY (DELTA):")
            for d in range(1, horizon + 1):
                d_da = np.mean([r["delta_per_day"][d]["directional_accuracy"] for r in all_res if r.get("delta_per_day") and d in r["delta_per_day"]])
                d_mae = np.mean([r["delta_per_day"][d]["mae"] for r in all_res if r.get("delta_per_day") and d in r["delta_per_day"]])
                print(f"  Day {d}: DA={d_da:.1%}, MAE={d_mae:.6f}")
                
    print("="*90)
    return {"watchlist": watchlist_name, "avg_mae": p_mae, "avg_da": p_da}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--watchlist", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--signal-type", type=str, choices=["macd", "signal_line"], default="macd")
    parser.add_argument("--architecture", type=str, default="bidirectional_gru")
    parser.add_argument("--input-days", type=int, default=30)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--forecast-horizon", type=int)
    parser.add_argument("--inference-forcast-horizon", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--arima-only", action="store_true")
    parser.add_argument("--breakdown-by-day", action="store_true")
    parser.add_argument("--lag-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()
    
    if args.list_models:
        mdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        if os.path.exists(mdir):
            for f in sorted(os.listdir(mdir)):
                if f.endswith(".mlpackage") or f.endswith(".pt"): print(f"  - {f}")
        return

    if not args.symbol and not args.watchlist: args.symbol = "MSFT"
    ex_list = [s.strip().upper() for s in args.exclude.split(",")] if args.exclude else []
    
    if args.watchlist:
        run_watchlist_evaluation(args.watchlist, args.signal_type, args.architecture, args.input_days, args.samples, args.forecast_horizon, args.inference_forcast_horizon, args.compare, args.arima_only, ex_list, args.breakdown_by_day, args.lag_test, args.verbose)
    else:
        run_evaluation(args.symbol.upper(), args.signal_type, args.architecture, args.input_days, args.samples, args.forecast_horizon, args.inference_forcast_horizon, args.compare, args.arima_only, args.verbose, None, args.breakdown_by_day, args.lag_test)

if __name__ == "__main__":
    main()
