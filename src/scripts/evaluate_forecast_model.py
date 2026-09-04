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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLUMN_MAPPING = {
    "macd": "MACD",
    "signal_line": "Signal_Line",
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "volume": "Volume",
    "ema12": "EMA12",
    "ema26": "EMA26",
    "ma20": "MA20",
    "ma50": "MA50",
}


def get_historical_data_for_features(
    symbol: str, 
    feature_names: List[str],
    signal_type: str, 
    end_date: datetime, 
    days_back: int = 60
) -> Tuple[np.ndarray, List[datetime]]:
    """Get historical multi-feature data from database.

    Reads directly from stock_cache (same data the UI uses) without triggering
    any Yahoo Finance fetches. If data is missing from the cache, the evaluation
    will simply have fewer samples rather than re-fetching live data.
    """
    from db_utils import fetch_bulk_from_cache
    calendar_days = int(days_back * 1.6)
    if any(f.lower() == "volume" for f in feature_names):
        # Relative volume needs a trailing 20-day window before the first valid
        # row; pad the fetch so trimming those NaN rows doesn't starve days_back.
        calendar_days += 40
    start_date = end_date - timedelta(days=calendar_days)
    bulk = fetch_bulk_from_cache([symbol], start_date, end_date)
    cached_df = bulk.get(symbol)
    if cached_df is None or cached_df.empty:
        return np.empty((0, len(feature_names)), dtype=np.float32), []
        
    primary_col = "MACD" if signal_type == "macd" else "Signal_Line"
    cols_data = []
    
    for f in feature_names:
        f_lower = f.lower()
        if f_lower in ("macd", "signal_line"):
            col = "MACD" if f_lower == "macd" else "Signal_Line"
            if col in cached_df.columns:
                cols_data.append(cached_df[col].values)
            else:
                return np.empty((0, len(feature_names)), dtype=np.float32), []
        elif f_lower == "delta":
            if primary_col in cached_df.columns:
                p_vals = cached_df[primary_col].values.astype(np.float32)
                deltas = np.zeros(len(cached_df), dtype=np.float32)
                deltas[1:] = p_vals[1:] - p_vals[:-1]
                cols_data.append(deltas)
            else:
                return np.empty((0, len(feature_names)), dtype=np.float32), []
        elif f_lower == "open-close":
            if "Open" in cached_df.columns and "Close" in cached_df.columns:
                o_vals = cached_df["Open"].values.astype(np.float32)
                c_vals = cached_df["Close"].values.astype(np.float32)
                cols_data.append(np.where(o_vals != 0, (c_vals - o_vals) / o_vals, 0.0).astype(np.float32))
            else:
                return np.empty((0, len(feature_names)), dtype=np.float32), []
        elif f_lower == "volume":
            # Relative volume (today's volume / trailing 20-day average) instead
            # of the raw count — see train_forecast_model.py's get_training_data.
            if "Volume" in cached_df.columns:
                vol_vals = cached_df["Volume"].values.astype(np.float32)
                vol_avg = cached_df["Volume"].rolling(window=20, min_periods=20).mean().values
                cols_data.append(np.where(vol_avg > 0, vol_vals / vol_avg, np.nan).astype(np.float32))
            else:
                return np.empty((0, len(feature_names)), dtype=np.float32), []
        else:
            db_col = COLUMN_MAPPING.get(f_lower, f_lower.upper())
            if db_col in cached_df.columns:
                cols_data.append(cached_df[db_col].values)
            else:
                return np.empty((0, len(feature_names)), dtype=np.float32), []

    matrix = np.column_stack(cols_data).astype(np.float32)
    dates = [idx.to_pydatetime() for idx in cached_df.index]
    
    valid_mask = ~np.isnan(matrix).any(axis=1) & ~np.isinf(matrix).any(axis=1)
    matrix = matrix[valid_mask]
    dates = [d for d, v in zip(dates, valid_mask) if v]
    return matrix, dates


def get_historical_data(
    symbol: str, 
    signal_type: str, 
    end_date: datetime, 
    days_back: int = 60
) -> Tuple[List[float], List[datetime]]:
    """Legacy helper for 1D single-signal historical data."""
    mat, dates = get_historical_data_for_features(symbol, [signal_type], signal_type, end_date, days_back)
    if len(mat) == 0: return [], []
    return mat[:, 0].tolist(), dates



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


def _usable_arima(values, horizon: int):
    """Return the first `horizon` ARIMA values, or None if the forecast is unusable.

    A failed fit is recorded as `[None] * horizon` by the worker loop. A plain
    truthiness check treats that list as a real forecast and hands Nones straight to
    `_compute_metrics`, which raises `TypeError: unsupported operand type(s) for -:
    'NoneType' and 'float'`. In a watchlist run that kills the whole pass — the loop
    does not catch exceptions from run_evaluation — so a single illiquid symbol can
    discard half an hour of work.
    """
    if not values:
        return None
    trimmed = values[:horizon]
    if not trimmed or any(v is None for v in trimmed):
        return None
    return trimmed


def load_model(architecture: str, signal_type: str, model_name: str = None, version: int = None):
    """Load the trained model (CoreML or PyTorch).

    When `model_name` is given, resolves `models/{model_name}_{version}.*`
    (latest version if `version` is None) instead of the legacy
    `{signal_type}_{architecture}_forecaster.*` naming.
    """
    from models.lstm_forecaster import get_model_path, get_pytorch_model_path
    coreml_path = get_model_path(signal_type, architecture, model_name=model_name, version=version)
    pytorch_path = get_pytorch_model_path(signal_type, architecture, model_name=model_name, version=version)

    if model_name:
        pin = f"version={version}" if version is not None else "latest version"
        print(f"Resolved model_name='{model_name}' ({pin}) -> {os.path.basename(coreml_path)}")


    if os.path.exists(coreml_path):
        try:
            from models.neural_forecast import CoreMLForecaster
            f = CoreMLForecaster(coreml_path, signal_type)
            if f.is_available:
                details = {
                    "hidden_size": getattr(f, "hidden_size", None),
                    "num_layers": getattr(f, "num_layers", None),
                    "batch_size": getattr(f, "batch_size", None),
                    "feature_names": getattr(f, "feature_names", ["macd"]),
                    "input_size": getattr(f, "input_size", 1),
                    "target_size": getattr(f, "target_size", 1)
                }
                return ("coreml", f, f.normalization_type, details)
        except Exception as e: print(f"Warning: CoreML load failed: {e}")
        
    if os.path.exists(pytorch_path):
        try:
            import torch
            from models.lstm_forecaster import MACDForecasterTrainer
            cp = torch.load(pytorch_path, map_location="cpu", weights_only=False)
            feature_names = cp.get("feature_names", ["macd", "delta"] if cp.get("include_delta", False) else ["macd"])
            trainer = MACDForecasterTrainer(
                seq_length=cp.get("seq_length", 30), 
                forecast_horizon=cp.get("forecast_horizon", 5), 
                hidden_size=cp.get("hidden_size", 64), 
                num_layers=cp.get("num_layers", 2),
                architecture=cp.get("architecture", architecture),
                feature_names=feature_names
            )
            trainer.load(pytorch_path)
            details = {
                "hidden_size": cp.get("hidden_size"),
                "num_layers": cp.get("num_layers"),
                "batch_size": cp.get("batch_size"),
                "feature_names": getattr(trainer, "feature_names", feature_names),
                "input_size": getattr(trainer, "input_size", len(feature_names)),
                "target_size": getattr(trainer, "target_size", 1)
            }
            return ("pytorch", trainer, cp.get("normalization_type", "global"), details)
        except Exception as e: print(f"Warning: PyTorch load failed: {e}")
        
    return (None, None, "unknown", {})


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
        prev_a = baselines[i]
        for p, a in zip(p_seq, a_seq):
            if (p > prev_a) == (a > prev_a): correct += 1
            total += 1
            prev_a = a
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
                prev_val = baselines[i] if day_idx == 0 else a_seq[day_idx-1]
                if (p_seq[day_idx] > prev_val) == (a_seq[day_idx] > prev_val): correct += 1
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


def _rollout_predict(model, seq, prev_val, steps, inc_delta, fill="mean", delta_mode="predicted"):
    """Recursive (iterated) multi-step forecast: predict one day, append that prediction to
    the input window, drop the oldest row, re-predict — `steps` times, keeping only each
    run's day-1 output.

    Motivation: the direct multi-horizon head's day-1 output is by far its strongest
    (MACD DA ~56-65% vs ~40% at day 5), and the day-3+ DA is *below* chance, which is an
    anti-correlation signature rather than an out-of-signal one — an MSE-trained head
    collapses toward the anchor as the horizon grows, so in a trending window it calls the
    direction backwards. Re-anchoring on each step's own output carries the trend forward
    instead of flattening it. See docs/FORECAST_MODEL_IMPROVEMENTS.md (A4).

    Two caveats that must be kept in mind when reading the numbers this produces:
      - **Exposure bias.** The model only ever saw windows of real data during training, but
        by the final step most of the window's tail is its own output. Nothing here corrects
        for that; a fair version of this idea needs scheduled sampling at training time.
      - **Unpredictable feature columns.** Anything past macd/delta (open-close, volume, ...)
        cannot be forecast by this model at all, so `fill` decides what goes into those
        columns for the synthesized rows. The filler is computed once from the *original*
        window and reused, so every appended row carries the same value there.
    """
    window = np.array(seq, dtype=np.float32, copy=True)
    n_feat = window.shape[1]
    if fill == "zero":
        filler = np.zeros(n_feat, dtype=np.float32)
    elif fill == "hold":
        filler = window[-1].copy()
    else:  # "mean" — for open-close this is ~0, for relative volume ~1, i.e. each column's
           # own neutral value rather than a constant that means different things per feature
        filler = window.mean(axis=0)

    macd_out, delta_out = [], []
    for _ in range(steps):
        p = model.predict(window, prev_value=prev_val)
        macd_next = float(p[0, 0])
        has_delta_out = inc_delta and p.ndim > 1 and p.shape[1] > 1
        if has_delta_out and delta_mode == "predicted":
            delta_next = float(p[0, 1])
        else:
            # "diff": derive delta the way every other code path defines it — as the
            # difference of the primary series — instead of trusting the second head.
            delta_next = macd_next - float(window[-1, 0])
        macd_out.append(macd_next)
        delta_out.append(delta_next)

        row = filler.copy()
        row[0] = macd_next
        if inc_delta and n_feat > 1:
            row[1] = delta_next
        prev_val = float(window[-1, 0])
        window = np.vstack([window[1:], row[None, :]])

    return macd_out, (delta_out if inc_delta else None)


def run_evaluation(symbol: str, signal_type: str, architecture: str, input_days: int, num_samples: int, forecast_horizon: int = None, inference_forcast_horizon: int = None, compare_arima: bool = False, arima_only: bool = False, verbose: bool = True, cached_model: Any = None, breakdown_by_day: bool = False, lag_test: bool = False, model_name: str = None, version: int = None, rollout: bool = False, rollout_fill: str = "mean", rollout_delta: str = "predicted") -> Dict[str, Any]:
    """Run model evaluation for a single symbol."""
    from macd_utils import get_latest_market_date
    if arima_only: engine_type, model, normalization_type, details = "statsmodels", None, "N/A", {}
    elif cached_model: engine_type, model, normalization_type, details = cached_model
    else: engine_type, model, normalization_type, details = load_model(architecture, signal_type, model_name, version)
    if not arima_only and not model: return {"error": f"No model found for {model_name or f'{signal_type}_{architecture}'}"}
    
    inc_delta = getattr(model, "include_delta", False) if model else False
    feature_names = getattr(model, "feature_names", ["macd", "delta"] if inc_delta else ["macd"]) if not arima_only else [signal_type]
    in_size = getattr(model, "input_size", len(feature_names)) if model else 1
    seq_len = input_days if arima_only else model.seq_length
    fh = forecast_horizon if forecast_horizon else (model.forecast_horizon if model else 5)
    efh = min(inference_forcast_horizon if inference_forcast_horizon else fh, fh)
    
    model_info = {
        "architecture": "ARIMA" if arima_only else (model_name or architecture),
        "engine": "statsmodels" if arima_only else engine_type.upper(),
        "normalization": normalization_type,
        "features": "MACD" if arima_only else (" + ".join([f.upper() for f in feature_names])),
        "seq_len": seq_len,
        "forecast_horizon": fh,
        "eval_horizon": efh,
        "mode": (f"rollout (recursive, fill={rollout_fill}, delta={rollout_delta})"
                 if rollout and not arima_only else "direct"),
        "hidden_size": details.get("hidden_size"),
        "num_layers": details.get("num_layers"),
        "batch_size": details.get("batch_size")
    }

    # Use evaluation horizon for sampling constraints if smaller than model's trained horizon
    total_needed = seq_len + num_samples + efh + (1 if inc_delta else 0)
    fetch_until = get_latest_market_date()
    all_vals, all_dates = get_historical_data_for_features(symbol, feature_names, signal_type, fetch_until, total_needed + 10)
    
    if len(all_vals) < total_needed: return {"error": f"Not enough data: {len(all_vals)} < {total_needed}"}

    sample_data = []
    for i in range(num_samples):
        # We want to end the last sample exactly when we have efh days of actual data left.
        # Adding +1 to offset ensures we use the most recent available data window.
        s_idx = len(all_vals) - num_samples - efh + i - seq_len + 1
        e_idx = s_idx + seq_len
        if s_idx < 0: continue
        in_seq = all_vals[s_idx:e_idx, :]
        dt = all_dates[e_idx-1]
        
        # We try to get up to fh days of actuals for the model's output comparison,
        # but we only require at least efh days to be present.
        act_v = all_vals[e_idx : min(e_idx + fh, len(all_vals)), 0]
        act_d = [act_v[j] - (in_seq[-1, 0] if j==0 else act_v[j-1]) for j in range(len(act_v))]
        sample_data.append((i, in_seq, dt, act_v, act_d, all_dates[e_idx : min(e_idx + fh, len(all_vals))]))

    neural_macd, neural_delta = [], []
    for i, seq, dt, av, ad, ads in sample_data:
        if not arima_only:
            # Recalculate s_idx for prev_val consistency
            s_idx = len(all_vals) - num_samples - efh + i - seq_len + 1
            prev_val = float(all_vals[s_idx - 1, 0]) if inc_delta and s_idx > 0 else None
            if rollout:
                # efh sequential one-step predictions instead of one efh-step prediction
                rm, rd = _rollout_predict(model, seq, prev_val, efh, inc_delta, rollout_fill, rollout_delta)
                neural_macd.append(rm)
                neural_delta.append(rd)
            else:
                p = model.predict(seq, prev_value=prev_val)
                neural_macd.append(p[:efh, 0].tolist())
                if inc_delta and p.shape[1] > 1: neural_delta.append(p[:efh, 1].tolist())
                else: neural_delta.append(None)
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
        ap = _usable_arima(arima_res.get(i), efh) if (compare_arima or arima_only) else None
        pp = ap if arima_only else neural_macd[i]
        if pp is not None:
            mlen = min(len(pp), len(av[:efh]))
            p_preds.append(pp[:mlen]); actuals.append(av[:mlen]); bases.append(float(seq[-1, 0]))
            if neural_delta[i]: d_preds.append(neural_delta[i][:mlen]); d_actuals.append(ad[:mlen])
            if ap: a_preds.append(ap[:mlen])
            
            if verbose:
                print(f"Sample {len(p_preds)}: Ends {sample_data[i][2].strftime('%Y-%m-%d')} | Last MACD: {seq[-1, 0]:.4f}")
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
            d_bases = [float(sample_data[idx][1][-1, 0] - sample_data[idx][1][-2, 0]) for idx in range(len(d_preds))]
            res["delta_lag"] = _compute_lag_metrics(d_preds, d_actuals, d_bases)
            
    if breakdown_by_day:
        res["primary_per_day"] = _compute_per_day_metrics(p_preds, actuals, bases)
        if d_preds:
            res["delta_per_day"] = _compute_per_day_metrics(d_preds, d_actuals, [0.0]*len(d_preds))
        if a_preds:
            # Same bases as the model, so the two per-day tables are directly comparable.
            res["arima_per_day"] = _compute_per_day_metrics(a_preds, actuals[:len(a_preds)], bases[:len(a_preds)])

    if verbose:
        print("="*90 + "\nEVALUATION SUMMARY\n" + "="*90)
        print(f"Symbol:        {symbol}")
        print(f"Model:         {model_info['architecture']} ({model_info['engine']})")
        print(f"Features:      {model_info['features']}")
        print(f"Normalization: {model_info['normalization']}")
        print(f"Horizons:      Seq={model_info['seq_len']}, Forecast={model_info['forecast_horizon']}, Eval={model_info['eval_horizon']}")
        print(f"Mode:          {model_info.get('mode', 'direct')}")
        
        h, l, b = model_info.get("hidden_size"), model_info.get("num_layers"), model_info.get("batch_size")
        params = [f"Hidden={h if h else '?'}", f"Layers={l if l else '?'}", f"Batch={b if b else '?'}"]
        print(f"Parameters:    {', '.join(params)}")
            
        print(f"Samples:       {num_samples}")
        print("-" * 90)
        print(f"MACD:  MAE={m_macd['mae']:.6f}, RMSE={m_macd['rmse']:.6f}, DirAcc={m_macd['directional_accuracy']:.2%}")
        if m_delta: print(f"DELTA: MAE={m_delta['mae']:.6f}, RMSE={m_delta['rmse']:.6f}, DirAcc={m_delta['directional_accuracy']:.2%}")
        if m_arima: print(f"ARIMA: MAE={m_arima['mae']:.6f}, RMSE={m_arima['rmse']:.6f}, DirAcc={m_arima['directional_accuracy']:.2%}"
                          f"  (on {m_arima['total_directions']} of {m_macd['total_directions']} model comparisons)")
        
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
            if res.get("arima_per_day"):
                print("\nPER-DAY (ARIMA):")
                for d, m in sorted(res["arima_per_day"].items()): print(f"  Day {d}: DA={m['directional_accuracy']:.1%}, MAE={m['mae']:.6f}")
        print("="*90)

    return res


def run_watchlist_evaluation(watchlist_name: str, signal_type: str, architecture: str, input_days: int, num_samples: int, forecast_horizon: int = None, inference_forcast_horizon: int = None, compare_arima: bool = False, arima_only: bool = False, exclude_list: List[str] = None, breakdown_by_day: bool = False, lag_test: bool = False, verbose: bool = False, model_name: str = None, version: int = None, rollout: bool = False, rollout_fill: str = "mean", rollout_delta: str = "predicted") -> Dict[str, Any]:
    from db_utils import get_watchlist_symbols
    try: symbols = sorted(get_watchlist_symbols(watchlist_name))
    except Exception as e: return {"error": str(e)}
    if not symbols: return {"error": "Empty watchlist"}
    if exclude_list: symbols = [s for s in symbols if s.upper() not in exclude_list]

    cm = load_model(architecture, signal_type, model_name, version) if not arima_only else None
    print(f"\nWATCHLIST: {watchlist_name} | Samples: {num_samples}\n" + "="*90)

    all_res = []
    for i, sym in enumerate(symbols):
        if not verbose: print(f"[{i+1}/{len(symbols)}] {sym:<8}...", end="", flush=True)
        try:
            r = run_evaluation(sym, signal_type, architecture, input_days, num_samples, forecast_horizon, inference_forcast_horizon, compare_arima, arima_only, verbose, cm, breakdown_by_day, lag_test, model_name, version, rollout, rollout_fill, rollout_delta)
        except Exception as e:
            # One bad symbol must not discard the whole pass — a watchlist run is
            # ~30 min with --compare, and the loop below is the only place progress
            # is accumulated.
            r = {"error": f"{type(e).__name__}: {e}"}
        if "error" not in r:
            all_res.append(r)
            if not verbose: print(f" DONE (DA: {r['primary_metrics']['directional_accuracy']:.1%}, MAE: {r['primary_metrics']['mae']:.4f})")
        else:
            if not verbose: print(f" FAILED: {r['error']}")
    
    if not all_res: return {"error": "No symbols evaluated"}
    
    # AGGREGATE
    def _vals(key, subkey): return [r[key][subkey] for r in all_res if r.get(key)]
    def avg(key, subkey): return np.mean(_vals(key, subkey))
    def med(key, subkey):
        # MEAN WAPE is not the scale-free answer either: normalizing removes the price
        # effect but not divergence, and one blown-up fit dominates it (measured
        # 2026-09-04: a single symbol, ALGN, scored 63,469% ARIMA WAPE and pulled a
        # 72-symbol mean to 915% against a median of 24.8%). Compare MEDIAN WAPE.
        # Raw MAE is NOT scale-free: MACD magnitude scales with share price, so the
        # mean over symbols is dominated by expensive names (measured 2026-09-04:
        # the top 50 of 501 symbols carry 45% of the total MAE mass, led by NVR and
        # AZO). WAPE normalizes by |actual| and is the number to compare across
        # different symbol sets. The mean is taken over PER-SYMBOL scores, so divergent fits can
        # dominate it — measured on 2026-09-04, ARIMA's watchlist mean MAE was 3.68 while
        # its median across a 42-symbol sample was 0.72. Report both; when they disagree
        # by a lot, the mean is describing the tail, not the typical symbol.
        return np.median(_vals(key, subkey))
    
    p_mae, p_rmse, p_da = avg("primary_metrics", "mae"), avg("primary_metrics", "rmse"), avg("primary_metrics", "directional_accuracy")
    m_info = all_res[0]["model_info"]

    print("\n" + "="*90 + f"\nWATCHLIST SUMMARY: {watchlist_name}\n" + "="*90)
    print(f"Model:         {m_info['architecture']} ({m_info['engine']})")
    print(f"Features:      {m_info['features']}")
    print(f"Normalization: {m_info['normalization']}")
    print(f"Horizons:      Seq={m_info['seq_len']}, Forecast={m_info['forecast_horizon']}, Eval={m_info['eval_horizon']}")
    print(f"Mode:          {m_info.get('mode', 'direct')}")
    
    h, l, b = m_info.get("hidden_size"), m_info.get("num_layers"), m_info.get("batch_size")
    params = [f"Hidden={h if h else '?'}", f"Layers={l if l else '?'}", f"Batch={b if b else '?'}"]
    print(f"Parameters:    {', '.join(params)}")
        
    print(f"Symbols:       {len(all_res)} / {len(symbols)} (Samples/Symbol: {num_samples})")
    print("-" * 90)
    print(f"MACD:  MAE={p_mae:.6f}, RMSE={p_rmse:.6f}, DA={p_da:.2%}")
    print(f"       median MAE={med('primary_metrics','mae'):.6f}, "
          f"median DA={med('primary_metrics','directional_accuracy'):.2%}, "
          f"median WAPE={med('primary_metrics','wape'):.2f}% "
          f"(mean {avg('primary_metrics','wape'):.2f}%)")
    
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
        n_arima = sum(1 for r in all_res if r.get("arima_metrics"))
        a_mae, a_da = avg("arima_metrics", "mae"), avg("arima_metrics", "directional_accuracy")
        print(f"\nARIMA: MAE={a_mae:.6f}, DA={a_da:.2%}  (fitted on {n_arima}/{len(all_res)} symbols)")
        a_med = med("arima_metrics", "mae")
        print(f"       median MAE={a_med:.6f}, "
              f"median DA={med('arima_metrics','directional_accuracy'):.2%}, "
              f"median WAPE={med('arima_metrics','wape'):.2f}% "
              f"(mean {avg('arima_metrics','wape'):.2f}%)")
        if a_mae > 2 * a_med:
            print(f"       WARNING: mean MAE is {a_mae/a_med:.1f}x the median — the average is "
                  f"dominated by a minority of divergent ARIMA fits, not typical behaviour. "
                  f"Compare medians.")
        if n_arima < len(all_res):
            print(f"       NOTE: {len(all_res) - n_arima} symbols have model metrics but no ARIMA "
                  f"fit; the two rows above are averaged over different symbol sets.")

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

        if any(r.get("arima_per_day") for r in all_res):
            print("\nPER-DAY (ARIMA):")
            for d in range(1, horizon + 1):
                rows = [r["arima_per_day"][d] for r in all_res
                        if r.get("arima_per_day") and d in r["arima_per_day"]]
                if not rows:
                    continue
                print(f"  Day {d}: DA={np.mean([x['directional_accuracy'] for x in rows]):.1%}, "
                      f"MAE={np.mean([x['mae'] for x in rows]):.6f}")

    print("="*90)
    return {"watchlist": watchlist_name, "avg_mae": p_mae, "avg_da": p_da}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                         help="Path to the JSON training config used for the model (see "
                              "src/configs/). Pulls model_name and signal_type from it.")
    parser.add_argument("--model-name", type=str, default=None,
                         help="Look up models/{model_name}_{version}.* directly, without a config file.")
    parser.add_argument("--model-version", type=int, default=None,
                         help="Pin to this version instead of the latest (only with --config/--model-name).")
    parser.add_argument("--symbol", type=str)
    parser.add_argument("--watchlist", type=str)
    parser.add_argument("--exclude", type=str)
    parser.add_argument("--signal-type", type=str, choices=["macd", "signal_line"], default=None)
    parser.add_argument("--architecture", type=str, default="bidirectional_gru")
    parser.add_argument("--input-days", type=int, default=30)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--forecast-horizon", type=int)
    parser.add_argument("--inference-forcast-horizon", type=int)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--arima-only", action="store_true")
    parser.add_argument("--breakdown-by-day", action="store_true")
    parser.add_argument("--lag-test", action="store_true")
    parser.add_argument("--rollout", action="store_true",
                        help="Recursive multi-step forecasting: run the model once per forecast day, "
                             "feeding each run's day-1 prediction back in as input, instead of taking "
                             "all N days from a single direct prediction.")
    parser.add_argument("--rollout-fill", type=str, default="mean", choices=["mean", "hold", "zero"],
                        help="What to put in feature columns the model cannot predict (open-close, "
                             "volume, ...) when synthesizing a rolled-forward row. mean = that column's "
                             "mean over the input window (default), hold = its last observed value, "
                             "zero = 0.0. Ignored for macd/delta-only models.")
    parser.add_argument("--rollout-delta", type=str, default="predicted", choices=["predicted", "diff"],
                        help="Where the delta column of a rolled-forward row comes from: the model's own "
                             "delta head (default), or differencing the predicted primary series the way "
                             "every other code path defines delta.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--list-models", action="store_true")
    args = parser.parse_args()

    if args.list_models:
        mdir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")
        if os.path.exists(mdir):
            for f in sorted(os.listdir(mdir)):
                if f.endswith(".mlpackage") or f.endswith(".pt"): print(f"  - {f}")
        return

    model_name = args.model_name
    if args.config:
        from config_utils import load_json_config, get_model_name
        json_config = load_json_config(args.config)
        model_name = get_model_name(json_config)
        if args.signal_type is None:
            args.signal_type = json_config.get("signal_type")
    if args.signal_type is None:
        args.signal_type = "macd"

    if not args.symbol and not args.watchlist: args.symbol = "MSFT"
    ex_list = [s.strip().upper() for s in args.exclude.split(",")] if args.exclude else []

    if args.watchlist:
        run_watchlist_evaluation(args.watchlist, args.signal_type, args.architecture, args.input_days, args.samples, args.forecast_horizon, args.inference_forcast_horizon, args.compare, args.arima_only, ex_list, args.breakdown_by_day, args.lag_test, args.verbose, model_name, args.model_version, args.rollout, args.rollout_fill, args.rollout_delta)
    else:
        run_evaluation(args.symbol.upper(), args.signal_type, args.architecture, args.input_days, args.samples, args.forecast_horizon, args.inference_forcast_horizon, args.compare, args.arima_only, args.verbose, None, args.breakdown_by_day, args.lag_test, model_name, args.model_version, args.rollout, args.rollout_fill, args.rollout_delta)

if __name__ == "__main__":
    main()
