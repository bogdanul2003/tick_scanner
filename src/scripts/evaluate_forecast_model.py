#!/usr/bin/env python3
"""
Evaluation script for MACD/Signal Line Forecaster models.

This script loads a trained model and runs inference on historical data,
comparing predictions to actual values stored in the database.

Supports side-by-side comparison of Neural (LSTM/GRU) vs ARIMA predictions
against actual values using the --compare flag.

Usage:
    python evaluate_forecast_model.py [--symbol SYMBOL] [--signal-type TYPE] [--architecture ARCH]
    python evaluate_forecast_model.py --symbol AAPL --compare

Example:
    python evaluate_forecast_model.py --symbol AAPL --architecture bidirectional_gru
    python evaluate_forecast_model.py --symbol MSFT --compare --samples 20
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
    """
    Get historical MACD or Signal Line data from database.
    
    Args:
        symbol: Stock symbol
        signal_type: 'macd' or 'signal_line'
        end_date: End date for data retrieval
        days_back: Number of trading days to look back
        
    Returns:
        Tuple of (values list, dates list)
    """
    from macd_utils import get_macd_for_range
    
    # Multiply by 1.6 to account for weekends and holidays
    # (roughly 2 out of 7 days are non-trading, plus ~7-10 holidays/year)
    calendar_days = int(days_back * 1.6)
    start_date = end_date - timedelta(days=calendar_days)
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    
    field_name = "macd" if signal_type == "macd" else "signal_line"
    
    values = []
    dates = []
    for d in macd_data:
        if field_name in d and d[field_name] is not None:
            values.append(float(d[field_name]))
            # Handle both datetime objects and string dates
            date_val = d["date"]
            if isinstance(date_val, str):
                date_val = datetime.strptime(date_val, "%Y-%m-%d")
            dates.append(date_val)
    
    return values, dates


def _run_arima_forecast_worker(
    symbol: str,
    signal_type: str,
    end_date_str: str,
    forecast_horizon: int,
    days_past: int = 100
) -> np.ndarray:
    """
    Worker function for parallel ARIMA forecasting.
    Converts date string back to datetime.
    """
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return run_arima_forecast(symbol, signal_type, end_date, forecast_horizon, days_past)


def run_arima_forecast(
    symbol: str,
    signal_type: str,
    end_date: datetime,
    forecast_horizon: int,
    days_past: int = 100
) -> np.ndarray:
    """
    Run ARIMA forecast using the same production flow as the API.

    Uses arima_macd_positive_forecast from forecast_utils.py with dynamic
    windowing, grid search, and model caching — the same behavior as when
    calling the /forecast/macd/arima_positive endpoint.

    Args:
        symbol: Stock symbol
        signal_type: 'macd' or 'signal_line'
        end_date: The date to treat as "today" for the forecast
        forecast_horizon: Number of trading days to forecast
        days_past: Calendar days of history to feed ARIMA (default 100, same as API)

    Returns:
        Array of forecasted values (length = forecast_horizon)
    """
    from forecast_utils import arima_macd_positive_forecast

    result = arima_macd_positive_forecast(
        symbol,
        days_past=days_past,
        forecast_days=forecast_horizon,
        end_date=end_date,
        skip_cache=True,
        verbose=False
    )

    if "error" in result.get("details", {}):
        raise RuntimeError(result["details"]["error"])

    # Extract forecasted values from the result dict
    forecasted_macd = result.get("forecasted_macd", {})
    if isinstance(forecasted_macd, dict):
        values = list(forecasted_macd.values())
    else:
        values = list(forecasted_macd)

    if not values:
        raise RuntimeError("ARIMA returned no forecast values")

    return np.array(values[:forecast_horizon])


def load_model(architecture: str, signal_type: str):
    """
    Load the trained model.
    
    Args:
        architecture: Model architecture name
        signal_type: 'macd' or 'signal_line'
        
    Returns:
        Tuple of (engine_type, model, normalization_type)
    """
    from models.lstm_forecaster import get_model_path, get_pytorch_model_path
    
    # Try Core ML first (faster inference)
    coreml_path = get_model_path(signal_type, architecture)
    pytorch_path = get_pytorch_model_path(signal_type, architecture)
    
    if os.path.exists(coreml_path):
        try:
            from models.neural_forecast import CoreMLForecaster
            forecaster = CoreMLForecaster(coreml_path, signal_type)
            if forecaster.is_available:
                return ("coreml", forecaster, forecaster.normalization_type)
        except Exception as e:
            print(f"Warning: Could not load Core ML model: {e}")
    
    if os.path.exists(pytorch_path):
        try:
            import torch
            from models.lstm_forecaster import MACDForecasterTrainer
            
            # Load checkpoint once to get all info
            checkpoint = torch.load(pytorch_path, map_location="cpu")
            
            trainer = MACDForecasterTrainer(
                seq_length=checkpoint.get("seq_length", 30),
                forecast_horizon=checkpoint.get("forecast_horizon", 5),
                hidden_size=checkpoint.get("hidden_size", 64),
                architecture=checkpoint.get("architecture", architecture)
            )
            trainer.load(pytorch_path)
            
            # Extract normalization type directly from checkpoint (it's guaranteed to be loaded now)
            normalization_type = checkpoint.get("normalization_type", "global")
            
            return ("pytorch", trainer, normalization_type)
        except Exception as e:
            print(f"Warning: Could not load PyTorch model: {e}")
    
    return (None, None, "unknown")


def _compute_metrics(
    all_values: List[float],
    predictions: List[List[float]],
    actuals: List[List[float]],
    num_samples: int,
    model_forecast_horizon: int,
    seq_length: int,
    full_model_horizon: int = None
) -> Dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and directional accuracy for a set of predictions.
    
    Args:
        all_values: Historical MACD values
        predictions: List of prediction sequences (already sliced to eval horizon)
        actuals: List of actual values (already sliced to eval horizon)
        num_samples: Number of samples evaluated
        model_forecast_horizon: Number of forecast days in each prediction (after slicing)
        seq_length: Input sequence length
        full_model_horizon: Full model forecast horizon (used for positioning calculations)
                           If None, uses model_forecast_horizon
    """
    # If not provided, assume full_model_horizon == model_forecast_horizon
    if full_model_horizon is None:
        full_model_horizon = model_forecast_horizon
    
    all_pred = np.array([p for pred in predictions for p in pred])
    all_actual = np.array([a for act in actuals for a in act])

    mae = float(np.mean(np.abs(all_pred - all_actual)))
    rmse = float(np.sqrt(np.mean((all_pred - all_actual) ** 2)))
    mape = float(np.mean(np.abs((all_pred - all_actual) / (all_actual + 1e-8))) * 100)

    correct_directions = 0
    total_directions = 0
    for i, (pred, act) in enumerate(zip(predictions, actuals)):
        # Use full_model_horizon for positioning (not the sliced eval horizon)
        start_idx = len(all_values) - num_samples - full_model_horizon + i - seq_length
        last_input = all_values[start_idx + seq_length - 1]
        for p, a in zip(pred, act):
            if (p > last_input) == (a > last_input):
                correct_directions += 1
            total_directions += 1

    directional_accuracy = correct_directions / total_directions if total_directions > 0 else 0.0

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "directional_accuracy": directional_accuracy,
        "total_predictions": len(all_pred),
        "correct_directions": correct_directions,
        "total_directions": total_directions
    }


def _compute_per_day_metrics(
    all_values: List[float],
    predictions: List[List[float]],
    actuals: List[List[float]],
    num_samples: int,
    model_forecast_horizon: int,
    seq_length: int,
    full_model_horizon: int = None
) -> Dict[int, Dict[str, float]]:
    """
    Compute metrics broken down by forecast day (day 1, day 2, etc).
    
    Args:
        full_model_horizon: Full model forecast horizon (used for positioning calculations)
                           If None, uses model_forecast_horizon
    
    Returns:
        Dict mapping day number (1-indexed) to metrics dict
    """
    # If not provided, assume full_model_horizon == model_forecast_horizon
    if full_model_horizon is None:
        full_model_horizon = model_forecast_horizon
    
    per_day_metrics = {}
    
    for day_idx in range(model_forecast_horizon):
        day_num = day_idx + 1
        
        # Collect all predictions and actuals for this day across all samples
        day_preds = []
        day_actuals = []
        day_correct_dirs = 0
        day_total_dirs = 0
        
        for sample_idx, (pred, act) in enumerate(zip(predictions, actuals)):
            if day_idx < len(pred) and day_idx < len(act):
                day_preds.append(pred[day_idx])
                day_actuals.append(act[day_idx])
                
                # Directional accuracy check - use full_model_horizon for positioning
                start_idx = len(all_values) - num_samples - full_model_horizon + sample_idx - seq_length
                last_input = all_values[start_idx + seq_length - 1]
                if (pred[day_idx] > last_input) == (act[day_idx] > last_input):
                    day_correct_dirs += 1
                day_total_dirs += 1
        
        if day_preds:
            day_preds = np.array(day_preds)
            day_actuals = np.array(day_actuals)
            
            mae = float(np.mean(np.abs(day_preds - day_actuals)))
            rmse = float(np.sqrt(np.mean((day_preds - day_actuals) ** 2)))
            mape = float(np.mean(np.abs((day_preds - day_actuals) / (day_actuals + 1e-8))) * 100)
            da = day_correct_dirs / day_total_dirs if day_total_dirs > 0 else 0.0
            
            per_day_metrics[day_num] = {
                "mae": mae,
                "rmse": rmse,
                "mape": mape,
                "directional_accuracy": da,
                "total": day_total_dirs
            }
    
    return per_day_metrics


def run_evaluation(
    symbol: str,
    signal_type: str,
    architecture: str,
    input_days: int,
    num_samples: int,
    forecast_horizon: int = None,
    inference_forcast_horizon: int = None,
    compare_arima: bool = False,
    verbose: bool = True,
    cached_model: Any = None,
    breakdown_by_day: bool = False
) -> Dict[str, Any]:
    """
    Run model evaluation comparing predictions to actual values.

    Args:
        symbol: Stock symbol
        signal_type: 'macd' or 'signal_line'
        architecture: Model architecture
        input_days: Number of days of input data
        num_samples: Number of prediction samples to evaluate
        forecast_horizon: Number of days to forecast (if None, uses model's default)
        inference_forcast_horizon: Number of forecasted days to use for evaluation (if None, uses forecast_horizon)
        compare_arima: If True, also run ARIMA on each sample and compare
        verbose: If True, print detailed output
        cached_model: Optional tuple of (engine_type, model) to reuse

    Returns:
        Evaluation results dictionary
    """
    from macd_utils import get_latest_market_date

    # Load model (or use cached one)
    if cached_model is not None:
        engine_type, model, normalization_type = cached_model
    else:
        engine_type, model, normalization_type = load_model(architecture, signal_type)

    if model is None:
        return {
            "error": f"No model found for {signal_type}_{architecture}",
            "searched_paths": {
                "coreml": f"models/{signal_type}_{architecture}_forecaster.mlpackage",
                "pytorch": f"models/{signal_type}_{architecture}_forecaster.pt"
            }
        }

    # Get model parameters
    seq_length = model.seq_length
    # Use the passed forecast_horizon parameter if specified, otherwise use model's default
    model_forecast_horizon = forecast_horizon if forecast_horizon is not None else model.forecast_horizon
    # Use the passed inference_forcast_horizon if specified, otherwise use forecast_horizon
    eval_forcast_horizon = inference_forcast_horizon if inference_forcast_horizon is not None else model_forecast_horizon
    
    # Validate that eval_forcast_horizon doesn't exceed model_forecast_horizon
    if eval_forcast_horizon > model_forecast_horizon:
        return {
            "error": f"inference-forcast-horizon ({eval_forcast_horizon}) cannot exceed forecast-horizon ({model_forecast_horizon})"
        }

    # Get historical data
    end_date = get_latest_market_date()
    # Calculate total data points needed: we work backwards from end date
    # We need: seq_length (input) + num_samples (sample positions) + model_forecast_horizon (future values)
    total_days_needed = seq_length + num_samples + model_forecast_horizon

    all_values, all_dates = get_historical_data(
        symbol, signal_type, end_date, total_days_needed
    )

    if len(all_values) < seq_length + model_forecast_horizon + num_samples:
        return {
            "error": f"Not enough data. Have {len(all_values)} points, need at least {seq_length + model_forecast_horizon + num_samples}"
        }

    # Run predictions
    neural_predictions = []
    arima_predictions = []
    actuals = []
    input_end_dates = []
    input_baselines = []  # Store the last input value for DA calculation

    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    mode_label = "COMPARISON: Neural vs ARIMA" if compare_arima else f"Neural ({engine_type})"

    if verbose:
        print(f"\n{'='*90}")
        print(f"Running {num_samples} predictions on {symbol} {signal_label}")
        print(f"Mode: {mode_label}")
        print(f"Neural Model: {architecture} ({engine_type})")
        print(f"Input sequence: {seq_length} days, Forecast: {model_forecast_horizon} days", end="")
        if eval_forcast_horizon < model_forecast_horizon:
            print(f" (evaluating on {eval_forcast_horizon} days)")
        else:
            print()
        if compare_arima:
            print(f"ARIMA parallel execution: 4 cores")
        print(f"{'='*90}\n")

    # Prepare data for ARIMA forecasts (if comparing) to enable parallel execution
    arima_tasks = {}  # Maps task index to (i, input_end_date) for later reassembly
    sample_data = []  # Collect all sample data first
    
    for i in range(num_samples):
        # Calculate indices
        start_idx = len(all_values) - num_samples - model_forecast_horizon + i - seq_length
        end_idx = start_idx + seq_length
        actual_start_idx = end_idx
        actual_end_idx = actual_start_idx + model_forecast_horizon

        if start_idx < 0 or actual_end_idx > len(all_values):
            continue

        # Get input sequence
        input_sequence = np.array(all_values[start_idx:end_idx])
        input_end_date = all_dates[end_idx - 1]

        # Get actual future values
        actual_values = all_values[actual_start_idx:actual_end_idx]
        actual_dates = all_dates[actual_start_idx:actual_end_idx]

        sample_data.append((i, input_sequence, input_end_date, actual_values, actual_dates))
    
    # Pre-compute neural predictions and collect ARIMA tasks
    neural_all = []
    for i, input_sequence, input_end_date, actual_values, actual_dates in sample_data:
        neural_pred = model.predict(input_sequence).tolist()
        neural_all.append((i, input_sequence, input_end_date, actual_values, actual_dates, neural_pred))
    
    if compare_arima and not verbose:
        # Silent - let watchlist see final DONE message
        pass
    elif verbose and compare_arima:
        print(f"  ✓ Neural predictions complete ({len(neural_all)} samples)")
        print(f"  → Starting ARIMA forecasts in parallel (4 cores)...\n")
    
    # Collect ARIMA forecast parameters for parallel execution
    arima_forecast_params = []
    if compare_arima:
        for i, input_sequence, input_end_date, actual_values, actual_dates, neural_pred in neural_all:
            arima_forecast_params.append((
                i,
                symbol,
                signal_type,
                input_end_date.strftime("%Y-%m-%d"),
                model_forecast_horizon
            ))
    
    # Execute ARIMA forecasts in parallel (4 cores)
    arima_results = {}  # Maps i to arima_pred
    if compare_arima and arima_forecast_params:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}
            for task_idx, (i, sym, sig_type, end_date_str, fh) in enumerate(arima_forecast_params):
                future = executor.submit(
                    _run_arima_forecast_worker,
                    sym, sig_type, end_date_str, fh
                )
                futures[future] = i
            
            for future in as_completed(futures):
                i = futures[future]
                try:
                    arima_pred = future.result().tolist()
                    arima_results[i] = arima_pred
                except Exception as e:
                    if verbose:
                        # Find the sample number for error reporting
                        sample_num = next(idx for idx, (si, _, _, _, _) in enumerate(sample_data) if si == i) + 1
                        print(f"  ARIMA failed for sample {sample_num}: {e}")
                    arima_results[i] = [None] * model_forecast_horizon
    
    # Process results
    for i, input_sequence, input_end_date, actual_values, actual_dates, neural_pred in neural_all:

        # Get ARIMA result for this sample (from parallel execution)
        arima_pred = arima_results.get(i) if compare_arima else None
        
        # Slice predictions and actuals to evaluation forecast horizon
        neural_pred = neural_pred[:eval_forcast_horizon]
        actual_values = actual_values[:eval_forcast_horizon]
        actual_dates = actual_dates[:eval_forcast_horizon]
        if arima_pred is not None:
            arima_pred = arima_pred[:eval_forcast_horizon]
        
        # Ensure same length
        min_len = min(len(neural_pred), len(actual_values))
        neural_pred = neural_pred[:min_len]
        actual_values = actual_values[:min_len]
        if arima_pred is not None:
            arima_pred = arima_pred[:min_len]

        neural_predictions.append(neural_pred)
        if arima_pred is not None:
            arima_predictions.append(arima_pred)
        actuals.append(actual_values)
        input_end_dates.append(input_end_date)
        input_baselines.append(float(input_sequence[-1]))  # Store baseline for DA calculation

        # Print sample
        sample_num = len(neural_predictions)  # Current sample count
        if verbose:
            print(f"Sample {sample_num}: Input ends {input_end_date.strftime('%Y-%m-%d')}")
            print(f"  Last input value: {input_sequence[-1]:.4f}")

            if compare_arima and arima_pred is not None:
                # Side-by-side output
                header = f"  {'Day':<22} {'Actual':>10} {'Neural':>10} {'Err':>9} {'ARIMA':>10} {'Err':>9} {'Winner':>8}"
                print(header)
                print(f"  {'-'*len(header.strip())}")

                for j, date in enumerate(actual_dates[:min_len]):
                    act = actual_values[j]
                    n_pred = neural_pred[j]
                    a_pred = arima_pred[j]
                    n_err = abs(n_pred - act)
                    a_err = abs(a_pred - act) if a_pred is not None else float('inf')

                    if a_pred is not None:
                        winner = "Neural" if n_err < a_err else ("ARIMA" if a_err < n_err else "Tie")
                    else:
                        winner = "Neural"

                    a_pred_str = f"{a_pred:10.4f}" if a_pred is not None else "     N/A  "
                    a_err_str = f"{a_err:+9.4f}" if a_pred is not None else "     N/A "

                    print(f"  Day {j+1} ({date.strftime('%Y-%m-%d')}) {act:10.4f} {n_pred:10.4f} {n_err:+9.4f} {a_pred_str} {a_err_str} {winner:>8}")
            else:
                # Original neural-only output
                for j, (pred, act, date) in enumerate(zip(neural_pred, actual_values, actual_dates)):
                    error = pred - act
                    pct_error = abs(error / act) * 100 if act != 0 else 0
                    direction_match = "✓" if (pred > input_sequence[-1]) == (act > input_sequence[-1]) else "✗"
                    print(f"  Day {j+1} ({date.strftime('%Y-%m-%d')}): Pred={pred:8.4f}, Actual={act:8.4f}, "
                          f"Error={error:+.4f} ({pct_error:.1f}%) {direction_match}")
            print()

    # Calculate metrics for neural (using evaluation forecast horizon)
    neural_metrics = _compute_metrics(
        all_values, neural_predictions, actuals,
        num_samples, eval_forcast_horizon, seq_length,
        full_model_horizon=model_forecast_horizon
    )

    # Calculate per-day metrics if breakdown requested
    per_day_metrics = None
    if breakdown_by_day:
        per_day_metrics = _compute_per_day_metrics(
            all_values, neural_predictions, actuals,
            num_samples, eval_forcast_horizon, seq_length,
            full_model_horizon=model_forecast_horizon
        )

    # Calculate metrics for ARIMA if comparing
    arima_metrics = None
    if compare_arima and arima_predictions:
        # Filter out samples where ARIMA had None values
        valid_arima = []
        valid_actuals_for_arima = []
        for a_pred, act in zip(arima_predictions, actuals):
            if all(v is not None for v in a_pred):
                valid_arima.append(a_pred)
                valid_actuals_for_arima.append(act)

        if valid_arima:
            arima_metrics = _compute_metrics(
                all_values, valid_arima, valid_actuals_for_arima,
                num_samples, eval_forcast_horizon, seq_length,
                full_model_horizon=model_forecast_horizon
            )

    # Print summary
    if verbose:
        print("=" * 90)
        if compare_arima and arima_metrics:
            print("COMPARISON SUMMARY")
            print("=" * 90)
            print(f"Symbol:              {symbol}")
            print(f"Architecture:        {architecture}")
            print(f"Signal Type:         {signal_label}")
            print(f"Normalization:       {normalization_type}")
            print(f"Forecast Horizon:    {model_forecast_horizon} days", end="")
            if eval_forcast_horizon < model_forecast_horizon:
                print(f" (evaluating on {eval_forcast_horizon} days)")
            else:
                print()
            print(f"Input Days:          {seq_length} days")
            print(f"Samples Evaluated:   {num_samples}")
            print()
            print(f"{'Metric':<35} {'Neural (' + architecture + ')':>20} {'ARIMA':>20} {'Winner':>10}")
            print(f"{'-'*85}")

            comparisons = [
                ("MAE  (Mean Absolute Error)", "mae", False),
                ("RMSE (Root Mean Square Error)", "rmse", False),
                ("MAPE (Mean Absolute % Error)", "mape", False),
                ("Directional Accuracy", "directional_accuracy", True),
            ]

            neural_wins = 0
            arima_wins = 0

            for label, key, higher_is_better in comparisons:
                n_val = neural_metrics[key]
                a_val = arima_metrics[key]

                if higher_is_better:
                    winner = "Neural" if n_val > a_val else ("ARIMA" if a_val > n_val else "Tie")
                else:
                    winner = "Neural" if n_val < a_val else ("ARIMA" if a_val < n_val else "Tie")

                if winner == "Neural":
                    neural_wins += 1
                elif winner == "ARIMA":
                    arima_wins += 1

                if key == "directional_accuracy":
                    n_str = f"{n_val:.2%}"
                    a_str = f"{a_val:.2%}"
                elif key == "mape":
                    n_str = f"{n_val:.2f}%"
                    a_str = f"{a_val:.2f}%"
                else:
                    n_str = f"{n_val:.6f}"
                    a_str = f"{a_val:.6f}"

                print(f"{label:<35} {n_str:>20} {a_str:>20} {winner:>10}")

            print(f"{'-'*85}")
            overall = "Neural" if neural_wins > arima_wins else ("ARIMA" if arima_wins > neural_wins else "Tie")
            print(f"{'Overall Winner':<35} {'':>20} {'':>20} {overall:>10}")
            print(f"  (Neural: {neural_wins} wins, ARIMA: {arima_wins} wins)")
        else:
            print("EVALUATION SUMMARY")
            print("=" * 90)
            print(f"Symbol:              {symbol}")
            print(f"Architecture:        {architecture}")
            print(f"Signal Type:         {signal_label}")
            print(f"Normalization:       {normalization_type}")
            print(f"Forecast Horizon:    {model_forecast_horizon} days", end="")
            if eval_forcast_horizon < model_forecast_horizon:
                print(f" (evaluating on {eval_forcast_horizon} days)")
            else:
                print()
            print(f"Input Days:          {seq_length} days")
            print(f"Inference Engine:    {engine_type.upper()}")
            print(f"Samples Evaluated:   {num_samples}")
            print(f"Total Predictions:   {neural_metrics['total_predictions']}")
            print()
            print(f"MAE  (Mean Absolute Error):    {neural_metrics['mae']:.6f}")
            print(f"RMSE (Root Mean Square Error): {neural_metrics['rmse']:.6f}")
            print(f"MAPE (Mean Absolute % Error):  {neural_metrics['mape']:.2f}%")
            print(f"Directional Accuracy:          {neural_metrics['directional_accuracy']:.2%}")
        
        # Print per-day breakdown if requested
        if breakdown_by_day and per_day_metrics:
            print()
            print("PER-DAY METRICS BREAKDOWN")
            print("-" * 90)
            print(f"{'Day':>5} {'DA':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>10} {'Samples':>10}")
            print(f"{'-'*65}")
            for day in sorted(per_day_metrics.keys()):
                metrics = per_day_metrics[day]
                print(f"{day:>5} {metrics['directional_accuracy']:>7.1%} {metrics['mae']:>12.6f} "
                      f"{metrics['rmse']:>12.6f} {metrics['mape']:>9.2f}% {metrics['total']:>10}")
        
        print("=" * 90)

    result = {
        "symbol": symbol,
        "signal_type": signal_type,
        "architecture": architecture,
        "engine": engine_type,
        "normalization_type": normalization_type,
        "samples": num_samples,
        "total_predictions": neural_metrics["total_predictions"],
        "neural_metrics": neural_metrics,
        "predictions": neural_predictions,
        "actuals": actuals,
        "input_baselines": input_baselines  # Store baselines for DA calculation during aggregation
    }

    if per_day_metrics:
        result["per_day_metrics"] = per_day_metrics

    if compare_arima and arima_metrics:
        result["arima_metrics"] = arima_metrics
        result["arima_predictions"] = arima_predictions

    return result


def run_watchlist_evaluation(
    watchlist_name: str,
    signal_type: str,
    architecture: str,
    input_days: int,
    num_samples: int,
    forecast_horizon: int = None,
    inference_forcast_horizon: int = None,
    compare_arima: bool = False,
    exclude_list: List[str] = None,
    breakdown_by_day: bool = False
) -> Dict[str, Any]:
    """Run model evaluation for an entire watchlist."""
    from db_utils import get_watchlist_symbols

    try:
        symbols = sorted(get_watchlist_symbols(watchlist_name))
    except ValueError as e:
        return {"error": str(e)}

    if not symbols:
        return {"error": f"Watchlist '{watchlist_name}' is empty."}

    # Filter symbols based on exclude_list
    if exclude_list:
        exclude_set = {s.upper().strip() for s in exclude_list}
        symbols = [s for s in symbols if s.upper() not in exclude_set]
        if not symbols:
            return {"error": "All symbols in watchlist were excluded."}

    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    
    # Load model once for reuse
    cached_model = load_model(architecture, signal_type)
    
    print(f"\n{'='*90}")
    print(f"EVALUATING WATCHLIST: {watchlist_name} ({len(symbols)} symbols)")
    print(f"Signal Type:  {signal_label}")
    print(f"Architecture: {architecture}")
    print(f"Samples:      {num_samples} per symbol")
    if exclude_list:
        print(f"Excluded:     {', '.join(exclude_list)}")
    if compare_arima:
        print(f"Mode:         Neural vs ARIMA Comparison")
    print(f"{'='*90}\n")

    all_res = []
    for i, symbol in enumerate(symbols):
        print(f"[{i+1}/{len(symbols)}] Evaluating {symbol:<8}...", end="", flush=True)
        res = run_evaluation(
            symbol=symbol,
            signal_type=signal_type,
            architecture=architecture,
            input_days=input_days,
            num_samples=num_samples,
            forecast_horizon=forecast_horizon,
            inference_forcast_horizon=inference_forcast_horizon,
            compare_arima=compare_arima,
            verbose=False,
            cached_model=cached_model,
            breakdown_by_day=breakdown_by_day  # Compute per-day metrics for later aggregation
        )
        if "error" not in res:
            all_res.append(res)
            n_da = res['neural_metrics']['directional_accuracy']
            n_mae = res['neural_metrics']['mae']
            n_rmse = res['neural_metrics']['rmse']
            if compare_arima and 'arima_metrics' in res:
                a_da = res['arima_metrics']['directional_accuracy']
                a_mae = res['arima_metrics']['mae']
                a_rmse = res['arima_metrics']['rmse']
                print(f" DONE (Neural: DA {n_da:.1%}, MAE {n_mae:.4f}, RMSE {n_rmse:.4f} | ARIMA: DA {a_da:.1%}, MAE {a_mae:.4f}, RMSE {a_rmse:.4f})")
            else:
                print(f" DONE (DA: {n_da:.1%}, MAE: {n_mae:.4f}, RMSE: {n_rmse:.4f})")
        else:
            print(f" FAILED: {res['error']}")

    if not all_res:
        return {"error": "No symbols could be evaluated."}

    # Aggregate Neural Metrics
    neural_mae = np.mean([r["neural_metrics"]["mae"] for r in all_res])
    neural_rmse = np.mean([r["neural_metrics"]["rmse"] for r in all_res])
    neural_mape = np.mean([r["neural_metrics"]["mape"] for r in all_res])
    neural_total_correct = sum([r["neural_metrics"]["correct_directions"] for r in all_res])
    neural_total_dirs = sum([r["neural_metrics"]["total_directions"] for r in all_res])
    neural_da = neural_total_correct / neural_total_dirs if neural_total_dirs > 0 else 0

    # Aggregate ARIMA Metrics (if applicable)
    arima_metrics = None
    if compare_arima:
        valid_arima_res = [r for r in all_res if "arima_metrics" in r]
        if valid_arima_res:
            arima_mae = np.mean([r["arima_metrics"]["mae"] for r in valid_arima_res])
            arima_rmse = np.mean([r["arima_metrics"]["rmse"] for r in valid_arima_res])
            arima_mape = np.mean([r["arima_metrics"]["mape"] for r in valid_arima_res])
            arima_total_correct = sum([r["arima_metrics"]["correct_directions"] for r in valid_arima_res])
            arima_total_dirs = sum([r["arima_metrics"]["total_directions"] for r in valid_arima_res])
            arima_da = arima_total_correct / arima_total_dirs if arima_total_dirs > 0 else 0
            arima_metrics = {
                "mae": arima_mae,
                "rmse": arima_rmse,
                "mape": arima_mape,
                "directional_accuracy": arima_da
            }

    # Determine effective forecast horizons for display and get normalization type
    # Load model to get defaults if not specified
    normalization_type = "unknown"
    if forecast_horizon is None or inference_forcast_horizon is None:
        _, temp_model, normalization_type = load_model(architecture, signal_type)
        model_default_fh = temp_model.forecast_horizon if temp_model else 5
    else:
        model_default_fh = forecast_horizon
        # Still need to get normalization_type
        _, _, normalization_type = load_model(architecture, signal_type)
    
    display_fh = forecast_horizon if forecast_horizon is not None else model_default_fh
    display_eval_fh = inference_forcast_horizon if inference_forcast_horizon is not None else display_fh
    
    # Print Summary
    print("\n" + "=" * 90)
    if compare_arima and arima_metrics:
        print(f"WATCHLIST COMPARISON SUMMARY: {watchlist_name}")
        print("=" * 90)
        print(f"Symbols Evaluated:   {len(all_res)} / {len(symbols)}")
        print(f"Architecture:        {architecture}")
        print(f"Signal Type:         {signal_label}")
        print(f"Normalization:       {normalization_type}")
        print(f"Forecast Horizon:    {display_fh} days", end="")
        if display_eval_fh < display_fh:
            print(f" (evaluating on {display_eval_fh} days)")
        else:
            print()
        print(f"Input Days:          {input_days} days")
        print(f"Samples/Symbol:      {num_samples}")
        print()
        print(f"{'Metric':<35} {'Neural (' + architecture + ')':>20} {'ARIMA':>20} {'Winner':>10}")
        print(f"{'-'*85}")

        comparisons = [
            ("MAE  (Mean Absolute Error)", "mae", False),
            ("RMSE (Root Mean Square Error)", "rmse", False),
            ("MAPE (Mean Absolute % Error)", "mape", False),
            ("Directional Accuracy", "directional_accuracy", True),
        ]

        neural_wins = 0
        arima_wins = 0

        for label, key, higher_is_better in comparisons:
            n_val = neural_mae if key == "mae" else neural_rmse if key == "rmse" else neural_mape if key == "mape" else neural_da
            a_val = arima_metrics[key]

            if higher_is_better:
                winner = "Neural" if n_val > a_val else ("ARIMA" if a_val > n_val else "Tie")
            else:
                winner = "Neural" if n_val < a_val else ("ARIMA" if a_val < n_val else "Tie")

            if winner == "Neural":
                neural_wins += 1
            elif winner == "ARIMA":
                arima_wins += 1

            if key == "directional_accuracy":
                n_str = f"{n_val:.2%}"
                a_str = f"{a_val:.2%}"
            elif key == "mape":
                n_str = f"{n_val:.2f}%"
                a_str = f"{a_val:.2f}%"
            else:
                n_str = f"{n_val:.6f}"
                a_str = f"{a_val:.6f}"

            print(f"{label:<35} {n_str:>20} {a_str:>20} {winner:>10}")

        print(f"{'-'*85}")
        overall = "Neural" if neural_wins > arima_wins else ("ARIMA" if arima_wins > neural_wins else "Tie")
        print(f"{'Overall Winner':<35} {'':>20} {'':>20} {overall:>10}")
    else:
        print(f"WATCHLIST EVALUATION SUMMARY: {watchlist_name}")
        print("=" * 90)
        print(f"Symbols Evaluated:   {len(all_res)} / {len(symbols)}")
        print(f"Architecture:        {architecture}")
        print(f"Signal Type:         {signal_label}")
        print(f"Normalization:       {normalization_type}")
        print(f"Forecast Horizon:    {display_fh} days", end="")
        if display_eval_fh < display_fh:
            print(f" (evaluating on {display_eval_fh} days)")
        else:
            print()
        print(f"Input Days:          {input_days} days")
        print(f"Samples/Symbol:      {num_samples}")
        print(f"Total Predictions:   {neural_total_dirs}")
        print()
        print(f"MAE  (Mean Absolute Error):    {neural_mae:.6f}")
        print(f"RMSE (Root Mean Square Error): {neural_rmse:.6f}")
        print(f"MAPE (Mean Absolute % Error):  {neural_mape:.2f}%")
        print(f"Directional Accuracy:          {neural_da:.2%}")
    
    # Aggregate per-day metrics if requested
    aggregated_per_day = None
    if breakdown_by_day:
        # Collect all predictions, actuals, and baselines to compute per-day metrics once
        all_predictions_combined = []
        all_actuals_combined = []
        all_baselines_combined = []
        
        for res in all_res:
            if "predictions" in res and "actuals" in res and "input_baselines" in res:
                all_predictions_combined.extend(res["predictions"])
                all_actuals_combined.extend(res["actuals"])
                all_baselines_combined.extend(res["input_baselines"])
        
        if all_predictions_combined:
            # Compute per-day metrics on combined data
            aggregated_per_day = {}
            # Determine the max number of days across all predictions
            max_days = max(len(p) for p in all_predictions_combined) if all_predictions_combined else 0
            
            for day_idx in range(max_days):
                day_num = day_idx + 1
                day_preds = []
                day_actuals = []
                day_correct_dirs = 0
                day_total_dirs = 0
                
                for pred_seq, act_seq, baseline in zip(all_predictions_combined, all_actuals_combined, all_baselines_combined):
                    if day_idx < len(pred_seq) and day_idx < len(act_seq):
                        day_preds.append(pred_seq[day_idx])
                        day_actuals.append(act_seq[day_idx])
                        
                        # Directional accuracy: compare to the input baseline
                        if (pred_seq[day_idx] > baseline) == (act_seq[day_idx] > baseline):
                            day_correct_dirs += 1
                        day_total_dirs += 1
                
                if day_preds:
                    day_preds = np.array(day_preds)
                    day_actuals = np.array(day_actuals)
                    
                    mae = float(np.mean(np.abs(day_preds - day_actuals)))
                    rmse = float(np.sqrt(np.mean((day_preds - day_actuals) ** 2)))
                    mape = float(np.mean(np.abs((day_preds - day_actuals) / (day_actuals + 1e-8))) * 100)
                    da = day_correct_dirs / day_total_dirs if day_total_dirs > 0 else 0.0
                    
                    aggregated_per_day[day_num] = {
                        "mae": mae,
                        "rmse": rmse,
                        "mape": mape,
                        "directional_accuracy": da,
                        "total": len(day_preds)
                    }
            
            # Print per-day breakdown
            print()
            print("PER-DAY METRICS BREAKDOWN (across all symbols)")
            print("-" * 90)
            print(f"{'Day':>5} {'DA':>8} {'MAE':>12} {'RMSE':>12} {'MAPE':>10}")
            print(f"{'-'*57}")
            for day in sorted(aggregated_per_day.keys()):
                metrics = aggregated_per_day[day]
                print(f"{day:>5} {metrics['directional_accuracy']:>7.1%} {metrics['mae']:>12.6f} "
                      f"{metrics['rmse']:>12.6f} {metrics['mape']:>9.2f}%")
    
    print("=" * 90)

    result = {
        "watchlist": watchlist_name,
        "symbols_evaluated": len(all_res),
        "neural_metrics": {
            "mae": neural_mae,
            "rmse": neural_rmse,
            "mape": neural_mape,
            "directional_accuracy": neural_da
        },
        "arima_metrics": arima_metrics
    }
    
    if aggregated_per_day:
        result["per_day_metrics"] = aggregated_per_day
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MACD/Signal Line Forecaster models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate bidirectional GRU on AAPL MACD
  python evaluate_forecast_model.py --symbol AAPL --architecture bidirectional_gru

  # Evaluate on an entire watchlist
  python evaluate_forecast_model.py --watchlist sp500 --architecture bidirectional_gru

  # Compare neural model vs ARIMA on a watchlist
  python evaluate_forecast_model.py --watchlist sp500 --compare

  # Compare with more samples for better statistics
  python evaluate_forecast_model.py --symbol MSFT --compare --samples 30

  # Evaluate on signal line with more samples
  python evaluate_forecast_model.py --symbol MSFT --signal-type signal_line --samples 30
        """
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Stock symbol to evaluate"
    )
    parser.add_argument(
        "--watchlist",
        type=str,
        help="Watchlist name to evaluate"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        help="Comma-separated list of symbols to exclude from evaluation"
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        choices=["macd", "signal_line"],
        default="macd",
        help="Type of signal to evaluate: 'macd' or 'signal_line' (default: macd)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["stacked_lstm", "bidirectional_gru", "stacked_gru", "standard_lstm", "gru"],
        default="bidirectional_gru",
        help="Model architecture (default: bidirectional_gru)"
    )
    parser.add_argument(
        "--input-days",
        type=int,
        default=30,
        help="Number of input days for context (default: 30)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Number of prediction samples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=None,
        help="Number of days to forecast (default: uses model's trained forecast horizon)"
    )
    parser.add_argument(
        "--inference-forcast-horizon",
        type=int,
        default=None,
        help="Number of forecasted days to use for evaluation (default: same as forecast-horizon, must be <= forecast-horizon)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare neural model predictions against ARIMA on the same data"
    )
    parser.add_argument(
        "--breakdown-by-day",
        action="store_true",
        help="Show per-day metrics breakdown (DA, MAE, RMSE, MAPE for each forecast day)"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available trained models"
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        models_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models"
        )
        print("\nAvailable trained models:")
        print("-" * 50)
        
        if os.path.exists(models_dir):
            files = os.listdir(models_dir)
            coreml_models = [f for f in files if f.endswith(".mlpackage")]
            pytorch_models = [f for f in files if f.endswith(".pt")]
            
            if coreml_models:
                print("\nCore ML models (NPU inference):")
                for m in sorted(coreml_models):
                    print(f"  - {m}")
            
            if pytorch_models:
                print("\nPyTorch models:")
                for m in sorted(pytorch_models):
                    print(f"  - {m}")
            
            if not coreml_models and not pytorch_models:
                print("  No trained models found.")
                print("  Run train_forecast_model.py to train a model.")
        else:
            print(f"  Models directory not found: {models_dir}")
        
        print()
        return

    # Check that either symbol or watchlist is provided
    if not args.symbol and not args.watchlist:
        args.symbol = "MSFT" # Default back to MSFT if none provided
    
    exclude_list = [s.strip().upper() for s in args.exclude.split(",")] if args.exclude else []

    if args.watchlist:
        results = run_watchlist_evaluation(
            watchlist_name=args.watchlist,
            signal_type=args.signal_type,
            architecture=args.architecture,
            input_days=args.input_days,
            num_samples=args.samples,
            forecast_horizon=args.forecast_horizon,
            inference_forcast_horizon=args.inference_forcast_horizon,
            compare_arima=args.compare,
            exclude_list=exclude_list,
            breakdown_by_day=args.breakdown_by_day
        )
    else:
        # Run evaluation for single symbol
        mode_label = "Neural vs ARIMA Comparison" if args.compare else "MACD Forecaster Model Evaluation"
        print(f"\n{'='*70}")
        print(mode_label)
        print(f"{'='*70}")
        print(f"Symbol:       {args.symbol.upper()}")
        print(f"Signal Type:  {args.signal_type}")
        print(f"Architecture: {args.architecture}")
        print(f"Input Days:   {args.input_days}")
        print(f"Samples:      {args.samples}")
        if args.compare:
            print(f"Compare:      Neural ({args.architecture}) vs ARIMA")

        results = run_evaluation(
            symbol=args.symbol.upper(),
            signal_type=args.signal_type,
            architecture=args.architecture,
            input_days=args.input_days,
            num_samples=args.samples,
            forecast_horizon=args.forecast_horizon,
            inference_forcast_horizon=args.inference_forcast_horizon,
            compare_arima=args.compare,
            verbose=True,
            breakdown_by_day=args.breakdown_by_day
        )
    
    if "error" in results:
        print(f"\nError: {results['error']}")
        if "searched_paths" in results:
            print("Searched paths:")
            for engine, path in results["searched_paths"].items():
                print(f"  - {engine}: {path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
