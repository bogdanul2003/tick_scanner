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
        days_back: Number of days to look back
        
    Returns:
        Tuple of (values list, dates list)
    """
    from macd_utils import get_macd_for_range
    
    start_date = end_date - timedelta(days=days_back)
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
        end_date=end_date
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
        Loaded CoreMLForecaster or PyTorch model trainer
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
                return ("coreml", forecaster)
        except Exception as e:
            print(f"Warning: Could not load Core ML model: {e}")
    
    if os.path.exists(pytorch_path):
        try:
            import torch
            from models.lstm_forecaster import MACDForecasterTrainer
            
            # Load checkpoint to get architecture info
            checkpoint = torch.load(pytorch_path, map_location="cpu")
            
            trainer = MACDForecasterTrainer(
                seq_length=checkpoint.get("seq_length", 30),
                forecast_horizon=checkpoint.get("forecast_horizon", 5),
                hidden_size=checkpoint.get("hidden_size", 64),
                architecture=checkpoint.get("architecture", architecture)
            )
            trainer.load(pytorch_path)
            return ("pytorch", trainer)
        except Exception as e:
            print(f"Warning: Could not load PyTorch model: {e}")
    
    return (None, None)


def _compute_metrics(
    all_values: List[float],
    predictions: List[List[float]],
    actuals: List[List[float]],
    num_samples: int,
    model_forecast_horizon: int,
    seq_length: int
) -> Dict[str, float]:
    """Compute MAE, RMSE, MAPE, and directional accuracy for a set of predictions."""
    all_pred = np.array([p for pred in predictions for p in pred])
    all_actual = np.array([a for act in actuals for a in act])

    mae = float(np.mean(np.abs(all_pred - all_actual)))
    rmse = float(np.sqrt(np.mean((all_pred - all_actual) ** 2)))
    mape = float(np.mean(np.abs((all_pred - all_actual) / (all_actual + 1e-8))) * 100)

    correct_directions = 0
    total_directions = 0
    for i, (pred, act) in enumerate(zip(predictions, actuals)):
        start_idx = len(all_values) - num_samples - model_forecast_horizon + i - seq_length
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
        "total_predictions": len(all_pred)
    }


def run_evaluation(
    symbol: str,
    signal_type: str,
    architecture: str,
    input_days: int,
    num_samples: int,
    forecast_horizon: int = 5,
    compare_arima: bool = False
) -> Dict[str, Any]:
    """
    Run model evaluation comparing predictions to actual values.

    Args:
        symbol: Stock symbol
        signal_type: 'macd' or 'signal_line'
        architecture: Model architecture
        input_days: Number of days of input data
        num_samples: Number of prediction samples to evaluate
        forecast_horizon: Number of days to forecast
        compare_arima: If True, also run ARIMA on each sample and compare

    Returns:
        Evaluation results dictionary
    """
    from macd_utils import get_latest_market_date

    # Load model
    engine_type, model = load_model(architecture, signal_type)

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
    model_forecast_horizon = model.forecast_horizon

    # Get historical data
    end_date = get_latest_market_date()
    total_days_needed = input_days + num_samples + forecast_horizon + seq_length

    all_values, all_dates = get_historical_data(
        symbol, signal_type, end_date, total_days_needed
    )

    if len(all_values) < seq_length + forecast_horizon + num_samples:
        return {
            "error": f"Not enough data. Have {len(all_values)} points, need at least {seq_length + forecast_horizon + num_samples}"
        }

    # Run predictions
    neural_predictions = []
    arima_predictions = []
    actuals = []
    input_end_dates = []

    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    mode_label = "COMPARISON: Neural vs ARIMA" if compare_arima else f"Neural ({engine_type})"

    print(f"\n{'='*90}")
    print(f"Running {num_samples} predictions on {symbol} {signal_label}")
    print(f"Mode: {mode_label}")
    print(f"Neural Model: {architecture} ({engine_type})")
    print(f"Input sequence: {seq_length} days, Forecast: {model_forecast_horizon} days")
    print(f"{'='*90}\n")

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

        # Neural prediction
        neural_pred = model.predict(input_sequence).tolist()

        # ARIMA prediction (if comparing) — uses production flow with dynamic windowing
        arima_pred = None
        if compare_arima:
            try:
                arima_pred = run_arima_forecast(
                    symbol=symbol,
                    signal_type=signal_type,
                    end_date=input_end_date,
                    forecast_horizon=model_forecast_horizon
                ).tolist()
            except Exception as e:
                print(f"  ARIMA failed for sample {i+1}: {e}")
                arima_pred = [None] * model_forecast_horizon

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

        # Print sample
        print(f"Sample {i+1}: Input ends {input_end_date.strftime('%Y-%m-%d')}")
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

    # Calculate metrics for neural
    neural_metrics = _compute_metrics(
        all_values, neural_predictions, actuals,
        num_samples, model_forecast_horizon, seq_length
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
                num_samples, model_forecast_horizon, seq_length
            )

    # Print summary
    print("=" * 90)
    if compare_arima and arima_metrics:
        print("COMPARISON SUMMARY")
        print("=" * 90)
        print(f"Symbol:              {symbol}")
        print(f"Signal Type:         {signal_label}")
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
        print(f"Signal Type:         {signal_label}")
        print(f"Architecture:        {architecture}")
        print(f"Inference Engine:    {engine_type.upper()}")
        print(f"Samples Evaluated:   {num_samples}")
        print(f"Total Predictions:   {neural_metrics['total_predictions']}")
        print()
        print(f"MAE  (Mean Absolute Error):    {neural_metrics['mae']:.6f}")
        print(f"RMSE (Root Mean Square Error): {neural_metrics['rmse']:.6f}")
        print(f"MAPE (Mean Absolute % Error):  {neural_metrics['mape']:.2f}%")
        print(f"Directional Accuracy:          {neural_metrics['directional_accuracy']:.2%}")
    print("=" * 90)

    result = {
        "symbol": symbol,
        "signal_type": signal_type,
        "architecture": architecture,
        "engine": engine_type,
        "samples": num_samples,
        "total_predictions": neural_metrics["total_predictions"],
        "neural_metrics": neural_metrics,
        "predictions": neural_predictions,
        "actuals": actuals
    }

    if compare_arima and arima_metrics:
        result["arima_metrics"] = arima_metrics
        result["arima_predictions"] = arima_predictions

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MACD/Signal Line Forecaster models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate bidirectional GRU on AAPL MACD
  python evaluate_forecast_model.py --symbol AAPL --architecture bidirectional_gru

  # Compare neural model vs ARIMA side-by-side
  python evaluate_forecast_model.py --symbol AAPL --compare

  # Compare with more samples for better statistics
  python evaluate_forecast_model.py --symbol MSFT --compare --samples 30

  # Evaluate on signal line with more samples
  python evaluate_forecast_model.py --symbol MSFT --signal-type signal_line --samples 30

  # Compare multiple architectures
  python evaluate_forecast_model.py --symbol NVDA --architecture stacked_lstm
  python evaluate_forecast_model.py --symbol NVDA --architecture bidirectional_gru
        """
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default="MSFT",
        help="Stock symbol to evaluate (default: MSFT)"
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
        default=5,
        help="Number of days to forecast (default: 5)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare neural model predictions against ARIMA on the same data"
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
    
    # Run evaluation
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
        compare_arima=args.compare
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
