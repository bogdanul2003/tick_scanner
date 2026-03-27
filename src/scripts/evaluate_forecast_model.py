#!/usr/bin/env python3
"""
Evaluation script for MACD/Signal Line Forecaster models.

This script loads a trained model and runs inference on historical data,
comparing predictions to actual values stored in the database.

Usage:
    python evaluate_forecast_model.py [--symbol SYMBOL] [--signal-type TYPE] [--architecture ARCH]
    
Example:
    python evaluate_forecast_model.py --symbol AAPL --architecture bidirectional_gru
    python evaluate_forecast_model.py --symbol MSFT --signal-type signal_line --samples 20
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
            dates.append(d["date"])
    
    return values, dates


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


def run_evaluation(
    symbol: str,
    signal_type: str,
    architecture: str,
    input_days: int,
    num_samples: int,
    forecast_horizon: int = 5
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
    if engine_type == "coreml":
        seq_length = model.seq_length
        model_forecast_horizon = model.forecast_horizon
    else:  # pytorch
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
    predictions = []
    actuals = []
    input_end_dates = []
    
    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    
    print(f"\n{'='*70}")
    print(f"Running {num_samples} predictions on {symbol} {signal_label}")
    print(f"Model: {architecture} ({engine_type})")
    print(f"Input sequence: {seq_length} days, Forecast: {model_forecast_horizon} days")
    print(f"{'='*70}\n")
    
    for i in range(num_samples):
        # Calculate indices
        # Start from enough days back that we have actuals for comparison
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
        
        # Make prediction
        if engine_type == "coreml":
            predicted_values = model.predict(input_sequence).tolist()
        else:  # pytorch
            predicted_values = model.predict(input_sequence).tolist()
        
        # Ensure same length
        min_len = min(len(predicted_values), len(actual_values))
        predicted_values = predicted_values[:min_len]
        actual_values = actual_values[:min_len]
        
        predictions.append(predicted_values)
        actuals.append(actual_values)
        input_end_dates.append(input_end_date)
        
        # Print sample
        print(f"Sample {i+1}: Input ends {input_end_date.strftime('%Y-%m-%d')}")
        print(f"  Last input value: {input_sequence[-1]:.4f}")
        
        for j, (pred, act, date) in enumerate(zip(predicted_values, actual_values, actual_dates)):
            error = pred - act
            pct_error = abs(error / act) * 100 if act != 0 else 0
            direction_match = "✓" if (pred > input_sequence[-1]) == (act > input_sequence[-1]) else "✗"
            print(f"  Day {j+1} ({date.strftime('%Y-%m-%d')}): Pred={pred:8.4f}, Actual={act:8.4f}, "
                  f"Error={error:+.4f} ({pct_error:.1f}%) {direction_match}")
        print()
    
    # Calculate metrics
    all_pred = np.array([p for pred in predictions for p in pred])
    all_actual = np.array([a for act in actuals for a in act])
    
    mae = np.mean(np.abs(all_pred - all_actual))
    rmse = np.sqrt(np.mean((all_pred - all_actual) ** 2))
    mape = np.mean(np.abs((all_pred - all_actual) / (all_actual + 1e-8))) * 100
    
    # Direction accuracy
    correct_directions = 0
    total_directions = 0
    for i, (pred, act, input_date) in enumerate(zip(predictions, actuals, input_end_dates)):
        # Get the last input value for this sample
        start_idx = len(all_values) - num_samples - model_forecast_horizon + i - seq_length
        last_input = all_values[start_idx + seq_length - 1]
        
        for p, a in zip(pred, act):
            pred_up = p > last_input
            actual_up = a > last_input
            if pred_up == actual_up:
                correct_directions += 1
            total_directions += 1
    
    directional_accuracy = correct_directions / total_directions if total_directions > 0 else 0
    
    # Print summary
    print("=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Symbol:              {symbol}")
    print(f"Signal Type:         {signal_label}")
    print(f"Architecture:        {architecture}")
    print(f"Inference Engine:    {engine_type.upper()}")
    print(f"Samples Evaluated:   {num_samples}")
    print(f"Total Predictions:   {len(all_pred)}")
    print()
    print(f"MAE  (Mean Absolute Error):    {mae:.6f}")
    print(f"RMSE (Root Mean Square Error): {rmse:.6f}")
    print(f"MAPE (Mean Absolute % Error):  {mape:.2f}%")
    print(f"Directional Accuracy:          {directional_accuracy:.2%}")
    print("=" * 70)
    
    return {
        "symbol": symbol,
        "signal_type": signal_type,
        "architecture": architecture,
        "engine": engine_type,
        "samples": num_samples,
        "total_predictions": len(all_pred),
        "metrics": {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "directional_accuracy": float(directional_accuracy)
        },
        "predictions": predictions,
        "actuals": actuals
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MACD/Signal Line Forecaster models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate bidirectional GRU on AAPL MACD
  python evaluate_forecast_model.py --symbol AAPL --architecture bidirectional_gru

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
    print(f"\n{'='*70}")
    print("MACD Forecaster Model Evaluation")
    print(f"{'='*70}")
    print(f"Symbol:       {args.symbol.upper()}")
    print(f"Signal Type:  {args.signal_type}")
    print(f"Architecture: {args.architecture}")
    print(f"Input Days:   {args.input_days}")
    print(f"Samples:      {args.samples}")
    
    results = run_evaluation(
        symbol=args.symbol.upper(),
        signal_type=args.signal_type,
        architecture=args.architecture,
        input_days=args.input_days,
        num_samples=args.samples,
        forecast_horizon=args.forecast_horizon
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
