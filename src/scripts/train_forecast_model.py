#!/usr/bin/env python3
"""
Training script for MACD Forecaster models.

This script trains various neural network architectures on historical MACD data
and exports them to Core ML format for NPU inference on Apple Silicon.

Supported architectures:
    - stacked_lstm: 2-layer LSTM (default)
    - bidirectional_gru: Bidirectional GRU (best in research)
    - stacked_gru: 2-layer GRU
    - standard_lstm: Single-layer LSTM
    - gru: Single-layer GRU

Usage:
    python train_forecast_model.py [--architecture ARCH] [--symbols SYMBOLS] [--epochs EPOCHS]
    
Example:
    python train_forecast_model.py --architecture bidirectional_gru --epochs 100
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def get_training_data(symbols: list, days: int = 365, signal_type: str = "macd") -> np.ndarray:
    """
    Gather MACD or Signal Line data for training from multiple symbols.
    
    Args:
        symbols: List of stock symbols
        days: Number of days of history to use
        signal_type: Type of signal to gather - "macd" or "signal_line"
        
    Returns:
        Concatenated values from all symbols
    """
    from macd_utils import get_macd_for_range, get_latest_market_date
    from db_utils import get_watchlist_symbols
    
    all_values = []
    
    end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=days)
    
    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    print(f"Gathering {signal_label} data from {start_date} to {end_date}")
    print(f"Processing {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        try:
            macd_data = get_macd_for_range(symbol, start_date, end_date)
            if signal_type == "macd":
                series = [
                    d["macd"] for d in macd_data 
                    if "macd" in d and d["macd"] is not None
                ]
            else:  # signal_line
                series = [
                    d["signal_line"] for d in macd_data 
                    if "signal_line" in d and d["signal_line"] is not None
                ]
            if len(series) >= 50:  # Need enough data
                all_values.extend(series)
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(series)} data points")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: Error - {e}")
    
    print(f"\nTotal {signal_label} data points: {len(all_values)}")
    return np.array(all_values, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description="Train LSTM MACD Forecaster")
    parser.add_argument(
        "--symbols", 
        type=str, 
        default=None,
        help="Comma-separated list of symbols to train on"
    )
    parser.add_argument(
        "--watchlist",
        type=str,
        default="sp500",
        help="Watchlist name to use for training data (default: sp500)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data to use (default: 365)"
    )
    parser.add_argument(
        "--signal-type",
        type=str,
        choices=["macd", "signal_line"],
        default="macd",
        help="Type of signal to train on: 'macd' or 'signal_line' (default: macd)"
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=["stacked_lstm", "bidirectional_gru", "stacked_gru", "standard_lstm", "gru"],
        default="stacked_lstm",
        help="Model architecture: stacked_lstm, bidirectional_gru, stacked_gru, standard_lstm, gru (default: stacked_lstm)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=30,
        help="Input sequence length (default: 30)"
    )
    parser.add_argument(
        "--forecast-horizon",
        type=int,
        default=5,
        help="Number of days to forecast (default: 5)"
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=64,
        help="LSTM hidden size (default: 64)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for models"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Fraction of data to hold out for testing (default: 0.15)"
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Skip Core ML conversion (useful for testing on non-Mac)"
    )
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "models"
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model name based on signal type and architecture
    signal_label = "MACD" if args.signal_type == "macd" else "Signal Line"
    arch_label = args.architecture.replace("_", " ").title()
    model_name = f"{args.signal_type}_{args.architecture}"  # e.g., "macd_bidirectional_gru"
    
    print("=" * 60)
    print(f"{arch_label} {signal_label} Forecaster Training")
    print("=" * 60)
    print(f"Architecture: {args.architecture}")
    print(f"Signal type: {signal_label}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Forecast horizon: {args.forecast_horizon}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Test split: {args.test_split * 100:.0f}%")
    print()
    
    # Get symbols
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",")]
    else:
        from db_utils import get_watchlist_symbols
        symbols = get_watchlist_symbols(args.watchlist)
        if not symbols:
            print(f"Error: Watchlist '{args.watchlist}' not found or empty")
            print("Use --symbols to specify symbols manually")
            sys.exit(1)
    
    print(f"Using {len(symbols)} symbols for training")
    print()
    
    # Gather training data
    print("Gathering training data...")
    all_data = get_training_data(symbols, args.days, args.signal_type)
    
    if len(all_data) < 1000:
        print(f"Warning: Only {len(all_data)} data points. Need at least 1000 for good training.")
        if len(all_data) < 100:
            print("Error: Not enough data for training")
            sys.exit(1)
    
    # Split into train and test sets
    test_size = int(len(all_data) * args.test_split)
    train_data = all_data[:-test_size] if test_size > 0 else all_data
    test_data = all_data[-test_size:] if test_size > 0 else None
    
    print(f"\nData split:")
    print(f"  Total: {len(all_data)} data points")
    print(f"  Train: {len(train_data)} data points ({100 - args.test_split * 100:.0f}%)")
    if test_data is not None:
        print(f"  Test:  {len(test_data)} data points ({args.test_split * 100:.0f}%)")
    
    # Check for PyTorch
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        
        if torch.backends.mps.is_available():
            print("Using Apple Metal GPU (MPS) for training")
        elif torch.cuda.is_available():
            print("Using NVIDIA GPU (CUDA) for training")
        else:
            print("Using CPU for training")
    except ImportError:
        print("Error: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)
    
    # Create and train model
    print("\nInitializing model...")
    from models.lstm_forecaster import MACDForecasterTrainer
    
    trainer = MACDForecasterTrainer(
        seq_length=args.seq_length,
        forecast_horizon=args.forecast_horizon,
        hidden_size=args.hidden_size,
        learning_rate=args.learning_rate,
        architecture=args.architecture
    )
    
    print("\nTraining...")
    history = trainer.train(
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=True
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Evaluate on held-out test set
    if test_data is not None and len(test_data) > args.seq_length + args.forecast_horizon:
        print("\nEvaluating on held-out test set...")
        test_metrics = trainer.evaluate(test_data, verbose=True)
    else:
        print("\nSkipping test evaluation (not enough test data)")
        test_metrics = None
    
    # Save PyTorch model (filename includes signal type and architecture)
    pytorch_path = os.path.join(output_dir, f"{model_name}_forecaster.pt")
    trainer.save(pytorch_path)
    
    # Export to Core ML
    if not args.skip_coreml:
        try:
            import coremltools
            print(f"\nCore ML Tools version: {coremltools.__version__}")
            
            coreml_path = os.path.join(output_dir, f"{model_name}_forecaster.mlpackage")
            trainer.export_to_coreml(coreml_path)
            
            print("\n" + "=" * 60)
            print("Training complete!")
            print("=" * 60)
            print(f"PyTorch model: {pytorch_path}")
            print(f"Core ML model: {coreml_path}")
            if test_metrics:
                print(f"\nTest MAE: {test_metrics['mae']:.6f}")
                print(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")
            print("\nThe Core ML model will run on Apple Neural Engine (NPU)")
            
        except ImportError:
            print("\nWarning: coremltools not installed. Skipping Core ML export.")
            print("Install with: pip install coremltools")
            print(f"\nPyTorch model saved to: {pytorch_path}")
    else:
        print(f"\nPyTorch model saved to: {pytorch_path}")
        if test_metrics:
            print(f"Test MAE: {test_metrics['mae']:.6f}")
            print(f"Test Directional Accuracy: {test_metrics['directional_accuracy']:.2%}")
        print("Core ML export skipped (--skip-coreml)")


if __name__ == "__main__":
    main()
