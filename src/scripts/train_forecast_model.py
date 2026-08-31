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


def get_training_data(symbols: list, days: int = 365, signal_type: str = "macd", feature_names: list = None):
    """
    Gather multi-feature data for training from multiple symbols.

    Reads directly from stock_cache (same data the UI uses) without triggering
    any Yahoo Finance fetches. If data is missing from the cache for a symbol,
    that symbol is simply skipped rather than re-fetching live data.

    Args:
        symbols: List of stock symbols
        days: Number of days of history to use
        signal_type: Type of primary signal - "macd" or "signal_line"
        feature_names: List of features to gather (e.g. ['macd', 'delta', 'open', 'close', 'volume'])
        
    Returns:
        List of 2D numpy arrays of shape (points, num_features), one per symbol
    """
    from macd_utils import get_latest_market_date
    from db_utils import fetch_bulk_from_cache
    
    if feature_names is None:
        feature_names = [signal_type]
        
    per_symbol_data = []
    total_points = 0
    
    end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=days)
    
    signal_label = "MACD" if signal_type == "macd" else "Signal Line"
    primary_col = "MACD" if signal_type == "macd" else "Signal_Line"
    print(f"Gathering features ({', '.join(feature_names)}) from {start_date} to {end_date}")
    print(f"Processing {len(symbols)} symbols...")
    
    for i, symbol in enumerate(symbols):
        try:
            bulk = fetch_bulk_from_cache([symbol], start_date, end_date)
            cached_df = bulk.get(symbol)
            if cached_df is None or cached_df.empty:
                continue
                
            cols_data = []
            has_all_cols = True
            for f in feature_names:
                f_lower = f.lower()
                if f_lower in ("macd", "signal_line"):
                    col = "MACD" if f_lower == "macd" else "Signal_Line"
                    if col in cached_df.columns:
                        cols_data.append(cached_df[col].values)
                    else:
                        has_all_cols = False
                        break
                elif f_lower == "delta":
                    if primary_col in cached_df.columns:
                        p_vals = cached_df[primary_col].values.astype(np.float32)
                        deltas = np.zeros(len(cached_df), dtype=np.float32)
                        deltas[1:] = p_vals[1:] - p_vals[:-1]
                        cols_data.append(deltas)
                    else:
                        has_all_cols = False
                        break
                elif f_lower == "open-close":
                    if "Open" in cached_df.columns and "Close" in cached_df.columns:
                        o_vals = cached_df["Open"].values.astype(np.float32)
                        c_vals = cached_df["Close"].values.astype(np.float32)
                        cols_data.append(np.where(o_vals != 0, (c_vals - o_vals) / o_vals, 0.0).astype(np.float32))
                    else:
                        has_all_cols = False
                        break
                elif f_lower == "volume":
                    # Relative volume (today's volume / trailing 20-day average) instead
                    # of the raw count, so it's comparable across symbols with wildly
                    # different share counts under global normalization.
                    if "Volume" in cached_df.columns:
                        vol_vals = cached_df["Volume"].values.astype(np.float32)
                        vol_avg = cached_df["Volume"].rolling(window=20, min_periods=20).mean().values
                        cols_data.append(np.where(vol_avg > 0, vol_vals / vol_avg, np.nan).astype(np.float32))
                    else:
                        has_all_cols = False
                        break
                else:
                    db_col = COLUMN_MAPPING.get(f_lower, f_lower.upper())
                    if db_col in cached_df.columns:
                        cols_data.append(cached_df[db_col].values)
                    else:
                        has_all_cols = False
                        break

            if not has_all_cols or len(cols_data) == 0:
                continue
                
            matrix = np.column_stack(cols_data).astype(np.float32)
            # Filter out any rows with NaN/inf across any feature
            valid_mask = ~np.isnan(matrix).any(axis=1) & ~np.isinf(matrix).any(axis=1)
            matrix = matrix[valid_mask]
            
            if len(matrix) >= 50:  # Need enough data
                per_symbol_data.append(matrix)
                total_points += len(matrix)
                print(f"  [{i+1}/{len(symbols)}] {symbol}: {len(matrix)} data points")
        except Exception as e:
            print(f"  [{i+1}/{len(symbols)}] {symbol}: Error - {e}")
    
    print(f"\nTotal data points: {total_points} across {len(per_symbol_data)} symbols")
    return per_symbol_data


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
        "--normalization-type",
        type=str,
        choices=["global", "internal"],
        default="global",
        help="Normalization method: 'global' (dataset-wide stats) or 'internal' (per-sequence stats) (default: global)"
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
        "--lr-scheduler",
        action="store_true",
        help="Decay the learning rate via ReduceLROnPlateau, watching the same loss "
             "(val if available, else train) used for checkpoint selection."
    )
    parser.add_argument(
        "--lr-factor",
        type=float,
        default=0.5,
        help="Multiply the learning rate by this factor on each plateau (default: 0.5)"
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="Epochs with no improvement before decaying the learning rate (default: 10)"
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Floor the learning rate never decays below (default: 1e-6)"
    )
    parser.add_argument(
        "--checkpoint-warmup-epochs",
        type=int,
        default=0,
        help="Epochs 1..N train normally but are ineligible to become the 'best' "
             "checkpoint, and are excluded from LR scheduler plateau tracking too "
             "(default: 0, i.e. every epoch is eligible from the start)"
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
        "--split-strategy",
        type=str,
        choices=["symbol", "time"],
        default="symbol",
        help="How to split data into train/test sets: 'symbol' (separate stocks) or 'time' (past vs future for all stocks) (default: symbol)"
    )
    parser.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Skip Core ML conversion (useful for testing on non-Mac)"
    )
    parser.add_argument(
        "--include-delta",
        action="store_true",
        help="Include MACD delta (today - yesterday) as a feature and predict it"
    )
    parser.add_argument(
        "--residual-target",
        action="store_true",
        help="Train on (actual - drift baseline) instead of the actual values, where "
             "drift is macd[t+k] = macd[t] + (k+1)*delta[t]. Makes 'beat persistence' "
             "the training objective. Inference adds the drift back, so predictions "
             "stay in original units. Compare against scripts/persistence_baseline.py"
    )
    parser.add_argument(
        "--loss-decay-gamma",
        type=float,
        default=None,
        help="Exponential decay factor per forecast step for weighted MSE loss (e.g. 0.8). "
             "Discounts errors on far-horizon days so the model focuses on near-term accuracy. "
             "If omitted, standard unweighted MSE is used."
    )
    parser.add_argument(
        "--extra-features",
        type=str,
        default=None,
        help="Comma-separated list of additional features from stock_cache to use as inputs "
             "(e.g., 'open,close,volume' or 'open,high,low,close,volume,ma20,ma50'). "
             "'open-close' is a derived (close-open)/open ratio feature (distinct from the "
             "raw 'open'/'close' levels, which can be included alongside it). 'volume' is "
             "relative volume (today's volume / trailing 20-day average), not the raw count. "
             "Primary signal (MACD) and delta (if --include-delta) are always included."
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
    
    # Build list of feature names
    feature_names = [args.signal_type]
    if args.include_delta:
        feature_names.append("delta")
    if args.extra_features:
        extras = [f.strip().lower() for f in args.extra_features.split(",") if f.strip()]
        for ex in extras:
            if ex not in feature_names:
                feature_names.append(ex)
                
    # Determine model name based on signal type, architecture and features
    signal_label = "MACD" if args.signal_type == "macd" else "Signal Line"
    arch_label = args.architecture.replace("_", " ").title()
    delta_suffix = "_with_delta" if args.include_delta else ""
    residual_suffix = "_residual" if args.residual_target else ""
    extra_suffix = f"_with_{'_'.join([f.strip().lower() for f in args.extra_features.split(',') if f.strip()])}" if args.extra_features else ""
    # e.g. "macd_bidirectional_gru_with_delta_residual_with_open_close_volume".
    model_name = f"{args.signal_type}_{args.architecture}{delta_suffix}{residual_suffix}{extra_suffix}"
    
    print("=" * 60)
    print(f"{arch_label} {signal_label} Forecaster Training")
    print("=" * 60)
    print(f"Architecture: {args.architecture}")
    print(f"Signal type: {signal_label}")
    print(f"Features: {', '.join(feature_names)}")
    print(f"Normalization: {args.normalization_type}")
    print(f"Include Delta: {args.include_delta}")
    print(f"Residual Target: {args.residual_target}")
    if args.loss_decay_gamma is not None:
        print(f"Loss Decay Gamma: {args.loss_decay_gamma}")
    print(f"Output directory: {output_dir}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Forecast horizon: {args.forecast_horizon}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    if args.lr_scheduler:
        print(f"LR Scheduler: ReduceLROnPlateau (factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.lr_min})")
    if args.checkpoint_warmup_epochs > 0:
        print(f"Checkpoint warmup: {args.checkpoint_warmup_epochs} epochs (ineligible as 'best', excluded from LR scheduler tracking)")
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
    per_symbol_data = get_training_data(symbols, args.days, args.signal_type, feature_names=feature_names)
    
    if len(per_symbol_data) == 0:
        print("Error: No valid symbol data collected")
        sys.exit(1)
    
    # Calculate total data points across all symbols
    total_points = sum(len(data) for data in per_symbol_data)
    if total_points < 1000:
        print(f"Warning: Only {total_points} total data points. Need at least 1000 for good training.")
        if total_points < 100:
            print("Error: Not enough data for training")
            sys.exit(1)
    
    # Split data based on strategy
    if args.split_strategy == "time":
        train_data = []
        test_data = []
        for i, symbol_array in enumerate(per_symbol_data):
            # Each symbol is split temporally: first 85% for train, last 15% for test
            split_idx = int(len(symbol_array) * (1 - args.test_split))
            
            # Ensure each part is long enough for model sequence + forecast requirements
            if split_idx >= args.seq_length + args.forecast_horizon and \
               (len(symbol_array) - split_idx) >= args.seq_length + args.forecast_horizon:
                train_data.append(symbol_array[:split_idx])
                test_data.append(symbol_array[split_idx:])
            else:
                # If symbol doesn't have enough data to split, add it all to training
                train_data.append(symbol_array)
    else:
        # Default 'symbol' strategy: split by whole symbols (keeps symbol boundaries intact)
        test_size = int(len(per_symbol_data) * args.test_split)
        train_data = per_symbol_data[:-test_size] if test_size > 0 else per_symbol_data
        test_data = per_symbol_data[-test_size:] if test_size > 0 else None
    
    train_points = sum(len(data) for data in train_data)
    test_points = sum(len(data) for data in test_data) if test_data else 0
    
    print(f"\nData split (strategy: {args.split_strategy}):")
    if args.split_strategy == "time":
        print(f"  Total: {len(per_symbol_data)} symbols, {total_points} data points")
        print(f"  Train: {len(train_data)} symbols (part 1), {train_points} data points")
        print(f"  Test:  {len(test_data)} symbols (part 2), {test_points} data points")
    else:
        print(f"  Total: {len(per_symbol_data)} symbols, {total_points} data points")
        print(f"  Train: {len(train_data)} symbols, {train_points} data points")
        if test_data is not None:
            print(f"  Test:  {len(test_data)} symbols, {test_points} data points")
    
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
        architecture=args.architecture,
        normalization_type=args.normalization_type,
        include_delta=args.include_delta,
        residual_target=args.residual_target,
        loss_decay_gamma=args.loss_decay_gamma,
        feature_names=feature_names,
        lr_scheduler=args.lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min
    )
    
    print("\nTraining...")
    history = trainer.train(
        train_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        verbose=True,
        split_strategy=args.split_strategy,
        checkpoint_warmup_epochs=args.checkpoint_warmup_epochs
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print()
    print("Training Configuration:")
    print(f"  Architecture:       {args.architecture}")
    print(f"  Signal Type:        {signal_label}")
    print(f"  Features:           {', '.join(feature_names)}")
    print(f"  Normalization:      {args.normalization_type}")
    print(f"  Split Strategy:     {args.split_strategy}")
    print(f"  Sequence Length:    {args.seq_length}")
    print(f"  Forecast Horizon:   {args.forecast_horizon}")
    print(f"  Hidden Size:        {args.hidden_size}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch Size:         {args.batch_size}")
    print(f"  Learning Rate:      {args.learning_rate}")
    if args.lr_scheduler:
        print(f"  LR Scheduler:       ReduceLROnPlateau (factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.lr_min})")
    if args.checkpoint_warmup_epochs > 0:
        print(f"  Checkpoint Warmup:  {args.checkpoint_warmup_epochs} epochs")
    if args.loss_decay_gamma is not None:
        print(f"  Loss Decay Gamma:   {args.loss_decay_gamma}")

    # Evaluate on held-out test set
    if test_data is not None and len(test_data) > 0:
        # Check if any test symbol has enough data
        has_enough_data = any(len(data) > args.seq_length + args.forecast_horizon for data in test_data)
        if has_enough_data:
            print("\nEvaluating on held-out test set...")
            test_metrics = trainer.evaluate(test_data, verbose=True)
        else:
            print("\nSkipping test evaluation (not enough test data)")
            test_metrics = None
    else:
        print("\nSkipping test evaluation (no test symbols)")
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
            print()
            print("Model Parameters:")
            print(f"  Architecture:       {args.architecture}")
            print(f"  Signal Type:        {signal_label}")
            print(f"  Normalization:      {args.normalization_type}")
            print(f"  Sequence Length:    {args.seq_length}")
            print(f"  Forecast Horizon:   {args.forecast_horizon}")
            print(f"  Hidden Size:        {args.hidden_size}")
            if test_metrics:
                print(f"\nTest Metrics:")
                print(f"  MAE:                {test_metrics['mae']:.6f}")
                print(f"  Directional Acc:    {test_metrics['directional_accuracy']:.2%}")
            print("\nThe Core ML model will run on Apple Neural Engine (NPU)")
            
        except ImportError:
            print("\nWarning: coremltools not installed. Skipping Core ML export.")
            print("Install with: pip install coremltools")
            print(f"\nPyTorch model saved to: {pytorch_path}")
    else:
        print("\n" + "=" * 60)
        print("Training complete!")
        print("=" * 60)
        print(f"PyTorch model saved to: {pytorch_path}")
        print()
        print("Model Parameters:")
        print(f"  Architecture:       {args.architecture}")
        print(f"  Signal Type:        {signal_label}")
        print(f"  Normalization:      {args.normalization_type}")
        print(f"  Sequence Length:    {args.seq_length}")
        print(f"  Forecast Horizon:   {args.forecast_horizon}")
        print(f"  Hidden Size:        {args.hidden_size}")
        if test_metrics:
            print(f"\nTest Metrics:")
            print(f"  MAE:                {test_metrics['mae']:.6f}")
            print(f"  Directional Acc:    {test_metrics['directional_accuracy']:.2%}")
        print("\nCore ML export skipped (--skip-coreml)")


if __name__ == "__main__":
    main()
