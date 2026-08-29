"""
LSTM-based Time Series Forecaster for MACD prediction.

This module provides a PyTorch LSTM model that can be trained on MACD data
and exported to Core ML for NPU inference on Apple Silicon.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, List
import os


# Supported architectures
ARCHITECTURES = ["stacked_lstm", "bidirectional_gru", "stacked_gru", "standard_lstm", "gru"]


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting (MACD values).
    
    Architecture:
    - Input: sequence of MACD values (batch, seq_len, input_size)
    - LSTM layers with dropout
    - Fully connected output layer
    - Output: forecasted values (batch, forecast_horizon * input_size)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        """
        Initialize the LSTM forecaster.
        
        Args:
            input_size: Number of input features (1 for univariate MACD, 2 for MACD + Delta)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            forecast_horizon: Number of steps to forecast
        """
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * input_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Forecasted values of shape (batch, forecast_horizon * input_size)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        
        # Take the last timestep's output
        last_output = lstm_out[:, -1, :]
        
        # Project to forecast horizon
        forecast = self.fc(last_output)
        
        return forecast


class BidirectionalGRUForecaster(nn.Module):
    """
    Bidirectional GRU model for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        super(BidirectionalGRUForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * input_size
        
        # Bidirectional GRU layers
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Output projection (hidden_size * 2 because bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GRU forward (bidirectional)
        gru_out, _ = self.gru(x)

        # Correctly combine both directions:
        # - forward direction at last position  → has processed all inputs left-to-right
        # - backward direction at first position → has processed all inputs right-to-left
        forward_last   = gru_out[:, -1, :self.hidden_size]
        backward_first = gru_out[:, 0, self.hidden_size:]
        last_output = torch.cat([forward_last, backward_first], dim=1)

        # Project to forecast horizon
        forecast = self.fc(last_output)

        return forecast


class StackedGRUForecaster(nn.Module):
    """
    Stacked GRU model (non-bidirectional) for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        super(StackedGRUForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * input_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class StandardGRUForecaster(nn.Module):
    """
    Standard single-layer GRU for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        super(StandardGRUForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * input_size
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


class StandardLSTMForecaster(nn.Module):
    """
    Standard single-layer LSTM for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        forecast_horizon: int = 5
    ):
        super(StandardLSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * input_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, self.output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        forecast = self.fc(last_output)
        return forecast


def create_model(
    architecture: str = "stacked_lstm",
    input_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    forecast_horizon: int = 5
) -> nn.Module:
    """
    Factory function to create a forecaster model by architecture name.
    """
    architecture = architecture.lower()
    
    if architecture == "stacked_lstm":
        return LSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    elif architecture == "bidirectional_gru":
        return BidirectionalGRUForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    elif architecture == "stacked_gru":
        return StackedGRUForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    elif architecture == "standard_lstm":
        return StandardLSTMForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    elif architecture in ("gru", "standard_gru"):
        return StandardGRUForecaster(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=dropout,
            forecast_horizon=forecast_horizon
        )
    else:
        raise ValueError(
            f"Unknown architecture: {architecture}. "
            f"Choose from: {ARCHITECTURES}"
        )


class MACDForecasterTrainer:
    """Trainer for MACD forecaster models (LSTM, GRU, Bidirectional variants)."""
    
    def __init__(
        self,
        seq_length: int = 30,
        forecast_horizon: int = 5,
        hidden_size: int = 64,
        num_layers: int = 2,
        learning_rate: float = 0.001,
        architecture: str = "stacked_lstm",
        normalization_type: str = "global",
        include_delta: bool = False,
        residual_target: bool = False,
        device: str = None
    ):
        """
        Initialize the trainer.

        Args:
            seq_length: Length of input sequences
            forecast_horizon: Number of days to forecast
            hidden_size: Hidden layer size
            num_layers: Number of recurrent layers
            learning_rate: Learning rate for optimizer
            architecture: Model architecture
            normalization_type: 'global' (dataset-wide) or 'internal' (per-sequence)
            include_delta: If True, include MACD delta as a feature and predict it
            residual_target: If True, train on (actual - drift baseline) instead of
                the actual values, where drift is the linear extrapolation
                macd[t+k] = macd[t] + (k+1)*delta[t]. Makes "beat persistence" the
                literal training objective. Inference adds the drift back, so
                predictions remain in the original units.
            device: Device to train on
        """
        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.architecture = architecture
        self.normalization_type = normalization_type.lower()
        self.include_delta = include_delta
        self.residual_target = residual_target
        self.input_size = 2 if include_delta else 1
        self.output_size = forecast_horizon * self.input_size
        self.batch_size = None # Set during training
        
        # Auto-select device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        print(f"Architecture: {architecture}")
        print(f"Normalization: {self.normalization_type}")
        print(f"Include Delta: {self.include_delta}")
        print(f"Residual Target: {self.residual_target}")
        
        # Initialize model using factory function
        self.model = create_model(
            architecture=architecture,
            input_size=self.input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # For 'global' normalization (store as vectors for multi-variate support)
        self.mean = np.zeros(self.input_size, dtype=np.float32)
        self.std = np.ones(self.input_size, dtype=np.float32)
    
    def _drift_baseline(self, last: float, last_delta: float) -> np.ndarray:
        """
        Linear-extrapolation ("drift") persistence baseline for one window.

            macd[t+k]  = macd[t] + (k+1) * delta[t]
            delta[t+k] = delta[t]

        This is the baseline measured by scripts/persistence_baseline.py. In
        residual mode the model is trained to predict the error of THIS, so
        anything it learns is by construction information the baseline lacks.

        Returns shape (forecast_horizon, input_size) in raw units.
        """
        steps = np.arange(1, self.forecast_horizon + 1, dtype=np.float32)
        drift = np.empty((self.forecast_horizon, self.input_size), dtype=np.float32)
        drift[:, 0] = last + last_delta * steps
        if self.input_size > 1:
            drift[:, 1] = last_delta
        return drift

    def _calculate_deltas(self, series: np.ndarray) -> np.ndarray:
        """Calculate MACD delta (today - yesterday). First point is 0."""
        deltas = np.zeros_like(series)
        deltas[1:] = series[1:] - series[:-1]
        return deltas

    def prepare_sequences(
        self,
        data,
        fit: bool = True,
        return_drift: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training sequences from time series data.

        Args:
            data: List of 1D numpy arrays (one per symbol) or a single array
            fit: If True (training), compute and store normalization stats.
                 If False (evaluation), reuse existing self.mean/self.std.
            return_drift: If True, also return the drift baseline for each window,
                normalized into the same space the *level* target occupies, i.e.
                D = (drift_raw - mean) / std. That makes reconstruction additive:
                    level_norm = residual_norm + D
                Used by evaluate() to score in level space regardless of target mode.

        Returns (X, y) or (X, y, D) when return_drift is True.

        In residual mode the target is (y_raw - drift_raw) / std — scaled by std
        but with NO mean subtraction, because a residual is already centred near
        zero and drift's systematic overshoot at longer horizons is exactly the
        bias we want the model to learn rather than absorb into normalization.
        """
        all_X, all_y, all_D = [], [], []

        # Process each symbol into (raw, delta) pairs if needed
        processed_symbols = []
        input_data = data if isinstance(data, list) else [data]

        for symbol_series in input_data:
            if self.include_delta:
                deltas = self._calculate_deltas(symbol_series)
                # Stack to shape (points, 2)
                stacked = np.stack([symbol_series, deltas], axis=1)
                processed_symbols.append(stacked)
            else:
                # Shape (points, 1)
                processed_symbols.append(symbol_series.reshape(-1, 1))

        # 1. Handle dataset-wide stats for 'global' mode
        if self.normalization_type == "global" and fit:
            combined = np.concatenate(processed_symbols, axis=0)
            self.mean = np.mean(combined, axis=0)
            self.std = np.std(combined, axis=0) + 1e-8
        
        # 2. Process processed symbols into windows
        for symbol_data in processed_symbols:
            if len(symbol_data) < self.seq_length + self.forecast_horizon:
                continue
                
            for i in range(len(symbol_data) - self.seq_length - self.forecast_horizon + 1):
                X_raw = symbol_data[i:i + self.seq_length]
                y_raw = symbol_data[i + self.seq_length:i + self.seq_length + self.forecast_horizon]
                
                if self.normalization_type == "internal":
                    # Per-sequence normalization (per feature)
                    m = np.mean(X_raw, axis=0)
                    s = np.std(X_raw, axis=0) + 1e-8
                else:
                    # Dataset-wide normalization
                    m = self.mean
                    s = self.std

                X_norm = (X_raw - m) / s

                if self.residual_target or return_drift:
                    # Drift is computed from RAW values, before any normalization.
                    drift_raw = self._drift_baseline(
                        last=float(X_raw[-1, 0]),
                        last_delta=float(X_raw[-1, 0] - X_raw[-2, 0])
                    )

                if self.residual_target:
                    y_norm = (y_raw - drift_raw) / s
                else:
                    y_norm = (y_raw - m) / s

                all_X.append(X_norm)
                # Flatten target if multi-variate: (horizon, input_size) -> (horizon * input_size)
                all_y.append(y_norm.flatten())
                if return_drift:
                    all_D.append(((drift_raw - m) / s).flatten())

        if not all_X:
            # No window was long enough. Return empty tensors rather than
            # crashing in zip(*combined) below, so callers can check len().
            empty = (
                torch.empty((0, self.seq_length, self.input_size), dtype=torch.float32),
                torch.empty((0, self.output_size), dtype=torch.float32)
            )
            return empty + (empty[1],) if return_drift else empty

        # Shuffle (keep X/y/D aligned)
        if return_drift:
            combined = list(zip(all_X, all_y, all_D))
            np.random.shuffle(combined)
            all_X, all_y, all_D = zip(*combined)
        else:
            combined = list(zip(all_X, all_y))
            np.random.shuffle(combined)
            all_X, all_y = zip(*combined)

        # Convert to tensors
        X = torch.tensor(np.array(all_X), dtype=torch.float32)
        y = torch.tensor(np.array(all_y), dtype=torch.float32)
        if return_drift:
            return X, y, torch.tensor(np.array(all_D), dtype=torch.float32)
        
        return X, y
    
    def _split_series_by_symbol(self, series_list, val_fraction: float):
        """Hold out whole symbols. Measures generalization to UNSEEN symbols."""
        n_val = int(len(series_list) * val_fraction)
        if n_val >= 1 and (len(series_list) - n_val) >= 1:
            return series_list[:-n_val], series_list[-n_val:]
        return series_list, []

    def _split_series_by_time(self, series_list, val_fraction: float):
        """
        Split each symbol temporally. Measures generalization to LATER DATES on
        the same symbols — which is the production condition.

        A purge gap of forecast_horizon separates the two parts: training windows
        only touch indices < cut, so any gap >= 0 already guarantees index
        disjointness; the horizon-sized gap additionally stops the last training
        window's target period from sitting adjacent to the first validation input.
        """
        min_len = self.seq_length + self.forecast_horizon
        embargo = self.forecast_horizon
        train_part, val_part = [], []
        for s in series_list:
            cut = int(len(s) * (1 - val_fraction))
            if cut >= min_len and (len(s) - cut - embargo) >= min_len:
                train_part.append(s[:cut])
                val_part.append(s[cut + embargo:])
            else:
                train_part.append(s)  # too short to split — training only
        return train_part, val_part

    def _split_series(self, data, val_fraction: float, strategy: str = "symbol"):
        """
        Split per-symbol series into train/val parts BEFORE windowing.

        Windows are cut with stride 1, so window i and window i+1 share
        seq_length-1 of their inputs. Splitting *after* windowing (and after
        shuffling) puts near-duplicates of training windows into the validation
        set, which makes val loss meaningless. Splitting the raw series first
        guarantees no validation window shares an observation with a training one.

        `strategy` should match how the caller split off its test set, so that
        val is a valid early-stopping proxy for test:
            "time"   -> temporal split within each symbol (same symbols, later
                        dates — matches production)
            "symbol" -> hold out whole symbols (unseen symbols)
        Falls back to the other strategy, with a warning, if the requested one
        cannot produce a validation set.

        Returns (train_series, val_series); val_series is [] if no split is possible.
        """
        series_list = data if isinstance(data, list) else [data]

        if val_fraction <= 0:
            return series_list, []

        if strategy == "time":
            train_part, val_part = self._split_series_by_time(series_list, val_fraction)
            if val_part:
                return train_part, val_part
            print("WARNING: temporal validation split produced no data (symbols "
                  "too short); falling back to a by-symbol split. Val loss will "
                  "measure unseen-symbol generalization, not later dates.")
            return self._split_series_by_symbol(series_list, val_fraction)

        train_part, val_part = self._split_series_by_symbol(series_list, val_fraction)
        if val_part:
            return train_part, val_part
        print("WARNING: by-symbol validation split needs >= 2 symbols; "
              "falling back to a temporal split.")
        return self._split_series_by_time(series_list, val_fraction)

    def train(
        self,
        train_data,
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.2,
        verbose: bool = True,
        split_strategy: str = "symbol"
    ) -> dict:
        """
        Train the model on MACD data.

        split_strategy: how to carve the validation set — "time" (later dates,
            same symbols) or "symbol" (unseen symbols). Pass the same strategy
            used for the test split so val tracks test.
        """
        self.batch_size = batch_size

        train_series, val_series = self._split_series(
            train_data, validation_split, split_strategy
        )

        # fit=True stores the normalization stats — computed on TRAIN ONLY so the
        # validation set contributes nothing to them.
        X_train, y_train = self.prepare_sequences(train_series, fit=True)
        if val_series:
            X_val, y_val = self.prepare_sequences(val_series, fit=False)
        else:
            X_val, y_val = None, None

        if len(X_train) == 0:
            raise ValueError(
                f"No training windows produced. Each symbol needs at least "
                f"{self.seq_length + self.forecast_horizon} data points "
                f"(seq_length={self.seq_length} + forecast_horizon={self.forecast_horizon})."
            )

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            if verbose:
                axis = ("later dates, same symbols" if split_strategy == "time"
                        else "unseen symbols")
                print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)} "
                      f"(split before windowing — no overlap; val measures {axis})")
        else:
            print("WARNING: no validation set could be built; "
                  "selecting the checkpoint on train loss instead.")

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        
        for epoch in range(epochs):
            self.model.train()
            
            # Mini-batch training
            indices = torch.randperm(len(X_train))
            total_loss = 0.0
            num_batches = 0
            
            for i in range(0, len(X_train), batch_size):
                batch_idx = indices[i:i + batch_size]
                batch_X = X_train[batch_idx]
                batch_y = y_train[batch_idx]
                
                self.optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = self.criterion(predictions, batch_y)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            train_loss = total_loss / num_batches
            
            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val).item()
            else:
                val_loss = train_loss

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            selection_loss = val_loss if has_val else train_loss
            if selection_loss < best_val_loss:
                best_val_loss = selection_loss
                best_epoch = epoch + 1
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                metric = "val" if has_val else "train"
                print(f"Restored best checkpoint from epoch {best_epoch}/{epochs} "
                      f"({metric} loss {best_val_loss:.6f})")
                if has_val and best_epoch == epochs:
                    print("NOTE: best epoch is the last epoch — val loss never turned up. "
                          "The model is likely undertrained; try more epochs.")

        return history
    
    def evaluate(
        self,
        test_data,
        verbose: bool = True
    ) -> dict:
        """
        Evaluate the model on held-out test data.

        Metrics are always computed in *level* space, even in residual mode, so
        they stay comparable across target modes. In residual mode the drift
        baseline is scored on the identical windows and reported alongside — if
        the model's numbers do not beat the drift column, it has learned nothing
        that persistence did not already provide.
        """
        X_test, y_test, D_test = self.prepare_sequences(
            test_data, fit=False, return_drift=True
        )

        if len(X_test) == 0:
            return {"error": "Not enough test data"}

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        D_test = D_test.to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_test)

        # Move everything into normalized LEVEL space.
        # In residual mode the model and the target are both residuals, and
        # level_norm = residual_norm + D (see prepare_sequences).
        if self.residual_target:
            pred_level = predictions + D_test
            actual_level = y_test + D_test
        else:
            pred_level = predictions
            actual_level = y_test

        def _score(pred_t):
            """Per-feature metrics for one prediction tensor, in level space."""
            p = pred_t.cpu().numpy().reshape(-1, self.forecast_horizon, self.input_size)
            a = actual_level.cpu().numpy().reshape(-1, self.forecast_horizon, self.input_size)
            X_np = X_test.cpu().numpy()
            out = []
            for f in range(self.input_size):
                p_f, a_f = p[:, :, f], a[:, :, f]
                mse = np.mean((p_f - a_f) ** 2)
                # Directional accuracy on day 1, versus the last input value
                last_vals = X_np[:, -1, f]
                da = float(np.mean((p_f[:, 0] > last_vals) == (a_f[:, 0] > last_vals)))
                out.append({
                    "mae": float(np.mean(np.abs(p_f - a_f))),
                    "rmse": float(np.sqrt(mse)),
                    "directional_accuracy": da
                })
            return out

        feature_metrics = _score(pred_level)
        # Zero residual == the drift baseline exactly.
        drift_metrics = _score(D_test) if self.residual_target else None

        metrics = {
            "mae": feature_metrics[0]["mae"],  # MACD
            "rmse": feature_metrics[0]["rmse"],
            "directional_accuracy": feature_metrics[0]["directional_accuracy"],
            "test_samples": len(X_test),
            "features": feature_metrics,
            "drift_features": drift_metrics
        }

        if verbose:
            print("\n" + "=" * 62)
            print("Test Evaluation Results (normalized level space)")
            print("=" * 62)
            print(f"Test samples: {metrics['test_samples']}")
            if self.residual_target:
                print("Target mode:  residual (model predicts the error of drift)")

            labels = ["MACD", "Delta"] if self.input_size > 1 else ["Signal"]
            for i, label in enumerate(labels):
                m = feature_metrics[i]
                print(f"\n{label} Metrics:")
                if drift_metrics:
                    d = drift_metrics[i]
                    print(f"  {'':22s}{'model':>12s}{'drift':>12s}")
                    print(f"  {'MAE':22s}{m['mae']:>12.6f}{d['mae']:>12.6f}")
                    print(f"  {'RMSE':22s}{m['rmse']:>12.6f}{d['rmse']:>12.6f}")
                    print(f"  {'Directional Acc (D1)':22s}"
                          f"{m['directional_accuracy']:>11.2%}{d['directional_accuracy']:>12.2%}")
                    better = m["mae"] < d["mae"] and m["directional_accuracy"] > d["directional_accuracy"]
                    print(f"  -> model beats drift on both: {better}")
                else:
                    print(f"  MAE:                  {m['mae']:.6f}")
                    print(f"  RMSE:                 {m['rmse']:.6f}")
                    print(f"  Directional Acc (D1): {m['directional_accuracy']:.2%}")
            print("=" * 62)

        return metrics
    
    def predict(self, sequence: np.ndarray, prev_value: float = None) -> np.ndarray:
        """
        Make a prediction given an input sequence.
        Input sequence shape: (seq_length,)
        prev_value: the data point immediately before sequence[0], used to compute
                    the correct delta for position 0 (matches training behaviour).
        Returns: forecasted values (forecast_horizon, input_size)
        """
        self.model.eval()

        # Window the input FIRST, then resolve prev_value against the windowed
        # array — otherwise a caller-supplied prev_value points at the wrong
        # element once truncation shifts the window. Keeps this path consistent
        # with the Core ML path, which previously truncated while this one did not.
        # NOTE: kept in sync with CoreMLForecaster.predict in neural_forecast.py —
        # not shared, to avoid importing torch into the inference-only module.
        seq = np.asarray(sequence, dtype=np.float32)

        if len(seq) > self.seq_length:
            prev_value = float(seq[-(self.seq_length + 1)])
            seq = seq[-self.seq_length:]
        elif len(seq) < self.seq_length:
            print(f"WARNING: input has {len(seq)} points but the model needs "
                  f"{self.seq_length}; padding {self.seq_length - len(seq)} synthetic steps.")
            padding = np.full(self.seq_length - len(seq), seq[0], dtype=np.float32)
            seq = np.concatenate([padding, seq])
            prev_value = None   # seq[0] is synthetic
        sequence = seq

        # Prepare features
        if self.include_delta:
            deltas = self._calculate_deltas(sequence)
            if prev_value is not None:
                deltas[0] = sequence[0] - prev_value
            input_features = np.stack([sequence, deltas], axis=1)
        else:
            input_features = sequence.reshape(-1, 1)
            
        if self.normalization_type == "internal":
            m = np.mean(input_features, axis=0)
            s = np.std(input_features, axis=0) + 1e-8
        else:
            m = self.mean
            s = self.std
            
        # Normalize
        normalized = (input_features - m) / s
        
        # Prepare input
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0) # (1, seq_len, features)
        x = x.to(self.device)
        
        # Predict
        with torch.no_grad():
            pred = self.model(x)
        
        # Denormalize and reshape
        pred_np = pred.cpu().numpy()[0].reshape(self.forecast_horizon, self.input_size)

        if self.residual_target:
            # The model predicted the drift baseline's error. Denormalize the
            # residual by std only (no mean — it was never subtracted) and add
            # the raw drift back:  level = drift_raw + residual_norm * std
            drift_raw = self._drift_baseline(
                last=float(sequence[-1]),
                last_delta=float(sequence[-1] - sequence[-2])
            )
            forecast_all = drift_raw + pred_np * s
        else:
            forecast_all = pred_np * s + m

        return forecast_all
    
    def save(self, path: str):
        """Save model and normalization parameters."""
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "mean": self.mean,
            "std": self.std,
            "normalization_type": self.normalization_type,
            "seq_length": self.seq_length,
            "forecast_horizon": self.forecast_horizon,
            "hidden_size": self.model.hidden_size,
            "num_layers": self.model.num_layers,
            "batch_size": self.batch_size,
            "architecture": self.architecture,
            "include_delta": self.include_delta,
            "residual_target": self.residual_target,
            "input_size": self.input_size
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and normalization parameters."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.include_delta = checkpoint.get("include_delta", False)
        self.residual_target = checkpoint.get("residual_target", False)
        self.input_size = checkpoint.get("input_size", 1)
        self.seq_length = checkpoint["seq_length"]
        self.forecast_horizon = checkpoint["forecast_horizon"]
        self.normalization_type = checkpoint.get("normalization_type", "global")
        
        # Re-initialize model with correct input size
        self.model = create_model(
            architecture=checkpoint.get("architecture", self.architecture),
            input_size=self.input_size,
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2),
            forecast_horizon=self.forecast_horizon
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.mean = checkpoint.get("mean", 0.0)
        self.std = checkpoint.get("std", 1.0)
        
        print(f"Model loaded from {path} (Features: {self.input_size}, Norm: {self.normalization_type})")
    
    def export_to_coreml(self, output_path: str) -> str:
        """
        Export the model to Core ML format.
        """
        import coremltools as ct
        
        self.model.eval()
        self.model.to("cpu")
        example_input = torch.randn(1, self.seq_length, self.input_size)
        traced_model = torch.jit.trace(self.model, example_input)
        
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, self.seq_length, self.input_size), name="input_sequence")],
            outputs=[ct.TensorType(name="forecast")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL
        )
        
        mlmodel.author = "Tick Scanner"
        mlmodel.short_description = f"{self.architecture} MACD forecaster (Features: {self.input_size}, Norm: {self.normalization_type})"
        mlmodel.version = "1.2"
        
        mlmodel.user_defined_metadata["normalization_type"] = self.normalization_type
        mlmodel.user_defined_metadata["mean"] = str(self.mean.tolist())
        mlmodel.user_defined_metadata["std"] = str(self.std.tolist())
        mlmodel.user_defined_metadata["seq_length"] = str(self.seq_length)
        mlmodel.user_defined_metadata["forecast_horizon"] = str(self.forecast_horizon)
        mlmodel.user_defined_metadata["include_delta"] = str(self.include_delta)
        mlmodel.user_defined_metadata["residual_target"] = str(self.residual_target)
        mlmodel.user_defined_metadata["hidden_size"] = str(self.model.hidden_size)
        mlmodel.user_defined_metadata["num_layers"] = str(self.model.num_layers)
        if self.batch_size:
            mlmodel.user_defined_metadata["batch_size"] = str(self.batch_size)
        
        mlmodel.save(output_path)
        print(f"Core ML model saved to {output_path}")
        
        return output_path


def get_model_path(signal_type: str = "macd", architecture: str = "stacked_lstm") -> str:
    """
    Get the path to the Core ML model.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "models", f"{signal_type}_{architecture}_forecaster.mlpackage")


def get_pytorch_model_path(signal_type: str = "macd", architecture: str = "stacked_lstm") -> str:
    """
    Get the path to the PyTorch model checkpoint.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    return os.path.join(base_dir, "models", f"{signal_type}_{architecture}_forecaster.pt")
