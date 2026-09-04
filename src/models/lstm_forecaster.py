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
import re
from datetime import datetime


# Supported architectures
ARCHITECTURES = ["stacked_lstm", "bidirectional_gru", "stacked_gru", "standard_lstm", "gru"]


def seed_everything(seed: Optional[int]) -> Optional[int]:
    """Seed Python, NumPy and torch RNGs for a controlled comparison between runs.

    Call this ONCE, before any data shuffling or model construction — weight init
    and `np.random.shuffle` in prepare_sequences both draw from these generators.

    Returns the seed it applied, or None when `seed` is None (in which case the
    RNGs are left alone and the run is nondeterministic, the historical behaviour).

    NOT a bitwise-reproducibility guarantee: MPS and cuDNN kernels can reorder
    floating-point reductions between runs, so two seeded runs may still differ in
    the last digits and, after enough epochs, in which checkpoint wins. The purpose
    is to remove *initialization and shuffling* as a source of variance so that two
    arms of an experiment differ only in the thing being tested.
    """
    if seed is None:
        return None
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed


class LSTMForecaster(nn.Module):
    """
    LSTM model for time series forecasting (MACD values).
    
    Architecture:
    - Input: sequence of features (batch, seq_len, input_size)
    - LSTM layers with dropout
    - Fully connected output layer
    - Output: forecasted values (batch, forecast_horizon * target_size)
    """
    
    def __init__(
        self,
        input_size: int = 1,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
        auxiliary_direction: bool = False
    ):
        """
        Initialize the LSTM forecaster.

        Args:
            input_size: Number of input features
            target_size: Number of target features to forecast (1 for MACD, 2 for MACD + Delta)
            hidden_size: Number of LSTM hidden units
            num_layers: Number of LSTM layers
            dropout: Dropout rate between LSTM layers
            forecast_horizon: Number of steps to forecast
            auxiliary_direction: If True (A1), add a small auxiliary head predicting
                per-day direction logits alongside the magnitude forecast.
        """
        super(LSTMForecaster, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * target_size
        self.auxiliary_direction = auxiliary_direction

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
        if auxiliary_direction:
            self.direction_head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor):
        # LSTM forward
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        forecast = self.fc(last_output)
        if self.auxiliary_direction:
            return forecast, self.direction_head(last_output)
        return forecast


class BidirectionalGRUForecaster(nn.Module):
    """
    Bidirectional GRU model for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
        auxiliary_direction: bool = False
    ):
        super(BidirectionalGRUForecaster, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * target_size
        self.auxiliary_direction = auxiliary_direction

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
        if auxiliary_direction:
            self.direction_head = nn.Linear(hidden_size * 2, forecast_horizon)

    def forward(self, x: torch.Tensor):
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

        if self.auxiliary_direction:
            return forecast, self.direction_head(last_output)
        return forecast


class StackedGRUForecaster(nn.Module):
    """
    Stacked GRU model (non-bidirectional) for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
        auxiliary_direction: bool = False
    ):
        super(StackedGRUForecaster, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * target_size
        self.auxiliary_direction = auxiliary_direction

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
        if auxiliary_direction:
            self.direction_head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        forecast = self.fc(last_output)
        if self.auxiliary_direction:
            return forecast, self.direction_head(last_output)
        return forecast


class StandardGRUForecaster(nn.Module):
    """
    Standard single-layer GRU for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
        auxiliary_direction: bool = False
    ):
        super(StandardGRUForecaster, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * target_size
        self.auxiliary_direction = auxiliary_direction

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
        if auxiliary_direction:
            self.direction_head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        forecast = self.fc(last_output)
        if self.auxiliary_direction:
            return forecast, self.direction_head(last_output)
        return forecast


class StandardLSTMForecaster(nn.Module):
    """
    Standard single-layer LSTM for time series forecasting.
    """
    
    def __init__(
        self,
        input_size: int = 1,
        target_size: int = 1,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        forecast_horizon: int = 5,
        auxiliary_direction: bool = False
    ):
        super(StandardLSTMForecaster, self).__init__()

        self.input_size = input_size
        self.target_size = target_size
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.forecast_horizon = forecast_horizon
        self.output_size = forecast_horizon * target_size
        self.auxiliary_direction = auxiliary_direction

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
        if auxiliary_direction:
            self.direction_head = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x: torch.Tensor):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        forecast = self.fc(last_output)
        if self.auxiliary_direction:
            return forecast, self.direction_head(last_output)
        return forecast


def create_model(
    architecture: str = "stacked_lstm",
    input_size: int = 1,
    target_size: int = 1,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    forecast_horizon: int = 5,
    auxiliary_direction: bool = False
) -> nn.Module:
    """
    Factory function to create a forecaster model by architecture name.
    """
    architecture = architecture.lower()

    if architecture == "stacked_lstm":
        return LSTMForecaster(
            input_size=input_size,
            target_size=target_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=auxiliary_direction
        )
    elif architecture == "bidirectional_gru":
        return BidirectionalGRUForecaster(
            input_size=input_size,
            target_size=target_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=auxiliary_direction
        )
    elif architecture == "stacked_gru":
        return StackedGRUForecaster(
            input_size=input_size,
            target_size=target_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=auxiliary_direction
        )
    elif architecture == "standard_lstm":
        return StandardLSTMForecaster(
            input_size=input_size,
            target_size=target_size,
            hidden_size=hidden_size,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=auxiliary_direction
        )
    elif architecture in ("gru", "standard_gru"):
        return StandardGRUForecaster(
            input_size=input_size,
            target_size=target_size,
            hidden_size=hidden_size,
            dropout=dropout,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=auxiliary_direction
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
        loss_decay_gamma: float = None,
        loss_decay_gamma_delta: float = None,
        feature_names: List[str] = None,
        lr_scheduler: bool = False,
        lr_factor: float = 0.5,
        lr_patience: int = 10,
        lr_min: float = 1e-6,
        auxiliary_direction_lambda: float = None,
        predict_deltas_only: bool = False,
        seed: int = None,
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
            loss_decay_gamma: Exponential decay factor per forecast step for MSE loss
                (e.g., 0.8). When set < 1.0, discounts errors on further horizons
                so the model focuses on near-term prediction. Default is None (unweighted MSE).
                Applies to the primary signal (macd/signal_line) column; also applies to the
                delta column unless loss_decay_gamma_delta overrides it (A3).
            loss_decay_gamma_delta: Optional independent decay gamma for the delta column
                (only meaningful with include_delta=True). Defaults to loss_decay_gamma when
                omitted, so existing configs that only set loss_decay_gamma are unaffected.
            feature_names: Optional list of feature names (e.g. ['macd', 'delta', 'open', 'close', 'volume']).
                If omitted, defaults to ['macd', 'delta'] when include_delta is True, else ['macd'].
            lr_scheduler: If True, decay the learning rate via ReduceLROnPlateau, watching
                the same loss (val if available, else train) used for checkpoint selection.
            lr_factor: Multiply the learning rate by this factor on each plateau (default 0.5).
            lr_patience: Epochs with no improvement before decaying (default 10).
            lr_min: Floor the learning rate never decays below (default 1e-6).
            auxiliary_direction_lambda: Weight for an auxiliary BCE loss term on
                sign(future change) per forecast day (A1). None/<=0 disables it
                (default). Requires no other flag — works with or without include_delta.
            predict_deltas_only: If True (A2), the network predicts delta only and the
                primary signal is reconstructed as macd[t] + cumsum(predicted_deltas),
                anchored to the last known real value. Requires include_delta=True and
                normalization_type='global', and is incompatible with residual_target.
            seed: If set, seed Python/NumPy/torch RNGs here — before weight
                initialization — so two trainers built with the same seed start
                from identical weights and shuffle identically. None (default)
                leaves the RNGs untouched, i.e. the historical nondeterministic
                behaviour. See seed_everything() for what this does and does not
                guarantee.
            device: Device to train on
        """
        # Must run before the model is constructed below: weight init draws from
        # the torch RNG this seeds.
        self.seed = seed_everything(seed)

        self.seq_length = seq_length
        self.forecast_horizon = forecast_horizon
        self.architecture = architecture
        self.normalization_type = normalization_type.lower()
        self.residual_target = residual_target
        self.loss_decay_gamma = loss_decay_gamma
        self.loss_decay_gamma_delta = loss_decay_gamma_delta
        self.auxiliary_lambda = auxiliary_direction_lambda
        self.use_auxiliary_direction = auxiliary_direction_lambda is not None and auxiliary_direction_lambda > 0
        self.predict_deltas_only = predict_deltas_only

        # Configure feature names
        if feature_names is not None and len(feature_names) > 0:
            self.feature_names = [f.strip().lower() for f in feature_names]
            self.include_delta = "delta" in self.feature_names or include_delta
            if self.include_delta and "delta" not in self.feature_names:
                # Insert delta right after primary signal if missing
                self.feature_names.insert(1, "delta")
        else:
            self.include_delta = include_delta
            self.feature_names = ["macd", "delta"] if include_delta else ["macd"]
            
        self.input_size = len(self.feature_names)
        self.target_size = 2 if self.include_delta else 1
        self.output_size = forecast_horizon * self.target_size
        self.batch_size = None # Set during training

        if self.predict_deltas_only:
            if not self.include_delta:
                raise ValueError("predict_deltas_only requires include_delta=True (a real "
                                  "delta target is needed for the network's sole output).")
            if self.residual_target:
                raise ValueError("predict_deltas_only is not yet supported together with "
                                  "residual_target.")
            if self.normalization_type != "global":
                raise ValueError("predict_deltas_only currently requires "
                                  "normalization_type='global'.")
        self.network_target_size = 1 if self.predict_deltas_only else self.target_size

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
        print(f"Features ({self.input_size}): {', '.join(self.feature_names)}")
        print(f"Targets ({self.target_size}): {', '.join(self.feature_names[:self.target_size])}")
        print(f"Residual Target: {self.residual_target}")
        if self.predict_deltas_only:
            print("Predict Deltas Only: True (primary signal reconstructed via cumsum)")
        if self.use_auxiliary_direction:
            print(f"Auxiliary Direction Loss: lambda={self.auxiliary_lambda}")

        # Initialize model using factory function
        self.model = create_model(
            architecture=architecture,
            input_size=self.input_size,
            target_size=self.network_target_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            forecast_horizon=forecast_horizon,
            auxiliary_direction=self.use_auxiliary_direction
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        self.scheduler = None
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=lr_factor, patience=lr_patience, min_lr=lr_min
            )
        
        # Decay weights for horizon loss weighting (if enabled) — A3
        self._build_loss_weights()

        # For 'global' normalization (store as vectors for multi-variate support)
        self.mean = np.zeros(self.input_size, dtype=np.float32)
        self.std = np.ones(self.input_size, dtype=np.float32)

    def _build_loss_weights(self):
        """
        (Re)build self.loss_weights from self.loss_decay_gamma / self.loss_decay_gamma_delta (A3).
        Called from __init__ and load() — the single source of truth for this math so the two
        can't drift apart the way the old duplicated blocks could.

        loss_decay_gamma applies to the primary signal (macd/signal_line) column. Delta gets its
        own gamma via loss_decay_gamma_delta when set, else falls back to loss_decay_gamma — so a
        config that only ever sets loss_decay_gamma behaves identically to before A3 existed.

        Each column is normalized to its own mean of 1.0, so a column's gamma only redistributes
        weight *within* that column's horizon and never changes how much of the total loss budget
        the column gets. Normalizing the flattened matrix by one shared mean instead would let a
        fast-decaying column's small raw sum be divided by a denominator inflated by the other
        column, silently handing that other column a larger share of the gradient (with
        gamma_macd=0.5 / gamma_delta=0.9 over 5 days, macd's share drops from 50% to 32%).
        Under a single gamma both columns are identical, so the two schemes coincide — which is
        why this only ever affected the A3 two-gamma path.
        """
        gamma_macd = self.loss_decay_gamma
        gamma_delta = self.loss_decay_gamma_delta if self.loss_decay_gamma_delta is not None else self.loss_decay_gamma

        if self.target_size == 1 and self.loss_decay_gamma_delta is not None:
            print("WARNING: loss_decay_gamma_delta has no effect without a delta target "
                  "(--include-delta); ignoring.")

        for name, g in (("loss_decay_gamma", gamma_macd), ("loss_decay_gamma_delta", gamma_delta)):
            if g is not None and not (0.0 < g <= 1.0):
                raise ValueError(f"{name} must be in (0.0, 1.0], got {g}")

        if gamma_macd is None or gamma_macd >= 1.0:
            self.loss_weights = None
            return

        def day_weights(gamma):
            raw = np.array([gamma ** k for k in range(self.forecast_horizon)], dtype=np.float32)
            return raw / raw.mean()

        if self.target_size > 1:
            matrix = np.stack([day_weights(gamma_macd), day_weights(gamma_delta)], axis=1)
        else:
            matrix = day_weights(gamma_macd)[:, None]

        # Row-major flatten matches y_norm.flatten()'s (horizon, target_size) -> (horizon*target_size) layout.
        # Every column already has mean 1.0, so the flattened vector does too — no second division.
        flat_weights = matrix.flatten()
        self.loss_weights = torch.tensor(flat_weights, dtype=torch.float32, device=self.device)

        label = f"macd={gamma_macd}" + (f", delta={gamma_delta}" if self.target_size > 1 else "")
        print(f"Loss Decay Gamma: {label} (normalized weights: {np.round(flat_weights, 3).tolist()})")

    def _forward(self, x: torch.Tensor):
        """Run the model and normalize its output to (forecast, direction_logits_or_None)."""
        out = self.model(x)
        if isinstance(out, tuple):
            return out
        return out, None

    def _reconstruct_deltas_only(self, delta_pred_norm: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
        """
        A2: given the network's raw (batch, forecast_horizon) normalized delta
        predictions and the per-window raw anchor (last known primary-signal
        value), reconstruct a (batch, forecast_horizon * 2) normalized [macd, delta]
        tensor matching the standard target layout, so the rest of the loss/metric
        machinery (weighted MSE, loss_decay_gamma, evaluate()) is unchanged.

        Only valid in 'global' normalization mode (validated at __init__) — self.mean/
        self.std are fixed dataset-wide stats, not per-window.
        """
        delta_pred_raw = delta_pred_norm * float(self.std[1]) + float(self.mean[1])
        macd_pred_raw = anchor.unsqueeze(1) + torch.cumsum(delta_pred_raw, dim=1)
        macd_pred_norm = (macd_pred_raw - float(self.mean[0])) / float(self.std[0])
        combined = torch.stack([macd_pred_norm, delta_pred_norm], dim=-1)
        return combined.reshape(combined.shape[0], -1)

    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        direction_logits: torch.Tensor = None,
        direction_labels: torch.Tensor = None
    ) -> torch.Tensor:
        """Compute loss (weighted MSE if decay weights enabled, standard MSE otherwise),
        plus an auxiliary BCE term on forecast direction when A1 is enabled (A1)."""
        if self.loss_weights is not None:
            mse = torch.mean(self.loss_weights * (predictions - targets) ** 2)
        else:
            mse = self.criterion(predictions, targets)

        if self.use_auxiliary_direction and direction_logits is not None:
            bce = nn.functional.binary_cross_entropy_with_logits(direction_logits, direction_labels)
            return mse + self.auxiliary_lambda * bce
        return mse

    def _drift_baseline(self, last: float, last_delta: float) -> np.ndarray:
        """
        Linear-extrapolation ("drift") persistence baseline for one window.

            macd[t+k]  = macd[t] + (k+1) * delta[t]
            delta[t+k] = delta[t]

        Returns shape (forecast_horizon, target_size) in raw units.
        """
        steps = np.arange(1, self.forecast_horizon + 1, dtype=np.float32)
        drift = np.empty((self.forecast_horizon, self.target_size), dtype=np.float32)
        drift[:, 0] = last + last_delta * steps
        if self.target_size > 1:
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
    ):
        """
        Prepare training sequences from time series data.

        Args:
            data: List of numpy arrays (2D points x input_size, or 1D per symbol) or a single array
            fit: If True (training), compute and store normalization stats.
                 If False (evaluation), reuse existing self.mean/self.std.
            return_drift: If True, also return the drift baseline for each window,
                normalized into the same space the *level* target occupies.

        Returns a 5-tuple (X, y, D, dir_labels, anchor). D is populated when
        residual_target or return_drift is set, else None. dir_labels (A1) is
        populated when self.use_auxiliary_direction, else None. anchor (A2) is
        populated when self.predict_deltas_only, else None.
        """
        all_X, all_y, all_D, all_dir, all_anchor = [], [], [], [], []

        processed_symbols = []
        input_data = data if isinstance(data, list) else [data]

        for symbol_item in input_data:
            if isinstance(symbol_item, np.ndarray) and symbol_item.ndim == 2:
                processed_symbols.append(symbol_item.astype(np.float32))
            else:
                series_1d = np.asarray(symbol_item, dtype=np.float32)
                if self.include_delta:
                    deltas = self._calculate_deltas(series_1d)
                    stacked = np.stack([series_1d, deltas], axis=1)
                    processed_symbols.append(stacked)
                else:
                    processed_symbols.append(series_1d.reshape(-1, 1))

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
                y_raw = symbol_data[i + self.seq_length:i + self.seq_length + self.forecast_horizon, 0:self.target_size]
                
                if self.normalization_type == "internal":
                    # Per-sequence normalization (per feature)
                    m = np.mean(X_raw, axis=0)
                    s = np.std(X_raw, axis=0) + 1e-8
                else:
                    # Dataset-wide normalization
                    m = self.mean
                    s = self.std

                X_norm = (X_raw - m) / s
                s_target = s[:self.target_size]
                m_target = m[:self.target_size]

                if self.residual_target or return_drift:
                    # Drift is computed from RAW values, before any normalization.
                    last_delta = float(X_raw[-1, 0] - X_raw[-2, 0]) if len(X_raw) >= 2 else 0.0
                    drift_raw = self._drift_baseline(
                        last=float(X_raw[-1, 0]),
                        last_delta=last_delta
                    )

                if self.residual_target:
                    y_norm = (y_raw - drift_raw) / s_target
                else:
                    y_norm = (y_raw - m_target) / s_target

                all_X.append(X_norm)
                # Flatten target: (horizon, target_size) -> (horizon * target_size)
                all_y.append(y_norm.flatten())
                if return_drift:
                    all_D.append(((drift_raw - m_target) / s_target).flatten())
                if self.use_auxiliary_direction:
                    # sign(future change) on the primary signal column, independent
                    # of include_delta: day k's label compares y_raw[k] to the prior
                    # day's raw value (the last input point for k=0, else y_raw[k-1]).
                    prev = np.concatenate([X_raw[-1:, 0], y_raw[:-1, 0]])
                    all_dir.append((y_raw[:, 0] - prev > 0).astype(np.float32))
                if self.predict_deltas_only:
                    all_anchor.append(float(X_raw[-1, 0]))

        if not all_X:
            empty_X = torch.empty((0, self.seq_length, self.input_size), dtype=torch.float32)
            empty_y = torch.empty((0, self.output_size), dtype=torch.float32)
            empty_D = torch.empty((0, self.output_size), dtype=torch.float32) if (self.residual_target or return_drift) else None
            empty_dir = torch.empty((0, self.forecast_horizon), dtype=torch.float32) if self.use_auxiliary_direction else None
            empty_anchor = torch.empty((0,), dtype=torch.float32) if self.predict_deltas_only else None
            return empty_X, empty_y, empty_D, empty_dir, empty_anchor

        # Shuffle (keep every collected sequence aligned)
        indices = list(range(len(all_X)))
        np.random.shuffle(indices)
        all_X = [all_X[i] for i in indices]
        all_y = [all_y[i] for i in indices]
        if return_drift:
            all_D = [all_D[i] for i in indices]
        if self.use_auxiliary_direction:
            all_dir = [all_dir[i] for i in indices]
        if self.predict_deltas_only:
            all_anchor = [all_anchor[i] for i in indices]

        # Convert to tensors
        X = torch.tensor(np.array(all_X), dtype=torch.float32)
        y = torch.tensor(np.array(all_y), dtype=torch.float32)
        D = torch.tensor(np.array(all_D), dtype=torch.float32) if return_drift else None
        dir_labels = torch.tensor(np.array(all_dir), dtype=torch.float32) if self.use_auxiliary_direction else None
        anchor = torch.tensor(np.array(all_anchor), dtype=torch.float32) if self.predict_deltas_only else None
        return X, y, D, dir_labels, anchor
    
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
        split_strategy: str = "symbol",
        checkpoint_warmup_epochs: int = 0
    ) -> dict:
        """
        Train the model on MACD data.

        split_strategy: how to carve the validation set — "time" (later dates,
            same symbols) or "symbol" (unseen symbols). Pass the same strategy
            used for the test split so val tracks test.
        checkpoint_warmup_epochs: epochs 1..N are trained normally but ineligible
            to become the "best" checkpoint, and are excluded from the LR
            scheduler's plateau tracking too (so it doesn't anchor to, and decay
            LR against, an early accidental low before training has settled in).
        """
        self.batch_size = batch_size

        train_series, val_series = self._split_series(
            train_data, validation_split, split_strategy
        )

        # fit=True stores the normalization stats — computed on TRAIN ONLY so the
        # validation set contributes nothing to them.
        X_train, y_train, _, dir_train, anchor_train = self.prepare_sequences(train_series, fit=True)
        if val_series:
            X_val, y_val, _, dir_val, anchor_val = self.prepare_sequences(val_series, fit=False)
        else:
            X_val, y_val, dir_val, anchor_val = None, None, None, None

        if len(X_train) == 0:
            raise ValueError(
                f"No training windows produced. Each symbol needs at least "
                f"{self.seq_length + self.forecast_horizon} data points "
                f"(seq_length={self.seq_length} + forecast_horizon={self.forecast_horizon})."
            )

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        if dir_train is not None:
            dir_train = dir_train.to(self.device)
        if anchor_train is not None:
            anchor_train = anchor_train.to(self.device)

        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
            if dir_val is not None:
                dir_val = dir_val.to(self.device)
            if anchor_val is not None:
                anchor_val = anchor_val.to(self.device)
            if verbose:
                axis = ("later dates, same symbols" if split_strategy == "time"
                        else "unseen symbols")
                print(f"Train windows: {len(X_train)}, Val windows: {len(X_val)} "
                      f"(split before windowing — no overlap; val measures {axis})")
        else:
            print("WARNING: no validation set could be built; "
                  "selecting the checkpoint on train loss instead.")

        if checkpoint_warmup_epochs >= epochs:
            print(f"WARNING: checkpoint_warmup_epochs ({checkpoint_warmup_epochs}) >= epochs "
                  f"({epochs}); no epoch will ever be eligible as 'best' — the model will end "
                  f"training on its final-epoch weights, unselected.")

        history = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state = None
        best_epoch = 0
        interrupted = False

        try:
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
                    batch_dir = dir_train[batch_idx] if dir_train is not None else None
                    batch_anchor = anchor_train[batch_idx] if anchor_train is not None else None

                    self.optimizer.zero_grad()
                    predictions, direction_logits = self._forward(batch_X)
                    if self.predict_deltas_only:
                        predictions = self._reconstruct_deltas_only(predictions, batch_anchor)
                    loss = self._compute_loss(predictions, batch_y, direction_logits, batch_dir)
                    loss.backward()
                    self.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                train_loss = total_loss / num_batches

                # Validation
                if has_val:
                    self.model.eval()
                    with torch.no_grad():
                        val_pred, val_dir_logits = self._forward(X_val)
                        if self.predict_deltas_only:
                            val_pred = self._reconstruct_deltas_only(val_pred, anchor_val)
                        val_loss = self._compute_loss(val_pred, y_val, val_dir_logits, dir_val).item()
                else:
                    val_loss = train_loss

                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                selection_loss = val_loss if has_val else train_loss
                past_warmup = (epoch + 1) > checkpoint_warmup_epochs
                if past_warmup and selection_loss < best_val_loss:
                    best_val_loss = selection_loss
                    best_epoch = epoch + 1
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    if verbose:
                        ts = datetime.now().strftime("%H:%M:%S")
                        metric = "val" if has_val else "train"
                        print(f"[{ts}] New best checkpoint: epoch {best_epoch}/{epochs} ({metric} loss {best_val_loss:.6f})")

                if self.scheduler is not None and past_warmup:
                    self.scheduler.step(selection_loss)

                if verbose and (epoch + 1) % 10 == 0:
                    ts = datetime.now().strftime("%H:%M:%S")
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    print(f"[{ts}] Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, LR: {current_lr:.6g}")
        except KeyboardInterrupt:
            interrupted = True
            completed = len(history["train_loss"])
            print(f"\nInterrupted after {completed} epoch(s). "
                  f"{'Falling back to the best checkpoint found so far.' if best_state is not None else 'No checkpoint was eligible yet (still in warmup) — keeping current in-progress weights.'}")

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

        history["interrupted"] = interrupted
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
        X_test, y_test, D_test, _, anchor_test = self.prepare_sequences(
            test_data, fit=False, return_drift=True
        )

        if len(X_test) == 0:
            return {"error": "Not enough test data"}

        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        D_test = D_test.to(self.device)
        if anchor_test is not None:
            anchor_test = anchor_test.to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions, _ = self._forward(X_test)
            if self.predict_deltas_only:
                predictions = self._reconstruct_deltas_only(predictions, anchor_test)

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
            p = pred_t.cpu().numpy().reshape(-1, self.forecast_horizon, self.target_size)
            a = actual_level.cpu().numpy().reshape(-1, self.forecast_horizon, self.target_size)
            X_np = X_test.cpu().numpy()
            out = []
            for f in range(self.target_size):
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

            labels = ["MACD", "Delta"] if self.target_size > 1 else ["Signal"]
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
        Input sequence shape: (seq_length, input_size) or (seq_length,) for 1D.
        Returns: forecasted values (forecast_horizon, target_size)
        """
        self.model.eval()
        seq = np.asarray(sequence, dtype=np.float32)

        if seq.ndim == 2:
            if len(seq) > self.seq_length:
                seq = seq[-self.seq_length:]
            elif len(seq) < self.seq_length:
                padding = np.repeat(seq[0:1, :], self.seq_length - len(seq), axis=0)
                seq = np.vstack([padding, seq])
            input_features = seq
        else:
            # 1D input (legacy single feature / delta)
            if len(seq) > self.seq_length:
                prev_value = float(seq[-(self.seq_length + 1)])
                seq = seq[-self.seq_length:]
            elif len(seq) < self.seq_length:
                print(f"WARNING: input has {len(seq)} points but the model needs "
                      f"{self.seq_length}; padding {self.seq_length - len(seq)} synthetic steps.")
                padding = np.full(self.seq_length - len(seq), seq[0], dtype=np.float32)
                seq = np.concatenate([padding, seq])
                prev_value = None
            if self.include_delta:
                deltas = self._calculate_deltas(seq)
                if prev_value is not None:
                    deltas[0] = seq[0] - prev_value
                input_features = np.stack([seq, deltas], axis=1)
            else:
                input_features = seq.reshape(-1, 1)

        if self.normalization_type == "internal":
            m = np.mean(input_features, axis=0)
            s = np.std(input_features, axis=0) + 1e-8
        else:
            m = self.mean
            s = self.std

        normalized = (input_features - m) / s
        x = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred, _ = self._forward(x)

        if self.predict_deltas_only:
            # A2: network output is delta-only; reconstruct the primary signal as
            # anchor + cumsum(deltas). Requires normalization_type='global' (validated
            # at __init__), so s/m here are the fixed dataset-wide stats.
            delta_pred_norm = pred.cpu().numpy()[0].reshape(self.forecast_horizon)
            delta_pred_raw = delta_pred_norm * s[1] + m[1]
            anchor = float(input_features[-1, 0])
            macd_pred_raw = anchor + np.cumsum(delta_pred_raw)
            return np.stack([macd_pred_raw, delta_pred_raw], axis=1)

        pred_np = pred.cpu().numpy()[0].reshape(self.forecast_horizon, self.target_size)
        s_target = s[:self.target_size] if isinstance(s, np.ndarray) else s
        m_target = m[:self.target_size] if isinstance(m, np.ndarray) else m

        if self.residual_target:
            last = float(input_features[-1, 0])
            last_delta = float(input_features[-1, 0] - input_features[-2, 0]) if len(input_features) >= 2 else 0.0
            drift_raw = self._drift_baseline(last, last_delta)
            forecast_all = drift_raw + pred_np * s_target
        else:
            forecast_all = pred_np * s_target + m_target

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
            "loss_decay_gamma": self.loss_decay_gamma,
            "loss_decay_gamma_delta": self.loss_decay_gamma_delta,
            "feature_names": self.feature_names,
            "input_size": self.input_size,
            "target_size": self.target_size,
            "network_target_size": self.network_target_size,
            "auxiliary_direction_lambda": self.auxiliary_lambda,
            "predict_deltas_only": self.predict_deltas_only,
            "seed": self.seed
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and normalization parameters."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.include_delta = checkpoint.get("include_delta", False)
        self.residual_target = checkpoint.get("residual_target", False)
        self.loss_decay_gamma = checkpoint.get("loss_decay_gamma", None)
        self.loss_decay_gamma_delta = checkpoint.get("loss_decay_gamma_delta", None)
        self.seed = checkpoint.get("seed", None)
        self.feature_names = checkpoint.get("feature_names", ["macd", "delta"] if self.include_delta else ["macd"])
        self.input_size = checkpoint.get("input_size", len(self.feature_names))
        self.target_size = checkpoint.get("target_size", 2 if self.include_delta else 1)
        self.seq_length = checkpoint["seq_length"]
        self.forecast_horizon = checkpoint["forecast_horizon"]
        self.output_size = self.forecast_horizon * self.target_size
        self.normalization_type = checkpoint.get("normalization_type", "global")
        self.auxiliary_lambda = checkpoint.get("auxiliary_direction_lambda", None)
        self.use_auxiliary_direction = self.auxiliary_lambda is not None and self.auxiliary_lambda > 0
        self.predict_deltas_only = checkpoint.get("predict_deltas_only", False)
        self.network_target_size = checkpoint.get(
            "network_target_size", 1 if self.predict_deltas_only else self.target_size
        )

        self._build_loss_weights()

        # Re-initialize model with correct input size and target size
        self.model = create_model(
            architecture=checkpoint.get("architecture", self.architecture),
            input_size=self.input_size,
            target_size=self.network_target_size,
            hidden_size=checkpoint.get("hidden_size", 64),
            num_layers=checkpoint.get("num_layers", 2),
            forecast_horizon=self.forecast_horizon,
            auxiliary_direction=self.use_auxiliary_direction
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        raw_mean = checkpoint.get("mean", 0.0)
        raw_std = checkpoint.get("std", 1.0)
        if isinstance(raw_mean, (int, float)):
            self.mean = np.full(self.input_size, raw_mean, dtype=np.float32)
        else:
            self.mean = np.array(raw_mean, dtype=np.float32)
            if len(self.mean) < self.input_size:
                self.mean = np.pad(self.mean, (0, self.input_size - len(self.mean)))

        if isinstance(raw_std, (int, float)):
            self.std = np.full(self.input_size, raw_std, dtype=np.float32)
        else:
            self.std = np.array(raw_std, dtype=np.float32)
            if len(self.std) < self.input_size:
                self.std = np.pad(self.std, (0, self.input_size - len(self.std)), constant_values=1.0)
        
        print(f"Model loaded from {path} (Features: {self.feature_names}, Norm: {self.normalization_type})")
    
    def export_to_coreml(self, output_path: str) -> str:
        """
        Export the model to Core ML format.
        """
        import coremltools as ct
        
        self.model.eval()
        self.model.to("cpu")
        example_input = torch.randn(1, self.seq_length, self.input_size)

        # Core ML export always traces a single-output module: when A1's auxiliary
        # direction head is enabled, self.model(x) returns (forecast, direction_logits)
        # for training, but only the magnitude forecast is served in production.
        if self.use_auxiliary_direction:
            class _ForecastOnly(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(x)[0]

            export_module = _ForecastOnly(self.model).eval()
        else:
            export_module = self.model

        traced_model = torch.jit.trace(export_module, example_input)

        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=(1, self.seq_length, self.input_size), name="input_sequence")],
            outputs=[ct.TensorType(name="forecast")],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.ALL
        )
        
        mlmodel.author = "Tick Scanner"
        mlmodel.short_description = f"{self.architecture} MACD forecaster (Features: {', '.join(self.feature_names)}, Norm: {self.normalization_type})"
        mlmodel.version = "1.3"
        
        mlmodel.user_defined_metadata["normalization_type"] = self.normalization_type
        mlmodel.user_defined_metadata["mean"] = str(self.mean.tolist())
        mlmodel.user_defined_metadata["std"] = str(self.std.tolist())
        mlmodel.user_defined_metadata["seq_length"] = str(self.seq_length)
        mlmodel.user_defined_metadata["forecast_horizon"] = str(self.forecast_horizon)
        mlmodel.user_defined_metadata["include_delta"] = str(self.include_delta)
        mlmodel.user_defined_metadata["residual_target"] = str(self.residual_target)
        mlmodel.user_defined_metadata["feature_names"] = ",".join(self.feature_names)
        mlmodel.user_defined_metadata["input_size"] = str(self.input_size)
        mlmodel.user_defined_metadata["target_size"] = str(self.target_size)
        mlmodel.user_defined_metadata["predict_deltas_only"] = str(self.predict_deltas_only)
        mlmodel.user_defined_metadata["seed"] = str(self.seed)
        mlmodel.user_defined_metadata["hidden_size"] = str(self.model.hidden_size)
        mlmodel.user_defined_metadata["num_layers"] = str(self.model.num_layers)
        if self.batch_size:
            mlmodel.user_defined_metadata["batch_size"] = str(self.batch_size)
        
        mlmodel.save(output_path)
        print(f"Core ML model saved to {output_path}")
        
        return output_path


def get_latest_model_version(models_dir: str, model_name: str) -> int:
    """Highest existing version for model_name, or 0 if none exist yet."""
    if not os.path.isdir(models_dir):
        return 0
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)\.(pt|mlpackage)$")
    versions = {int(m.group(1)) for f in os.listdir(models_dir) if (m := pattern.match(f))}
    return max(versions) if versions else 0


def get_model_path(signal_type: str = "macd", architecture: str = "stacked_lstm",
                    model_name: str = None, version: int = None) -> str:
    """
    Get the path to the Core ML model.

    When `model_name` is given, resolves to `models/{model_name}_{version}.mlpackage`
    instead of the legacy `{signal_type}_{architecture}_forecaster.mlpackage` naming.
    If `version` is omitted, resolves to the latest existing version for that name
    (0 if none exist yet) — callers that need the *next* version (i.e. when saving a
    freshly trained model) must resolve it themselves via `get_latest_model_version`
    and pass it explicitly, so both the `.pt` and `.mlpackage` of one training run
    land on the same version number.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, "models")
    if model_name:
        resolved_version = version if version is not None else get_latest_model_version(models_dir, model_name)
        return os.path.join(models_dir, f"{model_name}_{resolved_version}.mlpackage")
    return os.path.join(models_dir, f"{signal_type}_{architecture}_forecaster.mlpackage")


def get_pytorch_model_path(signal_type: str = "macd", architecture: str = "stacked_lstm",
                            model_name: str = None, version: int = None) -> str:
    """
    Get the path to the PyTorch model checkpoint.

    See `get_model_path` for the `model_name`/`version` resolution rules.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    models_dir = os.path.join(base_dir, "models")
    if model_name:
        resolved_version = version if version is not None else get_latest_model_version(models_dir, model_name)
        return os.path.join(models_dir, f"{model_name}_{resolved_version}.pt")
    return os.path.join(models_dir, f"{signal_type}_{architecture}_forecaster.pt")
