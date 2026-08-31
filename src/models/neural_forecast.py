"""
Core ML Neural Forecaster for NPU inference on Apple Silicon.

This module provides a high-performance inference engine using Core ML
to run MACD forecasts on the Apple Neural Engine (NPU).
"""
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import ast

logger = logging.getLogger(__name__)

# Cache for loaded models
_model_cache: Dict[str, Any] = {}


def _trailing_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing (inclusive) rolling mean; NaN wherever fewer than `window` points precede it."""
    n = len(values)
    out = np.full(n, np.nan, dtype=np.float32)
    if n < window:
        return out
    csum = np.cumsum(np.insert(values.astype(np.float64), 0, 0.0))
    out[window - 1:] = ((csum[window:] - csum[:-window]) / window).astype(np.float32)
    return out


class CoreMLForecaster:
    """
    Core ML-based forecaster that runs on Apple Neural Engine (NPU).
    """
    
    def __init__(self, model_path: Optional[str] = None, signal_type: str = "macd"):
        """
        Initialize the Core ML forecaster.
        
        Args:
            model_path: Path to the .mlpackage model file.
            signal_type: Type of signal model - "macd" or "signal_line"
        """
        self.model = None
        self.mean = 0.0
        self.std = 1.0
        self.seq_length = 30
        self.forecast_horizon = 5
        self.normalization_type = "unknown"
        self.signal_type = signal_type
        self.include_delta = False
        self.residual_target = False
        self.feature_names = ["macd"]
        self.input_size = 1
        self.target_size = 1
        self.hidden_size = None
        self.num_layers = None
        self.batch_size = None

        if model_path is None:
            from models.lstm_forecaster import get_model_path
            model_path = get_model_path(signal_type)
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the Core ML model."""
        global _model_cache

        cached = _model_cache.get(self.model_path)
        if cached is not None:
            # The cache entry IS the attribute set produced by _read_metadata,
            # so restoring it wholesale keeps the cached instance identical to a
            # freshly parsed one. Adding a field to _read_metadata cannot leave
            # this path half-initialized the way an explicit copy-out could.
            self.__dict__.update(cached)
            logger.info(f"Loaded Core ML model from cache")
            return

        if not os.path.exists(self.model_path):
            logger.warning(f"Core ML model not found at {self.model_path}")
            return

        try:
            import coremltools as ct

            model = ct.models.MLModel(self.model_path)
            attrs = self._read_metadata(model)
        except ImportError:
            logger.error("coremltools not installed. Install with: pip install coremltools")
            return
        except Exception as e:
            # Nothing has been applied to self yet, so a parse failure leaves the
            # forecaster unavailable (is_available stays False) and callers fall
            # back to ARIMA rather than predicting with half-parsed metadata.
            logger.error(f"Failed to load Core ML model: {e}")
            return

        _model_cache[self.model_path] = attrs
        self.__dict__.update(attrs)

        logger.info(f"Loaded Core ML model from {self.model_path}")
        logger.info(f"  - Features: {', '.join(self.feature_names)} (Count: {self.input_size})")
        logger.info(f"  - Targets: {self.target_size}")
        logger.info(f"  - Sequence length: {self.seq_length}")
        logger.info(f"  - Forecast horizon: {self.forecast_horizon}")

    @staticmethod
    def _read_metadata(model) -> Dict[str, Any]:
        """
        Parse a loaded Core ML model into the attribute dict that both the
        instance and the module-level cache are populated from.

        The returned keys ARE attribute names: whatever is added here is restored
        on a cache hit for free.
        """
        metadata = model.user_defined_metadata

        # Handle list-based mean/std if multi-variate
        mean_str = metadata.get("mean", "0.0")
        std_str = metadata.get("std", "1.0")

        if mean_str.startswith("["):
            mean = np.array(ast.literal_eval(mean_str), dtype=np.float32)
            std = np.array(ast.literal_eval(std_str), dtype=np.float32)
        else:
            mean = float(mean_str)
            std = float(std_str)

        include_delta = metadata.get("include_delta", "False").lower() == "true"

        # Feature names and sizes
        feature_names_str = metadata.get("feature_names", "")
        if feature_names_str:
            feature_names = [f.strip().lower() for f in feature_names_str.split(",") if f.strip()]
        else:
            feature_names = ["macd", "delta"] if include_delta else ["macd"]

        # Additional model details if present
        try:
            hidden_size = int(metadata.get("hidden_size", 0))
            num_layers = int(metadata.get("num_layers", 0))
            batch_size_str = metadata.get("batch_size")
            batch_size = int(batch_size_str) if batch_size_str else None
        except (ValueError, TypeError):
            hidden_size = None
            num_layers = None
            batch_size = None

        return {
            "model": model,
            "mean": mean,
            "std": std,
            "seq_length": int(metadata.get("seq_length", 30)),
            "forecast_horizon": int(metadata.get("forecast_horizon", 5)),
            "normalization_type": metadata.get("normalization_type", "global"),
            "include_delta": include_delta,
            "residual_target": metadata.get("residual_target", "False").lower() == "true",
            "feature_names": feature_names,
            "input_size": int(metadata.get("input_size", len(feature_names))),
            "target_size": int(metadata.get("target_size", 2 if include_delta else 1)),
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_size": batch_size,
        }

    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None
    
    def _calculate_deltas(self, series: np.ndarray) -> np.ndarray:
        """Calculate deltas (today - yesterday)."""
        deltas = np.zeros_like(series)
        deltas[1:] = series[1:] - series[:-1]
        return deltas

    def predict(self, sequence: np.ndarray, prev_value: float = None) -> np.ndarray:
        """
        Make a prediction using the Core ML model (runs on NPU).
        Input sequence shape: (seq_length, input_size) or (seq_length,) for 1D.
        Returns: forecasted values (forecast_horizon, target_size)
        """
        if not self.is_available:
            raise RuntimeError("Core ML model not loaded")

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
                logger.warning(
                    "Input has %d points but the model needs %d; padding %d synthetic "
                    "steps. Increase days_past.",
                    len(seq), self.seq_length, self.seq_length - len(seq)
                )
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

        # Determine normalization parameters
        if self.normalization_type == "internal":
            m = np.mean(input_features, axis=0)
            s = np.std(input_features, axis=0) + 1e-8
        else:
            m = self.mean
            s = self.std

        # Normalize
        normalized = (input_features - m) / s
        
        # Reshape for model input: (1, seq_length, input_size)
        input_data = normalized.reshape(1, self.seq_length, self.input_size).astype(np.float32)
        
        # Run inference on NPU
        output = self.model.predict({"input_sequence": input_data})
        
        # Get forecast and denormalize (horizon, target_size)
        forecast_raw = output["forecast"][0].reshape(self.forecast_horizon, self.target_size)
        s_target = s[:self.target_size] if isinstance(s, np.ndarray) else s
        m_target = m[:self.target_size] if isinstance(m, np.ndarray) else m

        if self.residual_target:
            last = float(input_features[-1, 0])
            last_delta = float(input_features[-1, 0] - input_features[-2, 0]) if len(input_features) >= 2 else 0.0
            steps = np.arange(1, self.forecast_horizon + 1, dtype=np.float32)
            drift = np.empty((self.forecast_horizon, self.target_size), dtype=np.float32)
            drift[:, 0] = last + last_delta * steps
            if self.target_size > 1:
                drift[:, 1] = last_delta
            forecast_denorm = drift + forecast_raw * s_target
        else:
            forecast_denorm = forecast_raw * s_target + m_target

        return forecast_denorm
    
    def predict_batch(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Make predictions for multiple sequences (batch inference).
        """
        return [self.predict(seq) for seq in sequences]


class NeuralForecastService:
    """
    Service for neural network-based MACD forecasting using NPU.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        fallback_to_arima: bool = True,
        signal_type: str = "macd"
    ):
        """
        Initialize the neural forecast service.
        """
        self.fallback_to_arima = fallback_to_arima
        self.signal_type = signal_type
        self.forecaster: Optional[CoreMLForecaster] = None
        
        try:
            self.forecaster = CoreMLForecaster(model_path, signal_type)
        except Exception as e:
            logger.warning(f"Failed to initialize Core ML forecaster: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if neural forecasting is available."""
        return self.forecaster is not None and self.forecaster.is_available
    
    def forecast_macd(
        self,
        symbol: str,
        days_past: int = 100,   # calendar days; matches config.forecast_days_past
        forecast_days: int = 5
    ) -> Dict[str, Any]:
        """
        Forecast if MACD/Signal Line will become positive using neural network (NPU).
        """
        # Import here to avoid circular imports
        from macd_utils import get_macd_for_range, get_latest_market_date
        
        # Determine which field to use based on signal_type
        field_name = "macd" if self.signal_type == "macd" else "signal_line"
        signal_label = "MACD" if self.signal_type == "macd" else "Signal Line"
        
        # If NPU not available, fall back to ARIMA
        if not self.is_available:
            if self.fallback_to_arima:
                logger.info(f"Neural forecaster unavailable, falling back to ARIMA for {symbol}")
                from forecast_utils import arima_macd_positive_forecast
                return arima_macd_positive_forecast(symbol, days_past, forecast_days)
            else:
                return {
                    "will_become_positive": False,
                    "forecasted_values": [],
                    "details": {"error": "Neural forecaster not available"}
                }
        
        try:
            # Get historical data.
            # days_past is CALENDAR days but seq_length is TRADING days (~1.45x
            # ratio; 1.6 for safety). Sizing the window off days_past alone lets a
            # small value silently underfeed the model, and predict() then pads
            # with synthetic points. Widening is free: predict() truncates to the
            # last seq_length points anyway, and it gives predict() the extra
            # observation it needs to reconstruct delta[0].
            end_date = get_latest_market_date()
            has_volume_feature = any(f.lower() == "volume" for f in self.forecaster.feature_names)
            needed = self.forecaster.seq_length + 6 + (20 if has_volume_feature else 0)
            calendar_days = max(days_past, int(needed * 1.6) + 1)
            start_date = end_date - timedelta(days=calendar_days)
            macd_data = get_macd_for_range(symbol, start_date, end_date)
            
            if len(self.forecaster.feature_names) > (2 if self.forecaster.include_delta else 1):
                # Multi-feature input
                cols = []
                for f in self.forecaster.feature_names:
                    f_lower = f.lower()
                    if f_lower in ("macd", "signal_line"):
                        # Read the column this feature NAMES, not the service's
                        # primary signal: a MACD model carrying signal_line as an
                        # extra feature must get real Signal_Line values here.
                        # get_training_data maps the two independently, so keying
                        # off field_name fed the model the primary column twice.
                        cols.append(np.array(
                            [float(d[f_lower]) if d.get(f_lower) is not None else 0.0
                             for d in macd_data
                             if field_name in d and d[field_name] is not None],
                            dtype=np.float32
                        ))
                    elif f_lower == "delta":
                        primary_vals = [d[field_name] for d in macd_data if field_name in d and d[field_name] is not None]
                        deltas = np.zeros(len(primary_vals), dtype=np.float32)
                        deltas[1:] = np.array(primary_vals[1:]) - np.array(primary_vals[:-1])
                        cols.append(deltas)
                    elif f_lower == "open-close":
                        o_vals = np.array([float(d.get("open", 0.0) or 0.0) for d in macd_data if field_name in d and d[field_name] is not None], dtype=np.float32)
                        c_vals = np.array([float(d.get("close", 0.0) or 0.0) for d in macd_data if field_name in d and d[field_name] is not None], dtype=np.float32)
                        cols.append(np.where(o_vals != 0, (c_vals - o_vals) / o_vals, 0.0).astype(np.float32))
                    elif f_lower == "volume":
                        # Relative volume (today's volume / trailing 20-day average) —
                        # see train_forecast_model.py's get_training_data for rationale.
                        vol_vals = np.array([float(d.get("volume", 0.0) or 0.0) for d in macd_data if field_name in d and d[field_name] is not None], dtype=np.float32)
                        vol_avg = _trailing_mean(vol_vals, 20)
                        cols.append(np.where(vol_avg > 0, vol_vals / vol_avg, np.nan).astype(np.float32))
                    else:
                        cols.append(np.array([float(d.get(f_lower, 0.0) or 0.0) for d in macd_data if field_name in d and d[field_name] is not None], dtype=np.float32))
                series_input = np.column_stack(cols).astype(np.float32)
                if len(series_input) < 10:
                    return {
                        "will_become_positive": False,
                        "forecasted_values": [],
                        "details": {"error": f"Not enough {signal_label} data"}
                    }
                forecast = self.forecaster.predict(series_input)
                series = series_input[:, 0]
            else:
                series = np.array([
                    d[field_name] for d in macd_data 
                    if field_name in d and d[field_name] is not None
                ], dtype=np.float32)
                
                if len(series) < 10:
                    return {
                        "will_become_positive": False,
                        "forecasted_values": [],
                        "details": {"error": f"Not enough {signal_label} data"}
                    }
                
                # Run neural prediction on NPU
                forecast = self.forecaster.predict(series)
            
            # forecast shape is (horizon, target_size); extract MACD column (index 0)
            macd_forecast = forecast[:, 0] if forecast.ndim > 1 else forecast.flatten()
            forecasted_values = macd_forecast.tolist()

            # Determine if signal will become positive
            last_value = float(series[-1])
            will_become_positive = (
                last_value < 0 and any(v > 0 for v in forecasted_values)
            )
            
            # Build response with appropriate field names
            forecast_key = "forecasted_macd" if self.signal_type == "macd" else "forecasted_signal_line"
            last_key = "last_macd" if self.signal_type == "macd" else "last_signal_line"
            
            return {
                "will_become_positive": will_become_positive,
                forecast_key: {
                    f"Day {i+1}": float(v) for i, v in enumerate(forecasted_values)
                },
                "details": {
                    last_key: last_value,
                    "inference_engine": "Core ML NPU",
                    "model_type": "LSTM",
                    "signal_type": self.signal_type,
                    "features": self.forecaster.input_size
                }
            }
            
        except Exception as e:
            logger.error(f"Neural forecast failed for {symbol}: {e}")
            
            if self.fallback_to_arima:
                logger.info(f"Falling back to ARIMA for {symbol}")
                from forecast_utils import arima_macd_positive_forecast
                return arima_macd_positive_forecast(symbol, days_past, forecast_days)
            
            forecast_key = "forecasted_macd" if self.signal_type == "macd" else "forecasted_signal_line"
            return {
                "will_become_positive": False,
                forecast_key: [],
                "details": {"error": str(e), "signal_type": self.signal_type}
            }
    
    def forecast_batch(
        self,
        symbols: List[str],
        days_past: int = 100,   # calendar days; matches config.forecast_days_past
        forecast_days: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Forecast for multiple symbols using neural network.
        """
        results = {}
        
        for symbol in symbols:
            results[symbol.upper()] = self.forecast_macd(
                symbol, days_past, forecast_days
            )
        
        return results


def check_npu_availability() -> Dict[str, Any]:
    """
    Check if Apple Neural Engine (NPU) is available and get device info.
    """
    info = {
        "npu_available": False,
        "coreml_available": False,
        "mps_available": False,
        "device": "cpu"
    }
    
    try:
        import coremltools
        info["coreml_available"] = True
        info["coreml_version"] = coremltools.__version__
    except ImportError:
        pass
    
    try:
        import torch
        if torch.backends.mps.is_available():
            info["mps_available"] = True
            info["device"] = "mps"
    except ImportError:
        pass
    
    # Check for Apple Silicon
    import platform
    if platform.processor() == "arm":
        info["apple_silicon"] = True
        info["npu_available"] = info["coreml_available"]
    else:
        info["apple_silicon"] = False
    
    return info


# Default service instance (lazy-loaded)
_neural_forecast_service: Optional[NeuralForecastService] = None


def get_neural_forecast_service() -> NeuralForecastService:
    """Get the singleton neural forecast service instance."""
    global _neural_forecast_service
    if _neural_forecast_service is None:
        _neural_forecast_service = NeuralForecastService()
    return _neural_forecast_service
