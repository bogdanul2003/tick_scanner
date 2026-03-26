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

logger = logging.getLogger(__name__)

# Cache for loaded models
_model_cache: Dict[str, Any] = {}


class CoreMLForecaster:
    """
    Core ML-based forecaster that runs on Apple Neural Engine (NPU).
    
    This class provides fast inference for MACD predictions using
    a pre-trained LSTM model compiled for Core ML.
    """
    
    def __init__(self, model_path: Optional[str] = None, signal_type: str = "macd"):
        """
        Initialize the Core ML forecaster.
        
        Args:
            model_path: Path to the .mlpackage model file.
                       If None, uses the default model path based on signal_type.
            signal_type: Type of signal model - "macd" or "signal_line"
        """
        self.model = None
        self.mean = 0.0
        self.std = 1.0
        self.seq_length = 30
        self.forecast_horizon = 5
        self.signal_type = signal_type
        
        if model_path is None:
            from models.lstm_forecaster import get_model_path
            model_path = get_model_path(signal_type)
        
        self.model_path = model_path
        self._load_model()
    
    def _load_model(self):
        """Load the Core ML model."""
        global _model_cache
        
        if self.model_path in _model_cache:
            cached = _model_cache[self.model_path]
            self.model = cached["model"]
            self.mean = cached["mean"]
            self.std = cached["std"]
            self.seq_length = cached["seq_length"]
            self.forecast_horizon = cached["forecast_horizon"]
            logger.info(f"Loaded Core ML model from cache")
            return
        
        if not os.path.exists(self.model_path):
            logger.warning(f"Core ML model not found at {self.model_path}")
            return
        
        try:
            import coremltools as ct
            
            # Load the model
            self.model = ct.models.MLModel(self.model_path)
            
            # Load normalization parameters from metadata
            metadata = self.model.user_defined_metadata
            self.mean = float(metadata.get("mean", 0.0))
            self.std = float(metadata.get("std", 1.0))
            self.seq_length = int(metadata.get("seq_length", 30))
            self.forecast_horizon = int(metadata.get("forecast_horizon", 5))
            
            # Cache the model
            _model_cache[self.model_path] = {
                "model": self.model,
                "mean": self.mean,
                "std": self.std,
                "seq_length": self.seq_length,
                "forecast_horizon": self.forecast_horizon
            }
            
            logger.info(f"Loaded Core ML model from {self.model_path}")
            logger.info(f"  - Sequence length: {self.seq_length}")
            logger.info(f"  - Forecast horizon: {self.forecast_horizon}")
            logger.info(f"  - Normalization: mean={self.mean:.4f}, std={self.std:.4f}")
            
        except ImportError:
            logger.error("coremltools not installed. Install with: pip install coremltools")
        except Exception as e:
            logger.error(f"Failed to load Core ML model: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if the model is loaded and ready."""
        return self.model is not None
    
    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """
        Make a prediction using the Core ML model (runs on NPU).
        
        Args:
            sequence: 1D array of MACD values (length = seq_length)
            
        Returns:
            Forecasted values array (length = forecast_horizon)
        """
        if not self.is_available:
            raise RuntimeError("Core ML model not loaded")
        
        # Ensure correct length
        if len(sequence) < self.seq_length:
            # Pad with the first value if too short
            padding = np.full(self.seq_length - len(sequence), sequence[0])
            sequence = np.concatenate([padding, sequence])
        elif len(sequence) > self.seq_length:
            # Take the most recent values
            sequence = sequence[-self.seq_length:]
        
        # Normalize
        normalized = (sequence - self.mean) / self.std
        
        # Reshape for model input: (1, seq_length, 1)
        input_data = normalized.reshape(1, self.seq_length, 1).astype(np.float32)
        
        # Run inference on NPU
        output = self.model.predict({"input_sequence": input_data})
        
        # Get forecast and denormalize
        forecast = output["forecast"][0]
        forecast = forecast * self.std + self.mean
        
        return forecast
    
    def predict_batch(self, sequences: List[np.ndarray]) -> List[np.ndarray]:
        """
        Make predictions for multiple sequences (batch inference).
        
        Args:
            sequences: List of 1D arrays of MACD values
            
        Returns:
            List of forecasted value arrays
        """
        return [self.predict(seq) for seq in sequences]


class NeuralForecastService:
    """
    Service for neural network-based MACD forecasting using NPU.
    
    This service provides a drop-in replacement for ARIMA-based forecasting,
    using a pre-trained LSTM model running on Apple's Neural Engine.
    """
    
    def __init__(
        self, 
        model_path: Optional[str] = None, 
        fallback_to_arima: bool = True,
        signal_type: str = "macd"
    ):
        """
        Initialize the neural forecast service.
        
        Args:
            model_path: Path to Core ML model (uses default if None)
            fallback_to_arima: If True, fall back to ARIMA when NPU unavailable
            signal_type: Type of signal model - "macd" or "signal_line"
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
        days_past: int = 30,
        forecast_days: int = 5
    ) -> Dict[str, Any]:
        """
        Forecast if MACD/Signal Line will become positive using neural network (NPU).
        
        Args:
            symbol: Stock symbol
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast
            
        Returns:
            Forecast result dictionary compatible with ARIMA format
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
            # Get historical data
            end_date = get_latest_market_date()
            start_date = end_date - timedelta(days=days_past)
            macd_data = get_macd_for_range(symbol, start_date, end_date)
            
            series = np.array([
                d[field_name] for d in macd_data 
                if field_name in d and d[field_name] is not None
            ])
            
            if len(series) < 10:
                return {
                    "will_become_positive": False,
                    "forecasted_values": [],
                    "details": {"error": f"Not enough {signal_label} data"}
                }
            
            # Run neural prediction on NPU
            forecast = self.forecaster.predict(series)
            forecasted_values = forecast.tolist()
            
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
                    "signal_type": self.signal_type
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
        days_past: int = 30,
        forecast_days: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Forecast for multiple symbols using neural network.
        
        Args:
            symbols: List of stock symbols
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast
            
        Returns:
            Dictionary mapping symbols to forecast results
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
    
    Returns:
        Dictionary with NPU availability info
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
