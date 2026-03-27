"""Forecast service for ARIMA and Neural Network predictions."""
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Check NPU availability at import time
_npu_available = False
_neural_service = None

def _check_npu():
    """Check if NPU-based forecasting is available."""
    global _npu_available, _neural_service
    try:
        from models.neural_forecast import NeuralForecastService, check_npu_availability
        info = check_npu_availability()
        _npu_available = info.get("npu_available", False)
        if _npu_available:
            _neural_service = NeuralForecastService(fallback_to_arima=True)
            logger.info("NPU-based neural forecasting available")
    except Exception as e:
        logger.debug(f"NPU forecasting not available: {e}")


class ForecastService:
    """Service for running ARIMA and Neural Network forecasts on stock data."""
    
    def __init__(self, max_workers: int = None, use_neural: bool = True):
        """
        Initialize the forecast service.
        
        Args:
            max_workers: Maximum number of parallel workers for forecasting.
                        Defaults to CPU count.
            use_neural: Whether to use NPU-based neural forecasting when available.
        """
        self.max_workers = max_workers or os.cpu_count() or 4
        self.use_neural = use_neural
        
        # Initialize NPU check
        if use_neural:
            _check_npu()
    
    @property
    def neural_available(self) -> bool:
        """Check if neural forecasting is available."""
        return _npu_available and _neural_service is not None
    
    def run_macd_forecast(
        self, 
        symbol: str, 
        days_past: int, 
        forecast_days: int,
        engine: str = "auto"
    ) -> Dict[str, Any]:
        """
        Run forecast for MACD becoming positive.
        
        Args:
            symbol: Stock symbol
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            engine: Forecasting engine - "auto", "neural", or "arima"
            
        Returns:
            Forecast result dictionary
        """
        # Determine which engine to use
        use_neural = (
            engine == "neural" or 
            (engine == "auto" and self.use_neural and self.neural_available)
        )
        
        if use_neural and _neural_service is not None:
            try:
                return _neural_service.forecast_macd(symbol.upper(), days_past, forecast_days)
            except Exception as e:
                logger.warning(f"Neural forecast failed for {symbol}, falling back to ARIMA: {e}")
        
        # Fall back to ARIMA
        from forecast_utils import arima_macd_positive_forecast
        try:
            return arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
        except Exception as e:
            logger.error(f"MACD forecast failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def run_ma_forecast(
        self, 
        symbol: str, 
        days_past: int, 
        forecast_days: int
    ) -> Dict[str, Any]:
        """
        Run ARIMA forecast for MA20 crossing above MA50.
        
        Args:
            symbol: Stock symbol
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            
        Returns:
            Forecast result dictionary
        """
        from forecast_utils import arima_ma20_above_ma50_forecast
        try:
            return arima_ma20_above_ma50_forecast(symbol.upper(), days_past, forecast_days)
        except Exception as e:
            logger.error(f"MA forecast failed for {symbol}: {e}")
            return {"error": str(e)}
    
    def bulk_macd_forecast(
        self, 
        symbols: List[str], 
        days_past: int = 100, 
        forecast_days: int = 5,
        engine: str = "auto"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run MACD forecasts for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            engine: Forecasting engine - "auto", "neural", or "arima"
            
        Returns:
            Dictionary mapping symbols to forecast results, sorted by likelihood
        """
        # Determine which engine to use
        use_neural = (
            engine == "neural" or 
            (engine == "auto" and self.use_neural and self.neural_available)
        )
        
        results = {}
        
        if use_neural and _neural_service is not None:
            # Neural inference: NPU handles parallelism internally
            logger.info(f"Running neural forecast for {len(symbols)} symbols on NPU")
            results = _neural_service.forecast_batch(symbols, days_past, forecast_days)
        else:
            # ARIMA: use ProcessPoolExecutor for CPU parallelism
            logger.info(f"Running ARIMA forecast for {len(symbols)} symbols on {self.max_workers} workers")
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(
                        self._run_macd_forecast_worker, 
                        s, 
                        days_past, 
                        forecast_days
                    ): s for s in symbols
                }
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        results[symbol.upper()] = future.result()
                    except Exception as e:
                        results[symbol.upper()] = {"error": str(e)}
        
        return self._sort_forecast_results(results)
    
    def bulk_ma_forecast(
        self, 
        symbols: List[str], 
        days_past: int = 100, 
        forecast_days: int = 5,
        filter_positive: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run MA forecasts for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            filter_positive: Only return symbols where MA20 will be above MA50
            
        Returns:
            Dictionary mapping symbols to forecast results
        """
        results = {}
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(
                    self._run_ma_forecast_worker, 
                    s, 
                    days_past, 
                    forecast_days
                ): s for s in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol.upper()] = future.result()
                except Exception as e:
                    results[symbol.upper()] = {"error": str(e)}
        
        if filter_positive:
            return {
                sym: results[sym]
                for sym in results
                if isinstance(results[sym], dict) 
                and results[sym].get("ma20_will_be_above_ma50_and_macd_above_signal", False)
            }
        
        return results
    
    def _run_macd_forecast_worker(
        self, 
        symbol: str, 
        days_past: int, 
        forecast_days: int
    ) -> Dict[str, Any]:
        """Worker function for parallel MACD forecasting."""
        from forecast_utils import arima_macd_positive_forecast
        return arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
    
    def _run_ma_forecast_worker(
        self, 
        symbol: str, 
        days_past: int, 
        forecast_days: int
    ) -> Dict[str, Any]:
        """Worker function for parallel MA forecasting."""
        from forecast_utils import arima_ma20_above_ma50_forecast
        return arima_ma20_above_ma50_forecast(symbol.upper(), days_past, forecast_days)
    
    def _sort_forecast_results(
        self, 
        results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Sort forecast results by likelihood of becoming positive.
        
        Sorting priority:
        1. will_become_positive = True first
        2. Values are increasing
        3. First value is positive
        4. Earlier first positive index
        5. Lower first positive value (tie-breaker)
        """
        def get_sorting_keys(symbol: str):
            result = results[symbol]
            if "error" in result:
                return (False, False, False, float('-inf'), float('-inf'))
            
            will_become_positive = result.get("will_become_positive", False)
            forecast_values = result.get("forecast_values", [])
            details = result.get("details")
            last_macd = details.get("last_macd") if isinstance(details, dict) else None

            if isinstance(result.get("forecasted_macd"), dict):
                if not forecast_values:
                    forecast_values = list(result["forecasted_macd"].values())
            
            if last_macd is not None:
                if not isinstance(forecast_values, list):
                    forecast_values = list(forecast_values)
                forecast_values = [last_macd, *forecast_values]
            
            if not forecast_values:
                return (will_become_positive, False, False, float('-inf'), float('-inf'))
            
            valid_values = [v for v in forecast_values if v is not None]
            if len(valid_values) < 2:
                is_increasing = False
            else:
                is_increasing = all(
                    valid_values[i] < valid_values[i+1] 
                    for i in range(len(valid_values)-1)
                )
            
            first_value_positive = (
                forecast_values 
                and forecast_values[0] is not None 
                and forecast_values[0] > 0
            )
            
            first_positive_index = float('inf')
            first_positive_value = float('inf')
            for i, value in enumerate(forecast_values):
                if value is not None and value > 0:
                    first_positive_index = i
                    first_positive_value = value
                    break
            
            return (
                will_become_positive, 
                is_increasing, 
                first_value_positive, 
                first_positive_index, 
                first_positive_value
            )
        
        ordered_symbols = sorted(
            results.keys(),
            key=lambda sym: (
                not get_sorting_keys(sym)[0],  # will_become_positive first
                not get_sorting_keys(sym)[1],  # increasing values
                not get_sorting_keys(sym)[2],  # first value positive
                get_sorting_keys(sym)[3],      # earlier positive index
                get_sorting_keys(sym)[4]       # lower positive value
            )
        )
        
        return {sym: results[sym] for sym in ordered_symbols}
    
    def get_engine_status(self) -> Dict[str, Any]:
        """
        Get information about available forecasting engines.
        
        Returns:
            Dictionary with engine availability and status
        """
        status = {
            "arima_available": True,
            "neural_available": self.neural_available,
            "current_engine": "neural" if (self.use_neural and self.neural_available) else "arima",
            "max_workers": self.max_workers
        }
        
        if self.neural_available:
            try:
                from models.neural_forecast import check_npu_availability
                npu_info = check_npu_availability()
                status["npu_info"] = npu_info
            except Exception as e:
                status["npu_info"] = {"error": str(e)}
        
        return status


# Default service instance
forecast_service = ForecastService()


def get_forecast_engine_status() -> Dict[str, Any]:
    """Get the status of the forecast service engines."""
    return forecast_service.get_engine_status()
