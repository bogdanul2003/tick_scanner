"""Forecast service for ARIMA predictions."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ForecastService:
    """Service for running ARIMA forecasts on stock data."""
    
    def __init__(self, max_workers: int = 4):
        """
        Initialize the forecast service.
        
        Args:
            max_workers: Maximum number of parallel workers for forecasting
        """
        self.max_workers = max_workers
    
    def run_macd_forecast(
        self, 
        symbol: str, 
        days_past: int, 
        forecast_days: int
    ) -> Dict[str, Any]:
        """
        Run ARIMA forecast for MACD becoming positive.
        
        Args:
            symbol: Stock symbol
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            
        Returns:
            Forecast result dictionary
        """
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
        forecast_days: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run MACD forecasts for multiple symbols in parallel.
        
        Args:
            symbols: List of stock symbols
            days_past: Number of historical days to use
            forecast_days: Number of days to forecast ahead
            
        Returns:
            Dictionary mapping symbols to forecast results, sorted by likelihood
        """
        results = {}
        
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


# Default service instance
forecast_service = ForecastService()
