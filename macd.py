import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def dynamic_window(macd_series, base_window=45):
    """Adjust lookback based on recent volatility"""
    recent_std = macd_series[-10:].std()  # Last 10 days volatility
    if recent_std > 2.0:   return min(30, len(macd_series))  # High volatility
    elif recent_std < 0.5: return min(60, len(macd_series))  # Low volatility
    else:                  return min(base_window, len(macd_series))

def forecast_macd(full_series, steps=3):
    """
    Combined dynamic window + grid search for MACD forecasting
    :param full_series: Complete historical MACD Series
    :return: Forecast (array), window size (int), best_order (tuple)
    """
    # 1. Determine adaptive window size
    window_size = dynamic_window(full_series)
    train_data = full_series[-window_size:]
    
    # 2. Grid search over p,q (d=0 for MACD)
    best_aic, best_order = float('inf'), (0,0,0)
    best_model = None
    
    # Search space (keep small for stability)
    for p in range(3):  # 0-2
        for q in range(3):  # 0-2
            try:
                model = ARIMA(train_data, order=(p, 0, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, 0, q)
                    best_model = results
            except:
                continue
    
    # 3. Fallback to simple model if grid search fails
    if best_model is None:
        best_order = (1, 0, 0)
        best_model = ARIMA(train_data, order=best_order).fit()
    
    # 4. Forecast next 3 days
    forecast = best_model.forecast(steps=steps)
    return forecast, window_size, best_order

# ===== Usage Example =====
# full_macd_series = pd.Series([...])  # Your historical MACD
# forecast, window_used, order_used = forecast_macd(full_macd_series)