import pandas as pd
from datetime import datetime, timedelta
from macd_utils import get_macd_for_range


def arima_macd_positive_forecast(symbol: str, days_past: int = 30, forecast_days: int = 3):
    """
    Uses ARIMA (with dynamic window and grid search) to forecast if MACD will become positive in the next `forecast_days` days
    based on the past `days_past` days of MACD data.

    Returns:
        {
            "will_become_positive": bool,
            "forecasted_macd": [float, ...],
            "details": {
                "last_macd": float,
                "macd_series": [float, ...],
                "window_used": int,
                "order_used": tuple
            }
        }
    """
    import numpy as np
    import pandas as pd
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as e:
        import traceback
        print(f"ImportError in arima_macd_positive_forecast for {symbol}: {e}")
        traceback.print_exc()
        return {
            "will_become_positive": False,
            "forecasted_macd": [],
            "details": {"error": "statsmodels not installed"}
        }

    # Helper: dynamic window selection
    def dynamic_window(macd_series, base_window=days_past):
        recent_std = np.std(macd_series[-10:]) if len(macd_series) >= 10 else np.std(macd_series)
        if recent_std > 2.0:   return min(30, len(macd_series))
        elif recent_std < 0.5: return min(60, len(macd_series))
        else:                  return min(base_window, len(macd_series))

    end_date = datetime.now().date() - timedelta(days=5)
    start_date = end_date - timedelta(days=days_past-1)
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    macd_series = [d["macd"] for d in macd_data if "macd" in d and d["macd"] is not None]
    print(f"MACD data for {symbol} from {start_date} to {end_date} with length {len(macd_series)}: {macd_series}")

    if len(macd_series) < 10:
        return {
            "will_become_positive": False,
            "forecasted_macd": [],
            "details": {"error": "Not enough MACD data"}
        }

    # Dynamic window
    window_size = dynamic_window(macd_series)
    train_data = pd.Series(macd_series[-window_size:])
    print(f"Using dynamic window size: {window_size} for symbol {symbol}")

    # Grid search ARIMA(p,0,q)
    best_aic, best_order = float('inf'), (0,0,0)
    best_model = None
    for p in range(5):
        for q in range(5):
            try:
                model = ARIMA(train_data, order=(p, 0, q))
                results = model.fit()
                print(f"Fitting ARIMA({p},0,{q}) for {symbol} with AIC: {results.summary()}")
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, 0, q)
                    best_model = results
            except Exception:
                continue

    # Fallback if grid search fails
    if best_model is None:
        best_order = (1, 0, 0)
        best_model = ARIMA(train_data, order=best_order).fit()

    print(f"Best ARIMA order for {symbol}: {best_order} with AIC: {best_aic}")
    # Forecast
    forecast = best_model.forecast(steps=forecast_days)
    forecasted_macd_values = forecast.tolist()
    print(f"Forecasted MACD for {symbol} over the next {forecast_days} days: {forecasted_macd_values}")
    will_become_positive = any([v > 0 for v in forecasted_macd_values])

    # Add forecasted dates and build dict
    forecasted_dates = [(end_date + timedelta(days=i+1)).isoformat() for i in range(forecast_days)]
    forecasted_macd = {date: value for date, value in zip(forecasted_dates, forecasted_macd_values)}

    return {
        "will_become_positive": will_become_positive,
        "forecasted_macd": forecasted_macd,
        "forecasted_dates": forecasted_dates,
        "details": {
            "last_macd": float(macd_series[-1]),
            "macd_series": macd_series,
            "window_used": window_size,
            "order_used": best_order
        }
    }