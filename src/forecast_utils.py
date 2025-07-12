import pandas as pd
from datetime import datetime, timedelta
from macd_utils import get_macd_for_date, get_macd_for_range


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

    # --- ARIMA grid search caching logic ---
    from db_utils import get_arima_grid_cache, set_arima_grid_cache, get_cached_arima_model

    end_date = datetime.now().date() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_past-1)
    get_macd_for_date([symbol], end_date)  # Ensure we have data for the start date
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    macd_series = [d["macd"] for d in macd_data if "macd" in d and d["macd"] is not None]
    print(f"MACD data for {symbol} from {start_date} to {end_date} with length {len(macd_series)}")

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

    # Try to load ARIMA grid search from cache (forecast_util table)
    cached = get_arima_grid_cache(symbol, window_size, cache_days=30, with_model=True)

    best_model = None
    if cached:
        best_order = tuple(cached["best_order"])
        best_aic = cached["best_aic"]
        print(f"Loaded ARIMA grid search from cache for {symbol}: order={best_order}, aic={best_aic}, window={window_size}")
        # Try to load model from blob
        if cached.get("model_blob"):
            try:
                best_model = get_cached_arima_model(symbol, window_size, cache_days=30)
                print(f"Loaded ARIMA model from DB blob for {symbol}")
                # If model loaded, append missing data since search_date
                search_date = cached.get("search_date")
                if search_date:
                    # Find index of search_date in train_data
                    train_dates = pd.date_range(end=end_date, periods=window_size)
                    # search_date may be a datetime.date, convert to pd.Timestamp for comparison
                    search_date_pd = pd.Timestamp(search_date)
                    # Find where search_date is in train_dates
                    try:
                        idx = list(train_dates).index(search_date_pd)
                        # Data after search_date
                        missing_data = train_data.iloc[idx+1:]
                        if not missing_data.empty:
                            print(f"Appending {len(missing_data)} new data points to ARIMA model for {symbol}")
                            best_model = best_model.append(missing_data, refit=False)
                    except ValueError:
                        # search_date not in train_dates, fallback to refit
                        print(f"search_date {search_date} not in train_dates for {symbol}, refitting model.")
                        from statsmodels.tsa.arima.model import ARIMA
                        best_model = ARIMA(train_data, order=best_order).fit()
            except Exception as e:
                print(f"Failed to load ARIMA model blob for {symbol}: {e}")
        if best_model is None:
            from statsmodels.tsa.arima.model import ARIMA
            best_model = ARIMA(train_data, order=best_order).fit()
    else:
        # Grid search ARIMA(p,0,q)
        import warnings
        best_aic, best_order = float('inf'), (0,0,0)
        best_model = None
        for p in range(4):
            for q in range(4):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train_data, order=(p, 0, q))
                        results = model.fit()
                    print(f"Fitting ARIMA({p},0,{q}) for {symbol} with AIC: {results.aic}")
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

        # Save grid search result and model to cache
        set_arima_grid_cache(symbol, best_order, best_aic, window_size, model=best_model)

    # Forecast
    forecast = best_model.forecast(steps=forecast_days)
    forecasted_macd_values = forecast.tolist()
    print(f"Forecasted MACD for {symbol} over the next {forecast_days} days: {forecasted_macd_values}")

    last_macd = float(macd_series[-1])
    # will_become_positive: only if macd was negative and forecasted values become positive,
    # or if first forecasted value is negative and they become positive later
    will_become_positive = False
    if last_macd < 0 and any(v > 0 for v in forecasted_macd_values):
        will_become_positive = True
    elif forecasted_macd_values and forecasted_macd_values[0] < 0:
        for v in forecasted_macd_values[1:]:
            if v > 0:
                will_become_positive = True
                break

    # Add forecasted dates and build dict
    forecasted_dates = []
    next_date = end_date
    days_added = 0
    while days_added < forecast_days:
        next_date += timedelta(days=1)
        if next_date.weekday() < 5:  # 0=Monday, ..., 4=Friday
            forecasted_dates.append(next_date.isoformat())
            days_added += 1

    forecasted_macd = {date: value for date, value in zip(forecasted_dates, forecasted_macd_values)}

    return {
        "will_become_positive": will_become_positive,
        "forecasted_macd": forecasted_macd,
        "forecasted_dates": forecasted_dates,
        "details": {
            "last_macd": last_macd,
            "macd_series": macd_series,
            "window_used": window_size,
            "order_used": best_order
        }
    }