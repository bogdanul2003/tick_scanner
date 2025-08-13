import pandas as pd
from datetime import datetime, timedelta
from macd_utils import get_latest_market_date, get_macd_for_date, get_macd_for_range


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

    end_date = get_latest_market_date()
    print(f"Latest market date for {symbol}: {end_date}")
    start_date = end_date - timedelta(days=days_past-1)
    # get_macd_for_date([symbol], end_date)  # Ensure we have data for the start date
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
    cached = get_arima_grid_cache(symbol, window_size, cache_days=1, with_model=True, model_type="MACD_MODEL")

    best_model = None
    if cached:
        best_order = tuple(cached["best_order"])
        best_aic = cached["best_aic"]
        print(f"Loaded ARIMA grid search from cache for {symbol}: order={best_order}, aic={best_aic}, window={window_size}, date= {cached['search_date']}")
        # Try to load model from blob
        if cached.get("model_blob"):
            try:
                best_model = get_cached_arima_model(symbol, window_size, cache_days=1, model_type="MACD_MODEL")
                print(f"Loaded ARIMA model from DB blob for {symbol}")
                # If model loaded, append missing data since search_date
                search_date = cached.get("search_date")
                # Ensure search_date is not in the future compared to latest market date
                if search_date:
                    latest_market_date = get_latest_market_date()
                    # Convert both to pd.Timestamp for comparison
                    search_date_pd = pd.Timestamp(search_date)
                    latest_market_date_pd = pd.Timestamp(latest_market_date)
                    if search_date_pd > latest_market_date_pd:
                        print(f"search_date {search_date} is after latest market date {latest_market_date}, skipping append/refit for {symbol}")
                        search_date = get_latest_market_date()
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
                            print(f"Appending {len(missing_data)} new data points to ARIMA model for {symbol} dates: {missing_data}")
                            
                            # Fix: Safely handle different index types
                            try:
                                model_index = best_model.model.data.row_labels
                                
                                # Check if it's a DatetimeIndex first
                                if isinstance(model_index, pd.DatetimeIndex):
                                    model_index_freq = model_index.freq
                                else:
                                    # For RangeIndex and other types that don't have freq
                                    model_index_freq = None
                                    
                                # Get the dates needed for the new data
                                missing_dates = train_dates[idx+1:]
                                
                                # Create a new index with the proper frequency (or lack thereof)
                                if model_index_freq is None:
                                    # If model has no frequency, convert dates to a simple index without frequency
                                    missing_dates_index = pd.DatetimeIndex(missing_dates.values, freq=None)
                                else:
                                    missing_dates_index = missing_dates
                                
                                # Create properly indexed Series
                                missing_data_indexed = pd.Series(missing_data.values, index=missing_dates_index)
                                
                                best_model = best_model.append(missing_data_indexed, refit=False)
                            except Exception as e:
                                print(f"Failed to append new data to model for {symbol}: {e}")
                                # Fallback to refit if append fails
                                from statsmodels.tsa.arima.model import ARIMA
                                best_model = ARIMA(train_data, order=best_order).fit()
                    except ValueError as ve:
                        import traceback
                        
                        # Check if search_date is a weekday (0=Monday,...,4=Friday)
                        if hasattr(search_date_pd, "weekday") and search_date_pd.weekday() < 5:
                            # search_date not in train_dates, fallback to refit only if missing date is not a weekend
                            traceback.print_exc()
                            print(f"search_date {search_date} not in train_dates for {symbol}, checking if refit needed. Exception: {ve}")
                            from statsmodels.tsa.arima.model import ARIMA
                            best_model = ARIMA(train_data, order=best_order).fit()
                        else:
                            print(f"search_date {search_date} is a weekend, skipping refit for {symbol}")
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
        set_arima_grid_cache(symbol, best_order, best_aic, window_size, model=best_model, model_type="MACD_MODEL")

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


def arima_ma20_above_ma50_forecast(symbol: str, days_past: int = 60, forecast_days: int = 3):
    """
    Uses ARIMA to forecast if MA20 will become higher than MA50 in the next `forecast_days` days
    based on the past `days_past` days of MA20 and MA50 data.

    Returns:
        {
            "ma20_will_be_above_ma50": bool,
            "forecasted_ma20": {date: value, ...},
            "forecasted_ma50": {date: value, ...},
            "forecasted_dates": [date, ...],
            "details": {
                "last_ma20": float,
                "last_ma50": float,
                "ma20_series": [float, ...],
                "ma50_series": [float, ...],
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
        print(f"ImportError in arima_ma20_above_ma50_forecast for {symbol}: {e}")
        traceback.print_exc()
        return {
            "ma20_will_be_above_ma50": False,
            "forecasted_ma20": {},
            "forecasted_ma50": {},
            "forecasted_dates": [],
            "details": {"error": "statsmodels not installed"}
        }

    from db_utils import get_arima_grid_cache, set_arima_grid_cache, get_cached_arima_model
    from macd_utils import get_latest_market_date, get_macd_for_range

    end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=days_past-1)
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    ma20_series = [d["ma20"] for d in macd_data if "ma20" in d and d["ma20"] is not None]
    ma50_series = [d["ma50"] for d in macd_data if "ma50" in d and d["ma50"] is not None]

    if len(ma20_series) < 10 or len(ma50_series) < 10:
        return {
            "ma20_will_be_above_ma50": False,
            "forecasted_ma20": {},
            "forecasted_ma50": {},
            "forecasted_dates": [],
            "details": {"error": "Not enough MA20/MA50 data"}
        }

    # Dynamic window (reuse logic)
    def dynamic_window(series, base_window=days_past):
        recent_std = np.std(series[-10:]) if len(series) >= 10 else np.std(series)
        if recent_std > 2.0:   return min(30, len(series))
        elif recent_std < 0.5: return min(60, len(series))
        else:                  return min(base_window, len(series))

    window_size = min(dynamic_window(ma20_series), dynamic_window(ma50_series))
    train_ma20 = pd.Series(ma20_series[-window_size:])
    train_ma50 = pd.Series(ma50_series[-window_size:])

    # ARIMA for MA20
    cache_ma20 = get_arima_grid_cache(symbol, window_size, cache_days=1, with_model=True, model_type="MA20_MODEL")
    best_model_ma20 = None
    if cache_ma20:
        best_order_ma20 = tuple(cache_ma20["best_order"])
        if cache_ma20.get("model_blob"):
            try:
                best_model_ma20 = get_cached_arima_model(symbol, window_size, cache_days=1, model_type="MA20_MODEL")
                print(f"Loaded ARIMA model for MA20 from DB blob for {symbol}")
            except Exception:
                from statsmodels.tsa.arima.model import ARIMA
                best_model_ma20 = ARIMA(train_ma20, order=best_order_ma20).fit()
        if best_model_ma20 is None:
            from statsmodels.tsa.arima.model import ARIMA
            best_model_ma20 = ARIMA(train_ma20, order=best_order_ma20).fit()
    else:
        import warnings
        best_aic_ma20, best_order_ma20 = float('inf'), (0,0,0)
        best_model_ma20 = None
        for p in range(4):
            for q in range(4):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train_ma20, order=(p, 0, q))
                        results = model.fit()
                    if results.aic < best_aic_ma20:
                        best_aic_ma20 = results.aic
                        best_order_ma20 = (p, 0, q)
                        best_model_ma20 = results
                except Exception:
                    continue
        if best_model_ma20 is None:
            best_order_ma20 = (1, 0, 0)
            best_model_ma20 = ARIMA(train_ma20, order=best_order_ma20).fit()
        set_arima_grid_cache(symbol, best_order_ma20, best_aic_ma20, window_size, model=best_model_ma20, model_type="MA20_MODEL")
        print(f"Best ARIMA order for MA20 for {symbol}: {best_order_ma20} with AIC: {best_aic_ma20}")

    # ARIMA for MA50
    cache_ma50 = get_arima_grid_cache(symbol, window_size, cache_days=1, with_model=True, model_type="MA50_MODEL")
    best_model_ma50 = None
    if cache_ma50:
        best_order_ma50 = tuple(cache_ma50["best_order"])
        print(f"Loaded ARIMA grid search from cache for MA50 for {symbol}: order={best_order_ma50}, aic={cache_ma50['best_aic']}, window={window_size}, date= {cache_ma50['search_date']}")
        if cache_ma50.get("model_blob"):
            try:
                best_model_ma50 = get_cached_arima_model(symbol, window_size, cache_days=1, model_type="MA50_MODEL")
                print(f"Loaded ARIMA model for MA50 from DB blob for {symbol}")
            except Exception:
                from statsmodels.tsa.arima.model import ARIMA
                best_model_ma50 = ARIMA(train_ma50, order=best_order_ma50).fit()
        if best_model_ma50 is None:
            from statsmodels.tsa.arima.model import ARIMA
            best_model_ma50 = ARIMA(train_ma50, order=best_order_ma50).fit()
    else:
        import warnings
        best_aic_ma50, best_order_ma50 = float('inf'), (0,0,0)
        best_model_ma50 = None
        for p in range(4):
            for q in range(4):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(train_ma50, order=(p, 0, q))
                        results = model.fit()
                    if results.aic < best_aic_ma50:
                        best_aic_ma50 = results.aic
                        best_order_ma50 = (p, 0, q)
                        best_model_ma50 = results
                except Exception:
                    continue
        if best_model_ma50 is None:
            best_order_ma50 = (1, 0, 0)
            best_model_ma50 = ARIMA(train_ma50, order=best_order_ma50).fit()
        set_arima_grid_cache(symbol, best_order_ma50, best_aic_ma50, window_size, model=best_model_ma50, model_type="MA50_MODEL")
        print(f"Best ARIMA order for MA50 for {symbol}: {best_order_ma50} with AIC: {best_aic_ma50}")

    # Forecast
    forecast_ma20 = best_model_ma20.forecast(steps=forecast_days)
    forecast_ma50 = best_model_ma50.forecast(steps=forecast_days)
    forecasted_ma20_values = forecast_ma20.tolist()
    forecasted_ma50_values = forecast_ma50.tolist()

    # Build forecasted dates (weekdays only)
    forecasted_dates = []
    next_date = end_date
    days_added = 0
    while days_added < forecast_days:
        next_date += timedelta(days=1)
        if next_date.weekday() < 5:
            forecasted_dates.append(next_date.isoformat())
            days_added += 1

    forecasted_ma20 = {date: value for date, value in zip(forecasted_dates, forecasted_ma20_values)}
    forecasted_ma50 = {date: value for date, value in zip(forecasted_dates, forecasted_ma50_values)}

    # Check if currently MA20 is below MA50 and in any forecasted value MA20 becomes above MA50
    ma20_will_be_above_ma50 = False
    if float(ma20_series[-1]) < float(ma50_series[-1]):
        for m20, m50 in zip(forecasted_ma20_values, forecasted_ma50_values):
            if m20 > m50:
                ma20_will_be_above_ma50 = True
                break

    # Also check if current MACD is above signal line
    current_macd = None
    current_signal_line = None
    if macd_data and "macd" in macd_data[-1] and "signal_line" in macd_data[-1]:
        current_macd = macd_data[-1]["macd"]
        current_signal_line = macd_data[-1]["signal_line"]

    ma20_will_be_above_ma50_and_macd_above_signal = (
        ma20_will_be_above_ma50 and
        current_macd is not None and
        current_signal_line is not None and
        float(current_macd) > float(current_signal_line)
    )

    return {
        "ma20_will_be_above_ma50": ma20_will_be_above_ma50,
        "ma20_will_be_above_ma50_and_macd_above_signal": ma20_will_be_above_ma50_and_macd_above_signal,
        "forecasted_ma20": forecasted_ma20,
        "forecasted_ma50": forecasted_ma50,
        "forecasted_dates": forecasted_dates
    }