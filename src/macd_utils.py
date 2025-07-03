import pandas as pd
from datetime import timedelta
from fastapi import HTTPException
from db_utils import (
    get_cached_dates,
    fetch_from_cache,
    load_cached_data,
    get_missing_dates,
    save_bulk_to_cache
)
import yfinance as yf

def calculate_macd_and_signal(symbol: str, date: pd.Timestamp, cached_data: pd.DataFrame, missing_dates):
    # ...existing code from api.py...
    earliest_missing = min(missing_dates)
    latest_missing = max(missing_dates)
    lookback_buffer = 10
    fetch_start = earliest_missing - timedelta(days=lookback_buffer)
    fetch_end = latest_missing + timedelta(days=1)
    print(f"Fetching data from {fetch_start} to {fetch_end} for {symbol}")
    yf_ticker = yf.Ticker(symbol)
    new_data = yf_ticker.history(start=fetch_start, end=fetch_end, interval="1d")
    if new_data.empty:
        raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")
    new_data_close = new_data[['Close']].copy()
    if new_data_close.index.tz is not None:
        new_data_close.index = new_data_close.index.tz_convert(None)
    new_data_close.index = new_data_close.index.normalize()
    if not cached_data.empty and cached_data.index.tz is not None:
        cached_data.index = cached_data.index.tz_convert(None)
        cached_data.index = cached_data.index.normalize()
    if not cached_data.empty:
        all_data = pd.concat([cached_data, new_data_close])
        all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()
    else:
        all_data = new_data_close.copy()
    all_data = all_data[['Close']]
    all_data['EMA12'] = all_data['Close'].ewm(span=12, adjust=False).mean()
    all_data['EMA26'] = all_data['Close'].ewm(span=26, adjust=False).mean()
    all_data['MACD'] = all_data['EMA12'] - all_data['EMA26']
    all_data['Signal_Line'] = all_data['MACD'].ewm(span=9, adjust=False).mean()
    missing_dates_set = set(missing_dates)
    to_cache = all_data[all_data.index.map(lambda x: x.date() in missing_dates_set)]
    if not to_cache.empty:
        save_bulk_to_cache(symbol, to_cache)
    date_data = all_data[all_data.index.map(lambda x: x.date() == date)]
    if date_data.empty:
        date_data = all_data.iloc[-1:]
        actual_date = date_data.index[0].date()
        note = f"Using latest available data from {actual_date.isoformat()}"
    else:
        actual_date = date
        note = None
    return {
        "symbol": symbol,
        "date": actual_date.isoformat(),
        "close": float(date_data['Close'].iloc[0]),
        "macd": float(date_data['MACD'].iloc[0]),
        "signal_line": float(date_data['Signal_Line'].iloc[0]),
        "note": note
    }

def get_macd_for_date(symbol: str, date):
    # ...existing code from api.py...
    cached_data_row = fetch_from_cache(symbol, date)
    if cached_data_row:
        close, ema12, ema26, macd, signal_line = cached_data_row
        return {
            "symbol": symbol,
            "date": date.isoformat(),
            "close": close,
            "macd": macd,
            "signal_line": signal_line
        }
    start_date = date - timedelta(days=365)
    end_date = date
    cached_dates = get_cached_dates(symbol, start_date, end_date)
    print(f"Cached dates for {symbol} from {start_date} to {end_date}: {len(cached_dates)} dates found")
    missing_dates = get_missing_dates(symbol, start_date, end_date, cached_dates)
    print(f"Missing dates: {len(missing_dates)} days {missing_dates}")
    cached_data = load_cached_data(symbol)
    print(f"Loaded {len(cached_data)} days from cache for {symbol}")
    if missing_dates:
        return calculate_macd_and_signal(symbol, date, cached_data, missing_dates)
    raise HTTPException(status_code=404, detail=f"No data found for symbol: {symbol}")

def get_macd_for_range(symbol: str, start_date, end_date):
    """
    Retrieves MACD and Signal Line for a given symbol and date range, using cache if possible.
    Returns a list of daily MACD data for the interval [start_date, end_date].
    """
    # Try to get cached dates in the range
    cached_dates = get_cached_dates(symbol, start_date, end_date)
    missing_dates = get_missing_dates(symbol, start_date, end_date, cached_dates)
    cached_data = load_cached_data(symbol)
    results = []

    # If there are missing dates, fetch and cache them
    if missing_dates:
        calculate_macd_and_signal(symbol, end_date, cached_data, missing_dates)
        # Reload cache after update
        cached_data = load_cached_data(symbol)

    # Prepare the output for each date in the interval
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in date_range:
        row = fetch_from_cache(symbol, date.date())
        if row:
            close, ema12, ema26, macd, signal_line = row
            results.append({
                "symbol": symbol,
                "date": date.date().isoformat(),
                "close": close,
                "macd": macd,
                "signal_line": signal_line
            })
        else:
            results.append({
                "symbol": symbol,
                "date": date.date().isoformat(),
                "error": "No data"
            })
    return results
