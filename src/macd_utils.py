import pandas as pd
from datetime import datetime, timedelta
from fastapi import HTTPException
from db_utils import (
    fetch_bulk_from_cache,
    load_cached_data,
    get_missing_dates,
    save_bulk_to_cache,
    fetch_from_cache
)
import yfinance as yf

def get_macd_for_date(symbols: list, date):
    """
    Retrieves MACD and Signal Line for a list of symbols for a specific date.
    Returns a dict: {symbol: macd data dict} for the given date.
    """
    start_date = date - timedelta(days=365)
    end_date = date
    results = get_macd_for_range_bulk(symbols, start_date, end_date)
    output = {}
    for symbol in symbols:
        symbol_results = results.get(symbol, [])
        # Find the entry for the requested date
        entry = next((item for item in symbol_results if item.get("date") == date.isoformat()), None)
        if entry:
            output[symbol] = entry
        else:
            output[symbol] = {
                "symbol": symbol,
                "date": date.isoformat(),
                "error": "No data"
            }
    return output

def get_macd_for_range(symbol: str, start_date, end_date):
    """
    Retrieves MACD and Signal Line for a given symbol and date range, using cache if possible.
    Returns a list of daily MACD data for the interval [start_date, end_date].
    """
    # Replace get_cached_dates with fetch_bulk_from_cache
    bulk_cache = fetch_bulk_from_cache([symbol], start_date, end_date)
    cached_df = bulk_cache.get(symbol, pd.DataFrame())
    cached_dates = set(cached_df.index.date) if not cached_df.empty else set()
    missing_dates = get_missing_dates(symbol, start_date, end_date, cached_dates)
    # print(f"Missing dates for {symbol}: {missing_dates}")
    cached_data = cached_df if not cached_df.empty else load_cached_data(symbol)
    results = []

    # If there are missing dates, fetch and cache them using bulk function
    if missing_dates:
        calculate_macd_and_signal_bulk(
            [symbol],
            end_date,
            {symbol: cached_data},
            {symbol: missing_dates}
        )
        # Reload cache after update
        bulk_cache = fetch_bulk_from_cache([symbol], start_date, end_date)
        cached_df = bulk_cache.get(symbol, pd.DataFrame())

    # Prepare the output for each date in the interval
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for date in date_range:
        if not cached_df.empty and date in cached_df.index:
            row = cached_df.loc[date]
            results.append({
                "symbol": symbol,
                "date": date.date().isoformat(),
                "close": row['Close'],
                "macd": row['MACD'],
                "signal_line": row['Signal_Line']
            })
        else:
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

def get_macd_for_range_bulk(symbols: list, start_date, end_date):
    """
    Retrieves MACD and Signal Line for a list of symbols and date range, using cache if possible.
    Returns a dict: {symbol: [daily MACD data dicts]} for the interval [start_date, end_date].
    """
    # Fetch cached data for all symbols
    bulk_cache = fetch_bulk_from_cache(symbols, start_date, end_date)
    cached_data_dict = {}
    missing_dates_dict = {}
    for symbol in symbols:
        cached_df = bulk_cache.get(symbol, pd.DataFrame())
        cached_dates = set(cached_df.index.date) if not cached_df.empty else set()
        missing_dates = get_missing_dates(symbol, start_date, end_date, cached_dates)
        print(f"Missing dates for {symbol}: {missing_dates}")
        cached_data_dict[symbol] = cached_df if not cached_df.empty else load_cached_data(symbol)
        missing_dates_dict[symbol] = missing_dates

    # If there are missing dates for any symbol, fetch and cache them using bulk function
    if any(missing_dates_dict[symbol] for symbol in symbols):
        calculate_macd_and_signal_bulk(
            symbols,
            end_date,
            cached_data_dict,
            missing_dates_dict
        )
        # Reload cache after update
        bulk_cache = fetch_bulk_from_cache(symbols, start_date, end_date)

    # Prepare the output for each symbol and each date in the interval
    results = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for symbol in symbols:
        symbol_results = []
        cached_df = bulk_cache.get(symbol, pd.DataFrame())
        for date in date_range:
            if not cached_df.empty and date in cached_df.index:
                row = cached_df.loc[date]
                symbol_results.append({
                    "symbol": symbol,
                    "date": date.date().isoformat(),
                    "close": row['Close'],
                    "macd": row['MACD'],
                    "signal_line": row['Signal_Line']
                })
            else:
                row = fetch_from_cache(symbol, date.date())
                if row:
                    close, ema12, ema26, macd, signal_line = row
                    symbol_results.append({
                        "symbol": symbol,
                        "date": date.date().isoformat(),
                        "close": close,
                        "macd": macd,
                        "signal_line": signal_line
                    })
                else:
                    symbol_results.append({
                        "symbol": symbol,
                        "date": date.date().isoformat(),
                        "error": "No data"
                    })
        results[symbol] = symbol_results
    return results

def calculate_macd_and_signal_bulk(symbols: list, date: pd.Timestamp, cached_data_dict: dict, missing_dates_dict: dict):
    """
    For a list of symbols, finds the largest missing interval and fetches bulk data using yfinance.Tickers.
    Calculates and caches MACD and Signal Line for all missing dates for each symbol.
    Returns a dict: {symbol: {macd data dict}}
    Partition symbols by missing date intervals, grouping intervals that don't differ in more than 30 days.
    """
    from collections import defaultdict

    # Build intervals for each symbol (exclude empty intervals)
    interval_info = []
    for symbol in symbols:
        dates = missing_dates_dict.get(symbol, [])
        if dates:
            interval_info.append((symbol, min(dates), max(dates)))
    if not interval_info:
        raise HTTPException(status_code=400, detail="No missing dates for any symbol.")

    # Sort intervals by start date
    interval_info.sort(key=lambda x: x[1])

    # Partition intervals: group together if start/end within 30 days of current partition's start/end
    partitions = []
    current_partition = []
    current_start = None
    current_end = None
    for symbol, start, end in interval_info:
        if not current_partition:
            current_partition = [(symbol, start, end)]
            current_start = start
            current_end = end
        else:
            # If this interval is within 30 days of current partition's start/end, add to partition
            if abs((start - current_start).days) <= 30 and abs((end - current_end).days) <= 30:
                current_partition.append((symbol, start, end))
                # Update partition bounds
                current_start = min(current_start, start)
                current_end = max(current_end, end)
            else:
                partitions.append((current_partition, current_start, current_end))
                current_partition = [(symbol, start, end)]
                current_start = start
                current_end = end
    if current_partition:
        partitions.append((current_partition, current_start, current_end))

    results = {}
    for partition, partition_start, partition_end in partitions:
        partition_symbols = [item[0] for item in partition]
        # Add lookback buffer
        lookback_buffer = 10
        fetch_start = partition_start - timedelta(days=lookback_buffer)
        fetch_end = partition_end + timedelta(days=1)
        print(f"Fetching bulk data from {fetch_start} to {fetch_end} for symbols: {partition_symbols}")

        # Fetch bulk data using yfinance.Tickers
        yf_tickers = yf.Tickers(" ".join(partition_symbols))
        history = yf_tickers.history(start=fetch_start, end=fetch_end, interval="1d", group_by='ticker')
        print("history columns:", history)

        for symbol in partition_symbols:
            # --- Robust extraction of 'Close' price for each symbol ---
            symbol_data = None
            if isinstance(history.columns, pd.MultiIndex):
                try:
                    symbol_data = history['Close'][symbol].to_frame('Close')
                except Exception:
                    try:
                        symbol_data = history[symbol]['Close'].to_frame('Close')
                    except Exception:
                        results[symbol] = {"error": f"No data found for symbol: {symbol}"}
                        continue
            else:
                if 'Close' in history.columns:
                    symbol_data = history[['Close']]
                else:
                    results[symbol] = {"error": f"No data found for symbol: {symbol}"}
                    continue

            # Remove timezone and normalize index
            if symbol_data.index.tz is not None:
                symbol_data.index = symbol_data.index.tz_convert(None)
            symbol_data.index = symbol_data.index.normalize()

            # Merge with cached data if available
            cached_data = cached_data_dict.get(symbol, pd.DataFrame())
            if not cached_data.empty and cached_data.index.tz is not None:
                cached_data.index = cached_data.index.tz_convert(None)
                cached_data.index = cached_data.index.normalize()
            if not cached_data.empty:
                all_data = pd.concat([cached_data, symbol_data])
                all_data = all_data[~all_data.index.duplicated(keep='last')].sort_index()
            else:
                all_data = symbol_data.copy()
            all_data = all_data[['Close']]
            all_data['EMA12'] = all_data['Close'].ewm(span=12, adjust=False).mean()
            all_data['EMA26'] = all_data['Close'].ewm(span=26, adjust=False).mean()
            all_data['MACD'] = all_data['EMA12'] - all_data['EMA26']
            all_data['Signal_Line'] = all_data['MACD'].ewm(span=9, adjust=False).mean()

            # Cache only the missing dates for this symbol
            missing_dates_set = set(missing_dates_dict[symbol])
            to_cache = all_data[all_data.index.map(lambda x: x.date() in missing_dates_set)]
            if not to_cache.empty:
                save_bulk_to_cache(symbol, to_cache)

            # Prepare result for the requested date
            date_data = all_data[all_data.index.map(lambda x: x.date() == date)]
            if date_data.empty:
                date_data = all_data.iloc[-1:]
                actual_date = date_data.index[0].date()
                note = f"Using latest available data from {actual_date.isoformat()}"
            else:
                actual_date = date
                note = None
            results[symbol] = {
                "symbol": symbol,
                "date": actual_date.isoformat(),
                "close": float(date_data['Close'].iloc[0]),
                "macd": float(date_data['MACD'].iloc[0]),
                "signal_line": float(date_data['Signal_Line'].iloc[0]),
                "note": note
            }
    return results

def macd_crossover_signal(symbol: str, days: int, threshold: float = 0.05, threshold_pos_neg: float = 0.08):
    """
    Determines if the MACD line is about to cross over the signal line for a symbol
    in the last 'days' days, if a bullish crossover occurred in the past days//2,
    and if MACD or Signal Line are about to become positive or negative.

    Args:
        symbol: str
        days: int
        threshold: float - maximum allowed difference between MACD and signal line to consider "about to cross"

    Returns:
        {
            "about_to_cross": bool,
            "recent_crossover": bool,
            "bullish_macd_above_signal": bool,
            "about_to_become_positive": bool,
            "about_to_become_negative": bool,
            "details": {
                "last_macd": float,
                "last_signal": float,
                "prev_macd": float,
                "prev_signal": float,
                "crossover_dates": [date1, date2, ...]
            }
        }
    """

    end_date = datetime.now().date() - timedelta(days=1)
    start_date = datetime.now().date() - timedelta(days=days)
    # Get MACD data for the range
    macd_data = get_macd_for_range(symbol, start_date, end_date)
    if not macd_data or len(macd_data) < 2:
        return {
            "about_to_cross": False,
            "recent_crossover": False,
            "about_to_become_positive": False,
            "details": {"error": "Not enough data"}
        }

    # Filter out entries with errors
    macd_data = [d for d in macd_data if "macd" in d and "signal_line" in d]

    if len(macd_data) < 2:
        return {
            "about_to_cross": False,
            "recent_crossover": False,
            "about_to_become_positive": False,
            "details": {"error": "Not enough valid MACD data"}
        }

    # Check if MACD is about to cross above signal line (bullish crossover)
    last = macd_data[-1]
    prev = macd_data[-2]
    macd_diff = float(abs(last["macd"] - last["signal_line"]))
    about_to_cross = (
        bool(prev["macd"] < prev["signal_line"])
        and bool(last["macd"] > prev["macd"])
        and bool(last["macd"] < last["signal_line"])
        and macd_diff <= float(threshold)
    )

    # Check if MACD or Signal Line is about to become positive (currently negative and close to zero)
    about_to_become_positive = (
        (float(last["macd"]) < 0 and abs(float(last["macd"])) <= float(threshold))
        or (float(last["signal_line"]) < 0 and abs(float(last["signal_line"])) <= float(threshold))
    )

    # Check if MACD or Signal Line is about to become negative (currently positive and close to zero)
    about_to_become_negative = (
        (float(last["macd"]) > 0 and abs(float(last["macd"])) <= float(threshold))
        or (float(last["signal_line"]) > 0 and abs(float(last["signal_line"])) <= float(threshold))
    )

    # Check if MACD is currently above the signal line
    macd_above_signal = False
    if "macd" in last and "signal_line" in last:
        macd_above_signal = float(last["macd"]) > float(last["signal_line"])

    # Find bullish crossovers in the past days//2
    lookback = max(2, days // 2)
    crossover_dates = []
    for i in range(1, min(lookback, len(macd_data))):
        prev_row = macd_data[-i-1]
        curr_row = macd_data[-i]
        # Bullish crossover: MACD crosses from below to above signal line
        if prev_row["macd"] < prev_row["signal_line"] and curr_row["macd"] >= curr_row["signal_line"]:
            crossover_dates.append(curr_row["date"])
    recent_crossover = bool(len(crossover_dates) > 0)

    return {
        "about_to_cross": bool(about_to_cross),
        "recent_crossover": recent_crossover,
        "bullish_macd_above_signal": macd_above_signal,
        "about_to_become_positive": about_to_become_positive,
        "about_to_become_negative": about_to_become_negative,
        "details": {
            "last_macd": float(last["macd"]),
            "last_signal": float(last["signal_line"]),
            "prev_macd": float(prev["macd"]),
            "prev_signal": float(prev["signal_line"]),
            "crossover_dates": crossover_dates
        }
    }
