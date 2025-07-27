import pandas as pd
from datetime import datetime, time, timedelta
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
    # print(f"Preparing MACD data for symbols: {symbols} from {start_date} to {end_date}")
    results = {}
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    for symbol in symbols:
        symbol_results = []
        cached_df = bulk_cache.get(symbol, pd.DataFrame())
        for date in date_range:
            # Skip weekends (Saturday=5, Sunday=6)
            if date.weekday() >= 5:
                continue
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
                # print(f"Fetching missing data for {symbol} on {date.date()}")
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
    print(f"Calculating MACD and Signal Line for {len(symbols)} symbols on date {date}")
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
            # print(f"Starting new partition for symbol {symbol} with interval ({start}, {end})")
            current_partition = [(symbol, start, end)]
            current_start = start
            current_end = end
        else:
            # If this interval is within 30 days of current partition's start/end, add to partition
            if abs((start - current_start).days) <= 30 and abs((end - current_end).days) <= 30:
                # print(f"Adding symbol {symbol} to current partition ({current_start}, {current_end}) with interval ({start}, {end}) < 30 days")
                current_partition.append((symbol, start, end))
                # Update partition bounds
                current_start = min(current_start, start)
                current_end = max(current_end, end)
            else:
                # print(f"Closing current partition ({current_start}, {current_end}) and starting new one for symbol {symbol} with interval ({start}, {end})")
                partitions.append((current_partition, current_start, current_end))
                current_partition = [(symbol, start, end)]
                current_start = start
                current_end = end
    if current_partition:
        # print(f"Closing final partition ({current_start}, {current_end}) with symbols: {[item[0] for item in current_partition]}")
        partitions.append((current_partition, current_start, current_end))

    # print(f"Total partitions created: {len(partitions)} partitions: {partitions}")
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

def macd_crossover_signal(symbols: list, days: int, threshold: float = 0.05, threshold_pos_neg: float = 0.08):
    """
    Determines MACD crossover signals for a list of symbols.
    Returns a dict: {symbol: signal_result_dict}
    """
    end_date = get_latest_market_date()
    start_date = datetime.now().date() - timedelta(days=days)
    # Ensure we have data for the end date for all symbols
    # get_macd_for_date(symbols, end_date)
    # Get MACD data for the range for all symbols
    macd_bulk_data = get_macd_for_range_bulk(symbols, end_date - timedelta(days=365), end_date)
    print(f"Got MACD data for {len(macd_bulk_data)} symbols from {end_date - timedelta(days=365)} to {end_date}")

    # Extract from macd_bulk_data just the dates between start_date and end_date and assign it back to macd_bulk_data
    for symbol in macd_bulk_data:
        macd_bulk_data[symbol] = [
            entry for entry in macd_bulk_data[symbol]
            if "date" in entry and start_date <= datetime.fromisoformat(entry["date"]).date() <= end_date
        ]

    print(f"Exrtacted MACD data for {len(macd_bulk_data)} symbols from {start_date} to {end_date}")

    results = {}

    for symbol in symbols:
        try:
            macd_data = macd_bulk_data.get(symbol, [])
            # Filter out entries with errors
            macd_data = [d for d in macd_data if "macd" in d and "signal_line" in d]
            if not macd_data or len(macd_data) < 2:
                results[symbol] = {
                    "about_to_cross": False,
                    "recent_crossover": False,
                    "about_to_become_positive": False,
                    "details": {"error": "Not enough data"}
                }
                continue

            last = macd_data[-1]
            prev = macd_data[-2]
            macd_diff = float(abs(last["macd"] - last["signal_line"]))
            about_to_cross = (
                bool(prev["macd"] < prev["signal_line"])
                and bool(last["macd"] > prev["macd"])
                and bool(last["macd"] < last["signal_line"])
                and macd_diff <= float(threshold)
            )

            about_to_become_positive = (
                (float(last["macd"]) < 0 and abs(float(last["macd"])) <= float(threshold))
                or (float(last["signal_line"]) < 0 and abs(float(last["signal_line"])) <= float(threshold))
            )

            about_to_become_negative = (
                (float(last["macd"]) > 0 and abs(float(last["macd"])) <= float(threshold))
                or (float(last["signal_line"]) > 0 and abs(float(last["signal_line"])) <= float(threshold))
            )

            macd_above_signal = False
            if "macd" in last and "signal_line" in last:
                macd_above_signal = float(last["macd"]) > float(last["signal_line"])

            lookback = max(2, days // 2)
            crossover_dates = []
            for i in range(1, min(lookback, len(macd_data))):
                prev_row = macd_data[-i-1]
                curr_row = macd_data[-i]
                if prev_row["macd"] < prev_row["signal_line"] and curr_row["macd"] >= curr_row["signal_line"]:
                    crossover_dates.append(curr_row["date"])
            recent_crossover = bool(len(crossover_dates) > 0)

            # --- Add macd_just_became_positive signal ---
            macd_just_became_positive = False
            recent_positive_dates = []
            # Check today
            if len(macd_data) >= 2:
                last = macd_data[-1]
                prev = macd_data[-2]
                if prev["macd"] < 0 and last["macd"] > 0:
                    macd_just_became_positive = True
                elif len(macd_data) >= 3:
                    prev2 = macd_data[-3]
                    if prev2["macd"] < 0 and prev["macd"] < 0 and last["macd"] > 0:
                        macd_just_became_positive = True
            # Check last 5 days for the pattern
            for i in range(1, min(3, len(macd_data))):
                curr = macd_data[-i]
                if i >= 2:
                    prev = macd_data[-i-1]
                    if prev["macd"] < 0 and curr["macd"] > 0:
                        recent_positive_dates.append(curr["date"])
                    elif i >= 3:
                        prev2 = macd_data[-i-2]
                        if prev2["macd"] < 0 and prev["macd"] < 0 and curr["macd"] > 0:
                            recent_positive_dates.append(curr["date"])
            if recent_positive_dates:
                macd_just_became_positive = True

            results[symbol] = {
                "about_to_cross": bool(about_to_cross),
                "recent_crossover": recent_crossover,
                "bullish_macd_above_signal": macd_above_signal,
                "about_to_become_positive": about_to_become_positive,
                "about_to_become_negative": about_to_become_negative,
                "macd_just_became_positive": macd_just_became_positive,
                "details": {
                    "last_macd": float(last["macd"]),
                    "last_signal": float(last["signal_line"]),
                    "prev_macd": float(prev["macd"]),
                    "prev_signal": float(prev["signal_line"]),
                    "crossover_dates": crossover_dates,
                    "macd_just_became_positive_dates": recent_positive_dates
                }
            }
        except Exception as e:
            print(f"Error processing symbol {symbol}: {e}")
            continue

    # --- Order results as requested ---
    ordered_symbols = sorted(
        results.keys(),
        key=lambda sym: (
            not results[sym].get("macd_just_became_positive", False),
            not results[sym].get("about_to_cross", False),
            not results[sym].get("about_to_become_positive", False)
        )
    )
    ordered_results = {sym: results[sym] for sym in ordered_symbols}
    return ordered_results

def get_closing_prices_bulk(symbols: list, start_date, end_date):
    """
    Fetches closing prices for a list of symbols between start_date and end_date (inclusive).
    Returns a dict: {symbol: [{date: ..., close: ...}, ...]}
    """
    import yfinance as yf
    import pandas as pd

    yf_tickers = yf.Tickers(" ".join(symbols))
    history = yf_tickers.history(start=start_date, end=end_date + timedelta(days=1), interval="1d", group_by='ticker')
    results = {}

    for symbol in symbols:
        symbol_data = None
        if isinstance(history.columns, pd.MultiIndex):
            try:
                symbol_data = history['Close'][symbol].to_frame('Close')
            except Exception:
                try:
                    symbol_data = history[symbol]['Close'].to_frame('Close')
                except Exception:
                    results[symbol] = []
                    continue
        else:
            if 'Close' in history.columns:
                symbol_data = history[['Close']]
            else:
                results[symbol] = []
                continue

        # Remove timezone and normalize index
        if symbol_data.index.tz is not None:
            symbol_data.index = symbol_data.index.tz_convert(None)
        symbol_data.index = symbol_data.index.normalize()

        # Prepare output as list of dicts
        symbol_results = []
        for idx, row in symbol_data.iterrows():
            symbol_results.append({
                "date": idx.date().isoformat(),
                "close": float(row['Close']) if not pd.isna(row['Close']) else None
            })
        results[symbol] = symbol_results

    return results

def get_closing_prices(symbol: str, start_date, end_date):
    """
    Fetches closing prices for a single symbol between start_date and end_date (inclusive).
    Returns a list: [{date: ..., close: ...}, ...]
    """
    import yfinance as yf
    import pandas as pd

    ticker = yf.Ticker(symbol)
    history = ticker.history(start=start_date, end=end_date + timedelta(days=1), interval="1d")
    results = []

    if history.empty or 'Close' not in history.columns:
        return results

    # Remove timezone and normalize index
    if history.index.tz is not None:
        history.index = history.index.tz_convert(None)
    history.index = history.index.normalize()

    for idx, row in history.iterrows():
        results.append({
            "date": idx.date().isoformat(),
            "close": float(row['Close']) if not pd.isna(row['Close']) else None
        })

    return results

def get_latest_market_date():
    """
    Returns the latest date for which market data is available.
    If the US market is still open today or before market open, returns yesterday's date.
    Otherwise, returns today's date.
    """
    now = datetime.now()
    # US market hours: 9:30am to 4:00pm Eastern Time (ET)
    # We'll use UTC-5 for ET (not handling DST for simplicity)
    market_open = time(9, 30)
    market_close = time(16, 0)
    today = now.date()
    current_time = now.time()
    weekday = today.weekday()
    # If today is Saturday (5) or Sunday (6), return last Friday
    if weekday == 5:
        return today - timedelta(days=1)
    if weekday == 6:
        return today - timedelta(days=2)
    # If before market open, use previous trading day
    if current_time < market_open:
        if weekday == 0:  # Monday before open, return last Friday
            return today - timedelta(days=3)
        else:
            return today - timedelta(days=1)
    # If before market close, use previous trading day
    if current_time < market_close:
        if weekday == 0:  # Monday before close, return last Friday
            return today - timedelta(days=3)
        else:
            return today - timedelta(days=1)
    return today
