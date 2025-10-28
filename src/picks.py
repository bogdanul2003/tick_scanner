from db_utils import save_symbol_picks
from macd_utils import get_latest_market_date

def store_symbol_picks(date, watchlist_name, filtered_symbols):
    """
    Stores a row in symbol_picks table for the given date and watchlist.
    filtered_symbols is the output from macd_crossover_signal().
    filter_results is a dict: {signal_name: [symbols with signal True]}
    Excludes 'details' keys from signals.
    """
    from collections import defaultdict

    # Build filter_results: {signal_name: [symbols]}
    filter_results = defaultdict(list)
    for symbol, signals in filtered_symbols.items():
        for signal_name, value in signals.items():
            if signal_name == "details":
                continue
            if isinstance(value, bool) and value:
                filter_results[signal_name].append(symbol)
    # Convert defaultdict to regular dict for JSON serialization
    filter_results = dict(filter_results)
    save_symbol_picks(watchlist_name, filter_results, date)

def get_watchlist_bullish_signal(watchlist_name, days=30, threshold=0.05):
    """
    Returns the same output as api_watchlist_bullish_signal for the given watchlist.
    Also stores the bullish signals in symbol_picks.
    If a row exists in symbol_picks for the watchlist and date, use the cached DB result.
    """
    from db_utils import get_watchlist_symbols, get_symbol_picks
    from macd_utils import macd_crossover_signal
    from datetime import date as dt_date
    import traceback

    try:
        symbols = get_watchlist_symbols(watchlist_name)
        if not symbols:
            return {"watchlist": watchlist_name, "results": {}}
        today = get_latest_market_date()
        cached = get_symbol_picks(watchlist_name, today)
        # print(f"Fetched {len(symbols)} symbols for {watchlist_name} on {today} cached: {cached}")
        results = {}

        if cached is not None:
            # Rebuild the output format from the cached filter_results
            # cached: {signal_name: [symbols]}
            # We want: {symbol: {signal_name: True/False, ...}}
            print(f"Using cached results for {watchlist_name} on {today} for {len(cached)} signals")
            for symbol in symbols:
                symbol_signals = {}
                for signal_name, symbol_list in cached.items():
                    symbol_signals[signal_name] = symbol in symbol_list
                results[symbol] = symbol_signals
        else:
            # If not cached, compute and store
            results = macd_crossover_signal(symbols, days, threshold)
            store_symbol_picks(today, watchlist_name, results)

        # --- Order results as requested ---
        ordered_symbols = sorted(
            results.keys(),
            key=lambda sym: (
                not results[sym].get("macd_just_became_positive", False),
                not results[sym].get("ma20_just_became_above_ma50", False),
                not results[sym].get("bullish_macd_above_signal", False),
                not results[sym].get("about_to_cross", False),
                not results[sym].get("about_to_become_positive", False)
            )
        )
        ordered_results = {sym: results[sym] for sym in ordered_symbols}
        return {"watchlist": watchlist_name, "results": ordered_results}
    except ValueError as e:
        print(f"Exception: {e}")
        traceback.print_exc()

def get_company_names_for_symbols(symbols):
    """
    Given a list of symbols, return a dict {symbol: company_name}.
    Uses the symbol_properties table for cached names.
    For missing symbols, fetches company names using yfinance Tickers bulk API,
    stores them in the table, and returns the result.
    """
    from db_utils import get_company_names, upsert_symbol_property
    import yfinance as yf

    # Step 1: Query cached company names in bulk
    cached = get_company_names(symbols)
    missing = set(symbols) - set(cached.keys())

    # Step 2: For missing, fetch from yfinance and upsert
    if missing:
        tickers = yf.Tickers(list(missing))
        infos = {}
        try:
            infos = tickers.get_info()
        except AttributeError:
            # fallback for older yfinance versions
            infos = {sym: tickers.tickers[sym].info for sym in missing if sym in tickers.tickers}
        except Exception:
            infos = {}

        for sym in missing:
            info = infos.get(sym, {})
            company_name = info.get("shortName") or info.get("longName") or info.get("name") or ""
            upsert_symbol_property(sym, company_name)
            cached[sym] = company_name

    return {sym: cached.get(sym, "") for sym in symbols}

def get_company_names_and_prices_for_symbols(symbols):
    """
    Given a list of symbols, returns a dict {symbol: {"company_name": ..., "close": ...}}
    that contains both company names and latest closing prices.
    """
    from db_utils import fetch_latest_bulk_from_cache
    
    company_names = get_company_names_for_symbols(symbols)
    latest_cache = fetch_latest_bulk_from_cache(symbols)
    result = {}
    
    for sym in symbols:
        close = None
        df = latest_cache.get(sym)
        if df is not None and not df.empty:
            # Get the closing price from the latest row
            close = df.iloc[-1]["Close"]
        result[sym] = {
            "company_name": company_names.get(sym, ""),
            "close": close
        }
    
    return result

def get_company_names_from_bullish_signal_result(watchlist_name, days=30, threshold=0.05):
    """
    Given the watchlist name and optional days/threshold, call get_watchlist_bullish_signal(),
    and return a dict {symbol: {"company_name": ..., "close": ...}} for all symbols in the result that have both
    ma20_just_became_above_ma50 and bullish_macd_above_signal set to True.
    """
    result = get_watchlist_bullish_signal(watchlist_name, days, threshold)
    filtered_symbols = [
        sym for sym, signals in result.get("results", {}).items()
        if signals.get("ma20_just_became_above_ma50") and signals.get("bullish_macd_above_signal")
    ]
    
    return get_company_names_and_prices_for_symbols(filtered_symbols)