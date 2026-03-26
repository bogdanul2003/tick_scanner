"""Forecast-related API endpoints (ARIMA and Neural predictions)."""
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from typing import List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

router = APIRouter(prefix="/forecast", tags=["Forecast"])


@router.get("/engine/status")
async def get_forecast_engine_status():
    """
    Get information about available forecasting engines.
    
    Returns engine availability (ARIMA vs Neural/NPU) and configuration.
    """
    try:
        from services.forecast_service import get_forecast_engine_status
        return get_forecast_engine_status()
    except Exception as e:
        return {
            "arima_available": True,
            "neural_available": False,
            "error": str(e)
        }


def run_forecast(symbol, days_past, forecast_days):
    """Run MACD forecast for a single symbol."""
    from forecast_utils import arima_macd_positive_forecast
    try:
        return symbol.upper(), arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
    except Exception as e:
        print(f"Exception in run_forecast for {symbol}: {e}")
        traceback.print_exc()
        return symbol.upper(), {"error": str(e)}


def run_ma_forecast(symbol, days_past, forecast_days):
    """Run MA forecast for a single symbol."""
    from forecast_utils import arima_ma20_above_ma50_forecast
    try:
        return symbol.upper(), arima_ma20_above_ma50_forecast(symbol.upper(), days_past, forecast_days)
    except Exception as e:
        print(f"Exception in run_ma_forecast for {symbol}: {e}")
        traceback.print_exc()
        return symbol.upper(), {"error": str(e)}


def get_sorting_keys(symbol, results):
    """Get sorting keys for forecast results."""
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
        is_increasing = all(valid_values[i] < valid_values[i+1] for i in range(len(valid_values)-1))
    
    first_value_positive = False
    if forecast_values and forecast_values[0] is not None:
        first_value_positive = forecast_values[0] > 0
    
    first_positive_index = float('inf')
    first_positive_value = float('inf')
    for i, value in enumerate(forecast_values):
        if value is not None and value > 0:
            first_positive_index = i
            first_positive_value = value
            break
    
    return (will_become_positive, is_increasing, first_value_positive, first_positive_index, first_positive_value)


@router.post("/macd/arima_positive")
async def get_arima_macd_positive_forecast_bulk(
    symbols: List[str] = Body(..., embed=True),
    days_past: int = 100,
    forecast_days: int = 5
):
    """
    For a list of symbols, forecast if MACD will become positive in the next `forecast_days` days using ARIMA,
    based on the past `days_past` days of MACD data.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_forecast, symbol, days_past, forecast_days)
            for symbol in symbols
        ]
        for future in as_completed(futures):
            symbol, result = future.result()
            results[symbol] = result

    ordered_symbols = sorted(
        results.keys(),
        key=lambda sym: (
            not results[sym].get("will_become_positive", False),
            not get_sorting_keys(sym, results)[1],
            not get_sorting_keys(sym, results)[2],
            get_sorting_keys(sym, results)[3],
            get_sorting_keys(sym, results)[4]
        )
    )
    ordered_results = {sym: results[sym] for sym in ordered_symbols}
    return ordered_results


@router.post("/ma/arima_above_50")
async def get_arima_ma20_above_ma50_forecast_bulk(
    symbols: List[str] = Body(..., embed=True),
    days_past: int = 100,
    forecast_days: int = 5
):
    """
    For a list of symbols, forecast if MA20 will become higher than MA50 in the next `forecast_days` days using ARIMA,
    based on the past `days_past` days of MA20/MA50 data. Returns only symbols where ma20_will_be_above_ma50 is True.
    """
    results = {}
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_ma_forecast, symbol, days_past, forecast_days)
            for symbol in symbols
        ]
        for future in as_completed(futures):
            symbol, result = future.result()
            results[symbol] = result

    print(f"Results for {len(results)} symbols: {results.keys()} results: {results}")
    
    filtered_results = {
        sym: results[sym]
        for sym in results
        if isinstance(results[sym], dict) and results[sym].get("ma20_will_be_above_ma50_and_macd_above_signal", False)
    }
    return filtered_results


@router.post("/combined/{watchlist_name}")
async def get_combined_forecast(watchlist_name: str):
    """
    For a given watchlist, return symbols that have both will_become_positive and ma20_will_be_above_ma50 set to True
    in the stock_cache table for the current date.
    """
    try:
        from db_utils import get_watchlist_symbols, get_connection, put_connection
        from macd_utils import get_latest_market_date
        
        symbols = get_watchlist_symbols(watchlist_name)
        if not symbols:
            return {"symbols": []}
        
        current_date = get_latest_market_date()
        
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT symbol 
                    FROM stock_cache
                    WHERE symbol = ANY(%s) 
                    AND date = %s
                    AND will_become_positive = TRUE
                    AND ma20_will_be_above_ma50 = TRUE
                """, (symbols, current_date))
                matching_symbols = [row[0] for row in cur.fetchall()]
        finally:
            put_connection(conn)
        
        return {"symbols": matching_symbols, "date": current_date.isoformat()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in get_combined_forecast: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
