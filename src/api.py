from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
import pandas as pd
from datetime import datetime, timedelta, time
import yfinance as yf
from db_utils import (
    create_symbol_picks_table,
    create_symbol_properties_table,
    create_table,
    create_watchlist_tables,
    create_watchlist,
    delete_watchlist,
    add_symbol_to_watchlist,
    remove_symbol_from_watchlist,
    get_watchlist_symbols,
    get_all_watchlists_with_symbols,
    create_forecast_util_table
)
from macd_utils import get_latest_market_date, get_macd_for_date, macd_crossover_signal, get_closing_prices_bulk, get_closing_prices
from picks import get_watchlist_bullish_signal, get_company_names_from_bullish_signal_result
from concurrent.futures import ProcessPoolExecutor, as_completed
import io

app = FastAPI(title="Stock MACD API", description="API for retrieving MACD indicators for stocks")

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow the frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

@app.get("/macd/{symbol}", tags=["MACD"])
async def get_macd(symbol: str):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the current date.
    """
    try:
        symbol = symbol.upper()
        today = get_latest_market_date()
        # get_macd_for_date now expects a list of symbols
        result = get_macd_for_date([symbol], today)
        return result[symbol]
    except Exception as e:
        import traceback
        print(f"Exception in get_macd: {e}")  # Print the exception error
        traceback.print_exc()  # Print the stack trace
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

@app.post("/macd/bulk", tags=["MACD"])
async def get_macd_bulk(symbols: list[str] = Body(..., embed=True)):
    """
    Get the MACD and Signal Line for a list of ticker symbols for the current date.
    """
    today = get_latest_market_date()
    # get_macd_for_date now expects a list of symbols
    symbols = [s.upper() for s in symbols]
    result = get_macd_for_date(symbols, today)
    # Return a list of results for consistency with previous API
    return [result[symbol] for symbol in symbols]

@app.post("/macd/bullish_signal", tags=["MACD"])
async def macd_bullish_signal(
    symbols: list[str] = Body(..., embed=True),
    days: int = 30,
    threshold: float = 0.05
):
    """
    For a list of ticker symbols, check if there is a bullish MACD crossover signal in the past 'days' (default 30).
    """
    symbols = [s.upper() for s in symbols]
    results = {}
    for symbol in symbols:
        try:
            signal = macd_crossover_signal(symbol, days, threshold)
            results[symbol] = signal
        except Exception as e:
            import traceback
            print(f"Exception in macd_bullish_signal for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = {"error": str(e)}
    return results

@app.post("/watchlist", tags=["Watchlist"])
async def api_create_watchlist(name: str = Body(..., embed=True)):
    """
    Create a new watchlist with the given name.
    """
    watchlist_id = create_watchlist(name)
    if watchlist_id is None:
        raise HTTPException(status_code=400, detail="Watchlist already exists")
    return {"id": watchlist_id, "name": name}

@app.delete("/watchlist", tags=["Watchlist"])
async def api_delete_watchlist(name: str = Body(..., embed=True)):
    """
    Delete a watchlist by name.
    """
    delete_watchlist(name)
    return {"deleted": name}

@app.post("/watchlist/{watchlist_name}/add_symbol", tags=["Watchlist"])
async def api_add_symbol_to_watchlist(watchlist_name: str, symbols: list[str] = Body(..., embed=True)):
    """
    Add one or more symbols to a watchlist.
    """
    added = []
    errors = []
    for symbol in symbols:
        try:
            add_symbol_to_watchlist(watchlist_name, symbol.upper())
            added.append(symbol.upper())
            today = get_latest_market_date()
            # get_macd_for_date now expects a list of symbols
            get_macd_for_date([symbol.upper()], today)
        except ValueError as e:
            errors.append({"symbol": symbol, "error": str(e)})
    if errors and not added:
        raise HTTPException(status_code=404, detail=errors)
    return {"watchlist": watchlist_name, "symbols_added": added, "errors": errors}

@app.post("/watchlist/{watchlist_name}/remove_symbol", tags=["Watchlist"])
async def api_remove_symbol_from_watchlist(watchlist_name: str, symbols: list[str] = Body(..., embed=True)):
    """
    Remove one or more symbols from a watchlist.
    """
    removed = []
    errors = []
    for symbol in symbols:
        try:
            remove_symbol_from_watchlist(watchlist_name, symbol.upper())
            removed.append(symbol.upper())
        except ValueError as e:
            import traceback
            print(f"Exception in api_remove_symbol_from_watchlist for {symbol}: {e}")
            traceback.print_exc()
            errors.append({"symbol": symbol, "error": str(e)})
    if errors and not removed:
        raise HTTPException(status_code=404, detail=errors)
    return {"watchlist": watchlist_name, "symbols_removed": removed, "errors": errors}

@app.get("/watchlist/{watchlist_name}/symbols", tags=["Watchlist"])
async def api_get_watchlist_symbols(watchlist_name: str):
    """
    Get all symbols in a watchlist.
    """
    try:
        symbols = get_watchlist_symbols(watchlist_name)
        return {"watchlist": watchlist_name, "symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/watchlists", tags=["Watchlist"])
async def api_get_all_watchlists():
    """
    Get all watchlists with their symbols.
    """
    watchlists = get_all_watchlists_with_symbols()
    return {"watchlists": watchlists}

@app.get("/watchlist/{watchlist_name}", tags=["Watchlist"])
async def api_get_watchlist(watchlist_name: str):
    """
    Get a watchlist by name, including its symbols.
    """
    try:
        symbols = get_watchlist_symbols(watchlist_name)
        return {"watchlist": watchlist_name, "symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/watchlist/{watchlist_name}/bullish_signal", tags=["Watchlist"])
async def api_watchlist_bullish_signal(
    watchlist_name: str,
    days: int = Body(30, embed=True),
    threshold: float = Body(0.05, embed=True)
):
    """
    For a given watchlist, check if there is a bullish MACD crossover signal for each symbol.
    """
    try:
        return get_watchlist_bullish_signal(watchlist_name, days, threshold)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/macd/{symbol}/history", tags=["MACD"])
async def get_macd_history(symbol: str, days: int = 60):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the past 'days' (default 60).
    """
    try:
        symbol = symbol.upper()
        end_date = get_latest_market_date()
        start_date = end_date - timedelta(days=days-1)
        from macd_utils import get_macd_for_range
        result = get_macd_for_range(symbol, start_date, end_date)
        return result
    except Exception as e:
        import traceback
        print(f"Exception in get_macd_history: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

def run_forecast(symbol, days_past, forecast_days):
    from forecast_utils import arima_macd_positive_forecast
    try:
        return symbol.upper(), arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
    except Exception as e:
        import traceback
        print(f"Exception in get_arima_macd_positive_forecast for {symbol}: {e}")
        traceback.print_exc()
        return symbol.upper(), {"error": str(e)}
    
def run_ma_forecast(symbol, days_past, forecast_days):
    from forecast_utils import arima_ma20_above_ma50_forecast
    try:
        return symbol.upper(), arima_ma20_above_ma50_forecast(symbol.upper(), days_past, forecast_days)
    except Exception as e:
        import traceback
        print(f"Exception in arima_ma20_above_ma50_forecast for {symbol}: {e}")
        traceback.print_exc()
        return symbol.upper(), {"error": str(e)}

@app.post("/macd/arima_positive_forecast", tags=["MACD"])
async def get_arima_macd_positive_forecast_bulk(
    symbols: list[str] = Body(..., embed=True),
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

    # --- Order results with improved sorting logic ---
    def get_sorting_keys(symbol):
        result = results[symbol]
        # Handle error case
        if "error" in result:
            return (False, False, False, float('-inf'), float('-inf'))
        
        will_become_positive = result.get("will_become_positive", False)
        forecast_values = result.get("forecast_values", [])
        details = result.get("details")
        last_macd = details.get("last_macd") if isinstance(details, dict) else None

        # Convert forecasted_macd from dict to list if needed
        if isinstance(result.get("forecasted_macd"), dict):
            if not forecast_values:
                forecast_values = list(result["forecasted_macd"].values())
        
        if last_macd is not None:
            if not isinstance(forecast_values, list):
                forecast_values = list(forecast_values)
            forecast_values = [last_macd, *forecast_values]

        
        
        # No forecast values case
        if not forecast_values:
            return (will_become_positive, False, False, float('-inf'), float('-inf'))
        
        # Check if values are increasing
        # Filter out None values for comparison
        valid_values = [v for v in forecast_values if v is not None]
        if len(valid_values) < 2:
            is_increasing = False
        else:
            is_increasing = all(valid_values[i] < valid_values[i+1] for i in range(len(valid_values)-1))
        
        # Check if first value is positive
        first_value_positive = False
        if forecast_values and forecast_values[0] is not None:
            first_value_positive = forecast_values[0] > 0
        
        # Find index of first positive value
        first_positive_index = float('inf')
        first_positive_value = float('inf')
        for i, value in enumerate(forecast_values):
            if value is not None and value > 0:
                first_positive_index = i
                first_positive_value = value
                break
        
        return (will_become_positive, is_increasing, first_value_positive, first_positive_index, first_positive_value)

    ordered_symbols = sorted(
        results.keys(),
        key=lambda sym: (
            not results[sym].get("will_become_positive", False),  # Will become positive first
            not get_sorting_keys(sym)[1],  # Values are increasing
            not get_sorting_keys(sym)[2],  # First value is positive
            get_sorting_keys(sym)[3],      # Index of first positive value
            get_sorting_keys(sym)[4]       # Value of first positive (for tie-breaking)
        )
    )
    ordered_results = {sym: results[sym] for sym in ordered_symbols}
    return ordered_results

@app.post("/watchlist/upload", tags=["Watchlist"])
async def api_upload_watchlist(
    watchlist_name: str = Body(..., embed=True),
    file: UploadFile = File(...)
):
    """
    Create a new watchlist from a text file containing stock symbols (one per line).
    """
    try:
        content = await file.read()
        lines = content.decode("utf-8").splitlines()
        symbols = [line.strip().upper() for line in lines if line.strip()]
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols found in file")
        watchlist_id = create_watchlist(watchlist_name)
        if watchlist_id is None:
            raise HTTPException(status_code=400, detail="Watchlist already exists")
        added = []
        errors = []
        for symbol in symbols:
            try:
                add_symbol_to_watchlist(watchlist_name, symbol)
                added.append(symbol)
            except ValueError as e:
                errors.append({"symbol": symbol, "error": str(e)})
        return {
            "watchlist": watchlist_name,
            "symbols_added": added,
            "errors": errors
        }
    except Exception as e:
        import traceback
        print(f"Exception in api_upload_watchlist: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/closing_prices/bulk", tags=["Prices"])
async def get_closing_prices_bulk_api(
    symbols: list[str] = Body(..., embed=True),
    start_date: str = Body(..., embed=True),
    end_date: str = Body(..., embed=True)
):
    """
    Get closing prices for a list of symbols between start_date and end_date (inclusive).
    Dates must be in 'YYYY-MM-DD' format.
    """
    try:
        symbols = [s.upper() for s in symbols]
        from datetime import datetime
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        result = get_closing_prices_bulk(symbols, start, end)
        return result
    except Exception as e:
        import traceback
        print(f"Exception in get_closing_prices_bulk_api: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

@app.get("/closing_prices/{symbol}", tags=["Prices"])
async def get_closing_prices_api(
    symbol: str,
    start_date: str,
    end_date: str
):
    """
    Get closing prices for a single symbol between start_date and end_date (inclusive).
    Dates must be in 'YYYY-MM-DD' format.
    """
    try:
        symbol = symbol.upper()
        from datetime import datetime
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        result = get_closing_prices(symbol, start, end)
        return result
    except Exception as e:
        import traceback
        print(f"Exception in get_closing_prices_api: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

@app.post("/watchlist/{watchlist_name}/bullish_companies_csv", tags=["Watchlist"])
async def api_watchlist_bullish_companies_csv(
    watchlist_name: str,
    days: int = Body(30, embed=True),
    threshold: float = Body(0.05, embed=True)
):
    """
    For a given watchlist, return a CSV file with columns Name, Symbol, Open Price, and Amount for all companies
    that have both ma20_just_became_above_ma50 and bullish_macd_above_signal set to True.
    The filename includes the current date and time and the watchlist name.
    """
    try:
        # Get company names and closing prices for filtered symbols
        company_dict = get_company_names_from_bullish_signal_result(watchlist_name, days, threshold)
        # Prepare CSV
        output = io.StringIO()
        output.write("Name,Symbol,Open Price,Amount\n")
        for symbol, info in company_dict.items():
            name = info.get("company_name", "")
            close = info.get("close", "")
            # Format close to 2 decimals if it's a number
            if isinstance(close, (float, int)):
                close_str = f"{close:.2f}"
            else:
                close_str = close
            # Escape quotes and commas in name if needed
            safe_name = '"' + name.replace('"', '""') + '"' if ',' in name or '"' in name else name
            output.write(f"{safe_name},{symbol},{close_str},1\n")
        output.seek(0)
        # Generate filename with current date and time and watchlist name
        from datetime import datetime
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_watchlist = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in watchlist_name)
        filename = f"{now_str}_{safe_watchlist}_bullish_companies.csv"
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        import traceback
        print(f"Exception in api_watchlist_bullish_companies_csv: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ma/arima_ma20_above_ma50_forecast", tags=["MA"])
async def get_arima_ma20_above_ma50_forecast_bulk(
    symbols: list[str] = Body(..., embed=True),
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
    # Filter to only those where ma20_will_be_above_ma50_and_macd_above_signal is True
    filtered_results = {
        sym: results[sym]
        for sym in results
        if isinstance(results[sym], dict) and results[sym].get("ma20_will_be_above_ma50_and_macd_above_signal", False)
    }
    return filtered_results

@app.post("/watchlist/{watchlist_name}/combined_forecast", tags=["Watchlist"])
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

if __name__ == "__main__":
    import uvicorn

    # First ensure the table exists
    create_table()
    create_watchlist_tables()
    create_forecast_util_table()
    create_symbol_picks_table()
    create_symbol_properties_table()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)