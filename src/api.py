from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Add this import
import pandas as pd
from datetime import datetime, timedelta
from forecast_utils import arima_macd_positive_forecast
import yfinance as yf
from db_utils import (
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
from macd_utils import get_macd_for_date, macd_crossover_signal

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
        today = datetime.now().date() - timedelta(days=1)
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
    today = datetime.now().date() - timedelta(days=1)
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
            today = datetime.now().date() - timedelta(days=1)
            # get_macd_for_date now expects a list of symbols
            get_macd_for_date([symbol.upper()], today)
        except ValueError as e:
            errors.append({"symbol": symbol, "error": str(e)})
    if errors and not added:
        raise HTTPException(status_code=404, detail=errors)
    return {"watchlist": watchlist_name, "symbols_added": added, "errors": errors}

@app.post("/watchlist/{watchlist_name}/remove_symbol", tags=["Watchlist"])
async def api_remove_symbol_from_watchlist(watchlist_name: str, symbol: str = Body(..., embed=True)):
    """
    Remove a symbol from a watchlist.
    """
    try:
        remove_symbol_from_watchlist(watchlist_name, symbol.upper())
        return {"watchlist": watchlist_name, "symbol_removed": symbol.upper()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

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
        symbols = get_watchlist_symbols(watchlist_name)
        if not symbols:
            raise HTTPException(status_code=404, detail=f"No symbols found in watchlist '{watchlist_name}'")
        results = {}
        for symbol in symbols:
            try:
                signal = macd_crossover_signal(symbol, days, threshold)
                results[symbol] = signal
            except Exception as e:
                import traceback
                print(f"Exception in watchlist_bullish_signal for {symbol}: {e}")
                traceback.print_exc()
                results[symbol] = {"error": str(e)}
        return {"watchlist": watchlist_name, "results": results}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/macd/{symbol}/history", tags=["MACD"])
async def get_macd_history(symbol: str, days: int = 60):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the past 'days' (default 60).
    """
    try:
        symbol = symbol.upper()
        end_date = datetime.now().date() - timedelta(days=1)
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
    for symbol in symbols:
        try:
            result = arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
            results[symbol.upper()] = result
        except Exception as e:
            import traceback
            print(f"Exception in get_arima_macd_positive_forecast for {symbol}: {e}")
            traceback.print_exc()
            results[symbol.upper()] = {"error": str(e)}
    # print(f"ARIMA MACD positive forecast results: {results}")
    return results

if __name__ == "__main__":
    import uvicorn

    # First ensure the table exists
    create_table()
    create_watchlist_tables()
    create_forecast_util_table()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)