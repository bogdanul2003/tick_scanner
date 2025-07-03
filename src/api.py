from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
from db_utils import (
    create_table,
)
from macd_utils import calculate_macd_and_signal, get_macd_for_date

app = FastAPI(title="Stock MACD API", description="API for retrieving MACD indicators for stocks")

@app.get("/macd/{symbol}")
async def get_macd(symbol: str):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the current date.
    
    Parameters:
    - symbol: The stock ticker symbol (e.g., "AAPL", "MSFT")
    
    Returns:
    - JSON object with symbol, date, close price, MACD, and Signal Line values
    """
    try:
        symbol = symbol.upper()
        today = datetime.now().date() - timedelta(days=1)
        return get_macd_for_date(symbol, today)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )

@app.post("/macd/bulk")
async def get_macd_bulk(symbols: list[str] = Body(..., embed=True)):
    """
    Get the MACD and Signal Line for a list of ticker symbols for the current date.

    Request body:
    {
        "symbols": ["AAPL", "MSFT", ...]
    }

    Returns:
    - List of JSON objects with symbol, date, close price, MACD, and Signal Line values
    """
    today = datetime.now().date() - timedelta(days=1)
    results = []
    for symbol in symbols:
        try:
            result = get_macd_for_date(symbol.upper(), today)
        except Exception as e:
            result = {
                "symbol": symbol.upper(),
                "error": str(e)
            }
        results.append(result)
    return results

if __name__ == "__main__":
    import uvicorn

    # First ensure the table exists
    create_table()
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)