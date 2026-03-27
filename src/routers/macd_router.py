"""MACD-related API endpoints."""
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from datetime import timedelta
import traceback

from utils.sanitization import sanitize_record
from macd_utils import (
    get_latest_market_date,
    get_macd_for_date,
    macd_crossover_signal,
    get_macd_for_range,
)

router = APIRouter(prefix="/macd", tags=["MACD"])


@router.get("/{symbol}")
async def get_macd(symbol: str):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the current date.
    """
    try:
        symbol = symbol.upper()
        today = get_latest_market_date()
        result = get_macd_for_date([symbol], today)
        return result[symbol]
    except Exception as e:
        print(f"Exception in get_macd: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )


@router.post("/bulk")
async def get_macd_bulk(symbols: List[str] = Body(..., embed=True)):
    """
    Get the MACD and Signal Line for a list of ticker symbols for the current date.
    """
    today = get_latest_market_date()
    symbols = [s.upper() for s in symbols]
    result = get_macd_for_date(symbols, today)
    return [result[symbol] for symbol in symbols]


@router.post("/bullish_signal")
async def macd_bullish_signal(
    symbols: List[str] = Body(..., embed=True),
    days: int = 30,
    threshold: float = 0.05
):
    """
    For a list of ticker symbols, check if there is a bullish MACD crossover signal 
    in the past 'days' (default 30).
    """
    symbols = [s.upper() for s in symbols]
    results = {}
    for symbol in symbols:
        try:
            signal = macd_crossover_signal(symbol, days, threshold)
            results[symbol] = signal
        except Exception as e:
            print(f"Exception in macd_bullish_signal for {symbol}: {e}")
            traceback.print_exc()
            results[symbol] = {"error": str(e)}
    return results


@router.get("/{symbol}/history")
async def get_macd_history(symbol: str, days: int = 60):
    """
    Get the MACD and Signal Line for the specified ticker symbol for the past 'days' (default 60).
    """
    try:
        symbol = symbol.upper()
        end_date = get_latest_market_date()
        start_date = end_date - timedelta(days=days - 1)
        result = get_macd_for_range(symbol, start_date, end_date)
        sanitized_result = [sanitize_record(r) for r in result]
        return sanitized_result
    except Exception as e:
        print(f"Exception in get_macd_history: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )
