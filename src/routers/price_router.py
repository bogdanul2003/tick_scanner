"""Price-related API endpoints."""
from fastapi import APIRouter, Body
from fastapi.responses import JSONResponse
from typing import List
from datetime import datetime
import traceback

from macd_utils import get_closing_prices_bulk, get_closing_prices

router = APIRouter(prefix="/prices", tags=["Prices"])


@router.post("/closing/bulk")
async def get_closing_prices_bulk_api(
    symbols: List[str] = Body(..., embed=True),
    start_date: str = Body(..., embed=True),
    end_date: str = Body(..., embed=True)
):
    """
    Get closing prices for a list of symbols between start_date and end_date (inclusive).
    Dates must be in 'YYYY-MM-DD' format.
    """
    try:
        symbols = [s.upper() for s in symbols]
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        result = get_closing_prices_bulk(symbols, start, end)
        return result
    except Exception as e:
        print(f"Exception in get_closing_prices_bulk_api: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )


@router.get("/closing/{symbol}")
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
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        result = get_closing_prices(symbol, start, end)
        return result
    except Exception as e:
        print(f"Exception in get_closing_prices_api: {e}")
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"message": f"An error occurred: {str(e)}"}
        )
