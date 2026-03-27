"""Watchlist-related API endpoints."""
from fastapi import APIRouter, Body, HTTPException, Form, UploadFile, File
from fastapi.responses import StreamingResponse
from typing import List
from datetime import datetime
import io
import traceback

from db_utils import (
    create_watchlist,
    delete_watchlist,
    add_symbol_to_watchlist,
    remove_symbol_from_watchlist,
    get_watchlist_symbols,
    get_all_watchlists_with_symbols,
)
from macd_utils import get_latest_market_date, get_macd_for_date, refresh_watchlist_data
from picks import get_watchlist_bullish_signal, get_company_names_from_bullish_signal_result

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])


@router.post("")
async def api_create_watchlist(name: str = Body(..., embed=True)):
    """Create a new watchlist with the given name."""
    watchlist_id = create_watchlist(name)
    if watchlist_id is None:
        raise HTTPException(status_code=400, detail="Watchlist already exists")
    return {"id": watchlist_id, "name": name}


@router.delete("/{name}")
async def api_delete_watchlist(name: str):
    """Delete a watchlist by name."""
    delete_watchlist(name)
    return {"deleted": name}


@router.post("/{watchlist_name}/add_symbol")
async def api_add_symbol_to_watchlist(
    watchlist_name: str, 
    symbols: List[str] = Body(..., embed=True)
):
    """Add one or more symbols to a watchlist."""
    added = []
    errors = []
    for symbol in symbols:
        try:
            add_symbol_to_watchlist(watchlist_name, symbol.upper())
            added.append(symbol.upper())
            today = get_latest_market_date()
            get_macd_for_date([symbol.upper()], today)
        except ValueError as e:
            errors.append({"symbol": symbol, "error": str(e)})
    if errors and not added:
        raise HTTPException(status_code=404, detail=errors)
    return {"watchlist": watchlist_name, "symbols_added": added, "errors": errors}


@router.post("/{watchlist_name}/remove_symbol")
async def api_remove_symbol_from_watchlist(
    watchlist_name: str, 
    symbols: List[str] = Body(..., embed=True)
):
    """Remove one or more symbols from a watchlist."""
    removed = []
    errors = []
    for symbol in symbols:
        try:
            remove_symbol_from_watchlist(watchlist_name, symbol.upper())
            removed.append(symbol.upper())
        except ValueError as e:
            print(f"Exception in api_remove_symbol_from_watchlist for {symbol}: {e}")
            traceback.print_exc()
            errors.append({"symbol": symbol, "error": str(e)})
    if errors and not removed:
        raise HTTPException(status_code=404, detail=errors)
    return {"watchlist": watchlist_name, "symbols_removed": removed, "errors": errors}


@router.post("/{watchlist_name}/refresh")
async def api_refresh_watchlist(watchlist_name: str, days_back: int = 365):
    """Refresh missing/null OHLCV data for all symbols in a watchlist."""
    try:
        result = refresh_watchlist_data(watchlist_name, days_back=days_back)
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in api_refresh_watchlist: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{watchlist_name}/symbols")
async def api_get_watchlist_symbols(watchlist_name: str):
    """Get all symbols in a watchlist."""
    try:
        symbols = get_watchlist_symbols(watchlist_name)
        return {"watchlist": watchlist_name, "symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("s")
async def api_get_all_watchlists():
    """Get all watchlists with their symbols."""
    watchlists = get_all_watchlists_with_symbols()
    return {"watchlists": watchlists}


@router.get("/{watchlist_name}")
async def api_get_watchlist(watchlist_name: str):
    """Get a watchlist by name, including its symbols."""
    try:
        symbols = get_watchlist_symbols(watchlist_name)
        return {"watchlist": watchlist_name, "symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{watchlist_name}/bullish_signal")
async def api_watchlist_bullish_signal(
    watchlist_name: str,
    days: int = Body(30, embed=True),
    threshold: float = Body(0.05, embed=True)
):
    """For a given watchlist, check if there is a bullish MACD crossover signal for each symbol."""
    try:
        return get_watchlist_bullish_signal(watchlist_name, days, threshold)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def api_upload_watchlist(
    watchlist_name: str = Form(...),
    file: UploadFile = File(...)
):
    """Create a new watchlist from a text file containing stock symbols (one per line)."""
    try:
        content = await file.read()
        lines = content.decode("utf-8").splitlines()
        symbols = [line.strip().upper() for line in lines if line.strip()]
        if not symbols:
            raise HTTPException(status_code=400, detail="No symbols found in file")
        
        create_watchlist(watchlist_name)
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
        print(f"Exception in api_upload_watchlist: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{watchlist_name}/bullish_companies_csv")
async def api_watchlist_bullish_companies_csv(
    watchlist_name: str,
    days: int = Body(30, embed=True),
    threshold: float = Body(0.05, embed=True)
):
    """
    For a given watchlist, return a CSV file with columns Name, Symbol, Open Price, and Amount 
    for all companies that have both ma20_just_became_above_ma50 and bullish_macd_above_signal set to True.
    """
    try:
        company_dict = get_company_names_from_bullish_signal_result(watchlist_name, days, threshold)
        
        output = io.StringIO()
        output.write("Name,Symbol,Open Price,Amount\n")
        for symbol, info in company_dict.items():
            name = info.get("company_name", "")
            close = info.get("close", "")
            if isinstance(close, (float, int)):
                close_str = f"{close:.2f}"
            else:
                close_str = close
            safe_name = '"' + name.replace('"', '""') + '"' if ',' in name or '"' in name else name
            output.write(f"{safe_name},{symbol},{close_str},1\n")
        output.seek(0)
        
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_watchlist = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in watchlist_name)
        filename = f"{now_str}_{safe_watchlist}_bullish_companies.csv"
        
        return StreamingResponse(
            output,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        print(f"Exception in api_watchlist_bullish_companies_csv: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
