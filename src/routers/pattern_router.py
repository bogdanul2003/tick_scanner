"""Pattern detection API endpoints."""
from fastapi import APIRouter, Body, HTTPException
from typing import List
import traceback

from pattern_utils import scan_symbol_for_patterns, scan_watchlist_for_patterns

router = APIRouter(prefix="/patterns", tags=["Patterns"])


@router.get("/{symbol}")
async def get_symbol_patterns(
    symbol: str,
    days: int = 120,
    pattern_type: str = "all"
):
    """
    Scan a symbol for chart patterns (Head and Shoulders, Inverse Head and Shoulders).
    
    Args:
        symbol: Stock symbol to scan
        days: Number of days of historical data to analyze (default 120)
        pattern_type: 'all', 'head_and_shoulders', or 'inverse_head_and_shoulders'
    
    Returns:
        List of detected patterns with details
    """
    try:
        symbol = symbol.upper()
        patterns = scan_symbol_for_patterns(symbol, days, pattern_type)
        
        return {
            "symbol": symbol,
            "days_analyzed": days,
            "pattern_type": pattern_type,
            "patterns_found": len(patterns),
            "patterns": patterns
        }
    except Exception as e:
        print(f"Exception in get_symbol_patterns: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/bulk")
async def get_bulk_patterns(
    symbols: List[str] = Body(..., embed=True),
    days: int = 120,
    pattern_type: str = "all"
):
    """
    Scan multiple symbols for chart patterns.
    
    Args:
        symbols: List of stock symbols to scan
        days: Number of days of historical data to analyze (default 120)
        pattern_type: 'all', 'head_and_shoulders', or 'inverse_head_and_shoulders'
    
    Returns:
        Dictionary with symbols as keys and their detected patterns
    """
    try:
        results = {}
        for symbol in symbols:
            symbol = symbol.upper()
            try:
                patterns = scan_symbol_for_patterns(symbol, days, pattern_type)
                if patterns:
                    results[symbol] = patterns
            except Exception as e:
                print(f"Error scanning {symbol}: {e}")
                continue
        
        return {
            "days_analyzed": days,
            "pattern_type": pattern_type,
            "symbols_with_patterns": len(results),
            "results": results
        }
    except Exception as e:
        print(f"Exception in get_bulk_patterns: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/watchlist/{watchlist_name}")
async def get_watchlist_patterns(
    watchlist_name: str,
    days: int = Body(120, embed=True),
    pattern_type: str = Body("all", embed=True)
):
    """
    Scan all symbols in a watchlist for chart patterns.
    
    Args:
        watchlist_name: Name of the watchlist to scan
        days: Number of days of historical data to analyze (default 120)
        pattern_type: 'all', 'head_and_shoulders', or 'inverse_head_and_shoulders'
    
    Returns:
        Dictionary with symbols that have detected patterns
    """
    try:
        results = scan_watchlist_for_patterns(watchlist_name, days, pattern_type)
        
        # Count patterns by type
        pattern_counts = {"head_and_shoulders": 0, "inverse_head_and_shoulders": 0}
        for symbol_patterns in results.values():
            for pattern in symbol_patterns:
                if pattern["pattern_type"] in pattern_counts:
                    pattern_counts[pattern["pattern_type"]] += 1
        
        return {
            "watchlist": watchlist_name,
            "days_analyzed": days,
            "pattern_type_filter": pattern_type,
            "symbols_with_patterns": len(results),
            "pattern_counts": pattern_counts,
            "results": results
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        print(f"Exception in get_watchlist_patterns: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
