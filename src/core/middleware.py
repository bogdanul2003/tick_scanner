"""Error handling middleware and exception handlers."""
from fastapi import Request
from fastapi.responses import JSONResponse
import traceback
import logging

from utils.exceptions import (
    TickScannerError,
    WatchlistNotFoundError,
    SymbolNotFoundError,
    ForecastError,
    DataFetchError,
    DatabaseError,
    PatternDetectionError,
    ChartGenerationError,
)

logger = logging.getLogger(__name__)


async def tick_scanner_exception_handler(request: Request, exc: TickScannerError):
    """Handle all TickScanner custom exceptions."""
    
    if isinstance(exc, WatchlistNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "message": str(exc),
                "error_code": "WATCHLIST_NOT_FOUND",
                "detail": {"watchlist_name": exc.watchlist_name}
            }
        )
    
    if isinstance(exc, SymbolNotFoundError):
        return JSONResponse(
            status_code=404,
            content={
                "message": str(exc),
                "error_code": "SYMBOL_NOT_FOUND",
                "detail": {"symbol": exc.symbol}
            }
        )
    
    if isinstance(exc, ForecastError):
        return JSONResponse(
            status_code=422,
            content={
                "message": str(exc),
                "error_code": "FORECAST_ERROR",
                "detail": {"symbol": exc.symbol, "reason": exc.reason}
            }
        )
    
    if isinstance(exc, DataFetchError):
        return JSONResponse(
            status_code=502,
            content={
                "message": str(exc),
                "error_code": "DATA_FETCH_ERROR",
                "detail": {"symbol": exc.symbol, "source": exc.source}
            }
        )
    
    if isinstance(exc, DatabaseError):
        logger.error(f"Database error: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "message": "Database operation failed",
                "error_code": "DATABASE_ERROR"
            }
        )
    
    if isinstance(exc, (PatternDetectionError, ChartGenerationError)):
        return JSONResponse(
            status_code=500,
            content={
                "message": str(exc),
                "error_code": exc.__class__.__name__.upper(),
                "detail": {"symbol": exc.symbol, "reason": exc.reason}
            }
        )
    
    # Generic TickScanner error
    return JSONResponse(
        status_code=500,
        content={
            "message": str(exc),
            "error_code": "TICK_SCANNER_ERROR"
        }
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    traceback.print_exc()
    
    return JSONResponse(
        status_code=500,
        content={
            "message": "An unexpected error occurred",
            "error_code": "INTERNAL_ERROR"
        }
    )


def register_exception_handlers(app):
    """Register all exception handlers with the FastAPI app."""
    app.add_exception_handler(TickScannerError, tick_scanner_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)
