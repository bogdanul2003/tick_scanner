"""
Tick Scanner API - Stock Market Technical Analysis Platform

This is the main FastAPI application entry point. The API provides endpoints for:
- MACD technical indicator calculations
- Watchlist management
- Chart pattern detection (using neural networks)
- ARIMA-based forecasting
- Closing price retrieval

The application uses a modular router-based architecture for better maintainability.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add chart_scan to path for neural detector imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "chart_scan"))

# Import routers
from routers import (
    macd_router,
    watchlist_router,
    pattern_router,
    chart_router,
    price_router,
    forecast_router,
)

# Import core modules
from core.config import settings
from core.middleware import register_exception_handlers

# Import database initialization (legacy - for backwards compatibility)
from db_utils import (
    create_symbol_picks_table,
    create_symbol_properties_table,
    create_table,
    create_watchlist_tables,
    create_forecast_util_table,
)


# Initialize database tables
def init_database():
    """Initialize all required database tables."""
    logger.info("Initializing database tables...")
    create_table()
    create_watchlist_tables()
    create_forecast_util_table()
    create_symbol_picks_table()
    create_symbol_properties_table()
    logger.info("Database initialization complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan event handler."""
    # Startup
    logger.info("Starting Tick Scanner API...")
    init_database()
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Tick Scanner API...")


def create_app() -> FastAPI:
    """
    Application factory function.
    
    Creates and configures the FastAPI application with all middleware,
    routers, and static file mounts.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Tick Scanner API",
        description="Stock Market Technical Analysis Platform - MACD indicators, pattern detection, and forecasting",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Register exception handlers
    register_exception_handlers(app)
    
    # Mount static files for generated charts
    charts_dir = os.path.join(os.path.dirname(__file__), "..", "generated_charts")
    if os.path.exists(charts_dir):
        app.mount("/charts", StaticFiles(directory=charts_dir), name="charts")
    else:
        logger.warning(f"Charts directory not found: {charts_dir}")
    
    # Include routers
    app.include_router(macd_router)
    app.include_router(watchlist_router)
    app.include_router(pattern_router)
    app.include_router(chart_router)
    app.include_router(price_router)
    app.include_router(forecast_router)
    
    # Legacy endpoint aliases for backwards compatibility
    # These map old endpoints to new router structure
    
    @app.get("/watchlists", tags=["Watchlist"])
    async def get_all_watchlists_alias():
        """Alias for /watchlist/s - Get all watchlists."""
        from db_utils import get_all_watchlists_with_symbols
        watchlists = get_all_watchlists_with_symbols()
        return {"watchlists": watchlists}
    
    @app.post("/closing_prices/bulk", tags=["Prices"])
    async def closing_prices_bulk_alias(
        symbols: list[str],
        start_date: str,
        end_date: str
    ):
        """Alias for /prices/closing/bulk."""
        from macd_utils import get_closing_prices_bulk
        from datetime import datetime
        symbols = [s.upper() for s in symbols]
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        return get_closing_prices_bulk(symbols, start, end)
    
    @app.get("/closing_prices/{symbol}", tags=["Prices"])
    async def closing_prices_single_alias(symbol: str, start_date: str, end_date: str):
        """Alias for /prices/closing/{symbol}."""
        from macd_utils import get_closing_prices
        from datetime import datetime
        symbol = symbol.upper()
        start = datetime.fromisoformat(start_date).date()
        end = datetime.fromisoformat(end_date).date()
        return get_closing_prices(symbol, start, end)
    
    @app.post("/macd/arima_positive_forecast", tags=["Forecast"])
    async def arima_forecast_alias(
        symbols: list[str],
        days_past: int = 100,
        forecast_days: int = 5
    ):
        """Alias for /forecast/macd/arima_positive."""
        from services.forecast_service import forecast_service
        return forecast_service.bulk_macd_forecast(symbols, days_past, forecast_days)
    
    @app.post("/ma/arima_ma20_above_ma50_forecast", tags=["Forecast"])
    async def ma_forecast_alias(
        symbols: list[str],
        days_past: int = 100,
        forecast_days: int = 5
    ):
        """Alias for /forecast/ma/arima_above_50."""
        from services.forecast_service import forecast_service
        return forecast_service.bulk_ma_forecast(symbols, days_past, forecast_days)
    
    @app.post("/watchlist/{watchlist_name}/generate_charts", tags=["Charts"])
    async def generate_charts_alias(watchlist_name: str, selected_date: str = None):
        """Alias for /charts/watchlist/{watchlist_name}/generate."""
        from services.chart_service import chart_service
        return chart_service.generate_and_scan_watchlist(watchlist_name, selected_date)
    
    @app.get("/watchlist/{watchlist_name}/available_dates", tags=["Charts"])
    async def available_dates_alias(watchlist_name: str):
        """Alias for /charts/watchlist/{watchlist_name}/available_dates."""
        from db_utils import get_available_dates_for_watchlist
        dates = get_available_dates_for_watchlist(watchlist_name)
        return {"dates": dates}
    
    @app.post("/watchlist/{watchlist_name}/combined_forecast", tags=["Forecast"])
    async def combined_forecast_alias(watchlist_name: str):
        """Alias for /forecast/combined/{watchlist_name}."""
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
    
    return app


# Create the application instance
app = create_app()


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Tick Scanner API",
        "version": "2.0.0",
        "description": "Stock Market Technical Analysis Platform",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "api:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload
    )
