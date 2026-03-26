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
