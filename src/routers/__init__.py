# Routers package - FastAPI route modules by domain
from .macd_router import router as macd_router
from .watchlist_router import router as watchlist_router
from .pattern_router import router as pattern_router
from .chart_router import router as chart_router
from .price_router import router as price_router
from .forecast_router import router as forecast_router

__all__ = [
    "macd_router",
    "watchlist_router",
    "pattern_router",
    "chart_router",
    "price_router",
    "forecast_router",
]
