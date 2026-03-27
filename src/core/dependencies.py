"""FastAPI dependencies for dependency injection."""
from fastapi import Depends, HTTPException, status
from typing import Generator, Optional
import logging

from .database import get_db_connection, release_db_connection
from .config import settings

logger = logging.getLogger(__name__)


def get_db() -> Generator:
    """
    FastAPI dependency for database connection.
    
    Yields a connection and returns it to pool after request.
    
    Usage:
        @app.get("/items")
        def get_items(db = Depends(get_db)):
            with db.cursor() as cur:
                cur.execute("SELECT * FROM items")
                return cur.fetchall()
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        release_db_connection(conn)


def get_settings():
    """
    FastAPI dependency for application settings.
    
    Usage:
        @app.get("/config")
        def get_config(settings = Depends(get_settings)):
            return {"host": settings.api_host}
    """
    return settings


class RateLimiter:
    """
    Simple rate limiter dependency.
    
    Can be extended with Redis for distributed rate limiting.
    """
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self._requests = {}
    
    async def __call__(self, request_id: Optional[str] = None):
        """Check rate limit for the request."""
        # Placeholder - implement with Redis for production
        pass


# Common dependencies
rate_limiter = RateLimiter()
