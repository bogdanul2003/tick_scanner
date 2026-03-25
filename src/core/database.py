"""Database connection management and initialization."""
import psycopg2
from psycopg2 import pool
from contextlib import contextmanager
from typing import Optional, Generator
import logging

from .config import settings

logger = logging.getLogger(__name__)

# Global connection pool
_connection_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def init_database() -> None:
    """
    Initialize the database connection pool and create required tables.
    
    This should be called once at application startup.
    """
    global _connection_pool
    
    if _connection_pool is not None:
        logger.warning("Database already initialized, skipping...")
        return
    
    logger.info(f"Initializing database connection pool to {settings.db_host}:{settings.db_port}/{settings.db_name}")
    
    _connection_pool = psycopg2.pool.SimpleConnectionPool(
        minconn=settings.db_pool_min,
        maxconn=settings.db_pool_max,
        **settings.db_params
    )
    
    # Create tables
    _create_tables()
    
    logger.info("Database initialization complete")


def _create_tables() -> None:
    """Create all required database tables if they don't exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            # Stock cache table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stock_cache (
                    symbol TEXT,
                    date DATE,
                    open FLOAT,
                    high FLOAT,
                    low FLOAT,
                    close FLOAT,
                    volume BIGINT,
                    ema12 FLOAT,
                    ema26 FLOAT,
                    ma20 FLOAT,
                    ma50 FLOAT,
                    macd FLOAT,
                    signal_line FLOAT,
                    will_become_positive BOOLEAN DEFAULT FALSE,
                    ma20_will_be_above_ma50 BOOLEAN DEFAULT FALSE,
                    patterns JSONB,
                    PRIMARY KEY (symbol, date)
                )
            """)
            
            # Watchlists table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id SERIAL PRIMARY KEY,
                    name TEXT UNIQUE NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Watchlist symbols table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_symbols (
                    id SERIAL PRIMARY KEY,
                    watchlist_id INTEGER REFERENCES watchlists(id) ON DELETE CASCADE,
                    symbol TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(watchlist_id, symbol)
                )
            """)
            
            # Symbol picks table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS symbol_picks (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    filters JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, date)
                )
            """)
            
            # Forecast utility table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS forecast_util (
                    id SERIAL PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    model_type TEXT NOT NULL,
                    model_data BYTEA,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, model_type)
                )
            """)
            
            # Company names table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS company_names (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info("Database tables created/verified")
            
    except Exception as e:
        conn.rollback()
        logger.error(f"Error creating tables: {e}")
        raise
    finally:
        release_db_connection(conn)


def get_db_connection():
    """
    Get a connection from the pool.
    
    Returns:
        A database connection from the pool
        
    Raises:
        RuntimeError: If the database hasn't been initialized
    """
    global _connection_pool
    
    if _connection_pool is None:
        # Auto-initialize if not done
        init_database()
    
    return _connection_pool.getconn()


def release_db_connection(conn) -> None:
    """
    Return a connection to the pool.
    
    Args:
        conn: The connection to return
    """
    global _connection_pool
    
    if _connection_pool is not None and conn is not None:
        _connection_pool.putconn(conn)


@contextmanager
def db_connection() -> Generator:
    """
    Context manager for database connections.
    
    Automatically returns the connection to the pool when done.
    
    Usage:
        with db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM table")
    """
    conn = get_db_connection()
    try:
        yield conn
    finally:
        release_db_connection(conn)


@contextmanager
def db_cursor(commit: bool = True) -> Generator:
    """
    Context manager for database cursors with automatic commit/rollback.
    
    Args:
        commit: Whether to commit on success (default True)
    
    Usage:
        with db_cursor() as cur:
            cur.execute("INSERT INTO table VALUES (%s)", (value,))
    """
    with db_connection() as conn:
        cur = conn.cursor()
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()


def close_pool() -> None:
    """
    Close all connections in the pool.
    
    Should be called on application shutdown.
    """
    global _connection_pool
    
    if _connection_pool is not None:
        _connection_pool.closeall()
        _connection_pool = None
        logger.info("Database connection pool closed")
