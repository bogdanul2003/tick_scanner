"""Date utilities for market date calculations."""
from datetime import date, datetime, time, timedelta
from typing import Optional
import functools


def is_market_open_day(d: date) -> bool:
    """
    Check if a given date is a potential market trading day.
    
    This checks for weekends but does not account for holidays.
    
    Args:
        d: The date to check
        
    Returns:
        True if the date is a weekday (Mon-Fri)
    """
    return d.weekday() < 5  # Monday = 0, Friday = 4


def get_previous_market_day(d: date) -> date:
    """
    Get the previous market trading day.
    
    Args:
        d: The reference date
        
    Returns:
        The most recent weekday before the given date
    """
    prev = d - timedelta(days=1)
    while not is_market_open_day(prev):
        prev -= timedelta(days=1)
    return prev


@functools.lru_cache(maxsize=1)
def _get_cached_market_date(today_str: str) -> date:
    """
    Internal function to cache market date calculation.
    
    Cache is invalidated when the date changes.
    """
    today = datetime.strptime(today_str, "%Y-%m-%d").date()
    now = datetime.now()
    
    # Market closes at 4:00 PM ET (roughly 16:00)
    # If it's after market close, use today's date
    # If it's before market open or during weekend, use last trading day
    market_close = time(16, 0)
    
    if now.time() >= market_close and is_market_open_day(today):
        return today
    elif is_market_open_day(today) and now.time() >= time(9, 30):
        # Market is open, use today
        return today
    else:
        # Use previous trading day
        return get_previous_market_day(today)


def get_latest_market_date() -> date:
    """
    Get the most recent market trading date.
    
    If the market is currently open or has closed today, returns today.
    Otherwise returns the previous trading day.
    
    Returns:
        The most recent trading date
    """
    today_str = datetime.now().strftime("%Y-%m-%d")
    return _get_cached_market_date(today_str)


def get_date_range(days_back: int, end_date: Optional[date] = None) -> tuple[date, date]:
    """
    Calculate a date range going back from the end date.
    
    Args:
        days_back: Number of days to go back
        end_date: The end date (defaults to latest market date)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if end_date is None:
        end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=days_back - 1)
    return start_date, end_date


def parse_date(date_str: str) -> date:
    """
    Parse a date string in YYYY-MM-DD format.
    
    Args:
        date_str: Date string in ISO format
        
    Returns:
        Parsed date object
        
    Raises:
        ValueError: If the date string is invalid
    """
    return datetime.fromisoformat(date_str).date()


def format_date(d: date) -> str:
    """
    Format a date as YYYY-MM-DD string.
    
    Args:
        d: Date to format
        
    Returns:
        ISO formatted date string
    """
    return d.strftime("%Y-%m-%d")
