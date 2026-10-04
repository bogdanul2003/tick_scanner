"""Custom exceptions for the tick_scanner application."""


class TickScannerError(Exception):
    """Base exception for all tick_scanner errors."""
    pass


class WatchlistNotFoundError(TickScannerError):
    """Raised when a watchlist is not found."""
    
    def __init__(self, watchlist_name: str):
        self.watchlist_name = watchlist_name
        super().__init__(f"Watchlist '{watchlist_name}' not found")


class SymbolNotFoundError(TickScannerError):
    """Raised when a stock symbol is not found."""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        super().__init__(f"Symbol '{symbol}' not found")


class ForecastError(TickScannerError):
    """Raised when forecasting fails."""
    
    def __init__(self, symbol: str, reason: str = "Unknown error"):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Forecast failed for '{symbol}': {reason}")


class DataFetchError(TickScannerError):
    """Raised when data fetching from external API fails."""
    
    def __init__(self, symbol: str, source: str = "Yahoo Finance"):
        self.symbol = symbol
        self.source = source
        super().__init__(f"Failed to fetch data for '{symbol}' from {source}")


class DatabaseError(TickScannerError):
    """Raised when a database operation fails."""
    
    def __init__(self, operation: str, reason: str = "Unknown error"):
        self.operation = operation
        self.reason = reason
        super().__init__(f"Database error during '{operation}': {reason}")


class PatternDetectionError(TickScannerError):
    """Raised when pattern detection fails."""
    
    def __init__(self, symbol: str, reason: str = "Unknown error"):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Pattern detection failed for '{symbol}': {reason}")


class InvalidImageError(TickScannerError):
    """Raised when an uploaded file is not a usable image."""

    def __init__(self, filename: str, reason: str = "Unsupported or corrupt image"):
        self.filename = filename
        self.reason = reason
        super().__init__(f"Invalid image '{filename}': {reason}")


class ImageTooLargeError(TickScannerError):
    """Raised when an uploaded image exceeds the configured size limit."""

    def __init__(self, filename: str, byte_size: int, max_bytes: int):
        self.filename = filename
        self.byte_size = byte_size
        self.max_bytes = max_bytes
        super().__init__(
            f"Image '{filename}' is {byte_size} bytes, over the {max_bytes} byte limit"
        )


class ChartGenerationError(TickScannerError):
    """Raised when chart generation fails."""
    
    def __init__(self, symbol: str, reason: str = "Unknown error"):
        self.symbol = symbol
        self.reason = reason
        super().__init__(f"Chart generation failed for '{symbol}': {reason}")
