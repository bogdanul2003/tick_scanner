"""Response schemas for the API endpoints."""
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from datetime import date


class MacdResponse(BaseModel):
    """Response model for MACD data."""
    symbol: str
    date: str
    macd: Optional[float] = None
    signal_line: Optional[float] = None
    histogram: Optional[float] = None
    close: Optional[float] = None
    ema12: Optional[float] = None
    ema26: Optional[float] = None


class MacdHistoryResponse(BaseModel):
    """Response model for MACD history."""
    symbol: str
    date: str
    macd: Optional[float] = None
    signal_line: Optional[float] = None
    close: Optional[float] = None


class BullishSignalResponse(BaseModel):
    """Response model for bullish signal detection."""
    symbol: str
    has_signal: bool
    signal_type: Optional[str] = None
    days_ago: Optional[int] = None
    details: Optional[Dict[str, Any]] = None


class WatchlistResponse(BaseModel):
    """Response model for watchlist operations."""
    watchlist: str
    symbols: List[str]


class WatchlistListResponse(BaseModel):
    """Response model for listing all watchlists."""
    watchlists: List[Dict[str, Any]]


class WatchlistModifyResponse(BaseModel):
    """Response model for watchlist symbol modifications."""
    watchlist: str
    symbols_added: Optional[List[str]] = None
    symbols_removed: Optional[List[str]] = None
    errors: List[Dict[str, str]] = Field(default_factory=list)


class ForecastResponse(BaseModel):
    """Response model for forecast results."""
    symbol: str
    will_become_positive: bool
    forecast_values: Optional[List[float]] = None
    forecasted_macd: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None


class ChartGenerationResponse(BaseModel):
    """Response model for chart generation."""
    message: str
    watchlist: str
    date: str
    count: int
    bullish: List[str]
    bearish: List[str]
    images: List[str]
    status: str


class PatternDetail(BaseModel):
    """Detail model for a detected pattern."""
    pattern_type: str
    confidence: float
    head_price: Optional[float] = None
    left_shoulder_price: Optional[float] = None
    right_shoulder_price: Optional[float] = None
    neckline: Optional[float] = None


class PatternResponse(BaseModel):
    """Response model for pattern detection."""
    symbol: str
    days_analyzed: int
    pattern_type: str
    patterns_found: int
    patterns: List[PatternDetail]


class BulkPatternResponse(BaseModel):
    """Response model for bulk pattern detection."""
    days_analyzed: int
    pattern_type: str
    symbols_with_patterns: int
    results: Dict[str, List[PatternDetail]]


class ClosingPriceResponse(BaseModel):
    """Response model for closing prices."""
    symbol: str
    prices: Dict[str, float]  # date -> price


class ErrorResponse(BaseModel):
    """Standard error response model."""
    message: str
    detail: Optional[str] = None
    error_code: Optional[str] = None


class AvailableDatesResponse(BaseModel):
    """Response model for available dates."""
    dates: List[str]


class CombinedForecastResponse(BaseModel):
    """Response model for combined forecast."""
    symbols: List[str]
    date: str


class NoteWatchlistSummary(BaseModel):
    """Summary response model for a notes watchlist."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    symbol_count: int
    note_count: int


class NoteSymbolSummary(BaseModel):
    """Summary of notes for a symbol in a notes watchlist."""
    symbol: str
    note_count: int
    last_note_date: Optional[str] = None


class NoteWatchlistDetailResponse(BaseModel):
    """Detailed response model for a notes watchlist."""
    id: int
    name: str
    description: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    symbols: List[NoteSymbolSummary] = Field(default_factory=list)


class NoteImageResponse(BaseModel):
    """Response model for a note attachment image."""
    id: int
    filename: Optional[str] = None
    mime_type: str
    byte_size: int
    width: Optional[int] = None
    height: Optional[int] = None
    sort_order: int
    url: str
    thumb_url: str


class NoteResponse(BaseModel):
    """Response model for a note."""
    id: int
    watchlist_id: int
    symbol: str
    note_date: str
    body: str
    status: str = "INITIAL"
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    images: List[NoteImageResponse] = Field(default_factory=list)


