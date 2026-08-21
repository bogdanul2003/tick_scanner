"""Request schemas for the API endpoints."""
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional


class BulkSymbolsRequest(BaseModel):
    """Request model for bulk symbol operations."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")
    
    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class BullishSignalRequest(BaseModel):
    """Request model for bullish signal detection."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")
    days: int = Field(default=30, ge=1, le=365, description="Number of days to analyze")
    threshold: float = Field(default=0.05, ge=0, le=1, description="Signal threshold")
    
    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class ForecastRequest(BaseModel):
    """Request model for ARIMA forecasting."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")
    days_past: int = Field(default=100, ge=30, le=365, description="Days of historical data")
    forecast_days: int = Field(default=5, ge=1, le=30, description="Days to forecast")
    
    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class ChartGenerationRequest(BaseModel):
    """Request model for chart generation."""
    selected_date: Optional[str] = Field(
        default=None,
        description="Date for chart generation (YYYY-MM-DD format)"
    )
    
    @field_validator("selected_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class PatternRequest(BaseModel):
    """Request model for pattern detection."""
    days: int = Field(default=120, ge=30, le=365, description="Days of data to analyze")
    pattern_type: str = Field(
        default="all",
        description="Pattern type: 'all', 'head_and_shoulders', 'inverse_head_and_shoulders'"
    )
    
    @field_validator("pattern_type")
    @classmethod
    def validate_pattern_type(cls, v: str) -> str:
        valid_types = {"all", "head_and_shoulders", "inverse_head_and_shoulders"}
        if v.lower() not in valid_types:
            raise ValueError(f"pattern_type must be one of: {valid_types}")
        return v.lower()


class WatchlistSymbolsRequest(BaseModel):
    """Request model for adding/removing symbols from watchlist."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")
    
    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class WatchlistCreateRequest(BaseModel):
    """Request model for creating a watchlist."""
    name: str = Field(..., min_length=1, max_length=100, description="Watchlist name")


class ClosingPricesRequest(BaseModel):
    """Request model for getting closing prices."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")
    
    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]
    
    @field_validator("start_date", "end_date")
    @classmethod
    def validate_date_format(cls, v: str) -> str:
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")


class WatchlistBullishSignalRequest(BaseModel):
    """Request model for watchlist bullish signal check."""
    days: int = Field(default=30, ge=1, le=365, description="Number of days to analyze")
    threshold: float = Field(default=0.05, ge=0, le=1, description="Signal threshold")


class NoteWatchlistCreateRequest(BaseModel):
    """Request model for creating a notes watchlist."""
    name: str = Field(..., min_length=1, max_length=100, description="Notes watchlist name")
    description: Optional[str] = Field(default=None, description="Notes watchlist description")


class NoteWatchlistUpdateRequest(BaseModel):
    """Request model for updating a notes watchlist."""
    name: Optional[str] = Field(default=None, min_length=1, max_length=100, description="New notes watchlist name")
    description: Optional[str] = Field(default=None, description="New notes watchlist description")


class NoteSymbolsRequest(BaseModel):
    """Request model for adding symbols to a notes watchlist."""
    symbols: List[str] = Field(..., min_length=1, description="List of stock symbols")

    @field_validator("symbols")
    @classmethod
    def uppercase_symbols(cls, v: List[str]) -> List[str]:
        return [s.upper().strip() for s in v]


class NoteCreateRequest(BaseModel):
    """Request model for creating a note."""
    body: str = Field(..., description="Note content in Markdown")
    note_date: Optional[str] = Field(default=None, description="Note date (YYYY-MM-DD)")
    status: Optional[str] = Field(default="INITIAL", description="Note status: INITIAL, CONFIRMED, WRONG")

    @field_validator("note_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return "INITIAL"
        val = v.upper().strip()
        if val not in ("INITIAL", "CONFIRMED", "WRONG"):
            raise ValueError("status must be INITIAL, CONFIRMED, or WRONG")
        return val


class NoteUpdateRequest(BaseModel):
    """Request model for updating a note."""
    body: Optional[str] = Field(default=None, description="Updated note content in Markdown")
    note_date: Optional[str] = Field(default=None, description="Updated note date (YYYY-MM-DD)")
    status: Optional[str] = Field(default=None, description="Updated note status: INITIAL, CONFIRMED, WRONG")

    @field_validator("note_date")
    @classmethod
    def validate_date_format(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        from datetime import datetime
        try:
            datetime.strptime(v, "%Y-%m-%d")
            return v
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        val = v.upper().strip()
        if val not in ("INITIAL", "CONFIRMED", "WRONG"):
            raise ValueError("status must be INITIAL, CONFIRMED, or WRONG")
        return val


