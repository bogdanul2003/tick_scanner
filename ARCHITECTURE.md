# Tick Scanner - Architecture Documentation

## Project Overview

**Tick Scanner** is a stock market technical analysis platform that identifies bullish trading signals by combining multiple analytical approaches:

- **MACD Technical Indicators**: Detects bullish crossover signals
- **Moving Average Crossovers**: Identifies MA20/MA50 trend changes  
- **ARIMA Time-Series Forecasting**: Predicts future indicator values
- **Neural Network Chart Pattern Detection**: Uses YOLO-based ML to detect visual patterns

The platform scans watchlists of stocks and prioritizes opportunities based on bullish signals, providing traders with actionable insights.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User Interface                                  │
│                         React + Vite (localhost:5173)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      FastAPI Backend (api.py)                        │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │   MACD   │ │Watchlist │ │ Patterns │ │ Forecast │ │  Charts  │  │   │
│  │  │ Endpoints│ │ Endpoints│ │ Endpoints│ │ Endpoints│ │ Endpoints│  │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │   │
│  └───────┼────────────┼────────────┼────────────┼────────────┼─────────┘   │
│          │            │            │            │            │              │
│          ▼            ▼            ▼            ▼            ▼              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌──────────┐  │
│  │macd_utils  │ │ db_utils   │ │pattern_utils│ │forecast_   │ │charts_   │  │
│  │            │ │            │ │            │ │utils       │ │generator │  │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘ └──────────┘  │
│          │            │                              │            │         │
│          ▼            ▼                              │            ▼         │
│  ┌────────────────────────────────┐                 │    ┌──────────────┐  │
│  │     PostgreSQL Database        │◄────────────────┘    │ YOLO Neural  │  │
│  │  (stock_cache, watchlists...)  │                      │   Network    │  │
│  └────────────────────────────────┘                      └──────────────┘  │
│          ▲                                                                  │
│          │                                                                  │
│  ┌───────┴──────────┐                                                      │
│  │    yfinance      │ (Yahoo Finance API)                                  │
│  │  (yaho.py)       │                                                      │
│  └──────────────────┘                                                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### Backend (`src/`)

| File | Purpose | Lines |
|------|---------|-------|
| `api.py` | FastAPI server with 20+ REST endpoints | ~800 |
| `macd_utils.py` | MACD calculations, crossover detection, caching | ~400 |
| `db_utils.py` | PostgreSQL connection pooling, CRUD operations | ~500 |
| `forecast_utils.py` | ARIMA time-series forecasting | ~300 |
| `pattern_utils.py` | Head & Shoulders pattern detection | ~200 |
| `picks.py` | Bullish signal caching and retrieval | ~150 |
| `charts_generator.py` | Candlestick chart generation with mplfinance | ~200 |
| `yaho.py` / `yaho_back.py` | Yahoo Finance data fetching & backtesting | ~300 |

### ML Pipeline (`chart_scan/`)

| File | Purpose |
|------|---------|
| `detector_neural.py` | YOLO CoreML inference for chart patterns |
| `detector_gpu.py` | GPU-accelerated pattern detection |
| `convert.py` | PyTorch to CoreML model conversion |
| `model.mlpackage/` | Pre-trained YOLO model (CoreML format) |

### Frontend (`frontend/`)

| File | Purpose |
|------|---------|
| `App.jsx` | Main React components (~500 lines) |
| `index.jsx` / `main.jsx` | React entry points |
| `vite.config.js` | Vite development server config |

---

## Data Flow

```
1. User Creates/Selects Watchlist
         │
         ▼
2. API fetches price data from Yahoo Finance (yfinance)
         │
         ▼
3. Store OHLCV data + Calculate Indicators (MACD, MA, EMA)
         │
         ├──────────────────────────────────────────────┐
         ▼                                              ▼
4a. Detect Bullish Signals              4b. Generate Charts (mplfinance)
    (MACD crossovers, MA crossovers)              │
         │                                        ▼
         │                              4c. Neural Network Pattern Detection
         │                                        │
         ▼                                        ▼
5. Cache Results in PostgreSQL          Store Pattern Results
         │
         ▼
6. Frontend displays color-coded symbols
   - 🟢 Green: MACD about to cross signal
   - 🔴 Magenta: Recent MACD crossover  
   - 🟣 Purple: Both signals active
```

---

## Database Schema

### Main Table: `stock_cache`

| Column | Type | Description |
|--------|------|-------------|
| symbol | VARCHAR | Stock ticker symbol |
| date | DATE | Trading date |
| open, high, low, close | NUMERIC | OHLC prices |
| volume | BIGINT | Trading volume |
| ema12, ema26 | NUMERIC | Exponential moving averages |
| ma20, ma50 | NUMERIC | Simple moving averages |
| macd, signal_line | NUMERIC | MACD indicators |
| will_become_positive | BOOLEAN | ARIMA forecast flag |
| ma20_will_be_above_ma50 | BOOLEAN | MA crossover forecast |
| chart_patterns | JSONB | Detected visual patterns |

### Supporting Tables

- `watchlists` - User-defined stock lists
- `watchlist_symbols` - Watchlist membership
- `symbol_picks` - Cached bullish signal results
- `forecast_util` - Trained ARIMA model cache
- `company_names` - Company metadata

---

## Technology Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, Canvas API |
| Backend | FastAPI, Python 3.x, Pydantic |
| Database | PostgreSQL 15 (Docker) |
| ML/AI | YOLO (CoreML), ARIMA (statsmodels) |
| Data API | yfinance (Yahoo Finance) |
| Visualization | mplfinance, matplotlib, OpenCV |
| Analysis | pandas, numpy, scipy |

---

## API Endpoints Summary

### MACD Endpoints
- `GET /macd/{symbol}` - Get MACD for single symbol
- `POST /macd/bulk` - Get MACD for multiple symbols
- `GET /macd/{symbol}/history` - Get MACD history
- `POST /macd/bullish_signal` - Check for crossover signals
- `POST /macd/arima_positive_forecast` - Forecast MACD becoming positive

### Watchlist Endpoints
- `POST /watchlist` - Create watchlist
- `DELETE /watchlist/{name}` - Delete watchlist
- `GET /watchlist/{name}` - Get watchlist details
- `GET /watchlists` - Get all watchlists
- `POST /watchlist/{name}/add_symbol` - Add symbols
- `POST /watchlist/{name}/remove_symbol` - Remove symbols
- `POST /watchlist/{name}/bullish_signal` - Scan for signals
- `POST /watchlist/{name}/bullish_companies_csv` - Export CSV

### Pattern Endpoints
- `GET /patterns/{symbol}` - Detect patterns for symbol
- `POST /patterns/bulk` - Bulk pattern detection
- `POST /watchlist/{name}/patterns` - Scan watchlist patterns

### Chart Endpoints
- `GET /watchlist/{name}/available_dates` - Get available dates
- `POST /watchlist/{name}/generate_charts` - Generate & scan charts

---

# Refactoring Plan for `api.py`

## Current Problems

### 1. **Monolithic Design** (~800 lines)
The `api.py` file bundles:
- HTTP request/response handling (controllers)
- Business logic (services)
- Data transformation (utilities)
- Direct database access

### 2. **Business Logic in Controllers**
Complex operations embedded directly in route handlers:
```python
# Example: Sorting logic in endpoint
def get_sorting_keys(symbol):
    result = results[symbol]
    will_become_positive = result.get("will_become_positive", False)
    # ... 30+ lines of sorting logic
```

### 3. **Inline Helper Functions**
Utility functions scattered throughout:
- `sanitize_float()` / `sanitize_record()`
- `run_forecast()` / `run_ma_forecast()`
- `upgrade_to_volume_charts()`

### 4. **Direct Database Access in Endpoints**
```python
# Example: Direct connection in controller
@app.post("/watchlist/{name}/combined_forecast")
async def get_combined_forecast(watchlist_name: str):
    conn = get_connection()  # ❌ Direct DB access
    with conn.cursor() as cur:
        cur.execute(...)
```

### 5. **Repeated Error Handling**
Same try/except pattern repeated in every endpoint with no centralized handling.

### 6. **No Dependency Injection**
Functions imported inside endpoints rather than injected, making testing difficult.

---

## Proposed Architecture

```
src/
├── api.py                    # Slim controller layer (HTTP only)
├── routers/                  # Route modules by domain
│   ├── __init__.py
│   ├── macd_router.py        # MACD endpoints
│   ├── watchlist_router.py   # Watchlist endpoints
│   ├── pattern_router.py     # Pattern endpoints
│   ├── chart_router.py       # Chart endpoints
│   └── price_router.py       # Closing price endpoints
├── services/                 # Business logic layer
│   ├── __init__.py
│   ├── macd_service.py       # MACD calculations & signals
│   ├── watchlist_service.py  # Watchlist management
│   ├── forecast_service.py   # ARIMA forecasting
│   ├── pattern_service.py    # Pattern detection
│   └── chart_service.py      # Chart generation & detection
├── repositories/             # Data access layer
│   ├── __init__.py
│   ├── stock_repository.py   # stock_cache table operations
│   ├── watchlist_repository.py  # watchlist tables
│   └── forecast_repository.py   # forecast caching
├── models/                   # Pydantic models & DTOs
│   ├── __init__.py
│   ├── requests.py           # Request schemas
│   ├── responses.py          # Response schemas
│   └── domain.py             # Domain entities
├── utils/                    # Shared utilities
│   ├── __init__.py
│   ├── sanitization.py       # JSON sanitization helpers
│   ├── date_utils.py         # Market date helpers
│   └── exceptions.py         # Custom exceptions
├── core/                     # Core application setup
│   ├── __init__.py
│   ├── config.py             # Configuration management
│   ├── database.py           # Connection pool setup
│   └── dependencies.py       # FastAPI dependencies
├── db_utils.py               # ← Keep but migrate to repositories/
├── macd_utils.py             # ← Keep but extract to services/
├── forecast_utils.py         # ← Keep but integrate with services/
├── pattern_utils.py          # ← Keep but integrate with services/
└── ...
```

---

## Refactoring Steps

### Phase 1: Extract Utilities (Low Risk)

**1.1 Create `utils/sanitization.py`**
```python
# utils/sanitization.py
import math

def sanitize_float(value):
    """Convert NaN/Infinity to None for JSON serialization."""
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value

def sanitize_record(record: dict) -> dict:
    """Sanitize all float values in a dictionary."""
    return {
        k: sanitize_float(v) if isinstance(v, (float, int, type(None))) else v 
        for k, v in record.items()
    }
```

**1.2 Create `utils/exceptions.py`**
```python
# utils/exceptions.py
class WatchlistNotFoundError(Exception):
    """Raised when a watchlist is not found."""
    pass

class SymbolNotFoundError(Exception):
    """Raised when a symbol is not found."""
    pass

class ForecastError(Exception):
    """Raised when forecasting fails."""
    pass
```

---

### Phase 2: Extract Services (Medium Risk)

**2.1 Create `services/forecast_service.py`**
```python
# services/forecast_service.py
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

class ForecastService:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
    
    def run_macd_forecast(self, symbol: str, days_past: int, forecast_days: int):
        from forecast_utils import arima_macd_positive_forecast
        return arima_macd_positive_forecast(symbol.upper(), days_past, forecast_days)
    
    def run_ma_forecast(self, symbol: str, days_past: int, forecast_days: int):
        from forecast_utils import arima_ma20_above_ma50_forecast
        return arima_ma20_above_ma50_forecast(symbol.upper(), days_past, forecast_days)
    
    def bulk_macd_forecast(
        self, 
        symbols: List[str], 
        days_past: int = 100, 
        forecast_days: int = 5
    ) -> Dict[str, Any]:
        """Run MACD forecasts in parallel and return sorted results."""
        results = {}
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.run_macd_forecast, s, days_past, forecast_days): s 
                for s in symbols
            }
            for future in as_completed(futures):
                symbol = futures[future]
                try:
                    results[symbol.upper()] = future.result()
                except Exception as e:
                    results[symbol.upper()] = {"error": str(e)}
        
        return self._sort_forecast_results(results)
    
    def _sort_forecast_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Sort results by likelihood of becoming positive."""
        # Move sorting logic here from api.py
        ...
```

**2.2 Create `services/chart_service.py`**
```python
# services/chart_service.py
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class ChartService:
    def __init__(self, charts_base_dir: str = "../generated_charts"):
        self.charts_base_dir = charts_base_dir
    
    def generate_and_scan_watchlist(
        self, 
        watchlist_name: str, 
        selected_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate charts for watchlist and run neural detection."""
        from charts_generator import generate_charts_for_watchlist, generate_symbol_chart
        from detector_neural import run_detection
        
        today_str = selected_date or datetime.now().strftime('%Y-%m-%d')
        safe_watchlist = self._sanitize_name(watchlist_name)
        base_dir = os.path.join(self.charts_base_dir, safe_watchlist, today_str)
        
        # Check cache
        cached = self._load_cached_results(base_dir)
        if cached:
            return cached
        
        # Generate charts without volume
        generate_charts_for_watchlist(watchlist_name, selected_date=selected_date, show_volume=False)
        
        # Run pattern detection
        detections_3m = run_detection(
            os.path.join(base_dir, "3m"), 
            os.path.join(base_dir, "3m", "filtered"), 
            find_x=550
        )
        detections_6m = run_detection(
            os.path.join(base_dir, "6m"), 
            os.path.join(base_dir, "6m", "filtered"), 
            find_x=565
        )
        
        # Upgrade filtered charts with volume
        self._upgrade_charts_with_volume(detections_3m, "3m", 3, base_dir, today_str)
        self._upgrade_charts_with_volume(detections_6m, "6m", 6, base_dir, today_str)
        
        # Save patterns to DB
        self._save_detected_patterns(detections_3m + detections_6m, today_str)
        
        # Build result
        result = self._build_result(
            watchlist_name, today_str, safe_watchlist, 
            detections_3m, detections_6m
        )
        
        self._cache_results(base_dir, result)
        return result
    
    def _sanitize_name(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name)
    
    # ... other private methods
```

---

### Phase 3: Create Routers (Medium Risk)

**3.1 Create `routers/macd_router.py`**
```python
# routers/macd_router.py
from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from services.macd_service import MacdService
from services.forecast_service import ForecastService
from utils.sanitization import sanitize_record

router = APIRouter(prefix="/macd", tags=["MACD"])
macd_service = MacdService()
forecast_service = ForecastService()

@router.get("/{symbol}")
async def get_macd(symbol: str):
    """Get MACD for a single symbol."""
    try:
        return macd_service.get_macd(symbol)
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": str(e)})

@router.post("/bulk")
async def get_macd_bulk(symbols: List[str] = Body(..., embed=True)):
    """Get MACD for multiple symbols."""
    return macd_service.get_macd_bulk(symbols)

@router.post("/bullish_signal")
async def macd_bullish_signal(
    symbols: List[str] = Body(..., embed=True),
    days: int = 30,
    threshold: float = 0.05
):
    """Check for bullish MACD crossover signals."""
    return macd_service.check_bullish_signals(symbols, days, threshold)

@router.get("/{symbol}/history")
async def get_macd_history(symbol: str, days: int = 60):
    """Get MACD history for a symbol."""
    history = macd_service.get_history(symbol, days)
    return [sanitize_record(r) for r in history]

@router.post("/arima_positive_forecast")
async def get_arima_forecast(
    symbols: List[str] = Body(..., embed=True),
    days_past: int = 100,
    forecast_days: int = 5
):
    """Forecast if MACD will become positive."""
    return forecast_service.bulk_macd_forecast(symbols, days_past, forecast_days)
```

**3.2 Create `routers/watchlist_router.py`**
```python
# routers/watchlist_router.py
from fastapi import APIRouter, Body, HTTPException
from typing import List
from services.watchlist_service import WatchlistService

router = APIRouter(prefix="/watchlist", tags=["Watchlist"])
watchlist_service = WatchlistService()

@router.post("")
async def create_watchlist(name: str = Body(..., embed=True)):
    """Create a new watchlist."""
    watchlist_id = watchlist_service.create(name)
    if watchlist_id is None:
        raise HTTPException(status_code=400, detail="Watchlist already exists")
    return {"id": watchlist_id, "name": name}

@router.delete("/{name}")
async def delete_watchlist(name: str):
    """Delete a watchlist."""
    watchlist_service.delete(name)
    return {"deleted": name}

@router.get("/{name}/symbols")
async def get_symbols(name: str):
    """Get all symbols in a watchlist."""
    try:
        symbols = watchlist_service.get_symbols(name)
        return {"watchlist": name, "symbols": symbols}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

# ... other endpoints
```

**3.3 Update main `api.py`**
```python
# api.py (refactored)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routers import macd_router, watchlist_router, pattern_router, chart_router, price_router
from core.database import init_database

app = FastAPI(
    title="Stock MACD API", 
    description="API for retrieving MACD indicators for stocks"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files
app.mount("/charts", StaticFiles(directory="../generated_charts"), name="charts")

# Routers
app.include_router(macd_router.router)
app.include_router(watchlist_router.router)
app.include_router(pattern_router.router)
app.include_router(chart_router.router)
app.include_router(price_router.router)

@app.on_event("startup")
async def startup():
    init_database()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
```

---

### Phase 4: Create Models (Low Risk)

**4.1 Create `models/requests.py`**
```python
# models/requests.py
from pydantic import BaseModel
from typing import List, Optional

class BulkSymbolsRequest(BaseModel):
    symbols: List[str]

class BullishSignalRequest(BaseModel):
    symbols: List[str]
    days: int = 30
    threshold: float = 0.05

class ForecastRequest(BaseModel):
    symbols: List[str]
    days_past: int = 100
    forecast_days: int = 5

class ChartGenerationRequest(BaseModel):
    selected_date: Optional[str] = None

class PatternRequest(BaseModel):
    days: int = 120
    pattern_type: str = "all"
```

**4.2 Create `models/responses.py`**
```python
# models/responses.py
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class MacdResponse(BaseModel):
    symbol: str
    date: str
    macd: Optional[float]
    signal_line: Optional[float]
    histogram: Optional[float]

class WatchlistResponse(BaseModel):
    watchlist: str
    symbols: List[str]

class ChartGenerationResponse(BaseModel):
    message: str
    watchlist: str
    date: str
    count: int
    bullish: List[str]
    bearish: List[str]
    images: List[str]
    status: str
```

---

### Phase 5: Error Handling Middleware (Low Risk)

**5.1 Create `core/middleware.py`**
```python
# core/middleware.py
from fastapi import Request
from fastapi.responses import JSONResponse
from utils.exceptions import WatchlistNotFoundError, SymbolNotFoundError
import traceback

async def exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    if isinstance(exc, WatchlistNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    if isinstance(exc, SymbolNotFoundError):
        return JSONResponse(status_code=404, content={"detail": str(exc)})
    
    # Log unexpected errors
    print(f"Unhandled exception: {exc}")
    traceback.print_exc()
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
```

---

## Migration Strategy

### Week 1: Foundation ✅ COMPLETED
1. ✅ Create folder structure (`routers/`, `services/`, `models/`, `utils/`, `core/`)
2. ✅ Extract utility functions to `utils/`
3. ✅ Create Pydantic models in `models/`
4. ⏳ Add tests for extracted utilities

### Week 2: Services Layer ✅ COMPLETED
1. ✅ Create `ForecastService` (extract from `api.py`)
2. ✅ Create `ChartService` (extract from `api.py`)
3. ⏳ Create `MacdService` (wrapper around `macd_utils.py`)
4. ⏳ Create `WatchlistService` (wrapper around `db_utils.py` functions)
5. ⏳ Add unit tests for services

### Week 3: Routers ✅ COMPLETED
1. ✅ Create `macd_router.py` and migrate MACD endpoints
2. ✅ Create `watchlist_router.py` and migrate watchlist endpoints
3. ✅ Create `pattern_router.py` and migrate pattern endpoints
4. ✅ Create `chart_router.py` and migrate chart endpoints
5. ✅ Create `price_router.py` and migrate price endpoints
6. ✅ Create `forecast_router.py` and migrate forecast endpoints

### Week 4: Integration & Cleanup
1. ✅ Update main `api.py` to use routers
2. ✅ Add error handling middleware
3. ⏳ Create integration tests
4. ⏳ Remove duplicated code
5. ⏳ Update documentation

---

## Benefits After Refactoring

| Aspect | Before | After |
|--------|--------|-------|
| **Testability** | Hard to unit test | Services can be tested independently |
| **Maintainability** | 800+ line monolith | Small, focused modules |
| **Code Reuse** | Copy-paste patterns | Shared services |
| **Error Handling** | Repeated try/except | Centralized middleware |
| **Team Scaling** | Merge conflicts | Independent routers |
| **Debugging** | Mixed concerns | Clear layer boundaries |

---

## File Dependencies (Current State)

```
api.py
├── macd_utils.py
│   └── db_utils.py
├── db_utils.py (direct imports)
├── picks.py
│   ├── macd_utils.py
│   └── db_utils.py
├── forecast_utils.py
│   └── db_utils.py
├── pattern_utils.py
│   └── db_utils.py
├── charts_generator.py
│   └── db_utils.py
└── detector_neural.py (chart_scan/)
```

---

## Running the Application

### Development Setup

```bash
# Start PostgreSQL
cd src
docker-compose up -d

# Install dependencies
pip install -r requirements.txt

# Start backend
cd src
uvicorn api:app --reload --port 8000

# Start frontend (separate terminal)
cd frontend
npm install
npm run dev
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgres://postgres:postgres@localhost:5432/postgres` | PostgreSQL connection |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API bind port |

---

## Next Steps

1. **Implement Phase 1** - Extract utilities (1-2 days)
2. **Add Tests** - pytest setup with fixtures (1 day)
3. **Implement Phase 2** - Create services (3-4 days)
4. **Review & Iterate** - Team review and adjustments (1-2 days)
