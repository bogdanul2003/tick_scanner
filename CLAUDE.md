# CLAUDE.md

## Project Overview

Tick Scanner is a stock market technical analysis platform that identifies bullish trading signals. It combines MACD indicators, moving average crossovers, ARIMA forecasting, LSTM/GRU neural network predictions (via Apple Neural Engine), and YOLO-based chart pattern detection.

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Pydantic
- **Frontend**: React 19, Vite 7
- **Database**: PostgreSQL 15 (Docker)
- **ML Training**: PyTorch (LSTM/GRU), exported to Core ML (.mlpackage)
- **ML Inference**: Core ML on Apple Neural Engine (NPU), YOLO for chart patterns
- **Data Source**: Yahoo Finance (yfinance)
- **Charting**: mplfinance, matplotlib, OpenCV

## Project Structure

```
src/
  api.py                      # FastAPI entry point
  routers/                    # API route modules (macd, watchlist, chart, forecast, pattern, price)
  services/                   # Business logic (forecast_service, chart_service)
  models/                     # Pydantic DTOs + ML models (lstm_forecaster, neural_forecast)
  core/                       # Config, database, middleware, dependencies
  utils/                      # Sanitization, exceptions, date helpers
  scripts/                    # train_forecast_model.py, evaluate_forecast_model.py
  macd_utils.py, db_utils.py, forecast_utils.py, pattern_utils.py, charts_generator.py, picks.py
chart_scan/                   # YOLO pattern detection (detector_neural.py, detector_gpu.py)
frontend/src/                 # React app (App.jsx is the main component)
watchlists/                   # Stock symbol lists (sp500.txt, etc.)
models/                       # Trained Core ML / PyTorch model files
```

## Running the App

```bash
# Database
cd src && docker-compose up -d

# Backend (from src/)
python api.py
# or: uvicorn api:app --reload --port 8000

# Frontend (from frontend/)
npm run dev
```

- Backend: http://localhost:8000 (docs at /docs)
- Frontend: http://localhost:5173

## Key Commands

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install

# Train a forecast model (from src/)
python scripts/train_forecast_model.py --architecture bidirectional_gru --epochs 100

# Evaluate a model (from src/)
python scripts/evaluate_forecast_model.py --symbol AAPL --samples 10
```

## Architecture Notes

- The project was recently refactored from a monolithic `api.py` into a layered architecture (routers -> services -> utils). The refactoring is mostly complete on the `refactor1` branch.
- `db_utils.py` uses `psycopg2` SimpleConnectionPool (min=1, max=15).
- Neural forecasting auto-selects engine: Core ML (NPU) when available, ARIMA (CPU) as fallback.
- Supported neural architectures: `bidirectional_gru` (recommended), `stacked_gru`, `gru`, `standard_lstm`, `stacked_lstm`.
- Chart pattern detection uses YOLO via Core ML. The model is at `chart_scan/model.mlpackage`.

## Database

PostgreSQL with main table `stock_cache` (symbol, date, OHLC, volume, EMA/MA indicators, MACD, signal_line, forecast flags, chart_patterns JSONB). Supporting tables: `watchlists`, `watchlist_symbols`, `symbol_picks`, `forecast_util`, `company_names`.

Default connection: `postgres://postgres:postgres@localhost:5432/postgres`

## Code Conventions

- Backend scripts are run from the `src/` directory.
- Routers use FastAPI's `APIRouter` with prefix and tags.
- Pydantic models for request/response validation in `src/models/requests.py` and `src/models/responses.py`.
- Float sanitization (NaN/Infinity -> None) via `utils/sanitization.py` before JSON responses.
- Custom exceptions in `utils/exceptions.py`, handled by centralized middleware.
- macOS-specific: uses matplotlib Agg backend (headless), Core ML for NPU inference.
