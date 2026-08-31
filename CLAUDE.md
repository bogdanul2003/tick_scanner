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
  routers/                    # API route modules (macd, watchlist, chart, forecast, pattern, price, notes)
  services/                   # Business logic (forecast_service, chart_service)
  models/                     # Pydantic DTOs + ML models (lstm_forecaster, neural_forecast)
  core/                       # Config, database, middleware, dependencies
  utils/                      # Sanitization, exceptions, date helpers
  scripts/                    # train_forecast_model.py, evaluate_forecast_model.py
  tests/                      # unittest suite (no DB / Core ML needed — both stubbed)
  macd_utils.py, db_utils.py, forecast_utils.py, pattern_utils.py, charts_generator.py, picks.py
  notes_db.py                 # Notes Dashboard tables + CRUD
chart_scan/                   # YOLO pattern detection (detector_neural.py, detector_gpu.py)
frontend/src/
  App.jsx                     # Shell: hash-routed dashboard switch (#/, #/macd, #/notes)
  api.js                      # Shared API base URL + fetch helpers
  pages/HomePage.jsx          # Start page
  dashboards/MacdDashboard.jsx  # MACD/forecast/pattern UI
  dashboards/notes/           # Notes Dashboard components
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

# Run the tests (from src/)
python -m unittest discover tests
```

## Tests

`src/tests/` uses the stdlib `unittest` module — there is no pytest dependency and
no test runner config. The suite stubs `coremltools` and `macd_utils` in
`sys.modules`, so it needs no Postgres, no `.mlpackage`, and no NPU; it runs
anywhere in well under a second.

```bash
# From src/ — whole suite
python -m unittest discover tests

# One module / one test
python -m unittest tests.test_neural_forecast_cache
python -m unittest tests.test_neural_forecast_features.MultiFeatureInputTest.test_signal_line_feature_reads_the_signal_line_column

# From the repo root (the test files add src/ to sys.path themselves)
python -m unittest discover src/tests
```

Current coverage is the neural forecaster's inference plumbing only — the parts
where a silent wrong answer is indistinguishable from a right one:

| Module | Covers |
|--------|--------|
| `test_neural_forecast_cache.py` | `CoreMLForecaster._load_model`: a cache-hit instance must be identical to a freshly parsed one; a failed parse must leave the forecaster unavailable rather than predicting with default normalization stats |
| `test_neural_forecast_features.py` | The multi-feature matrix `forecast_macd` builds: each feature reads the column it names, `delta` tracks the primary signal, columns stay length-aligned |

When adding a feature to the `--extra-features` path, note it is built in three
places (`scripts/train_forecast_model.py`, `scripts/evaluate_forecast_model.py`,
`models/neural_forecast.py`) — `test_neural_forecast_features.py` pins only the
inference one.

## Architecture Notes

- The project was recently refactored from a monolithic `api.py` into a layered architecture (routers -> services -> utils). The refactoring is mostly complete on the `refactor1` branch.
- `db_utils.py` uses `psycopg2` SimpleConnectionPool (min=1, max=15).
- Neural forecasting auto-selects engine: Core ML (NPU) when available, ARIMA (CPU) as fallback.
- Supported neural architectures: `bidirectional_gru` (recommended), `stacked_gru`, `gru`, `standard_lstm`, `stacked_lstm`.
- Chart pattern detection uses YOLO via Core ML. The model is at `chart_scan/model.mlpackage`.

## Database

PostgreSQL with main table `stock_cache` (symbol, date, OHLC, volume, EMA/MA indicators, MACD, signal_line, forecast flags, chart_patterns JSONB). Supporting tables: `watchlists`, `watchlist_symbols`, `symbol_picks`, `forecast_util`, `company_names`.

Notes Dashboard tables (independent of the market-data tables): `note_watchlists`, `note_watchlist_symbols`, `symbol_notes`, `note_images` (image bytes as BYTEA plus a WEBP thumbnail).

Default connection: `postgres://postgres:postgres@localhost:5432/postgres`

## Code Conventions

- Backend scripts are run from the `src/` directory.
- Routers use FastAPI's `APIRouter` with prefix and tags.
- Pydantic models for request/response validation in `src/models/requests.py` and `src/models/responses.py`.
- Float sanitization (NaN/Infinity -> None) via `utils/sanitization.py` before JSON responses.
- Custom exceptions in `utils/exceptions.py`, handled by centralized middleware.
- macOS-specific: uses matplotlib Agg backend (headless), Core ML for NPU inference.

## Bullish MACD Signal Columns & History Lines

In the **"Bullish MACD Signal for [Watchlist]"** view and its historical trend graph (**"Column counts – last X months"**), stock symbols are classified into 5 columns and tracked over time. Each line / column represents:

1. **MACD gets positive** (Green `#27ae60`) – history key `macd_gets_positive`:
   - **Meaning**: Stocks where MACD has recently crossed above zero from negative territory AND MACD is currently above the signal line.
   - **Calculation**: `macd_just_became_positive == True` (crossed 0 from below in recent days) **AND** `bullish_macd_above_signal == True` (`macd > signal_line`).

2. **Recently crossed** (Purple `#8e44ad`) – history key `recently_crossed`:
   - **Meaning**: Stocks that had a bullish MACD crossover above the signal line within the lookback window (past 15 trading days) AND MACD is already positive.
   - **Calculation**: `recent_crossover == True` **AND** `macd_is_positive == True` (`macd > 0`).

3. **MACD crossed** (Orange `#e67e22`) – history key `macd_crossed`:
   - **Meaning**: Stocks where MACD is currently above the signal line, regardless of when the crossover happened. This is the superset that catches symbols whose crossover predates the `recent_crossover` lookback window.
   - **Calculation**: `bullish_macd_above_signal == True` (`macd > signal_line`).

4. **MACD under signal positive** (Blue `#2980b9`) – history key `under_signal_positive`:
   - **Meaning**: Stocks where MACD is currently below the signal line but remains positive.
   - **Calculation**: `bullish_macd_above_signal == False` (`macd <= signal_line`) **AND** `macd_is_positive == True` (`macd > 0`).

5. **MACD under signal negative** (Grey `#7f8c8d`) – history key `under_signal_negative`:
   - **Meaning**: Stocks where MACD is currently below the signal line and also in negative territory.
   - **Calculation**: `bullish_macd_above_signal == False` (`macd <= signal_line`) **AND** `macd_is_positive == False` (`macd <= 0`).

*Note*: A symbol may appear in multiple columns if it satisfies multiple conditions — in particular, every symbol in columns 1 and 2 that is above the signal line also appears in column 3. Column 3 is exactly the complement of columns 4 and 5 combined, so `macd_crossed + under_signal_positive + under_signal_negative` equals the number of evaluated symbols. Historical daily totals are stored in `symbol_picks` and backfilled from `stock_cache` for 1, 3, 6, and 12-month intervals.
