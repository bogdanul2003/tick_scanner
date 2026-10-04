# Tick Scanner

A stock market technical analysis platform that identifies bullish trading signals using MACD indicators, moving average crossovers, ARIMA forecasting, and neural network-based pattern detection.

## Features

- **MACD Technical Indicators**: Detects bullish crossover signals
- **Moving Average Crossovers**: Identifies MA20/MA50 trend changes
- **Neural Network Forecasting**: The bullish MACD forecast is served by a seed ensemble of GRU models (three versions of one config, forecasts averaged) running through Core ML; pick the ensemble in the UI, run it, or retrain it from there
- **ARIMA Time-Series Forecasting**: Still used for the MA20/MA50 forecast, and selectable as a legacy engine for the MACD forecast
- **Forecast History**: Every forecast run is stored, so accuracy can be measured against what actually happened
- **Chart Pattern Detection**: YOLO-based ML to detect visual patterns (Head & Shoulders, etc.)
- **Watchlist Management**: Track and analyze groups of stocks

## Quick Start

### Prerequisites

- Python 3.10 to 3.13 (Core ML Tools has no Python 3.14 build yet: apple/coremltools#2646)
- PostgreSQL 15+
- Node.js 18+ (for frontend)
- macOS with Apple Silicon (for NPU inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/tick_scanner.git
cd tick_scanner

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd frontend && npm install && cd ..

# Start PostgreSQL (Docker)
cd src && docker-compose up -d && cd ..
```

### Running the Application

```bash
# Start the backend API
cd src && python api.py

# Start the frontend (in a separate terminal)
cd frontend && npm run dev
```

- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Frontend**: http://localhost:5173

### Running the Tests

The suite lives in `src/tests/` and uses Python's built-in `unittest` — nothing to
install beyond `requirements.txt`. Core ML and the database layer are stubbed, so
the tests need no PostgreSQL, no trained `.mlpackage`, and no Apple Neural Engine.

```bash
# Whole suite (from src/)
cd src && python -m unittest discover tests

# Or from the repo root
python -m unittest discover src/tests

# A single module, with per-test names
cd src && python -m unittest tests.test_neural_forecast_cache -v
```

Expected output:

```
----------------------------------------------------------------------
Ran 154 tests in 0.659s

OK
```

| Test module | What it pins down |
|-------------|-------------------|
| `test_neural_forecast_cache.py` | The `CoreMLForecaster` metadata cache: an instance built from the cache must be identical to one that parsed the `.mlpackage`, and a model that fails to parse must report itself unavailable instead of forecasting with default normalization stats |
| `test_neural_forecast_features.py` | The multi-feature input matrix (`--extra-features` models): every feature reads the DB column it names, `delta` tracks the primary signal, and all columns stay length-aligned |
| `test_ensemble_service.py` | Seed ensembles: the ensemble is the mean of its members and refuses mismatched ones, the registry and retrain state resolve to the right versions, and the ensemble forecast returns the ARIMA endpoint's shape and records the run |
| `test_ensemble_training.py` | The retrain job: served versions switch only when every seed trained, one job runs at a time, and an existing model version is never overwritten |

The table lists the modules most relevant to serving forecasts; `src/tests/` has more
(training, splits, seeding, model versioning).

To add a test, drop a `test_*.py` file into `src/tests/`; discovery picks it up with
no registration step. The existing modules show the two stubbing patterns —
`mock.patch.dict(sys.modules, ...)` for `coremltools` / `macd_utils`, and a small
recording double in place of the forecaster.

---

## Neural Forecasting System

The platform includes a GRU/LSTM-based neural forecasting system that runs through Core ML on Apple Silicon.

### Bullish Forecast: ensembles, retraining and history

**Show Bullish Forecast** in the MACD dashboard opens a panel with a model dropdown and two buttons.

- **Model dropdown** — the selectable seed ensembles, each showing the versions it serves, its last measured MAE and directional accuracy, and when it was trained. The default is `gru_v1_residual_macdonly_warm5` (versions 1, 2, 3). "ARIMA (legacy)" is the last option.
- **Run forecast** — forecasts every symbol in the watchlist with the chosen model and lists the results most-bullish first. The whole sp500 watchlist takes about 4 seconds with an ensemble.
- **Retrain** — after a confirmation, retrains the chosen ensemble in the background: refreshes the price data, trains seeds 42, 7 and 123 in parallel on the CPU (10-15 minutes), evaluates the result and switches to the new versions. The current versions keep serving until every seed has trained, and old versions are never deleted. Retraining on unchanged data reproduces the same models, so it is only useful once new trading days are in the database.

An ensemble is three versions (seeds) of one training config whose forecasts are averaged. Which ensembles exist is defined in `src/configs/ensembles.json`; what a retrain changed is recorded in `models/ensemble_state.json`. `models/` is not in git, so on a fresh checkout every ensemble shows as "not trained" until it is retrained (or trained from the command line, below).

**Forecast history.** Every run — ensemble or ARIMA — is stored in the `forecast_predictions` table: the model, the symbol, the market date it was forecast from, the five predicted values, and which model versions (and training date) produced them. Running the same model again from the same market date replaces the earlier run. Once the forecast days have happened, the scoring query in `docs/ENSEMBLE_PRODUCTION_PLAN.md` gives MAE and directional accuracy per model and forecast day.

**How good is it?** Measured 2026-10-04 on the sp500 watchlist, the default ensemble's forecast error is about 21% lower than naive drift's on average and about 22% lower than ARIMA's on the typical symbol, but its direction calls (51%) are no better than drift's (58%). See `src/models/MODEL_CARD.md` §11 before relying on the "will become positive" flag.

The rest of this section describes training and evaluating single models from the command line. `src/models/MODEL_CARD.md` is the up-to-date reference for configs, commands and measured results — in particular, train on the CPU (`--device cpu`, about 3.6x faster than the GPU at this model size) and compare configs on at least three seeds.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Forecasting Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Training (GPU)              Inference (NPU)                    │
│  ┌──────────────┐           ┌──────────────┐                   │
│  │   PyTorch    │           │   Core ML    │                   │
│  │  LSTM/GRU    │  ──────►  │   Model      │                   │
│  │   Training   │  export   │  (.mlpackage)│                   │
│  └──────────────┘           └──────────────┘                   │
│        │                           │                            │
│        ▼                           ▼                            │
│  ┌──────────────┐           ┌──────────────┐                   │
│  │  Apple MPS   │           │ Apple Neural │                   │
│  │    (GPU)     │           │   Engine     │                   │
│  └──────────────┘           └──────────────┘                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Model Architectures

Based on research (NIFTY50 MACD forecasting study), multiple architectures are available:

| Architecture | Description | Research RMSE | Best For |
|--------------|-------------|---------------|----------|
| `bidirectional_gru` | Bidirectional GRU | **19.57** | Recommended - best accuracy |
| `stacked_gru` | 2-layer GRU | 24.40 | Good balance of speed/accuracy |
| `gru` | Standard single-layer GRU | 29.01 | Fastest training |
| `standard_lstm` | Single-layer LSTM | 32.89 | Baseline |
| `stacked_lstm` | 2-layer LSTM (default) | 115.46* | Legacy default |

*Note: Stacked LSTM underperformed in research but may work better with larger datasets.

### Model Types

| Signal Type | Architecture | Input | Output | Filename |
|-------------|--------------|-------|--------|----------|
| MACD | bidirectional_gru | 30 days MACD | 5-day forecast | `macd_bidirectional_gru_forecaster.mlpackage` |
| MACD | stacked_lstm | 30 days MACD | 5-day forecast | `macd_stacked_lstm_forecaster.mlpackage` |
| Signal Line | bidirectional_gru | 30 days Signal | 5-day forecast | `signal_line_bidirectional_gru_forecaster.mlpackage` |

### Training the Models

#### Prerequisites for Training

```bash
# Install PyTorch (required for training only)
pip install torch

# Verify Core ML Tools is installed
pip install coremltools
```

#### Train MACD Forecaster

```bash
cd src

# Train with Bidirectional GRU (recommended - best accuracy)
python scripts/train_forecast_model.py --architecture bidirectional_gru --epochs 100

# Train with Stacked GRU (good balance)
python scripts/train_forecast_model.py --architecture stacked_gru --epochs 100

# Train with default Stacked LSTM
python scripts/train_forecast_model.py --epochs 100

# Train on specific watchlist
python scripts/train_forecast_model.py --architecture bidirectional_gru --watchlist my_watchlist --epochs 100

# Train on specific symbols
python scripts/train_forecast_model.py --architecture bidirectional_gru --symbols "AAPL,MSFT,GOOG,AMZN" --epochs 100
```

#### Train Signal Line Forecaster

```bash
cd src

# Train Signal Line model with Bidirectional GRU
python scripts/train_forecast_model.py --signal-type signal_line --architecture bidirectional_gru --epochs 100
```

#### Compare Architectures

```bash
cd src

# Train all architectures to compare performance
python scripts/train_forecast_model.py --architecture bidirectional_gru --epochs 100
python scripts/train_forecast_model.py --architecture stacked_gru --epochs 100
python scripts/train_forecast_model.py --architecture stacked_lstm --epochs 100
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--architecture` | stacked_lstm | Model architecture: `bidirectional_gru`, `stacked_gru`, `gru`, `standard_lstm`, `stacked_lstm` |
| `--symbols` | None | Comma-separated list of symbols |
| `--watchlist` | sp500 | Watchlist name to use for training data |
| `--days` | 365 | Days of historical data to use |
| `--signal-type` | macd | Type of signal: `macd` or `signal_line` |
| `--epochs` | 100 | Number of training epochs |
| `--seq-length` | 30 | Input sequence length (days) |
| `--forecast-horizon` | 5 | Number of days to forecast |
| `--hidden-size` | 64 | Hidden layer size |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 0.001 | Optimizer learning rate |
| `--test-split` | 0.15 | Fraction of data for testing |
| `--output-dir` | models/ | Output directory for trained models |
| `--skip-coreml` | False | Skip Core ML export |

#### Training Output

```
============================================================
Bidirectional Gru MACD Forecaster Training
============================================================
Architecture: bidirectional_gru
Signal type: MACD
Output directory: /path/to/models
Sequence length: 30
Forecast horizon: 5
...

Using device: mps
Architecture: bidirectional_gru

Epoch 10/100 - Train Loss: 0.045678, Val Loss: 0.052341
Epoch 20/100 - Train Loss: 0.032456, Val Loss: 0.041234
...

==================================================
Test Evaluation Results
==================================================
Test samples: 1500
MAE (Mean Absolute Error): 0.023456
RMSE (Root Mean Square Error): 0.045678
Directional Accuracy: 68.50%
Positive Prediction Accuracy: 72.30%
==================================================

Model saved to models/macd_bidirectional_gru_forecaster.pt
Core ML model saved to models/macd_bidirectional_gru_forecaster.mlpackage
```

### Evaluating Models

After training, you can evaluate model performance by comparing predictions to actual values stored in the database.

#### Basic Evaluation

```bash
cd src

# Evaluate bidirectional GRU model on MSFT (defaults)
python scripts/evaluate_forecast_model.py

# Evaluate on specific symbol
python scripts/evaluate_forecast_model.py --symbol AAPL

# Evaluate stacked LSTM for comparison
python scripts/evaluate_forecast_model.py --symbol AAPL --architecture stacked_lstm

# Evaluate signal line model
python scripts/evaluate_forecast_model.py --symbol NVDA --signal-type signal_line

# More samples for better statistics
python scripts/evaluate_forecast_model.py --symbol GOOGL --samples 30
```

#### List Available Models

```bash
python scripts/evaluate_forecast_model.py --list-models
```

#### Evaluation Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbol` | MSFT | Stock symbol to evaluate |
| `--signal-type` | macd | Type of signal: `macd` or `signal_line` |
| `--architecture` | bidirectional_gru | Model architecture to use |
| `--input-days` | 30 | Number of input context days |
| `--samples` | 10 | Number of prediction samples to evaluate |
| `--forecast-horizon` | 5 | Number of days to forecast |
| `--list-models` | - | List available trained models |

#### Evaluation Output

```
======================================================================
Running 10 predictions on AAPL MACD
Model: bidirectional_gru (coreml)
Input sequence: 30 days, Forecast: 5 days
======================================================================

Sample 1: Input ends 2026-03-10
  Last input value: -0.5234
  Day 1 (2026-03-11): Pred= -0.4123, Actual= -0.4567, Error=+0.0444 (9.7%) ✓
  Day 2 (2026-03-12): Pred= -0.2891, Actual= -0.3012, Error=+0.0121 (4.0%) ✓
  Day 3 (2026-03-13): Pred= -0.1234, Actual= -0.0987, Error=-0.0247 (25.0%) ✓
  Day 4 (2026-03-14): Pred=  0.0567, Actual=  0.1234, Error=-0.0667 (54.1%) ✓
  Day 5 (2026-03-17): Pred=  0.2345, Actual=  0.2890, Error=-0.0545 (18.9%) ✓

...

======================================================================
EVALUATION SUMMARY
======================================================================
Symbol:              AAPL
Signal Type:         MACD
Architecture:        bidirectional_gru
Inference Engine:    COREML
Samples Evaluated:   10
Total Predictions:   50

MAE  (Mean Absolute Error):    0.234567
RMSE (Root Mean Square Error): 0.345678
MAPE (Mean Absolute % Error):  12.34%
Directional Accuracy:          68.50%
======================================================================
```

The ✓ and ✗ symbols indicate whether the model correctly predicted the direction of change relative to the last input value.

### Using the Neural Forecaster

#### Check Engine Status

```bash
curl http://localhost:8000/forecast/engine/status
```

Response:
```json
{
  "arima_available": true,
  "neural_available": true,
  "current_engine": "neural",
  "max_workers": 10,
  "npu_info": {
    "npu_available": true,
    "coreml_available": true,
    "apple_silicon": true,
    "device": "mps"
  }
}
```

#### Automatic Engine Selection

The dashboard does not use this path: it calls the ensemble endpoints described above.
`forecast_service` remains for programmatic use and selects an engine automatically:

1. **Neural (NPU)**: Used when Core ML model exists and NPU is available
2. **ARIMA (CPU)**: Fallback when neural model is unavailable

```python
# The service automatically uses NPU when available
from services.forecast_service import forecast_service

# Single symbol forecast
result = forecast_service.run_macd_forecast("AAPL", days_past=30, forecast_days=5)

# Bulk forecast (uses NPU for batch inference)
results = forecast_service.bulk_macd_forecast(["AAPL", "MSFT", "GOOG"])
```

### Model Files

After training, models are saved to the `models/` directory:

```
models/
├── macd_forecaster.pt           # PyTorch checkpoint
├── macd_forecaster.mlpackage/   # Core ML model (NPU inference)
├── signal_line_forecaster.pt
└── signal_line_forecaster.mlpackage/
```

---

## Project Structure

```
tick_scanner/
├── src/
│   ├── api.py                 # FastAPI application
│   ├── macd_utils.py          # MACD calculations
│   ├── db_utils.py            # Database utilities
│   ├── forecast_utils.py      # ARIMA forecasting
│   ├── charts_generator.py    # Chart generation
│   ├── routers/               # API route modules
│   ├── configs/               # Training configs + ensembles.json (the ensemble registry)
│   ├── services/              # Business logic services
│   │   ├── forecast_service.py
│   │   ├── ensemble_service.py   # Ensemble registry, bulk inference, forecast history
│   │   ├── ensemble_training.py  # Background retrain job
│   │   └── chart_service.py
│   ├── models/                # Data models & neural networks
│   │   ├── lstm_forecaster.py # LSTM training & export
│   │   └── neural_forecast.py # NPU inference
│   ├── scripts/
│   │   ├── train_forecast_model.py      # Train LSTM/GRU models
│   │   ├── evaluate_forecast_model.py   # Evaluate model accuracy
│   │   ├── evaluate_ensemble.py         # Evaluate the average of several model versions
│   │   └── persistence_baseline.py      # Naive flat/drift baselines to compare against
│   └── tests/                 # unittest suite (DB and Core ML stubbed)
├── chart_scan/
│   ├── detector_neural.py     # YOLO pattern detection
│   └── model.mlpackage/       # YOLO Core ML model
├── frontend/                   # React frontend
├── models/                     # Trained forecast models
├── watchlists/                 # Watchlist files
├── requirements.txt
└── docs/
    ├── ARCHITECTURE.md        # Full architecture docs
    ├── FORECAST_*.md          # Forecaster fix plan, improvement proposals and results log
    └── ENSEMBLE_PRODUCTION_PLAN.md  # Ensemble serving, retraining and forecast history
```

---

## API Endpoints

### Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/forecast/engine/status` | Get forecasting engine status |
| GET | `/forecast/ensembles` | Selectable seed ensembles, the default, availability, metrics and training date |
| POST | `/forecast/macd/ensemble` | Bulk MACD forecast with a seed ensemble (`{symbols, ensemble_id}`) |
| POST | `/forecast/ensembles/{id}/retrain` | Retrain an ensemble in the background (one job at a time) |
| GET | `/forecast/ensembles/retrain/status` | Progress of the running or most recent retrain |
| POST | `/forecast/macd/arima_positive` | Bulk MACD forecast with ARIMA (legacy engine) |
| POST | `/forecast/ma/arima_above_50` | Bulk MA20/MA50 forecast |
| POST | `/forecast/combined/{watchlist}` | Combined forecast |

### Charts

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/charts/watchlist/{name}/generate` | Generate charts with pattern detection |
| POST | `/charts/watchlist/{name}/bulk_generate` | Bulk generate for multiple days |
| GET | `/charts/watchlist/{name}/available_dates` | Get available dates |

### MACD

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/macd/{symbol}` | Get MACD for symbol |
| POST | `/macd/bulk` | Bulk MACD lookup |
| GET | `/macd/{symbol}/history` | Get MACD history |

See full API documentation at http://localhost:8000/docs

---

## Performance

### Neural Forecasting vs ARIMA

| Metric | ARIMA | Neural ensemble (3 seeds) | Naive drift |
|--------|-------|---------------------------|-------------|
| 500 symbols | ~100s (parallel) | ~4s (measured, 501 symbols) | — |
| Directional accuracy, 5-day | 46.4% | 51.3% | 57.8% |
| Typical forecast error (median MAE) | 0.577 | 0.452 | — |
| Mean MAE | 1.682 | 0.901 | 1.147 |

Accuracy measured 2026-10-04 on the sp500 watchlist, 20 samples per symbol
(`src/models/MODEL_CARD.md` §11). The ensemble is the most accurate on the size of the
move; simply extrapolating the last day's change (drift) is the most accurate on its
direction.

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | 4 cores | 8+ cores |
| RAM | 8GB | 16GB+ |
| GPU | - | Apple Silicon (MPS) |
| NPU | - | Apple Neural Engine |

---

## License

MIT License
