# Tick Scanner

A stock market technical analysis platform that identifies bullish trading signals using MACD indicators, moving average crossovers, ARIMA forecasting, and neural network-based pattern detection.

## Features

- **MACD Technical Indicators**: Detects bullish crossover signals
- **Moving Average Crossovers**: Identifies MA20/MA50 trend changes
- **ARIMA Time-Series Forecasting**: Predicts future MACD and Signal Line values
- **Neural Network Forecasting**: LSTM-based predictions running on Apple Neural Engine (NPU)
- **Chart Pattern Detection**: YOLO-based ML to detect visual patterns (Head & Shoulders, etc.)
- **Watchlist Management**: Track and analyze groups of stocks

## Quick Start

### Prerequisites

- Python 3.10+
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

---

## Neural Forecasting System

The platform includes an LSTM-based neural forecasting system that can run on Apple's Neural Engine (NPU) for fast inference.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Neural Forecasting Pipeline                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Training (GPU)              Inference (NPU)                    │
│  ┌──────────────┐           ┌──────────────┐                   │
│  │   PyTorch    │           │   Core ML    │                   │
│  │    LSTM      │  ──────►  │   Model      │                   │
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

### Model Types

| Model | Input | Output | Filename |
|-------|-------|--------|----------|
| MACD Forecaster | 30 days of MACD values | 5-day MACD forecast | `macd_forecaster.mlpackage` |
| Signal Line Forecaster | 30 days of Signal Line values | 5-day Signal Line forecast | `signal_line_forecaster.mlpackage` |

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

# Train on default watchlist (sp500)
python scripts/train_forecast_model.py --epochs 100

# Train on specific watchlist
python scripts/train_forecast_model.py --watchlist my_watchlist --epochs 100

# Train on specific symbols
python scripts/train_forecast_model.py --symbols "AAPL,MSFT,GOOG,AMZN" --epochs 100
```

#### Train Signal Line Forecaster

```bash
cd src

# Train Signal Line model instead of MACD
python scripts/train_forecast_model.py --signal-type signal_line --epochs 100
```

#### Training Options

| Option | Default | Description |
|--------|---------|-------------|
| `--symbols` | None | Comma-separated list of symbols |
| `--watchlist` | sp500 | Watchlist name to use for training data |
| `--days` | 365 | Days of historical data to use |
| `--signal-type` | macd | Type of signal: `macd` or `signal_line` |
| `--epochs` | 100 | Number of training epochs |
| `--seq-length` | 30 | Input sequence length (days) |
| `--forecast-horizon` | 5 | Number of days to forecast |
| `--hidden-size` | 64 | LSTM hidden layer size |
| `--batch-size` | 32 | Training batch size |
| `--learning-rate` | 0.001 | Optimizer learning rate |
| `--test-split` | 0.15 | Fraction of data for testing |
| `--output-dir` | models/ | Output directory for trained models |
| `--skip-coreml` | False | Skip Core ML export |

#### Training Output

```
============================================================
LSTM MACD Forecaster Training
============================================================
Signal type: MACD
Output directory: /path/to/models
Sequence length: 30
Forecast horizon: 5
...

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

Model saved to models/macd_forecaster.pt
Core ML model saved to models/macd_forecaster.mlpackage
```

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

The forecast service automatically selects the best available engine:

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
│   ├── services/              # Business logic services
│   │   ├── forecast_service.py
│   │   └── chart_service.py
│   ├── models/                # Data models & neural networks
│   │   ├── lstm_forecaster.py # LSTM training & export
│   │   └── neural_forecast.py # NPU inference
│   └── scripts/
│       └── train_forecast_model.py
├── chart_scan/
│   ├── detector_neural.py     # YOLO pattern detection
│   └── model.mlpackage/       # YOLO Core ML model
├── frontend/                   # React frontend
├── models/                     # Trained forecast models
├── watchlists/                 # Watchlist files
├── requirements.txt
└── ARCHITECTURE.md            # Full architecture docs
```

---

## API Endpoints

### Forecasting

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/forecast/engine/status` | Get forecasting engine status |
| POST | `/forecast/macd/arima_positive` | Bulk MACD forecast |
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

| Metric | ARIMA | Neural (NPU) |
|--------|-------|--------------|
| Single Symbol | ~200ms | ~5ms |
| 500 Symbols | ~100s (parallel) | ~2.5s |
| Accuracy (Directional) | ~65% | ~68-72% |

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
