"""
Seed-ensemble MACD forecasting: the ensemble registry and bulk inference.

An ensemble is several versions (seeds) of one training config whose forecasts are
averaged. `configs/ensembles.json` says which ensembles exist and which versions each
one ships with; `models/ensemble_state.json` records what a retrain changed (versions
now served, when they were trained, their measured metrics) and overrides the
registry's versions. See docs/ENSEMBLE_PRODUCTION_PLAN.md.
"""
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIGS_DIR = os.path.join(SRC_DIR, "configs")
REGISTRY_PATH = os.path.join(CONFIGS_DIR, "ensembles.json")
MODELS_DIR = os.path.join(os.path.dirname(SRC_DIR), "models")
STATE_PATH = os.path.join(MODELS_DIR, "ensemble_state.json")

DEFAULT_SEEDS = [42, 7, 123]


class EnsembleNotFoundError(ValueError):
    """The requested ensemble id is not in the registry."""


class EnsembleUnavailableError(RuntimeError):
    """The ensemble is registered but its model files are missing or unloadable."""


def load_registry(registry_path: str = None, configs_dir: str = None) -> Dict[str, Any]:
    """
    Load and validate the ensemble registry.

    Each returned entry carries the fields read from its training config
    (`model_name`, `signal_type`, `watchlist`) alongside the registry's own.
    """
    registry_path = registry_path or REGISTRY_PATH
    configs_dir = configs_dir or CONFIGS_DIR
    with open(registry_path) as f:
        raw = json.load(f)

    entries = raw.get("ensembles") or []
    if not entries:
        raise ValueError("Ensemble registry lists no ensembles")

    seen = set()
    ensembles = []
    for entry in entries:
        ensemble_id = entry.get("id")
        if not ensemble_id:
            raise ValueError("Ensemble registry entry is missing 'id'")
        if ensemble_id in seen:
            raise ValueError(f"Duplicate ensemble id in registry: {ensemble_id}")
        seen.add(ensemble_id)

        versions = entry.get("versions") or []
        if not versions or not all(isinstance(v, int) and v > 0 for v in versions):
            raise ValueError(f"Ensemble '{ensemble_id}' needs a non-empty list of positive integer versions")

        config_path = os.path.join(configs_dir, entry.get("config", ""))
        if not os.path.isfile(config_path):
            raise ValueError(f"Ensemble '{ensemble_id}' points at a missing config: {entry.get('config')}")
        with open(config_path) as f:
            config = json.load(f)
        if not config.get("model_name"):
            raise ValueError(f"Config for ensemble '{ensemble_id}' has no model_name")
        signal_type = config.get("signal_type") or "macd"
        if signal_type != "macd":
            raise ValueError(f"Ensemble '{ensemble_id}' is a {signal_type} model; only MACD ensembles are served")

        ensembles.append({
            "id": ensemble_id,
            "label": entry.get("label") or ensemble_id,
            "config": entry["config"],
            "config_path": config_path,
            "model_name": config["model_name"],
            "signal_type": signal_type,
            "watchlist": config.get("watchlist"),
            "versions": list(versions),
            "metrics": entry.get("metrics"),
        })

    default_id = raw.get("default") or ensembles[0]["id"]
    if default_id not in seen:
        raise ValueError(f"Registry default '{default_id}' is not a listed ensemble")

    return {"default": default_id, "seeds": raw.get("seeds") or DEFAULT_SEEDS, "ensembles": ensembles}


def load_state(state_path: str = None) -> Dict[str, Any]:
    """Per-ensemble retrain state; empty when nothing has been retrained here."""
    state_path = state_path or STATE_PATH
    if not os.path.isfile(state_path):
        return {}
    try:
        with open(state_path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.error(f"Ignoring unreadable ensemble state at {state_path}: {e}")
        return {}


def save_state(state: Dict[str, Any], state_path: str = None) -> None:
    """Write the retrain state atomically, so a reader never sees a half-written file."""
    state_path = state_path or STATE_PATH
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    tmp_path = f"{state_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, state_path)


def _member_path(model_name: str, version: int, models_dir: str) -> str:
    return os.path.join(models_dir, f"{model_name}_{version}.mlpackage")


def _local_iso(moment: datetime) -> str:
    """ISO timestamp with the local UTC offset, so it means the same instant in the DB."""
    return moment.astimezone().isoformat(timespec="seconds")


def _files_trained_at(model_name: str, versions: List[int], models_dir: str) -> Optional[str]:
    """
    When the newest member was trained, read off its files; None if none exist.

    Model files carry no training date of their own, so for an ensemble that was not
    retrained through the app a file time is the only record of it. The .pt
    checkpoint is the one to read: loading a Core ML model rewrites the
    .mlpackage's Manifest.json, so the package's own time is "last served", not
    "trained". Its Data directory is left alone and stands in when the .pt is gone.
    """
    times = []
    for v in versions:
        package = _member_path(model_name, v, models_dir)
        for candidate in (os.path.join(models_dir, f"{model_name}_{v}.pt"), os.path.join(package, "Data")):
            if os.path.exists(candidate):
                times.append(os.path.getmtime(candidate))
                break
    return _local_iso(datetime.fromtimestamp(max(times))) if times else None


def resolve_ensemble(ensemble_id: Optional[str] = None, registry: Dict[str, Any] = None,
                     state: Dict[str, Any] = None, models_dir: str = None) -> Dict[str, Any]:
    """
    One registry entry with retrain state applied and availability worked out.

    `ensemble_id` None means the registry default.
    """
    registry = registry or load_registry()
    state = load_state() if state is None else state
    models_dir = models_dir or MODELS_DIR
    ensemble_id = ensemble_id or registry["default"]

    entry = next((e for e in registry["ensembles"] if e["id"] == ensemble_id), None)
    if entry is None:
        raise EnsembleNotFoundError(f"Unknown ensemble: {ensemble_id}")

    resolved = dict(entry)
    retrained = state.get(ensemble_id) or {}
    if retrained.get("versions"):
        resolved["versions"] = list(retrained["versions"])
        resolved["metrics"] = retrained.get("metrics")
    recorded = retrained.get("trained_at") if retrained.get("versions") else None
    resolved["trained_at"] = (
        _local_iso(datetime.fromisoformat(recorded)) if recorded
        else _files_trained_at(resolved["model_name"], resolved["versions"], models_dir)
    )
    resolved["missing_versions"] = [
        v for v in resolved["versions"]
        if not os.path.exists(_member_path(resolved["model_name"], v, models_dir))
    ]
    resolved["available"] = not resolved["missing_versions"]
    resolved["is_default"] = ensemble_id == registry["default"]
    return resolved


def describe_ensembles() -> Dict[str, Any]:
    """What the UI needs to build its dropdown."""
    registry = load_registry()
    state = load_state()
    public_keys = ("id", "label", "model_name", "watchlist", "versions", "metrics",
                   "trained_at", "available", "missing_versions", "is_default")
    return {
        "default": registry["default"],
        "ensembles": [
            {k: resolved[k] for k in public_keys}
            for resolved in (resolve_ensemble(e["id"], registry, state) for e in registry["ensembles"])
        ],
    }


def load_forecaster(ensemble: Dict[str, Any], models_dir: str = None):
    """Build the EnsembleForecaster for a resolved registry entry."""
    from models.neural_forecast import CoreMLForecaster, EnsembleForecaster

    if not ensemble["available"]:
        raise EnsembleUnavailableError(
            f"Ensemble '{ensemble['id']}' is not trained: missing "
            f"{ensemble['model_name']} version(s) {ensemble['missing_versions']}"
        )

    models_dir = models_dir or MODELS_DIR
    members = [
        CoreMLForecaster(_member_path(ensemble["model_name"], v, models_dir), ensemble["signal_type"])
        for v in ensemble["versions"]
    ]
    unloaded = [v for v, m in zip(ensemble["versions"], members) if not m.is_available]
    if unloaded:
        raise EnsembleUnavailableError(
            f"Ensemble '{ensemble['id']}' could not load {ensemble['model_name']} version(s) {unloaded}"
        )
    return EnsembleForecaster(members)


def _failed(message: str) -> Dict[str, Any]:
    """A per-symbol failure, shaped like the ARIMA path's."""
    return {"will_become_positive": False, "forecasted_macd": [], "details": {"error": message}}


def forecast_symbols(symbols: List[str], ensemble_id: Optional[str] = None,
                     days_past: int = 100, forecast_days: int = 5) -> Dict[str, Dict[str, Any]]:
    """
    Ensemble MACD forecast for each symbol, in the shape arima_macd_positive_forecast
    returns, so callers and the UI treat both engines alike.

    Like the ARIMA path, each symbol's will_become_positive is written to stock_cache
    for the latest market date: the combined-forecast endpoint reads it from there.
    The run itself is recorded in forecast_predictions (see record_forecast_run).
    """
    from db_utils import cache_macd_positive_forecast
    from forecast_utils import macd_will_become_positive, next_weekday_dates, sanitize_data
    from macd_utils import get_latest_market_date, get_macd_for_range_bulk
    from models.neural_forecast import NeuralForecastService

    ensemble = resolve_ensemble(ensemble_id)
    forecaster = load_forecaster(ensemble)
    service = NeuralForecastService(
        forecaster=forecaster, fallback_to_arima=False, signal_type=ensemble["signal_type"]
    )

    symbols = [s.upper() for s in symbols]
    end_date = get_latest_market_date()
    start_date = end_date - timedelta(days=service.calendar_days_needed(days_past))
    rows_by_symbol = get_macd_for_range_bulk(symbols, start_date, end_date)

    horizon = min(forecast_days, forecaster.forecast_horizon)
    results = {}
    for symbol in symbols:
        raw = service.forecast_macd(symbol, days_past, forecast_days, macd_data=rows_by_symbol.get(symbol, []))
        details = raw.get("details") or {}
        forecast = raw.get("forecasted_macd")
        if details.get("error") or not isinstance(forecast, dict) or not forecast:
            results[symbol] = _failed(details.get("error") or "No forecast produced")
            continue

        values = list(forecast.values())[:horizon]
        dates = next_weekday_dates(end_date, len(values))
        last_macd = details["last_macd"]
        will_become_positive = macd_will_become_positive(last_macd, values)
        results[symbol] = sanitize_data({
            "will_become_positive": will_become_positive,
            "forecasted_macd": dict(zip(dates, values)),
            "forecasted_dates": dates,
            "details": {
                "last_macd": last_macd,
                "inference_engine": "Core ML ensemble",
                "ensemble": ensemble["id"],
                "model_name": ensemble["model_name"],
                "versions": ensemble["versions"],
            },
        })
        try:
            cache_macd_positive_forecast(symbol, end_date, will_become_positive)
        except Exception as e:
            logger.error(f"Failed to cache will_become_positive for {symbol}: {e}")

    record_forecast_run(ensemble["id"], end_date, results, model_name=ensemble["model_name"],
                        model_versions=ensemble["versions"], model_trained_at=ensemble.get("trained_at"))
    return results


def record_forecast_run(model: str, as_of_date, results: Dict[str, Dict[str, Any]], model_name: str = None,
                        model_versions: List[int] = None, model_trained_at: str = None) -> int:
    """
    Store what a forecast run predicted, so its accuracy can be checked once the
    forecast days have happened. `model` is an ensemble id or 'arima'. Re-running the
    same model from the same market date replaces the earlier run.

    Never raises: failing to record a run must not fail the forecast it describes.
    Returns the number of rows written.
    """
    try:
        from db_utils import save_forecast_predictions
        from forecast_utils import forecast_prediction_rows
        return save_forecast_predictions(
            model, as_of_date, forecast_prediction_rows(results),
            model_name=model_name, model_versions=model_versions, model_trained_at=model_trained_at
        )
    except Exception as e:
        logger.error(f"Failed to record forecast run for {model} as of {as_of_date}: {e}")
        return 0
