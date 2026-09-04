"""
Shared JSON training-config loading/merging for train_forecast_model.py and
evaluate_forecast_model.py.

Precedence for every hyperparameter: CLI (if explicitly passed) > JSON config
value > DEFAULT_CONFIG. A CLI value counts as "explicitly passed" when it is
not None, so every hyperparameter flag in train_forecast_model.py must have
its argparse `default=None` for this to work — the real default lives here
instead, in DEFAULT_CONFIG.
"""
import json

# Mirrors train_forecast_model.py's argparse defaults. `model_name` is
# deliberately absent: it only makes sense alongside a full config (JSON-only,
# required whenever --config is used).
DEFAULT_CONFIG = {
    "symbols": None,
    "watchlist": "sp500",
    "days": 365,
    "signal_type": "macd",
    "architecture": "stacked_lstm",
    "normalization_type": "global",
    "epochs": 100,
    "seq_length": 30,
    "forecast_horizon": 5,
    "hidden_size": 64,
    "batch_size": 32,
    "learning_rate": 0.001,
    "lr_scheduler": False,
    "lr_factor": 0.5,
    "lr_patience": 10,
    "lr_min": 1e-6,
    "checkpoint_warmup_epochs": 0,
    "output_dir": None,
    "test_split": 0.15,
    "split_strategy": "symbol",
    "skip_coreml": False,
    "include_delta": False,
    "residual_target": False,
    "loss_decay_gamma": None,
    "loss_decay_gamma_delta": None,
    "extra_features": None,
    "auxiliary_direction_lambda": None,
    "predict_deltas_only": False,
    "seed": None,
    "allow_empty_test_set": False,
}


def load_json_config(path: str) -> dict:
    """Load a training config JSON file."""
    with open(path) as f:
        return json.load(f)


def get_model_name(json_config: dict) -> str:
    """Extract the required `model_name` field from a loaded config JSON."""
    model_name = json_config.get("model_name")
    if not model_name:
        raise ValueError("Config JSON must include a non-empty 'model_name' field")
    return model_name


def resolve_config(cli_values: dict, json_config: dict = None, defaults: dict = None) -> dict:
    """
    Merge CLI flag values, a JSON config dict, and defaults with precedence
    CLI (non-None) > JSON > defaults.

    `cli_values` and the returned dict share `defaults`' keys (DEFAULT_CONFIG
    unless overridden). `json_config` may be None or {} when no --config was
    given, in which case this is equivalent to "CLI > defaults".
    """
    if defaults is None:
        defaults = DEFAULT_CONFIG
    json_config = json_config or {}
    resolved = {}
    for key, default_value in defaults.items():
        cli_value = cli_values.get(key)
        if cli_value is not None:
            resolved[key] = cli_value
        elif key in json_config:
            resolved[key] = json_config[key]
        else:
            resolved[key] = default_value
    return resolved
