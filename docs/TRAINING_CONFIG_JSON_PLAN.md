# Move forecast training config from CLI flags to a JSON file

Plan only — not yet implemented. Written 2026-08-30, branch `better_scripts`.

## Context

`src/scripts/train_forecast_model.py` has grown to **25 CLI flags** (`--symbols`,
`--watchlist`, `--days`, `--signal-type`, `--architecture`, `--normalization-type`, `--epochs`,
`--seq-length`, `--forecast-horizon`, `--hidden-size`, `--batch-size`, `--learning-rate`,
`--lr-scheduler`, `--lr-factor`, `--lr-patience`, `--lr-min`, `--checkpoint-warmup-epochs`,
`--output-dir`, `--test-split`, `--split-strategy`, `--skip-coreml`, `--include-delta`,
`--residual-target`, `--loss-decay-gamma`, `--extra-features`) — every recent training run in
this session has been a 6-8 line shell command that's easy to mistype and impossible to diff/version
cleanly. On the evaluation side, `src/scripts/evaluate_forecast_model.py` locates the model file
via `--architecture`, which today has to be the *exact* filename suffix string that
`train_forecast_model.py` computed and printed (e.g.
`bidirectional_gru_with_delta_residual_with_open-close_volume`) — there's no name you choose, you
have to reconstruct it by hand from the training flags, which is its own source of copy-paste
errors.

Goal: move all training hyperparameters into a JSON config file, and give each trained model an
explicit, user-chosen `model_name` that the evaluation script can use directly to find the model
file — no more manually rebuilding the architecture-suffix string.

## Current mechanics (for reference)

- Model files are always saved as `models/{signal_type}_{architecture}_forecaster.{pt,mlpackage}`
  via `get_model_path()` / `get_pytorch_model_path()` in `src/models/lstm_forecaster.py:1085-1098`.
  `train_forecast_model.py`'s `main()` builds `args.architecture` + suffixes
  (`_with_delta`, `_residual`, `_with_<extras>`) into `model_name` (`train_forecast_model.py:333-340`)
  before calling `get_model_path`/`get_pytorch_model_path` with that combined string.
- `evaluate_forecast_model.py::load_model(architecture, signal_type)` (line 144) calls those same
  two path functions with whatever `--architecture` string you pass — it has to match exactly.
- The full training config (seq_length, forecast_horizon, hidden_size, feature_names, etc.) is
  already persisted inside the `.pt` checkpoint (`MACDForecasterTrainer.save()`,
  `lstm_forecaster.py:967-987`) and the Core ML `user_defined_metadata`
  (`export_to_coreml()`), and read back by `load_model()` — so the *values* already round-trip
  correctly today. What's missing is a stable, chosen *name* to find the file by, and a
  non-CLI place to define the inputs in the first place.

## Design

### 1. JSON config schema — flat, one key per existing CLI flag, plus `model_name`

```json
{
  "model_name": "gru_v3_openclose_volume",
  "symbols": null,
  "watchlist": "sp500",
  "days": 600,
  "signal_type": "macd",
  "architecture": "bidirectional_gru",
  "normalization_type": "global",
  "epochs": 300,
  "seq_length": 30,
  "forecast_horizon": 5,
  "hidden_size": 32,
  "batch_size": 64,
  "learning_rate": 0.001,
  "lr_scheduler": true,
  "lr_factor": 0.5,
  "lr_patience": 10,
  "lr_min": 1e-6,
  "checkpoint_warmup_epochs": 20,
  "output_dir": null,
  "test_split": 0.15,
  "split_strategy": "time",
  "skip_coreml": false,
  "include_delta": true,
  "residual_target": true,
  "loss_decay_gamma": 0.8,
  "extra_features": "open-close,volume"
}
```

Flat and 1:1 with today's flag names (snake_case) — no schema redesign, so mapping is mechanical
and low-risk. `model_name` is the one genuinely new field.

Config files live in a new `src/configs/` directory (parallel to `src/scripts/`, `src/models/`),
e.g. `src/configs/gru_v3_openclose_volume.json`. Naming the file after the model is convention,
not enforced.

### 2. Precedence: JSON is primary, CLI flags become optional overrides

`train_forecast_model.py` gains `--config <path>` (required for the new workflow, but the script
still runs config-free with just CLI flags for quick one-offs — see Migration below). When
`--config` is given:

- Load the JSON into a dict.
- For every existing hyperparameter flag, change its argparse `default=` to `None` and move the
  *real* default (100 epochs, 0.001 lr, etc.) into a `DEFAULT_CONFIG` dict in a new small shared
  module, `src/scripts/config_utils.py`.
- Resolve each field with precedence **CLI (if explicitly passed) > JSON > DEFAULT_CONFIG**:
  `value = cli_value if cli_value is not None else json_config.get(key, DEFAULT_CONFIG[key])`.
- `model_name` has no CLI flag of its own reason to exist standalone (it's meaningless without a
  full training config) — it's JSON-only, required when `--config` is used.

This keeps the door open for the common "rerun the same config with one tweak" pattern
(`--config base.json --epochs 500`) without reintroducing a wall of flags for routine runs.

`config_utils.py` also exposes the merge function so `evaluate_forecast_model.py` can reuse the
exact same loading/precedence logic (see below) instead of duplicating it.

### 3. Model naming: auto-versioned, `{model_name}_{version}`

When a `model_name` is resolved (from JSON), files are saved as
`models/{model_name}_{version}.pt` / `.mlpackage`, where `version` starts at `1` and
auto-increments — **no overwrite, ever, and no fail-fast prompt**. Re-running the same JSON
twice just produces `_1` then `_2`. This replaces the current
`delta_suffix`/`residual_suffix`/`extra_suffix` concatenation (`train_forecast_model.py:333-340`)
for the JSON-driven path; the old auto-suffix naming stays exactly as-is when running without
`--config` (see backward-compat note below).

Add next to `get_model_path`/`get_pytorch_model_path` in `src/models/lstm_forecaster.py:1085-1098`:

```python
def get_latest_model_version(models_dir: str, model_name: str) -> int:
    """Highest existing version for model_name, or 0 if none exist yet."""
    pattern = re.compile(rf"^{re.escape(model_name)}_(\d+)\.(pt|mlpackage)$")
    versions = {int(m.group(1)) for f in os.listdir(models_dir) if (m := pattern.match(f))}
    return max(versions) if versions else 0
```

Both `get_model_path`/`get_pytorch_model_path` gain optional `model_name`/`version` params; when
`model_name` is given and `version` is `None`, they resolve to
`get_latest_model_version(...) + 1` for **saving** (training) — training always calls this fresh
right before `save()`/`export_to_coreml()` so it can't race against itself within one run. For
**loading** (evaluation, design #4), the same helper with no `+1` gives the latest existing
version.

Running `train_forecast_model.py` **without** `--config` keeps today's exact behavior (CLI flags,
auto-suffix filename, plain overwrite-on-save) — fully backward compatible, no forced migration
for quick experiments.

### 4. Evaluation script: defaults to the latest version, `--model-version` to pin one

`evaluate_forecast_model.py` gains:

- `--config <path>` — load the *same* JSON used for training (via the shared `config_utils`
  loader) and pull `model_name` and `signal_type` from it, so evaluating a model is just
  `python evaluate_forecast_model.py --config src/configs/gru_v3.json --watchlist sp500 --samples
  20 --lag-test`. Run-specific flags (`--watchlist`, `--samples`, `--verbose`,
  `--breakdown-by-day`, `--lag-test`, `--compare`, `--arima-only`, `--forecast-horizon`,
  `--inference-forcast-horizon`) stay as plain CLI flags — they're properties of the *evaluation
  run*, not the model, and there are few enough of them that JSON would be overkill.
- `--model-name <name>` — standalone alternative for quick lookups without the JSON file on hand.
- `--model-version <N>` — optional; when omitted, resolves to
  `get_latest_model_version(models_dir, model_name)` (the newest trained version for that name).
  When given, pins to that exact version instead.

`load_model()` (`evaluate_forecast_model.py:144`) changes to accept optional
`model_name`/`version`; when `model_name` is present it calls `get_model_path`/
`get_pytorch_model_path` with the resolved version instead of building the path from
`architecture`. **`--architecture` keeps working exactly as today** (unchanged code path) for full
backward compatibility with the many already-trained `.pt`/`.mlpackage` files already in
`models/`, none of which follow the new `_{version}` convention — those stay reachable exactly as
before, by their existing literal architecture-suffix string.

## Files touched

| File | Change |
|---|---|
| `src/scripts/config_utils.py` (new) | `DEFAULT_CONFIG` dict, JSON loader, CLI/JSON/default merge function shared by both scripts |
| `src/scripts/train_forecast_model.py` | `--config` flag; all hyperparameter flags' `default=` → `None`; merge via `config_utils`; auto-versioned `model_name`-based path resolution in `main()` |
| `src/scripts/evaluate_forecast_model.py` | `--config`, `--model-name`, `--model-version` flags; `load_model()` accepts `model_name`/`version`, defaults to latest |
| `src/models/lstm_forecaster.py` | `get_latest_model_version()` (new); `get_model_path()` / `get_pytorch_model_path()` gain optional `model_name`/`version` params (additive, lines 1085-1098) |
| `src/configs/` (new dir) | example config JSON(s), e.g. the `--days 600` run from this session, so there's a working reference from day one |

## Verification

- Run `python scripts/train_forecast_model.py --config src/configs/<example>.json` from `src/`
  (per this repo's convention) with a small `--epochs` override for a quick smoke run, confirm the
  printed config summary matches the JSON, and confirm `models/<model_name>_1.pt` /
  `_1.mlpackage` are created.
- Re-run the same config a second time unchanged and confirm it saves as `_2` without touching
  `_1` — no overwrite, no prompt.
- Run `python scripts/evaluate_forecast_model.py --config src/configs/<example>.json --watchlist
  sp500 --samples 5` and confirm it picks `_2` (the latest) by default; re-run with
  `--model-version 1` and confirm it picks `_1` instead.
- Run the **existing** flag-only invocations from earlier in this session (no `--config`) against
  both scripts and confirm output is byte-for-byte the same as before (backward-compat check).
