# Ensemble Forecast in Production — Plan

Replace the ARIMA call behind **Show Bullish Forecast** with a selectable seed ensemble of
the neural MACD forecaster, and allow retraining an ensemble from the UI.

Plan date: 2026-10-04. Branch: `better_scripts`. Background and measurements:
`src/models/MODEL_CARD.md` §9 and §11, `docs/FORECAST_MODEL_IMPROVEMENTS.md`
("Results log — 2026-10-04").

## Status (2026-10-04): implemented

Everything in this plan is built, plus forecast history (§6), which was added afterwards.

Verified:
- Unit tests: 154 passing (`test_ensemble_service.py`, `test_ensemble_training.py` are new).
- The ensemble endpoint reproduces `evaluate_ensemble.py`'s predictions exactly for the same
  window, on all three registered ensembles, and forecasts the 501-symbol sp500 watchlist
  in about 4 seconds with no failures.
- A real retrain of `hidden16` through the API: three seeds trained in parallel, the
  ensemble was evaluated, the served versions switched from 1-3 to 4-6 (11m41s). A second
  retrain request during it was refused.
- Ensemble and ARIMA runs are stored in `forecast_predictions`; a second run from the same
  market date replaced the first.

Not verified: the panel has not been exercised by hand in a browser (it builds, and the
edited component lints clean).

What turned out differently from the plan below:
- ARIMA runs are stored in the forecast history too, under the model id `arima`.
- The dropdown also shows each ensemble's training date. It comes from the retrain state,
  or from the newest member's `.pt` file time — not the `.mlpackage`, whose time changes
  every time the model is loaded.
- Retraining on unchanged data reproduces the same models (the `hidden16` retrain scored
  exactly what it had before), so Retrain only matters once new trading days have arrived.
- "Where things stand" below describes the state before this work.

---

## Where things stand

- The button calls `POST /forecast/macd/arima_positive`, which runs ARIMA directly. It does
  not go through `ForecastService` or `NeuralForecastService`, so no neural model is ever
  served, whatever is in `models/`.
- `NeuralForecastService` can only load the legacy path
  `models/macd_stacked_lstm_forecaster.mlpackage`; every model measured so far is
  config-based and versioned (`models/{model_name}_{version}.*`).
- Measured 2026-10-04 on sp500 (20 samples/symbol): the three-seed ensemble of
  `gru_v1_residual_macdonly_warm5` has MACD DA 51.3% and median MAE 0.452; ARIMA has 46.4%
  and 0.577; drift has 57.8% DA.

## What gets built

### 1. Ensemble registry

`src/configs/ensembles.json` lists the selectable ensembles. Each entry pins a config and
exact versions; one is the default.

| id | Config | Versions | Default |
|---|---|---|---|
| `macdonly_warm5` | `gru_v1_residual_macdonly_warm5.json` | 1, 2, 3 | yes |
| `hidden16` | `experiments/gru_v1_exp_hidden16.json` | 1, 2, 3 | |
| `seq60_d750` | `gru_v1_residual_seq60_d750.json` | 2, 3, 4 | |

Versions are pinned rather than "latest three" because stray runs share a model name
(`seq60_d750` v1 is from an interrupted earlier run).

The registry is static and lives in git. What a retrain changes — the versions currently
served, when they were trained, their measured metrics — is written to
`models/ensemble_state.json`, next to the model files it describes (`models/` is
gitignored). State overrides the registry's versions when present.

### 2. Backend inference

- `EnsembleForecaster` moves from `scripts/evaluate_ensemble.py` into
  `models/neural_forecast.py`, shared by the app and the evaluation script.
- `GET /forecast/ensembles` — registry entries plus, for each: whether its model files
  exist, served versions, trained-at, last measured MAE/DA, and the default id.
- `POST /forecast/macd/ensemble` — body `{symbols, ensemble_id}`. Returns what the ARIMA
  endpoint returns (`will_become_positive`, `forecasted_macd` keyed by date,
  `forecasted_dates`, `details.last_macd`), in the same order, so the result list renders
  unchanged.
- Price data is loaded once in bulk for all symbols.
- `will_become_positive` is still written to `stock_cache`: the Combined Forecast button
  reads it. It uses the same rule as the ARIMA path.
- `POST /forecast/macd/arima_positive` is kept and offered as "ARIMA (legacy)" in the
  dropdown. The MA20>MA50 forecast is a separate ARIMA model and is untouched.

### 3. Retraining

- `POST /forecast/ensembles/{id}/retrain` starts a background job;
  `GET /forecast/ensembles/retrain/status` reports it. One job at a time.
- The job: refresh the training data in bulk → train seeds 42, 7, 123 on CPU in parallel
  (single-threaded each, ~10-15 min) → evaluate the new ensemble on the config's
  watchlist → record versions and metrics in `models/ensemble_state.json`.
- `train_forecast_model.py` gains `--model-version`, so the job reserves three version
  numbers up front. Without it, parallel runs of one config can resolve the same version.
- The served versions switch only if all three seeds succeed. Old versions are never
  deleted: they keep serving during the retrain and remain for rollback.
- Job state is persisted to a file, because the dev server runs with auto-reload and
  would lose in-memory state.

### 4. Frontend (`WatchlistBullishForecast` in `MacdDashboard.jsx`)

- Opening the panel runs nothing. It shows the ensemble dropdown (default preselected),
  **Run forecast** and **Retrain**.
- Run forecast calls the new endpoint and renders the list exactly as today, with a line
  naming the ensemble used.
- Retrain asks for confirmation, then polls the status and shows the stage and elapsed
  time. Run stays usable on the current versions meanwhile.
- An ensemble whose model files are missing shows as "not trained", with only Retrain
  enabled — the state of any fresh checkout.

### 5. Tests and docs

- `src/tests/` (unittest, Core ML stubbed): registry validation, ensemble averaging and
  member-mismatch refusal, response shape and ordering, retrain job state transitions with
  the subprocess stubbed, `--model-version` refusing to overwrite.
- Update `MODEL_CARD.md` §9, `docs/ARCHITECTURE.md`, `CLAUDE.md`.

### 6. Forecast history (added 2026-10-04)

Every run of `POST /forecast/macd/ensemble` and `POST /forecast/macd/arima_positive` is
recorded in the `forecast_predictions` table, so accuracy can be measured on forecasts
that were actually made rather than only recomputed after the fact.

| Column | Meaning |
|---|---|
| `model` | Ensemble id, or `arima` |
| `symbol`, `as_of_date` | What was forecast, and the last market date it was forecast from |
| `horizon_day` | 1..5 trading days ahead |
| `target_date` | The nominal date shown in the UI (weekdays; holidays are not skipped) |
| `predicted_macd`, `last_macd`, `will_become_positive` | The forecast, the MACD it started from, and the flag shown |
| `model_name`, `model_versions`, `model_trained_at` | Which models made it (null for ARIMA) |
| `run_at` | When the run happened |

- Primary key `(model, symbol, as_of_date, horizon_day)`. A later run of the same model
  from the same market date replaces the earlier one for the symbols it covers, so only
  the last run of a day is kept. Symbols whose forecast failed are not stored.
- `model_trained_at` comes from `models/ensemble_state.json` for an ensemble retrained
  through the app, otherwise from the newest member's `.pt` file time. Model files carry
  no training date of their own, and the `.mlpackage`'s time is not usable: loading a
  Core ML model rewrites its `Manifest.json`.
- Recording never fails a forecast: an error is logged and the forecast is still returned.

Scoring, once the forecast days have happened and the watchlist has been refreshed. It
matches on trading-day offset, not `target_date`, and uses the evaluation script's
direction rule (did the prediction and the actual move the same way from the previous
actual value):

```sql
WITH actuals AS (
  SELECT symbol, date, macd,
         ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
  FROM stock_cache WHERE macd IS NOT NULL
)
SELECT p.model, p.horizon_day, COUNT(*) AS forecasts,
       ROUND(AVG(ABS(p.predicted_macd - t.macd))::numeric, 4) AS mae,
       ROUND(AVG(((p.predicted_macd > prev.macd) = (t.macd > prev.macd))::int)::numeric, 4) AS directional_accuracy
FROM forecast_predictions p
JOIN actuals o    ON o.symbol = p.symbol AND o.date = p.as_of_date
JOIN actuals t    ON t.symbol = p.symbol AND t.rn = o.rn + p.horizon_day
JOIN actuals prev ON prev.symbol = p.symbol AND prev.rn = t.rn - 1
GROUP BY p.model, p.horizon_day
ORDER BY p.model, p.horizon_day;
```

It pools all forecasts, where the evaluation scripts average per symbol first, so the two
will differ slightly. Only days that were actually forecast are scored: to build a
history, run the forecast on each trading day.

## Order of work

1. Registry, shared ensemble class, the two inference endpoints. Check endpoint output for
   a few symbols against `evaluate_ensemble.py`.
2. Frontend dropdown and Run forecast.
3. Retrain job and `--model-version`.
4. Frontend Retrain and polling.
5. Tests and docs.

## Decisions taken

- ARIMA stays selectable as a legacy option.
- A successful retrain switches the served versions automatically; no manual promote step.
- Drift is not offered as an option.

## Known limits

- **Direction is still weaker than drift.** The ensemble beats ARIMA on both measures,
  but `will_become_positive` is a direction call and drift scored 57.8% DA against 51.3%.
- **Retraining uses the config's watchlist (sp500)**, not the watchlist being viewed.
  Forecasts for symbols outside sp500 work but are unmeasured.
- **Metrics move with each retrain**, since each uses newer data; they will drift from the
  model card's figures.
