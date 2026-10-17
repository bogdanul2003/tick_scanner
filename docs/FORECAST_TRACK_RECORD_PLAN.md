# Forecast Track Record in the Bullish Forecast Panel — Plan

Show, in the **Bullish MACD Forecast for ...** panel, how the selected model's stored
forecasts compared with what MACD then did on each of the last five trading days.

Plan date: 2026-10-06. **Status: implemented 2026-10-06** — backend verified against
the database (the endpoint reproduces `docs/FORECAST_LIVE_TRACKING.md`'s day-1 figures)
and covered by unit tests; the panel builds and lints clean but has not been exercised
by hand in a browser. Builds on the forecast history table (`forecast_predictions`,
`docs/ENSEMBLE_PRODUCTION_PLAN.md` §6) and on the scoring rules used in
`docs/FORECAST_LIVE_TRACKING.md`.

## What it shows

Under the model dropdown, a small table for the selected model and the watchlist being
viewed: one column per trading day (the last five settled ones), two rows.

```
Track record, last 5 trading days — macdonly_warm5
                    Sep 29   Sep 30   Oct 1    Oct 2    Oct 5
All symbols          NaN      NaN      NaN      NaN     72.5%
                                                      drift 75.1%
Top 100 best match   NaN      NaN      NaN      NaN     ≤24% off
                                                      69 beat drift
```

(The figures are the default ensemble's on sp500 as of 2026-10-06.)

### Row 1 — all symbols

- **Score**: share of the watchlist's symbols where the forecast for that day moved in
  the right direction: forecast and actual MACD on the same side of the previous day's
  actual MACD. Same rule as the evaluation scripts.
- **Under it**: drift's score on the same forecasts (MACD keeps changing by its last
  daily change). Green when the model beats drift, red when it does not.
- **Hover**: symbols scored, mean and median error, drift's mean error, how many days
  ahead the forecasts were made.

### Row 2 — top 100 best match

- **Selection**: per day, the 100 symbols whose forecast came closest, ranked by error
  as a share of how far MACD moved since the forecast's starting day.
- **Shows**: the error of the last symbol in the list ("≤24% off": all of them were
  within 24% of the move) and how many of them beat drift. A direction score is not
  shown here: it is near 100% by construction.
- **Click**: lists that day's symbols with forecast, actual, error and drift's error,
  best first.
- **Small watchlists**: with fewer than 100 scored symbols the row uses the best half
  and is labelled accordingly ("Top 21 of 42").

### NaN

A day shows `NaN` in both rows when no stored forecast of the selected model targets it.

## Rules

1. **One forecast per symbol and day: the most recent one made before it.** A day can
   have been forecast up to five times (1 to 5 trading days ahead). The latest is used
   and its lead time shown.
2. **Only forecasts made by the versions currently served.** After a retrain, earlier
   days are `NaN` for that ensemble until new runs accumulate. ARIMA has no versions.
3. **No backfill.** Only forecasts stored before the outcome count; nothing is
   recomputed after the fact.
4. **Days are trading days**, taken from the dates present in `stock_cache` up to the
   latest settled market date, so holidays do not shift anything.
5. The top-100 list is chosen after the outcome is known. It says how close the best
   forecasts were, not which symbols will be forecast well next.

## Build

- **`src/db_utils.py`** — `get_forecast_track_rows`: one query returning, for a model
  and a symbol list, every stored forecast that targets one of the last N trading days,
  with the actual MACD on that day, the day before it, and on the forecast's starting
  day and the day before that (for drift).
- **`src/services/forecast_track_record.py`** — `summarize_track_record`: a pure
  function from those rows to the per-day entries; `get_track_record`: resolves the
  model's current versions and calls the two.
- **`src/routers/forecast_router.py`** — `POST /forecast/macd/track_record` with
  `{symbols, model, days, top_n}`. `model` is an ensemble id or `arima`; omitted means
  the default ensemble. Missing days come back with `available: false` and null values.
- **`frontend/src/dashboards/MacdDashboard.jsx`** — a `ForecastTrackRecord` component in
  `WatchlistBullishForecast`, loaded when the panel opens and when the model or its
  served versions change. The forecast result list is untouched.
- **Tests** (`src/tests/test_forecast_track_record.py`) — a full day, a missing day,
  latest-forecast selection, drift, ranking order and cut-off, a watchlist smaller than
  100, a symbol whose MACD did not move, the version filter.
- **Docs** — endpoint and test rows in `docs/ARCHITECTURE.md`; a pointer in
  `docs/ENSEMBLE_PRODUCTION_PLAN.md` §6.

## Later

- Symbols that land in the top 100 on most days ("reliably well forecast"), once
  several days of history exist.
- A per-symbol marker in the forecast result list for those symbols.
