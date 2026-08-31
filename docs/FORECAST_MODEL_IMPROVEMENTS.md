# Forecast Model — Proposed Future Improvements

Candidate improvements for the MACD neural forecaster (`src/models/lstm_forecaster.py`,
`src/scripts/train_forecast_model.py`, `src/models/neural_forecast.py`), surfaced while
investigating why directional accuracy (DA) stays flat-to-worse across feature-engineering
attempts even as MAE/RMSE sometimes improve. None of these are implemented yet — this is a
proposal list, not a status log (see `docs/FORECAST_FIXES_PLAN.md` for that).

Audit date: 2026-08-30. Branch: `better_scripts`.

---

## Context: what prompted this list

Across several `bidirectional_gru` / `--residual-target` runs on the sp500 watchlist (hidden=32,
seq_length=30, forecast_horizon=5), a consistent pattern emerged:

- **MAE/RMSE and DA don't move together.** The best-ever MAE/RMSE run (`--extra-features
  open-close,volume`, `--days 600`) had MACD DA of 50.06% — the *worst* of four comparable runs —
  while an earlier, worse-MAE run scored 51.95%. Optimizing the current MSE-based loss does not
  reliably optimize the metric that actually matters for a bullish-signal product.
- **The model overfits within a handful of epochs regardless of target mode.** Both level-target
  (`Model B`, best epoch 3/200) and residual-target (`Model C`, best epoch 5/200) training runs
  peak almost immediately (see `src/models/MODEL_CARD.md` §11), and recent runs in this session
  showed the same pattern (best epoch 2/300, 4/300, 14/300 across different `--days`/feature
  configs). This points at limited genuinely-learnable multi-day signal, not a validation-noise
  artifact — confirmed against both thin (~500 windows) and healthy (~9,300 windows) validation
  sets.
- **Days 3-5 directional accuracy is below chance in every run measured so far** (typically
  38-45%), while delta's DA stays healthy (57-65%) across the same horizons. MACD and delta are
  not equally learnable, but they're currently trained and weighted identically.

The improvements below target these three observations directly, rather than adding more input
features (see `docs/FORECAST_FIXES_PLAN.md` and `src/models/MODEL_CARD.md` for the feature-side
history — `open`, `close`, `volume`, `open-close`, relative-`volume` have all been tried with
flat-to-negative results on DA).

---

## A. Output & loss function

### A1. Auxiliary directional (classification) loss — highest priority

**Problem:** MSE optimizes magnitude error. Its optimal answer under uncertainty is the mean,
which is exactly the documented "damping" failure mode (`MODEL_CARD.md` §12: "because the
MSE-optimal answer there is the mean, it actively teaches damping"). DA is a classification-like
metric; nothing in the current loss directly rewards getting the sign right.

**Proposal:** Add a small auxiliary head off the shared GRU representation that predicts
`sign(future delta)` per forecast day via binary cross-entropy, combined with the existing MSE:

```
loss = mse(predictions, targets) + lambda * bce(direction_logits, sign(actual_delta))
```

This directly optimizes the thing being evaluated (DA) instead of hoping it falls out of a
magnitude-accuracy objective as a side effect. `lambda` would need its own tuning pass.

**Effort:** Moderate — one more small `Linear` head sharing the GRU's final hidden state, one more
loss term, no change to existing target/normalization plumbing.

### A2. MACD/delta output consistency constraint

**Problem:** By construction, real data satisfies `delta[t+k] = macd[t+k] - macd[t+k-1]`, but the
model's `macd` and `delta` output columns come from the same flat `Linear` projection
(`BidirectionalGRUForecaster.forward()`, `lstm_forecaster.py:124-138`) with no architectural link
enforcing that relationship on *predictions*. The network has to learn the consistency
approximately through correlated gradients rather than getting it for free — wasted capacity in a
~28.5K-parameter model that already overfits in a handful of epochs.

**Proposal (pick one):**
- **Architectural (preferred):** predict `forecast_horizon` delta values only, then construct the
  MACD forecast as `macd[t] + cumsum(predicted_deltas)`, anchored to the known last value. Reduces
  the model's effective output degrees of freedom from `forecast_horizon * 2` to
  `forecast_horizon`, which should help given the fast-overfit signature.
- **Soft penalty (fallback if the architectural change interacts badly with `--residual-target`'s
  drift baseline):** add `Σ(pred_macd[k] - pred_macd[k-1] - pred_delta[k])²` to the loss.

**Effort:** Moderate. Touches `BidirectionalGRUForecaster.forward()` (and the other architecture
classes if consistency is desired across all of them), `MACDForecasterTrainer.prepare_sequences`/
`predict`, and the residual-target drift-baseline math in both `lstm_forecaster.py` and
`neural_forecast.py` (`CoreMLForecaster.predict`'s residual branch).

### A3. Per-target horizon-decay weighting

**Problem:** `loss_decay_gamma` currently discounts by forecast day only, applied identically to
both output columns (`lstm_forecaster.py:451-458`: `np.repeat(day_weights, target_size)` —
confirmed to produce the same weight for MACD and delta at a given day). But `MODEL_CARD.md`'s own
measured table shows delta beats drift on DA at every horizon (57-65%, days 1-5) while MACD only
wins day 1 and falls below chance by day 3. Equal weighting spends gradient on a MACD target
that's already known not to generalize past day 1.

**Proposal:** Replace the flat `day_weights` vector with a real `(forecast_horizon, target_size)`
matrix, allowing MACD's later-horizon weight to decay faster than delta's. Could start with two
independent `gamma` values (`--loss-decay-gamma-macd`, `--loss-decay-gamma-delta`) before
considering anything more adaptive (e.g. weights informed by `persistence_baseline.py`'s measured
per-day error).

**Effort:** Small — confined to `_compute_loss()` and the weight-construction block in `__init__`.

---

## B. Additional input features

Ordered by expected value; all are input-only (auxiliary), never targets — see
`src/models/MODEL_CARD.md` for why `target_size` currently maxes at 2 and what that constrains.

### B1. `macd_signal_dist` = `macd - signal_line`

Directly encodes the quantity this app's own bullish-signal logic keys off (`CLAUDE.md`'s column
definitions: `bullish_macd_above_signal == True` iff `macd > signal_line`). Cheap: one new
`elif f_lower == "macd_signal_dist":` branch mirroring `open-close`/`volume` in
`train_forecast_model.py::get_training_data`, `evaluate_forecast_model.py::get_historical_data_for_features`,
and `neural_forecast.py::forecast_macd`'s multi-feature branch.

### B2. MA20/MA50 trend-strength ratio — `(MA20 - MA50) / MA50`

Same idea as `open-close`: a scale-invariant crossover-distance indicator on a slower timescale
than MACD. Mirrors the app's own MA20-vs-MA50 signal concept, computed honestly from present-only
data (contrast with B4 below).

### B3. Price-vs-average mean-reversion ratio — `(Close - MA20) / MA20`

Classic overbought/oversold indicator, cheap to add, same ratio-based scale-invariance pattern.

### B4. `chart_patterns` (JSONB, YOLO-detected patterns) — biggest lever, most effort

The only candidate that's a genuinely different information source rather than another transform
of the same OHLCV numbers everything else derives from. Requires real encoding work (categorical
pattern labels, likely with confidence scores, need one-hot or embedding treatment) before it can
feed into the GRU input sequence alongside the numeric features — not a quick `--extra-features`
add like B1-B3. Worth revisiting once A1-A3 are evaluated.

### ⚠️ Do not use: `will_become_positive`, `ma20_will_be_above_ma50`

These `stock_cache` columns are **not observed data** — they're the app's own ARIMA forecast
outputs, written back by `cache_macd_positive_forecast()` / `cache_ma20_above_ma50_forecast()`
(`db_utils.py:847-876`), keyed to the future date they predict *for*. Using them as an input
feature would be label leakage: the model would look implausibly good in evaluation and be useless
in production. They're not currently reachable through the training pipeline (`fetch_bulk_from_cache`'s
SELECT doesn't include them, `COLUMN_MAPPING` doesn't map to them) — this note exists so nobody
wires them in later without realizing what they are.

---

## Suggested order

1. **A1** (auxiliary directional loss) — most directly targets the persistent MAE/DA mismatch.
2. **A3** (per-target decay weighting) — small, low-risk, complements A1.
3. **A2** (output consistency) — bigger change, evaluate once A1/A3 results are in.
4. **B1-B3** — cheap, can be tried in parallel with the above.
5. **B4** — only after the loss/output changes are evaluated; largest effort.
