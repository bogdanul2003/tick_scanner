# Forecast Model — Proposed Future Improvements

Candidate improvements for the MACD neural forecaster (`src/models/lstm_forecaster.py`,
`src/scripts/train_forecast_model.py`, `src/models/neural_forecast.py`), surfaced while
investigating why directional accuracy (DA) stays flat-to-worse across feature-engineering
attempts even as MAE/RMSE sometimes improve. None of these are implemented yet — this is a
proposal list, not a status log (see `docs/FORECAST_FIXES_PLAN.md` for that).

Audit date: 2026-08-30. Branch: `better_scripts`.

**Update 2026-10-04:** A1, A2 and A3 have since been implemented and tried (see
`src/models/MODEL_CARD.md` §5-§6). A multi-seed re-measurement on 2026-10-04 is recorded in
"Results log — 2026-10-04" at the end of this file; it changes the status of A1 and the
suggested order. The proposal text below is left as written.

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

---

## Results log — 2026-10-04

Protocol: fresh database, sp500 (501 symbols), data through 2026-10-02, `--days 600`,
`evaluate_forecast_model.py --watchlist sp500 --samples 20 --breakdown-by-day --lag-test`, Core ML
inference. Every model was retrained on CPU (`--device cpu`, torch 2.14.1, Python 3.13) with an
explicit seed. Raw outputs: `eval_*_2026-10-03.txt` / `eval_*_2026-10-04.txt` in the repo root.
These numbers are comparable to each other, not to the pre-October tables (different window).

Baselines on this window: `drift` MACD DA 57.81%, MAE 1.1466, per-day DA
80.8 / 62.6 / 52.1 / 47.7 / 45.7. `flat` MAE 1.2565.

### 1. Seed variance dominates every DA comparison

| Config | DA seed 42 / 7 / 123 | Mean DA | Mean MAE |
|---|---|---|---|
| `gru_v1_residual_macdonly_warm5` | 56.65 / 50.86 / 46.63 | 51.4% | 0.935 |
| `gru_v1_exp_hidden16` | 55.28 / 47.97 / 49.78 | 51.0% | 0.924 |
| `gru_v1_residual_seq60_d750` | 52.68 / 47.46 / 48.59 | 49.6% | 0.925 |

Seeds of one config span 5-10pp of aggregate MACD DA, nearly all on days 3-5; day 1 is stable.
The noise floor in `src/configs/experiments/README.md` (TS-017: sd 0.77pp, range 1.71pp) does not
hold on this data. Seed 42 was the best seed in all three configs. Two single-seed "wins" —
`hidden16` at 55.3% and `macdonly_warm5` at 56.7% — both fell back to ~51% on reseeding, and
`seq60_d750`'s MAE of 0.901 did not repeat (0.930, 0.942). **Consequence: the single-run
observations in "Context" above (e.g. 50.06% vs 51.95%) are inside this spread and do not
establish a difference.**

### 2. A1 (auxiliary directional loss) — no benefit on the current recipe

`gru_v1_residual_macdonly_warm5_aux03` = `macdonly_warm5` + `auxiliary_direction_lambda: 0.3`,
seeds 42 / 7 / 123:

| | With A1 | Without |
|---|---|---|
| MACD DA by seed | 45.27 / 46.33 / 48.18 | 56.65 / 50.86 / 46.63 |
| Mean MACD DA | 46.6% | 51.4% |
| Mean MACD MAE | 0.956 | 0.935 |
| Mean per-day DA | 78.2 / 50.5 / 36.8 / 34.3 / 33.3 | 81.0 / 56.1 / 42.1 / 39.3 / 38.3 |

Worse in two of three seeds and on every forecast day on average. With n=3 against a control
that itself spans 10pp this is not proof of harm, but there is no sign of the benefit recorded
earlier (which was measured with `loss_decay_gamma` on, apparently from single runs). Only
`lambda=0.3` was tested.

### 3. Hyperparameters and input window — nothing outside the seed spread

Single seed (42) unless noted. `gru_v1_exp_baseline` (the `plainloss_warm5` recipe): 49.82% /
MAE 0.941.

| Change | MACD DA | MACD MAE | Best epoch |
|---|---|---|---|
| `lr_patience` 2 | identical to baseline | | 7 — same checkpoint; the scheduler fires after it |
| `learning_rate` 3e-4 | 48.42% | 0.901 | 6 |
| `learning_rate` 1e-4 | 47.46% | 0.920 | 13 |
| `hidden_size` 16 (3 seeds) | 51.0% mean | 0.924 mean | — |
| `hidden_size` 16 + lr 3e-4 | 49.29% | 0.900 | 8 |
| `hidden_size` 8 | 44.19% | 0.982 | 10 |
| `seq_length` 15 / 20 / 45 | 46.64 / 48.46 / 49.09% | 0.964 / 0.947 / 0.969 | 8 / 6 / 18 |
| `seq_length` 60, `days` 750 (3 seeds) | 49.6% mean | 0.925 mean | 6 / 8 / 8 |
| drop `open-close` (3 seeds) | 51.4% mean | 0.935 mean | 7 / 6 / 8 |

Lower learning rates did not move the optimum away from epoch 6-13 in any useful way, and the
~0.90 MAE they produced is unconfirmed. Dropping `open-close` cost nothing, so B-list features
derived from the same price series (B1-B3) have a low prior.

Horizons 2 and 3 (`gru_v1_residual_h2_w0`, `gru_v1_residual_h3`): ~5% better day-1/2 MAE than the
5-day models, direction no better than drift at the same horizon (day 2: 61.3% vs 62.9%; day 3:
42.7% vs 52.7%).

### 4. Seed ensembles — better MAE, same DA

`src/scripts/evaluate_ensemble.py` averages the forecasts of several versions of one model.

| Family | Ensemble MAE | Members' mean MAE | Ensemble DA | Members' mean DA |
|---|---|---|---|---|
| `macdonly_warm5` v1,2,3 | 0.901 | 0.935 | 51.28% | 51.38% |
| `hidden16` v1,2,3 | 0.906 | 0.924 | 51.48% | 51.01% |
| `seq60_d750` v2,3,4 | 0.906 | 0.925 | 49.41% | 49.58% |

The ensemble's DA equals the members' average, so seeds do not cancel each other's day 3-5
errors: the weak long-horizon direction is a bias the seeds share, not noise between them.

### 5. ARIMA comparison — ARIMA is behind both the neural model and drift

`evaluate_forecast_model.py --compare` on `gru_v1_residual_macdonly_warm5` v2 (seed 7), ARIMA
fitted on 501/501 symbols. Raw output: `eval_arima_compare_2026-10-04.txt`.

| | Mean MAE | Median MAE | Median WAPE | MACD DA | Day 1 / 2 / 3 / 4 / 5 DA |
|---|---|---|---|---|---|
| Neural, 3-seed ensemble | 0.901 | 0.452 | 21.6% | 51.28% | 81.0 / 56.4 / 42.1 / 38.9 / 38.0 |
| Neural, v2 | 0.928 | 0.456 | 22.1% | 50.86% | 80.9 / 55.4 / 41.8 / 38.6 / 37.6 |
| ARIMA | 1.682 | 0.577 | 27.2% | 46.41% | 77.7 / 48.9 / 36.9 / 34.8 / 33.7 |
| `drift` | 1.147 | — | — | 57.81% | 80.8 / 62.6 / 52.1 / 47.7 / 45.7 |

Neural median MAE is ~21% below ARIMA's; ARIMA's mean is 2.9x its median (divergent fits), so
compare medians. ARIMA trails the neural model by 3-6.5pp of DA on every day and drift by 3-15pp.
Production currently falls back to ARIMA because it cannot load config-based models
(`MODEL_CARD.md` §9) — the weakest of the options measured here.

### 6. Training cost

At this model size CPU is ~3.6x faster than MPS (4.7 vs 17.2 s/epoch, Apple M5), and four
single-threaded CPU runs in parallel give ~3.5x throughput. Details in `MODEL_CARD.md` §12.

### 7. In production

The `gru_v1_residual_macdonly_warm5` three-seed ensemble is what "Show Bullish Forecast"
serves by default as of 2026-10-04, replacing ARIMA; ensembles can be chosen and retrained
from the UI, and every forecast run is stored in `forecast_predictions`
(`docs/ENSEMBLE_PRODUCTION_PLAN.md`, `MODEL_CARD.md` §9). An end-to-end retrain of `hidden16`
on unchanged data reproduced the previous ensemble's metrics exactly, so seeded CPU training
is reproducible and a retrain only matters once new data has arrived.

### Suggested order, revised 2026-10-04

0. Score the frozen default ensemble on live data after two to four weeks (`MODEL_CARD.md`
   §13) — every result above is from a single window ending 2026-10-02.

1. Use at least three seeds per arm for any DA claim; treat earlier single-run rankings as open.
2. Checkpoint selection on DA (evaluate several early checkpoints) — the seed spread and the
   checkpoint effect live on the same days.
3. **B4** (`chart_patterns`) — the only input that is not a function of the same price series.
4. **B1-B3** — cheap, but low prior after the `open-close` result.
5. **A1** at other `lambda` values — only with three seeds per value; 0.3 showed no benefit.
