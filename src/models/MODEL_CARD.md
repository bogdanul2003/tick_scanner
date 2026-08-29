# MACD Forecaster — Model Card

Reference for the neural MACD forecaster in `lstm_forecaster.py` (training) and
`neural_forecast.py` (inference). Covers what the model consumes, what it
predicts, how to read its metrics, and what it has actually measured.

Last updated 2026-08-29. Current best model: **Model C** (§11) —
`macd_bidirectional_gru_with_delta_residual_forecaster`.

---

## 1. What the model sees

`get_training_data` (`../scripts/train_forecast_model.py:56-73`) queries
`stock_cache` per symbol and extracts **one column** — `macd` (or `signal_line`
with `--signal-type signal_line`) — dropping nulls. The result is one 1-D array
per symbol in date order. Symbols with fewer than 50 points are skipped.

**That is the entire input.** Close price, volume, MA20 and MA50 are all present
in `stock_cache` and unused. MACD is itself `EMA12(close) - EMA26(close)` — a
single scalar summarizing price momentum, and a low-pass filter over price.

Consequence worth internalizing: the model can only learn the filter's inertia
plus whatever autocorrelation price retains. It cannot recover information the
smoother destroyed. That is the ceiling on this feature set.

## 2. Windowing

Stride-1 sliding window (`lstm_forecaster.py:479`):

- **input** = `seq_length` consecutive MACD values (default 30 ≈ 6 calendar weeks)
- **target** = the next `forecast_horizon` values (default 5 ≈ 1 week)

A ~363-point symbol yields `363 - 30 - 5 + 1 = 329` windows. Consecutive windows
share 29 of 30 inputs — which is why any split must happen on the raw series
*before* windowing (see §7).

### Worked example (real MSFT data, window ending 2026-08-03)

```
input X  : 30 MACD values, last = 14.8577  (previous day 9.3003)
target y : 19.4541, 22.4068, 25.4540, 27.5616, 29.3831
```

## 3. The `--include-delta` feature

Each timestep becomes a pair `(macd, macd - previous_macd)`:

```
X[-1] = (14.8577, 5.5574)        # 5.5574 = 14.8577 - 9.3003
y     = [(19.4541, 4.5964), (22.4068, 2.9527), (25.4540, 3.0472),
         (27.5616, 2.1077), (29.3831, 1.8214)]
```

**The delta column is a deterministic function of the MACD column — it adds no
information.** It is a reparameterization that hands the network the first
difference instead of making it learn subtraction, plus a second prediction head
(so `input_size` and the output width both double). Useful as an inductive bias;
not new data.

`_calculate_deltas` sets element 0 to zero. At training time this is applied to
the full per-symbol series before windowing, so every window but the first gets a
true delta at position 0. At inference the `prev_value` argument reconstructs it —
see §9 for the open bug.

## 4. Normalization

| `--normalization-type` | What the model can see |
|---|---|
| `global` | values standardized by dataset-wide mean/std (fit on **train only**) — the model *can* distinguish MACD at +30 from MACD near 0 |
| `internal` | each window standardized by **its own** mean/std — the model sees only the *shape* of the last 30 days and cannot tell where the level is |

`internal` is a poor fit for this product: `will_become_positive` is a question
about the **level** (does MACD cross zero), and internal normalization deletes
exactly that information from the model's input. The level is still restored on
denormalization, so outputs are in correct units — but the model cannot condition
on it.

Note: with `internal`, `self.mean`/`self.std` are never assigned and the saved
metadata contains the `[0.0, 0.0]` / `[1.0, 1.0]` initializers. Harmless while
`normalization_type` is read correctly (the internal path recomputes per window),
but it is a landmine — `_load_model` defaults the key to `"global"` when absent.

## 5. Architecture and shapes

Five architectures, selected by `--architecture`, all the same shape: recurrent
encoder → take a final hidden state → 2-layer MLP head.

`stacked_lstm` (2-layer LSTM), `bidirectional_gru`, `stacked_gru`,
`standard_lstm` (1 layer), `gru` (1 layer). Factory: `create_model()`.

`bidirectional_gru` combines the two directions correctly
(`lstm_forecaster.py:140-142`): forward state at the **last** timestep (has read
`x[0..T-1]`) concatenated with backward state at the **first** timestep (has read
`x[T-1..0]`). Both are full-sequence summaries. There is **no future leakage** —
the input window is entirely historical and fully observed at inference too.

Shapes for `bidirectional_gru`, `hidden=64`, `layers=2`, `include_delta`, `horizon=5`:

| stage | shape |
|---|---|
| input | `(batch, 30, 2)` |
| bidirectional GRU ×2 layers | `(batch, 30, 128)` |
| forward@t=29 ⊕ backward@t=0 | `(batch, 128)` |
| `Linear(128,64) → ReLU → Dropout(0.2) → Linear(64,10)` | `(batch, 10)` |
| reshape (C-order) | `(5 days, 2 features)`, interleaved `[d1_macd, d1_delta, d2_macd, …]` |

**Parameter count** for that configuration: **109,514**

```
layer 1, per direction : 3*64*2 + 3*64*64 + 2*(3*64)   = 13,056   x2 = 26,112
layer 2, per direction : 3*64*128 + 3*64*64 + 2*(3*64) = 37,248   x2 = 74,496
head                   : 128*64+64 + 64*10+10                    =  8,906
```

With `hidden=32` the same configuration is **28,138** parameters.

Watch the **windows-per-parameter ratio** — but note it was not the binding
constraint here:

| run | params | train windows | ratio | best epoch |
|---|---|---|---|---|
| Model B (`hidden=64`) | 109,514 | 109,855 | 1.0 | 3 / 200 |
| Model C (`hidden=32`) | 28,138 | 106,114 | 3.8 | 5 / 200 |

Quartering capacity moved the overfitting point from epoch 3 to epoch 5. The
learnable signal is genuinely thin — the model finds it within a handful of
epochs and memorizes after that. More capacity is not the lever; more
*information* is (§14).

## 6. Objective — and the mismatch that matters

```python
self.criterion = nn.MSELoss()
loss = self.criterion(predictions, batch_y)
```

Optimizer is Adam (`--learning-rate`, default 1e-3). One scalar loss: mean squared
error over all `forecast_horizon * input_size` outputs.

So the training objective is literally **"minimize squared error of the MACD level
over the next 5 days."**

Production asks a different question (`neural_forecast.py:287-289`):

```python
will_become_positive = (last_value < 0 and any(v > 0 for v in forecasted_values))
```

That is a **sign** question. Nothing in training optimizes for it. MSE is
indifferent to sign near zero and penalizes magnitude everywhere, so a model can
post good MAE and poor crossing accuracy — which is exactly what §11 measures.

Two further consequences of MSE on this target:

- MACD is heavily autocorrelated, so "last value plus recent slope" is already
  near-optimal for MSE. That *is* the drift baseline, so MSE pulls the model
  toward it (see the lag ratio in §11).
- MSE on an unpredictable horizon converges to the conditional mean, producing a
  damped, mean-reverting curve. Hence models that predict a turn while the actual
  series keeps trending.

## 7. Data splits

Two independent splits, and they must measure the same axis.

**Test split** — `train_forecast_model.py:254-273`, `--split-strategy` /
`--test-split` (default 0.15):

- `symbol`: hold out whole symbols
- `time`: last fraction of each symbol's series (same symbols, later dates)

**Validation split** — `_split_series`, `validation_split=0.2`, follows
`--split-strategy` so val is a valid early-stopping proxy for test. Temporal
splits use a purge gap of `forecast_horizon`; training windows only touch indices
`< cut`, so any gap ≥ 0 guarantees index disjointness and the horizon-sized gap
adds temporal separation. Falls back to the other strategy with a warning if the
requested one cannot produce a val set.

Both splits happen on the **raw series before windowing**. Splitting after
windowing (and after `prepare_sequences` shuffles) puts near-duplicates of
training windows into validation and makes val loss meaningless.

### Minimum data for a `time` split

With `W = seq_length + forecast_horizon`, `t = test_split`, `N` = trading days
per symbol, both sides must independently satisfy:

```
train side:  floor(N(1-t)) >= W
test  side:  N - floor(N(1-t)) >= W      <-- binding
```

Rule of thumb: **`--days >= 1.45 * W / test_split`** (empirical calendar→trading
ratio is 1.45). Raising `--forecast-horizon` raises the floor linearly.

A `--days 365 --forecast-horizon 10 --test-split 0.15` run produced **zero** test
symbols: 251 points/symbol, `int(251*0.85)=213`, test part `38 < 40`. It reported
this as a normal split and skipped test evaluation silently.

## 8. Residual target (`--residual-target`)

Same inputs, same architecture, same MSE. Only the target changes:

```
standard:  target = y
residual:  target = y - drift(X)
```

where the **drift** baseline is a straight line through the last two input points,
extended forward:

```
drift[k] = macd[t] + (k+1) * delta[t]      # level channel
drift[k] = delta[t]                        # delta channel
```

| symbol | meaning |
|---|---|
| `t` | index of the **last day of the input window** — "now" |
| `macd[t]` | MACD on that day |
| `delta[t]` | `macd[t] - macd[t-1]`, the most recent one-day change |
| `k` | forecast day, **zero-based**: `k=0` is tomorrow |
| `X` | the input window (drift only needs its last two values) |
| `y` | the actual future values |

`(k+1)` rather than `k` because `k=0` is one day ahead and gets one step of change.

Worked on the §2 window (`macd[t]=14.8577`, `delta[t]=5.5574`):

| k | day | drift | actual y | residual `y - drift` |
|---|---|---|---|---|
| 0 | +1 | 20.42 | 19.45 | −0.96 |
| 1 | +2 | 25.97 | 22.41 | −3.57 |
| 2 | +3 | 31.53 | 25.45 | −6.08 |
| 3 | +4 | 37.09 | 27.56 | −9.53 |
| 4 | +5 | 42.64 | 29.38 | −13.26 |

The model is trained to output that last column — "drift will be too high by this
much." Anything it learns is by construction information persistence lacked.
(This window is unusually bad for drift; its sp500 average Day-5 error is 2.61.)

Note residuals grow with horizon. Unweighted long horizons are therefore
dominated by far days — see §12.

**Scaling.** The residual is divided by `std` with **no mean subtraction**: a
residual is already centred near zero, and drift's systematic overshoot is the
bias the model should learn rather than have absorbed into normalization.
Reconstruction at inference:

```
level_raw = drift_raw + residual_norm * std
```

A model emitting exactly zero therefore reproduces drift exactly — the anchor
that makes any deviation a genuine correction. `prepare_sequences(return_drift=True)`
returns `D = (drift_raw - mean)/std` so that `level_norm = residual_norm + D`,
which `evaluate()` uses to score in level space and to score drift on the same
windows.

## 9. Artifacts and inference

```
models/{signal_type}_{architecture}[_with_delta][_residual]_forecaster.pt
models/{signal_type}_{architecture}[_with_delta][_residual]_forecaster.mlpackage
```

Pass the **whole suffixed string** as `--architecture` to
`evaluate_forecast_model.py` (e.g. `bidirectional_gru_with_delta_residual`) — the
path is built from that string.

Core ML export traces the model (`torch.jit.trace`), converts at FP16 with
`compute_units=ct.ComputeUnit.ALL`, and stores everything needed to reproduce
preprocessing in `user_defined_metadata`: `normalization_type`, `mean`, `std`,
`seq_length`, `forecast_horizon`, `include_delta`, `residual_target`,
`hidden_size`, `num_layers`, `batch_size`.

`ComputeUnit.ALL` is a *permission*, not a guarantee — recurrent layers are not
reliably ANE-resident, so the `"inference_engine": "Core ML NPU"` string in the
response (`neural_forecast.py:302`) is hardcoded, not measured.

### Input handling (fixed 2026-08-29)

Both `predict` implementations now **window the input first, then resolve
`prev_value` against the windowed array**:

- **longer than `seq_length`** → `prev_value` is derived internally as
  `seq[-(seq_length+1)]`, and any caller-supplied value is discarded (it referred
  to the untruncated `sequence[0]` and would point at the wrong element)
- **shorter** → pad, log a warning, and set `prev_value = None` because `seq[0]`
  is now synthetic
- **exactly `seq_length`** → both branches skipped, caller's value preserved (this
  is the eval script's path, so its numbers are unchanged)

`forecast_macd` also sizes its fetch off the model rather than trusting
`days_past`, which is calendar days while `seq_length` is trading days:

```python
needed = self.forecaster.seq_length + 6          # +6: prev_value + holiday clustering
calendar_days = max(days_past, int(needed * 1.6) + 1)
```

Widening is free — `predict` truncates to the last `seq_length` points anyway.
`forecast_macd` and `forecast_batch` defaults raised from `days_past=30` to 100.

The windowing block is **duplicated** in `lstm_forecaster.py` and
`neural_forecast.py` (with cross-reference comments) rather than shared, so the
inference-only module does not have to import torch. Same reasoning as the
duplicated drift arithmetic. Keep them in sync.

### Remaining known issue

**Production loads one fixed path.** `get_model_path` defaults
`architecture="stacked_lstm"` and `NeuralForecastService` never passes one, so the
API only ever loads `models/macd_stacked_lstm_forecaster.mlpackage`. Model C's
artifact is `macd_bidirectional_gru_with_delta_residual_forecaster.mlpackage` and
is therefore **invisible to production** — `forecast_macd` silently falls back to
ARIMA. Fix: give `get_model_path` the suffix parameters and source the
architecture from `core/config.py`.

See `../../FORECAST_FIXES_PLAN.md` for status of all fixes.

## 10. Interpreting the metrics

All three compare predictions `p` against actuals `a`, pooled over every
(window, forecast-day) pair.

**MAE — Mean Absolute Error.** Average of `|p - a|`.
- Units: **MACD points**. `MAE=1.407` means "typically off by 1.41 MACD points."
- Errors count proportionally: off by 10 is twice as bad as off by 5.
- **Scale-dependent.** MSFT's MACD ranges 9→30, so 1.41 is modest there; for a $20
  stock whose MACD swings ±0.5 it is catastrophic. Averaged across 500 symbols the
  aggregate is dominated by high-priced names — so MAE is unsafe for cross-symbol
  comparison. Normalize per symbol first if you need that.

**RMSE — Root Mean Squared Error.** `sqrt(mean((p - a)^2))`.
- Same units, always ≥ MAE. Squares first, so **large misses dominate**: off by 10
  is *four* times as bad as off by 5.
- The useful read is the **RMSE/MAE ratio** — error concentration. 1.0 means
  uniform errors; higher means a few big blowups carry the total.

**DA — Directional Accuracy.** How often the predicted *direction* matched reality.

```python
prev_a = last observed value        # the "baseline" for this window
for p, a in zip(predictions, actuals):
    if (p > prev_a) == (a > prev_a): correct += 1
    prev_a = a                      # reference walks onto the ACTUAL each day
```

- 50% is a coin flip. Above = real directional information; **below = systematically
  wrong** (its inverse would be above 50%).
- **Scale-free**, so unlike MAE it is safely comparable across symbols. This is the
  metric `will_become_positive` actually depends on.

**Lag ratio** (`--lag-test`) = `MAE(pred vs actuals shifted back one day) / MAE(pred vs actuals)`.
- `> 1.0` — predictions are farther from a lagged copy than from the truth: the
  model is adding information. **This is the bar.**
- `< 1.0` — predictions track yesterday's value more faithfully than today's
  target. The model is a damped echo.

### Metric traps — read these before trusting a number

1. **`flat`'s MACD DA is degenerate.** `flat` predicts `p = macd[t]`, which on day 1
   *is* the reference, so `p > prev_a` is `p > p` = False always. It can only vote
   "not up." Its 29.84% measures how often MACD fell, not skill. Ignore that cell.
2. **Delta DA is an artifact for days ≥ 2.** The walking reference becomes the
   previous *actual delta*, so predicting zero scores well whenever deltas
   mean-revert. On a synthetic random walk (zero predictability) `flat` scores 66%
   aggregate and 70-75% on days 2-5. **Only Day-1 delta DA is a valid skill
   measure** — there the reference is a fixed 0.0, making it a genuine sign test.
3. **Aggregates hide everything.** `flat` and `drift` have nearly identical
   aggregate delta DA (59.56% vs 59.02%) while their Day-1 values are 47.7% and
   78.4%. Always read the per-day tables.
4. **`trainer.evaluate()` DA is Day-1 only**, measured against the last *input*
   value — not comparable to `evaluate_forecast_model.py`'s multi-day aggregate,
   and it inherits trap #2 on the delta channel. Its MAE/RMSE columns are the
   trustworthy part.
5. **Direction and magnitude trade off.** `drift` has better DA than `flat`
   (54.36% vs 29.84%) but worse MAE (1.465 vs 1.407): it commits to a direction so
   it *can* be right, but overshoots when trends decelerate.
6. **The lag ratio has no measured baseline.** `persistence_baseline.py` does not
   compute lag metrics, so it is unknown what `flat` and `drift` score. For a series
   as autocorrelated as MACD, a ratio below 1.0 may be partly unavoidable at short
   horizons rather than a defect. Treat 0.82x as *uninterpreted* until a baseline
   exists — `_compute_lag_metrics` already exists in the eval script, so adding it
   is small.
7. **`evaluate()`'s `beats drift on both` flag is too blunt.** Model C's run
   reported `False` solely on a DA difference of 78.05% vs 78.22% — 0.17pp on
   10,804 samples, where the standard error is ~0.4pp. That is a tie, while the
   model beat drift by 22% on MAE and 25% on RMSE. Read the MAE/RMSE rows, not
   the flag.
8. **The drift column's Delta DA in `evaluate()` is degenerate.** drift predicts
   `delta[t+k] = delta[t]` exactly, and the trainer measures DA against the last
   observed delta, so `p > prev` is `last_delta > last_delta` = False always —
   the same failure as trap #1. Its 49.32% is not a comparison point.

### Quick reference

| question | metric |
|---|---|
| Does it know which way? | **DA** (Day-1 for the delta channel) |
| How far off, typically? | **MAE** |
| Uniform errors or occasional disasters? | **RMSE/MAE ratio** |
| Is it more than an echo? | **lag ratio > 1.0** |
| Comparing across symbols? | **DA only** |
| Comparing two models? | **per-day**, same horizon, never aggregates |

## 11. Measured results

sp500, 500 symbols, 20 samples/symbol, `seq_length=30`, `forecast_horizon=5`,
measured 2026-08-28. Baselines from `../scripts/persistence_baseline.py`
(no model, no training).

### Baselines

| | MACD MAE | MACD RMSE | MACD DA | DELTA MAE | DELTA DA |
|---|---|---|---|---|---|
| `flat` (`macd[t+k]=macd[t]`) | 1.4068 | 1.8315 | 29.84%* | 0.5316 | 59.56% |
| `drift` (`+ (k+1)·delta[t]`) | 1.4653 | 2.1122 | 54.36% | 0.6078 | 59.02% |

\* degenerate — see trap #1.

### Per-day directional accuracy

| Day | flat MACD | drift MACD | flat DELTA | drift DELTA |
|---|---|---|---|---|
| 1 | 47.7% | **78.4%** | 47.7% | **78.4%** |
| 2 | 21.0% | 58.7% | 61.4% | 51.5% |
| 3 | 23.9% | 48.9% | 62.8% | 54.7% |
| 4 | 27.1% | 44.3% | 63.4% | 55.0% |
| 5 | 29.4% | 41.5% | 62.5% | 55.6% |

### Models measured so far

| | Model A | Model B | **Model C** |
|---|---|---|---|
| architecture | bidirectional_gru | bidirectional_gru | bidirectional_gru |
| hidden / layers | 32 / 2 | 64 / 2 | 32 / 2 |
| normalization | internal | global | global |
| horizon | 10 | 5 | 5 |
| target | level | level | **residual** |
| checkpoint | epoch 100/100 (leaked val) | epoch 3/200 | epoch 5/200 |
| MACD MAE | 1.7947 | 1.1902 | **1.1410** |
| MACD RMSE | 2.3682 | **1.5922** | 1.6054 |
| MACD DA | 47.48% | 44.68% | **50.75%** |
| DELTA MAE / DA | 0.4932 / 64.41% | 0.4544 / 65.39% | 0.4751 / 63.02% |
| lag ratio (MACD) | 0.93x | 0.82x | 0.82x |
| **Day-1 MACD MAE** | 0.5876 | 0.5788 | **0.3849** |
| **Day-1 MACD DA** | 67.3% | 59.9% | **78.4%** |
| Day-1 DELTA DA | 77.9% | 78.2% | 78.5% |

### Per-day MACD MAE — the metric that matters for a forecast curve

| Day | **Model C** | Model B | drift | flat |
|---|---|---|---|---|
| 1 | **0.3849** | 0.5788 | 0.4011 | 0.5563 |
| 2 | **0.7701** | 0.8771 | 0.8940 | 1.0461 |
| 3 | **1.1469** | 1.2115 | 1.4298 | 1.4673 |
| 4 | 1.5137 | **1.5022** | 1.9966 | 1.8217 |
| 5 | 1.8894 | **1.7815** | 2.6050 | 2.1428 |
| **aggregate** | **1.1410** | 1.1902 | 1.4653 | 1.4068 |

### Per-day directional accuracy, Model C vs drift

| Day | Model C MACD | drift MACD | Model C DELTA |
|---|---|---|---|
| 1 | **78.4%** | 78.4% | 78.5% |
| 2 | 52.9% | **58.7%** | 57.1% |
| 3 | 43.1% | **48.9%** | 59.6% |
| 4 | 40.3% | **44.3%** | 60.0% |
| 5 | 39.1% | **41.5%** | 59.9% |

### Verdict

**Model C is the best model measured, and the first to beat every baseline on the
accuracy of the predicted curve.**

- Lowest aggregate MACD MAE of anything tested: **1.1410** — 4.1% better than
  Model B, 19% better than `flat`, 22% better than `drift`.
- Wins days 1, 2 and 3 on MAE; loses day 4 by 0.8% (noise) and day 5 by 6% to
  Model B.
- **The residual target closed the day-1 gap, which is what it was built for.**
  Day-1 MAE went 0.5788 → **0.3849** (33% better) and now beats drift's 0.4011.
  Day-1 DA went 59.9% → **78.4%**, from 18.5 points behind drift to a dead tie.
- Aggregate MACD DA rose 44.68% → **50.75%** — above chance for the first time.

What did not improve:

- **Days 3-5 direction is still below chance** (43.1%, 40.3%, 39.1%), and drift
  still wins days 2-5 on DA. The damping at longer horizons is unchanged.
- **RMSE is marginally worse than Model B** (1.6054 vs 1.5922; ratio 1.41 vs 1.34),
  i.e. slightly more blowup-prone — expected, since anchoring to drift inherits
  some of drift's overshoot.
- **Lag ratio unchanged at 0.82x** — but see trap #6 in §10: this number has no
  measured baseline, so treat it as uninterpreted rather than as a failing grade.

### How to read the curve this model produces

Accurate in magnitude, **sluggish in turns**. Two biases to hold in mind:

1. **It systematically understates trends.** In one MSFT window the actual ran
   +21.0 over seven days while the model predicted +9.5 — 45% of the move — then
   turned down at day 6 while the actual kept climbing. The sub-chance DA at days
   3-5 *is* that damping.
2. **It stays close to a lagged copy of recent values.** Combined with (1), the
   output is conservative and late rather than random.

For a watch signal that is a defensible profile — bounded error, no wild
extrapolations — provided you know it under-reacts. Day 1 is now genuinely good
(0.3849 MAE, 78.4% DA); days 3-5 should be read as "roughly where things are"
rather than as a directional call.

### Context on the 78%

Day-1 sign accuracy near 78% is largely mechanical: MACD is `EMA12 - EMA26`, so
day-over-day sign persistence reflects the smoother's inertia. Model C matching it
means the model has learned to reproduce that inertia *and* beat it on magnitude —
not that it found a market edge. And drift's own MACD DA falls below chance by day
3, so **nothing measured here predicts MACD 3-5 days out directionally.**

## 12. Practical notes

**Device.** Training auto-selects MPS → CUDA → CPU. The ANE is inference-only
(reachable only through Core ML, which has no training API), so it cannot be used
for the backward pass. At 28K-110K parameters and batch 64 these kernels are small
enough that CPU may beat MPS — worth timing with `--epochs 5`.

**Horizon and loss weighting.** MSE averages over all outputs, and per-day error
grows steeply (0.588 → 2.711 across 10 days in Model A). Squared, days 6-10 carry
roughly **77%** of the loss — so an unweighted `--forecast-horizon 10` spends most
of its gradient on horizons where nothing is predictable, and because the
MSE-optimal answer there is the mean, it actively teaches damping. Residual mode
makes this sharper still (residuals grow with horizon). If you want a long horizon,
add per-horizon target standardization or explicit decay weights first.

**Reading the training curve.** With splits done correctly, val loss should sit
*above* train and eventually stop improving. Val below train, declining in
lockstep, is the leakage signature. `train()` reports the winning epoch and warns
when the best epoch is the last one (undertrained).

## 13. Commands

```bash
# Train (from src/)
python scripts/train_forecast_model.py \
  --architecture bidirectional_gru --epochs 200 \
  --normalization-type global --forecast-horizon 5 \
  --split-strategy time --test-split 0.15 \
  --include-delta --residual-target \
  --hidden-size 32 --batch-size 64 --days 730

# Persistence baselines (no model needed — run this first)
python scripts/persistence_baseline.py \
  --watchlist sp500 --samples 20 --seq-length 30 --forecast-horizon 5

# Evaluate — pass the FULL suffixed architecture string
python scripts/evaluate_forecast_model.py \
  --watchlist sp500 --architecture bidirectional_gru_with_delta_residual \
  --samples 20 --lag-test --breakdown-by-day

python scripts/evaluate_forecast_model.py --list-models
```

**Evaluation and the baselines read `stock_cache` directly** — `get_historical_data`
uses `fetch_bulk_from_cache`, so they never trigger a Yahoo Finance fetch and never
write to the DB. Missing data simply yields fewer samples. This means paired
before/after comparisons are safe without freezing the database.

**Training does fetch.** `get_training_data` goes through `get_macd_for_range`,
which fetches and writes missing dates **one symbol at a time** — 500 sequential
yfinance calls whenever the cache is a day stale. Prime it in one bulk call first:

```bash
python -c "
from datetime import timedelta
from macd_utils import get_macd_for_range_bulk, get_latest_market_date
from db_utils import get_watchlist_symbols
end = get_latest_market_date()
get_macd_for_range_bulk(get_watchlist_symbols('sp500'), end - timedelta(days=140), end)
"
```

Note the training example above uses `--residual-target` and `--hidden-size 32`,
which is the Model C configuration. The `--architecture` string passed to the
evaluator (`bidirectional_gru_with_delta_residual`) must carry both suffixes,
because the artifact path is built from it.

## 14. If you continue

Ranked by cost, cheapest first:

1. **Residual target** (implemented, §8) — makes "beat persistence" the literal
   objective. One run tells you whether this architecture adds anything.
2. **Add features.** The model sees only MACD. Close, volume, MA20 and MA50 are
   already in `stock_cache`; cross-sectional/market context could be added. This is
   where the real headroom is — a model reading only a smoothed function of price
   cannot know more than the smoother.
3. **Reframe as classification** on the decision you care about: "does MACD cross
   zero within k days?", cross-entropy, measured class balance, drift-implied rate
   as the baseline. Removes the objective/decision mismatch in §6.

Change **one variable per run** — a four-change run cannot be attributed.
