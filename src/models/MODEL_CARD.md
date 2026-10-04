# MACD Forecaster — Model Card

Reference for the neural MACD forecaster in `lstm_forecaster.py` (training) and
`neural_forecast.py` (inference). Covers what the model consumes, what it
predicts, how to read its metrics, and what it has actually measured.

Last updated 2026-10-04. **Recommended configuration: `gru_v1_residual_macdonly_warm5`**
(residual target, `macd,delta`, plain MSE, `checkpoint_warmup_epochs=5`) — config at
`../configs/gru_v1_residual_macdonly_warm5.json`. A 2026-10-04 multi-seed
re-measurement (§11) found that **no configuration is distinguishable from another on
DA** — seeds of one config spread by 7-10pp — so this is picked for simplicity, not
because it won. Averaging three seeds of it gives the best MAE measured (0.901).
`gru_v1_residual_plainloss_warm5` (the same recipe plus `open-close`) was named best on
2026-09-03 from single runs; that ranking did not survive reseeding.

**For the full investigation log — every experiment, why it was run, and what it
ruled in or out — see `../../docs/FORECAST_MODEL_IMPROVEMENTS.md`.** That file is
the live, append-only record; this card is a snapshot reference and does not try
to duplicate it. Where the two could drift, that file is authoritative for recent
results and this section-by-section reference is authoritative for the mechanics
of the code.

---

## 1. What the model sees

`get_training_data` (`../scripts/train_forecast_model.py`) queries `stock_cache`
per symbol. The primary column is `macd` (or `signal_line` with `--signal-type
signal_line`), plus whatever `--extra-features` / the JSON config's
`extra_features` key names — see `--include-delta` (§3) and the feature list
below. Symbols with fewer than 50 points are skipped.

| Feature | Meaning | Status |
|---|---|---|
| `macd` | primary signal, always present | — |
| `delta` | `macd[t] - macd[t-1]`, via `--include-delta` | recommended on — see §3 |
| `open-close` | `(close - open) / open`, scale-invariant daily return | optional — called a modest win on 2026-09-03 from single runs; the 2026-10-04 seeded runs show no benefit over `macd,delta` alone (§11), though it was only run on one seed there |
| `volume` | 20-day relative volume | tried, flat-to-negative on DA, not currently used |

**Close price (level), MA20 and MA50 are present in `stock_cache` and unused.**
MACD is itself `EMA12(close) - EMA26(close)` — a low-pass filter over price. The
model can only learn the filter's inertia plus whatever autocorrelation price
retains; it cannot recover information the smoother destroyed. `open-close`
partially escapes this ceiling by carrying same-day return, not a MACD-derived
quantity. `docs/FORECAST_MODEL_IMPROVEMENTS.md`'s section B (`macd_signal_dist`,
MA20/MA50 trend ratio, price-vs-MA20 ratio, `chart_patterns`) is the current list
of untried inputs.

## 2. Windowing

Stride-1 sliding window (`lstm_forecaster.py`):

- **input** = `seq_length` consecutive values (default 30 ≈ 6 calendar weeks)
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
new information.** It is a reparameterization that hands the network the first
difference instead of making it learn subtraction, plus a second prediction head.
Useful as an inductive bias; not new data.

**The delta head is the single best-performing part of this model.** Across every
configuration measured, it beats both drift's delta column and a naive
"repeat the last observed delta" baseline on MAE and DA, and holds 60-65% DA flat
across all five forecast days — while MACD's own direction falls below chance by
day 3 in every model tried (§11). If a future product decision has to choose one
output to trust, it should be this one.

`_calculate_deltas` sets element 0 to zero. At training time this is applied to
the full per-symbol series before windowing, so every window but the first gets a
true delta at position 0. At inference the `prev_value` argument reconstructs it —
see §9.

## 4. Normalization

| `--normalization-type` | What the model can see |
|---|---|
| `global` | values standardized by dataset-wide mean/std (fit on **train only**) — the model *can* distinguish MACD at +30 from MACD near 0 |
| `internal` | each window standardized by **its own** mean/std — the model sees only the *shape* of the last 30 days and cannot tell where the level is |

`internal` is a poor fit here: the model needs to condition on level to have any
chance of a correct sign, and internal normalization deletes exactly that
information from its input. The level is still restored on denormalization, so
outputs are in correct units — but the model cannot condition on it during
inference. Every config in current use is `global`.

Note: with `internal`, `self.mean`/`self.std` are never assigned and the saved
metadata contains the `[0.0, 0.0]` / `[1.0, 1.0]` initializers. Harmless while
`normalization_type` is read correctly (the internal path recomputes per window),
but it is a landmine — `_load_model` defaults the key to `"global"` when absent.

## 5. Architecture and shapes

Five architectures, selected by `--architecture`, all the same shape: recurrent
encoder → take a final hidden state → 2-layer MLP head, optionally plus a second
head for the auxiliary directional loss.

`stacked_lstm` (2-layer LSTM), `bidirectional_gru`, `stacked_gru`,
`standard_lstm` (1 layer), `gru` (1 layer). Factory: `create_model()`.
`bidirectional_gru` is the architecture used by every model in §11.

`bidirectional_gru` combines the two directions correctly: forward state at the
**last** timestep (has read `x[0..T-1]`) concatenated with backward state at the
**first** timestep (has read `x[T-1..0]`). Both are full-sequence summaries. There
is **no future leakage** — the input window is entirely historical and fully
observed at inference too.

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

With `hidden=32` the same configuration is **28,138** parameters — every model in
§11 uses `hidden=32`. Adding an `open-close` column raises `input_size` to 3,
which only affects layer 1's `3*hidden*input_size` term (+192 params at
`hidden=32`); everything downstream is unchanged.

Watch the **windows-per-parameter ratio** — but note it was not the binding
constraint here:

| run | params | train windows | ratio | best epoch |
|---|---|---|---|---|
| Model B (`hidden=64`) | 109,514 | 109,855 | 1.0 | 3 / 200 |
| Model C (`hidden=32`) | 28,138 | 106,114 | 3.8 | 5 / 200 |
| `plainloss_warm5` (`hidden=32`, 3 features) | ~28,330 | 122,680 | 4.3 | 9 / 40 |

Quartering capacity moved the overfitting point from epoch 3 to epoch 5; adding
`open-close` and 500k more training windows did not move it further — every model
measured so far peaks inside the first 10 epochs regardless of capacity or data
volume (§7). The learnable signal is genuinely thin; more capacity is not the
lever, more *information* is (§14).

### Optional: auxiliary directional head (`--auxiliary-direction-lambda`)

A second `Linear(feature_dim, forecast_horizon)` head off the same shared
representation, trained with binary cross-entropy on `sign(future delta)` per
forecast day, added to the MSE loss as `mse + lambda * bce`. It was described on
2026-09-03 as the one loss change that reliably helped, at `lambda=0.3` tuned under
a since-removed `loss_decay_gamma` handicap. **Re-tested 2026-10-04 on the current
recipe with three seeds, `lambda=0.3` showed no benefit**: 46.6% mean MACD DA against
51.4% without it, and ~2% worse MAE (§11). No current config should enable it; other
`lambda` values are untested. At inference the head is stripped before Core ML export
(`_ForecastOnly` wrapper), so it costs nothing in production.

### Shelved: `predict_deltas_only` (A2)

An architectural variant that predicts only the delta column and reconstructs
MACD as `anchor + cumsum(deltas)`, forcing output consistency by construction.
Ablated and found to *worsen* DA despite improving MAE — the compounding error
this was meant to fix was never primarily an inter-head-inconsistency problem.
The code remains (`network_target_size`, `_reconstruct_deltas_only`) but no
current config uses it.

## 6. Objective, and what the loss actually optimizes

```python
self.criterion = nn.MSELoss()
loss = self.criterion(predictions, batch_y) [+ lambda * bce_loss]  # optional A1 term
```

Optimizer is Adam (`--learning-rate`, default 1e-3). The base loss is mean
squared error over all `forecast_horizon * target_size` outputs, optionally
combined with the auxiliary directional BCE term above.

**The product objective is the quality of the 5-day forecast curve — MACD
directional accuracy (DA) and MAE, evaluated with `evaluate_forecast_model.py`.**
(`will_become_positive`, computed in `neural_forecast.py` as `last_value < 0 and
any(v > 0 for v in forecasted_values)`, is a minor front-end convenience flag —
**explicitly not a target to optimize for**, per project direction. An earlier
draft of this card suggested reframing training as classification on that flag;
that direction has been ruled out and should not be revisited without new
instruction.)

**The central empirical finding of the whole investigation: MSE-side
improvements and DA have moved in opposite directions in essentially every
experiment run**, not just as a one-off tension:

- Loss decay weighting (`loss_decay_gamma`, tried as a per-day discount and as
  independent per-column decay) **worsened DA at every setting tried** — removing
  it recovered 2-4pp of DA at exactly the horizon days it had discounted. It
  should be left `null`.
- Residual vs. level targets moved MAE and long-horizon DA by the *largest*
  margins measured, in opposite directions: residual mode cut MAE ~17% and
  recovered day-1 DA to match drift, but made days 3-5 DA the worst on record at
  the time.
- The same tension recurs **inside checkpoint selection**, even under plain MSE
  with no decay weighting: the epoch with the lowest validation loss is not
  reliably the epoch with the best long-horizon DA (§7).

Two things follow. First, `loss_decay_gamma` should stay off. Second, the
auxiliary directional BCE head (§5) is the only lever tried so far that pushes on
DA directly rather than hoping it falls out of a magnitude objective. It was
recorded here as a modest win, but a three-seed re-test on the current recipe found
none (§5, §11) — as of 2026-10-04 no loss-side change has a confirmed DA benefit.

MSE also has two structural properties worth keeping in mind independent of any
of the above:

- MACD is heavily autocorrelated, so "last value plus recent slope" (drift) is
  already close to MSE-optimal, which is why it is such a strong baseline (§11).
- MSE on an unpredictable horizon converges toward the conditional mean —
  producing damped, mean-reverting predictions when the actual series keeps
  trending. This is *not* the same failure as the day-3+ DA collapse (a separate
  regression-dilution effect — the model's predicted variance is well-calibrated,
  but its correlation with the actual move collapses; see
  `docs/FORECAST_MODEL_IMPROVEMENTS.md`'s 2026-09-03 dispersion analysis for the
  measurement), but it is the mechanism behind drift's own over-extrapolation at
  long horizons.

## 7. Data splits, training loop, and checkpoint selection

Two independent splits, and they must measure the same axis.

**Test split** — `train_forecast_model.py`, `--split-strategy` / `--test-split`
(default 0.15):

- `symbol`: hold out whole symbols
- `time`: last fraction of each symbol's series (same symbols, later dates) —
  used by every current config

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

### `checkpoint_warmup_epochs` and `ReduceLROnPlateau`

Two mechanisms sit around the training loop:

- `--lr-scheduler` enables `ReduceLROnPlateau` (factor/patience/min configurable)
  on validation loss.
- `checkpoint_warmup_epochs` makes epochs before that count ineligible to be
  saved as "best," to stop the selector locking onto a spuriously low early-epoch
  val loss before the model has actually learned anything.

**Every architecture/feature combination measured so far converges within the
first 5-10 epochs and overfits steadily after.** Model B peaked at epoch 3/200,
Model C at 5/200, and the current best models at 6/40 and 9/40. A
`checkpoint_warmup_epochs` set higher than that — 30 was used through most of
this investigation — can and did **discard the true optimum**: two runs found
their lowest validation loss inside the warmup window and shipped a checkpoint
3.8% and 8.2% worse on MAE than what training had actually found. Lowering the
warmup to 5 fixed this. **Given how early the optimum sits, `--epochs 40-50` with
`checkpoint_warmup_epochs` around 5 is now the practical range** — the 200-300
epoch runs earlier in this investigation spent the large majority of their time
overfitting.

**Caution, not yet resolved: the lowest-val-loss checkpoint is not reliably the
best-DA checkpoint, even under plain MSE.** One run's warmup fix picked a lower
validation loss than its own predecessor and improved MAE and day-1 DA, but
landed on the *worst* days-4-5 DA recorded for that configuration. This is the
§6 MSE-vs-DA tension recurring one level below the loss function, in which epoch
gets selected as "best" at all. The current mitigation is to watchlist-evaluate
the model, not trust val loss alone; a systematic fix (evaluating several early
checkpoints on DA rather than saving only the single lowest-val-loss one) has not
been built. See `docs/FORECAST_MODEL_IMPROVEMENTS.md`'s "Suggested order" for the
proposal.

### A `--days 365 --forecast-horizon 10 --test-split 0.15` run produced **zero** test symbols

251 points/symbol, `int(251*0.85)=213`, test part `38 < 40`. It reported this as a
normal split and skipped test evaluation silently — a reminder to check the
"Total data points" / symbol counts printed at the start of training.

## 8. Residual target (`residual_target: true`) — the standard configuration

Same inputs, same architecture, same loss. Only the target changes:

```
level:     target = y
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

**This is now the standard configuration, not an experimental option.** Every
current best model uses `residual_target: true`; it reliably recovers day-1 DA to
match drift and beats drift on MAE at every horizon day. Its remaining weakness —
days 3-5 DA still trails drift — is the open problem tracked in
`docs/FORECAST_MODEL_IMPROVEMENTS.md`.

## 9. Artifacts, naming, and inference

**Two naming schemes exist. The app serves config-based models through seed
ensembles; the legacy scheme is only reachable through `NeuralForecastService`'s
default path, which nothing in the UI calls.**

**Config-based (current, recommended)** — `train_forecast_model.py --config
path/to/config.json`. `model_name` comes from the JSON config
(`src/configs/*.json`, see `../scripts/config_utils.py`); the version is resolved
once per run via `get_latest_model_version(models_dir, model_name) + 1`, so the
`.pt` and `.mlpackage` from one run always share a version number:

```
models/{model_name}_{version}.pt
models/{model_name}_{version}.mlpackage
```

`evaluate_forecast_model.py --config path/to/config.json` (or `--model-name` /
`--model-version` directly) resolves the same way, defaulting to the latest
version when none is given, and prints the resolution
(`Resolved model_name='X' (latest version) -> X_N.mlpackage`).

**Legacy (CLI flags only, no `--config`)** — auto-built suffix name:

```
models/{signal_type}_{architecture}[_with_delta][_residual][_with_{extra_features}]_forecaster.pt
models/{signal_type}_{architecture}[_with_delta][_residual][_with_{extra_features}]_forecaster.mlpackage
```

Pass the **whole suffixed string** as `--architecture` to
`evaluate_forecast_model.py` to load one of these (e.g.
`bidirectional_gru_with_delta_residual`).

**Production serves seed ensembles (since 2026-10-04).** Until then the "Show
Bullish Forecast" button called ARIMA directly and no config-based model was
reachable from the app — the gap this section used to flag as the most
consequential in the pipeline. Now:

- `../configs/ensembles.json` lists the selectable ensembles. Each pins a config
  and exact versions (seeds 42 / 7 / 123); the default is
  `gru_v1_residual_macdonly_warm5` v1,2,3. Versions are pinned rather than "latest
  three" because stray runs share a model name.
- `POST /forecast/macd/ensemble` (`../services/ensemble_service.py`) averages the
  members' forecasts with `EnsembleForecaster` (`neural_forecast.py`) and returns
  the ARIMA endpoint's shape. It reproduces `evaluate_ensemble.py`'s predictions
  exactly for the same window, and forecasts all 501 sp500 symbols in about 4 s.
  It writes `will_become_positive` to `stock_cache` the way the ARIMA path does,
  so the combined forecast reflects whichever engine ran last.
- `POST /forecast/ensembles/{id}/retrain` (`../services/ensemble_training.py`)
  retrains an ensemble in the background: bulk data refresh, three seeds in
  parallel on CPU with reserved `--model-version` numbers, evaluation, then the
  served versions switch — only if every seed produced a Core ML model. What a
  retrain changes is recorded in `models/ensemble_state.json`, which overrides the
  registry's versions; old versions stay on disk.
- ARIMA remains selectable in the UI as a legacy option. §11 measured it as worse
  than both the ensemble and drift.

`NeuralForecastService()` with no arguments still resolves to the legacy
`models/macd_stacked_lstm_forecaster.mlpackage` and falls back to ARIMA when that
file is absent; `ForecastService` (`../services/forecast_service.py`) still uses
it, but no UI path goes through either.

**Every forecast run is stored** in the `forecast_predictions` table (one row per
model, symbol, market date and forecast day; `model` is an ensemble id or `arima`),
with the model versions and training date that produced it. A later run of the same
model from the same market date replaces the earlier one. This is the record to score
a frozen model against as time passes — see §13.

**Training date.** Model files carry none. For an ensemble retrained through the app
it is recorded in `models/ensemble_state.json`; otherwise it is read from the newest
member's `.pt` file time. The `.mlpackage`'s own time is "last served", not
"trained": loading a Core ML model rewrites its `Manifest.json`.

**A retrain on unchanged data changes nothing.** The 2026-10-04 end-to-end test
retrained `hidden16` (seeds 42 / 7 / 123, CPU, 11m41s) and the new versions 4-6
scored exactly what versions 1-3 had (MAE 0.905742, DA 51.48%): seeded CPU training
is reproducible. Retraining is only worth doing once new trading days are in the
database — and then the new models are different models, so score a frozen ensemble
by its pinned version numbers.

Design and decisions: `../../docs/ENSEMBLE_PRODUCTION_PLAN.md`.

Core ML export traces the model (`torch.jit.trace`), converts at FP16 with
`compute_units=ct.ComputeUnit.ALL`, and stores everything needed to reproduce
preprocessing in `user_defined_metadata`: `normalization_type`, `mean`, `std`,
`seq_length`, `forecast_horizon`, `include_delta`, `residual_target`,
`predict_deltas_only`, `hidden_size`, `num_layers`, `batch_size`.

`ComputeUnit.ALL` is a *permission*, not a guarantee — recurrent layers are not
reliably ANE-resident, so the `"inference_engine": "Core ML NPU"` string in the
response is hardcoded, not measured.

### Input handling

Both `predict` implementations window the input first, then resolve `prev_value`
against the windowed array:

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

The windowing block is **duplicated** in `lstm_forecaster.py` and
`neural_forecast.py` (with cross-reference comments) rather than shared, so the
inference-only module does not have to import torch. Same reasoning as the
duplicated drift arithmetic. Keep them in sync.

### `--rollout` (recursive multi-step inference, `evaluate_forecast_model.py`)

An alternative inference mode for evaluation only: instead of one direct 5-step
prediction, run the model `forecast_horizon` times, feeding each run's day-1
output back into the window (`_rollout_predict`). Motivated by the day-1 head
being far stronger than the direct multi-day head. **Measured and falsified,
repeatedly** — on four separate models (level-target and residual-target,
macd-only and with `open-close`), rollout reproduces the direct head's per-day DA
to within ~1pp while making MAE 5-15% worse from compounding error. The day-3+
collapse is present in the one-step prediction itself, not introduced by how
multi-day predictions are assembled. The flag remains in the codebase (tested,
`--rollout-fill`, `--rollout-delta`) but no config should use it for a shipping
decision — see `docs/FORECAST_MODEL_IMPROVEMENTS.md`'s A4 follow-ups for the full
numbers.

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
- **Scale-free**, so unlike MAE it is safely comparable across symbols.

**Lag ratio** (`--lag-test`) = `MAE(pred vs actuals shifted back one day) / MAE(pred vs actuals)`.
- `> 1.0` — predictions are farther from a lagged copy than from the truth: the
  model is adding information. **This is the bar.**
- `< 1.0` — predictions track yesterday's value more faithfully than today's
  target. The model is a damped echo.

(This card's formula and sign convention have always been correct. A different
document, `docs/FORECAST_MODEL_IMPROVEMENTS.md`, initially read this ratio
backwards in several 2026-09-01 entries and carries a standing correction —
consult it, not this card, if you see a LAG claim there that looks inverted.)

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
   it *can* be right, but overshoots when trends decelerate. This turned out to be
   a special case of a much more general pattern — see §6.
6. **The lag ratio now has a measured baseline, and it needs a caveat.**
   `persistence_baseline.py` / a throwaway audit script (2026-09-03, sp500, 500
   symbols, `evaluate_forecast_model.py`'s exact metric functions) measured
   `flat`/persistence at **0.69x**, `drift` at **0.95x**, and repeat-last-delta at
   **0.95x** — i.e. drift itself is close to a lag copy, which is expected given
   how autocorrelated MACD is. Trained models measured under the same protocol on
   2026-09-03 scored **3.0-4.2x** on the MACD channel, comfortably clearing that
   bar. But **Model C's originally-reported ratio (§11) was 0.82x** — below the
   drift baseline measured later, and roughly 4-5x lower than the later models'
   ratios despite similar MAE/DA. This gap has not been root-caused (possible
   protocol difference between when Model C was measured and now, rather than a
   genuine 4x change in lag-like behavior). **Do not directly compare Model C's
   0.82x against any ratio measured after 2026-09-01 without re-measuring Model C
   under the current script first.**
7. **`evaluate()`'s `beats drift on both` flag is too blunt.** Model C's run
   reported `False` solely on a DA difference of 78.05% vs 78.22% — 0.17pp on
   10,804 samples, where the standard error is ~0.4pp. That is a tie, while the
   model beat drift by 22% on MAE and 25% on RMSE. Read the MAE/RMSE rows, not
   the flag.
8. **The drift column's Delta DA in `evaluate()` is degenerate.** drift predicts
   `delta[t+k] = delta[t]` exactly, and the trainer measures DA against the last
   observed delta, so `p > prev` is `last_delta > last_delta` = False always —
   the same failure as trap #1. Its 49.32% is not a comparison point.
9. **`will_become_positive` (the zero-crossing flag) is a different metric from
   DA, and much rarer.** It fires on a base rate of ~16% of currently-negative
   windows. "Never fire" scores 84% raw accuracy against a trained model's ~85%,
   so accuracy is nearly uninformative for it — use precision/recall/F1 if this
   flag is ever evaluated again, and remember it is **not the training or
   evaluation objective** (§6).

### Quick reference

| question | metric |
|---|---|
| Does it know which way? | **DA** (Day-1 for the delta channel) |
| How far off, typically? | **MAE** |
| Uniform errors or occasional disasters? | **RMSE/MAE ratio** |
| Is it more than an echo? | **lag ratio > 1.0**, compared against drift's own ~0.95x, not just against 1.0 |
| Comparing across symbols? | **DA only** |
| Comparing two models? | **per-day**, same horizon, never aggregates |

## 11. Measured results

sp500, 500 symbols, 20 samples/symbol, `seq_length=30`, `forecast_horizon=5`.
Baselines from `../scripts/persistence_baseline.py` / re-measured 2026-09-03
through `evaluate_forecast_model.py`'s exact metric functions.

### Baselines (naive, no model, no training)

| | MACD MAE | MACD DA | MACD LAG | DELTA MAE | DELTA DA | DELTA LAG |
|---|---|---|---|---|---|---|
| `flat` / persistence (`macd[t+k]=macd[t]`) | 1.313-1.407 | 29.8-30.1%* | 0.69x | 0.53 | 59.6% | — |
| **`drift`** (`+ (k+1)·delta[t]`) | 1.369-1.465 | **54.4-55.2%** | 0.95x | 0.56-0.61 | 59.0% | 0.95x |

\* degenerate on day 1 — see trap #1. Small MAE/DA ranges above reflect two
separate measurement passes (2026-08-28 and 2026-09-03); treat as the same
baseline, not two different ones.

### Per-day directional accuracy (drift and flat)

| Day | flat MACD | drift MACD | flat DELTA | drift DELTA |
|---|---|---|---|---|
| 1 | 47.7% | **78.4-78.6%** | 47.7% | **78.4%** |
| 2 | 21.0% | 58.7-60.4% | 61.4% | 51.5% |
| 3 | 23.9% | 48.9-50.5% | 62.8% | 54.7% |
| 4 | 27.1% | 44.3-44.8% | 63.4% | 55.0% |
| 5 | 29.4% | 41.5-41.9% | 62.5% | 55.6% |

**Drift has not been beaten on aggregate DA by any trained model.** It remains
the strongest baseline in this entire investigation on that metric.

### Model lineage

| | Model C (08-28) | `plain_macdonly` (09-03) | **`plainloss_warm5` (09-03, best at the time)** |
|---|---|---|---|
| Features | macd, delta | macd, delta | macd, delta, open-close |
| Target | residual | residual | residual |
| Loss | plain MSE | plain MSE | plain MSE |
| `checkpoint_warmup_epochs` | 5 (200 total) | 30 (100 total) | **5 (40 total)** |
| Best epoch | 5/200 | 49/100 | **9/40** |
| MACD MAE | 1.1410 | 1.0707 | **1.0356** |
| MACD DA | 50.75% | 50.21% | **51.52%** |
| Day 1 / 2 / 3 / 4 / 5 DA | 78.4 / 52.9 / 43.1 / 40.3 / 39.1 | 78.5 / 54.4 / 42.5 / 38.5 / 37.2 | **78.4 / 55.2 / 45.3 / 41.2 / 37.5** |
| Delta MAE / DA | 0.4751 / 63.02% | 0.4463 / 64.92% | 0.4316 / 64.04% |

The path from Model C to the current best ran through several dead ends worth
knowing about even though none of them are the shipping configuration:
`loss_decay_gamma` (a per-day loss discount) turned out to be a regression —
turning it off is most of the gap between Model C and `plain_macdonly`;
`checkpoint_warmup_epochs=30` discarded the true optimum in two separate
training runs, worth 4-8% MAE, before being lowered to 5. Full detail, including
the falsified A2/A3/A4 candidates and the checkpoint-selection nuance, is in
`docs/FORECAST_MODEL_IMPROVEMENTS.md`.

Everything above this line was measured on or before 2026-09-03 from **single,
unseeded runs**. The next subsection shows why that matters.

### 2026-10-04 re-measurement (multi-seed)

Fresh database, sp500 (501 symbols), data through 2026-10-02, `--days 600`, 20
samples/symbol, Core ML inference. All models retrained on CPU (`--device cpu`,
torch 2.14.1, Python 3.13) with explicit seeds. Raw outputs are the
`eval_*_2026-10-0{3,4}.txt` files in the repo root. Numbers are **not** comparable
to the tables above — different data window — only to each other.

**Baselines on this window**

| | MACD MAE | MACD DA | Day 1 / 2 / 3 / 4 / 5 DA | Day 1 / 2 / 3 / 4 / 5 MAE |
|---|---|---|---|---|
| `flat` | 1.2565 | 30.84%* | — | 0.480 / 0.904 / 1.291 / 1.641 / 1.967 |
| **`drift`** | 1.1466 | **57.81%** | 80.8 / 62.6 / 52.1 / 47.7 / 45.7 | 0.311 / 0.684 / 1.117 / 1.576 / 2.044 |

\* degenerate, see trap #1.

**Seed variance is the headline finding.** Three configs were each trained on seeds
42, 7 and 123:

| Config | DA seed 42 / 7 / 123 | Mean DA | MAE seed 42 / 7 / 123 | Mean MAE |
|---|---|---|---|---|
| `gru_v1_residual_macdonly_warm5` | 56.65 / 50.86 / 46.63 | 51.4% | 0.936 / 0.928 / 0.940 | 0.935 |
| `gru_v1_exp_hidden16` | 55.28 / 47.97 / 49.78 | 51.0% | 0.925 / 0.922 / 0.924 | 0.924 |
| `gru_v1_residual_seq60_d750` | 52.68 / 47.46 / 48.59 | 49.6% | 0.901 / 0.930 / 0.942 | 0.925 |

- **Aggregate MACD DA spans 5-10pp between seeds of one config**, almost all of it
  on days 3-5 (`macdonly_warm5` day 5: 46.2 / 37.6 / 31.2). The noise floor quoted
  in `../configs/experiments/README.md` (sd 0.77pp, range 1.71pp) badly understates
  this. **A single run cannot rank two configs on DA.**
- MAE is steadier but not fixed: range 0.003 for `hidden16`, 0.041 for `seq60_d750`.
- Seed 42 was the best DA seed in all three configs, which is why single-seed
  tables at that seed looked more promising than they were.
- Day-1 DA is the stable part: `macdonly_warm5` scored 80.9-81.2% on every seed,
  level with drift (80.8%). Two other runs had an unexplained low day 1
  (`hidden16` seed 7: 73.2%; `seq60_d750` seed 123: 76.3%).

**Single-seed results (seed 42) — read with the spread above in mind**

| Config | Change from `plainloss_warm5` | MACD DA | MACD MAE |
|---|---|---|---|
| `gru_v1_exp_baseline` | none (adds `open-close` vs `macdonly`) | 49.82% | 0.941 |
| `gru_v1_exp_lr_patience2` | `lr_patience` 2 | identical to baseline — same checkpoint, the scheduler fires after the best epoch | |
| `gru_v1_exp_lr_3e4` | `learning_rate` 3e-4 | 48.42% | 0.901 |
| `gru_v1_exp_lr_1e4` | `learning_rate` 1e-4 | 47.46% | 0.920 |
| `gru_v1_exp_hidden16_lr3e4` | hidden 16, lr 3e-4 | 49.29% | 0.900 |
| `gru_v1_exp_hidden8` | hidden 8 | 44.19% | 0.982 |
| `gru_v1_residual_seq15` | `seq_length` 15 | 46.64% | 0.964 |
| `gru_v1_residual_seq20` | `seq_length` 20 | 48.46% | 0.947 |
| `gru_v1_residual_seq45` | `seq_length` 45 | 49.09% | 0.969 |

None of these is outside the seed spread on DA. The ~0.90 MAE of the lower-LR arms
is unconfirmed (one seed; `seq60_d750`'s 0.901 did not repeat).

**Shorter horizons** (seed 42; compare per day against drift at the same horizon,
never on the aggregate):

| | Day 1 DA / MAE | Day 2 DA / MAE | Day 3 DA / MAE |
|---|---|---|---|
| `gru_v1_residual_h2_w0` (= `plainloss_warm5_h2`, same checkpoint) | 80.6% / 0.283 | 61.3% / 0.594 | — |
| drift, horizon 2 | 80.4% / 0.309 | 62.9% / 0.683 | — |
| `gru_v1_residual_h3` | 79.3% / 0.281 | 57.9% / 0.575 | 42.7% / 0.885 |
| drift, horizon 3 | 80.8% / 0.306 | 63.4% / 0.679 | 52.7% / 1.118 |

About 5% better day-1/2 MAE than the 5-day models; direction no better than drift.

**Auxiliary direction loss** (`gru_v1_residual_macdonly_warm5_aux03`: the
`macdonly_warm5` recipe plus `auxiliary_direction_lambda: 0.3`, same three seeds):

| | Seed 42 | Seed 7 | Seed 123 | Mean |
|---|---|---|---|---|
| MACD DA with aux / without | 45.27 / 56.65 | 46.33 / 50.86 | 48.18 / 46.63 | 46.6% / 51.4% |
| MACD MAE with aux / without | 0.963 / 0.936 | 0.977 / 0.928 | 0.929 / 0.940 | 0.956 / 0.935 |

Lower DA on every forecast day on average (78.2 / 50.5 / 36.8 / 34.3 / 33.3 vs
81.0 / 56.1 / 42.1 / 39.3 / 38.3). Not proof it hurts at n=3, but no sign it helps.

**Seed ensembles** (`../scripts/evaluate_ensemble.py`: mean of the three seeds'
forecasts per window, scored by the stock evaluation code):

| Family | Ensemble MAE | Members' mean (best) MAE | Ensemble DA | Members' mean DA |
|---|---|---|---|---|
| `macdonly_warm5` v1,2,3 | **0.901** | 0.935 (0.928) | 51.28% | 51.38% |
| `hidden16` v1,2,3 | 0.906 | 0.924 (0.922) | 51.48% | 51.01% |
| `seq60_d750` v2,3,4 | 0.906 | 0.925 (0.901) | 49.41% | 49.58% |

Averaging buys 2-4% MAE and removes the dependence on a lucky seed. It does
**nothing** for DA — the ensemble lands on the members' average, so the seeds are
not cancelling each other's day 3-5 errors; the weak direction is a shared bias,
not seed noise.

**ARIMA comparison** (`evaluate_forecast_model.py --compare` on
`gru_v1_residual_macdonly_warm5` v2, the seed-7 model closest to the family mean;
ARIMA fitted on 501/501 symbols; `eval_arima_compare_2026-10-04.txt`):

| | Mean MAE | Median MAE | Median WAPE | MACD DA | Day 1 / 2 / 3 / 4 / 5 DA |
|---|---|---|---|---|---|
| Neural, 3-seed ensemble | **0.901** | **0.452** | **21.6%** | 51.28% | 81.0 / 56.4 / 42.1 / 38.9 / 38.0 |
| Neural, v2 | 0.928 | 0.456 | 22.1% | 50.86% | 80.9 / 55.4 / 41.8 / 38.6 / 37.6 |
| ARIMA | 1.682 | 0.577 | 27.2% | 46.41% | 77.7 / 48.9 / 36.9 / 34.8 / 33.7 |
| `drift` | 1.147 | — | — | **57.81%** | 80.8 / 62.6 / 52.1 / 47.7 / 45.7 |

- **ARIMA is the weakest of the four.** On the typical symbol the neural model's MAE
  is ~21% lower (median 0.456 vs 0.577). ARIMA's mean MAE is 2.9x its median — a
  minority of divergent fits dominate it — so compare medians (its per-day mean MAE,
  0.891 on day 1 rising to 2.503 on day 5, is inflated the same way).
- ARIMA's DA is 3-6.5pp below the neural model on every day and 3-15pp below drift.
  Its 46.41% aggregate is level with the worst neural seed measured (46.63%).
- `persistence_baseline.py` prints no medians, so drift has none here.
- **Production serves ARIMA today** (§9: it cannot load any config-based model), i.e.
  the weakest option measured — the neural model is better on magnitude and plain
  drift is better on direction.

### Verdict

**As of 2026-10-04 no configuration is measurably better than another.** Hidden
size, learning rate, scheduler patience, input window, the `open-close` feature and
the auxiliary direction loss all land at 46-51% aggregate MACD DA on a three-seed
mean, or inside the seed spread where only one seed was run, with MAE 0.90-0.97.
The one arm that looks worse on both measures is `hidden_size` 8 (one seed).
`gru_v1_residual_macdonly_warm5` is recommended as the simplest of them (two
inputs) with the best three-seed mean by a small margin; its three-seed ensemble
has the best MAE measured (0.901, 21% below drift). ARIMA, the production
fallback, is behind it on both MAE and DA.

The 2026-09-03 verdict — `gru_v1_residual_plainloss_warm5` best on DA (51.52%) and
MAE (1.0356) — came from single unseeded runs on an earlier window and should be
read as one draw from the spread documented above, not as a ranking.

What has not improved, and is the open problem:

- **Drift still leads aggregate DA by about 6pp** (57.81% vs ~51% three-seed means)
  and on every individual day 2-5; models match it only on day 1. No trained model
  or ensemble has beaten it on aggregate DA. The models' reliable advantage is
  magnitude: MAE ~18-21% below drift.
- **Days 3-5 direction remains the weakest part of every model measured.** A
  2026-09-03 dispersion analysis found the model's predicted variance at these
  horizons is well-calibrated to the actual variance (ratio ≈1.0-1.1), but its
  *correlation* with the actual move collapses (r ≈0.5 at day 1 → r ≈0.2 at day
  5) — regression dilution, not magnitude damping. See
  `docs/FORECAST_MODEL_IMPROVEMENTS.md` for the measurement and what was tried
  in response.

### Context on the ~78% day-1 figure

Day-1 sign accuracy near 78% is largely mechanical: MACD is `EMA12 - EMA26`, so
day-over-day sign persistence reflects the smoother's inertia. Every current
model matching that number means it has learned to reproduce that inertia *and*
beat it on magnitude — not that it found a market edge. Drift's own MACD DA falls
below chance by day 3, so directional predictability past day 2 remains
genuinely scarce in this feature set, not merely unmodeled.

## 12. Practical notes

**Device.** Training auto-selects MPS → CUDA → CPU. The ANE is inference-only
(reachable only through Core ML, which has no training API), so it cannot be used
for the backward pass. Pass `--device {mps,cuda,cpu}` (or `"device"` in the config)
to override the auto-selection.

**At this model size CPU beats MPS.** Measured 2026-10-04 on an Apple M5 (torch
2.14.1), bidirectional GRU, hidden 32, batch 64, synthetic data: 4.7 s/epoch on CPU
against 17.2 s/epoch on MPS — about 3.6x. GPU utilization sits near 30% on MPS
because each batch is a few tiny kernels separated by Python overhead. A real
40-epoch sp500 run takes about 9 minutes on CPU. CPU and MPS runs differ in the
third decimal of the loss, so do not mix devices inside one comparison.

**Run several trainings in parallel on CPU.** One thread per process is as fast as
the default (4.6 vs 4.8 s/epoch), and four at once on the four performance cores
cost only ~5.3 s/epoch each — about 3.5x the throughput. Set
`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`; without them four parallel runs slow
to ~7.0 s/epoch each. Runs of the **same** `model_name` resolve their version number
just before saving, so stagger or serialize those; different model names are safe.

**Horizon and loss weighting — do not add per-day decay weighting.** An earlier
version of this note recommended per-horizon decay weights as a mitigation for
unweighted long-horizon loss dominating gradient. That was tried
(`loss_decay_gamma`, both as a uniform per-day discount and as independent
per-column decay) and **measured to make DA worse at every setting tried, including
mild ones** — see §6. If a genuinely long horizon (`--forecast-horizon` well past
5) is attempted, per-horizon *target standardization* (normalizing each day's
target scale independently) is a less-tested and more promising alternative to
loss re-weighting; explicit decay weights are not recommended based on current
evidence.

**Reading the training curve.** With splits done correctly, val loss should sit
*above* train and eventually stop improving. Val below train, declining in
lockstep, is the leakage signature. `train()` reports the winning epoch and warns
when the best epoch is the last one (undertrained). Given §7's finding, also
**watchlist-evaluate the selected checkpoint** rather than trusting that the
lowest val loss is the best model for DA.

## 13. Commands

```bash
# Train the recommended configuration (from src/). Always seed, and train at least
# three seeds before comparing configs — see §11's seed-variance finding.
for s in 42 7 123; do
  python scripts/train_forecast_model.py --config configs/gru_v1_residual_macdonly_warm5.json \
    --seed $s --device cpu
done

# Evaluate one version (latest if --model-version is omitted)
python scripts/evaluate_forecast_model.py --config configs/gru_v1_residual_macdonly_warm5.json \
  --model-version 1 --watchlist sp500 --samples 20 --lag-test --breakdown-by-day

# Evaluate the average of several versions (seed ensemble)
python scripts/evaluate_ensemble.py --model-name gru_v1_residual_macdonly_warm5 \
  --versions 1,2,3 --watchlist sp500 --samples 20 --lag-test --breakdown-by-day

# Persistence/drift baselines (no model needed — run this first for any new comparison)
python scripts/persistence_baseline.py \
  --watchlist sp500 --samples 20 --seq-length 30 --forecast-horizon 5

# List all trained models on disk
python scripts/evaluate_forecast_model.py --list-models
```

**Checking a frozen ensemble as time passes.** Two ways, both after refreshing the
watchlist so the new trading days are in `stock_cache`:

```bash
# Recompute what the pinned versions would have forecast, and score it. Use
# --samples = (trading days since the last day the models were trained on) - 4, so
# every forecast counted was made on or after that day: 6 after two weeks, 16 after four.
python scripts/evaluate_ensemble.py --model-name gru_v1_residual_macdonly_warm5 \
  --versions 1,2,3 --watchlist sp500 --samples 6 --breakdown-by-day --lag-test
python scripts/persistence_baseline.py \
  --watchlist sp500 --samples 6 --seq-length 30 --forecast-horizon 5
```

Or score the forecasts that were actually run from the UI, with the SQL query in
`docs/ENSEMBLE_PRODUCTION_PLAN.md` §6 against `forecast_predictions`. That only covers
days on which the forecast was run, and it pools all forecasts where the scripts
average per symbol first, so the two differ slightly. Either way the bar is drift on
the same window: the ensemble should keep an MAE roughly 20% below drift's. Two weeks
is about 3,000 forecasts but stocks move together, so treat it as a noisy read; a
month is more convincing.

`train_forecast_model.py --model-version N` saves under an explicit version (refused
if it exists). The retrain job uses it to reserve versions for parallel runs.

Legacy CLI-only form (no `--config`, still supported — see §9 for the resulting
filename):

```bash
python scripts/train_forecast_model.py \
  --architecture bidirectional_gru --epochs 40 --checkpoint-warmup-epochs 5 \
  --normalization-type global --forecast-horizon 5 \
  --split-strategy time --test-split 0.15 \
  --include-delta --residual-target --extra-features open-close \
  --hidden-size 32 --batch-size 64 --days 600

python scripts/evaluate_forecast_model.py \
  --watchlist sp500 --architecture bidirectional_gru_with_delta_residual_with_open_close \
  --samples 20 --lag-test --breakdown-by-day
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

## 14. If you continue

The live, ranked list of what to try next lives in `docs/FORECAST_MODEL_IMPROVEMENTS.md`'s
"Suggested order" (updated after every experiment) — refer to it rather than this
section, which would otherwise duplicate it and go stale the way this whole card
did between 2026-08-29 and 2026-09-03. As of the 2026-10-04 update, the top items are:

1. **Run every comparison on at least three seeds.** Single-run DA differences
   under ~10pp are inside the seed spread (§11); most earlier rankings in this
   investigation were made on one unseeded run each and should be treated as open.
2. Evaluate multiple early checkpoints on the watchlist rather than trusting the
   single lowest-val-loss one (§7's open caution). The seed spread sits almost
   entirely on days 3-5, the same place checkpoint choice moves.
3. Add naive baselines (persistence, drift, repeat-delta) to
   `evaluate_forecast_model.py` directly, so every future result is printed next
   to the bar it needs to beat rather than requiring a separate audit script.
4. `chart_patterns` — the only genuinely different information source available,
   and the most work.
5. Cheap new input features (`macd_signal_dist`, MA20/MA50 trend ratio,
   price-vs-MA20 ratio) — §1's B-list. Low prior: like `open-close`, they are
   derived from the same price series.
6. Seed ensembles are now what production serves (§9): 2-4% better MAE and
   steadier than a single seed (§11). The direction gap to drift is unchanged.
7. Check the frozen default ensemble against live data after two to four weeks
   (§13). Every number in §11 is from one window ending 2026-10-02; whether the MAE
   advantage over drift holds out of sample is still unmeasured.

**Tried 2026-10-04 with no confirmed benefit** (§11): `hidden_size` 16 and 8;
`learning_rate` 3e-4 and 1e-4; `lr_patience` 2 (a no-op — it fires after the best
epoch); `seq_length` 15, 20, 45, 60; dropping `open-close`;
`auxiliary_direction_lambda` 0.3 on the current recipe; forecast horizons 2 and 3;
three-seed averaging (helps MAE, not DA).

**Ruled out, do not revisit without new evidence:** per-day/per-column loss decay
weighting (`loss_decay_gamma`); the delta-only architectural consistency variant
(`predict_deltas_only`); recursive/rollout inference (`--rollout`); reframing
training as zero-crossing classification on `will_become_positive` (explicitly
not the objective — see §6).

Change **one variable per run** — a four-change run cannot be attributed.
