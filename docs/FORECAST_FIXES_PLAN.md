# Forecast Pipeline Fix Plan

Fixes for four defects found in the MACD forecaster training / inference pipeline.
Audit date: 2026-08-26. Branch at time of audit: `better_scripts` (clean).

---

## STATUS (updated 2026-08-28)

| Fix | State |
|-----|-------|
| 1 — validation split leaked through overlapping windows | **DONE** |
| 1b — validation axis did not match the test axis | **DONE** (found after Fix 1 landed) |
| 2 — production inference never passed `prev_value` | pending |
| 3 — `prev_value` applied after truncation | pending |
| 4 — `days_past` calendar vs trading days | pending |
| 5 — housekeeping (5a done with Fix 1; 5b, 5c pending) | partial |
| 6 — hard-fail when `--split-strategy time` yields no test set | pending |
| 7 — ~~honor `forecast_days` in the neural path~~ | **WITHDRAWN** — `forecast_days` is ARIMA-only by design |

**Read the "Findings" section at the bottom before doing any more model work.**
The measurement work these fixes enabled shows the current neural forecaster does
not beat a two-line arithmetic baseline. Fixes 2-4 are still correct and worth
doing, but retuning this model is not.

Files touched:

| File | Fixes |
|------|-------|
| `src/models/lstm_forecaster.py` | 1, 3, 5 |
| `src/models/neural_forecast.py` | 2, 3, 4 |
| `src/models/requests.py` | 4 |

**Ordering matters.** Fix 1 changes which weights get exported, so it must land and be
retrained before Fixes 2–4 can be measured. Fixes 2 and 3 are one coherent change and
must land together — Fix 2 alone walks straight into the Fix 3 trap.

---

## Fix 1 — Validation split leaks through overlapping windows  **[DONE]**

Implemented in `src/models/lstm_forecaster.py`. Deviations from the plan below:

- `_split_series` was later split into `_split_series_by_time` /
  `_split_series_by_symbol` plus a dispatcher taking a `strategy` argument (Fix 1b).
- The purge gap is `forecast_horizon`, **not** `seq_length + forecast_horizon`.
  The larger gap left too few points for a temporal val split on ~308-point
  series and silently fell back to a by-symbol split. Training windows only touch
  indices `< cut`, so any gap >= 0 already guarantees index disjointness; the
  horizon-sized gap is the standard purge.
- `train()` prints which axis val measures, and reports the winning epoch plus a
  note when the best epoch is the last one (undertrained signal).

Verified effect on a real sp500 run:

| | before | after |
|---|---|---|
| val vs train | val *below* train at all checkpoints | val 6-8x *above* train |
| best epoch | 200/200 (always last) | 3/200 |

### Problem

`prepare_sequences` shuffles every window across every symbol (`lstm_forecaster.py:470-473`),
and `train()` then slices the validation set off that shuffled array
(`lstm_forecaster.py:496-499`). Windows use stride 1 (`lstm_forecaster.py:451`), so window
*i* and *i+1* share 29 of 30 input points and 4 of 5 targets. After a global shuffle nearly
every validation window has a near-duplicate in the training set.

The leaked val loss drives checkpoint selection (`lstm_forecaster.py:543-545`), so the
exported model is the most *overfit* one, not the best generalizing one. Secondary leak:
global normalization stats are fit over train+val together.

Held-out **test** metrics are unaffected (that split happens in `train_forecast_model.py`
before windowing), so test numbers stay comparable across this fix.

### Change

Split the per-symbol series *before* windowing, then window each side independently.

**1a.** Add a helper to `MACDForecasterTrainer` (place above `train()`):

```python
def _split_series(self, data, val_fraction: float):
    """
    Split per-symbol series into train/val parts BEFORE windowing, so no
    validation window shares observations with a training window.

    Prefers a by-symbol split (cleanest — whole symbols held out). Falls back
    to a temporal split with a purge gap when there are too few symbols.
    Returns (train_series, val_series); val_series is [] when no split is possible.
    """
    series_list = data if isinstance(data, list) else [data]
    min_len = self.seq_length + self.forecast_horizon

    if val_fraction <= 0:
        return series_list, []

    # Preferred: hold out whole symbols.
    n_val = int(len(series_list) * val_fraction)
    if n_val >= 1 and (len(series_list) - n_val) >= 1:
        return series_list[:-n_val], series_list[-n_val:]

    # Fallback: temporal split per symbol with a purge gap. The gap guarantees
    # no shared observation between the last train window and the first val window.
    embargo = self.seq_length + self.forecast_horizon
    train_part, val_part = [], []
    for s in series_list:
        cut = int(len(s) * (1 - val_fraction))
        if cut >= min_len and (len(s) - cut - embargo) >= min_len:
            train_part.append(s[:cut])
            val_part.append(s[cut + embargo:])
        else:
            train_part.append(s)  # too short to split — training only
    return train_part, val_part
```

**1b.** Guard `prepare_sequences` against an empty result. Currently
`zip(*combined)` at `lstm_forecaster.py:473` raises `ValueError: not enough values to
unpack` on an empty window list, which fires *before* `evaluate()`'s own
`len(X_test) == 0` check at line 566. Insert before the `# Shuffle` block:

```python
    if not all_X:
        empty_x = torch.empty((0, self.seq_length, self.input_size), dtype=torch.float32)
        empty_y = torch.empty((0, self.output_size), dtype=torch.float32)
        return empty_x, empty_y
```

**1c.** Replace `lstm_forecaster.py:492-504` — the duplicate call, the shuffled split,
and the device moves:

```python
        self.batch_size = batch_size

        train_series, val_series = self._split_series(train_data, validation_split)

        # fit=True stores normalization stats — TRAIN ONLY, never val.
        X_train, y_train = self.prepare_sequences(train_series, fit=True)
        if val_series:
            X_val, y_val = self.prepare_sequences(val_series, fit=False)
        else:
            X_val, y_val = None, None

        if len(X_train) == 0:
            raise ValueError(
                f"No training windows produced. Need at least "
                f"{self.seq_length + self.forecast_horizon} points per symbol."
            )

        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)
        has_val = X_val is not None and len(X_val) > 0
        if has_val:
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)
        else:
            print("WARNING: no validation set — selecting checkpoint on train loss.")
```

Note this also deletes the duplicated `prepare_sequences` call (Fix 5a) as a side effect.

**1d.** Make the validation block and checkpoint selection tolerate `has_val == False`.
Replace `lstm_forecaster.py:534-545`:

```python
            # Validation
            if has_val:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(X_val)
                    val_loss = self.criterion(val_pred, y_val).item()
            else:
                val_loss = train_loss

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            selection_loss = val_loss if has_val else train_loss
            if selection_loss < best_val_loss:
                best_val_loss = selection_loss
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
```

### Expected effect

Val loss will jump substantially and stop tracking train loss — that is the fix working,
not a regression. Test MAE / directional accuracy should improve or hold. If test metrics
get *worse*, the previous checkpoint was winning on leaked val and this is the honest number.

### Verify

- Val loss curve visibly separates from train loss in the per-10-epoch output.
- Assert disjointness once, manually: with `--symbols A,B,C,D,E --epochs 1`, confirm
  `_split_series` returns 4 train / 1 val symbol.
- Test-set block at `train_forecast_model.py:343-351` still runs and reports.

---

## Fix 1b — Validation axis did not match the test axis  **[DONE]**

### Problem

Fix 1 as first written always preferred a by-symbol validation split. With
`--split-strategy time` the test set is temporal (same symbols, later dates) —
which is also the production condition — so val was measuring a *different*
generalization axis than the thing being selected on. Early stopping picked
epoch 3/200 on the harder unseen-symbol metric.

Evidence: at that checkpoint, test combined MSE was ~0.559 while val MSE was
0.955 — a 1.7x gap purely from the axis difference. (Note the printed
`Test MAE 0.145` is MACD-channel-only *and* MAE, while train/val are MSE over
both channels; converting properly gives test MACD MSE 0.150, delta 0.967.)

### Change

`_split_series(data, val_fraction, strategy)` dispatches to
`_split_series_by_time` or `_split_series_by_symbol`, each falling back to the
other with a warning when it cannot produce a validation set.
`train()` takes `split_strategy`; `train_forecast_model.py` passes
`args.split_strategy`.

Verified at production dimensions (501 symbols x 308 points, horizon 5):

```
time   : train 501 syms x 246 pts | val 501 syms x 57 pts
         train idx 0..245, val idx 251..307, gap=5
         -> 106212 train windows, 11523 val windows
symbol : train 401 syms | val 100 syms
```

---

## Fix 2 — Production inference never supplies `prev_value` (delta channel starts at a fake 0)

### Problem

`_calculate_deltas` zeroes element 0 (`lstm_forecaster.py:406-409`). At training time it is
applied to the **full per-symbol series before windowing**
(`lstm_forecaster.py:432`), so every window but the first gets a real non-zero
`delta[0]`. The model learns that position is real data.

`CoreMLForecaster.predict` has a `prev_value` parameter to reconstruct it
(`neural_forecast.py:166-167`), but the production caller omits it
(`neural_forecast.py:280`):

```python
forecast = self.forecaster.predict(series)   # prev_value=None -> deltas[0] = 0
```

`evaluate_forecast_model.py:210` *does* pass it. So offline metrics are measured on
correctly-fed data and production is not.

The needed value is already in memory and discarded: `series` is ~69 points and gets
truncated to the last 30, so the true pre-window value is `series[-31]`.

### Change

Handled entirely inside `predict` by Fix 3 below — no caller change required. After Fix 3,
`self.forecaster.predict(series)` at `neural_forecast.py:280` derives the correct
`prev_value` itself.

### Verify

Log `deltas[0]` for one symbol before and after; it must go from `0.0` to a non-zero value
matching `series[-30] - series[-31]`.

---

## Fix 3 — `prev_value` is applied after truncation/padding (latent wrong-value trap)

### Problem

`neural_forecast.py:156-167` reshapes the sequence, *then* applies `prev_value` to
`sequence[0]` of the **new** array:

```python
if len(sequence) < self.seq_length:
    padding = np.full(self.seq_length - len(sequence), sequence[0])
    sequence = np.concatenate([padding, sequence])
elif len(sequence) > self.seq_length:
    sequence = sequence[-self.seq_length:]
...
if prev_value is not None:
    deltas[0] = sequence[0] - prev_value
```

The docstring defines `prev_value` as the point immediately before the *caller's*
`sequence[0]`. Once truncation shifts the window, `prev_value` refers to a point ~39
positions away and `deltas[0]` becomes large garbage. In the padding branch `sequence[0]`
is itself synthetic.

Neither current caller trips it (evaluate passes an exactly-`seq_length` slice; production
passes `None`) — but Fix 2 would.

Separately: `MACDForecasterTrainer.predict` (`lstm_forecaster.py:631-672`) does no
truncation at all, so the PyTorch and Core ML paths return different answers for the same
long input.

### Change

**3a.** Replace the reshape block in `CoreMLForecaster.predict`
(`neural_forecast.py:155-170`) so the contract is unambiguous — `predict` owns the
windowing and derives `prev_value` itself:

```python
        seq = np.asarray(sequence, dtype=np.float32)

        if len(seq) > self.seq_length:
            # Derive the true pre-window value BEFORE truncating. A caller-supplied
            # prev_value refers to the untruncated sequence[0] and would be wrong here.
            prev_value = float(seq[-(self.seq_length + 1)])
            seq = seq[-self.seq_length:]
        elif len(seq) < self.seq_length:
            logger.warning(
                "Input has %d points but model needs %d; padding %d synthetic steps. "
                "Increase days_past.",
                len(seq), self.seq_length, self.seq_length - len(seq),
            )
            pad = np.full(self.seq_length - len(seq), seq[0], dtype=np.float32)
            seq = np.concatenate([pad, seq])
            prev_value = None   # seq[0] is synthetic; a real delta is meaningless
        sequence = seq
```

Exact-length input falls through both branches and keeps a caller-supplied `prev_value`,
so `evaluate_forecast_model.py` stays correct.

**3b.** Apply the same block to `MACDForecasterTrainer.predict`
(`lstm_forecaster.py:631-648`) so both engines agree. Import `logging` and add a
module-level `logger` there, or use `print` to match that file's existing style.

### Verify

- `evaluate_forecast_model.py --symbol MSFT --samples 10` produces **identical** numbers
  before and after (its inputs are exact-length — this is the regression guard).
- A long-input call now returns the same forecast as an exact-length call on the same
  final 30 points: `predict(series)` == `predict(series[-30:], prev_value=series[-31])`.
- Padding now emits a warning instead of failing silently.

---

## Fix 4 — `days_past` is calendar days, `seq_length` is trading days

### Problem

`neural_forecast.py:263-265`:

```python
end_date = get_latest_market_date()
start_date = end_date - timedelta(days=days_past)
```

30 calendar days ≈ 21 trading days against a model wanting 30 → ~30% of the window becomes
replicated `sequence[0]`, and with `include_delta` that region's delta channel is all zeros.

The default API path is safe (`config.py:33` and `requests.py:31` both default to 100,
≈69 trading days → truncates). Two entry points are not:

- `requests.py:31` allows `ge=30` — a client passing the documented minimum silently gets
  a 30%-fabricated window.
- `NeuralForecastService.forecast_macd` itself defaults to `days_past=30`
  (`neural_forecast.py:235`), so any internal caller omitting the argument hits it.

`evaluate_forecast_model.py:31` already compensates (`calendar_days = int(days_back * 1.6)`);
production has no equivalent.

### Change

**4a.** Change the default at `neural_forecast.py:235` from `30` to `100` to match
`config.py:33`. Do the same for `forecast_batch` at `neural_forecast.py:327`.

**4b.** Widen the fetch window in `forecast_macd`, replacing `neural_forecast.py:263-265`:

```python
            end_date = get_latest_market_date()

            # days_past is calendar days; seq_length is trading days (~1.6x ratio).
            # +6 covers the prev_value point plus holiday clustering.
            needed = self.forecaster.seq_length if self.is_available else 0
            calendar_days = max(days_past, int((needed + 6) * 1.6) + 1)
            start_date = end_date - timedelta(days=calendar_days)
            macd_data = get_macd_for_range(symbol, start_date, end_date)
```

Widening the fetch is safe: `predict` truncates to the last `seq_length` points anyway,
and `last_value = float(series[-1])` at `neural_forecast.py:286` is unaffected by a longer
history.

**4c.** Raise the floor at `requests.py:31` so the minimum cannot produce a padded window:

```python
    days_past: int = Field(default=100, ge=60, le=365, description="Days of historical data (calendar days; ~1.6x trading days)")
```

`ge=60` ≈ 41 trading days, comfortably above `seq_length=30`. This is an API contract
change — a client currently sending 30–59 will now get a 422. Given the only in-repo
callers are `forecast_router.py` defaults (100), the blast radius is external clients only.
If that is unacceptable, keep `ge=30` and rely on 4b, which already guarantees a full
window regardless of what the client sends.

### Verify

- `POST /forecast` with `days_past=60` produces no padding warning from Fix 3a.
- Confirm the model no longer pads on the default path: warning absent at `days_past=100`.

---

## Fix 5 — Housekeeping (not part of the four, low risk)

**5a.** Duplicate `prepare_sequences` call at `lstm_forecaster.py:493-494` — the second
overwrites the first, doubling data-prep time and burning an extra shuffle. Already removed
by Fix 1c.

**5b.** `export_to_coreml` moves the model to CPU (`lstm_forecaster.py:724`) and never
restores `self.device`, so `trainer.predict()` after an export raises a device mismatch.
Add before the `return` at `lstm_forecaster.py:754`:

```python
        self.model.to(self.device)
```

**5c.** `get_training_data` filters out `None` MACD values
(`train_forecast_model.py:60-63`), so DB gaps collapse silently and a delta can span
multiple days while the model reads it as a one-day change. Affects the delta channel
specifically. Minimum: count and report dropped rows per symbol. Proper fix: assert date
contiguity, or split a symbol into separate series at each gap.

---

## Execution order

Each step is a separate commit so effects stay attributable.

| # | Commit | Contents | Retrain? |
|---|--------|----------|----------|
| 0 | baseline | none — record current metrics | yes, tag as baseline |
| 1 | `fix: split validation set before windowing` | Fix 1 (a–d), 5a | yes |
| 2 | `fix: correct delta feature at inference` | Fix 3 (a, b) → enables Fix 2 | no |
| 3 | `fix: size fetch window in trading days` | Fix 4 (a–c) | no |
| 4 | `chore: forecaster housekeeping` | Fix 5b, 5c | no |

Commits 2–4 are inference-only and can be measured against the commit-1 model without
retraining.

## Verification protocol

There is currently no trained model (`models/` does not exist), so establish the baseline
first. Use a small, fast config for iteration:

```bash
cd src
docker-compose up -d   # DB must be populated for get_macd_for_range

# Step 0 — baseline on current (unfixed) code
python scripts/train_forecast_model.py \
  --architecture stacked_lstm --days 730 --epochs 40 --split-strategy time
python scripts/evaluate_forecast_model.py \
  --watchlist sp500 --samples 20 --breakdown-by-day --lag-test > /tmp/baseline.txt

# after each commit, same two commands, then diff the summaries
```

Use `--architecture stacked_lstm` and **no** `--include-delta` so the artifact lands at the
one path the running API actually loads (`models/macd_stacked_lstm_forecaster.mlpackage` —
see the separate naming issue below). Add `--include-delta` runs afterwards to exercise the
delta fixes, evaluating with `--architecture stacked_lstm_with_delta`.

Record for each step: final train loss, final val loss, test MAE, test directional
accuracy, and the `--lag-test` ratio.

**The `--lag-test` ratio is the number that matters.** A ratio near `1.0x` means the model
is reproducing the last observed value and has learned nothing useful, regardless of how
good MAE looks. Watch it across all four commits.

## Fix 6 — `--split-strategy time` can silently produce an empty test set

### Problem

`train_forecast_model.py:262-268` requires **both** sides of the split to hold
`seq_length + forecast_horizon` points, and falls through to
`train_data.append(symbol_array)` otherwise — with no warning. A real run
reported `Test: 0 symbols (part 2), 0 data points` and then
`Skipping test evaluation`, producing zero honest metrics.

Arithmetic that caused it: 365 days -> ~251 trading points/symbol;
`int(251 x 0.85) = 213`; test part `251 - 213 = 38`; required
`30 + 10 = 40`. Missed by two points, and *no* symbol could pass because
clearing it needs N >= 261 while 365 calendar days caps out around 251.

Requirement, with `W = seq_length + forecast_horizon`:

```
train side:  floor(N(1-t)) >= W
test  side:  N - floor(N(1-t)) >= W      <-- binding
```

Rule of thumb: `--days >= 1.45 * W / test_split`. Raising `--forecast-horizon`
raises the floor linearly. Empirical calendar->trading ratio is 1.45.

### Change

Raise a hard error (or fall back to `--split-strategy symbol` with a loud
warning) when the time split yields no test symbols. Silent fallthrough is the
bug, not the arithmetic.

---

## Findings — read before further model work

Measured 2026-08-28 on sp500, 500 symbols, 20 samples/symbol, seq 30, horizon 5.
Baselines from `src/scripts/persistence_baseline.py` (new; no model, no training):

- `flat`  — `macd[t+k] = macd[t]`, `delta[t+k] = 0`
- `drift` — `macd[t+k] = macd[t] + (k+1)*delta[t]`, `delta[t+k] = delta[t]`

### 1. The model does not beat trivial arithmetic on direction

| Day-1 metric | drift | model (bi-GRU, h64, global, ep3) |
|---|---|---|
| MACD DA | **78.4%** | 59.9% |
| MACD MAE | **0.4011** | 0.5788 |
| delta DA | **78.37%** | 78.2% |

drift wins on MACD direction at *every* horizon (78.4/58.7/48.9/44.3/41.5 vs
59.9/48.0/41.0/38.1/36.5). On the one clean metric — Day-1 delta sign — the model
ties the baseline. The model's only wins are magnitude at days 3-5 and delta MAE,
both of which come from damping rather than information. Its lag ratio is 0.82x
(below 1.0 = closer to a lagged copy than to the target).

### 2. The delta-DA metric is an artifact past Day 1

`_compute_metrics` walks its reference onto the *previous actual delta* for days
>= 2, so predicting zero scores well whenever deltas mean-revert:

| Day | flat (predict 0) | model |
|---|---|---|
| 2 | 61.4% | 60.0% |
| 3 | 62.8% | 62.1% |
| 4 | 63.4% | 63.6% |
| 5 | 62.5% | 62.9% |

Confirmed on a synthetic random walk, where flat scores 66% aggregate. **Only
Day-1 delta DA is a valid skill measure** (there the reference is a fixed 0.0).
The model's 65.39% aggregate delta DA is not evidence of anything.

### 3. Why 78% is not skill

MACD is `EMA12 - EMA26`. Its first difference is heavily filtered, so ~78%
day-over-day sign persistence is the smoother's inertia. Note drift's own MACD DA
drops below chance by day 3 — nothing here, model or baseline, predicts MACD 3-5
days out.

### 4. Consequences

- Do **not** wire this model into `will_become_positive`; drift is strictly better
  on direction.
- Do not retune this architecture. Two runs with different normalization,
  capacity, horizon and training duration all converged to damped persistence,
  which is where MSE minimization on a smoothed near-random-walk goes.
- Ways forward, cheapest first: **residual target** (implemented — see below);
  **add features** (close/volume/MA20/MA50 already in `stock_cache`, currently
  unused — a model seeing only a smoothed function of price cannot know more than
  the smoother); **reframe as classification** on "does MACD cross zero within k
  days" with cross-entropy on the actual decision.

### 5. Residual target (implemented)

`--residual-target` trains on `actual - drift_prediction` instead of `actual`, so
"beat the persistence baseline" becomes the literal training objective rather than
something checked afterwards. Inference adds the drift back, so
`evaluate_forecast_model.py` numbers stay directly comparable to the table above.
`trainer.evaluate()` additionally prints drift-alone metrics on the same test set,
giving an in-training answer to "did the model beat drift".

Artifacts get a `_residual` filename suffix so they cannot overwrite a
level-target model.

---

## Out of scope

The model-path mismatch documented earlier is **not** covered here: `get_model_path`
defaults `architecture="stacked_lstm"` (`lstm_forecaster.py:757`) and
`NeuralForecastService` never passes one (`neural_forecast.py:47-48`,
`forecast_service.py:21`), so the API only ever loads
`models/macd_stacked_lstm_forecaster.mlpackage`, and `--include-delta` output
(`_with_delta` suffix) is invisible to it. Separate change: give `get_model_path` an
`include_delta` parameter and source the architecture from `core/config.py`.
