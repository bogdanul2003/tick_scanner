# Forecast Live Tracking

How the Bullish MACD Forecast did against what the market then did. One section per
forecast run, filled in as its five forecast days close.

This is the out-of-sample record. The figures in `src/models/MODEL_CARD.md` §11 are
backtests on a window the models were chosen on; the figures here are forecasts stored
before the outcome was known (`forecast_predictions` table, see
`docs/ENSEMBLE_PRODUCTION_PLAN.md` §6).

## How to read the numbers

- **Scope**: the 501 symbols of the `sp500` watchlist.
- **Day**: trading days after the forecast's starting close. Day 1 is the next session.
- **MAE / median error**: absolute difference between forecast and actual MACD, in
  dollars. The mean is dominated by high-priced symbols; the median is the typical symbol.
- **Direction (DA)**: share of symbols where the forecast and the actual MACD are on the
  same side of the previous day's *actual* MACD. Same rule as the evaluation scripts.
- **Drift**: the naive baseline — MACD keeps changing by the last day's change. A model
  that does not beat drift adds nothing.
- **Beats drift on**: share of symbols where the model's error is smaller than drift's.
- All symbols move on the same market day, so one day's result is one correlated
  sample, not 501 independent ones. Judge on several runs.

Backtest reference (default ensemble, window ending 2026-10-02): all-days MAE 0.901,
median 0.452, DA 51.3%; drift MAE 1.147, DA 57.8%. Day 1 alone is about 78-81% DA for
both, falling to about 40% for the model by day 5.

## Summary

| Forecast from | Model | Days scored | MAE | Drift MAE | DA | Drift DA |
|---|---|---|---|---|---|---|
| 2026-10-02 | `macdonly_warm5` (default) | 5 of 5 | 0.868 | 1.097 | 41.9% | 47.9% |
| 2026-10-02 | `hidden16` | 5 of 5 | 0.838 | 1.097 | 44.7% | 47.9% |
| 2026-10-05 | `macdonly_warm5` (default) | 1 of 5 | 0.309 | 0.315 | 80.0% | 83.2% |
| 2026-10-05 | `hidden16` | 1 of 5 | 0.299 | 0.315 | 84.4% | 83.2% |
| 2026-10-07 | `macdonly_warm5` (default) | 3 (days 1-3 only) | 0.721 | 0.741 | 43.6% | 51.8% |
| 2026-10-07 | `hidden16` | 3 (days 1-3 only) | 0.709 | 0.741 | 48.1% | 51.8% |

Update a row when more days of that run have closed (MAE and DA over all scored days),
and add a row for each new run.

---

## Run: forecast from the 2026-10-02 close

- Forecast made 2026-10-04 (Sunday), 16:21-16:24 UTC, for 518 symbols; 501 are in sp500.
- `macdonly_warm5`: `gru_v1_residual_macdonly_warm5` versions 1, 2, 3, trained 2026-10-04.
- `hidden16`: `gru_v1_exp_hidden16` versions 4, 5, 6, trained 2026-10-04.
- Both were trained on data up to 2026-10-02, so every day below is out of sample.

### Per day

| Day | Date | Model | MAE | Median error | DA | Drift MAE | Drift DA | Beats drift on |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-05 | `macdonly_warm5` | 0.251 | 0.106 | 72.5% | 0.273 | 75.1% | 50.1% |
| 1 | 2026-10-05 | `hidden16` | 0.237 | 0.098 | 77.3% | 0.273 | 75.1% | 66.7% |
| 2 | 2026-10-06 | `macdonly_warm5` | 0.565 | 0.241 | 53.5% | 0.608 | 57.9% | 60.5% |
| 2 | 2026-10-06 | `hidden16` | 0.549 | 0.240 | 54.9% | 0.608 | 57.9% | 55.9% |
| 3 | 2026-10-07 | `macdonly_warm5` | 0.803 | 0.317 | 34.9% | 0.958 | 46.3% | 62.3% |
| 3 | 2026-10-07 | `hidden16` | 0.775 | 0.341 | 38.1% | 0.958 | 46.3% | 60.9% |
| 4 | 2026-10-08 | `macdonly_warm5` | 1.158 | 0.490 | 27.7% | 1.543 | 30.5% | 61.9% |
| 4 | 2026-10-08 | `hidden16` | 1.121 | 0.495 | 28.9% | 1.543 | 30.5% | 65.7% |
| 5 | 2026-10-09 | `macdonly_warm5` | 1.565 | 0.655 | 20.6% | 2.105 | 29.7% | 60.1% |
| 5 | 2026-10-09 | `hidden16` | 1.508 | 0.658 | 24.4% | 2.105 | 29.7% | 65.9% |

Flat baseline (MACD stays at its 2026-10-02 value) on day 1: MAE 0.426.

Days 2 to 5 are scored on 499 symbols: PSKY and WBD have no MACD cached for 2026-10-06
to 2026-10-08. Days 3 to 5 are matched by market date, not by each symbol's own row
count, so PSKY's 2026-10-09 bar is not scored as an earlier day (the query under
"Scoring a run" would do that for a symbol with missing days).

### "Will become positive" flag (`macdonly_warm5`)

384 symbols had a negative MACD on 2026-10-02. The model flagged 10 of them to turn
positive within five days.

| As of | Flagged symbols now positive | Symbols that crossed above zero | Of those, flagged |
|---|---|---|---|
| Day 1 (2026-10-05) | 6 of 10 | 8 | 6 |
| Day 2 (2026-10-06) | 8 of 10 | 15 | 8 |
| Day 5 (2026-10-09) | 9 of 10 | 46 | 9 |

On day 1 specifically the model forecast 6 crossings; 5 of them happened.

Final: 9 of the 10 flags came true (CHRW did not) and all 9 were still positive on day
5. The flag is precise but covers little: 46 symbols crossed, 37 of them unflagged.

### Per-symbol ranking, day 1 (`macdonly_warm5`)

`forecast_day1_ranking_2026-10-05.csv` (project root) ranks all 501 symbols from best
to worst forecast, by error as a share of how far MACD moved that day. Per symbol it
has the price and its move, MACD on the starting day, forecast, actual, the error (in
dollars, as % of the move and as % of price), drift's error, and whether the model
beat drift and called the direction.

- About the first 100 are close calls: the error is 12% of the move at rank 50, 24% at
  rank 100 and 72% at rank 250.
- The top 250 all have the direction right (an error smaller than the move implies
  it), but beat drift less often than the bottom 251: 116 of 250 against 135 of 251.
  They are mostly steady trends, where drift is good too.
- The largest dollar errors are on high-priced symbols and on one-day price jumps
  (PTC +33.5%, CHRW -10.8%), which drift missed equally. The 12 largest errors are 26%
  of the total error. The model's own misses, where drift was much closer, are NVR,
  SNDK and MPWR.
- CTVA ($12, MACD -10.7) looks like an unadjusted split or spin-off in the cached
  history, not a forecast error.

### Per-symbol ranking, day 2 (`macdonly_warm5`)

`forecast_day2_ranking_2026-10-06.csv` (project root): the same ranking for the
two-day-ahead forecast, 499 symbols. The move and the error share are measured from the
2026-10-02 MACD; direction is judged against the 2026-10-05 actual; drift is two steps
of the 2026-10-02 daily change.

- The error is 16% of the two-day move at rank 50, 33% at rank 100 and 69% at rank 250
  (day 1: 12%, 24%, 72%).
- 53 symbols are in the top 100 on both days. With two days this is not yet evidence
  that some symbols are reliably well forecast; chance alone would give about 20.
- The model beats drift on 113 of the top 250 and 189 of the other 249: as on day 1,
  its edge over drift is on the symbols where neither is close, not on the best matches.
- The largest dollar errors are again high-priced symbols. Where the trend simply
  continued, drift was far closer (SNDK 9.35 against 4.09, LITE 7.94 against 2.07, FIX
  7.47 against 2.13, MPWR 6.74 against 0.71); where it turned, the model was closer (AZO
  8.35 against 12.64, NVR 7.25 against 13.75). The 12 largest errors are 27% of the total.

### Per-symbol ranking, days 3 to 5 (`macdonly_warm5`)

`forecast_day3_ranking_2026-10-07.csv`, `forecast_day4_ranking_2026-10-08.csv` and
`forecast_day5_ranking_2026-10-09.csv` (project root): the same ranking for the three-,
four- and five-day-ahead forecasts, 499 symbols each. Columns as in the day 2 file: the
move and the error share are measured from the 2026-10-02 MACD, direction is judged
against the previous day's actual (the second MACD column), and drift is that many
steps of the 2026-10-02 daily change.

| | Day 3 | Day 4 | Day 5 |
|---|---|---|---|
| Error as % of the move at rank 50 / 100 / 250 | 14% / 35% / 71% | 23% / 42% / 82% | 29% / 47% / 87% |
| Direction right in the top 100 | 77 | 72 | 52 |
| Beats drift, top 250 / other 249 | 130 / 181 | 127 / 182 | 120 / 180 |
| Share of total error in the 12 largest | 29% | 27% | 27% |

- The close calls thin out with the horizon: the error at rank 100 goes from 24% of the
  move on day 1 to 47% on day 5.
- From day 3 a small error against the five-day move no longer implies the right
  direction, because direction is judged on the last day's change alone.
- 24 symbols are in the top 100 on all five days (BKNG, CASY, CCL, CDNS, COO, DVA, EIX,
  EMR, EXR, FDX, FRT, FTV, INCY, JCI, KEYS, MTB, NSC, PNR, RCL, SNPS, SW, SWKS, UNP,
  VRTX). The five days are one forecast path against one week of prices, so this is
  one run's result, not five; it needs other runs to mean anything.
- The edge over drift stays on the symbols where neither is close, as on days 1 and 2.
- The largest dollar errors are the same symbols every day. NVR (47.2 on day 5, drift
  37.3) and SNDK (36.8, drift 14.3) are the model's own misses; on AZO (20.5 against
  38.7) and LITE (12.3 against 24.0) it was much closer than drift. STX is the clearest
  case of a trend that simply continued: 13.1 against 0.4 on day 5.

### Notes

- Days 3 to 5: error stays below drift and the gap widens (16%, 25%, 26% for the
  default; 19%, 27%, 28% for `hidden16`), with the lower error on 60-66% of symbols.
  Direction falls to 21-24% by day 5, against the backtest's roughly 40%, and is behind
  drift on every day (by 2 to 11 points); drift itself is at 30% on days 4 and 5.
  Below 50% means the forecast sat on the wrong side of the previous day's actual more
  often than not. `hidden16` is ahead of the default on error and direction on all
  three days.
- Day 2: error is still below drift for both ensembles (7% and 10%), and the model has
  the lower error on 60.5% of symbols (default). Direction is 3 to 4 points behind
  drift for both. Error roughly doubled from day 1, as did drift's.
- Day 1: both ensembles had lower error than drift (8% and 13%). On direction
  `hidden16` was 2 points ahead of drift and the default 2.6 points behind.
  `hidden16` was ahead of the default on every measure; the backtests had them level.
- The 2026-10-05 bars were fetched 15-25 minutes after the close, while the settle
  buffer was 15 minutes (30 minutes since). They may not be final; re-score day 1
  if any are refetched.

---

## Run: forecast from the 2026-10-05 close

- Forecast made 2026-10-05, 21:10-21:34 UTC, for the sp500 symbols.
- Same model versions as the run above (`macdonly_warm5` v1-3, `hidden16` v4-6); no
  retrain in between.

### Per day

| Day | Date | Model | MAE | Median error | DA | Drift MAE | Drift DA | Beats drift on |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-06 | `macdonly_warm5` | 0.309 | 0.108 | 80.0% | 0.315 | 83.2% | 37.9% |
| 1 | 2026-10-06 | `hidden16` | 0.299 | 0.094 | 84.4% | 0.315 | 83.2% | 57.1% |
| 2 | 2026-10-07 | | | | | | | |
| 3 | 2026-10-08 | | | | | | | |
| 4 | 2026-10-09 | | | | | | | |
| 5 | 2026-10-12 | | | | | | | |

Scored on 499 symbols (PSKY and WBD have no MACD cached for 2026-10-06).

### Notes

- Day 1: the default ensemble's error is 2% below drift's but it has the lower error on
  only 37.9% of symbols, and its direction is 3.2 points behind drift. `hidden16` is
  ahead of the default on every measure again, and ahead of drift on error (5%) and
  direction (1.2 points).

---

## Run: forecast from the 2026-10-07 close (days 1 to 3 only)

- Forecast made 2026-10-07, 20:44-20:46 UTC, for the sp500 symbols.
- Same model versions as the runs above (`macdonly_warm5` v1-3, `hidden16` v4-6); no
  retrain in between.
- Tracked for the first three forecast days only. It is the latest run whose first
  three days have all closed (as of 2026-10-13).

### Per day

| Day | Date | Model | MAE | Median error | DA | Drift MAE | Drift DA | Beats drift on |
|---|---|---|---|---|---|---|---|---|
| 1 | 2026-10-08 | `macdonly_warm5` | 0.381 | 0.174 | 69.1% | 0.391 | 72.3% | 40.5% |
| 1 | 2026-10-08 | `hidden16` | 0.363 | 0.152 | 74.7% | 0.391 | 72.3% | 61.3% |
| 2 | 2026-10-09 | `macdonly_warm5` | 0.654 | 0.286 | 40.9% | 0.695 | 48.1% | 52.3% |
| 2 | 2026-10-09 | `hidden16` | 0.656 | 0.300 | 43.3% | 0.695 | 48.1% | 49.3% |
| 3 | 2026-10-12 | `macdonly_warm5` | 1.130 | 0.488 | 20.8% | 1.136 | 35.1% | 51.5% |
| 3 | 2026-10-12 | `hidden16` | 1.107 | 0.473 | 26.3% | 1.136 | 35.1% | 51.9% |

Flat baseline (MACD stays at its 2026-10-07 value): MAE 0.484, 0.920 and 1.456 on days
1, 2 and 3.

Scored on 499 symbols, matched by market date (PSKY and WBD have no MACD cached for
2026-10-07 and 2026-10-08).

### "Will become positive" flag (`macdonly_warm5`)

371 symbols had a negative MACD on 2026-10-07. The model flagged 14 of them to turn
positive within five days.

| As of | Flagged symbols now positive | Symbols that crossed above zero | Of those, flagged |
|---|---|---|---|
| Day 1 (2026-10-08) | 7 of 14 | 10 | 7 |
| Day 2 (2026-10-09) | 13 of 14 | 27 | 13 |
| Day 3 (2026-10-12) | 13 of 14 | 46 | 13 |

AVGO is the one flag not yet true; the flag's own horizon is five days.

### Per-symbol ranking, days 1 to 3 (`macdonly_warm5`)

`forecast_run2026-10-07_day1_ranking_2026-10-08.csv`,
`forecast_run2026-10-07_day2_ranking_2026-10-09.csv` and
`forecast_run2026-10-07_day3_ranking_2026-10-12.csv` (project root), 499 symbols each.
Same columns and rules as the 2026-10-02 run's files, measured from the 2026-10-07
MACD. The file names carry the run date because the target dates overlap with the
earlier run's files.

| | Day 1 | Day 2 | Day 3 |
|---|---|---|---|
| Error as % of the move at rank 50 / 100 / 250 | 19% / 39% / 81% | 18% / 36% / 74% | 27% / 47% / 80% |
| Direction right in the top 100 | 100 | 97 | 67 |
| Beats drift, top 250 / other 249 | 79 / 123 | 91 / 170 | 94 / 163 |
| Share of total error in the 12 largest | 32% | 29% | 32% |

- Fewer close calls than in the 2026-10-02 run: the error at rank 100 was 24%, 33% and
  35% of the move on that run's days 1 to 3.
- 49 symbols are in the top 100 on all three days, 15 of them utilities (AEP, CMS, CNP,
  DTE, EIX, ETR, EVRG, FE, NI, PCG, PEG, PNW, PPL, SO, WEC). Five were also in the
  2026-10-02 run's all-five-days list: CASY, CCL, EIX, INCY, RCL. That run's days 1 to
  3 overlap this one's by no market day, so this is the first cross-run repeat, but
  five names is what chance could give.
- The largest errors are high-priced chip and storage symbols that fell 7-12% over the
  three days: SNDK (40.8 on day 3, drift 23.5), MU (15.7, drift 16.3), MPWR (13.3,
  drift 17.0), LITE (10.5, drift 20.4). NVR is again the model's own miss on day 3
  (29.4 against drift's 4.4).
- CTVA is in the top three errors every day (about 17) on a $13 price: the same
  unadjusted history noted for the earlier run, not a forecast error.

### Notes

- Error: the default is below drift on all three days, but only just (3%, 6%, 0.5%),
  and has the lower error on 40.5% of symbols on day 1 and about half after. In the
  2026-10-02 run the margin on days 1 to 3 was 8%, 7% and 16%.
- Direction: the default is behind drift on every day and the gap grows (3, 7 and 14
  points). 20.8% on day 3 is lower than the 34.9% of the earlier run's day 3.
- `hidden16` is ahead of the default on error on days 1 and 3 and level on day 2, and
  ahead on direction every day; it beats drift on direction on day 1 only.
- Newer runs, first three days not yet closed: from 2026-10-08 (two days scored) both
  ensembles have a higher error than drift (0.457 and 0.448 against 0.436); from
  2026-10-09 (one day) also (0.326 and 0.315 against 0.289). The 2026-10-12 run has no
  scored day yet.

---

## Across runs so far (2026-10-06)

Three day-level results from two runs, all in the same week, so correlated:

- **Error**: both ensembles below drift every time, by 2-8% (default) and 5-13%
  (`hidden16`).
- **Direction**: the default is behind drift every time (2.6, 4.4 and 3.2 points);
  `hidden16` is ahead on both day-1 results and behind on day 2.
- **`hidden16` against the default**: ahead on error and direction in all three. The
  backtests could not separate them; this is the first, still thin, sign that they differ.

---

## Should the model be retrained?

The live record above says how the model is doing, not whether a retrain would change
that. The **Freshness backtest** button in the "Bullish MACD Forecast" panel answers the
second question for the selected ensemble. It takes about 30-40 minutes on the CPU and
changes nothing that is served.

What it does:

1. Takes the last 20 trading days whose five forecast days have all closed as the
   evaluation window.
2. Trains the ensemble's config (3 seeds each) four times, as if on an earlier day: with
   data ending the day before the window, and 4, 8 and 13 weeks before that
   (`train_forecast_model.py --end-date`). The pipeline is the retrain's own, splits
   included, so the copies are what a retrain on those days would have produced.
3. Scores every copy, the served ensemble, drift and flat on the same forecasts
   (`scripts/evaluate_freshness.py`). Error is |forecast - actual| as a % of the stock's
   price, so the highest-priced symbols do not dominate as they do the dollar MAE.
4. Fits how much the error grows per week of data age and multiplies it by the served
   ensemble's age. The 95% range covers both which days were in the window (resampled
   as whole days, since all symbols move on the same day) and which seeds each copy
   got (from the spread of its single-seed members). Seeds matter more than one would
   guess: in the first run single seeds of one copy differed by up to 18%, and a copy
   came out 4% better than the fresh one by luck.

How to read the verdict:

- **Retrain**: the expected loss from age is at least 2% and its range is above zero.
- **No retrain needed**: even the top of the range is below 2%. The report says when,
  at the measured rate, a retrain would start to gain 2%.
- **Inconclusive**: in between; run it again in a few weeks.
- Separately it says whether the served ensemble beats drift on the window. If it does
  not and age does not matter, retraining the same config will not fix it: the config
  or the market is the problem, and a different config is the thing to try.

A retrain also re-draws the seeds, so even when it is worth doing its result can be
worse by luck; the report gives that range. Check the new versions' track record before
relying on them.

The report also shows which dates the served ensemble actually learned from. With the
time split, weights are fitted on roughly the first two thirds of the 600-day window,
the checkpoint is chosen on the next part, and the last 15% is held out, so a model
trained today has not fitted anything from the last six months or so. A retrain moves
that whole window forward by the model's age, no more.

The job file is `models/freshness_backtest_job.json`; the training and scoring logs are
`train_logs/backtest_<job id>_*.log`. The backtest models are deleted when the job
finishes.

## Scoring a run

Run after the watchlist has been refreshed for the day (from the repo root). It scores
every stored run and every forecast day that has closed, matching by trading-day
offset so holidays do not shift anything.

```bash
docker exec -i tick_scanner_db psql -U postgres -d ticks <<'SQL'
WITH sp AS (
  SELECT ws.symbol FROM watchlist_symbols ws
  JOIN watchlists w ON w.id = ws.watchlist_id WHERE w.name = 'sp500'
),
actuals AS (
  SELECT symbol, date, macd,
         ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY date) AS rn
  FROM stock_cache WHERE macd IS NOT NULL
),
scored AS (
  SELECT p.model, p.as_of_date, p.horizon_day, t.date AS actual_date,
         p.predicted_macd AS pred, t.macd AS actual, prev.macd AS prev_actual,
         o.macd + p.horizon_day * (o.macd - o1.macd) AS drift
  FROM forecast_predictions p
  JOIN sp USING (symbol)
  JOIN actuals o    ON o.symbol = p.symbol AND o.date = p.as_of_date
  JOIN actuals o1   ON o1.symbol = p.symbol AND o1.rn = o.rn - 1
  JOIN actuals t    ON t.symbol = p.symbol AND t.rn = o.rn + p.horizon_day
  JOIN actuals prev ON prev.symbol = p.symbol AND prev.rn = t.rn - 1
)
SELECT model, as_of_date, horizon_day AS day, MAX(actual_date) AS scored_on, COUNT(*) AS symbols,
       ROUND(AVG(ABS(pred - actual))::numeric, 4) AS mae,
       ROUND((PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ABS(pred - actual)))::numeric, 4) AS median_err,
       ROUND(100 * AVG(((pred > prev_actual) = (actual > prev_actual))::int)::numeric, 1) AS da_pct,
       ROUND(AVG(ABS(drift - actual))::numeric, 4) AS drift_mae,
       ROUND(100 * AVG(((drift > prev_actual) = (actual > prev_actual))::int)::numeric, 1) AS drift_da_pct,
       ROUND(100 * AVG((ABS(pred - actual) < ABS(drift - actual))::int)::numeric, 1) AS beats_drift_pct
FROM scored
GROUP BY model, as_of_date, horizon_day
ORDER BY as_of_date, model, horizon_day;
SQL
```

For the all-days figures in the Summary table, drop `horizon_day` from the `SELECT`
and `GROUP BY`.

The flag table (replace the date and model as needed):

```bash
docker exec -i tick_scanner_db psql -U postgres -d ticks <<'SQL'
WITH sp AS (
  SELECT ws.symbol FROM watchlist_symbols ws
  JOIN watchlists w ON w.id = ws.watchlist_id WHERE w.name = 'sp500'
),
run AS (
  SELECT DISTINCT p.symbol, p.last_macd, p.will_become_positive
  FROM forecast_predictions p JOIN sp USING (symbol)
  WHERE p.model = 'macdonly_warm5' AND p.as_of_date = '2026-10-02'
),
after AS (
  SELECT r.symbol, r.will_become_positive, BOOL_OR(s.macd > 0) AS crossed, MAX(s.date) AS through
  FROM run r JOIN stock_cache s ON s.symbol = r.symbol AND s.date > '2026-10-02'
  WHERE r.last_macd < 0
  GROUP BY r.symbol, r.will_become_positive
)
SELECT MAX(through) AS through, COUNT(*) AS negative_at_start,
       COUNT(*) FILTER (WHERE will_become_positive) AS flagged,
       COUNT(*) FILTER (WHERE will_become_positive AND crossed) AS flagged_now_positive,
       COUNT(*) FILTER (WHERE crossed) AS crossed,
       COUNT(*) FILTER (WHERE crossed AND will_become_positive) AS crossed_and_flagged
FROM after;
SQL
```

"Crossed" here means positive on any day since the start, so run it only up to day 5
of the run being scored.

## Keeping the record useful

- Run the forecast from the UI on each trading day you want tracked. Only the last run
  per model and starting date is stored.
- Forecasts are stored only when a forecast is run; a day with no run leaves no row and
  cannot be scored later.
- Record a retrain here when it happens: the stored rows carry the model versions and
  training date, so runs before and after can be told apart.
