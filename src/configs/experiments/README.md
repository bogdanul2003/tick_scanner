# Hyperparameter experiment set (2026-09-04)

Seven arms branched from `../gru_v1_residual_plainloss_warm5.json` — the current best
recipe (residual target, `macd + delta + open-close`, plain MSE, warmup 5). Everything
outside the column below is identical across all seven.

**Every arm is pinned to `seed: 42`**, so differences between them are not confounded by
initialization. That is the whole point of running a control arm here: comparing a seeded
experiment against the unseeded `gru_v1_residual_plainloss_warm5_1` would reintroduce the
~0.8pp run-to-run noise that TS-017 measured.

| Config | Changed from best | Hypothesis |
|---|---|---|
| `gru_v1_exp_baseline` | *nothing* (control at seed 42) | Same-seed reference for the other six |
| `gru_v1_exp_lr_patience2` | `lr_patience: 10 → 2` | The LR scheduler currently **never fires before the checkpoint is chosen** — best epoch is 6-9, patience 10 means the first decay lands around epoch 17-20. This is the only change that makes `lr_factor` mean anything at all |
| `gru_v1_exp_lr_3e4` | `learning_rate: 1e-3 → 3e-4` | An optimum arriving at epoch 6 of 40 is the signature of an LR too high for the loss surface |
| `gru_v1_exp_lr_1e4` | `learning_rate: 1e-3 → 1e-4` | Same, further. ⚠️ **May not converge inside the 40-epoch cap** — if its best epoch is at or near 40, read it as undertrained, not as evidence against low LR |
| `gru_v1_exp_hidden16` | `hidden_size: 32 → 16` | 28,330 params already overfit by epoch 6; capacity may be wasted rather than scarce |
| `gru_v1_exp_hidden8` | `hidden_size: 32 → 8` | Same hypothesis, pushed to ~4x smaller again |
| `gru_v1_exp_hidden16_lr3e4` | `hidden_size: 16`, `learning_rate: 3e-4` | Combination — **run only if both singles look promising**; a combo is not interpretable when the singles are inside the noise |

`hidden_size` going *down* is deliberate. Raising it has a poor prior: more capacity on a
model that peaks at epoch 6 overfits sooner, and the TFT analysis concluded the binding
constraint is signal scarcity, not capacity.

## Run

```bash
# from src/ — ~30 min per arm, ~3.5h for all seven (alphabetical order puts baseline first)
for c in configs/experiments/*.json; do
  ../venv/bin/python scripts/train_forecast_model.py --config "$c"
done
```

## Evaluate

```bash
# from src/ — matches the protocol every table in FORECAST_MODEL_IMPROVEMENTS.md uses
for c in configs/experiments/*.json; do
  n=$(basename "$c" .json)
  echo "===== $n ====="
  ../venv/bin/python scripts/evaluate_forecast_model.py --model-name "$n" \
    --watchlist sp500 --samples 20 --breakdown-by-day --lag-test
done
```

## Reading the results

The measured noise floor (TS-017, n=4) is **sd 0.77pp / range 1.71pp on aggregate DA**, and
per-day `sd` runs 0.32 (day 1) to 1.34 (day 4).

- Shared seed removes the initialization component, so these arms are more comparable than
  four independent runs would be — but **not** noise-free: data shuffling still differs once
  `hidden_size` or LR changes the trajectory, and MPS kernels are not bitwise deterministic.
- Treat anything under **~2pp aggregate DA** as unresolved and worth 2-3 seeds before it is
  written up. Day-1 differences are the exception — day 1 is ~4x more stable and can resolve
  under 1pp.
- Also record the **best epoch** for every arm. For `lr_patience2` and the low-LR arms the
  epoch is as informative as the DA: it says directly whether the change moved the optimum
  away from epoch 6, which is the mechanism being tested.

## Results (2026-10-04)

All seven arms trained at seed 42 on CPU (`--device cpu`) and evaluated with the protocol above
on data through 2026-10-02. Full output: `eval_experiments_2026-10-03.txt` in the repo root.
Drift on the same window: 57.81% aggregate MACD DA, MAE 1.147, per-day DA
80.8 / 62.6 / 52.1 / 47.7 / 45.7.

| Config | MACD DA | MACD MAE | Day 1 / 2 / 3 / 4 / 5 DA | Best epoch |
|---|---|---|---|---|
| `gru_v1_exp_baseline` | 49.82% | 0.941 | 79.8 / 56.5 / 41.4 / 36.3 / 35.1 | 7 |
| `gru_v1_exp_lr_patience2` | 49.82% | 0.941 | identical to baseline | 7 |
| `gru_v1_exp_lr_3e4` | 48.42% | 0.901 | 78.8 / 53.7 / 40.6 / 34.7 / 34.3 | 6 |
| `gru_v1_exp_lr_1e4` | 47.46% | 0.920 | 79.7 / 50.6 / 38.5 / 34.6 / 33.8 | 13 |
| `gru_v1_exp_hidden16` | 55.28% | 0.925 | 80.6 / 62.5 / 48.5 / 43.0 / 41.8 | not captured |
| `gru_v1_exp_hidden8` | 44.19% | 0.982 | 76.7 / 46.2 / 34.2 / 32.5 / 31.4 | 10 |
| `gru_v1_exp_hidden16_lr3e4` | 49.29% | 0.900 | 80.8 / 56.0 / 39.9 / 35.4 / 34.3 | 8 |

- **`lr_patience2` is a no-op.** It saved the same checkpoint as the baseline: with a 5-epoch
  warmup and the optimum at epoch 7, even patience 2 fires after the best epoch.
- **Lower learning rates did not move the optimum much** (epoch 6 at 3e-4, 13 at 1e-4, neither
  near the 40-epoch cap) and gave lower MAE with slightly lower DA.
- **`hidden16` looked like a 5.5pp win and was not.** Reseeded at 7 and 123 it scored 47.97% and
  49.78% (`eval_hidden16_seeds_2026-10-04.txt`); three-seed mean 51.0%, MAE 0.924.

### The noise floor above is too optimistic

The TS-017 figure (sd 0.77pp / range 1.71pp) does not hold here. Three configs trained on seeds
42 / 7 / 123 spanned **5-10pp** of aggregate DA each, almost entirely on days 3-5, and seed 42 was
the best seed in all three. Treat any single-seed DA difference under ~10pp as unresolved and
run at least three seeds per arm. MAE is steadier (range 0.003-0.041 across seeds).

See `../../models/MODEL_CARD.md` §11 and `../../../docs/FORECAST_MODEL_IMPROVEMENTS.md`
("Results log — 2026-10-04") for the other configs, the auxiliary-loss test and seed ensembles.
