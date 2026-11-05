"""
Freshness backtest: does a seed ensemble get worse as its training data ages, and so
would retraining it help?

The job trains the ensemble's config several times, each as if on an earlier day (its
data ends AGES_WEEKS before the evaluation window), and scores every copy on the same
recent forecast origins, which none of them has seen. If the copies trained on older
data are clearly worse, accuracy decays with age and the served ensemble, which is
`served_age_weeks` old, has lost about slope x age; if they are no worse, retraining
would only swap one seed draw for another.

The served ensemble and the drift and flat baselines are scored on the same origins,
so the result also says whether the model beats drift on recent data at all; that is a
different question from freshness, and retraining the same config does not answer it.

Like the retrain job (services/ensemble_training.py), one job runs at a time and its
progress is kept in a file, so an auto-reloading dev server does not lose it. See
docs/FORECAST_LIVE_TRACKING.md, "Should the model be retrained?".
"""
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from services import ensemble_service
from services.ensemble_service import MODELS_DIR, SRC_DIR

logger = logging.getLogger(__name__)

JOB_PATH = os.path.join(MODELS_DIR, "freshness_backtest_job.json")
WORK_DIR = os.path.join(MODELS_DIR, "backtest")
LOG_DIR = os.path.join(os.path.dirname(SRC_DIR), "train_logs")

ACTIVE_STATES = ("refreshing_data", "training", "scoring")
# How much older than the evaluation window each copy's data is. 0 is the freshest
# model possible for the window; 13 weeks is a quarter.
AGES_WEEKS = (0, 4, 8, 13)
ORIGINS = 20  # forecast origins scored, each with all of its forecast days closed
HORIZON = 5
MAX_PARALLEL = 6  # training processes at once, one thread each (10 cores here)
BOOTSTRAP_REPS = 2000
# The smallest expected gain in error that makes a retrain worth it.
MIN_EFFECT_PCT = 2.0
# train_forecast_model.py carves this share of the training part off as validation.
VALIDATION_SPLIT = 0.2
POLL_SECONDS = 2

_start_lock = threading.Lock()


class BacktestInProgressError(RuntimeError):
    """A backtest or a retrain is already running."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_job() -> Optional[Dict[str, Any]]:
    if not os.path.isfile(JOB_PATH):
        return None
    try:
        with open(JOB_PATH) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.error(f"Ignoring unreadable backtest job file at {JOB_PATH}: {e}")
        return None


def _write_job(job: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(JOB_PATH), exist_ok=True)
    tmp_path = f"{JOB_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, JOB_PATH)


def is_active() -> bool:
    return get_status()["state"] in ACTIVE_STATES


def get_status() -> Dict[str, Any]:
    """
    The current or most recent backtest, or {"state": "idle"} if there has been none.
    A job left active by another server process is reported as failed.
    """
    job = _read_job()
    if job is None:
        return {"state": "idle"}

    if job["state"] in ACTIVE_STATES and job.get("server_pid") != os.getpid():
        job["state"] = "failed"
        job["error"] = "The server restarted while this backtest was running; start it again."
        job["finished_at"] = _now()
        _write_job(job)

    end = job.get("finished_at") or _now()
    job["elapsed_seconds"] = int(
        (datetime.fromisoformat(end) - datetime.fromisoformat(job["started_at"])).total_seconds()
    )
    return job


def start_backtest(ensemble_id: Optional[str] = None) -> Dict[str, Any]:
    """Start the freshness backtest of `ensemble_id` in the background; return the job."""
    from services import ensemble_training

    with _start_lock:
        if is_active():
            raise BacktestInProgressError("A backtest is already running")
        if ensemble_training.get_status()["state"] in ensemble_training.ACTIVE_STATES:
            raise BacktestInProgressError("A retrain is running; start the backtest when it has finished")

        registry = ensemble_service.load_registry()
        ensemble = ensemble_service.resolve_ensemble(ensemble_id, registry)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = {
            "id": f"{ensemble['id']}_{stamp}",
            "ensemble_id": ensemble["id"],
            "label": ensemble["label"],
            "model_name": ensemble["model_name"],
            "served_versions": ensemble["versions"],
            "served_trained_at": ensemble.get("trained_at"),
            "state": "refreshing_data",
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "warning": None,
            "server_pid": os.getpid(),
            "seeds": list(registry["seeds"]),
            "ages_weeks": list(AGES_WEEKS),
            "runs": [],
            "result": None,
        }
        _write_job(job)
        threading.Thread(target=_run_job, args=(job, ensemble), daemon=True).start()
        return job


def _run_job(job: Dict[str, Any], ensemble: Dict[str, Any]) -> None:
    """Drive one backtest to completed/failed. Never raises."""
    from services.ensemble_training import _config_symbols, _refresh_training_data
    from db_utils import get_trading_days, get_watchlist_symbols
    from macd_utils import get_latest_market_date

    work_dir = os.path.join(WORK_DIR, job["id"])
    try:
        with open(ensemble["config_path"]) as f:
            config = json.load(f)
        _refresh_training_data(config)
        symbols = _config_symbols(config, get_watchlist_symbols)

        latest = get_latest_market_date()
        lookback = int(config.get("days", 365)) + 7 * max(AGES_WEEKS) + 120
        calendar = get_trading_days(symbols, latest - timedelta(days=lookback), latest)
        window = plan_window(calendar, ORIGINS, HORIZON, AGES_WEEKS)
        job["window"] = {k: (v.isoformat() if isinstance(v, date) else v) for k, v in window.items()
                         if k not in ("cutoffs", "calendar")}
        job["cutoffs"] = {str(a): c.isoformat() for a, c in window["cutoffs"].items()}

        job["runs"] = [
            {"age_weeks": age, "cutoff": window["cutoffs"][age].isoformat(), "seed": seed,
             "version": k + 1, "state": "pending",
             "dir": os.path.join(work_dir, f"age{age}"),
             "log": os.path.join(LOG_DIR, f"backtest_{job['id']}_age{age}_seed{seed}.log")}
            for age in AGES_WEEKS for k, seed in enumerate(job["seeds"])
        ]
        job["state"] = "training"
        _write_job(job)
        _train_all(job, ensemble)

        job["state"] = "scoring"
        _write_job(job)
        raw = _score(job, ensemble, symbols, calendar, window, work_dir)

        served_cutoff = _served_cutoff(job.get("served_trained_at"), calendar)
        served_age = (latest - served_cutoff).days / 7.0 if served_cutoff else None
        job["result"] = summarize_backtest(
            raw, AGES_WEEKS, served_age_weeks=served_age,
            cutoffs={a: c.isoformat() for a, c in window["cutoffs"].items()},
        )
        job["result"]["served"] = {
            "versions": job["served_versions"],
            "trained_at": job.get("served_trained_at"),
            "cutoff": served_cutoff.isoformat() if served_cutoff else None,
            "age_weeks_now": round(served_age, 1) if served_age is not None else None,
            "data_periods": data_periods(calendar, served_cutoff, config) if served_cutoff else None,
        }
        job["state"] = "completed"
        # The backtest models answer one question and are not served. The raw scores
        # are kept with the logs, so the stats can be recomputed without retraining.
        shutil.copy(os.path.join(work_dir, "scores.json"), os.path.join(LOG_DIR, f"backtest_{job['id']}_scores.json"))
        shutil.rmtree(work_dir, ignore_errors=True)
    except Exception as e:
        logger.error(f"Freshness backtest {job['id']} failed: {e}")
        job["state"] = "failed"
        job["error"] = str(e)
    job["finished_at"] = _now()
    _write_job(job)


def plan_window(calendar: List[date], origins: int, horizon: int, ages_weeks) -> Dict[str, Any]:
    """
    The evaluation origins (the last `origins` trading days whose `horizon` forecast
    days have all closed) and each age's training cutoff: the trading day before the
    first origin, moved back by the age and snapped to a trading day.
    """
    if len(calendar) < origins + horizon + 1:
        raise ValueError("Not enough cached trading days for the backtest window")
    chosen = calendar[-(origins + horizon):-horizon]
    fresh_cutoff = calendar[calendar.index(chosen[0]) - 1]
    cutoffs = {}
    for age in ages_weeks:
        target = fresh_cutoff - timedelta(days=7 * age)
        earlier = [d for d in calendar if d <= target]
        if not earlier:
            raise ValueError(f"No cached trading day on or before {target} for the {age}-week copy")
        cutoffs[age] = earlier[-1]
    return {"origins": [d.isoformat() for d in chosen], "first_origin": chosen[0],
            "last_origin": chosen[-1], "last_target": calendar[-1], "calendar": calendar,
            "cutoffs": cutoffs}


def data_periods(calendar: List[date], cutoff: date, config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Which dates a model trained on `cutoff` with `config` actually learned from, by the
    training script's splits: weights are fitted on the first part, the checkpoint is
    chosen on a later validation part, and with the time strategy the last
    test_split is held out entirely. Approximate: it assumes a symbol with a bar on
    every trading day.
    """
    start = cutoff - timedelta(days=int(config.get("days", 365)))
    days = [d for d in calendar if start <= d <= cutoff]
    if len(days) < 10:
        return None
    if (config.get("split_strategy") or "symbol") != "time":
        return {"weights_from": days[0].isoformat(), "weights_to": cutoff.isoformat()}
    test_cut = int(len(days) * (1 - float(config.get("test_split", 0.15))))
    val_cut = int(test_cut * (1 - VALIDATION_SPLIT))
    embargo = int(config.get("forecast_horizon", HORIZON))
    return {
        "weights_from": days[0].isoformat(),
        "weights_to": days[val_cut - 1].isoformat(),
        "checkpoint_from": days[min(val_cut + embargo, test_cut - 1)].isoformat(),
        "checkpoint_to": days[test_cut - 1].isoformat(),
        "held_out_from": days[test_cut].isoformat() if test_cut < len(days) else None,
        "held_out_to": cutoff.isoformat(),
    }


def _served_cutoff(trained_at: Optional[str], calendar: List[date]) -> Optional[date]:
    """The last trading day on or before the served ensemble was trained."""
    if not trained_at:
        return None
    trained = datetime.fromisoformat(trained_at).date()
    earlier = [d for d in calendar if d <= trained]
    return earlier[-1] if earlier else None


def _train_all(job: Dict[str, Any], ensemble: Dict[str, Any]) -> None:
    """Train every (age, seed) copy, MAX_PARALLEL at a time; raise if any fails."""
    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    pending = list(job["runs"])
    running = []
    failed = None
    while (pending or running) and failed is None:
        while pending and len(running) < MAX_PARALLEL:
            run = pending.pop(0)
            os.makedirs(run["dir"], exist_ok=True)
            log = open(run["log"], "w")
            proc = subprocess.Popen(
                [sys.executable, os.path.join("scripts", "train_forecast_model.py"),
                 "--config", ensemble["config_path"], "--seed", str(run["seed"]),
                 "--device", "cpu", "--end-date", run["cutoff"], "--output-dir", run["dir"],
                 "--model-version", str(run["version"]), "--skip-coreml"],
                cwd=SRC_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
            )
            run["state"] = "training"
            running.append((run, proc, log))
            _write_job(job)

        time.sleep(POLL_SECONDS)
        for entry in list(running):
            run, proc, log = entry
            if proc.poll() is None:
                continue
            log.close()
            running.remove(entry)
            if proc.returncode == 0 and os.path.exists(_member_pt(run, job["model_name"])):
                run["state"] = "done"
            else:
                run["state"] = "failed"
                failed = (f"Training the {run['age_weeks']}-week copy, seed {run['seed']}, failed "
                          f"(exit code {proc.returncode}); see {run['log']}")
            _write_job(job)

    if failed is not None:
        for run, proc, log in running:
            proc.terminate()
            proc.wait()
            log.close()
            run["state"] = "cancelled"
        for run in pending:
            run["state"] = "cancelled"
        _write_job(job)
        raise RuntimeError(failed)


def _member_pt(run: Dict[str, Any], model_name: str) -> str:
    return os.path.join(run["dir"], f"{model_name}_{run['version']}.pt")


def _score(job, ensemble, symbols, calendar, window, work_dir) -> Dict[str, Any]:
    """Run scripts/evaluate_freshness.py on every age and the served ensemble."""
    arms = []
    for age in AGES_WEEKS:
        arms.append({"id": f"age{age}", "members": [
            _member_pt(r, job["model_name"]) for r in job["runs"] if r["age_weeks"] == age]})
    served = [os.path.join(MODELS_DIR, f"{job['model_name']}_{v}.pt") for v in job["served_versions"]]
    if all(os.path.exists(p) for p in served):
        arms.append({"id": "served", "members": served})
    else:
        job["warning"] = "The served ensemble's .pt checkpoints are missing, so it is not scored."

    first = date.fromisoformat(window["origins"][0])
    plan = {
        "symbols": symbols,
        "origins": window["origins"],
        "horizon": HORIZON,
        "calendar": [d.isoformat() for d in calendar if d >= first],
        "arms": arms,
    }
    os.makedirs(work_dir, exist_ok=True)
    plan_path = os.path.join(work_dir, "plan.json")
    out_path = os.path.join(work_dir, "scores.json")
    log_path = os.path.join(LOG_DIR, f"backtest_{job['id']}_score.log")
    with open(plan_path, "w") as f:
        json.dump(plan, f)
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, os.path.join("scripts", "evaluate_freshness.py"),
             "--plan", plan_path, "--json-out", out_path],
            cwd=SRC_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Scoring failed (exit code {result.returncode}); see {log_path}")
    with open(out_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Statistics: pure functions of the scoring output, so they can be tested alone.
# ---------------------------------------------------------------------------

def _totals(series: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
    """Per-origin totals over all forecast days, and per-day totals over all origins."""
    arrays = {k: np.asarray(series[k], dtype=float) for k in ("n", "err", "err_pct", "dir")}
    return {
        "by_origin": {k: v.sum(axis=1) for k, v in arrays.items()},
        "by_day": {k: v.sum(axis=0) for k, v in arrays.items()},
    }


def _ratio(num, den):
    return float(num / den) if den else None


def _describe(series: Dict[str, List[List[float]]]) -> Dict[str, Any]:
    t = _totals(series)
    o, d = t["by_origin"], t["by_day"]
    n = o["n"].sum()
    return {
        "forecasts": int(n),
        "err_pct": _ratio(o["err_pct"].sum(), n),
        "mae": _ratio(o["err"].sum(), n),
        "direction_accuracy": _ratio(o["dir"].sum(), n),
        "per_day": [
            {"day": h + 1, "err_pct": _ratio(d["err_pct"][h], d["n"][h]), "mae": _ratio(d["err"][h], d["n"][h]),
             "direction_accuracy": _ratio(d["dir"][h], d["n"][h])}
            for h in range(len(d["n"]))
        ],
    }


def _resampled_error(by_origin, counts: np.ndarray) -> np.ndarray:
    """Error (% of price) of one series on each bootstrap draw of origin days."""
    return (counts @ by_origin["err_pct"]) / (counts @ by_origin["n"])


def _slope(ages, rel):
    """Least-squares slope through the origin: relative error change per week of age."""
    ages = np.asarray(ages, dtype=float)
    return (rel * ages).sum(axis=-1) / (ages ** 2).sum()


def _interval(values) -> List[float]:
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def summarize_backtest(raw: Dict[str, Any], ages_weeks, served_age_weeks: Optional[float] = None,
                       cutoffs: Optional[Dict[int, str]] = None, reps: int = BOOTSTRAP_REPS,
                       seed: int = 0) -> Dict[str, Any]:
    """
    Stats and a verdict from evaluate_freshness.py's output.

    The decision metric is error as a % of the origin day's close: MACD scales with
    price, so a dollar MAE is mostly the few highest-priced symbols. Uncertainty has
    two sources and the intervals carry both: which days happened to be in the window
    (resampled as whole days, since every symbol moves on the same day) and which
    seeds each copy happened to get (each copy's error is perturbed by its seed noise,
    estimated from the spread of its single-seed members). Without the second, a copy
    that is better by seed luck looks significantly better.
    """
    series = raw["series"]
    ages = [a for a in ages_weeks if f"age{a}" in series]
    if 0 not in ages or len(ages) < 2:
        raise ValueError("The backtest needs the 0-week copy and at least one older one")
    older = [a for a in ages if a > 0]
    origins = raw["origins"]
    n_origins = len(origins)

    arms = {}
    for a in ages:
        arm = _describe(series[f"age{a}"])
        arm["age_weeks"] = a
        arm["cutoff"] = (cutoffs or {}).get(a)
        members = sorted(k for k in series if k.startswith(f"age{a}#"))
        arm["seeds_err_pct"] = [_describe(series[k])["err_pct"] for k in members]
        arms[a] = arm

    totals = {a: _totals(series[f"age{a}"])["by_origin"] for a in ages}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_origins, size=(reps, n_origins))
    counts = np.zeros((reps, n_origins))
    np.add.at(counts, (np.arange(reps)[:, None], draws), 1)

    # Seed noise of one ensemble's error, relative: the spread of its single-seed
    # members over the square root of their number, averaged over the copies.
    seed_noise = []
    for a in ages:
        singles = [v for v in arms[a]["seeds_err_pct"] if v is not None]
        if len(singles) >= 2:
            seed_noise.append(np.std(singles, ddof=1) / math.sqrt(len(singles)) / arms[a]["err_pct"])
    noise = float(np.mean(seed_noise)) if seed_noise else 0.0

    fresh = arms[0]["err_pct"]
    perturbed = {a: _resampled_error(totals[a], counts) * (1 + rng.normal(0, noise, reps)) for a in ages}
    rel_draws = {}
    for a in older:
        rel_draws[a] = perturbed[a] / perturbed[0] - 1
        arms[a]["vs_fresh_pct"] = 100 * (arms[a]["err_pct"] / fresh - 1)
        arms[a]["vs_fresh_ci"] = [100 * v for v in _interval(rel_draws[a])]
    arms[0]["vs_fresh_pct"] = 0.0
    arms[0]["vs_fresh_ci"] = None

    rel_point = np.array([arms[a]["err_pct"] / fresh - 1 for a in older])
    slope = float(_slope(older, rel_point))
    slope_draws = _slope(older, np.stack([rel_draws[a] for a in older], axis=1))

    oldest = max(older)
    fresh_by_origin = totals[0]["err_pct"] / totals[0]["n"]
    oldest_by_origin = totals[oldest]["err_pct"] / totals[oldest]["n"]
    staleness = {
        "per_week_pct": 100 * slope,
        "per_week_ci": [100 * v for v in _interval(slope_draws)],
        "ensemble_noise_pct": 100 * noise if seed_noise else None,
        # How far a retrained ensemble's error can land from the old one's by seed
        # luck alone (95%): two ensembles, each with that noise.
        "seed_luck_pct": 100 * 1.96 * math.sqrt(2) * noise if seed_noise else None,
        "threshold_pct": MIN_EFFECT_PCT,
        "fresh_beats_oldest_days": int((fresh_by_origin < oldest_by_origin).sum()),
        "oldest_age_weeks": oldest,
        "days": n_origins,
        "cost_now_pct": None,
        "cost_now_ci": None,
    }
    if served_age_weeks is not None:
        staleness["cost_now_pct"] = 100 * slope * served_age_weeks
        staleness["cost_now_ci"] = [100 * v * served_age_weeks for v in _interval(slope_draws)]

    baselines = {name: _describe(series[name]) for name in ("drift", "flat") if name in series}
    served = _describe(series["served"]) if "served" in series else None
    if served and "drift" in baselines:
        served["vs_drift_err_pct"] = served["err_pct"] / baselines["drift"]["err_pct"]
        served["vs_drift_mae"] = served["mae"] / baselines["drift"]["mae"]
    if "drift" in baselines:
        arms[0]["vs_drift_err_pct"] = arms[0]["err_pct"] / baselines["drift"]["err_pct"]

    result = {
        "window": {"first_origin": origins[0], "last_origin": origins[-1], "origins": n_origins,
                   "symbols": raw.get("symbols_scored"), "horizon": raw.get("horizon")},
        "arms": [arms[a] for a in ages],
        "served_scores": served,
        "baselines": baselines,
        "staleness": staleness,
    }
    result["verdict"] = verdict(result, served_age_weeks)
    return result


def verdict(result: Dict[str, Any], served_age_weeks: Optional[float]) -> Dict[str, Any]:
    """
    'retrain', 'not_needed' or 'inconclusive', with the reasons in plain words.

    Retrain when the error a model of the served ensemble's age is expected to have
    lost is at least MIN_EFFECT_PCT and its 95% interval is above zero. Not needed when
    even the top of that interval is below MIN_EFFECT_PCT.
    """
    s = result["staleness"]
    served = result.get("served_scores")
    threshold = s["threshold_pct"]
    reasons = []

    if served_age_weeks is None or s["cost_now_pct"] is None:
        code = "inconclusive"
        reasons.append("The served ensemble's training date is unknown, so its age cannot be judged.")
    else:
        cost, (lo, hi) = s["cost_now_pct"], s["cost_now_ci"]
        reasons.append(
            f"Each week of data age changed the error by {s['per_week_pct']:+.2f}% "
            f"(95% range {s['per_week_ci'][0]:+.2f}% to {s['per_week_ci'][1]:+.2f}%). "
            f"At {served_age_weeks:.1f} weeks the served ensemble is expected to be {cost:+.1f}% "
            f"off a fresh one (range {lo:+.1f}% to {hi:+.1f}%)."
        )
        if cost >= threshold and lo > 0:
            code = "retrain"
        elif hi < threshold:
            code = "not_needed"
        else:
            code = "inconclusive"
        if s["per_week_pct"] > 0 and code != "retrain":
            weeks = threshold / s["per_week_pct"] - served_age_weeks
            if 0 < weeks <= 52:
                reasons.append(f"At this rate, a retrain would gain {threshold:.0f}% in about {weeks:.0f} more week(s).")
            elif weeks > 52:
                reasons.append(f"At this rate, a retrain would not gain {threshold:.0f}% within a year.")
        elif s["per_week_pct"] <= 0:
            reasons.append("Copies trained on older data did no worse on this window: no sign of ageing.")
        if s.get("seed_luck_pct"):
            reasons.append(
                f"A retrain also re-draws the seeds: by luck alone its error can land up to "
                f"{s['seed_luck_pct']:.0f}% better or worse than now. Check it after retraining."
            )

    reasons.append(
        f"The {s['days']}-day comparison: the fresh copy had the lower error on "
        f"{s['fresh_beats_oldest_days']} of {s['days']} days against the {s['oldest_age_weeks']}-week copy."
    )

    if served and served.get("vs_drift_err_pct") is not None:
        ratio = served["vs_drift_err_pct"]
        if ratio >= 1:
            reasons.append(
                f"The served ensemble does not beat drift on this window (error {ratio:.2f}x drift's)."
                + (" Retraining the same config will not fix that; it is the config or the market, not age."
                   if code != "retrain" else "")
            )
        else:
            reasons.append(f"The served ensemble's error is {ratio:.2f}x drift's on this window ({100 * (1 - ratio):.0f}% better).")

    headline = {
        "retrain": "Retrain: the model has measurably lost accuracy with age.",
        "not_needed": "No retrain needed: fresher data would not measurably help.",
        "inconclusive": "Inconclusive: the effect of age is within the noise. Run again in a few weeks.",
    }[code]
    return {"code": code, "headline": headline, "reasons": reasons}
