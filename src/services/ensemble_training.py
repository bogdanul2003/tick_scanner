"""
Background retraining of a seed ensemble.

One job at a time: refresh the training data, train one model per seed in parallel on
CPU, evaluate the averaged ensemble, then point the ensemble at the new versions. The
versions being served are only switched once every seed has produced a model; old
versions stay on disk and keep serving until then.

The job's progress is kept in a file rather than in memory because the dev server runs
with auto-reload: a restart would otherwise leave the UI polling a job nobody remembers.
See docs/ENSEMBLE_PRODUCTION_PLAN.md.
"""
import json
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from services import ensemble_service
from services.ensemble_service import MODELS_DIR, SRC_DIR

logger = logging.getLogger(__name__)

JOB_PATH = os.path.join(MODELS_DIR, "ensemble_retrain_job.json")
LOG_DIR = os.path.join(os.path.dirname(SRC_DIR), "train_logs")

ACTIVE_STATES = ("refreshing_data", "training", "evaluating")
EVAL_SAMPLES = 20
POLL_SECONDS = 2

_start_lock = threading.Lock()


class RetrainInProgressError(RuntimeError):
    """A retrain job is already running; only one runs at a time."""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _read_job() -> Optional[Dict[str, Any]]:
    if not os.path.isfile(JOB_PATH):
        return None
    try:
        with open(JOB_PATH) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        logger.error(f"Ignoring unreadable retrain job file at {JOB_PATH}: {e}")
        return None


def _write_job(job: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(JOB_PATH), exist_ok=True)
    tmp_path = f"{JOB_PATH}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(job, f, indent=2)
    os.replace(tmp_path, JOB_PATH)


def get_status() -> Dict[str, Any]:
    """
    The current or most recent retrain job, or {"state": "idle"} if there has been none.

    A job recorded as active by a different server process is reported as failed: the
    thread driving it died with that process, so it will never finish.
    """
    job = _read_job()
    if job is None:
        return {"state": "idle"}

    if job["state"] in ACTIVE_STATES and job.get("server_pid") != os.getpid():
        job["state"] = "failed"
        job["error"] = "The server restarted while this job was running; start it again."
        job["finished_at"] = _now()
        _write_job(job)

    end = job.get("finished_at") or _now()
    job["elapsed_seconds"] = int(
        (datetime.fromisoformat(end) - datetime.fromisoformat(job["started_at"])).total_seconds()
    )
    return job


def start_retrain(ensemble_id: str) -> Dict[str, Any]:
    """Start retraining `ensemble_id` in the background and return the new job."""
    from models.lstm_forecaster import get_latest_model_version

    with _start_lock:
        if get_status()["state"] in ACTIVE_STATES:
            raise RetrainInProgressError("A retrain is already running")

        registry = ensemble_service.load_registry()
        ensemble = ensemble_service.resolve_ensemble(ensemble_id, registry)
        seeds = list(registry["seeds"])

        # Reserve the versions up front: runs of one config that finish together
        # would otherwise each resolve the same "latest + 1".
        first = get_latest_model_version(MODELS_DIR, ensemble["model_name"]) + 1
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        job = {
            "id": f"{ensemble['id']}_{stamp}",
            "ensemble_id": ensemble["id"],
            "label": ensemble["label"],
            "model_name": ensemble["model_name"],
            "state": "refreshing_data",
            "started_at": _now(),
            "finished_at": None,
            "error": None,
            "warning": None,
            "server_pid": os.getpid(),
            "previous_versions": ensemble["versions"],
            "new_versions": [first + i for i in range(len(seeds))],
            "metrics": None,
            "runs": [
                {
                    "seed": seed,
                    "version": first + i,
                    "state": "pending",
                    "log": os.path.join(LOG_DIR, f"retrain_{ensemble['id']}_{stamp}_seed{seed}.log"),
                }
                for i, seed in enumerate(seeds)
            ],
        }
        _write_job(job)
        threading.Thread(target=_run_job, args=(job, ensemble), daemon=True).start()
        return job


def _run_job(job: Dict[str, Any], ensemble: Dict[str, Any]) -> None:
    """Drive one retrain job to completed/failed. Never raises."""
    try:
        with open(ensemble["config_path"]) as f:
            config = json.load(f)

        _refresh_training_data(config)

        job["state"] = "training"
        _write_job(job)
        _train_all(job, ensemble)

        job["state"] = "evaluating"
        _write_job(job)
        try:
            job["metrics"] = _evaluate(job, config)
        except Exception as e:
            # The models trained; a failed measurement is not a reason to discard them.
            logger.error(f"Ensemble evaluation failed for {job['id']}: {e}")
            job["warning"] = f"Trained, but evaluation failed: {e}"

        state = ensemble_service.load_state()
        state[job["ensemble_id"]] = {
            "versions": job["new_versions"],
            "previous_versions": job["previous_versions"],
            "seeds": [run["seed"] for run in job["runs"]],
            "trained_at": _now(),
            "metrics": job["metrics"],
        }
        ensemble_service.save_state(state)
        job["state"] = "completed"
    except Exception as e:
        logger.error(f"Retrain job {job['id']} failed: {e}")
        job["state"] = "failed"
        job["error"] = str(e)
    job["finished_at"] = _now()
    _write_job(job)


def _refresh_training_data(config: Dict[str, Any]) -> None:
    """
    Bring the cache up to date for the config's symbols in one bulk call.

    The training script fetches missing dates one symbol at a time, and here several
    copies of it start at once; priming the cache first means they only read.
    """
    from db_utils import get_watchlist_symbols
    from macd_utils import get_latest_market_date, get_macd_for_range_bulk

    symbols = _config_symbols(config, get_watchlist_symbols)
    end_date = get_latest_market_date()
    get_macd_for_range_bulk(symbols, end_date - timedelta(days=int(config.get("days", 365))), end_date)


def _config_symbols(config: Dict[str, Any], get_watchlist_symbols) -> List[str]:
    symbols = config.get("symbols")
    if symbols:
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]
        return [s.upper() for s in symbols]
    return sorted(get_watchlist_symbols(config.get("watchlist") or "sp500"))


def _train_all(job: Dict[str, Any], ensemble: Dict[str, Any]) -> None:
    """Train every seed in parallel; raise if any run fails or leaves no Core ML model."""
    os.makedirs(LOG_DIR, exist_ok=True)
    # One thread per process: on CPU that is as fast as the default for this model
    # size, and without it parallel runs contend for the same cores (MODEL_CARD §12).
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")

    procs = []
    for run in job["runs"]:
        log = open(run["log"], "w")
        proc = subprocess.Popen(
            [sys.executable, os.path.join("scripts", "train_forecast_model.py"),
             "--config", ensemble["config_path"], "--seed", str(run["seed"]),
             "--device", "cpu", "--model-version", str(run["version"])],
            cwd=SRC_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
        run["state"] = "training"
        procs.append((run, proc, log))
    _write_job(job)

    failed = None
    while any(run["state"] == "training" for run, _, _ in procs) and failed is None:
        time.sleep(POLL_SECONDS)
        changed = False
        for run, proc, log in procs:
            if run["state"] != "training" or proc.poll() is None:
                continue
            log.close()
            produced = os.path.exists(os.path.join(MODELS_DIR, f"{job['model_name']}_{run['version']}.mlpackage"))
            if proc.returncode == 0 and produced:
                run["state"] = "done"
            else:
                run["state"] = "failed"
                reason = (f"exit code {proc.returncode}" if proc.returncode != 0
                          else "no Core ML model was produced")
                failed = f"Training seed {run['seed']} failed ({reason}); see {run['log']}"
            changed = True
        if changed:
            _write_job(job)

    if failed is not None:
        for run, proc, log in procs:
            if run["state"] == "training":
                proc.terminate()
                proc.wait()
                log.close()
                run["state"] = "cancelled"
        raise RuntimeError(failed)


def _evaluate(job: Dict[str, Any], config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Score the new ensemble on the config's watchlist; None when it has none."""
    watchlist = config.get("watchlist")
    if not watchlist or config.get("symbols"):
        return None

    json_out = os.path.join(LOG_DIR, f"retrain_{job['id']}_eval.json")
    log_path = os.path.join(LOG_DIR, f"retrain_{job['id']}_eval.log")
    env = dict(os.environ, OMP_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONUNBUFFERED="1")
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, os.path.join("scripts", "evaluate_ensemble.py"),
             "--model-name", job["model_name"],
             "--versions", ",".join(str(v) for v in job["new_versions"]),
             "--watchlist", watchlist, "--samples", str(EVAL_SAMPLES),
             "--breakdown-by-day", "--json-out", json_out],
            cwd=SRC_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        raise RuntimeError(f"exit code {result.returncode}; see {log_path}")

    with open(json_out) as f:
        measured = json.load(f)
    return {
        "mae": round(measured["mae"], 4),
        "directional_accuracy": round(measured["directional_accuracy"], 4),
        "measured": datetime.now().date().isoformat(),
    }
