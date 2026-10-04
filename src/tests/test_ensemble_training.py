"""
Tests for the background ensemble retrain job (services/ensemble_training.py).

The invariant that matters: the versions an ensemble serves change only when every
seed produced a Core ML model. A failed or half-finished retrain must leave the
ensemble exactly as it was.

No training runs — subprocess, the data refresh and the evaluation are stubbed, and
the job is driven synchronously.
Run from src/:
    python -m unittest tests.test_ensemble_training
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services import ensemble_service as es
from services import ensemble_training as et
from scripts.train_forecast_model import resolve_save_version


class _FakeProc:
    """A finished training subprocess; optionally leaves the Core ML model behind."""

    def __init__(self, cmd, models_dir, returncode=0, produce=True):
        self.returncode = returncode
        version = cmd[cmd.index("--model-version") + 1]
        if produce:
            os.makedirs(os.path.join(models_dir, f"model_a_{version}.mlpackage"))

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def wait(self):
        return self.returncode


class _JobFixture(unittest.TestCase):

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.models_dir = os.path.join(tmp.name, "models")
        os.makedirs(self.models_dir)
        config_path = os.path.join(tmp.name, "a.json")
        with open(config_path, "w") as f:
            json.dump({"model_name": "model_a", "watchlist": "sp500", "days": 600}, f)

        self.ensemble = {"id": "a", "label": "A", "model_name": "model_a", "config_path": config_path,
                         "versions": [1, 2, 3], "signal_type": "macd"}
        registry = {"default": "a", "seeds": [42, 7, 123], "ensembles": [self.ensemble]}

        self.popen_outcomes = {}  # seed -> kwargs for _FakeProc
        self.launched = []

        def fake_popen(cmd, **kwargs):
            self.launched.append((cmd, kwargs))
            seed = int(cmd[cmd.index("--seed") + 1])
            return _FakeProc(cmd, self.models_dir, **self.popen_outcomes.get(seed, {}))

        patches = [
            mock.patch.object(et, "JOB_PATH", os.path.join(self.models_dir, "job.json")),
            mock.patch.object(et, "LOG_DIR", os.path.join(tmp.name, "logs")),
            mock.patch.object(et, "MODELS_DIR", self.models_dir),
            mock.patch.object(et, "POLL_SECONDS", 0),
            mock.patch.object(es, "STATE_PATH", os.path.join(self.models_dir, "state.json")),
            mock.patch.object(es, "load_registry", lambda: registry),
            mock.patch.object(es, "resolve_ensemble", lambda ensemble_id, registry=None: self._resolve(ensemble_id)),
            mock.patch.object(et, "_refresh_training_data", lambda config: None),
            mock.patch.object(et, "_evaluate", lambda job, config: {"mae": 0.9, "directional_accuracy": 0.51}),
            mock.patch.object(et.subprocess, "Popen", fake_popen),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def _resolve(self, ensemble_id):
        if ensemble_id != "a":
            raise es.EnsembleNotFoundError(f"Unknown ensemble: {ensemble_id}")
        return self.ensemble

    def _start(self):
        """start_retrain with the worker thread run inline."""
        class _InlineThread:
            def __init__(self, target, args, daemon):
                self._run = lambda: target(*args)

            def start(self):
                self._run()

        with mock.patch.object(et.threading, "Thread", _InlineThread):
            return et.start_retrain("a")


class RetrainJobTest(_JobFixture):

    def test_successful_retrain_switches_the_served_versions(self):
        for v in (1, 2, 3):
            os.makedirs(os.path.join(self.models_dir, f"model_a_{v}.mlpackage"))

        self._start()
        status = et.get_status()

        self.assertEqual(status["state"], "completed")
        self.assertEqual(status["new_versions"], [4, 5, 6])  # reserved above the existing 1-3
        self.assertEqual([r["state"] for r in status["runs"]], ["done", "done", "done"])
        self.assertEqual(status["metrics"]["mae"], 0.9)

        served = es.load_state()["a"]
        self.assertEqual(served["versions"], [4, 5, 6])
        self.assertEqual(served["previous_versions"], [1, 2, 3])
        self.assertEqual(served["seeds"], [42, 7, 123])

    def test_each_seed_trains_on_cpu_single_threaded_with_its_reserved_version(self):
        self._start()
        self.assertEqual(len(self.launched), 3)
        for (cmd, kwargs), seed, version in zip(self.launched, (42, 7, 123), (1, 2, 3)):
            self.assertEqual(cmd[cmd.index("--seed") + 1], str(seed))
            self.assertEqual(cmd[cmd.index("--model-version") + 1], str(version))
            self.assertEqual(cmd[cmd.index("--device") + 1], "cpu")
            self.assertEqual(kwargs["env"]["OMP_NUM_THREADS"], "1")

    def test_a_failed_seed_leaves_the_ensemble_untouched(self):
        self.popen_outcomes[7] = {"returncode": 1, "produce": False}
        self._start()
        status = et.get_status()

        self.assertEqual(status["state"], "failed")
        self.assertIn("seed 7", status["error"])
        self.assertEqual(es.load_state(), {})

    def test_a_run_that_exits_cleanly_without_a_coreml_model_counts_as_failed(self):
        # train_forecast_model.py exits 0 when coremltools is missing: .pt only.
        self.popen_outcomes[123] = {"produce": False}
        self._start()
        status = et.get_status()

        self.assertEqual(status["state"], "failed")
        self.assertIn("no Core ML model", status["error"])
        self.assertEqual(es.load_state(), {})

    def test_failed_evaluation_still_switches_but_records_a_warning(self):
        def boom(job, config):
            raise RuntimeError("exit code 1")

        with mock.patch.object(et, "_evaluate", boom):
            self._start()
        status = et.get_status()

        self.assertEqual(status["state"], "completed")
        self.assertIn("evaluation failed", status["warning"])
        self.assertIsNone(es.load_state()["a"]["metrics"])
        self.assertEqual(es.load_state()["a"]["versions"], [1, 2, 3])

    def test_unknown_ensemble_starts_nothing(self):
        with self.assertRaises(es.EnsembleNotFoundError):
            et.start_retrain("zzz")
        self.assertEqual(et.get_status(), {"state": "idle"})


class RetrainStatusTest(_JobFixture):

    def _write_active_job(self, server_pid):
        et._write_job({"id": "a_x", "ensemble_id": "a", "state": "training", "server_pid": server_pid,
                       "started_at": "2026-10-04T10:00:00", "finished_at": None})

    def test_idle_before_any_job(self):
        self.assertEqual(et.get_status(), {"state": "idle"})

    def test_only_one_job_at_a_time(self):
        self._write_active_job(os.getpid())
        with self.assertRaises(et.RetrainInProgressError):
            et.start_retrain("a")
        self.assertEqual(self.launched, [])

    def test_job_orphaned_by_a_server_restart_is_reported_failed(self):
        self._write_active_job(os.getpid() + 1)
        status = et.get_status()
        self.assertEqual(status["state"], "failed")
        self.assertIn("restarted", status["error"])
        # and it no longer blocks a new job
        self._start()
        self.assertEqual(et.get_status()["state"], "completed")


class ResolveSaveVersionTest(unittest.TestCase):

    def test_defaults_to_the_next_free_version(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(resolve_save_version(d, "m"), 1)
            open(os.path.join(d, "m_4.pt"), "w").close()
            self.assertEqual(resolve_save_version(d, "m"), 5)

    def test_requested_version_is_used_when_free(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "m_4.pt"), "w").close()
            self.assertEqual(resolve_save_version(d, "m", 7), 7)

    def test_existing_version_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "m_4.pt"), "w").close()
            with self.assertRaisesRegex(ValueError, "already exists"):
                resolve_save_version(d, "m", 4)
            with self.assertRaises(ValueError):
                resolve_save_version(d, "m", 0)


if __name__ == "__main__":
    unittest.main()
