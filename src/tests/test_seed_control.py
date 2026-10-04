"""
Tests for TS-017 (docs/plans/SEED_CONTROL_PLAN.md) — seed control via
models.lstm_forecaster.seed_everything and MACDForecasterTrainer(seed=...).

No training run, no DB, no Core ML: this only checks that the RNGs are seeded
before weight initialization and that the seed survives a save/load round trip.
Run from src/:
    python -m unittest tests.test_seed_control
"""
import os
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lstm_forecaster import MACDForecasterTrainer, seed_everything


def _make_trainer(**overrides):
    kwargs = dict(seq_length=5, forecast_horizon=5, hidden_size=4, num_layers=1,
                  architecture="gru")
    kwargs.update(overrides)
    return MACDForecasterTrainer(**kwargs)


def _weights(trainer):
    return torch.cat([p.detach().flatten().cpu() for p in trainer.model.parameters()])


class SeedEverythingTest(unittest.TestCase):
    def test_none_is_a_no_op_and_reports_it(self):
        self.assertIsNone(seed_everything(None))

    def test_returns_the_applied_seed(self):
        self.assertEqual(seed_everything(123), 123)

    def test_seeds_all_three_generators(self):
        seed_everything(7)
        first = (random.random(), np.random.rand(), torch.rand(1).item())
        seed_everything(7)
        second = (random.random(), np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_different_seeds_give_different_streams(self):
        seed_everything(1)
        a = np.random.rand(5)
        seed_everything(2)
        b = np.random.rand(5)
        self.assertFalse(np.array_equal(a, b))


class TrainerSeedTest(unittest.TestCase):
    """The point of the ticket: two runs of the same config must start identically."""

    def test_same_seed_gives_identical_initial_weights(self):
        self.assertTrue(torch.equal(_weights(_make_trainer(seed=42)),
                                    _weights(_make_trainer(seed=42))))

    def test_different_seed_gives_different_initial_weights(self):
        self.assertFalse(torch.equal(_weights(_make_trainer(seed=42)),
                                     _weights(_make_trainer(seed=43))))

    def test_seed_applies_regardless_of_prior_rng_state(self):
        """Constructing with a seed must not depend on what consumed randomness
        earlier — otherwise 'same seed' would still drift between runs."""
        seed_everything(999)
        [np.random.rand() for _ in range(17)]
        a = _weights(_make_trainer(seed=42))
        seed_everything(111)
        b = _weights(_make_trainer(seed=42))
        self.assertTrue(torch.equal(a, b))

    def test_unseeded_default_is_preserved(self):
        """seed=None must remain the historical nondeterministic behaviour, so
        existing configs are unaffected."""
        self.assertIsNone(_make_trainer().seed)

    def test_shuffle_is_covered_by_the_seed(self):
        """prepare_sequences shuffles with np.random.shuffle; the seed must reach it."""
        seed_everything(5)
        a = np.arange(20)
        np.random.shuffle(a)
        seed_everything(5)
        b = np.arange(20)
        np.random.shuffle(b)
        self.assertTrue(np.array_equal(a, b))


class SeedPersistenceTest(unittest.TestCase):
    def test_seed_survives_save_load_round_trip(self):
        trainer = _make_trainer(seed=2024)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            trainer.save(path)
            reloaded = _make_trainer()
            reloaded.load(path)
        self.assertEqual(reloaded.seed, 2024)

    def test_checkpoint_without_a_seed_key_loads_as_none(self):
        """Every model trained before TS-017 lacks the key; loading must not raise."""
        trainer = _make_trainer(seed=7)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pt")
            trainer.save(path)
            cp = torch.load(path, map_location="cpu", weights_only=False)
            del cp["seed"]
            torch.save(cp, path)
            reloaded = _make_trainer()
            reloaded.load(path)
        self.assertIsNone(reloaded.seed)


if __name__ == "__main__":
    unittest.main()
