"""Deterministic random number generation for reproducible experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class ExperimentRNG:
    """Thin wrapper around NumPy Generator with seed tracking."""

    seed: int
    _rng: np.random.Generator

    @classmethod
    def from_seed(cls, seed: int) -> ExperimentRNG:
        return cls(seed=seed, _rng=np.random.default_rng(seed))

    def integers(self, low: int, high: int | None = None, size: int | None = None) -> int | np.ndarray:
        return self._rng.integers(low, high, size=size)

    def choice(self, options: list | tuple | np.ndarray, size: int | None = None):
        return self._rng.choice(options, size=size)

    def random(self) -> float:
        return float(self._rng.random())

    def shuffle(self, sequence: list) -> None:
        self._rng.shuffle(sequence)

    def spawn(self, name: str) -> ExperimentRNG:
        """Derive a child RNG from seed and a stable name."""
        digest = hashlib.sha256(f"{self.seed}:{name}".encode()).hexdigest()
        child_seed = int(digest[:8], 16)
        return ExperimentRNG.from_seed(child_seed)
