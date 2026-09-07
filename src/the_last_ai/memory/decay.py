"""Exponential memory strength decay."""

from __future__ import annotations

import math


def decayed_strength(strength_0: float, elapsed: int, decay_lambda: float) -> float:
    """strength(t) = strength_0 * exp(-lambda * elapsed_time)."""
    if elapsed <= 0:
        return max(0.0, strength_0)
    if decay_lambda <= 0:
        return max(0.0, strength_0)
    return max(0.0, strength_0 * math.exp(-decay_lambda * elapsed))
