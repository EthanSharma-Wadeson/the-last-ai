"""Relevance-based memory retrieval."""

from __future__ import annotations

from the_last_ai.memory.entity import EntityRepresentation
from the_last_ai.types import Position


def retrieval_score(
    memory: EntityRepresentation,
    *,
    tick: int,
    query_position: Position,
    decay_lambda: float,
    location_weight: float = 0.35,
    recency_weight: float = 0.25,
    importance_weight: float = 0.40,
) -> float:
    """Score a memory for retrieval given the agent's current context."""
    strength = memory.strength_at(tick, decay_lambda)
    if strength <= 0:
        return 0.0

    if memory.expected_location is None:
        location_score = 0.0
    else:
        dist = query_position.manhattan(Position(*memory.expected_location))
        location_score = 1.0 / (1.0 + dist)

    elapsed = max(0, tick - memory.last_seen)
    recency_score = 1.0 / (1.0 + elapsed)

    importance = 0.5 * memory.familiarity + 0.5 * max(0.0, memory.interaction_value)
    importance = max(0.0, min(1.0, importance))

    return (
        location_weight * location_score
        + recency_weight * recency_score
        + importance_weight * importance
    ) * strength
