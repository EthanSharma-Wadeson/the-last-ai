"""Per-entity internal representations."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.memory.decay import decayed_strength
from the_last_ai.types import JSONDict


@dataclass
class EntityRepresentation:
    entity_id: str
    familiarity: float = 0.0
    last_seen: int = 0
    expected_location: tuple[int, int] | None = None
    interaction_value: float = 0.0
    uncertainty: float = 1.0
    strength: float = 0.0
    sighting_count: int = 0
    interaction_count: int = 0
    retrieval_count: int = 0

    def strength_at(self, tick: int, decay_lambda: float) -> float:
        elapsed = max(0, tick - self.last_seen)
        return decayed_strength(self.strength, elapsed, decay_lambda)

    def as_vector(self, tick: int, decay_lambda: float) -> tuple[float, ...]:
        """Compact numerical representation for similarity comparisons."""
        loc_x, loc_y = self.expected_location if self.expected_location else (-1.0, -1.0)
        return (
            self.familiarity,
            self.interaction_value,
            1.0 - self.uncertainty,
            self.strength_at(tick, decay_lambda),
            float(loc_x),
            float(loc_y),
            float(self.sighting_count),
        )

    def to_dict(self) -> JSONDict:
        return {
            "entity_id": self.entity_id,
            "familiarity": self.familiarity,
            "last_seen": self.last_seen,
            "expected_location": list(self.expected_location) if self.expected_location else None,
            "interaction_value": self.interaction_value,
            "uncertainty": self.uncertainty,
            "strength": self.strength,
            "sighting_count": self.sighting_count,
            "interaction_count": self.interaction_count,
            "retrieval_count": self.retrieval_count,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> EntityRepresentation:
        loc = data.get("expected_location")
        return cls(
            entity_id=str(data["entity_id"]),
            familiarity=float(data.get("familiarity", 0.0)),
            last_seen=int(data.get("last_seen", 0)),
            expected_location=(int(loc[0]), int(loc[1])) if loc else None,
            interaction_value=float(data.get("interaction_value", 0.0)),
            uncertainty=float(data.get("uncertainty", 1.0)),
            strength=float(data.get("strength", 0.0)),
            sighting_count=int(data.get("sighting_count", 0)),
            interaction_count=int(data.get("interaction_count", 0)),
            retrieval_count=int(data.get("retrieval_count", 0)),
        )
