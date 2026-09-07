"""World-model prediction structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from the_last_ai.types import JSONDict


class PredictionKind(str, Enum):
    AGENT_POSITION = "agent_position"
    AGENT_DIRECTION = "agent_direction"
    RESOURCE_PRESENCE = "resource_presence"
    AGENT_PRESENCE = "agent_presence"  # will a remembered agent be visible?
    SOCIAL_INTENT = "social_intent"


@dataclass(slots=True)
class Prediction:
    kind: PredictionKind
    subject_id: str  # entity id, or "env" for environment
    tick_made: int
    tick_target: int
    value: object
    confidence: float
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "kind": self.kind.value,
            "subject_id": self.subject_id,
            "tick_made": self.tick_made,
            "tick_target": self.tick_target,
            "value": self.value,
            "confidence": self.confidence,
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class PredictionError:
    prediction: Prediction
    observed_value: object
    error: float  # 0 = perfect, 1 = maximally wrong
    tick: int

    def to_dict(self) -> JSONDict:
        return {
            "prediction": self.prediction.to_dict(),
            "observed_value": self.observed_value,
            "error": self.error,
            "tick": self.tick,
        }
