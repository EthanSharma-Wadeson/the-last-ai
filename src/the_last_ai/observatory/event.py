"""Structured life-history events for the observatory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from the_last_ai.types import JSONDict


class LifeEventType(str, Enum):
    BIRTH = "birth"
    SURVIVAL = "survival"
    DISAPPEARANCE_SELF = "disappearance_self"
    DISAPPEARANCE_OTHER = "disappearance_other"
    POPULATION_CHANGE = "population_change"
    RESOURCE_DISCOVERED = "resource_discovered"
    RESOURCE_COLLECTED = "resource_collected"
    ENTITY_ENCOUNTERED = "entity_encountered"
    ENTITY_ABSENT_EXPECTED = "entity_absent_expected"
    ACTION = "action"
    SOCIAL_INTERACTION = "social_interaction"
    COMMUNICATION_SENT = "communication_sent"
    COMMUNICATION_RECEIVED = "communication_received"
    COMMUNICATION_VERIFIED = "communication_verified"
    RELATIONSHIP_UPDATED = "relationship_updated"
    MEMORY_ACTIVATED = "memory_activated"
    PREDICTION_ERROR = "prediction_error"
    AFFECTIVE_CHANGE = "affective_change"
    INVESTIGATION = "investigation"
    DECISION = "decision"
    COLLAPSE_STAGE = "collapse_stage"
    FINAL_STATE = "final_state"
    NOTE = "note"


@dataclass
class LifeEvent:
    """One inspectable computational experience in an agent's life."""

    tick: int
    agent_id: str
    event_type: LifeEventType
    event_id: str = field(default_factory=lambda: uuid4().hex[:12])
    state_before: JSONDict = field(default_factory=dict)
    state_after: JSONDict = field(default_factory=dict)
    context: JSONDict = field(default_factory=dict)
    summary: str = ""  # short factual caption for timelines (not interpretive prose)

    def to_dict(self) -> JSONDict:
        return {
            "event_id": self.event_id,
            "tick": self.tick,
            "agent_id": self.agent_id,
            "event_type": self.event_type.value,
            "state_before": dict(self.state_before),
            "state_after": dict(self.state_after),
            "context": dict(self.context),
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> LifeEvent:
        return cls(
            event_id=str(data.get("event_id", uuid4().hex[:12])),
            tick=int(data.get("tick", 0)),
            agent_id=str(data.get("agent_id", "")),
            event_type=LifeEventType(str(data.get("event_type", "note"))),
            state_before=dict(data.get("state_before") or {}),
            state_after=dict(data.get("state_after") or {}),
            context=dict(data.get("context") or {}),
            summary=str(data.get("summary") or ""),
        )
