"""Individual Agent Life Observatory — instrumentation & biography layer."""

from the_last_ai.observatory.affect import (
    AffectiveState,
    affect_label,
    compute_affect,
    relationship_label,
)
from the_last_ai.observatory.event import LifeEvent, LifeEventType
from the_last_ai.observatory.export import export_agent_life, write_agent_biography
from the_last_ai.observatory.history import AgentLifeHistory
from the_last_ai.observatory.recorder import ObservatoryRecorder
from the_last_ai.observatory.summary import generate_life_summary

__all__ = [
    "AffectiveState",
    "AgentLifeHistory",
    "LifeEvent",
    "LifeEventType",
    "ObservatoryRecorder",
    "affect_label",
    "compute_affect",
    "export_agent_life",
    "generate_life_summary",
    "relationship_label",
    "write_agent_biography",
]
