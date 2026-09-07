"""World event system for controlled, logged simulation changes."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from the_last_ai.types import JSONDict, Position


class EventType(str, Enum):
    AGENT_APPEAR = "agent_appear"
    AGENT_DISAPPEAR = "agent_disappear"
    AGENT_RESTORE = "agent_restore"
    RESOURCE_SPAWN = "resource_spawn"
    RESOURCE_REMOVE = "resource_remove"
    ENVIRONMENT_CHANGE = "environment_change"
    CUSTOM = "custom"


@dataclass(slots=True)
class WorldEvent:
    tick: int
    event_type: EventType
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSONDict:
        return {
            "tick": self.tick,
            "event_type": self.event_type.value,
            "payload": self.payload,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> WorldEvent:
        return cls(
            tick=int(data["tick"]),
            event_type=EventType(data["event_type"]),
            payload=dict(data.get("payload", {})),
        )


@dataclass
class EventLog:
    events: list[WorldEvent] = field(default_factory=list)

    def record(self, event: WorldEvent) -> None:
        self.events.append(event)

    def appear(self, tick: int, agent_id: str, position: Position) -> None:
        self.record(
            WorldEvent(
                tick=tick,
                event_type=EventType.AGENT_APPEAR,
                payload={"agent_id": agent_id, "position": position.as_tuple()},
            )
        )

    def disappear(self, tick: int, agent_id: str, *, reversible: bool = True) -> None:
        self.record(
            WorldEvent(
                tick=tick,
                event_type=EventType.AGENT_DISAPPEAR,
                payload={"agent_id": agent_id, "reversible": reversible},
            )
        )

    def restore(self, tick: int, agent_id: str, position: Position) -> None:
        self.record(
            WorldEvent(
                tick=tick,
                event_type=EventType.AGENT_RESTORE,
                payload={"agent_id": agent_id, "position": position.as_tuple()},
            )
        )

    def resource_spawn(self, tick: int, position: Position) -> None:
        self.record(
            WorldEvent(
                tick=tick,
                event_type=EventType.RESOURCE_SPAWN,
                payload={"position": position.as_tuple()},
            )
        )

    def resource_remove(self, tick: int, position: Position, *, agent_id: str | None = None) -> None:
        payload: dict[str, Any] = {"position": position.as_tuple()}
        if agent_id is not None:
            payload["agent_id"] = agent_id
        self.record(
            WorldEvent(
                tick=tick,
                event_type=EventType.RESOURCE_REMOVE,
                payload=payload,
            )
        )

    def to_dict(self) -> JSONDict:
        return {"events": [event.to_dict() for event in self.events]}

    @classmethod
    def from_dict(cls, data: JSONDict) -> EventLog:
        return cls(events=[WorldEvent.from_dict(item) for item in data.get("events", [])])
