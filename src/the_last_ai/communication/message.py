"""Structured communication messages (not free-form strings)."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from the_last_ai.types import JSONDict


@dataclass(slots=True)
class Message:
    """A primitive symbolic utterance from sender to receiver."""

    sender_id: str
    receiver_id: str
    tokens: list[str]
    tick: int
    message_id: str = field(default_factory=lambda: uuid4().hex[:12])
    sender_position: tuple[int, int] | None = None
    receiver_position: tuple[int, int] | None = None
    claimed_position: tuple[int, int] | None = None
    concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "message_id": self.message_id,
            "tick": self.tick,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "tokens": list(self.tokens),
            "concepts": list(self.concepts),
            "sender_position": list(self.sender_position) if self.sender_position else None,
            "receiver_position": list(self.receiver_position) if self.receiver_position else None,
            "claimed_position": list(self.claimed_position) if self.claimed_position else None,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> Message:
        def _pos(value) -> tuple[int, int] | None:
            if not value:
                return None
            return (int(value[0]), int(value[1]))

        return cls(
            message_id=str(data.get("message_id", uuid4().hex[:12])),
            sender_id=str(data["sender_id"]),
            receiver_id=str(data["receiver_id"]),
            tokens=[str(t) for t in data.get("tokens", [])],
            tick=int(data.get("tick", 0)),
            sender_position=_pos(data.get("sender_position")),
            receiver_position=_pos(data.get("receiver_position")),
            claimed_position=_pos(data.get("claimed_position")),
            concepts=[str(c) for c in data.get("concepts", [])],
        )

    def render_line(self) -> str:
        content = " ".join(self.tokens)
        return f"{self.sender_id}: {content} → {self.receiver_id}"
