"""Finite communication memory with optional forgetting by age/capacity."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from the_last_ai.communication.message import Message
from the_last_ai.types import JSONDict


@dataclass
class CommunicationRecord:
    """One remembered communication episode (may outlive the other agent)."""

    message_id: str
    tick: int
    sender_id: str
    receiver_id: str
    tokens: list[str]
    concepts: list[str] = field(default_factory=list)
    claimed_position: tuple[int, int] | None = None
    sender_reliability_at_receipt: float = 0.5
    verified: bool | None = None  # None = not yet checked
    outcome: str | None = None
    role: str = "receiver"  # sender | receiver
    historical: bool = False  # True once counterpart is no longer active

    def to_dict(self) -> JSONDict:
        return {
            "message_id": self.message_id,
            "tick": self.tick,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "tokens": list(self.tokens),
            "concepts": list(self.concepts),
            "claimed_position": list(self.claimed_position) if self.claimed_position else None,
            "sender_reliability_at_receipt": self.sender_reliability_at_receipt,
            "verified": self.verified,
            "outcome": self.outcome,
            "role": self.role,
            "historical": self.historical,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> CommunicationRecord:
        claimed = data.get("claimed_position")
        return cls(
            message_id=str(data["message_id"]),
            tick=int(data.get("tick", 0)),
            sender_id=str(data["sender_id"]),
            receiver_id=str(data["receiver_id"]),
            tokens=[str(t) for t in data.get("tokens", [])],
            concepts=[str(c) for c in data.get("concepts", [])],
            claimed_position=(int(claimed[0]), int(claimed[1])) if claimed else None,
            sender_reliability_at_receipt=float(data.get("sender_reliability_at_receipt", 0.5)),
            verified=data.get("verified"),
            outcome=data.get("outcome"),
            role=str(data.get("role", "receiver")),
            historical=bool(data.get("historical", False)),
        )

    @classmethod
    def from_message(
        cls,
        message: Message,
        *,
        role: str,
        reliability: float,
    ) -> CommunicationRecord:
        return cls(
            message_id=message.message_id,
            tick=message.tick,
            sender_id=message.sender_id,
            receiver_id=message.receiver_id,
            tokens=list(message.tokens),
            concepts=list(message.concepts),
            claimed_position=message.claimed_position,
            sender_reliability_at_receipt=reliability,
            role=role,
        )


@dataclass
class CommunicationMemory:
    """Bounded buffer of communication records (finite; drops oldest)."""

    capacity: int = 48
    records: deque[CommunicationRecord] = field(default_factory=deque)

    def __post_init__(self) -> None:
        if not isinstance(self.records, deque):
            self.records = deque(self.records, maxlen=max(1, self.capacity))
        else:
            self.records = deque(self.records, maxlen=max(1, self.capacity))

    def add(self, record: CommunicationRecord) -> None:
        self.records.append(record)

    def mark_historical(self, absent_ids: set[str]) -> int:
        """Flag records involving disappeared agents; do not delete them."""
        marked = 0
        for record in self.records:
            if record.sender_id in absent_ids or record.receiver_id in absent_ids:
                if not record.historical:
                    record.historical = True
                    marked += 1
        return marked

    def find(self, message_id: str) -> CommunicationRecord | None:
        for record in self.records:
            if record.message_id == message_id:
                return record
        return None

    def token_frequencies(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            for token in record.tokens:
                counts[token] = counts.get(token, 0) + 1
        return counts

    def summary(self) -> JSONDict:
        verified = sum(1 for r in self.records if r.verified is True)
        failed = sum(1 for r in self.records if r.verified is False)
        historical = sum(1 for r in self.records if r.historical)
        return {
            "capacity": self.capacity,
            "count": len(self.records),
            "verified_count": verified,
            "failed_count": failed,
            "historical_count": historical,
            "active_count": len(self.records) - historical,
            "token_frequencies": self.token_frequencies(),
            "recent": [r.to_dict() for r in list(self.records)[-8:]],
        }

    def to_dict(self) -> JSONDict:
        return {
            "capacity": self.capacity,
            "records": [r.to_dict() for r in self.records],
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> CommunicationMemory:
        capacity = int(data.get("capacity", 48))
        mem = cls(capacity=capacity)
        for item in data.get("records", []):
            mem.add(CommunicationRecord.from_dict(item))
        return mem
