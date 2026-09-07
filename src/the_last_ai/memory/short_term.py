"""Minimal short-term memory for recent observations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from the_last_ai.types import JSONDict, Observation


@dataclass
class ShortTermMemory:
    """High-capacity buffer with optional rapid age-based decay."""

    capacity: int = 32
    max_age: int | None = 24
    _items: deque[Observation] = field(default_factory=deque, init=False, repr=False)

    def __post_init__(self) -> None:
        self._items = deque(maxlen=max(1, self.capacity))

    def push(self, observation: Observation) -> None:
        self._items.append(observation)
        self._prune(observation.tick)

    def _prune(self, current_tick: int) -> None:
        if self.max_age is None:
            return
        while self._items and (current_tick - self._items[0].tick) > self.max_age:
            self._items.popleft()

    def recent(self, n: int | None = None) -> list[Observation]:
        if n is None:
            return list(self._items)
        if n <= 0:
            return []
        return list(self._items)[-n:]

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def to_dict(self) -> JSONDict:
        return {
            "capacity": self.capacity,
            "max_age": self.max_age,
            "size": len(self._items),
            "last_tick": self._items[-1].tick if self._items else None,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> ShortTermMemory:
        return cls(
            capacity=int(data.get("capacity", 32)),
            max_age=data.get("max_age", 24),
        )
