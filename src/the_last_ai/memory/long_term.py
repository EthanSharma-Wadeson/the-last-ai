"""Long-term entity memory with capacity limits, decay, and retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.memory.entity import EntityRepresentation
from the_last_ai.memory.retrieval import retrieval_score
from the_last_ai.types import JSONDict, Position


@dataclass
class MemoryConfig:
    capacity: int = 16  # 0 = unlimited
    decay_lambda: float = 0.002
    forget_threshold: float = 0.01
    familiarity_rate: float = 0.08
    value_learning_rate: float = 0.1
    location_ema: float = 0.4
    strength_boost: float = 0.25
    interaction_proximity: int = 1
    retrieval_top_k: int = 3
    seek_strength_threshold: float = 0.2
    seek_probability: float = 0.55

    def to_dict(self) -> JSONDict:
        return {
            "capacity": self.capacity,
            "decay_lambda": self.decay_lambda,
            "forget_threshold": self.forget_threshold,
            "familiarity_rate": self.familiarity_rate,
            "value_learning_rate": self.value_learning_rate,
            "location_ema": self.location_ema,
            "strength_boost": self.strength_boost,
            "interaction_proximity": self.interaction_proximity,
            "retrieval_top_k": self.retrieval_top_k,
            "seek_strength_threshold": self.seek_strength_threshold,
            "seek_probability": self.seek_probability,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> MemoryConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass
class LongTermMemory:
    config: MemoryConfig = field(default_factory=MemoryConfig)
    entities: dict[str, EntityRepresentation] = field(default_factory=dict)
    replacement_count: int = 0
    retrieval_events: int = 0
    forgotten_ids: list[str] = field(default_factory=list)

    def get(self, entity_id: str) -> EntityRepresentation | None:
        return self.entities.get(entity_id)

    def observe_entity(
        self,
        entity_id: str,
        absolute_position: Position,
        *,
        tick: int,
        distance: int,
    ) -> EntityRepresentation:
        memory = self.entities.get(entity_id)
        if memory is None:
            self._ensure_capacity()
            memory = EntityRepresentation(entity_id=entity_id)
            self.entities[entity_id] = memory

        memory.sighting_count += 1
        memory.last_seen = tick
        memory.familiarity = min(1.0, memory.familiarity + self.config.familiarity_rate)
        memory.uncertainty = max(0.05, memory.uncertainty * 0.85)
        # Asymptotic strength growth so repeated interaction remains informative.
        memory.strength = min(
            1.0,
            memory.strength + self.config.strength_boost * (1.0 - memory.strength),
        )
        # Blend in interaction history so socially proximal entities stay stronger.
        memory.strength = min(
            1.0,
            max(memory.strength, 0.5 * memory.familiarity + 0.5 * memory.interaction_value),
        )

        loc = absolute_position.as_tuple()
        if memory.expected_location is None:
            memory.expected_location = loc
        else:
            alpha = self.config.location_ema
            ex, ey = memory.expected_location
            memory.expected_location = (
                int(round((1 - alpha) * ex + alpha * loc[0])),
                int(round((1 - alpha) * ey + alpha * loc[1])),
            )

        if distance <= self.config.interaction_proximity:
            memory.interaction_count += 1
            delta = self.config.value_learning_rate * (1.0 - memory.interaction_value)
            memory.interaction_value = min(1.0, memory.interaction_value + delta)

        return memory

    def _ensure_capacity(self) -> None:
        if self.config.capacity <= 0:
            return
        while len(self.entities) >= self.config.capacity:
            # Replace weakest (lowest strength * familiarity) memory.
            victim_id = min(
                self.entities,
                key=lambda eid: (
                    self.entities[eid].strength * (0.5 + self.entities[eid].familiarity),
                    self.entities[eid].last_seen,
                    eid,
                ),
            )
            del self.entities[victim_id]
            self.replacement_count += 1
            self.forgotten_ids.append(victim_id)

    def tick_decay(self, tick: int) -> list[str]:
        """Apply forgetting based on decayed strength. Returns removed ids."""
        removed: list[str] = []
        for entity_id, memory in list(self.entities.items()):
            current = memory.strength_at(tick, self.config.decay_lambda)
            # Store absolute strength refreshed at last_seen for persistence bookkeeping.
            # Soft-forget when decayed strength falls below threshold.
            if current < self.config.forget_threshold and memory.sighting_count > 0:
                # Only forget if not seen very recently.
                if tick - memory.last_seen > 0:
                    del self.entities[entity_id]
                    removed.append(entity_id)
                    self.forgotten_ids.append(entity_id)
        return removed

    def retrieve(
        self,
        query_position: Position,
        *,
        tick: int,
        top_k: int | None = None,
        exclude_visible: set[str] | None = None,
    ) -> list[EntityRepresentation]:
        top_k = self.config.retrieval_top_k if top_k is None else top_k
        exclude_visible = exclude_visible or set()
        scored: list[tuple[float, str, EntityRepresentation]] = []
        for entity_id, memory in self.entities.items():
            if entity_id in exclude_visible:
                continue
            score = retrieval_score(
                memory,
                tick=tick,
                query_position=query_position,
                decay_lambda=self.config.decay_lambda,
            )
            if score > 0:
                scored.append((score, entity_id, memory))
        scored.sort(key=lambda item: (-item[0], item[1]))
        selected = [memory for _, _, memory in scored[:top_k]]
        for memory in selected:
            memory.retrieval_count += 1
        self.retrieval_events += len(selected)
        return selected

    def strength_map(self, tick: int) -> dict[str, float]:
        return {
            entity_id: memory.strength_at(tick, self.config.decay_lambda)
            for entity_id, memory in self.entities.items()
        }

    def summary(self, tick: int) -> JSONDict:
        strengths = self.strength_map(tick)
        return {
            "entity_count": len(self.entities),
            "replacement_count": self.replacement_count,
            "retrieval_events": self.retrieval_events,
            "forgotten_count": len(self.forgotten_ids),
            "mean_strength": (sum(strengths.values()) / len(strengths)) if strengths else 0.0,
            "strengths": strengths,
            "familiarities": {eid: m.familiarity for eid, m in self.entities.items()},
            "interaction_values": {eid: m.interaction_value for eid, m in self.entities.items()},
        }

    def to_dict(self) -> JSONDict:
        return {
            "config": self.config.to_dict(),
            "entities": {eid: mem.to_dict() for eid, mem in self.entities.items()},
            "replacement_count": self.replacement_count,
            "retrieval_events": self.retrieval_events,
            "forgotten_ids": list(self.forgotten_ids),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> LongTermMemory:
        ltm = cls(config=MemoryConfig.from_dict(data.get("config", {})))
        for eid, mem_data in data.get("entities", {}).items():
            ltm.entities[eid] = EntityRepresentation.from_dict(mem_data)
        ltm.replacement_count = int(data.get("replacement_count", 0))
        ltm.retrieval_events = int(data.get("retrieval_events", 0))
        ltm.forgotten_ids = list(data.get("forgotten_ids", []))
        return ltm
