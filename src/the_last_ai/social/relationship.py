"""Learned directed relationships: experience → trust / utility / strength."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.social.types import InteractionKind
from the_last_ai.types import JSONDict


@dataclass
class Relationship:
    """A's learned representation of interacting with B (not a friendship label)."""

    other_id: str
    interaction_count: int = 0
    positive_interactions: int = 0
    negative_interactions: int = 0
    last_interaction_tick: int = 0
    predicted_utility: float = 0.0
    trust: float = 0.5  # neutral prior; learned from outcomes
    uncertainty: float = 1.0
    strength: float = 0.0
    cooperate_count: int = 0
    compete_count: int = 0
    share_count: int = 0
    communicate_count: int = 0
    # Information-channel reliability (distinct from moral/social trust).
    successful_messages: int = 0
    failed_messages: int = 0
    information_reliability: float = 0.5

    def update_information_reliability(
        self,
        *,
        success: bool,
        lr: float = 0.2,
    ) -> None:
        """Update measurable information reliability from verification outcomes."""
        target = 1.0 if success else 0.0
        self.information_reliability = (1.0 - lr) * self.information_reliability + lr * target
        self.information_reliability = max(0.0, min(1.0, self.information_reliability))
        if success:
            self.successful_messages += 1
        else:
            self.failed_messages += 1

    def update_from_outcome(
        self,
        *,
        tick: int,
        valence: float,
        energy_delta: float,
        intent: InteractionKind | None,
        utility_lr: float = 0.2,
        trust_lr: float = 0.15,
        strength_boost: float = 0.12,
    ) -> None:
        self.interaction_count += 1
        self.last_interaction_tick = tick

        if valence > 0.05:
            self.positive_interactions += 1
        elif valence < -0.05:
            self.negative_interactions += 1

        if intent == InteractionKind.COOPERATE:
            self.cooperate_count += 1
        elif intent == InteractionKind.COMPETE:
            self.compete_count += 1
        elif intent == InteractionKind.SHARE:
            self.share_count += 1
        elif intent == InteractionKind.COMMUNICATE:
            self.communicate_count += 1

        # Predicted utility: EMA of experienced valence / energy.
        signal = 0.6 * valence + 0.4 * max(-1.0, min(1.0, energy_delta / 10.0))
        self.predicted_utility = (1 - utility_lr) * self.predicted_utility + utility_lr * signal

        # Trust moves toward 1 with positive outcomes, toward 0 with negative.
        target_trust = 0.5 + 0.5 * max(-1.0, min(1.0, valence))
        self.trust = (1 - trust_lr) * self.trust + trust_lr * target_trust
        self.trust = max(0.0, min(1.0, self.trust))

        self.uncertainty = max(0.05, self.uncertainty * 0.88)
        self.strength = min(1.0, self.strength + strength_boost * (1.0 - self.strength))
        # Strength also tracks consistency of positive history.
        if self.interaction_count > 0:
            pos_rate = self.positive_interactions / self.interaction_count
            self.strength = min(1.0, max(self.strength, 0.5 * pos_rate + 0.5 * self.trust))

    def cooperation_rate(self) -> float:
        if self.interaction_count == 0:
            return 0.0
        return self.cooperate_count / self.interaction_count

    def conflict_rate(self) -> float:
        if self.interaction_count == 0:
            return 0.0
        return self.compete_count / self.interaction_count

    def to_dict(self) -> JSONDict:
        return {
            "other_id": self.other_id,
            "interaction_count": self.interaction_count,
            "positive_interactions": self.positive_interactions,
            "negative_interactions": self.negative_interactions,
            "last_interaction_tick": self.last_interaction_tick,
            "predicted_utility": self.predicted_utility,
            "trust": self.trust,
            "uncertainty": self.uncertainty,
            "strength": self.strength,
            "cooperate_count": self.cooperate_count,
            "compete_count": self.compete_count,
            "share_count": self.share_count,
            "communicate_count": self.communicate_count,
            "successful_messages": self.successful_messages,
            "failed_messages": self.failed_messages,
            "information_reliability": self.information_reliability,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> Relationship:
        return cls(
            other_id=str(data["other_id"]),
            interaction_count=int(data.get("interaction_count", 0)),
            positive_interactions=int(data.get("positive_interactions", 0)),
            negative_interactions=int(data.get("negative_interactions", 0)),
            last_interaction_tick=int(data.get("last_interaction_tick", 0)),
            predicted_utility=float(data.get("predicted_utility", 0.0)),
            trust=float(data.get("trust", 0.5)),
            uncertainty=float(data.get("uncertainty", 1.0)),
            strength=float(data.get("strength", 0.0)),
            cooperate_count=int(data.get("cooperate_count", 0)),
            compete_count=int(data.get("compete_count", 0)),
            share_count=int(data.get("share_count", 0)),
            communicate_count=int(data.get("communicate_count", 0)),
            successful_messages=int(data.get("successful_messages", 0)),
            failed_messages=int(data.get("failed_messages", 0)),
            information_reliability=float(data.get("information_reliability", 0.5)),
        )


@dataclass
class RelationshipStore:
    """Directed relationships from one agent to others."""

    relationships: dict[str, Relationship] = field(default_factory=dict)

    def get(self, other_id: str) -> Relationship | None:
        return self.relationships.get(other_id)

    def get_or_create(self, other_id: str) -> Relationship:
        if other_id not in self.relationships:
            self.relationships[other_id] = Relationship(other_id=other_id)
        return self.relationships[other_id]

    def update_from_outcome(
        self,
        other_id: str,
        *,
        tick: int,
        valence: float,
        energy_delta: float,
        intent: InteractionKind | None,
    ) -> Relationship:
        rel = self.get_or_create(other_id)
        rel.update_from_outcome(
            tick=tick,
            valence=valence,
            energy_delta=energy_delta,
            intent=intent,
        )
        return rel

    def strongest(self, n: int = 3) -> list[Relationship]:
        ranked = sorted(
            self.relationships.values(),
            key=lambda r: (-r.strength, -r.predicted_utility, r.other_id),
        )
        return ranked[:n]

    def summary(self) -> JSONDict:
        return {
            "count": len(self.relationships),
            "mean_trust": (
                sum(r.trust for r in self.relationships.values()) / len(self.relationships)
                if self.relationships
                else 0.5
            ),
            "mean_strength": (
                sum(r.strength for r in self.relationships.values()) / len(self.relationships)
                if self.relationships
                else 0.0
            ),
            "relationships": {oid: r.to_dict() for oid, r in self.relationships.items()},
        }

    def to_dict(self) -> JSONDict:
        return {"relationships": {oid: r.to_dict() for oid, r in self.relationships.items()}}

    @classmethod
    def from_dict(cls, data: JSONDict) -> RelationshipStore:
        store = cls()
        for oid, rel_data in data.get("relationships", {}).items():
            store.relationships[oid] = Relationship.from_dict(rel_data)
        return store
