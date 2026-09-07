"""Social agent: bidirectional interaction is a core behavioural primitive."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.agents.features import action_toward, encode_state
from the_last_ai.agents.memory_agent import MemoryAgent, _direction_to_target
from the_last_ai.agents.motivation import MotivationConfig, evaluate_drives
from the_last_ai.agents.value_learner import LearningConfig
from the_last_ai.communication.memory import CommunicationMemory
from the_last_ai.communication.system import PendingClaim
from the_last_ai.loss.tracker import LossTracker
from the_last_ai.memory.long_term import MemoryConfig
from the_last_ai.rng import ExperimentRNG
from the_last_ai.social.relationship import RelationshipStore
from the_last_ai.social.types import ACTION_TO_INTERACTION, InteractionKind, InteractionOutcome
from the_last_ai.types import (
    MOVEMENT_ACTIONS,
    SOCIAL_ACTION_VALUES,
    Action,
    JSONDict,
    Observation,
    Position,
)


@dataclass
class SocialConfig:
    """Policy parameters for social action selection (not hard-coded friend/enemy labels)."""

    interact_radius: int = 1
    approach_trust_threshold: float = 0.55
    avoid_trust_threshold: float = 0.35
    interact_probability: float = 0.7
    cooperate_bias: float = 0.0  # experimental control knob; 0 = purely learned
    enable_indirect_memory_hook: bool = False  # Phase 3.5: association transfer via shared contacts
    # Symbolic communication layer (payloads + memory + reliability).
    enable_symbolic_communication: bool = True
    max_message_tokens: int = 2
    communication_energy_cost: float = 0.5
    communication_bias: float = 0.0  # raises communicate_score for controlled trials

    def to_dict(self) -> JSONDict:
        return {
            "interact_radius": self.interact_radius,
            "approach_trust_threshold": self.approach_trust_threshold,
            "avoid_trust_threshold": self.avoid_trust_threshold,
            "interact_probability": self.interact_probability,
            "cooperate_bias": self.cooperate_bias,
            "enable_indirect_memory_hook": self.enable_indirect_memory_hook,
            "enable_symbolic_communication": self.enable_symbolic_communication,
            "max_message_tokens": self.max_message_tokens,
            "communication_energy_cost": self.communication_energy_cost,
            "communication_bias": self.communication_bias,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> SocialConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


class SocialAgent(MemoryAgent):
    """
    Learns directed relationships from interaction outcomes.

    A observes B → chooses approach / avoid / social action → outcome updates
    both agents' relationship stores → future behaviour changes.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        energy: float = 100.0,
        perception_radius: int = 3,
        stm_capacity: int = 32,
        motivation: MotivationConfig | None = None,
        learning: LearningConfig | None = None,
        memory_config: MemoryConfig | None = None,
        social_config: SocialConfig | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            energy=energy,
            perception_radius=perception_radius,
            stm_capacity=stm_capacity,
            motivation=motivation,
            learning=learning,
            memory_config=memory_config,
        )
        self.social_config = social_config or SocialConfig()
        self.relationships = RelationshipStore()
        self.social_outcomes_received: int = 0
        self.approach_actions: int = 0
        self.avoid_actions: int = 0
        self.social_action_counts: dict[str, int] = {a.value: 0 for a in SOCIAL_ACTION_VALUES}
        self.last_interaction_partner: str | None = None
        self.indirect_association_sources: dict[str, list[str]] = {}
        self.secondhand_entities: dict[str, JSONDict] = {}

        # Symbolic communication pathway.
        self.communication_memory = CommunicationMemory(capacity=48)
        self.pending_claims: list[PendingClaim] = []
        self.pending_investigation: Position | None = None
        self.investigation_priority: float = 0.0
        self.investigation_source: str | None = None
        self.investigation_message_id: str | None = None
        self.messages_sent_count: int = 0
        self.messages_received_count: int = 0
        self.verification_count: int = 0
        self.successful_verification_count: int = 0
        self.investigation_actions: int = 0
        self.last_sent_message: JSONDict | None = None
        self.last_received_message: JSONDict | None = None
        self.loss_tracker = LossTracker()

    @property
    def symbolic_communication_enabled(self) -> bool:
        return bool(self.social_config.enable_symbolic_communication)

    def _nearest_visible(self, observation: Observation, radius: int | None = None):
        radius = self.social_config.interact_radius if radius is None else radius
        nearby = [v for v in observation.visible_agents if v.distance <= radius]
        if not nearby:
            return None
        return min(nearby, key=lambda v: (v.distance, v.agent_id))

    def _choose_social_kind(self, other_id: str, rng: ExperimentRNG) -> InteractionKind:
        rel = self.relationships.get(other_id)
        trust = rel.trust if rel else 0.5
        utility = rel.predicted_utility if rel else 0.0
        uncertainty = rel.uncertainty if rel else 1.0
        info_rel = rel.information_reliability if rel else 0.5

        # Experimental bias can force cooperation for controlled trials.
        bias = self.social_config.cooperate_bias
        cooperate_score = trust + max(0.0, utility) + bias + 0.1 * (1.0 - uncertainty)
        compete_score = (1.0 - trust) + max(0.0, -utility) - bias
        share_score = trust * 0.8 + max(0.0, utility) * 0.5 + bias * 0.5
        # Communicate competes with other actions; boosted by uncertainty / info need.
        communicate_score = (
            0.35
            + 0.4 * uncertainty
            + 0.25 * (1.0 - info_rel)
            + self.social_config.communication_bias
        )
        if not self.symbolic_communication_enabled:
            communicate_score *= 0.5

        scores = {
            InteractionKind.COOPERATE: cooperate_score,
            InteractionKind.COMPETE: compete_score,
            InteractionKind.SHARE: share_score,
            InteractionKind.COMMUNICATE: communicate_score,
        }
        # Softmax-like stochastic choice among top options for exploration.
        kinds = list(InteractionKind)
        if rng.random() < 0.1:
            return kinds[int(rng.integers(0, len(kinds)))]
        best = max(scores.values())
        tied = [kind for kind, score in scores.items() if abs(score - best) < 1e-9]
        return tied[int(rng.integers(0, len(tied)))]

    def _maybe_investigate_from_communication(
        self, observation: Observation, rng: ExperimentRNG
    ) -> Action | None:
        """Move toward a location suggested by received communication (weighted by reliability)."""
        target = self.pending_investigation
        if target is None:
            return None
        priority = self.investigation_priority
        if priority < 0.35:
            return None
        # High-priority claims almost always redirect behaviour; low-priority remain stochastic.
        act_probability = min(0.98, 0.45 + 0.55 * priority)
        if rng.random() > act_probability:
            return None
        if observation.position.manhattan(target) == 0:
            self.investigation_actions += 1
            return Action.STAY
        direction = _direction_to_target(observation.position, target.as_tuple())
        action = action_toward(direction)
        if action is None or action not in MOVEMENT_ACTIONS:
            return None
        self.investigation_actions += 1
        return action

    def _social_or_spatial_action(self, observation: Observation, rng: ExperimentRNG) -> Action | None:
        # Prefer interacting with adjacent agents when relationship warrants it.
        adjacent = self._nearest_visible(observation, radius=self.social_config.interact_radius)
        if adjacent is not None and rng.random() < self.social_config.interact_probability:
            kind = self._choose_social_kind(adjacent.agent_id, rng)
            self.last_interaction_partner = adjacent.agent_id
            action = Action(kind.value)
            self.social_action_counts[action.value] = self.social_action_counts.get(action.value, 0) + 1
            return action

        # Approach high-trust / high-utility visible agents; avoid low-trust ones.
        if observation.visible_agents:
            scored = []
            for visible in observation.visible_agents:
                rel = self.relationships.get(visible.agent_id)
                trust = rel.trust if rel else 0.5
                utility = rel.predicted_utility if rel else 0.0
                scored.append((trust + utility, trust, visible))
            scored.sort(key=lambda item: (-item[0], item[2].agent_id))
            _, trust, target = scored[0]
            direction = _direction_to_target(
                observation.position,
                observation.position.offset(*target.relative_position).as_tuple(),
            )
            if trust >= self.social_config.approach_trust_threshold or (
                self.relationships.get(target.agent_id)
                and self.relationships.get(target.agent_id).predicted_utility > 0.15
            ):
                action = action_toward(direction)
                if action is not None:
                    self.approach_actions += 1
                    return action
            if trust <= self.social_config.avoid_trust_threshold:
                # Move opposite to target.
                invert = {
                    "north": "south",
                    "south": "north",
                    "east": "west",
                    "west": "east",
                    "here": "stay",
                }
                action = action_toward(invert.get(direction, "stay"))
                if action is not None:
                    self.avoid_actions += 1
                    return action
        return None

    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        self._update_entity_memories(observation)
        pos = observation.position.as_tuple()
        self.visit_counts[pos] += 1
        self.last_motivation = evaluate_drives(
            observation,
            visit_count=self.visit_counts[pos],
            config=self.motivation_config,
        )

        # Communication-influenced investigation competes with other action sources.
        investigate = self._maybe_investigate_from_communication(observation, rng)
        if investigate is not None:
            return investigate

        social_action = self._social_or_spatial_action(observation, rng)
        if social_action is not None:
            return social_action

        # Fall back to memory-seeking for absent partners, then Q-learning.
        memory_action = self._maybe_seek_remembered(observation, rng)
        if memory_action is not None:
            return memory_action

        state = encode_state(observation)
        if rng.random() < self.epsilon:
            return Action(rng.choice(MOVEMENT_ACTIONS))
        return self._best_action(state, rng)

    def update_loss_tracker(self, *, tick: int, active_ids: set[str]) -> None:
        """Refresh computational-loss trajectories for historical entities."""
        for oid in self.relationships.relationships:
            if oid in active_ids:
                self.loss_tracker.ensure_active(oid)
        self.loss_tracker.on_tick(self, tick=tick, active_ids=active_ids)

    def mark_absent_communicators(self, absent_ids: set[str]) -> int:
        """Flag historical communication records; relationships stay as memory, not live links."""
        return self.communication_memory.mark_historical(absent_ids)

    def communication_summary(self, *, active_ids: set[str] | None = None) -> JSONDict:
        active_ids = active_ids or set()
        rels = self.relationships.relationships
        comm_rels = {
            oid: {
                "communicate_count": r.communicate_count,
                "successful_messages": r.successful_messages,
                "failed_messages": r.failed_messages,
                "information_reliability": r.information_reliability,
                "active_counterpart": oid in active_ids,
            }
            for oid, r in rels.items()
            if r.communicate_count > 0
            or r.successful_messages > 0
            or r.failed_messages > 0
            or abs(r.information_reliability - 0.5) > 1e-9
        }
        return {
            "enabled": self.symbolic_communication_enabled,
            "messages_sent": self.messages_sent_count,
            "messages_received": self.messages_received_count,
            "verification_count": self.verification_count,
            "successful_verification_count": self.successful_verification_count,
            "investigation_actions": self.investigation_actions,
            "pending_investigation": (
                list(self.pending_investigation.as_tuple())
                if self.pending_investigation
                else None
            ),
            "pending_claims": [c.to_dict() for c in self.pending_claims],
            "memory": self.communication_memory.summary(),
            "communication_relationships": comm_rels,
            "last_sent_message": self.last_sent_message,
            "last_received_message": self.last_received_message,
        }

    def apply_social_outcome(self, outcome: InteractionOutcome) -> None:
        """Update relationship + entity memory from a resolved interaction (both sides call this)."""
        other_id, energy_delta, valence, intent = outcome.for_agent(self.agent_id)
        if energy_delta:
            self.apply_energy_gain(energy_delta)

        rel = self.relationships.update_from_outcome(
            other_id,
            tick=outcome.tick,
            valence=valence,
            energy_delta=energy_delta,
            intent=intent,
        )
        self.social_outcomes_received += 1
        self.last_interaction_partner = other_id

        # Keep entity memory aligned with social experience.
        memory = self.ltm.get(other_id)
        if memory is None:
            from the_last_ai.memory.entity import EntityRepresentation

            memory = EntityRepresentation(entity_id=other_id)
            self.ltm.entities[other_id] = memory
        memory.interaction_value = max(
            0.0,
            min(1.0, 0.7 * memory.interaction_value + 0.3 * max(0.0, rel.predicted_utility)),
        )
        memory.familiarity = max(memory.familiarity, rel.strength)
        memory.strength = max(memory.strength, rel.strength)
        memory.uncertainty = min(memory.uncertainty, rel.uncertainty)
        memory.last_seen = max(memory.last_seen, outcome.tick)
        if energy_delta != 0 or valence != 0:
            memory.interaction_count += 1

    def relationship_summary(self) -> JSONDict:
        summary = self.relationships.summary()
        summary.update(
            {
                "social_outcomes_received": self.social_outcomes_received,
                "approach_actions": self.approach_actions,
                "avoid_actions": self.avoid_actions,
                "social_action_counts": dict(self.social_action_counts),
                "last_interaction_partner": self.last_interaction_partner,
                "indirect_association_sources": {
                    k: list(v) for k, v in self.indirect_association_sources.items()
                },
                "secondhand_entities": dict(self.secondhand_entities),
            }
        )
        return summary

    def to_dict(self) -> JSONDict:
        data = super().to_dict()
        data.update(
            {
                "social_config": self.social_config.to_dict(),
                "relationships": self.relationships.to_dict(),
                "social_outcomes_received": self.social_outcomes_received,
                "approach_actions": self.approach_actions,
                "avoid_actions": self.avoid_actions,
                "social_action_counts": dict(self.social_action_counts),
                "last_interaction_partner": self.last_interaction_partner,
                "indirect_association_sources": {
                    k: list(v) for k, v in self.indirect_association_sources.items()
                },
                "secondhand_entities": dict(self.secondhand_entities),
                "communication_memory": self.communication_memory.to_dict(),
                "messages_sent_count": self.messages_sent_count,
                "messages_received_count": self.messages_received_count,
                "verification_count": self.verification_count,
                "successful_verification_count": self.successful_verification_count,
                "investigation_actions": self.investigation_actions,
                "pending_claims": [c.to_dict() for c in self.pending_claims],
                "pending_investigation": (
                    list(self.pending_investigation.as_tuple())
                    if self.pending_investigation
                    else None
                ),
                "last_sent_message": self.last_sent_message,
                "last_received_message": self.last_received_message,
                "loss_tracker": self.loss_tracker.to_dict(),
            }
        )
        return data

    def load_state(self, data: JSONDict) -> None:
        super().load_state(data)
        self.social_config = SocialConfig.from_dict(data.get("social_config", {}))
        self.relationships = RelationshipStore.from_dict(data.get("relationships", {}))
        self.social_outcomes_received = int(data.get("social_outcomes_received", 0))
        self.approach_actions = int(data.get("approach_actions", 0))
        self.avoid_actions = int(data.get("avoid_actions", 0))
        self.social_action_counts = dict(data.get("social_action_counts", {}))
        self.last_interaction_partner = data.get("last_interaction_partner")
        self.indirect_association_sources = {
            k: list(v) for k, v in data.get("indirect_association_sources", {}).items()
        }
        self.secondhand_entities = dict(data.get("secondhand_entities", {}))
        self.communication_memory = CommunicationMemory.from_dict(
            data.get("communication_memory", {"capacity": 48, "records": []})
        )
        self.messages_sent_count = int(data.get("messages_sent_count", 0))
        self.messages_received_count = int(data.get("messages_received_count", 0))
        self.verification_count = int(data.get("verification_count", 0))
        self.successful_verification_count = int(data.get("successful_verification_count", 0))
        self.investigation_actions = int(data.get("investigation_actions", 0))
        self.pending_claims = [
            PendingClaim(**{k: v for k, v in c.items() if k in PendingClaim.__dataclass_fields__})
            if isinstance(c, dict) and "claimed_position" in c and isinstance(c["claimed_position"], tuple)
            else PendingClaim(
                message_id=str(c["message_id"]),
                sender_id=str(c["sender_id"]),
                claimed_position=(
                    int(c["claimed_position"][0]),
                    int(c["claimed_position"][1]),
                ),
                tick_received=int(c.get("tick_received", 0)),
                concepts=list(c.get("concepts", [])),
                status=str(c.get("status", "pending")),
            )
            for c in data.get("pending_claims", [])
        ]
        inv = data.get("pending_investigation")
        self.pending_investigation = Position(int(inv[0]), int(inv[1])) if inv else None
        self.last_sent_message = data.get("last_sent_message")
        self.last_received_message = data.get("last_received_message")
        if data.get("loss_tracker"):
            self.loss_tracker = LossTracker.from_dict(data["loss_tracker"])
        else:
            self.loss_tracker = LossTracker()


class CooperativePartner(SocialAgent):
    """Controlled partner that prefers cooperation — useful for Experiment 3."""

    def __init__(self, agent_id: str, **kwargs) -> None:
        social = kwargs.pop("social_config", None) or SocialConfig(cooperate_bias=1.5, interact_probability=0.9)
        super().__init__(agent_id, social_config=social, **kwargs)
