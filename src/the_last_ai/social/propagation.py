"""Indirect information transfer via shared social contacts.

This module implements a measurable association-transfer mechanism.
Results should be described as computational effects of network structure,
not immediately labelled as "socially propagated memory."
"""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.memory.entity import EntityRepresentation
from the_last_ai.social.types import InteractionKind, InteractionOutcome
from the_last_ai.types import JSONDict


@dataclass
class IndirectTransferRecord:
    tick: int
    receiver_id: str
    partner_id: str
    about_entity_id: str
    path_strength: float
    association_boost: float
    secondhand_memory: bool
    transferred_location: list[int] | None = None

    def to_dict(self) -> JSONDict:
        return {
            "tick": self.tick,
            "receiver_id": self.receiver_id,
            "partner_id": self.partner_id,
            "about_entity_id": self.about_entity_id,
            "path_strength": self.path_strength,
            "association_boost": self.association_boost,
            "secondhand_memory": self.secondhand_memory,
            "transferred_location": self.transferred_location,
        }


@dataclass
class PropagationConfig:
    min_shared_strength: float = 0.25
    min_shared_interactions: int = 5
    association_lr: float = 0.2
    secondhand_strength_scale: float = 0.35
    secondhand_uncertainty: float = 0.85
    require_communicate_for_memory: bool = True

    def to_dict(self) -> JSONDict:
        return {
            "min_shared_strength": self.min_shared_strength,
            "min_shared_interactions": self.min_shared_interactions,
            "association_lr": self.association_lr,
            "secondhand_strength_scale": self.secondhand_strength_scale,
            "secondhand_uncertainty": self.secondhand_uncertainty,
            "require_communicate_for_memory": self.require_communicate_for_memory,
        }


@dataclass
class PropagationLog:
    records: list[IndirectTransferRecord] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {"records": [r.to_dict() for r in self.records], "count": len(self.records)}


def shared_absent_contacts(
    agent_a,
    agent_b,
    *,
    absent_ids: set[str],
    min_strength: float,
    min_interactions: int = 5,
) -> list[tuple[str, float, float]]:
    """Return (entity_id, strength_a, strength_b) for shared absent contacts."""
    shared: list[tuple[str, float, float]] = []
    rels_a = getattr(agent_a, "relationships", None)
    rels_b = getattr(agent_b, "relationships", None)
    if rels_a is None or rels_b is None:
        return shared

    for entity_id in sorted(absent_ids):
        ra = rels_a.get(entity_id)
        rb = rels_b.get(entity_id)
        if ra is None or rb is None:
            continue
        if ra.strength < min_strength or rb.strength < min_strength:
            continue
        if ra.interaction_count < min_interactions or rb.interaction_count < min_interactions:
            continue
        shared.append((entity_id, ra.strength, rb.strength))
    return shared


def _ensure_entity(agent, entity_id: str) -> EntityRepresentation:
    memory = agent.ltm.get(entity_id)
    if memory is None:
        memory = EntityRepresentation(entity_id=entity_id)
        agent.ltm.entities[entity_id] = memory
    return memory


def apply_association_transfer(
    receiver,
    partner_id: str,
    *,
    about_id: str,
    receiver_strength_to_about: float,
    partner_strength_to_about: float,
    partner_utility_to_about: float,
    partner_trust_to_about: float,
    tick: int,
    config: PropagationConfig,
) -> float:
    """
    Bias receiver→partner relationship using shared association with an absent entity.

    Path strength ≈ product of both sides' relationship strength to the absent contact.
    """
    path = receiver_strength_to_about * partner_strength_to_about
    boost = config.association_lr * path

    rel = receiver.relationships.get_or_create(partner_id)
    # Pull predicted utility / trust slightly toward partner's stance on the shared contact,
    # scaled by how strongly both were linked to that contact.
    target_utility = 0.5 * partner_utility_to_about + 0.5 * path
    rel.predicted_utility = (1 - boost) * rel.predicted_utility + boost * target_utility
    target_trust = 0.5 + 0.5 * (partner_trust_to_about - 0.5) * path
    rel.trust = max(0.0, min(1.0, (1 - boost) * rel.trust + boost * target_trust))
    rel.uncertainty = max(0.05, rel.uncertainty * (1.0 - 0.15 * path))
    rel.strength = min(1.0, rel.strength + 0.5 * boost * (1.0 - rel.strength))
    rel.last_interaction_tick = max(rel.last_interaction_tick, tick)

    # Track that this relationship was influenced by an absent third party.
    if not hasattr(receiver, "indirect_association_sources"):
        receiver.indirect_association_sources = {}
    sources = receiver.indirect_association_sources.setdefault(partner_id, [])
    if about_id not in sources:
        sources.append(about_id)

    return boost


def apply_secondhand_entity_memory(
    receiver,
    *,
    about_id: str,
    donor_memory: EntityRepresentation | None,
    donor_rel_strength: float,
    receiver_rel_strength: float,
    tick: int,
    config: PropagationConfig,
) -> bool:
    """Install or reinforce a high-uncertainty secondhand entity representation."""
    if donor_memory is None:
        return False
    path = donor_rel_strength * receiver_rel_strength
    if path < config.min_shared_strength * config.min_shared_strength:
        return False

    memory = _ensure_entity(receiver, about_id)
    was_new = memory.sighting_count == 0 and memory.strength < 1e-9
    transfer = config.secondhand_strength_scale * path

    memory.strength = min(1.0, max(memory.strength, transfer))
    memory.familiarity = max(memory.familiarity, 0.5 * transfer)
    memory.uncertainty = max(memory.uncertainty, config.secondhand_uncertainty)
    memory.interaction_value = max(
        memory.interaction_value,
        0.5 * donor_memory.interaction_value * path,
    )
    if donor_memory.expected_location is not None:
        if memory.expected_location is None:
            memory.expected_location = donor_memory.expected_location
        else:
            # Soft blend toward donor's expected location.
            ex, ey = memory.expected_location
            dx, dy = donor_memory.expected_location
            memory.expected_location = (
                int(round(0.7 * ex + 0.3 * dx)),
                int(round(0.7 * ey + 0.3 * dy)),
            )
    memory.last_seen = max(memory.last_seen, tick)

    if not hasattr(receiver, "secondhand_entities"):
        receiver.secondhand_entities = {}
    receiver.secondhand_entities[about_id] = {
        "via": "shared_contact_communication",
        "path_strength": path,
        "tick": tick,
        "was_new": was_new,
    }
    return True


def propagate_after_interaction(
    agent_a,
    agent_b,
    outcome: InteractionOutcome,
    *,
    absent_ids: set[str],
    config: PropagationConfig | None = None,
    log: PropagationLog | None = None,
) -> list[IndirectTransferRecord]:
    """
    After A↔C interact, transfer association (and optionally secondhand memory)
    for shared absent contacts such as B.
    """
    config = config or PropagationConfig()
    log = log or PropagationLog()
    created: list[IndirectTransferRecord] = []

    if not absent_ids:
        return created
    if not getattr(getattr(agent_a, "social_config", None), "enable_indirect_memory_hook", False):
        return created
    if not getattr(getattr(agent_b, "social_config", None), "enable_indirect_memory_hook", False):
        return created

    shared = shared_absent_contacts(
        agent_a,
        agent_b,
        absent_ids=absent_ids,
        min_strength=config.min_shared_strength,
        min_interactions=config.min_shared_interactions,
    )
    if not shared:
        return created

    communicate = outcome.resolved_kind.startswith("communicate") or (
        outcome.intent_a == InteractionKind.COMMUNICATE
        or outcome.intent_b == InteractionKind.COMMUNICATE
    )

    for about_id, strength_a, strength_b in shared:
        rel_a = agent_a.relationships.get(about_id)
        rel_b = agent_b.relationships.get(about_id)
        assert rel_a is not None and rel_b is not None

        # A receives association transfer about shared contact via B-link through C.
        boost_a = apply_association_transfer(
            agent_a,
            agent_b.agent_id,
            about_id=about_id,
            receiver_strength_to_about=strength_a,
            partner_strength_to_about=strength_b,
            partner_utility_to_about=rel_b.predicted_utility,
            partner_trust_to_about=rel_b.trust,
            tick=outcome.tick,
            config=config,
        )
        secondhand_a = False
        loc_a = None
        if communicate or not config.require_communicate_for_memory:
            secondhand_a = apply_secondhand_entity_memory(
                agent_a,
                about_id=about_id,
                donor_memory=agent_b.ltm.get(about_id),
                donor_rel_strength=strength_b,
                receiver_rel_strength=strength_a,
                tick=outcome.tick,
                config=config,
            )
            mem = agent_a.ltm.get(about_id)
            if mem and mem.expected_location:
                loc_a = list(mem.expected_location)

        record_a = IndirectTransferRecord(
            tick=outcome.tick,
            receiver_id=agent_a.agent_id,
            partner_id=agent_b.agent_id,
            about_entity_id=about_id,
            path_strength=strength_a * strength_b,
            association_boost=boost_a,
            secondhand_memory=secondhand_a,
            transferred_location=loc_a,
        )
        log.records.append(record_a)
        created.append(record_a)

        # Symmetric: C also updates from A's association with B.
        boost_b = apply_association_transfer(
            agent_b,
            agent_a.agent_id,
            about_id=about_id,
            receiver_strength_to_about=strength_b,
            partner_strength_to_about=strength_a,
            partner_utility_to_about=rel_a.predicted_utility,
            partner_trust_to_about=rel_a.trust,
            tick=outcome.tick,
            config=config,
        )
        secondhand_b = False
        loc_b = None
        if communicate or not config.require_communicate_for_memory:
            secondhand_b = apply_secondhand_entity_memory(
                agent_b,
                about_id=about_id,
                donor_memory=agent_a.ltm.get(about_id),
                donor_rel_strength=strength_a,
                receiver_rel_strength=strength_b,
                tick=outcome.tick,
                config=config,
            )
            mem = agent_b.ltm.get(about_id)
            if mem and mem.expected_location:
                loc_b = list(mem.expected_location)

        record_b = IndirectTransferRecord(
            tick=outcome.tick,
            receiver_id=agent_b.agent_id,
            partner_id=agent_a.agent_id,
            about_entity_id=about_id,
            path_strength=strength_a * strength_b,
            association_boost=boost_b,
            secondhand_memory=secondhand_b,
            transferred_location=loc_b,
        )
        log.records.append(record_b)
        created.append(record_b)

    return created
