"""Phase 3.5 — social memory propagation tests."""

from the_last_ai.agents.social_agent import SocialAgent, SocialConfig
from the_last_ai.experiments.social_memory_propagation import run_social_memory_propagation
from the_last_ai.social.propagation import (
    PropagationConfig,
    PropagationLog,
    propagate_after_interaction,
    shared_absent_contacts,
)
from the_last_ai.social.resolve import resolve_pair
from the_last_ai.social.types import InteractionKind


def _linked_agents():
    cfg = SocialConfig(enable_indirect_memory_hook=True, cooperate_bias=1.5)
    a = SocialAgent("A", social_config=cfg)
    c = SocialAgent("C", social_config=cfg)
    # Both have strong learned links to absent B.
    for agent in (a, c):
        agent.relationships.update_from_outcome(
            "B",
            tick=10,
            valence=0.8,
            energy_delta=6.0,
            intent=InteractionKind.COOPERATE,
        )
        for t in range(11, 25):
            agent.relationships.update_from_outcome(
                "B",
                tick=t,
                valence=0.7,
                energy_delta=5.0,
                intent=InteractionKind.COOPERATE,
            )
        mem = agent.ltm.entities.setdefault(
            "B",
            __import__("the_last_ai.memory.entity", fromlist=["EntityRepresentation"]).EntityRepresentation(
                entity_id="B"
            ),
        )
        mem.strength = 0.9
        mem.familiarity = 0.9
        mem.expected_location = (4, 4)
        mem.last_seen = 24
    return a, c


def test_shared_absent_contacts_detected():
    a, c = _linked_agents()
    shared = shared_absent_contacts(
        a, c, absent_ids={"B"}, min_strength=0.2, min_interactions=5
    )
    assert shared and shared[0][0] == "B"


def test_propagation_boosts_ac_and_logs_transfer():
    a, c = _linked_agents()
    outcome = resolve_pair(
        tick=30,
        agent_a="A",
        agent_b="C",
        intent_a=InteractionKind.COOPERATE,
        intent_b=InteractionKind.COOPERATE,
        energy_a=80,
        energy_b=80,
    )
    # Direct outcome first (as simulation does).
    a.apply_social_outcome(outcome)
    c.apply_social_outcome(outcome)

    before = a.relationships.get("C").predicted_utility
    log = PropagationLog()
    created = propagate_after_interaction(
        a,
        c,
        outcome,
        absent_ids={"B"},
        config=PropagationConfig(
            min_shared_strength=0.2,
            min_shared_interactions=5,
            require_communicate_for_memory=False,
        ),
        log=log,
    )
    assert created
    assert a.relationships.get("C").predicted_utility >= before
    assert "B" in a.indirect_association_sources.get("C", [])
    assert "B" in a.secondhand_entities or a.ltm.get("B") is not None
    assert log.records


def test_no_propagation_without_shared_contact():
    cfg = SocialConfig(enable_indirect_memory_hook=True)
    a = SocialAgent("A", social_config=cfg)
    c = SocialAgent("C", social_config=cfg)
    a.relationships.update_from_outcome(
        "B", tick=1, valence=0.8, energy_delta=5.0, intent=InteractionKind.COOPERATE
    )
    outcome = resolve_pair(
        tick=2,
        agent_a="A",
        agent_b="C",
        intent_a=InteractionKind.COOPERATE,
        intent_b=InteractionKind.COOPERATE,
        energy_a=50,
        energy_b=50,
    )
    a.apply_social_outcome(outcome)
    c.apply_social_outcome(outcome)
    created = propagate_after_interaction(
        a,
        c,
        outcome,
        absent_ids={"B"},
        config=PropagationConfig(min_shared_interactions=5),
    )
    assert created == []


def test_experiment_4_triangle_exceeds_controls(tmp_path):
    result = run_social_memory_propagation(
        seed=0,
        strong_ticks=30,
        weak_ticks=1,
        test_ticks=40,
        output_dir=tmp_path / "exp4",
    )
    summary = result["summary"]
    # Triangle condition should produce indirect transfer events; controls should not.
    assert summary["experiment_propagation_events"] > 0
    assert result["comparison"]["control_ab_only"]["test"]["propagation_events"] == 0
    assert result["comparison"]["control_weak"]["test"]["propagation_events"] == 0
    assert summary["deltas_vs_ab_only"]["propagation_events"] > 0
    assert summary["deltas_vs_weak"]["propagation_events"] > 0
    assert "B" in summary["experiment_indirect_sources"]
    # Utility/strength toward C should be influenced relative to at least one control.
    assert (
        summary["deltas_vs_ab_only"]["delta_predicted_utility"] > 0
        or summary["deltas_vs_weak"]["delta_predicted_utility"] > 0
        or summary["deltas_vs_ab_only"]["delta_strength"] > 0
    )
