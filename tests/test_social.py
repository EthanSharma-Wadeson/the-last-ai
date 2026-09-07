"""Social cognition tests."""

from the_last_ai.agents.social_agent import CooperativePartner, SocialAgent, SocialConfig
from the_last_ai.experiments.sudden_disappearance import run_sudden_disappearance
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.social.graph import SocialGraph
from the_last_ai.social.relationship import RelationshipStore
from the_last_ai.social.resolve import resolve_pair
from the_last_ai.social.types import InteractionKind
from the_last_ai.types import Action, Observation, Position, VisibleAgent


def test_mutual_cooperate_benefits_both():
    outcome = resolve_pair(
        tick=1,
        agent_a="a",
        agent_b="b",
        intent_a=InteractionKind.COOPERATE,
        intent_b=InteractionKind.COOPERATE,
        energy_a=50,
        energy_b=50,
    )
    assert outcome.energy_delta_a > 0
    assert outcome.energy_delta_b > 0
    assert outcome.valence_a > 0
    assert outcome.valence_b > 0


def test_compete_exploits_cooperator():
    outcome = resolve_pair(
        tick=1,
        agent_a="a",
        agent_b="b",
        intent_a=InteractionKind.COMPETE,
        intent_b=InteractionKind.COOPERATE,
        energy_a=50,
        energy_b=50,
    )
    assert outcome.energy_delta_a > 0
    assert outcome.energy_delta_b < 0
    assert outcome.valence_b < 0


def test_trust_is_learned_from_outcomes():
    store = RelationshipStore()
    store.update_from_outcome(
        "b", tick=1, valence=0.7, energy_delta=6.0, intent=InteractionKind.COOPERATE
    )
    store.update_from_outcome(
        "b", tick=2, valence=0.7, energy_delta=6.0, intent=InteractionKind.COOPERATE
    )
    rel = store.get("b")
    assert rel is not None
    assert rel.trust > 0.5
    assert rel.predicted_utility > 0
    assert rel.strength > 0
    assert rel.positive_interactions == 2


def test_negative_outcomes_reduce_trust():
    store = RelationshipStore()
    store.update_from_outcome(
        "b", tick=1, valence=-0.6, energy_delta=-5.0, intent=InteractionKind.COMPETE
    )
    store.update_from_outcome(
        "b", tick=2, valence=-0.6, energy_delta=-5.0, intent=InteractionKind.COMPETE
    )
    rel = store.get("b")
    assert rel is not None
    assert rel.trust < 0.5
    assert rel.negative_interactions == 2


def test_bidirectional_updates_in_simulation():
    sim = Simulation.create(
        SimulationConfig(
            seed=0,
            n_agents=2,
            width=8,
            height=6,
            agent_type="SocialAgent",
            perception_radius=3,
        )
    )
    # Place adjacent for guaranteed interaction opportunities.
    ids = sim.active_agent_ids()
    sim.world.agent_positions[ids[0]] = Position(2, 2)
    sim.world.agent_positions[ids[1]] = Position(3, 2)
    for agent in sim.agents.values():
        if isinstance(agent, SocialAgent):
            agent.social_config = SocialConfig(cooperate_bias=1.5, interact_probability=1.0)

    sim.run(30)
    a, b = ids
    assert isinstance(sim.agents[a], SocialAgent)
    assert isinstance(sim.agents[b], SocialAgent)
    assert sim.agents[a].relationships.get(b) is not None
    assert sim.agents[b].relationships.get(a) is not None
    assert sim.social_graph.weight(a, b) > 0
    assert sim.social_graph.weight(b, a) > 0
    assert len(sim.interaction_log) > 0


def test_social_graph_two_hop_scaffold():
    graph = SocialGraph()
    graph.set_edge("a", "b", 0.8)
    graph.set_edge("b", "c", 0.7)
    hops = graph.two_hop_neighbors("a")
    assert any(node == "c" and via == "b" for node, _, via in hops)


def test_social_agent_roundtrip(tmp_path):
    sim = Simulation.create(
        SimulationConfig(seed=3, n_agents=3, agent_type="SocialAgent", width=10, height=8)
    )
    sim.run(12)
    path = sim.save(tmp_path / "social.json")
    loaded = Simulation.load(path)
    assert loaded.config.agent_type == "SocialAgent"
    original = sim.agents["agent_000"]
    restored = loaded.agents["agent_000"]
    assert isinstance(original, SocialAgent)
    assert isinstance(restored, SocialAgent)
    assert set(restored.relationships.relationships) == set(original.relationships.relationships)


def test_experiment_3_familiar_stronger_relationship(tmp_path):
    result = run_sudden_disappearance(
        seed=1,
        familiar_ticks=80,
        unfamiliar_ticks=5,
        post_ticks=30,
        output_dir=tmp_path / "exp3",
    )
    deltas = result["comparison"]["deltas"]
    assert deltas["interaction_count"] > 0
    assert deltas["relationship_strength"] > 0
    assert deltas["trust"] > 0 or deltas["predicted_utility"] > 0
    assert deltas["graph_weight"] > 0


def test_cooperative_partner_prefers_cooperate():
    partner = CooperativePartner("p")
    rng = ExperimentRNG.from_seed(0)
    obs = Observation(
        tick=0,
        position=Position(2, 2),
        energy=80,
        local_cells=tuple(tuple(0 for _ in range(7)) for _ in range(7)),
        visible_agents=(VisibleAgent("o", (1, 0), 1),),
        perception_radius=3,
    )
    actions = [partner.select_action(obs, rng) for _ in range(20)]
    assert sum(1 for a in actions if a == Action.COOPERATE) >= 10
