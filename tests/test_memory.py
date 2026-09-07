"""Memory system tests."""

from the_last_ai.agents.memory_agent import MemoryAgent
from the_last_ai.experiments.entity_representation import run_entity_representation
from the_last_ai.memory.decay import decayed_strength
from the_last_ai.memory.long_term import LongTermMemory, MemoryConfig
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Observation, Position, VisibleAgent
from the_last_ai.world.grid import Grid


def test_exponential_decay():
    assert decayed_strength(1.0, 0, 0.01) == 1.0
    later = decayed_strength(1.0, 100, 0.01)
    assert 0.0 < later < 1.0


def test_entity_memory_updates_and_retrieves():
    ltm = LongTermMemory(MemoryConfig(capacity=4, decay_lambda=0.001))
    ltm.observe_entity("a", Position(3, 3), tick=1, distance=1)
    ltm.observe_entity("a", Position(4, 3), tick=2, distance=1)
    ltm.observe_entity("b", Position(8, 8), tick=2, distance=3)
    mem = ltm.get("a")
    assert mem is not None
    assert mem.sighting_count == 2
    assert mem.familiarity > 0
    assert mem.interaction_count >= 1
    retrieved = ltm.retrieve(Position(3, 3), tick=3, top_k=1)
    assert retrieved[0].entity_id == "a"
    assert ltm.retrieval_events >= 1


def test_capacity_replacement():
    ltm = LongTermMemory(MemoryConfig(capacity=2, strength_boost=0.05, familiarity_rate=0.01))
    ltm.observe_entity("a", Position(1, 1), tick=1, distance=5)
    ltm.observe_entity("b", Position(2, 2), tick=2, distance=5)
    ltm.observe_entity("c", Position(3, 3), tick=3, distance=5)
    assert len(ltm.entities) == 2
    assert ltm.replacement_count >= 1


def test_decay_can_forget_weak_memories():
    ltm = LongTermMemory(MemoryConfig(decay_lambda=0.5, forget_threshold=0.2, strength_boost=0.15))
    ltm.observe_entity("ghost", Position(2, 2), tick=0, distance=4)
    assert "ghost" in ltm.entities
    removed = ltm.tick_decay(20)
    assert "ghost" in removed
    assert "ghost" not in ltm.entities


def test_memory_agent_forms_representation():
    agent = MemoryAgent("observer", perception_radius=3)
    rng = ExperimentRNG.from_seed(0)
    obs = Observation(
        tick=5,
        position=Position(4, 4),
        energy=80.0,
        local_cells=tuple(tuple(0 for _ in range(7)) for _ in range(7)),
        visible_agents=(
            VisibleAgent(agent_id="partner", relative_position=(1, 0), distance=1),
        ),
        perception_radius=3,
    )
    agent.select_action(obs, rng)
    assert agent.ltm.get("partner") is not None
    assert agent.entity_strength("partner", tick=5) > 0


def test_memory_agent_roundtrip(tmp_path):
    sim = Simulation.create(
        SimulationConfig(seed=4, n_agents=3, agent_type="MemoryAgent", width=10, height=8)
    )
    sim.run(15)
    path = sim.save(tmp_path / "mem.json")
    loaded = Simulation.load(path)
    assert loaded.config.agent_type == "MemoryAgent"
    original = sim.agents["agent_000"]
    restored = loaded.agents["agent_000"]
    assert isinstance(original, MemoryAgent)
    assert isinstance(restored, MemoryAgent)
    assert set(restored.ltm.entities) == set(original.ltm.entities)


def test_experiment_2_familiar_stronger_than_unfamiliar(tmp_path):
    result = run_entity_representation(
        seed=2,
        familiar_ticks=100,
        unfamiliar_ticks=5,
        post_ticks=40,
        output_dir=tmp_path / "exp2",
    )
    deltas = result["comparison"]["deltas"]
    assert deltas["familiarity_at_removal"] > 0
    assert (
        deltas["strength_at_removal"] > 0
        or result["comparison"]["familiar"]["strength_at_removal"]
        >= result["comparison"]["unfamiliar"]["strength_at_removal"]
    )
    assert result["comparison"]["familiar"]["memory_persisted"] or deltas["final_strength"] >= 0
