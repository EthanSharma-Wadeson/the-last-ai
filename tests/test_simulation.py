"""Simulation and persistence tests."""

from pathlib import Path

from the_last_ai.simulation.engine import Simulation, SimulationConfig


def test_simulation_is_deterministic():
    config = SimulationConfig(seed=42, n_agents=4, width=12, height=8)

    sim_a = Simulation.create(config)
    sim_a.run(30)
    positions_a = {aid: pos.as_tuple() for aid, pos in sim_a.world.agent_positions.items()}

    sim_b = Simulation.create(config)
    sim_b.run(30)
    positions_b = {aid: pos.as_tuple() for aid, pos in sim_b.world.agent_positions.items()}

    assert positions_a == positions_b
    assert sim_a.metrics.summary() == sim_b.metrics.summary()


def test_save_and_load_restores_world_state(tmp_path: Path):
    config = SimulationConfig(seed=7, n_agents=3, width=10, height=8)
    sim = Simulation.create(config)
    sim.run(15)
    path = sim.save(tmp_path / "snap.json")

    loaded = Simulation.load(path)
    assert loaded.world.tick == sim.world.tick
    assert loaded.world.agent_positions == sim.world.agent_positions
    assert set(loaded.agents) == set(sim.agents)
    assert loaded.agents["agent_000"].state.age == sim.agents["agent_000"].state.age


def test_disappearance_reduces_population():
    sim = Simulation.create(SimulationConfig(seed=1, n_agents=3, width=10, height=8))
    sim.run(5)
    victim = sim.active_agent_ids()[0]
    sim.disappear(victim)
    assert victim not in sim.world.agent_positions
    assert len(sim.active_agent_ids()) == 2


def test_perception_excludes_distant_agents():
    sim = Simulation.create(
        SimulationConfig(seed=0, n_agents=2, width=20, height=12, perception_radius=1)
    )
    # Place agents far apart by moving them via world state.
    ids = sim.active_agent_ids()
    sim.world.agent_positions[ids[0]] = sim.world.grid.free_positions()[0]
    far = [p for p in sim.world.grid.free_positions() if p.manhattan(sim.world.agent_positions[ids[0]]) > 5]
    sim.world.agent_positions[ids[1]] = far[0]

    obs = sim.agents[ids[0]].observe(sim.world)
    assert all(v.agent_id != ids[1] for v in obs.visible_agents)
