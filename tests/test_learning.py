"""Motivation, features, and value-learning tests."""

from the_last_ai.agents.features import encode_state, nearest_resource_cue
from the_last_ai.agents.motivation import MotivationConfig, compute_reward_with_blocked, evaluate_drives
from the_last_ai.agents.value_learner import ValueBasedAgent
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Action, CellType, Observation, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _obs(*, energy: float = 40.0, resources: bool = False, agents=()) -> Observation:
    radius = 1
    cells = [
        [int(CellType.EMPTY), int(CellType.EMPTY), int(CellType.EMPTY)],
        [int(CellType.EMPTY), int(CellType.EMPTY), int(CellType.EMPTY)],
        [int(CellType.EMPTY), int(CellType.EMPTY), int(CellType.EMPTY)],
    ]
    if resources:
        cells[1][2] = int(CellType.RESOURCE)
    return Observation(
        tick=0,
        position=Position(2, 2),
        energy=energy,
        local_cells=tuple(tuple(row) for row in cells),
        visible_agents=tuple(agents),
        perception_radius=radius,
    )


def test_energy_drive_rises_when_energy_is_low():
    low = evaluate_drives(_obs(energy=10), visit_count=1, config=MotivationConfig())
    high = evaluate_drives(_obs(energy=90), visit_count=1, config=MotivationConfig())
    assert low.energy_drive > high.energy_drive


def test_resource_collection_and_regen():
    world = World(grid=Grid.empty(8, 6), resource_energy=15.0, resource_regen_interval=2)
    world.place_resource(Position(2, 2))
    world.place_agent("a", Position(2, 2))
    gained = world.collect_resources()
    assert gained["a"] == 15.0
    assert world.grid.get(Position(2, 2)).name == "EMPTY"
    world.apply_action("a", Action.MOVE_EAST)
    world.advance_tick()
    assert world.regenerate_resources() == 0
    world.advance_tick()
    world.advance_tick()
    assert world.regenerate_resources() == 1
    assert world.grid.get(Position(2, 2)).name == "RESOURCE"


def test_value_agent_updates_q_table():
    agent = ValueBasedAgent("learner")
    rng = ExperimentRNG.from_seed(0)
    before = _obs(energy=30.0, resources=True)
    action = agent.select_action(before, rng)
    after = _obs(energy=55.0)
    agent.learn(before, action, reward=0.5, next_observation=after, energy_gained=25.0, moved=True)
    assert agent.update_count == 1
    assert len(agent.q_values) >= 1
    assert agent.epsilon < agent.learning_config.epsilon


def test_value_learner_outperforms_random_on_resources():
    shared = dict(
        seed=11,
        n_agents=3,
        n_resources=12,
        width=14,
        height=10,
        initial_energy=35.0,
        energy_cost_per_tick=0.25,
        resource_energy=30.0,
        resource_regen_interval=8,
    )
    random_sim = Simulation.create(SimulationConfig(agent_type="RandomAgent", **shared))
    learner_sim = Simulation.create(SimulationConfig(agent_type="ValueBasedAgent", **shared))
    random_metrics = random_sim.run(250)
    learner_metrics = learner_sim.run(250)

    # Learning should improve resource foraging and/or cumulative reward.
    assert (
        learner_metrics.summary()["total_resources_collected"]
        >= random_metrics.summary()["total_resources_collected"]
        or learner_metrics.summary()["mean_reward"] > random_metrics.summary()["mean_reward"]
    )


def test_encode_state_is_hashable():
    state = encode_state(_obs(energy=42.0, resources=True))
    assert isinstance(state, tuple)
    assert hash(state) is not None
    direction, dist = nearest_resource_cue(_obs(energy=42.0, resources=True))
    assert direction in {"east", "west", "north", "south", "here", "none"}


def test_reward_increases_with_energy_gain():
    before = _obs(energy=20)
    after = _obs(energy=45)
    low = compute_reward_with_blocked(
        before, after, energy_gained=0.0, moved=True, new_cell=False, blocked=False, config=MotivationConfig()
    )
    high = compute_reward_with_blocked(
        before, after, energy_gained=25.0, moved=True, new_cell=False, blocked=False, config=MotivationConfig()
    )
    assert high > low


def test_value_agent_roundtrip(tmp_path):
    sim = Simulation.create(
        SimulationConfig(
            seed=2,
            n_agents=2,
            n_resources=4,
            agent_type="ValueBasedAgent",
            width=10,
            height=8,
        )
    )
    sim.run(20)
    path = sim.save(tmp_path / "learner.json")
    loaded = Simulation.load(path)
    assert loaded.config.agent_type == "ValueBasedAgent"
    assert loaded.agents["agent_000"].update_count == sim.agents["agent_000"].update_count
