"""World-model / predictive agent tests."""

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.experiments.gradual_population import run_gradual_population_reduction
from the_last_ai.experiments.world_model_prediction import run_world_model_experiment
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import CellType, Observation, Position, VisibleAgent
from the_last_ai.world_model.agent_model import EntityDynamicsModel
from the_last_ai.world_model.environment import EnvironmentModel
from the_last_ai.world_model.model import WorldModel
from the_last_ai.world_model.types import Prediction, PredictionKind


def _obs_with_agent(*, tick: int, me: Position, other: Position | None = None) -> Observation:
    radius = 2
    size = 2 * radius + 1
    cells = [[int(CellType.EMPTY) for _ in range(size)] for _ in range(size)]
    visible = ()
    if other is not None:
        visible = (
            VisibleAgent(
                agent_id="B",
                relative_position=(other.x - me.x, other.y - me.y),
                distance=me.manhattan(other),
            ),
        )
    return Observation(
        tick=tick,
        position=me,
        energy=80,
        local_cells=tuple(tuple(row) for row in cells),
        visible_agents=visible,
        perception_radius=radius,
    )


def test_entity_direction_learning():
    model = EntityDynamicsModel(entity_id="B")
    model.update_sighting(Position(5, 5), tick=1)
    model.update_sighting(Position(6, 5), tick=2)
    model.update_sighting(Position(7, 5), tick=3)
    direction, prob = model.likely_direction()
    assert direction == "east"
    assert prob > 0.5
    pred = model.predict_next_position(3)
    assert pred is not None
    assert pred.value == [8, 5]


def test_world_model_scores_position_error():
    wm = WorldModel()
    obs1 = _obs_with_agent(tick=0, me=Position(4, 4), other=Position(5, 4))
    wm.observe_and_update(obs1)
    preds = wm.form_predictions(obs1)
    assert any(p.kind == PredictionKind.AGENT_POSITION for p in preds)

    # B moves east as predicted or not.
    obs2 = _obs_with_agent(tick=1, me=Position(4, 4), other=Position(6, 4))
    errors = wm.observe_and_update(obs2)
    assert wm.total_scored >= 1
    assert any(e.prediction.kind == PredictionKind.AGENT_POSITION for e in errors)


def test_presence_prediction_after_absence():
    wm = WorldModel()
    for t, x in enumerate([5, 6, 7]):
        obs = _obs_with_agent(tick=t, me=Position(4, 4), other=Position(x, 4))
        wm.observe_and_update(obs)
        wm.form_predictions(obs)
    # B disappears from view.
    obs_missing = _obs_with_agent(tick=10, me=Position(4, 4), other=None)
    errors = wm.observe_and_update(obs_missing)
    presence_errors = [e for e in errors if e.prediction.kind == PredictionKind.AGENT_PRESENCE]
    assert presence_errors
    assert "B" in wm.agents.entities


def test_predictive_agent_in_simulation():
    sim = Simulation.create(
        SimulationConfig(
            seed=0,
            n_agents=4,
            n_resources=6,
            width=14,
            height=10,
            agent_type="PredictiveAgent",
        )
    )
    sim.run(40)
    agent = sim.agents["agent_000"]
    assert isinstance(agent, PredictiveAgent)
    assert agent.prediction_cycles == 40
    assert agent.world_model.total_scored >= 0
    summary = agent.world_model_summary()
    assert "mean_error_ema" in summary
    cf = agent.counterfactual_absence("agent_001")
    assert "query" in cf


def test_predictive_agent_roundtrip(tmp_path):
    sim = Simulation.create(
        SimulationConfig(seed=2, n_agents=3, n_resources=3, agent_type="PredictiveAgent", width=10, height=8)
    )
    sim.run(15)
    path = sim.save(tmp_path / "pred.json")
    loaded = Simulation.load(path)
    assert loaded.config.agent_type == "PredictiveAgent"
    original = sim.agents["agent_000"]
    restored = loaded.agents["agent_000"]
    assert isinstance(original, PredictiveAgent)
    assert isinstance(restored, PredictiveAgent)
    assert restored.world_model.total_scored == original.world_model.total_scored


def test_world_model_experiment(tmp_path):
    result = run_world_model_experiment(
        seed=0,
        n_ticks=60,
        n_agents=4,
        n_resources=5,
        disappear_at=35,
        output_dir=tmp_path / "wm",
    )
    changing = result["comparison"]["changing_world"]
    assert changing["total_scored"] > 0
    assert changing["removed_agent"] is not None
    assert changing["removed_agent"] in changing["world_model"]["entity_models"]


def test_gradual_population_reduction(tmp_path):
    result = run_gradual_population_reduction(
        seed=0,
        initial_agents=8,
        schedule=[4, 2, 1],
        ticks_between=10,
        n_resources=6,
        output_dir=tmp_path / "exp5",
    )
    snap = result["snapshot"]
    assert snap["final_population"] == 1
    assert snap["survivor_world_model"]["tracked_entities"] >= 0
