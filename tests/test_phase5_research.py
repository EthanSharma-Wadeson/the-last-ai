"""Phase 5 — restoration, counterfactuals, Last AI, architecture comparison."""

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.experiments.architecture_compare import run_architecture_comparison
from the_last_ai.experiments.counterfactual_reasoning import run_counterfactual_reasoning
from the_last_ai.experiments.restoration import run_restoration
from the_last_ai.experiments.the_last_ai import run_the_last_ai
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Position
from the_last_ai.world_model.planning import evaluate_action_options, simulate_action_sequence


def test_multi_step_internal_simulation():
    sim = Simulation.create(
        SimulationConfig(
            seed=0,
            n_agents=4,
            n_resources=4,
            agent_type="PredictiveAgent",
            width=12,
            height=10,
        )
    )
    for i, aid in enumerate(sim.active_agent_ids()):
        sim.world.agent_positions[aid] = Position(3 + i, 3)
    sim.run(25)
    agent = sim.agents["agent_000"]
    assert isinstance(agent, PredictiveAgent)
    agent.observe(sim.world)
    seq = simulate_action_sequence(
        agent.world_model, start=agent.last_observation.position, actions=["east", "east"]
    )
    assert len(seq.predicted_outcomes) == 3  # start + 2 steps
    plan = evaluate_action_options(
        agent.world_model, start=agent.last_observation.position, horizon=1
    )
    assert plan["options_evaluated"] == 5
    assert agent.imagine_sequence(["north", "west"])["query"].startswith("sequence:")


def test_restoration_persists_then_adapts(tmp_path):
    result = run_restoration(
        seed=1,
        formation_ticks=20,
        absence_ticks=12,
        restoration_ticks=20,
        output_dir=tmp_path / "exp6",
    )["result"]
    assert result["at_removal"]["world_model_has_B"] or result["at_removal"]["relationship"]
    assert result["during_absence"]["memory_persisted"] or result["during_absence"]["relationship_persisted"]
    assert result["during_absence"]["residual_expectation_rate"] >= 0.0
    assert result["after_restoration"]["final_relationship"] is not None


def test_counterfactual_experiment(tmp_path):
    result = run_counterfactual_reasoning(
        seed=0,
        n_ticks=30,
        n_agents=4,
        n_resources=4,
        output_dir=tmp_path / "exp7",
    )["result"]
    assert result["plan_horizon_2"]["options_evaluated"] > 0
    assert result["imagine_east_east_north"]["predicted_outcomes"]


def test_the_last_ai_small_collapse(tmp_path):
    result = run_the_last_ai(
        seed=0,
        initial_agents=8,
        schedule=[4, 2, 1],
        ticks_between=8,
        final_ticks=10,
        n_resources=6,
        width=16,
        height=12,
        output_dir=tmp_path / "last",
        compare_control=True,
    )
    report = result["report"]
    assert report["final_population"] == 1
    assert any(e["event"] == "collapse_to_1" for e in report["events"])
    assert report["control_comparison"] is not None
    assert report["architecture_frozen"] is True


def test_architecture_comparison_smoke(tmp_path):
    result = run_architecture_comparison(
        seeds=[0, 1],
        initial_agents=6,
        schedule=[3, 1],
        ticks_between=5,
        final_ticks=6,
        architectures=["RandomAgent", "MemoryAgent", "PredictiveAgent"],
        output_dir=tmp_path / "cmp",
    )
    assert "RandomAgent" in result["aggregates"]
    assert result["aggregates"]["PredictiveAgent"]["trials"] == 2
    assert result["report"]["architecture_frozen"] is True
