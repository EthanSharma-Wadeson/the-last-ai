"""Phase 4 experiment — predictive world model vs non-predictive control."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialAgent, SocialConfig
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Position


def _run_predictive(
    *,
    seed: int,
    n_ticks: int,
    n_agents: int,
    n_resources: int,
    disappear_at: int | None,
) -> dict:
    config = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        n_resources=n_resources,
        width=16,
        height=12,
        agent_type="PredictiveAgent",
        perception_radius=4,
        initial_energy=80.0,
        resource_energy=20.0,
        resource_regen_interval=10,
    )
    sim = Simulation.create(config)
    # Cluster agents so behavioural models have signal.
    origin = Position(4, 4)
    for i, agent_id in enumerate(sim.active_agent_ids()):
        sim.world.agent_positions[agent_id] = origin.offset(i % 3, i // 3)

    for agent in sim.agents.values():
        if isinstance(agent, SocialAgent):
            agent.social_config = SocialConfig(cooperate_bias=0.6, interact_probability=0.7)

    pre_error = []
    post_error = []
    target_id = None

    for t in range(n_ticks):
        if disappear_at is not None and t == disappear_at:
            # Remove one non-observer agent.
            ids = [aid for aid in sim.active_agent_ids() if aid != "agent_000"]
            if ids:
                target_id = ids[0]
                sim.disappear(target_id)
        sim.step()
        observer = sim.agents["agent_000"]
        assert isinstance(observer, PredictiveAgent)
        err = observer.world_model.mean_error_ema
        if disappear_at is None or t < disappear_at:
            pre_error.append(err)
        else:
            post_error.append(err)

    observer = sim.agents["agent_000"]
    assert isinstance(observer, PredictiveAgent)
    absence = None
    if target_id is not None:
        absence = observer.counterfactual_absence(target_id)

    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "disappear_at": disappear_at,
        "removed_agent": target_id,
        "final_mean_error_ema": observer.world_model.mean_error_ema,
        "total_scored": observer.world_model.total_scored,
        "pre_disappearance_mean_error": (
            sum(pre_error) / len(pre_error) if pre_error else None
        ),
        "post_disappearance_mean_error": (
            sum(post_error) / len(post_error) if post_error else None
        ),
        "world_model": observer.world_model_summary(),
        "counterfactual_absence": absence,
        "counterfactual_move_east": observer.counterfactual_move("east"),
    }


def run_world_model_experiment(
    *,
    seed: int = 0,
    n_ticks: int = 150,
    n_agents: int = 5,
    n_resources: int = 8,
    disappear_at: int = 90,
    output_dir: str | Path = "outputs/experiment_world_model",
) -> dict:
    """
    Core Phase 4 question:
    Can an agent learn an internal predictive model of a changing world,
    including other agents, and continue using that model when parts disappear?
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stable = _run_predictive(
        seed=seed,
        n_ticks=n_ticks,
        n_agents=n_agents,
        n_resources=n_resources,
        disappear_at=None,
    )
    changing = _run_predictive(
        seed=seed + 1,
        n_ticks=n_ticks,
        n_agents=n_agents,
        n_resources=n_resources,
        disappear_at=disappear_at,
    )

    comparison = {
        "seed": seed,
        "question": (
            "Can an agent learn an internal predictive model of a changing world, "
            "including other agents, and continue using that model when parts disappear?"
        ),
        "stable_world": stable,
        "changing_world": changing,
        "notes": {
            "error_should_generally_fall_with_experience": True,
            "after_disappearance_presence_error_may_rise": True,
            "model_of_removed_agent_should_persist": changing.get("removed_agent")
            in changing["world_model"].get("entity_models", {}),
        },
    }

    path = output_dir / f"comparison_seed{seed}.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "comparison": comparison,
        "output_dir": str(output_dir),
        "comparison_path": str(path),
    }


if __name__ == "__main__":
    result = run_world_model_experiment()
    changing = result["comparison"]["changing_world"]
    print(
        json.dumps(
            {
                "final_mean_error_ema": changing["final_mean_error_ema"],
                "pre": changing["pre_disappearance_mean_error"],
                "post": changing["post_disappearance_mean_error"],
                "tracked_entities": changing["world_model"]["tracked_entities"],
            },
            indent=2,
        )
    )
