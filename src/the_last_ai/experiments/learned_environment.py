"""Experiment 1 — learned environment vs random baseline (docs/EXPERIMENTS.md)."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.simulation.engine import Simulation, SimulationConfig


def _run_condition(
    *,
    agent_type: str,
    seed: int,
    n_ticks: int,
    n_agents: int,
    n_resources: int,
    width: int,
    height: int,
) -> dict:
    config = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        n_resources=n_resources,
        width=width,
        height=height,
        agent_type=agent_type,
        initial_energy=40.0,
        energy_cost_per_tick=0.2,
        resource_energy=25.0,
        resource_regen_interval=12,
    )
    sim = Simulation.create(config)
    metrics = sim.run(n_ticks)
    return {
        "agent_type": agent_type,
        "seed": seed,
        "summary": metrics.summary(),
        "agents": sim.agent_learning_summary(),
        "final_grid": sim.render(),
    }


def run_learned_environment(
    *,
    seed: int = 0,
    n_ticks: int = 300,
    n_agents: int = 4,
    n_resources: int = 10,
    width: int = 16,
    height: int = 12,
    output_dir: str | Path = "outputs/experiment_1",
) -> dict:
    """Compare RandomAgent and ValueBasedAgent in a resource-rich world."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_result = _run_condition(
        agent_type="RandomAgent",
        seed=seed,
        n_ticks=n_ticks,
        n_agents=n_agents,
        n_resources=n_resources,
        width=width,
        height=height,
    )
    learner_result = _run_condition(
        agent_type="ValueBasedAgent",
        seed=seed,
        n_ticks=n_ticks,
        n_agents=n_agents,
        n_resources=n_resources,
        width=width,
        height=height,
    )

    comparison = {
        "seed": seed,
        "n_ticks": n_ticks,
        "n_agents": n_agents,
        "n_resources": n_resources,
        "random": random_result["summary"],
        "value_based": learner_result["summary"],
        "delta": {
            "final_mean_energy": (
                learner_result["summary"]["final_mean_energy"]
                - random_result["summary"]["final_mean_energy"]
            ),
            "total_resources_collected": (
                learner_result["summary"]["total_resources_collected"]
                - random_result["summary"]["total_resources_collected"]
            ),
            "mean_reward": (
                learner_result["summary"]["mean_reward"]
                - random_result["summary"]["mean_reward"]
            ),
        },
        "learner_agents": learner_result["agents"],
    }

    path = output_dir / f"comparison_seed{seed}.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    (output_dir / f"random_grid_seed{seed}.txt").write_text(
        random_result["final_grid"] + "\n", encoding="utf-8"
    )
    (output_dir / f"learner_grid_seed{seed}.txt").write_text(
        learner_result["final_grid"] + "\n", encoding="utf-8"
    )

    return {
        "comparison": comparison,
        "output_dir": str(output_dir),
        "comparison_path": str(path),
    }


if __name__ == "__main__":
    result = run_learned_environment()
    print(json.dumps(result["comparison"]["delta"], indent=2))
