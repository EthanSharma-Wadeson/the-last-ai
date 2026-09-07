"""Experiment 0 — random baseline (docs/EXPERIMENTS.md)."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.simulation.engine import Simulation, SimulationConfig


def run_random_baseline(
    *,
    seed: int = 0,
    n_ticks: int = 200,
    n_agents: int = 8,
    width: int = 20,
    height: int = 12,
    output_dir: str | Path = "outputs/experiment_0",
) -> dict:
    """Multiple agents move randomly; records movement, interactions, energy, population."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        seed=seed,
        n_agents=n_agents,
        width=width,
        height=height,
        agent_type="RandomAgent",
    )
    sim = Simulation.create(config)
    metrics = sim.run(n_ticks)

    snapshot_path = sim.save(output_dir / f"snapshot_seed{seed}.json")
    summary = metrics.summary()
    summary_path = output_dir / f"summary_seed{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    ascii_path = output_dir / f"final_grid_seed{seed}.txt"
    ascii_path.write_text(sim.render() + "\n", encoding="utf-8")

    return {
        "seed": seed,
        "n_ticks": n_ticks,
        "n_agents": n_agents,
        "summary": summary,
        "snapshot_path": str(snapshot_path),
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    result = run_random_baseline()
    print(json.dumps(result["summary"], indent=2))
