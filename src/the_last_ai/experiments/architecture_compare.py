"""Multi-seed architecture comparison — research mode after architecture freeze."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialAgent, SocialConfig
from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.experiments.the_last_ai import run_the_last_ai
from the_last_ai.simulation.engine import Simulation, SimulationConfig


ARCHITECTURES = [
    "RandomAgent",
    "ValueBasedAgent",
    "MemoryAgent",
    "SocialAgent",
    "PredictiveAgent",
]


def _configure(sim: Simulation, agent_type: str) -> None:
    for agent in sim.agents.values():
        if isinstance(agent, (SocialAgent, PredictiveAgent)):
            agent.social_config = SocialConfig(
                cooperate_bias=0.4,
                interact_probability=0.55,
                enable_indirect_memory_hook=agent_type == "PredictiveAgent",
            )


def _run_collapse_for_architecture(
    *,
    agent_type: str,
    seed: int,
    initial_agents: int,
    schedule: list[int],
    ticks_between: int,
    final_ticks: int,
) -> dict:
    if agent_type == "PredictiveAgent":
        # Use the full Last AI runner for the predictive case.
        result = run_the_last_ai(
            seed=seed,
            initial_agents=initial_agents,
            schedule=schedule,
            ticks_between=ticks_between,
            final_ticks=final_ticks,
            n_resources=max(8, initial_agents // 3),
            width=max(24, int(initial_agents**0.5) * 4 + 8),
            height=max(18, int(initial_agents**0.5) * 3 + 6),
            output_dir=f"outputs/architecture_compare/_tmp_pred_{seed}",
            compare_control=True,
        )
        report = result["report"]
        final_event = next(e for e in report["events"] if e["event"] == "final_survivor")
        return {
            "agent_type": agent_type,
            "seed": seed,
            "final_population": report["final_population"],
            "summary": final_event["summary"],
            "control_delta": (report.get("control_comparison") or {}).get(
                "delta_survivor_minus_control"
            ),
        }

    width = max(24, int(initial_agents**0.5) * 4 + 8)
    height = max(18, int(initial_agents**0.5) * 3 + 6)
    sim = Simulation.create(
        SimulationConfig(
            seed=seed,
            n_agents=initial_agents,
            n_resources=max(8, initial_agents // 3),
            width=width,
            height=height,
            agent_type=agent_type,
            perception_radius=4,
        )
    )
    _configure(sim, agent_type)
    survivor_id = "agent_000"
    sim.run(ticks_between)
    for target in schedule:
        sim.run(ticks_between)
        active = [aid for aid in sim.active_agent_ids() if aid != survivor_id]
        active.sort(reverse=True)
        while len(sim.active_agent_ids()) > target and active:
            sim.disappear(active.pop())
    sim.run(final_ticks)
    mind = capture_agent_mind(sim.agents[survivor_id], sim=sim, label="final")
    return {
        "agent_type": agent_type,
        "seed": seed,
        "final_population": len(sim.active_agent_ids()),
        "summary": {
            "entity_memory_count": mind.get("entity_memory", {}).get("entity_count", 0)
            if mind.get("entity_memory")
            else 0,
            "relationships": mind.get("relationships", {}).get("count", 0)
            if mind.get("relationships")
            else 0,
            "tracked_entities": mind.get("world_model", {}).get("tracked_entities", 0)
            if mind.get("world_model")
            else 0,
            "prediction_error_ema": mind.get("prediction_error_ema"),
            "graph_edges": mind.get("social_graph", {}).get("edge_count", 0)
            if mind.get("social_graph")
            else 0,
            "total_reward": mind.get("total_reward", 0.0),
            "unique_cells": mind.get("behaviour", {}).get("unique_cells_visited", 0)
            if mind.get("behaviour")
            else 0,
        },
        "control_delta": None,
    }


def _aggregate(values: list[float | int | None]) -> dict:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "stdev": None, "n": 0}
    return {
        "mean": statistics.mean(clean),
        "stdev": statistics.pstdev(clean) if len(clean) > 1 else 0.0,
        "n": len(clean),
        "min": min(clean),
        "max": max(clean),
    }


def run_architecture_comparison(
    *,
    seeds: list[int] | None = None,
    initial_agents: int = 24,
    schedule: list[int] | None = None,
    ticks_between: int = 15,
    final_ticks: int = 20,
    architectures: list[str] | None = None,
    output_dir: str | Path = "outputs/architecture_compare",
) -> dict:
    """
    Frozen-architecture research trials.

    Compare Random / Learning / Memory / Social / Predictive survivors after collapse.
    """
    seeds = seeds or list(range(5))
    schedule = schedule or [12, 6, 3, 1]
    architectures = architectures or list(ARCHITECTURES)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for agent_type in architectures:
        for seed in seeds:
            row = _run_collapse_for_architecture(
                agent_type=agent_type,
                seed=seed,
                initial_agents=initial_agents,
                schedule=schedule,
                ticks_between=ticks_between,
                final_ticks=final_ticks,
            )
            rows.append(row)

    by_arch: dict[str, list[dict]] = {}
    for row in rows:
        by_arch.setdefault(row["agent_type"], []).append(row)

    metrics = [
        "entity_memory_count",
        "relationships",
        "tracked_entities",
        "prediction_error_ema",
        "graph_edges",
        "total_reward",
        "unique_cells",
    ]
    aggregates = {}
    for agent_type, arch_rows in by_arch.items():
        aggregates[agent_type] = {
            metric: _aggregate([r["summary"].get(metric) for r in arch_rows])
            for metric in metrics
        }
        aggregates[agent_type]["trials"] = len(arch_rows)

    report = {
        "question": (
            "Across architectures, what structure remains in the final survivor "
            "after population collapse?"
        ),
        "architecture_frozen": True,
        "seeds": seeds,
        "initial_agents": initial_agents,
        "schedule": schedule,
        "ticks_between": ticks_between,
        "final_ticks": final_ticks,
        "architectures": architectures,
        "aggregates": aggregates,
        "trials": rows,
        "next_steps": [
            "Increase seeds to hundreds for publication-grade confidence intervals",
            "Plot tracked_entities / relationships / prediction_error vs architecture",
            "Document failure cases where survivors retain nothing structured",
            "Only then consider neural world models or larger procedural worlds",
        ],
    }

    path = output_dir / "comparison.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    summary_path = output_dir / "aggregates.json"
    summary_path.write_text(json.dumps(aggregates, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "report": report,
        "aggregates": aggregates,
        "output_dir": str(output_dir),
        "path": str(path),
    }


if __name__ == "__main__":
    out = run_architecture_comparison(
        seeds=[0, 1],
        initial_agents=10,
        schedule=[5, 2, 1],
        ticks_between=8,
        final_ticks=10,
    )
    print(json.dumps(out["aggregates"], indent=2))
