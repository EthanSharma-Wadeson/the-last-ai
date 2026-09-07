"""The Last AI — centrepiece population-collapse experiment."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.experiments.snapshots import capture_agent_mind, mind_delta
from the_last_ai.observatory.export import export_agent_life
from the_last_ai.simulation.engine import Simulation, SimulationConfig


DEFAULT_SCHEDULE = [50, 25, 10, 5, 2, 1]


def _configure_predictive(sim: Simulation) -> None:
    for agent in sim.agents.values():
        if isinstance(agent, PredictiveAgent):
            agent.social_config = SocialConfig(
                cooperate_bias=0.45,
                interact_probability=0.6,
                enable_indirect_memory_hook=True,
                enable_symbolic_communication=True,
                communication_bias=0.35,
            )


def run_the_last_ai(
    *,
    seed: int = 0,
    initial_agents: int = 100,
    schedule: list[int] | None = None,
    ticks_between: int = 30,
    final_ticks: int = 40,
    n_resources: int = 40,
    width: int = 40,
    height: int = 30,
    output_dir: str | Path = "outputs/the_last_ai",
    compare_control: bool = True,
) -> dict:
    """
    Centrepiece question:

    When a population disappears, what structure of that population remains
    encoded in the final surviving artificial mind?

    Schedule default: 100 → 50 → 25 → 10 → 5 → 2 → 1
    """
    schedule = schedule or list(DEFAULT_SCHEDULE)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    events_dir = output_dir / f"events_seed{seed}"
    events_dir.mkdir(parents=True, exist_ok=True)

    config = SimulationConfig(
        seed=seed,
        n_agents=initial_agents,
        n_resources=n_resources,
        width=width,
        height=height,
        agent_type="PredictiveAgent",
        perception_radius=4,
        initial_energy=100.0,
        resource_energy=18.0,
        resource_regen_interval=12,
    )
    sim = Simulation.create(config)
    _configure_predictive(sim)

    survivor_id = "agent_000"
    sim.observatory.focus({survivor_id})
    event_log: list[dict] = []
    mind_series: list[dict] = []

    def _record(event_name: str) -> dict:
        survivor = sim.agents[survivor_id]
        assert isinstance(survivor, PredictiveAgent)
        mind = capture_agent_mind(survivor, sim=sim, label=event_name)
        path = events_dir / f"{event_name}_tick{sim.world.tick}.json"
        path.write_text(json.dumps(mind, indent=2, sort_keys=True), encoding="utf-8")
        entry = {
            "event": event_name,
            "tick": sim.world.tick,
            "population": len(sim.active_agent_ids()),
            "mind_path": str(path),
            "summary": {
                "entity_memory_count": mind.get("entity_memory", {}).get("entity_count", 0),
                "relationships": mind.get("relationships", {}).get("count", 0),
                "tracked_entities": mind.get("world_model", {}).get("tracked_entities", 0),
                "prediction_error_ema": mind.get("prediction_error_ema"),
                "graph_edges": mind.get("social_graph", {}).get("edge_count", 0),
                "messages_sent": mind.get("communication", {}).get("messages_sent", 0),
                "messages_received": mind.get("communication", {}).get("messages_received", 0),
                "communication_memory_count": mind.get("communication", {})
                .get("memory", {})
                .get("count", 0),
                "historical_communication_count": mind.get("communication", {})
                .get("memory", {})
                .get("historical_count", 0),
            },
        }
        event_log.append(entry)
        mind_series.append(mind)
        return mind

    _record("start")
    sim.run(ticks_between)
    _record("after_initial_interaction")

    for target_pop in schedule:
        sim.run(ticks_between)
        active = [aid for aid in sim.active_agent_ids() if aid != survivor_id]
        # Prefer removing least-interacted agents with the survivor first? Keep simple: LIFO by id.
        active.sort(reverse=True)
        while len(sim.active_agent_ids()) > target_pop and active:
            sim.disappear(active.pop())
        sim.observatory.on_collapse_stage(
            sim,
            survivor_id=survivor_id,
            population=len(sim.active_agent_ids()),
            label=f"collapse_to_{target_pop}",
        )
        _record(f"collapse_to_{target_pop}")

    sim.run(final_ticks)
    final_mind = _record("final_survivor")
    sim.observatory.finalize_survivor(sim, survivor_id)

    # Control: inexperienced PredictiveAgent with same architecture, no history.
    control_comparison = None
    if compare_control:
        control = Simulation.create(
            SimulationConfig(
                seed=seed + 999,
                n_agents=1,
                n_resources=n_resources // 4 or 1,
                width=min(20, width),
                height=min(16, height),
                agent_type="PredictiveAgent",
                perception_radius=4,
            )
        )
        _configure_predictive(control)
        control.run(final_ticks)
        control_mind = capture_agent_mind(
            control.agents["agent_000"], sim=control, label="inexperienced_control"
        )
        control_comparison = {
            "control": {
                "entity_memory_count": control_mind.get("entity_memory", {}).get("entity_count", 0),
                "relationships": control_mind.get("relationships", {}).get("count", 0),
                "tracked_entities": control_mind.get("world_model", {}).get("tracked_entities", 0),
                "prediction_error_ema": control_mind.get("prediction_error_ema"),
            },
            "survivor": {
                "entity_memory_count": final_mind.get("entity_memory", {}).get("entity_count", 0),
                "relationships": final_mind.get("relationships", {}).get("count", 0),
                "tracked_entities": final_mind.get("world_model", {}).get("tracked_entities", 0),
                "prediction_error_ema": final_mind.get("prediction_error_ema"),
            },
            "delta_survivor_minus_control": {
                "entity_memory_count": final_mind.get("entity_memory", {}).get("entity_count", 0)
                - control_mind.get("entity_memory", {}).get("entity_count", 0),
                "relationships": final_mind.get("relationships", {}).get("count", 0)
                - control_mind.get("relationships", {}).get("count", 0),
                "tracked_entities": final_mind.get("world_model", {}).get("tracked_entities", 0)
                - control_mind.get("world_model", {}).get("tracked_entities", 0),
            },
        }

    # Trajectory of structure across collapse events.
    trajectory = []
    for i in range(1, len(mind_series)):
        trajectory.append(
            {
                "from": mind_series[i - 1]["label"],
                "to": mind_series[i]["label"],
                "delta": mind_delta(mind_series[i - 1], mind_series[i]),
            }
        )

    report = {
        "question": (
            "When a population disappears, what structure of that population remains "
            "encoded in the final surviving artificial mind?"
        ),
        "seed": seed,
        "initial_agents": initial_agents,
        "schedule": schedule,
        "ticks_between": ticks_between,
        "final_ticks": final_ticks,
        "events": event_log,
        "trajectory_deltas": trajectory,
        "final_population": len(sim.active_agent_ids()),
        "survivor_id": survivor_id,
        "control_comparison": control_comparison,
        "architecture_frozen": True,
        "note": (
            "Prototype architecture is frozen for research trials. Prefer multi-seed "
            "comparisons over adding new cognitive features."
        ),
    }

    report_path = output_dir / f"report_seed{seed}.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    sim.save(output_dir / f"final_sim_seed{seed}.json")

    # Export survivor life observatory (JSON authoritative + TXT/HTML presentation).
    survivor_history = sim.observatory.histories.get(survivor_id)
    observatory_paths = None
    if survivor_history is not None:
        observatory_dir = output_dir / f"observatory_seed{seed}"
        observatory_paths = export_agent_life(
            survivor_history,
            observatory_dir,
            mind=final_mind,
            active_ids=set(sim.active_agent_ids()),
            meta={
                "experiment": "the_last_ai",
                "seed": seed,
                "schedule": schedule,
                "initial_agents": initial_agents,
                "survivor_id": survivor_id,
                "question": report["question"],
            },
        )
        report["observatory"] = {
            "survivor_id": survivor_id,
            "paths": observatory_paths,
            "overview": survivor_history.overview(
                active_ids=set(sim.active_agent_ids()),
                final_tick=sim.world.tick,
            ),
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    from the_last_ai.experiments.interpret import write_interpretation

    interpretation_path = write_interpretation(report_path, output_path=output_dir / f"report_seed{seed}.md")

    return {
        "report": report,
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "interpretation_path": str(interpretation_path),
        "observatory_paths": observatory_paths,
        "events_dir": str(events_dir),
    }


# Backwards-compatible wrapper used by earlier CLI / tests.
def run_gradual_population_reduction(**kwargs):
    # Map old kwargs.
    if "output_dir" not in kwargs:
        kwargs["output_dir"] = "outputs/experiment_5"
    if "initial_agents" not in kwargs:
        kwargs["initial_agents"] = 20
    if "schedule" not in kwargs:
        kwargs["schedule"] = [10, 5, 2, 1]
    result = run_the_last_ai(**kwargs)
    # Preserve old return shape keys where possible.
    report = result["report"]
    survivor_event = next(e for e in report["events"] if e["event"] == "final_survivor")
    return {
        "snapshot": {
            "seed": report["seed"],
            "initial_agents": report["initial_agents"],
            "schedule": report["schedule"],
            "final_population": report["final_population"],
            "survivor_id": report["survivor_id"],
            "survivor_world_model": {"tracked_entities": survivor_event["summary"]["tracked_entities"],
                                    "mean_error_ema": survivor_event["summary"]["prediction_error_ema"]},
            "survivor_relationships": {"count": survivor_event["summary"]["relationships"]},
            "history": report["events"],
            "question": report["question"],
        },
        "output_dir": result["output_dir"],
        "snapshot_path": result["report_path"],
        "report": report,
    }


if __name__ == "__main__":
    # Small local demo; full 100-agent run via CLI.
    result = run_the_last_ai(
        initial_agents=12,
        schedule=[6, 3, 1],
        ticks_between=12,
        final_ticks=15,
        n_resources=8,
        width=18,
        height=14,
    )
    print(json.dumps(result["report"]["control_comparison"], indent=2))
