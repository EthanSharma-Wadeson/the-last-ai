"""Experiment 7 — counterfactual / internal multi-step simulation."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Position


def run_counterfactual_reasoning(
    *,
    seed: int = 0,
    n_ticks: int = 80,
    n_agents: int = 5,
    n_resources: int = 8,
    output_dir: str | Path = "outputs/experiment_7",
) -> dict:
    """
    Train a predictive agent, then ask multi-step internal questions:

    If I do X → what happens next → and then what?
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim = Simulation.create(
        SimulationConfig(
            seed=seed,
            n_agents=n_agents,
            n_resources=n_resources,
            width=16,
            height=12,
            agent_type="PredictiveAgent",
            perception_radius=4,
        )
    )
    # Cluster for behavioural signal.
    for i, aid in enumerate(sim.active_agent_ids()):
        sim.world.agent_positions[aid] = Position(4 + (i % 3), 4 + (i // 3))
        agent = sim.agents[aid]
        if isinstance(agent, PredictiveAgent):
            agent.social_config = SocialConfig(cooperate_bias=0.5, interact_probability=0.7)

    sim.run(n_ticks)
    agent = sim.agents["agent_000"]
    assert isinstance(agent, PredictiveAgent)

    # Ensure last_observation exists for planning.
    agent.observe(sim.world)

    known = list(agent.world_model.agents.entities.keys())
    goal = known[0] if known else None

    sequence_east = agent.imagine_sequence(["east", "east", "north"])
    sequence_stay = agent.imagine_sequence(["stay", "stay"])
    plan = agent.plan(horizon=2, goal_entity=goal)
    absent_plan = None
    if goal is not None:
        absent_plan = agent.imagine_absent_then_act(goal, ["east", "south"])

    result = {
        "seed": seed,
        "n_ticks": n_ticks,
        "tracked_entities": len(known),
        "goal_entity": goal,
        "imagine_east_east_north": sequence_east,
        "imagine_stay_stay": sequence_stay,
        "plan_horizon_2": plan,
        "imagine_absent_then_act": absent_plan,
        "world_model_summary": agent.world_model_summary(),
        "question": "Can the agent evaluate multi-step internal futures from its world model?",
    }

    path = output_dir / f"counterfactual_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return {"result": result, "output_dir": str(output_dir), "path": str(path)}


if __name__ == "__main__":
    out = run_counterfactual_reasoning()
    r = out["result"]
    print(
        json.dumps(
            {
                "tracked_entities": r["tracked_entities"],
                "best_plan": (r["plan_horizon_2"] or {}).get("best"),
            },
            indent=2,
        )
    )
