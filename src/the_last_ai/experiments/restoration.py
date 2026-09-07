"""Experiment 6 — restoration: does B's representation persist, then re-adapt?"""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.predictive_agent import PredictiveAgent
from the_last_ai.agents.social_agent import SocialConfig
from the_last_ai.experiments.snapshots import capture_agent_mind
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Action, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _force_cooperate(sim: Simulation, a: str, b: str, n_ticks: int) -> None:
    for _ in range(n_ticks):
        origin = sim.world.agent_positions[a]
        sim.world.agent_positions[b] = origin.offset(1, 0)
        observations = {
            aid: sim.agents[aid].observe(sim.world) for aid in sim.active_agent_ids()
        }
        actions = {aid: Action.STAY for aid in sim.active_agent_ids()}
        actions[a] = Action.COOPERATE
        actions[b] = Action.COOPERATE
        outcomes = sim._resolve_social_interactions(actions, dict(sim.world.agent_positions))
        sim.interaction_log.extend(outcomes)
        sim.world.advance_tick()
        sim.social_graph.sync_population(sim.agents)
        next_obs = {
            aid: sim.agents[aid].observe(sim.world, cognitive=False)
            for aid in sim.active_agent_ids()
        }
        for aid in (a, b):
            sim.agents[aid].learn(
                observations[aid],
                actions[aid],
                reward=0.2,
                next_observation=next_obs[aid],
            )
            sim.agents[aid].update_after_tick()


def run_restoration(
    *,
    seed: int = 0,
    formation_ticks: int = 40,
    absence_ticks: int = 30,
    restoration_ticks: int = 40,
    output_dir: str | Path = "outputs/experiment_6",
) -> dict:
    """
    A ↔ B → B disappears → A keeps predicting B → B returns → measure re-adaptation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = ExperimentRNG.from_seed(seed)
    grid = Grid.empty(14, 10, bordered=True)
    world = World(grid=grid)
    cfg = SocialConfig(cooperate_bias=1.2, interact_probability=0.9, enable_indirect_memory_hook=False)
    agents = {
        "A": PredictiveAgent("A", perception_radius=4, social_config=cfg),
        "B": PredictiveAgent("B", perception_radius=4, social_config=cfg),
    }
    world.place_agent("A", Position(4, 4))
    world.place_agent("B", Position(5, 4))
    sim = Simulation(
        config=SimulationConfig(seed=seed, n_agents=2, agent_type="PredictiveAgent", width=14, height=10),
        world=world,
        agents=agents,
        rng=rng,
        metrics=MetricsRecorder(),
    )

    # Stage 1: formation
    _force_cooperate(sim, "A", "B", formation_ticks)
    a = sim.agents["A"]
    assert isinstance(a, PredictiveAgent)
    after_formation = capture_agent_mind(a, sim=sim, label="after_formation")
    presence_pred_at_removal = a.counterfactual_absence("B")
    rel_at_removal = a.relationships.get("B")
    mem_strength_at_removal = a.entity_strength("B")
    model_b = a.world_model.agents.entities.get("B")

    # Stage 2: disappearance
    sim.disappear("B")
    absence_presence_errors: list[float] = []
    for _ in range(absence_ticks):
        # Keep A observing / predicting alone.
        a.observe(sim.world)
        # Score whether model still predicts B present.
        pred = a.counterfactual_absence("B")
        outcomes = pred.get("predicted_outcomes") or []
        if outcomes:
            # If model still predicts presence=True while B is gone, that is residual expectation.
            predicted_present = bool(outcomes[0].get("presence_prediction"))
            absence_presence_errors.append(1.0 if predicted_present else 0.0)
        a.update_after_tick()
        sim.world.advance_tick()

    during_absence = capture_agent_mind(a, sim=sim, label="during_absence")
    residual_expectation_rate = (
        sum(absence_presence_errors) / len(absence_presence_errors)
        if absence_presence_errors
        else 0.0
    )

    # Stage 3: restoration
    sim.restore("B", Position(5, 4))
    recognition_tick = None
    trust_series: list[float] = []
    error_series: list[float] = []
    for i in range(restoration_ticks):
        origin = sim.world.agent_positions["A"]
        sim.world.agent_positions["B"] = origin.offset(1, 0)
        _force_cooperate(sim, "A", "B", 1)
        rel = a.relationships.get("B")
        trust_series.append(rel.trust if rel else 0.5)
        error_series.append(a.world_model.mean_error_ema)
        if recognition_tick is None and a.ltm.get("B") is not None:
            # First post-restoration sighting / interaction already happened.
            recognition_tick = i

    after_restoration = capture_agent_mind(a, sim=sim, label="after_restoration")
    rel_final = a.relationships.get("B")

    # Adaptation speed: ticks until trust recovers to 90% of pre-removal trust.
    target_trust = (rel_at_removal.trust if rel_at_removal else 0.5) * 0.9
    adapt_ticks = None
    for i, trust in enumerate(trust_series):
        if trust >= target_trust:
            adapt_ticks = i + 1
            break

    result = {
        "seed": seed,
        "formation_ticks": formation_ticks,
        "absence_ticks": absence_ticks,
        "restoration_ticks": restoration_ticks,
        "at_removal": {
            "relationship": rel_at_removal.to_dict() if rel_at_removal else None,
            "memory_strength": mem_strength_at_removal,
            "world_model_has_B": model_b is not None,
            "presence_prediction": presence_pred_at_removal,
        },
        "during_absence": {
            "residual_expectation_rate": residual_expectation_rate,
            "memory_persisted": "B" in (during_absence.get("entity_representations") or {}),
            "relationship_persisted": "B"
            in ((during_absence.get("relationships") or {}).get("relationships") or {}),
            "prediction_error_ema": during_absence.get("prediction_error_ema"),
        },
        "after_restoration": {
            "recognition_tick": recognition_tick,
            "adaptation_ticks_to_90pct_trust": adapt_ticks,
            "final_relationship": rel_final.to_dict() if rel_final else None,
            "final_memory_strength": a.entity_strength("B"),
            "trust_series": trust_series,
            "error_series": error_series,
        },
        "minds": {
            "after_formation": {
                "relationships": after_formation.get("relationships", {}).get("count"),
                "tracked_entities": after_formation.get("world_model", {}).get("tracked_entities"),
            },
            "during_absence": {
                "relationships": during_absence.get("relationships", {}).get("count"),
                "tracked_entities": during_absence.get("world_model", {}).get("tracked_entities"),
            },
            "after_restoration": {
                "relationships": after_restoration.get("relationships", {}).get("count"),
                "tracked_entities": after_restoration.get("world_model", {}).get("tracked_entities"),
            },
        },
        "interpretation_note": (
            "Residual expectation during absence indicates representation persistence. "
            "Adaptation ticks after return measure re-integration speed."
        ),
    }

    path = output_dir / f"restoration_seed{seed}.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    from the_last_ai.experiments.interpret import write_interpretation

    interpretation_path = write_interpretation(
        path, output_path=output_dir / f"restoration_seed{seed}.md"
    )
    return {
        "result": result,
        "output_dir": str(output_dir),
        "path": str(path),
        "interpretation_path": str(interpretation_path),
    }


if __name__ == "__main__":
    out = run_restoration()
    r = out["result"]
    print(
        json.dumps(
            {
                "residual_expectation_rate": r["during_absence"]["residual_expectation_rate"],
                "adaptation_ticks": r["after_restoration"]["adaptation_ticks_to_90pct_trust"],
                "memory_persisted": r["during_absence"]["memory_persisted"],
            },
            indent=2,
        )
    )
