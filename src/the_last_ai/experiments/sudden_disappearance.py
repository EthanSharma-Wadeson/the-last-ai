"""Experiment 3 — sudden disappearance after bidirectional social interaction."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.social_agent import CooperativePartner, SocialAgent, SocialConfig
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import ACTION_DELTA, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _build_social_world(
    *,
    seed: int,
    partner_id: str,
    partner_type: str,
    bystander_id: str | None = "bystander",
    width: int = 12,
    height: int = 10,
    cooperate_bias_observer: float = 0.8,
) -> tuple[Simulation, str, str]:
    """A (observer) adjacent to B (partner); optional distant bystander for exploration."""
    rng = ExperimentRNG.from_seed(seed)
    grid = Grid.empty(width, height, bordered=True)
    world = World(grid=grid)

    observer_id = "observer"
    observer = SocialAgent(
        observer_id,
        perception_radius=4,
        social_config=SocialConfig(
            cooperate_bias=cooperate_bias_observer,
            interact_probability=0.85,
        ),
    )

    if partner_type == "CooperativePartner":
        partner = CooperativePartner(partner_id, perception_radius=4)
    else:
        partner = SocialAgent(
            partner_id,
            perception_radius=4,
            social_config=SocialConfig(cooperate_bias=0.8, interact_probability=0.85),
        )

    agents = {observer_id: observer, partner_id: partner}
    world.place_agent(observer_id, Position(4, 4))
    world.place_agent(partner_id, Position(5, 4))

    if bystander_id is not None:
        bystander = SocialAgent(
            bystander_id,
            perception_radius=4,
            social_config=SocialConfig(interact_probability=0.5),
        )
        world.place_agent(bystander_id, Position(width - 3, height - 3))
        agents[bystander_id] = bystander

    config = SimulationConfig(
        seed=seed,
        width=width,
        height=height,
        n_agents=len(agents),
        perception_radius=4,
        agent_type="SocialAgent",
    )
    sim = Simulation(
        config=config,
        world=world,
        agents=agents,
        rng=rng,
        metrics=MetricsRecorder(),
    )
    return sim, observer_id, partner_id


def _toward_location(sim: Simulation, observer_id: str, target: tuple[int, int] | None) -> bool:
    if target is None:
        return False
    observer = sim.agents[observer_id]
    action = observer.last_action
    if action is None or observer_id not in sim.world.agent_positions:
        return False
    pos = sim.world.agent_positions[observer_id]
    dx, dy = ACTION_DELTA[action]
    nxt = pos.offset(dx, dy)
    goal = Position(*target)
    return nxt.manhattan(goal) < pos.manhattan(goal)


def _run_condition(
    *,
    seed: int,
    label: str,
    interaction_ticks: int,
    post_ticks: int,
) -> dict:
    sim, observer_id, partner_id = _build_social_world(
        seed=seed,
        partner_id="partner",
        partner_type="CooperativePartner",
    )
    observer = sim.agents[observer_id]
    assert isinstance(observer, SocialAgent)

    sim.run(interaction_ticks)

    rel = observer.relationships.get(partner_id)
    relationship_at_removal = rel.to_dict() if rel else None
    memory_strength_at_removal = observer.entity_strength(partner_id)
    graph_weight_at_removal = sim.social_graph.weight(observer_id, partner_id)
    reverse_weight_at_removal = sim.social_graph.weight(partner_id, observer_id)
    expected_location = None
    mem = observer.ltm.get(partner_id)
    if mem and mem.expected_location:
        expected_location = list(mem.expected_location)
    last_partner_pos = sim.world.agent_positions[partner_id].as_tuple()

    interactions_before = sum(
        1
        for outcome in sim.interaction_log
        if {outcome.agent_a, outcome.agent_b} == {observer_id, partner_id}
    )

    sim.disappear(partner_id)

    strength_series: list[dict] = []
    approach_former_location = 0
    bystander_approaches = 0
    social_actions_post = 0

    for _ in range(post_ticks):
        before_partner_rel_strength = (
            observer.relationships.get(partner_id).strength
            if observer.relationships.get(partner_id)
            else 0.0
        )
        sim.step()
        tick = sim.world.tick
        rel_now = observer.relationships.get(partner_id)
        strength_series.append(
            {
                "tick": tick,
                "relationship_strength": rel_now.strength if rel_now else 0.0,
                "predicted_utility": rel_now.predicted_utility if rel_now else 0.0,
                "trust": rel_now.trust if rel_now else 0.5,
                "memory_strength": observer.entity_strength(partner_id, tick),
            }
        )
        target = tuple(expected_location) if expected_location else last_partner_pos
        if _toward_location(sim, observer_id, target):
            approach_former_location += 1
        if "bystander" in sim.world.agent_positions and _toward_location(
            sim,
            observer_id,
            sim.world.agent_positions["bystander"].as_tuple(),
        ):
            bystander_approaches += 1
        if observer.last_action and observer.last_action.value in {
            "cooperate",
            "compete",
            "share",
            "communicate",
        }:
            social_actions_post += 1
        # Relationship strength for absent partners is frozen (no new outcomes);
        # keep series informative via memory decay already tracked.
        _ = before_partner_rel_strength

    rel_final = observer.relationships.get(partner_id)
    return {
        "label": label,
        "interaction_ticks": interaction_ticks,
        "post_ticks": post_ticks,
        "interactions_with_partner": interactions_before,
        "relationship_at_removal": relationship_at_removal,
        "memory_strength_at_removal": memory_strength_at_removal,
        "graph_weight_observer_to_partner": graph_weight_at_removal,
        "graph_weight_partner_to_observer": reverse_weight_at_removal,
        "relationship_final": rel_final.to_dict() if rel_final else None,
        "memory_strength_final": observer.entity_strength(partner_id),
        "approach_former_location": approach_former_location,
        "bystander_approaches": bystander_approaches,
        "social_actions_post": social_actions_post,
        "seek_attempts": observer.seek_attempts,
        "social_graph": sim.social_graph.summary(),
        "two_hop_from_observer": [
            {"node": n, "path_strength": s, "via": v}
            for n, s, v in sim.social_graph.two_hop_neighbors(observer_id)
        ],
        "strength_series": strength_series,
        "observer_summary": observer.relationship_summary(),
    }


def run_sudden_disappearance(
    *,
    seed: int = 0,
    familiar_ticks: int = 100,
    unfamiliar_ticks: int = 6,
    post_ticks: int = 60,
    output_dir: str | Path = "outputs/experiment_3",
) -> dict:
    """
    Familiar: A ↔ B with repeated positive interactions, then B disappears.
    Unfamiliar control: A ↔ C with minimal interaction, then C disappears.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    familiar = _run_condition(
        seed=seed,
        label="familiar_partner",
        interaction_ticks=familiar_ticks,
        post_ticks=post_ticks,
    )
    unfamiliar = _run_condition(
        seed=seed + 17,
        label="unfamiliar_partner",
        interaction_ticks=unfamiliar_ticks,
        post_ticks=post_ticks,
    )

    def _rel_field(result: dict, field: str, default: float = 0.0) -> float:
        rel = result.get("relationship_at_removal") or {}
        return float(rel.get(field, default))

    comparison = {
        "seed": seed,
        "familiar": {k: v for k, v in familiar.items() if k != "strength_series"},
        "unfamiliar": {k: v for k, v in unfamiliar.items() if k != "strength_series"},
        "deltas": {
            "relationship_strength": _rel_field(familiar, "strength") - _rel_field(unfamiliar, "strength"),
            "trust": _rel_field(familiar, "trust", 0.5) - _rel_field(unfamiliar, "trust", 0.5),
            "predicted_utility": _rel_field(familiar, "predicted_utility")
            - _rel_field(unfamiliar, "predicted_utility"),
            "interaction_count": _rel_field(familiar, "interaction_count")
            - _rel_field(unfamiliar, "interaction_count"),
            "memory_strength_at_removal": (
                familiar["memory_strength_at_removal"] - unfamiliar["memory_strength_at_removal"]
            ),
            "approach_former_location": (
                familiar["approach_former_location"] - unfamiliar["approach_former_location"]
            ),
            "graph_weight": (
                familiar["graph_weight_observer_to_partner"]
                - unfamiliar["graph_weight_observer_to_partner"]
            ),
        },
    }

    path = output_dir / f"comparison_seed{seed}.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")
    series_path = output_dir / f"series_seed{seed}.json"
    series_path.write_text(
        json.dumps(
            {
                "familiar": familiar["strength_series"],
                "unfamiliar": unfamiliar["strength_series"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return {
        "comparison": comparison,
        "output_dir": str(output_dir),
        "comparison_path": str(path),
        "series_path": str(series_path),
    }


if __name__ == "__main__":
    result = run_sudden_disappearance()
    print(json.dumps(result["comparison"]["deltas"], indent=2))
