"""Experiment 4 — network-structure effects on A↔C after B disappears.

Phase 3.5: Social memory propagation (controlled).

Do not over-interpret results as "socially propagated memory" until the
network-structure contrast against controls is established.
"""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.social_agent import SocialAgent, SocialConfig
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.social.propagation import PropagationConfig
from the_last_ai.types import ACTION_DELTA, Action, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _social_cfg(*, indirect: bool, cooperate_bias: float = 1.2) -> SocialConfig:
    return SocialConfig(
        interact_radius=1,
        interact_probability=0.95,
        cooperate_bias=cooperate_bias,
        enable_indirect_memory_hook=indirect,
    )


def _make_triangle_sim(
    *,
    seed: int,
    enable_indirect: bool,
    width: int = 14,
    height: int = 10,
) -> Simulation:
    rng = ExperimentRNG.from_seed(seed)
    grid = Grid.empty(width, height, bordered=True)
    world = World(grid=grid)

    agents = {
        "A": SocialAgent("A", perception_radius=5, social_config=_social_cfg(indirect=enable_indirect)),
        "B": SocialAgent("B", perception_radius=5, social_config=_social_cfg(indirect=enable_indirect)),
        "C": SocialAgent("C", perception_radius=5, social_config=_social_cfg(indirect=enable_indirect)),
    }
    # Start separated; stages reposition pairs.
    world.place_agent("A", Position(2, 2))
    world.place_agent("B", Position(7, 5))
    world.place_agent("C", Position(width - 3, height - 3))

    sim = Simulation(
        config=SimulationConfig(
            seed=seed,
            width=width,
            height=height,
            n_agents=3,
            perception_radius=5,
            agent_type="SocialAgent",
        ),
        world=world,
        agents=agents,
        rng=rng,
        metrics=MetricsRecorder(),
        propagation_config=PropagationConfig(
            min_shared_strength=0.2,
            min_shared_interactions=5,
            association_lr=0.25,
            require_communicate_for_memory=False,
        ),
    )
    return sim


def _place_pair(sim: Simulation, left: str, right: str, origin: Position) -> None:
    """Place two agents adjacent; park the third far away if present."""
    sim.world.agent_positions[left] = origin
    sim.world.agent_positions[right] = origin.offset(1, 0)
    others = [aid for aid in sim.world.agent_positions if aid not in {left, right}]
    if others:
        far = Position(sim.world.grid.width - 3, sim.world.grid.height - 3)
        # Avoid colliding with the pair.
        if far.manhattan(origin) < 4:
            far = Position(2, sim.world.grid.height - 3)
        sim.world.agent_positions[others[0]] = far


def _force_pair_cooperate(sim: Simulation, left: str, right: str, n_ticks: int) -> int:
    """
    Controlled interaction: keep pair adjacent and force mutual cooperate each tick.
    Returns number of resolved interactions between the pair.
    """
    count_before = sum(
        1
        for o in sim.interaction_log
        if {o.agent_a, o.agent_b} == {left, right}
    )
    for _ in range(n_ticks):
        # Re-assert adjacency every tick so random movement cannot separate them.
        origin = sim.world.agent_positions[left]
        sim.world.agent_positions[right] = origin.offset(1, 0)

        observations = {
            agent_id: sim.agents[agent_id].observe(sim.world)
            for agent_id in sim.active_agent_ids()
        }
        actions = {
            agent_id: Action.STAY
            for agent_id in sim.active_agent_ids()
        }
        actions[left] = Action.COOPERATE
        actions[right] = Action.COOPERATE

        # Minimal step: no movement, resolve social, dynamics, learn.
        social_outcomes = sim._resolve_social_interactions(actions, dict(sim.world.agent_positions))
        sim.interaction_log.extend(social_outcomes)
        energy_gains = sim.world.collect_resources()
        for agent_id, amount in energy_gains.items():
            sim.agents[agent_id].apply_energy_gain(amount)
        sim.world.advance_tick()
        sim.world.regenerate_resources()
        sim.social_graph.sync_population(sim.agents)

        next_obs = {
            agent_id: sim.agents[agent_id].observe(sim.world)
            for agent_id in sim.active_agent_ids()
        }
        for agent_id in sim.active_agent_ids():
            agent = sim.agents[agent_id]
            agent.learn(
                observations[agent_id],
                actions[agent_id],
                reward=0.1 if agent_id in {left, right} else 0.0,
                next_observation=next_obs[agent_id],
                moved=False,
                blocked=False,
            )
            agent.update_after_tick()

        sim.metrics.record_tick(
            tick=sim.world.tick,
            positions=dict(sim.world.agent_positions),
            actions=actions,
            energies={aid: sim.agents[aid].state.energy for aid in sim.world.agent_positions},
            interactions=len(social_outcomes),
            rewards={aid: 0.0 for aid in sim.world.agent_positions},
            resources_collected=len(energy_gains),
        )

    count_after = sum(
        1
        for o in sim.interaction_log
        if {o.agent_a, o.agent_b} == {left, right}
    )
    return count_after - count_before


def _free_pair_phase(sim: Simulation, left: str, right: str, n_ticks: int) -> dict:
    """Allow natural social policy between a placed pair; measure A→C style metrics for left."""
    _place_pair(sim, left, right, Position(5, 4))
    observer: SocialAgent = sim.agents[left]  # type: ignore[assignment]
    partner_id = right

    before_rel = observer.relationships.get(partner_id)
    before_snapshot = before_rel.to_dict() if before_rel else None
    before_utility = before_rel.predicted_utility if before_rel else 0.0
    before_trust = before_rel.trust if before_rel else 0.5
    before_strength = before_rel.strength if before_rel else 0.0

    interactions = 0
    cooperations = 0
    communicates = 0
    b_location_approaches = 0
    b_mem = observer.ltm.get("B")
    b_location = b_mem.expected_location if b_mem else None

    for _ in range(n_ticks):
        # Keep them in interaction range but allow policy choice.
        origin = sim.world.agent_positions[left]
        if sim.world.agent_positions[right].manhattan(origin) > 1:
            sim.world.agent_positions[right] = origin.offset(1, 0)

        positions_before = dict(sim.world.agent_positions)
        observations = {
            agent_id: sim.agents[agent_id].observe(sim.world)
            for agent_id in sim.active_agent_ids()
        }
        actions = {}
        for agent_id in sim.active_agent_ids():
            if agent_id in {left, right}:
                actions[agent_id] = sim.agents[agent_id].select_action(
                    observations[agent_id], sim.rng
                )
            else:
                actions[agent_id] = Action.STAY
            sim.agents[agent_id].last_action = actions[agent_id]

        # Apply only movement for the pair; ignore third party.
        for agent_id in (left, right):
            action = actions[agent_id]
            if action in {Action.COOPERATE, Action.COMPETE, Action.SHARE, Action.COMMUNICATE, Action.INTERACT}:
                continue
            sim.world.apply_action(agent_id, action)

        social_outcomes = sim._resolve_social_interactions(actions, positions_before)
        sim.interaction_log.extend(social_outcomes)
        for outcome in social_outcomes:
            if {outcome.agent_a, outcome.agent_b} == {left, right}:
                interactions += 1
                intents = {outcome.intent_a, outcome.intent_b}
                from the_last_ai.social.types import InteractionKind

                if InteractionKind.COOPERATE in intents:
                    cooperations += 1
                if InteractionKind.COMMUNICATE in intents or outcome.resolved_kind.startswith(
                    "communicate"
                ):
                    communicates += 1

        sim.world.advance_tick()
        sim.social_graph.sync_population(sim.agents)

        next_obs = {
            agent_id: sim.agents[agent_id].observe(sim.world)
            for agent_id in sim.active_agent_ids()
        }
        for agent_id in (left, right):
            agent = sim.agents[agent_id]
            agent.learn(
                observations[agent_id],
                actions[agent_id],
                reward=0.0,
                next_observation=next_obs[agent_id],
            )
            agent.update_after_tick()

        if b_location is not None:
            action = observer.last_action
            if action is not None:
                pos = sim.world.agent_positions[left]
                dx, dy = ACTION_DELTA[action]
                nxt = pos.offset(dx, dy)
                goal = Position(*b_location)
                if nxt.manhattan(goal) < pos.manhattan(goal):
                    b_location_approaches += 1

    after_rel = observer.relationships.get(partner_id)
    return {
        "interactions": interactions,
        "cooperations": cooperations,
        "communicates": communicates,
        "interaction_probability": interactions / max(1, n_ticks),
        "cooperation_rate": cooperations / max(1, interactions) if interactions else 0.0,
        "relationship_before": before_snapshot,
        "relationship_after": after_rel.to_dict() if after_rel else None,
        "delta_strength": (after_rel.strength if after_rel else 0.0) - before_strength,
        "delta_trust": (after_rel.trust if after_rel else 0.5) - before_trust,
        "delta_predicted_utility": (
            (after_rel.predicted_utility if after_rel else 0.0) - before_utility
        ),
        "b_location_approaches": b_location_approaches,
        "indirect_sources_for_partner": list(
            observer.indirect_association_sources.get(partner_id, [])
        ),
        "secondhand_b": observer.secondhand_entities.get("B"),
        "propagation_events": len(sim.propagation_log.records),
        "memory_strength_b": observer.entity_strength("B"),
        "graph_A_to_C": sim.social_graph.weight("A", "C"),
        "graph_C_to_A": sim.social_graph.weight("C", "A"),
    }


def _relationship_snapshot(sim: Simulation) -> dict:
    out = {}
    for aid, agent in sim.agents.items():
        if isinstance(agent, SocialAgent):
            out[aid] = agent.relationships.summary()
    return out


def run_condition(
    *,
    seed: int,
    label: str,
    form_ab_ticks: int,
    form_bc_ticks: int,
    test_ticks: int,
    enable_indirect: bool,
) -> dict:
    """
    Stages:
      1. Formation AB / BC (controlled cooperate)
      2. Remove B
      3. Test A↔C free social policy
    """
    sim = _make_triangle_sim(seed=seed, enable_indirect=enable_indirect)

    formation = {"ab_interactions": 0, "bc_interactions": 0}

    if form_ab_ticks > 0:
        _place_pair(sim, "A", "B", Position(3, 3))
        formation["ab_interactions"] = _force_pair_cooperate(sim, "A", "B", form_ab_ticks)

    if form_bc_ticks > 0:
        _place_pair(sim, "B", "C", Position(8, 3))
        formation["bc_interactions"] = _force_pair_cooperate(sim, "B", "C", form_bc_ticks)

    # Ensure A and C did not form a direct relationship during formation.
    # (They were kept far apart; clear any incidental edge just in case.)
    for agent_id, other in (("A", "C"), ("C", "A")):
        agent = sim.agents[agent_id]
        assert isinstance(agent, SocialAgent)
        agent.relationships.relationships.pop(other, None)

    pre_removal = {
        "A_to_B": sim.agents["A"].relationships.get("B").to_dict()
        if sim.agents["A"].relationships.get("B")
        else None,
        "C_to_B": sim.agents["C"].relationships.get("B").to_dict()
        if sim.agents["C"].relationships.get("B")
        else None,
        "A_to_C": sim.agents["A"].relationships.get("C").to_dict()
        if sim.agents["A"].relationships.get("C")
        else None,
        "graph": sim.social_graph.summary(),
    }

    b_last_pos = sim.world.agent_positions["B"].as_tuple()
    sim.disappear("B")

    test = _free_pair_phase(sim, "A", "C", test_ticks)

    return {
        "label": label,
        "seed": seed,
        "form_ab_ticks": form_ab_ticks,
        "form_bc_ticks": form_bc_ticks,
        "test_ticks": test_ticks,
        "enable_indirect": enable_indirect,
        "formation": formation,
        "pre_removal": pre_removal,
        "b_last_position": list(b_last_pos),
        "test": test,
        "final_relationships": _relationship_snapshot(sim),
        "propagation_log_count": len(sim.propagation_log.records),
        "two_hop_from_A_pre_test_note": "B removed; A-C edge forms only in test phase",
    }


def run_social_memory_propagation(
    *,
    seed: int = 0,
    strong_ticks: int = 40,
    weak_ticks: int = 1,
    test_ticks: int = 50,
    output_dir: str | Path = "outputs/experiment_4",
) -> dict:
    """
    Control 1: weak A-B and B-C (no meaningful B relationships)
    Control 2: strong A↔B only (no B↔C bridge)
    Experiment: strong A↔B↔C, then A↔C after B disappears
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    control_weak = run_condition(
        seed=seed,
        label="control_weak_chain",
        form_ab_ticks=weak_ticks,
        form_bc_ticks=weak_ticks,
        test_ticks=test_ticks,
        enable_indirect=True,
    )
    control_ab_only = run_condition(
        seed=seed + 1,
        label="control_ab_only",
        form_ab_ticks=strong_ticks,
        form_bc_ticks=0,
        test_ticks=test_ticks,
        enable_indirect=True,
    )
    experiment = run_condition(
        seed=seed + 2,
        label="experiment_triangle",
        form_ab_ticks=strong_ticks,
        form_bc_ticks=strong_ticks,
        test_ticks=test_ticks,
        enable_indirect=True,
    )

    def _metric(result: dict, key: str) -> float:
        return float(result["test"].get(key, 0.0))

    comparison = {
        "seed": seed,
        "design": {
            "control_1": "weak A-B-C then B disappears then A↔C",
            "control_2": "strong A↔B only then B disappears then A↔C",
            "experiment": "strong A↔B↔C then B disappears then A↔C",
            "interpretation_note": (
                "Compare experiment against both controls before attributing "
                "effects to network-structured indirect information."
            ),
        },
        "control_weak": control_weak,
        "control_ab_only": control_ab_only,
        "experiment": experiment,
        "deltas_experiment_minus_weak": {
            "interaction_probability": _metric(experiment, "interaction_probability")
            - _metric(control_weak, "interaction_probability"),
            "cooperation_rate": _metric(experiment, "cooperation_rate")
            - _metric(control_weak, "cooperation_rate"),
            "delta_strength": _metric(experiment, "delta_strength")
            - _metric(control_weak, "delta_strength"),
            "delta_predicted_utility": _metric(experiment, "delta_predicted_utility")
            - _metric(control_weak, "delta_predicted_utility"),
            "propagation_events": _metric(experiment, "propagation_events")
            - _metric(control_weak, "propagation_events"),
            "b_location_approaches": _metric(experiment, "b_location_approaches")
            - _metric(control_weak, "b_location_approaches"),
        },
        "deltas_experiment_minus_ab_only": {
            "interaction_probability": _metric(experiment, "interaction_probability")
            - _metric(control_ab_only, "interaction_probability"),
            "cooperation_rate": _metric(experiment, "cooperation_rate")
            - _metric(control_ab_only, "cooperation_rate"),
            "delta_strength": _metric(experiment, "delta_strength")
            - _metric(control_ab_only, "delta_strength"),
            "delta_predicted_utility": _metric(experiment, "delta_predicted_utility")
            - _metric(control_ab_only, "delta_predicted_utility"),
            "propagation_events": _metric(experiment, "propagation_events")
            - _metric(control_ab_only, "propagation_events"),
            "b_location_approaches": _metric(experiment, "b_location_approaches")
            - _metric(control_ab_only, "b_location_approaches"),
        },
    }

    # Strip bulky nested dumps for the written comparison summary? Keep full for research.
    path = output_dir / f"comparison_seed{seed}.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "deltas_vs_weak": comparison["deltas_experiment_minus_weak"],
        "deltas_vs_ab_only": comparison["deltas_experiment_minus_ab_only"],
        "experiment_propagation_events": experiment["test"]["propagation_events"],
        "experiment_indirect_sources": experiment["test"]["indirect_sources_for_partner"],
        "experiment_A_to_C_after": experiment["test"]["relationship_after"],
    }
    summary_path = output_dir / f"summary_seed{seed}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return {
        "comparison": comparison,
        "summary": summary,
        "output_dir": str(output_dir),
        "comparison_path": str(path),
        "summary_path": str(summary_path),
    }


if __name__ == "__main__":
    result = run_social_memory_propagation()
    print(json.dumps(result["summary"], indent=2))
