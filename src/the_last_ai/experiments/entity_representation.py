"""Experiment 2 — entity representation persistence after disappearance."""

from __future__ import annotations

import json
from pathlib import Path

from the_last_ai.agents.base import RandomAgent
from the_last_ai.agents.memory_agent import MemoryAgent
from the_last_ai.memory.long_term import MemoryConfig
from the_last_ai.metrics.recorder import MetricsRecorder
from the_last_ai.rng import ExperimentRNG
from the_last_ai.simulation.engine import Simulation, SimulationConfig
from the_last_ai.types import Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def _make_pair_simulation(
    *,
    seed: int,
    memory_config: MemoryConfig,
    width: int = 12,
    height: int = 10,
    perception_radius: int = 4,
) -> tuple[Simulation, str, str]:
    """Create observer (MemoryAgent) + partner (RandomAgent) placed adjacently."""
    rng = ExperimentRNG.from_seed(seed)
    grid = Grid.empty(width, height, bordered=True)
    world = World(grid=grid)
    observer_id = "observer"
    partner_id = "partner"

    observer = MemoryAgent(
        observer_id,
        perception_radius=perception_radius,
        memory_config=memory_config,
    )
    partner = RandomAgent(partner_id, perception_radius=perception_radius)

    world.place_agent(observer_id, Position(4, 4))
    world.place_agent(partner_id, Position(5, 4))

    config = SimulationConfig(
        seed=seed,
        width=width,
        height=height,
        n_agents=2,
        perception_radius=perception_radius,
        agent_type="MemoryAgent",
    )
    sim = Simulation(
        config=config,
        world=world,
        agents={observer_id: observer, partner_id: partner},
        rng=rng,
        metrics=MetricsRecorder(),
    )
    return sim, observer_id, partner_id


def _approach_toward_expected(sim: Simulation, observer_id: str, entity_id: str) -> bool:
    observer = sim.agents[observer_id]
    assert isinstance(observer, MemoryAgent)
    memory = observer.ltm.get(entity_id)
    if memory is None or memory.expected_location is None:
        return False
    if observer_id not in sim.world.agent_positions:
        return False
    pos = sim.world.agent_positions[observer_id]
    expected = Position(*memory.expected_location)
    action = observer.last_action
    if action is None:
        return False
    from the_last_ai.types import ACTION_DELTA

    dx, dy = ACTION_DELTA[action]
    next_pos = pos.offset(dx, dy)
    return next_pos.manhattan(expected) < pos.manhattan(expected)


def _run_condition(
    *,
    seed: int,
    label: str,
    interaction_ticks: int,
    post_ticks: int,
    memory_config: MemoryConfig,
) -> dict:
    sim, observer_id, partner_id = _make_pair_simulation(seed=seed, memory_config=memory_config)
    observer = sim.agents[observer_id]
    assert isinstance(observer, MemoryAgent)

    # Interaction phase
    sim.run(interaction_ticks)
    strength_at_removal = observer.entity_strength(partner_id)
    familiarity_at_removal = (
        observer.ltm.get(partner_id).familiarity if observer.ltm.get(partner_id) else 0.0
    )
    expected_at_removal = None
    mem = observer.ltm.get(partner_id)
    if mem and mem.expected_location:
        expected_at_removal = list(mem.expected_location)

    # Disappearance
    last_partner_pos = sim.world.agent_positions[partner_id].as_tuple()
    sim.disappear(partner_id)

    # Post-disappearance tracking
    strength_series: list[tuple[int, float]] = []
    approach_count = 0
    location_revisits = 0
    for _ in range(post_ticks):
        sim.step()
        tick = sim.world.tick
        strength_series.append((tick, observer.entity_strength(partner_id, tick)))
        if _approach_toward_expected(sim, observer_id, partner_id):
            approach_count += 1
        obs_pos = sim.world.agent_positions[observer_id].as_tuple()
        if expected_at_removal and obs_pos == tuple(expected_at_removal):
            location_revisits += 1
        elif obs_pos == last_partner_pos:
            location_revisits += 1

    final_strength = observer.entity_strength(partner_id)
    memory_persisted = partner_id in observer.ltm.entities

    return {
        "label": label,
        "interaction_ticks": interaction_ticks,
        "post_ticks": post_ticks,
        "strength_at_removal": strength_at_removal,
        "familiarity_at_removal": familiarity_at_removal,
        "final_strength": final_strength,
        "memory_persisted": memory_persisted,
        "approach_count": approach_count,
        "location_revisits": location_revisits,
        "seek_attempts": observer.seek_attempts,
        "memory_guided_actions": observer.memory_guided_actions,
        "retrieval_events": observer.ltm.retrieval_events,
        "replacement_count": observer.ltm.replacement_count,
        "strength_series": strength_series,
        "memory_summary": observer.memory_summary(),
        "expected_location_at_removal": expected_at_removal,
        "last_partner_position": list(last_partner_pos),
    }


def run_entity_representation(
    *,
    seed: int = 0,
    familiar_ticks: int = 120,
    unfamiliar_ticks: int = 8,
    post_ticks: int = 80,
    output_dir: str | Path = "outputs/experiment_2",
) -> dict:
    """
    Condition A: disappear after minimal interaction.
    Condition B: disappear after extensive interaction.
    Also compare decaying fixed-capacity memory vs unlimited slow-decay memory.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decaying = MemoryConfig(capacity=8, decay_lambda=0.01, forget_threshold=0.02)
    unlimited = MemoryConfig(capacity=0, decay_lambda=0.001, forget_threshold=0.0)

    unfamiliar = _run_condition(
        seed=seed,
        label="unfamiliar_minimal_interaction",
        interaction_ticks=unfamiliar_ticks,
        post_ticks=post_ticks,
        memory_config=decaying,
    )
    familiar = _run_condition(
        seed=seed + 1,
        label="familiar_extensive_interaction",
        interaction_ticks=familiar_ticks,
        post_ticks=post_ticks,
        memory_config=decaying,
    )
    familiar_unlimited = _run_condition(
        seed=seed + 1,
        label="familiar_unlimited_slow_decay",
        interaction_ticks=familiar_ticks,
        post_ticks=post_ticks,
        memory_config=unlimited,
    )

    comparison = {
        "seed": seed,
        "unfamiliar": {
            k: v for k, v in unfamiliar.items() if k != "strength_series"
        },
        "familiar": {
            k: v for k, v in familiar.items() if k != "strength_series"
        },
        "familiar_unlimited": {
            k: v for k, v in familiar_unlimited.items() if k != "strength_series"
        },
        "deltas": {
            "strength_at_removal": familiar["strength_at_removal"] - unfamiliar["strength_at_removal"],
            "final_strength": familiar["final_strength"] - unfamiliar["final_strength"],
            "approach_count": familiar["approach_count"] - unfamiliar["approach_count"],
            "familiarity_at_removal": (
                familiar["familiarity_at_removal"] - unfamiliar["familiarity_at_removal"]
            ),
        },
        "architecture_deltas": {
            "final_strength_unlimited_minus_decaying": (
                familiar_unlimited["final_strength"] - familiar["final_strength"]
            ),
            "memory_persisted_unlimited": familiar_unlimited["memory_persisted"],
            "memory_persisted_decaying": familiar["memory_persisted"],
        },
    }

    path = output_dir / f"comparison_seed{seed}.json"
    path.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding="utf-8")

    series_path = output_dir / f"strength_series_seed{seed}.json"
    series_path.write_text(
        json.dumps(
            {
                "unfamiliar": unfamiliar["strength_series"],
                "familiar": familiar["strength_series"],
                "familiar_unlimited": familiar_unlimited["strength_series"],
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
    result = run_entity_representation()
    print(json.dumps(result["comparison"]["deltas"], indent=2))
