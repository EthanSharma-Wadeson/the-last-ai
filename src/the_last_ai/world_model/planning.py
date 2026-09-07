"""Multi-step internal simulation (Experiment 7 / Phase 4.4 deepened)."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.types import JSONDict, Position
from the_last_ai.world_model.counterfactual import CounterfactualResult
from the_last_ai.world_model.model import WorldModel


DELTA = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
    "stay": (0, 0),
}


@dataclass
class ImaginedState:
    my_position: Position
    entity_positions: dict[str, Position]
    step: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> JSONDict:
        return {
            "step": self.step,
            "my_position": list(self.my_position.as_tuple()),
            "entity_positions": {k: list(v.as_tuple()) for k, v in self.entity_positions.items()},
            "notes": list(self.notes),
        }


def _roll_entity(world_model: WorldModel, entity_id: str, current: Position) -> Position:
    model = world_model.agents.entities.get(entity_id)
    if model is None:
        return current
    direction, _ = model.likely_direction()
    dx, dy = DELTA.get(direction, (0, 0))
    return current.offset(dx, dy)


def simulate_action_sequence(
    world_model: WorldModel,
    *,
    start: Position,
    actions: list[str],
    entity_ids: list[str] | None = None,
) -> CounterfactualResult:
    """
    Evaluate: if I do X, then Y, then Z — what happens next?

    Rolls the agent's own position by chosen actions and advances other entities
    using their learned likely directions. This is an internal simulation, not
    ground truth.
    """
    entity_ids = entity_ids or list(world_model.agents.entities.keys())
    state = ImaginedState(
        my_position=start,
        entity_positions={},
        step=0,
        notes=["internal simulation; not environment ground truth"],
    )
    for eid in entity_ids:
        model = world_model.agents.entities.get(eid)
        if model and model.last_absolute_position:
            state.entity_positions[eid] = Position(*model.last_absolute_position)

    trajectory: list[JSONDict] = [state.to_dict()]

    for action in actions:
        dx, dy = DELTA.get(action, (0, 0))
        state.my_position = state.my_position.offset(dx, dy)
        state.step += 1
        state.notes = [f"I {action}"]
        for eid in list(state.entity_positions.keys()):
            state.entity_positions[eid] = _roll_entity(
                world_model, eid, state.entity_positions[eid]
            )
        # Distances after this imagined step.
        distances = {
            eid: state.my_position.manhattan(pos) for eid, pos in state.entity_positions.items()
        }
        frame = state.to_dict()
        frame["distances"] = distances
        frame["nearest_entity"] = min(distances, key=distances.get) if distances else None
        trajectory.append(frame)

    return CounterfactualResult(
        query=f"sequence:{'>'.join(actions)}",
        predicted_outcomes=trajectory,
        notes="Multi-step internal rollout using learned entity dynamics.",
    )


def evaluate_action_options(
    world_model: WorldModel,
    *,
    start: Position,
    horizon: int = 2,
    goal_entity: str | None = None,
) -> JSONDict:
    """
    Compare short action sequences by predicted distance to a goal entity
    (or mean distance to all known entities if goal is None).
    """
    options = ["stay", "north", "south", "east", "west"]
    # Depth-1 and depth-2 sequences.
    sequences: list[list[str]] = [[a] for a in options]
    if horizon >= 2:
        sequences.extend([[a, b] for a in options for b in options])

    scored: list[JSONDict] = []
    for seq in sequences:
        result = simulate_action_sequence(world_model, start=start, actions=seq)
        final = result.predicted_outcomes[-1]
        distances = final.get("distances", {})
        if goal_entity and goal_entity in distances:
            score = float(distances[goal_entity])
        elif distances:
            score = sum(distances.values()) / len(distances)
        else:
            score = 0.0
        scored.append({"actions": seq, "predicted_score": score, "final": final})

    scored.sort(key=lambda item: (item["predicted_score"], "".join(item["actions"])))
    return {
        "horizon": horizon,
        "goal_entity": goal_entity,
        "best": scored[0] if scored else None,
        "options_evaluated": len(scored),
        "top3": scored[:3],
    }


def simulate_disappearance_then_act(
    world_model: WorldModel,
    *,
    start: Position,
    missing_entity: str,
    actions: list[str],
) -> CounterfactualResult:
    """Counterfactual: if entity is gone, then I take actions — what remains?"""
    # Work on a shallow conceptual copy: ignore missing entity during rollout.
    filtered_ids = [eid for eid in world_model.agents.entities if eid != missing_entity]
    result = simulate_action_sequence(
        world_model, start=start, actions=actions, entity_ids=filtered_ids
    )
    result.query = f"if_absent:{missing_entity}|sequence:{'>'.join(actions)}"
    result.notes = (
        f"Imagines world without {missing_entity}, then rolls action sequence. "
        "Tests whether planning still uses residual models of others."
    )
    return result
