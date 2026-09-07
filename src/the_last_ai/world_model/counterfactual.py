"""Lightweight counterfactual / internal simulation scaffold (Phase 4.4)."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.types import JSONDict, Position
from the_last_ai.world_model.model import WorldModel


@dataclass
class CounterfactualResult:
    query: str
    predicted_outcomes: list[JSONDict]
    notes: str

    def to_dict(self) -> JSONDict:
        return {
            "query": self.query,
            "predicted_outcomes": self.predicted_outcomes,
            "notes": self.notes,
        }


def simulate_if_i_move(world_model: WorldModel, direction: str, current: Position) -> CounterfactualResult:
    """Ask: if I move in `direction`, where do I expect nearby agents to be?"""
    delta = {
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
        "stay": (0, 0),
    }.get(direction, (0, 0))
    my_next = current.offset(*delta)
    outcomes: list[JSONDict] = []
    for entity_id, model in world_model.agents.entities.items():
        pred = model.predict_next_position(tick=model.last_seen_tick)
        if pred is None:
            continue
        outcomes.append(
            {
                "entity_id": entity_id,
                "predicted_position": pred.value,
                "confidence": pred.confidence,
                "my_next_position": list(my_next.as_tuple()),
                "predicted_distance": Position(int(pred.value[0]), int(pred.value[1])).manhattan(
                    my_next
                )
                if isinstance(pred.value, list)
                else None,
            }
        )
    return CounterfactualResult(
        query=f"if_i_move:{direction}",
        predicted_outcomes=outcomes,
        notes="Scaffold only — does not roll out multi-step futures yet.",
    )


def simulate_if_entity_absent(world_model: WorldModel, entity_id: str) -> CounterfactualResult:
    """Ask: if entity disappears, what predictions remain about them?"""
    model = world_model.agents.entities.get(entity_id)
    if model is None:
        return CounterfactualResult(
            query=f"if_absent:{entity_id}",
            predicted_outcomes=[],
            notes="No dynamics model for entity.",
        )
    presence = model.predict_presence(model.last_seen_tick, model.last_absolute_position)
    return CounterfactualResult(
        query=f"if_absent:{entity_id}",
        predicted_outcomes=[
            {
                "entity_id": entity_id,
                "presence_prediction": presence.value,
                "confidence": presence.confidence,
                "likely_direction": model.likely_direction()[0],
                "last_position": list(model.last_absolute_position)
                if model.last_absolute_position
                else None,
            }
        ],
        notes="Uses existing behavioural model; absence does not erase predictions.",
    )
