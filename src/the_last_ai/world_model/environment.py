"""Environmental dynamics model (resources and local cell persistence)."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.types import CellType, JSONDict, Observation
from the_last_ai.world_model.types import Prediction, PredictionKind


@dataclass
class ResourceCellStats:
    seen_present: int = 0
    next_still_present: int = 0
    next_absent: int = 0

    @property
    def persistence_rate(self) -> float:
        total = self.next_still_present + self.next_absent
        if total == 0:
            return 0.5
        return self.next_still_present / total


@dataclass
class EnvironmentModel:
    """Tracks local resource persistence without neural nets."""

    cells: dict[tuple[int, int], ResourceCellStats] = field(default_factory=dict)
    _pending_resources: dict[tuple[int, int], bool] = field(default_factory=dict)
    updates: int = 0

    def observe(self, observation: Observation) -> None:
        """Register currently visible resources and resolve previous-tick pending checks."""
        radius = observation.perception_radius
        absolute_resources: dict[tuple[int, int], bool] = {}
        for dy, row in enumerate(observation.local_cells):
            for dx, cell in enumerate(row):
                abs_pos = (
                    observation.position.x + dx - radius,
                    observation.position.y + dy - radius,
                )
                absolute_resources[abs_pos] = cell == int(CellType.RESOURCE)

        for pos, was_present in list(self._pending_resources.items()):
            if pos not in absolute_resources:
                continue
            stats = self.cells.setdefault(pos, ResourceCellStats())
            if was_present:
                if absolute_resources[pos]:
                    stats.next_still_present += 1
                else:
                    stats.next_absent += 1
            self.updates += 1
        self._pending_resources = {
            pos: present for pos, present in absolute_resources.items() if present
        }
        for pos, present in absolute_resources.items():
            if present:
                self.cells.setdefault(pos, ResourceCellStats()).seen_present += 1

    def predict_resources(self, observation: Observation) -> list[Prediction]:
        predictions: list[Prediction] = []
        radius = observation.perception_radius
        for dy, row in enumerate(observation.local_cells):
            for dx, cell in enumerate(row):
                if cell != int(CellType.RESOURCE):
                    continue
                abs_pos = (
                    observation.position.x + dx - radius,
                    observation.position.y + dy - radius,
                )
                stats = self.cells.get(abs_pos)
                rate = stats.persistence_rate if stats else 0.5
                predict_present = rate >= 0.5
                predictions.append(
                    Prediction(
                        kind=PredictionKind.RESOURCE_PRESENCE,
                        subject_id=f"resource:{abs_pos[0]},{abs_pos[1]}",
                        tick_made=observation.tick,
                        tick_target=observation.tick + 1,
                        value=predict_present,
                        confidence=rate if predict_present else 1.0 - rate,
                        meta={"absolute_position": list(abs_pos), "persistence_rate": rate},
                    )
                )
        return predictions

    def score_resource_prediction(
        self,
        prediction: Prediction,
        observation: Observation,
    ) -> float | None:
        meta_pos = prediction.meta.get("absolute_position")
        if not meta_pos:
            return None
        abs_pos = (int(meta_pos[0]), int(meta_pos[1]))
        radius = observation.perception_radius
        rel_x = abs_pos[0] - observation.position.x
        rel_y = abs_pos[1] - observation.position.y
        if abs(rel_x) > radius or abs(rel_y) > radius:
            return None
        cell = observation.local_cells[rel_y + radius][rel_x + radius]
        observed = cell == int(CellType.RESOURCE)
        predicted = bool(prediction.value)
        return 0.0 if predicted == observed else 1.0

    def to_dict(self) -> JSONDict:
        return {
            "updates": self.updates,
            "cells": {
                f"{x},{y}": {
                    "seen_present": s.seen_present,
                    "next_still_present": s.next_still_present,
                    "next_absent": s.next_absent,
                    "persistence_rate": s.persistence_rate,
                }
                for (x, y), s in self.cells.items()
            },
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> EnvironmentModel:
        model = cls(updates=int(data.get("updates", 0)))
        for key, stats in data.get("cells", {}).items():
            x_str, y_str = key.split(",")
            model.cells[(int(x_str), int(y_str))] = ResourceCellStats(
                seen_present=int(stats.get("seen_present", 0)),
                next_still_present=int(stats.get("next_still_present", 0)),
                next_absent=int(stats.get("next_absent", 0)),
            )
        return model
