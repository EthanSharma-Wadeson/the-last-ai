"""Integrated internal world model: predict → compare → update."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.types import CellType, JSONDict, Observation
from the_last_ai.world_model.agent_model import AgentBehaviourModels
from the_last_ai.world_model.environment import EnvironmentModel
from the_last_ai.world_model.types import Prediction, PredictionError, PredictionKind


@dataclass
class WorldModel:
    """
    Non-neural predictive model of environment + other agents.

    Core loop: form predictions → observe outcome → compute error → update parameters.
    """

    environment: EnvironmentModel = field(default_factory=EnvironmentModel)
    agents: AgentBehaviourModels = field(default_factory=AgentBehaviourModels)
    pending: list[Prediction] = field(default_factory=list)
    errors: list[PredictionError] = field(default_factory=list)
    mean_error_ema: float = 0.5
    total_scored: int = 0

    def _resource_positions(self, observation: Observation) -> set[tuple[int, int]]:
        radius = observation.perception_radius
        positions: set[tuple[int, int]] = set()
        for dy, row in enumerate(observation.local_cells):
            for dx, cell in enumerate(row):
                if cell == int(CellType.RESOURCE):
                    positions.add(
                        (
                            observation.position.x + dx - radius,
                            observation.position.y + dy - radius,
                        )
                    )
        return positions

    def observe_and_update(self, observation: Observation) -> list[PredictionError]:
        """Verify pending predictions against the new observation, then ingest it."""
        scored = self.verify(observation)
        self.environment.observe(observation)
        self.agents.observe_visible(
            observation,
            resource_absolute_positions=self._resource_positions(observation),
        )
        return scored

    def form_predictions(self, observation: Observation) -> list[Prediction]:
        predictions = []
        predictions.extend(self.environment.predict_resources(observation))
        predictions.extend(self.agents.predict_all(observation))
        self.pending = predictions
        return predictions

    def verify(self, observation: Observation) -> list[PredictionError]:
        scored: list[PredictionError] = []
        remaining: list[Prediction] = []
        for prediction in self.pending:
            if prediction.tick_target > observation.tick:
                remaining.append(prediction)
                continue
            error_value: float | None
            observed: object
            if prediction.kind == PredictionKind.RESOURCE_PRESENCE:
                error_value = self.environment.score_resource_prediction(prediction, observation)
                observed = prediction.meta.get("absolute_position")
            elif prediction.kind == PredictionKind.AGENT_POSITION:
                error_value = self.agents.score_position_prediction(prediction, observation)
                visible = {v.agent_id: v for v in observation.visible_agents}
                if prediction.subject_id in visible:
                    abs_pos = observation.position.offset(
                        *visible[prediction.subject_id].relative_position
                    )
                    observed = list(abs_pos.as_tuple())
                else:
                    observed = None
            elif prediction.kind == PredictionKind.AGENT_PRESENCE:
                error_value = self.agents.score_presence_prediction(prediction, observation)
                observed = any(
                    v.agent_id == prediction.subject_id for v in observation.visible_agents
                )
            else:
                error_value = None
                observed = None

            if error_value is None:
                continue
            record = PredictionError(
                prediction=prediction,
                observed_value=observed,
                error=error_value,
                tick=observation.tick,
            )
            scored.append(record)
            self.errors.append(record)
            self.total_scored += 1
            self.mean_error_ema = 0.9 * self.mean_error_ema + 0.1 * error_value

        self.pending = remaining
        # Keep error history bounded.
        if len(self.errors) > 2000:
            self.errors = self.errors[-1000:]
        return scored

    def summary(self) -> JSONDict:
        by_kind: dict[str, list[float]] = {}
        for err in self.errors[-200:]:
            by_kind.setdefault(err.prediction.kind.value, []).append(err.error)
        kind_means = {
            kind: (sum(vals) / len(vals) if vals else 0.0) for kind, vals in by_kind.items()
        }
        return {
            "pending_predictions": len(self.pending),
            "total_scored": self.total_scored,
            "mean_error_ema": self.mean_error_ema,
            "recent_error_by_kind": kind_means,
            "tracked_entities": len(self.agents.entities),
            "tracked_resource_cells": len(self.environment.cells),
            "entity_models": {
                eid: {
                    "confidence": model.confidence(),
                    "likely_direction": model.likely_direction()[0],
                    "observations": model.observations,
                    "last_position": list(model.last_absolute_position)
                    if model.last_absolute_position
                    else None,
                }
                for eid, model in self.agents.entities.items()
            },
        }

    def to_dict(self) -> JSONDict:
        return {
            "environment": self.environment.to_dict(),
            "agents": self.agents.to_dict(),
            "pending": [p.to_dict() for p in self.pending],
            "mean_error_ema": self.mean_error_ema,
            "total_scored": self.total_scored,
            "recent_errors": [e.to_dict() for e in self.errors[-100:]],
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> WorldModel:
        model = cls(
            environment=EnvironmentModel.from_dict(data.get("environment", {})),
            agents=AgentBehaviourModels.from_dict(data.get("agents", {})),
            mean_error_ema=float(data.get("mean_error_ema", 0.5)),
            total_scored=int(data.get("total_scored", 0)),
        )
        for pred in data.get("pending", []):
            model.pending.append(
                Prediction(
                    kind=PredictionKind(pred["kind"]),
                    subject_id=str(pred["subject_id"]),
                    tick_made=int(pred["tick_made"]),
                    tick_target=int(pred["tick_target"]),
                    value=pred["value"],
                    confidence=float(pred["confidence"]),
                    meta=dict(pred.get("meta", {})),
                )
            )
        return model
