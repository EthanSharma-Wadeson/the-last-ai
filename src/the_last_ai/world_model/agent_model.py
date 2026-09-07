"""Per-entity behavioural dynamics models."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from the_last_ai.types import JSONDict, Observation, Position, VisibleAgent
from the_last_ai.world_model.types import Prediction, PredictionKind


DIRECTION_LABELS = {
    (0, 0): "stay",
    (0, -1): "north",
    (0, 1): "south",
    (1, 0): "east",
    (-1, 0): "west",
}


def _delta_to_label(dx: int, dy: int) -> str:
    # Compress to cardinal / stay.
    if dx == 0 and dy == 0:
        return "stay"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def _label_to_delta(label: str) -> tuple[int, int]:
    mapping = {
        "stay": (0, 0),
        "north": (0, -1),
        "south": (0, 1),
        "east": (1, 0),
        "west": (-1, 0),
    }
    return mapping.get(label, (0, 0))


@dataclass
class EntityDynamicsModel:
    """Learned behavioural pattern for one observed entity."""

    entity_id: str
    direction_counts: Counter = field(default_factory=Counter)
    resource_approach_count: int = 0
    observations: int = 0
    last_absolute_position: tuple[int, int] | None = None
    last_seen_tick: int = 0
    presence_when_expected: int = 0
    absence_when_expected: int = 0
    social_cooperate_toward_self: int = 0
    social_compete_toward_self: int = 0

    def confidence(self) -> float:
        # More observations → higher confidence, asymptotic.
        return min(0.95, self.observations / (self.observations + 5))

    def likely_direction(self) -> tuple[str, float]:
        if not self.direction_counts:
            return "stay", 0.2
        total = sum(self.direction_counts.values())
        best_label, best_count = max(
            self.direction_counts.items(), key=lambda item: (item[1], item[0])
        )
        return best_label, best_count / total

    def update_sighting(
        self,
        absolute_position: Position,
        *,
        tick: int,
        moved_toward_resource: bool = False,
    ) -> None:
        if self.last_absolute_position is not None:
            dx = absolute_position.x - self.last_absolute_position[0]
            dy = absolute_position.y - self.last_absolute_position[1]
            # Only count unit steps / stays as directional evidence.
            if max(abs(dx), abs(dy)) <= 1:
                self.direction_counts[_delta_to_label(dx, dy)] += 1
        self.last_absolute_position = absolute_position.as_tuple()
        self.last_seen_tick = tick
        self.observations += 1
        if moved_toward_resource:
            self.resource_approach_count += 1

    def predict_next_position(self, tick: int) -> Prediction | None:
        if self.last_absolute_position is None:
            return None
        direction, prob = self.likely_direction()
        dx, dy = _label_to_delta(direction)
        x, y = self.last_absolute_position
        predicted = (x + dx, y + dy)
        return Prediction(
            kind=PredictionKind.AGENT_POSITION,
            subject_id=self.entity_id,
            tick_made=tick,
            tick_target=tick + 1,
            value=list(predicted),
            confidence=self.confidence() * prob,
            meta={
                "likely_direction": direction,
                "direction_probability": prob,
                "last_position": list(self.last_absolute_position),
            },
        )

    def predict_presence(self, tick: int, expected_location: tuple[int, int] | None) -> Prediction:
        """Predict whether this entity will be observed next tick (used after disappearance too)."""
        total = self.presence_when_expected + self.absence_when_expected
        if total == 0:
            rate = 0.7 if self.observations > 0 else 0.3
        else:
            rate = self.presence_when_expected / total
        return Prediction(
            kind=PredictionKind.AGENT_PRESENCE,
            subject_id=self.entity_id,
            tick_made=tick,
            tick_target=tick + 1,
            value=rate >= 0.5,
            confidence=abs(rate - 0.5) * 2 * self.confidence(),
            meta={
                "presence_rate": rate,
                "expected_location": list(expected_location) if expected_location else None,
            },
        )

    def register_presence_outcome(self, present: bool) -> None:
        if present:
            self.presence_when_expected += 1
        else:
            self.absence_when_expected += 1

    def to_dict(self) -> JSONDict:
        return {
            "entity_id": self.entity_id,
            "direction_counts": dict(self.direction_counts),
            "resource_approach_count": self.resource_approach_count,
            "observations": self.observations,
            "last_absolute_position": list(self.last_absolute_position)
            if self.last_absolute_position
            else None,
            "last_seen_tick": self.last_seen_tick,
            "presence_when_expected": self.presence_when_expected,
            "absence_when_expected": self.absence_when_expected,
            "social_cooperate_toward_self": self.social_cooperate_toward_self,
            "social_compete_toward_self": self.social_compete_toward_self,
            "confidence": self.confidence(),
            "likely_direction": self.likely_direction()[0],
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> EntityDynamicsModel:
        loc = data.get("last_absolute_position")
        return cls(
            entity_id=str(data["entity_id"]),
            direction_counts=Counter(data.get("direction_counts", {})),
            resource_approach_count=int(data.get("resource_approach_count", 0)),
            observations=int(data.get("observations", 0)),
            last_absolute_position=(int(loc[0]), int(loc[1])) if loc else None,
            last_seen_tick=int(data.get("last_seen_tick", 0)),
            presence_when_expected=int(data.get("presence_when_expected", 0)),
            absence_when_expected=int(data.get("absence_when_expected", 0)),
            social_cooperate_toward_self=int(data.get("social_cooperate_toward_self", 0)),
            social_compete_toward_self=int(data.get("social_compete_toward_self", 0)),
        )


@dataclass
class AgentBehaviourModels:
    entities: dict[str, EntityDynamicsModel] = field(default_factory=dict)

    def get_or_create(self, entity_id: str) -> EntityDynamicsModel:
        if entity_id not in self.entities:
            self.entities[entity_id] = EntityDynamicsModel(entity_id=entity_id)
        return self.entities[entity_id]

    def observe_visible(
        self,
        observation: Observation,
        *,
        resource_absolute_positions: set[tuple[int, int]],
    ) -> None:
        visible_ids = set()
        for visible in observation.visible_agents:
            visible_ids.add(visible.agent_id)
            absolute = observation.position.offset(*visible.relative_position)
            toward_resource = False
            model = self.get_or_create(visible.agent_id)
            if model.last_absolute_position is not None and resource_absolute_positions:
                prev = Position(*model.last_absolute_position)
                # Approaching if distance to nearest resource decreased.
                prev_dist = min(
                    prev.manhattan(Position(x, y)) for x, y in resource_absolute_positions
                )
                new_dist = min(
                    absolute.manhattan(Position(x, y)) for x, y in resource_absolute_positions
                )
                toward_resource = new_dist < prev_dist
            model.update_sighting(absolute, tick=observation.tick, moved_toward_resource=toward_resource)

        # Presence outcomes for entities we expected to maybe see.
        for entity_id, model in self.entities.items():
            if model.last_absolute_position is None:
                continue
            # Only score presence when we are near the last known area.
            last = Position(*model.last_absolute_position)
            if observation.position.manhattan(last) <= observation.perception_radius + 1:
                model.register_presence_outcome(entity_id in visible_ids)

    def predict_all(self, observation: Observation) -> list[Prediction]:
        predictions: list[Prediction] = []
        for entity_id, model in self.entities.items():
            pos_pred = model.predict_next_position(observation.tick)
            if pos_pred is not None:
                predictions.append(pos_pred)
            expected = model.last_absolute_position
            predictions.append(model.predict_presence(observation.tick, expected))
        return predictions

    def score_position_prediction(
        self,
        prediction: Prediction,
        observation: Observation,
    ) -> float | None:
        subject = prediction.subject_id
        visible = {v.agent_id: v for v in observation.visible_agents}
        if subject not in visible:
            # Cannot score position if not visible — leave to presence prediction.
            return None
        v = visible[subject]
        absolute = observation.position.offset(*v.relative_position)
        predicted = prediction.value
        if not isinstance(predicted, (list, tuple)) or len(predicted) != 2:
            return None
        dist = absolute.manhattan(Position(int(predicted[0]), int(predicted[1])))
        # 0 if exact, approaches 1 as distance grows.
        return min(1.0, dist / 3.0)

    def score_presence_prediction(
        self,
        prediction: Prediction,
        observation: Observation,
    ) -> float:
        present = any(v.agent_id == prediction.subject_id for v in observation.visible_agents)
        predicted = bool(prediction.value)
        return 0.0 if predicted == present else 1.0

    def to_dict(self) -> JSONDict:
        return {eid: model.to_dict() for eid, model in self.entities.items()}

    @classmethod
    def from_dict(cls, data: JSONDict) -> AgentBehaviourModels:
        models = cls()
        for eid, payload in data.items():
            models.entities[eid] = EntityDynamicsModel.from_dict(payload)
        return models
