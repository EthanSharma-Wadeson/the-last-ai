"""Value-based agent with long-term entity memory."""

from __future__ import annotations

from the_last_ai.agents.features import action_toward, encode_state
from the_last_ai.agents.motivation import MotivationConfig, evaluate_drives
from the_last_ai.agents.value_learner import LearningConfig, ValueBasedAgent
from the_last_ai.memory.long_term import LongTermMemory, MemoryConfig
from the_last_ai.rng import ExperimentRNG
from the_last_ai.types import MOVEMENT_ACTIONS, Action, JSONDict, Observation, Position


def _direction_to_target(current: Position, target: tuple[int, int]) -> str:
    dx = target[0] - current.x
    dy = target[1] - current.y
    if dx == 0 and dy == 0:
        return "here"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


class MemoryAgent(ValueBasedAgent):
    """Learner that maintains decaying entity representations and seeks remembered agents."""

    def __init__(
        self,
        agent_id: str,
        *,
        energy: float = 100.0,
        perception_radius: int = 3,
        stm_capacity: int = 32,
        motivation: MotivationConfig | None = None,
        learning: LearningConfig | None = None,
        memory_config: MemoryConfig | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            energy=energy,
            perception_radius=perception_radius,
            stm_capacity=stm_capacity,
            motivation=motivation,
            learning=learning,
        )
        self.ltm = LongTermMemory(config=memory_config or MemoryConfig())
        self.last_retrieved_ids: list[str] = []
        self.seek_attempts: int = 0
        self.memory_guided_actions: int = 0

    def _update_entity_memories(self, observation: Observation) -> None:
        for visible in observation.visible_agents:
            absolute = observation.position.offset(*visible.relative_position)
            self.ltm.observe_entity(
                visible.agent_id,
                absolute,
                tick=observation.tick,
                distance=visible.distance,
            )

    def _maybe_seek_remembered(self, observation: Observation, rng: ExperimentRNG) -> Action | None:
        visible_ids = {v.agent_id for v in observation.visible_agents}
        retrieved = self.ltm.retrieve(
            observation.position,
            tick=observation.tick,
            exclude_visible=visible_ids,
        )
        self.last_retrieved_ids = [m.entity_id for m in retrieved]
        if not retrieved:
            return None

        best = retrieved[0]
        strength = best.strength_at(observation.tick, self.ltm.config.decay_lambda)
        if strength < self.ltm.config.seek_strength_threshold:
            return None
        if best.expected_location is None:
            return None
        if rng.random() > self.ltm.config.seek_probability:
            return None

        direction = _direction_to_target(observation.position, best.expected_location)
        action = action_toward(direction)
        if action is None or action not in MOVEMENT_ACTIONS:
            return None
        self.seek_attempts += 1
        self.memory_guided_actions += 1
        return action

    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        self._update_entity_memories(observation)
        pos = observation.position.as_tuple()
        self.visit_counts[pos] += 1
        self.last_motivation = evaluate_drives(
            observation,
            visit_count=self.visit_counts[pos],
            config=self.motivation_config,
        )

        memory_action = self._maybe_seek_remembered(observation, rng)
        if memory_action is not None:
            return memory_action

        state = encode_state(observation)
        if rng.random() < self.epsilon:
            return Action(rng.choice(MOVEMENT_ACTIONS))
        return self._best_action(state, rng)

    def update_after_tick(self) -> None:
        super().update_after_tick()
        tick = self.last_observation.tick if self.last_observation else self.state.age
        self.ltm.tick_decay(tick)

    def entity_strength(self, entity_id: str, tick: int | None = None) -> float:
        memory = self.ltm.get(entity_id)
        if memory is None:
            return 0.0
        if tick is None:
            tick = self.last_observation.tick if self.last_observation else self.state.age
        return memory.strength_at(tick, self.ltm.config.decay_lambda)

    def memory_summary(self, tick: int | None = None) -> JSONDict:
        if tick is None:
            tick = self.last_observation.tick if self.last_observation else self.state.age
        summary = self.ltm.summary(tick)
        summary.update(
            {
                "seek_attempts": self.seek_attempts,
                "memory_guided_actions": self.memory_guided_actions,
                "last_retrieved_ids": list(self.last_retrieved_ids),
            }
        )
        return summary

    def to_dict(self) -> JSONDict:
        data = super().to_dict()
        data.update(
            {
                "ltm": self.ltm.to_dict(),
                "seek_attempts": self.seek_attempts,
                "memory_guided_actions": self.memory_guided_actions,
                "last_retrieved_ids": list(self.last_retrieved_ids),
            }
        )
        return data

    def load_state(self, data: JSONDict) -> None:
        super().load_state(data)
        self.ltm = LongTermMemory.from_dict(data.get("ltm", {}))
        self.seek_attempts = int(data.get("seek_attempts", 0))
        self.memory_guided_actions = int(data.get("memory_guided_actions", 0))
        self.last_retrieved_ids = list(data.get("last_retrieved_ids", []))
