"""Value-based (tabular Q-learning) agent with motivation-shaped rewards."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from the_last_ai.agents.base import Agent
from the_last_ai.agents.features import encode_state
from the_last_ai.agents.motivation import (
    MotivationConfig,
    MotivationState,
    evaluate_drives,
)
from the_last_ai.rng import ExperimentRNG
from the_last_ai.types import MOVEMENT_ACTIONS, Action, JSONDict, Observation


@dataclass
class LearningConfig:
    learning_rate: float = 0.15
    discount: float = 0.9
    epsilon: float = 0.25
    epsilon_decay: float = 0.997
    min_epsilon: float = 0.05

    def to_dict(self) -> JSONDict:
        return {
            "learning_rate": self.learning_rate,
            "discount": self.discount,
            "epsilon": self.epsilon,
            "epsilon_decay": self.epsilon_decay,
            "min_epsilon": self.min_epsilon,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> LearningConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


class ValueBasedAgent(Agent):
    """Epsilon-greedy tabular Q-learner driven by computational motivation."""

    def __init__(
        self,
        agent_id: str,
        *,
        energy: float = 100.0,
        perception_radius: int = 3,
        stm_capacity: int = 32,
        motivation: MotivationConfig | None = None,
        learning: LearningConfig | None = None,
    ) -> None:
        super().__init__(
            agent_id,
            energy=energy,
            perception_radius=perception_radius,
            stm_capacity=stm_capacity,
        )
        self.motivation_config = motivation or MotivationConfig()
        self.learning_config = learning or LearningConfig()
        self.epsilon = self.learning_config.epsilon
        self.q_values: dict[tuple, dict[str, float]] = {}
        self.visit_counts: Counter[tuple[int, int]] = Counter()
        self.last_motivation: MotivationState | None = None
        self.total_reward: float = 0.0
        self.resources_collected: int = 0
        self.update_count: int = 0

    def _q(self, state: tuple, action: Action) -> float:
        return self.q_values.setdefault(state, {}).get(action.value, 0.0)

    def _set_q(self, state: tuple, action: Action, value: float) -> None:
        self.q_values.setdefault(state, {})[action.value] = value

    def _best_action(self, state: tuple, rng: ExperimentRNG) -> Action:
        scored = [(self._q(state, action), action.value, action) for action in MOVEMENT_ACTIONS]
        # Tie-break deterministically via RNG among equal values.
        max_value = max(item[0] for item in scored)
        tied = [action for value, _, action in scored if value == max_value]
        if len(tied) == 1:
            return tied[0]
        return Action(rng.choice(tied))

    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        pos = observation.position.as_tuple()
        self.visit_counts[pos] += 1
        self.last_motivation = evaluate_drives(
            observation,
            visit_count=self.visit_counts[pos],
            config=self.motivation_config,
        )
        state = encode_state(observation)
        if rng.random() < self.epsilon:
            return Action(rng.choice(MOVEMENT_ACTIONS))
        return self._best_action(state, rng)

    def learn(
        self,
        observation: Observation,
        action: Action,
        reward: float | None = None,
        next_observation: Observation | None = None,
        *,
        energy_gained: float = 0.0,
        moved: bool = False,
        blocked: bool = False,
    ) -> None:
        if next_observation is None:
            return

        before_pos = observation.position.as_tuple()
        after_pos = next_observation.position.as_tuple()
        new_cell = self.visit_counts[after_pos] == 0 and after_pos != before_pos

        if reward is None:
            from the_last_ai.agents.motivation import compute_reward_with_blocked

            reward = compute_reward_with_blocked(
                observation,
                next_observation,
                energy_gained=energy_gained,
                moved=moved,
                new_cell=new_cell,
                blocked=blocked,
                config=self.motivation_config,
            )

        state = encode_state(observation)
        next_state = encode_state(next_observation)
        old = self._q(state, action)
        next_max = max(self._q(next_state, a) for a in MOVEMENT_ACTIONS)
        td_target = reward + self.learning_config.discount * next_max
        updated = old + self.learning_config.learning_rate * (td_target - old)
        self._set_q(state, action, updated)

        self.total_reward += reward
        self.update_count += 1
        if energy_gained > 0:
            self.resources_collected += 1

        self.epsilon = max(
            self.learning_config.min_epsilon,
            self.epsilon * self.learning_config.epsilon_decay,
        )

    def apply_energy_gain(self, amount: float) -> None:
        self.state.energy = min(
            self.motivation_config.max_energy,
            max(0.0, self.state.energy + amount),
        )

    def to_dict(self) -> JSONDict:
        data = super().to_dict()
        data.update(
            {
                "motivation": self.motivation_config.to_dict(),
                "learning": self.learning_config.to_dict(),
                "epsilon": self.epsilon,
                "q_values": {
                    "|".join(str(part) for part in state): values
                    for state, values in self.q_values.items()
                },
                "visit_counts": {
                    f"{x},{y}": count for (x, y), count in self.visit_counts.items()
                },
                "total_reward": self.total_reward,
                "resources_collected": self.resources_collected,
                "update_count": self.update_count,
                "last_motivation": self.last_motivation.to_dict() if self.last_motivation else None,
            }
        )
        return data

    def load_state(self, data: JSONDict) -> None:
        super().load_state(data)
        self.motivation_config = MotivationConfig.from_dict(data.get("motivation", {}))
        self.learning_config = LearningConfig.from_dict(data.get("learning", {}))
        self.epsilon = float(data.get("epsilon", self.learning_config.epsilon))
        self.q_values = {}
        for key, values in data.get("q_values", {}).items():
            parts = key.split("|")
            state = (int(parts[0]), parts[1], int(parts[2]), parts[3], int(parts[4]))
            self.q_values[state] = {k: float(v) for k, v in values.items()}
        self.visit_counts = Counter()
        for key, count in data.get("visit_counts", {}).items():
            x_str, y_str = key.split(",")
            self.visit_counts[(int(x_str), int(y_str))] = int(count)
        self.total_reward = float(data.get("total_reward", 0.0))
        self.resources_collected = int(data.get("resources_collected", 0))
        self.update_count = int(data.get("update_count", 0))
