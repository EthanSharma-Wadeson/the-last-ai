"""Computational motivation drives and reward signals."""

from __future__ import annotations

from dataclasses import dataclass

from the_last_ai.types import CellType, JSONDict, Observation


@dataclass
class MotivationConfig:
    """Weights for energy maintenance, social proximity, and exploration."""

    energy_weight: float = 1.0
    social_weight: float = 0.25
    exploration_weight: float = 0.15
    energy_threshold: float = 50.0
    max_energy: float = 100.0
    social_radius: int = 2
    step_cost: float = 0.01
    blocked_penalty: float = 0.05

    def to_dict(self) -> JSONDict:
        return {
            "energy_weight": self.energy_weight,
            "social_weight": self.social_weight,
            "exploration_weight": self.exploration_weight,
            "energy_threshold": self.energy_threshold,
            "max_energy": self.max_energy,
            "social_radius": self.social_radius,
            "step_cost": self.step_cost,
            "blocked_penalty": self.blocked_penalty,
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> MotivationConfig:
        return cls(**{k: data[k] for k in cls.__dataclass_fields__ if k in data})


@dataclass(slots=True)
class MotivationState:
    energy_drive: float
    social_drive: float
    exploration_drive: float

    def to_dict(self) -> JSONDict:
        return {
            "energy_drive": self.energy_drive,
            "social_drive": self.social_drive,
            "exploration_drive": self.exploration_drive,
        }


def evaluate_drives(
    observation: Observation,
    *,
    visit_count: int,
    config: MotivationConfig,
) -> MotivationState:
    """Map observation and visit history into drive strengths in [0, 1]."""
    energy_frac = observation.energy / max(config.max_energy, 1e-6)
    energy_drive = max(0.0, min(1.0, 1.0 - energy_frac))
    if observation.energy >= config.energy_threshold:
        energy_drive *= 0.5

    nearby = [a for a in observation.visible_agents if a.distance <= config.social_radius]
    if nearby:
        social_drive = 1.0 / (1.0 + min(a.distance for a in nearby))
    else:
        social_drive = 0.0

    # High drive when the current cell is unfamiliar.
    exploration_drive = 1.0 / (1.0 + visit_count)

    return MotivationState(
        energy_drive=energy_drive,
        social_drive=social_drive,
        exploration_drive=exploration_drive,
    )


def _resource_visible(observation: Observation) -> bool:
    return any(cell == CellType.RESOURCE for row in observation.local_cells for cell in row)


def compute_reward(
    observation_before: Observation,
    observation_after: Observation,
    *,
    energy_gained: float,
    moved: bool,
    new_cell: bool,
    config: MotivationConfig,
) -> float:
    """Scalar reward used by value-based learners."""
    reward = 0.0

    # Energy maintenance: collecting resources and remaining above threshold.
    if energy_gained > 0:
        reward += config.energy_weight * (energy_gained / max(config.max_energy, 1e-6))
    if observation_after.energy < config.energy_threshold:
        reward -= config.energy_weight * 0.02
    if _resource_visible(observation_before) and energy_gained > 0:
        reward += 0.05 * config.energy_weight

    # Social proximity: reward being near other agents.
    near_before = sum(1 for a in observation_before.visible_agents if a.distance <= config.social_radius)
    near_after = sum(1 for a in observation_after.visible_agents if a.distance <= config.social_radius)
    if near_after > 0:
        reward += config.social_weight * (0.05 + 0.02 * near_after)
    if near_after > near_before:
        reward += config.social_weight * 0.03

    # Exploration: prefer unfamiliar cells.
    if new_cell:
        reward += config.exploration_weight * 0.1

    reward -= config.step_cost
    if not moved and observation_before.position == observation_after.position:
        # Staying intentionally is cheap; failed moves still incur the step cost only.
        # Blocked moves are indistinguishable from STAY here unless caller flags them.
        pass

    return reward


def compute_reward_with_blocked(
    observation_before: Observation,
    observation_after: Observation,
    *,
    energy_gained: float,
    moved: bool,
    new_cell: bool,
    blocked: bool,
    config: MotivationConfig,
) -> float:
    reward = compute_reward(
        observation_before,
        observation_after,
        energy_gained=energy_gained,
        moved=moved,
        new_cell=new_cell,
        config=config,
    )
    if blocked:
        reward -= config.blocked_penalty
    return reward
