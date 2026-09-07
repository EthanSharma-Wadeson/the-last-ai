"""Agent base class and random baseline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from the_last_ai.memory.short_term import ShortTermMemory
from the_last_ai.perception.local import perceive
from the_last_ai.rng import ExperimentRNG
from the_last_ai.types import MOVEMENT_ACTIONS, Action, JSONDict, Observation
from the_last_ai.world.world import World


@dataclass
class AgentState:
    agent_id: str
    energy: float = 100.0
    age: int = 0
    perception_radius: int = 3
    energy_cost_per_tick: float = 0.1
    alive: bool = True
    max_energy: float = 100.0


class Agent(ABC):
    """Artificial organism-like computational system."""

    def __init__(
        self,
        agent_id: str,
        *,
        energy: float = 100.0,
        perception_radius: int = 3,
        stm_capacity: int = 32,
        max_energy: float = 100.0,
    ) -> None:
        self.state = AgentState(
            agent_id=agent_id,
            energy=energy,
            perception_radius=perception_radius,
            max_energy=max_energy,
        )
        self.memory = ShortTermMemory(capacity=stm_capacity)
        self.last_observation: Observation | None = None
        self.last_action: Action | None = None
        self.total_reward: float = 0.0
        self.resources_collected: int = 0

    @property
    def agent_id(self) -> str:
        return self.state.agent_id

    def observe(self, world: World, *, cognitive: bool = True) -> Observation:
        observation = perceive(
            world,
            agent_id=self.agent_id,
            energy=self.state.energy,
            perception_radius=self.state.perception_radius,
        )
        self.last_observation = observation
        if cognitive:
            self.memory.push(observation)
        return observation

    def update_after_tick(self) -> None:
        self.state.age += 1
        self.state.energy = max(0.0, self.state.energy - self.state.energy_cost_per_tick)

    def apply_energy_gain(self, amount: float) -> None:
        self.state.energy = min(self.state.max_energy, max(0.0, self.state.energy + amount))

    @abstractmethod
    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        raise NotImplementedError

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
        """Default: no learning. Subclasses override."""
        if energy_gained > 0:
            self.resources_collected += 1
        if reward is not None:
            self.total_reward += reward

    def to_dict(self) -> JSONDict:
        return {
            "type": self.__class__.__name__,
            "agent_id": self.agent_id,
            "energy": self.state.energy,
            "age": self.state.age,
            "perception_radius": self.state.perception_radius,
            "energy_cost_per_tick": self.state.energy_cost_per_tick,
            "max_energy": self.state.max_energy,
            "alive": self.state.alive,
            "memory": self.memory.to_dict(),
            "last_action": self.last_action.value if self.last_action else None,
            "total_reward": self.total_reward,
            "resources_collected": self.resources_collected,
        }

    def load_state(self, data: JSONDict) -> None:
        self.state.energy = float(data["energy"])
        self.state.age = int(data["age"])
        self.state.perception_radius = int(data["perception_radius"])
        self.state.energy_cost_per_tick = float(data.get("energy_cost_per_tick", 0.1))
        self.state.max_energy = float(data.get("max_energy", 100.0))
        self.state.alive = bool(data.get("alive", True))
        self.memory = ShortTermMemory.from_dict(data.get("memory", {}))
        last = data.get("last_action")
        self.last_action = Action(last) if last else None
        self.total_reward = float(data.get("total_reward", 0.0))
        self.resources_collected = int(data.get("resources_collected", 0))


class RandomAgent(Agent):
    """Baseline agent: selects movement actions uniformly at random."""

    def select_action(self, observation: Observation, rng: ExperimentRNG) -> Action:
        return Action(rng.choice(MOVEMENT_ACTIONS))


def create_agent(agent_type: str, agent_id: str, **kwargs) -> Agent:
    # Local imports avoid circular dependency at module load.
    from the_last_ai.agents.memory_agent import MemoryAgent
    from the_last_ai.agents.predictive_agent import PredictiveAgent
    from the_last_ai.agents.social_agent import CooperativePartner, SocialAgent
    from the_last_ai.agents.value_learner import ValueBasedAgent

    mapping = {
        "RandomAgent": RandomAgent,
        "ValueBasedAgent": ValueBasedAgent,
        "MemoryAgent": MemoryAgent,
        "SocialAgent": SocialAgent,
        "CooperativePartner": CooperativePartner,
        "PredictiveAgent": PredictiveAgent,
    }
    if agent_type not in mapping:
        raise ValueError(f"Unknown agent type: {agent_type}")
    return mapping[agent_type](agent_id, **kwargs)
