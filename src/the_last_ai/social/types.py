"""Social interaction kinds and outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from the_last_ai.types import Action, JSONDict


class InteractionKind(str, Enum):
    COOPERATE = "cooperate"
    COMPETE = "compete"
    SHARE = "share"
    COMMUNICATE = "communicate"


# Actions that attempt a social interaction with a nearby agent.
SOCIAL_ACTIONS = (
    Action.COOPERATE,
    Action.COMPETE,
    Action.SHARE,
    Action.COMMUNICATE,
)

ACTION_TO_INTERACTION: dict[Action, InteractionKind] = {
    Action.COOPERATE: InteractionKind.COOPERATE,
    Action.COMPETE: InteractionKind.COMPETE,
    Action.SHARE: InteractionKind.SHARE,
    Action.COMMUNICATE: InteractionKind.COMMUNICATE,
}

INTERACTION_TO_ACTION: dict[InteractionKind, Action] = {
    kind: action for action, kind in ACTION_TO_INTERACTION.items()
}


@dataclass(slots=True)
class InteractionOutcome:
    """Result of a resolved pairwise interaction (both sides updated)."""

    tick: int
    agent_a: str
    agent_b: str
    intent_a: InteractionKind | None
    intent_b: InteractionKind | None
    energy_delta_a: float
    energy_delta_b: float
    valence_a: float  # roughly [-1, 1]
    valence_b: float
    resolved_kind: str
    messages: list[JSONDict] | None = None

    def for_agent(self, agent_id: str) -> tuple[str, float, float, InteractionKind | None]:
        """Return (other_id, energy_delta, valence, own_intent)."""
        if agent_id == self.agent_a:
            return self.agent_b, self.energy_delta_a, self.valence_a, self.intent_a
        if agent_id == self.agent_b:
            return self.agent_a, self.energy_delta_b, self.valence_b, self.intent_b
        raise KeyError(agent_id)

    def to_dict(self) -> JSONDict:
        data: JSONDict = {
            "tick": self.tick,
            "agent_a": self.agent_a,
            "agent_b": self.agent_b,
            "intent_a": self.intent_a.value if self.intent_a else None,
            "intent_b": self.intent_b.value if self.intent_b else None,
            "energy_delta_a": self.energy_delta_a,
            "energy_delta_b": self.energy_delta_b,
            "valence_a": self.valence_a,
            "valence_b": self.valence_b,
            "resolved_kind": self.resolved_kind,
        }
        if self.messages:
            data["messages"] = list(self.messages)
        return data
