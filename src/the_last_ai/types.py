"""Shared types and enumerations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any


class CellType(IntEnum):
    EMPTY = 0
    WALL = 1
    RESOURCE = 2


class Action(Enum):
    STAY = "stay"
    MOVE_NORTH = "move_north"
    MOVE_SOUTH = "move_south"
    MOVE_EAST = "move_east"
    MOVE_WEST = "move_west"
    INTERACT = "interact"  # legacy generic; prefer explicit social actions
    COOPERATE = "cooperate"
    COMPETE = "compete"
    SHARE = "share"
    COMMUNICATE = "communicate"


# (dx, dy) with y increasing southward on the grid.
ACTION_DELTA: dict[Action, tuple[int, int]] = {
    Action.STAY: (0, 0),
    Action.MOVE_NORTH: (0, -1),
    Action.MOVE_SOUTH: (0, 1),
    Action.MOVE_EAST: (1, 0),
    Action.MOVE_WEST: (-1, 0),
    Action.INTERACT: (0, 0),
    Action.COOPERATE: (0, 0),
    Action.COMPETE: (0, 0),
    Action.SHARE: (0, 0),
    Action.COMMUNICATE: (0, 0),
}

MOVEMENT_ACTIONS = (
    Action.STAY,
    Action.MOVE_NORTH,
    Action.MOVE_SOUTH,
    Action.MOVE_EAST,
    Action.MOVE_WEST,
)

SOCIAL_ACTION_VALUES = (
    Action.COOPERATE,
    Action.COMPETE,
    Action.SHARE,
    Action.COMMUNICATE,
)


@dataclass(frozen=True, slots=True)
class Position:
    x: int
    y: int

    def offset(self, dx: int, dy: int) -> Position:
        return Position(self.x + dx, self.y + dy)

    def manhattan(self, other: Position) -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def as_tuple(self) -> tuple[int, int]:
        return (self.x, self.y)


@dataclass(frozen=True, slots=True)
class VisibleAgent:
    agent_id: str
    relative_position: tuple[int, int]
    distance: int


@dataclass(slots=True)
class Observation:
    tick: int
    position: Position
    energy: float
    local_cells: tuple[tuple[int, ...], ...]
    visible_agents: tuple[VisibleAgent, ...]
    perception_radius: int


JSONDict = dict[str, Any]
