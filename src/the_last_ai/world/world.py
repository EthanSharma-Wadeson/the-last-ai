"""World state: grid, agents present, and dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field

from the_last_ai.types import ACTION_DELTA, Action, CellType, JSONDict, Position, SOCIAL_ACTION_VALUES
from the_last_ai.world.events import EventLog
from the_last_ai.world.grid import Grid


@dataclass
class World:
    grid: Grid
    tick: int = 0
    agent_positions: dict[str, Position] = field(default_factory=dict)
    removed_agents: dict[str, Position] = field(default_factory=dict)
    event_log: EventLog = field(default_factory=EventLog)
    resource_sites: list[Position] = field(default_factory=list)
    resource_energy: float = 20.0
    resource_regen_interval: int = 15
    _resource_cleared_at: dict[tuple[int, int], int] = field(default_factory=dict)

    def place_agent(self, agent_id: str, position: Position) -> None:
        if not self.grid.is_passable(position):
            raise ValueError(f"Cannot place agent {agent_id} on impassable cell {position}")
        if position in self.agent_positions.values():
            occupied = [aid for aid, pos in self.agent_positions.items() if pos == position]
            raise ValueError(f"Cell {position} already occupied by {occupied}")
        self.agent_positions[agent_id] = position
        self.removed_agents.pop(agent_id, None)
        self.event_log.appear(self.tick, agent_id, position)

    def place_resource(self, position: Position, *, register_site: bool = True) -> None:
        if not self.grid.is_passable(position):
            raise ValueError(f"Cannot place resource on impassable cell {position}")
        self.grid.set(position, CellType.RESOURCE)
        if register_site and position not in self.resource_sites:
            self.resource_sites.append(position)
        self._resource_cleared_at.pop(position.as_tuple(), None)
        self.event_log.resource_spawn(self.tick, position)

    def remove_agent(self, agent_id: str, *, reversible: bool = True) -> Position:
        if agent_id not in self.agent_positions:
            raise KeyError(f"Unknown active agent: {agent_id}")
        position = self.agent_positions.pop(agent_id)
        if reversible:
            self.removed_agents[agent_id] = position
        self.event_log.disappear(self.tick, agent_id, reversible=reversible)
        return position

    def restore_agent(self, agent_id: str, position: Position | None = None) -> Position:
        if agent_id in self.agent_positions:
            raise ValueError(f"Agent {agent_id} is already present")
        if position is None:
            if agent_id not in self.removed_agents:
                raise KeyError(f"No stored position for removed agent {agent_id}")
            position = self.removed_agents[agent_id]
        if not self.grid.is_passable(position):
            raise ValueError(f"Cannot restore agent {agent_id} onto impassable cell {position}")
        if position in self.agent_positions.values():
            position = self._find_nearby_free(position)
        self.agent_positions[agent_id] = position
        self.removed_agents.pop(agent_id, None)
        self.event_log.restore(self.tick, agent_id, position)
        return position

    def _find_nearby_free(self, origin: Position) -> Position:
        for radius in range(0, max(self.grid.width, self.grid.height)):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    candidate = origin.offset(dx, dy)
                    if self.grid.is_passable(candidate) and candidate not in self.agent_positions.values():
                        return candidate
        raise RuntimeError("No free cell available for agent restoration")

    def apply_action(self, agent_id: str, action: Action) -> bool:
        """Apply a movement or stay action. Returns True if the agent moved."""
        if agent_id not in self.agent_positions:
            return False
        if action == Action.INTERACT or action in SOCIAL_ACTION_VALUES:
            return False

        current = self.agent_positions[agent_id]
        dx, dy = ACTION_DELTA[action]
        target = current.offset(dx, dy)

        if not self.grid.is_passable(target):
            return False
        if target in self.agent_positions.values():
            return False

        self.agent_positions[agent_id] = target
        return True

    def collect_resources(self) -> dict[str, float]:
        """Agents standing on resources collect them and gain energy."""
        gained: dict[str, float] = {}
        for agent_id, position in list(self.agent_positions.items()):
            if self.grid.get(position) != CellType.RESOURCE:
                continue
            self.grid.set(position, CellType.EMPTY)
            self._resource_cleared_at[position.as_tuple()] = self.tick
            self.event_log.resource_remove(self.tick, position, agent_id=agent_id)
            gained[agent_id] = self.resource_energy
        return gained

    def regenerate_resources(self) -> int:
        """Respawn resources at known sites after the regen interval."""
        if self.resource_regen_interval <= 0:
            return 0
        spawned = 0
        for site in self.resource_sites:
            if self.grid.get(site) == CellType.RESOURCE:
                continue
            if site in self.agent_positions.values():
                continue
            cleared_at = self._resource_cleared_at.get(site.as_tuple())
            if cleared_at is None or (self.tick - cleared_at) < self.resource_regen_interval:
                continue
            self.grid.set(site, CellType.RESOURCE)
            self._resource_cleared_at.pop(site.as_tuple(), None)
            self.event_log.resource_spawn(self.tick, site)
            spawned += 1
        return spawned

    def agents_within(self, center: Position, radius: int, *, exclude: str | None = None) -> list[tuple[str, Position]]:
        found: list[tuple[str, Position]] = []
        for agent_id, position in self.agent_positions.items():
            if exclude is not None and agent_id == exclude:
                continue
            if center.manhattan(position) <= radius:
                found.append((agent_id, position))
        return found

    def advance_tick(self) -> None:
        self.tick += 1

    def to_dict(self) -> JSONDict:
        return {
            "tick": self.tick,
            "grid": self.grid.to_dict(),
            "agent_positions": {aid: pos.as_tuple() for aid, pos in self.agent_positions.items()},
            "removed_agents": {aid: pos.as_tuple() for aid, pos in self.removed_agents.items()},
            "event_log": self.event_log.to_dict(),
            "resource_sites": [pos.as_tuple() for pos in self.resource_sites],
            "resource_energy": self.resource_energy,
            "resource_regen_interval": self.resource_regen_interval,
            "resource_cleared_at": {f"{x},{y}": tick for (x, y), tick in self._resource_cleared_at.items()},
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> World:
        cleared: dict[tuple[int, int], int] = {}
        for key, tick in data.get("resource_cleared_at", {}).items():
            x_str, y_str = key.split(",")
            cleared[(int(x_str), int(y_str))] = int(tick)
        return cls(
            grid=Grid.from_dict(data["grid"]),
            tick=int(data["tick"]),
            agent_positions={
                aid: Position(int(pos[0]), int(pos[1])) for aid, pos in data["agent_positions"].items()
            },
            removed_agents={
                aid: Position(int(pos[0]), int(pos[1])) for aid, pos in data.get("removed_agents", {}).items()
            },
            event_log=EventLog.from_dict(data.get("event_log", {})),
            resource_sites=[
                Position(int(pos[0]), int(pos[1])) for pos in data.get("resource_sites", [])
            ],
            resource_energy=float(data.get("resource_energy", 20.0)),
            resource_regen_interval=int(data.get("resource_regen_interval", 15)),
            _resource_cleared_at=cleared,
        )

    def render_ascii(self) -> str:
        return self.grid.render_ascii(self.agent_positions)
