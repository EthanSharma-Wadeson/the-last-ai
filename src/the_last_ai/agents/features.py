"""Discrete state encoding for tabular value learning."""

from __future__ import annotations

from the_last_ai.types import ACTION_DELTA, Action, CellType, Observation


def energy_bin(energy: float, *, bin_size: float = 20.0, n_bins: int = 5) -> int:
    return max(0, min(n_bins - 1, int(energy // bin_size)))


def _direction_from_delta(dx: int, dy: int) -> str:
    if dx == 0 and dy == 0:
        return "here"
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "south" if dy > 0 else "north"


def nearest_resource_cue(observation: Observation) -> tuple[str, int]:
    """Return (direction, distance_bin) for the nearest visible resource."""
    radius = observation.perception_radius
    best: tuple[int, int, int] | None = None  # dist, dx, dy
    for dy, row in enumerate(observation.local_cells):
        for dx, cell in enumerate(row):
            if cell != CellType.RESOURCE:
                continue
            rel_x = dx - radius
            rel_y = dy - radius
            dist = abs(rel_x) + abs(rel_y)
            candidate = (dist, rel_x, rel_y)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        return ("none", 3)
    dist, rel_x, rel_y = best
    dist_bin = 0 if dist == 0 else 1 if dist <= 2 else 2 if dist <= 4 else 3
    return (_direction_from_delta(rel_x, rel_y), dist_bin)


def nearest_agent_cue(observation: Observation) -> tuple[str, int]:
    if not observation.visible_agents:
        return ("none", 3)
    nearest = min(observation.visible_agents, key=lambda a: (a.distance, a.agent_id))
    dx, dy = nearest.relative_position
    dist_bin = 0 if nearest.distance == 0 else 1 if nearest.distance <= 2 else 2 if nearest.distance <= 4 else 3
    return (_direction_from_delta(dx, dy), dist_bin)


def encode_state(observation: Observation) -> tuple:
    """Compact discrete state used as a Q-table key."""
    e_bin = energy_bin(observation.energy)
    resource_dir, resource_dist = nearest_resource_cue(observation)
    agent_dir, agent_dist = nearest_agent_cue(observation)
    return (e_bin, resource_dir, resource_dist, agent_dir, agent_dist)


def action_toward(direction: str) -> Action | None:
    mapping = {
        "north": Action.MOVE_NORTH,
        "south": Action.MOVE_SOUTH,
        "east": Action.MOVE_EAST,
        "west": Action.MOVE_WEST,
        "here": Action.STAY,
    }
    return mapping.get(direction)


def is_action_aligned(action: Action, direction: str) -> bool:
    target = action_toward(direction)
    return target is not None and action == target


def local_cell_in_direction(observation: Observation, action: Action) -> int | None:
    """Cell type one step in the action direction, if within the local window."""
    dx, dy = ACTION_DELTA[action]
    if dx == 0 and dy == 0:
        radius = observation.perception_radius
        return observation.local_cells[radius][radius]
    radius = observation.perception_radius
    x = radius + dx
    y = radius + dy
    if not (0 <= y < len(observation.local_cells)):
        return None
    row = observation.local_cells[y]
    if not (0 <= x < len(row)):
        return None
    return int(row[x])
