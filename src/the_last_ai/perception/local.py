"""Local perception: agents only see nearby cells and agents."""

from __future__ import annotations

from the_last_ai.types import Observation, VisibleAgent
from the_last_ai.world.world import World


def perceive(
    world: World,
    *,
    agent_id: str,
    energy: float,
    perception_radius: int,
) -> Observation:
    """Build an observation from only locally available world information."""
    if agent_id not in world.agent_positions:
        raise KeyError(f"Agent {agent_id} is not present in the world")

    position = world.agent_positions[agent_id]
    local_cells = world.grid.local_window(position, perception_radius)

    visible: list[VisibleAgent] = []
    for other_id, other_pos in world.agents_within(position, perception_radius, exclude=agent_id):
        relative = (other_pos.x - position.x, other_pos.y - position.y)
        visible.append(
            VisibleAgent(
                agent_id=other_id,
                relative_position=relative,
                distance=position.manhattan(other_pos),
            )
        )
    visible.sort(key=lambda item: (item.distance, item.agent_id))

    return Observation(
        tick=world.tick,
        position=position,
        energy=energy,
        local_cells=local_cells,
        visible_agents=tuple(visible),
        perception_radius=perception_radius,
    )
