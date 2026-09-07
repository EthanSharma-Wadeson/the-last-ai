"""Grid and world behaviour tests."""

from the_last_ai.types import Action, Position
from the_last_ai.world.grid import Grid
from the_last_ai.world.world import World


def test_bordered_grid_has_walls():
    grid = Grid.empty(5, 4, bordered=True)
    assert grid.get(Position(0, 0)).name == "WALL"
    assert grid.get(Position(2, 1)).name == "EMPTY"
    assert not grid.is_passable(Position(0, 1))


def test_agent_placement_and_movement():
    world = World(grid=Grid.empty(8, 6))
    world.place_agent("a", Position(2, 2))
    assert world.apply_action("a", Action.MOVE_EAST)
    assert world.agent_positions["a"] == Position(3, 2)


def test_wall_blocks_movement():
    world = World(grid=Grid.empty(8, 6))
    world.place_agent("a", Position(1, 1))
    assert not world.apply_action("a", Action.MOVE_WEST)
    assert world.agent_positions["a"] == Position(1, 1)


def test_disappear_and_restore_are_reversible():
    world = World(grid=Grid.empty(8, 6))
    world.place_agent("a", Position(3, 3))
    world.remove_agent("a", reversible=True)
    assert "a" not in world.agent_positions
    restored = world.restore_agent("a")
    assert restored == Position(3, 3)
    assert "a" in world.agent_positions
    assert any(e.event_type.value == "agent_disappear" for e in world.event_log.events)
    assert any(e.event_type.value == "agent_restore" for e in world.event_log.events)


def test_world_roundtrip_dict():
    world = World(grid=Grid.empty(6, 5))
    world.place_agent("a", Position(2, 2))
    world.advance_tick()
    restored = World.from_dict(world.to_dict())
    assert restored.tick == 1
    assert restored.agent_positions["a"] == Position(2, 2)
