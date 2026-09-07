"""2D grid representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from the_last_ai.types import CellType, JSONDict, Position


@dataclass
class Grid:
    width: int
    height: int
    cells: np.ndarray

    @classmethod
    def empty(cls, width: int, height: int, *, bordered: bool = True) -> Grid:
        cells = np.full((height, width), CellType.EMPTY, dtype=np.int8)
        grid = cls(width=width, height=height, cells=cells)
        if bordered:
            grid.cells[0, :] = CellType.WALL
            grid.cells[-1, :] = CellType.WALL
            grid.cells[:, 0] = CellType.WALL
            grid.cells[:, -1] = CellType.WALL
        return grid

    def in_bounds(self, position: Position) -> bool:
        return 0 <= position.x < self.width and 0 <= position.y < self.height

    def get(self, position: Position) -> CellType:
        if not self.in_bounds(position):
            return CellType.WALL
        return CellType(int(self.cells[position.y, position.x]))

    def set(self, position: Position, cell_type: CellType) -> None:
        if not self.in_bounds(position):
            raise IndexError(f"Position out of bounds: {position}")
        self.cells[position.y, position.x] = int(cell_type)

    def is_passable(self, position: Position) -> bool:
        return self.get(position) != CellType.WALL

    def free_positions(self) -> list[Position]:
        positions: list[Position] = []
        for y in range(self.height):
            for x in range(self.width):
                if self.cells[y, x] != CellType.WALL:
                    positions.append(Position(x, y))
        return positions

    def local_window(self, center: Position, radius: int) -> tuple[tuple[int, ...], ...]:
        """Return a square window of cell types centered on `center`.

        Out-of-bounds cells are reported as walls.
        """
        rows: list[tuple[int, ...]] = []
        for dy in range(-radius, radius + 1):
            row: list[int] = []
            for dx in range(-radius, radius + 1):
                row.append(int(self.get(center.offset(dx, dy))))
            rows.append(tuple(row))
        return tuple(rows)

    def to_dict(self) -> JSONDict:
        return {
            "width": self.width,
            "height": self.height,
            "cells": self.cells.tolist(),
        }

    @classmethod
    def from_dict(cls, data: JSONDict) -> Grid:
        cells = np.asarray(data["cells"], dtype=np.int8)
        return cls(width=int(data["width"]), height=int(data["height"]), cells=cells)

    def render_ascii(self, agents: dict[str, Position] | None = None) -> str:
        overlay: dict[tuple[int, int], str] = {}
        if agents:
            for agent_id, pos in agents.items():
                label = agent_id[-1].upper() if agent_id else "?"
                overlay[pos.as_tuple()] = label

        lines: list[str] = []
        for y in range(self.height):
            chars: list[str] = []
            for x in range(self.width):
                key = (x, y)
                if key in overlay:
                    chars.append(overlay[key])
                else:
                    cell = CellType(int(self.cells[y, x]))
                    if cell == CellType.WALL:
                        chars.append("#")
                    elif cell == CellType.RESOURCE:
                        chars.append("*")
                    else:
                        chars.append(".")
            lines.append("".join(chars))
        return "\n".join(lines)
