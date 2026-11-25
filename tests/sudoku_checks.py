"""Shared assertions for Sudoku grids."""

from __future__ import annotations

from collections.abc import Sequence

DIGITS = set(range(1, 10))


def rows_of(grid: Sequence[int] | Sequence[Sequence[int]]) -> list[list[int]]:
    """Return a 9x9 list from a flat 81-digit grid or a 9x9 nested grid."""
    values = [int(value) for value in (grid.ravel() if hasattr(grid, "ravel") else _flatten(grid))]
    assert len(values) == 81, f"expected 81 cells, got {len(values)}"
    return [values[start : start + 9] for start in range(0, 81, 9)]


def _flatten(grid: Sequence) -> list[int]:
    if grid and isinstance(grid[0], (list, tuple)):
        return [value for row in grid for value in row]
    return list(grid)


def is_valid_grid(grid: Sequence[int] | Sequence[Sequence[int]]) -> bool:
    """True when every row, column and 3x3 box holds the digits 1-9 exactly once."""
    rows = rows_of(grid)
    columns = [list(column) for column in zip(*rows)]
    boxes = [
        [rows[r][c] for r in range(top, top + 3) for c in range(left, left + 3)]
        for top in (0, 3, 6)
        for left in (0, 3, 6)
    ]
    return all(set(unit) == DIGITS for unit in rows + columns + boxes)
