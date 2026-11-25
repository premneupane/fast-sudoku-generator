"""Randomized backtracking engine for complete Sudoku grids."""

from __future__ import annotations

import random

from fast_sudoku_generator.patterns import ALL_DIGITS


def generate_grid(rng: random.Random) -> list[list[int]]:
    """Generate one valid completed grid with randomized MRV backtracking."""
    grid = [0] * 81
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    def search() -> bool:
        best_cells: list[tuple[int, int]] = []
        best_count = 10
        for cell, value in enumerate(grid):
            if value:
                continue
            row, col = divmod(cell, 9)
            box = (row // 3) * 3 + col // 3
            mask = ALL_DIGITS & ~(row_used[row] | col_used[col] | box_used[box])
            count = mask.bit_count()
            if count == 0:
                return False
            if count < best_count:
                best_count = count
                best_cells = [(cell, mask)]
            elif count == best_count:
                best_cells.append((cell, mask))

        if not best_cells:
            return True

        cell, mask = rng.choice(best_cells)
        row, col = divmod(cell, 9)
        box = (row // 3) * 3 + col // 3
        digits = [digit for digit in range(1, 10) if mask & (1 << (digit - 1))]
        rng.shuffle(digits)
        for digit in digits:
            bit = 1 << (digit - 1)
            grid[cell] = digit
            row_used[row] |= bit
            col_used[col] |= bit
            box_used[box] |= bit
            if search():
                return True
            grid[cell] = 0
            row_used[row] ^= bit
            col_used[col] ^= bit
            box_used[box] ^= bit
        return False

    if not search():
        raise RuntimeError("Search failed to find a Sudoku completion")
    return [grid[start : start + 9] for start in range(0, 81, 9)]
