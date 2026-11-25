#!/usr/bin/env python3
"""Write random Sudoku-symmetry copies of each grid in a JSON file.

Input and output grids are flat lists of 81 digits in row order, one grid per
line, the same layout sudoku_generator.py writes with --format json. Input
grids given as 9x9 nested lists are also accepted.

Example:
    python3 utils/generate_equivalent_grids.py
    python3 utils/generate_equivalent_grids.py --count 5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "grids.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "equiv_grids.json"


def random_equivalent_grid(grid: Sequence[Sequence[int]], rng: random.Random) -> list[list[int]]:
    """Apply a random standard Sudoku symmetry to one solved grid."""
    if len(grid) != 9 or any(len(row) != 9 for row in grid):
        raise ValueError("grid must be a 9 by 9 solved Sudoku grid")
    if any(value not in range(1, 10) for row in grid for value in row):
        raise ValueError("grid entries must be digits 1 through 9")

    transformed = [list(row) for row in grid]
    if rng.randrange(2):
        transformed = [list(row) for row in zip(*transformed)]

    bands = list(range(3))
    rng.shuffle(bands)
    row_order: list[int] = []
    for band in bands:
        within_band = list(range(3))
        rng.shuffle(within_band)
        row_order.extend(3 * band + row for row in within_band)
    transformed = [transformed[row] for row in row_order]

    stacks = list(range(3))
    rng.shuffle(stacks)
    column_order: list[int] = []
    for stack in stacks:
        within_stack = list(range(3))
        rng.shuffle(within_stack)
        column_order.extend(3 * stack + col for col in within_stack)
    transformed = [[row[col] for col in column_order] for row in transformed]

    digits = list(range(1, 10))
    rng.shuffle(digits)
    return [[digits[value - 1] for value in row] for row in transformed]


def to_rows(grid: Sequence[int] | Sequence[Sequence[int]]) -> list[list[int]]:
    """Return a 9x9 grid from a flat 81-digit list or a 9x9 nested list."""
    if len(grid) == 81 and all(isinstance(value, int) for value in grid):
        return [list(grid[start : start + 9]) for start in range(0, 81, 9)]
    return [list(row) for row in grid]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="input JSON file (default: data/grids.json)")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="output JSON file (default: data/equiv_grids.json)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="random equivalent copies to create per input grid (default: 1)",
    )
    parser.add_argument("--seed", type=int, help="seed for repeatable output")
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different files")

    with args.input.open(encoding="utf-8") as source:
        grids = json.load(source)
    if not isinstance(grids, list):
        parser.error("input JSON must contain a list of Sudoku grids")

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_count = 0
    with args.output.open("w", encoding="utf-8") as destination:
        destination.write("[\n")
        for grid in grids:
            rows = to_rows(grid)
            for _ in range(args.count):
                if output_count:
                    destination.write(",\n")
                flat = [value for row in random_equivalent_grid(rows, rng) for value in row]
                destination.write(json.dumps(flat, separators=(", ", ":")))
                output_count += 1
        destination.write("\n]\n")

    print(f"Wrote {output_count:,} equivalent grids to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
