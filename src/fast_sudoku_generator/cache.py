"""Build the base-grid cache used by the vectorized method.

Each grid starts from one of the 36,288 normalized top bands in the Felgenhauer-Jarvis
table (downloaded on first use) and is completed by randomized backtracking. The
weighted cache picks bands in proportion to their completion counts.

Examples:
    nby9 build-cache                         weighted, 1,000,000 grids
    nby9 build-cache --type u --seed 42      unweighted
"""

from __future__ import annotations

import argparse
import bisect
import random
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from fast_sudoku_generator.data import grid_cache_path, pattern_file
from fast_sudoku_generator.patterns import ALL_DIGITS, load_pattern_table

TYPE_ALIASES = {
    "weighted": "weighted",
    "w": "weighted",
    "unweighted": "unweighted",
    "u": "unweighted",
}


def generate_completion(first_band: Sequence[str], rng: random.Random) -> list[list[int]]:
    """Complete a fixed top band from the lookup-table format."""
    if len(first_band) != 3:
        raise ValueError("first_band must contain exactly three six-digit strings")

    prefixes = ("123", "456", "789")
    grid = [0] * 81
    row_used = [0] * 9
    col_used = [0] * 9
    box_used = [0] * 9

    for row_index, suffix in enumerate(first_band):
        suffix = str(suffix)
        if len(suffix) != 6 or any(char not in "123456789" for char in suffix):
            raise ValueError("each first-band entry must be six digits from 1 to 9")
        for col_index, digit in enumerate(map(int, prefixes[row_index] + suffix)):
            bit = 1 << (digit - 1)
            box = (row_index // 3) * 3 + col_index // 3
            if row_used[row_index] & bit or col_used[col_index] & bit or box_used[box] & bit:
                raise ValueError("first_band conflicts with Sudoku row, column, or box rules")
            grid[row_index * 9 + col_index] = digit
            row_used[row_index] |= bit
            col_used[col_index] |= bit
            box_used[box] |= bit

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
        raise ValueError("first_band is valid locally but has no Sudoku completion")
    return [grid[start : start + 9] for start in range(0, 81, 9)]


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--count", type=int, default=1_000_000, help="number of grids (default: 1,000,000)")
    parser.add_argument(
        "--type",
        dest="grid_type",
        choices=tuple(TYPE_ALIASES),
        default="weighted",
        metavar="{weighted,w,unweighted,u}",
        help="pattern selection rule; w/weighted uses completion counts (default: weighted)",
    )
    parser.add_argument("--pattern-file", type=Path, help="band table to use (default: the downloaded table)")
    parser.add_argument("-o", "--output", type=Path, help="cache output path (default: the data folder)")
    parser.add_argument("--seed", type=int, help="seed for a repeatable cache")
    parser.add_argument("--progress-every", type=int, default=50_000, help="progress interval")


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    started = time.perf_counter()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")

    grid_type = TYPE_ALIASES[args.grid_type]
    output_path = args.output or grid_cache_path(grid_type)
    table = load_pattern_table(args.pattern_file or pattern_file())
    if len(table.patterns) != 36_288:
        parser.error("the cache builder requires the 36,288-row v36288 file")

    rng = random.Random(args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grids = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint8, shape=(args.count, 81))
    cumulative = table.cumulative_weights
    total_weight = cumulative[-1]
    generation_started = time.perf_counter()

    for index in range(args.count):
        if grid_type == "weighted":
            ticket = rng.randrange(total_weight)
            pattern_index = bisect.bisect_right(cumulative, ticket)
        else:
            pattern_index = rng.randrange(len(table.patterns))
        grid = generate_completion(table.patterns[pattern_index], rng)
        grids[index] = np.asarray(grid, dtype=np.uint8).reshape(81)
        completed = index + 1
        if completed % args.progress_every == 0 or completed == args.count:
            elapsed = time.perf_counter() - generation_started
            print(
                f"Generated {completed:,}/{args.count:,} {grid_type} grids ({elapsed:.1f} s)",
                file=sys.stderr,
                flush=True,
            )

    grids.flush()
    del grids
    print(f"{grid_type.title()} cache saved to {output_path}", file=sys.stderr)
    print(f"Total end-to-end time: {time.perf_counter() - started:.3f} seconds", file=sys.stderr)
    return 0
