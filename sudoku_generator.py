#!/usr/bin/env python3
"""Generate completed 9x9 Sudoku grids using several sampling methods.

Usage:
    python3 sudoku_generator.py -n 1
    python3 sudoku_generator.py -n 1000000 --method b --format npy
    python3 sudoku_generator.py -n 1000000 --method v --format npy

Text output to a terminal is shown as 9x9 grids and is capped at 10 grids unless
--force is given. Files, pipes and JSON get one 81-digit list per grid per line.

Methods: backtracking (b) and vectorized (v). The vectorized method draws
from a prebuilt grid cache and applies Sudoku symmetries in batches. Build a
weighted or unweighted cache with utils/generate_cached_grids.py.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from src.backtracking import generate_grid
from src.vectorized import iter_vectorized_batches

DEFAULT_GRID_FILE = Path(__file__).resolve().parent / "data" / "cached-grids.npy"
DEFAULT_METHOD_OUTPUTS = {
    "backtracking": Path(__file__).resolve().parent / "data" / "backtracking.npy",
    "vectorized": Path(__file__).resolve().parent / "data" / "vectorized.npy",
}
METHODS = ("backtracking", "vectorized")
TERMINAL_LIMIT = 10
METHOD_ALIASES = {
    "b": "backtracking",
    "v": "vectorized",
    **{method: method for method in METHODS},
}


def parse_method(value: str) -> str:
    try:
        return METHOD_ALIASES[value]
    except KeyError as exc:
        valid = ", ".join((*METHODS, "b", "v"))
        raise argparse.ArgumentTypeError(f"choose one of: {valid}") from exc


def format_grid(grid: Sequence[Sequence[int]]) -> str:
    lines = []
    for row_index, row in enumerate(grid):
        if row_index and row_index % 3 == 0:
            lines.append("")
        chunks = [" ".join(map(str, row[start : start + 3])) for start in (0, 3, 6)]
        lines.append("   ".join(chunks))
    return "\n".join(lines)


def report_timings(generation_seconds: float, count: int, start_time: float) -> None:
    end_to_end_seconds = time.perf_counter() - start_time
    print(file=sys.stderr)
    print(f"Grid generation time: {generation_seconds:.3f} seconds ({count:,} grids)", file=sys.stderr)
    print(f"Total end-to-end time: {end_to_end_seconds:.3f} seconds", file=sys.stderr)
    print(file=sys.stderr)


def format_line(flat_grid: np.ndarray) -> str:
    """Format one grid as a single-line list of its 81 digits in row-major order."""
    return "[" + ", ".join(map(str, flat_grid.tolist())) + "]"


def iter_batches(
    args: argparse.Namespace, rng: random.Random, grid_file: Path | None
) -> Iterator[tuple[np.ndarray, float]]:
    """Yield (batch, seconds): (k, 81) uint8 grids and the time spent generating them."""
    if args.method == "vectorized":
        batches = iter_vectorized_batches(args.count, grid_file=grid_file, seed=args.seed)
        while True:
            generation_start = time.perf_counter()
            try:
                batch = next(batches)
            except StopIteration:
                return
            yield batch, time.perf_counter() - generation_start
    else:
        for _ in range(args.count):
            generation_start = time.perf_counter()
            grid = generate_grid(rng)
            seconds = time.perf_counter() - generation_start
            yield np.asarray(grid, dtype=np.uint8).reshape(1, 81), seconds


def _write_npy(args: argparse.Namespace, rng: random.Random, grid_file: Path | None) -> tuple[float, Path]:
    output_path = Path(args.output) if args.output else DEFAULT_METHOD_OUTPUTS[args.method]
    boards = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint8, shape=(args.count, 81))
    generation_seconds = 0.0
    start = 0
    for batch, seconds in iter_batches(args, rng, grid_file):
        generation_seconds += seconds
        boards[start : start + len(batch)] = batch
        start += len(batch)
    boards.flush()
    del boards
    return generation_seconds, output_path


def main() -> int:
    end_to_end_start = time.perf_counter()
    parser = argparse.ArgumentParser(description="Generate valid completed Sudoku grids.")
    parser.add_argument("-n", "--count", type=int, default=1, help="number of grids (default: 1)")
    parser.add_argument(
        "--method",
        type=parse_method,
        default="backtracking",
        metavar="METHOD",
        help="backtracking (b) or vectorized (v)",
    )
    parser.add_argument("--grid-file", type=Path, help="completed-grid cache for v (default: data/cached-grids.npy)")
    parser.add_argument("--seed", type=int, help="seed for repeatable output")
    parser.add_argument("--format", choices=("text", "json", "npy"), default="text")
    parser.add_argument(
        "-o",
        "--output",
        help="output file (npy defaults to data/backtracking.npy for b and data/vectorized.npy for v)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=f"allow more than {TERMINAL_LIMIT} grids to print to the terminal",
    )
    args = parser.parse_args()

    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.grid_file and args.method != "vectorized":
        parser.error("--grid-file is used only with vectorized")

    to_terminal = args.format != "npy" and args.output is None and sys.stdout.isatty()
    if to_terminal and args.count > TERMINAL_LIMIT and not args.force:
        parser.error(
            f"refusing to print {args.count:,} grids to the terminal (limit {TERMINAL_LIMIT}); "
            "write to a file with -o FILE or --format npy, redirect the output, or pass --force"
        )
    if args.format == "npy" and args.output is None:
        default_output = DEFAULT_METHOD_OUTPUTS[args.method]
        print(f"Saving NumPy dataset to {default_output}", file=sys.stderr)

    rng = random.Random(args.seed)
    grid_file = (args.grid_file or DEFAULT_GRID_FILE) if args.method == "vectorized" else None
    if grid_file and not grid_file.is_file():
        parser.error(
            f"grid cache not found: {grid_file}; build it with python3 utils/generate_cached_grids.py --type w"
        )
    if args.format == "npy":
        generation_seconds, output_path = _write_npy(args, rng, grid_file)
        print(f"Dataset saved to {output_path}", file=sys.stderr)
        report_timings(generation_seconds, args.count, end_to_end_start)
        return 0

    # The 9x9 layout is only for a terminal; files and pipes get one 81-digit list per line.
    pretty = args.format == "text" and to_terminal
    if args.output:
        output = open(args.output, "w", encoding="utf-8")
        close_output = True
    else:
        output = sys.stdout
        close_output = False

    generation_seconds = 0.0
    index = 0
    try:
        if pretty:
            output.write("\n")
        elif args.format == "json" and args.count != 1:
            output.write("[\n")
        for batch, seconds in iter_batches(args, rng, grid_file):
            generation_seconds += seconds
            for flat_grid in batch:
                if pretty:
                    if index:
                        output.write("\n")
                    if args.count > 1:
                        output.write(f"Grid {index + 1}:\n")
                    output.write(format_grid(flat_grid.reshape(9, 9).tolist()) + "\n")
                elif args.format == "json" and args.count != 1:
                    output.write((",\n" if index else "") + format_line(flat_grid))
                else:
                    output.write(format_line(flat_grid) + "\n")
                index += 1
        if pretty:
            output.write("\n")
        elif args.format == "json" and args.count != 1:
            output.write("\n]\n")
    finally:
        if close_output:
            output.close()

    report_timings(generation_seconds, args.count, end_to_end_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
