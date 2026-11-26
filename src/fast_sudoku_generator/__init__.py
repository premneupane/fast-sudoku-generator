"""Generate completed 9x9 Sudoku grids quickly and measure how samples are distributed.

Python API::

    from fast_sudoku_generator import generate_grids

    grids = generate_grids(1000, seed=42)  # backtracking, shape (1000, 81)
    grids = generate_grids(1_000_000, method="vectorized")  # needs `nby9 build-cache` once
"""

from __future__ import annotations

import random
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from fast_sudoku_generator.backtracking import generate_grid
from fast_sudoku_generator.data import DataError, grid_cache_path
from fast_sudoku_generator.vectorized import iter_vectorized_batches

try:
    __version__ = version("nby9")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0"

__all__ = ["DataError", "__version__", "generate_grid", "generate_grids", "iter_vectorized_batches"]


def generate_grids(
    count: int = 1,
    method: str = "backtracking",
    seed: int | None = None,
    grid_file: str | Path | None = None,
) -> np.ndarray:
    """Return ``count`` completed grids as a ``(count, 81)`` uint8 array in row order.

    ``method`` is ``"backtracking"`` (``"b"``) or ``"vectorized"`` (``"v"``). The vectorized
    method samples from a base-grid cache: ``grid_file`` if given, otherwise the weighted
    cache built by ``nby9 build-cache``. Results match the command line for the
    same method, count, seed and cache.
    """
    if count < 1:
        raise ValueError("count must be at least 1")
    method = {"b": "backtracking", "v": "vectorized"}.get(method, method)
    if method == "backtracking":
        rng = random.Random(seed)
        return np.asarray([generate_grid(rng) for _ in range(count)], dtype=np.uint8).reshape(count, 81)
    if method == "vectorized":
        cache = Path(grid_file) if grid_file is not None else grid_cache_path("weighted")
        if not cache.is_file():
            raise DataError(f"grid cache not found: {cache}; build it with `nby9 build-cache`")
        return np.concatenate(list(iter_vectorized_batches(count, grid_file=cache, seed=seed)))
    raise ValueError("method must be 'backtracking' (b) or 'vectorized' (v)")
