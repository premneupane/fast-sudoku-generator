"""Generate batches of transformed grids sampled from a completed-grid cache."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np


def _generate_chunk(
    base_grids: np.ndarray,
    source_indices: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Transform a batch of selected cached grids with legal Sudoku symmetries."""
    count = len(source_indices)
    source = np.asarray(base_grids[source_indices]).reshape(count, 9, 9)

    row_blocks = np.argsort(rng.random((count, 3)), axis=1)
    rows_within_blocks = np.argsort(rng.random((count, 3, 3)), axis=2)
    row_order = (3 * row_blocks[:, :, None] + rows_within_blocks).reshape(count, 9)

    column_stacks = np.argsort(rng.random((count, 3)), axis=1)
    columns_within_stacks = np.argsort(rng.random((count, 3, 3)), axis=2)
    column_order = (3 * column_stacks[:, :, None] + columns_within_stacks).reshape(count, 9)

    board_indices = np.arange(count)[:, None, None]
    boards = source[board_indices, row_order[:, :, None], column_order[:, None, :]]

    transpose_mask = rng.random(count) < 0.5
    boards[transpose_mask] = boards[transpose_mask].transpose(0, 2, 1)

    digit_maps = np.argsort(rng.random((count, 9)), axis=1) + 1
    boards = digit_maps[board_indices, boards - 1]
    return boards.reshape(count, 81).astype(np.uint8, copy=False)


def iter_vectorized_batches(
    count: int,
    grid_file: str | Path,
    seed: int | None = None,
    batch_size: int = 65_536,
) -> Iterator[np.ndarray]:
    """Yield ``count`` flattened uint8 grids, drawing bases from a .npy cache."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    base_grids = np.load(grid_file, mmap_mode="r")
    if base_grids.ndim != 2 or base_grids.shape[1] != 81 or len(base_grids) == 0:
        raise ValueError(f"base-grid file must contain a nonempty (count, 81) array: {grid_file}")
    rng = np.random.default_rng(seed)
    remaining = count
    order = np.empty(0, dtype=np.int64)
    offset = 0
    while remaining:
        if offset == len(order):
            order = rng.permutation(len(base_grids))
            offset = 0
        chunk_count = min(remaining, batch_size, len(order) - offset)
        source_indices = order[offset : offset + chunk_count]
        yield _generate_chunk(base_grids, source_indices, rng)
        offset += chunk_count
        remaining -= chunk_count
