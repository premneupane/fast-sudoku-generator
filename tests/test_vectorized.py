import numpy as np
import pytest

from src.vectorized import iter_vectorized_batches
from tests.sudoku_checks import is_valid_grid


def collect(count, cache, **kwargs):
    return np.concatenate(list(iter_vectorized_batches(count, grid_file=cache, **kwargs)))


def test_yields_requested_count_as_uint8(small_cache):
    grids = collect(120, small_cache, seed=1)
    assert grids.shape == (120, 81)
    assert grids.dtype == np.uint8


def test_all_outputs_are_valid(small_cache):
    assert all(is_valid_grid(grid) for grid in collect(200, small_cache, seed=2))


def test_count_larger_than_cache_reuses_bases_without_duplicates(small_cache):
    grids = collect(500, small_cache, seed=3)  # cache holds 50 grids
    assert len(np.unique(grids, axis=0)) == 500


def test_respects_batch_size(small_cache):
    sizes = [len(batch) for batch in iter_vectorized_batches(30, grid_file=small_cache, seed=4, batch_size=7)]
    assert sum(sizes) == 30
    assert max(sizes) <= 7


def test_same_seed_is_reproducible(small_cache):
    assert np.array_equal(collect(64, small_cache, seed=5), collect(64, small_cache, seed=5))


def test_outputs_are_transformed_not_copied(small_cache):
    base = {grid.tobytes() for grid in np.load(small_cache)}
    grids = collect(50, small_cache, seed=6)
    assert sum(grid.tobytes() in base for grid in grids) < 5


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"count": 5, "batch_size": 0}])
def test_rejects_bad_arguments(small_cache, kwargs):
    with pytest.raises(ValueError):
        next(iter_vectorized_batches(grid_file=small_cache, **kwargs))


def test_rejects_badly_shaped_cache(tmp_path):
    bad = tmp_path / "bad.npy"
    np.save(bad, np.ones((3, 9), dtype=np.uint8))
    with pytest.raises(ValueError):
        next(iter_vectorized_batches(1, grid_file=bad))
