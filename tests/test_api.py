"""Tests for the public Python API."""

import json

import numpy as np
import pytest

import fast_sudoku_generator as fsg
from fast_sudoku_generator.cli import main
from tests.sudoku_checks import is_valid_grid


@pytest.mark.parametrize("method", ["backtracking", "b"])
def test_backtracking_grids(method):
    grids = fsg.generate_grids(20, method=method, seed=1)
    assert grids.shape == (20, 81) and grids.dtype == np.uint8
    assert all(is_valid_grid(grid) for grid in grids)


@pytest.mark.parametrize("method", ["vectorized", "v"])
def test_vectorized_grids(small_cache, method):
    grids = fsg.generate_grids(200, method=method, seed=1, grid_file=small_cache)
    assert grids.shape == (200, 81)
    assert all(is_valid_grid(grid) for grid in grids)


def test_seed_is_reproducible(small_cache):
    assert np.array_equal(fsg.generate_grids(5, seed=3), fsg.generate_grids(5, seed=3))
    a = fsg.generate_grids(50, method="v", seed=3, grid_file=small_cache)
    assert np.array_equal(a, fsg.generate_grids(50, method="v", seed=3, grid_file=small_cache))


def test_matches_command_line(tmp_path):
    output = tmp_path / "grids.txt"
    main(["generate", "-n", "4", "--seed", "8", "-o", str(output)])
    from_cli = [json.loads(line) for line in output.read_text().splitlines()]
    assert fsg.generate_grids(4, seed=8).tolist() == from_cli


def test_missing_cache_raises_data_error():
    with pytest.raises(fsg.DataError, match="build-cache"):
        fsg.generate_grids(5, method="vectorized")


@pytest.mark.parametrize("kwargs", [{"count": 0}, {"method": "other"}])
def test_bad_arguments(kwargs):
    with pytest.raises(ValueError):
        fsg.generate_grids(**kwargs)


def test_version_is_set():
    assert fsg.__version__ != "0.0.0"
