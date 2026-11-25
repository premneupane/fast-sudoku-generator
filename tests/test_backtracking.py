import random

from fast_sudoku_generator.backtracking import generate_grid
from tests.sudoku_checks import is_valid_grid


def test_generates_valid_grids():
    rng = random.Random(1)
    for _ in range(25):
        assert is_valid_grid(generate_grid(rng))


def test_returns_nine_rows_of_nine():
    grid = generate_grid(random.Random(1))
    assert len(grid) == 9 and all(len(row) == 9 for row in grid)


def test_same_seed_same_grid():
    assert generate_grid(random.Random(42)) == generate_grid(random.Random(42))


def test_different_seeds_differ():
    grids = {tuple(map(tuple, generate_grid(random.Random(seed)))) for seed in range(10)}
    assert len(grids) == 10
