import random

import numpy as np
import pytest

from utils.generate_cached_grids import generate_completion
from tests.sudoku_checks import is_valid_grid


def test_completion_keeps_the_top_band(pattern_table):
    band = pattern_table.patterns[123]
    grid = generate_completion(band, random.Random(0))
    expected = [[int(d) for d in prefix + suffix] for prefix, suffix in zip(("123", "456", "789"), band)]
    assert grid[:3] == expected
    assert is_valid_grid(grid)


def test_completions_are_valid_for_sampled_patterns(pattern_table):
    rng = random.Random(1)
    for index in rng.sample(range(len(pattern_table.patterns)), 25):
        assert is_valid_grid(generate_completion(pattern_table.patterns[index], rng))


@pytest.mark.parametrize(
    "band",
    [
        ("456789", "789123"),  # too few rows
        ("456789", "789123", "12345"),  # short row
        ("456789", "789123", "12345x"),  # bad character
        ("456789", "456789", "123456"),  # column conflict
    ],
)
def test_rejects_invalid_bands(band):
    with pytest.raises(ValueError):
        generate_completion(band, random.Random(0))


def test_cli_writes_cache(tmp_path, monkeypatch):
    import sys

    from utils import generate_cached_grids

    output = tmp_path / "cache.npy"
    monkeypatch.setattr(sys, "argv", ["prog", "--count", "20", "--seed", "1", "-o", str(output)])
    assert generate_cached_grids.main() == 0
    grids = np.load(output)
    assert grids.shape == (20, 81) and grids.dtype == np.uint8
    assert all(is_valid_grid(grid) for grid in grids)
