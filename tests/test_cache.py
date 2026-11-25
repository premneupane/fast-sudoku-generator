import random

import numpy as np
import pytest

from fast_sudoku_generator.cache import generate_completion
from fast_sudoku_generator.cli import main
from fast_sudoku_generator.data import ENV_PATTERN_FILE, grid_cache_path
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


def test_cli_writes_cache(tmp_path, pattern_path):
    output = tmp_path / "cache.npy"
    args = ["build-cache", "--count", "20", "--seed", "1", "-o", str(output), "--pattern-file", str(pattern_path)]
    assert main(args) == 0
    grids = np.load(output)
    assert grids.shape == (20, 81) and grids.dtype == np.uint8
    assert all(is_valid_grid(grid) for grid in grids)


@pytest.mark.parametrize("kind, flag", [("weighted", "w"), ("unweighted", "u")])
def test_cli_default_output_goes_to_data_dir(pattern_path, monkeypatch, kind, flag):
    monkeypatch.setenv(ENV_PATTERN_FILE, str(pattern_path))
    assert main(["build-cache", "--type", flag, "--count", "5", "--seed", "2"]) == 0
    assert np.load(grid_cache_path(kind)).shape == (5, 81)


def test_cli_without_band_table_reports_error(monkeypatch, capsys):
    # no table in the (isolated) data folder and downloads pointed at a missing file
    monkeypatch.setattr("fast_sudoku_generator.data.PATTERN_URL", "file:///nonexistent/results2long.txt")
    assert main(["build-cache", "--count", "5"]) == 1
    assert "fetch-data --from" in capsys.readouterr().err
