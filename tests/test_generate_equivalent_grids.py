import json
import random
import sys

import pytest

from src.backtracking import generate_grid
from tests.sudoku_checks import is_valid_grid
from utils import generate_equivalent_grids
from utils.generate_equivalent_grids import random_equivalent_grid, to_rows


def test_equivalent_grids_are_valid():
    rng = random.Random(0)
    grid = generate_grid(rng)
    for _ in range(25):
        assert is_valid_grid(random_equivalent_grid(grid, rng))


def test_to_rows_accepts_flat_and_nested():
    grid = generate_grid(random.Random(1))
    flat = [value for row in grid for value in row]
    assert to_rows(flat) == grid
    assert to_rows(grid) == grid


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        random_equivalent_grid([[1] * 9] * 8, random.Random(0))


def test_cli_reads_and_writes_flat_lines(tmp_path, monkeypatch):
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    grids = [[v for row in generate_grid(random.Random(s)) for v in row] for s in range(2)]
    source.write_text(json.dumps(grids))
    monkeypatch.setattr(sys, "argv", ["prog", "--input", str(source), "--output", str(target), "--count", "3"])
    assert generate_equivalent_grids.main() == 0
    lines = target.read_text().splitlines()
    assert lines[0] == "[" and lines[-1] == "]" and len(lines) == 2 + 6
    output = json.loads(target.read_text())
    assert len(output) == 6 and all(len(grid) == 81 and is_valid_grid(grid) for grid in output)


def test_default_placeholder_input_is_usable():
    grids = json.loads(generate_equivalent_grids.DEFAULT_INPUT.read_text())
    assert len(grids) >= 1 and all(len(grid) == 81 and is_valid_grid(grid) for grid in grids)
