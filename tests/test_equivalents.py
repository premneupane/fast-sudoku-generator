import json
import random

import pytest

from fast_sudoku_generator.backtracking import generate_grid
from fast_sudoku_generator.cli import main
from fast_sudoku_generator.equivalents import random_equivalent_grid, to_rows
from tests.conftest import PROJECT_ROOT
from tests.sudoku_checks import is_valid_grid


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


def test_cli_reads_and_writes_flat_lines(tmp_path):
    source = tmp_path / "in.json"
    target = tmp_path / "out.json"
    grids = [[v for row in generate_grid(random.Random(s)) for v in row] for s in range(2)]
    source.write_text(json.dumps(grids))
    assert main(["equivalents", str(source), "-o", str(target), "--count", "3"]) == 0
    lines = target.read_text().splitlines()
    assert lines[0] == "[" and lines[-1] == "]" and len(lines) == 2 + 6
    output = json.loads(target.read_text())
    assert len(output) == 6 and all(len(grid) == 81 and is_valid_grid(grid) for grid in output)


def test_cli_writes_to_stdout_by_default(capsys):
    assert main(["equivalents", str(PROJECT_ROOT / "examples" / "grids.json"), "--seed", "1"]) == 0
    grids = json.loads(capsys.readouterr().out)
    assert len(grids) == 1 and is_valid_grid(grids[0])


def test_example_grids_file_is_valid():
    grids = json.loads((PROJECT_ROOT / "examples" / "grids.json").read_text())
    assert len(grids) >= 1 and all(len(grid) == 81 and is_valid_grid(grid) for grid in grids)
