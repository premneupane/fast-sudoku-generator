"""End-to-end tests for `fast-sudoku generate` and the top-level command."""

import io
import json
import subprocess
import sys

import numpy as np
import pytest

from fast_sudoku_generator import __version__, generate
from fast_sudoku_generator.cli import main
from tests.sudoku_checks import is_valid_grid


class FakeTerminal(io.StringIO):
    def isatty(self):
        return True


def run(monkeypatch, *args, terminal=False):
    """Run `fast-sudoku generate ARGS`; return (exit code, stdout text)."""
    stdout = FakeTerminal() if terminal else io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    code = main(["generate", *map(str, args)])
    return code, stdout.getvalue()


@pytest.mark.parametrize("method", ["b", "v"])
def test_text_file_has_one_flat_grid_per_line(monkeypatch, tmp_path, small_cache, method):
    output = tmp_path / "grids.txt"
    extra = ["--grid-file", small_cache] if method == "v" else []
    run(monkeypatch, "-n", 5, "--method", method, "--seed", 1, "-o", output, *extra)
    lines = output.read_text().splitlines()
    assert len(lines) == 5
    for line in lines:
        grid = json.loads(line)
        assert len(grid) == 81 and is_valid_grid(grid)


def test_json_many_grids_is_an_array_one_per_line(monkeypatch, tmp_path):
    output = tmp_path / "grids.json"
    run(monkeypatch, "-n", 4, "--format", "json", "--seed", 1, "-o", output)
    text = output.read_text()
    assert text.splitlines()[0] == "[" and len(text.splitlines()) == 6
    grids = json.loads(text)
    assert len(grids) == 4 and all(len(g) == 81 and is_valid_grid(g) for g in grids)


def test_json_single_grid_is_a_bare_list(monkeypatch):
    _, out = run(monkeypatch, "--format", "json", "--seed", 1)
    grid = json.loads(out)
    assert len(grid) == 81 and is_valid_grid(grid)


def test_npy_output(monkeypatch, tmp_path, small_cache):
    output = tmp_path / "grids.npy"
    run(monkeypatch, "-n", 300, "--method", "v", "--grid-file", small_cache, "--format", "npy", "-o", output)
    grids = np.load(output)
    assert grids.shape == (300, 81) and grids.dtype == np.uint8
    assert all(is_valid_grid(grid) for grid in grids)


def test_seed_is_reproducible(monkeypatch):
    assert run(monkeypatch, "-n", 3, "--seed", 9)[1] == run(monkeypatch, "-n", 3, "--seed", 9)[1]


def test_terminal_shows_9x9_layout(monkeypatch):
    _, out = run(monkeypatch, "-n", 2, "--seed", 1, terminal=True)
    assert "Grid 1:" in out and "Grid 2:" in out
    assert "[" not in out


def test_terminal_limit_is_ten():
    assert generate.TERMINAL_LIMIT == 10  # documented in the README


@pytest.mark.parametrize("fmt", ["text", "json"])
def test_terminal_limit(monkeypatch, fmt):
    limit = generate.TERMINAL_LIMIT
    assert run(monkeypatch, "-n", limit, "--format", fmt, terminal=True)[0] == 0
    with pytest.raises(SystemExit) as error:
        run(monkeypatch, "-n", limit + 1, "--format", fmt, terminal=True)
    assert error.value.code == 2


def test_force_lifts_terminal_limit(monkeypatch):
    _, out = run(monkeypatch, "-n", 11, "--force", terminal=True)
    assert out.count("Grid ") == 11


def test_no_limit_when_not_a_terminal(monkeypatch):
    _, out = run(monkeypatch, "-n", 25)
    assert len(out.splitlines()) == 25


@pytest.mark.parametrize(
    "args",
    [["-n", 0], ["--method", "x"], ["--grid-file", "cache.npy"]],
)
def test_bad_arguments_exit_with_usage_error(monkeypatch, args):
    with pytest.raises(SystemExit) as error:
        run(monkeypatch, *args)
    assert error.value.code == 2


@pytest.mark.parametrize("args", [["--method", "v"], ["--method", "v", "--grid-file", "missing.npy"]])
def test_missing_cache_reports_how_to_build_it(monkeypatch, capsys, args):
    code, _ = run(monkeypatch, *args)
    assert code == 1
    assert "fast-sudoku build-cache" in capsys.readouterr().err


def test_npy_default_output_is_current_folder(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    run(monkeypatch, "-n", 3, "--format", "npy")
    assert np.load(tmp_path / "backtracking.npy").shape == (3, 81)


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out.strip() == f"fast-sudoku {__version__}"


def test_runs_as_python_module():
    result = subprocess.run(
        [sys.executable, "-m", "fast_sudoku_generator", "generate", "--format", "json", "--seed", "3"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert is_valid_grid(json.loads(result.stdout))
