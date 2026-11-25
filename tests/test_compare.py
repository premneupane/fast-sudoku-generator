import math
import random

import numpy as np
import pytest

from fast_sudoku_generator.backtracking import generate_grid
from fast_sudoku_generator.cli import main
from fast_sudoku_generator.compare import (
    _encode_pattern,
    _normalized_band_codes,
    compare_distributions,
    duplicate_grid_count,
    grid_pattern_counts,
)


def test_identical_distributions_have_zero_divergence():
    metrics = compare_distributions([1, 2, 3], [2, 4, 6])
    assert metrics["kl_p_to_q_bits"] == pytest.approx(0)
    assert metrics["js_divergence_bits"] == pytest.approx(0)
    assert metrics["total_variation"] == pytest.approx(0)


def test_known_values():
    metrics = compare_distributions([1, 0], [0.5, 0.5])
    assert metrics["entropy_p_bits"] == pytest.approx(0)
    assert metrics["entropy_q_bits"] == pytest.approx(1)
    assert metrics["kl_p_to_q_bits"] == pytest.approx(1)
    assert math.isinf(metrics["kl_q_to_p_bits"])
    assert metrics["total_variation"] == pytest.approx(0.5)


def test_mappings_are_aligned_by_key():
    metrics = compare_distributions({"a": 1, "b": 1}, {"b": 1, "a": 1})
    assert metrics["total_variation"] == pytest.approx(0)


@pytest.mark.parametrize(
    "p, q, error",
    [
        ([1, 2], [1, 2, 3], ValueError),
        ([1, -1], [1, 1], ValueError),
        ([0, 0], [1, 1], ValueError),
        ({"a": 1}, [1], TypeError),
    ],
)
def test_rejects_bad_inputs(p, q, error):
    with pytest.raises(error):
        compare_distributions(p, q)


def test_band_code_matches_table_row_for_canonical_band(pattern_table):
    band = pattern_table.patterns[0]
    grid = np.asarray(generate_grid(random.Random(0)), dtype=np.uint8)
    grid[:3] = [[int(d) for d in prefix + suffix] for prefix, suffix in zip(("123", "456", "789"), band)]
    assert int(_normalized_band_codes(grid[None])[0]) == _encode_pattern(band)


def test_band_code_ignores_relabeling_and_column_moves():
    grid = np.asarray(generate_grid(random.Random(7)), dtype=np.uint8)
    relabel = np.asarray([0, 5, 3, 9, 1, 8, 2, 7, 4, 6], dtype=np.uint8)
    moved = relabel[grid]
    moved = moved[:, [0, 1, 2, 7, 6, 8, 4, 3, 5]]  # swap stacks 2/3 and permute columns within them
    codes = _normalized_band_codes(np.stack([grid, moved]))
    assert codes[0] == codes[1]


def test_counts_cover_every_grid(small_cache, pattern_table):
    counts, size = grid_pattern_counts(small_cache, pattern_table)
    assert size == 50 and counts.sum() == 50


def test_duplicate_count(tmp_path, small_cache):
    grids = np.load(small_cache)
    path = tmp_path / "dupes.npy"
    np.save(path, np.concatenate([grids, grids[:7]]))
    assert duplicate_grid_count(path) == 7
    assert duplicate_grid_count(small_cache) == 0


def test_cli_compares_two_grid_files(pattern_path, small_cache, tmp_path, capsys):
    other = tmp_path / "other.npy"
    np.save(other, np.load(small_cache)[::-1])
    assert main(["compare", str(small_cache), str(other), "--pattern-file", str(pattern_path)]) == 0
    report = capsys.readouterr().out
    assert "Top-band bucket comparison" in report
    assert "Total variation distance" in report


def test_cli_compares_two_json_distributions(tmp_path, capsys):
    p, q = tmp_path / "p.json", tmp_path / "q.json"
    p.write_text("[1, 1]")
    q.write_text("[1, 3]")
    assert main(["compare", str(p), str(q)]) == 0
    assert "Total variation distance: 25.00%" in capsys.readouterr().out


def test_cli_rejects_mixed_inputs(tmp_path):
    with pytest.raises(SystemExit) as error:
        main(["compare", str(tmp_path / "a.npy"), str(tmp_path / "b.json")])
    assert error.value.code == 2
