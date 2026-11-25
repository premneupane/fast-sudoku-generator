import math

import pytest

from src.shared_helper import load_pattern_table

TOTAL_GRIDS = 6_670_903_752_021_072_936_960


def test_table_has_36288_rows(pattern_table):
    assert len(pattern_table.patterns) == len(pattern_table.weights) == 36_288


def test_first_row_matches_file(pattern_table):
    assert pattern_table.patterns[0] == ("456789", "789123", "123456")
    assert pattern_table.weights[0] == 108_374_976


def test_cumulative_weights_end_at_total(pattern_table):
    assert pattern_table.cumulative_weights[-1] == sum(pattern_table.weights)


def test_weights_take_44_distinct_values(pattern_table):
    assert len(set(pattern_table.weights)) == 44


def test_weights_reproduce_total_grid_count(pattern_table):
    assert sum(pattern_table.weights) * 72**2 * math.factorial(9) == TOTAL_GRIDS


def test_malformed_line_raises(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("[456789,789123] => 5\n")
    with pytest.raises(ValueError, match="Cannot parse"):
        load_pattern_table(path)


def test_wrong_row_count_raises(tmp_path):
    path = tmp_path / "short.txt"
    path.write_text("[456789,789123,123456] => 108374976\n")
    with pytest.raises(ValueError, match="36,288"):
        load_pattern_table(path)
