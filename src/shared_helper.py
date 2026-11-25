"""Constants and shared lookup-table helpers for Sudoku tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ALL_DIGITS = (1 << 9) - 1


@dataclass(frozen=True)
class PatternTable:
    patterns: list[tuple[str, str, str]]
    weights: list[int]
    cumulative_weights: list[int]


def load_pattern_table(path: str | Path) -> PatternTable:
    """Read the 36,288 weighted top-band patterns from v36288."""
    pattern_re = re.compile(r"^\[([1-9]{6}),([1-9]{6}),([1-9]{6})\]\s*=>\s*(\d+)\s*$")
    patterns: list[tuple[str, str, str]] = []
    weights: list[int] = []
    cumulative_weights: list[int] = []
    running_total = 0

    with open(path, "r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line or line == "\x1a":
                continue
            match = pattern_re.fullmatch(line)
            if not match:
                raise ValueError(f"Cannot parse pattern data at {path}:{line_number}: {line!r}")
            weight = int(match.group(4))
            if weight <= 0:
                raise ValueError(f"Pattern weight must be positive at {path}:{line_number}")
            patterns.append((match.group(1), match.group(2), match.group(3)))
            weights.append(weight)
            running_total += weight
            cumulative_weights.append(running_total)

    if len(patterns) != 36_288:
        raise ValueError(f"Expected 36,288 v36288 pattern rows, found {len(patterns)} in {path}")
    return PatternTable(patterns, weights, cumulative_weights)
