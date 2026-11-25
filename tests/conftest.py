"""Shared fixtures. Tests never depend on the large, git-ignored .npy files in data/."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest

from src.backtracking import generate_grid
from src.shared_helper import PatternTable, load_pattern_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATTERN_FILE = PROJECT_ROOT / "data" / "v36288.txt"


@pytest.fixture(scope="session")
def pattern_table() -> PatternTable:
    return load_pattern_table(PATTERN_FILE)


@pytest.fixture(scope="session")
def small_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 50-grid base cache in the same (N, 81) uint8 format the cache builder writes."""
    rng = random.Random(0)
    grids = np.asarray([generate_grid(rng) for _ in range(50)], dtype=np.uint8).reshape(50, 81)
    path = tmp_path_factory.mktemp("cache") / "cache.npy"
    np.save(path, grids)
    return path
