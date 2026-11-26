"""Shared fixtures.

Tests never touch the network or the user's real data folder: every test runs with
FAST_SUDOKU_DATA_DIR pointing at a temporary folder. Tests that need the real
Felgenhauer-Jarvis band table use the `pattern_path` fixture, which looks for an already
downloaded copy and skips (or fails on CI) if there is none.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytest

from fast_sudoku_generator.backtracking import generate_grid
from fast_sudoku_generator.data import ENV_DATA_DIR, ENV_PATTERN_FILE, DataError, pattern_file
from fast_sudoku_generator.patterns import PatternTable, load_pattern_table

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _find_downloaded_table() -> Path | None:
    """Resolved once, before any test changes the environment."""
    try:
        return pattern_file(download=False)
    except DataError:
        return None


DOWNLOADED_TABLE = _find_downloaded_table()


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "fast-sudoku-data"
    monkeypatch.setenv(ENV_DATA_DIR, str(data_dir))
    monkeypatch.delenv(ENV_PATTERN_FILE, raising=False)
    return data_dir


@pytest.fixture(scope="session")
def pattern_path() -> Path:
    if DOWNLOADED_TABLE is None:
        message = "band table not downloaded; run `uv run nby9 fetch-data` first"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    return DOWNLOADED_TABLE


@pytest.fixture(scope="session")
def pattern_table(pattern_path: Path) -> PatternTable:
    return load_pattern_table(pattern_path)


@pytest.fixture(scope="session")
def small_cache(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 50-grid base cache in the same (N, 81) uint8 format the cache builder writes."""
    rng = random.Random(0)
    grids = np.asarray([generate_grid(rng) for _ in range(50)], dtype=np.uint8).reshape(50, 81)
    path = tmp_path_factory.mktemp("cache") / "cache.npy"
    np.save(path, grids)
    return path
