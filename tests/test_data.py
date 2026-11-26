"""Tests for locating, downloading and verifying data files. None of them use the network:
downloads are exercised with file:// URLs and a fake table whose checksum is patched in."""

import gzip
import hashlib
from pathlib import Path

import pytest

from fast_sudoku_generator import data
from fast_sudoku_generator.cli import main

FAKE_TABLE = b"[456789,789123,123456] => 108374976\n"


@pytest.fixture
def fake_table(tmp_path, monkeypatch) -> Path:
    """A stand-in table file, with PATTERN_SHA256 patched to match it."""
    path = tmp_path / "results2long.txt"
    path.write_bytes(FAKE_TABLE)
    monkeypatch.setattr(data, "PATTERN_SHA256", hashlib.sha256(FAKE_TABLE).hexdigest())
    return path


def leftovers(folder: Path) -> list[Path]:
    return list(folder.glob(".v36288-*")) if folder.exists() else []


@pytest.mark.parametrize(
    "platform, env, expected",
    [
        ("darwin", {}, "Library/Caches/fast-sudoku-generator"),
        ("linux", {}, ".cache/fast-sudoku-generator"),
        ("linux", {"XDG_CACHE_HOME": "{home}/xdg"}, "xdg/fast-sudoku-generator"),
        ("win32", {"LOCALAPPDATA": "{home}/local"}, "local/fast-sudoku-generator"),
    ],
)
def test_default_data_dir_per_platform(tmp_path, monkeypatch, platform, env, expected):
    monkeypatch.delenv(data.ENV_DATA_DIR)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(data.sys, "platform", platform)
    monkeypatch.setattr(data.Path, "home", classmethod(lambda cls: tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, value.format(home=tmp_path))
    assert data.data_dir() == tmp_path / expected


def test_data_dir_env_override(isolated_data_dir):
    assert data.data_dir() == isolated_data_dir


def test_grid_cache_paths(isolated_data_dir):
    assert data.grid_cache_path() == isolated_data_dir / "cached-grids.npy"
    assert data.grid_cache_path("unweighted") == isolated_data_dir / "cached-grids-unweighted.npy"
    with pytest.raises(ValueError):
        data.grid_cache_path("other")


def test_missing_table_without_download_explains_fix():
    with pytest.raises(data.DataError, match="nby9 fetch-data"):
        data.pattern_file(download=False)


def test_env_pattern_file_is_used_as_is(tmp_path, monkeypatch):
    own = tmp_path / "my-table.txt"
    own.write_text("anything")  # not checksum-checked: may be a table you generated yourself
    monkeypatch.setenv(data.ENV_PATTERN_FILE, str(own))
    assert data.pattern_file() == own


def test_env_pattern_file_must_exist(tmp_path, monkeypatch):
    monkeypatch.setenv(data.ENV_PATTERN_FILE, str(tmp_path / "missing.txt"))
    with pytest.raises(data.DataError, match="missing file"):
        data.pattern_file()


def test_fetch_from_local_copy(fake_table, isolated_data_dir):
    path = data.fetch_pattern_file(source=fake_table)
    assert path == isolated_data_dir / data.PATTERN_FILENAME
    assert path.read_bytes() == FAKE_TABLE
    assert not leftovers(isolated_data_dir)


def test_download_on_first_use(fake_table, monkeypatch, isolated_data_dir):
    monkeypatch.setattr(data, "PATTERN_URL", fake_table.as_uri())
    assert data.pattern_file().read_bytes() == FAKE_TABLE
    # second call finds the cached copy and does not download again
    monkeypatch.setattr(data, "PATTERN_URL", "file:///nonexistent")
    assert data.pattern_file().read_bytes() == FAKE_TABLE


def test_checksum_mismatch_keeps_nothing(fake_table, tmp_path, isolated_data_dir):
    wrong = tmp_path / "wrong.txt"
    wrong.write_bytes(FAKE_TABLE + b"extra\n")
    with pytest.raises(data.DataError, match="checksum mismatch"):
        data.fetch_pattern_file(source=wrong)
    assert not (isolated_data_dir / data.PATTERN_FILENAME).exists()
    assert not leftovers(isolated_data_dir)


def test_force_with_bad_file_keeps_good_copy(fake_table, tmp_path, isolated_data_dir):
    good = data.fetch_pattern_file(source=fake_table)
    wrong = tmp_path / "wrong.txt"
    wrong.write_bytes(b"junk\n")
    with pytest.raises(data.DataError):
        data.fetch_pattern_file(source=wrong, force=True)
    assert good.read_bytes() == FAKE_TABLE


def test_existing_copy_is_not_replaced_without_force(fake_table, tmp_path):
    first = data.fetch_pattern_file(source=fake_table)
    other = tmp_path / "other.txt"
    other.write_bytes(b"junk\n")
    assert data.fetch_pattern_file(source=other) == first  # no error: nothing fetched


def test_unreachable_source_suggests_manual_download(isolated_data_dir):
    with pytest.raises(data.DataError, match="fetch-data --from"):
        data.fetch_pattern_file(source="file:///nonexistent/results2long.txt")
    assert not leftovers(isolated_data_dir)


def test_real_table_matches_pinned_checksum(pattern_path):
    assert data.sha256_of(pattern_path) == data.PATTERN_SHA256


def test_cli_fetch_data(fake_table, capsys, isolated_data_dir):
    assert main(["fetch-data", "--from", str(fake_table)]) == 0
    assert capsys.readouterr().out.strip() == str(isolated_data_dir / data.PATTERN_FILENAME)


def test_cli_fetch_data_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.txt"
    bad.write_text("junk")
    assert main(["fetch-data", "--from", str(bad)]) == 1
    assert "checksum mismatch" in capsys.readouterr().err


def test_default_url_is_the_gzipped_table():
    assert data.PATTERN_URL.endswith("/results2long.gz")


def test_gzipped_download_is_decompressed(fake_table, tmp_path, monkeypatch):
    packed = tmp_path / "results2long.gz"
    packed.write_bytes(gzip.compress(FAKE_TABLE))
    monkeypatch.setattr(data, "PATTERN_URL", packed.as_uri())
    assert data.pattern_file().read_bytes() == FAKE_TABLE


def test_gzipped_local_copy(fake_table, tmp_path):
    packed = tmp_path / "results2long.gz"
    packed.write_bytes(gzip.compress(FAKE_TABLE))
    assert data.fetch_pattern_file(source=packed).read_bytes() == FAKE_TABLE


def test_corrupt_gzip_is_rejected(fake_table, tmp_path, isolated_data_dir):
    broken = tmp_path / "broken.gz"
    broken.write_bytes(gzip.compress(FAKE_TABLE)[:-8])
    with pytest.raises(data.DataError, match="not a valid gzip file"):
        data.fetch_pattern_file(source=broken)
    assert not (isolated_data_dir / data.PATTERN_FILENAME).exists()
