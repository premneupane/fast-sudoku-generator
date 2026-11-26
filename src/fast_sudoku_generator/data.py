"""Locate, download and cache the data files the generator needs.

Two kinds of file live in a per-user data folder:

* ``v36288.txt``: Bertram Felgenhauer and Frazer Jarvis's table of the 36,288 normalized
  top bands and their completion counts (``results2long``). It is not bundled with this
  package. It is downloaded once (gzipped) from Frazer Jarvis's site, decompressed and
  checked against a known SHA-256 hash.
* ``cached-grids.npy`` / ``cached-grids-unweighted.npy``: base-grid caches for the
  vectorized method, built locally with ``nby9 build-cache``.

Environment variables:

* ``FAST_SUDOKU_DATA_DIR``: use this folder instead of the default per-user cache folder.
* ``FAST_SUDOKU_PATTERN_FILE``: use this band table instead of the downloaded one. It is
  not checksum-checked, so it can point at a table you generated yourself.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

APP_NAME = "fast-sudoku-generator"
PATTERN_FILENAME = "v36288.txt"
PATTERN_URL = "https://www.afjarvis.org.uk/sudoku/results2long.gz"
PATTERN_PAGE = "https://www.afjarvis.org.uk/sudoku/bertram.html"
# SHA-256 of the decompressed table (results2long.txt).
PATTERN_SHA256 = "3f1868fb77647e33e7149855176d4cb78013a43cc82ab170f73d1d095f228d1d"
GZIP_MAGIC = b"\x1f\x8b"
ENV_DATA_DIR = "FAST_SUDOKU_DATA_DIR"
ENV_PATTERN_FILE = "FAST_SUDOKU_PATTERN_FILE"
GRID_CACHE_FILENAMES = {
    "weighted": "cached-grids.npy",
    "unweighted": "cached-grids-unweighted.npy",
}


class DataError(RuntimeError):
    """A required data file is missing, could not be downloaded or failed its checksum."""


def data_dir() -> Path:
    """Return the folder for downloaded and generated data (not created here)."""
    override = os.environ.get(ENV_DATA_DIR)
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        base = home / "Library" / "Caches"
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME") or home / ".cache")
    return base / APP_NAME


def grid_cache_path(kind: str = "weighted") -> Path:
    """Return the default path of the weighted or unweighted base-grid cache."""
    try:
        return data_dir() / GRID_CACHE_FILENAMES[kind]
    except KeyError as exc:
        raise ValueError(f"cache kind must be one of: {', '.join(GRID_CACHE_FILENAMES)}") from exc


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pattern_file(download: bool = True) -> Path:
    """Return the path of the band table, downloading it on first use if allowed."""
    override = os.environ.get(ENV_PATTERN_FILE)
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise DataError(f"{ENV_PATTERN_FILE} points to a missing file: {path}")
        return path
    path = data_dir() / PATTERN_FILENAME
    if path.is_file():
        return path
    if not download:
        raise DataError(f"band table not found at {path}; run `nby9 fetch-data` to download it")
    return fetch_pattern_file()


def fetch_pattern_file(source: str | Path | None = None, force: bool = False) -> Path:
    """Download (or copy from ``source``) the band table into the data folder and verify it.

    ``source`` may be a URL or a local file path, plain or gzipped. Without it the gzipped
    file is downloaded from Frazer Jarvis's site. Gzipped input is decompressed, and the
    table is checked against ``PATTERN_SHA256`` before it is kept.
    """
    target = data_dir() / PATTERN_FILENAME
    if target.is_file() and not force:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)

    location = str(source) if source is not None else PATTERN_URL
    is_url = "://" in location
    if is_url:
        print(
            f"Downloading the Felgenhauer-Jarvis band table (one time, about 150 KB)\n  from {location}",
            file=sys.stderr,
        )

    try:
        if is_url:
            request = urllib.request.Request(location, headers={"User-Agent": APP_NAME})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        else:
            payload = Path(location).read_bytes()
    except OSError as exc:
        raise DataError(
            f"could not get the band table from {location}: {exc}\n"
            f"Download results2long.gz by hand from {PATTERN_PAGE} and run\n"
            "  nby9 fetch-data --from /path/to/results2long.gz"
        ) from exc

    if payload[:2] == GZIP_MAGIC:
        try:
            payload = gzip.decompress(payload)
        except (OSError, EOFError) as exc:
            raise DataError(f"{location} is not a valid gzip file: {exc}") from exc

    actual = hashlib.sha256(payload).hexdigest()
    if actual != PATTERN_SHA256:
        hint = (
            "The file may have changed upstream; please open an issue."
            if is_url
            else "This is not the expected results2long table."
        )
        raise DataError(f"checksum mismatch for {location}\n  expected {PATTERN_SHA256}\n  got      {actual}\n{hint}")

    # Write next to the target, then rename, so a half-written file is never left in place.
    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".v36288-", suffix=".part")
    temp_path = Path(temp_name)
    try:
        with os.fdopen(handle, "wb") as destination:
            destination.write(payload)
        os.replace(temp_path, target)
    finally:
        temp_path.unlink(missing_ok=True)

    print(f"Saved band table to {target}", file=sys.stderr)
    return target
