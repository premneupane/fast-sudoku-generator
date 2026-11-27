# nby9: fast Sudoku grid generator

Generate large numbers of completed 9×9 Sudoku grids quickly, and measure how the samples are distributed.

There are two generation methods:

- **Backtracking (`b`)** builds each grid from scratch with a randomized depth-first search. It needs no setup, but it
  is slow for large batches.
- **Vectorized (`v`)** samples base grids from a prebuilt cache and applies random Sudoku symmetries to whole batches
  at once with NumPy. It is about 170 times faster.

The package also builds the cache, compares the distributions of generated datasets, and makes symmetry copies of
grids. Everything is available as the `nby9` command and as a Python API.

## Install

```bash
pip install nby9
```

or, to install just the command with [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/):

```bash
uv tool install nby9
pipx install nby9
```

Requires Python 3.10 or later. The only dependency is NumPy. The package installs the `nby9` command and the
`fast_sudoku_generator` Python module.

## Quick start

```bash
# One grid, printed as a 9x9 layout
nby9 generate

# A thousand reproducible grids, one per line, to a file
nby9 generate -n 1000 --seed 42 -o grids.txt

# For the fast method, build the base-grid cache once (about 7 minutes, 78 MB)
nby9 build-cache

# A million grids as a NumPy array
nby9 generate -n 1000000 --method v --format npy -o grids.npy
```

The first command that needs the Felgenhauer–Jarvis band table downloads it (about 150 KB, once). See "Data files" below.

## Commands

Run `nby9 COMMAND --help` for every option.

### `nby9 generate`

| Option | Description |
| --- | --- |
| `-n`, `--count` | Number of grids to generate (default: 1). |
| `--method` | `backtracking` / `b` (default) or `vectorized` / `v`. |
| `--grid-file` | Base-grid cache for the vectorized method (default: the weighted cache from `build-cache`). |
| `--seed` | Seed for repeatable output. |
| `--format` | `text` (default), `json` or `npy`. |
| `-o`, `--output` | Output file. Text and JSON default to standard output; `npy` defaults to `backtracking.npy` or `vectorized.npy` in the current folder. |
| `--force` | Allow more than 10 grids to print to the terminal. |

Output formats:

- **text**: in a terminal, grids are laid out in 3×3 boxes and numbered when there is more than one. Written to a file
  or a pipe, each grid is one line holding a list of its 81 digits in row order, for example `[9, 7, 8, ..., 1, 2]`.
- **json**: each grid is a flat list of 81 digits in row order. One grid is written as a bare list; more than one is
  written as a JSON array with one grid per line.
- **npy**: a `(count, 81)` `uint8` array, written through a memory map so large datasets do not have to fit in memory.

To keep the terminal readable, text and JSON output to a terminal is limited to 10 grids. Larger requests stop with an
error that suggests `-o FILE` or `--format npy`. Redirected or piped output (`> grids.txt`, `| head`) is not limited,
and `--force` lifts the limit. Timing information goes to standard error, so standard output stays clean for piping.

### `nby9 build-cache`

Builds the base-grid cache that the vectorized method samples from. Each grid starts from one of the 36,288 normalized
top bands in the band table, and the remaining six rows are completed with randomized backtracking.

```bash
nby9 build-cache                       # weighted, 1,000,000 grids
nby9 build-cache --type u --seed 42    # unweighted
```

- `--type weighted` (`w`, the default) picks each top band in proportion to its number of completions.
- `--type unweighted` (`u`) picks each top band with equal probability.
- `--count` (default 1,000,000), `--seed`, `-o/--output`, `--pattern-file` and `--progress-every` are also available.

Caches are saved in the data folder unless you pass `-o`. A one-million-grid cache is about 78 MB.

### `nby9 compare`

Compares two `.npy` grid datasets (shape `(N, 81)` or `(N, 9, 9)`) with each other and with the reference distribution
from the band table:

```bash
nby9 compare backtracking.npy vectorized.npy
```

Each grid's top band is normalized (digits relabeled so the first box reads 1–9, then the second and third boxes are
reduced) and mapped to one of the 36,288 buckets. The report gives, for each sample, the grid count, duplicate grids,
buckets observed, entropy, KL divergence, Jensen–Shannon divergence and total-variation distance from the reference,
and then the same divergences between the two samples.

These metrics describe the top-band distribution only, not the probability of complete grids. Exact-grid entropy is not
reported, because with samples of a million grids it mostly reflects sample size.

It also compares two JSON weight distributions (lists in matching order, or objects keyed by category):

```bash
nby9 compare observed.json target.json
```

### `nby9 equivalents`

Reads a JSON list of solved grids and writes random symmetry-equivalent copies of each one, using the same
transformations as the vectorized method. Grids are flat lists of 81 digits in row order, one per line, in the same
layout as `generate --format json`; 9×9 nested lists are also accepted as input.
[`examples/grids.json`](https://github.com/nby9/fast-training-data-generator/blob/main/examples/grids.json) holds one
example grid.

```bash
nby9 equivalents grids.json --count 5 --seed 42 -o copies.json
```

Without `-o`, the copies are written to standard output.

### `nby9 fetch-data`

Downloads the band table now instead of on first use, for example before going offline. `--from FILE` installs a copy
you already have (`results2long.gz` or `results2long.txt`), and `--force` replaces an existing copy. Either way, the file is checked against its known SHA-256
hash before it is kept.

## Python API

```python
from fast_sudoku_generator import generate_grids

grids = generate_grids(1000, seed=42)  # backtracking, shape (1000, 81), dtype uint8
grids = generate_grids(1_000_000, method="vectorized")  # needs `nby9 build-cache` once
grid = grids[0].reshape(9, 9)
```

`generate_grids(count, method="backtracking", seed=None, grid_file=None)` returns grids as rows of 81 digits in row
order. For the same method, count, seed and cache, it returns the same grids as `nby9 generate`. A missing cache
raises `fast_sudoku_generator.DataError`.

## Data files

The package keeps two kinds of file in a per-user data folder:

| Platform | Default folder |
| --- | --- |
| macOS | `~/Library/Caches/fast-sudoku-generator` |
| Linux | `~/.cache/fast-sudoku-generator` (or `$XDG_CACHE_HOME/fast-sudoku-generator`) |
| Windows | `%LOCALAPPDATA%\fast-sudoku-generator` |

- **`v36288.txt`**, the band table. It is not bundled with the package. On first use the gzipped version,
  `results2long.gz` on [Frazer Jarvis's page](https://www.afjarvis.org.uk/sudoku/bertram.html), is downloaded,
  decompressed and checked against a known SHA-256 hash.
- **`cached-grids.npy`** and **`cached-grids-unweighted.npy`**, the base-grid caches built by `nby9 build-cache`.

Two environment variables change this:

- `FAST_SUDOKU_DATA_DIR`: use this folder instead.
- `FAST_SUDOKU_PATTERN_FILE`: use this band table instead of the downloaded one. It is not checksum-checked, so it can
  point at a table you generated yourself.

**Offline machines:** download `results2long.gz` by hand from the page above, copy it over, and run
`nby9 fetch-data --from results2long.gz`. A decompressed `results2long.txt` works too.

## How the methods work

### Backtracking

The search fills the grid one cell at a time. Row, column and box usage are tracked as 9-bit masks. At each step the
search picks a cell with the fewest legal digits (the minimum-remaining-values heuristic), breaking ties at random,
tries its candidates in random order and backtracks on a dead end.

### Vectorized

The base-grid cache is read with a memory map and walked through in a random order. Each base grid is transformed with
a random combination of the validity-preserving Sudoku symmetries:

- permuting the three bands, and the three rows within each band;
- permuting the three stacks, and the three columns within each stack;
- transposing the grid (with probability 0.5);
- relabeling the digits 1–9.

All of this is done with NumPy indexing on batches of up to 65,536 grids. If you ask for more grids than the cache
holds, the cache is reshuffled and reused, and the random transformations keep the outputs distinct.

### Why sample from the band table

The goal is a uniform sample: every one of the 6,670,903,752,021,072,936,960 valid grids equally likely. Plain
randomized backtracking does not give that, because the order of the search favors some grids over others. In a uniform
sample, a top band appears in proportion to how many ways it can be completed. Drawing the top band with the
completion-count weights from the band table therefore matches that part of the target distribution, and the symmetry
transformations used by the vectorized method map each grid to another valid grid without favoring any of them.

This is an imperfect proxy for uniform sampling:

- Only the top band is weighted correctly. The other six rows are filled by randomized backtracking, which does not pick
  uniformly among the completions of that band.
- The vectorized method can only produce symmetry variants of grids in the cache. A one-million-grid cache covers at
  most a million of the 5,472,730,538 essentially different grids, however many outputs you generate.
- `nby9 compare` checks the top-band distribution only, so it can show bias in the top band but not in the rest
  of the grid.

### Performance

Times for generating 100,000 grids to `.npy` with `--seed 1`:

| Method | Generation time | Rate |
| --- | ---: | ---: |
| Backtracking | 77.9 s | about 1,300 grids/s |
| Vectorized | 0.46 s | about 220,000 grids/s |

The vectorized method was about 170 times faster. These times come from one run on a 4-core arm64 Linux VM with Python
3.10 and NumPy 2.2, using the default one-million-grid weighted cache. They do not include the one-time cache build,
which takes about 0.4 ms per grid (roughly 7 minutes for the default million). Your numbers will vary with hardware.

## The band table

The band table comes from Bertram Felgenhauer and Frazer Jarvis's 2005 enumeration of Sudoku grids, published in
*Mathematics of Sudoku I* (Felgenhauer and Jarvis, 2006). It was computed by brute force with Bertram Felgenhauer's
programs and is published on [Frazer Jarvis's Sudoku pages](https://www.afjarvis.org.uk/sudoku/bertram.html) as
`results2long.txt` (and gzipped as `results2long.gz`). Credit for the data goes to the authors. This package does not redistribute it; it downloads it
from the authors' page when first needed.

**What it represents.** Fix the top-left box as `123 / 456 / 789`. There are 2,612,736 ways to fill the rest of the top
band. Sorting the columns of boxes 2 and 3 by their top-row entries and putting the two boxes in order of their first
entries does not change the number of completions, and it groups those fillings into 36,288 classes of 72. Each line of
the file is one class with its completion count. The counts take only 44 distinct values, which correspond to
Felgenhauer and Jarvis's 44 equivalence classes. As a check, the sum of all counts × 72² × 9! gives exactly the total
number of grids above.

**How to read it.** Each line has this form:

```
[456789,789123,123456] => 108374976
```

The three six-digit groups are columns 4–9 of rows 1, 2 and 3. Columns 1–3 are always `123`, `456` and `789`, so this
line is the top band

```
1 2 3 | 4 5 6 | 7 8 9
4 5 6 | 7 8 9 | 1 2 3
7 8 9 | 1 2 3 | 4 5 6
```

and the number after `=>` (108,374,976) is its completion count, which is the weight used for sampling. The file ends
with a DOS end-of-file character (`0x1A`), which the loader skips.

## Development

```bash
git clone https://github.com/nby9/fast-training-data-generator.git
cd fast-training-data-generator
uv sync                          # creates .venv with the package (editable) and dev tools
uv run nby9 fetch-data    # the band table, needed by some tests
uv run pytest
```

Dependencies live in `pyproject.toml` and exact versions are locked in `uv.lock`. The `dev` group adds `ruff`,
`pytest`, `pre-commit` and `mdformat`.

### Tests

The suite in `tests/` runs in about a second and never touches the network or your real data folder: every test runs
with `FAST_SUDOKU_DATA_DIR` pointing at a temporary folder and builds its own small cache. Tests that need the real band
table use an already downloaded copy; without one they are skipped locally and fail on CI. The suite covers:

- both generation methods: every grid is valid, seeds reproduce output, and the vectorized method handles batches and
  counts larger than its cache;
- the Python API, including that it matches the command line;
- the band-table loader, including a check that the counts reproduce the published total number of grids;
- downloading and verifying the band table (with local files only), data folders and environment variables;
- top-band completion and cache building;
- the distribution metrics and top-band normalization;
- symmetry copies;
- the command line: output formats, the 10-grid terminal limit, `--force`, errors and `--version`.

### Pre-commit hooks

`.pre-commit-config.yaml` runs three checks before every commit: `ruff check` (lint), `ruff format` (formatting) and the
pytest suite. Set them up once per clone:

```bash
uv sync
uv run pre-commit install
```

After that, `git commit` stops if any check fails. `ruff format` rewrites badly formatted files itself, so stage the
changes and commit again. To run every hook without committing:

```bash
uv run pre-commit run --all-files
```

### Continuous integration and releases

GitHub Actions runs `.github/workflows/tests.yml` on every push to `main` or `scripts` and on every pull request:

- the pre-commit lint and format hooks;
- the test suite on Python 3.10–3.13, with the band table downloaded once and cached;
- a build of the wheel, a check that it does not contain the band table, and a smoke test of the installed wheel in a
  clean environment.

`.github/workflows/publish.yml` publishes with PyPI Trusted Publishing, so no API tokens are stored. Running it by hand
publishes to TestPyPI. Publishing a GitHub release tagged `vX.Y.Z` (matching the version in `pyproject.toml`) publishes
to PyPI.

## Script version

The original standalone scripts, with no install step, live on the
[`scripts` branch](https://github.com/nby9/fast-training-data-generator/tree/scripts) and in the
[`scripts-v0.1.0` release](https://github.com/nby9/fast-training-data-generator/releases/tag/scripts-v0.1.0).

## License

The code is released under the
[MIT License](https://github.com/nby9/fast-training-data-generator/blob/main/LICENSE). The band table is Bertram
Felgenhauer and Frazer Jarvis's published data; it is not part of this package and is not covered by its license.
