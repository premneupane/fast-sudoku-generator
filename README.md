# fast-sudoku-generator

Generate large numbers of completed 9×9 Sudoku grids quickly, and measure how the samples are distributed.

The generator has two methods:

- **Backtracking (`b`)** builds each grid from scratch with a randomized depth-first search. It is simple and needs no
  setup, but it is slow for large batches.
- **Vectorized (`v`)** samples base grids from a prebuilt cache and applies random Sudoku symmetries to whole batches at
  once with NumPy. It is much faster for large batches.

The `utils/` folder holds the tools that build the cache for the vectorized method and compare the distributions of
generated datasets.

## Requirements

- [uv](https://docs.astral.sh/uv/)
- Python 3.10 or later (the code uses `int.bit_count()`). uv installs the version pinned in `.python-version` (3.12.3)
  if you don't have it.

Dependencies live in `pyproject.toml` and exact versions are locked in `uv.lock`. The only runtime dependency is NumPy;
the `dev` group adds `ruff`, `pytest`, `pre-commit` and `mdformat`.

```bash
uv sync                      # creates .venv with the runtime and dev dependencies
source .venv/bin/activate    # optional: lets you run the python3 commands below directly
```

Without activating, prefix any command with `uv run`, for example `uv run python sudoku_generator.py -n 5`.

## Quick start

```bash
# One grid, printed to the terminal
python3 sudoku_generator.py

# Five reproducible grids as JSON
python3 sudoku_generator.py -n 5 --format json --seed 42

# A thousand grids to a text file, one grid per line
python3 sudoku_generator.py -n 1000 -o grids.txt

# The vectorized method needs a grid cache first (one-time step)
python3 utils/generate_cached_grids.py --type weighted

# One million grids saved as a NumPy array
python3 sudoku_generator.py -n 1000000 --method v --format npy
```

Timing information is printed to stderr, so stdout stays clean for piping.

To keep the terminal readable, text and JSON output to a terminal is limited to 10 grids. Larger requests stop with an
error that suggests `-o FILE` or `--format npy`. Redirected or piped output (`> grids.txt`, `| head`) is not limited,
and `--force` lifts the limit.

## `sudoku_generator.py`

| Option | Description |
| --- | --- |
| `-n`, `--count` | Number of grids to generate (default: 1). |
| `--method` | `backtracking` / `b` (default) or `vectorized` / `v`. |
| `--grid-file` | Grid cache for the vectorized method (default: `data/cached-grids.npy`). |
| `--seed` | Seed for repeatable output. |
| `--format` | `text` (default), `json` or `npy`. |
| `-o`, `--output` | Output file. Text and JSON default to stdout. `npy` defaults to `data/backtracking.npy` or `data/vectorized.npy`. |
| `--force` | Allow more than 10 grids to print to the terminal. |

Output formats:

- **text**: in a terminal, grids are laid out in 3×3 boxes and numbered when there is more than one. Written to a file
  or a pipe, each grid is one line holding a list of its 81 digits in row order, for example `[9, 7, 8, ..., 1, 2]`.
- **json**: each grid is a flat list of 81 digits in row order. One grid is written as a bare list; more than one is
  written as a JSON array with one grid per line.
- **npy**: a `(count, 81)` `uint8` array, written through a memory map so large datasets do not have to fit in memory.

## How the methods work

### Backtracking

`src/backtracking.py` fills the grid one cell at a time. Row, column and box usage are tracked as 9-bit masks. At each
step the search picks a cell with the fewest legal digits (the minimum-remaining-values heuristic), breaking ties at
random, tries its candidates in random order and backtracks on a dead end.

### Vectorized

`src/vectorized.py` reads the base-grid cache with a memory map and walks through it in a random order. Each base grid
is transformed with a random combination of the validity-preserving Sudoku symmetries:

- permuting the three bands, and the three rows within each band;
- permuting the three stacks, and the three columns within each stack;
- transposing the grid (with probability 0.5);
- relabeling the digits 1–9.

All of this is done with NumPy indexing on batches of up to 65,536 grids. If you ask for more grids than the cache
holds, the cache is reshuffled and reused, and the random transformations keep the outputs distinct.

### Performance

Times for generating 100,000 grids to `.npy` with `--seed 1`:

| Method | Generation time | Rate |
| --- | ---: | ---: |
| Backtracking | 77.9 s | about 1,300 grids/s |
| Vectorized | 0.46 s | about 220,000 grids/s |

The vectorized method was about 170 times faster. These times come from one run on a 4-core arm64 Linux VM with Python
3.10 and NumPy 2.2, using the default one-million-grid weighted cache. They do not include the one-time cache build,
which takes about 0.4 ms per grid (roughly 7 minutes for the default million). Your numbers will vary with hardware.

## Utilities

### `utils/generate_cached_grids.py`: build the base-grid cache

Builds the cache that the vectorized method draws from. Each grid starts from one of the 36,288 normalized top-band
patterns in `data/v36288.txt`, and the remaining six rows are completed with randomized backtracking.

```bash
python3 utils/generate_cached_grids.py --type weighted            # -> data/cached-grids.npy
python3 utils/generate_cached_grids.py --type unweighted --seed 42 # -> data/cached-grids-unweighted.npy
```

- `--type weighted` (`w`, the default) picks each top band in proportion to its number of completions, the weight listed
  for it in `v36288.txt`.
- `--type unweighted` (`u`) picks each top band with equal probability.
- `--count` (default 1,000,000), `--seed`, `-o/--output`, `--pattern-file` and `--progress-every` are also available.

A one-million-grid cache is about 78 MB.

#### Why sample from `v36288.txt`

The goal is a uniform sample: every one of the 6,670,903,752,021,072,936,960 valid grids equally likely. Plain randomized
backtracking does not give that, because the order of the search favors some grids over others. In a uniform sample, a
top band appears in proportion to how many ways it can be completed. Drawing the top band with the completion-count
weights from `v36288.txt` therefore matches that part of the target distribution, and the symmetry transformations used
by the vectorized method map each grid to another valid grid without favoring any of them.

This is an imperfect proxy for uniform sampling:

- Only the top band is weighted correctly. The other six rows are filled by randomized backtracking, which does not pick
  uniformly among the completions of that band.
- The vectorized method can only produce symmetry variants of grids in the cache. A one-million-grid cache covers at
  most a million of the 5,472,730,538 essentially different grids, however many outputs you generate.
- `compare_distributions.py` checks the top-band distribution only, so it can show bias in the top band but not in the
  rest of the grid.

### `utils/compare_distributions.py`: compare samples

Compares two `.npy` grid datasets (shape `(N, 81)` or `(N, 9, 9)`) with each other and with the reference distribution
from `v36288.txt`:

```bash
python3 utils/compare_distributions.py data/backtracking.npy data/vectorized.npy
```

Each grid's top band is normalized (digits relabeled so the first box reads 1–9, then the second and third boxes are
reduced) and mapped to one of the 36,288 buckets. The report gives, for each sample, the grid count, duplicate grids,
buckets observed, entropy, KL divergence, Jensen–Shannon divergence and total-variation distance from the reference,
and then the same divergences between the two samples.

These metrics describe the top-band distribution only, not the probability of complete grids. Exact-grid entropy is not
reported, because with samples of a million grids it mostly reflects sample size.

The script also compares two JSON weight distributions (lists in matching order, or objects keyed by category):

```bash
python3 utils/compare_distributions.py observed.json target.json
```

### `utils/generate_equivalent_grids.py`: symmetry copies of a grid list

Reads a JSON list of solved grids and writes random symmetry-equivalent copies of each one, using the same
transformations as the vectorized method in pure Python.

Grids are flat lists of 81 digits in row order, one per line, in the same layout as `--format json`. Input grids
written as 9×9 nested lists are also accepted. `data/grids.json` holds a single placeholder grid so the default command
runs; replace it with your own grids.

```bash
python3 utils/generate_equivalent_grids.py --input data/grids.json --output data/equiv_grids.json --count 5 --seed 42
```

## Data source: `data/v36288.txt`

`v36288.txt` comes from Bertram Felgenhauer and Frazer Jarvis's 2005 enumeration of Sudoku grids, published in
*Mathematics of Sudoku I* (Felgenhauer and Jarvis, 2006). The table was computed by brute force and is available from
Frazer Jarvis's Sudoku pages at <http://www.afjarvis.staff.shef.ac.uk/sudoku/bertram>. Credit for the data goes to the
authors.

**What it represents.** Fix the top-left box as `123 / 456 / 789`. There are 2,612,736 ways to fill the rest of the top
band. Sorting the columns of boxes 2 and 3 by their top-row entries and putting the two boxes in order of their first
entries does not change the number of completions, and it groups those fillings into 36,288 classes of 72. Each line of
the file is one class with its completion count. The counts take only 44 distinct values, which correspond to Felgenhauer
and Jarvis's 44 equivalence classes. As a check, the sum of all counts × 72² × 9! gives exactly the total number of
grids above.

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
with a DOS end-of-file character (`0x1A`), which `src/shared_helper.py` skips.

## Tests

```bash
uv run pytest
```

The suite in `tests/` runs in under a second and needs none of the large `.npy` files. It builds its own 50-grid cache
in a temporary folder. It covers:

- both generation engines: every grid is valid, seeds reproduce output, and the vectorized method handles batches and
  counts larger than its cache;
- the `v36288.txt` loader, including a check that the counts reproduce the published total number of grids;
- top-band completion and cache building;
- the distribution metrics and top-band normalization;
- the symmetry-copy utility and the `data/grids.json` placeholder;
- the command line: output formats, the 10-grid terminal limit, `--force` and argument errors.

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

### Continuous integration

GitHub Actions (`.github/workflows/tests.yml`) installs the locked dependencies with uv, runs the same pre-commit hooks
for lint and formatting, and runs the test suite on Python 3.10–3.13, for every push to `main` and every pull request.

## Repository layout

```
sudoku_generator.py               command-line entry point
src/
  backtracking.py                 randomized MRV backtracking engine
  vectorized.py                   batched symmetry sampling from a grid cache
  shared_helper.py                bitmask constant and v36288 table loader
utils/
  generate_cached_grids.py        builds the base-grid cache
  compare_distributions.py        distribution metrics for grid datasets
  generate_equivalent_grids.py    symmetry copies of grids in a JSON file
tests/                            pytest suite (no large data needed)
.github/workflows/tests.yml       CI: pre-commit lint/format and tests on Python 3.10-3.13
.pre-commit-config.yaml           local hooks: ruff check, ruff format, pytest
pyproject.toml                    dependencies and tool settings (pytest, ruff, mdformat)
uv.lock                           exact dependency versions, managed by uv
data/
  v36288.txt                      36,288 top-band patterns with completion counts (committed)
  grids.json                      one placeholder grid, input for generate_equivalent_grids.py (committed)
  *.npy                           generated datasets and caches (git-ignored)
```

## Notes

- Grids are always valid completed Sudokus. Neither method produces puzzles (grids with cells removed).
- Use the same `--seed` with the same method, count and cache to reproduce output exactly.
- Generated `.npy` files are large and git-ignored. Rebuild them with the commands above.

## License

The code is released under the [MIT License](LICENSE). `data/v36288.txt` is Bertram Felgenhauer and Frazer Jarvis's
published enumeration data, included with credit (see [Data source](#data-source-datav36288txt)); the MIT license covers
this project's code, not their data.
