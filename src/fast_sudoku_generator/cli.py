"""The ``nby9`` command line."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from fast_sudoku_generator import __version__, cache, compare, equivalents, generate
from fast_sudoku_generator.data import DataError, fetch_pattern_file

COMMANDS = (
    ("generate", generate, "generate completed Sudoku grids"),
    ("build-cache", cache, "build the base-grid cache used by the vectorized method"),
    ("compare", compare, "compare two grid datasets or two JSON distributions"),
    ("equivalents", equivalents, "write random symmetry copies of grids in a JSON file"),
)


def _configure_fetch_data(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--from",
        dest="source",
        metavar="FILE_OR_URL",
        help="use a local copy of results2long.txt (or another URL) instead of downloading",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing copy")


def _run_fetch_data(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    path = fetch_pattern_file(source=args.source, force=args.force)
    print(path)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nby9",
        description="Generate completed 9x9 Sudoku grids quickly and measure how samples are distributed.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module, help_text in COMMANDS:
        sub = subparsers.add_parser(
            name,
            help=help_text,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.configure(sub)
        sub.set_defaults(_run=module.run, _parser=sub)
    sub = subparsers.add_parser(
        "fetch-data",
        help="download the band table now (normally done on first use)",
        description="Download the Felgenhauer-Jarvis band table into the data folder and verify its checksum.",
    )
    _configure_fetch_data(sub)
    sub.set_defaults(_run=_run_fetch_data, _parser=sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args._run(args, args._parser)
    except DataError as exc:
        print(f"nby9 {args.command}: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
