"""Compare two grid datasets, or two probability distributions in JSON.

For JSON inputs, each file contains either a list of weights in matching
category order or an object mapping category names to weights.

For two .npy Sudoku datasets, the command compares empirical distributions of
normalized top-band buckets against each other and against the completion-count
reference distribution from the Felgenhauer-Jarvis band table. Exact-grid entropy is intentionally
not reported: with million-grid samples it is dominated by sample size.

Examples:
    fast-sudoku compare observed.json target.json
    fast-sudoku compare sample1.npy sample2.npy
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from fast_sudoku_generator.data import pattern_file
from fast_sudoku_generator.patterns import PatternTable, load_pattern_table

Distribution = Sequence[float] | Mapping[str, float]


def _aligned_values(p: Distribution, q: Distribution) -> tuple[list[float], list[float]]:
    """Align mapping categories or validate that sequence categories line up."""
    if isinstance(p, Mapping) and isinstance(q, Mapping):
        keys = list(p)
        keys.extend(key for key in q if key not in p)
        return [float(p.get(key, 0.0)) for key in keys], [float(q.get(key, 0.0)) for key in keys]
    if isinstance(p, Mapping) or isinstance(q, Mapping):
        raise TypeError("both distributions must be mappings or both must be sequences")
    p_values, q_values = [float(value) for value in p], [float(value) for value in q]
    if len(p_values) != len(q_values):
        raise ValueError("sequence distributions must have the same length and category order")
    return p_values, q_values


def _normalize(values: Sequence[float], name: str) -> list[float]:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError(f"{name} must contain finite, nonnegative values")
    total = sum(values)
    if total <= 0:
        raise ValueError(f"{name} must have positive total weight")
    return [value / total for value in values]


def entropy_bits(probabilities: Sequence[float]) -> float:
    """Return Shannon entropy in bits; probabilities should sum to one."""
    return -sum(probability * math.log2(probability) for probability in probabilities if probability)


def kl_divergence_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """Return D_KL(p || q) in bits; infinity if q omits p's support."""
    if any(pv > 0 and qv == 0 for pv, qv in zip(p, q)):
        return math.inf
    return sum(pv * math.log2(pv / qv) for pv, qv in zip(p, q) if pv > 0)


def total_variation(p: Sequence[float], q: Sequence[float]) -> float:
    """Return total-variation distance, one half of the L1 distance."""
    return 0.5 * sum(abs(pv - qv) for pv, qv in zip(p, q))


def jensen_shannon_divergence_bits(p: Sequence[float], q: Sequence[float]) -> float:
    """Return the finite, symmetric Jensen-Shannon divergence in bits."""
    midpoint = [(pv + qv) / 2 for pv, qv in zip(p, q)]
    return 0.5 * kl_divergence_bits(p, midpoint) + 0.5 * kl_divergence_bits(q, midpoint)


def compare_distributions(p: Distribution, q: Distribution) -> dict[str, float]:
    """Normalize two weight vectors and return entropy, KL, TV, and JS metrics."""
    p_values, q_values = _aligned_values(p, q)
    p_probabilities = _normalize(p_values, "p")
    q_probabilities = _normalize(q_values, "q")
    return {
        "entropy_p_bits": entropy_bits(p_probabilities),
        "entropy_q_bits": entropy_bits(q_probabilities),
        "kl_p_to_q_bits": kl_divergence_bits(p_probabilities, q_probabilities),
        "kl_q_to_p_bits": kl_divergence_bits(q_probabilities, p_probabilities),
        "total_variation": total_variation(p_probabilities, q_probabilities),
        "js_divergence_bits": jensen_shannon_divergence_bits(p_probabilities, q_probabilities),
    }


def _encode_pattern(pattern: Sequence[str]) -> int:
    """Encode 18 base-9 digits as one integer for fast bucket lookup."""
    code = 0
    for row in pattern:
        for digit in row:
            code = code * 9 + int(digit) - 1
    return code


def _normalized_band_codes(grids: np.ndarray) -> np.ndarray:
    """Normalize B1, then lexicographically reduce B2/B3 as in v36288."""
    count = len(grids)
    top_band = grids[:, :3, :]
    first_box = top_band[:, :, :3].reshape(count, 9)

    # Relabel the digits so the upper-left box reads 1..9 in row-major order.
    digit_labels = np.zeros((count, 10), dtype=np.uint8)
    row_indices = np.arange(count)[:, None]
    digit_labels[row_indices, first_box] = np.arange(1, 10, dtype=np.uint8)[None, :]
    normalized = digit_labels[np.arange(count)[:, None, None], top_band]

    # Sort columns within B2 and B3 by their top-row entries, then order the
    # two boxes by their first entries. These 72 reductions identify a v36288 bucket.
    order_b2 = np.argsort(normalized[:, 0, 3:6], axis=1)
    order_b3 = np.argsort(normalized[:, 0, 6:9], axis=1)
    b2 = np.take_along_axis(normalized[:, :, 3:6], order_b2[:, None, :], axis=2)
    b3 = np.take_along_axis(normalized[:, :, 6:9], order_b3[:, None, :], axis=2)
    swap = b2[:, 0, 0] > b3[:, 0, 0]
    left_box = b2.copy()
    right_box = b3.copy()
    left_box[swap] = b3[swap]
    right_box[swap] = b2[swap]
    suffixes = np.concatenate((left_box, right_box), axis=2).reshape(count, 18)

    powers = np.asarray([9**power for power in range(17, -1, -1)], dtype=np.uint64)
    return (suffixes.astype(np.uint64) - 1) @ powers


def grid_pattern_counts(path: Path, table: PatternTable, batch_size: int = 50_000) -> tuple[np.ndarray, int]:
    """Count normalized top-band buckets from an (N,81) or (N,9,9) .npy dataset."""
    grids = np.load(path, mmap_mode="r")
    if grids.ndim == 2 and grids.shape[1] == 81:
        grid_count = len(grids)
    elif grids.ndim == 3 and grids.shape[1:] == (9, 9):
        grid_count = len(grids)
    else:
        raise ValueError(f"expected {path} to contain an (N,81) or (N,9,9) grid array")
    if grid_count == 0:
        raise ValueError(f"grid dataset is empty: {path}")

    code_to_index = {_encode_pattern(pattern): index for index, pattern in enumerate(table.patterns)}
    counts = np.zeros(len(table.patterns), dtype=np.int64)
    for start in range(0, grid_count, batch_size):
        batch = np.asarray(grids[start : start + batch_size]).reshape(-1, 9, 9)
        if np.any(batch < 1) or np.any(batch > 9):
            raise ValueError(f"grid dataset contains digits outside 1..9: {path}")
        codes, frequencies = np.unique(_normalized_band_codes(batch), return_counts=True)
        for code, frequency in zip(codes, frequencies):
            index = code_to_index.get(int(code))
            if index is None:
                raise ValueError(f"normalized top band is absent from the v36288 lookup: {path}")
            counts[index] += int(frequency)
    return counts, grid_count


def duplicate_grid_count(path: Path) -> int:
    """Count repeated grid rows beyond the first copy of each distinct grid."""
    grids = np.load(path, mmap_mode="r")
    if grids.ndim == 2 and grids.shape[1] == 81:
        grid_count = len(grids)
    elif grids.ndim == 3 and grids.shape[1:] == (9, 9):
        grid_count = len(grids)
    else:
        raise ValueError(f"expected {path} to contain an (N,81) or (N,9,9) grid array")
    if grid_count == 0:
        raise ValueError(f"grid dataset is empty: {path}")
    unique_count = len(np.unique(grids.reshape(grid_count, 81), axis=0))
    return grid_count - unique_count


def _display(value: float) -> str:
    return "infinity" if math.isinf(value) else f"{value:.4f}"


def _compare_grid_files(p_path: Path, q_path: Path, pattern_path: Path) -> None:
    table = load_pattern_table(pattern_path)
    if len(table.patterns) != 36_288:
        raise ValueError("grid comparison requires the full v36288 pattern table")
    p_counts, p_size = grid_pattern_counts(p_path, table)
    q_counts, q_size = grid_pattern_counts(q_path, table)
    p_duplicates = duplicate_grid_count(p_path)
    q_duplicates = duplicate_grid_count(q_path)
    target = _normalize(table.weights, "reference")
    p_metrics = compare_distributions(p_counts, table.weights)
    q_metrics = compare_distributions(q_counts, table.weights)
    between = compare_distributions(p_counts, q_counts)
    target_entropy = entropy_bits(target)

    print("Top-band bucket comparison")
    print("  Buckets: 36,288 (normalized using the v36288 lookup)")
    print(f"  Reference entropy: {target_entropy:.2f} bits")
    print(f"  Sample 1: {p_path}")
    print(f"  Sample 2: {q_path}")
    print()

    rows = (
        ("Grids", f"{p_size:,}", f"{q_size:,}"),
        ("Duplicate grids", f"{p_duplicates:,}", f"{q_duplicates:,}"),
        ("Observed buckets", f"{np.count_nonzero(p_counts):,}", f"{np.count_nonzero(q_counts):,}"),
        ("Entropy (H), bits", f"{p_metrics['entropy_p_bits']:.2f}", f"{q_metrics['entropy_p_bits']:.2f}"),
        ("KL(sample || reference), bits", f"{p_metrics['kl_p_to_q_bits']:.4f}", f"{q_metrics['kl_p_to_q_bits']:.4f}"),
        (
            "Jensen-Shannon(sample, reference), bits",
            f"{p_metrics['js_divergence_bits']:.4f}",
            f"{q_metrics['js_divergence_bits']:.4f}",
        ),
        (
            "TV(sample, reference)",
            f"{100 * p_metrics['total_variation']:.2f}%",
            f"{100 * q_metrics['total_variation']:.2f}%",
        ),
    )
    label_width = max(38, max(len(label) for label, _, _ in rows))
    value_width = 20
    column_gap = "  "
    print(f"{'Measure':<{label_width}}{column_gap}{'Sample 1':>{value_width}}{column_gap}{'Sample 2':>{value_width}}")
    print("-" * (label_width + len(column_gap) * 2 + 2 * value_width))
    for label, p_value, q_value in rows:
        print(f"{label:<{label_width}}{column_gap}{p_value:>{value_width}}{column_gap}{q_value:>{value_width}}")

    print()
    print("Between-sample comparison")
    print("-" * 56)
    between_rows = (
        ("KL(Sample 1 || Sample 2), bits", _display(between["kl_p_to_q_bits"])),
        ("KL(Sample 2 || Sample 1), bits", _display(between["kl_q_to_p_bits"])),
        ("Jensen-Shannon divergence, bits", f"{between['js_divergence_bits']:.4f}"),
        ("Total variation distance", f"{100 * between['total_variation']:.2f}%"),
    )
    for label, value in between_rows:
        print(f"{label:<42}{value:>14}")

    print()
    print("Scope: these metrics compare normalized top-band buckets, not complete-grid probabilities.")
    print()


def _compare_json_files(p_path: Path, q_path: Path) -> None:
    with p_path.open(encoding="utf-8") as file_p:
        p = json.load(file_p)
    with q_path.open(encoding="utf-8") as file_q:
        q = json.load(file_q)
    metrics = compare_distributions(p, q)

    print(f"Sample 1: {p_path}")
    print(f"Sample 2: {q_path}")
    print()
    print("Entropy")
    print(f"  Sample 1: {metrics['entropy_p_bits']:.2f} bits")
    print(f"  Sample 2: {metrics['entropy_q_bits']:.2f} bits")
    print()
    print("Between-sample comparison")
    print(f"  KL(Sample 1 || Sample 2): {_display(metrics['kl_p_to_q_bits'])} bits")
    print(f"  KL(Sample 2 || Sample 1): {_display(metrics['kl_q_to_p_bits'])} bits")
    print(f"  Jensen-Shannon divergence: {metrics['js_divergence_bits']:.4f} bits")
    print(f"  Total variation distance: {100 * metrics['total_variation']:.2f}%")
    print()


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("p", type=Path, help="first JSON distribution or .npy grid dataset")
    parser.add_argument("q", type=Path, help="second JSON distribution or .npy grid dataset")
    parser.add_argument(
        "--pattern-file",
        type=Path,
        help="band table used as the reference for .npy datasets (default: the downloaded table)",
    )


def run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    p_is_npy = args.p.suffix.lower() == ".npy"
    q_is_npy = args.q.suffix.lower() == ".npy"
    if p_is_npy != q_is_npy:
        parser.error("both inputs must be .npy grid datasets or both must be JSON distributions")
    print()
    if p_is_npy:
        _compare_grid_files(args.p, args.q, args.pattern_file or pattern_file())
    else:
        _compare_json_files(args.p, args.q)
    return 0
