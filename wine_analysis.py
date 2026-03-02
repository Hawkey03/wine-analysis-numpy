"""
Wine CSV analysis (NumPy)
Loads a classic wine dataset from a CSV file and demonstrates:

- parsing CSV into NumPy arrays
- slicing (classes vs attributes)
- basic statistics (axis-aware)
- sorting features and keeping alignment
- standardization (zero mean, unit variance)
- row-wise dot products and cosine similarity (NumPy-only)

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass(frozen=True)
class WineData:
    names: np.ndarray        # (13,) strings
    classes: np.ndarray      # (n,) ints
    attributes: np.ndarray   # (n, 13) floats


def load_wine_csv(csv_path: Path) -> WineData:
    """Load wine.csv where col0 is class label and col1.. are numeric features."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open() as f:
        header = f.readline().strip().split(",")

        # Basic sanity check
        if len(header) < 2:
            raise ValueError("Header looks too short. Expected class + features.")

        rows: list[list[float]] = []
        for line in f:
            if not line.strip():
                continue
            rows.append([float(x) for x in line.strip().split(",")])

    data = np.array(rows, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D numeric data, got shape {data.shape}")

    names = np.array(header[1:], dtype=str)
    classes = data[:, 0].astype(int)
    attributes = data[:, 1:].astype(float)

    # Shape checks (these are the kind of asserts people like to see)
    if names.shape != (13,):
        raise ValueError(f"Expected 13 feature names, got {names.shape}")
    if attributes.shape[1] != 13:
        raise ValueError(f"Expected 13 feature columns, got {attributes.shape}")

    return WineData(names=names, classes=classes, attributes=attributes)


def sort_features(names: np.ndarray, attributes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sort feature names alphabetically and reorder attribute columns accordingly."""
    idx = np.argsort(names)
    return names[idx], attributes[:, idx]


def standardize(attributes: np.ndarray) -> np.ndarray:
    """Z-score standardization per feature (column-wise)."""
    mu = attributes.mean(axis=0)
    sigma = attributes.std(axis=0)
    if np.any(sigma == 0):
        raise ValueError("At least one feature has zero std; cannot standardize.")
    return (attributes - mu) / sigma


def rowwise_dot(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dot product for corresponding rows, without np.dot or loops."""
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    return (a * b).sum(axis=1)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity using NumPy only."""
    denom = np.sqrt((x * x).sum()) * np.sqrt((y * y).sum())
    if denom == 0:
        raise ValueError("Zero vector encountered; cosine similarity undefined.")
    return float((x * y).sum() / denom)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("wine.csv"))
    args = parser.parse_args()

    wine = load_wine_csv(args.csv)

    # Basic stats
    total_sum = wine.attributes.sum()
    col_means = wine.attributes.mean(axis=0)
    max_of_row_mins = wine.attributes.min(axis=1).max()
    avg_of_col_max = wine.attributes.max(axis=0).mean()

    # Transpose stats (demonstrates axis understanding)
    attrs_T = wine.attributes.T
    col_means_via_T = attrs_T.mean(axis=1)  # same as mean(axis=0) on original
    max_of_row_mins_via_T = attrs_T.min(axis=1).max()

    # Sorting with alignment
    names_sorted, attrs_sorted = sort_features(wine.names, wine.attributes)

    # Standardization
    attrs_norm = standardize(attrs_sorted)

    # Matrix ops
    slice1 = attrs_sorted[6:10]
    slice2 = attrs_sorted[76:80]
    dots = rowwise_dot(slice1, slice2)

    cos_raw = cosine_similarity(attrs_sorted[7], attrs_sorted[77])
    cos_norm = cosine_similarity(attrs_norm[7], attrs_norm[77])

    # Report (short, “exec summary” style)
    np.set_printoptions(suppress=True, precision=4)
    print("Wine dataset loaded")
    print(f"- n_samples: {wine.attributes.shape[0]}")
    print(f"- n_features: {wine.attributes.shape[1]}")
    print()
    print("Statistics")
    print(f"- sum(all attributes): {total_sum:.4f}")
    print(f"- max(row minimums): {max_of_row_mins:.4f}")
    print(f"- mean(column maximums): {avg_of_col_max:.4f}")
    print()
    print("Axis sanity checks")
    print(f"- mean(axis=0) matches transpose mean(axis=1): {np.allclose(col_means, col_means_via_T)}")
    print(f"- max(row minimums) via transpose: {max_of_row_mins_via_T:.4f}")
    print()
    print("Feature sorting")
    print(f"- first 5 sorted feature names: {names_sorted[:5]}")
    print()
    print("Matrix operations")
    print(f"- rowwise dot products (4 rows): {dots}")
    print()
    print("Cosine similarity")
    print(f"- raw features cos_sim(7,77): {cos_raw:.6f}")
    print(f"- standardized cos_sim(7,77): {cos_norm:.6f}")


if __name__ == "__main__":
    main()