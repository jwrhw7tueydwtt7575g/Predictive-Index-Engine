from __future__ import annotations

import argparse
import os
import sys

import joblib
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import load_csv_keys
from rmi import RMIIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained RMI index")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--key", default="timestamp_key", help="Key column name")
    parser.add_argument("--model", required=True, help="Model file (joblib)")
    parser.add_argument("--samples", type=int, default=10000, help="Number of random probes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = load_csv_keys(args.data, args.key)
    index: RMIIndex = joblib.load(args.model)

    rng = np.random.default_rng(123)
    sample_idx = rng.integers(0, len(keys), size=min(args.samples, len(keys)))

    window_sizes = []
    misses = 0
    for i in sample_idx:
        key = int(keys[i])
        pred, low, high = index.predict_bounds(key)
        window_sizes.append(high - low + 1)
        found = index.search(key)
        if found != i:
            misses += 1

    print(f"Samples: {len(sample_idx)}")
    print(f"Average window size: {float(np.mean(window_sizes)):.2f}")
    print(f"Misses: {misses}")


if __name__ == "__main__":
    main()
