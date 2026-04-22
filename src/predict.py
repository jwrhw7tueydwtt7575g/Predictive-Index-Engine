from __future__ import annotations

import argparse
import os
import sys

import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from rmi import RMIIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query a trained RMI index")
    parser.add_argument("--model", required=True, help="Model file (joblib)")
    parser.add_argument("--key", type=int, required=True, help="Key to search")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index: RMIIndex = joblib.load(args.model)
    pred, low, high = index.predict_bounds(args.key)
    found = index.search(args.key)

    print(f"Predicted position: {pred}")
    print(f"Search window: [{low}, {high}]")
    if found is None:
        print("Exact match: not found")
    else:
        print(f"Exact match position: {found}")


if __name__ == "__main__":
    main()
