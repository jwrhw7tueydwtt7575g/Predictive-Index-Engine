from __future__ import annotations

import argparse
import os
import sys

import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from data import load_csv_keys
from rmi import RMIIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a two-stage RMI index")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--key", default="timestamp_key", help="Key column name")
    parser.add_argument("--experts", type=int, default=200, help="Number of experts")
    parser.add_argument(
        "--model-type",
        choices=["linear", "nn"],
        default="nn",
        help="Model type for stage 1 and experts",
    )
    parser.add_argument(
        "--hidden",
        default="64",
        help="Hidden layer sizes for NN, e.g. '64' or '128,64'",
    )
    parser.add_argument("--max-iter", type=int, default=500, help="NN max iterations")
    parser.add_argument("--model", required=True, help="Output model file (joblib)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = load_csv_keys(args.data, args.key)
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    index = RMIIndex(
        n_experts=args.experts,
        model_type=args.model_type,
        hidden_layer_sizes=hidden,
        max_iter=args.max_iter,
    ).fit(keys)
    joblib.dump(index, args.model)
    print(f"Trained index on {len(keys)} keys")
    print(f"Error bounds: min_err={index.min_err}, max_err={index.max_err}")


if __name__ == "__main__":
    main()
