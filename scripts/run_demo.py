from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import generate_timestamps, save_timestamps_csv
from src.rmi import RMIIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="End-to-end RMI demo on timestamps")
    parser.add_argument("--rows", type=int, default=200000, help="Number of rows")
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
    parser.add_argument("--out", default="data/demo_timestamps.csv", help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = generate_timestamps(args.rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_timestamps_csv(keys, args.out)

    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())
    index = RMIIndex(
        n_experts=args.experts,
        model_type=args.model_type,
        hidden_layer_sizes=hidden,
        max_iter=args.max_iter,
    ).fit(keys)
    print(f"Trained on {len(keys)} keys")
    print(f"Error bounds: min_err={index.min_err}, max_err={index.max_err}")

    probe_key = int(keys[len(keys) // 2])
    pred, low, high = index.predict_bounds(probe_key)
    found = index.search(probe_key)

    print(f"Probe key: {probe_key}")
    print(f"Predicted position: {pred}")
    print(f"Search window: [{low}, {high}]")
    print(f"Exact match position: {found}")


if __name__ == "__main__":
    main()
