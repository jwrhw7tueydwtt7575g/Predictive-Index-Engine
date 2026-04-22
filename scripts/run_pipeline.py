from __future__ import annotations

import argparse
import os
import sys

import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.pipeline import run_pipeline, step_07_lookup, step_09_visualise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full RMI pipeline (steps 01–09)")
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
    parser.add_argument("--model", default="data/rmi.joblib", help="Output model file")
    parser.add_argument("--plots", default="plots", help="Plot file prefix")
    parser.add_argument("--probe", type=int, help="Optional key to lookup")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    artifacts = run_pipeline(
        data_path=args.data,
        key_column=args.key,
        n_experts=args.experts,
        model_type=args.model_type,
        hidden_layer_sizes=hidden,
        max_iter=args.max_iter,
    )

    os.makedirs(os.path.dirname(args.model), exist_ok=True)
    joblib.dump(artifacts.index, args.model)
    print(f"Saved model to {args.model}")
    print(f"Error bounds: min_err={artifacts.index.min_err}, max_err={artifacts.index.max_err}")

    if args.probe is not None:
        pred, low, high, found = step_07_lookup(artifacts.index, args.probe)
        print(f"Predicted position: {pred}")
        print(f"Search window: [{low}, {high}]")
        print(f"Exact match position: {found}")

    step_09_visualise(artifacts.keys, artifacts.index, out_prefix=args.plots)
    print(f"Saved plots: {args.plots}_cdf.png, {args.plots}_errors.png, {args.plots}_speed.png")


if __name__ == "__main__":
    main()
