from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import generate_timestamps, save_timestamps_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic timestamp dataset")
    parser.add_argument("--rows", type=int, default=1000000, help="Number of rows")
    parser.add_argument("--out", required=True, help="Output CSV path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    keys = generate_timestamps(args.rows)
    save_timestamps_csv(keys, args.out)
    print(f"Wrote {len(keys)} rows to {args.out}")


if __name__ == "__main__":
    main()
