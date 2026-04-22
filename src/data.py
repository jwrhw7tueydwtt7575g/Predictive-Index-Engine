from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def generate_timestamps(
    rows: int,
    start_ts: int = 1672531200,
    end_ts: int = 1704067199,
    seed: int = 42,
) -> np.ndarray:
    if rows <= 0:
        raise ValueError("rows must be positive")
    if start_ts >= end_ts:
        raise ValueError("start_ts must be less than end_ts")

    rng = np.random.default_rng(seed)
    keys = np.array([], dtype=np.int64)
    while keys.size < rows:
        sample = rng.integers(start_ts, end_ts + 1, size=rows, dtype=np.int64)
        keys = np.unique(np.concatenate([keys, sample]))
    keys = np.sort(keys)[:rows]
    return keys


def save_timestamps_csv(keys: np.ndarray, out_path: str) -> None:
    df = pd.DataFrame({"timestamp_key": keys})
    df.to_csv(out_path, index=False)


def load_csv_keys(path: str, key_column: str) -> np.ndarray:
    df = pd.read_csv(path, usecols=[key_column])
    keys = df[key_column].dropna().to_numpy(dtype=np.int64)
    keys = np.unique(keys)
    keys.sort()
    return keys


def train_test_split_keys(keys: np.ndarray, test_ratio: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be between 0 and 1")
    n = len(keys)
    split = int(n * (1.0 - test_ratio))
    return keys[:split], keys[split:]
