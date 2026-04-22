from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np

from src.data import load_csv_keys
from src.rmi import RMIIndex


@dataclass
class PipelineArtifacts:
    keys: np.ndarray
    index: RMIIndex
    cdf_x: np.ndarray
    cdf_y: np.ndarray


def step_01_generate_sort_data(keys: Iterable[int]) -> np.ndarray:
    arr = np.asarray(list(keys), dtype=np.int64)
    arr = arr[~np.isnan(arr)] if arr.dtype.kind == "f" else arr
    arr = np.unique(arr)
    arr.sort()
    if len(arr) == 0:
        raise ValueError("No valid keys after deduplication")
    return arr


def step_02_build_cdf_pairs(keys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = len(keys)
    x = keys.reshape(-1, 1)
    y = np.arange(n, dtype=float) / float(n)
    return x, y


def step_03_to_06_train_and_bounds(
    keys: np.ndarray,
    n_experts: int,
    model_type: str,
    hidden_layer_sizes: tuple[int, ...],
    max_iter: int,
) -> RMIIndex:
    index = RMIIndex(
        n_experts=n_experts,
        model_type=model_type,
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
    )
    index.fit(keys)
    return index


def step_07_lookup(index: RMIIndex, key: int) -> Tuple[int, int, int, int | None]:
    pred, low, high = index.predict_bounds(key)
    found = index.search(key)
    return pred, low, high, found


def step_09_visualise(
    keys: np.ndarray,
    index: RMIIndex,
    samples: int = 20000,
    out_prefix: str = "plots",
) -> None:
    n = len(keys)
    x = keys
    y = np.arange(n, dtype=float) / float(n)

    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.title("CDF of Keys")
    plt.xlabel("Key")
    plt.ylabel("CDF")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cdf.png", dpi=150)
    plt.close()

    rng = np.random.default_rng(123)
    sample_idx = rng.integers(0, n, size=min(samples, n))
    sample_keys = keys[sample_idx]

    preds = []
    for key in sample_keys:
        pred, _, _, _ = step_07_lookup(index, int(key))
        preds.append(pred)
    preds = np.array(preds, dtype=int)
    errors = sample_idx - preds

    plt.figure(figsize=(8, 4))
    plt.hist(errors, bins=100)
    plt.title("Prediction Error Histogram")
    plt.xlabel("Error (true_index - predicted)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_errors.png", dpi=150)
    plt.close()

    # Speed comparison: bounded search vs bisect
    def rmi_probe() -> None:
        for key in sample_keys:
            index.search(int(key))

    def bisect_probe() -> None:
        for key in sample_keys:
            _ = np.searchsorted(keys, int(key), side="left")

    t0 = time.perf_counter()
    rmi_probe()
    rmi_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    bisect_probe()
    bisect_time = time.perf_counter() - t1

    plt.figure(figsize=(6, 4))
    plt.bar(["RMI", "Bisect"], [rmi_time, bisect_time])
    plt.title("Speed Comparison")
    plt.ylabel("Total time (s)")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_speed.png", dpi=150)
    plt.close()


def run_pipeline(
    data_path: str,
    key_column: str,
    n_experts: int,
    model_type: str,
    hidden_layer_sizes: tuple[int, ...],
    max_iter: int,
) -> PipelineArtifacts:
    raw_keys = load_csv_keys(data_path, key_column)
    keys = step_01_generate_sort_data(raw_keys)
    _, _ = step_02_build_cdf_pairs(keys)
    index = step_03_to_06_train_and_bounds(
        keys, n_experts, model_type, hidden_layer_sizes, max_iter
    )
    cdf_x = keys
    cdf_y = np.arange(len(keys), dtype=float) / float(len(keys))
    return PipelineArtifacts(keys=keys, index=index, cdf_x=cdf_x, cdf_y=cdf_y)
