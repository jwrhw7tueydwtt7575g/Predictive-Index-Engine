from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


@dataclass
class ExpertModel:
    model: Optional[RegressorMixin]
    constant: Optional[float]

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is not None:
            return self.model.predict(x)
        return np.full((x.shape[0],), self.constant, dtype=float)


class RMIIndex:
    def __init__(
        self,
        n_experts: int = 100,
        model_type: str = "nn",
        hidden_layer_sizes: tuple[int, ...] = (64,),
        max_iter: int = 500,
    ) -> None:
        if n_experts <= 0:
            raise ValueError("n_experts must be positive")
        if model_type not in {"linear", "nn"}:
            raise ValueError("model_type must be 'linear' or 'nn'")
        self.n_experts = n_experts
        self.model_type = model_type
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.stage1: Optional[RegressorMixin] = None
        self.experts: List[ExpertModel] = []
        self.keys: Optional[np.ndarray] = None
        self.min_err: int = 0
        self.max_err: int = 0

    def _make_model(self):
        if self.model_type == "linear":
            return LinearRegression()
        return MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42,
        )

    def fit(self, keys: np.ndarray) -> "RMIIndex":
        keys = np.asarray(keys, dtype=np.int64)
        keys = np.unique(keys)
        keys.sort()
        self.keys = keys
        n = len(keys)
        if n == 0:
            raise ValueError("keys must not be empty")

        x = keys.reshape(-1, 1)
        y = np.arange(n, dtype=float) / float(n)

        stage1 = self._make_model()
        stage1.fit(x, y)
        self.stage1 = stage1

        # Assign keys to experts
        stage1_pred = np.clip(stage1.predict(x), 0.0, 1.0)
        expert_ids = np.floor(stage1_pred * self.n_experts).astype(int)
        expert_ids = np.clip(expert_ids, 0, self.n_experts - 1)

        self.experts = []
        for expert_id in range(self.n_experts):
            mask = expert_ids == expert_id
            if not np.any(mask):
                self.experts.append(ExpertModel(model=None, constant=0.0))
                continue
            x_sub = x[mask]
            y_sub = np.nonzero(mask)[0].astype(float)
            if len(y_sub) < 2:
                self.experts.append(ExpertModel(model=None, constant=float(y_sub[0])))
                continue
            model = self._make_model()
            model.fit(x_sub, y_sub)
            self.experts.append(ExpertModel(model=model, constant=None))

        self._compute_error_bounds()
        return self

    def _compute_error_bounds(self) -> None:
        if self.keys is None or self.stage1 is None:
            raise ValueError("model not fitted")

        x = self.keys.reshape(-1, 1)
        n = len(self.keys)
        pred = self._predict_positions(x)
        true_idx = np.arange(n, dtype=int)
        errors = true_idx - pred
        self.min_err = int(errors.min())
        self.max_err = int(errors.max())

    def _route_expert(self, x: np.ndarray) -> np.ndarray:
        if self.stage1 is None:
            raise ValueError("model not fitted")
        stage1_pred = np.clip(self.stage1.predict(x), 0.0, 1.0)
        expert_ids = np.floor(stage1_pred * self.n_experts).astype(int)
        return np.clip(expert_ids, 0, self.n_experts - 1)

    def _predict_positions(self, x: np.ndarray) -> np.ndarray:
        expert_ids = self._route_expert(x)
        preds = np.zeros((x.shape[0],), dtype=float)
        for expert_id in range(self.n_experts):
            mask = expert_ids == expert_id
            if not np.any(mask):
                continue
            expert = self.experts[expert_id]
            preds[mask] = expert.predict(x[mask])
        return np.round(preds).astype(int)

    def predict_bounds(self, key: int) -> Tuple[int, int, int]:
        if self.keys is None:
            raise ValueError("model not fitted")
        x = np.array([[key]], dtype=np.int64)
        pred = int(self._predict_positions(x)[0])
        low = max(0, pred + self.min_err)
        high = min(len(self.keys) - 1, pred + self.max_err)
        return pred, low, high

    def search(self, key: int) -> Optional[int]:
        if self.keys is None:
            raise ValueError("model not fitted")
        _, low, high = self.predict_bounds(key)
        segment = self.keys[low : high + 1]
        idx = int(np.searchsorted(segment, key, side="left"))
        global_idx = low + idx
        if global_idx < len(self.keys) and self.keys[global_idx] == key:
            return global_idx
        return None
