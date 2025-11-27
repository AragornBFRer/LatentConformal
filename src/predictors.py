"""Regression utilities for latent conformal experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import numpy as np
from sklearn.ensemble import RandomForestRegressor


def _ensure_2d(X: np.ndarray) -> np.ndarray:
    arr = np.asarray(X, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    return arr


class Predictor:
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def variant_name(self) -> str:
        return self.__class__.__name__


@dataclass(frozen=True)
class RandomForestParams:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int
    n_jobs: int
    random_state: int | None = None


class RandomForestMeanPredictor(Predictor):
    def __init__(self, params: RandomForestParams) -> None:
        self.params = params
        self._model: RandomForestRegressor | None = None

    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            min_samples_leaf=self.params.min_samples_leaf,
            n_jobs=self.params.n_jobs,
            random_state=self.params.random_state,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "RandomForestMeanPredictor":
        model = self._build_model()
        model.fit(_ensure_2d(X), np.asarray(Y, dtype=float))
        self._model = model
        return self

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict(_ensure_2d(X))


class RandomForestQuantilePredictor(Predictor):
    def __init__(self, params: RandomForestParams) -> None:
        self.params = params
        self._model: RandomForestRegressor | None = None
        self._leaf_targets: List[Dict[int, np.ndarray]] = []

    def _build_model(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            n_estimators=self.params.n_estimators,
            max_depth=self.params.max_depth,
            min_samples_leaf=self.params.min_samples_leaf,
            n_jobs=self.params.n_jobs,
            random_state=self.params.random_state,
            bootstrap=True,
        )

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "RandomForestQuantilePredictor":
        X_arr = _ensure_2d(X)
        y_arr = np.asarray(Y, dtype=float).ravel()
        model = self._build_model()
        model.fit(X_arr, y_arr)
        self._model = model
        self._leaf_targets = []
        for tree in model.estimators_:
            leaves = tree.apply(X_arr)
            mapping: Dict[int, List[float]] = {}
            for leaf_id, target in zip(leaves, y_arr):
                mapping.setdefault(int(leaf_id), []).append(float(target))
            self._leaf_targets.append(
                {leaf: np.sort(np.asarray(vals, dtype=float)) for leaf, vals in mapping.items()}
            )
        return self

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        return self._model.predict(_ensure_2d(X))

    def predict_quantiles(self, X: np.ndarray, quantiles: Iterable[float]) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted")
        X_arr = _ensure_2d(X)
        q_arr = np.asarray(list(quantiles), dtype=float)
        if q_arr.size == 0:
            raise ValueError("At least one quantile must be specified")
        if np.any((q_arr < 0.0) | (q_arr > 1.0)):
            raise ValueError("Quantiles must lie in [0, 1]")
        n_samples = X_arr.shape[0]
        preds = np.zeros((n_samples, q_arr.size), dtype=float)

        leaves = np.vstack([tree.apply(X_arr) for tree in self._model.estimators_])
        fallback_preds: Optional[np.ndarray] = None

        for idx in range(n_samples):
            samples: List[np.ndarray] = []
            for tree_idx, leaf_id in enumerate(leaves[:, idx]):
                leaf_vals = self._leaf_targets[tree_idx].get(int(leaf_id))
                if leaf_vals is not None and leaf_vals.size:
                    samples.append(leaf_vals)
            if samples:
                pool = np.concatenate(samples)
            else:
                if fallback_preds is None:
                    fallback_preds = np.vstack([
                        tree.predict(X_arr) for tree in self._model.estimators_
                    ])
                pool = fallback_preds[:, idx]
            preds[idx, :] = np.quantile(pool, q_arr)
        return preds


