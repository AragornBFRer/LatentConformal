"""Regression models using oracle or EM-imputed cluster means."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _augment(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    return np.hstack([np.ones((n, 1)), X])


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    return out


def _solve_linear(design: np.ndarray, target: np.ndarray, ridge_alpha: float) -> np.ndarray:
    gram = design.T @ design
    rhs = design.T @ target
    if ridge_alpha > 0.0 and design.shape[1] > 1:
        penalty = ridge_alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0  # leave the intercept unpenalized
        gram = gram + penalty
    try:
        sol = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        sol = np.linalg.pinv(gram) @ rhs
    return sol


class Predictor:
    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs):  # pragma: no cover - interface
        raise NotImplementedError

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError

    def variant_name(self) -> str:
        return self.__class__.__name__


class IgnoreZPredictor(Predictor):
    def __init__(self, ridge_alpha: float = 0.0) -> None:
        self.ridge_alpha = ridge_alpha
        self.coef: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, **kwargs) -> "IgnoreZPredictor":
        design = _augment(X)
        self.coef = _solve_linear(design, Y, self.ridge_alpha)
        return self

    def predict_mean(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("Model not fitted")
        design = _augment(X)
        return design @ self.coef


class OracleZPredictor(Predictor):
    def __init__(self, ridge_alpha: float = 0.0) -> None:
        self.ridge_alpha = ridge_alpha
        self.intercept: float | None = None
        self.theta: np.ndarray | None = None
        self.b: np.ndarray | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, z_star: np.ndarray, **kwargs) -> "OracleZPredictor":
        z_feat = _ensure_2d(z_star)
        design = np.hstack([_augment(X), z_feat])
        coef = _solve_linear(design, Y, self.ridge_alpha)
        self.intercept = float(coef[0])
        self.theta = coef[1 : 1 + X.shape[1]]
        self.b = coef[1 + X.shape[1] :]
        return self

    def predict_mean(self, X: np.ndarray, *, z_star: np.ndarray, **kwargs) -> np.ndarray:
        if self.intercept is None or self.theta is None or self.b is None:
            raise RuntimeError("Model not fitted")
        z_feat = _ensure_2d(z_star)
        base = self.intercept + X @ self.theta if X.size else np.full(z_feat.shape[0], self.intercept)
        return base + (z_feat @ self.b)


@dataclass
class SoftParams:
    intercept: float
    theta: np.ndarray
    b: np.ndarray


class EMSoftPredictor(Predictor):
    def __init__(self, ridge_alpha: float = 0.0) -> None:
        self.ridge_alpha = ridge_alpha
        self.params: SoftParams | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, *, z_feat: np.ndarray, **kwargs) -> "EMSoftPredictor":
        z_feature = _ensure_2d(z_feat)
        design = np.hstack([_augment(X), z_feature])
        coef = _solve_linear(design, Y, self.ridge_alpha)
        intercept = float(coef[0])
        theta = coef[1 : 1 + X.shape[1]]
        b = coef[1 + X.shape[1] :]
        self.params = SoftParams(intercept=intercept, theta=theta, b=b)
        return self

    def predict_mean(self, X: np.ndarray, *, z_feat: np.ndarray, **kwargs) -> np.ndarray:
        if self.params is None:
            raise RuntimeError("Model not fitted")
        z_feature = _ensure_2d(z_feat)
        base = self.params.intercept + X @ self.params.theta if X.size else np.full(z_feature.shape[0], self.params.intercept)
        return base + (z_feature @ self.params.b)
