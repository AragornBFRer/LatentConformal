"""Evaluation metrics for conformal intervals and imputations."""
from __future__ import annotations

import numpy as np


def coverage(y: np.ndarray, mu: np.ndarray, q: float | np.ndarray) -> float:
    y_arr = np.asarray(y, dtype=float)
    mu_arr = np.asarray(mu, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim == 0:
        q_arr = np.full_like(y_arr, float(q_arr))
    if q_arr.shape != y_arr.shape:
        raise ValueError("q must be scalar or match y shape")
    inside = (y_arr >= mu_arr - q_arr) & (y_arr <= mu_arr + q_arr)
    return float(np.mean(inside))


def avg_length(q: float | np.ndarray) -> float:
    q_arr = np.asarray(q, dtype=float)
    if q_arr.ndim == 0:
        return float(2.0 * q_arr)
    return float(np.mean(2.0 * q_arr))


def hard_accuracy(zhat: np.ndarray, z_true: np.ndarray) -> float:
    return float(np.mean(zhat.astype(int) == z_true.astype(int)))


def mean_max_tau(tau: np.ndarray) -> float:
    return float(np.mean(np.max(tau, axis=1)))


def cross_entropy(tau: np.ndarray, z_true: np.ndarray, eps: float = 1e-12) -> float:
    idx = z_true.astype(int)
    probs = tau[np.arange(tau.shape[0]), idx]
    probs = np.clip(probs, eps, 1.0)
    return float(-np.mean(np.log(probs)))


def z_feature_mse(z_est: np.ndarray, z_true: np.ndarray) -> float:
    est = np.asarray(z_est, dtype=float)
    truth = np.asarray(z_true, dtype=float)
    if est.ndim == 1:
        est = est.reshape(-1, 1)
    if truth.ndim == 1:
        truth = truth.reshape(-1, 1)
    diff = est - truth
    return float(np.mean(np.sum(diff**2, axis=1)))
