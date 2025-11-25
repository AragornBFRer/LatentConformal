"""Synthetic data generation aligned with the documented XRZY model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import DGPConfig, RunConfig


@dataclass
class DatasetSplit:
    X: np.ndarray
    R: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


def _cluster_params(dgp_cfg: DGPConfig, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha = np.asarray(dgp_cfg.alpha[:K], dtype=float)
    sigma = np.asarray(dgp_cfg.sigma[:K], dtype=float)
    mu_r = np.asarray(dgp_cfg.mu_r[:K], dtype=float)
    if mu_r.ndim == 1:
        mu_r = mu_r.reshape(-1, 1)
    if alpha.size != K or sigma.size != K or mu_r.shape[0] != K:
        raise ValueError("Cluster parameter lists must provide at least K entries")
    return alpha, sigma, mu_r


def generate_data(
    dgp_cfg: DGPConfig,
    run_cfg: RunConfig,
    n_train: int,
    n_cal: int,
    n_test: int,
    rng: np.random.Generator,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, Dict[str, np.ndarray]]:
    K = int(run_cfg.K)
    d_r = int(dgp_cfg.d_R)
    d_x = int(dgp_cfg.d_X)
    total = n_train + n_cal + n_test

    alpha, sigma, mu_r = _cluster_params(dgp_cfg, K)
    eta = np.asarray(dgp_cfg.eta, dtype=float)
    if eta.size != d_x:
        raise ValueError(f"eta vector has length {eta.size}, expected {d_x}")
    eta0 = float(dgp_cfg.eta0)

    pi = np.full(K, 1.0 / K)
    z = rng.choice(K, size=total, p=pi)

    X = rng.normal(loc=0.0, scale=1.0, size=(total, d_x)) if d_x > 0 else np.zeros((total, 0))
    R = np.zeros((total, d_r)) if d_r > 0 else np.zeros((total, 0))
    if d_r > 0:
        for k in range(K):
            mask = z == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            center = mu_r[k]
            R[mask] = rng.normal(loc=center, scale=1.0, size=(n_k, d_r))

    linear = (X @ eta) if d_x > 0 else np.zeros(total)
    alpha_shift = alpha[z]
    noise = rng.normal(loc=0.0, scale=sigma[z], size=total)
    Y = eta0 + linear + alpha_shift + noise

    train_slice = slice(0, n_train)
    cal_slice = slice(n_train, n_train + n_cal)
    test_slice = slice(n_train + n_cal, total)

    train = DatasetSplit(X=X[train_slice], R=R[train_slice], Y=Y[train_slice], Z=z[train_slice])
    cal = DatasetSplit(X=X[cal_slice], R=R[cal_slice], Y=Y[cal_slice], Z=z[cal_slice])
    test = DatasetSplit(X=X[test_slice], R=R[test_slice], Y=Y[test_slice], Z=z[test_slice])

    info = {
        "pi": pi,
        "means_r": mu_r,
        "alpha": alpha,
        "sigma": sigma,
        "eta0": eta0,
        "eta": eta,
    }

    return train, cal, test, info
