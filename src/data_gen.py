"""Synthetic data generation for latent-cluster experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .config import DGPConfig, RunConfig
from .utils import simplex_vertices


@dataclass
class DatasetSplit:
    X: np.ndarray
    R: np.ndarray
    Y: np.ndarray
    Z: np.ndarray


def _make_covariance(dgp: DGPConfig, rho: float, use_full_S: bool) -> np.ndarray:
    d_r = dgp.d_R
    d_x = dgp.d_X
    sigma2 = dgp.sigma_s ** 2
    if use_full_S:
        cov_rr = sigma2 * np.eye(d_r)
        cov_xx = sigma2 * np.eye(d_x)
        if dgp.cov_type_true == "full" and rho != 0.0:
            cov_rx = rho * sigma2 * np.eye(d_r, d_x)
        else:
            cov_rx = np.zeros((d_r, d_x))
        top = np.hstack([cov_rr, cov_rx])
        bottom = np.hstack([cov_rx.T, cov_xx])
        cov = np.vstack([top, bottom])
        if dgp.cov_type_true == "diag":
            cov = np.diag(np.diag(cov))
    else:
        cov = sigma2 * np.eye(d_r)
        if dgp.cov_type_true == "diag":
            cov = np.diag(np.diag(cov))
    cov += 1e-9 * np.eye(cov.shape[0])
    return cov


def generate_data(
    dgp_cfg: DGPConfig,
    run_cfg: RunConfig,
    n_train: int,
    n_cal: int,
    n_test: int,
    rng: np.random.Generator,
) -> Tuple[DatasetSplit, DatasetSplit, DatasetSplit, Dict[str, np.ndarray]]:
    K = run_cfg.K
    d_r = dgp_cfg.d_R
    d_x = dgp_cfg.d_X
    total = n_train + n_cal + n_test

    # delta scales the simplex of latent means in the R-space
    means_r = simplex_vertices(K, d_r, run_cfg.delta, allow_smaller_dim=True)

    use_full_S = dgp_cfg.use_S.upper() == "RX"
    means_x = np.zeros((K, d_x))
    if dgp_cfg.mu_x_shift != 0.0 and d_x > 0:
        # optional X-mean shift also scales with delta
        shift = dgp_cfg.mu_x_shift * run_cfg.delta
        base = np.zeros((K, d_x))
        base[:, 0] = np.linspace(-(K - 1) / 2.0, (K - 1) / 2.0, K)
        base -= base.mean(axis=0, keepdims=True)
        means_x = shift * base
    if use_full_S:
        means_full = np.zeros((K, d_r + d_x))
        means_full[:, :d_r] = means_r
        means_full[:, d_r:] = means_x
    else:
        means_full = None

    pi = np.full(K, 1.0 / K)
    z = rng.choice(K, size=total, p=pi)

    R = np.empty((total, d_r))
    X = np.empty((total, d_x))

    if use_full_S:
        cov = _make_covariance(dgp_cfg, run_cfg.rho, True)
        for k in range(K):
            mask = z == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            R_and_X = rng.multivariate_normal(mean=means_full[k], cov=cov, size=n_k)
            R[mask] = R_and_X[:, :d_r]
            X[mask] = R_and_X[:, d_r:]
    else:
        cov_r = _make_covariance(dgp_cfg, run_cfg.rho, False)
        for k in range(K):
            mask = z == k
            n_k = int(mask.sum())
            if n_k == 0:
                continue
            R[mask] = rng.multivariate_normal(mean=means_r[k], cov=cov_r, size=n_k)
        # X independent standard normal
        X[:] = rng.normal(size=(total, d_x))

    theta = rng.normal(scale=1.0 / max(1, np.sqrt(d_x)), size=d_x)
    if d_r > 0:
        b = rng.normal(scale=1.0 / max(1, np.sqrt(d_r)), size=d_r)
        norm_b = np.linalg.norm(b)
        if run_cfg.b_scale <= 0.0:
            b = np.zeros_like(b)
        elif norm_b > 0:
            b = (run_cfg.b_scale / norm_b) * b
        else:
            b = np.zeros_like(b)
        mu_r_lookup = means_r
        cluster_contrib = mu_r_lookup[z] @ b
    else:
        b = np.zeros(0, dtype=float)
        cluster_contrib = np.zeros(total)

    noise = rng.normal(scale=run_cfg.sigma_y, size=total)
    linear = X @ theta if d_x > 0 else np.zeros(total)
    Y = linear + cluster_contrib + noise

    train_slice = slice(0, n_train)
    cal_slice = slice(n_train, n_train + n_cal)
    test_slice = slice(n_train + n_cal, total)

    train = DatasetSplit(X=X[train_slice], R=R[train_slice], Y=Y[train_slice], Z=z[train_slice])
    cal = DatasetSplit(X=X[cal_slice], R=R[cal_slice], Y=Y[cal_slice], Z=z[cal_slice])
    test = DatasetSplit(X=X[test_slice], R=R[test_slice], Y=Y[test_slice], Z=z[test_slice])

    info = {
        "pi": pi,
        "means_r": means_r,
        "means_x": means_x,
        "theta": theta,
        "b": b,
        "sigma_y": run_cfg.sigma_y,
        "rho": run_cfg.rho,
    }
    if means_full is not None:
        info["means_full"] = means_full

    return train, cal, test, info
