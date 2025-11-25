### No longer used ###
"""Expectation-maximisation for Gaussian mixture models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .utils import logsumexp


@dataclass
class GMMParams:
    pi: np.ndarray
    means: np.ndarray
    covs: np.ndarray
    cov_type: str
    converged: bool
    n_iter: int
    log_likelihood: float

    def as_dict(self) -> dict:
        return {
            "pi": self.pi,
            "means": self.means,
            "covs": self.covs,
            "cov_type": self.cov_type,
            "converged": self.converged,
            "n_iter": self.n_iter,
            "log_likelihood": self.log_likelihood,
        }


def _init_means(data: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = data.shape[0]
    if n < K:
        raise ValueError("Not enough samples to initialise means")
    idx = rng.choice(n, size=K, replace=False)
    means = data[idx].copy()
    for _ in range(5):
        dists = np.sum((data[:, None, :] - means[None, :, :]) ** 2, axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                means[k] = data[mask].mean(axis=0)
            else:
                means[k] = data[rng.integers(0, n)]
    return means


def _estimate_log_gaussian_full(data: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> np.ndarray:
    d = data.shape[1]
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6 * np.eye(d)
        chol = np.linalg.cholesky(cov)
    solve = np.linalg.solve(chol, (data - mean).T)
    quad = np.sum(solve ** 2, axis=0)
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _estimate_log_gaussian_diag(data: np.ndarray, mean: np.ndarray, diag_cov: np.ndarray) -> np.ndarray:
    prec = 1.0 / diag_cov
    quad = np.sum(((data - mean) ** 2) * prec, axis=1)
    log_det = np.sum(np.log(diag_cov))
    d = data.shape[1]
    return -0.5 * (d * np.log(2.0 * np.pi) + log_det + quad)


def _estimate_log_prob(
    data: np.ndarray,
    pi: np.ndarray,
    means: np.ndarray,
    covs: np.ndarray | np.ndarray,
    cov_type: str,
) -> np.ndarray:
    n, d = data.shape
    K = means.shape[0]
    log_prob = np.empty((n, K))
    for k in range(K):
        if cov_type == "full":
            cov_k = covs[k]
            log_pdf = _estimate_log_gaussian_full(data, means[k], cov_k)
        elif cov_type == "diag":
            log_pdf = _estimate_log_gaussian_diag(data, means[k], covs[k])
        elif cov_type == "tied":
            log_pdf = _estimate_log_gaussian_full(data, means[k], covs)
        elif cov_type == "tied_diag":
            log_pdf = _estimate_log_gaussian_diag(data, means[k], covs)
        else:
            raise ValueError(f"Unsupported cov_type '{cov_type}'")
        log_prob[:, k] = np.log(pi[k] + 1e-16) + log_pdf
    return log_prob


def _m_step(
    data: np.ndarray,
    tau: np.ndarray,
    cov_type: str,
    reg_covar: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, d = data.shape
    K = tau.shape[1]
    Nk = tau.sum(axis=0) + 1e-16
    pi = Nk / n
    means = (tau.T @ data) / Nk[:, None]
    if cov_type == "tied":
        cov = np.zeros((d, d))
        for k in range(K):
            diff = data - means[k]
            weights = tau[:, k][:, None]
            cov += (weights * diff).T @ diff
        cov /= max(float(n), 1e-12)
        cov += reg_covar * np.eye(d)
        return pi, means, cov
    if cov_type == "tied_diag":
        cov = np.zeros(d)
        for k in range(K):
            diff = data - means[k]
            weights = tau[:, k]
            cov += np.sum((diff ** 2) * weights[:, None], axis=0)
        cov /= max(float(n), 1e-12)
        cov += reg_covar
        return pi, means, cov

    covs = []
    for k in range(K):
        diff = data - means[k]
        weights = tau[:, k][:, None]
        if cov_type == "full":
            cov = (weights * diff).T @ diff / max(Nk[k], 1e-12)
            cov += reg_covar * np.eye(d)
        elif cov_type == "diag":
            cov = np.sum(weights * (diff ** 2), axis=0) / max(Nk[k], 1e-12)
            cov += reg_covar
        else:
            raise ValueError(f"Unsupported cov_type '{cov_type}'")
        covs.append(cov)
    covs = np.array(covs)
    return pi, means, covs


def _fit_gmm_em_single(
    data: np.ndarray,
    K: int,
    *,
    cov_type: str,
    max_iter: int,
    tol: float,
    reg_covar: float,
    init: str,
    rng: np.random.Generator,
) -> GMMParams:
    data = np.asarray(data, dtype=float)
    n, d = data.shape

    if init.lower() == "kmeans":
        means = _init_means(data, K, rng)
    else:
        means = data[rng.choice(n, size=K, replace=False)]
    cov_init = np.cov(data.T) + reg_covar * np.eye(d)
    if cov_type == "diag":
        covs = np.tile(np.diag(cov_init), (K, 1))
    elif cov_type == "tied_diag":
        covs = np.diag(cov_init)
    elif cov_type == "tied":
        covs = cov_init
    elif cov_type == "full":
        covs = np.tile(cov_init, (K, 1, 1))
    else:
        raise ValueError(f"Unsupported cov_type '{cov_type}'")
    pi = np.full(K, 1.0 / K)

    log_prob = _estimate_log_prob(data, pi, means, covs, cov_type)
    prev_ll = np.sum(logsumexp(log_prob, axis=1))
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        log_resp = log_prob - logsumexp(log_prob, axis=1)[:, None]
        tau = np.exp(log_resp)
        pi, means, covs = _m_step(data, tau, cov_type, reg_covar)
        log_prob = _estimate_log_prob(data, pi, means, covs, cov_type)
        ll = np.sum(logsumexp(log_prob, axis=1))
        improvement = ll - prev_ll
        if improvement <= tol * (1.0 + abs(prev_ll)):
            converged = True
            n_iter = it
            break
        prev_ll = ll
        n_iter = it
    else:
        ll = prev_ll

    final_ll = float(np.sum(logsumexp(log_prob, axis=1)))
    return GMMParams(
        pi=pi,
        means=means,
        covs=covs,
        cov_type=cov_type,
        converged=converged,
        n_iter=n_iter,
        log_likelihood=final_ll,
    )


def fit_gmm_em(
    data: np.ndarray,
    K: int,
    *,
    cov_type: str = "full",
    max_iter: int = 200,
    tol: float = 1e-5,
    reg_covar: float = 1e-6,
    init: str = "kmeans",
    n_init: int = 1,
    r_dim: int | None = None,
    rng: np.random.Generator | None = None,
) -> GMMParams:
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("data must be a 2D array")
    if n_init <= 0:
        raise ValueError("n_init must be positive")

    base_rng = rng if rng is not None else np.random.default_rng()
    best_params: GMMParams | None = None
    best_ll = -np.inf
    best_spread = np.inf

    for init_idx in range(n_init):
        if init_idx == 0:
            local_rng = base_rng
        else:
            seed = int(base_rng.integers(0, np.iinfo(np.int64).max))
            local_rng = np.random.default_rng(seed)
        params = _fit_gmm_em_single(
            data,
            K,
            cov_type=cov_type,
            max_iter=max_iter,
            tol=tol,
            reg_covar=reg_covar,
            init=init,
            rng=local_rng,
        )
        ll = params.log_likelihood

        spread = np.inf
        if r_dim and r_dim > 0 and params.means.shape[1] >= r_dim and K > 1:
            means_r = params.means[:, :r_dim]
            if means_r.shape[0] > 1:
                diff = means_r[:, None, :] - means_r[None, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                tri = dist_matrix[np.triu_indices(dist_matrix.shape[0], k=1)]
                if tri.size > 0:
                    spread = float(np.std(tri))

        if best_params is None:
            best_params = params
            best_ll = ll
            best_spread = spread
            continue

        ll_tol = 1e-3 * max(1.0, abs(best_ll))
        if ll > best_ll + ll_tol:
            best_params = params
            best_ll = ll
            best_spread = spread
        elif abs(ll - best_ll) <= ll_tol and spread < best_spread:
            best_params = params
            best_ll = ll
            best_spread = spread

    # type checker: best_params is set when n_init > 0
    assert best_params is not None
    return best_params


def gmm_responsibilities(data: np.ndarray, params: GMMParams) -> np.ndarray:
    log_prob = _estimate_log_prob(data, params.pi, params.means, params.covs, params.cov_type)
    log_resp = log_prob - logsumexp(log_prob, axis=1)[:, None]
    tau = np.exp(log_resp)
    tau /= tau.sum(axis=1, keepdims=True)
    return tau
