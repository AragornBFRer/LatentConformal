"""EM algorithm tailored to the documented XRZY simulation model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .utils import logsumexp


@dataclass
class DocEMParams:
    eta0: float
    eta: np.ndarray
    mu_r: np.ndarray
    alpha: np.ndarray
    sigma: np.ndarray
    pi: np.ndarray
    converged: bool
    n_iter: int
    log_likelihood: float


def _ensure_matrix(arr: ArrayLike) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        return out.reshape(-1, 1)
    return out


def _init_mu_r(R: np.ndarray, K: int, rng: np.random.Generator) -> np.ndarray:
    n = R.shape[0]
    if n < K:
        raise ValueError("Need at least K samples to initialise mu_R")
    idx = rng.choice(n, size=K, replace=False)
    centers = R[idx].copy()
    for _ in range(5):
        dists = np.linalg.norm(R[:, None, :] - centers[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        for k in range(K):
            mask = labels == k
            if mask.any():
                centers[k] = R[mask].mean(axis=0)
            else:
                centers[k] = R[rng.integers(0, n)]
    return centers


def _log_joint_prob(
    X: np.ndarray,
    R: np.ndarray,
    Y: np.ndarray,
    eta0: float,
    eta: np.ndarray,
    mu_r: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
    pi: np.ndarray,
) -> np.ndarray:
    n = Y.size
    base = np.full(n, eta0, dtype=float)
    if X.size:
        base += X @ eta
    K = alpha.size
    log_prob = np.empty((n, K), dtype=float)
    log_two_pi = np.log(2.0 * np.pi)
    for k in range(K):
        mu_y = base + alpha[k]
        var_y = sigma[k] ** 2
        log_p_y = -0.5 * (np.log(2.0 * np.pi * var_y) + ((Y - mu_y) ** 2) / var_y)
        if R.size:
            diff = R - mu_r[k]
            quad = np.sum(diff ** 2, axis=1)
            log_p_r = -0.5 * (R.shape[1] * log_two_pi + quad)
        else:
            log_p_r = 0.0
        log_prob[:, k] = np.log(pi[k] + 1e-16) + log_p_y + log_p_r
    return log_prob


def _responsibilities(log_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    log_norm = logsumexp(log_prob, axis=1)
    gamma = np.exp(log_prob - log_norm[:, None])
    return gamma, log_norm


def _update_eta(
    X: np.ndarray,
    Y: np.ndarray,
    gamma: np.ndarray,
    alpha: np.ndarray,
    sigma: np.ndarray,
) -> Tuple[float, np.ndarray]:
    inv_sigma2 = 1.0 / (sigma ** 2)
    weights = gamma @ inv_sigma2
    weights = np.maximum(weights, 1e-12)
    numerator = np.sum(gamma * ((Y[:, None] - alpha[None, :]) * inv_sigma2[None, :]), axis=1)
    y_star = numerator / weights
    design = np.ones((Y.size, 1 + (X.shape[1] if X.ndim == 2 else 0)), dtype=float)
    if X.size:
        design[:, 1:] = X
    W_design = design * weights[:, None]
    gram = design.T @ W_design
    rhs = design.T @ (weights * y_star)
    try:
        coef = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(gram) @ rhs
    eta0 = float(coef[0])
    eta = coef[1:] if coef.size > 1 else np.zeros(0, dtype=float)
    return eta0, eta


def _update_mu_r(R: np.ndarray, gamma: np.ndarray, prev_mu: np.ndarray) -> np.ndarray:
    if R.size == 0:
        return prev_mu
    Nk = gamma.sum(axis=0)
    updated = prev_mu.copy()
    for k in range(gamma.shape[1]):
        if Nk[k] <= 1e-12:
            continue
        weights = gamma[:, k][:, None]
        updated[k] = np.sum(weights * R, axis=0) / Nk[k]
    return updated


def fit_doc_em(
    X: ArrayLike,
    R: ArrayLike,
    Y: ArrayLike,
    *,
    alpha: ArrayLike,
    sigma: ArrayLike,
    pi: ArrayLike | None = None,
    max_iter: int = 200,
    tol: float = 1e-6,
    rng: np.random.Generator | None = None,
) -> Tuple[DocEMParams, np.ndarray]:
    rng = rng or np.random.default_rng()
    X_mat = _ensure_matrix(X)
    R_mat = _ensure_matrix(R)
    Y_vec = np.asarray(Y, dtype=float).reshape(-1)
    n = Y_vec.size
    if X_mat.shape[0] != n or R_mat.shape[0] != n:
        raise ValueError("X, R, and Y must share the same number of samples")
    alpha_arr = np.asarray(alpha, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    if alpha_arr.ndim != 1 or sigma_arr.ndim != 1:
        raise ValueError("alpha and sigma must be one-dimensional")
    if alpha_arr.size != sigma_arr.size:
        raise ValueError("alpha and sigma must have the same length")
    K = alpha_arr.size
    if pi is None:
        pi_arr = np.full(K, 1.0 / K)
    else:
        pi_arr = np.asarray(pi, dtype=float)
        if pi_arr.size != K:
            raise ValueError("pi must have length K")
        total = float(np.sum(pi_arr))
        if total <= 0:
            raise ValueError("pi must sum to a positive value")
        pi_arr = pi_arr / total

    eta0 = 0.0
    eta = np.zeros(X_mat.shape[1], dtype=float)
    mu_r = _init_mu_r(R_mat, K, rng)

    log_prob = _log_joint_prob(X_mat, R_mat, Y_vec, eta0, eta, mu_r, alpha_arr, sigma_arr, pi_arr)
    prev_ll = float(np.sum(logsumexp(log_prob, axis=1)))
    converged = False
    n_iter = 0

    for it in range(1, max_iter + 1):
        gamma, _ = _responsibilities(log_prob)
        eta0, eta = _update_eta(X_mat, Y_vec, gamma, alpha_arr, sigma_arr)
        mu_r = _update_mu_r(R_mat, gamma, mu_r)
        log_prob = _log_joint_prob(X_mat, R_mat, Y_vec, eta0, eta, mu_r, alpha_arr, sigma_arr, pi_arr)
        ll = float(np.sum(logsumexp(log_prob, axis=1)))
        if abs(ll - prev_ll) <= tol * (1.0 + abs(prev_ll)):
            converged = True
            n_iter = it
            prev_ll = ll
            break
        prev_ll = ll
        n_iter = it
    else:
        ll = prev_ll
        n_iter = max_iter

    gamma, _ = _responsibilities(log_prob)
    params = DocEMParams(
        eta0=eta0,
        eta=eta,
        mu_r=mu_r,
        alpha=alpha_arr,
        sigma=sigma_arr,
        pi=pi_arr,
        converged=converged,
        n_iter=n_iter,
        log_likelihood=float(prev_ll),
    )
    return params, gamma


def responsibilities_with_y(
    X: ArrayLike,
    R: ArrayLike,
    Y: ArrayLike,
    params: DocEMParams,
) -> np.ndarray:
    X_mat = _ensure_matrix(X)
    R_mat = _ensure_matrix(R)
    Y_vec = np.asarray(Y, dtype=float).reshape(-1)
    if X_mat.shape[0] != Y_vec.size or R_mat.shape[0] != Y_vec.size:
        raise ValueError("X, R, and Y must align")
    log_prob = _log_joint_prob(
        X_mat,
        R_mat,
        Y_vec,
        params.eta0,
        params.eta,
        params.mu_r,
        params.alpha,
        params.sigma,
        params.pi,
    )
    gamma, _ = _responsibilities(log_prob)
    return gamma


def responsibilities_from_r(R: ArrayLike, params: DocEMParams) -> np.ndarray:
    R_mat = _ensure_matrix(R)
    if params.mu_r.shape[1] != R_mat.shape[1]:
        raise ValueError("R dimension does not match mu_r")
    K = params.pi.size
    log_two_pi = np.log(2.0 * np.pi)
    log_prob = np.empty((R_mat.shape[0], K), dtype=float)
    for k in range(K):
        diff = R_mat - params.mu_r[k]
        quad = np.sum(diff ** 2, axis=1)
        log_prob[:, k] = np.log(params.pi[k] + 1e-16) - 0.5 * (R_mat.shape[1] * log_two_pi + quad)
    gamma, _ = _responsibilities(log_prob)
    return gamma
