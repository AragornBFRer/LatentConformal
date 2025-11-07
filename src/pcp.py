"""Posterior conformal prediction tailored to the XRZY setting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class ConstantProbabilityEstimator:
    def __init__(self, prob: float) -> None:
        self.prob = float(np.clip(prob, 0.0, 1.0))

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ConstantProbabilityEstimator":  # pragma: no cover - trivial
        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        n = np.shape(X)[0]
        p = np.full(n, self.prob, dtype=float)
        return np.column_stack([1.0 - p, p])


@dataclass
class PCPConfig:
    n_thresholds: int = 9
    logistic_cv_folds: int = 5
    max_clusters: int = 15
    cluster_r2_tol: float = 0.05
    factor_max_iter: int = 400
    factor_tol: float = 1e-4
    precision_grid: Tuple[int, ...] = (20, 50, 100, 200, 500)
    precision_trials: int = 64
    clip_eps: float = 1e-8
    proj_lr: float = 0.2
    proj_max_iter: int = 200
    proj_tol: float = 1e-6


@dataclass
class PCPModel:
    thresholds: np.ndarray
    tau_models: List
    gamma: np.ndarray
    pi_cal: np.ndarray
    log_pi_cal: np.ndarray
    scores_cal: np.ndarray
    alpha: float
    clip_eps: float
    precision: int
    precision_fallbacks: Tuple[int, ...]
    proj_lr: float
    proj_max_iter: int
    proj_tol: float
    n_clusters: int
    cluster_r2: float

    def _ensure_matrix(self, R: ArrayLike) -> np.ndarray:
        arr = np.asarray(R, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr

    def _tau_eval(self, R: ArrayLike) -> np.ndarray:
        R_mat = self._ensure_matrix(R)
        preds = []
        for model in self.tau_models:
            prob = model.predict_proba(R_mat)[:, 1]
            preds.append(np.clip(prob, self.clip_eps, 1.0 - self.clip_eps))
        return np.column_stack(preds)

    def _project_membership(self, tau_vec: np.ndarray) -> np.ndarray:
        K = self.gamma.shape[0]
        pi = np.full(K, 1.0 / K)
        gamma = self.gamma
        for _ in range(self.proj_max_iter):
            pred = pi @ gamma
            grad = (pred - tau_vec) @ gamma.T
            new_pi = _project_simplex(pi - self.proj_lr * grad)
            if np.linalg.norm(new_pi - pi, ord=1) <= self.proj_tol:
                pi = new_pi
                break
            pi = new_pi
        return np.clip(pi, 0.0, 1.0)

    def _weights_from_counts(self, L: np.ndarray, log_pi_test: np.ndarray) -> Tuple[np.ndarray, float]:
        log_phi = self.log_pi_cal @ L
        log_phi_inf = float(np.dot(L, log_pi_test))
        max_log = max(float(np.max(log_phi)), log_phi_inf)
        phi = np.exp(log_phi - max_log)
        phi_inf = float(np.exp(log_phi_inf - max_log))
        denom = float(np.sum(phi) + phi_inf)
        if denom == 0.0:
            return np.zeros_like(log_phi), 1.0
        weights = phi / denom
        return weights, phi_inf / denom

    def _quantile_from_membership(self, pi_test: np.ndarray, rng: np.random.Generator) -> float:
        pi_test = np.asarray(pi_test, dtype=float)
        pi_test = np.clip(pi_test, self.clip_eps, 1.0)
        pi_test /= np.sum(pi_test)
        log_pi_test = np.log(pi_test)
        candidates = (self.precision,) + tuple(self.precision_fallbacks)
        for m in candidates:
            if m <= 0:
                continue
            L = rng.multinomial(m, pi_test)
            weights, w_inf = self._weights_from_counts(L, log_pi_test)
            if w_inf > self.alpha + 1e-12:
                continue
            q = _weighted_quantile(self.scores_cal, weights, self.alpha, w_inf)
            if np.isfinite(q):
                return float(q)
        return float(np.inf)

    def quantiles(self, R_test: ArrayLike, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        R_mat = self._ensure_matrix(R_test)
        tau_test = self._tau_eval(R_mat)
        out = np.empty(R_mat.shape[0], dtype=float)
        for idx in range(R_mat.shape[0]):
            pi_test = self._project_membership(tau_test[idx])
            out[idx] = self._quantile_from_membership(pi_test, rng)
        return out


def train_pcp(
    R_cal: ArrayLike,
    scores_cal: ArrayLike,
    *,
    alpha: float,
    rng: np.random.Generator,
    config: PCPConfig | None = None,
) -> PCPModel:
    if config is None:
        config = PCPConfig()
    R_mat = _ensure_matrix_np(R_cal)
    scores_arr = np.asarray(scores_cal, dtype=float)
    thresholds = _quantile_grid(scores_arr, config.n_thresholds)
    tau_cal, tau_models = _fit_tau_models(R_mat, scores_arr, thresholds, config, rng)
    pi_cal, gamma, r2, n_clusters = _select_clusters(tau_cal, config, rng)
    log_pi_cal = np.log(np.clip(pi_cal, config.clip_eps, 1.0))
    precision, fallbacks = _select_precision(pi_cal, log_pi_cal, alpha, config, rng)
    model = PCPModel(
        thresholds=thresholds,
        tau_models=tau_models,
        gamma=gamma,
        pi_cal=pi_cal,
        log_pi_cal=log_pi_cal,
        scores_cal=scores_arr,
        alpha=alpha,
        clip_eps=config.clip_eps,
        precision=precision,
        precision_fallbacks=fallbacks,
        proj_lr=config.proj_lr,
        proj_max_iter=config.proj_max_iter,
        proj_tol=config.proj_tol,
        n_clusters=n_clusters,
        cluster_r2=r2,
    )
    return model


def _ensure_matrix_np(arr: ArrayLike) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    if out.ndim == 1:
        out = out.reshape(-1, 1)
    return out


def _quantile_grid(scores: np.ndarray, s: int) -> np.ndarray:
    if s <= 0:
        raise ValueError("Number of thresholds must be positive")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if scores.size == 0:
        raise ValueError("Need calibration scores to build thresholds")
    try:
        qs = np.linspace(1.0 / (s + 1), s / (s + 1), s)
        thresh = np.quantile(scores, qs, method="linear")
    except TypeError:  # numpy < 1.22
        thresh = np.quantile(scores, qs, interpolation="linear")
    thresh = np.asarray(thresh, dtype=float)
    eps = 1e-6 * max(1.0, float(np.max(np.abs(scores))))
    for idx in range(1, thresh.size):
        if thresh[idx] <= thresh[idx - 1]:
            thresh[idx] = min(thresh[idx - 1] + eps, float(np.max(scores)))
    return thresh


def _fit_tau_models(
    R_cal: np.ndarray,
    scores_cal: np.ndarray,
    thresholds: np.ndarray,
    config: PCPConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, List]:
    n = R_cal.shape[0]
    tau_mat = np.empty((n, thresholds.size), dtype=float)
    models: List = []
    base_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, solver="lbfgs"),
    )
    kf = KFold(n_splits=config.logistic_cv_folds, shuffle=True, random_state=int(rng.integers(0, 2**32 - 1)))
    for t, thr in enumerate(thresholds):
        labels = (scores_cal <= thr).astype(int)
        if np.all(labels == labels[0]):
            prob = float(labels[0])
            preds = np.full(n, prob, dtype=float)
            model = ConstantProbabilityEstimator(prob)
        else:
            preds = np.zeros(n, dtype=float)
            for train_idx, val_idx in kf.split(R_cal):
                est = clone(base_model)
                est.fit(R_cal[train_idx], labels[train_idx])
                preds[val_idx] = est.predict_proba(R_cal[val_idx])[:, 1]
            model = clone(base_model)
            model.fit(R_cal, labels)
        tau_mat[:, t] = np.clip(preds, config.clip_eps, 1.0 - config.clip_eps)
        models.append(model)
    return tau_mat, models


def _select_clusters(
    tau_cal: np.ndarray,
    config: PCPConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float, int]:
    best_pi = None
    best_gamma = None
    best_r2 = -np.inf
    prev_r2 = -np.inf
    selected_k = 1
    for k in range(1, config.max_clusters + 1):
        pi, gamma, r2 = _factorize_tau(tau_cal, k, config, rng)
        if pi is None:
            break
        if k == 1 or r2 - prev_r2 >= config.cluster_r2_tol:
            best_pi = pi
            best_gamma = gamma
            best_r2 = r2
            selected_k = k
        else:
            break
        prev_r2 = r2
    if best_pi is None or best_gamma is None:
        raise RuntimeError("PCP factorization failed")
    return best_pi, best_gamma, best_r2, selected_k


def _factorize_tau(
    tau: np.ndarray,
    K: int,
    config: PCPConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray | None, np.ndarray | None, float]:
    n, s = tau.shape
    if K == 0:
        return None, None, float("nan")
    km = KMeans(n_clusters=K, n_init=10, random_state=int(rng.integers(0, 2**32 - 1)))
    km.fit(tau)
    gamma = km.cluster_centers_.astype(float)
    gamma = np.clip(gamma, config.clip_eps, 1.0 - config.clip_eps)
    gamma = np.maximum.accumulate(gamma, axis=1)
    pi = rng.random((n, K))
    pi = pi / np.sum(pi, axis=1, keepdims=True)
    prev_loss = np.inf
    eps = 1e-12
    for _ in range(config.factor_max_iter):
        numerator = tau @ gamma.T
        denominator = (pi @ gamma) @ gamma.T + eps
        pi *= numerator / denominator
        pi = np.clip(pi, config.clip_eps, None)
        pi = pi / np.sum(pi, axis=1, keepdims=True)
        numerator_g = pi.T @ tau
        denominator_g = (pi.T @ pi) @ gamma + eps
        gamma *= numerator_g / denominator_g
        gamma = np.clip(gamma, config.clip_eps, 1.0 - config.clip_eps)
        gamma = np.maximum.accumulate(gamma, axis=1)
        residual = tau - pi @ gamma
        loss = float(np.sum(residual ** 2))
        if not np.isfinite(loss):
            break
        if abs(prev_loss - loss) <= config.factor_tol * max(1.0, prev_loss):
            break
        prev_loss = loss
    total_var = float(np.sum((tau - np.mean(tau, axis=0, keepdims=True)) ** 2))
    if total_var == 0.0:
        r2 = 1.0
    else:
        r2 = 1.0 - float(np.sum((tau - pi @ gamma) ** 2)) / total_var
    return pi, gamma, r2


def _select_precision(
    pi_cal: np.ndarray,
    log_pi_cal: np.ndarray,
    alpha: float,
    config: PCPConfig,
    rng: np.random.Generator,
) -> Tuple[int, Tuple[int, ...]]:
    n = pi_cal.shape[0]
    for m in config.precision_grid:
        w_inf_vals = []
        for _ in range(config.precision_trials):
            idx = int(rng.integers(0, n))
            pi_test = np.clip(pi_cal[idx], config.clip_eps, 1.0)
            pi_test /= np.sum(pi_test)
            log_pi_test = np.log(pi_test)
            L = rng.multinomial(m, pi_test)
            _, w_inf = _weights_from_counts_static(log_pi_cal, log_pi_test, L)
            w_inf_vals.append(w_inf)
        mean_w_inf = float(np.mean(w_inf_vals))
        max_w_inf = float(np.max(w_inf_vals))
        if mean_w_inf <= alpha * 0.6 and max_w_inf <= alpha * 1.2:
            fallbacks = tuple(sorted([val for val in config.precision_grid if val < m], reverse=True))
            return m, fallbacks
    fallbacks = tuple(sorted(config.precision_grid[:-1], reverse=True))
    return config.precision_grid[-1], fallbacks


def _weights_from_counts_static(
    log_pi_cal: np.ndarray,
    log_pi_test: np.ndarray,
    L: np.ndarray,
) -> Tuple[np.ndarray, float]:
    log_phi = log_pi_cal @ L
    log_phi_inf = float(np.dot(L, log_pi_test))
    max_log = max(float(np.max(log_phi)), log_phi_inf)
    phi = np.exp(log_phi - max_log)
    phi_inf = float(np.exp(log_phi_inf - max_log))
    denom = float(np.sum(phi) + phi_inf)
    if denom == 0.0:
        return np.zeros_like(log_phi), 1.0
    weights = phi / denom
    return weights, phi_inf / denom


def _project_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    if v.ndim != 1:
        raise ValueError("simplex projection expects a 1-D array")
    if v.size == 1:
        return np.array([1.0], dtype=float)
    sorted_v = np.sort(v)[::-1]
    css = np.cumsum(sorted_v)
    rho = np.nonzero(sorted_v + (1 - css) / (np.arange(v.size) + 1) > 0)[0]
    if rho.size == 0:
        theta = (css[0] - 1) / 1
    else:
        j = rho[-1]
        theta = (css[j] - 1) / (j + 1)
    w = np.maximum(v - theta, 0.0)
    sum_w = np.sum(w)
    if sum_w == 0.0:
        return np.full_like(w, 1.0 / w.size)
    return w / sum_w


def _weighted_quantile(
    scores: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    w_inf: float,
) -> float:
    if w_inf > alpha - 1e-12:
        return float(np.inf)
    order = np.argsort(scores)
    sorted_scores = scores[order]
    sorted_weights = weights[order]
    cumsum = np.cumsum(sorted_weights)
    threshold = 1.0 - alpha
    if cumsum.size == 0 or cumsum[-1] + 1e-12 < threshold:
        return float(np.inf)
    idx = int(np.searchsorted(cumsum, threshold, side="left"))
    if idx >= sorted_scores.size:
        return float(np.inf)
    return float(sorted_scores[idx])