"""PCP weighting using pre-computed membership probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import ArrayLike

from .pcp import PCPConfig, _select_precision, _weighted_quantile


@dataclass
class MembershipPCPModel:
    scores_cal: np.ndarray
    pi_cal: np.ndarray
    log_pi_cal: np.ndarray
    alpha: float
    clip_eps: float
    precision: int
    precision_fallbacks: Tuple[int, ...]
    n_clusters: int

    @classmethod
    def fit(
        cls,
        pi_cal: ArrayLike,
        scores_cal: ArrayLike,
        *,
        alpha: float,
        rng: np.random.Generator,
        config: PCPConfig | None = None,
    ) -> "MembershipPCPModel":
        if config is None:
            config = PCPConfig()
        pi = np.asarray(pi_cal, dtype=float)
        if pi.ndim != 2:
            raise ValueError("pi_cal must be a 2D array")
        if pi.shape[1] == 0:
            raise ValueError("pi_cal must have at least one cluster column")
        pi = np.clip(pi, config.clip_eps, None)
        row_sums = pi.sum(axis=1, keepdims=True)
        zero_mask = row_sums.squeeze() == 0.0
        if np.any(zero_mask):
            pi[zero_mask] = 1.0
            row_sums = pi.sum(axis=1, keepdims=True)
        pi /= row_sums
        log_pi = np.log(np.clip(pi, config.clip_eps, 1.0))
        scores = np.asarray(scores_cal, dtype=float)
        if scores.ndim != 1:
            raise ValueError("scores_cal must be one-dimensional")
        precision, fallbacks = _select_precision(pi, log_pi, alpha, config, rng)
        return cls(
            scores_cal=scores,
            pi_cal=pi,
            log_pi_cal=log_pi,
            alpha=alpha,
            clip_eps=config.clip_eps,
            precision=precision,
            precision_fallbacks=fallbacks,
            n_clusters=pi.shape[1],
        )

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
        pi = np.asarray(pi_test, dtype=float)
        if pi.ndim != 1 or pi.size != self.n_clusters:
            raise ValueError("pi_test must be a 1D array matching the number of clusters")
        pi = np.clip(pi, self.clip_eps, None)
        total = float(np.sum(pi))
        if not np.isfinite(total) or total <= 0.0:
            pi = np.full(self.n_clusters, 1.0 / self.n_clusters)
        else:
            pi /= total
        log_pi_test = np.log(np.clip(pi, self.clip_eps, 1.0))
        candidates = (self.precision,) + tuple(self.precision_fallbacks)
        for m in candidates:
            if m <= 0:
                continue
            L = rng.multinomial(m, pi)
            weights, w_inf = self._weights_from_counts(L, log_pi_test)
            if w_inf > self.alpha + 1e-12:
                continue
            q = _weighted_quantile(self.scores_cal, weights, self.alpha, w_inf)
            if np.isfinite(q):
                return float(q)
        return float(np.inf)

    def quantiles(self, pi_test: ArrayLike, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        arr = np.asarray(pi_test, dtype=float)
        if arr.ndim == 1:
            return np.array([self._quantile_from_membership(arr, rng)], dtype=float)
        if arr.ndim != 2 or arr.shape[1] != self.n_clusters:
            raise ValueError("pi_test must be a 1D or 2D array with matching cluster dimension")
        out = np.empty(arr.shape[0], dtype=float)
        for idx in range(arr.shape[0]):
            out[idx] = self._quantile_from_membership(arr[idx], rng)
        return out