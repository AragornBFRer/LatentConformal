"""Experiment orchestration for latent conformal simulations."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, PCPVariantOptions, RunConfig, iter_run_configs, load_config
from .conformal import split_conformal
from .data_gen import DatasetSplit, generate_data
from .doc_em import fit_doc_em, responsibilities_from_r, responsibilities_with_y
from .metrics import avg_length, coverage, cross_entropy, mean_max_tau, z_feature_mse
from .predictors import EMSoftPredictor, IgnoreZPredictor, OracleZPredictor, XRZYPredictor
from .utils import ensure_dir, rng_from_seed
from .pcp import PCPConfig, train_pcp
from .pcp_membership import MembershipPCPModel

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm may be unavailable at runtime
    tqdm = None


def _combo_seed(run_cfg: RunConfig) -> int:
    payload = (
        f"{run_cfg.seed}|{run_cfg.K}|{run_cfg.delta}|{run_cfg.rho}|{run_cfg.sigma_y}|"
        f"{run_cfg.b_scale}|{int(run_cfg.use_x_in_em)}"
    ).encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _variant_to_pcp_config(variant: PCPVariantOptions) -> PCPConfig:
    return PCPConfig(
        n_thresholds=variant.n_thresholds,
        logistic_cv_folds=variant.logistic_cv_folds,
        max_clusters=variant.max_clusters,
        cluster_r2_tol=variant.cluster_r2_tol,
        factor_max_iter=variant.factor_max_iter,
        factor_tol=variant.factor_tol,
        precision_grid=variant.precision_grid,
        precision_trials=variant.precision_trials,
        clip_eps=variant.clip_eps,
        proj_lr=variant.proj_lr,
        proj_max_iter=variant.proj_max_iter,
        proj_tol=variant.proj_tol,
    )


def _concat_features(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    X_arr = np.asarray(X, dtype=float)
    R_arr = np.asarray(R, dtype=float)
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    if R_arr.ndim == 1:
        R_arr = R_arr.reshape(-1, 1)
    parts = []
    if X_arr.size:
        parts.append(X_arr)
    if R_arr.size:
        parts.append(R_arr)
    if not parts:
        raise ValueError("At least one of X or R must provide features for PCP")
    base_n = parts[0].shape[0]
    for part in parts[1:]:
        if part.shape[0] != base_n:
            raise ValueError("Feature blocks must share the same number of samples")
    return parts[0] if len(parts) == 1 else np.hstack(parts)


def _smooth_memberships(pi: np.ndarray, smoothing: float) -> np.ndarray:
    arr = np.asarray(pi, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Membership matrix must be 2D")
    if smoothing <= 0.0:
        return arr
    smoothing = min(max(float(smoothing), 0.0), 1.0 - 1e-9)
    K = arr.shape[1]
    uniform = 1.0 / K if K > 0 else 0.0
    arr = (1.0 - smoothing) * arr + smoothing * uniform
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return arr / row_sums


def _run_single(cfg: ExperimentConfig, run_cfg: RunConfig) -> Dict[str, float]:
    rng = rng_from_seed(_combo_seed(run_cfg))
    train, cal, test, true_params = generate_data(
        cfg.dgp_cfg,
        run_cfg,
        cfg.global_cfg.n_train,
        cfg.global_cfg.n_cal,
        cfg.global_cfg.n_test,
        rng,
    )

    alpha = np.asarray(true_params["alpha"], dtype=float)
    sigma = np.asarray(true_params["sigma"], dtype=float)
    pi_prior = np.asarray(true_params["pi"], dtype=float)

    em_params, tau_train = fit_doc_em(
        train.X,
        train.R,
        train.Y,
        alpha=alpha,
        sigma=sigma,
        pi=pi_prior,
        max_iter=cfg.em_cfg.max_iter,
        tol=cfg.em_cfg.tol,
        rng=rng,
    )

    tau_cal = responsibilities_with_y(cal.X, cal.R, cal.Y, em_params)
    tau_test = responsibilities_from_r(test.R, em_params)

    means_r_true = true_params["means_r"]
    z_star_train = means_r_true[train.Z]
    z_star_cal = means_r_true[cal.Z]
    z_star_test = means_r_true[test.Z]

    mu_hat_r = em_params.mu_r
    z_soft_train = tau_train @ mu_hat_r
    z_soft_cal = tau_cal @ mu_hat_r
    z_soft_test = tau_test @ mu_hat_r

    oracle = OracleZPredictor(ridge_alpha=cfg.model_cfg.ridge_alpha_oracle).fit(
        train.X, train.Y, z_star=z_star_train
    )
    soft = EMSoftPredictor(ridge_alpha=cfg.model_cfg.ridge_alpha_soft).fit(
        train.X, train.Y, z_feat=z_soft_train
    )
    ignore = IgnoreZPredictor(ridge_alpha=cfg.model_cfg.ridge_alpha_ignore).fit(train.X, train.Y)

    scores = {
        "oracle": np.abs(cal.Y - oracle.predict_mean(cal.X, z_star=z_star_cal)),
        "soft": np.abs(cal.Y - soft.predict_mean(cal.X, z_feat=z_soft_cal)),
        "ignore": np.abs(cal.Y - ignore.predict_mean(cal.X)),
    }

    qhat = {name: split_conformal(vals, cfg.global_cfg.alpha) for name, vals in scores.items()}

    preds_test = {
        "oracle": oracle.predict_mean(test.X, z_star=z_star_test),
        "soft": soft.predict_mean(test.X, z_feat=z_soft_test),
        "ignore": ignore.predict_mean(test.X),
    }

    xrzy = XRZYPredictor(
        n_estimators=cfg.model_cfg.pcp_rf_n_estimators,
        max_depth=cfg.model_cfg.pcp_rf_max_depth,
        min_samples_leaf=cfg.model_cfg.pcp_rf_min_samples_leaf,
        n_jobs=cfg.model_cfg.pcp_rf_n_jobs,
        random_state=int(rng.integers(0, 2**32 - 1)),
    ).fit(train.X, train.Y, R=train.R)
    mu_cal_pcp = xrzy.predict_mean(cal.X, R=cal.R)
    mu_test_pcp = xrzy.predict_mean(test.X, R=test.R)
    scores_pcp = np.abs(cal.Y - mu_cal_pcp)

    results = {}
    for name in preds_test:
        q = qhat[name]
        mu = preds_test[name]
        results[f"coverage_{name}"] = coverage(test.Y, mu, q)
        results[f"length_{name}"] = avg_length(q)

    length_oracle = results["length_oracle"]
    denom = length_oracle if length_oracle != 0 else 1e-12
    results["len_ratio_soft"] = results["length_soft"] / denom
    results["len_ratio_ignore"] = results["length_ignore"] / denom

    # XRZY-specific PCP (R-only membership)
    results["coverage_pcp"] = np.nan
    results["length_pcp"] = np.nan
    results["len_ratio_pcp"] = np.nan
    results["pcp_frac_inf"] = np.nan
    results["pcp_precision"] = np.nan
    results["pcp_clusters"] = np.nan
    results["pcp_cluster_r2"] = np.nan
    qhat_pcp = np.full(test.Y.shape, np.nan)
    if cfg.pcp_cfg.xrzy.enabled:
        pcp_model = train_pcp(
            cal.R,
            scores_pcp,
            alpha=cfg.global_cfg.alpha,
            rng=rng,
            config=_variant_to_pcp_config(cfg.pcp_cfg.xrzy),
        )
        qhat_pcp = pcp_model.quantiles(test.R, rng)
        results["coverage_pcp"] = coverage(test.Y, mu_test_pcp, qhat_pcp)
        results["length_pcp"] = avg_length(qhat_pcp)
        results["len_ratio_pcp"] = results["length_pcp"] / denom
        results["pcp_frac_inf"] = float(np.mean(~np.isfinite(qhat_pcp)))
        results["pcp_precision"] = float(pcp_model.precision)
        results["pcp_clusters"] = float(pcp_model.n_clusters)
        results["pcp_cluster_r2"] = float(pcp_model.cluster_r2)

    # PCP-base using (X, R) features
    results["coverage_pcp_base"] = np.nan
    results["length_pcp_base"] = np.nan
    results["len_ratio_pcp_base"] = np.nan
    results["pcp_base_frac_inf"] = np.nan
    results["pcp_base_precision"] = np.nan
    results["pcp_base_clusters"] = np.nan
    results["pcp_base_cluster_r2"] = np.nan
    if cfg.pcp_cfg.base.enabled:
        base_features_cal = _concat_features(cal.X, cal.R)
        base_features_test = _concat_features(test.X, test.R)
        pcp_base_model = train_pcp(
            base_features_cal,
            scores_pcp,
            alpha=cfg.global_cfg.alpha,
            rng=rng,
            config=_variant_to_pcp_config(cfg.pcp_cfg.base),
        )
        qhat_pcp_base = pcp_base_model.quantiles(base_features_test, rng)
        results["coverage_pcp_base"] = coverage(test.Y, mu_test_pcp, qhat_pcp_base)
        results["length_pcp_base"] = avg_length(qhat_pcp_base)
        results["len_ratio_pcp_base"] = results["length_pcp_base"] / denom
        results["pcp_base_frac_inf"] = float(np.mean(~np.isfinite(qhat_pcp_base)))
        results["pcp_base_precision"] = float(pcp_base_model.precision)
        results["pcp_base_clusters"] = float(pcp_base_model.n_clusters)
        results["pcp_base_cluster_r2"] = float(pcp_base_model.cluster_r2)

    # EM-PCP using memberships from the doc-style EM fit
    results["coverage_em_pcp"] = np.nan
    results["length_em_pcp"] = np.nan
    results["len_ratio_em_pcp"] = np.nan
    results["em_pcp_frac_inf"] = np.nan
    results["em_pcp_precision"] = np.nan
    results["em_pcp_clusters"] = np.nan
    results["em_pcp_cluster_r2"] = np.nan
    if cfg.pcp_cfg.em.enabled:
        pi_cal_em = tau_cal.copy()
        pi_test_em = tau_test.copy()
        smoothing = getattr(cfg.pcp_cfg.em, "membership_smoothing", 0.0)
        if smoothing > 0.0:
            pi_cal_em = _smooth_memberships(pi_cal_em, smoothing)
            pi_test_em = _smooth_memberships(pi_test_em, smoothing)
        membership_cfg = _variant_to_pcp_config(cfg.pcp_cfg.em)
        em_model = MembershipPCPModel.fit(
            pi_cal_em,
            scores_pcp,
            alpha=cfg.global_cfg.alpha,
            rng=rng,
            config=membership_cfg,
        )
        qhat_em = em_model.quantiles(pi_test_em, rng)
        results["coverage_em_pcp"] = coverage(test.Y, mu_test_pcp, qhat_em)
        results["length_em_pcp"] = avg_length(qhat_em)
        results["len_ratio_em_pcp"] = results["length_em_pcp"] / denom
        results["em_pcp_frac_inf"] = float(np.mean(~np.isfinite(qhat_em)))
        results["em_pcp_precision"] = float(em_model.precision)
        results["em_pcp_clusters"] = float(pi_cal_em.shape[1])
        results["em_pcp_cluster_r2"] = np.nan

    results["mean_max_tau"] = mean_max_tau(tau_test)
    results["cross_entropy"] = cross_entropy(tau_test, test.Z)
    results["z_feature_mse"] = z_feature_mse(z_soft_test, z_star_test)

    results.update(
        {
            "seed": run_cfg.seed,
            "K": run_cfg.K,
            "delta": run_cfg.delta,
            "rho": run_cfg.rho,
            "sigma_y": run_cfg.sigma_y,
            "b_scale": run_cfg.b_scale,
            "use_x_in_em": run_cfg.use_x_in_em,
            "em_converged": em_params.converged,
            "em_iter": em_params.n_iter,
            "em_loglik": em_params.log_likelihood,
        }
    )

    return results

def _with_progress(seq: Iterable[RunConfig], total: int | None = None) -> Iterable[RunConfig]:
    if tqdm is None:
        return seq
    return tqdm(seq, total=total, desc="Running trials", unit="trial")


def run_experiment(
    cfg_path: str,
    *,
    seeds: Sequence[int] | None = None,
    results_path_override: str | Path | None = None,
) -> pd.DataFrame:
    cfg = load_config(cfg_path)
    if seeds is not None:
        seed_list = [int(s) for s in seeds]
        if not seed_list:
            raise ValueError("Seed override must contain at least one value")
        seed_list = list(dict.fromkeys(seed_list))
        cfg = replace(cfg, global_cfg=replace(cfg.global_cfg, seeds=seed_list))
    if results_path_override is not None:
        cfg = replace(cfg, io_cfg=replace(cfg.io_cfg, results_csv=str(results_path_override)))
    key_cols = ["seed", "K", "delta", "rho", "sigma_y", "b_scale", "use_x_in_em"]
    rows: List[Dict[str, float]] = []
    run_cfgs = list(iter_run_configs(cfg))
    valid_keys = set()
    for run_cfg in _with_progress(run_cfgs, total=len(run_cfgs)):
        rows.append(_run_single(cfg, run_cfg))
        valid_keys.add(
            (
                run_cfg.seed,
                run_cfg.K,
                run_cfg.delta,
                run_cfg.rho,
                run_cfg.sigma_y,
                run_cfg.b_scale,
                run_cfg.use_x_in_em,
            )
        )

    df = pd.DataFrame(rows)
    df["__key"] = list(zip(*(df[col] for col in key_cols)))

    results_path = Path(cfg.io_cfg.results_csv)
    ensure_dir(results_path)
    if results_path.exists():
        prev = pd.read_csv(results_path)
        prev["__key"] = list(zip(*(prev[col] for col in key_cols)))
        df = pd.concat([prev, df], ignore_index=True)

    df = df[df["__key"].isin(valid_keys)].copy()
    df.drop_duplicates(subset="__key", keep="last", inplace=True)
    df.sort_values(key_cols, inplace=True)
    df.drop(columns="__key", inplace=True)
    for legacy_col in ["len_gap_soft", "len_gap_ignore"]:
        if legacy_col in df.columns:
            df.drop(columns=legacy_col, inplace=True)
    df.to_csv(results_path, index=False)
    df.attrs["results_path"] = str(results_path)
    return df
