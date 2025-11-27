"""Experiment orchestration for latent conformal simulations."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import (
    ExperimentConfig,
    PCPVariantOptions,
    RunConfig,
    iter_run_configs,
    load_config,
)
from .conformal import split_conformal
from .data_gen import DatasetSplit, generate_data
from .doc_em import fit_doc_em, responsibilities_from_r, responsibilities_with_y
from .metrics import avg_length, coverage, cross_entropy, mean_max_tau, z_feature_mse
from .predictors import (
    RandomForestMeanPredictor,
    RandomForestParams,
    RandomForestQuantilePredictor,
)
from .utils import ensure_dir, rng_from_seed
from .pcp import PCPConfig, train_pcp
from .pcp_membership import MembershipPCPModel

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm may be unavailable at runtime
    tqdm = None


PCP_INPUT_SPECS = {
    "pcp_xr": ("X", "R"),
    "pcp_xz": ("X", "Z"),
    "pcp_xzhat": ("X", "Zhat"),
    "pcp_xrzhat": ("X", "R", "Zhat"),
}


def _combo_seed(run_cfg: RunConfig) -> int:
    payload = (
        f"{run_cfg.seed}|{run_cfg.K}|{run_cfg.delta}|{int(run_cfg.use_x_in_em)}"
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


def _sample_rf_params(model_cfg, rng: np.random.Generator) -> RandomForestParams:
    return RandomForestParams(
        n_estimators=model_cfg.rf_n_estimators,
        max_depth=model_cfg.rf_max_depth,
        min_samples_leaf=model_cfg.rf_min_samples_leaf,
        n_jobs=model_cfg.rf_n_jobs,
        random_state=int(rng.integers(0, 2**31 - 1)),
    )


def _ensure_feature_array(arr: np.ndarray) -> np.ndarray:
    block = np.asarray(arr, dtype=float)
    if block.ndim == 1:
        block = block.reshape(-1, 1)
    return block


def _stack_feature_components(
    split: DatasetSplit,
    components: tuple[str, ...],
    *,
    K: int,
    tau_split: np.ndarray | None = None,
) -> np.ndarray:
    parts: List[np.ndarray] = []
    for comp in components:
        if comp == "X":
            block = _ensure_feature_array(split.X)
        elif comp == "R":
            block = _ensure_feature_array(split.R)
        elif comp == "Z":
            block = np.eye(K)[np.asarray(split.Z, dtype=int)]
        elif comp == "Zhat":
            if tau_split is None:
                raise ValueError("Zhat features require membership probabilities for the split")
            block = _ensure_feature_array(tau_split)
        else:
            raise ValueError(f"Unknown feature component '{comp}'")
        if block.size:
            parts.append(block)
    if not parts:
        raise ValueError("At least one non-empty component is required for PCP features")
    return np.hstack(parts)


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

    em_params, _ = fit_doc_em(
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

    tau_train = responsibilities_with_y(train.X, train.R, train.Y, em_params)
    tau_cal = responsibilities_with_y(cal.X, cal.R, cal.Y, em_params)
    tau_test = responsibilities_from_r(test.R, em_params)

    mu_hat_r = em_params.mu_r
    means_r_true = true_params["means_r"]
    z_star_test = means_r_true[test.Z]
    z_soft_test = tau_test @ mu_hat_r

    features_train = _concat_features(train.X, train.R)
    features_cal = _concat_features(cal.X, cal.R)
    features_test = _concat_features(test.X, test.R)

    splits = {"train": train, "cal": cal, "test": test}
    taus = {"train": tau_train, "cal": tau_cal, "test": tau_test}
    pcp_feature_bundles: Dict[str, Dict[str, np.ndarray]] = {}
    for name, components in PCP_INPUT_SPECS.items():
        needs_hat = "Zhat" in components
        bundle: Dict[str, np.ndarray] = {}
        for split_name, split_data in splits.items():
            tau_split = taus[split_name] if needs_hat else None
            bundle[split_name] = _stack_feature_components(
                split_data,
                components,
                K=run_cfg.K,
                tau_split=tau_split,
            )
        pcp_feature_bundles[name] = bundle

    # CQR-ignoreZ interval via RandomForest quantile regression
    alpha_lo = 0.5 * cfg.global_cfg.alpha
    alpha_hi = 1.0 - alpha_lo
    quantile_levels = (alpha_lo, alpha_hi)
    cqr_params = _sample_rf_params(cfg.model_cfg, rng)
    cqr_model = RandomForestQuantilePredictor(cqr_params).fit(features_train, train.Y)
    cal_bounds = cqr_model.predict_quantiles(features_cal, quantile_levels)
    scores_cqr = np.maximum(cal_bounds[:, 0] - cal.Y, cal.Y - cal_bounds[:, 1])
    q_cqr = split_conformal(scores_cqr, cfg.global_cfg.alpha)
    test_bounds = cqr_model.predict_quantiles(features_test, quantile_levels)
    lower_cqr = test_bounds[:, 0] - q_cqr
    upper_cqr = test_bounds[:, 1] + q_cqr
    center_cqr = 0.5 * (lower_cqr + upper_cqr)
    radius_cqr = 0.5 * (upper_cqr - lower_cqr)

    mu_cal_joint = cqr_model.predict_mean(features_cal)
    mu_test_joint = cqr_model.predict_mean(features_test)
    scores_joint = np.abs(cal.Y - mu_cal_joint)

    results = {}
    results["coverage_cqr_ignore"] = coverage(test.Y, center_cqr, radius_cqr)
    results["length_cqr_ignore"] = avg_length(radius_cqr)

    pcp_variant_names = list(PCP_INPUT_SPECS.keys())
    for name in pcp_variant_names:
        results[f"coverage_{name}"] = np.nan
        results[f"length_{name}"] = np.nan
        results[f"{name}_frac_inf"] = np.nan
        results[f"{name}_precision"] = np.nan
        results[f"{name}_clusters"] = np.nan
        results[f"{name}_cluster_r2"] = np.nan

    if cfg.pcp_cfg.base.enabled:
        pcp_config = _variant_to_pcp_config(cfg.pcp_cfg.base)
        for name in pcp_variant_names:
            bundle = pcp_feature_bundles[name]
            rf_params = _sample_rf_params(cfg.model_cfg, rng)
            predictor = RandomForestMeanPredictor(rf_params).fit(bundle["train"], train.Y)
            mu_cal = predictor.predict_mean(bundle["cal"])
            mu_test_variant = predictor.predict_mean(bundle["test"])
            scores_variant = np.abs(cal.Y - mu_cal)
            pcp_model = train_pcp(
                bundle["cal"],
                scores_variant,
                alpha=cfg.global_cfg.alpha,
                rng=rng,
                config=pcp_config,
            )
            qhat = pcp_model.quantiles(bundle["test"], rng)
            results[f"coverage_{name}"] = coverage(test.Y, mu_test_variant, qhat)
            results[f"length_{name}"] = avg_length(qhat)
            results[f"{name}_frac_inf"] = float(np.mean(~np.isfinite(qhat)))
            results[f"{name}_precision"] = float(pcp_model.precision)
            results[f"{name}_clusters"] = float(pcp_model.n_clusters)
            results[f"{name}_cluster_r2"] = float(pcp_model.cluster_r2)

    # EM-PCP using memberships from the doc-style EM fit
    results["coverage_em_pcp"] = np.nan
    results["length_em_pcp"] = np.nan
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
            scores_joint,
            alpha=cfg.global_cfg.alpha,
            rng=rng,
            config=membership_cfg,
        )
        qhat_em = em_model.quantiles(pi_test_em, rng)
        results["coverage_em_pcp"] = coverage(test.Y, mu_test_joint, qhat_em)
        results["length_em_pcp"] = avg_length(qhat_em)
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
    key_cols = ["seed", "K", "delta", "use_x_in_em"]
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
    legacy_cols = [
        "len_gap_soft",
        "len_gap_ignore",
        "coverage_oracle",
        "length_oracle",
        "coverage_soft",
        "length_soft",
        "coverage_pcp",
        "length_pcp",
        "len_ratio_soft",
        "len_ratio_ignore",
        "len_ratio_pcp",
        "len_ratio_pcp_base",
        "len_ratio_em_pcp",
    ]
    for legacy_col in legacy_cols:
        if legacy_col in df.columns:
            df.drop(columns=legacy_col, inplace=True)
    df.to_csv(results_path, index=False)
    df.attrs["results_path"] = str(results_path)
    return df
