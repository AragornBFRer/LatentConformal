"""Experiment orchestration for latent conformal simulations."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .config import ExperimentConfig, RunConfig, iter_run_configs, load_config
from .conformal import split_conformal
from .data_gen import DatasetSplit, generate_data
from .em_gmm import fit_gmm_em, gmm_responsibilities
from .metrics import avg_length, coverage, cross_entropy, mean_max_tau, z_feature_mse
from .predictors import EMSoftPredictor, IgnoreZPredictor, OracleZPredictor, XRZYPredictor
from .utils import ensure_dir, rng_from_seed
from .pcp import PCPConfig, train_pcp

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


def _build_em_features(split: DatasetSplit, use_x: bool) -> np.ndarray:
    if use_x:
        return np.hstack([split.R, split.X])
    return split.R


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

    K_fit = cfg.em_cfg.K_fit or run_cfg.K
    S_train = _build_em_features(train, run_cfg.use_x_in_em)
    S_cal = _build_em_features(cal, run_cfg.use_x_in_em)
    S_test = _build_em_features(test, run_cfg.use_x_in_em)

    em_params = fit_gmm_em(
        S_train,
        K_fit,
        cov_type=cfg.em_cfg.cov_type_fit,
        max_iter=cfg.em_cfg.max_iter,
        tol=cfg.em_cfg.tol,
        reg_covar=cfg.em_cfg.reg_covar,
        init=cfg.em_cfg.init,
        n_init=cfg.em_cfg.n_init,
        r_dim=cfg.dgp_cfg.d_R,
        rng=rng,
    )

    tau_train = gmm_responsibilities(S_train, em_params)
    tau_cal = gmm_responsibilities(S_cal, em_params)
    tau_test = gmm_responsibilities(S_test, em_params)

    means_r_true = true_params["means_r"]
    z_star_train = means_r_true[train.Z]
    z_star_cal = means_r_true[cal.Z]
    z_star_test = means_r_true[test.Z]

    em_means = em_params.means
    if run_cfg.use_x_in_em:
        mu_hat_r = em_means[:, : cfg.dgp_cfg.d_R]
    else:
        mu_hat_r = em_means
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

    xrzy = XRZYPredictor(ridge_alpha=cfg.model_cfg.ridge_alpha_xrzy).fit(train.X, train.Y, R=train.R)
    mu_cal_pcp = xrzy.predict_mean(cal.X, R=cal.R)
    mu_test_pcp = xrzy.predict_mean(test.X, R=test.R)
    scores_pcp = np.abs(cal.Y - mu_cal_pcp)
    pcp_model = train_pcp(
        cal.R,
        scores_pcp,
        alpha=cfg.global_cfg.alpha,
        rng=rng,
        config=PCPConfig(),
    )
    qhat_pcp = pcp_model.quantiles(test.R, rng)

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

    results["coverage_pcp"] = coverage(test.Y, mu_test_pcp, qhat_pcp)
    results["length_pcp"] = avg_length(qhat_pcp)
    results["len_ratio_pcp"] = results["length_pcp"] / denom
    results["pcp_frac_inf"] = float(np.mean(~np.isfinite(qhat_pcp)))
    results["pcp_precision"] = float(pcp_model.precision)
    results["pcp_clusters"] = float(pcp_model.n_clusters)
    results["pcp_cluster_r2"] = float(pcp_model.cluster_r2)

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
