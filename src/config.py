"""Configuration loading and grid expansion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class GlobalConfig:
    seeds: List[int]
    n_train: int
    n_cal: int
    n_test: int
    alpha: float


@dataclass(frozen=True)
class DGPConfig:
    K_list: List[int]
    d_R: int
    d_X: int
    use_S: str
    delta_list: List[float]
    sigma_s: float
    cov_type_true: str
    rho_list: List[float]
    sigma_y_list: List[float]
    b_scale_list: List[float]
    mu_x_shift: float
    alpha: List[float]
    sigma: List[float]
    mu_r: List[List[float]]
    eta0: float
    eta: List[float]


@dataclass(frozen=True)
class EMConfig:
    K_fit: Optional[int]
    cov_type_fit: str
    reg_covar: float
    max_iter: int
    tol: float
    init: str
    n_init: int
    use_X_in_em: List[bool]


@dataclass(frozen=True)
class ModelConfig:
    ridge_alpha_oracle: float
    ridge_alpha_soft: float
    ridge_alpha_ignore: float
    ridge_alpha_xrzy: float
    pcp_rf_n_estimators: int
    pcp_rf_max_depth: Optional[int]
    pcp_rf_min_samples_leaf: int
    pcp_rf_n_jobs: int


@dataclass(frozen=True)
class IOConfig:
    results_csv: str
    artifacts_dir: str


@dataclass(frozen=True)
class ExperimentConfig:
    global_cfg: GlobalConfig
    dgp_cfg: DGPConfig
    em_cfg: EMConfig
    model_cfg: ModelConfig
    io_cfg: IOConfig
    pcp_cfg: "PCPFamilyConfig"


@dataclass(frozen=True)
class RunConfig:
    seed: int
    K: int
    delta: float
    rho: float
    sigma_y: float
    b_scale: float
    use_x_in_em: bool


@dataclass(frozen=True)
class PCPVariantOptions:
    enabled: bool
    n_thresholds: int
    logistic_cv_folds: int
    max_clusters: int
    cluster_r2_tol: float
    factor_max_iter: int
    factor_tol: float
    precision_grid: Tuple[int, ...]
    precision_trials: int
    clip_eps: float
    proj_lr: float
    proj_max_iter: int
    proj_tol: float
    membership_smoothing: float


@dataclass(frozen=True)
class PCPFamilyConfig:
    xrzy: PCPVariantOptions
    base: PCPVariantOptions
    em: PCPVariantOptions


def _expand_range(value) -> List:
    if isinstance(value, dict):
        if "values" in value:
            vals = value["values"]
            return list(vals) if isinstance(vals, (list, tuple, set)) else [vals]
        if "start" in value and ("stop" in value or "count" in value):
            start = value.get("start")
            if start is None:
                raise ValueError("Range spec missing 'start'")
            step = value.get("step", 1)
            if step == 0:
                raise ValueError("Range spec requires non-zero 'step'")
            count = value.get("count")
            if count is not None:
                count = int(count)
                seq = [start + step * i for i in range(count)]
            else:
                stop = value.get("stop")
                if stop is None:
                    raise ValueError("Range spec missing 'stop'")
                seq = []
                current = start
                forward = step > 0
                limit = int(value.get("max_terms", 100000))
                cmp = (lambda c: c < stop) if forward else (lambda c: c > stop)
                for _ in range(limit):
                    if not cmp(current):
                        break
                    seq.append(current)
                    current = current + step
                else:
                    raise ValueError("Range spec exceeded max_terms without reaching stop")
            if all(abs(float(x) - round(float(x))) < 1e-12 for x in seq):
                return [int(round(float(x))) for x in seq]
            return [float(x) for x in seq]
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def _parse_precision_grid(raw_grid, default: Tuple[int, ...]) -> Tuple[int, ...]:
    if raw_grid is None:
        return default
    if isinstance(raw_grid, (list, tuple)):
        return tuple(int(v) for v in raw_grid)
    return tuple([int(raw_grid)])


def _parse_cluster_scalar_list(
    raw_values,
    target_len: int,
    label: str,
    default_factory,
) -> List[float]:
    if target_len <= 0:
        return []
    if raw_values is None:
        return [float(default_factory(idx)) for idx in range(target_len)]
    seq = list(raw_values)
    if len(seq) < target_len:
        raise ValueError(f"'{label}' requires at least {target_len} entries, got {len(seq)}")
    return [float(seq[idx]) for idx in range(target_len)]


def _parse_cluster_vector_list(
    raw_values,
    target_len: int,
    dim: int,
    label: str,
    default_factory,
) -> List[List[float]]:
    if target_len <= 0:
        return []
    if raw_values is None:
        return [list(default_factory(idx)) for idx in range(target_len)]
    seq = list(raw_values)
    if len(seq) < target_len:
        raise ValueError(f"'{label}' requires at least {target_len} rows, got {len(seq)}")
    result: List[List[float]] = []
    for idx in range(target_len):
        entry = seq[idx]
        if isinstance(entry, (list, tuple)):
            row = [float(v) for v in entry]
        else:
            row = [float(entry)]
        if len(row) == 1 and dim > 1:
            row = row + [0.0] * (dim - 1)
        if len(row) != dim:
            raise ValueError(
                f"Each row in '{label}' must have {dim} values; row {idx} has {len(row)}"
            )
        result.append(row)
    return result


def _parse_eta_vector(raw_values, dim: int, default: Iterable[float]) -> List[float]:
    seq = list(default) if raw_values is None else list(raw_values)
    if len(seq) < dim:
        if raw_values is None:
            seq = seq + [0.0] * (dim - len(seq))
        else:
            raise ValueError(f"'eta' requires at least {dim} entries, got {len(seq)}")
    return [float(seq[idx]) for idx in range(dim)]


def _parse_optional_int(value) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "none", "null"}:
            return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"Expected optional integer, got {value!r}") from exc


def _parse_pcp_variant(raw: dict | None, *, enabled_default: bool) -> PCPVariantOptions:
    section = raw or {}
    precision_default = (20, 50, 100, 200, 500)
    return PCPVariantOptions(
        enabled=bool(section.get("enabled", enabled_default)),
        n_thresholds=int(section.get("n_thresholds", 9)),
        logistic_cv_folds=int(section.get("logistic_cv_folds", 5)),
        max_clusters=int(section.get("max_clusters", 15)),
        cluster_r2_tol=float(section.get("cluster_r2_tol", 0.05)),
        factor_max_iter=int(section.get("factor_max_iter", 400)),
        factor_tol=float(section.get("factor_tol", 1e-4)),
        precision_grid=_parse_precision_grid(section.get("precision_grid"), precision_default),
        precision_trials=int(section.get("precision_trials", 64)),
        clip_eps=float(section.get("clip_eps", 1e-8)),
        proj_lr=float(section.get("proj_lr", 0.2)),
        proj_max_iter=int(section.get("proj_max_iter", 200)),
        proj_tol=float(section.get("proj_tol", 1e-6)),
        membership_smoothing=float(section.get("membership_smoothing", 0.0)),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    g_raw = raw.get("global", {})
    d_raw = raw.get("dgp", {})
    e_raw = raw.get("em_fit", {})
    m_raw = raw.get("model", {})
    io_raw = raw.get("io", {})
    p_raw = raw.get("pcp", {})

    global_cfg = GlobalConfig(
        seeds=_expand_range(g_raw.get("seeds", [1])),
        n_train=int(g_raw.get("n_train", 1000)),
        n_cal=int(g_raw.get("n_cal", 1000)),
        n_test=int(g_raw.get("n_test", 2000)),
        alpha=float(g_raw.get("alpha", 0.1)),
    )

    rho_source = d_raw.get("rho_list")
    if rho_source is None:
        rho_source = d_raw.get("rho_rx", [0.0])
    sigma_y_source = d_raw.get("sigma_y_list")
    if sigma_y_source is None:
        sigma_y_source = d_raw.get("sigma_y", [1.0])
    b_scale_source = d_raw.get("b_scale_list")
    if b_scale_source is None:
        b_scale_source = d_raw.get("beta_spread_list", [1.0])

    K_list = _expand_range(d_raw.get("K_list", [2]))
    if not K_list:
        raise ValueError("dgp.K_list must contain at least one value")
    d_R = int(d_raw.get("d_R", 4))
    d_X = int(d_raw.get("d_X", 6))
    max_K = max(int(k) for k in K_list)

    alpha_vals = _parse_cluster_scalar_list(
        d_raw.get("alpha"),
        max_K,
        "alpha",
        lambda idx: idx + 1,
    )
    sigma_vals = _parse_cluster_scalar_list(
        d_raw.get("sigma"),
        max_K,
        "sigma",
        lambda idx: 2 ** idx,
    )

    def _default_mu_r(idx: int) -> List[float]:
        row = [0.0] * d_R
        if d_R > 0:
            if max_K <= 1:
                center = 0.0
            else:
                start = -3.0
                end = 3.0
                step = (end - start) / (max_K - 1)
                center = start + step * idx
            row[0] = center
        return row

    mu_r_vals = _parse_cluster_vector_list(
        d_raw.get("mu_r"),
        max_K,
        d_R,
        "mu_r",
        _default_mu_r,
    )

    eta_vals = _parse_eta_vector(d_raw.get("eta"), d_X, [1.0, -0.5, 0.8])
    eta0_val = float(d_raw.get("eta0", 0.5))

    dgp_cfg = DGPConfig(
        K_list=K_list,
        d_R=d_R,
        d_X=d_X,
        use_S=str(d_raw.get("use_S", "R")),
        delta_list=_expand_range(d_raw.get("delta_list", [1.0])),
        sigma_s=float(d_raw.get("sigma_s", 1.0)),
        cov_type_true=str(d_raw.get("cov_type_true", "full")),
        rho_list=_expand_range(rho_source),
        sigma_y_list=_expand_range(sigma_y_source),
        b_scale_list=_expand_range(b_scale_source),
        mu_x_shift=float(d_raw.get("mu_x_shift", 0.0)),
        alpha=alpha_vals,
        sigma=sigma_vals,
        mu_r=mu_r_vals,
        eta0=eta0_val,
        eta=eta_vals,
    )

    em_cfg = EMConfig(
        K_fit=e_raw.get("K_fit"),
        cov_type_fit=str(e_raw.get("cov_type_fit", "full")),
        reg_covar=float(e_raw.get("reg_covar", 1e-6)),
        max_iter=int(e_raw.get("max_iter", 200)),
        tol=float(e_raw.get("tol", 1e-5)),
        init=str(e_raw.get("init", "kmeans")),
        n_init=int(e_raw.get("n_init", 1)),
        use_X_in_em=[bool(v) for v in _expand_range(e_raw.get("use_X_in_em", [False]))],
    )

    max_depth_val = _parse_optional_int(m_raw.get("pcp_rf_max_depth"))
    model_cfg = ModelConfig(
        ridge_alpha_oracle=float(m_raw.get("ridge_alpha_oracle", 0.0)),
        ridge_alpha_soft=float(m_raw.get("ridge_alpha_soft", m_raw.get("ridge_alpha", 0.0))),
        ridge_alpha_ignore=float(m_raw.get("ridge_alpha_ignore", 0.0)),
        ridge_alpha_xrzy=float(
            m_raw.get(
                "ridge_alpha_xrzy",
                m_raw.get("ridge_alpha_ignore", m_raw.get("ridge_alpha", 0.0)),
            )
        ),
        pcp_rf_n_estimators=int(m_raw.get("pcp_rf_n_estimators", 200)),
        pcp_rf_max_depth=max_depth_val,
        pcp_rf_min_samples_leaf=int(m_raw.get("pcp_rf_min_samples_leaf", 5)),
        pcp_rf_n_jobs=int(m_raw.get("pcp_rf_n_jobs", -1)),
    )

    io_cfg = IOConfig(
        results_csv=str(io_raw.get("results_csv", "experiments/results/results.csv")),
        artifacts_dir=str(io_raw.get("artifacts_dir", "experiments/artifacts")),
    )

    pcp_cfg = PCPFamilyConfig(
        xrzy=_parse_pcp_variant(p_raw.get("xrzy"), enabled_default=True),
        base=_parse_pcp_variant(p_raw.get("base"), enabled_default=False),
        em=_parse_pcp_variant(p_raw.get("em"), enabled_default=False),
    )

    return ExperimentConfig(global_cfg, dgp_cfg, em_cfg, model_cfg, io_cfg, pcp_cfg)


def iter_run_configs(cfg: ExperimentConfig) -> Iterator[RunConfig]:
    for seed in cfg.global_cfg.seeds:
        for K in cfg.dgp_cfg.K_list:
            for delta in cfg.dgp_cfg.delta_list:
                for rho in cfg.dgp_cfg.rho_list:
                    for sigma_y in cfg.dgp_cfg.sigma_y_list:
                        for b_scale in cfg.dgp_cfg.b_scale_list:
                            for use_x in cfg.em_cfg.use_X_in_em:
                                yield RunConfig(
                                    seed=int(seed),
                                    K=int(K),
                                    delta=float(delta),
                                    rho=float(rho),
                                    sigma_y=float(sigma_y),
                                    b_scale=float(b_scale),
                                    use_x_in_em=bool(use_x),
                                )
