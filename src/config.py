"""Configuration loading and grid expansion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

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


@dataclass(frozen=True)
class RunConfig:
    seed: int
    K: int
    delta: float
    rho: float
    sigma_y: float
    b_scale: float
    use_x_in_em: bool


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


def load_config(path: str | Path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text())

    g_raw = raw.get("global", {})
    d_raw = raw.get("dgp", {})
    e_raw = raw.get("em_fit", {})
    m_raw = raw.get("model", {})
    io_raw = raw.get("io", {})

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

    dgp_cfg = DGPConfig(
        K_list=_expand_range(d_raw.get("K_list", [2])),
        d_R=int(d_raw.get("d_R", 4)),
        d_X=int(d_raw.get("d_X", 6)),
        use_S=str(d_raw.get("use_S", "R")),
        delta_list=_expand_range(d_raw.get("delta_list", [1.0])),
        sigma_s=float(d_raw.get("sigma_s", 1.0)),
        cov_type_true=str(d_raw.get("cov_type_true", "full")),
        rho_list=_expand_range(rho_source),
        sigma_y_list=_expand_range(sigma_y_source),
        b_scale_list=_expand_range(b_scale_source),
        mu_x_shift=float(d_raw.get("mu_x_shift", 0.0)),
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
    )

    io_cfg = IOConfig(
        results_csv=str(io_raw.get("results_csv", "experiments/results/results.csv")),
        artifacts_dir=str(io_raw.get("artifacts_dir", "experiments/artifacts")),
    )

    return ExperimentConfig(global_cfg, dgp_cfg, em_cfg, model_cfg, io_cfg)


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
