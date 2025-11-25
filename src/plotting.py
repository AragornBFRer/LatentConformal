"""Visualization utilities for latent conformal experiment outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D

from .utils import ensure_dir

sns.set_theme(style="whitegrid")

ID_COLS = ["seed", "K", "delta", "rho", "sigma_y", "b_scale", "use_x_in_em"]
USE_LABEL = {False: "EM-R", True: "EM-RX"}
USE_ROW_LABEL = {
    False: "EM-R (R-only responsibilities)",
    True: "EM-RX (R and X in EM)",
}
VARIANT_LABELS = {
    "cqr_ignore": "CQR-ignoreZ",
    "pcp_base": "PCP-base",
    "em_pcp": "EM-PCP",
}
IMPUTATION_LABELS = {
    "mean_max_tau": "Mean max responsibility",
    "z_feature_mse": "Z-feature MSE",
}
AXIS_LABELS = {
    "delta": "Mixture separation δ",
    "b_scale": "Latent effect scale ||b||",
    "sigma_y": "Outcome noise σ_y",
    "rho": "Responsibility correlation ρ",
}

@dataclass(frozen=True)
class MetricSpec:
    source: str
    ylabel: str
    filename: str
    title: str
    label_map: Dict[str, str] | None = None
    scalar_label: str | None = None
    x_candidates: List[str] | None = None
    variant_filter: List[str] | None = None


def _melt_metrics(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        raise ValueError(f"No columns found with prefix '{prefix}'")
    melted = df.melt(
        id_vars=[col for col in ID_COLS if col in df.columns],
        value_vars=cols,
        var_name="metric",
        value_name="value",
    )
    melted["variant"] = melted["metric"].str.removeprefix(prefix)
    melted.drop(columns="metric", inplace=True)
    return melted


def _prepare_tidy(df: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    if spec.scalar_label:
        if spec.source not in df.columns:
            raise ValueError(f"Column '{spec.source}' not found in results")
        tidy = df[[col for col in ID_COLS if col in df.columns] + [spec.source]].copy()
        tidy.rename(columns={spec.source: "value"}, inplace=True)
        tidy["variant"] = spec.scalar_label
    else:
        tidy = _melt_metrics(df, spec.source)
        if spec.variant_filter:
            tidy = tidy[tidy["variant"].isin(spec.variant_filter)].copy()
        if "use_x_in_em" in tidy.columns:
            tidy["use_label"] = tidy["use_x_in_em"].map(USE_ROW_LABEL)
        else:
            tidy["use_label"] = USE_ROW_LABEL.get(False, "EM-R (R-only responsibilities)")
    if spec.variant_filter:
        tidy.reset_index(drop=True, inplace=True)
        mapping = spec.label_map or VARIANT_LABELS
        tidy["variant"] = tidy["variant"].map(mapping).fillna(
            tidy["variant"].str.replace("_", " ").str.title()
        )

    if "use_label" not in tidy.columns:
        tidy["use_label"] = tidy.get("use_x_in_em", False).map(USE_ROW_LABEL)
    if "b_scale" in tidy.columns:
        tidy["b_label"] = tidy["b_scale"].map(lambda v: f"||b||={v}")
    if "sigma_y" in tidy.columns:
        tidy["sigma_label"] = tidy["sigma_y"].map(lambda v: f"σ_y={v}")
    if "delta" in tidy.columns:
        tidy["delta_label"] = tidy["delta"].map(lambda v: f"δ={v}")
    if "rho" in tidy.columns:
        tidy["rho_label"] = tidy["rho"].map(lambda v: f"ρ={v}")
    sort_cols = [
        col
        for col in ["use_label", "variant", "sigma_y", "b_scale", "delta"]
        if col in tidy.columns
    ]
    if sort_cols:
        tidy.sort_values(sort_cols, inplace=True)
    return tidy


def _choose_axis_field(tidy: pd.DataFrame, candidates: List[str] | None) -> str:
    if candidates:
        for field in candidates:
            if field in tidy.columns and tidy[field].nunique() > 1:
                return field
    if "delta" in tidy.columns:
        return "delta"
    for fallback in ["b_scale", "sigma_y", "rho"]:
        if fallback in tidy.columns:
            return fallback
    excluded = {"value", "variant", "use_label", "b_label", "sigma_label", "delta_label", "rho_label"}
    for col in tidy.columns:
        if col not in excluded:
            return col
    return "value"


def _series_fields(tidy: pd.DataFrame) -> List[str]:
    candidates = ["use_label", "K", "rho", "sigma_y", "b_scale"]
    fields: List[str] = []
    for field in candidates:
        if field in tidy.columns and tidy[field].nunique() > 1:
            fields.append(field)
    return fields


def _format_series_label(row: pd.Series, fields: List[str]) -> str:
    parts = [str(row["variant"])]
    for field in fields:
        value = row[field]
        if pd.isna(value):
            continue
        if field == "use_label":
            parts.append(str(value))
        elif field == "K":
            parts.append(f"K={int(value)}")
        elif field == "rho":
            parts.append(f"ρ={value}")
        elif field == "sigma_y":
            parts.append(f"σ={value}")
        elif field == "b_scale":
            parts.append(f"||b||={value}")
        else:
            parts.append(f"{field}={value}")
    return " · ".join(parts)


def _plot_metric_single(
    tidy: pd.DataFrame,
    spec: MetricSpec,
    out_dir: Path,
    *,
    reference: float | None = None,
) -> None:
    if tidy.empty:
        return

    axis_field = _choose_axis_field(tidy, spec.x_candidates)
    axis_label = AXIS_LABELS.get(axis_field, axis_field.replace("_", " ").title())

    df = tidy.copy()
    series_fields = _series_fields(df)
    df["series_label"] = df.apply(lambda row: _format_series_label(row, series_fields), axis=1)
    series_order = list(dict.fromkeys(df["series_label"]))
    palette_base = sns.color_palette("tab10", max(len(series_order), 3))
    color_map = {label: palette_base[i % len(palette_base)] for i, label in enumerate(series_order)}

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(
        data=df,
        x=axis_field,
        y="value",
        hue="series_label",
        hue_order=series_order,
        style="series_label",
        style_order=series_order,
        markers=True,
        dashes=False,
        linewidth=1.8,
        errorbar="sd",
        palette=color_map,
        ax=ax,
    )

    ax.set_xlabel(axis_label)
    ax.set_ylabel(spec.ylabel)
    ax.set_title(spec.title)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    if reference is not None:
        ax.axhline(reference, color="#2c3e50", linestyle=":", linewidth=1.1)

    ax.legend(title="Series", loc="upper left", bbox_to_anchor=(1.02, 1))

    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / spec.filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_imputation_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        MetricSpec(
            "mean_max_tau",
            "Mean max responsibility",
            "mean_max_tau_vs_grid.png",
            "Mean max responsibility across grid",
            scalar_label=IMPUTATION_LABELS["mean_max_tau"],
            x_candidates=["delta"],
        ),
        MetricSpec(
            "z_feature_mse",
            "Z-feature MSE",
            "z_feature_mse_vs_grid.png",
            "Z-feature MSE across grid",
            scalar_label=IMPUTATION_LABELS["z_feature_mse"],
            x_candidates=["delta"],
        ),
    ]

    for spec in specs:
        tidy = _prepare_tidy(df, spec)
        _plot_metric_single(tidy, spec, out_dir)


def _plot_em_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["em_iter"], bins=20, kde=False, ax=ax, color="#4c72b0")
    ax.set_xlabel("EM iterations to converge")
    ax.set_ylabel("Count")
    ax.set_title("EM iteration counts")
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / "em_iterations_hist.png", dpi=300)
    plt.close(fig)


def generate_all_plots(results_csv: str | Path, out_dir: str | Path, alpha: float | None = None) -> None:
    results_path = Path(results_csv)
    output_dir = Path(out_dir)
    df = pd.read_csv(results_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.empty:
        raise ValueError("Results CSV is empty; nothing to plot")

    metric_specs = [
        MetricSpec(
            "coverage_",
            "Coverage",
            "coverage_vs_grid.png",
            "Coverage across parameter grid",
            label_map=VARIANT_LABELS,
            x_candidates=["delta"],
        ),
        MetricSpec(
            "length_",
            "Interval length",
            "length_vs_grid.png",
            "Interval length across parameter grid",
            label_map=VARIANT_LABELS,
            x_candidates=["delta"],
        ),
        MetricSpec(
            "length_",
            "Interval length",
            "length_vs_grid_pcp.png",
            "PCP baseline interval length across parameter grid",
            label_map={"pcp_base": "PCP-base", "em_pcp": "EM-PCP"},
            x_candidates=["delta"],
            variant_filter=["pcp_base", "em_pcp"],
        ),
    ]

    reference = 1.0 - float(alpha) if alpha is not None else None

    for spec in metric_specs:
        tidy = _prepare_tidy(df, spec)
        if spec.filename == "length_vs_grid_pcp.png":
            tidy = tidy[tidy.get("use_x_in_em", False) == False]
            tidy = tidy.copy()
            tidy["use_label"] = USE_ROW_LABEL[False]
        ref_line = None
        if spec.source == "coverage_":
            ref_line = reference
        _plot_metric_single(tidy, spec, output_dir, reference=ref_line)

    _plot_imputation_metrics(df, output_dir)
    _plot_em_diagnostics(df, output_dir)
