"""Visualization utilities for latent conformal experiment outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D

from .utils import ensure_dir

sns.set_theme(style="whitegrid")

ID_COLS = ["seed", "K", "delta", "rho", "sigma_y", "b_scale", "use_x_in_em"]
B_MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
USE_LABEL = {False: "EM-R", True: "EM-RX"}
USE_ROW_LABEL = {
    False: "EM-R (R-only responsibilities)",
    True: "EM-RX (R and X in EM)",
}
VARIANT_LABELS = {"oracle": "Oracle-Z", "soft": "EM-soft", "ignore": "Ignore-Z"}
IMPUTATION_LABELS = {
    "mean_max_tau": "Mean max responsibility",
    "z_feature_mse": "Z-feature MSE",
}


@dataclass(frozen=True)
class MetricSpec:
    source: str
    ylabel: str
    filename: str
    title: str
    label_map: Dict[str, str] | None = None
    scalar_label: str | None = None


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


def _panel_label(row: pd.Series) -> str:
    return f"K={int(row['K'])}, ρ={row['rho']}, σ={row['sigma_y']}"


def _prepare_tidy(df: pd.DataFrame, spec: MetricSpec) -> pd.DataFrame:
    if spec.scalar_label:
        if spec.source not in df.columns:
            raise ValueError(f"Column '{spec.source}' not found in results")
        tidy = df[[col for col in ID_COLS if col in df.columns] + [spec.source]].copy()
        tidy.rename(columns={spec.source: "value"}, inplace=True)
        tidy["variant"] = spec.scalar_label
    else:
        tidy = _melt_metrics(df, spec.source)
        mapping = spec.label_map or VARIANT_LABELS
        tidy["variant"] = tidy["variant"].map(mapping).fillna(
            tidy["variant"].str.replace("_", " ").str.title()
        )

    tidy["use_label"] = tidy["use_x_in_em"].map(USE_ROW_LABEL)
    tidy["panel"] = tidy.apply(_panel_label, axis=1)
    tidy["b_label"] = tidy["b_scale"].map(lambda v: f"||b||={v}")
    tidy.sort_values(["panel", "use_label", "variant", "b_label", "delta"], inplace=True)
    return tidy


def _factor_grid(n: int) -> tuple[int, int]:
    if n <= 0:
        return (1, 1)
    best_rows, best_cols = n, 1
    for cols in range(1, int(math.sqrt(n)) + 1):
        rows = math.ceil(n / cols)
        area = rows * cols
        best_area = best_rows * best_cols
        diff = abs(rows - cols)
        best_diff = abs(best_rows - best_cols)
        if area < best_area or (area == best_area and diff < best_diff):
            best_rows, best_cols = rows, cols
    return best_rows, best_cols


def _plot_metric_grid(tidy: pd.DataFrame, spec: MetricSpec, out_dir: Path, *, reference: float | None = None) -> None:
    if tidy.empty:
        return

    panel_order = sorted(tidy["panel"].unique())
    use_order = list(dict.fromkeys(tidy["use_label"]))
    variant_order = list(dict.fromkeys(tidy["variant"]))
    style_order = list(dict.fromkeys(tidy["b_label"]))

    palette_base = sns.color_palette("tab10", max(len(variant_order), 3))
    variant_palette = {
        variant: palette_base[i % len(palette_base)] for i, variant in enumerate(variant_order)
    }

    marker_map = {label: B_MARKERS[i % len(B_MARKERS)] for i, label in enumerate(style_order)}

    combos = [(use, panel) for use in use_order for panel in panel_order]
    n_panels = len(combos)
    n_rows, n_cols = _factor_grid(n_panels)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4.0, n_rows * 2.6),
        squeeze=False,
    )

    for idx, (use, panel) in enumerate(combos):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        subset = tidy[(tidy["use_label"] == use) & (tidy["panel"] == panel)]
        if subset.empty:
            ax.axis("off")
            continue

        sns.lineplot(
            data=subset,
            x="delta",
            y="value",
            hue="variant",
            hue_order=variant_order,
            palette=variant_palette,
            style="b_label",
            style_order=style_order,
            markers=marker_map,
            dashes=False,
            errorbar="sd",
            ax=ax,
            linewidth=1.8,
            markersize=5.5,
        )

        if col == 0:
            ax.set_ylabel(spec.ylabel)
        else:
            ax.set_ylabel("")

        if row == n_rows - 1:
            ax.set_xlabel("Mixture separation δ")
        else:
            ax.set_xlabel("")

        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        if reference is not None:
            ax.axhline(reference, color="#2c3e50", linestyle=":", linewidth=1.1)

        if ax.legend_ is not None:
            ax.legend_.remove()

        ax.set_title(f"{panel}\n{use}", fontsize=9)

    for idx in range(n_panels, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis("off")

    legend_handles: List[Line2D] = []
    legend_labels: List[str] = []

    for variant in variant_order:
        handle = Line2D([0, 1], [0, 0], color=variant_palette[variant], linewidth=2.6)
        legend_handles.append(handle)
        legend_labels.append(f"{variant} (line color)")

    for label in style_order:
        handle = Line2D(
            [0],
            [0],
            marker=marker_map[label],
            color="#3c3c3c",
            linestyle="",
            markersize=7,
            markerfacecolor="#3c3c3c",
        )
        legend_handles.append(handle)
        legend_labels.append(f"{label} (marker)")

    if reference is not None:
        legend_handles.append(
            Line2D([0, 1], [0, 0], color="#2c3e50", linestyle=":", linewidth=1.3)
        )
        legend_labels.append(f"Target coverage = {reference:.2f}")

    ncol = min(4, max(1, len(legend_handles)))

    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.12, hspace=0.5, wspace=0.24)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=ncol,
        frameon=False,
        fontsize=9,
    )

    fig.suptitle(spec.title, fontsize=13, y=0.985)
    fig.text(
        0.5,
        0.955,
        "Line color identifies predictor; marker denotes ||b||; shaded band is ±1 SD across seeds.",
        ha="center",
        va="center",
        fontsize=9,
    )

    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / spec.filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_imputation_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        MetricSpec("mean_max_tau", "Mean max responsibility", "mean_max_tau_vs_delta.png", "Mean max responsibility vs separation", scalar_label=IMPUTATION_LABELS["mean_max_tau"]),
        MetricSpec("z_feature_mse", "Z-feature MSE", "z_feature_mse_vs_delta.png", "Z-feature MSE vs separation", scalar_label=IMPUTATION_LABELS["z_feature_mse"]),
    ]

    for spec in specs:
        tidy = _prepare_tidy(df, spec)
        _plot_metric_grid(tidy, spec, out_dir)


def _plot_scatter_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    tidy = df.copy()
    tidy["use_label"] = tidy["use_x_in_em"].map(USE_LABEL)
    tidy["panel"] = tidy.apply(_panel_label, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)
    scatter_kwargs = {
        "hue": "use_label",
        "style": "b_scale",
        "size": "delta",
        "palette": "tab10",
        "sizes": (40, 140),
        "alpha": 0.75,
    }

    sns.scatterplot(data=tidy, x="mean_max_tau", y="len_gap_soft", ax=axes[0], **scatter_kwargs)
    axes[0].set_xlabel("Mean max responsibility")
    axes[0].set_ylabel("Soft length gap")
    axes[0].axhline(0.0, color="#2c3e50", linestyle=":", linewidth=1)
    axes[0].set_title("Length gap vs τ sharpness")
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    sns.scatterplot(data=tidy, x="z_feature_mse", y="len_gap_soft", ax=axes[1], **scatter_kwargs)
    axes[1].set_xlabel("Z-feature MSE")
    axes[1].axhline(0.0, color="#2c3e50", linestyle=":", linewidth=1)
    axes[1].set_title("Length gap vs feature error")
    axes[1].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    if axes[0].legend_:
        axes[0].legend_.remove()
    handles, labels = axes[1].get_legend_handles_labels()
    if axes[1].legend_:
        axes[1].legend_.remove()
    if handles:
        axes[1].legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False)

    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / "len_gap_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_len_gap_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    grouped = list(df.groupby("use_x_in_em"))
    if not grouped:
        return

    agg = df.groupby(["use_x_in_em", "delta", "rho"])["len_gap_soft"].mean()
    max_abs = float(agg.abs().max()) if not agg.empty else 0.0
    vmax = max(max_abs, 1e-6)

    n_cols = len(grouped)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.2 * n_cols, 3.6), squeeze=False)
    axes = axes[0]

    for idx, (use, group) in enumerate(grouped):
        pivot = (
            group.groupby(["delta", "rho"])["len_gap_soft"].mean()
            .sort_index()
            .unstack("rho")
            .sort_index(axis=1)
        )
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            cbar=idx == n_cols - 1,
            cbar_kws={"label": "Mean soft length gap"},
            ax=axes[idx],
        )
        axes[idx].set_title(f"Soft length gap\n{USE_LABEL[use]}")
        axes[idx].set_xlabel("ρ")
        axes[idx].set_ylabel("δ")

    fig.suptitle("Soft length gap vs separation (δ) and correlation (ρ)", fontsize=13, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    ensure_dir(out_dir)
    fig.savefig(Path(out_dir) / "len_gap_heatmap.png", dpi=300)
    plt.close(fig)


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
        MetricSpec("coverage_", "Coverage", "coverage_vs_delta.png", "Coverage vs separation", label_map=VARIANT_LABELS),
        MetricSpec("len_gap_", "Length gap vs oracle", "length_gap_vs_delta.png", "Length gap vs separation", label_map=VARIANT_LABELS),
    ]

    reference = 1.0 - float(alpha) if alpha is not None else None

    for spec in metric_specs:
        tidy = _prepare_tidy(df, spec)
        _plot_metric_grid(tidy, spec, output_dir, reference=reference if spec.source == "coverage_" else None)

    _plot_imputation_metrics(df, output_dir)
    _plot_scatter_relationships(df, output_dir)
    _plot_len_gap_heatmap(df, output_dir)
    _plot_em_diagnostics(df, output_dir)
