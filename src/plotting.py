"""Visualization utilities for latent conformal experiment outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .utils import ensure_dir

sns.set_theme(style="whitegrid")

ID_COLS = ["seed", "K", "delta", "beta_spread", "use_x_in_em"]
USE_LABEL = {False: "EM-R", True: "EM-RX"}


@dataclass(frozen=True)
class MetricSpec:
    prefix: str
    ylabel: str
    filename: str
    title: str


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


def _aggregate(melted: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["K", "beta_spread", "delta", "use_x_in_em", "variant"]
    agg = (
        melted.groupby(group_cols)
        .agg(mean_value=("value", "mean"), std_value=("value", "std"))
        .reset_index()
    )
    agg["use_label"] = agg["use_x_in_em"].map(USE_LABEL)
    agg.sort_values(["K", "beta_spread", "variant", "use_x_in_em", "delta"], inplace=True)
    return agg


def _subplot_grid(unique_k: Iterable[int], unique_beta: Iterable[float]) -> Tuple[plt.Figure, np.ndarray]:
    rows = len(unique_k)
    cols = len(unique_beta)
    figsize = (4 * cols, 3.2 * rows)
    fig, axes = plt.subplots(rows, cols, sharex=True, sharey=False, figsize=figsize)
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr[np.newaxis, :]
    return fig, axes_arr


def _line_plot_by_delta(agg: pd.DataFrame, spec: MetricSpec, out_dir: Path, *, reference: float | None = None) -> None:
    use_variants = list(dict.fromkeys(agg["variant"].tolist()))
    palette = sns.color_palette("tab10", len(use_variants))
    color_map = dict(zip(use_variants, palette))

    unique_k = sorted(agg["K"].unique())
    unique_beta = sorted(agg["beta_spread"].unique())
    fig, axes = _subplot_grid(unique_k, unique_beta)

    for r, K in enumerate(unique_k):
        for c, beta in enumerate(unique_beta):
            ax = axes[r, c]
            subset = agg[(agg["K"] == K) & (agg["beta_spread"] == beta)]
            if subset.empty:
                ax.set_visible(False)
                continue
            for variant in use_variants:
                sub_variant = subset[subset["variant"] == variant]
                if sub_variant.empty:
                    continue
                for use_val, style in ((False, "-"), (True, "--")):
                    sv = sub_variant[sub_variant["use_x_in_em"] == use_val]
                    if sv.empty:
                        continue
                    x = sv["delta"].to_numpy()
                    y = sv["mean_value"].to_numpy()
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    ax.plot(
                        x,
                        y,
                        linestyle=style,
                        marker="o",
                        color=color_map[variant],
                        label=f"{variant} ({USE_LABEL[use_val]})",
                    )
                    std = sv["std_value"].to_numpy()[order]
                    if np.isfinite(std).any():
                        ax.fill_between(
                            x,
                            y - std,
                            y + std,
                            color=color_map[variant],
                            alpha=0.12,
                            linewidth=0,
                        )
            ax.set_title(f"K={K}, β spread={beta}")
            ax.set_xlabel("Mixture separation δ")
            ax.set_ylabel(spec.ylabel)
            if reference is not None:
                ax.axhline(reference, color="black", linestyle=":", linewidth=1, label="target")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=3)
    fig.suptitle(spec.title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    ensure_dir(out_dir)
    fig.savefig(out_dir / spec.filename, dpi=300)
    plt.close(fig)


def _plot_imputation_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = ["acc_hard", "mean_max_tau", "cross_entropy"]
    titles = {
        "acc_hard": "Hard accuracy",
        "mean_max_tau": "Mean max responsibility",
        "cross_entropy": "Cross-entropy",
    }
    unique_k = sorted(df["K"].unique())
    unique_beta = sorted(df["beta_spread"].unique())

    fig, axes = _subplot_grid(unique_k, unique_beta)
    for r, K in enumerate(unique_k):
        for c, beta in enumerate(unique_beta):
            ax = axes[r, c]
            subset = df[(df["K"] == K) & (df["beta_spread"] == beta)]
            if subset.empty:
                ax.set_visible(False)
                continue
            for metric in metrics:
                grouped = (
                    subset.groupby(["delta", "use_x_in_em"])
                    [metric]
                    .agg(["mean", "std"])
                    .reset_index()
                )
                for use_val, style in ((False, "-"), (True, "--")):
                    sub_use = grouped[grouped["use_x_in_em"] == use_val]
                    if sub_use.empty:
                        continue
                    x = sub_use["delta"].to_numpy()
                    y = sub_use["mean"].to_numpy()
                    order = np.argsort(x)
                    color = sns.color_palette("tab10", len(metrics))[metrics.index(metric)]
                    ax.plot(
                        x[order],
                        y[order],
                        linestyle=style,
                        marker="o",
                        color=color,
                        label=f"{titles[metric]} ({USE_LABEL[use_val]})",
                    )
                    std = sub_use["std"].to_numpy()[order]
                    if np.isfinite(std).any():
                        ax.fill_between(
                            x[order],
                            y[order] - std,
                            y[order] + std,
                            alpha=0.12,
                            color=color,
                            linewidth=0,
                        )
            ax.set_title(f"K={K}, β spread={beta}")
            ax.set_xlabel("Mixture separation δ")
            ax.set_ylabel("Value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)
    fig.suptitle("Imputation metrics vs δ", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    ensure_dir(out_dir)
    fig.savefig(out_dir / "imputation_metrics_vs_delta.png", dpi=300)
    plt.close(fig)


def _plot_scatter_relationships(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    color_map = {False: "#1f77b4", True: "#ff7f0e"}
    for use_val, color in color_map.items():
        subset = df[df["use_x_in_em"] == use_val]
        ax.scatter(
            subset["acc_hard"],
            subset["len_gap_soft"],
            alpha=0.6,
            label=USE_LABEL[use_val],
            color=color,
        )
    ax.set_xlabel("Hard accuracy")
    ax.set_ylabel("Soft length gap")
    ax.set_title("Length efficiency vs imputation quality")
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.legend()
    ensure_dir(out_dir)
    fig.tight_layout()
    fig.savefig(out_dir / "len_gap_vs_accuracy.png", dpi=300)
    plt.close(fig)


def _plot_em_diagnostics(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df["em_iter"], bins=20, kde=False, ax=ax, color="#4c72b0")
    ax.set_xlabel("EM iterations to converge")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of EM iteration counts")
    fig.tight_layout()
    ensure_dir(out_dir)
    fig.savefig(out_dir / "em_iterations_hist.png", dpi=300)
    plt.close(fig)


def generate_all_plots(results_csv: str | Path, out_dir: str | Path, alpha: float | None = None) -> None:
    results_path = Path(results_csv)
    output_dir = Path(out_dir)
    df = pd.read_csv(results_path)
    if df.empty:
        raise ValueError("Results CSV is empty; nothing to plot")

    metric_specs = [
        MetricSpec("coverage_", "Coverage", "coverage_vs_delta.png", "Coverage vs separation"),
        MetricSpec("length_", "Interval length", "length_vs_delta.png", "Interval length vs separation"),
        MetricSpec("len_gap_", "Length gap vs oracle", "length_gap_vs_delta.png", "Length gap vs separation"),
    ]

    reference = None
    if alpha is not None:
        reference = 1.0 - float(alpha)

    for spec in metric_specs:
        melted = _melt_metrics(df, spec.prefix)
        agg = _aggregate(melted)
        _line_plot_by_delta(agg, spec, output_dir, reference=reference if spec.prefix == "coverage_" else None)

    _plot_imputation_metrics(df, output_dir)
    _plot_scatter_relationships(df, output_dir)
    _plot_em_diagnostics(df, output_dir)
