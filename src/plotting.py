"""Visualization utilities for latent conformal experiment outputs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .utils import ensure_dir

sns.set_theme(style="whitegrid")

ID_COLS = ["seed", "K", "delta", "rho", "sigma_y", "b_scale", "use_x_in_em"]
USE_LABEL = {False: "EM-R", True: "EM-RX"}
VARIANT_LABELS = {"oracle": "Oracle-Z", "soft": "EM-soft", "ignore": "Ignore-Z"}
IMPUTATION_LABELS = {
    "mean_max_tau": "Mean max responsibility",
    "cross_entropy": "Cross-entropy",
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

    tidy["use_label"] = tidy["use_x_in_em"].map(USE_LABEL)
    tidy["panel"] = tidy.apply(_panel_label, axis=1)
    tidy["b_label"] = tidy["b_scale"].map(lambda v: f"||b||={v}")
    tidy.sort_values(["panel", "use_label", "variant", "b_label", "delta"], inplace=True)
    return tidy


def _plot_metric_grid(tidy: pd.DataFrame, spec: MetricSpec, out_dir: Path, *, reference: float | None = None) -> None:
    if tidy.empty:
        return

    g = sns.relplot(
        data=tidy,
        x="delta",
        y="value",
        hue="variant",
        style="b_label",
        col="use_label",
        row="panel",
        kind="line",
        estimator="mean",
        errorbar="sd",
        facet_kws={"sharey": False, "sharex": True, "margin_titles": True},
        markers=True,
        dashes=False,
        height=2.8,
        aspect=1.35,
        legend=True,
    )

    g.set_axis_labels("Mixture separation δ", spec.ylabel)
    g.set_titles(col_template="{col_name}", row_template="{row_name}")

    for axes_row in g.axes:
        for ax in axes_row:
            if ax is None:
                continue
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
            if reference is not None:
                ax.axhline(reference, color="#2c3e50", linestyle=":", linewidth=1)

    if g._legend is not None:
        legend = g._legend
        legend.set_title("")
        labels = [text.get_text() for text in legend.texts]
        legend.remove()

        axes = [ax for ax in g.axes.flat if ax is not None]
        handle_lookup: Dict[str, object] = {}
        for ax in axes:
            axis_handles, axis_labels = ax.get_legend_handles_labels()
            for handle, label in zip(axis_handles, axis_labels):
                if label and label not in handle_lookup:
                    handle_lookup[label] = handle

        handles = [handle_lookup[label] for label in labels if label in handle_lookup]
        if not handles:
            handles = [h for h in handle_lookup.values()]
            labels = [l for l in handle_lookup.keys()]

        ncol = max(1, min(len(labels), 4))
        g.fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=ncol,
            frameon=False,
        )

    g.fig.subplots_adjust(top=0.92, hspace=0.25)
    g.fig.suptitle(spec.title, fontsize=13)

    ensure_dir(out_dir)
    g.fig.savefig(Path(out_dir) / spec.filename, dpi=300)
    plt.close(g.fig)


def _plot_imputation_metrics(df: pd.DataFrame, out_dir: Path) -> None:
    specs = [
        MetricSpec("mean_max_tau", "Mean max responsibility", "mean_max_tau_vs_delta.png", "Mean max responsibility vs separation", scalar_label=IMPUTATION_LABELS["mean_max_tau"]),
        MetricSpec("cross_entropy", "Cross-entropy", "cross_entropy_vs_delta.png", "Cross-entropy vs separation", scalar_label=IMPUTATION_LABELS["cross_entropy"]),
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
    if df.empty:
        raise ValueError("Results CSV is empty; nothing to plot")

    metric_specs = [
        MetricSpec("coverage_", "Coverage", "coverage_vs_delta.png", "Coverage vs separation", label_map=VARIANT_LABELS),
        MetricSpec("length_", "Interval length", "length_vs_delta.png", "Interval length vs separation", label_map=VARIANT_LABELS),
        MetricSpec("len_gap_", "Length gap vs oracle", "length_gap_vs_delta.png", "Length gap vs separation", label_map=VARIANT_LABELS),
    ]

    reference = 1.0 - float(alpha) if alpha is not None else None

    for spec in metric_specs:
        tidy = _prepare_tidy(df, spec)
        _plot_metric_grid(tidy, spec, output_dir, reference=reference if spec.source == "coverage_" else None)

    _plot_imputation_metrics(df, output_dir)
    _plot_scatter_relationships(df, output_dir)
    _plot_em_diagnostics(df, output_dir)
