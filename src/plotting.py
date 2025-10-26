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
AXIS_LABELS = {
    "delta": "Mixture separation δ",
    "b_scale": "Latent effect scale ||b||",
    "sigma_y": "Outcome noise σ_y",
    "rho": "Responsibility correlation ρ",
}
STYLE_DESCRIPTIONS = {
    "b_label": "Marker encodes latent scale ||b||",
    "sigma_label": "Marker encodes outcome noise σ_y",
    "delta_label": "Marker encodes mixture separation δ",
    "rho_label": "Marker encodes responsibility correlation ρ",
}
STYLE_FIELD_MAP = {
    "b_label": "b_scale",
    "sigma_label": "sigma_y",
    "delta_label": "delta",
    "rho_label": "rho",
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
    if "b_scale" in tidy.columns:
        tidy["b_label"] = tidy["b_scale"].map(lambda v: f"||b||={v}")
    if "sigma_y" in tidy.columns:
        tidy["sigma_label"] = tidy["sigma_y"].map(lambda v: f"σ_y={v}")
    if "delta" in tidy.columns:
        tidy["delta_label"] = tidy["delta"].map(lambda v: f"δ={v}")
    if "rho" in tidy.columns:
        tidy["rho_label"] = tidy["rho"].map(lambda v: f"ρ={v}")
    sort_cols = [col for col in ["panel", "use_label", "variant", "sigma_y", "b_scale", "delta"] if col in tidy.columns]
    if sort_cols:
        tidy.sort_values(sort_cols, inplace=True)
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
    excluded = {"value", "variant", "use_label", "panel", "b_label", "sigma_label", "delta_label", "rho_label"}
    for col in tidy.columns:
        if col not in excluded:
            return col
    return "value"


def _plot_metric_grid(tidy: pd.DataFrame, spec: MetricSpec, out_dir: Path, *, reference: float | None = None) -> None:
    if tidy.empty:
        return

    panel_order = sorted(tidy["panel"].unique())
    use_order = list(dict.fromkeys(tidy["use_label"]))
    variant_order = list(dict.fromkeys(tidy["variant"]))
    axis_field = _choose_axis_field(tidy, spec.x_candidates)
    axis_label = AXIS_LABELS.get(axis_field, axis_field.replace("_", " ").title())

    def _style_choice(df: pd.DataFrame) -> str | None:
        alternatives = ["b_label", "sigma_label", "delta_label", "rho_label"]
        # Prefer styles that vary and are not tied to the x-axis
        for field in alternatives:
            base_field = STYLE_FIELD_MAP.get(field, field)
            if base_field == axis_field:
                continue
            if field in df.columns and df[field].nunique() > 1:
                return field
        return None

    style_field = _style_choice(tidy)
    style_order: List[str] = []
    if style_field:
        style_order = list(dict.fromkeys(tidy[style_field]))

    palette_base = sns.color_palette("tab10", max(len(variant_order), 3))
    variant_palette = {
        variant: palette_base[i % len(palette_base)] for i, variant in enumerate(variant_order)
    }
    marker_map: Dict[str, str] = {}
    if style_order:
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

        subset = subset.sort_values(axis_field)

        plot_kwargs = {
            "data": subset,
            "x": axis_field,
            "y": "value",
            "hue": "variant",
            "hue_order": variant_order,
            "palette": variant_palette,
            "errorbar": "sd",
            "ax": ax,
            "linewidth": 1.8,
            "dashes": False,
        }

        if style_field:
            plot_kwargs.update(
                {
                    "style": style_field,
                    "style_order": style_order,
                    "markers": marker_map,
                    "markersize": 5.5,
                }
            )
        else:
            plot_kwargs.update({"markers": False})

        sns.lineplot(**plot_kwargs)

        if col == 0:
            ax.set_ylabel(spec.ylabel)
        else:
            ax.set_ylabel("")

        if row == n_rows - 1:
            ax.set_xlabel(axis_label)
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

    if style_order:
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
    footnote_parts = ["Line color identifies predictor"]
    if style_field:
        footnote_parts.append(STYLE_DESCRIPTIONS.get(style_field, "Marker encodes secondary parameter"))
    footnote_parts.append("Shaded band is ±1 SD across seeds.")
    fig.text(
        0.5,
        0.955,
        "; ".join(footnote_parts),
        ha="center",
        va="center",
        fontsize=9,
    )

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
            x_candidates=["b_scale", "sigma_y", "delta"],
        ),
        MetricSpec(
            "z_feature_mse",
            "Z-feature MSE",
            "z_feature_mse_vs_grid.png",
            "Z-feature MSE across grid",
            scalar_label=IMPUTATION_LABELS["z_feature_mse"],
            x_candidates=["b_scale", "sigma_y", "delta"],
        ),
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

    sns.scatterplot(data=tidy, x="mean_max_tau", y="len_ratio_soft", ax=axes[0], **scatter_kwargs)
    axes[0].set_xlabel("Mean max responsibility")
    axes[0].set_ylabel("Soft/oracle length ratio")
    axes[0].axhline(1.0, color="#2c3e50", linestyle=":", linewidth=1)
    axes[0].set_title("Length ratio vs τ sharpness")
    axes[0].grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    sns.scatterplot(data=tidy, x="z_feature_mse", y="len_ratio_soft", ax=axes[1], **scatter_kwargs)
    axes[1].set_xlabel("Z-feature MSE")
    axes[1].axhline(1.0, color="#2c3e50", linestyle=":", linewidth=1)
    axes[1].set_title("Length ratio vs feature error")
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
    fig.savefig(Path(out_dir) / "len_ratio_diagnostics.png", dpi=300, bbox_inches="tight")
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
        MetricSpec(
            "coverage_",
            "Coverage",
            "coverage_vs_grid.png",
            "Coverage across parameter grid",
            label_map=VARIANT_LABELS,
            x_candidates=["b_scale", "sigma_y", "delta"],
        ),
        MetricSpec(
            "len_ratio_",
            "Length / oracle length",
            "length_ratio_vs_grid.png",
            "Length ratio across parameter grid",
            label_map=VARIANT_LABELS,
            x_candidates=["b_scale", "sigma_y", "delta"],
        ),
    ]

    reference = 1.0 - float(alpha) if alpha is not None else None

    for spec in metric_specs:
        tidy = _prepare_tidy(df, spec)
        ref_line = None
        if spec.source == "coverage_":
            ref_line = reference
        elif spec.source == "len_ratio_":
            ref_line = 1.0
        _plot_metric_grid(tidy, spec, output_dir, reference=ref_line)

    _plot_imputation_metrics(df, output_dir)
    _plot_scatter_relationships(df, output_dir)
    _plot_em_diagnostics(df, output_dir)
