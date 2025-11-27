from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import load_config
from src.plotting import (
    MetricSpec,
    VARIANT_COLORS,
    VARIANT_LABELS,
    USE_ROW_LABEL,
    AXIS_LABELS,
    _prepare_tidy,
    _choose_axis_field,
)
from src.utils import ensure_dir

import warnings

warnings.filterwarnings("ignore")

sns.set_theme(style="whitegrid")
sns.set_context("talk")

LABEL_FONTSIZE = 16
TICK_FONTSIZE = 13
LEGEND_FONTSIZE = 13

PCP_VARIANTS = ["pcp_xr", "pcp_xz", "pcp_xzhat", "pcp_xrzhat", "em_pcp"]

COVERAGE_SPEC = MetricSpec(
    source="coverage_",
    ylabel="Coverage",
    filename="",
    title="",
    label_map=VARIANT_LABELS,
    x_candidates=["delta"],
    variant_filter=PCP_VARIANTS,
)

LENGTH_SPEC = MetricSpec(
    source="length_",
    ylabel="Interval length",
    filename="",
    title="",
    label_map=VARIANT_LABELS,
    x_candidates=["delta"],
    variant_filter=PCP_VARIANTS,
)


def _prepare_metric(
    df: pd.DataFrame,
    spec: MetricSpec,
    *,
    use_x_in_em: bool | None,
) -> Tuple[pd.DataFrame, str, str]:
    tidy = _prepare_tidy(df, spec)
    if tidy.empty:
        raise ValueError(f"No data available for metric '{spec.source}'")

    if use_x_in_em is not None and "use_x_in_em" in tidy.columns:
        tidy = tidy[tidy["use_x_in_em"] == use_x_in_em].copy()
        if tidy.empty:
            raise ValueError(
                f"No rows remain for metric '{spec.source}' with use_x_in_em={use_x_in_em}"
            )
        tidy["use_label"] = USE_ROW_LABEL[use_x_in_em]

    tidy["series_label"] = tidy["variant"]
    axis_field = _choose_axis_field(tidy, spec.x_candidates)
    axis_label = AXIS_LABELS.get(axis_field, axis_field.replace("_", " ").title())
    return tidy, axis_field, axis_label


def _style_tick_labels(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(TICK_FONTSIZE)
        label.set_fontweight("bold")


def _collect_series_info(tidies: Iterable[pd.DataFrame]) -> Tuple[List[str], Dict[str, str]]:
    key_to_label: Dict[str, str] = {}
    keys_in_order: List[str] = []
    for tidy in tidies:
        if tidy.empty:
            continue
        for key, label in zip(tidy["variant_key"], tidy["series_label"]):
            if key not in key_to_label:
                key_to_label[key] = label
            if key not in keys_in_order:
                keys_in_order.append(key)

    if not keys_in_order:
        raise ValueError("Unable to identify any variants to plot")

    ordered_keys: List[str] = []
    for key in PCP_VARIANTS:
        if key in key_to_label:
            ordered_keys.append(key)
    for key in keys_in_order:
        if key not in ordered_keys:
            ordered_keys.append(key)

    series_order = [key_to_label[key] for key in ordered_keys]
    label_to_key = {label: key for key, label in key_to_label.items()}
    return series_order, label_to_key


def _build_color_map(series_order: List[str], label_to_key: Dict[str, str]) -> Dict[str, str]:
    palette = sns.color_palette("tab10", max(len(series_order), 3))
    color_map: Dict[str, str] = {}
    fallback_idx = 0

    for idx, label in enumerate(series_order):
        key = label_to_key[label]
        if key in VARIANT_COLORS:
            color_map[label] = VARIANT_COLORS[key]
        else:
            color_map[label] = palette[fallback_idx % len(palette)]
            fallback_idx += 1

    return color_map


def _plot_metric_axis(
    ax: plt.Axes,
    tidy: pd.DataFrame,
    axis_field: str,
    axis_label: str,
    y_label: str,
    *,
    series_order: List[str],
    color_map: Dict[str, str],
    reference: float | None = None,
    clamp_high_coverage: bool = False,
) -> None:
    sns.lineplot(
        data=tidy,
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
        palette={label: color_map[label] for label in series_order},
        ax=ax,
    )

    ax.set_xlabel(axis_label, fontsize=LABEL_FONTSIZE, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=LABEL_FONTSIZE, fontweight="bold")
    _style_tick_labels(ax)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)

    if reference is not None:
        ax.axhline(reference, color="#2c3e50", linestyle=":", linewidth=1.1)

    if clamp_high_coverage:
        min_val = float(np.nanmin(tidy["value"]))
        if min_val >= 0.7:
            max_val = float(np.nanmax(tidy["value"]))
            upper = max_val + 0.01
            if upper <= 0.71:
                upper = 0.75
            ax.set_ylim(0.7, min(1.0, upper))


def _create_shared_legend(fig: plt.Figure, axes: List[plt.Axes]) -> None:
    handles: List = []
    labels: List[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        if h and l:
            handles.extend(h)
            labels.extend(l)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    unique_entries: List[Tuple] = []
    seen: set[str] = set()
    for handle, label in zip(handles, labels):
        if not label or label == "series_label" or label in seen:
            continue
        seen.add(label)
        unique_entries.append((handle, label))

    if not unique_entries:
        return

    handles_clean, labels_clean = zip(*unique_entries)
    legend = fig.legend(
        handles_clean,
        labels_clean,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=len(labels_clean),
        frameon=False,
        fontsize=LEGEND_FONTSIZE,
    )
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontweight("bold")


def plot_combined_metrics(
    results_csv: Path,
    output_path: Path,
    *,
    alpha: float | None,
    use_x_in_em: bool | None = False,
) -> None:
    df = pd.read_csv(results_csv)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if df.empty:
        raise ValueError("Results CSV is empty; nothing to plot")

    coverage_tidy, coverage_axis_field, coverage_axis_label = _prepare_metric(
        df, COVERAGE_SPEC, use_x_in_em=use_x_in_em
    )
    length_tidy, length_axis_field, length_axis_label = _prepare_metric(
        df, LENGTH_SPEC, use_x_in_em=use_x_in_em
    )

    series_order, label_to_key = _collect_series_info([coverage_tidy, length_tidy])
    color_map = _build_color_map(series_order, label_to_key)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    reference = 1.0 - alpha if alpha is not None else None
    _plot_metric_axis(
        axes[0],
        coverage_tidy,
        coverage_axis_field,
        coverage_axis_label,
        "Coverage",
        series_order=series_order,
        color_map=color_map,
        reference=reference,
        clamp_high_coverage=True,
    )
    _plot_metric_axis(
        axes[1],
        length_tidy,
        length_axis_field,
        length_axis_label,
        "Interval length",
        series_order=series_order,
        color_map=color_map,
    )

    for ax in axes:
        ax.set_title("")

    _create_shared_legend(fig, list(axes))

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate combined coverage/length plots for latent conformal experiments",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/gmm_em.yaml",
        help="Path to experiment configuration (provides defaults and alpha)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Override path to results CSV (defaults to config io.results_csv)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="experiments/plots",
        help="Output directory for the combined PDF",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="pcp_metrics_combined.pdf",
        help="Filename for the combined PDF (must end with .pdf)",
    )
    parser.add_argument(
        "--use-x-in-em",
        choices=["auto", "true", "false"],
        default="false",
        help="Which EM responsibility setting to include (default: false)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_path = Path(args.results or cfg.io_cfg.results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    out_dir = Path(args.out)
    filename = args.filename
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Output filename must end with .pdf")
    output_path = out_dir / filename

    use_x_setting: bool | None
    if args.use_x_in_em == "auto":
        use_x_setting = None
    elif args.use_x_in_em == "true":
        use_x_setting = True
    else:
        use_x_setting = False

    plot_combined_metrics(results_path, output_path, alpha=cfg.global_cfg.alpha, use_x_in_em=use_x_setting)
    print(f"Saved combined plot to {output_path}")


if __name__ == "__main__":
    main()
