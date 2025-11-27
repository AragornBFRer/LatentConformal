from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from src.config import load_config
from src.experiment import run_experiment
from src.plotting import generate_all_plots


def _parse_run_spec(spec: str) -> Tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(
            f"Run specification '{spec}' must be of the form <identifier>=<config-path>"
        )
    identifier, cfg_path = spec.split("=", 1)
    identifier = identifier.strip()
    cfg_path = cfg_path.strip()
    if not identifier:
        raise argparse.ArgumentTypeError("Identifier portion of run spec cannot be empty")
    if not cfg_path:
        raise argparse.ArgumentTypeError("Config path portion of run spec cannot be empty")
    return identifier, cfg_path


def _parse_run_specs(specs: Sequence[str]) -> List[Tuple[str, str]]:
    runs: List[Tuple[str, str]] = []
    seen = set()
    for spec in specs:
        identifier, cfg_path = _parse_run_spec(spec)
        if identifier in seen:
            raise argparse.ArgumentTypeError(f"Duplicate identifier '{identifier}'")
        seen.add(identifier)
        runs.append((identifier, cfg_path))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run multiple latent-conformal experiments, saving each config's"
            " outputs under identifier-specific folders."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run spec of the form <identifier>=<config-path>. May be provided multiple times.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Override seeds with range(1, trials+1) for every run.",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="experiments/results",
        help="Base directory under which identifier-specific results CSVs are stored.",
    )
    parser.add_argument(
        "--plots-root",
        type=str,
        default="experiments/plots",
        help="Base directory under which identifier-specific plots will be saved.",
    )
    parser.add_argument(
        "--results-name",
        type=str,
        default="results.csv",
        help="Filename to use for each results CSV inside its identifier folder.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation after each run (useful for headless environments).",
    )
    args = parser.parse_args()

    try:
        runs = _parse_run_specs(args.run)
    except argparse.ArgumentTypeError as exc:
        parser.error(str(exc))

    seeds = None
    if args.trials is not None:
        if args.trials <= 0:
            parser.error("--trials must be positive when specified")
        seeds = range(1, args.trials + 1)

    results_root = Path(args.results_root)
    plots_root = Path(args.plots_root)
    filename = args.results_name

    for identifier, cfg_path in runs:
        cfg_path_obj = Path(cfg_path)
        if not cfg_path_obj.exists():
            raise FileNotFoundError(f"Config not found for identifier '{identifier}': {cfg_path}")
        cfg = load_config(cfg_path_obj)
        results_dir = results_root / identifier
        results_dir.mkdir(parents=True, exist_ok=True)
        results_csv = results_dir / filename

        print(f"[multi-run] Starting '{identifier}' → {cfg_path}")
        df = run_experiment(
            str(cfg_path_obj),
            seeds=seeds,
            results_path_override=results_csv,
        )
        results_path = df.attrs.get("results_path", str(results_csv))
        print(f"[multi-run] Finished '{identifier}'. Results → {results_path}")

        if args.skip_plots:
            continue
        plot_dir = plots_root / identifier
        plot_dir.mkdir(parents=True, exist_ok=True)
        generate_all_plots(results_csv, str(plot_dir), alpha=cfg.global_cfg.alpha)
        print(f"[multi-run] Plots for '{identifier}' saved to {plot_dir}")


if __name__ == "__main__":
    main()
