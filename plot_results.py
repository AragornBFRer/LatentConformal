from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.plotting import generate_all_plots

import warnings
warnings.filterwarnings("ignore")

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots for latent conformal experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/configs/gmm_em.yaml",
        help="Path to experiment configuration (used for defaults and alpha)",
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
        help="Output directory for generated figures",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_path = Path(args.results or cfg.io_cfg.results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    generate_all_plots(results_path, args.out, alpha=cfg.global_cfg.alpha)
    print(f"Saved plots to {args.out}")


if __name__ == "__main__":
    main()
