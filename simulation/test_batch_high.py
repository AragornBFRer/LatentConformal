import numpy as np
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore")

from eval import run_simulation
from eval_utils import print_results

alpha = 0.1
random_seed = 20
num_samples = 500  # Reduced sample size to make p > n more pronounced
num_trials = 100

gmm_params = {
    "K": 3,
    "d": 800,  # High dimensional: p = 800 >> n = 500
    "mean_scale": 4.0,
    "temperature": 0.75,
    "cluster_spread": 1.0,
    "response_noise": 0.6,
    "deterministic_margin": 0.2,
}

pcp_params = {
    "pcp_fold": 10,
    "pcp_grid": 20,
}

rf_params = {
    "n_estimators": 100,         # Fewer trees for high-dim efficiency
    "max_features": "sqrt",      # Even more important in high-dim to prevent overfitting  
    "max_depth": 10,             # Limit depth to prevent overfitting in high-dim
    "min_samples_leaf": 10,      # Larger leaf size to regularize
    "min_samples_split": 20,     # Higher split threshold to prevent overfitting
    "bootstrap": True,
    "oob_score": True,           # use OOB to estimate generalization (no val split)
    "n_jobs": -1,
}

split_params = {
    "train_ratio": 0.375,
    "val_ratio": 0.125,
}

output_dir = Path("out") / "low dim (large batch)"
output_dir.mkdir(parents=True, exist_ok=True)

print("Basic simulation parameters:")
print(f"Sample size: {num_samples}")
print(f"Alpha (significance level): {alpha}")
print(f"Gaussian mixture: K={gmm_params['K']}, d={gmm_params['d']}")
print(f"Random forest trees: {rf_params['n_estimators']}")
print(f"Number of trials: {num_trials}")

rng = np.random.default_rng(random_seed)
trial_records = []
diagnostic_records = []

for trial_idx in range(num_trials):
    seed = int(rng.integers(0, 1_000_000_000))
    print(f"\n{'=' * 70}")
    print(f"Trial {trial_idx + 1}/{num_trials} (seed={seed})")

    results, _, _, _, diagnostics = run_simulation(
        num_samples=num_samples,
        alpha=alpha,
        random_seed=seed,
        rf_params=rf_params,
        **gmm_params,
        **pcp_params,
        **split_params,
    )

    print_results(results, alpha)

    scp_metrics = results.get("SCP", {})
    scp_coverage = float(scp_metrics.get("coverage_rate", np.nan))
    scp_length = scp_metrics.get("avg_length", np.nan)
    scp_length = float(scp_length) if scp_length is not None else np.nan

    for name, result in results.items():
        coverage = float(result.get("coverage_rate", np.nan))
        avg_length = result.get("avg_length", np.nan)
        avg_length = float(avg_length) if avg_length is not None else np.nan

        if np.isfinite(scp_length) and np.isfinite(avg_length) and avg_length > 0:
            length_eff = scp_length / avg_length
        else:
            length_eff = np.nan

        trial_records.append(
            {
                "trial": trial_idx + 1,
                "seed": seed,
                "method": name,
                "coverage_rate": coverage,
                "avg_length": avg_length,
                "coverage_vs_scp": coverage - scp_coverage if np.isfinite(scp_coverage) else np.nan,
                "length_efficiency_vs_scp": length_eff,
            }
        )

    true_clusters = diagnostics["true_clusters_test"]
    ambiguous_mask = diagnostics["ambiguous_mask_test"]
    noisy_labels = diagnostics["noisy_labels_test"]

    diagnostic_records.append(
        {
            "trial": trial_idx + 1,
            "seed": seed,
            "noisy_accuracy": float(np.mean(noisy_labels == true_clusters)),
            "ambiguous_rate": float(np.mean(ambiguous_mask)),
        }
    )

stats_df = pd.DataFrame(trial_records)
diagnostics_df = pd.DataFrame(diagnostic_records)
identifier = f"samples{num_samples}_trials{num_trials}_temperature{gmm_params['temperature']}_margin{gmm_params['deterministic_margin']}_K{gmm_params['K']}_d{gmm_params['d']}"

# Merge diagnostics with trial stats on trial and seed
merged_df = stats_df.merge(diagnostics_df, on=['trial', 'seed'], how='left')

results_path = output_dir / f"result_{identifier}.csv"
merged_df.to_csv(results_path, index=False)

print(f"\nStored results at: {results_path}")
