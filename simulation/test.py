import numpy as np
import warnings

warnings.filterwarnings("ignore")

from eval_const import OUTPATH_FIG
from eval_utils import print_results, plot_results
from eval import run_simulation


alpha = 0.1
random_seed = 17
num_samples = 2000

gmm_params = {
    "K": 3,
    "d": 4,
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
    "n_estimators": 500,         # more trees → stabler predictions
    "max_features": "sqrt",      # good bias/variance tradeoff for many problems
    "max_depth": None,           # let trees expand, regularized by min_samples_*
    "min_samples_leaf": 5,       # prevents tiny leaves (reduces variance)
    "min_samples_split": 10,
    "bootstrap": True,
    "oob_score": True,           # use OOB to estimate generalization (no val split)
    "n_jobs": -1,
}

split_params = {
    "train_ratio": 0.375,
    "val_ratio": 0.125,
}

print("Basic simulation parameters:")
print(f"Sample size: {num_samples}")
print(f"Alpha (significance level): {alpha}")
print(f"Gaussian mixture: K={gmm_params['K']}, d={gmm_params['d']}")
print(f"Random forest trees: {rf_params['n_estimators']}")


# Run simulation
results, X_test_0, Y_test, predictions_test, diagnostics = run_simulation(
    num_samples=num_samples,
    alpha=alpha,
    random_seed=random_seed,
    rf_params=rf_params,
    **gmm_params,
    **pcp_params,
    **split_params,
)

# Print results
print_results(results, alpha)

print(f"\n{'=' * 70}")
print("ANALYSIS")
print(f"{'=' * 70}")

scp_coverage = results['SCP']['coverage_rate']
scp_length = results['SCP']['avg_length']

print(f"- SCP (baseline): {scp_coverage:.3f} coverage, {scp_length:.2f} average length")

for name, result in results.items():
    if name == 'SCP':
        continue
    cov_diff = result['coverage_rate'] - scp_coverage
    len_ratio = scp_length / result['avg_length'] if result['avg_length'] else float('inf')
    print(f"- {name}: {result['coverage_rate']:.3f} coverage ({cov_diff:+.3f} vs SCP), "
          f"{len_ratio:.2f}x length efficiency")

true_clusters = diagnostics['true_clusters_test']
ambiguous_mask = diagnostics['ambiguous_mask_test']
noisy_labels = diagnostics['noisy_labels_test']

noisy_accuracy = np.mean(noisy_labels == true_clusters)
ambiguous_rate = np.mean(ambiguous_mask)

print(f"- Noisy label accuracy (test): {noisy_accuracy:.3f}")
print(f"- Ambiguous region rate (test): {ambiguous_rate:.3f}")


feature_indices = list(range(min(3, X_test_0.shape[1]))) or [0]

if len(feature_indices) < 3:
    for feat_idx in feature_indices:
        print(f"\nGenerating plots for feature {feat_idx}...")
        plot_results(
            results,
            X_test_0,
            Y_test,
            predictions_test,
            feature_idx=feat_idx,
            num_samples=num_samples,
            n_tree=rf_params['n_estimators'],
            seed=random_seed,
            setting=f"gmm_K{gmm_params['K']}_d{gmm_params['d']}",
            temperature=gmm_params['temperature'],
            true_clusters=true_clusters,
            ambiguous_mask=ambiguous_mask,
            noisy_labels=noisy_labels,
            save=True,
        )
else:
    print("\nRandomly selecting 3 features for plotting...")
    selected_feats = np.random.choice(feature_indices, size=3, replace=False)
    for feat_idx in selected_feats:
        print(f"\nGenerating plots for feature {feat_idx}...")
        plot_results(
            results,
            X_test_0,
            Y_test,
            predictions_test,
            feature_idx=feat_idx,
            num_samples=num_samples,
            n_tree=rf_params['n_estimators'],
            seed=random_seed,
            setting=f"gmm_K{gmm_params['K']}_d{gmm_params['d']}",
            temperature=gmm_params['temperature'],
            true_clusters=true_clusters,
            ambiguous_mask=ambiguous_mask,
            noisy_labels=noisy_labels,
            save=True,
            show=False,
        )

out_fig = OUTPATH_FIG.format(num_samples=num_samples)
print(f"\nSimulation completed! Check '{out_fig}' for plots.")
