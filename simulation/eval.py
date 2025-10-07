"""
Simulation study comparing multiple conformal prediction methods.
1. SCP (Split Conformal Prediction) - standard conformal prediction baseline
2. Cluster-wise Oracle - conditional on discrete feature clusters (Equation 5)
3. PCP - complete posterior conformal prediction implementation
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

from pcp import simulate_data, train_val_test_split, SCP, PCP
from eval_const import OUTPATH_FIG
from eval_utils import print_results, plot_results


class ClusterwiseOracle:
    """Oracle method that conditions on the latent clusters used to generate the data."""

    def __init__(self, true_clusters):
        self.name = "Cluster-wise Oracle"
        self.true_clusters = np.asarray(true_clusters)
        self.cluster_residuals = {}
        self.alpha_used = None
        self.missing_clusters_ = set()

    def fit_clusters(self, R_val, val_indices):
        cluster_ids = self.true_clusters[val_indices]
        self.cluster_residuals = {}
        for cluster_id, residual in zip(cluster_ids, R_val):
            self.cluster_residuals.setdefault(int(cluster_id), []).append(residual)
        for cluster_id in self.cluster_residuals:
            self.cluster_residuals[cluster_id] = np.asarray(self.cluster_residuals[cluster_id])

    def calibrate(self, R_val, R_test, alpha, val_indices, test_indices):
        self.alpha_used = alpha
        self.fit_clusters(R_val, val_indices)

        global_quantile = np.quantile(R_val, 1 - alpha)
        quantiles = []
        coverage = []
        missing_clusters = set()

        test_cluster_ids = self.true_clusters[test_indices]

        for i, cluster_id in enumerate(test_cluster_ids):
            residuals = self.cluster_residuals.get(int(cluster_id))
            if residuals is None or residuals.size <= 5:
                raise ValueError(f"Cluster {cluster_id} not found in validation set or not sufficiently populated.")
            else:
                quantile = np.quantile(residuals, 1 - alpha)

            quantiles.append(float(quantile))
            coverage.append(float(R_test[i] <= quantile))

        self.missing_clusters_ = missing_clusters
        return quantiles, coverage


def run_simulation(num_samples=1200, alpha=0.1, random_seed=42, pcp_fold=20, pcp_grid=20,
                   K=3, d=2, mean_scale=4.0, temperature=1.0, cluster_spread=1.0, response_noise=0.5,
                   deterministic_margin=0.2, rf_params=None,
                   train_ratio=0.4, val_ratio=0.3):
    """Run the GMM-based simulation comparing SCP, Oracle, and PCP."""

    print(f"Running GMM simulation with {num_samples} samples, alpha={alpha}, K={K}, d={d}, temperature={temperature}")
    print("=" * 70)

    X, Y, meta = simulate_data(
        num_samples,
        K=K,
        d=d,
        mean_scale=mean_scale,
        temperature=temperature,
        cluster_spread=cluster_spread,
        response_noise=response_noise,
        deterministic_margin=deterministic_margin,
        random_state=random_seed,
        return_meta=True,
    )

    X_train, X_val, X_test, Y_train, Y_val, Y_test, \
    X_val_0, idx_val, X_test_0, idx_test = train_val_test_split(
        X, Y, p=train_ratio, p2=val_ratio, return_index=True, random_state=random_seed
    )

    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    rf = RandomForestRegressor(
        random_state=random_seed,
        **(rf_params if rf_params is not None else {})
    )
    rf.fit(X_train, Y_train)

    predictions_val = rf.predict(X_val)
    R_val = np.abs(Y_val - predictions_val)
    predictions_test = rf.predict(X_test)
    R_test = np.abs(Y_test - predictions_test)

    print(f"Model trained. Val RMSE: {np.sqrt(np.mean((Y_val - predictions_val) ** 2)):.3f}")
    print(f"Test RMSE: {np.sqrt(np.mean((Y_test - predictions_test) ** 2)):.3f}")

    results = {}

    # Baseline: standard split conformal
    print("\n-- Running SCP...")
    scp_quantiles, scp_coverage = SCP(R_val, R_test, alpha)
    results['SCP'] = {
        'quantiles': scp_quantiles,
        'coverage': scp_coverage,
        'avg_length': np.mean(scp_quantiles) * 2,
        'coverage_rate': np.mean(scp_coverage),
    }

    # Oracle that sees the true latent clusters
    print("-- Running Cluster-wise Oracle...")
    cluster_oracle = ClusterwiseOracle(meta['true_clusters'])
    cluster_quantiles, cluster_coverage = cluster_oracle.calibrate(
        R_val, R_test, alpha, idx_val, idx_test
    )

    val_true_clusters = meta['true_clusters'][idx_val]
    test_true_clusters = meta['true_clusters'][idx_test]
    val_noisy = meta['noisy_labels'][idx_val]
    test_noisy = meta['noisy_labels'][idx_test]
    val_ambiguous = meta['ambiguous_mask'][idx_val]
    test_ambiguous = meta['ambiguous_mask'][idx_test]

    val_unique, val_counts = np.unique(val_true_clusters, return_counts=True)
    test_unique, test_counts = np.unique(test_true_clusters, return_counts=True)

    print(f"   Validation clusters: {len(val_unique)} (avg size {np.mean(val_counts):.1f})")
    print(f"   Test clusters:       {len(test_unique)} (avg size {np.mean(test_counts):.1f})")
    print(f"   Noisy label accuracy (val/test): {np.mean(val_noisy == val_true_clusters):.3f} / {np.mean(test_noisy == test_true_clusters):.3f}")
    print(f"   Ambiguous region rate (val/test): {np.mean(val_ambiguous):.3f} / {np.mean(test_ambiguous):.3f}")
    if cluster_oracle.missing_clusters_:
        print(f"   Warning: Missing clusters in validation set {sorted(cluster_oracle.missing_clusters_)}; using global quantile fallback.")

    results['Cluster-wise Oracle'] = {
        'quantiles': cluster_quantiles,
        'coverage': cluster_coverage,
        'avg_length': np.mean(cluster_quantiles) * 2,
        'coverage_rate': np.mean(cluster_coverage),
    }

    # PCP with externally supplied (noisy) cluster probabilities
    print("-- Running PCP (with pre-labeled clusters)...")
    val_cluster_probs = meta['noisy_membership'][idx_val]
    test_cluster_probs = meta['noisy_membership'][idx_test]

    pcp_model = PCP(fold=pcp_fold, grid=pcp_grid)
    pcp_model.train(X_val, R_val, info=False, cluster_probs=val_cluster_probs)

    pcp_quantiles, pcp_coverage = pcp_model.calibrate(
        X_val,
        R_val,
        X_test,
        R_test,
        alpha,
        return_pi=False,
        finite=True,
        max_iter=5,
        tol=0.01,
        cluster_probs_val=val_cluster_probs,
        cluster_probs_test=test_cluster_probs,
    )

    results['PCP'] = {
        'quantiles': pcp_quantiles,
        'coverage': pcp_coverage,
        'avg_length': np.mean(pcp_quantiles) * 2,
        'coverage_rate': np.mean(pcp_coverage),
    }
    print("   PCP completed successfully!")

    diagnostics = {
        'true_clusters_test': test_true_clusters,
        'ambiguous_mask_test': test_ambiguous,
        'noisy_labels_test': test_noisy,
        'temperature': temperature,
    }

    return results, X_test_0, Y_test, predictions_test, diagnostics

