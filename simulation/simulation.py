"""
Simulation study comparing multiple conformal prediction methods.
1. SCP (Split Conformal Prediction) - standard conformal prediction baseline
2. Naive Oracle - uses global score functions (naive baseline, not true conditional)
3. Cluster-wise Oracle - conditional on discrete feature clusters (Equation 5)
4. PCP - complete posterior conformal prediction implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Import from the existing pcp.py
from pcp import simulate_data, train_val_test_split, SCP, PCP


OUTPATH_FIG = 'out/simulation_comparison_samples{num_samples}.png'


class NaiveOracle:
    """
    Naive Oracle method using global score functions
    
    Uses global empirical distribution of residuals without conditioning on features.
    This is NOT the true conditional distribution but rather a naive baseline that
    ignores feature-dependent heteroscedasticity.
    """
    
    def __init__(self):
        self.name = "Naive Oracle"
    
    def calibrate(self, X_val, R_val, X_test, R_test, predictions_test, alpha):
        """Generate prediction intervals using oracle method"""
        n_val = len(R_val)
        quantiles = []
        coverage = []
        
        for i in range(len(X_test)):
            # Use empirical quantile from validation set as proxy for oracle
            # Add small noise to break ties
            extended_residuals = np.append(R_val, 0)
            
            # Binary search for the quantile
            low, high = 0, np.max(R_val) * 2
            epsilon = 1e-6
            
            for _ in range(10000):  # Limit iterations
                if high - low < epsilon:
                    break
                    
                mid = (low + high) / 2
                extended_residuals[-1] = mid
                
                # Calculate empirical probability
                p_empirical = np.mean(extended_residuals >= mid)
                
                if p_empirical > alpha:
                    high = mid
                else:
                    low = mid
            
            quantile = (low + high) / 2
            quantiles.append(quantile)
            coverage.append(1 if R_test[i] <= quantile else 0)
        
        return quantiles, coverage


class ClusterwiseOracle:
    """
    Cluster-wise calibration oracle method (Equation 5)
    
    Each discrete X value forms its own cluster. For continuous features,
    we discretize them into bins to create discrete clusters.
    """

    def __init__(self, setting=1):
        self.cluster_boundaries = None
        self.name = "Cluster-wise Oracle"
        self.setting = setting

    def _discretize_features(self, X):
        """Create clusters based on distinct X values,
        points with identical X values are grouped together"""
        cluster_ids = []
        for i in range(X.shape[0]):
            cluster_ids.append(tuple(X[i]))
        return cluster_ids
    
    def fit_clusters(self, X_val, R_val):
        """Fit clusters on validation data"""
        # Discretize features to create clusters
        cluster_ids = self._discretize_features(X_val)
        
        # Store residuals by cluster
        self.cluster_residuals = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in self.cluster_residuals:
                self.cluster_residuals[cluster_id] = []
            self.cluster_residuals[cluster_id].append(R_val[i])
        
        # Convert to numpy arrays
        for cluster_id in self.cluster_residuals:
            self.cluster_residuals[cluster_id] = np.array(self.cluster_residuals[cluster_id])
    
    def calibrate(self, X_val, R_val, X_test, R_test, predictions_test, alpha):
        """Generate prediction intervals using cluster-wise oracle method"""
        self.fit_clusters(X_val, R_val)
        
        quantiles = []
        coverage = []
        
        # Get cluster assignments for test points
        test_cluster_ids = self._discretize_features(X_test)
        
        for i, cluster_id in enumerate(test_cluster_ids):
            # Get residuals for this cluster
            if cluster_id in self.cluster_residuals and len(self.cluster_residuals[cluster_id]) > 0:
                cluster_resid = self.cluster_residuals[cluster_id]
                # Need at least 2 points to compute meaningful quantile
                if len(cluster_resid) >= 2:
                    quantile = np.quantile(cluster_resid, 1-alpha)
                else:
                    # Use the single residual or fall back to global
                    quantile = np.quantile(R_val, 1-alpha)
            else:
                raise ValueError("Error in calibrating for the ClusterwiseOracle:\n",
                                 f"-->  Test point cluster {cluster_id} not seen in validation set.")

            quantiles.append(quantile)
            coverage.append(1 if R_test[i] <= quantile else 0)
        
        return quantiles, coverage


def run_simulation(num_samples=1200, alpha=0.1, setting=1, random_seed=42, n_tree=100):
    """Run the complete simulation comparing all methods"""
    print(f"Running simulation with {num_samples} samples, alpha={alpha}, setting={setting}")
    print("=" * 70)
    
    np.random.seed(random_seed)
    
    X, Y = simulate_data(num_samples, setting)
    X_train, X_val, X_test, Y_train, Y_val, Y_test, X_test_0 = train_val_test_split(X, Y, 1/3)
    
    print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    RF = RandomForestRegressor(random_state=random_seed, n_estimators=n_tree)
    RF.fit(X_train, Y_train)
    
    # Get predictions and residuals
    predictions_val = RF.predict(X_val)
    R_val = np.abs(Y_val - predictions_val)
    predictions_test = RF.predict(X_test)
    R_test = np.abs(Y_test - predictions_test)
    
    print(f"Model trained. Val RMSE: {np.sqrt(np.mean((Y_val - predictions_val)**2)):.3f}")
    print(f"Test RMSE: {np.sqrt(np.mean((Y_test - predictions_test)**2)):.3f}")
    

    results = {}
    
    # -- SCP Method (baseline)
    print("\n-- Running SCP...")
    scp_quantiles, scp_coverage = SCP(R_val, R_test, alpha)
    
    results['SCP'] = {
        'quantiles': scp_quantiles,
        'coverage': scp_coverage,
        'avg_length': np.mean(scp_quantiles) * 2,
        'coverage_rate': np.mean(scp_coverage)
    }
    
    # -- Naive Oracle Method
    if False:
        print("-- Running Naive Oracle...")
        naive_oracle = NaiveOracle()
        naive_quantiles, naive_coverage = naive_oracle.calibrate(X_val, R_val, X_test, R_test, 
                                                                predictions_test, alpha)
        
        results['Naive Oracle'] = {
            'quantiles': naive_quantiles,
            'coverage': naive_coverage,
            'avg_length': np.mean(naive_quantiles) * 2,
            'coverage_rate': np.mean(naive_coverage)
        }
    
    # --Cluster-wise Oracle Method
    if setting != 1:
        print("-- Running Cluster-wise Oracle...")
        cluster_oracle = ClusterwiseOracle(setting=setting)
        cluster_quantiles, cluster_coverage = cluster_oracle.calibrate(X_val, R_val, X_test, R_test,
                                                                    predictions_test, alpha)
        
        results['Cluster-wise Oracle'] = {
            'quantiles': cluster_quantiles,
            'coverage': cluster_coverage,
            'avg_length': np.mean(cluster_quantiles) * 2,
            'coverage_rate': np.mean(cluster_coverage)
        }
    
    # PCP Method
    print("-- Running PCP...")    
    pcp_model = PCP(fold=20, grid=20)
    pcp_model.train(X_val, R_val, info=False)
    
    # Generate prediction intervals 
    pcp_quantiles, pcp_coverage = pcp_model.calibrate(
        X_val, R_val, X_test, R_test, alpha, 
        return_pi=False, finite=True, max_iter=5, tol=0.01
    )
    
    results['PCP'] = {
        'quantiles': pcp_quantiles,
        'coverage': pcp_coverage,
        'avg_length': np.mean(pcp_quantiles) * 2,
        'coverage_rate': np.mean(pcp_coverage)
    }
    print("   PCP completed successfully!")

    return results, X_test_0, Y_test, predictions_test


def print_results(results, alpha):
    """Print comparison results"""
    print(f"\n{'='*70}")
    print("SIMULATION RESULTS")
    print(f"{'='*70}")
    print(f"Target coverage: {1-alpha:.1%}")
    print(f"{'Method':<20} {'Coverage Rate':<15} {'Avg Length':<15} {'Efficiency':<10}")
    print("-" * 70)
    
    baseline_length = None
    for method_name, result in results.items():
        coverage_rate = result['coverage_rate']
        avg_length = result['avg_length']
        
        if baseline_length is None:
            baseline_length = avg_length
            efficiency = 1.0
        else:
            efficiency = baseline_length / avg_length if avg_length > 0 else float('inf')
        
        print(f"{method_name:<20} {coverage_rate:<15.3f} {avg_length:<15.3f} {efficiency:<10.3f}")
    
    print("-" * 70)


def plot_results(results, X_test_0, Y_test, predictions_test, feature_idx=0, 
                 num_samples=None, n_tree=None, seed=None, setting=None):
    """Plot prediction intervals for visual comparison"""
    
    # Sort by feature for better visualization
    sort_idx = np.argsort(X_test_0[:, feature_idx])
    x_vals = X_test_0[sort_idx, feature_idx]
    y_vals = Y_test[sort_idx]
    pred_vals = predictions_test[sort_idx]
    
    # Filter out failed methods for plotting
    plot_results_dict = {k: v for k, v in results.items() if not k.endswith('(Failed)')}
    n_methods = len(plot_results_dict)
    
    # Determine subplot layout
    if n_methods == 2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes = axes.flatten()
    elif n_methods <= 4:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
    
    for i, (method_name, result) in enumerate(plot_results_dict.items()):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        quantiles_sorted = np.array(result['quantiles'])[sort_idx]
        
        # Handle infinite quantiles for display
        finite_mask = np.isfinite(quantiles_sorted)
        if np.any(finite_mask):
            max_finite_quantile = np.max(quantiles_sorted[finite_mask]) * 1.5
            quantiles_sorted = np.where(np.isfinite(quantiles_sorted), quantiles_sorted, max_finite_quantile)
        
        lower_bounds = pred_vals - quantiles_sorted
        upper_bounds = pred_vals + quantiles_sorted
        
        # Plot data points and prediction intervals
        ax.scatter(x_vals, y_vals, alpha=0.4, s=8, color='gray', label='True values')
        ax.fill_between(x_vals, lower_bounds, upper_bounds, alpha=0.3, color=colors[i % len(colors)])
        ax.plot(x_vals, pred_vals, 'k--', alpha=0.7, label='Predictions', linewidth=1)
        
        avg_length = result["avg_length"] if np.isfinite(result["avg_length"]) else "∞"
        ax.set_title(f'{method_name}\nCoverage: {result["coverage_rate"]:.3f}, Length: {avg_length}')
        ax.set_xlabel(f'Feature {feature_idx}')
        ax.set_ylabel('Response')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    
    outpath = OUTPATH_FIG.format(num_samples=num_samples)
    if n_tree is not None:
        outpath = outpath.replace('.png', f'_ntree{n_tree}.png')
    if seed is not None:
        outpath = outpath.replace('.png', f'_seed{seed}.png')
    if setting is not None:
        outpath = outpath.replace('out/', f'out/setting{setting}/')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Main simulation function"""
    # Simulation parameters
    num_samples = 1200
    alpha = 0.1
    setting = 1
    random_seed = 31
    n_tree = 120
    
    print("Simulation Study: Comparing Conformal Prediction Methods")
    print(f"Sample size: {num_samples}")
    print(f"Alpha (significance level): {alpha}")
    print(f"Data setting: {setting}")
    
    # Run simulation
    results, X_test_0, Y_test, predictions_test = run_simulation(
        num_samples=num_samples, 
        alpha=alpha, 
        setting=setting, 
        random_seed=random_seed,
        n_tree=n_tree,
    )
    
    # Print results
    print_results(results, alpha)
    
    # Generate analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    scp_coverage = results['SCP']['coverage_rate']
    scp_length = results['SCP']['avg_length']
    
    print(f"- SCP (baseline): {scp_coverage:.3f} coverage, {scp_length:.1f} average length")
    
    for name, result in results.items():
        if name != 'SCP':
            cov_diff = result['coverage_rate'] - scp_coverage
            len_ratio = scp_length / result['avg_length']
            print(f"- {name}: {result['coverage_rate']:.3f} coverage (+{cov_diff:+.3f}), "
                  f"{len_ratio:.2f}x length efficiency")
    
    print(f"- Naive Oracle shows performance with global (non-conditional) scoring")
    print(f"- Cluster-wise Oracle conditions on discrete feature bins")
    if 'PCP' in results:
        print(f"- PCP uses complete posterior conformal prediction methodology")
    else:
        print(f"- PCP failed due to numerical instability with this sample size")

    # Plot results
    print(f"\nGenerating plots...")
    plot_results(results, X_test_0, Y_test, predictions_test, 
                 num_samples=num_samples, n_tree=n_tree, seed=random_seed, setting=setting)

    out_fig = OUTPATH_FIG.format(num_samples=num_samples)
    print(f"\nSimulation completed! Check '{out_fig}' for plots.")

    return results


if __name__ == "__main__":
    results = main()