"""
Simulation study comparing multiple conformal prediction methods.
1. SCP (Split Conformal Prediction) - standard conformal prediction baseline
2. Cluster-wise Oracle - conditional on discrete feature clusters (Equation 5)
3. PCP - complete posterior conformal prediction implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from pcp import simulate_data, train_val_test_split, SCP, PCP

from eval_const import OUTPATH_FIG
from eval_utils import print_results, plot_results


class ClusterwiseOracle:
    """
    Cluster-wise calibration oracle method (Equation 5)
    
    For settings 1-3: Each discrete X value forms its own cluster based on feature thresholds.
    For settings 4-5: Uses the true cluster assignments from the data generation process.
    """

    def __init__(self, setting=1, meta=None):
        self.cluster_boundaries = None
        self.name = "Cluster-wise Oracle"
        self.setting = setting
        self.meta = meta  # Metadata containing true cluster assignments for settings 4-5

    def _get_ground_truth_clusters(self, X, indices=None):
        """Create clusters based on ground truth from data generation process
        
        The oracle knows the true clustering structure from how the data was generated:
        - Setting 1: Continuous variance (no discrete clusters - should not use this method)
        - Setting 2: Two clusters based on X[:, 0] <= 5 vs X[:, 0] > 5
        - Setting 3: Three clusters based on X[:, 0] <= 3, (3,6], X[:, 0] > 6
        - Setting 4: Clustered sparse linear model - uses true cluster assignments from metadata
        - Setting 5: Block-wise clusters - uses true cluster assignments from metadata
        
        Args:
            X: Feature matrix (may be standardized)
            indices: Original indices in the full dataset (for settings 4-5)
        """
        if self.setting in [1, 2, 3]:
            # Original settings use feature-based clustering
            x_first_feature = X[:, 0]  # The first feature determines the clustering
            
            if self.setting == 1:
                # Setting 1 has continuous variance change, not discrete clusters
                # Fall back to creating many fine-grained clusters based on first feature
                # Discretize into bins for approximation
                bins = np.linspace(0, 8, 21)  # 20 bins across [0,8] range
                cluster_ids = np.digitize(x_first_feature, bins)
            elif self.setting == 2:
                # Setting 2: Two clusters - X[:, 0] <= 5 vs X[:, 0] > 5
                cluster_ids = (x_first_feature > 5).astype(int)
            elif self.setting == 3:
                # Setting 3: Three clusters - X[:, 0] <= 3, (3,6], X[:, 0] > 6
                cluster_ids = np.zeros(len(x_first_feature), dtype=int)
                cluster_ids[x_first_feature <= 3] = 0
                cluster_ids[(x_first_feature > 3) & (x_first_feature <= 6)] = 1
                cluster_ids[x_first_feature > 6] = 2
                
        elif self.setting in [4, 5]:
            # High-dimensional settings use true cluster assignments from metadata
            if self.meta is None or 'clusters' not in self.meta:
                raise ValueError(f"Setting {self.setting} requires metadata with true cluster assignments.")
            
            if indices is None:
                raise ValueError(f"Setting {self.setting} requires original data indices to map cluster assignments.")
            
            # Extract the cluster assignments for the given indices
            cluster_ids = self.meta['clusters'][indices]
            
        else:
            raise ValueError(f"Unknown setting: {self.setting}. Supported settings are 1, 2, 3, 4, 5.")
            
        return cluster_ids.tolist()
    
    def fit_clusters(self, X_val, R_val, val_indices=None):
        """Fit clusters on validation data using ground truth clustering"""
        # Use ground truth clustering based on data generation process
        cluster_ids = self._get_ground_truth_clusters(X_val, val_indices)
        
        # Store residuals by cluster
        self.cluster_residuals = {}
        for i, cluster_id in enumerate(cluster_ids):
            if cluster_id not in self.cluster_residuals:
                self.cluster_residuals[cluster_id] = []
            self.cluster_residuals[cluster_id].append(R_val[i])
        
        # Convert to numpy arrays
        for cluster_id in self.cluster_residuals:
            self.cluster_residuals[cluster_id] = np.array(self.cluster_residuals[cluster_id])
    
    def calibrate(self, X_val, R_val, X_test, R_test, predictions_test, alpha, 
                  val_indices=None, test_indices=None):
        """Generate prediction intervals using cluster-wise oracle method"""
        self.fit_clusters(X_val, R_val, val_indices)
        
        quantiles = []
        coverage = []
        
        # Get cluster assignments for test points using ground truth
        test_cluster_ids = self._get_ground_truth_clusters(X_test, test_indices)
        
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
    
    # For settings 4 and 5, we need the metadata containing true cluster assignments
    if setting in [4, 5]:
        X, Y, meta = simulate_data(num_samples, setting, return_meta=True, random_state=random_seed)
    else:
        X, Y = simulate_data(num_samples, setting)
        meta = None
        
    if setting in [4, 5]:
        # For high-dimensional settings, we need the indices to map cluster assignments
        # First get the normal split
        X_train, X_val, X_test, Y_train, Y_val, Y_test, X_val_0, X_test_0 = train_val_test_split(X, Y, p=1/4, p2=1/4, random_state=random_seed)
        
        # Now get the indices by calling with return_index=True
        _, _, _, _, _, _, _, idx_test = train_val_test_split(X, Y, p=1/4, p2=1/4, return_index=True, random_state=random_seed)
        
        # We need to compute idx_val manually
        np.random.seed(random_seed)
        n = X.shape[0]
        train_size = int(n * 1/4)
        val_size = int(n * 1/4)
        indices = np.random.permutation(n)
        idx_val = indices[train_size:train_size + val_size]
    else:
        X_train, X_val, X_test, Y_train, Y_val, Y_test, X_val_0, X_test_0 = train_val_test_split(X, Y, p=1/4, p2=1/4)
        idx_val, idx_test = None, None
    
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
    
    # --Cluster-wise Oracle Method
    print("-- Running Cluster-wise Oracle...")
    cluster_oracle = ClusterwiseOracle(setting=setting, meta=meta)
    cluster_quantiles, cluster_coverage = cluster_oracle.calibrate(
        X_val_0, R_val, X_test_0, R_test, predictions_test, alpha,
        val_indices=idx_val, test_indices=idx_test)
    
    # show cluster information using original features
    val_clusters = cluster_oracle._get_ground_truth_clusters(X_val_0, idx_val)
    unique_clusters, cluster_counts = np.unique(val_clusters, return_counts=True)
    print(f"   Validation clusters found: {len(unique_clusters)} clusters")
    for cluster_id, count in zip(unique_clusters, cluster_counts):
        print(f"   Cluster {cluster_id}: {count} samples")
    
    if setting in [1, 2, 3]:
        print(f"   X_val_0 first feature range: [{X_val_0[:, 0].min():.2f}, {X_val_0[:, 0].max():.2f}]")
    
    test_clusters = cluster_oracle._get_ground_truth_clusters(X_test_0, idx_test)
    unique_test_clusters, test_cluster_counts = np.unique(test_clusters, return_counts=True)
    print(f"   Test clusters found: {len(unique_test_clusters)} clusters")
    for cluster_id, count in zip(unique_test_clusters, test_cluster_counts):
        print(f"   Test Cluster {cluster_id}: {count} samples")
    
    if setting in [1, 2, 3]:
        print(f"   X_test_0 first feature range: [{X_test_0[:, 0].min():.2f}, {X_test_0[:, 0].max():.2f}]")
    elif setting in [4, 5]:
        print(f"   X_val_0 shape: {X_val_0.shape}, X_test_0 shape: {X_test_0.shape}")
        if meta is not None:
            print(f"   Total clusters in metadata: {len(np.unique(meta['clusters']))}")
            print(f"   Cluster distribution in val: {dict(zip(*np.unique(val_clusters, return_counts=True)))}")
            print(f"   Cluster distribution in test: {dict(zip(*np.unique(test_clusters, return_counts=True)))}")

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


def main():
    """Main simulation function"""
    # Simulation parameters
    num_samples = 2000
    alpha = 0.1
    setting = 5  # Test the new high-dimensional setting with block-wise clusters
    random_seed = 42
    n_tree = 100
    
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
    
    print(f"- Cluster-wise Oracle conditions on true clusters from data generation")
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