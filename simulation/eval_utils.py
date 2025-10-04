import matplotlib.pyplot as plt
import numpy as np

from eval_utils import OUTPATH_FIG


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

