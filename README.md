LatentCP
---

> [!NOTE]
> It's an ongoing project. A polished version will be released later. If you find this project interesting, you might also want to check out [MDCP](https://github.com/AragornBFRer/MDCP)!

## Setup

The simulator:

- **Auxiliary feature:** $R \mid Z = k \sim \mathcal{N}(\mu_{R,k}, 1)$ with $\mu_R = (-3, -1, 1, 3)$.
	$\alpha = (1, 2, 3, 4)$, and $\varepsilon_Z \sim \mathcal{N}(0, \sigma_Z^2)$ with $\sigma = (1, 2, 4, 8)$.
- **Outcome:** $Y = \eta_0 + \eta^\top X + \alpha_Z + \varepsilon_Z$, where $\eta_0 = 0.5$, $\eta = (1, -0.5, 0.8)$, $\alpha = (1, 2, 3, 4)$, and $\varepsilon_Z \sim \mathcal{N}(0, \sigma_Z^2)$ with $\sigma = (1, 2, 4, 8)$.

	The actual shift used in any run is $\tilde{\alpha}_k = \delta \cdot \alpha_k$ where $\delta$ is a list of scaling factors for $\alpha_k$
	(default: five values `[0.5, 0.75, 1.0, 1.5, 2.0]`). This “mixture separation”
	knob lets you sweep how distinct the clusters are without redefining $\alpha$.
- No leakage: `R` only informs `Y` through the latent cluster.

Those constants can be overridden through the `dgp` block in the YAML (see
`alpha`, `sigma`, `mu_r`, `eta0`, `eta`).

EM routine:

`src/doc_em.py` implements the EM algorithm:

- E-step scores jointly log-likelihoods for `(R, Y)` given cluster `k`.
- M-step performs the weighted least squares update for `(η₀, η)` and updates
	`μ_{R,k}` via responsibility-weighted means.
- Responsibilities on calibration data use the observed `(R, Y)`, whereas test
	memberships fall back to the R-only formula $\pi_k(R) \propto \pi_k \mathcal{N}(R \mid \mu_{R,k}, 1)$.
- EM-PCP reuses these memberships instead of fitting a separate joint GMM.

---

## 1. Run the experiment

Quick start (Linux/macOS)

```bash
chmod +x run.sh
# args: <config> <num_trials> <plots_dir> <results_csv>
./run.sh experiments/configs/gmm_em.yaml 50 experiments/plots experiments/results/gmm_em_results.csv
```

Windows PowerShell

```powershell
python main.py --config experiments/configs/gmm_em.yaml --trials 50
```

Arguments mirror the CLI in `main.py`:

- `--config`: path to a YAML experiment spec.
- `--trials`: overrides the set of seeds defined in the YAML (`1..trials`).
- `--results`: optional destination CSV (defaults to `io.results_csv`).

Sequential execution: multi-config launcher

Sweep several YAML configs without manually juggling output folders. Use `multi_main.py`:

```bash
python multi_main.py \
		--run experiment_1=experiments/configs/sample_gmm_em_1.yaml \
		--run experiment_2=experiments/configs/sample_gmm_em_2.yaml \
		--trials 10 \
		--results-root experiments/results \
		--plots-root experiments/plots
```

Config:

Default configs live in `experiments/configs/gmm_em.yaml`.


Key facts:

- Each `--run` flag takes `<identifier>=<config-path>`. Identifiers determine the
	subfolders (e.g., `experiments/results/experiment_1/results.csv`).
- `--trials` works like the single-config CLI: it replaces the YAML seeds with
	`range(1, trials+1)` for every run.
- Use `--skip-plots` if you only need the CSVs; otherwise the launcher calls
	`generate_all_plots` for each identifier and saves them under
	`plots-root/<identifier>/`.
- Outputs are naturally segmented, so you can start the same command on multiple
	machines or schedule different identifiers independently without overwriting
	one another.
---

## 2. Choose which baselines to compare

The YAML file now contains a `pcp` section controlling each posterior-conformal variant:

```yaml
pcp:
	base:
		enabled: true   # residual-driven PCP on (X,R)
	em:
		enabled: false  # EM-PCP using joint memberships over (X,R,Y)
```

Flip any `enabled` flag to `false` to skip that baseline. Every sub-block also inherits the tunable hyperparameters defined in `src/config.py`:

| Field | Meaning |
| --- | --- |
| `n_thresholds` | Number of residual quantile levels per PCP variant |
| `max_clusters` / `cluster_r2_tol` | Factorization rank + R² tolerance for template selection |
| `precision_grid` / `precision_trials` | Grid-search candidates for the multinomial precision `m` |
| `clip_eps`, `proj_lr`, `proj_max_iter`, `proj_tol` | Numerical safeguards for simplex projection |

Per-variant overrides look like:

```yaml
pcp:
	base:
		enabled: true
		n_thresholds: 15
		precision_grid: [20, 40, 80]
```

Other key YAML knobs:

- `global`: seeds, split sizes, target `alpha`.
- `dgp`: latent structure (`K_list`, `delta_list`, `alpha`, `sigma`, `mu_r`, `eta0`, `eta`).
- `em_fit`: how to fit responsibilities (`use_X_in_em`, covariance model, iterations).
- `model`: RandomForest hyperparameters shared by CQR-ignoreZ and the PCP residual model.

Included predictors / objects in this repo:

| Name | What it means / uses |
| --- | --- |
| **CQR-ignoreZ** | RandomForest quantile regressor on `[X; R]` with conformalized quantile regression calibration. Still available, but mainly serves as an external reference. |
| **Oracle (true μ, Z)** | **Fully oracle split-conformal baseline**: uses the simulator's ground-truth regression function $\mu(x,z)=\eta_0+\eta^\top x+\alpha_z$ with the **true** latent label $Z$ revealed. No regressor is fit; we calibrate $\hat q$ on $|Y-\mu(X,Z)|$ and report intervals $[\mu(X,Z)\pm\hat q]$. Columns: `coverage_oracle_full`, `length_oracle_full`. |
| **PCP (X,R)** | RandomForest mean model on `[X; R]` produces residuals `|y - μ_rf(x,r)|` that drive the PCP clustering/reweighting pipeline. |
| **PCP (X,Z)** | Oracle PCP that augments the regressors and clustering features with the true latent one-hot $Z$. Helpful for gauging the ceiling when the latent label is known. |
| **PCP (X,$\hat{z}$)** | Uses doc-EM responsibilities $\hat{z} = \pi(x,r,y)$ (soft memberships) instead of the oracle $Z$, measuring how much benefit calibrated memberships alone provide. |
| **PCP (X,R,$\hat{z}$)** | Blends observed $R$ with the soft memberships $\hat{z}$, letting PCP cluster on both auxiliary signals simultaneously. |
| **EM-PCP** | The membership-aware conformal predictor that weights residuals by $\pi(x,r,y)$ via `MembershipPCPModel`, producing sets $C_{\text{em}}(x,r) = \{ y : |y - \mu_{\text{joint}}(x,r)| \le q_{\text{em}}(\pi(x,r,y)) \}$. |
| **EM-R / EM-RX** | Responsibility pipelines feeding EM-PCP: EM-R fits the GMM on $R$ alone (features $\tau_R(r)$), while EM-RX fits on $[R; X]$ ($\tau_{RX}(x,r)$), toggled by `em_fit.use_X_in_em`. |
| **MembershipPCPModel** | Consumes membership weights $\pi$ and outputs quantiles $q(\pi)$ by multinomial-precision weighting, enabling set construction $C(\pi) = \{ y : |y - \mu_{\text{joint}}| \le q(\pi) \}$. |

---

## 3. Visualize metrics

After the CSV is produced, call:

```bash
python plot_results.py --config experiments/configs/gmm_em.yaml --results experiments/results/gmm_em_results.csv --out experiments/plots
```

The plotter automatically discovers all columns in the CSV. When PCP-base or EM-PCP are enabled you will see extra curves (e.g., `coverage_pcp_base`, `coverage_em_pcp`, diagnostic histograms). Output figures land in `experiments/plots/`; use the table below as a cheat sheet:

| Plot | Description |
| --- | --- |
| `coverage_vs_grid.png` | Coverage for every baseline vs. the sweep variable (δ by default). The dotted line is the target coverage `1 - α`. |
| `length_vs_grid.png` | Absolute interval length for each baseline; lower is better. |
| `length_vs_grid_pcp.png` | PCP-only comparison (EM-PCP vs. all PCP baselines) restricted to EM-R runs so you can focus on how each feature choice impacts set length. |
| `mean_max_tau_vs_grid.png` | Average sharpness of EM responsibilities. High values mean clusters are well separated. |
| `z_feature_mse_vs_grid.png` | MSE between the EM-soft features and the oracle latent means. Lower values indicate better feature recovery. |
| `em_iterations_hist.png` | Histogram of EM iteration counts, useful for spotting difficult configurations. |
