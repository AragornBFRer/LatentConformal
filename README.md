Default configs live in `experiments/configs/gmm_em.yaml`.

---

## XRZY simulation recap

The simulator now mirrors the specification in `sim_model-dgp.md`:

- **Latent clusters:** `K = 4` with uniform prior.
- **Observed covariates:** `X ∈ ℝ³` sampled i.i.d. from `𝒩(0, I₃)`.
- **Auxiliary feature:** `R | Z = k ∼ 𝒩(μ_{R,k}, 1)` with `μ_R = (-3, -1, 1, 3)`.
- **Outcome:** `Y = η₀ + ηᵀ X + α_Z + ε_Z`, where `η₀ = 0.5`, `η = (1, -0.5, 0.8)`,
	`α = (1, 2, 3, 4)`, and `ε_Z ∼ 𝒩(0, σ_Z²)` with `σ = (1, 2, 4, 8)`.
- No leakage: `R` only informs `Y` through the latent cluster.

Those constants can be overridden through the `dgp` block in the YAML (see
`alpha`, `sigma`, `mu_r`, `eta0`, `eta`).

### EM routine

`src/doc_em.py` implements the EM algorithm described in `sim_algo.md`:

- E-step scores jointly log-likelihoods for `(R, Y)` given cluster `k`.
- M-step performs the weighted least squares update for `(η₀, η)` and updates
	`μ_{R,k}` via responsibility-weighted means.
- Responsibilities on calibration data use the observed `(R, Y)`, whereas test
	memberships fall back to the R-only formula `π_k(R) ∝ π_k 𝒩(R | μ_{R,k}, 1)`.
- EM-PCP reuses these memberships instead of fitting a separate joint GMM.

The soft predictor and diagnostics (`mean_max_tau`, `z_feature_mse`) now rely on
these doc-style responsibilities.

---

## 1. Run the XRZY baselines

### One-click runner (Linux/macOS)

```bash
chmod +x run.sh
# args: <config> <num_trials> <plots_dir> <results_csv>
./run.sh experiments/configs/gmm_em.yaml 50 experiments/plots experiments/results/gmm_em_results.csv
```

Environment knobs for HPC machines:

- `RESULTS_PATH` (4th arg) lets you stream runs into a single CSV.
- `OMP_NUM_THREADS` / `MKL_NUM_THREADS` default to `1` inside `run.sh` to avoid MKL/KMeans issues on shared nodes—you can override before invoking the script.
- Set `SKIP_PLOTS=1 ./run.sh ...` if you only want the CSV (plots can be heavy on headless clusters).

### Windows PowerShell

```powershell
python main.py --config experiments/configs/gmm_em.yaml --trials 50
```

Arguments mirror the CLI in `main.py`:

- `--config`: path to a YAML experiment spec.
- `--trials`: overrides the set of seeds defined in the YAML (`1..trials`).
- `--results`: optional destination CSV (defaults to `io.results_csv`).

---

## 2. Choose which baselines to compare

The YAML file now contains a `pcp` section controlling each posterior-conformal variant:

```yaml
pcp:
	xrzy:
		enabled: true   # responsibilities based on R only (current default)
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
- `dgp`: latent structure (`K_list`, `delta_list`, `rho_list`, `sigma_y_list`, `b_scale_list`).
- `em_fit`: how to fit responsibilities (`use_X_in_em`, covariance model, iterations).
- `model`: ridge penalties for oracle/soft/ignore/XRZY regressors.

Included predictors / objects in this repo:

| Name | What it means / uses |
| --- | --- |
| **Oracle-Z** | Linear regressor trained with the true latent centroids `μ_Z`. Serves as the unattainable lower bound for interval length. |
| **EM-soft** | Same architecture as Oracle but replaces `μ_Z` with responsibility-weighted estimates from the EM fit (R-only or [R;X], depending on `use_X_in_em`). |
| **Ignore-Z** | Baseline that regresses `Y` on `X` only, ignoring any latent structure; mirrors standard split conformal. |
| **XRZY PCP** | Posterior-conformal method that clusters calibration residual CDFs as a function of `R` only (responsibilities from the `XRZYPredictor`). |
| **PCP-base** | Residual-driven PCP that uses both `X` and `R` features to fit the conditional residual CDF grid, then factorizes templates to produce adaptive weights. |
| **EM-PCP** | PCP variant that first fits a joint Gaussian mixture over `(R, X, Y)` to get memberships `π_k(x,r,y)` and then reweights calibration residuals with the `MembershipPCPModel`. |
| **EM-R / EM-RX** | Two responsibility pipelines. EM-R feeds only `R` into the GMM, EM-RX stacks `(R, X)`; both are configured by `em_fit.use_X_in_em`. |
| **XRZYPredictor** | Ridge regressor for `μ(X, R)` used inside XRZY PCP and as the base mean for PCP-base / EM-PCP intervals. |
| **MembershipPCPModel** | Lightweight adapter that consumes membership matrices (e.g., from EM-PCP) and performs the multinomial precision sampling + weighted quantile selection. |

**Doc-style updates:**

- The XRZY predictor backing PCP now defaults to a `RandomForestRegressor`
	(`model.pcp_rf_*` knobs) to avoid oracle leakage.
- EM-soft features, EM diagnostics, and EM-PCP all draw memberships from
	`src/doc_em.py`, ensuring consistency with the simulator assumptions.

Run a subset of baselines by pairing CLI overrides with YAML edits. Example: EM-PCP only.

```bash
python main.py --config experiments/configs/gmm_em.yaml --results experiments/results/em_pcp_only.csv
```

With `pcp.xrzy.enabled=false`, `pcp.base.enabled=false`, `pcp.em.enabled=true` inside the YAML.

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
| `length_ratio_vs_grid.png` | Interval length divided by the oracle’s length; smaller is better. Helpful for spotting which conformal baseline is most efficient. |
| `length_ratio_vs_grid_pcp.png` | PCP-only comparison (EM-PCP vs. PCP-base) restricted to EM-R runs so you can see their gap without extra panels. |
| `len_ratio_diagnostics.png` | Scatter diagnostics showing how soft conformal length ratios correlate with EM quality metrics (mean max τ and Z-feature MSE). |
| `mean_max_tau_vs_grid.png` | Average sharpness of EM responsibilities. High values mean clusters are well separated. |
| `z_feature_mse_vs_grid.png` | MSE between the EM-soft features and the oracle latent means. Lower values indicate better feature recovery. |
| `em_iterations_hist.png` | Histogram of EM iteration counts, useful for spotting difficult configurations. |

Set `SKIP_PLOTS=1` during the run and execute the plotting command separately once results are ready.

---

## 4. Quick troubleshooting & tips

- Want shorter debug cycles? Shrink `global.n_train`, `n_cal`, `n_test`, and limit `dgp` lists to a single value.
- The EM step can be expensive for EM-PCP; lower `em_fit.max_iter` / `n_init` during experimentation.
- Large `precision_grid` entries increase PCP runtime. Start with `[20, 50, 100]` before scaling up.
- Reuse previous CSVs by pointing `--results` to the same path; the runner will deduplicate by key `(seed, K, delta, rho, sigma_y, b_scale, use_x_in_em)`.
