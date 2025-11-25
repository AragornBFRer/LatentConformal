Default configs live in `experiments/configs/gmm_em.yaml`.

---

## XRZY simulation recap

The simulator now mirrors the specification in `sim_model-dgp.md`:

- **Auxiliary feature:** `R | Z = k ∼ 𝒩(μ_{R,k}, 1)` with `μ_R = (-3, -1, 1, 3)`.
	`α = (1, 2, 3, 4)`, and `ε_Z ∼ 𝒩(0, σ_Z²)` with `σ = (1, 2, 4, 8)`.
	**Outcome:** `Y = η₀ + ηᵀ X + α_Z + ε_Z`, where `η₀ = 0.5`, `η = (1, -0.5, 0.8)`,
	`α = (1, 2, 3, 4)`, and `ε_Z ∼ 𝒩(0, σ_Z²)` with `σ = (1, 2, 4, 8)`.
	The actual shift used in any run is `α̃_k = δ · α_k` where `δ ∈ dgp.delta_list`
	(default: five values `[0.5, 0.75, 1.0, 1.5, 2.0]`). This “mixture separation”
	knob lets you sweep how distinct the clusters are without redefining `α`.
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

## 1. Run the remaining baselines

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
- `model`: ridge penalty for the linear regressors (shared by ignore and joint predictors).

Included predictors / objects in this repo:

| Name | What it means / uses |
| --- | --- |
| **Ignore-Z** | Models `μ_ignore(x) = θ₀ + θᵀ x` only, so its split-conformal band is `C_ignore(x) = { y : |y - μ_ignore(x)| ≤ q_ignore }`; no latent information enters. |
| **PCP-base** | Fits a joint linear predictor `μ_joint(x,r) = θ₀ + θᵀ [x; r]`, uses residuals `|y - μ_joint(x,r)|`, and conditions posterior conformal quantiles on the same `[x; r]` block, yielding `C_base(x,r) = { y : |y - μ_joint(x,r)| ≤ q_base(x,r) }`. |
| **EM-PCP** | Uses doc-EM memberships `π(x,r,y)` to weight the joint residuals via `MembershipPCPModel`, giving `C_em(x,r) = { y : |y - μ_joint(x,r)| ≤ q_em(π(x,r,y)) }`. |
| **EM-R / EM-RX** | Responsibility pipelines feeding EM-PCP: EM-R fits the GMM on `R` alone (features `τ_R(r)`), while EM-RX fits on `[R; X]` (`τ_RX(x,r)`), toggled by `em_fit.use_X_in_em`. |
| **MembershipPCPModel** | Consumes membership weights `π` and outputs quantiles `q(π)` by multinomial-precision weighting, enabling set construction `C(π) = { y : |y - μ_joint| ≤ q(π) }`. |

**Doc-style updates:**

- Joint residuals for PCP now come from a single linear model on `[X; R]`, avoiding dependencies on oracle-style regressors.
- EM diagnostics and EM-PCP continue to draw memberships from
	`src/doc_em.py`, ensuring consistency with the simulator assumptions.

Run a subset of baselines by pairing CLI overrides with YAML edits. Example: EM-PCP only.

```bash
python main.py --config experiments/configs/gmm_em.yaml --results experiments/results/em_pcp_only.csv
```

With `pcp.base.enabled=false`, `pcp.em.enabled=true` inside the YAML.

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
| `length_vs_grid_pcp.png` | PCP-only comparison (EM-PCP vs. PCP-base) restricted to EM-R runs so you can see their gap without extra panels. |
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
