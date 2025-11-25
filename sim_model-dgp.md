## Data Generating Process Specification

Consider a data generating process with the following structure:
$$
Y = \eta_0 + \bm{\eta}^\top \bm{X} + \alpha_Z + \varepsilon_Z
$$
where $\alpha_k$ and $\sigma_k$ are distinct for each cluster, and there is no $R \to Y$ leakage.

### Fixed Settings

- **Number of clusters:** $K = 4$
- **Feature dimension:** $p = 3$
- **Cluster shifts:** $\bm{\alpha} = (\alpha_1, \alpha_2, \alpha_3, \alpha_4) = (1, 2, 3, 4)$
- **Cluster standard deviations:** $\bm{\sigma} = (\sigma_1, \sigma_2, \sigma_3, \sigma_4) = (1, 2, 4, 8)$  
  (these are standard deviations; variances are $\sigma_k^2$)
- **Cluster prior:** $\bm{\pi} = \left(\frac{1}{4}, \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\right)$
- **$R$ model:**  
  $R \mid Z = k \sim \mathcal{N}(\mu_{R, k}, 1)$, where $\bm{\mu}_R = (-3, -1, +1, +3)$
- **$X$ model:**  
  $\bm{X} \sim \mathcal{N}(0, I_p)$, where $I_p$ is the $p \times p$ identity
- **$X \to Y$ part:**  
  $m(\bm{X}) = \eta_0 + \bm{\eta}^\top \bm{X}$, with $\eta_0 = 0.5$, $\bm{\eta} = (1.0, -0.5, 0.8)$
- **Outcome model:**  
  $Y \mid \bm{X}, Z = k \sim \mathcal{N}(m(\bm{X}) + \alpha_k, \sigma_k^2)$

---

### Data Generating Process (for $i = 1, \ldots, n$)

1. **Sample cluster:**
   $$
   Z_i \sim \text{Categorical}(\pi_1, \ldots, \pi_4)
   $$
2. **Sample features:**
   $$
   \bm{X}_i \sim \mathcal{N}(0, I_3)
   $$
   That is, $\bm{X}_i = (X_{i1}, X_{i2}, X_{i3})$ with each $X_{ij} \sim \mathcal{N}(0, 1)$ independently.
3. **Sample auxiliary $R$ (cluster-only; no effect on $Y$ given $Z$):**
   $$
   R_i \sim \mathcal{N}(\mu_{R, Z_i}, 1)
   $$
4. **Compute $X$-part of $Y$:**
   $$
   \begin{align*}
       m_i &= \eta_0 + \bm{\eta}^\top \bm{X}_i \\
           &= 0.5 + 1.0 \cdot X_{i1} - 0.5 \cdot X_{i2} + 0.8 \cdot X_{i3}
   \end{align*}
   $$
5. **Sample outcome:**
   $$
   Y_i \sim \mathcal{N}(m_i + \alpha_{Z_i}, \sigma_{Z_i}^2)
   $$

Return dataset $\mathcal{D} = \{ (\bm{X}_i, R_i, Y_i, Z_i) \}_{i=1}^n$.

---

### Notes

- Distinct mean shifts: $\bm{\alpha} = (1, 2, 3, 4)$
- Distinct residual scales: $\bm{\sigma} = (1, 2, 4, 8)$
- No leakage: $R$ is sampled from $Z$ only and is not used in $Y \mid \bm{X}, Z$
- To change distinctness, adjust $\bm{\alpha}$ and $\bm{\sigma}$ sequences (e.g., $\alpha_k = k$, $\sigma_k = 2^{k-1}$)

---

### Reference Implementation (Python/Numpy)

```python
import numpy as np

def simulate(n, seed=123):
    rng = np.random.default_rng(seed)

    # Fixed parameters
    K = 4
    p = 3
    pi = np.array([0.25, 0.25, 0.25, 0.25])
    alpha = np.array([1., 2., 3., 4.])           # alpha_k
    sigma = np.array([1., 2., 4., 8.])           # sigma_k (std dev)
    muR = np.array([-3., -1., 1., 3.])           # (mu)R_k
    eta0 = 0.5
    eta = np.array([1.0, -0.5, 0.8])             # length p=3

    # Sample Z
    Z = rng.choice(np.arange(K), size=n, p=pi)   # values in {0,1,2,3}

    # Sample X ~ N(0, I_p)
    X = rng.normal(0., 1., size=(n, p))

    # Sample R | Z
    R = rng.normal(muR[Z], 1.0)

    # Compute m(X)
    m = eta0 + X @ eta

    # Sample Y | X, Z
    Y = rng.normal(m + alpha[Z], sigma[Z])

    # Convert Z to 1..4 if you prefer 1-based indexing
    Z_1based = Z + 1
    return X, R, Y, Z_1based
```

*Structure visualization only, not production code.* If we want a different $K$ or spacing, we should be able to replace:

- $\alpha_k$ by a new sequence (e.g., $\alpha_k = k$ for $k=1,\ldots,K$)
- $\sigma_k$ by a new sequence (e.g., $\sigma_k = 2^{k-1}$)
- $\mu_{R, k}$ by well-separated means (e.g., linearly spaced over an interval)

