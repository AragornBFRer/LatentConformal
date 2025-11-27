## 1) Model

- **Clusters:** $Z \in \{1,2,3,4\}$ ($K=4$)
- **X-part (linear):** $m(X; \eta) = \eta_0 + \eta^T X$
- **Fixed cluster shifts:** $\alpha_1=1,\, \alpha_2=2,\, \alpha_3=3,\, \alpha_4=4$ ($\alpha_k = k$)
- **Fixed cluster scales:** $\sigma_1=1,\, \sigma_2=2,\, \sigma_3=4,\, \sigma_4=8$ ($\sigma_k = 2^{k-1}$)
- **Prior over clusters:** $\pi_k = 1/4$
- **Outcome:**
  - $Y \mid X, Z=k \sim \mathcal{N}\big(\text{mean} = m(X;\eta) + \alpha_k,\, \text{variance} = \sigma_k^2 \big)$
- **R-only cluster model (simple Gaussian, learn only the mean):**
  - $R \mid Z=k \sim \mathcal{N}(\mu_{R_k},\, 1)$
- **No leakage:** $R$ enters only via $Z$; $R$ is never used in $Y\,|\,X, Z$

**Unknown parameters to estimate:** $\eta_0,\, \eta,\, \mu_{R_1},\,\ldots,\,\mu_{R_4}$

---

## 2) Training

**Data:** $\{(x_i, r_i, y_i)\}$ for $i=1,\ldots,n$.  
We use the **EM Algorithm** for modeling.

### Initialization

- $\eta_0 \leftarrow 0,\, \eta \leftarrow 0$
- $\mu_{R_k} \leftarrow$ 1D $k$-means centers of $\{r_i\}$ with $K=4$ (or use evenly spaced quantiles)

---

### Repeat until convergence (e.g., change in log-likelihood $<$ tol):

#### **E-step**

For each $i, k$, compute:

- **y-mean:**

  $$
  \mu_{Y_{ik}} = \eta_0 + \eta^T x_i + \alpha_k
  $$

- **Log-likelihoods:**

  $$
  \log p_{Y_{ik}} = -0.5\left[ \log(2\pi \sigma_k^2) + \frac{(y_i - \mu_{Y_{ik}})^2}{\sigma_k^2} \right]
  $$
  $$
  \log p_{R_{ik}} = -0.5\left[ \log(2\pi \cdot 1) + (r_i - \mu_{R_k})^2 \right]
  $$

- **Total log-responsibility:**

  $$
  \ell_{ik} = \log \pi_k + \log p_{Y_{ik}} + \log p_{R_{ik}} \quad (\pi_k = \log(1/4) \text{ is constant})
  $$

- **Responsibilities:**

  $$
  \gamma_{ik} = \frac{\exp(\ell_{ik})}{\sum_{\ell=1}^4 \exp(\ell_{i\ell})}
  $$

---

#### **M-step**

- **Update $\eta_0, \eta$** (weighted least squares; one pass, closed form):

  For each $i$:

  $$
  w_i = \sum_k \frac{\gamma_{ik}}{\sigma_k^2}
  $$
  $$
  y^*_i = \frac{\sum_k \gamma_{ik} (y_i - \alpha_k)/\sigma_k^2}{w_i}
  $$

  Let $\tilde{X}$ be the $n \times (p+1)$ design with rows $[1,\, x_i^T]$, $\tilde{y}$ the vector with entries $y^*_i$, and $W = \mathrm{diag}(w_i)$.

  Solve

  $$
  \begin{bmatrix}
    \eta_0 \\ \eta
  \end{bmatrix}
  = (\tilde{X}^T W \tilde{X})^{-1} \tilde{X}^T W \tilde{y}
  $$

- **Update $\mu_{R_k}$** (weighted means):

  $$
  n_k = \sum_i \gamma_{ik}
  $$
  $$
  \mu_{R_k} = \frac{\sum_i \gamma_{ik} r_i}{n_k}
  $$

---

**Monitoring (optional):**

$$
\mathcal{L} = \sum_i \log \left[ \sum_k \pi_k \cdot \mathcal{N}(y_i \mid \eta_0 + \eta^T x_i + \alpha_k,\, \sigma_k^2) \cdot \mathcal{N}(r_i \mid \mu_{R_k},\, 1) \right]
$$

---

## 3) Prediction

Test-time; $X$ and $R$ observed

- For given $(x, r)$:

  - **Compute cluster weights using $R$ only:**
    $$
    w_k \propto \pi_k \cdot \mathcal{N}(r \mid \mu_{R_k},\, 1)\ ; \quad \sum_k w_k = 1
    $$

  - **Predictive mean:**
    $$
    \mathbb{E}[Y \mid x, r] = (\eta_0 + \eta^T x) + \sum_k w_k \alpha_k
    $$

  - **Predictive variance:**
    $$
    \bar{\alpha} = \sum_k w_k \alpha_k
    $$
    $$
    \mathrm{Var}[Y \mid x, r] = \sum_k w_k \left[ \sigma_k^2 + (\alpha_k - \bar{\alpha})^2 \right]
    $$
