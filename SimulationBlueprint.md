\section*{Simulation Blueprint: EM-based Z Imputation and Conformal Prediction for Y}

\subsection*{1) Purpose and Framework}

\paragraph{Purpose}
\begin{itemize}
    \item Quantify how EM-based $Z$ imputation accuracy (driven by mixture separation and model misspecification) affects:
    \begin{itemize}
        \item Conformal coverage for $Y$
        \item Efficiency (average interval length), relative to oracle-$Z$ and ignore-$Z$ baselines
    \end{itemize}
    \item Compare hard plug-in vs soft marginalization over $Z$.
\end{itemize}

\paragraph{Framework}
\begin{itemize}
    \item \textbf{DGP:} $Z \sim \text{Categorical}(\pi)$. Given $Z$, $(R, X)$ are Gaussian; $Y$ depends on $X$ and $Z$ plus noise.
    \item $Z$ is unobserved in all phases; estimate $Z$ (posteriors $\tau$) via EM on $R$ or on $[R; X]$.
    \item \textbf{Pipelines:}
    \begin{itemize}
        \item Oracle-$Z$ (benchmark)
        \item EM-hard: $\hat{z} = \arg\max_k \tau_k$
        \item EM-soft: use $\tau$ as $\hat{p}(Z \mid \cdot)$
        \item Ignore-$Z$: do not use $Z$
    \end{itemize}
    \item \textbf{Conformalization:} split conformal with absolute residual scores, calibrated using exactly the same pipeline as test time.
    \item \textbf{Knobs:}
    \begin{itemize}
        \item Mixture separation (controls EM accuracy)
        \item Number of clusters $K$
        \item Class-specific $Y$ effect magnitudes $\{\beta_k\}$
        \item Whether EM uses $R$ or $[R; X]$
        \item Covariance structure (shared vs class-specific, diagonal vs full)
        \item Sample sizes
    \end{itemize}
    \item \textbf{Metrics:}
    \begin{itemize}
        \item Coverage, average length
        \item Imputation accuracy $A = \mathbb{P}(\hat{z} = Z)$, cross-entropy / entropy of $\tau$
        \item Efficiency gaps vs oracle
    \end{itemize}
\end{itemize}

\subsection*{2) Specific Design in Theoretical Notation}

\paragraph{Latent clusters}
\begin{itemize}
    \item $K \geq 2$, $\pi = (\pi_1, \ldots, \pi_K)$, $\Sigma_k$ positive definite.
    \item For each $i$:
    \[
    Z_i \sim \text{Categorical}(\pi_1, \ldots, \pi_K)
    \]
\end{itemize}

\paragraph{Observables (Gaussian mixture)}

Stack 
\[
S_i := \begin{bmatrix} R_i \\ X_i \end{bmatrix} \in \mathbb{R}^{d_R + d_X}
\]

Conditional on $Z_i = k$:
\[
S_i \mid Z_i = k \sim \mathcal{N}(m_k, \Sigma_k)
\]
Partition
\[
m_k = \begin{bmatrix} \mu_{Rk} \\ \mu_{Xk} \end{bmatrix}, \quad
\Sigma_k = \begin{bmatrix} \Sigma_{RR,k} & \Sigma_{RX,k} \\ \Sigma_{XR,k} & \Sigma_{XX,k} \end{bmatrix}
\]

Control cluster separability primarily via $\|\mu_{Rk} - \mu_{R\ell}\|$ and optionally via $\mu_{Xk}$; control $X$–$R$ correlation via $\Sigma_{RX,k}$.

Two EM variants:
\begin{itemize}
    \item EM-$R$: $S_i := R_i$ only (fit GMM on $R$)
    \item EM-$RX$: $S_i := [R_i; X_i]$ (fit GMM on both)
\end{itemize}

\paragraph{Outcome model}

\[
Y_i \mid X_i, Z_i = k: \quad Y_i = \theta^\top X_i + \beta_k + \varepsilon_i
\]
with 
\[
\varepsilon_i \sim \mathcal{N}(0, \sigma^2), \text{ independent of } (X_i, R_i, Z_i)
\]
Parameters: $\theta \in \mathbb{R}^{d_X}$, class intercepts $\beta_1, \ldots, \beta_K$.

\paragraph{Data splits and sizes}

Generate $n_{\text{train}}, n_{\text{cal}}, n_{\text{test}}$ i.i.d.\ samples.

Typical: $n_{\text{train}}=2000$, $n_{\text{cal}}=2000$, $n_{\text{test}}=10000$.

\paragraph{EM estimation of $Z$ (on training set; then applied to cal/test)}

\begin{itemize}
    \item Fit on $S=R$ (EM-$R$) or $S=[R; X]$ (EM-$RX$).
    \item Initialize $\{\pi_k, m_k, \Sigma_k\}$ (e.g., $k$-means on $S$).
    \item Iterate until convergence:
    \[
    \text{E-step:} \quad \tau_{ik} \propto \pi_k \mathcal{N}(S_i \mid m_k, \Sigma_k), \quad \sum_k \tau_{ik} = 1
    \]
    \[
    \text{M-step:} \quad N_k = \sum_i \tau_{ik}
    \]
    \[
    \pi_k = \frac{N_k}{n_{\text{train}}}
    \]
    \[
    m_k = \frac{1}{N_k} \sum_i \tau_{ik} S_i
    \]
    \[
    \Sigma_k = \frac{1}{N_k} \sum_i \tau_{ik} (S_i - m_k)(S_i - m_k)^\top + \lambda I \quad (\text{optional ridge } \lambda > 0)
    \]
    \item For cal/test points $j$, compute $\hat{\tau}_{jk}$ with fitted $\{\hat{\pi}_k, \hat{m}_k, \hat{\Sigma}_k\}$.
    \item Hard labels: $\hat{z}_j = \arg\max_k \hat{\tau}_{jk}$.
\end{itemize}

\paragraph{Y-models and prediction functions}

\begin{itemize}
    \item \textbf{Oracle-$Z$:}
    \[
    \text{Fit OLS: } Y \sim X + \text{cluster dummies}(Z) \implies \hat{\mu}_{\text{orc}}(x, k) = \hat{\theta}^\top x + \hat{\beta}_k
    \]
    \item \textbf{EM-hard:}
    Use $\hat{z}$ on training data; fit OLS: $Y \sim X + \text{cluster dummies}(\hat{z})$
    \[
    \hat{\mu}_{\text{hard}}(x, s) = \hat{\theta}^\top x + \hat{\beta}_{\hat{z}(s)}
    \]
    \item \textbf{EM-soft (intercept-shift form):}
    Fit $\theta$ and $\{\beta_k\}$ by minimizing weighted squared loss
    \[
    L(\theta, \beta) = \sum_i \sum_k \tau_{ik} (Y_i - \theta^\top X_i - \beta_k)^2
    \]
    Closed forms:
    \[
    \hat{\theta} = \left(\sum_i w_i X_i X_i^\top \right)^{-1} \sum_i w_i X_i \left(Y_i - \sum_k \tau_{ik} \hat{\beta}_k \right), \quad w_i = \sum_k \tau_{ik}
    \]
    \[
    \hat{\beta}_k = \frac{1}{\sum_i \tau_{ik}} \sum_i \tau_{ik} (Y_i - \hat{\theta}^\top X_i)
    \]
    Iterate $\hat{\theta} \leftrightarrow \hat{\beta}_k$ a few times or solve jointly.
    
    Predict:
    \[
    \hat{\mu}_{\text{soft}}(x, s) = \hat{\theta}^\top x + \sum_k \hat{\tau}_k(s) \hat{\beta}_k
    \]
    \item \textbf{Ignore-$Z$:}
    Fit OLS: $Y \sim X$,
    \[
    \hat{\mu}_{\text{ign}}(x) = \hat{\theta}^\top x
    \]
\end{itemize}

\textbf{Note:} For EM-soft with class-specific slopes, replace $\beta_k$ by $\theta_k$ and use per-class weighted regressions:
\[
\hat{\theta}_k = \arg\min_{\theta_k} \sum_i \tau_{ik} (Y_i - \theta_k^\top X_i)^2, \quad \hat{\mu}_{\text{soft}}(x,s) = \sum_k \hat{\tau}_k(s) \hat{\theta}_k^\top x.
\]

\paragraph{Conformal prediction (split conformal, absolute residuals)}

On calibration set, compute same $\hat{\tau}$ (from fitted EM) and corresponding predictions $\hat{\mu}_{\text{variant}}$, then scores:
\[
S_i = |Y_i - \hat{\mu}_{\text{variant}}(X_i, S_i)|
\]

Let $\hat{q}_{1-\alpha}$ be the $(1-\alpha)(1 + \frac{1}{n_{\text{cal}}})$-empirical quantile of $\{S_i\}$.

For a test point $(x,s)$:
\[
C_{\alpha}(x,s) = \left[\hat{\mu}_{\text{variant}}(x,s) - \hat{q}_{1-\alpha}, \quad \hat{\mu}_{\text{variant}}(x,s) + \hat{q}_{1-\alpha}\right].
\]

\paragraph{Parameters to vary (controls imputation difficulty and $Y$ signal)}

\begin{itemize}
    \item Separation $\delta$ in $R$: set $\mu_{Rk}$ on a simplex with pairwise distance $\delta$. Larger $\delta \implies$ higher EM accuracy.
    \item Optional separation in $X$: $\|\mu_{Xk} - \mu_{X\ell}\| \in \{0, \text{small}\}$.
    \item Covariance:
    \begin{itemize}
        \item Shared spherical: $\Sigma_k = \sigma_S^2 I$
        \item Shared diagonal or class-specific full covariance (misspecification if EM assumes spherical)
    \end{itemize}
    \item Cross-covariance $\Sigma_{RX,k}$ to control $\mathrm{Corr}(X,R \mid Z)$
    \item $K \in \{2, 3, 5\}$
    \item $\beta$ spread: e.g., $K=3$ with $\beta = (-\beta, 0, +\beta)$, $\beta \in \{0,0.5,1,2\}$
    \item Noise $\sigma \in \{0.5, 1.0\}$
    \item EM fitting choice: EM-$R$ vs EM-$RX$
    \item Sample sizes $n_{\text{train}}, n_{\text{cal}}$
    \item Regularization $\lambda$ for $\Sigma_k$
\end{itemize}

\paragraph{Metrics}

\begin{itemize}
    \item Imputation quality:
    \[
    \text{Hard accuracy } A = \mathbb{P}(\hat{z} = Z) \text{ on test}
    \]
    \[
    \text{Soft quality: mean max responsibility } \bar{A}_{\text{conf}} = \mathbb{E}\left[\max_k \hat{\tau}_k\right], \quad \text{and cross-entropy } -\mathbb{E}[\log \hat{\tau}_Z]
    \]
    \item Conformal:
    \[
    \text{Coverage: } \mathbb{P}[Y \in C_{\alpha}(X,S)] \text{ for } \alpha \in \{0.1, 0.2\}
    \]
    \[
    \text{Average length: } \mathbb{E}[\text{length}(C_{\alpha})]
    \]
    \[
    \text{Conditional efficiency: length vs max}_k \hat{\tau}_k \text{ bins; length given correct vs incorrect hard labels}
    \]
    \item Efficiency gaps vs oracle:
    \[
    \Delta \mathrm{LEN}_{\text{variant}} = \mathrm{LEN}_{\text{variant}} - \mathrm{LEN}_{\text{oracle}}
    \]
\end{itemize}