# 2. Statistical Inference

## Overview

Statistical inference uses sample data to make conclusions about populations. Two main frameworks: Frequentist and Bayesian.

---

## 2.1 Point Estimation

### Estimators

**Sample Mean:**
```
x_bar = (1/n)sumx_i
```

**Sample Variance:**
```
s^2 = (1/(n-1))sum(x_i - x_bar)^2
```

**Sample Standard Deviation:**
```
s = sqrts^2
```

### Properties of Estimators

**Bias:** E[theta_hat] - theta (estimator is unbiased if bias = 0)

**Consistency:** theta_hat --> theta as n --> inf

**Efficiency:** Lower variance among unbiased estimators

**Mean Squared Error (MSE):** E[(theta_hat - theta)^2] = Var(theta_hat) + Bias^2

```python
import numpy as np

# Generate sample
np.random.seed(42)
true_mean = 10
true_std = 2
sample = np.random.normal(true_mean, true_std, size=100)

# Point estimates
mean_est = np.mean(sample)
var_est = np.var(sample, ddof=1)  # ddof=1 for unbiased
std_est = np.std(sample, ddof=1)

print(f"True mean: {true_mean}, Estimate: {mean_est:.3f}")
print(f"True std: {true_std}, Estimate: {std_est:.3f}")

# Bias of estimator
# Sample mean is unbiased: E[x_bar] = mu
```

---

## 2.2 Maximum Likelihood Estimation (MLE)

Finds parameter values that maximize the likelihood function.

**Likelihood:** L(theta|data) = P(data|theta)

**Log-Likelihood:** l(theta) = log L(theta)

**MLE:** theta_hat_MLE = argmax L(theta)

### Example: MLE for Normal Distribution

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# Generate data from N(5, 2)
np.random.seed(42)
data = np.random.normal(loc=5, scale=2, size=100)

# Negative log-likelihood (we minimize)
def neg_log_likelihood(params):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(norm.logpdf(data, loc=mu, scale=sigma))

# MLE
result = minimize(neg_log_likelihood, x0=[0, 1], method='Nelder-Mead')
mu_mle, sigma_mle = result.x

print(f"MLE mu: {mu_mle:.3f}, sigma: {sigma_mle:.3f}")
print(f"Sample mean: {np.mean(data):.3f}, std: {np.std(data, ddof=1):.3f}")

# For normal distribution, MLE coincides with sample statistics
```

### MLE for Logistic Regression

```python
from scipy.optimize import minimize
import numpy as np

# Binary classification data
X = np.array([[1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([0, 0, 1, 1])

# Negative log-likelihood
def neg_log_likelihood_logistic(beta):
    z = X @ beta
    return np.sum(np.log(1 + np.exp(z)) - y * z)

# MLE
result = minimize(neg_log_likelihood_logistic, x0=[0, 0])
beta_mle = result.x

print(f"MLE coefficients: {beta_mle}")

# Predictions
def predict_proba(X, beta):
    return 1 / (1 + np.exp(-X @ beta))

probs = predict_proba(X, beta_mle)
print(f"Predicted probabilities: {probs}")
```

---

## 2.3 Method of Moments

Equates sample moments with population moments.

**k-th moment:** E[X^k]

**Sample k-th moment:** (1/n)sumx_i^k

### Example: Gamma Distribution

```python
import numpy as np
from scipy.stats import gamma

# Generate data from Gamma(alpha=2, beta=3)
np.random.seed(42)
data = gamma.rvs(a=2, scale=3, size=1000)

# Method of Moments
# E[X] = alphabeta, Var(X) = alphabeta^2
sample_mean = np.mean(data)
sample_var = np.var(data, ddof=1)

# Solve for alpha and beta
beta_mom = sample_var / sample_mean
alpha_mom = sample_mean / beta_mom

print(f"True: alpha=2, beta=3")
print(f"Method of Moments: alpha={alpha_mom:.3f}, beta={beta_mom:.3f}")
```

---

## 2.4 Interval Estimation

### Confidence Intervals

**95% CI for mean (known sigma):**
```
x_bar +/- 1.96(sigma/sqrtn)
```

**95% CI for mean (unknown sigma):**
```
x_bar +/- t_(n-1,0.025)(s/sqrtn)
```

**95% CI for proportion:**
```
p_hat +/- 1.96sqrt(p_hat(1-p_hat)/n)
```

### Interpretation

If we repeat sampling many times, 95% of constructed intervals will contain the true parameter.

**IMPORTANT:** The parameter is fixed; the interval is random.

```python
import numpy as np
from scipy import stats

# Sample data
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=50)

# Sample statistics
n = len(data)
mean = np.mean(data)
std = np.std(data, ddof=1)
se = std / np.sqrt(n)

# 95% CI using t-distribution
alpha = 0.05
t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
margin = t_crit * se

ci_lower = mean - margin
ci_upper = mean + margin

print(f"Sample mean: {mean:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
print(f"True mean (10) in CI: {ci_lower <= 10 <= ci_upper}")
```

### Bootstrap Confidence Intervals (2025 Best Practice)

**Non-parametric approach** when distribution unknown.

```python
import numpy as np

def bootstrap_ci(data, n_bootstrap=10000, ci=95):
    """Bootstrap confidence interval"""
    bootstrap_means = []
    n = len(data)

    for _ in range(n_bootstrap):
        # Resample with replacement
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(np.mean(resample))

    # Percentile method
    alpha = (100 - ci) / 2
    lower = np.percentile(bootstrap_means, alpha)
    upper = np.percentile(bootstrap_means, 100 - alpha)

    return lower, upper

# Example
np.random.seed(42)
data = np.random.exponential(scale=5, size=50)  # Non-normal

# Bootstrap CI
lower, upper = bootstrap_ci(data, n_bootstrap=10000, ci=95)
print(f"Bootstrap 95% CI: [{lower:.3f}, {upper:.3f}]")

# Compare with normal approximation (less accurate for non-normal)
mean = np.mean(data)
se = np.std(data, ddof=1) / np.sqrt(len(data))
normal_ci = (mean - 1.96*se, mean + 1.96*se)
print(f"Normal approx CI: [{normal_ci[0]:.3f}, {normal_ci[1]:.3f}]")
```

### Advanced Bootstrap: BCa Intervals (2025)

**Bias-Corrected and Accelerated** - adjusts for bias and skewness.

```python
from scipy import stats
import numpy as np

def bca_ci(data, statistic=np.mean, n_bootstrap=10000, ci=95):
    """BCa confidence interval"""
    n = len(data)

    # Bootstrap distribution
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))
    bootstrap_stats = np.array(bootstrap_stats)

    # Original statistic
    theta_hat = statistic(data)

    # Bias correction
    z0 = stats.norm.ppf(np.mean(bootstrap_stats < theta_hat))

    # Acceleration (jackknife)
    jackknife_stats = []
    for i in range(n):
        jk_sample = np.delete(data, i)
        jackknife_stats.append(statistic(jk_sample))
    jackknife_stats = np.array(jackknife_stats)

    jk_mean = np.mean(jackknife_stats)
    numerator = np.sum((jk_mean - jackknife_stats)**3)
    denominator = 6 * (np.sum((jk_mean - jackknife_stats)**2))**1.5
    acceleration = numerator / denominator if denominator != 0 else 0

    # Adjusted percentiles
    alpha = (100 - ci) / 100
    z_alpha = stats.norm.ppf([alpha/2, 1-alpha/2])

    adjusted_percentiles = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
    adjusted_percentiles *= 100

    lower = np.percentile(bootstrap_stats, adjusted_percentiles[0])
    upper = np.percentile(bootstrap_stats, adjusted_percentiles[1])

    return lower, upper

# Example
np.random.seed(42)
data = np.random.lognormal(mean=0, sigma=1, size=100)

lower, upper = bca_ci(data, n_bootstrap=5000)
print(f"BCa 95% CI: [{lower:.3f}, {upper:.3f}]")
```

---

## 2.5 Bootstrapping and Resampling (2025 Methods)

### Block Bootstrap for Time Series

Preserves autocorrelation structure.

```python
import numpy as np

def block_bootstrap(data, block_size, n_bootstrap=1000):
    """Block bootstrap for time series"""
    n = len(data)
    n_blocks = int(np.ceil(n / block_size))

    bootstrap_means = []
    for _ in range(n_bootstrap):
        # Sample blocks
        blocks = []
        for _ in range(n_blocks):
            start = np.random.randint(0, n - block_size + 1)
            blocks.append(data[start:start+block_size])

        # Concatenate and trim
        resampled = np.concatenate(blocks)[:n]
        bootstrap_means.append(np.mean(resampled))

    return bootstrap_means

# Time series with autocorrelation
np.random.seed(42)
ar_data = np.zeros(200)
ar_data[0] = np.random.normal()
for i in range(1, 200):
    ar_data[i] = 0.7 * ar_data[i-1] + np.random.normal()

# Block bootstrap with block size 10
bootstrap_dist = block_bootstrap(ar_data, block_size=10, n_bootstrap=1000)
ci = np.percentile(bootstrap_dist, [2.5, 97.5])
print(f"Block Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Cluster Bootstrap (2025)

For clustered data (e.g., students in schools).

```python
import numpy as np
import pandas as pd

def cluster_bootstrap(data, cluster_col, value_col, n_bootstrap=1000):
    """Bootstrap at cluster level"""
    clusters = data[cluster_col].unique()
    bootstrap_means = []

    for _ in range(n_bootstrap):
        # Resample clusters (not individuals)
        sampled_clusters = np.random.choice(clusters, size=len(clusters), replace=True)

        # Get all data from sampled clusters
        resampled = []
        for cluster in sampled_clusters:
            resampled.extend(data[data[cluster_col] == cluster][value_col].values)

        bootstrap_means.append(np.mean(resampled))

    return bootstrap_means

# Example: Students in schools
df = pd.DataFrame({
    'school': np.repeat([1, 2, 3, 4, 5], 20),
    'score': np.random.normal(loc=np.repeat([70, 75, 80, 85, 90], 20), scale=10)
})

bootstrap_dist = cluster_bootstrap(df, 'school', 'score', n_bootstrap=1000)
ci = np.percentile(bootstrap_dist, [2.5, 97.5])
print(f"Cluster Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
```

---

## 2.6 Permutation Tests (2025)

Test hypotheses by randomly permuting data.

```python
import numpy as np

def permutation_test(group1, group2, n_permutations=10000):
    """Two-sample permutation test"""
    # Observed difference in means
    obs_diff = np.mean(group1) - np.mean(group2)

    # Combine data
    combined = np.concatenate([group1, group2])
    n1 = len(group1)

    # Permutation distribution
    perm_diffs = []
    for _ in range(n_permutations):
        # Shuffle
        shuffled = np.random.permutation(combined)
        perm_group1 = shuffled[:n1]
        perm_group2 = shuffled[n1:]

        perm_diff = np.mean(perm_group1) - np.mean(perm_group2)
        perm_diffs.append(perm_diff)

    # P-value (two-tailed)
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(obs_diff))

    return p_value, obs_diff, perm_diffs

# Example
np.random.seed(42)
treatment = np.random.normal(loc=10.5, scale=2, size=50)
control = np.random.normal(loc=10, scale=2, size=50)

p_value, obs_diff, perm_diffs = permutation_test(treatment, control)

print(f"Observed difference: {obs_diff:.3f}")
print(f"P-value: {p_value:.4f}")
print(f"Significant at alpha=0.05: {p_value < 0.05}")
```

---

## Resources

- **Classic:**
  - "All of Statistics" by Wasserman
  - "Statistical Inference" by Casella & Berger

- **Modern/Bootstrap:**
  - "An Introduction to the Bootstrap" by Efron & Tibshirani
  - "Computer Age Statistical Inference" by Efron & Hastie (2016)

- **2025 Applications:**
  - NumberAnalytics: Advanced Bootstrap Techniques
  - DataCamp: Modern Resampling Methods
