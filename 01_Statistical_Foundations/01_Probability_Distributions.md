# Probability Distributions

## Table of Contents

- [1. Introduction to Probability Distributions](#1-introduction-to-probability-distributions)
- [2. Discrete Probability Distributions](#2-discrete-probability-distributions)
- [3. Continuous Probability Distributions](#3-continuous-probability-distributions)
- [4. Distribution Fitting and Goodness-of-Fit Tests](#4-distribution-fitting-and-goodness-of-fit-tests)
- [5. Central Limit Theorem](#5-central-limit-theorem)
- [6. Multivariate Distributions](#6-multivariate-distributions)
- [7. Mixture Models](#7-mixture-models)
- [8. Distribution Selection Guide](#8-distribution-selection-guide)
- [9. Bayesian Conjugate Priors](#9-bayesian-conjugate-priors)
- [10. Resources and References](#10-resources-and-references)

---

## 1. Introduction to Probability Distributions

### 1.1 Random Variables

A **random variable** is a function that maps outcomes from a sample space to real numbers.

**Discrete Random Variable**: Takes countable values (integers, finite set)
- Examples: coin flips, dice rolls, number of customers, defect counts

**Continuous Random Variable**: Takes values from continuous intervals
- Examples: height, temperature, time, stock prices

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Discrete random variable example: dice roll
discrete_rv = np.random.randint(1, 7, size=1000)
print(f"Discrete RV sample: {discrete_rv[:10]}")

# Continuous random variable example: normal distribution
continuous_rv = np.random.normal(loc=100, scale=15, size=1000)
print(f"Continuous RV sample: {continuous_rv[:10]}")
```

### 1.2 Probability Mass Function (PMF)

For **discrete** random variables, the PMF gives the probability of each specific value:

**Definition**: P(X = x) for all possible values x

**Properties**:
- 0 <= P(X = x) <= 1 for all x
- Sum of all probabilities = 1

```python
from scipy.stats import binom

# Binomial PMF: 10 coin flips with p=0.5
n, p = 10, 0.5
dist = binom(n, p)

# Calculate PMF for all possible values
x_values = np.arange(0, n+1)
pmf_values = dist.pmf(x_values)

# Verify sums to 1
print(f"Sum of PMF: {pmf_values.sum():.6f}")

# Visualize PMF
plt.figure(figsize=(10, 4))
plt.stem(x_values, pmf_values, basefmt=' ')
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title('Binomial PMF (n=10, p=0.5)')
plt.grid(alpha=0.3)
plt.show()
```

### 1.3 Probability Density Function (PDF)

For **continuous** random variables, the PDF describes the relative likelihood of values:

**Definition**: f(x) such that P(a <= X <= b) = integral from a to b of f(x)dx

**Properties**:
- f(x) >= 0 for all x
- Integral over entire domain = 1
- P(X = exact value) = 0 for continuous distributions

```python
from scipy.stats import norm

# Standard normal PDF
dist = norm(loc=0, scale=1)

# Generate x values
x = np.linspace(-4, 4, 1000)
pdf_values = dist.pdf(x)

# PDF value at specific point
print(f"PDF at x=0: {dist.pdf(0):.4f}")
print(f"PDF at x=1: {dist.pdf(1):.4f}")

# Probability of interval using integration
prob_interval = dist.cdf(1) - dist.cdf(-1)
print(f"P(-1 <= X <= 1) = {prob_interval:.4f}")

# Visualize PDF
plt.figure(figsize=(10, 4))
plt.plot(x, pdf_values, linewidth=2)
plt.fill_between(x, pdf_values, where=(x >= -1) & (x <= 1), alpha=0.3)
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Standard Normal PDF')
plt.grid(alpha=0.3)
plt.show()
```

### 1.4 Cumulative Distribution Function (CDF)

The CDF gives the probability that X is less than or equal to a value:

**Definition**: F(x) = P(X <= x)

**Properties**:
- Non-decreasing function
- lim as x --> -infinity: F(x) = 0
- lim as x --> infinity: F(x) = 1
- P(a < X <= b) = F(b) - F(a)

```python
from scipy.stats import expon

# Exponential CDF with rate parameter lambda=0.5
rate = 0.5
dist = expon(scale=1/rate)

x = np.linspace(0, 10, 1000)
cdf_values = dist.cdf(x)

# Calculate probabilities using CDF
print(f"P(X <= 2) = {dist.cdf(2):.4f}")
print(f"P(X > 2) = {1 - dist.cdf(2):.4f}")
print(f"P(1 < X <= 3) = {dist.cdf(3) - dist.cdf(1):.4f}")

# Visualize CDF
plt.figure(figsize=(10, 4))
plt.plot(x, cdf_values, linewidth=2)
plt.xlabel('x')
plt.ylabel('F(x)')
plt.title('Exponential CDF (lambda=0.5)')
plt.grid(alpha=0.3)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Median')
plt.legend()
plt.show()
```

### 1.5 Expected Value (Mean)

The **expected value** is the long-run average value of a random variable:

**Discrete**: E[X] = sum of x * P(X = x) for all x

**Continuous**: E[X] = integral of x * f(x) dx

**Properties**:
- E[aX + b] = a * E[X] + b (linearity)
- E[X + Y] = E[X] + E[Y] (even if dependent)
- E[X * Y] = E[X] * E[Y] (only if independent)

```python
from scipy.stats import poisson

# Poisson distribution with lambda=3.5
lam = 3.5
dist = poisson(lam)

# Theoretical mean
theoretical_mean = dist.mean()
print(f"Theoretical E[X] = {theoretical_mean:.2f}")

# Empirical mean from samples
samples = dist.rvs(size=100000)
empirical_mean = samples.mean()
print(f"Empirical E[X] = {empirical_mean:.2f}")

# Expected value of transformation
# E[2X + 5]
theoretical_transform = 2 * theoretical_mean + 5
empirical_transform = (2 * samples + 5).mean()
print(f"E[2X + 5] theoretical: {theoretical_transform:.2f}")
print(f"E[2X + 5] empirical: {empirical_transform:.2f}")
```

### 1.6 Variance and Standard Deviation

**Variance** measures the spread of a distribution:

**Definition**: Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2

**Standard Deviation**: sigma = sqrt(Var(X))

**Properties**:
- Var(aX + b) = a^2 * Var(X)
- Var(X + Y) = Var(X) + Var(Y) + 2*Cov(X,Y)
- Var(X + Y) = Var(X) + Var(Y) if X and Y are independent

```python
from scipy.stats import gamma

# Gamma distribution
shape, scale = 2, 2
dist = gamma(a=shape, scale=scale)

# Theoretical variance
theoretical_var = dist.var()
theoretical_std = dist.std()
print(f"Theoretical Var(X) = {theoretical_var:.4f}")
print(f"Theoretical SD(X) = {theoretical_std:.4f}")

# Empirical variance from samples
samples = dist.rvs(size=100000)
empirical_var = samples.var()
empirical_std = samples.std()
print(f"Empirical Var(X) = {empirical_var:.4f}")
print(f"Empirical SD(X) = {empirical_std:.4f}")

# Variance of linear transformation
# Var(3X + 7) = 9 * Var(X)
transformed_var_theory = 9 * theoretical_var
transformed_var_empirical = (3 * samples + 7).var()
print(f"Var(3X + 7) theoretical: {transformed_var_theory:.4f}")
print(f"Var(3X + 7) empirical: {transformed_var_empirical:.4f}")
```

---

## 2. Discrete Probability Distributions

### 2.1 Bernoulli Distribution

Models a **single** binary trial with two possible outcomes (success/failure).

**Parameters**:
- p: probability of success (0 <= p <= 1)

**PMF**: P(X = k) = p^k * (1-p)^(1-k) for k in {0, 1}

**Statistics**:
- Mean: p
- Variance: p(1-p)
- Mode: 1 if p > 0.5, else 0

**Use Cases**:
- Single coin flip
- Binary classification outcome
- A/B test single conversion
- Click/no-click on ad
- Defective/non-defective product

```python
from scipy.stats import bernoulli

# Bernoulli with p=0.7 (70% success rate)
p = 0.7
dist = bernoulli(p)

# PMF
print(f"P(X = 1) = {dist.pmf(1):.2f}")  # 0.70
print(f"P(X = 0) = {dist.pmf(0):.2f}")  # 0.30

# Generate samples
samples = dist.rvs(size=1000, random_state=42)
print(f"Sample success rate: {samples.mean():.3f}")

# Statistics
print(f"Mean: {dist.mean()}")
print(f"Variance: {dist.var():.4f}")
print(f"Std Dev: {dist.std():.4f}")

# Parameter estimation from data
observed_data = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 1])
p_hat = observed_data.mean()  # MLE estimate
print(f"Estimated p: {p_hat:.2f}")
```

### 2.2 Binomial Distribution

Models the **number of successes** in n independent Bernoulli trials.

**Parameters**:
- n: number of trials (positive integer)
- p: probability of success on each trial

**PMF**: P(X = k) = C(n,k) * p^k * (1-p)^(n-k) for k = 0, 1, ..., n

**Statistics**:
- Mean: n*p
- Variance: n*p*(1-p)
- Mode: floor((n+1)*p)

**Use Cases**:
- Number of heads in n coin flips
- Quality control (defects in batch)
- Click-through rate (k clicks in n impressions)
- Free throw success (k made out of n attempts)
- Survey responses (k yes out of n respondents)

```python
from scipy.stats import binom

# 20 trials with 30% success rate
n, p = 20, 0.3
dist = binom(n, p)

# PMF - probability of exactly k successes
for k in [0, 5, 10, 15, 20]:
    print(f"P(X = {k:2d}) = {dist.pmf(k):.6f}")

# CDF - probability of at most k successes
print(f"\nP(X <= 5) = {dist.cdf(5):.4f}")
print(f"P(X <= 10) = {dist.cdf(10):.4f}")

# Survival function - probability of more than k successes
print(f"\nP(X > 8) = {dist.sf(8):.4f}")

# Statistics
print(f"\nMean: {dist.mean()}")
print(f"Variance: {dist.var()}")
print(f"Std Dev: {dist.std():.4f}")

# Confidence interval (95%)
lower, upper = dist.ppf(0.025), dist.ppf(0.975)
print(f"95% CI: [{lower:.0f}, {upper:.0f}]")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.4f}")

# Parameter estimation from data
observed_successes = 12
n_trials = 20
p_mle = observed_successes / n_trials
print(f"\nMLE estimate of p: {p_mle:.3f}")

# Confidence interval for p using Wilson score interval
from statsmodels.stats.proportion import proportion_confint
ci_lower, ci_upper = proportion_confint(observed_successes, n_trials, method='wilson')
print(f"95% CI for p: [{ci_lower:.3f}, {ci_upper:.3f}]")
```

### 2.3 Poisson Distribution

Models the **count of events** occurring in a fixed interval of time or space when events occur independently at a constant average rate.

**Parameter**:
- lambda (rate parameter): average number of events per interval

**PMF**: P(X = k) = (lambda^k * e^(-lambda)) / k! for k = 0, 1, 2, ...

**Statistics**:
- Mean: lambda
- Variance: lambda (variance equals mean!)
- Mode: floor(lambda)

**Use Cases**:
- Website visits per hour
- Customer arrivals per day
- Emails received per hour
- Defects per manufacturing unit
- Rare disease cases per year
- Radioactive decay counts

**Connection to Binomial**: Poisson is limiting case of Binomial(n, p) as n --> infinity, p --> 0, n*p --> lambda

```python
from scipy.stats import poisson

# Average of 4.5 events per hour
lam = 4.5
dist = poisson(lam)

# PMF
for k in [0, 2, 4, 6, 8, 10]:
    print(f"P(X = {k:2d}) = {dist.pmf(k):.6f}")

# CDF
print(f"\nP(X <= 3) = {dist.cdf(3):.4f}")
print(f"P(X > 7) = {1 - dist.cdf(7):.4f}")

# Statistics
print(f"\nMean: {dist.mean()}")
print(f"Variance: {dist.var()}")
print(f"Mean = Variance: {dist.mean() == dist.var()}")

# Interval probability
print(f"P(2 <= X <= 6) = {dist.cdf(6) - dist.cdf(1):.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)

# Parameter estimation from data (MLE)
observed_counts = np.array([3, 5, 4, 6, 2, 5, 4, 7, 3, 5])
lambda_mle = observed_counts.mean()
print(f"\nMLE estimate of lambda: {lambda_mle:.2f}")

# Goodness of fit test
from scipy.stats import chisquare
observed_freq = np.bincount(observed_counts)
expected_dist = poisson(lambda_mle)
expected_freq = expected_dist.pmf(np.arange(len(observed_freq))) * len(observed_counts)
chi2, p_value = chisquare(observed_freq, expected_freq)
print(f"Chi-square test: p-value = {p_value:.4f}")

# Poisson approximation to binomial
# When n is large and p is small
n_large, p_small = 1000, 0.005
binomial_dist = binom(n_large, p_small)
poisson_approx = poisson(n_large * p_small)

k_test = 3
print(f"\nBinomial P(X={k_test}): {binomial_dist.pmf(k_test):.6f}")
print(f"Poisson approx P(X={k_test}): {poisson_approx.pmf(k_test):.6f}")
```

### 2.4 Geometric Distribution

Models the **number of trials** until the first success in repeated Bernoulli trials.

**Parameter**:
- p: probability of success on each trial

**PMF**: P(X = k) = (1-p)^(k-1) * p for k = 1, 2, 3, ...

**Statistics**:
- Mean: 1/p
- Variance: (1-p) / p^2
- Mode: 1

**Memoryless Property**: P(X > s+t | X > s) = P(X > t)

**Use Cases**:
- Number of attempts until first success
- Time until first customer arrival
- Number of tests until first pass
- Insurance claims until first large claim

```python
from scipy.stats import geom

# 25% success rate per trial
p = 0.25
dist = geom(p)

# PMF - probability of first success on k-th trial
for k in [1, 2, 3, 4, 5, 10]:
    print(f"P(X = {k:2d}) = {dist.pmf(k):.6f}")

# CDF - probability of success within k trials
print(f"\nP(X <= 5) = {dist.cdf(5):.4f}")
print(f"P(X > 10) = {dist.sf(10):.4f}")

# Statistics
print(f"\nMean (expected trials): {dist.mean():.2f}")
print(f"Variance: {dist.var():.2f}")

# Memoryless property demonstration
# P(X > 5 + 3 | X > 5) = P(X > 3)
prob_conditional = dist.sf(8) / dist.sf(5)
prob_marginal = dist.sf(3)
print(f"\nMemoryless property:")
print(f"P(X > 8 | X > 5) = {prob_conditional:.4f}")
print(f"P(X > 3) = {prob_marginal:.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.2f}")

# Parameter estimation
observed_trials = np.array([2, 1, 4, 1, 3, 5, 1, 2, 1, 6])
p_mle = 1 / observed_trials.mean()
print(f"MLE estimate of p: {p_mle:.3f}")
```

### 2.5 Negative Binomial Distribution

Models the **number of trials** until the r-th success in repeated Bernoulli trials.

**Parameters**:
- r: number of successes (positive integer)
- p: probability of success on each trial

**PMF**: P(X = k) = C(k-1, r-1) * p^r * (1-p)^(k-r) for k = r, r+1, r+2, ...

**Statistics**:
- Mean: r/p
- Variance: r(1-p) / p^2
- Note: Geometric is special case when r=1

**Use Cases**:
- Number of trials until r-th success
- Over-dispersed count data (variance > mean)
- Modeling heterogeneity in Poisson processes
- Customer churn modeling

```python
from scipy.stats import nbinom

# Wait for 5 successes with p=0.3
r, p = 5, 0.3
# scipy uses different parameterization: n=r, p=p, but counts failures
dist = nbinom(n=r, p=p)

# PMF for number of failures before r successes
for k in [0, 5, 10, 15, 20]:
    print(f"P({k} failures before {r} successes) = {dist.pmf(k):.6f}")

# Statistics
print(f"\nMean failures: {dist.mean():.2f}")
print(f"Variance: {dist.var():.2f}")

# Alternative parameterization: total trials until r successes
# Mean total trials = r/p
print(f"Mean total trials: {r/p:.2f}")

# Over-dispersion example
# NegBin can model count data with variance > mean
lam = 5
r_param = 3
p_param = r_param / (r_param + lam)
nb_dist = nbinom(n=r_param, p=p_param)
print(f"\nNegBin with mean ~ {lam}:")
print(f"Mean: {nb_dist.mean():.2f}")
print(f"Variance: {nb_dist.var():.2f}")
print(f"Variance/Mean ratio: {nb_dist.var()/nb_dist.mean():.2f}")

# Compare to Poisson
poisson_dist = poisson(lam)
print(f"\nPoisson with lambda={lam}:")
print(f"Mean: {poisson_dist.mean():.2f}")
print(f"Variance: {poisson_dist.var():.2f}")
print(f"Variance/Mean ratio: {poisson_dist.var()/poisson_dist.mean():.2f}")
```

### 2.6 Hypergeometric Distribution

Models the number of successes in a sample drawn **without replacement** from a finite population.

**Parameters**:
- N: population size
- K: number of success states in population
- n: number of draws (sample size)

**PMF**: P(X = k) = C(K,k) * C(N-K, n-k) / C(N, n)

**Statistics**:
- Mean: n * K/N
- Variance: n * (K/N) * (1 - K/N) * (N-n)/(N-1)

**Use Cases**:
- Quality control sampling without replacement
- Card games (e.g., poker hands)
- Survey sampling from finite population
- Defect detection in batch

**Connection to Binomial**: When N is large relative to n, Hypergeometric approaches Binomial(n, K/N)

```python
from scipy.stats import hypergeom

# Population: 50 items, 20 defective, sample 10
N, K, n = 50, 20, 10
dist = hypergeom(M=N, n=K, N=n)  # scipy uses M, n, N notation

# PMF - probability of k defectives in sample
for k in range(0, min(n, K)+1, 2):
    print(f"P(X = {k}) = {dist.pmf(k):.6f}")

# Statistics
print(f"\nMean: {dist.mean():.2f}")
print(f"Expected proportion: {K/N:.2f}")
print(f"Variance: {dist.var():.4f}")

# CDF
print(f"\nP(X <= 3) = {dist.cdf(3):.4f}")
print(f"P(X >= 5) = {dist.sf(4):.4f}")

# Compare to Binomial approximation
p = K / N
binomial_approx = binom(n, p)
print(f"\nComparison to Binomial({n}, {p:.2f}):")
print(f"Hypergeometric P(X=4): {dist.pmf(4):.6f}")
print(f"Binomial P(X=4): {binomial_approx.pmf(4):.6f}")

# Practical example: quality control
# Lot of 1000 items, 50 defective, sample 20
N_lot, K_defect, n_sample = 1000, 50, 20
qc_dist = hypergeom(M=N_lot, n=K_defect, N=n_sample)

# Acceptance criterion: reject lot if >= 3 defectives found
reject_threshold = 3
prob_reject = qc_dist.sf(reject_threshold - 1)
print(f"\nQuality Control Example:")
print(f"Probability of rejecting lot: {prob_reject:.4f}")

# Generate samples
samples = dist.rvs(size=1000, random_state=42)
print(f"Empirical mean: {samples.mean():.2f}")
```

---

## 3. Continuous Probability Distributions

### 3.1 Normal (Gaussian) Distribution

The most important distribution in statistics, characterized by its bell-shaped curve.

**Parameters**:
- mu: mean (location parameter)
- sigma^2: variance (scale parameter)

**PDF**: f(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x - mu)^2 / (2*sigma^2))

**Statistics**:
- Mean: mu
- Variance: sigma^2
- Mode: mu
- Median: mu
- Skewness: 0 (symmetric)
- Kurtosis: 0 (excess kurtosis)

**Empirical Rule (68-95-99.7 rule)**:
- ~68% of data within 1 sigma of mu
- ~95% of data within 2 sigma of mu
- ~99.7% of data within 3 sigma of mu

**Use Cases**:
- Measurement errors
- Heights, weights, IQ scores
- Financial returns (approximately)
- Test scores
- Natural phenomena (CLT)
- ML: feature distributions after standardization

```python
from scipy.stats import norm

# Normal distribution with mu=100, sigma=15
mu, sigma = 100, 15
dist = norm(loc=mu, scale=sigma)

# PDF evaluation
x_test = 110
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF - probability X <= x
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")
print(f"P(X > {x_test}) = {dist.sf(x_test):.4f}")

# Empirical rule verification
print(f"\nEmpirical Rule:")
print(f"P(mu - sigma <= X <= mu + sigma) = {dist.cdf(mu + sigma) - dist.cdf(mu - sigma):.4f}")
print(f"P(mu - 2*sigma <= X <= mu + 2*sigma) = {dist.cdf(mu + 2*sigma) - dist.cdf(mu - 2*sigma):.4f}")
print(f"P(mu - 3*sigma <= X <= mu + 3*sigma) = {dist.cdf(mu + 3*sigma) - dist.cdf(mu - 3*sigma):.4f}")

# Quantiles (inverse CDF)
print(f"\n95th percentile: {dist.ppf(0.95):.2f}")
print(f"5th percentile: {dist.ppf(0.05):.2f}")

# 95% confidence interval
ci_lower, ci_upper = dist.ppf(0.025), dist.ppf(0.975)
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Standard normal (Z-score)
standard_normal = norm(loc=0, scale=1)
z_score = (x_test - mu) / sigma
print(f"\nZ-score for x={x_test}: {z_score:.2f}")
print(f"P(Z <= {z_score:.2f}) = {standard_normal.cdf(z_score):.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.2f}")
print(f"Empirical std: {samples.std():.2f}")

# Parameter estimation from data
sample_data = np.random.normal(75, 10, size=100)
mu_mle = sample_data.mean()
sigma_mle = sample_data.std(ddof=1)  # Use n-1 for unbiased estimate
print(f"\nMLE estimates:")
print(f"mu_hat = {mu_mle:.2f}")
print(f"sigma_hat = {sigma_mle:.2f}")

# Confidence interval for mean
from scipy.stats import t as t_dist
n = len(sample_data)
se = sigma_mle / np.sqrt(n)
t_critical = t_dist.ppf(0.975, df=n-1)
ci_mean_lower = mu_mle - t_critical * se
ci_mean_upper = mu_mle + t_critical * se
print(f"95% CI for mu: [{ci_mean_lower:.2f}, {ci_mean_upper:.2f}]")
```

### 3.2 Uniform Distribution

Equal probability density over a specified interval [a, b].

**Parameters**:
- a: minimum value
- b: maximum value

**PDF**: f(x) = 1/(b-a) for a <= x <= b, else 0

**CDF**: F(x) = (x-a)/(b-a) for a <= x <= b

**Statistics**:
- Mean: (a + b) / 2
- Variance: (b - a)^2 / 12
- Mode: any value in [a, b]

**Use Cases**:
- Random number generation
- Monte Carlo simulation
- Prior distribution (non-informative Bayesian)
- Modeling complete uncertainty
- Initialization in ML algorithms

```python
from scipy.stats import uniform

# Uniform on [5, 15]
a, b = 5, 15
dist = uniform(loc=a, scale=b-a)  # scipy uses loc and scale

# PDF is constant on support
x_values = [a, (a+b)/2, b]
for x in x_values:
    print(f"PDF at x={x}: {dist.pdf(x):.4f}")

# CDF is linear on support
print(f"\nP(X <= 8) = {dist.cdf(8):.4f}")
print(f"P(X <= 12) = {dist.cdf(12):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.2f}")
print(f"Variance: {dist.var():.4f}")

# Quantiles
for q in [0.25, 0.5, 0.75]:
    print(f"{int(q*100)}th percentile: {dist.ppf(q):.2f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.2f}")
print(f"Empirical min: {samples.min():.2f}")
print(f"Empirical max: {samples.max():.2f}")

# Standard uniform [0, 1]
standard_uniform = uniform(loc=0, scale=1)
u_samples = standard_uniform.rvs(size=5)
print(f"\nStandard uniform samples: {u_samples}")

# Inverse transform sampling
# Generate from any distribution using uniform samples
normal_from_uniform = norm.ppf(u_samples)
print(f"Normal samples from uniform: {normal_from_uniform}")
```

### 3.3 Exponential Distribution

Models time between events in a Poisson process (memoryless waiting time).

**Parameter**:
- lambda: rate parameter (lambda > 0)

**PDF**: f(x) = lambda * exp(-lambda * x) for x >= 0

**CDF**: F(x) = 1 - exp(-lambda * x)

**Statistics**:
- Mean: 1 / lambda
- Variance: 1 / lambda^2
- Mode: 0
- Median: ln(2) / lambda

**Memoryless Property**: P(X > s+t | X > s) = P(X > t)

**Use Cases**:
- Time between arrivals (Poisson process)
- Survival analysis
- Time to failure (reliability engineering)
- Service time in queuing theory
- Radioactive decay

```python
from scipy.stats import expon

# Mean waiting time of 10 minutes (lambda = 1/10 = 0.1)
mean_time = 10
lam = 1 / mean_time
dist = expon(scale=mean_time)  # scipy uses scale = 1/lambda

# PDF
x_test = 15
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")
print(f"P(X > {x_test}) = {dist.sf(x_test):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.2f}")
print(f"Variance: {dist.var():.2f}")
print(f"Median: {dist.median():.2f}")
print(f"Theoretical median: {np.log(2) * mean_time:.2f}")

# Memoryless property
s, t = 5, 10
prob_conditional = dist.sf(s + t) / dist.sf(s)
prob_marginal = dist.sf(t)
print(f"\nMemoryless property:")
print(f"P(X > {s+t} | X > {s}) = {prob_conditional:.4f}")
print(f"P(X > {t}) = {prob_marginal:.4f}")

# Quantiles
print(f"\n25th percentile: {dist.ppf(0.25):.2f}")
print(f"75th percentile: {dist.ppf(0.75):.2f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.2f}")

# Parameter estimation
observed_times = np.array([12.3, 8.7, 15.2, 6.4, 9.8, 11.5, 7.3, 14.1])
lambda_mle = 1 / observed_times.mean()
print(f"\nMLE estimate of lambda: {lambda_mle:.4f}")
print(f"Estimated mean time: {1/lambda_mle:.2f}")

# Relationship to Poisson
# If time between events ~ Exp(lambda), then number of events ~ Poisson(lambda*t)
time_interval = 30
expected_events = lam * time_interval
poisson_dist = poisson(expected_events)
print(f"\nIn {time_interval} minutes, expected {expected_events:.1f} events")
print(f"P(exactly 3 events) = {poisson_dist.pmf(3):.4f}")
```

### 3.4 Gamma Distribution

Generalizes the exponential distribution; models sum of exponential random variables.

**Parameters**:
- alpha (shape): shape parameter (alpha > 0)
- beta (rate) or theta (scale): beta = 1/theta

**PDF**: f(x) = (beta^alpha / Gamma(alpha)) * x^(alpha-1) * exp(-beta*x) for x >= 0

**Statistics**:
- Mean: alpha / beta = alpha * theta
- Variance: alpha / beta^2 = alpha * theta^2
- Mode: (alpha - 1) / beta for alpha >= 1

**Special Cases**:
- alpha = 1: Exponential distribution
- alpha = n/2, beta = 1/2: Chi-squared distribution with n degrees of freedom

**Use Cases**:
- Waiting time for multiple events
- Insurance claims
- Rainfall modeling
- Bayesian prior for rate parameters

```python
from scipy.stats import gamma

# Gamma with shape=2, scale=3
shape, scale = 2, 3
dist = gamma(a=shape, scale=scale)

# PDF
x_test = 5
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.2f}")
print(f"Variance: {dist.var():.2f}")
print(f"Std Dev: {dist.std():.2f}")

# Mode (for shape >= 1)
if shape >= 1:
    mode = (shape - 1) * scale
    print(f"Mode: {mode:.2f}")

# Quantiles
for q in [0.25, 0.5, 0.75, 0.95]:
    print(f"{int(q*100)}th percentile: {dist.ppf(q):.2f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.2f}")
print(f"Empirical variance: {samples.var():.2f}")

# Sum of exponentials
# Gamma(alpha=n, beta) is sum of n independent Exp(beta) random variables
n_exponentials = 5
rate = 0.5
gamma_dist = gamma(a=n_exponentials, scale=1/rate)
exp_dist = expon(scale=1/rate)

# Simulate sum of exponentials
exp_samples = exp_dist.rvs(size=(10000, n_exponentials), random_state=42)
sum_exp_samples = exp_samples.sum(axis=1)
gamma_samples = gamma_dist.rvs(size=10000, random_state=42)

print(f"\nSum of {n_exponentials} exponentials:")
print(f"Gamma dist mean: {gamma_dist.mean():.2f}")
print(f"Simulated sum mean: {sum_exp_samples.mean():.2f}")

# Parameter estimation using MLE
from scipy.optimize import minimize

def gamma_neg_log_likelihood(params, data):
    alpha, beta = params
    if alpha <= 0 or beta <= 0:
        return np.inf
    return -np.sum(stats.gamma.logpdf(data, a=alpha, scale=1/beta))

observed_data = gamma(a=3, scale=2).rvs(size=100, random_state=42)
result = minimize(gamma_neg_log_likelihood, x0=[2, 0.5], args=(observed_data,),
                  method='L-BFGS-B', bounds=[(0.01, None), (0.01, None)])
alpha_mle, beta_mle = result.x
print(f"\nMLE estimates:")
print(f"alpha_hat = {alpha_mle:.4f}")
print(f"beta_hat = {beta_mle:.4f}")
print(f"scale_hat = {1/beta_mle:.4f}")
```

### 3.5 Beta Distribution

Continuous distribution on [0, 1], extremely flexible for modeling proportions and probabilities.

**Parameters**:
- alpha: shape parameter (alpha > 0)
- beta: shape parameter (beta > 0)

**PDF**: f(x) = (x^(alpha-1) * (1-x)^(beta-1)) / B(alpha, beta) for 0 <= x <= 1

where B(alpha, beta) is the beta function

**Statistics**:
- Mean: alpha / (alpha + beta)
- Variance: (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
- Mode: (alpha - 1) / (alpha + beta - 2) for alpha, beta > 1

**Shape Properties**:
- alpha = beta = 1: Uniform[0, 1]
- alpha = beta > 1: Symmetric, bell-shaped
- alpha < beta: Skewed right
- alpha > beta: Skewed left
- alpha, beta < 1: U-shaped

**Use Cases**:
- Bayesian prior for binomial probability
- Proportion modeling
- Success rates, conversion rates
- Project completion percentages
- Order statistics

```python
from scipy.stats import beta

# Beta(2, 5) - skewed towards 0
alpha, beta_param = 2, 5
dist = beta(a=alpha, b=beta_param)

# PDF
x_values = np.linspace(0, 1, 5)
print("PDF values:")
for x in x_values:
    print(f"f({x:.2f}) = {dist.pdf(x):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.4f}")
print(f"Theoretical mean: {alpha / (alpha + beta_param):.4f}")
print(f"Variance: {dist.var():.6f}")

# Mode (for alpha, beta > 1)
if alpha > 1 and beta_param > 1:
    mode = (alpha - 1) / (alpha + beta_param - 2)
    print(f"Mode: {mode:.4f}")

# Quantiles
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    print(f"{int(q*100)}th percentile: {dist.ppf(q):.4f}")

# Different shapes visualization
shapes = [
    (0.5, 0.5, 'U-shaped'),
    (1, 1, 'Uniform'),
    (2, 2, 'Symmetric bell'),
    (2, 5, 'Right skewed'),
    (5, 2, 'Left skewed'),
    (10, 10, 'Concentrated center')
]

x = np.linspace(0, 1, 200)
for a, b, desc in shapes:
    dist_temp = beta(a, b)
    print(f"\nBeta({a}, {b}) - {desc}:")
    print(f"  Mean: {dist_temp.mean():.4f}")
    print(f"  Std: {dist_temp.std():.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)
print(f"\nEmpirical mean: {samples.mean():.4f}")
print(f"Empirical std: {samples.std():.4f}")

# Bayesian update example (see section 9 for more)
# Prior: Beta(2, 2) - weakly informative
prior_alpha, prior_beta = 2, 2
prior = beta(prior_alpha, prior_beta)

# Observe: 7 successes in 10 trials
successes, trials = 7, 10
failures = trials - successes

# Posterior: Beta(alpha + successes, beta + failures)
post_alpha = prior_alpha + successes
post_beta = prior_beta + failures
posterior = beta(post_alpha, post_beta)

print(f"\nBayesian Update:")
print(f"Prior mean: {prior.mean():.4f}")
print(f"Posterior mean: {posterior.mean():.4f}")
print(f"MLE (frequentist): {successes/trials:.4f}")
```

### 3.6 Log-Normal Distribution

Distribution of a random variable whose logarithm is normally distributed.

**Parameters**:
- mu: mean of log(X)
- sigma: standard deviation of log(X)

**PDF**: f(x) = (1 / (x * sigma * sqrt(2*pi))) * exp(-(ln(x) - mu)^2 / (2*sigma^2)) for x > 0

**Statistics**:
- Mean: exp(mu + sigma^2/2)
- Variance: (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
- Mode: exp(mu - sigma^2)
- Median: exp(mu)

**Use Cases**:
- Income distributions
- Stock prices
- File sizes
- Particle sizes
- Survival times
- ML: positive-valued features with multiplicative effects

```python
from scipy.stats import lognorm

# Log-normal with mu=2, sigma=0.5 for log(X)
mu, sigma = 2, 0.5
dist = lognorm(s=sigma, scale=np.exp(mu))  # scipy parameterization

# PDF
x_test = 10
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.4f}")
print(f"Theoretical mean: {np.exp(mu + sigma**2/2):.4f}")
print(f"Median: {dist.median():.4f}")
print(f"Theoretical median: {np.exp(mu):.4f}")
print(f"Mode: {np.exp(mu - sigma**2):.4f}")
print(f"Variance: {dist.var():.4f}")

# Relationship to normal distribution
samples_lognormal = dist.rvs(size=10000, random_state=42)
samples_log_transformed = np.log(samples_lognormal)

print(f"\nLog-transformed samples:")
print(f"Mean of log(X): {samples_log_transformed.mean():.4f} (expected: {mu})")
print(f"Std of log(X): {samples_log_transformed.std():.4f} (expected: {sigma})")

# Verify normality of log-transformed data
from scipy.stats import shapiro
stat, p_value = shapiro(samples_log_transformed[:5000])  # Use subset for speed
print(f"Shapiro-Wilk test on log(X): p-value = {p_value:.4f}")

# Quantiles
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    print(f"{int(q*100)}th percentile: {dist.ppf(q):.4f}")

# Parameter estimation
observed_data = lognorm(s=0.6, scale=np.exp(1.5)).rvs(size=100, random_state=42)

# Method 1: Transform to normal and estimate
log_data = np.log(observed_data)
mu_mle = log_data.mean()
sigma_mle = log_data.std(ddof=1)
print(f"\nMLE estimates:")
print(f"mu_hat = {mu_mle:.4f}")
print(f"sigma_hat = {sigma_mle:.4f}")

# Method 2: Use scipy fit
shape_fit, loc_fit, scale_fit = lognorm.fit(observed_data, floc=0)
print(f"\nScipy fit:")
print(f"sigma_hat = {shape_fit:.4f}")
print(f"mu_hat (via scale) = {np.log(scale_fit):.4f}")
```

### 3.7 Student's t-Distribution

Heavy-tailed distribution used for small sample inference when population variance is unknown.

**Parameter**:
- nu (degrees of freedom): nu > 0

**PDF**: f(x) = (Gamma((nu+1)/2) / (sqrt(nu*pi) * Gamma(nu/2))) * (1 + x^2/nu)^(-(nu+1)/2)

**Statistics**:
- Mean: 0 (for standard t)
- Variance: nu / (nu - 2) for nu > 2
- Approaches standard normal as nu --> infinity

**Use Cases**:
- Confidence intervals for small samples
- t-tests
- Regression inference
- Robust statistics (heavy tails)

```python
from scipy.stats import t as t_dist

# t-distribution with 5 degrees of freedom
df = 5
dist = t_dist(df)

# PDF comparison with normal
x_range = np.linspace(-4, 4, 100)
t_pdf = dist.pdf(x_range)
normal_pdf = norm(0, 1).pdf(x_range)

print("PDF at x=0:")
print(f"t({df}): {dist.pdf(0):.6f}")
print(f"Normal: {norm.pdf(0):.6f}")

print("\nPDF at x=2 (heavier tails):")
print(f"t({df}): {dist.pdf(2):.6f}")
print(f"Normal: {norm.pdf(2):.6f}")

# Statistics
print(f"\nMean: {dist.mean():.4f}")
if df > 2:
    print(f"Variance: {dist.var():.4f}")
    print(f"Theoretical variance: {df / (df - 2):.4f}")

# Critical values for confidence intervals
for alpha in [0.10, 0.05, 0.01]:
    t_critical = dist.ppf(1 - alpha/2)
    print(f"\nt-critical (alpha={alpha}): {t_critical:.4f}")

# Confidence interval example
sample_data = np.array([12.3, 14.1, 11.8, 13.5, 12.9, 14.7, 13.2])
n = len(sample_data)
sample_mean = sample_data.mean()
sample_std = sample_data.std(ddof=1)
se = sample_std / np.sqrt(n)

df_sample = n - 1
t_crit = t_dist.ppf(0.975, df_sample)
ci_lower = sample_mean - t_crit * se
ci_upper = sample_mean + t_crit * se

print(f"\n95% CI for mean:")
print(f"Sample mean: {sample_mean:.2f}")
print(f"CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# Convergence to normal as df increases
for df_test in [1, 2, 5, 10, 30, 100]:
    dist_test = t_dist(df_test)
    print(f"\ndf={df_test:3d}: P(|X| > 2) = {2 * dist_test.sf(2):.6f}")
print(f"Normal:  P(|X| > 2) = {2 * norm.sf(2):.6f}")
```

### 3.8 Chi-Squared Distribution

Distribution of sum of squared standard normal random variables.

**Parameter**:
- k (degrees of freedom): k > 0

**PDF**: f(x) = (1 / (2^(k/2) * Gamma(k/2))) * x^(k/2 - 1) * exp(-x/2) for x >= 0

**Statistics**:
- Mean: k
- Variance: 2*k
- Mode: max(k - 2, 0)

**Relationship**: Chi-squared(k) is Gamma(k/2, 1/2)

**Use Cases**:
- Goodness-of-fit tests
- Tests of independence
- Variance estimation
- Feature selection (chi-squared test)

```python
from scipy.stats import chi2

# Chi-squared with 5 degrees of freedom
df = 5
dist = chi2(df)

# PDF
x_test = 6
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")

# Statistics
print(f"\nMean: {dist.mean():.4f} (expected: {df})")
print(f"Variance: {dist.var():.4f} (expected: {2*df})")
if df >= 2:
    print(f"Mode: {df - 2:.4f}")

# Critical values for hypothesis testing
for alpha in [0.10, 0.05, 0.01]:
    chi2_critical = dist.ppf(1 - alpha)
    print(f"\nChi-squared critical value (alpha={alpha}): {chi2_critical:.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)

# Verify it's sum of squared normals
normal_samples = norm(0, 1).rvs(size=(10000, df), random_state=42)
sum_squared_normals = (normal_samples ** 2).sum(axis=1)

print(f"\nVerification (sum of squared normals):")
print(f"Chi-squared samples mean: {samples.mean():.2f}")
print(f"Sum of squared normals mean: {sum_squared_normals.mean():.2f}")

# Goodness-of-fit test example
# Test if observed frequencies match expected
observed = np.array([18, 23, 16, 25, 18])
expected = np.array([20, 20, 20, 20, 20])

chi2_statistic = ((observed - expected) ** 2 / expected).sum()
df_test = len(observed) - 1
p_value = chi2(df_test).sf(chi2_statistic)

print(f"\nGoodness-of-fit test:")
print(f"Chi-squared statistic: {chi2_statistic:.4f}")
print(f"Degrees of freedom: {df_test}")
print(f"p-value: {p_value:.4f}")

# Sample variance distribution
# If X1, ..., Xn ~ N(mu, sigma^2), then (n-1)*S^2/sigma^2 ~ Chi-squared(n-1)
true_sigma = 5
n = 20
normal_dist = norm(0, true_sigma)

# Simulate many sample variances
n_simulations = 10000
sample_variances = []
for _ in range(n_simulations):
    sample = normal_dist.rvs(size=n, random_state=np.random.randint(0, 10000))
    sample_variances.append(sample.var(ddof=1))

sample_variances = np.array(sample_variances)
scaled_variances = (n - 1) * sample_variances / (true_sigma ** 2)

print(f"\nSample variance distribution:")
print(f"Mean of scaled variances: {scaled_variances.mean():.2f} (expected: {n-1})")
print(f"Variance of scaled variances: {scaled_variances.var():.2f} (expected: {2*(n-1)})")
```

### 3.9 F-Distribution

Ratio of two scaled chi-squared distributions; used in ANOVA and regression.

**Parameters**:
- d1: degrees of freedom for numerator
- d2: degrees of freedom for denominator

**PDF**: Complex form involving beta function

**Statistics**:
- Mean: d2 / (d2 - 2) for d2 > 2
- Variance: Complex formula

**Use Cases**:
- ANOVA F-test
- Regression F-test
- Comparing variances
- Model comparison

```python
from scipy.stats import f as f_dist

# F-distribution with d1=5, d2=10
d1, d2 = 5, 10
dist = f_dist(d1, d2)

# PDF
x_test = 2.5
print(f"PDF at x={x_test}: {dist.pdf(x_test):.6f}")

# CDF
print(f"P(X <= {x_test}) = {dist.cdf(x_test):.4f}")

# Statistics
if d2 > 2:
    print(f"\nMean: {dist.mean():.4f}")
    print(f"Theoretical mean: {d2 / (d2 - 2):.4f}")
print(f"Variance: {dist.var():.4f}")

# Critical values for hypothesis testing
for alpha in [0.10, 0.05, 0.01]:
    f_critical = dist.ppf(1 - alpha)
    print(f"\nF-critical (alpha={alpha}): {f_critical:.4f}")

# Generate samples
samples = dist.rvs(size=10000, random_state=42)

# Verify as ratio of chi-squared distributions
chi2_1 = chi2(d1).rvs(size=10000, random_state=42)
chi2_2 = chi2(d2).rvs(size=10000, random_state=43)
f_from_chi2 = (chi2_1 / d1) / (chi2_2 / d2)

print(f"\nVerification (ratio of chi-squared):")
print(f"F-dist samples mean: {samples.mean():.4f}")
print(f"Ratio of chi-squared mean: {f_from_chi2.mean():.4f}")

# ANOVA example
# Test if three groups have equal means
group1 = np.array([23, 25, 27, 24, 26])
group2 = np.array([30, 32, 29, 31, 33])
group3 = np.array([18, 20, 19, 21, 17])

from scipy.stats import f_oneway
f_statistic, p_value = f_oneway(group1, group2, group3)

k = 3  # number of groups
n = len(group1) + len(group2) + len(group3)
df_between = k - 1
df_within = n - k

print(f"\nANOVA Test:")
print(f"F-statistic: {f_statistic:.4f}")
print(f"df1={df_between}, df2={df_within}")
print(f"p-value: {p_value:.6f}")

# Variance ratio test
var1, var2 = 25.0, 16.0
f_ratio = var1 / var2
df1_var, df2_var = 9, 9
p_value_var = 2 * min(f_dist(df1_var, df2_var).cdf(f_ratio),
                      f_dist(df1_var, df2_var).sf(f_ratio))
print(f"\nVariance Ratio Test:")
print(f"F = {f_ratio:.4f}")
print(f"p-value (two-tailed): {p_value_var:.4f}")
```

---

## 4. Distribution Fitting and Goodness-of-Fit Tests

### 4.1 Maximum Likelihood Estimation (MLE)

MLE finds parameters that maximize the likelihood of observing the data.

**Likelihood Function**: L(theta | data) = product of f(x_i | theta) for all observations

**Log-Likelihood**: log L(theta | data) = sum of log f(x_i | theta)

**MLE**: theta_hat = argmax L(theta | data)

```python
import numpy as np
from scipy import stats, optimize

# Example: Fit normal distribution to data
np.random.seed(42)
true_mu, true_sigma = 50, 10
data = np.random.normal(true_mu, true_sigma, size=200)

# Manual MLE for normal distribution
def normal_neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    return -np.sum(stats.norm.logpdf(data, loc=mu, scale=sigma))

# Optimize
result = optimize.minimize(normal_neg_log_likelihood, x0=[45, 8], args=(data,),
                          method='L-BFGS-B', bounds=[(None, None), (0.01, None)])
mu_mle, sigma_mle = result.x

print("Normal Distribution MLE:")
print(f"True parameters: mu={true_mu}, sigma={true_sigma}")
print(f"MLE estimates: mu={mu_mle:.4f}, sigma={sigma_mle:.4f}")
print(f"Sample mean: {data.mean():.4f}")
print(f"Sample std: {data.std(ddof=1):.4f}")

# Analytical MLE for normal (closed form)
mu_analytical = data.mean()
sigma_analytical = np.sqrt(((data - mu_analytical) ** 2).sum() / len(data))
print(f"Analytical MLE: mu={mu_analytical:.4f}, sigma={sigma_analytical:.4f}")
```

### 4.2 Fitting Distributions with scipy.stats

scipy.stats provides a convenient `fit` method for all distributions.

```python
# Fit various distributions to data
distributions = [
    ('norm', stats.norm),
    ('expon', stats.expon),
    ('gamma', stats.gamma),
    ('lognorm', stats.lognorm),
    ('beta', stats.beta)
]

# Generate test data (gamma distributed)
true_dist = stats.gamma(a=3, scale=2)
data = true_dist.rvs(size=500, random_state=42)

print("Fitting multiple distributions to data:\n")
fit_results = []

for name, dist_class in distributions:
    # Fit distribution
    if name == 'beta':
        # Beta needs data in [0, 1]
        data_scaled = (data - data.min()) / (data.max() - data.min())
        params = dist_class.fit(data_scaled)
        fitted_dist = dist_class(*params)
    else:
        params = dist_class.fit(data)
        fitted_dist = dist_class(*params)

    # Calculate negative log-likelihood
    if name == 'beta':
        nll = -fitted_dist.logpdf(data_scaled).sum()
    else:
        nll = -fitted_dist.logpdf(data).sum()

    # Calculate AIC and BIC
    k = len(params)  # number of parameters
    n = len(data)
    aic = 2 * k + 2 * nll
    bic = k * np.log(n) + 2 * nll

    fit_results.append((name, params, nll, aic, bic))

    print(f"{name:10s}: AIC={aic:8.2f}, BIC={bic:8.2f}, NLL={nll:8.2f}")

# Best fit by AIC
best_fit = min(fit_results, key=lambda x: x[3])
print(f"\nBest fit by AIC: {best_fit[0]}")

# Specialized fitting with constraints
# Fix location parameter
params_fixed = stats.gamma.fit(data, floc=0)
print(f"\nGamma fit with fixed loc=0:")
print(f"Parameters: {params_fixed}")

# Fit with initial guess
params_guess = stats.gamma.fit(data, f0=2)
print(f"Gamma fit with initial guess: {params_guess}")
```

### 4.3 Q-Q Plots (Quantile-Quantile Plots)

Q-Q plots compare quantiles of data against theoretical distribution quantiles.

```python
import matplotlib.pyplot as plt
from scipy import stats

# Generate data from different distributions
np.random.seed(42)
normal_data = stats.norm(50, 10).rvs(size=200)
lognormal_data = stats.lognorm(s=0.5, scale=np.exp(3)).rvs(size=200)
uniform_data = stats.uniform(0, 100).rvs(size=200)

def qq_plot_analysis(data, dist, title):
    """Create Q-Q plot and return correlation coefficient"""
    # Generate theoretical quantiles
    sorted_data = np.sort(data)
    n = len(data)
    theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, n))

    # Calculate correlation (measure of fit)
    corr = np.corrcoef(theoretical_quantiles, sorted_data)[0, 1]

    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(theoretical_quantiles, sorted_data, alpha=0.6)
    plt.plot([theoretical_quantiles.min(), theoretical_quantiles.max()],
             [theoretical_quantiles.min(), theoretical_quantiles.max()],
             'r--', linewidth=2, label='Perfect fit')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title(f'{title}\nCorrelation: {corr:.4f}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    return corr

# Q-Q plot for normal data vs normal distribution
corr1 = qq_plot_analysis(normal_data, stats.norm(50, 10), 'Normal Data vs Normal Distribution')
print(f"Normal data vs Normal dist: r = {corr1:.4f}")

# Q-Q plot for lognormal data vs normal distribution (should be poor fit)
corr2 = qq_plot_analysis(lognormal_data, stats.norm(lognormal_data.mean(), lognormal_data.std()),
                         'Lognormal Data vs Normal Distribution')
print(f"Lognormal data vs Normal dist: r = {corr2:.4f}")

# scipy built-in Q-Q plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

stats.probplot(normal_data, dist="norm", plot=axes[0])
axes[0].set_title('Normal Data')

stats.probplot(lognormal_data, dist="norm", plot=axes[1])
axes[1].set_title('Lognormal Data')

stats.probplot(np.log(lognormal_data), dist="norm", plot=axes[2])
axes[2].set_title('Log-transformed Lognormal Data')

plt.tight_layout()
plt.show()
```

### 4.4 Kolmogorov-Smirnov Test

Tests if sample comes from a specified distribution by comparing empirical CDF to theoretical CDF.

**Test Statistic**: D = max|F_n(x) - F(x)| where F_n is empirical CDF, F is theoretical CDF

**Null Hypothesis**: Data follows the specified distribution

```python
from scipy.stats import kstest, ks_2samp

# Generate test data
np.random.seed(42)
normal_data = stats.norm(0, 1).rvs(size=100)
uniform_data = stats.uniform(0, 1).rvs(size=100)

# One-sample KS test: compare data to theoretical distribution
print("One-Sample Kolmogorov-Smirnov Test:\n")

# Test normal data against normal distribution
statistic, p_value = kstest(normal_data, 'norm', args=(0, 1))
print(f"Normal data vs N(0,1):")
print(f"  KS statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Reject H0: {p_value < 0.05}")

# Test normal data against uniform distribution (should reject)
statistic, p_value = kstest(normal_data, 'uniform')
print(f"\nNormal data vs Uniform:")
print(f"  KS statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")
print(f"  Reject H0: {p_value < 0.05}")

# Test with fitted parameters
data_to_test = stats.gamma(a=2, scale=3).rvs(size=200, random_state=42)
params_fit = stats.gamma.fit(data_to_test)
statistic, p_value = kstest(data_to_test, 'gamma', args=params_fit)
print(f"\nGamma data vs fitted Gamma:")
print(f"  KS statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")

# Two-sample KS test: compare two empirical distributions
sample1 = stats.norm(0, 1).rvs(size=100, random_state=42)
sample2 = stats.norm(0.5, 1).rvs(size=100, random_state=43)
sample3 = stats.norm(0, 1).rvs(size=100, random_state=44)

print(f"\nTwo-Sample Kolmogorov-Smirnov Test:\n")

# Same distribution
statistic, p_value = ks_2samp(sample1, sample3)
print(f"Sample from N(0,1) vs Sample from N(0,1):")
print(f"  KS statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")

# Different distributions
statistic, p_value = ks_2samp(sample1, sample2)
print(f"\nSample from N(0,1) vs Sample from N(0.5,1):")
print(f"  KS statistic: {statistic:.4f}")
print(f"  p-value: {p_value:.4f}")

# Visualize empirical CDF vs theoretical CDF
def plot_ks_test(data, dist, params, title):
    sorted_data = np.sort(data)
    n = len(data)
    empirical_cdf = np.arange(1, n+1) / n
    theoretical_cdf = dist.cdf(sorted_data, *params)

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_data, empirical_cdf, label='Empirical CDF', linewidth=2)
    plt.plot(sorted_data, theoretical_cdf, label='Theoretical CDF', linewidth=2)
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

plot_ks_test(normal_data, stats.norm, (0, 1), 'KS Test: Normal Data vs N(0,1)')
```

### 4.5 Anderson-Darling Test

More sensitive goodness-of-fit test that gives more weight to tail deviations.

**Test Statistic**: A^2 = -n - (1/n) * sum of weighted squared deviations

**Null Hypothesis**: Data follows the specified distribution

```python
from scipy.stats import anderson

# Generate test data
np.random.seed(42)
normal_data = stats.norm(50, 10).rvs(size=200)
exponential_data = stats.expon(scale=5).rvs(size=200)

print("Anderson-Darling Test:\n")

# Test normal data
result = anderson(normal_data, dist='norm')
print(f"Testing normal data:")
print(f"  Statistic: {result.statistic:.4f}")
print(f"  Critical values: {result.critical_values}")
print(f"  Significance levels: {result.significance_level}%")

for i, (critical, significance) in enumerate(zip(result.critical_values, result.significance_level)):
    if result.statistic < critical:
        print(f"  At {significance}% level: Fail to reject (data appears normal)")
    else:
        print(f"  At {significance}% level: Reject (data not normal)")

# Test exponential data against normal (should reject)
result = anderson(exponential_data, dist='norm')
print(f"\nTesting exponential data against normal:")
print(f"  Statistic: {result.statistic:.4f}")
print(f"  At 5% level: {'Reject' if result.statistic > result.critical_values[2] else 'Fail to reject'}")

# Test exponential data against exponential
result = anderson(exponential_data, dist='expon')
print(f"\nTesting exponential data against exponential:")
print(f"  Statistic: {result.statistic:.4f}")
print(f"  At 5% level: {'Reject' if result.statistic > result.critical_values[2] else 'Fail to reject'}")

# Comparison of normality tests
from scipy.stats import shapiro, normaltest

print(f"\nComparison of Normality Tests on normal data:")

# Shapiro-Wilk test
stat_sw, p_sw = shapiro(normal_data)
print(f"Shapiro-Wilk: statistic={stat_sw:.4f}, p-value={p_sw:.4f}")

# D'Agostino-Pearson test
stat_dp, p_dp = normaltest(normal_data)
print(f"D'Agostino-Pearson: statistic={stat_dp:.4f}, p-value={p_dp:.4f}")

# Anderson-Darling test
result_ad = anderson(normal_data, dist='norm')
print(f"Anderson-Darling: statistic={result_ad.statistic:.4f}")

print(f"\nComparison of Normality Tests on exponential data:")
stat_sw, p_sw = shapiro(exponential_data)
print(f"Shapiro-Wilk: p-value={p_sw:.4f}")

stat_dp, p_dp = normaltest(exponential_data)
print(f"D'Agostino-Pearson: p-value={p_dp:.4f}")

result_ad = anderson(exponential_data, dist='norm')
print(f"Anderson-Darling: statistic={result_ad.statistic:.4f} (critical at 5%: {result_ad.critical_values[2]:.4f})")
```

---

## 5. Central Limit Theorem

### 5.1 Statement and Intuition

**Central Limit Theorem (CLT)**: For independent, identically distributed random variables X1, X2, ..., Xn with mean mu and variance sigma^2, the sample mean approaches a normal distribution as n increases:

sqrt(n) * (X_bar - mu) / sigma --> N(0, 1)

Equivalently: X_bar approximately N(mu, sigma^2/n)

**Key Points**:
- Works for ANY underlying distribution (with finite mean and variance)
- Convergence rate depends on original distribution
- Typically n >= 30 is sufficient for good approximation
- More skewed distributions need larger n

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Demonstrate CLT with different underlying distributions
def demonstrate_clt(dist, dist_name, sample_sizes=[1, 5, 10, 30]):
    """Demonstrate CLT for a given distribution"""
    n_simulations = 10000

    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(16, 8))

    for idx, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(n_simulations):
            sample = dist.rvs(size=n)
            sample_means.append(sample.mean())

        sample_means = np.array(sample_means)

        # Standardize sample means
        mu_true = dist.mean()
        sigma_true = dist.std()
        standardized_means = np.sqrt(n) * (sample_means - mu_true) / sigma_true

        # Plot histogram of sample means
        axes[0, idx].hist(sample_means, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, idx].set_title(f'n = {n}')
        axes[0, idx].set_ylabel('Density' if idx == 0 else '')
        axes[0, idx].grid(alpha=0.3)

        # Add theoretical normal
        x_range = np.linspace(sample_means.min(), sample_means.max(), 100)
        theoretical_normal = stats.norm(mu_true, sigma_true / np.sqrt(n))
        axes[0, idx].plot(x_range, theoretical_normal.pdf(x_range),
                         'r-', linewidth=2, label='Theoretical N')
        axes[0, idx].legend()

        # Plot histogram of standardized means
        axes[1, idx].hist(standardized_means, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[1, idx].set_xlabel('Standardized Sample Mean')
        axes[1, idx].set_ylabel('Density' if idx == 0 else '')
        axes[1, idx].grid(alpha=0.3)

        # Add standard normal
        x_range = np.linspace(-4, 4, 100)
        axes[1, idx].plot(x_range, stats.norm(0, 1).pdf(x_range),
                         'r-', linewidth=2, label='N(0,1)')
        axes[1, idx].set_xlim(-4, 4)
        axes[1, idx].legend()

    fig.suptitle(f'Central Limit Theorem: {dist_name}', fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()

# Test with various distributions
print("Demonstrating CLT with different distributions:\n")

# 1. Exponential (highly skewed)
demonstrate_clt(stats.expon(scale=2), 'Exponential Distribution')

# 2. Uniform
demonstrate_clt(stats.uniform(0, 10), 'Uniform Distribution')

# 3. Binomial
demonstrate_clt(stats.binom(n=10, p=0.3), 'Binomial Distribution')
```

### 5.2 Proof by Simulation

```python
# Rigorous simulation to verify CLT
def verify_clt(dist, dist_name, n_values=[5, 10, 30, 100], n_simulations=100000):
    """Verify CLT convergence for different sample sizes"""
    mu_true = dist.mean()
    sigma_true = dist.std()

    print(f"\nCLT Verification for {dist_name}:")
    print(f"True mean: {mu_true:.4f}, True std: {sigma_true:.4f}\n")

    results = []

    for n in n_values:
        # Generate sample means
        sample_means = np.array([dist.rvs(size=n).mean()
                                 for _ in range(n_simulations)])

        # Theoretical distribution of sample mean
        theoretical_mean = mu_true
        theoretical_std = sigma_true / np.sqrt(n)

        # Empirical statistics
        empirical_mean = sample_means.mean()
        empirical_std = sample_means.std()

        # Kolmogorov-Smirnov test against normal
        ks_stat, ks_pvalue = stats.kstest(
            sample_means,
            lambda x: stats.norm.cdf(x, mu_true, theoretical_std)
        )

        # Shapiro-Wilk test for normality (on subsample due to size limit)
        sw_stat, sw_pvalue = stats.shapiro(sample_means[:5000])

        print(f"n = {n:3d}:")
        print(f"  Empirical mean: {empirical_mean:.6f} (expected: {theoretical_mean:.6f})")
        print(f"  Empirical std:  {empirical_std:.6f} (expected: {theoretical_std:.6f})")
        print(f"  KS test p-value: {ks_pvalue:.6f}")
        print(f"  SW test p-value: {sw_pvalue:.6f}")
        print()

        results.append({
            'n': n,
            'mean_error': abs(empirical_mean - theoretical_mean),
            'std_error': abs(empirical_std - theoretical_std),
            'ks_pvalue': ks_pvalue
        })

    return results

# Test with highly skewed distribution
expo_dist = stats.expon(scale=5)
expo_results = verify_clt(expo_dist, "Exponential(scale=5)",
                          n_values=[5, 10, 30, 100, 500])

# Test with discrete distribution
binom_dist = stats.binom(n=10, p=0.2)
binom_results = verify_clt(binom_dist, "Binomial(n=10, p=0.2)",
                           n_values=[5, 10, 30, 100, 500])
```

### 5.3 Practical Implications

```python
# Practical applications of CLT

# 1. Confidence intervals for means
def calculate_ci_using_clt(data, confidence=0.95):
    """Calculate confidence interval using CLT"""
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)
    se = std / np.sqrt(n)

    # Use normal approximation (CLT)
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    margin = z_critical * se

    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, ci_lower, ci_upper

# Example with non-normal data
np.random.seed(42)
non_normal_data = stats.expon(scale=10).rvs(size=100)

mean_est, ci_low, ci_high = calculate_ci_using_clt(non_normal_data, 0.95)
print(f"Confidence Interval (95%) using CLT:")
print(f"Sample mean: {mean_est:.2f}")
print(f"CI: [{ci_low:.2f}, {ci_high:.2f}]")

# Verify coverage
true_mean = 10
n_simulations = 10000
coverage_count = 0

for _ in range(n_simulations):
    sample = stats.expon(scale=10).rvs(size=100)
    _, ci_l, ci_h = calculate_ci_using_clt(sample, 0.95)
    if ci_l <= true_mean <= ci_h:
        coverage_count += 1

coverage_rate = coverage_count / n_simulations
print(f"\nEmpirical coverage rate: {coverage_rate:.4f} (expected: 0.95)")

# 2. Sample size determination
def required_sample_size(sigma, margin_error, confidence=0.95):
    """Calculate required sample size for desired margin of error"""
    z_critical = stats.norm.ppf((1 + confidence) / 2)
    n = (z_critical * sigma / margin_error) ** 2
    return int(np.ceil(n))

sigma_estimate = 15
desired_margin = 2
n_required = required_sample_size(sigma_estimate, desired_margin, 0.95)
print(f"\nSample Size Calculation:")
print(f"To achieve margin of error = {desired_margin} with 95% confidence")
print(f"Given sigma ~ {sigma_estimate}")
print(f"Required sample size: {n_required}")

# 3. Hypothesis testing using CLT
def z_test(data, mu0, alternative='two-sided'):
    """Perform Z-test using CLT"""
    n = len(data)
    mean = data.mean()
    std = data.std(ddof=1)
    se = std / np.sqrt(n)

    z_statistic = (mean - mu0) / se

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    elif alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_statistic)
    elif alternative == 'less':
        p_value = stats.norm.cdf(z_statistic)

    return z_statistic, p_value

# Test if mean > 100
test_data = stats.norm(105, 20).rvs(size=50, random_state=42)
z_stat, p_val = z_test(test_data, mu0=100, alternative='greater')
print(f"\nZ-test (H0: mu <= 100 vs H1: mu > 100):")
print(f"Z-statistic: {z_stat:.4f}")
print(f"p-value: {p_val:.4f}")
print(f"Reject H0 at alpha=0.05: {p_val < 0.05}")
```

### 5.4 When CLT Fails

```python
# Cases where CLT doesn't apply or converges slowly

print("Cases Where CLT Fails or Converges Slowly:\n")

# 1. Heavy-tailed distributions (infinite variance)
# Cauchy distribution has infinite variance
def test_clt_failure_cauchy(n_values=[10, 100, 1000, 10000]):
    """Demonstrate CLT failure for Cauchy distribution"""
    cauchy_dist = stats.cauchy(loc=0, scale=1)
    n_simulations = 1000

    print("1. Cauchy Distribution (infinite variance):")
    for n in n_values:
        sample_means = [cauchy_dist.rvs(size=n).mean()
                       for _ in range(n_simulations)]
        sample_means = np.array(sample_means)

        # Sample mean std should decrease as 1/sqrt(n) if CLT holds
        # For Cauchy, it doesn't
        print(f"  n={n:5d}: Std of sample means = {sample_means.std():.4f}")

    print("  --> Std does NOT decrease with n (CLT fails)\n")

test_clt_failure_cauchy()

# 2. Strong dependencies violate independence assumption
def test_clt_with_autocorrelation():
    """Demonstrate CLT issues with autocorrelated data"""
    n = 1000
    n_simulations = 1000

    # Generate AR(1) process with strong autocorrelation
    rho = 0.9  # High autocorrelation

    sample_means_ar = []
    for _ in range(n_simulations):
        # AR(1): X_t = rho * X_{t-1} + epsilon_t
        epsilon = np.random.normal(0, 1, n)
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = rho * x[t-1] + epsilon[t]
        sample_means_ar.append(x.mean())

    sample_means_ar = np.array(sample_means_ar)

    # Compare to independent data
    sample_means_ind = [np.random.normal(0, 1, n).mean()
                       for _ in range(n_simulations)]
    sample_means_ind = np.array(sample_means_ind)

    print("2. Autocorrelated Data (AR(1) with rho=0.9):")
    print(f"  Std with independence: {sample_means_ind.std():.6f}")
    print(f"  Std with autocorrelation: {sample_means_ar.std():.6f}")
    print(f"  Ratio: {sample_means_ar.std() / sample_means_ind.std():.2f}")
    print("  --> Variance inflated by autocorrelation\n")

test_clt_with_autocorrelation()

# 3. Very small sample sizes with highly skewed distributions
def test_clt_small_n_skewed():
    """Demonstrate slow CLT convergence for skewed distributions"""
    # Highly skewed exponential
    exp_dist = stats.expon(scale=1)
    n_simulations = 10000

    print("3. Small n with Highly Skewed Distribution:")
    for n in [3, 5, 10, 30]:
        sample_means = [exp_dist.rvs(size=n).mean()
                       for _ in range(n_simulations)]

        # Test normality with Shapiro-Wilk
        stat, p_value = stats.shapiro(sample_means[:5000])

        # Also check skewness
        from scipy.stats import skew
        skewness = skew(sample_means)

        print(f"  n={n:2d}: SW p-value={p_value:.4f}, Skewness={skewness:.4f}")

    print("  --> Normality improves with larger n\n")

test_clt_small_n_skewed()

# 4. Finite population correction needed
def test_finite_population_correction():
    """Demonstrate when finite population correction is needed"""
    # Small population
    population = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    N = len(population)
    pop_mean = population.mean()
    pop_std = population.std()

    n = 5  # Sample size (50% of population)
    n_simulations = 10000

    # Sampling without replacement
    sample_means = []
    for _ in range(n_simulations):
        sample = np.random.choice(population, size=n, replace=False)
        sample_means.append(sample.mean())

    sample_means = np.array(sample_means)

    # Standard error without FPC
    se_without_fpc = pop_std / np.sqrt(n)

    # Standard error with FPC
    fpc = np.sqrt((N - n) / (N - 1))
    se_with_fpc = se_without_fpc * fpc

    # Empirical standard error
    se_empirical = sample_means.std()

    print("4. Finite Population Correction:")
    print(f"  Population size N={N}, Sample size n={n}")
    print(f"  SE without FPC: {se_without_fpc:.4f}")
    print(f"  SE with FPC: {se_with_fpc:.4f}")
    print(f"  Empirical SE: {se_empirical:.4f}")
    print(f"  --> FPC needed when n/N > 0.05")

test_finite_population_correction()
```

---

## 6. Multivariate Distributions

### 6.1 Multivariate Normal Distribution

The multivariate normal (Gaussian) distribution is the extension of the normal distribution to multiple dimensions.

**Parameters**:
- mu: mean vector (d-dimensional)
- Sigma: covariance matrix (d x d, positive semi-definite)

**PDF**: f(x) = (1 / sqrt((2*pi)^d * det(Sigma))) * exp(-0.5 * (x - mu)^T * Sigma^(-1) * (x - mu))

**Properties**:
- Marginal distributions are normal
- Conditional distributions are normal
- Linear combinations are normal
- Uncorrelated implies independent (special property)

```python
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# Bivariate normal distribution
mean = np.array([0, 0])
cov = np.array([[1, 0.7],
                [0.7, 1]])

dist = multivariate_normal(mean, cov)

# PDF evaluation
x_test = np.array([0.5, 0.5])
print(f"PDF at {x_test}: {dist.pdf(x_test):.6f}")

# Generate samples
samples = dist.rvs(size=1000, random_state=42)
print(f"Sample shape: {samples.shape}")
print(f"Sample mean: {samples.mean(axis=0)}")
print(f"Sample covariance:\n{np.cov(samples.T)}")

# Visualize 2D distribution
x = np.linspace(-3, 3, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = dist.pdf(pos)

plt.figure(figsize=(12, 5))

# Contour plot
plt.subplot(1, 2, 1)
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Density')
plt.scatter(samples[:100, 0], samples[:100, 1], alpha=0.3, c='red', s=10)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Bivariate Normal PDF')
plt.axis('equal')

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax = plt.subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Density')
ax.set_title('Bivariate Normal PDF (3D)')

plt.tight_layout()
plt.show()
```

### 6.2 Covariance Matrix

The **covariance matrix** Sigma encodes all pairwise covariances between variables.

**Structure**:
- Diagonal elements: variances Sigma_ii = Var(X_i)
- Off-diagonal elements: covariances Sigma_ij = Cov(X_i, X_j)
- Symmetric: Sigma = Sigma^T
- Positive semi-definite: x^T * Sigma * x >= 0 for all x

```python
# Create covariance matrix from correlation matrix
def corr_to_cov(corr_matrix, std_devs):
    """Convert correlation matrix to covariance matrix"""
    D = np.diag(std_devs)
    return D @ corr_matrix @ D

# Example: 3D multivariate normal
corr = np.array([[1.0, 0.5, 0.3],
                 [0.5, 1.0, 0.4],
                 [0.3, 0.4, 1.0]])
std = np.array([2.0, 1.5, 1.0])
cov_matrix = corr_to_cov(corr, std)

print("Correlation matrix:")
print(corr)
print("\nCovariance matrix:")
print(cov_matrix)

# Properties of covariance matrix
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"\nEigenvalues: {eigenvalues}")
print(f"All positive (positive definite): {np.all(eigenvalues > 0)}")

# Determinant
det_cov = np.linalg.det(cov_matrix)
print(f"Determinant: {det_cov:.4f}")

# Generate samples
mean_3d = np.array([0, 0, 0])
dist_3d = multivariate_normal(mean_3d, cov_matrix)
samples_3d = dist_3d.rvs(size=5000, random_state=42)

# Verify empirical covariance
empirical_cov = np.cov(samples_3d.T)
print(f"\nEmpirical covariance matrix:")
print(empirical_cov)

# Partial correlation (conditioning)
# P(X1 | X3) for bivariate normal
def conditional_distribution(mean, cov, condition_idx, condition_value):
    """Compute conditional distribution parameters"""
    # Partition into conditioned and conditioning variables
    n = len(mean)
    idx_cond = [condition_idx]
    idx_other = [i for i in range(n) if i != condition_idx]

    mu_1 = mean[idx_other]
    mu_2 = mean[idx_cond]

    Sigma_11 = cov[np.ix_(idx_other, idx_other)]
    Sigma_12 = cov[np.ix_(idx_other, idx_cond)]
    Sigma_22 = cov[np.ix_(idx_cond, idx_cond)]

    # Conditional mean and covariance
    mu_cond = mu_1 + Sigma_12 @ np.linalg.inv(Sigma_22) @ (condition_value - mu_2)
    Sigma_cond = Sigma_11 - Sigma_12 @ np.linalg.inv(Sigma_22) @ Sigma_12.T

    return mu_cond, Sigma_cond

# Example: condition on X3 = 1.5
mu_cond, Sigma_cond = conditional_distribution(mean_3d, cov_matrix, 2, np.array([1.5]))
print(f"\nConditional mean given X3=1.5: {mu_cond}")
print(f"Conditional covariance: {Sigma_cond}")
```

### 6.3 Mahalanobis Distance

The **Mahalanobis distance** measures distance from a point to a distribution, accounting for correlations.

**Formula**: D_M(x) = sqrt((x - mu)^T * Sigma^(-1) * (x - mu))

**Properties**:
- Scale-invariant
- Accounts for correlation structure
- For standard normal: reduces to Euclidean distance
- D_M^2 follows chi-squared(d) distribution

```python
from scipy.spatial.distance import mahalanobis

# 2D example with correlation
mean = np.array([2, 3])
cov = np.array([[2, 1.5],
                [1.5, 3]])

dist = multivariate_normal(mean, cov)

# Test points
points = np.array([
    [2, 3],      # At mean
    [3, 4],      # Nearby
    [5, 7],      # Further away
    [0, 0]       # Far away
])

# Compute Mahalanobis distance
cov_inv = np.linalg.inv(cov)
for point in points:
    d_mahal = mahalanobis(point, mean, cov_inv)
    d_euclidean = np.linalg.norm(point - mean)
    print(f"Point {point}: Mahalanobis={d_mahal:.4f}, Euclidean={d_euclidean:.4f}")

# Outlier detection using Mahalanobis distance
samples = dist.rvs(size=1000, random_state=42)
distances = np.array([mahalanobis(x, mean, cov_inv) for x in samples])

# Chi-squared critical value for 95% (df=2)
from scipy.stats import chi2
threshold = np.sqrt(chi2.ppf(0.95, df=2))
outliers = distances > threshold
print(f"\nOutliers detected: {outliers.sum()} / {len(samples)}")
print(f"Expected (5%): {0.05 * len(samples):.0f}")

# Visualize
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(samples[~outliers, 0], samples[~outliers, 1], alpha=0.5, label='Normal')
plt.scatter(samples[outliers, 0], samples[outliers, 1], c='red', alpha=0.8, label='Outlier')
plt.scatter(*mean, c='green', s=200, marker='*', label='Mean', zorder=5)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Outlier Detection via Mahalanobis Distance')
plt.legend()
plt.axis('equal')

plt.subplot(1, 2, 2)
plt.hist(distances, bins=50, density=True, alpha=0.7, edgecolor='black')
x_range = np.linspace(0, distances.max(), 100)
plt.plot(x_range, chi2(df=2).pdf(x_range**2) * 2 * x_range, 'r-', linewidth=2, label='Chi-squared(2)')
plt.axvline(threshold, color='green', linestyle='--', linewidth=2, label='95% threshold')
plt.xlabel('Mahalanobis Distance')
plt.ylabel('Density')
plt.title('Distribution of Mahalanobis Distances')
plt.legend()

plt.tight_layout()
plt.show()
```

### 6.4 Applications in Machine Learning

```python
# 1. Anomaly detection with multivariate normal
from sklearn.covariance import EmpiricalCovariance, MinCovDet

# Generate normal data with some outliers
np.random.seed(42)
n_samples, n_features = 500, 2
X_normal = np.random.randn(n_samples, n_features)
X_normal = X_normal @ np.array([[2, 0.5], [0.5, 1]])  # Add correlation

# Add outliers
n_outliers = 50
X_outliers = np.random.uniform(low=-6, high=6, size=(n_outliers, n_features))
X = np.vstack([X_normal, X_outliers])

# Fit empirical covariance
emp_cov = EmpiricalCovariance()
emp_cov.fit(X_normal)

# Compute Mahalanobis distances
distances = emp_cov.mahalanobis(X)
threshold = chi2.ppf(0.99, df=n_features)
is_outlier = distances > threshold

print(f"Detected outliers: {is_outlier.sum()}")
print(f"True outliers: {n_outliers}")

# 2. Whitening transformation (decorrelation)
def whiten_data(X):
    """Apply ZCA whitening to decorrelate data"""
    # Center data
    X_centered = X - X.mean(axis=0)

    # Compute covariance
    cov = np.cov(X_centered.T)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Whitening matrix
    W = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues + 1e-5)) @ eigenvectors.T

    # Apply whitening
    X_whitened = X_centered @ W

    return X_whitened, W

X_whitened, W = whiten_data(X_normal)
print(f"\nWhitened data covariance:")
print(np.cov(X_whitened.T))
print(f"Should be identity matrix")

# 3. Dimensionality reduction with PCA
from sklearn.decomposition import PCA

# High-dimensional multivariate normal
n_dim = 10
mean_high = np.zeros(n_dim)
cov_high = np.eye(n_dim)
# Add structure
for i in range(n_dim-1):
    cov_high[i, i+1] = 0.5
    cov_high[i+1, i] = 0.5

dist_high = multivariate_normal(mean_high, cov_high)
samples_high = dist_high.rvs(size=1000, random_state=42)

# Apply PCA
pca = PCA(n_components=3)
samples_reduced = pca.fit_transform(samples_high)

print(f"\nPCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Cumulative variance: {pca.explained_variance_ratio_.cumsum()}")
```

---

## 7. Mixture Models

### 7.1 Gaussian Mixture Models (GMM)

A **mixture model** represents a distribution as a weighted sum of component distributions.

**GMM Definition**: p(x) = sum_{k=1}^K pi_k * N(x | mu_k, Sigma_k)

where:
- K: number of components
- pi_k: mixture weights (sum to 1)
- mu_k, Sigma_k: component parameters

```python
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Generate synthetic mixture data
np.random.seed(42)
n_samples = 1000

# Component 1: N([0, 0], I)
X1 = np.random.randn(n_samples // 3, 2) * 0.5 + np.array([0, 0])

# Component 2: N([3, 3], I)
X2 = np.random.randn(n_samples // 3, 2) * 0.8 + np.array([3, 3])

# Component 3: N([-2, 3], I)
X3 = np.random.randn(n_samples // 3, 2) * 0.6 + np.array([-2, 3])

X_mixture = np.vstack([X1, X2, X3])
true_labels = np.hstack([np.zeros(len(X1)), np.ones(len(X2)), 2*np.ones(len(X3))])

# Fit GMM with 3 components
gmm = GaussianMixture(n_components=3, random_state=42, covariance_type='full')
gmm.fit(X_mixture)

# Predict cluster assignments
predicted_labels = gmm.predict(X_mixture)
probabilities = gmm.predict_proba(X_mixture)

print("GMM Parameters:")
print(f"Weights: {gmm.weights_}")
print(f"Means:\n{gmm.means_}")
print(f"Converged: {gmm.converged_}")
print(f"Number of iterations: {gmm.n_iter_}")

# Log-likelihood
log_likelihood = gmm.score(X_mixture)
print(f"\nLog-likelihood: {log_likelihood:.4f}")
print(f"BIC: {gmm.bic(X_mixture):.4f}")
print(f"AIC: {gmm.aic(X_mixture):.4f}")

# Visualize results
plt.figure(figsize=(15, 5))

# Original data with true labels
plt.subplot(1, 3, 1)
plt.scatter(X_mixture[:, 0], X_mixture[:, 1], c=true_labels, cmap='viridis', alpha=0.6)
plt.title('True Clusters')
plt.xlabel('X1')
plt.ylabel('X2')

# Predicted labels
plt.subplot(1, 3, 2)
plt.scatter(X_mixture[:, 0], X_mixture[:, 1], c=predicted_labels, cmap='viridis', alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', s=300, marker='X',
            edgecolors='black', linewidth=2, label='Centroids')
plt.title('GMM Predictions')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()

# Probability contours
plt.subplot(1, 3, 3)
x_min, x_max = X_mixture[:, 0].min() - 1, X_mixture[:, 0].max() + 1
y_min, y_max = X_mixture[:, 1].min() - 1, X_mixture[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
Z = -gmm.score_samples(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=10, linewidths=2)
plt.scatter(X_mixture[:, 0], X_mixture[:, 1], c=predicted_labels, cmap='viridis', alpha=0.4)
plt.title('GMM Decision Boundaries')
plt.xlabel('X1')
plt.ylabel('X2')

plt.tight_layout()
plt.show()
```

### 7.2 Expectation-Maximization (EM) Algorithm

The **EM algorithm** iteratively finds MLE for mixture models with latent variables.

**E-step**: Compute responsibilities (posterior probabilities of component membership)

**M-step**: Update parameters using weighted MLE

```python
# Manual implementation of EM for GMM
class SimpleGMM:
    def __init__(self, n_components, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize(self, X):
        """Initialize parameters randomly"""
        n_samples, n_features = X.shape

        # Random initialization of means
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means = X[random_idx]

        # Initialize covariances as identity
        self.covariances = np.array([np.eye(n_features) for _ in range(self.n_components)])

        # Equal weights
        self.weights = np.ones(self.n_components) / self.n_components

    def e_step(self, X):
        """E-step: compute responsibilities"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            dist = multivariate_normal(self.means[k], self.covariances[k])
            responsibilities[:, k] = self.weights[k] * dist.pdf(X)

        # Normalize
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        """M-step: update parameters"""
        n_samples, n_features = X.shape

        for k in range(self.n_components):
            # Effective number of points in cluster
            Nk = responsibilities[:, k].sum()

            # Update weights
            self.weights[k] = Nk / n_samples

            # Update means
            self.means[k] = (responsibilities[:, k][:, np.newaxis] * X).sum(axis=0) / Nk

            # Update covariances
            diff = X - self.means[k]
            self.covariances[k] = (responsibilities[:, k][:, np.newaxis, np.newaxis] *
                                   diff[:, :, np.newaxis] * diff[:, np.newaxis, :]).sum(axis=0) / Nk

            # Add small regularization
            self.covariances[k] += np.eye(n_features) * 1e-6

    def log_likelihood(self, X):
        """Compute log-likelihood"""
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            dist = multivariate_normal(self.means[k], self.covariances[k])
            likelihood += self.weights[k] * dist.pdf(X)

        return np.log(likelihood).sum()

    def fit(self, X):
        """Fit GMM using EM algorithm"""
        self.initialize(X)

        prev_log_likelihood = -np.inf
        self.log_likelihoods = []

        for iteration in range(self.max_iter):
            # E-step
            responsibilities = self.e_step(X)

            # M-step
            self.m_step(X, responsibilities)

            # Compute log-likelihood
            log_likelihood = self.log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)

            # Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_log_likelihood = log_likelihood

        return self

# Test custom GMM
gmm_custom = SimpleGMM(n_components=3, max_iter=100)
gmm_custom.fit(X_mixture)

print("\nCustom GMM Results:")
print(f"Final weights: {gmm_custom.weights}")
print(f"Final means:\n{gmm_custom.means}")

# Plot convergence
plt.figure(figsize=(10, 4))
plt.plot(gmm_custom.log_likelihoods, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Log-Likelihood')
plt.title('EM Algorithm Convergence')
plt.grid(alpha=0.3)
plt.show()
```

### 7.3 Model Selection with BIC

**Bayesian Information Criterion (BIC)** balances fit quality and model complexity.

**Formula**: BIC = -2 * log_likelihood + k * log(n)

where k is number of parameters, n is sample size

**Lower BIC is better** (penalizes complexity more than AIC)

```python
# Compare GMMs with different numbers of components
n_components_range = range(1, 8)
bic_scores = []
aic_scores = []
models = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42,
                          covariance_type='full', max_iter=200)
    gmm.fit(X_mixture)

    bic_scores.append(gmm.bic(X_mixture))
    aic_scores.append(gmm.aic(X_mixture))
    models.append(gmm)

# Find optimal number of components
optimal_components_bic = n_components_range[np.argmin(bic_scores)]
optimal_components_aic = n_components_range[np.argmin(aic_scores)]

print(f"Optimal components (BIC): {optimal_components_bic}")
print(f"Optimal components (AIC): {optimal_components_aic}")

# Visualize model selection
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(n_components_range, bic_scores, 'o-', linewidth=2, markersize=8)
plt.axvline(optimal_components_bic, color='r', linestyle='--', label=f'Optimal ({optimal_components_bic})')
plt.xlabel('Number of Components')
plt.ylabel('BIC')
plt.title('BIC vs Number of Components')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(n_components_range, aic_scores, 'o-', linewidth=2, markersize=8, color='orange')
plt.axvline(optimal_components_aic, color='r', linestyle='--', label=f'Optimal ({optimal_components_aic})')
plt.xlabel('Number of Components')
plt.ylabel('AIC')
plt.title('AIC vs Number of Components')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Different covariance types
covariance_types = ['full', 'tied', 'diag', 'spherical']
results = {}

for cov_type in covariance_types:
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    gmm.fit(X_mixture)
    results[cov_type] = {
        'BIC': gmm.bic(X_mixture),
        'AIC': gmm.aic(X_mixture),
        'n_params': gmm.n_parameters
    }

print("\nCovariance Type Comparison:")
for cov_type, metrics in results.items():
    print(f"{cov_type:10s}: BIC={metrics['BIC']:8.2f}, AIC={metrics['AIC']:8.2f}, params={metrics['n_params']}")
```

---

## 8. Distribution Selection Guide

### 8.1 Decision Framework

```python
# Automated distribution suggestion based on data characteristics
def suggest_distribution(data):
    """Suggest appropriate distributions based on data properties"""
    data = np.array(data)
    suggestions = []

    # Basic statistics
    min_val, max_val = data.min(), data.max()
    mean_val, var_val = data.mean(), data.var()
    is_discrete = np.all(data == data.astype(int))
    is_binary = np.all(np.isin(data, [0, 1]))
    is_positive = np.all(data >= 0)
    is_bounded_01 = np.all((data >= 0) & (data <= 1))

    print("Data Characteristics:")
    print(f"  Range: [{min_val:.4f}, {max_val:.4f}]")
    print(f"  Mean: {mean_val:.4f}, Variance: {var_val:.4f}")
    print(f"  Discrete: {is_discrete}, Binary: {is_binary}")
    print(f"  Positive: {is_positive}, Bounded [0,1]: {is_bounded_01}")

    if is_binary:
        suggestions.append(("Bernoulli", "Binary outcomes (0/1)"))
    elif is_discrete and is_positive:
        var_mean_ratio = var_val / mean_val if mean_val > 0 else 0
        print(f"  Variance/Mean: {var_mean_ratio:.4f}")
        suggestions.append(("Poisson", "Count data, events in interval"))
        if var_mean_ratio > 1.5:
            suggestions.append(("Negative Binomial", "Overdispersed counts"))
    elif is_bounded_01 and not is_discrete:
        suggestions.append(("Beta", "Proportions on [0,1]"))
    elif is_positive and not is_discrete:
        from scipy.stats import skew
        skewness = skew(data)
        print(f"  Skewness: {skewness:.4f}")
        if abs(skewness) < 0.5:
            suggestions.append(("Gamma", "Positive, moderate skew"))
        else:
            suggestions.append(("Exponential/Log-Normal", "Right-skewed"))
    else:
        from scipy.stats import skew, kurtosis
        skewness, kurt = skew(data), kurtosis(data)
        print(f"  Skewness: {skewness:.4f}, Kurtosis: {kurt:.4f}")
        if abs(skewness) < 0.5 and abs(kurt) < 1:
            suggestions.append(("Normal", "Symmetric, bell-shaped"))
        elif abs(kurt) > 2:
            suggestions.append(("Student t", "Heavy tails"))

    print("\nSuggested Distributions:")
    for i, (name, reason) in enumerate(suggestions, 1):
        print(f"  {i}. {name:20s} - {reason}")
    return suggestions

# Test examples
count_data = np.random.poisson(5, 200)
suggest_distribution(count_data)
```

### 8.2 Common Use Cases

```python
distribution_guide = {
    "Binary outcomes": ["Bernoulli"],
    "Successes in n trials": ["Binomial"],
    "Events per interval": ["Poisson"],
    "Time between events": ["Exponential"],
    "Proportions/probabilities": ["Beta"],
    "Measurement errors": ["Normal"],
    "Income distribution": ["Log-Normal"],
    "Survival times": ["Exponential", "Weibull"],
    "Count overdispersion": ["Negative Binomial"],
    "Small sample inference": ["Student t"],
}

print("Distribution Use Case Guide:\n")
for use_case, dists in distribution_guide.items():
    print(f"{use_case:30s}: {', '.join(dists)}")
```

---

## 9. Bayesian Conjugate Priors

### 9.1 Beta-Binomial Conjugacy

**Prior**: Beta(alpha, beta)
**Likelihood**: Binomial(n, p)
**Posterior**: Beta(alpha + k, beta + n - k)

```python
from scipy.stats import beta, binom

def beta_binomial_update(prior_alpha, prior_beta, n_success, n_total):
    """Bayesian update for binomial proportion"""
    prior = beta(prior_alpha, prior_beta)
    post_alpha = prior_alpha + n_success
    post_beta = prior_beta + (n_total - n_success)
    posterior = beta(post_alpha, post_beta)

    p_mle = n_success / n_total
    p_bayes = posterior.mean()
    ci = posterior.ppf([0.025, 0.975])

    print(f"Prior: Beta({prior_alpha}, {prior_beta}), Mean={prior.mean():.4f}")
    print(f"Data: {n_success}/{n_total} successes")
    print(f"Posterior: Beta({post_alpha}, {post_beta}), Mean={p_bayes:.4f}")
    print(f"95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")
    print(f"MLE: {p_mle:.4f}")

    # Visualize
    p_range = np.linspace(0, 1, 200)
    plt.figure(figsize=(10, 5))
    plt.plot(p_range, prior.pdf(p_range), label='Prior', linewidth=2)
    plt.plot(p_range, posterior.pdf(p_range), label='Posterior', linewidth=2)
    plt.axvline(p_mle, color='red', linestyle='--', label=f'MLE ({p_mle:.3f})')
    plt.axvline(p_bayes, color='green', linestyle='--', label=f'Bayes ({p_bayes:.3f})')
    plt.xlabel('Probability p')
    plt.ylabel('Density')
    plt.title('Beta-Binomial Conjugacy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return posterior

# Example: coin flip inference
posterior = beta_binomial_update(prior_alpha=2, prior_beta=2, n_success=7, n_total=10)
```

### 9.2 Gamma-Poisson Conjugacy

**Prior**: Gamma(alpha, beta)
**Likelihood**: Poisson(lambda)
**Posterior**: Gamma(alpha + sum_x, beta + n)

```python
from scipy.stats import gamma, poisson

def gamma_poisson_update(prior_alpha, prior_beta, observations):
    """Bayesian update for Poisson rate"""
    prior = gamma(prior_alpha, scale=1/prior_beta)
    n = len(observations)
    total = observations.sum()
    post_alpha = prior_alpha + total
    post_beta = prior_beta + n
    posterior = gamma(post_alpha, scale=1/post_beta)

    lambda_mle = observations.mean()
    lambda_bayes = posterior.mean()
    ci = posterior.ppf([0.025, 0.975])

    print(f"Prior: Gamma({prior_alpha}, {prior_beta}), Mean={prior.mean():.4f}")
    print(f"Data: n={n}, sum={total}, mean={lambda_mle:.4f}")
    print(f"Posterior: Gamma({post_alpha}, {post_beta}), Mean={lambda_bayes:.4f}")
    print(f"95% Credible Interval: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return posterior

# Example: event counts
visit_counts = np.random.poisson(8, 20)
posterior_gamma = gamma_poisson_update(prior_alpha=2, prior_beta=0.2, observations=visit_counts)
```

### 9.3 Conjugate Prior Reference

```python
conjugate_table = {
    "Likelihood": ["Binomial", "Poisson", "Normal (mean)", "Exponential"],
    "Parameter": ["p", "lambda", "mu", "lambda"],
    "Conjugate Prior": ["Beta", "Gamma", "Normal", "Gamma"],
    "Posterior": ["Beta", "Gamma", "Normal", "Gamma"],
}

import pandas as pd
print("\nConjugate Prior Pairs:\n")
print(pd.DataFrame(conjugate_table).to_string(index=False))
```

---

## 10. Resources and References

### 10.1 Essential References

```python
references = {
    "Textbooks": [
        "All of Statistics - Larry Wasserman",
        "Statistical Inference - Casella & Berger",
        "Bayesian Data Analysis - Gelman et al.",
    ],
    "Online": [
        "Seeing Theory - https://seeing-theory.brown.edu/",
        "SciPy Stats Docs - https://docs.scipy.org/doc/scipy/reference/stats.html",
    ],
    "Libraries": [
        "scipy.stats: Distributions and tests",
        "sklearn.mixture: Gaussian Mixture Models",
        "statsmodels: Statistical modeling",
    ],
}

print("RESOURCES:\n")
for category, items in references.items():
    print(f"{category}:")
    for item in items:
        print(f"  - {item}")
    print()
```

### 10.2 Production Best Practices

```python
best_practices = """
PRODUCTION BEST PRACTICES:

1. Distribution Fitting:
   - Visualize data (histogram, Q-Q plot) before fitting
   - Test multiple candidates, use AIC/BIC for selection
   - Validate with goodness-of-fit tests (KS, Anderson-Darling)
   - Consider domain knowledge over pure statistics

2. Parameter Estimation:
   - Use MLE for large samples
   - Consider Bayesian methods for small samples
   - Bootstrap confidence intervals when needed
   - Check parameter identifiability

3. Outlier Detection:
   - Use robust methods (Mahalanobis, GMM)
   - Set thresholds based on domain expertise
   - Investigate before removing
   - Document decisions

4. Code Quality:
   - Use scipy.stats for standard distributions
   - Validate inputs (NaNs, infinities)
   - Add type hints and docstrings
   - Write unit tests

5. Performance:
   - Cache distribution objects
   - Use vectorized operations
   - Profile before optimizing
"""

print(best_practices)
```

---

