# 1. Probability Theory and Distributions

## 1.1 Fundamental Concepts

### Probability Axioms

- **Non-negativity**: P(A) ≥ 0 for any event A
- **Normalization**: P(Ω) = 1 where Ω is the sample space
- **Countable Additivity**: For mutually exclusive events A₁, A₂, ...: P(∪Aᵢ) = ΣP(Aᵢ)

### Random Variables

**Discrete Random Variables:** Take countable values (e.g., coin flips, dice rolls)

**Continuous Random Variables:** Take values from continuous intervals (e.g., height, temperature)

### Joint, Marginal, and Conditional Probability

**Joint Probability:** P(A, B) - probability of both events occurring

**Marginal Probability:** P(A) = ΣP(A, B) for all B

**Conditional Probability:** P(A|B) = P(A, B) / P(B)

---

## 1.2 Common Probability Distributions

### Discrete Distributions

#### Bernoulli Distribution

Models binary outcomes (success/failure, 0/1)

**Parameters:**
- p: probability of success

**PMF:** P(X = x) = p^x (1-p)^(1-x) for x ∈ {0, 1}

**Statistics:**
- Mean: p
- Variance: p(1-p)

**Use Cases:**
- Single coin flip
- Binary classification
- A/B test conversion

```python
import numpy as np
from scipy.stats import bernoulli

# Create Bernoulli distribution
p = 0.7
dist = bernoulli(p)

# Sample
samples = dist.rvs(size=1000)

# PMF
print(f"P(X=1) = {dist.pmf(1)}")  # 0.7
print(f"P(X=0) = {dist.pmf(0)}")  # 0.3
```

---

#### Binomial Distribution

Models number of successes in n independent Bernoulli trials

**Parameters:**
- n: number of trials
- p: probability of success

**PMF:** P(X = k) = C(n,k) × p^k × (1-p)^(n-k)

**Statistics:**
- Mean: np
- Variance: np(1-p)

**Use Cases:**
- Free throw shooting percentage
- Quality control
- Click-through rates

```python
from scipy.stats import binom

# 10 trials, p=0.3
n, p = 10, 0.3
dist = binom(n, p)

# Probability of exactly 3 successes
print(f"P(X=3) = {dist.pmf(3):.4f}")

# Probability of at most 2 successes
print(f"P(X≤2) = {dist.cdf(2):.4f}")

# Expected value
print(f"E[X] = {dist.mean()}")  # 3.0
```

---

#### Poisson Distribution

Models count of events in fixed interval

**Parameter:**
- λ: rate parameter

**PMF:** P(X = k) = (λ^k × e^(-λ)) / k!

**Statistics:**
- Mean: λ
- Variance: λ

**Use Cases:**
- Website traffic
- Customer arrivals
- Rare events

```python
from scipy.stats import poisson

# Average rate of 5 events per hour
lam = 5
dist = poisson(lam)

# Probability of exactly 3 events
print(f"P(X=3) = {dist.pmf(3):.4f}")

# Probability of more than 7 events
print(f"P(X>7) = {1 - dist.cdf(7):.4f}")
```

---

### Continuous Distributions

#### Normal (Gaussian) Distribution

Most important distribution in statistics

**Parameters:**
- μ: mean
- σ²: variance

**PDF:** f(x) = (1/(σ√(2π))) × exp(-(x-μ)²/(2σ²))

**Central Limit Theorem:** Sum of independent random variables approaches normal distribution

**Use Cases:**
- Measurement errors
- Heights, weights
- Test scores
- Feature distributions in ML

```python
from scipy.stats import norm
import matplotlib.pyplot as plt

# Standard normal (μ=0, σ=1)
dist = norm(loc=0, scale=1)

# PDF at x=1
print(f"f(1) = {dist.pdf(1):.4f}")

# CDF (area under curve to left of x=1)
print(f"P(X≤1) = {dist.cdf(1):.4f}")

# 95% confidence interval
lower, upper = dist.ppf(0.025), dist.ppf(0.975)
print(f"95% CI: [{lower:.2f}, {upper:.2f}]")

# Visualization
x = np.linspace(-4, 4, 1000)
plt.plot(x, dist.pdf(x))
plt.title('Standard Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.show()
```

---

#### Uniform Distribution

Equal probability for all values in interval [a, b]

**PDF:** f(x) = 1/(b-a) for a ≤ x ≤ b

**Statistics:**
- Mean: (a+b)/2
- Variance: (b-a)²/12

**Use Cases:**
- Random sampling
- Simulation
- Initialization

```python
from scipy.stats import uniform

# Uniform on [0, 1]
dist = uniform(loc=0, scale=1)

# Sample
samples = dist.rvs(size=1000)

# PDF is constant
print(f"f(0.5) = {dist.pdf(0.5)}")  # 1.0
```

---

#### Exponential Distribution

Models time between events in Poisson process

**Parameter:**
- λ: rate

**PDF:** f(x) = λe^(-λx) for x ≥ 0

**Statistics:**
- Mean: 1/λ
- Variance: 1/λ²

**Memoryless Property:** P(X > s+t | X > s) = P(X > t)

**Use Cases:**
- Waiting times
- Survival analysis
- Time to failure

```python
from scipy.stats import expon

# Mean time = 5 minutes (λ = 1/5 = 0.2)
mean_time = 5
dist = expon(scale=mean_time)

# Probability wait time > 10 minutes
print(f"P(X>10) = {1 - dist.cdf(10):.4f}")

# Median wait time
print(f"Median = {dist.median():.2f} minutes")
```

---

#### Beta Distribution

Continuous distribution on [0, 1]

**Parameters:**
- α, β: shape parameters

**Use Cases:**
- Bayesian prior for probabilities
- Proportions
- Success rates

```python
from scipy.stats import beta

# Beta(2, 5) - skewed left
dist = beta(a=2, b=5)

# Mean probability
print(f"E[X] = {dist.mean():.3f}")  # 2/(2+5) = 0.286

# Used as Bayesian prior
# If prior is Beta(α, β) and observe k successes in n trials,
# posterior is Beta(α+k, β+n-k)
```

---

## 1.3 Transformations of Random Variables

### Law of the Unconscious Statistician (LOTUS)

If Y = g(X), then:
```
E[Y] = E[g(X)] = ∫ g(x) f_X(x) dx
```

### Change of Variables

If Y = g(X) where g is monotonic and differentiable:
```
f_Y(y) = f_X(g⁻¹(y)) |dg⁻¹/dy|
```

---

## 1.4 Multivariate Distributions

### Joint Distribution

For random variables X and Y:
```
f(x, y) = joint PDF
```

### Marginal Distribution

```
f_X(x) = ∫ f(x, y) dy
f_Y(y) = ∫ f(x, y) dx
```

### Conditional Distribution

```
f(x|y) = f(x, y) / f_Y(y)
```

### Independence

X and Y are independent if:
```
f(x, y) = f_X(x) × f_Y(y)
```

---

## 1.5 Expectation and Moments

### Expected Value

**Discrete:** E[X] = Σ x P(X = x)

**Continuous:** E[X] = ∫ x f(x) dx

### Variance

```
Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²
```

### Covariance and Correlation

**Covariance:**
```
Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = E[XY] - E[X]E[Y]
```

**Correlation:**
```
ρ(X, Y) = Cov(X, Y) / (σ_X σ_Y)
```

Range: [-1, 1]
- ρ = 1: perfect positive linear relationship
- ρ = 0: no linear relationship
- ρ = -1: perfect negative linear relationship

```python
import numpy as np

# Sample data
X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])

# Covariance
cov_matrix = np.cov(X, Y)
print(f"Cov(X,Y) = {cov_matrix[0, 1]:.2f}")

# Correlation
corr = np.corrcoef(X, Y)[0, 1]
print(f"ρ(X,Y) = {corr:.3f}")
```

---

## Resources

- **Books:**
  - "All of Statistics" by Larry Wasserman
  - "Probability and Statistics" by DeGroot & Schervish
  - "Introduction to Probability" by Blitzstein & Hwang

- **Online:**
  - Khan Academy: Probability and Statistics
  - Seeing Theory: https://seeing-theory.brown.edu/
