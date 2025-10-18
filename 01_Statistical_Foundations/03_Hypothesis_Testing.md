# 3. Hypothesis Testing

## 3.1 Hypothesis Testing Framework

### Components

**Null Hypothesis (H₀):** Default assumption (e.g., no effect, no difference)

**Alternative Hypothesis (H₁ or Hₐ):** What we want to prove

**Test Statistic:** Calculated from data

**P-value:** Probability of observing data as extreme as observed, assuming H₀ is true

**Significance Level (α):** Threshold for rejecting H₀ (commonly 0.05)

### Decision Rules

- **Reject H₀** if p-value < α
- **Fail to reject H₀** if p-value ≥ α

### Type I and Type II Errors

**Type I Error (False Positive):** Reject H₀ when it's true
- P(Type I) = α

**Type II Error (False Negative):** Fail to reject H₀ when it's false
- P(Type II) = β

**Statistical Power:** 1 - β
- Probability of correctly rejecting false H₀
- Typical target: 80%

```python
import numpy as np
from scipy import stats

# Illustrate Type I and Type II errors
np.random.seed(42)

# Scenario 1: H₀ is true (μ=10)
null_true_data = np.random.normal(loc=10, scale=2, size=100)
t_stat, p_value = stats.ttest_1samp(null_true_data, 10)
print(f"H₀ true, p-value: {p_value:.4f}, Reject H₀: {p_value < 0.05}")

# Scenario 2: H₁ is true (μ=11, but we test against 10)
alternative_true_data = np.random.normal(loc=11, scale=2, size=100)
t_stat, p_value = stats.ttest_1samp(alternative_true_data, 10)
print(f"H₁ true, p-value: {p_value:.4f}, Reject H₀: {p_value < 0.05}")
```

---

## 3.2 Common Hypothesis Tests

### T-Test

#### One-Sample t-test

Compare sample mean to known value.

**Hypotheses:**
- H₀: μ = μ₀
- H₁: μ ≠ μ₀ (two-tailed)

**Test statistic:**
```
t = (x̄ - μ₀) / (s/√n)
```

**Degrees of freedom:** n-1

```python
import numpy as np
from scipy import stats

# Test if mean height is 170 cm
np.random.seed(42)
heights = np.random.normal(loc=172, scale=5, size=30)

# One-sample t-test
t_stat, p_value = stats.ttest_1samp(heights, 170)

print(f"Sample mean: {np.mean(heights):.2f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Reject H₀ (μ=170) at α=0.05: {p_value < 0.05}")
```

---

#### Two-Sample Independent t-test

Compare means of two groups.

**Hypotheses:**
- H₀: μ₁ = μ₂
- H₁: μ₁ ≠ μ₂

**Assumptions:**
- Independence
- Normality
- Equal variances (or use Welch's t-test)

```python
import numpy as np
from scipy import stats

# A/B test: Compare conversion rates
np.random.seed(42)
control = np.random.normal(loc=10, scale=2, size=100)
treatment = np.random.normal(loc=10.5, scale=2, size=100)

# Two-sample t-test (equal variance assumed)
t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=True)

print(f"Control mean: {np.mean(control):.3f}")
print(f"Treatment mean: {np.mean(treatment):.3f}")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_value:.4f}")

# Welch's t-test (unequal variances)
t_stat_welch, p_value_welch = stats.ttest_ind(treatment, control, equal_var=False)
print(f"\nWelch's p-value: {p_value_welch:.4f}")

# Effect size (Cohen's d)
pooled_std = np.sqrt((np.var(control, ddof=1) + np.var(treatment, ddof=1)) / 2)
cohens_d = (np.mean(treatment) - np.mean(control)) / pooled_std
print(f"Cohen's d: {cohens_d:.3f}")
```

---

#### Paired t-test

Compare means of related samples.

**Hypotheses:**
- H₀: μ_diff = 0
- H₁: μ_diff ≠ 0

**Use Cases:**
- Before/after measurements
- Matched pairs

```python
import numpy as np
from scipy import stats

# Before/after study
np.random.seed(42)
before = np.random.normal(loc=120, scale=10, size=50)
# Treatment reduces blood pressure by 5 points on average
after = before - 5 + np.random.normal(loc=0, scale=3, size=50)

# Paired t-test
t_stat, p_value = stats.ttest_rel(after, before)

print(f"Mean before: {np.mean(before):.2f}")
print(f"Mean after: {np.mean(after):.2f}")
print(f"Mean difference: {np.mean(after - before):.2f}")
print(f"p-value: {p_value:.4f}")
print(f"Treatment effective: {p_value < 0.05}")
```

---

### Chi-Square Tests

#### Goodness of Fit Test

Does data match expected distribution?

**Hypotheses:**
- H₀: Data follows specified distribution
- H₁: Data does not follow specified distribution

**Test statistic:**
```
χ² = Σ((Observed - Expected)² / Expected)
```

```python
import numpy as np
from scipy import stats

# Test if die is fair
np.random.seed(42)
rolls = np.random.choice([1, 2, 3, 4, 5, 6], size=600, p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.2])

# Observed frequencies
observed = np.bincount(rolls)[1:]  # Counts for 1-6

# Expected (fair die)
expected = np.array([100, 100, 100, 100, 100, 100])

# Chi-square test
chi2_stat, p_value = stats.chisquare(observed, expected)

print(f"Observed: {observed}")
print(f"Expected: {expected}")
print(f"χ² statistic: {chi2_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Die is biased: {p_value < 0.05}")
```

---

#### Test of Independence

Are two categorical variables independent?

**Hypotheses:**
- H₀: Variables are independent
- H₁: Variables are dependent

```python
import numpy as np
from scipy import stats

# Contingency table: Gender vs Product Preference
#            Product A  Product B
# Male          30        20
# Female        25        35

contingency_table = np.array([[30, 20],
                               [25, 35]])

# Chi-square test of independence
chi2_stat, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print("Observed:")
print(contingency_table)
print("\nExpected:")
print(expected)
print(f"\nχ² statistic: {chi2_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"Gender and preference are dependent: {p_value < 0.05}")
```

---

### ANOVA (Analysis of Variance)

#### One-Way ANOVA

Compare means of 3+ groups.

**Hypotheses:**
- H₀: μ₁ = μ₂ = ... = μₖ
- H₁: At least one mean differs

**F-statistic:**
```
F = (Between-group variance) / (Within-group variance)
```

```python
import numpy as np
from scipy import stats

# Compare 3 teaching methods
np.random.seed(42)
method_a = np.random.normal(loc=75, scale=10, size=30)
method_b = np.random.normal(loc=80, scale=10, size=30)
method_c = np.random.normal(loc=78, scale=10, size=30)

# One-way ANOVA
f_stat, p_value = stats.f_oneway(method_a, method_b, method_c)

print(f"Method A mean: {np.mean(method_a):.2f}")
print(f"Method B mean: {np.mean(method_b):.2f}")
print(f"Method C mean: {np.mean(method_c):.2f}")
print(f"\nF-statistic: {f_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"At least one method differs: {p_value < 0.05}")
```

---

#### Post-hoc Tests

After finding significance in ANOVA, determine which groups differ.

**Tukey HSD (Honestly Significant Difference):**

```python
from scipy.stats import tukey_hsd
import numpy as np

# Use same data from above
result = tukey_hsd(method_a, method_b, method_c)

print("\nTukey HSD pairwise comparisons:")
print(result)
print(f"\nPairwise p-values:\n{result.pvalue}")

# Bonferroni correction (alternative)
# For k comparisons, use α/k
k_comparisons = 3  # A vs B, A vs C, B vs C
bonferroni_alpha = 0.05 / k_comparisons
print(f"\nBonferroni-corrected α: {bonferroni_alpha:.4f}")
```

---

### Non-Parametric Tests

When normality assumptions violated.

#### Mann-Whitney U Test

Non-parametric alternative to two-sample t-test.

```python
import numpy as np
from scipy import stats

# Heavily skewed data
np.random.seed(42)
group1 = np.random.exponential(scale=5, size=50)
group2 = np.random.exponential(scale=7, size=50)

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')

print(f"Group 1 median: {np.median(group1):.2f}")
print(f"Group 2 median: {np.median(group2):.2f}")
print(f"U-statistic: {u_stat:.3f}")
print(f"p-value: {p_value:.4f}")
```

---

#### Wilcoxon Signed-Rank Test

Non-parametric alternative to paired t-test.

```python
import numpy as np
from scipy import stats

# Before/after with non-normal differences
np.random.seed(42)
before = np.random.lognormal(mean=2, sigma=0.5, size=30)
after = before * 0.9 + np.random.lognormal(mean=0, sigma=0.2, size=30)

# Wilcoxon signed-rank test
w_stat, p_value = stats.wilcoxon(after, before)

print(f"Median before: {np.median(before):.2f}")
print(f"Median after: {np.median(after):.2f}")
print(f"W-statistic: {w_stat:.3f}")
print(f"p-value: {p_value:.4f}")
```

---

#### Kruskal-Wallis Test

Non-parametric alternative to one-way ANOVA.

```python
import numpy as np
from scipy import stats

# Non-normal groups
np.random.seed(42)
group1 = np.random.exponential(scale=3, size=30)
group2 = np.random.exponential(scale=4, size=30)
group3 = np.random.exponential(scale=5, size=30)

# Kruskal-Wallis test
h_stat, p_value = stats.kruskal(group1, group2, group3)

print(f"H-statistic: {h_stat:.3f}")
print(f"p-value: {p_value:.4f}")
print(f"At least one group differs: {p_value < 0.05}")
```

---

## 3.3 Multiple Testing Correction

### The Problem

Testing multiple hypotheses increases false discovery rate.

If α = 0.05 and we run 100 independent tests:
```
Expected false positives = 100 × 0.05 = 5
```

### Bonferroni Correction

**Adjusted α:** α/k for k tests

**Conservative:** Controls family-wise error rate (FWER)

```python
import numpy as np
from scipy import stats

# 10 hypothesis tests
np.random.seed(42)
p_values = []

for i in range(10):
    # All null hypotheses are true
    data = np.random.normal(loc=0, scale=1, size=30)
    _, p = stats.ttest_1samp(data, 0)
    p_values.append(p)

print("Original p-values:")
print(np.round(p_values, 4))

# Bonferroni correction
alpha = 0.05
bonferroni_alpha = alpha / len(p_values)

print(f"\nBonferroni α: {bonferroni_alpha:.4f}")
print(f"Significant (original α=0.05): {np.sum(np.array(p_values) < alpha)}")
print(f"Significant (Bonferroni): {np.sum(np.array(p_values) < bonferroni_alpha)}")
```

---

### False Discovery Rate (FDR)

**Benjamini-Hochberg procedure:**

Less conservative than Bonferroni, controls proportion of false discoveries.

```python
import numpy as np
from statsmodels.stats.multitest import multipletests

# Same p-values from above
p_values = np.array(p_values)

# FDR correction
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

print("\nOriginal p-values:")
print(np.round(p_values, 4))
print("\nFDR-adjusted p-values:")
print(np.round(p_adjusted, 4))
print(f"\nRejections (FDR): {np.sum(reject)}")
```

---

## 3.4 Effect Size

P-value tells if effect exists; effect size tells how large.

### Cohen's d

Standardized mean difference.

```
d = (μ₁ - μ₂) / σ_pooled
```

**Interpretation:**
- Small: d = 0.2
- Medium: d = 0.5
- Large: d = 0.8

```python
import numpy as np

def cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

# Example
np.random.seed(42)
control = np.random.normal(loc=10, scale=2, size=100)
treatment = np.random.normal(loc=11, scale=2, size=100)

d = cohens_d(treatment, control)
print(f"Cohen's d: {d:.3f}")
print(f"Effect size: {'Small' if d < 0.5 else 'Medium' if d < 0.8 else 'Large'}")
```

---

## 3.5 Power Analysis

Calculate required sample size to detect effect.

```python
from statsmodels.stats.power import ttest_power, tt_solve_power

# Given: effect size, α, desired power → find n
effect_size = 0.5  # Cohen's d
alpha = 0.05
power = 0.80

n = tt_solve_power(effect_size=effect_size, alpha=alpha, power=power, alternative='two-sided')

print(f"Required sample size per group: {int(np.ceil(n))}")

# Given: n, effect size, α → find power
power_achieved = ttest_power(effect_size=effect_size, nobs=50, alpha=alpha, alternative='two-sided')
print(f"Power with n=50: {power_achieved:.3f}")
```

---

## Resources

- **Classic:**
  - "Design and Analysis of Experiments" by Montgomery
  - "Statistical Methods for Research Workers" by Fisher

- **Modern:**
  - "Modern Statistics for Modern Biology" by Holmes & Huber
  - "Statistical Rethinking" by McElreath (Bayesian approach)

- **Online:**
  - Statsmodels documentation: https://www.statsmodels.org/
  - SciPy stats: https://docs.scipy.org/doc/scipy/reference/stats.html
