# 7. Statistical Tests: Complete Guide

## Overview

Statistical tests help you determine if differences or relationships in your data are real or just due to chance. Choosing the wrong test leads to invalid conclusions.

**This guide covers:**
- When to use each statistical test
- Test assumptions and requirements
- Common pitfalls and solutions
- A/B testing best practices
- Test selection flowcharts

---

## 7.1 Statistical Test Selection Flowchart

```
What are you comparing?

├─ TWO CATEGORICAL VARIABLES
│  └─ Chi-Square Test of Independence
│
├─ ONE CATEGORICAL, ONE NUMERICAL
│  ├─ 2 Groups?
│  │  ├─ Data normally distributed? → Independent t-test
│  │  └─ NOT normal or ordinal? → Mann-Whitney U test
│  │
│  └─ 3+ Groups?
│     ├─ Data normally distributed? → One-Way ANOVA
│     └─ NOT normal or ordinal? → Kruskal-Wallis test
│
├─ TWO NUMERICAL VARIABLES
│  ├─ Linear relationship? → Pearson Correlation
│  └─ Monotonic (not linear)? → Spearman Correlation
│
└─ BEFORE/AFTER MEASUREMENTS (same subjects)
   ├─ Data normally distributed? → Paired t-test
   └─ NOT normal? → Wilcoxon Signed-Rank test
```

---

## 7.2 Chi-Square Test

### What It Tests
Whether two categorical variables are independent or associated.

### ✅ When to Use

1. **Both variables are categorical**
   - Example: Gender (Male/Female) vs Product Preference (A/B/C)
   - Example: Treatment (Drug/Placebo) vs Outcome (Cured/Not Cured)

2. **Want to test independence**
   - "Is smoking status independent of lung cancer?"
   - "Is marketing channel associated with conversion?"

3. **Have contingency table data**
   - Counts or frequencies in each cell

### ❌ When NOT to Use

1. **Small sample sizes**
   - Expected count < 5 in more than 20% of cells
   - **Solution:** Use Fisher's Exact Test

2. **Continuous variables**
   - Chi-square requires categorical data
   - **Solution:** Bin continuous variables or use correlation tests

3. **Paired/dependent observations**
   - Example: Before/after on same subjects
   - **Solution:** McNemar's test

### Assumptions

1. ✅ **Independence**: Each observation is independent
2. ✅ **Expected frequencies ≥ 5**: In at least 80% of cells
3. ✅ **Mutually exclusive categories**: Each observation in only one cell
4. ✅ **Random sampling**: Data collected randomly

### Code Example

```python
import pandas as pd
from scipy.stats import chi2_contingency

# Create contingency table
contingency_table = pd.crosstab(df['gender'], df['product_preference'])
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")
print(f"\nExpected frequencies:\n{expected}")

# Check assumption: expected frequencies ≥ 5
if (expected < 5).sum() / expected.size > 0.2:
    print("⚠️ Warning: >20% of cells have expected count < 5")
    print("Consider using Fisher's Exact Test instead")

# Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Reject null hypothesis (p={p_value:.4f} < {alpha})")
    print("Variables are associated")
else:
    print(f"\n❌ Fail to reject null hypothesis (p={p_value:.4f} ≥ {alpha})")
    print("No evidence of association")

# Effect size: Cramér's V
import numpy as np
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
print(f"\nCramér's V (effect size): {cramers_v:.4f}")
# Interpretation: 0.1=small, 0.3=medium, 0.5=large
```

### Interpretation

- **p-value < 0.05**: Variables are associated (dependent)
- **p-value ≥ 0.05**: No evidence of association (independent)
- **Cramér's V**: Effect size (0=no association, 1=perfect association)

---

## 7.3 T-Tests

### Types of T-Tests

| Type | Use Case | Example |
|------|----------|---------|
| **One-Sample** | Compare sample mean to known value | "Is average height different from 170cm?" |
| **Independent** | Compare means of 2 independent groups | "Do males earn more than females?" |
| **Paired** | Compare means before/after (same subjects) | "Did treatment improve scores?" |

---

### 7.3.1 Independent T-Test

**What:** Compare means between two independent groups

### ✅ When to Use

1. **Comparing 2 groups only**
   - Male vs Female salaries
   - Treatment vs Control outcomes

2. **Continuous numerical data**
   - Test scores, heights, prices, etc.

3. **Groups are independent**
   - Different subjects in each group

### ❌ When NOT to Use

1. **More than 2 groups**
   - **Solution:** Use ANOVA

2. **Data not normally distributed**
   - **Solution:** Mann-Whitney U test

3. **Paired observations**
   - Example: Before/after on same people
   - **Solution:** Paired t-test

4. **Unequal variances**
   - **Solution:** Welch's t-test (default in Python)

### Assumptions

1. ✅ **Independence**: Observations are independent
2. ✅ **Normality**: Data in each group is normally distributed
   - Check with: Shapiro-Wilk test, Q-Q plot, histogram
   - Robust if n > 30 per group (Central Limit Theorem)
3. ✅ **Equal variances** (for standard t-test)
   - Check with: Levene's test
   - If violated, use Welch's t-test

### Code Example

```python
from scipy import stats
import numpy as np

# Sample data
group_a = np.random.normal(100, 15, 50)  # Mean=100
group_b = np.random.normal(110, 15, 50)  # Mean=110

# 1. Check normality assumption
_, p_norm_a = stats.shapiro(group_a)
_, p_norm_b = stats.shapiro(group_b)
print(f"Normality test - Group A: p={p_norm_a:.4f}")
print(f"Normality test - Group B: p={p_norm_b:.4f}")

if p_norm_a < 0.05 or p_norm_b < 0.05:
    print("⚠️ Data not normal, consider Mann-Whitney U test")

# 2. Check equal variances
_, p_levene = stats.levene(group_a, group_b)
print(f"\nLevene's test for equal variances: p={p_levene:.4f}")

# 3. Perform t-test
if p_levene < 0.05:
    # Welch's t-test (unequal variances)
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=False)
    print("Using Welch's t-test (unequal variances)")
else:
    # Standard t-test
    t_stat, p_value = stats.ttest_ind(group_a, group_b, equal_var=True)
    print("Using standard t-test (equal variances)")

print(f"\nt-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# 4. Effect size: Cohen's d
mean_a, mean_b = np.mean(group_a), np.mean(group_b)
std_pooled = np.sqrt((np.std(group_a, ddof=1)**2 + np.std(group_b, ddof=1)**2) / 2)
cohens_d = (mean_a - mean_b) / std_pooled
print(f"Cohen's d (effect size): {cohens_d:.4f}")
# Interpretation: 0.2=small, 0.5=medium, 0.8=large

# 5. Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Reject null hypothesis (p={p_value:.4f} < {alpha})")
    print(f"Group means are significantly different")
    print(f"Mean A: {mean_a:.2f}, Mean B: {mean_b:.2f}")
else:
    print(f"\n❌ Fail to reject null hypothesis (p={p_value:.4f} ≥ {alpha})")
    print("No significant difference between groups")
```

---

### 7.3.2 Paired T-Test

**What:** Compare means before/after on same subjects

### ✅ When to Use

1. **Before/after measurements**
   - Weight before/after diet
   - Test scores before/after training

2. **Matched pairs**
   - Twins, siblings, matched controls

3. **Repeated measures on same subjects**

### Code Example

```python
# Before/after data (same subjects)
before = np.array([120, 135, 142, 128, 130, 145, 138, 140])
after = np.array([115, 130, 138, 125, 127, 142, 135, 136])

# 1. Check normality of differences
differences = before - after
_, p_norm = stats.shapiro(differences)
print(f"Normality of differences: p={p_norm:.4f}")

if p_norm < 0.05:
    print("⚠️ Differences not normal, consider Wilcoxon test")

# 2. Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

print(f"\nt-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Mean difference: {np.mean(differences):.2f}")

# 3. Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Significant difference (p={p_value:.4f} < {alpha})")
    if np.mean(differences) > 0:
        print("Scores decreased after treatment")
    else:
        print("Scores increased after treatment")
else:
    print(f"\n❌ No significant difference (p={p_value:.4f} ≥ {alpha})")
```

---

## 7.4 ANOVA (Analysis of Variance)

### What It Tests
Whether means differ across 3 or more groups

### ✅ When to Use

1. **Comparing 3+ groups**
   - Drug A vs Drug B vs Drug C vs Placebo
   - Compare performance across 5 departments

2. **Continuous numerical outcome**

3. **Groups are independent**

### ❌ When NOT to Use

1. **Only 2 groups**
   - **Solution:** Use t-test (simpler)

2. **Data not normally distributed**
   - **Solution:** Kruskal-Wallis test

3. **Paired/repeated measures**
   - **Solution:** Repeated measures ANOVA

### Assumptions

1. ✅ **Independence**: Observations are independent
2. ✅ **Normality**: Each group is normally distributed
3. ✅ **Homogeneity of variance**: Groups have equal variances

### Code Example

```python
from scipy import stats

# Three groups
group1 = np.random.normal(100, 10, 30)
group2 = np.random.normal(105, 10, 30)
group3 = np.random.normal(110, 10, 30)

# 1. Check assumptions
# Normality (on each group)
_, p1 = stats.shapiro(group1)
_, p2 = stats.shapiro(group2)
_, p3 = stats.shapiro(group3)
print(f"Normality: Group1 p={p1:.3f}, Group2 p={p2:.3f}, Group3 p={p3:.3f}")

# Homogeneity of variance
_, p_levene = stats.levene(group1, group2, group3)
print(f"Levene's test: p={p_levene:.4f}")

if p_levene < 0.05:
    print("⚠️ Variances not equal, consider Welch's ANOVA or transformation")

# 2. Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(group1, group2, group3)

print(f"\nF-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# 3. Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Reject null hypothesis (p={p_value:.4f} < {alpha})")
    print("At least one group mean is different")
    print("\n⚠️ ANOVA doesn't tell you WHICH groups differ")
    print("→ Perform post-hoc tests (e.g., Tukey HSD)")
else:
    print(f"\n❌ Fail to reject null hypothesis (p={p_value:.4f} ≥ {alpha})")
    print("No significant difference between groups")

# 4. Post-hoc test: Tukey HSD (if ANOVA significant)
if p_value < alpha:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    # Combine data
    data = np.concatenate([group1, group2, group3])
    groups = ['Group1']*30 + ['Group2']*30 + ['Group3']*30

    tukey = pairwise_tukeyhsd(data, groups, alpha=0.05)
    print("\nTukey HSD post-hoc test:")
    print(tukey)
```

---

## 7.5 Non-Parametric Tests

**When to use:** Data violates normality assumption or is ordinal

---

### 7.5.1 Mann-Whitney U Test

**What:** Non-parametric alternative to independent t-test

### ✅ When to Use

1. **Comparing 2 independent groups**
2. **Data NOT normally distributed**
3. **Ordinal data** (ranked, but not interval)
4. **Small sample sizes**
5. **Outliers present**

### Code Example

```python
from scipy import stats

group_a = [23, 45, 67, 34, 89, 12, 56, 78]  # Non-normal
group_b = [34, 56, 78, 90, 45, 67, 23, 100]

# Mann-Whitney U test
u_stat, p_value = stats.mannwhitneyu(group_a, group_b, alternative='two-sided')

print(f"U-statistic: {u_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Significant difference (p={p_value:.4f} < {alpha})")
    print("Groups have different distributions")
else:
    print(f"\n❌ No significant difference (p={p_value:.4f} ≥ {alpha})")
```

**Advantages:**
- No normality assumption
- Robust to outliers
- Works with small samples

**Disadvantages:**
- Less powerful than t-test if data IS normal
- Tests distribution differences, not just means

---

### 7.5.2 Kruskal-Wallis Test

**What:** Non-parametric alternative to ANOVA

### ✅ When to Use

1. **Comparing 3+ groups**
2. **Data NOT normally distributed**
3. **Ordinal data**

### Code Example

```python
group1 = [23, 45, 67, 34, 89]
group2 = [34, 56, 78, 90, 45]
group3 = [12, 56, 78, 23, 100]

# Kruskal-Wallis test
h_stat, p_value = stats.kruskal(group1, group2, group3)

print(f"H-statistic: {h_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Significant difference (p={p_value:.4f} < {alpha})")
    print("At least one group is different")
    print("→ Perform post-hoc tests (e.g., Dunn's test)")
else:
    print(f"\n❌ No significant difference (p={p_value:.4f} ≥ {alpha})")
```

---

### 7.5.3 Wilcoxon Signed-Rank Test

**What:** Non-parametric alternative to paired t-test

### ✅ When to Use

1. **Paired/before-after data**
2. **Data NOT normally distributed**

### Code Example

```python
before = [23, 45, 67, 34, 89, 12, 56]
after = [25, 43, 70, 36, 85, 14, 58]

# Wilcoxon signed-rank test
w_stat, p_value = stats.wilcoxon(before, after)

print(f"W-statistic: {w_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret
alpha = 0.05
if p_value < alpha:
    print(f"\n✅ Significant change (p={p_value:.4f} < {alpha})")
else:
    print(f"\n❌ No significant change (p={p_value:.4f} ≥ {alpha})")
```

---

## 7.6 A/B Testing: Best Practices

### What It Is
Comparing two versions (A vs B) to determine which performs better.

---

### 7.6.1 Sample Size Calculation

**CRITICAL:** Calculate sample size BEFORE running test!

### Key Parameters

1. **Baseline conversion rate**: Current conversion rate (e.g., 5%)
2. **Minimum Detectable Effect (MDE)**: Smallest difference worth detecting (e.g., 10% relative lift)
3. **Statistical significance (α)**: Probability of false positive (typically 0.05 = 5%)
4. **Statistical power (1-β)**: Probability of detecting true effect (typically 0.80 = 80%)

### Code Example

```python
import scipy.stats as stats
import numpy as np

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """
    Calculate required sample size for A/B test

    Args:
        baseline_rate: Current conversion rate (e.g., 0.05 for 5%)
        mde: Minimum detectable effect (e.g., 0.10 for 10% relative lift)
        alpha: Significance level (default 0.05)
        power: Statistical power (default 0.80)

    Returns:
        Required sample size per variant
    """
    # Effect size
    p1 = baseline_rate
    p2 = baseline_rate * (1 + mde)  # Relative lift

    # Pooled proportion
    p_pooled = (p1 + p2) / 2

    # Z-scores
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # Two-tailed
    z_beta = stats.norm.ppf(power)

    # Sample size formula
    n = ((z_alpha * np.sqrt(2 * p_pooled * (1 - p_pooled)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))**2) / ((p2 - p1)**2)

    return int(np.ceil(n))

# Example: 5% baseline, want to detect 10% relative lift
baseline = 0.05
mde = 0.10  # 10% relative lift

sample_size = calculate_sample_size(baseline, mde)
print(f"Baseline conversion rate: {baseline*100:.1f}%")
print(f"Target conversion rate: {baseline*(1+mde)*100:.1f}%")
print(f"Required sample size per variant: {sample_size:,}")
print(f"Total sample size needed: {sample_size*2:,}")
```

---

### 7.6.2 Test Duration

**Best Practices:**

1. **Minimum 2 weeks**
   - Capture weekly patterns
   - Account for day-of-week effects

2. **Maximum 6-8 weeks**
   - Diminishing returns after that
   - External factors start affecting results

3. **Run through complete business cycles**
   - E-commerce: Include weekday + weekend
   - B2B: Full week (avoid partial weeks)

### Calculate Duration

```python
def calculate_test_duration(sample_size_per_variant, daily_visitors):
    """
    Calculate how long test will run

    Args:
        sample_size_per_variant: From sample size calculation
        daily_visitors: Average daily visitors

    Returns:
        Days needed to complete test
    """
    total_needed = sample_size_per_variant * 2
    days = total_needed / daily_visitors
    return np.ceil(days)

# Example
visitors_per_day = 5000
sample_needed = 12000  # per variant

days = calculate_test_duration(sample_needed, visitors_per_day)
print(f"Daily visitors: {visitors_per_day:,}")
print(f"Sample needed per variant: {sample_needed:,}")
print(f"Estimated test duration: {int(days)} days ({days/7:.1f} weeks)")
```

---

### 7.6.3 Analyzing A/B Test Results

```python
def analyze_ab_test(conversions_a, visitors_a, conversions_b, visitors_b):
    """
    Analyze A/B test with statistical significance

    Args:
        conversions_a: Number of conversions in variant A
        visitors_a: Number of visitors in variant A
        conversions_b: Number of conversions in variant B
        visitors_b: Number of visitors in variant B

    Returns:
        Dictionary with test results
    """
    # Conversion rates
    rate_a = conversions_a / visitors_a
    rate_b = conversions_b / visitors_b

    # Pooled proportion
    p_pooled = (conversions_a + conversions_b) / (visitors_a + visitors_b)

    # Standard error
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/visitors_a + 1/visitors_b))

    # Z-score
    z_score = (rate_b - rate_a) / se

    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

    # Relative lift
    relative_lift = ((rate_b - rate_a) / rate_a) * 100

    # Confidence interval (95%)
    margin = 1.96 * se
    ci_lower = (rate_b - rate_a) - margin
    ci_upper = (rate_b - rate_a) + margin

    results = {
        'conversion_rate_a': rate_a,
        'conversion_rate_b': rate_b,
        'relative_lift': relative_lift,
        'z_score': z_score,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': p_value < 0.05
    }

    return results

# Example
conv_a, visit_a = 250, 5000  # 5.0% conversion
conv_b, visit_b = 275, 5000  # 5.5% conversion

results = analyze_ab_test(conv_a, visit_a, conv_b, visit_b)

print(f"Variant A: {results['conversion_rate_a']*100:.2f}% ({conv_a}/{visit_a})")
print(f"Variant B: {results['conversion_rate_b']*100:.2f}% ({conv_b}/{visit_b})")
print(f"\nRelative lift: {results['relative_lift']:.2f}%")
print(f"P-value: {results['p_value']:.4f}")
print(f"95% CI: [{results['ci_lower']*100:.2f}%, {results['ci_upper']*100:.2f}%]")

if results['significant']:
    print(f"\n✅ Statistically significant (p < 0.05)")
    print(f"Variant B {'increases' if results['relative_lift'] > 0 else 'decreases'} "
          f"conversion by {abs(results['relative_lift']):.2f}%")
else:
    print(f"\n❌ Not statistically significant (p ≥ 0.05)")
    print("Continue test or accept no difference")
```

---

### 7.6.4 Common A/B Testing Pitfalls

### Pitfall 1: Peeking (Multiple Testing)

❌ **Wrong:**
```python
# Checking results every day and stopping when p < 0.05
# This inflates false positive rate!
```

✅ **Correct:**
```python
# Wait until planned sample size is reached
# Use sequential testing methods if must peek
```

### Pitfall 2: Ignoring Sample Size

❌ **Wrong:**
```python
# "Let's run test for 1 week regardless of traffic"
```

✅ **Correct:**
```python
# Calculate required sample size first
# Run until sample size is reached (even if > 1 week)
```

### Pitfall 3: Testing Too Many Variants

❌ **Wrong:**
```python
# Testing A vs B vs C vs D vs E simultaneously
# Needs 5x more traffic, increases false positive rate
```

✅ **Correct:**
```python
# Test 2 variants at a time
# Use multivariate testing frameworks if needed
```

### Pitfall 4: Stopping Too Early

❌ **Wrong:**
```python
# Stopping at p=0.049 after 3 days
```

✅ **Correct:**
```python
# Run for at least 2 weeks
# Reach planned sample size
# Consider business cycles
```

### Pitfall 5: Ignoring Novelty Effect

**Issue:** Users click new design out of curiosity, not genuine preference

✅ **Solution:**
- Run test for at least 2-4 weeks
- Segment new vs returning users
- Monitor metrics over time

---

## 7.7 Statistical Test Comparison Table

| Test | # Groups | Data Type | Paired | Parametric | When to Use |
|------|----------|-----------|---------|------------|-------------|
| **Chi-Square** | 2+ | Categorical | No | Non-parametric | Test independence of categorical variables |
| **Independent t-test** | 2 | Continuous | No | Yes | Compare means, normal data |
| **Paired t-test** | 2 | Continuous | Yes | Yes | Before/after, normal data |
| **Mann-Whitney U** | 2 | Ordinal/Continuous | No | Non-parametric | Compare 2 groups, non-normal |
| **Wilcoxon** | 2 | Ordinal/Continuous | Yes | Non-parametric | Paired data, non-normal |
| **One-Way ANOVA** | 3+ | Continuous | No | Yes | Compare 3+ groups, normal data |
| **Kruskal-Wallis** | 3+ | Ordinal/Continuous | No | Non-parametric | Compare 3+ groups, non-normal |
| **Pearson Correlation** | 2 | Continuous | - | Yes | Linear relationship, normal |
| **Spearman Correlation** | 2 | Ordinal/Continuous | - | Non-parametric | Monotonic relationship |

---

## 7.8 Complete Statistical Test Decision Tree

```
Step 1: What type of variables?

├─ TWO CATEGORICAL → Chi-Square Test
│
├─ ONE CATEGORICAL + ONE CONTINUOUS
│  └─ How many groups?
│     ├─ 2 GROUPS
│     │  └─ Paired or independent?
│     │     ├─ Independent
│     │     │  └─ Normal? → t-test | Not normal? → Mann-Whitney U
│     │     └─ Paired
│     │        └─ Normal? → Paired t-test | Not normal? → Wilcoxon
│     │
│     └─ 3+ GROUPS
│        └─ Normal? → ANOVA | Not normal? → Kruskal-Wallis
│
└─ TWO CONTINUOUS
   └─ Linear relationship? → Pearson | Monotonic? → Spearman
```

---

## 7.9 Summary Checklist

### Before Running Any Test:
- [ ] Define null hypothesis (H0) and alternative hypothesis (H1)
- [ ] Set significance level (α = 0.05 typically)
- [ ] Calculate required sample size (for A/B tests)
- [ ] Check test assumptions
- [ ] Choose appropriate test based on data type

### When Running Test:
- [ ] Check assumptions are met (normality, independence, etc.)
- [ ] Use appropriate test statistic
- [ ] Calculate p-value
- [ ] Report effect size (Cohen's d, Cramér's V, etc.)
- [ ] Construct confidence intervals

### After Test:
- [ ] Interpret p-value correctly
- [ ] Don't confuse statistical significance with practical significance
- [ ] Report effect size and confidence intervals
- [ ] Consider business impact, not just statistical significance

### For A/B Tests Specifically:
- [ ] Calculate sample size before starting
- [ ] Run for at least 2 weeks
- [ ] Don't peek early (or use sequential testing)
- [ ] Account for multiple comparisons
- [ ] Consider business metrics beyond primary metric

---

## Resources & Further Reading

**Statistical Testing:**
- "Statistics" by Freedman, Pisani, Purves
- "Statistical Methods" by Snedecor & Cochran

**A/B Testing:**
- "Trustworthy Online Controlled Experiments" by Kohavi et al.
- Evan Miller's A/B testing tools: https://www.evanmiller.org/ab-testing/

**Online Tools:**
- Sample size calculator: https://www.statsig.com/calculator
- Effect size calculator: https://www.psychometrica.de/effect_size.html

**Python Libraries:**
- scipy.stats: https://docs.scipy.org/doc/scipy/reference/stats.html
- statsmodels: https://www.statsmodels.org/
- pingouin: https://pingouin-stats.org/

---

**Last Updated:** 2025-10-12
**Next Section:** Deep Learning Basics (Phase 5)
