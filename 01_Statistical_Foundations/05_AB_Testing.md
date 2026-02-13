# 5. Experimental Design and A/B Testing

## Overview

A/B testing is the gold standard for causal inference in product development. This guide covers classical methods and cutting-edge 2025 techniques including sequential testing and always valid inference.

---

## 5.1 A/B Testing Framework

### Seven-Step Process

#### Step 1: Understand the Problem

- Define clear business goal
- Identify success metric
- Understand user journey and funnel
- Ask clarifying questions about context

**Example Questions:**
- What user behavior are we trying to change?
- What is the current baseline?
- What's the minimum improvement worth implementing?

---

#### Step 2: Define Hypotheses

**Null Hypothesis (H_0):** No difference between variants
- Example: Average revenue per user is same for control and treatment

**Alternative Hypothesis (H_1):** There is a difference
- Can be one-sided or two-sided

**Set Parameters:**
- **Significance level (alpha):** Typically 0.05
- **Statistical power (1-beta):** Typically 0.80
- **Minimum Detectable Effect (MDE):** Smallest practical difference

```python
import numpy as np
from scipy import stats

def calculate_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """
    Calculate required sample size for A/B test

    Args:
        baseline_rate: Current conversion rate (e.g., 0.10 for 10%)
        mde: Minimum detectable effect (relative change, e.g., 0.05 for 5%)
        alpha: Significance level
        power: Statistical power

    Returns:
        Sample size per variant
    """
    # Standard normal quantiles
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)

    # Treatment rate
    treatment_rate = baseline_rate * (1 + mde)

    # Pooled proportion
    p = (baseline_rate + treatment_rate) / 2

    # Sample size formula
    n = (2 * p * (1 - p) * (z_alpha + z_beta)**2) / (baseline_rate - treatment_rate)**2

    return int(np.ceil(n))

# Example: 10% baseline, want to detect 5% relative improvement
baseline = 0.10
mde = 0.05  # 5% relative = 10% --> 10.5%

n = calculate_sample_size(baseline, mde)
print(f"Baseline rate: {baseline:.1%}")
print(f"Target rate: {baseline*(1+mde):.1%}")
print(f"Required sample size per variant: {n:,}")
```

---

#### Step 3: Design the Experiment

**Randomization Unit:**
- User level (most common)
- Session level
- Device level
- Geography level

**Target Population:**
- All users
- Specific segment (e.g., mobile users, returning customers)

**Duration:**
- Minimum 1-2 weeks to capture day-of-week effects
- Consider seasonality
- Account for novelty effects

**Traffic Split:**
- Usually 50/50
- Can use 90/10 for risky changes

```python
import numpy as np

class ABTestDesign:
    def __init__(self, baseline_rate, mde, alpha=0.05, power=0.80, split=0.5):
        self.baseline_rate = baseline_rate
        self.mde = mde
        self.alpha = alpha
        self.power = power
        self.split = split

        # Calculate sample size
        self.n_per_variant = calculate_sample_size(baseline_rate, mde, alpha, power)
        self.n_total = int(self.n_per_variant / split)

    def estimate_duration(self, daily_traffic):
        """Estimate test duration in days"""
        return np.ceil(self.n_total / daily_traffic)

    def summary(self, daily_traffic):
        duration = self.estimate_duration(daily_traffic)

        print(f"A/B Test Design Summary")
        print(f"=" * 50)
        print(f"Baseline rate: {self.baseline_rate:.1%}")
        print(f"MDE: {self.mde:.1%} (relative)")
        print(f"Target rate: {self.baseline_rate * (1 + self.mde):.1%}")
        print(f"Significance level (alpha): {self.alpha}")
        print(f"Power (1-beta): {self.power}")
        print(f"Traffic split: {self.split:.0%} / {1-self.split:.0%}")
        print(f"\nSample size per variant: {self.n_per_variant:,}")
        print(f"Total sample size: {self.n_total:,}")
        print(f"Estimated duration: {int(duration)} days @ {daily_traffic:,}/day")

# Example
design = ABTestDesign(baseline_rate=0.10, mde=0.05)
design.summary(daily_traffic=10000)
```

---

#### Step 4: Run the Experiment

**Implementation Checklist:**
- [x] Implement random assignment
- [x] Log all exposures and outcomes
- [x] Set up monitoring dashboard
- [x] Do NOT peek at p-values (increases Type I error)

**Assignment Mechanism:**

```python
import hashlib

def assign_variant(user_id, experiment_id, salt=''):
    """Deterministic random assignment using hash"""
    # Hash user_id + experiment_id
    hash_input = f"{user_id}{experiment_id}{salt}".encode()
    hash_output = hashlib.md5(hash_input).hexdigest()

    # Convert to number 0-1
    hash_number = int(hash_output, 16) / (16**32)

    # Assign variant
    return 'treatment' if hash_number < 0.5 else 'control'

# Example
user_ids = [f"user_{i}" for i in range(10)]
for user_id in user_ids:
    variant = assign_variant(user_id, experiment_id='exp_001')
    print(f"{user_id}: {variant}")
```

---

#### Step 5: Validity Checks (Critical!)

**1. Instrumentation Check**

Verify data collection working properly.

```python
def check_instrumentation(control_count, treatment_count, expected_total):
    """Check if logging is working"""
    actual_total = control_count + treatment_count
    missing_rate = 1 - (actual_total / expected_total)

    print(f"Expected: {expected_total:,}")
    print(f"Actual: {actual_total:,}")
    print(f"Missing: {missing_rate:.1%}")

    if missing_rate > 0.05:
        print("[WARNING] WARNING: More than 5% data missing!")
    else:
        print("[x] Instrumentation looks good")
```

**2. Sample Ratio Mismatch (SRM)**

Verify randomization is balanced.

```python
from scipy import stats

def check_srm(control_count, treatment_count, expected_split=0.5):
    """
    Sample Ratio Mismatch check using chi-square test

    Returns p-value. Low p-value indicates randomization issue.
    """
    total = control_count + treatment_count

    # Expected counts
    expected_control = total * expected_split
    expected_treatment = total * (1 - expected_split)

    # Chi-square test
    observed = [control_count, treatment_count]
    expected = [expected_control, expected_treatment]

    chi2_stat, p_value = stats.chisquare(observed, expected)

    print(f"Control: {control_count:,} (expected {expected_control:,.0f})")
    print(f"Treatment: {treatment_count:,} (expected {expected_treatment:,.0f})")
    print(f"Chi-square p-value: {p_value:.4f}")

    if p_value < 0.001:
        print("[WARNING] WARNING: Significant SRM detected!")
        print("   Randomization may be broken. Do not trust results.")
    else:
        print("[x] No SRM detected")

    return p_value

# Example
check_srm(control_count=50247, treatment_count=49753, expected_split=0.5)
```

**3. AA Test**

Run control vs. control to verify no systematic bias.

```python
def aa_test(control_data, holdout_data):
    """
    AA test: Compare two control groups

    Should see no significant difference
    """
    t_stat, p_value = stats.ttest_ind(control_data, holdout_data)

    print(f"Control mean: {np.mean(control_data):.4f}")
    print(f"Holdout mean: {np.mean(holdout_data):.4f}")
    print(f"P-value: {p_value:.4f}")

    if p_value < 0.05:
        print("[WARNING] WARNING: AA test shows significant difference!")
        print("   This suggests measurement or assignment issues.")
    else:
        print("[x] AA test passed")

# Example
np.random.seed(42)
control = np.random.binomial(1, 0.10, 10000)
holdout = np.random.binomial(1, 0.10, 10000)

aa_test(control, holdout)
```

**4. Novelty Effect Check**

Compare new vs. returning users.

```python
def novelty_check(new_users_treatment, returning_users_treatment,
                  new_users_control, returning_users_control):
    """Check if effect differs between new and returning users"""

    # Treatment effect for new users
    new_lift = np.mean(new_users_treatment) - np.mean(new_users_control)

    # Treatment effect for returning users
    returning_lift = np.mean(returning_users_treatment) - np.mean(returning_users_control)

    print(f"New users lift: {new_lift:.4f}")
    print(f"Returning users lift: {returning_lift:.4f}")

    if abs(new_lift - returning_lift) / abs(returning_lift) > 0.5:
        print("[WARNING] Large novelty effect detected")
        print("   Consider running longer or analyzing segments separately")
```

---

#### Step 6: Interpret Results

**Calculate Lift and Confidence Interval:**

```python
from scipy import stats
import numpy as np

def analyze_ab_test(control_successes, control_total,
                     treatment_successes, treatment_total,
                     alpha=0.05):
    """
    Analyze A/B test results

    Returns:
        Dictionary with test statistics, lift, and confidence interval
    """
    # Rates
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total

    # Absolute lift
    absolute_lift = treatment_rate - control_rate

    # Relative lift
    relative_lift = absolute_lift / control_rate

    # Standard error
    se = np.sqrt(
        control_rate * (1 - control_rate) / control_total +
        treatment_rate * (1 - treatment_rate) / treatment_total
    )

    # Z-test
    z_stat = absolute_lift / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval
    z_crit = stats.norm.ppf(1 - alpha/2)
    ci_lower = absolute_lift - z_crit * se
    ci_upper = absolute_lift + z_crit * se

    # Relative CI
    ci_lower_rel = ci_lower / control_rate
    ci_upper_rel = ci_upper / control_rate

    results = {
        'control_rate': control_rate,
        'treatment_rate': treatment_rate,
        'absolute_lift': absolute_lift,
        'relative_lift': relative_lift,
        'p_value': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'ci_lower_rel': ci_lower_rel,
        'ci_upper_rel': ci_upper_rel,
        'significant': p_value < alpha
    }

    return results

def print_results(results):
    """Pretty print A/B test results"""
    print("A/B Test Results")
    print("=" * 60)
    print(f"Control rate:    {results['control_rate']:.2%}")
    print(f"Treatment rate:  {results['treatment_rate']:.2%}")
    print(f"\nAbsolute lift:   {results['absolute_lift']:.2%}")
    print(f"Relative lift:   {results['relative_lift']:+.2%}")
    print(f"\n95% CI (absolute): [{results['ci_lower']:.2%}, {results['ci_upper']:.2%}]")
    print(f"95% CI (relative): [{results['ci_lower_rel']:+.2%}, {results['ci_upper_rel']:+.2%}]")
    print(f"\nP-value:         {results['p_value']:.4f}")
    print(f"Significant:     {'[x] YES' if results['significant'] else '[ ] NO'}")

# Example
results = analyze_ab_test(
    control_successes=1000, control_total=10000,
    treatment_successes=1150, treatment_total=10000
)
print_results(results)
```

---

#### Step 7: Launch Decision

**Decision Matrix:**

```python
def launch_decision(results, mde, cost_estimate=None):
    """
    Make launch decision based on results

    Args:
        results: Output from analyze_ab_test
        mde: Minimum detectable effect (as decimal)
        cost_estimate: Optional implementation cost
    """
    ci_lower_rel = results['ci_lower_rel']
    ci_upper_rel = results['ci_upper_rel']
    relative_lift = results['relative_lift']

    print("Launch Decision Framework")
    print("=" * 60)

    if ci_lower_rel > mde:
        decision = "LAUNCH [x]"
        reason = f"CI fully above MDE ({mde:.1%})"
    elif ci_upper_rel < 0:
        decision = "DO NOT LAUNCH [ ]"
        reason = "CI fully negative"
    elif relative_lift > 0 and results['significant']:
        if ci_lower_rel > 0:
            decision = "LIKELY LAUNCH [WARNING]"
            reason = "Positive and significant, but CI includes values below MDE"
        else:
            decision = "RERUN TEST cycle"
            reason = "Positive but CI includes zero"
    else:
        decision = "DO NOT LAUNCH [ ]"
        reason = "Not significant or negative lift"

    print(f"Decision: {decision}")
    print(f"Reason: {reason}")

    if cost_estimate:
        expected_value = relative_lift * 1000000  # Example revenue impact
        roi = (expected_value - cost_estimate) / cost_estimate
        print(f"\nExpected value: ${expected_value:,.0f}")
        print(f"Implementation cost: ${cost_estimate:,.0f}")
        print(f"ROI: {roi:.1%}")

# Example
launch_decision(results, mde=0.05)
```

---

## 5.2 Sequential Testing (2025 Best Practice)

### The Peeking Problem

**Problem:** Looking at results before test completes inflates Type I error rate.

**Traditional Solution:** Pre-define sample size, never peek.

**2025 Solution:** Always Valid Inference (AVI)

---

### Always Valid Inference

Allows continuous monitoring without inflating false positive rate.

**Key Idea:** Adjust p-values and confidence intervals to remain valid at any stopping time.

```python
import numpy as np
from scipy import stats

class AlwaysValidInference:
    """
    Always Valid Inference for A/B testing

    Based on: Johari et al. (2017) "Always Valid Inference"
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.rho = alpha / 2  # Mixing parameter

    def always_valid_p_value(self, n_control, n_treatment,
                              sum_control, sum_treatment,
                              sumsq_control, sumsq_treatment):
        """
        Compute always valid p-value using mixture sequential probability ratio test

        Args:
            n_control, n_treatment: Sample sizes
            sum_control, sum_treatment: Sum of outcomes
            sumsq_control, sumsq_treatment: Sum of squared outcomes
        """
        # Sample means and variances
        mean_control = sum_control / n_control
        mean_treatment = sum_treatment / n_treatment

        var_control = (sumsq_control - sum_control**2 / n_control) / (n_control - 1)
        var_treatment = (sumsq_treatment - sum_treatment**2 / n_treatment) / (n_treatment - 1)

        # Pooled variance
        var_pooled = ((n_control - 1) * var_control +
                       (n_treatment - 1) * var_treatment) / (n_control + n_treatment - 2)

        # Z-statistic
        se = np.sqrt(var_pooled * (1/n_control + 1/n_treatment))
        z = (mean_treatment - mean_control) / se

        # Always valid p-value (conservative approximation)
        # True implementation requires computing mixture over effect sizes
        # Simplified version: use adjusted critical value
        adjusted_z = z / np.sqrt(1 + self.rho * np.log(max(n_control, n_treatment)))

        p_value = 2 * (1 - stats.norm.cdf(abs(adjusted_z)))

        return p_value

    def can_stop(self, p_value):
        """Check if we can stop the test"""
        return p_value < self.alpha or p_value > 1 - self.alpha

# Example: Sequential monitoring
np.random.seed(42)

# True effect: treatment increases rate from 10% to 10.5%
def simulate_sequential_test(n_max=10000, check_every=500):
    """Simulate sequential A/B test with AVI"""
    avi = AlwaysValidInference(alpha=0.05)

    # Generate all data upfront (for simulation)
    control = np.random.binomial(1, 0.10, n_max)
    treatment = np.random.binomial(1, 0.105, n_max)

    # Sequential monitoring
    for n in range(check_every, n_max + 1, check_every):
        # Current data
        c_data = control[:n]
        t_data = treatment[:n]

        # Compute statistics
        p_value = avi.always_valid_p_value(
            n, n,
            np.sum(c_data), np.sum(t_data),
            np.sum(c_data**2), np.sum(t_data**2)
        )

        print(f"n={n:5d} | p-value={p_value:.4f} | " +
              f"Control: {np.mean(c_data):.3f} | " +
              f"Treatment: {np.mean(t_data):.3f}")

        if avi.can_stop(p_value):
            print(f"\n[x] Can stop at n={n} (p={p_value:.4f})")
            break

    return n, p_value

final_n, final_p = simulate_sequential_test()
```

---

### Sequential Testing Frameworks (2025)

**1. Always Valid Inference (AVI)**
- Works under minimal restrictions
- Good for newcomers to sequential testing
- Reliably bounded false positive rates

**2. Group Sequential Tests (GST)**
- Optimal for pre-planned interim analyses
- Exploits correlation structure
- Used in clinical trials

**3. mSPRT (mixture Sequential Probability Ratio Test)**
- Optimal power properties
- More complex to implement

```python
# Practical sequential testing wrapper
class SequentialABTest:
    """Easy-to-use sequential A/B test"""
    def __init__(self, alpha=0.05, min_sample_size=1000):
        self.alpha = alpha
        self.min_sample_size = min_sample_size
        self.avi = AlwaysValidInference(alpha=alpha)

        # Running statistics
        self.n_control = 0
        self.n_treatment = 0
        self.sum_control = 0.0
        self.sum_treatment = 0.0
        self.sumsq_control = 0.0
        self.sumsq_treatment = 0.0

    def add_observation(self, variant, outcome):
        """Add new observation"""
        if variant == 'control':
            self.n_control += 1
            self.sum_control += outcome
            self.sumsq_control += outcome**2
        else:
            self.n_treatment += 1
            self.sum_treatment += outcome
            self.sumsq_treatment += outcome**2

    def get_result(self):
        """Get current test result"""
        if self.n_control < self.min_sample_size or self.n_treatment < self.min_sample_size:
            return {
                'can_conclude': False,
                'reason': 'Minimum sample size not reached',
                'n_control': self.n_control,
                'n_treatment': self.n_treatment
            }

        # Compute p-value
        p_value = self.avi.always_valid_p_value(
            self.n_control, self.n_treatment,
            self.sum_control, self.sum_treatment,
            self.sumsq_control, self.sumsq_treatment
        )

        mean_control = self.sum_control / self.n_control
        mean_treatment = self.sum_treatment / self.n_treatment

        can_conclude = self.avi.can_stop(p_value)

        return {
            'can_conclude': can_conclude,
            'p_value': p_value,
            'significant': p_value < self.alpha,
            'mean_control': mean_control,
            'mean_treatment': mean_treatment,
            'relative_lift': (mean_treatment - mean_control) / mean_control,
            'n_control': self.n_control,
            'n_treatment': self.n_treatment
        }

# Example usage
test = SequentialABTest(alpha=0.05, min_sample_size=1000)

# Simulate streaming data
np.random.seed(42)
for i in range(5000):
    # Random assignment
    variant = 'treatment' if np.random.random() < 0.5 else 'control'

    # Outcome (treatment has small positive effect)
    rate = 0.105 if variant == 'treatment' else 0.10
    outcome = np.random.binomial(1, rate)

    test.add_observation(variant, outcome)

    # Check every 500 observations
    if (i + 1) % 500 == 0:
        result = test.get_result()
        print(f"n={i+1:5d} | {result}")

        if result['can_conclude']:
            print("\n[x] Test concluded!")
            break
```

---

## 5.3 Success Metrics

### Qualities of Good Metrics

- **Measurable**: Can be collected through instrumentation
- **Attributable**: Clear link between treatment and effect
- **Sensitive**: Low variability, detectable signal
- **Timely**: Observable in reasonable timeframe

### Common Metrics

**Revenue Metrics:**
- Revenue per user
- Average order value
- Lifetime value

**Engagement Metrics:**
- Click-through rate
- Time on site
- Pages per session

**Conversion Metrics:**
- Conversion rate
- Completion rate
- Sign-up rate

**Retention Metrics:**
- Return rate
- Churn rate
- Day-N retention

### Primary vs. Guardrail Metrics

**Primary:** What you're trying to improve

**Guardrail:** Ensure other important metrics don't degrade

```python
def evaluate_metrics(control_metrics, treatment_metrics, guardrail_threshold=-0.02):
    """
    Evaluate primary and guardrail metrics

    Args:
        control_metrics: Dict of metric_name -> list of values
        treatment_metrics: Dict of metric_name -> list of values
        guardrail_threshold: Maximum acceptable relative decline
    """
    results = {}

    for metric_name in control_metrics.keys():
        c_mean = np.mean(control_metrics[metric_name])
        t_mean = np.mean(treatment_metrics[metric_name])

        relative_change = (t_mean - c_mean) / c_mean

        # T-test
        t_stat, p_value = stats.ttest_ind(treatment_metrics[metric_name],
                                            control_metrics[metric_name])

        results[metric_name] = {
            'control_mean': c_mean,
            'treatment_mean': t_mean,
            'relative_change': relative_change,
            'p_value': p_value,
            'significant': p_value < 0.05
        }

    print("Metric Evaluation")
    print("=" * 80)

    for metric, res in results.items():
        status = "[x]" if res['relative_change'] > 0 else "[ ]"
        print(f"{metric:20s} | {status} | " +
              f"Control: {res['control_mean']:8.3f} | " +
              f"Treatment: {res['treatment_mean']:8.3f} | " +
              f"Change: {res['relative_change']:+6.1%} | " +
              f"p={res['p_value']:.4f}")

    return results

# Example
control = {
    'revenue': np.random.gamma(2, 5, 1000),
    'engagement': np.random.gamma(3, 2, 1000),
    'retention': np.random.binomial(1, 0.6, 1000)
}

treatment = {
    'revenue': np.random.gamma(2.1, 5, 1000),  # +5% improvement
    'engagement': np.random.gamma(3, 2, 1000),  # No change
    'retention': np.random.binomial(1, 0.58, 1000)  # -2% decline (guardrail)
}

results = evaluate_metrics(control, treatment)
```

---

## 5.4 Common Pitfalls

### Multiple Comparisons

Testing many hypotheses increases false discovery rate.

**Solution:** Bonferroni or FDR correction

```python
from statsmodels.stats.multitest import multipletests

# Multiple metrics tested
p_values = [0.03, 0.08, 0.01, 0.15, 0.04]

# FDR correction
reject, p_adjusted, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

for i, (p, p_adj, rej) in enumerate(zip(p_values, p_adjusted, reject)):
    print(f"Metric {i+1}: p={p:.3f}, p_adj={p_adj:.3f}, reject={rej}")
```

---

### Simpson's Paradox

Aggregated data shows opposite trend from segments.

**Solution:** Always segment by important dimensions (device, geography, user type).

```python
# Simpson's Paradox example
def simpsons_paradox_example():
    # Segment A: Treatment wins
    segment_a = pd.DataFrame({
        'variant': ['control']*100 + ['treatment']*100,
        'converted': [1]*20 + [0]*80 + [1]*30 + [0]*70
    })

    # Segment B: Treatment wins
    segment_b = pd.DataFrame({
        'variant': ['control']*500 + ['treatment']*100,
        'converted': [1]*100 + [0]*400 + [1]*30 + [0]*70
    })

    # Combined: Control wins (Simpson's Paradox!)
    combined = pd.concat([segment_a, segment_b])

    print("Segment A:")
    print(segment_a.groupby('variant')['converted'].mean())

    print("\nSegment B:")
    print(segment_b.groupby('variant')['converted'].mean())

    print("\nCombined (paradox!):")
    print(combined.groupby('variant')['converted'].mean())
```

---

## Resources

**Classic:**
- "Trustworthy Online Controlled Experiments" by Kohavi, Tang, Xu (2020)
- "Design and Analysis of Experiments" by Montgomery

**2025 Sequential Testing:**
- Johari et al. "Always Valid Inference" (2017)
- Spotify Engineering: Sequential Testing Frameworks (2023)
- Netflix: Sequential A/B Testing (blog series)

**Tools:**
- Statsmodels: Power analysis
- Eppo: Modern A/B testing platform (uses GAVI)
- GrowthBook: Open-source feature flagging + experimentation
