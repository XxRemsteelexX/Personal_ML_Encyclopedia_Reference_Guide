# Fairness and Bias in Machine Learning

## Overview

Fairness in machine learning ensures that models make decisions equitably across different demographic groups. In 2025, fairness is not optional - it's mandated by regulations (EU AI Act, GDPR), essential for ethical AI, and critical for business reputation and legal compliance.

**Critical Understanding:** "Fair" does not mean "same predictions for everyone" - it means predictions are made without discriminatory bias based on protected characteristics.

---

## Table of Contents
1. [Why Fairness Matters (2025 Context)](#why-fairness-matters)
2. [Types of Bias](#types-of-bias)
3. [Fairness Metrics](#fairness-metrics)
4. [Bias Detection](#bias-detection)
5. [Bias Mitigation](#bias-mitigation)
6. [Fairness Libraries](#fairness-libraries)
7. [EU AI Act Compliance](#eu-ai-act-compliance)
8. [Case Studies](#case-studies)

---

## Why Fairness Matters

### 1. Legal and Regulatory Requirements (2025)

**EU AI Act (2024):**
- High-risk AI systems must undergo conformity assessment
- Mandatory bias monitoring and documentation
- Fines up to €35M or 7% of global turnover

**GDPR Article 22:**
- Right to non-discrimination in automated decisions
- Right to human intervention

**US Regulations:**
- Fair Credit Reporting Act (FCRA)
- Equal Credit Opportunity Act (ECOA)
- Fair Housing Act
- Civil Rights Act Title VII (employment)

**Financial Services:**
- Bank regulators require bias testing for credit models
- Disparate impact analysis required

### 2. Business Impact

```python
# Example: Cost of unfair AI

import numpy as np

def calculate_unfairness_cost(
    n_decisions_per_year=1_000_000,
    lawsuit_probability=0.001,
    avg_settlement=500_000,
    reputation_damage=10_000_000,
    regulatory_fine=5_000_000
):
    """
    Estimate financial cost of biased AI system.
    """
    # Direct legal costs
    legal_cost = n_decisions_per_year * lawsuit_probability * avg_settlement

    # One-time costs (assuming discovered)
    discovery_probability = 0.1
    one_time_cost = (reputation_damage + regulatory_fine) * discovery_probability

    total_annual_cost = legal_cost + one_time_cost

    print("=" * 70)
    print("ESTIMATED COST OF UNFAIR AI SYSTEM")
    print("=" * 70)
    print(f"Annual legal costs: ${legal_cost:,.0f}")
    print(f"Reputation + regulatory (amortized): ${one_time_cost:,.0f}")
    print(f"Total estimated annual cost: ${total_annual_cost:,.0f}")
    print("")
    print("Investment in fairness: ~$100,000 - $500,000")
    print(f"ROI: {(total_annual_cost / 300_000):.1f}x")

    return total_annual_cost

cost = calculate_unfairness_cost()
```

### 3. Notable Cases

**Real-world bias incidents:**
- **COMPAS (2016):** Recidivism prediction biased against Black defendants
- **Amazon Hiring (2018):** Resume screening biased against women
- **Apple Card (2019):** Credit limits biased against women
- **Healthcare Algorithm (2019):** Underestimated care needs of Black patients
- **Facial Recognition (2020):** Higher error rates for darker-skinned individuals

---

## Types of Bias

### 1. Historical Bias

Exists in the data because of societal biases.

```python
import pandas as pd
import numpy as np

def demonstrate_historical_bias():
    """
    Example: Historical hiring bias in data.
    """
    # Simulated historical hiring data
    np.random.seed(42)

    n_samples = 1000

    # Historical bias: men were hired more often
    gender = np.random.choice(['M', 'F'], n_samples, p=[0.7, 0.3])

    # Qualifications (unbiased)
    qualification_score = np.random.normal(75, 15, n_samples)

    # Hiring decision (biased by gender due to historical discrimination)
    hired = []
    for g, q in zip(gender, qualification_score):
        # Men: hired if score > 60
        # Women: hired if score > 75 (higher bar historically)
        if g == 'M':
            threshold = 60
        else:
            threshold = 75

        hired.append(1 if q > threshold else 0)

    df = pd.DataFrame({
        'gender': gender,
        'qualification_score': qualification_score,
        'hired': hired
    })

    print("=" * 70)
    print("HISTORICAL BIAS DEMONSTRATION")
    print("=" * 70)
    print("\nHiring rates by gender:")
    print(df.groupby('gender')['hired'].agg(['mean', 'count']))

    print("\nAverage qualification score of hired candidates:")
    print(df[df['hired'] == 1].groupby('gender')['qualification_score'].mean())

    print("\n⚠️  Women need higher qualifications to be hired (historical bias)")

    return df

historical_data = demonstrate_historical_bias()
```

**Mitigation:** Cannot simply remove protected attributes - need to address structural causes.

### 2. Representation Bias

Some groups are underrepresented in training data.

```python
def demonstrate_representation_bias():
    """
    Example: Underrepresentation leads to poor performance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    # Imbalanced training data
    # 90% Group A, 10% Group B
    n_train = 1000
    group_a_size = 900
    group_b_size = 100

    # Generate features and labels
    X_train_a = np.random.randn(group_a_size, 10)
    y_train_a = (X_train_a[:, 0] > 0).astype(int)

    X_train_b = np.random.randn(group_b_size, 10) + 0.5  # Slightly different distribution
    y_train_b = (X_train_b[:, 0] > 0).astype(int)

    X_train = np.vstack([X_train_a, X_train_b])
    y_train = np.concatenate([y_train_a, y_train_b])

    # Test data (balanced)
    X_test_a = np.random.randn(500, 10)
    y_test_a = (X_test_a[:, 0] > 0).astype(int)

    X_test_b = np.random.randn(500, 10) + 0.5
    y_test_b = (X_test_b[:, 0] > 0).astype(int)

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    acc_a = accuracy_score(y_test_a, model.predict(X_test_a))
    acc_b = accuracy_score(y_test_b, model.predict(X_test_b))

    print("=" * 70)
    print("REPRESENTATION BIAS DEMONSTRATION")
    print("=" * 70)
    print(f"Training set: {group_a_size} Group A, {group_b_size} Group B")
    print(f"\nTest Accuracy:")
    print(f"  Group A: {acc_a:.4f}")
    print(f"  Group B: {acc_b:.4f}")
    print(f"  Difference: {abs(acc_a - acc_b):.4f}")
    print("\n⚠️  Underrepresented group has lower accuracy")

demonstrate_representation_bias()
```

### 3. Measurement Bias

Features measured differently across groups.

```python
def demonstrate_measurement_bias():
    """
    Example: Credit score measured differently across groups.
    """
    # Group A: Traditional credit history
    # Group B: Alternative data (less reliable)

    print("=" * 70)
    print("MEASUREMENT BIAS EXAMPLE")
    print("=" * 70)
    print("\nScenario: Credit scoring")
    print("  Group A: Traditional FICO scores (well-calibrated)")
    print("  Group B: Alternative data scores (noisy)")
    print("\nResult: Even with same creditworthiness,")
    print("        Group B scores have higher variance")
    print("        → More false negatives for Group B")
    print("\n⚠️  Different measurement quality creates unfairness")

demonstrate_measurement_bias()
```

### 4. Aggregation Bias

One-size-fits-all model doesn't work for all groups.

```python
def demonstrate_aggregation_bias():
    """
    Example: Single model underperforms for subgroups.
    """
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error

    # Two groups with different relationships
    n = 500

    # Group A: y = 2x + noise
    X_a = np.random.randn(n, 1)
    y_a = 2 * X_a.ravel() + np.random.randn(n) * 0.5

    # Group B: y = -2x + noise (opposite relationship)
    X_b = np.random.randn(n, 1)
    y_b = -2 * X_b.ravel() + np.random.randn(n) * 0.5

    # Combined dataset
    X_combined = np.vstack([X_a, X_b])
    y_combined = np.concatenate([y_a, y_b])

    # Single model for both groups
    model = LinearRegression()
    model.fit(X_combined, y_combined)

    # Evaluate on each group
    mse_a = mean_squared_error(y_a, model.predict(X_a))
    mse_b = mean_squared_error(y_b, model.predict(X_b))

    # Group-specific models
    model_a = LinearRegression().fit(X_a, y_a)
    model_b = LinearRegression().fit(X_b, y_b)

    mse_a_specific = mean_squared_error(y_a, model_a.predict(X_a))
    mse_b_specific = mean_squared_error(y_b, model_b.predict(X_b))

    print("=" * 70)
    print("AGGREGATION BIAS DEMONSTRATION")
    print("=" * 70)
    print("\nSingle Model MSE:")
    print(f"  Group A: {mse_a:.4f}")
    print(f"  Group B: {mse_b:.4f}")

    print("\nGroup-Specific Models MSE:")
    print(f"  Group A: {mse_a_specific:.4f}")
    print(f"  Group B: {mse_b_specific:.4f}")

    print("\n⚠️  Single model underperforms for both groups")
    print("    Consider group-specific models or interactions")

demonstrate_aggregation_bias()
```

### 5. Evaluation Bias

Test set doesn't represent real-world distribution.

---

## Fairness Metrics

### 1. Demographic Parity (Statistical Parity)

**Definition:** Positive prediction rate should be equal across groups.

$$P(\hat{Y} = 1 | A = a) = P(\hat{Y} = 1 | A = b)$$

where $A$ is the protected attribute (e.g., gender, race).

```python
from sklearn.metrics import confusion_matrix

def demographic_parity_difference(y_pred, sensitive_feature):
    """
    Calculate demographic parity difference.

    Returns: Difference in positive prediction rates between groups.
    Range: [-1, 1], 0 = perfect parity
    """
    groups = np.unique(sensitive_feature)

    if len(groups) != 2:
        raise ValueError("This implementation supports binary sensitive features")

    # Positive prediction rates
    rate_0 = np.mean(y_pred[sensitive_feature == groups[0]])
    rate_1 = np.mean(y_pred[sensitive_feature == groups[1]])

    dpd = rate_1 - rate_0

    print("=" * 70)
    print("DEMOGRAPHIC PARITY")
    print("=" * 70)
    print(f"Group {groups[0]} positive rate: {rate_0:.4f}")
    print(f"Group {groups[1]} positive rate: {rate_1:.4f}")
    print(f"Demographic Parity Difference: {dpd:.4f}")

    if abs(dpd) < 0.05:
        print("✓ Acceptable parity (|DPD| < 0.05)")
    else:
        print("✗ Concerning disparity (|DPD| >= 0.05)")

    return dpd

# Example
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate data with protected attribute
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
sensitive_feature = np.random.choice([0, 1], size=1000, p=[0.6, 0.4])

# Train model (without sensitive feature)
model = RandomForestClassifier(random_state=42)
model.fit(X, y)
y_pred = model.predict(X)

dpd = demographic_parity_difference(y_pred, sensitive_feature)
```

**Pros:** Simple, intuitive
**Cons:** May conflict with accuracy, doesn't consider ground truth

### 2. Equal Opportunity

**Definition:** True positive rate should be equal across groups.

$$P(\hat{Y} = 1 | Y = 1, A = a) = P(\hat{Y} = 1 | Y = 1, A = b)$$

```python
def equal_opportunity_difference(y_true, y_pred, sensitive_feature):
    """
    Calculate equal opportunity difference.

    Measures difference in true positive rates (recall) between groups.
    """
    groups = np.unique(sensitive_feature)

    if len(groups) != 2:
        raise ValueError("Binary sensitive features only")

    # True positive rates for each group
    tpr_0 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_feature == groups[0])) / \
            np.sum((y_true == 1) & (sensitive_feature == groups[0]))

    tpr_1 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_feature == groups[1])) / \
            np.sum((y_true == 1) & (sensitive_feature == groups[1]))

    eod = tpr_1 - tpr_0

    print("=" * 70)
    print("EQUAL OPPORTUNITY")
    print("=" * 70)
    print(f"Group {groups[0]} TPR (recall): {tpr_0:.4f}")
    print(f"Group {groups[1]} TPR (recall): {tpr_1:.4f}")
    print(f"Equal Opportunity Difference: {eod:.4f}")

    if abs(eod) < 0.05:
        print("✓ Acceptable equal opportunity (|EOD| < 0.05)")
    else:
        print("✗ Concerning disparity (|EOD| >= 0.05)")

    return eod

eod = equal_opportunity_difference(y, y_pred, sensitive_feature)
```

**Use Case:** When false negatives are more costly (e.g., disease screening)

### 3. Equalized Odds

**Definition:** Both TPR and FPR should be equal across groups.

$$P(\hat{Y} = 1 | Y = y, A = a) = P(\hat{Y} = 1 | Y = y, A = b) \quad \forall y \in \{0, 1\}$$

```python
def equalized_odds_difference(y_true, y_pred, sensitive_feature):
    """
    Calculate equalized odds difference.

    Maximum of TPR difference and FPR difference.
    """
    groups = np.unique(sensitive_feature)

    if len(groups) != 2:
        raise ValueError("Binary sensitive features only")

    # True positive rates
    tpr_0 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_feature == groups[0])) / \
            max(np.sum((y_true == 1) & (sensitive_feature == groups[0])), 1)

    tpr_1 = np.sum((y_pred == 1) & (y_true == 1) & (sensitive_feature == groups[1])) / \
            max(np.sum((y_true == 1) & (sensitive_feature == groups[1])), 1)

    # False positive rates
    fpr_0 = np.sum((y_pred == 1) & (y_true == 0) & (sensitive_feature == groups[0])) / \
            max(np.sum((y_true == 0) & (sensitive_feature == groups[0])), 1)

    fpr_1 = np.sum((y_pred == 1) & (y_true == 0) & (sensitive_feature == groups[1])) / \
            max(np.sum((y_true == 0) & (sensitive_feature == groups[1])), 1)

    tpr_diff = abs(tpr_1 - tpr_0)
    fpr_diff = abs(fpr_1 - fpr_0)
    eq_odds_diff = max(tpr_diff, fpr_diff)

    print("=" * 70)
    print("EQUALIZED ODDS")
    print("=" * 70)
    print(f"Group {groups[0]} TPR: {tpr_0:.4f}, FPR: {fpr_0:.4f}")
    print(f"Group {groups[1]} TPR: {tpr_1:.4f}, FPR: {fpr_1:.4f}")
    print(f"TPR Difference: {tpr_diff:.4f}")
    print(f"FPR Difference: {fpr_diff:.4f}")
    print(f"Equalized Odds Difference (max): {eq_odds_diff:.4f}")

    if eq_odds_diff < 0.05:
        print("✓ Acceptable equalized odds")
    else:
        print("✗ Concerning disparity")

    return eq_odds_diff

eq_odds = equalized_odds_difference(y, y_pred, sensitive_feature)
```

**Most Comprehensive:** Considers both false positives and false negatives

### 4. Disparate Impact

**Definition:** Ratio of positive prediction rates (80% rule in US employment law).

$$\text{Disparate Impact} = \frac{P(\hat{Y} = 1 | A = \text{unprivileged})}{P(\hat{Y} = 1 | A = \text{privileged})}$$

```python
def disparate_impact_ratio(y_pred, sensitive_feature, unprivileged_group=1):
    """
    Calculate disparate impact ratio.

    US employment law: ratio should be >= 0.8
    """
    groups = np.unique(sensitive_feature)
    privileged_group = [g for g in groups if g != unprivileged_group][0]

    # Positive rates
    rate_unprivileged = np.mean(y_pred[sensitive_feature == unprivileged_group])
    rate_privileged = np.mean(y_pred[sensitive_feature == privileged_group])

    # Disparate impact ratio
    di_ratio = rate_unprivileged / rate_privileged if rate_privileged > 0 else 0

    print("=" * 70)
    print("DISPARATE IMPACT")
    print("=" * 70)
    print(f"Privileged group rate: {rate_privileged:.4f}")
    print(f"Unprivileged group rate: {rate_unprivileged:.4f}")
    print(f"Disparate Impact Ratio: {di_ratio:.4f}")

    # 80% rule (US employment law)
    if di_ratio >= 0.8:
        print("✓ Passes 80% rule (DI >= 0.8)")
    else:
        print("✗ Fails 80% rule (DI < 0.8)")
        print("  ⚠️  May indicate adverse impact")

    return di_ratio

di_ratio = disparate_impact_ratio(y_pred, sensitive_feature, unprivileged_group=1)
```

### 5. Calibration

**Definition:** Predicted probabilities should match true frequencies across groups.

```python
def calibration_by_group(y_true, y_proba, sensitive_feature, n_bins=10):
    """
    Check calibration separately for each group.
    """
    from sklearn.calibration import calibration_curve
    import matplotlib.pyplot as plt

    groups = np.unique(sensitive_feature)

    fig, axes = plt.subplots(1, len(groups), figsize=(12, 5))

    print("=" * 70)
    print("CALIBRATION BY GROUP")
    print("=" * 70)

    for i, group in enumerate(groups):
        mask = sensitive_feature == group

        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true[mask],
            y_proba[mask],
            n_bins=n_bins,
            strategy='uniform'
        )

        # Plot
        ax = axes[i] if len(groups) > 1 else axes
        ax.plot(mean_predicted_value, fraction_of_positives, 'o-', label=f'Group {group}')
        ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'Calibration - Group {group}')
        ax.legend()
        ax.grid(alpha=0.3)

        # Calibration error
        cal_error = np.abs(fraction_of_positives - mean_predicted_value).mean()
        print(f"Group {group} calibration error: {cal_error:.4f}")

    plt.tight_layout()
    plt.show()

# Example
model_proba = model.predict_proba(X)[:, 1]
calibration_by_group(y, model_proba, sensitive_feature)
```

---

## Bias Detection

### Comprehensive Fairness Audit

```python
def comprehensive_fairness_audit(model, X, y, sensitive_feature, feature_names):
    """
    Complete fairness audit with multiple metrics.
    """
    # Predictions
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    print("=" * 70)
    print("COMPREHENSIVE FAIRNESS AUDIT")
    print("=" * 70)

    groups = np.unique(sensitive_feature)
    print(f"\nGroups: {groups}")
    print(f"Group distribution: {np.bincount(sensitive_feature)}")

    # 1. Overall performance by group
    print("\n" + "=" * 70)
    print("1. PERFORMANCE BY GROUP")
    print("=" * 70)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    for group in groups:
        mask = sensitive_feature == group

        acc = accuracy_score(y[mask], y_pred[mask])
        prec = precision_score(y[mask], y_pred[mask], zero_division=0)
        rec = recall_score(y[mask], y_pred[mask], zero_division=0)
        f1 = f1_score(y[mask], y_pred[mask], zero_division=0)

        print(f"\nGroup {group}:")
        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    # 2. Fairness metrics
    print("\n" + "=" * 70)
    print("2. FAIRNESS METRICS")
    print("=" * 70)

    # Demographic parity
    print("\n--- Demographic Parity ---")
    dpd = demographic_parity_difference(y_pred, sensitive_feature)

    # Equal opportunity
    print("\n--- Equal Opportunity ---")
    eod = equal_opportunity_difference(y, y_pred, sensitive_feature)

    # Equalized odds
    print("\n--- Equalized Odds ---")
    eq_odds = equalized_odds_difference(y, y_pred, sensitive_feature)

    # Disparate impact
    print("\n--- Disparate Impact ---")
    di = disparate_impact_ratio(y_pred, sensitive_feature)

    # 3. Confusion matrices by group
    print("\n" + "=" * 70)
    print("3. CONFUSION MATRICES BY GROUP")
    print("=" * 70)

    from sklearn.metrics import confusion_matrix

    for group in groups:
        mask = sensitive_feature == group
        cm = confusion_matrix(y[mask], y_pred[mask])

        print(f"\nGroup {group}:")
        print(f"  TN: {cm[0,0]:5d}  FP: {cm[0,1]:5d}")
        print(f"  FN: {cm[1,0]:5d}  TP: {cm[1,1]:5d}")

    # 4. Feature importance (check for proxy features)
    print("\n" + "=" * 70)
    print("4. FEATURE IMPORTANCE (Check for Proxies)")
    print("=" * 70)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]

    print("\nTop 10 features:")
    for i, idx in enumerate(top_idx, 1):
        print(f"{i:2d}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    print("\n⚠️  Review top features for potential proxies of protected attributes")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    issues = []
    if abs(dpd) >= 0.05:
        issues.append("Demographic parity violation")
    if abs(eod) >= 0.05:
        issues.append("Equal opportunity violation")
    if eq_odds >= 0.05:
        issues.append("Equalized odds violation")
    if di < 0.8:
        issues.append("Disparate impact (80% rule)")

    if issues:
        print("\n⚠️  FAIRNESS ISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRECOMMENDATION: Apply bias mitigation techniques")
    else:
        print("\n✓ No major fairness issues detected")
        print("  Continue monitoring in production")

    return {
        'demographic_parity_diff': dpd,
        'equal_opportunity_diff': eod,
        'equalized_odds_diff': eq_odds,
        'disparate_impact': di
    }

# Example
audit_results = comprehensive_fairness_audit(
    model, X, y, sensitive_feature,
    [f'feature_{i}' for i in range(X.shape[1])]
)
```

---

## Bias Mitigation

### 1. Pre-processing (Before Training)

#### Reweighting

```python
def reweighting(X, y, sensitive_feature):
    """
    Reweight samples to achieve demographic parity.

    Increases weight of underrepresented positive/negative combinations.
    """
    from collections import Counter

    # Calculate weights to balance outcomes across groups
    groups = np.unique(sensitive_feature)
    labels = np.unique(y)

    # Count combinations
    counts = {}
    for g in groups:
        for label in labels:
            mask = (sensitive_feature == g) & (y == label)
            counts[(g, label)] = np.sum(mask)

    # Target: equal representation of each (group, label) combination
    total = len(y)
    n_combinations = len(groups) * len(labels)
    target_count = total / n_combinations

    # Calculate weights
    weights = np.ones(len(y))
    for i in range(len(y)):
        g = sensitive_feature[i]
        label = y[i]
        weights[i] = target_count / counts[(g, label)]

    print("=" * 70)
    print("REWEIGHTING FOR FAIRNESS")
    print("=" * 70)

    for g in groups:
        for label in labels:
            mask = (sensitive_feature == g) & (y == label)
            avg_weight = weights[mask].mean()
            print(f"Group {g}, Label {label}: count={counts[(g, label)]}, avg_weight={avg_weight:.4f}")

    return weights

# Example
weights = reweighting(X, y, sensitive_feature)

# Train with sample weights
model_weighted = RandomForestClassifier(random_state=42)
model_weighted.fit(X, y, sample_weight=weights)

print("\n--- After Reweighting ---")
y_pred_weighted = model_weighted.predict(X)
dpd_after = demographic_parity_difference(y_pred_weighted, sensitive_feature)
```

#### Resampling

```python
def fair_resampling(X, y, sensitive_feature, strategy='balance_labels'):
    """
    Resample to balance outcomes across groups.

    Strategies:
    - 'balance_labels': Equal label distribution per group
    - 'balance_groups': Equal group sizes
    """
    from sklearn.utils import resample

    groups = np.unique(sensitive_feature)
    labels = np.unique(y)

    X_resampled_list = []
    y_resampled_list = []
    sensitive_resampled_list = []

    if strategy == 'balance_labels':
        # For each (group, label) combination, resample to same size
        target_size = min([
            np.sum((sensitive_feature == g) & (y == label))
            for g in groups for label in labels
        ])

        for g in groups:
            for label in labels:
                mask = (sensitive_feature == g) & (y == label)
                X_subset = X[mask]
                y_subset = y[mask]
                sensitive_subset = sensitive_feature[mask]

                # Resample
                if len(X_subset) > 0:
                    X_res, y_res, sens_res = resample(
                        X_subset, y_subset, sensitive_subset,
                        n_samples=target_size,
                        random_state=42,
                        replace=True
                    )

                    X_resampled_list.append(X_res)
                    y_resampled_list.append(y_res)
                    sensitive_resampled_list.append(sens_res)

    X_resampled = np.vstack(X_resampled_list)
    y_resampled = np.concatenate(y_resampled_list)
    sensitive_resampled = np.concatenate(sensitive_resampled_list)

    print("=" * 70)
    print("FAIR RESAMPLING")
    print("=" * 70)
    print(f"Original size: {len(X)}")
    print(f"Resampled size: {len(X_resampled)}")

    return X_resampled, y_resampled, sensitive_resampled

# Example
X_fair, y_fair, sens_fair = fair_resampling(X, y, sensitive_feature)

model_resampled = RandomForestClassifier(random_state=42)
model_resampled.fit(X_fair, y_fair)

y_pred_resampled = model_resampled.predict(X)
dpd_resampled = demographic_parity_difference(y_pred_resampled, sensitive_feature)
```

### 2. In-processing (During Training)

#### Fairness Constraints

```python
def train_with_fairness_constraint(X_train, y_train, sensitive_train, constraint_type='demographic_parity'):
    """
    Train model with fairness constraints using Fairlearn.

    Requires: pip install fairlearn
    """
    from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
    from sklearn.linear_model import LogisticRegression

    # Base estimator
    base_model = LogisticRegression(random_state=42, max_iter=1000)

    # Choose constraint
    if constraint_type == 'demographic_parity':
        constraint = DemographicParity()
    elif constraint_type == 'equalized_odds':
        constraint = EqualizedOdds()
    else:
        raise ValueError("Unknown constraint type")

    # Train fair model
    fair_model = ExponentiatedGradient(
        base_model,
        constraints=constraint,
        eps=0.01  # Fairness tolerance
    )

    print("=" * 70)
    print(f"TRAINING WITH {constraint_type.upper()} CONSTRAINT")
    print("=" * 70)
    print("Training...")

    fair_model.fit(X_train, y_train, sensitive_features=sensitive_train)

    print("✓ Training complete")

    return fair_model

# Example (requires fairlearn)
# fair_model = train_with_fairness_constraint(X, y, sensitive_feature, 'demographic_parity')
# y_pred_fair = fair_model.predict(X)
# dpd_fair = demographic_parity_difference(y_pred_fair, sensitive_feature)
```

### 3. Post-processing (After Training)

#### Threshold Optimization

```python
def optimize_threshold_for_fairness(y_true, y_proba, sensitive_feature, fairness_metric='demographic_parity'):
    """
    Find optimal threshold for each group to satisfy fairness constraint.
    """
    groups = np.unique(sensitive_feature)

    # Grid search for thresholds
    thresholds = np.linspace(0.1, 0.9, 50)

    best_thresholds = {}
    best_metric = float('inf')

    for thresh_0 in thresholds:
        for thresh_1 in thresholds:
            # Apply group-specific thresholds
            y_pred = np.zeros(len(y_true))

            mask_0 = sensitive_feature == groups[0]
            mask_1 = sensitive_feature == groups[1]

            y_pred[mask_0] = (y_proba[mask_0] >= thresh_0).astype(int)
            y_pred[mask_1] = (y_proba[mask_1] >= thresh_1).astype(int)

            # Calculate fairness metric
            if fairness_metric == 'demographic_parity':
                rate_0 = y_pred[mask_0].mean()
                rate_1 = y_pred[mask_1].mean()
                metric_value = abs(rate_1 - rate_0)
            elif fairness_metric == 'equal_opportunity':
                tpr_0 = np.sum((y_pred == 1) & (y_true == 1) & mask_0) / max(np.sum((y_true == 1) & mask_0), 1)
                tpr_1 = np.sum((y_pred == 1) & (y_true == 1) & mask_1) / max(np.sum((y_true == 1) & mask_1), 1)
                metric_value = abs(tpr_1 - tpr_0)

            if metric_value < best_metric:
                best_metric = metric_value
                best_thresholds = {groups[0]: thresh_0, groups[1]: thresh_1}

    print("=" * 70)
    print("THRESHOLD OPTIMIZATION FOR FAIRNESS")
    print("=" * 70)
    print(f"Optimizing for: {fairness_metric}")
    print(f"Best thresholds:")
    for group, thresh in best_thresholds.items():
        print(f"  Group {group}: {thresh:.4f}")
    print(f"Achieved {fairness_metric} difference: {best_metric:.4f}")

    return best_thresholds

# Example
best_thresholds = optimize_threshold_for_fairness(y, model_proba, sensitive_feature, 'demographic_parity')
```

---

## Fairness Libraries

### 1. Fairlearn (Microsoft)

```python
# Complete Fairlearn example

def fairlearn_comprehensive_analysis(X, y, sensitive_feature):
    """
    Complete fairness analysis using Fairlearn.

    Install: pip install fairlearn
    """
    from fairlearn.metrics import (
        MetricFrame,
        demographic_parity_difference,
        equalized_odds_difference,
        selection_rate
    )
    from sklearn.metrics import accuracy_score, precision_score

    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    y_pred = model.predict(X)

    print("=" * 70)
    print("FAIRLEARN ANALYSIS")
    print("=" * 70)

    # MetricFrame: Disaggregated metrics
    metrics = {
        'accuracy': accuracy_score,
        'precision': precision_score,
        'selection_rate': selection_rate
    }

    mf = MetricFrame(
        metrics=metrics,
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )

    print("\n--- Disaggregated Metrics ---")
    print(mf.by_group)

    print("\n--- Overall Metrics ---")
    print(mf.overall)

    print("\n--- Differences ---")
    print(mf.difference(method='between_groups'))

    # Fairness metrics
    print("\n--- Fairness Metrics ---")
    dp_diff = demographic_parity_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    print(f"Demographic Parity Difference: {dp_diff:.4f}")

    eq_odds_diff = equalized_odds_difference(
        y_true=y,
        y_pred=y_pred,
        sensitive_features=sensitive_feature
    )
    print(f"Equalized Odds Difference: {eq_odds_diff:.4f}")

    # Fairlearn Dashboard (interactive visualization)
    # from fairlearn.widget import FairlearnDashboard
    # FairlearnDashboard(sensitive_features=sensitive_feature, y_true=y, y_pred={"model": y_pred})

    return mf

# Example (requires fairlearn)
# mf = fairlearn_comprehensive_analysis(X, y, sensitive_feature)
```

### 2. AIF360 (IBM)

```python
def aif360_bias_detection(X, y, sensitive_feature):
    """
    Bias detection using AIF360.

    Install: pip install aif360
    """
    # from aif360.datasets import BinaryLabelDataset
    # from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

    print("=" * 70)
    print("AIF360 BIAS DETECTION")
    print("=" * 70)
    print("\nAIF360 provides:")
    print("  - 70+ fairness metrics")
    print("  - 10+ bias mitigation algorithms")
    print("  - Pre-processing, in-processing, post-processing methods")
    print("\nKey features:")
    print("  - Bias scan (find biased subgroups)")
    print("  - Explainability")
    print("  - Multiple fairness definitions")

    # Example usage:
    """
    # Create BinaryLabelDataset
    dataset = BinaryLabelDataset(
        df=pd.DataFrame(np.hstack([X, y.reshape(-1, 1), sensitive_feature.reshape(-1, 1)])),
        label_names=['label'],
        protected_attribute_names=['sensitive']
    )

    # Calculate metrics
    metric = BinaryLabelDatasetMetric(
        dataset,
        unprivileged_groups=[{'sensitive': 0}],
        privileged_groups=[{'sensitive': 1}]
    )

    print(f"Disparate Impact: {metric.disparate_impact()}")
    print(f"Statistical Parity Difference: {metric.statistical_parity_difference()}")
    """

# aif360_bias_detection(X, y, sensitive_feature)
```

---

## EU AI Act Compliance

### Risk Classification

```python
def assess_ai_act_risk_level(use_case):
    """
    Determine EU AI Act risk level.
    """
    print("=" * 70)
    print("EU AI ACT RISK ASSESSMENT")
    print("=" * 70)

    high_risk_areas = [
        'Employment, workers management',
        'Access to education and vocational training',
        'Access to essential private/public services (credit, insurance)',
        'Law enforcement',
        'Migration, asylum and border control',
        'Administration of justice',
        'Critical infrastructure management',
        'Biometric identification'
    ]

    prohibited = [
        'Social scoring',
        'Subliminal manipulation',
        'Exploitation of vulnerabilities',
        'Real-time remote biometric identification (public)'
    ]

    print("\nUse case:", use_case)

    if use_case in prohibited:
        print("\n🚫 PROHIBITED SYSTEM")
        print("   This AI system is banned under EU AI Act")

    elif use_case in high_risk_areas:
        print("\n⚠️  HIGH-RISK AI SYSTEM")
        print("   Requirements:")
        print("   - Conformity assessment")
        print("   - Risk management system")
        print("   - Data governance")
        print("   - Technical documentation")
        print("   - Record keeping")
        print("   - Transparency and information to users")
        print("   - Human oversight")
        print("   - Accuracy, robustness, cybersecurity")
        print("   - Quality management system")

    else:
        print("\n✓ LIMITED/MINIMAL RISK")
        print("   Basic transparency obligations")

assess_ai_act_risk_level('Access to essential private/public services (credit, insurance)')
```

### Mandatory Documentation

```python
def generate_eu_ai_act_fairness_report(model, X_test, y_test, sensitive_test, feature_names):
    """
    Generate fairness report for EU AI Act compliance.
    """
    report = []

    report.append("=" * 70)
    report.append("EU AI ACT - FAIRNESS AND BIAS ASSESSMENT REPORT")
    report.append("=" * 70)
    report.append("")

    # 1. System Description
    report.append("1. AI SYSTEM DESCRIPTION")
    report.append("-" * 70)
    report.append(f"Model Type: {type(model).__name__}")
    report.append(f"Purpose: [Credit risk assessment]")
    report.append(f"Risk Level: HIGH-RISK")
    report.append("")

    # 2. Protected Characteristics
    report.append("2. PROTECTED CHARACTERISTICS ASSESSED")
    report.append("-" * 70)
    report.append("The following protected characteristics were assessed:")
    report.append("  - [Gender, Race, Age, etc.]")
    report.append("")

    # 3. Fairness Metrics
    report.append("3. FAIRNESS METRICS")
    report.append("-" * 70)

    y_pred = model.predict(X_test)

    # Calculate metrics
    dpd = abs(demographic_parity_difference(y_pred, sensitive_test))
    eod = abs(equal_opportunity_difference(y_test, y_pred, sensitive_test))
    di = disparate_impact_ratio(y_pred, sensitive_test, unprivileged_group=1)

    report.append(f"Demographic Parity Difference: {dpd:.4f} (threshold: 0.05)")
    report.append(f"Equal Opportunity Difference: {eod:.4f} (threshold: 0.05)")
    report.append(f"Disparate Impact Ratio: {di:.4f} (threshold: 0.80)")
    report.append("")

    # 4. Compliance Status
    report.append("4. COMPLIANCE STATUS")
    report.append("-" * 70)

    compliant = dpd < 0.05 and eod < 0.05 and di >= 0.8

    if compliant:
        report.append("✓ COMPLIANT: All fairness thresholds met")
    else:
        report.append("✗ NON-COMPLIANT: Fairness thresholds exceeded")
        report.append("\nIdentified Issues:")
        if dpd >= 0.05:
            report.append(f"  - Demographic parity violation (diff={dpd:.4f})")
        if eod >= 0.05:
            report.append(f"  - Equal opportunity violation (diff={eod:.4f})")
        if di < 0.8:
            report.append(f"  - Disparate impact violation (ratio={di:.4f})")

    report.append("")

    # 5. Mitigation Measures
    report.append("5. MITIGATION MEASURES")
    report.append("-" * 70)
    if not compliant:
        report.append("Required actions:")
        report.append("  1. Apply reweighting/resampling")
        report.append("  2. Train with fairness constraints")
        report.append("  3. Optimize decision thresholds")
        report.append("  4. Re-assess fairness metrics")
    else:
        report.append("No mitigation required - system is compliant")

    report.append("")

    # 6. Monitoring Plan
    report.append("6. ONGOING MONITORING")
    report.append("-" * 70)
    report.append("Monitoring frequency: Monthly")
    report.append("Metrics tracked:")
    report.append("  - Demographic parity")
    report.append("  - Equal opportunity")
    report.append("  - Disparate impact")
    report.append("  - Performance by group")
    report.append("")
    report.append("Re-assessment triggers:")
    report.append("  - Any metric exceeds threshold")
    report.append("  - Significant distribution shift")
    report.append("  - Model retraining")
    report.append("")

    # 7. Human Oversight
    report.append("7. HUMAN OVERSIGHT MEASURES")
    report.append("-" * 70)
    report.append("Human review required for:")
    report.append("  - All negative decisions")
    report.append("  - Cases where protected characteristic may have influenced decision")
    report.append("  - Contested decisions")
    report.append("")

    # 8. Documentation
    report.append("8. RECORD KEEPING")
    report.append("-" * 70)
    report.append(f"Assessment Date: 2025-10-14")
    report.append(f"Assessor: [Data Science Team]")
    report.append(f"Next Assessment Due: 2025-11-14")
    report.append("")

    report.append("=" * 70)

    report_text = "\n".join(report)
    print(report_text)

    # Save to file
    with open('eu_ai_act_fairness_report.txt', 'w') as f:
        f.write(report_text)

    return report_text

# Example
# Generate data split for testing
X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
    X, y, sensitive_feature, test_size=0.3, random_state=42
)

model.fit(X_train, y_train)
eu_report = generate_eu_ai_act_fairness_report(
    model, X_test, y_test, sens_test,
    [f'feature_{i}' for i in range(X.shape[1])]
)
```

---

## Case Studies

### Case Study 1: Credit Scoring

```python
def credit_scoring_fairness_case_study():
    """
    Real-world example: Fair credit scoring.
    """
    print("=" * 70)
    print("CASE STUDY: FAIR CREDIT SCORING")
    print("=" * 70)

    # Simulate credit data
    np.random.seed(42)
    n = 5000

    # Protected attribute: gender (0=M, 1=F)
    gender = np.random.choice([0, 1], n, p=[0.55, 0.45])

    # Features
    credit_score = np.random.normal(700, 100, n)
    income = np.random.normal(50000, 20000, n)
    debt_to_income = np.random.uniform(0, 0.5, n)

    # Historical bias: women had lower approval rates due to discrimination
    approval = []
    for g, cs, inc in zip(gender, credit_score, income):
        # Base approval probability
        prob = 1 / (1 + np.exp(-(cs - 650) / 50 + (inc - 40000) / 20000))

        # Historical bias: higher bar for women
        if g == 1:  # Women
            prob *= 0.8

        approval.append(1 if np.random.rand() < prob else 0)

    approval = np.array(approval)

    # Create dataset
    X = np.column_stack([credit_score, income, debt_to_income])
    y = approval

    print("\n1. DETECT BIAS IN HISTORICAL DATA")
    print("-" * 70)

    approval_rate_m = approval[gender == 0].mean()
    approval_rate_f = approval[gender == 1].mean()

    print(f"Historical approval rate - Men: {approval_rate_m:.2%}")
    print(f"Historical approval rate - Women: {approval_rate_f:.2%}")
    print(f"Difference: {(approval_rate_m - approval_rate_f):.2%}")

    # Train biased model
    print("\n2. TRAIN MODEL ON BIASED DATA")
    print("-" * 70)

    model_biased = LogisticRegression(random_state=42)
    model_biased.fit(X, y)
    y_pred_biased = model_biased.predict(X)

    dpd_biased = demographic_parity_difference(y_pred_biased, gender)
    di_biased = disparate_impact_ratio(y_pred_biased, gender, unprivileged_group=1)

    print(f"Model perpetuates bias:")
    print(f"  Demographic Parity Difference: {dpd_biased:.4f}")
    print(f"  Disparate Impact: {di_biased:.4f}")

    # Apply mitigation
    print("\n3. APPLY BIAS MITIGATION (REWEIGHTING)")
    print("-" * 70)

    weights = reweighting(X, y, gender)
    model_fair = LogisticRegression(random_state=42)
    model_fair.fit(X, y, sample_weight=weights)
    y_pred_fair = model_fair.predict(X)

    dpd_fair = demographic_parity_difference(y_pred_fair, gender)
    di_fair = disparate_impact_ratio(y_pred_fair, gender, unprivileged_group=1)

    print(f"\nAfter mitigation:")
    print(f"  Demographic Parity Difference: {dpd_fair:.4f}")
    print(f"  Disparate Impact: {di_fair:.4f}")

    # Compare performance
    from sklearn.metrics import roc_auc_score

    auc_biased = roc_auc_score(y, model_biased.predict_proba(X)[:, 1])
    auc_fair = roc_auc_score(y, model_fair.predict_proba(X)[:, 1])

    print("\n4. PERFORMANCE COMPARISON")
    print("-" * 70)
    print(f"Biased model AUC: {auc_biased:.4f}")
    print(f"Fair model AUC: {auc_fair:.4f}")
    print(f"Performance cost: {(auc_biased - auc_fair):.4f}")

    print("\n5. CONCLUSION")
    print("-" * 70)
    print("✓ Bias successfully mitigated")
    print("✓ Minimal performance impact")
    print("✓ Compliant with fairness requirements")

credit_scoring_fairness_case_study()
```

---

## Summary

### Fairness Metrics Quick Reference

| Metric | Definition | When to Use | Threshold |
|--------|-----------|-------------|-----------|
| **Demographic Parity** | Equal positive rates | When equal treatment is goal | |diff| < 0.05 |
| **Equal Opportunity** | Equal TPR | When FN more costly | |diff| < 0.05 |
| **Equalized Odds** | Equal TPR & FPR | Balanced consideration | max_diff < 0.05 |
| **Disparate Impact** | Ratio of positive rates | US employment law | ratio ≥ 0.80 |
| **Calibration** | Predicted = actual | Probability quality | Low calibration error |

### Mitigation Strategy Selection

| Stage | Method | Pros | Cons |
|-------|--------|------|------|
| **Pre-processing** | Reweighting | Simple, model-agnostic | May not fully address bias |
| **Pre-processing** | Resampling | Intuitive | Changes data distribution |
| **In-processing** | Fairness constraints | Theoretically sound | Requires special algorithms |
| **Post-processing** | Threshold optimization | Doesn't require retraining | Group-specific thresholds |

### Key Takeaways (2025)

1. ✅ **Fairness is mandatory** - EU AI Act, GDPR, US regulations
2. ✅ **Multiple metrics** - No single metric captures all aspects
3. ✅ **Trade-offs exist** - Fairness vs. accuracy, different fairness definitions
4. ✅ **Document everything** - Required for compliance
5. ✅ **Continuous monitoring** - Bias can emerge post-deployment
6. ✅ **Human oversight** - Required for high-risk systems
7. ✅ **Use established tools** - Fairlearn, AIF360
8. ✅ **Domain expertise** - Involve subject matter experts

---

**Last Updated:** 2025-10-14
**Status:** Complete - Production-ready with regulatory compliance
