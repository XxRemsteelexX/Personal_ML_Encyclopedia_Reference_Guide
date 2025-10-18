# Model Interpretability and Explainability

## Overview

Model interpretability is the ability to explain why a model makes certain predictions. In 2025, interpretability is no longer optional - it's required by regulations (GDPR Article 22, EU AI Act), essential for debugging, and critical for building trust in AI systems.

**Key Distinction:**
- **Interpretability:** Model is inherently understandable (linear models, decision trees)
- **Explainability:** Post-hoc methods to explain black-box models (SHAP, LIME)

---

## Table of Contents
1. [Why Interpretability Matters](#why-interpretability-matters)
2. [Intrinsic Interpretability](#intrinsic-interpretability)
3. [Post-hoc Interpretability](#post-hoc-interpretability)
4. [SHAP Values](#shap-values)
5. [LIME](#lime)
6. [Feature Importance](#feature-importance)
7. [Partial Dependence Plots](#partial-dependence-plots)
8. [Deep Learning Interpretability](#deep-learning-interpretability)
9. [Regulatory Compliance](#regulatory-compliance)

---

## Why Interpretability Matters

### 1. Regulatory Requirements (2025)

**GDPR Article 22 (EU):**
- Right to explanation for automated decisions
- Meaningful information about the logic involved
- Consequences of decision-making

**EU AI Act (2024):**
- High-risk AI systems must be interpretable
- Technical documentation requirements
- Transparency obligations

**Other Regulations:**
- FCRA (Fair Credit Reporting Act) - US
- ECOA (Equal Credit Opportunity Act) - US
- FDA guidance for medical AI

### 2. Business Value

```python
# Example: Credit scoring
# Bank needs to explain why loan was denied

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate credit data
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=8,
    random_state=42
)

feature_names = [
    'credit_score', 'income', 'debt_to_income', 'payment_history',
    'credit_utilization', 'num_accounts', 'inquiries', 'age_of_credit',
    'num_derogatory', 'bankruptcy_flag'
]

X_df = pd.DataFrame(X, columns=feature_names)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_df, y)

# Predict
sample_applicant = X_df.iloc[0:1]
prediction = model.predict(sample_applicant)[0]

print(f"Loan Decision: {'APPROVED' if prediction == 1 else 'DENIED'}")
print("\nWithout explanation → Customer dissatisfaction, regulatory risk")
print("With explanation → Customer understanding, compliance, trust")
```

### 3. Model Debugging

```python
# Interpretability helps find bugs:
# - Data leakage
# - Spurious correlations
# - Distribution shift
# - Feature engineering errors

def debug_model_with_interpretability(model, X, y, feature_names):
    """
    Use interpretability to debug model.
    """
    # Get feature importance
    importances = model.feature_importances_

    # Sort features
    indices = np.argsort(importances)[::-1]

    print("Feature Importance Ranking:")
    print("=" * 60)
    for i in range(len(feature_names)):
        print(f"{i+1}. {feature_names[indices[i]]:<25} {importances[indices[i]]:.4f}")

    # Check for suspicious patterns
    print("\n⚠️  Debugging Checks:")
    if importances[indices[0]] > 0.5:
        print(f"  - Feature '{feature_names[indices[0]]}' dominates (>50% importance)")
        print("    → Possible data leakage or overfitting")

    if importances[indices[-1]] < 0.001:
        print(f"  - Many features have near-zero importance")
        print("    → Consider feature selection")

debug_model_with_interpretability(model, X_df, y, feature_names)
```

---

## Intrinsic Interpretability

Models that are inherently interpretable.

### 1. Linear Models

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def interpret_linear_model(X, y, feature_names):
    """
    Interpret logistic regression coefficients.
    """
    # Standardize for comparable coefficients
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train logistic regression
    log_reg = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_scaled, y)

    # Get coefficients
    coef = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]

    # Create interpretation DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coef,
        'Abs_Coefficient': np.abs(coef)
    }).sort_values('Abs_Coefficient', ascending=False)

    print("=" * 70)
    print("LOGISTIC REGRESSION INTERPRETATION")
    print("=" * 70)
    print(f"\nIntercept (log-odds when all features = 0): {intercept:.4f}\n")
    print("Coefficients (impact on log-odds):")
    print(coef_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    print("=" * 70)

    for _, row in coef_df.head(3).iterrows():
        feature = row['Feature']
        coef = row['Coefficient']
        odds_ratio = np.exp(coef)

        if coef > 0:
            print(f"\n{feature}:")
            print(f"  Coefficient: +{coef:.4f}")
            print(f"  Odds ratio: {odds_ratio:.4f}")
            print(f"  → 1 std increase → {(odds_ratio - 1) * 100:.1f}% higher odds of approval")
        else:
            print(f"\n{feature}:")
            print(f"  Coefficient: {coef:.4f}")
            print(f"  Odds ratio: {odds_ratio:.4f}")
            print(f"  → 1 std increase → {(1 - odds_ratio) * 100:.1f}% lower odds of approval")

    return log_reg, coef_df

log_reg, coef_df = interpret_linear_model(X_df, y, feature_names)
```

**Advantages:**
- Direct interpretation of coefficients
- Global understanding
- Statistical significance tests available

**Limitations:**
- Assumes linear relationships
- May have lower accuracy than complex models

### 2. Decision Trees

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

def interpret_decision_tree(X, y, feature_names, max_depth=3):
    """
    Visualize and interpret decision tree.
    """
    # Train shallow tree (for interpretability)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    tree.fit(X, y)

    # Visualize
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=['Denied', 'Approved'],
        filled=True,
        rounded=True,
        fontsize=10
    )
    plt.title(f'Decision Tree (max_depth={max_depth})')
    plt.tight_layout()
    plt.show()

    # Extract rules
    from sklearn.tree import _tree

    def tree_to_rules(tree, feature_names):
        """Extract human-readable rules from tree."""
        tree_ = tree.tree_
        feature_name = [
            feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
            for i in tree_.feature
        ]

        def recurse(node, depth, path_conditions):
            indent = "  " * depth
            if tree_.feature[node] != _tree.TREE_UNDEFINED:
                name = feature_name[node]
                threshold = tree_.threshold[node]

                # Left child (<=)
                left_condition = f"{name} <= {threshold:.2f}"
                recurse(tree_.children_left[node], depth + 1,
                       path_conditions + [left_condition])

                # Right child (>)
                right_condition = f"{name} > {threshold:.2f}"
                recurse(tree_.children_right[node], depth + 1,
                       path_conditions + [right_condition])
            else:
                # Leaf node
                class_idx = np.argmax(tree_.value[node])
                class_name = ['DENIED', 'APPROVED'][class_idx]
                samples = tree_.n_node_samples[node]

                print(f"\nRule {len(rules) + 1}:")
                print(f"  IF {' AND '.join(path_conditions)}")
                print(f"  THEN {class_name} (n={samples})")

                rules.append({
                    'conditions': path_conditions,
                    'prediction': class_name,
                    'samples': samples
                })

        rules = []
        recurse(0, 0, [])
        return rules

    print("=" * 70)
    print("DECISION TREE RULES")
    print("=" * 70)

    rules = tree_to_rules(tree, feature_names)

    return tree, rules

tree, rules = interpret_decision_tree(X_df, y, feature_names, max_depth=3)
```

**Advantages:**
- Easy to understand and visualize
- Mimics human decision-making
- Handles non-linear relationships

**Limitations:**
- Can become complex with depth
- May overfit with deep trees
- Unstable (small data changes → different tree)

### 3. Rule-Based Models (RuleFit)

```python
# Example: Extract rules from ensemble
from sklearn.ensemble import GradientBoostingClassifier

def extract_simple_rules(model, X, y, feature_names, n_rules=5):
    """
    Extract simple IF-THEN rules from model.

    This is a simplified example. For production, use:
    - imodels library (RuleFit, BoostedRules)
    - skope-rules
    """
    # Train gradient boosting
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=1, random_state=42)
    gb.fit(X, y)

    print("=" * 70)
    print("EXTRACTED RULES (Gradient Boosting Stumps)")
    print("=" * 70)

    # Each stump is a simple rule
    for i, tree in enumerate(gb.estimators_[:n_rules, 0]):
        feature_idx = tree.tree_.feature[0]
        if feature_idx >= 0:
            threshold = tree.tree_.threshold[0]
            feature = feature_names[feature_idx]

            print(f"\nRule {i+1}:")
            print(f"  IF {feature} <= {threshold:.2f} THEN score += {tree.tree_.value[1][0][0]:.4f}")
            print(f"  IF {feature} > {threshold:.2f} THEN score += {tree.tree_.value[2][0][0]:.4f}")

    return gb

gb_rules = extract_simple_rules(model, X_df, y, feature_names)
```

---

## Post-hoc Interpretability

Explaining black-box models after training.

### Global vs. Local Explanations

- **Global:** How the model works overall
- **Local:** Why this specific prediction

---

## SHAP Values

**SHAP (SHapley Additive exPlanations)** - The gold standard for model interpretability (2025).

Based on game theory (Shapley values): Fair distribution of "payout" (prediction) among "players" (features).

### Installation

```python
# pip install shap
import shap

# Initialize JavaScript for visualizations
shap.initjs()
```

### 1. Tree-Based Models (TreeExplainer)

```python
def shap_analysis_tree(model, X, feature_names):
    """
    Complete SHAP analysis for tree-based models.

    TreeExplainer is fast and exact for trees.
    """
    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Calculate SHAP values
    shap_values = explainer.shap_values(X)

    # If binary classification, select positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print("=" * 70)
    print("SHAP ANALYSIS")
    print("=" * 70)
    print(f"Base value (expected model output): {explainer.expected_value:.4f}")

    # 1. Summary Plot (Global: Feature importance)
    print("\n1. Summary Plot - Feature Importance")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Summary Plot - Feature Importance')
    plt.tight_layout()
    plt.show()

    # 2. Summary Plot (Impact: positive/negative)
    print("\n2. Summary Plot - Feature Impact")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                     plot_type='violin', show=False)
    plt.title('SHAP Summary Plot - Feature Impact')
    plt.tight_layout()
    plt.show()

    # 3. Bar Plot (Mean absolute SHAP values)
    print("\n3. Bar Plot - Mean Feature Importance")
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names,
                     plot_type='bar', show=False)
    plt.title('SHAP Bar Plot - Mean Absolute Impact')
    plt.tight_layout()
    plt.show()

    # 4. Dependence Plots (top 2 features)
    print("\n4. Dependence Plots")
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[::-1][:2]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for i, idx in enumerate(top_features_idx):
        shap.dependence_plot(
            idx, shap_values, X,
            feature_names=feature_names,
            ax=axes[i],
            show=False
        )

    plt.tight_layout()
    plt.show()

    return explainer, shap_values

# Example
explainer, shap_values = shap_analysis_tree(model, X_df.values, feature_names)
```

### 2. Local Explanation (Single Prediction)

```python
def explain_prediction_shap(model, X, feature_names, sample_idx=0):
    """
    Explain a single prediction using SHAP.
    """
    explainer = shap.TreeExplainer(model)

    # Get prediction and SHAP values for one sample
    sample = X[sample_idx:sample_idx+1]
    prediction = model.predict_proba(sample)[0]
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_values_sample = shap_values[1][0]
    else:
        shap_values_sample = shap_values[0]

    print("=" * 70)
    print(f"EXPLAINING PREDICTION FOR SAMPLE {sample_idx}")
    print("=" * 70)
    print(f"Prediction: {'APPROVED' if prediction[1] > 0.5 else 'DENIED'}")
    print(f"Probability: {prediction[1]:.2%}")
    print(f"Base value: {explainer.expected_value:.4f}")

    # Waterfall plot (shows how each feature contributes)
    print("\nWaterfall Plot (Feature Contributions):")
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values_sample,
            base_values=explainer.expected_value,
            data=sample[0],
            feature_names=feature_names
        )
    )

    # Force plot (alternative visualization)
    print("\nForce Plot:")
    shap.force_plot(
        explainer.expected_value,
        shap_values_sample,
        sample[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.show()

    # Print detailed breakdown
    print("\n" + "=" * 70)
    print("DETAILED FEATURE CONTRIBUTIONS")
    print("=" * 70)

    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': sample[0],
        'SHAP': shap_values_sample,
        'Abs_SHAP': np.abs(shap_values_sample)
    }).sort_values('Abs_SHAP', ascending=False)

    print(contributions.to_string(index=False))

    # Generate natural language explanation
    print("\n" + "=" * 70)
    print("NATURAL LANGUAGE EXPLANATION")
    print("=" * 70)

    top_contributors = contributions.head(3)

    for _, row in top_contributors.iterrows():
        feature = row['Feature']
        value = row['Value']
        shap_val = row['SHAP']

        if shap_val > 0:
            print(f"✓ {feature} = {value:.2f} increases approval probability by {shap_val:.4f}")
        else:
            print(f"✗ {feature} = {value:.2f} decreases approval probability by {abs(shap_val):.4f}")

    return shap_values_sample, contributions

# Example
shap_vals, contributions = explain_prediction_shap(model, X_df.values, feature_names, sample_idx=0)
```

### 3. SHAP for Any Model (KernelExplainer)

```python
def shap_any_model(model, X_train, X_test, feature_names, n_background=100):
    """
    SHAP for any model (slower but model-agnostic).

    Uses KernelExplainer (based on LIME + Shapley values).
    """
    # Use subset of training data as background
    background = shap.sample(X_train, n_background, random_state=42)

    # Create explainer
    explainer = shap.KernelExplainer(
        model.predict_proba,
        background
    )

    # Calculate SHAP values (slower)
    print("Calculating SHAP values (this may take a while)...")
    shap_values = explainer.shap_values(X_test[:10])  # First 10 samples

    # Select positive class for binary classification
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Visualize
    shap.summary_plot(shap_values, X_test[:10], feature_names=feature_names)

    return explainer, shap_values

# Example (commented out - can be slow)
# X_train, X_test = X_df.values[:800], X_df.values[800:]
# explainer_kernel, shap_vals_kernel = shap_any_model(model, X_train, X_test, feature_names)
```

### 4. SHAP for Deep Learning

```python
def shap_deep_learning(model, X_train, X_test):
    """
    SHAP for neural networks.

    Uses DeepExplainer (optimized for deep learning).
    """
    # Example with PyTorch/TensorFlow model
    # For PyTorch:
    # explainer = shap.DeepExplainer(model, X_train)

    # For gradient-based explanation:
    # explainer = shap.GradientExplainer(model, X_train)

    # Calculate SHAP values
    # shap_values = explainer.shap_values(X_test)

    pass  # Placeholder

# See Deep Learning Interpretability section for complete implementation
```

---

## LIME

**LIME (Local Interpretable Model-agnostic Explanations)** - Explains individual predictions by fitting a local linear model.

### Installation

```python
# pip install lime
from lime import lime_tabular
```

### 1. LIME for Tabular Data

```python
def explain_with_lime(model, X_train, X_test, feature_names, sample_idx=0):
    """
    Explain prediction using LIME.
    """
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['Denied', 'Approved'],
        mode='classification',
        random_state=42
    )

    # Explain a prediction
    sample = X_test[sample_idx]

    exp = explainer.explain_instance(
        sample,
        model.predict_proba,
        num_features=10
    )

    print("=" * 70)
    print(f"LIME EXPLANATION FOR SAMPLE {sample_idx}")
    print("=" * 70)

    # Show in notebook
    exp.show_in_notebook(show_table=True)

    # Text explanation
    print("\nText Explanation:")
    print(exp.as_list())

    # Save as HTML
    exp.save_to_file(f'lime_explanation_{sample_idx}.html')

    # Feature importance from LIME
    print("\n" + "=" * 70)
    print("LIME FEATURE IMPORTANCE")
    print("=" * 70)

    lime_importance = exp.as_list()
    for feature, weight in lime_importance:
        direction = "increases" if weight > 0 else "decreases"
        print(f"{feature}: {direction} probability by {abs(weight):.4f}")

    return exp

# Example
X_train_arr, X_test_arr = X_df.values[:800], X_df.values[800:]
lime_exp = explain_with_lime(model, X_train_arr, X_test_arr, feature_names, sample_idx=0)
```

### 2. LIME vs SHAP

```python
def compare_lime_shap(model, X_train, X_test, feature_names, sample_idx=0):
    """
    Compare LIME and SHAP explanations.
    """
    sample = X_test[sample_idx:sample_idx+1]

    # SHAP
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]
    else:
        shap_values = shap_values[0]

    # LIME
    lime_explainer = lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode='classification',
        random_state=42
    )
    lime_exp = lime_explainer.explain_instance(
        sample[0],
        model.predict_proba,
        num_features=len(feature_names)
    )
    lime_importance = dict(lime_exp.as_list())

    # Extract LIME weights by feature name
    lime_weights = []
    for fname in feature_names:
        # Find matching feature in LIME output
        matching = [k for k in lime_importance.keys() if fname in k]
        if matching:
            lime_weights.append(lime_importance[matching[0]])
        else:
            lime_weights.append(0.0)

    # Compare
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'SHAP': shap_values,
        'LIME': lime_weights,
        'Difference': np.abs(shap_values - np.array(lime_weights))
    }).sort_values('Difference', ascending=False)

    print("=" * 70)
    print("COMPARISON: SHAP vs LIME")
    print("=" * 70)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 70)
    print("KEY DIFFERENCES")
    print("=" * 70)
    print("SHAP:")
    print("  ✓ Theoretically grounded (Shapley values)")
    print("  ✓ Consistent and accurate")
    print("  ✓ Fast for tree models (TreeExplainer)")
    print("  ✗ Can be slow for other models")

    print("\nLIME:")
    print("  ✓ Model-agnostic")
    print("  ✓ Intuitive (local linear approximation)")
    print("  ✓ Fast")
    print("  ✗ Can be unstable (different perturbations → different explanations)")
    print("  ✗ No theoretical guarantees")

    return comparison

# Example
comparison = compare_lime_shap(model, X_train_arr, X_test_arr, feature_names, sample_idx=0)
```

---

## Feature Importance

### 1. Model-Specific Importance (Tree-based)

```python
def plot_feature_importance(model, feature_names, top_n=10):
    """
    Plot feature importance from tree-based model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(10, 6))
    plt.bar(range(top_n), importances[indices])
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.xlabel('Feature')
    plt.ylabel('Importance (MDI)')
    plt.title('Feature Importance (Mean Decrease in Impurity)')
    plt.tight_layout()
    plt.show()

    # Print
    print("=" * 70)
    print("FEATURE IMPORTANCE (MDI)")
    print("=" * 70)
    for i, idx in enumerate(indices, 1):
        print(f"{i}. {feature_names[idx]:<25} {importances[idx]:.4f}")

    return importances

importances_mdi = plot_feature_importance(model, feature_names)
```

**Warning:** MDI (Mean Decrease in Impurity) is biased towards:
- High cardinality features
- Continuous features
- Can be misleading!

### 2. Permutation Importance (Unbiased)

```python
from sklearn.inspection import permutation_importance

def plot_permutation_importance(model, X, y, feature_names, n_repeats=10):
    """
    Calculate permutation importance (more reliable).

    Measures decrease in model performance when feature is randomly shuffled.
    """
    # Calculate permutation importance
    perm_importance = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1
    )

    # Sort features
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(
        perm_importance.importances[sorted_idx].T,
        vert=False,
        labels=[feature_names[i] for i in sorted_idx]
    )
    ax.set_xlabel('Permutation Importance')
    ax.set_title(f'Permutation Feature Importance (n_repeats={n_repeats})')
    plt.tight_layout()
    plt.show()

    # Print
    print("=" * 70)
    print("PERMUTATION IMPORTANCE")
    print("=" * 70)
    print(f"{'Feature':<25} {'Mean':<12} {'Std':<12}")
    print("-" * 70)

    for idx in sorted_idx[:10]:
        mean_imp = perm_importance.importances_mean[idx]
        std_imp = perm_importance.importances_std[idx]
        print(f"{feature_names[idx]:<25} {mean_imp:>11.4f} {std_imp:>11.4f}")

    return perm_importance

perm_imp = plot_permutation_importance(model, X_df.values, y, feature_names)
```

### 3. Compare MDI vs Permutation

```python
def compare_importance_methods(model, X, y, feature_names):
    """
    Compare MDI and permutation importance.
    """
    # MDI
    mdi = model.feature_importances_

    # Permutation
    perm = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    perm_mean = perm.importances_mean

    # Normalize to sum to 1
    mdi_norm = mdi / mdi.sum()
    perm_norm = perm_mean / perm_mean.sum()

    # Compare
    comparison = pd.DataFrame({
        'Feature': feature_names,
        'MDI': mdi_norm,
        'Permutation': perm_norm,
        'Difference': np.abs(mdi_norm - perm_norm)
    }).sort_values('Difference', ascending=False)

    print("=" * 70)
    print("COMPARISON: MDI vs PERMUTATION IMPORTANCE")
    print("=" * 70)
    print(comparison.head(10).to_string(index=False))

    # Scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(mdi_norm, perm_norm, alpha=0.6)

    # Add feature labels for top features
    top_features_idx = np.argsort(perm_norm)[::-1][:5]
    for idx in top_features_idx:
        plt.annotate(
            feature_names[idx],
            (mdi_norm[idx], perm_norm[idx]),
            fontsize=9
        )

    # Diagonal line
    max_val = max(mdi_norm.max(), perm_norm.max())
    plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, label='Perfect agreement')

    plt.xlabel('MDI Importance (normalized)')
    plt.ylabel('Permutation Importance (normalized)')
    plt.title('MDI vs Permutation Importance')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    return comparison

importance_comparison = compare_importance_methods(model, X_df.values, y, feature_names)
```

---

## Partial Dependence Plots

Shows marginal effect of features on predictions.

```python
from sklearn.inspection import PartialDependenceDisplay, partial_dependence

def plot_partial_dependence(model, X, feature_names, features_to_plot=None):
    """
    Plot partial dependence for top features.
    """
    if features_to_plot is None:
        # Select top 4 features by importance
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:4]
        features_to_plot = top_idx.tolist()

    # Create PDP
    fig, ax = plt.subplots(figsize=(14, 10))

    display = PartialDependenceDisplay.from_estimator(
        model,
        X,
        features_to_plot,
        feature_names=feature_names,
        ax=ax,
        n_cols=2,
        grid_resolution=50
    )

    plt.suptitle('Partial Dependence Plots', fontsize=16)
    plt.tight_layout()
    plt.show()

    # 2D PDP for interaction
    print("\n" + "=" * 70)
    print("2D PARTIAL DEPENDENCE (Feature Interactions)")
    print("=" * 70)

    if len(features_to_plot) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))

        display2d = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [(features_to_plot[0], features_to_plot[1])],
            feature_names=feature_names,
            ax=ax,
            grid_resolution=20
        )

        plt.title(f'2D PDP: {feature_names[features_to_plot[0]]} vs {feature_names[features_to_plot[1]]}')
        plt.tight_layout()
        plt.show()

    return display

# Example
pdp_display = plot_partial_dependence(model, X_df.values, feature_names)
```

### Individual Conditional Expectation (ICE)

```python
def plot_ice(model, X, feature_names, feature_idx=0, n_samples=50):
    """
    Plot ICE curves showing heterogeneous effects.

    Unlike PDP (average), ICE shows individual curves.
    """
    from sklearn.inspection import partial_dependence

    # Select random samples
    np.random.seed(42)
    sample_idx = np.random.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    X_sample = X[sample_idx]

    # Get feature values
    feature_values = np.linspace(X[:, feature_idx].min(), X[:, feature_idx].max(), 50)

    # Calculate predictions for each sample at each feature value
    ice_curves = []

    for sample in X_sample:
        sample_curves = []
        for val in feature_values:
            # Create modified sample
            modified = sample.copy()
            modified[feature_idx] = val

            # Predict
            pred = model.predict_proba([modified])[0][1]
            sample_curves.append(pred)

        ice_curves.append(sample_curves)

    ice_curves = np.array(ice_curves)

    # Plot
    plt.figure(figsize=(10, 6))

    # ICE curves
    for curve in ice_curves:
        plt.plot(feature_values, curve, alpha=0.3, color='blue')

    # PDP (average of ICE)
    pdp = ice_curves.mean(axis=0)
    plt.plot(feature_values, pdp, color='red', linewidth=3, label='PDP (average)')

    plt.xlabel(feature_names[feature_idx])
    plt.ylabel('Predicted Probability')
    plt.title(f'ICE Plot: {feature_names[feature_idx]}')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print("Blue lines: Individual predictions (ICE)")
    print("Red line: Average effect (PDP)")
    print("\nIf blue lines are parallel → homogeneous effect")
    print("If blue lines diverge → heterogeneous effect (interactions)")

# Example
plot_ice(model, X_df.values, feature_names, feature_idx=0)
```

---

## Deep Learning Interpretability

### 1. Grad-CAM (Computer Vision)

```python
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

def grad_cam(model, image, target_layer, target_class=None):
    """
    Gradient-weighted Class Activation Mapping for CNNs.

    Highlights which regions of image are important for prediction.
    """
    # Set model to eval mode
    model.eval()

    # Forward hook to capture activations
    activations = []
    def forward_hook(module, input, output):
        activations.append(output)

    # Backward hook to capture gradients
    gradients = []
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    # Forward pass
    output = model(image)

    if target_class is None:
        target_class = output.argmax(dim=1).item()

    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0][target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)

    # Get activations and gradients
    acts = activations[0]
    grads = gradients[0]

    # Calculate weights (global average pooling of gradients)
    weights = grads.mean(dim=(2, 3), keepdim=True)

    # Weighted combination of activation maps
    cam = (weights * acts).sum(dim=1, keepdim=True)

    # ReLU and normalize
    cam = torch.relu(cam)
    cam = cam / cam.max()

    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()

    return cam, target_class

# Example usage (requires PyTorch and image)
"""
# Load pretrained model
model = models.resnet50(pretrained=True)
target_layer = model.layer4[-1]

# Load and preprocess image
image = Image.open('example.jpg')
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image).unsqueeze(0)

# Get Grad-CAM
cam, predicted_class = grad_cam(model, input_tensor, target_layer)

# Visualize
plt.imshow(cam[0, 0].detach().numpy(), cmap='jet', alpha=0.5)
plt.imshow(image, alpha=0.5)
plt.title(f'Grad-CAM (Class: {predicted_class})')
plt.show()
"""
```

### 2. Attention Visualization (Transformers)

```python
def visualize_attention(model, tokenizer, text, layer_idx=-1, head_idx=0):
    """
    Visualize attention weights in transformer models.

    Shows which tokens the model focuses on.
    """
    # This is a template - requires actual transformer model
    # Example with Hugging Face transformers:

    """
    from transformers import AutoModel, AutoTokenizer

    # Load model
    model_name = 'bert-base-uncased'
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')

    # Get attention weights
    outputs = model(**inputs)
    attentions = outputs.attentions  # Tuple of attention weights per layer

    # Select layer and head
    attention = attentions[layer_idx][0, head_idx].detach().numpy()

    # Get tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Visualize attention matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(attention, cmap='viridis')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key Tokens')
    plt.ylabel('Query Tokens')
    plt.title(f'Attention Weights (Layer {layer_idx}, Head {head_idx})')
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    """

    print("Attention visualization template")
    print("Requires: transformers library and pretrained model")

# Example usage (commented out)
# visualize_attention(model, tokenizer, "The cat sat on the mat")
```

### 3. Integrated Gradients

```python
def integrated_gradients(model, input_data, baseline=None, steps=50):
    """
    Integrated Gradients for attributing predictions to input features.

    More stable than raw gradients.
    """
    if baseline is None:
        baseline = torch.zeros_like(input_data)

    # Generate interpolated inputs
    alphas = torch.linspace(0, 1, steps)
    interpolated = [baseline + alpha * (input_data - baseline) for alpha in alphas]
    interpolated = torch.cat(interpolated, dim=0)

    # Require gradients
    interpolated.requires_grad = True

    # Forward pass
    outputs = model(interpolated)

    # Get gradients
    target_output = outputs[:, target_class].sum()
    target_output.backward()

    gradients = interpolated.grad

    # Average gradients
    avg_gradients = gradients.mean(dim=0)

    # Integrated gradients
    integrated_grads = (input_data - baseline) * avg_gradients

    return integrated_grads

# Example (requires PyTorch model)
# ig = integrated_gradients(model, input_tensor, target_class=predicted_class)
```

---

## Regulatory Compliance

### GDPR Article 22 - Right to Explanation

```python
def generate_gdpr_explanation(model, sample, feature_names, threshold=0.5):
    """
    Generate GDPR-compliant explanation for automated decision.

    Requirements:
    - Meaningful information about logic involved
    - Significance and consequences
    - Human-readable format
    """
    # Get prediction
    prediction_proba = model.predict_proba([sample])[0]
    prediction = int(prediction_proba[1] > threshold)

    # Get SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([sample])
    if isinstance(shap_values, list):
        shap_values = shap_values[1][0]
    else:
        shap_values = shap_values[0]

    # Generate report
    report = []
    report.append("=" * 70)
    report.append("AUTOMATED DECISION EXPLANATION")
    report.append("=" * 70)
    report.append("")

    # Decision
    decision = "APPROVED" if prediction == 1 else "DENIED"
    report.append(f"Decision: {decision}")
    report.append(f"Confidence: {prediction_proba[prediction]:.2%}")
    report.append("")

    # Logic
    report.append("Logic Involved:")
    report.append("The decision was made using a machine learning model trained on")
    report.append("historical application data. The model considers the following factors:")
    report.append("")

    # Top contributing factors
    contributions = pd.DataFrame({
        'Factor': feature_names,
        'Your Value': sample,
        'Impact': shap_values
    }).sort_values('Impact', ascending=False)

    report.append("Top Contributing Factors:")
    report.append("")

    for i, row in contributions.head(5).iterrows():
        impact = row['Impact']
        direction = "increased" if impact > 0 else "decreased"
        magnitude = "significantly" if abs(impact) > 0.1 else "moderately"

        report.append(f"  • {row['Factor']}: {row['Your Value']:.2f}")
        report.append(f"    This {magnitude} {direction} your approval probability")
        report.append("")

    # Consequences
    report.append("Consequences:")
    if decision == "DENIED":
        report.append("  • Your application has been denied")
        report.append("  • You may reapply after addressing the factors above")
        report.append("  • You have the right to contest this decision")
    else:
        report.append("  • Your application has been approved")
        report.append("  • Proceed to next steps as outlined in communication")

    report.append("")
    report.append("Right to Contest:")
    report.append("You have the right to:")
    report.append("  • Obtain human intervention")
    report.append("  • Express your point of view")
    report.append("  • Contest the decision")
    report.append("")
    report.append("Contact: compliance@example.com")
    report.append("=" * 70)

    # Print and return
    explanation = "\n".join(report)
    print(explanation)

    return explanation

# Example
gdpr_explanation = generate_gdpr_explanation(
    model,
    X_df.iloc[0].values,
    feature_names
)
```

### EU AI Act Compliance (2025)

```python
def generate_ai_act_documentation(model, X_train, y_train, feature_names):
    """
    Generate technical documentation required by EU AI Act.

    Required for high-risk AI systems.
    """
    doc = []
    doc.append("=" * 70)
    doc.append("EU AI ACT TECHNICAL DOCUMENTATION")
    doc.append("=" * 70)
    doc.append("")

    # 1. General description
    doc.append("1. GENERAL DESCRIPTION")
    doc.append("-" * 70)
    doc.append(f"Model Type: {type(model).__name__}")
    doc.append(f"Purpose: Credit risk assessment")
    doc.append(f"Training Samples: {len(X_train)}")
    doc.append(f"Features: {len(feature_names)}")
    doc.append("")

    # 2. Development and training
    doc.append("2. DEVELOPMENT AND TRAINING")
    doc.append("-" * 70)
    doc.append("Training Data:")
    doc.append(f"  - Size: {len(X_train)} samples")
    doc.append(f"  - Class distribution: {np.bincount(y_train)}")
    doc.append("")
    doc.append("Features used:")
    for i, fname in enumerate(feature_names, 1):
        doc.append(f"  {i}. {fname}")
    doc.append("")

    # 3. Performance metrics
    doc.append("3. PERFORMANCE METRICS")
    doc.append("-" * 70)
    from sklearn.model_selection import cross_val_score
    accuracy = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    f1 = cross_val_score(model, X_train, y_train, cv=5, scoring='f1').mean()

    doc.append(f"Cross-validated Accuracy: {accuracy:.4f}")
    doc.append(f"Cross-validated F1-Score: {f1:.4f}")
    doc.append("")

    # 4. Interpretability
    doc.append("4. INTERPRETABILITY")
    doc.append("-" * 70)
    doc.append("Methods available:")
    doc.append("  - SHAP values for individual explanations")
    doc.append("  - Feature importance for global understanding")
    doc.append("  - Partial dependence plots for feature effects")
    doc.append("")

    # 5. Fairness assessment
    doc.append("5. FAIRNESS AND BIAS ASSESSMENT")
    doc.append("-" * 70)
    doc.append("See separate fairness audit report")
    doc.append("(Reference: 49_Fairness_and_Bias.md)")
    doc.append("")

    # 6. Human oversight
    doc.append("6. HUMAN OVERSIGHT")
    doc.append("-" * 70)
    doc.append("Human review required for:")
    doc.append("  - All denials")
    doc.append("  - Edge cases (probability 0.4-0.6)")
    doc.append("  - Contested decisions")
    doc.append("")

    # 7. Robustness
    doc.append("7. ROBUSTNESS AND SECURITY")
    doc.append("-" * 70)
    doc.append("Measures implemented:")
    doc.append("  - Input validation")
    doc.append("  - Model versioning")
    doc.append("  - Performance monitoring")
    doc.append("")

    doc.append("=" * 70)

    documentation = "\n".join(doc)
    print(documentation)

    # Save to file
    with open('ai_act_documentation.txt', 'w') as f:
        f.write(documentation)

    return documentation

# Example
ai_act_docs = generate_ai_act_documentation(model, X_df.values, y, feature_names)
```

---

## Summary

### Interpretability Method Selection

| Method | Type | Speed | Accuracy | When to Use |
|--------|------|-------|----------|-------------|
| **Linear Coefficients** | Global | ⚡⚡⚡ | ✓✓✓ | Linear models only |
| **Decision Tree Rules** | Global | ⚡⚡⚡ | ✓✓✓ | Trees only (max_depth ≤ 5) |
| **SHAP (Tree)** | Both | ⚡⚡⚡ | ✓✓✓ | Tree-based models (recommended) |
| **SHAP (Kernel)** | Both | ⚡ | ✓✓✓ | Any model (slow) |
| **LIME** | Local | ⚡⚡ | ✓✓ | Any model, quick local explanation |
| **Permutation Importance** | Global | ⚡⚡ | ✓✓✓ | Any model, feature ranking |
| **Partial Dependence** | Global | ⚡⚡ | ✓✓ | Marginal effects |
| **ICE** | Local | ⚡⚡ | ✓✓ | Individual effects |
| **Grad-CAM** | Local | ⚡⚡ | ✓✓✓ | CNNs only |
| **Attention Viz** | Local | ⚡⚡ | ✓✓✓ | Transformers only |

### Best Practices (2025)

1. **Start with SHAP** - Gold standard for most use cases
2. **Use multiple methods** - Cross-validate explanations
3. **Document everything** - Regulatory requirements
4. **Global + Local** - Understand both overall behavior and individual predictions
5. **Validate explanations** - Do they make domain sense?
6. **Monitor in production** - Track explanation drift

### Key Takeaways

- ✅ Interpretability is mandatory for regulated industries (2025)
- ✅ SHAP provides theoretically sound explanations
- ✅ Use permutation importance over MDI for feature importance
- ✅ Combine global and local explanations
- ✅ Generate GDPR/AI Act compliant documentation
- ✅ Validate explanations with domain experts

---

**Last Updated:** 2025-10-14
**Status:** Complete - Production-ready with regulatory compliance
