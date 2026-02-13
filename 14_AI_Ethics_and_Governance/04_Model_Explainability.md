# Model Explainability and Interpretability

## Table of Contents
- [1. Introduction to Model Explainability](#1-introduction-to-model-explainability)
  - [1.1 Why Explainability Matters](#11-why-explainability-matters)
  - [1.2 Interpretability vs Explainability](#12-interpretability-vs-explainability)
  - [1.3 Regulatory Requirements](#13-regulatory-requirements)
- [2. Feature Importance Methods](#2-feature-importance-methods)
  - [2.1 Permutation Importance](#21-permutation-importance)
  - [2.2 Impurity-Based Importance](#22-impurity-based-importance)
  - [2.3 Drop-Column Importance](#23-drop-column-importance)
  - [2.4 Complete Implementation](#24-complete-implementation)
- [3. SHAP (SHapley Additive exPlanations)](#3-shap-shapley-additive-explanations)
  - [3.1 Game Theory Foundation](#31-game-theory-foundation)
  - [3.2 TreeSHAP](#32-treeshap)
  - [3.3 KernelSHAP](#33-kernelshap)
  - [3.4 DeepSHAP](#34-deepshap)
  - [3.5 SHAP Visualizations](#35-shap-visualizations)
  - [3.6 Production SHAP Implementation](#36-production-shap-implementation)
- [4. LIME (Local Interpretable Model-agnostic Explanations)](#4-lime-local-interpretable-model-agnostic-explanations)
  - [4.1 How LIME Works](#41-how-lime-works)
  - [4.2 Tabular Explanations](#42-tabular-explanations)
  - [4.3 Text Explanations](#43-text-explanations)
  - [4.4 Image Explanations](#44-image-explanations)
  - [4.5 Complete LIME Implementation](#45-complete-lime-implementation)
- [5. Grad-CAM and Saliency Maps](#5-grad-cam-and-saliency-maps)
  - [5.1 Gradient-Based Visualization Theory](#51-gradient-based-visualization-theory)
  - [5.2 Vanilla Gradients and Saliency Maps](#52-vanilla-gradients-and-saliency-maps)
  - [5.3 Grad-CAM Implementation](#53-grad-cam-implementation)
  - [5.4 Guided Grad-CAM](#54-guided-grad-cam)
- [6. Integrated Gradients](#6-integrated-gradients)
  - [6.1 Attribution Theory](#61-attribution-theory)
  - [6.2 Implementation for Neural Networks](#62-implementation-for-neural-networks)
  - [6.3 Integrated Gradients for NLP](#63-integrated-gradients-for-nlp)

---

## 1. Introduction to Model Explainability

### 1.1 Why Explainability Matters

**Model explainability** is the ability to understand and communicate how ML models make decisions. Critical use cases include:

**Healthcare**: Physicians need to understand why a model recommends a diagnosis or treatment. Black-box predictions without justification are medically and legally unacceptable.

**Finance**: Loan denials, credit scoring, and fraud detection require explanations for regulatory compliance (FCRA, ECOA) and customer trust.

**Criminal Justice**: COMPAS and similar risk assessment tools face scrutiny. Explanations are necessary to identify and mitigate bias.

**Autonomous Systems**: When self-driving cars make decisions, understanding the reasoning is critical for safety validation and debugging.

**Business Trust**: Stakeholders won't deploy models they don't understand. Explainability drives adoption.

**Debugging and Improvement**: Explanations reveal when models learn spurious correlations, helping improve data quality and feature engineering.

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

# Example: Model without explainability
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction without explanation - not acceptable in regulated industries
prediction = model.predict(X_test[0:1])
print(f"Prediction: {prediction[0]}")  # Output: 1 or 0 - but WHY?

# This is where explainability techniques become essential
```

### 1.2 Interpretability vs Explainability

**Interpretability**: Models that are inherently understandable by humans. The model structure itself provides transparency.

- **Linear Regression**: Coefficients directly show feature contributions
- **Decision Trees**: Visual rule paths are human-readable
- **Rule-Based Systems**: If-then logic is explicit
- **Generalized Additive Models (GAMs)**: Separate effect functions per feature

**Explainability**: Post-hoc methods to understand complex black-box models. Applied after training.

- **SHAP**: Explains any model using game theory
- **LIME**: Local surrogate models
- **Grad-CAM**: Visual explanations for CNNs
- **Attention Weights**: Highlight important inputs in transformers

**Key Distinction**: Interpretable models are transparent by design. Explainability techniques make opaque models more understandable.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Interpretable model - coefficients are the explanation
interpretable_model = LogisticRegression()
interpretable_model.fit(X_train, y_train)
print("Feature coefficients (interpretable):")
print(interpretable_model.coef_[0][:5])  # Direct interpretation

# Black-box model - requires explainability techniques
blackbox_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500)
blackbox_model.fit(X_train, y_train)
# No direct interpretation - need SHAP, LIME, etc.
```

**Trade-offs**:
- **Accuracy vs Interpretability**: Complex models (deep learning, gradient boosting) often outperform simple interpretable models
- **Global vs Local**: Some methods explain individual predictions (local), others explain overall model behavior (global)
- **Fidelity**: Explanations are approximations; they may not perfectly represent model behavior

### 1.3 Regulatory Requirements

**GDPR (General Data Protection Regulation)** - Article 15: Right to explanation for automated decision-making. Individuals can request meaningful information about the logic involved.

**EU AI Act** - Article 13: High-risk AI systems must provide transparency, logging, and explanations. Requirements include:
- Technical documentation of model architecture
- Explanation of model outputs
- Human oversight capabilities
- Audit trails for decisions

**Fair Credit Reporting Act (FCRA)** - Adverse action notices must explain why credit was denied, including key factors.

**Equal Credit Opportunity Act (ECOA)** - Creditors must provide specific reasons for credit denials.

**HIPAA** - Healthcare decisions require audit trails and justifications.

```python
class RegulatoryCompliantModel:
    """
    Wrapper ensuring regulatory compliance for ML predictions.
    Implements GDPR Article 15 and EU AI Act Article 13 requirements.
    """

    def __init__(self, model, feature_names, explainer_type='shap'):
        self.model = model
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.audit_log = []

    def predict_with_explanation(self, X, user_id=None):
        """
        Make prediction with mandatory explanation logging.

        Args:
            X: Input features (numpy array or pandas DataFrame)
            user_id: Optional user identifier for audit trail

        Returns:
            dict with prediction, explanation, and audit information
        """
        # Make prediction
        prediction = self.model.predict(X)
        prediction_proba = self.model.predict_proba(X) if hasattr(self.model, 'predict_proba') else None

        # Generate explanation (implementation depends on explainer_type)
        explanation = self._generate_explanation(X)

        # Create audit record
        audit_record = {
            'timestamp': pd.Timestamp.now(),
            'user_id': user_id,
            'prediction': prediction[0],
            'probability': prediction_proba[0].tolist() if prediction_proba is not None else None,
            'top_features': explanation['top_features'],
            'model_version': '1.0.0'  # Track model versions
        }

        self.audit_log.append(audit_record)

        return {
            'prediction': prediction[0],
            'probability': prediction_proba[0] if prediction_proba is not None else None,
            'explanation': explanation,
            'audit_id': len(self.audit_log) - 1,
            'human_review_contact': 'compliance@company.com'
        }

    def _generate_explanation(self, X):
        """Generate explanation based on configured method"""
        # Placeholder - actual implementation in later sections
        return {
            'top_features': ['feature_1', 'feature_2', 'feature_3'],
            'contributions': [0.3, 0.2, 0.15]
        }

    def get_audit_trail(self, user_id=None):
        """Retrieve audit logs for GDPR Article 15 requests"""
        if user_id:
            return [log for log in self.audit_log if log['user_id'] == user_id]
        return self.audit_log

    def generate_adverse_action_notice(self, audit_id):
        """Generate FCRA/ECOA compliant adverse action notice"""
        record = self.audit_log[audit_id]

        notice = f"""
ADVERSE ACTION NOTICE

Your application was denied on {record['timestamp']}.

PRIMARY REASONS:
"""
        for i, (feature, contrib) in enumerate(zip(record['top_features'][:4],
                                                    [0.3, 0.2, 0.15, 0.1])):
            notice += f"{i+1}. {feature.replace('_', ' ').title()}\n"

        notice += """
You have the right to:
1. Request human review of this decision
2. Access your data used in this decision (GDPR Article 15)
3. Request correction of inaccurate data (GDPR Article 16)

Contact: compliance@company.com
        """
        return notice

# Example usage
compliant_model = RegulatoryCompliantModel(model, feature_names=[f'feature_{i}' for i in range(20)])
result = compliant_model.predict_with_explanation(X_test[0:1], user_id='user_12345')

print("Prediction:", result['prediction'])
print("Explanation:", result['explanation'])

# Generate adverse action notice if needed
if result['prediction'] == 0:  # Denial
    notice = compliant_model.generate_adverse_action_notice(result['audit_id'])
    print(notice)
```

---

## 2. Feature Importance Methods

### 2.1 Permutation Importance

**Permutation importance** measures feature importance by randomly shuffling a feature's values and observing the decrease in model performance. Features that cause large performance drops when shuffled are important.

**Algorithm**:
1. Compute baseline model performance on validation set
2. For each feature:
   - Randomly shuffle feature values
   - Compute new model performance
   - Importance = baseline_score - shuffled_score
3. Repeat shuffling multiple times for statistical robustness

**Advantages**:
- Model-agnostic (works with any model)
- Captures feature importance for the prediction task
- Accounts for feature interactions

**Disadvantages**:
- Computationally expensive (requires multiple model evaluations)
- Can be unreliable with correlated features
- Requires held-out validation set

```python
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Compute permutation importance
perm_importance = permutation_importance(
    rf_model,
    X_test,
    y_test,
    n_repeats=30,  # Shuffle each feature 30 times
    random_state=42,
    scoring='accuracy'
)

# Create DataFrame for analysis
feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]
perm_importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance_mean': perm_importance.importances_mean,
    'importance_std': perm_importance.importances_std
}).sort_values('importance_mean', ascending=False)

print("Top 10 most important features (permutation):")
print(perm_importance_df.head(10))

# Visualize
plt.figure(figsize=(10, 6))
plt.barh(perm_importance_df.head(15)['feature'],
         perm_importance_df.head(15)['importance_mean'],
         xerr=perm_importance_df.head(15)['importance_std'])
plt.xlabel('Permutation Importance')
plt.title('Feature Importance - Permutation Method')
plt.tight_layout()
plt.savefig('permutation_importance.png', dpi=300)
```

**Custom Implementation with Different Metrics**:

```python
def permutation_importance_custom(model, X, y, metric_fn, n_repeats=10, random_state=42):
    """
    Custom permutation importance supporting any metric.

    Args:
        model: Trained model with predict method
        X: Feature matrix (numpy array or DataFrame)
        y: True labels
        metric_fn: Function(y_true, y_pred) --> score (higher is better)
        n_repeats: Number of permutation repeats per feature
        random_state: Random seed

    Returns:
        dict: Feature importances with mean and std
    """
    np.random.seed(random_state)

    # Baseline score
    baseline_pred = model.predict(X)
    baseline_score = metric_fn(y, baseline_pred)

    n_features = X.shape[1]
    importance_scores = np.zeros((n_features, n_repeats))

    for feature_idx in range(n_features):
        for repeat in range(n_repeats):
            # Copy data and permute feature
            X_permuted = X.copy()
            X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])

            # Compute score with permuted feature
            permuted_pred = model.predict(X_permuted)
            permuted_score = metric_fn(y, permuted_pred)

            # Importance is drop in performance
            importance_scores[feature_idx, repeat] = baseline_score - permuted_score

    return {
        'importances_mean': importance_scores.mean(axis=1),
        'importances_std': importance_scores.std(axis=1),
        'importances': importance_scores
    }

# Example: Use custom metric (F1 score)
from sklearn.metrics import f1_score

custom_perm_importance = permutation_importance_custom(
    rf_model,
    X_test,
    y_test,
    metric_fn=lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
    n_repeats=20
)

print("Permutation importance (F1 metric):")
for i, (mean, std) in enumerate(zip(custom_perm_importance['importances_mean'],
                                     custom_perm_importance['importances_std'])):
    print(f"Feature {i}: {mean:.4f} (+/- {std:.4f})")
```

### 2.2 Impurity-Based Importance

**Impurity-based importance** (Gini importance or MDI - Mean Decrease Impurity) measures feature importance based on how much each feature decreases impurity (Gini or entropy) in tree-based models.

**Calculation for Random Forests**:
- For each tree, sum the decrease in impurity from all splits using that feature
- Weight by number of samples reaching each split
- Average across all trees in forest

**Advantages**:
- Fast (computed during training)
- No additional model evaluations needed
- Built into scikit-learn tree models

**Disadvantages**:
- **Biased toward high-cardinality features** (features with many unique values)
- **Unreliable with correlated features** (importance spread among correlated features)
- Only applicable to tree-based models

```python
# Impurity-based importance (built into RandomForest)
impurity_importance = rf_model.feature_importances_

impurity_df = pd.DataFrame({
    'feature': feature_names,
    'importance': impurity_importance
}).sort_values('importance', ascending=False)

print("Top 10 features (impurity-based):")
print(impurity_df.head(10))

# Comparison: Permutation vs Impurity
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Permutation importance
axes[0].barh(perm_importance_df.head(10)['feature'],
             perm_importance_df.head(10)['importance_mean'])
axes[0].set_xlabel('Permutation Importance')
axes[0].set_title('Permutation Importance (Model-Agnostic)')

# Impurity importance
axes[1].barh(impurity_df.head(10)['feature'],
             impurity_df.head(10)['importance'])
axes[1].set_xlabel('Impurity-Based Importance')
axes[1].set_title('Impurity Importance (Tree-Specific)')

plt.tight_layout()
plt.savefig('importance_comparison.png', dpi=300)
```

**Why differences occur**:

```python
# Demonstrate bias with high-cardinality feature
from sklearn.datasets import make_classification

# Create dataset with one high-cardinality random feature
X_biased, y_biased = make_classification(n_samples=1000, n_features=5,
                                          n_informative=3, random_state=42)

# Add random high-cardinality feature (should be unimportant)
X_biased = np.column_stack([X_biased, np.random.randint(0, 100, size=1000)])
feature_names_biased = [f'feature_{i}' for i in range(5)] + ['random_high_card']

# Train model
rf_biased = RandomForestClassifier(n_estimators=100, random_state=42)
rf_biased.fit(X_biased, y_biased)

# Compare importances
print("Impurity-based importance:")
for name, imp in zip(feature_names_biased, rf_biased.feature_importances_):
    print(f"{name}: {imp:.4f}")

# Permutation importance won't be fooled
perm_imp_biased = permutation_importance(rf_biased, X_biased, y_biased,
                                          n_repeats=10, random_state=42)
print("\nPermutation importance:")
for name, imp in zip(feature_names_biased, perm_imp_biased.importances_mean):
    print(f"{name}: {imp:.4f}")
```

### 2.3 Drop-Column Importance

**Drop-column importance** measures how much model performance decreases when a feature is completely removed from training.

**Algorithm**:
1. Train model with all features --> baseline performance
2. For each feature:
   - Train new model without that feature
   - Measure performance drop
   - Importance = baseline - performance_without_feature

**Advantages**:
- Captures true importance for the model
- Accounts for feature interactions
- Model-agnostic

**Disadvantages**:
- **Very expensive**: Requires training N+1 models (N features + 1 baseline)
- Not feasible for high-dimensional data
- Results depend on training randomness

```python
from sklearn.base import clone

def drop_column_importance(model, X_train, y_train, X_val, y_val,
                           metric_fn, feature_names=None):
    """
    Compute drop-column feature importance.

    Args:
        model: Unfitted model (will be cloned and trained multiple times)
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric_fn: Scoring function (higher is better)
        feature_names: Optional feature names

    Returns:
        DataFrame with feature importance scores
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(n_features)]

    # Baseline: train with all features
    baseline_model = clone(model)
    baseline_model.fit(X_train, y_train)
    baseline_score = metric_fn(y_val, baseline_model.predict(X_val))

    print(f"Baseline score (all features): {baseline_score:.4f}")

    # Drop each feature and measure impact
    importance_scores = []

    for i in range(n_features):
        # Create dataset without feature i
        feature_mask = np.ones(n_features, dtype=bool)
        feature_mask[i] = False

        X_train_dropped = X_train[:, feature_mask]
        X_val_dropped = X_val[:, feature_mask]

        # Train model without this feature
        dropped_model = clone(model)
        dropped_model.fit(X_train_dropped, y_train)
        dropped_score = metric_fn(y_val, dropped_model.predict(X_val_dropped))

        # Importance is performance drop
        importance = baseline_score - dropped_score
        importance_scores.append({
            'feature': feature_names[i],
            'importance': importance,
            'score_without': dropped_score
        })

        print(f"Without {feature_names[i]}: {dropped_score:.4f} (importance: {importance:.4f})")

    return pd.DataFrame(importance_scores).sort_values('importance', ascending=False)

# Example usage (on small dataset due to computational cost)
X_small, y_small = make_classification(n_samples=500, n_features=8,
                                        n_informative=6, random_state=42)
X_train_s, X_val_s, y_train_s, y_val_s = train_test_split(
    X_small, y_small, test_size=0.2, random_state=42
)

drop_importance = drop_column_importance(
    model=RandomForestClassifier(n_estimators=50, random_state=42),
    X_train=X_train_s,
    y_train=y_train_s,
    X_val=X_val_s,
    y_val=y_val_s,
    metric_fn=accuracy_score,
    feature_names=[f'feature_{i}' for i in range(8)]
)

print("\nDrop-column importance ranking:")
print(drop_importance)
```

### 2.4 Complete Implementation

**Production-ready feature importance class** supporting multiple methods:

```python
class FeatureImportanceAnalyzer:
    """
    Comprehensive feature importance analysis with multiple methods.
    Supports permutation, impurity, and drop-column importance.
    """

    def __init__(self, model, feature_names=None):
        """
        Args:
            model: Trained model (for permutation/impurity) or untrained (for drop-column)
            feature_names: List of feature names
        """
        self.model = model
        self.feature_names = feature_names
        self.importance_results = {}

    def compute_permutation_importance(self, X, y, n_repeats=30,
                                       scoring='accuracy', random_state=42):
        """Compute permutation importance"""
        perm_imp = permutation_importance(
            self.model, X, y, n_repeats=n_repeats,
            scoring=scoring, random_state=random_state
        )

        self.importance_results['permutation'] = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(X.shape[1])],
            'importance_mean': perm_imp.importances_mean,
            'importance_std': perm_imp.importances_std
        }).sort_values('importance_mean', ascending=False)

        return self.importance_results['permutation']

    def compute_impurity_importance(self):
        """Compute impurity-based importance (tree models only)"""
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError("Model does not support impurity-based importance")

        self.importance_results['impurity'] = pd.DataFrame({
            'feature': self.feature_names or [f'feature_{i}' for i in range(len(self.model.feature_importances_))],
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        return self.importance_results['impurity']

    def compute_drop_column_importance(self, X_train, y_train, X_val, y_val,
                                       metric_fn=accuracy_score):
        """Compute drop-column importance (requires retraining)"""
        from sklearn.base import clone

        # Baseline
        if not hasattr(self.model, 'predict'):
            raise ValueError("Model must be trained or clonable")

        baseline_model = clone(self.model) if hasattr(self.model, 'get_params') else self.model
        baseline_model.fit(X_train, y_train)
        baseline_score = metric_fn(y_val, baseline_model.predict(X_val))

        # Drop each feature
        n_features = X_train.shape[1]
        importance_scores = []

        for i in range(n_features):
            feature_mask = np.ones(n_features, dtype=bool)
            feature_mask[i] = False

            dropped_model = clone(baseline_model)
            dropped_model.fit(X_train[:, feature_mask], y_train)
            dropped_score = metric_fn(y_val, dropped_model.predict(X_val[:, feature_mask]))

            importance_scores.append({
                'feature': (self.feature_names or [f'feature_{i}' for i in range(n_features)])[i],
                'importance': baseline_score - dropped_score
            })

        self.importance_results['drop_column'] = pd.DataFrame(importance_scores).sort_values(
            'importance', ascending=False
        )

        return self.importance_results['drop_column']

    def plot_comparison(self, top_n=15, figsize=(15, 5)):
        """Plot comparison of all computed importance methods"""
        methods = list(self.importance_results.keys())
        n_methods = len(methods)

        if n_methods == 0:
            print("No importance results computed yet")
            return

        fig, axes = plt.subplots(1, n_methods, figsize=figsize)
        if n_methods == 1:
            axes = [axes]

        for i, method in enumerate(methods):
            df = self.importance_results[method].head(top_n)
            importance_col = 'importance_mean' if method == 'permutation' else 'importance'

            axes[i].barh(df['feature'], df[importance_col])
            axes[i].set_xlabel('Importance')
            axes[i].set_title(f'{method.replace("_", " ").title()} Importance')
            axes[i].invert_yaxis()

        plt.tight_layout()
        return fig

    def get_top_features(self, method='permutation', top_n=10):
        """Get top N most important features for a given method"""
        if method not in self.importance_results:
            raise ValueError(f"Method '{method}' not computed yet")

        return self.importance_results[method].head(top_n)['feature'].tolist()

    def export_report(self, filepath='feature_importance_report.html'):
        """Export comprehensive HTML report"""
        html = "<html><head><title>Feature Importance Report</title></head><body>"
        html += "<h1>Feature Importance Analysis Report</h1>"

        for method, df in self.importance_results.items():
            html += f"<h2>{method.replace('_', ' ').title()} Importance</h2>"
            html += df.head(20).to_html()

        html += "</body></html>"

        with open(filepath, 'w') as f:
            f.write(html)

        print(f"Report exported to {filepath}")

# Complete example
analyzer = FeatureImportanceAnalyzer(rf_model, feature_names=feature_names)

# Compute all importance types
perm_imp = analyzer.compute_permutation_importance(X_test, y_test, n_repeats=20)
imp_imp = analyzer.compute_impurity_importance()

print("Top 10 features (Permutation):")
print(perm_imp.head(10))

print("\nTop 10 features (Impurity):")
print(imp_imp.head(10))

# Visualize comparison
analyzer.plot_comparison(top_n=12)
plt.savefig('importance_methods_comparison.png', dpi=300)

# Export report
analyzer.export_report('feature_importance_report.html')
```

---

## 3. SHAP (SHapley Additive exPlanations)

### 3.1 Game Theory Foundation

**SHAP** is based on **Shapley values** from cooperative game theory. The Shapley value fairly distributes a payout among players based on their contributions to a coalition.

**In ML context**:
- **Players**: Features
- **Game**: Prediction task
- **Payout**: Model prediction
- **Contribution**: How much does each feature contribute to the prediction?

**Shapley Value Formula**:

For feature i:

```
phi_i = sum over all feature subsets S not containing i of:
    [|S|! * (M - |S| - 1)! / M!] * [f(S union {i}) - f(S)]
```

Where:
- M = total number of features
- S = subset of features
- f(S) = model prediction using only features in S
- phi_i = SHAP value for feature i

**Properties**:
1. **Efficiency**: Sum of all SHAP values = prediction - baseline
2. **Symmetry**: Features with identical contributions get same values
3. **Dummy**: Features with no effect get SHAP value of 0
4. **Additivity**: For ensemble models, SHAP values add linearly

```python
import shap
import xgboost as xgb

# Create sample dataset
X, y = make_classification(n_samples=1000, n_features=10, n_informative=7, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

feature_names = [f'feature_{i}' for i in range(10)]
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Train XGBoost model
xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, random_state=42)
xgb_model.fit(X_train_df, y_train)

# Create SHAP explainer
explainer = shap.Explainer(xgb_model)

# Compute SHAP values for test set
shap_values = explainer(X_test_df)

print("SHAP values shape:", shap_values.values.shape)  # (n_samples, n_features)
print("Base value (expected prediction):", explainer.expected_value)

# Verify efficiency property: prediction = base_value + sum(shap_values)
sample_idx = 0
prediction = xgb_model.predict_proba(X_test_df.iloc[[sample_idx]])[:, 1][0]
shap_sum = explainer.expected_value + shap_values.values[sample_idx].sum()

print(f"\nSample {sample_idx}:")
print(f"Model prediction (probability): {prediction:.4f}")
print(f"Base value + SHAP sum: {shap_sum:.4f}")
print(f"Difference (should be ~0): {abs(prediction - shap_sum):.6f}")
```

**Computational Challenge**: Exact Shapley values require 2^M coalitions. For 50 features, that's 1,125,899,906,842,624 evaluations - infeasible!

**Solution**: Approximation algorithms - TreeSHAP, KernelSHAP, DeepSHAP.

### 3.2 TreeSHAP

**TreeSHAP** is an exact, efficient algorithm for tree-based models (Random Forest, XGBoost, LightGBM, CatBoost). Computes exact Shapley values in polynomial time.

**Key Insight**: Instead of evaluating all 2^M coalitions, TreeSHAP traverses tree structures and computes expected values efficiently.

**Algorithm**:
1. For each tree, compute feature contributions by tracking decision paths
2. Weight contributions by probability of reaching each node
3. Average across all trees in ensemble

**Time Complexity**: O(TLD^2) where T=trees, L=leaves, D=depth. Much faster than exponential 2^M.

```python
# TreeSHAP with XGBoost
explainer_tree = shap.TreeExplainer(xgb_model)
shap_values_tree = explainer_tree.shap_values(X_test_df)

print("TreeSHAP values shape:", shap_values_tree.shape)

# Single prediction explanation
sample_idx = 0
print(f"\nExplanation for sample {sample_idx}:")
print(f"Prediction: {xgb_model.predict_proba(X_test_df.iloc[[sample_idx]])[:, 1][0]:.4f}")
print(f"Base value: {explainer_tree.expected_value:.4f}")

for feature, shap_val, feat_val in zip(feature_names,
                                        shap_values_tree[sample_idx],
                                        X_test_df.iloc[sample_idx]):
    print(f"{feature}: {feat_val:.3f} --> SHAP {shap_val:+.4f}")
```

**TreeSHAP for different tree models**:

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import lightgbm as lgb

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_df, y_train)
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test_df)

# LightGBM
lgb_model = lgb.LGBMClassifier(n_estimators=100, random_state=42)
lgb_model.fit(X_train_df, y_train)
explainer_lgb = shap.TreeExplainer(lgb_model)
shap_values_lgb = explainer_lgb.shap_values(X_test_df)

# Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(X_train_df, y_train)
explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_test_df)

print("TreeSHAP works efficiently across all tree-based models")
print(f"Random Forest SHAP: {shap_values_rf[1].shape}")
print(f"LightGBM SHAP: {shap_values_lgb.shape}")
print(f"GradientBoosting SHAP: {shap_values_gb.shape}")
```

### 3.3 KernelSHAP

**KernelSHAP** is a model-agnostic approximation method. Works with ANY model (neural networks, SVMs, etc.) by treating it as a black box.

**Algorithm**:
1. Sample coalitions of features (subsets)
2. For each coalition, replace missing features with background values
3. Get model predictions for each coalition
4. Fit weighted linear regression to approximate Shapley values

**Key Parameters**:
- `nsamples`: Number of coalition samples (higher = more accurate but slower)
- `background`: Background dataset for feature replacement (typically training set sample)

```python
# KernelSHAP for any model (example: neural network)
from sklearn.neural_network import MLPClassifier

# Train neural network
mlp_model = MLPClassifier(hidden_layer_sizes=(50, 30), max_iter=500, random_state=42)
mlp_model.fit(X_train_df, y_train)

# KernelSHAP requires background data
# Use k-means summary of training data for efficiency
background = shap.kmeans(X_train_df, 50)  # 50 representative samples

# Create KernelSHAP explainer
explainer_kernel = shap.KernelExplainer(mlp_model.predict_proba, background)

# Compute SHAP values (slower than TreeSHAP)
# Only explain first 10 samples for speed
shap_values_kernel = explainer_kernel.shap_values(X_test_df.iloc[:10])

print("KernelSHAP for neural network:")
print(f"SHAP values shape: {shap_values_kernel[1].shape}")

# Compare computational time
import time

start = time.time()
_ = explainer_tree.shap_values(X_test_df[:100])
tree_time = time.time() - start

start = time.time()
_ = explainer_kernel.shap_values(X_test_df[:10])
kernel_time = time.time() - start

print(f"\nTreeSHAP (100 samples): {tree_time:.2f}s")
print(f"KernelSHAP (10 samples): {kernel_time:.2f}s")
print("TreeSHAP is ~{:.0f}x faster per sample".format(kernel_time * 10 / tree_time))
```

**Optimizing KernelSHAP**:

```python
# Use sampling strategies to reduce computation time

# Strategy 1: Reduce background dataset size
background_small = shap.sample(X_train_df, 25)  # Use 25 samples

# Strategy 2: Reduce nsamples (coalition samples)
explainer_fast = shap.KernelExplainer(mlp_model.predict_proba, background_small)
shap_values_fast = explainer_fast.shap_values(X_test_df.iloc[:5], nsamples=100)  # Default is auto

# Strategy 3: Use GPU-accelerated models when possible
# KernelSHAP calls model.predict many times - fast models are critical

print("Fast KernelSHAP configuration:")
print(f"Background size: 25")
print(f"Samples explained: 5")
print(f"Coalition samples: 100")
```

### 3.4 DeepSHAP

**DeepSHAP** is optimized for deep neural networks. Combines ideas from DeepLIFT and Shapley values.

**Key Idea**: Backpropagate contributions through the network using reference activations from background dataset.

**Advantages over KernelSHAP for deep learning**:
- Much faster (single backward pass vs many forward passes)
- Handles non-linearities better
- Works with PyTorch and TensorFlow

```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Create PyTorch model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size=50):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 30)
        self.fc3 = nn.Linear(30, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNN(input_size=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
X_test_tensor = torch.FloatTensor(X_test).to(device)

# Training loop
model.train()
for epoch in range(50):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

model.eval()

# DeepSHAP explainer
background_tensor = torch.FloatTensor(X_train[:50]).to(device)
test_tensor = torch.FloatTensor(X_test[:10]).to(device)

explainer_deep = shap.DeepExplainer(model, background_tensor)
shap_values_deep = explainer_deep.shap_values(test_tensor)

print("DeepSHAP for PyTorch neural network:")
print(f"SHAP values shape: {np.array(shap_values_deep).shape}")

# Verify additivity
sample_idx = 0
model_output = model(test_tensor[sample_idx:sample_idx+1]).detach().cpu().numpy()[0]
expected_value = np.array(explainer_deep.expected_value)
shap_contribution = np.array(shap_values_deep)[:, sample_idx, :].sum(axis=1)

print(f"\nModel output: {model_output}")
print(f"Expected value: {expected_value}")
print(f"Expected + SHAP: {expected_value + shap_contribution}")
```

**DeepSHAP for CNN (Computer Vision)**:

```python
import torchvision.models as models

# Load pre-trained ResNet
resnet = models.resnet18(pretrained=True).eval()

# Use GPU if available
if torch.cuda.is_available():
    resnet = resnet.cuda()

# Dummy image data (batch_size=1, channels=3, height=224, width=224)
dummy_input = torch.randn(1, 3, 224, 224)
if torch.cuda.is_available():
    dummy_input = dummy_input.cuda()

# Background: use random images or dataset samples
background_images = torch.randn(10, 3, 224, 224)
if torch.cuda.is_available():
    background_images = background_images.cuda()

# DeepSHAP for CNN
explainer_cnn = shap.DeepExplainer(resnet, background_images)
shap_values_cnn = explainer_cnn.shap_values(dummy_input)

print(f"DeepSHAP for CNN:")
print(f"Input shape: {dummy_input.shape}")
print(f"SHAP values shape: {np.array(shap_values_cnn).shape}")
print("Note: SHAP values are at pixel level for images")
```

### 3.5 SHAP Visualizations

SHAP provides multiple visualization types for different analysis needs.

**1. Waterfall Plot** - Explains individual predictions by showing how each feature pushes prediction from base value.

```python
# Waterfall plot - single prediction
shap.plots.waterfall(shap_values[0])
plt.savefig('shap_waterfall.png', dpi=300, bbox_inches='tight')

# Interpretation: Shows cumulative effect of features
# Base value (e.g., 0.5) + feature contributions --> final prediction
```

**2. Force Plot** - Interactive visualization showing feature contributions.

```python
# Force plot - single prediction
shap.plots.force(shap_values[0])

# Force plot - multiple predictions
shap.plots.force(shap_values[:100])  # Interactive visualization
```

**3. Summary Plot (Beeswarm)** - Shows feature importance globally and distribution of impacts.

```python
# Summary plot - global feature importance
shap.summary_plot(shap_values, X_test_df)
plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')

# Interpretation:
# - Features sorted by importance (top to bottom)
# - Color shows feature value (red=high, blue=low)
# - X-axis shows SHAP value (impact on prediction)
# - Example: Red dots on right means high feature value --> high positive impact
```

**4. Summary Plot (Bar)** - Mean absolute SHAP values (global importance).

```python
# Bar plot - average feature importance
shap.summary_plot(shap_values, X_test_df, plot_type='bar')
plt.savefig('shap_bar.png', dpi=300, bbox_inches='tight')
```

**5. Dependence Plot** - Shows relationship between feature value and SHAP value, colored by interaction feature.

```python
# Dependence plot - shows feature effect and interactions
shap.dependence_plot('feature_0', shap_values.values, X_test_df)
plt.savefig('shap_dependence.png', dpi=300, bbox_inches='tight')

# Automatically detect interaction
shap.dependence_plot('feature_0', shap_values.values, X_test_df,
                     interaction_index='auto')  # Finds best interaction feature
plt.savefig('shap_dependence_interaction.png', dpi=300, bbox_inches='tight')
```

**6. Decision Plot** - Shows prediction paths for multiple samples.

```python
# Decision plot - compare multiple predictions
shap.decision_plot(explainer.expected_value, shap_values.values[:20], X_test_df.iloc[:20])
plt.savefig('shap_decision.png', dpi=300, bbox_inches='tight')
```

**Custom SHAP Visualization**:

```python
def plot_shap_top_features(shap_values, feature_names, top_n=10, figsize=(10, 6)):
    """
    Custom bar plot showing top N most important features.

    Args:
        shap_values: SHAP values array (n_samples, n_features)
        feature_names: List of feature names
        top_n: Number of top features to show
        figsize: Figure size
    """
    # Compute mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': mean_abs_shap
    }).sort_values('importance', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=figsize)
    plt.barh(importance_df['feature'], importance_df['importance'])
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    return importance_df

# Usage
top_features = plot_shap_top_features(shap_values, feature_names, top_n=10)
plt.savefig('custom_shap_importance.png', dpi=300)
print(top_features)
```

### 3.6 Production SHAP Implementation

Complete production-ready SHAP explainer with caching, batch processing, and logging.

```python
import joblib
import hashlib
import json
from pathlib import Path

class ProductionSHAPExplainer:
    """
    Production-ready SHAP explainer with caching and logging.

    Features:
    - Caches explainer objects to avoid recomputation
    - Batch processing for efficiency
    - Logging and audit trails
    - Multiple visualization exports
    - API-ready JSON outputs
    """

    def __init__(self, model, background_data, model_type='tree',
                 cache_dir='./shap_cache', model_version='1.0'):
        """
        Args:
            model: Trained model
            background_data: Background dataset for KernelSHAP (ignored for TreeSHAP)
            model_type: 'tree', 'kernel', or 'deep'
            cache_dir: Directory for caching explainers
            model_version: Model version for cache invalidation
        """
        self.model = model
        self.background_data = background_data
        self.model_type = model_type
        self.model_version = model_version
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        # Create or load explainer
        self.explainer = self._get_or_create_explainer()

        # Logging
        self.explanations_log = []

    def _get_cache_key(self):
        """Generate cache key based on model type and version"""
        key_string = f"{self.model_type}_{self.model_version}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def _get_or_create_explainer(self):
        """Load cached explainer or create new one"""
        cache_key = self._get_cache_key()
        cache_path = self.cache_dir / f"explainer_{cache_key}.pkl"

        if cache_path.exists():
            print(f"Loading cached explainer from {cache_path}")
            return joblib.load(cache_path)

        print("Creating new SHAP explainer...")
        if self.model_type == 'tree':
            explainer = shap.TreeExplainer(self.model)
        elif self.model_type == 'kernel':
            explainer = shap.KernelExplainer(self.model.predict_proba, self.background_data)
        elif self.model_type == 'deep':
            explainer = shap.DeepExplainer(self.model, self.background_data)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        # Cache explainer
        joblib.dump(explainer, cache_path)
        print(f"Explainer cached to {cache_path}")

        return explainer

    def explain_batch(self, X, batch_size=100):
        """
        Explain predictions for batch of samples.

        Args:
            X: Features to explain (DataFrame or numpy array)
            batch_size: Process in batches for memory efficiency

        Returns:
            SHAP values for all samples
        """
        n_samples = X.shape[0]
        all_shap_values = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = X.iloc[start_idx:end_idx] if hasattr(X, 'iloc') else X[start_idx:end_idx]

            if self.model_type == 'tree':
                batch_shap = self.explainer.shap_values(batch)
            else:
                batch_shap = self.explainer.shap_values(batch)

            all_shap_values.append(batch_shap)
            print(f"Processed batch {start_idx}-{end_idx}/{n_samples}")

        # Concatenate batches
        if isinstance(all_shap_values[0], list):  # Multi-class
            return [np.vstack([batch[i] for batch in all_shap_values])
                    for i in range(len(all_shap_values[0]))]
        else:
            return np.vstack(all_shap_values)

    def explain_single(self, instance, feature_names=None, return_json=True):
        """
        Explain single prediction with detailed output.

        Args:
            instance: Single sample to explain
            feature_names: Optional feature names
            return_json: Return JSON-serializable dict

        Returns:
            Explanation dict or SHAP values
        """
        # Get SHAP values
        if self.model_type == 'tree':
            shap_vals = self.explainer.shap_values(instance)
        else:
            shap_vals = self.explainer.shap_values(instance)

        # For binary classification, extract positive class
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_vals = shap_vals[1]

        if not return_json:
            return shap_vals

        # Create detailed explanation
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(shap_vals[0]))]

        feature_contributions = [
            {
                'feature': name,
                'value': float(instance.iloc[0, i] if hasattr(instance, 'iloc') else instance[0][i]),
                'shap_value': float(shap_vals[0][i]),
                'abs_shap': float(abs(shap_vals[0][i]))
            }
            for i, name in enumerate(feature_names)
        ]

        # Sort by absolute SHAP value
        feature_contributions.sort(key=lambda x: x['abs_shap'], reverse=True)

        explanation = {
            'base_value': float(self.explainer.expected_value),
            'prediction': float(self.model.predict_proba(instance)[0][1]) if hasattr(self.model, 'predict_proba')
                         else float(self.model.predict(instance)[0]),
            'feature_contributions': feature_contributions,
            'top_5_features': feature_contributions[:5],
            'model_version': self.model_version,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        # Log explanation
        self.explanations_log.append(explanation)

        return explanation

    def export_visualizations(self, shap_values, X, output_dir='./shap_plots',
                             feature_names=None):
        """
        Export all SHAP visualizations to files.

        Args:
            shap_values: Computed SHAP values
            X: Feature data
            output_dir: Directory to save plots
            feature_names: Optional feature names
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame) and feature_names is not None:
            X = pd.DataFrame(X, columns=feature_names)

        # Summary plot (beeswarm)
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(output_path / 'summary_beeswarm.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Summary plot (bar)
        shap.summary_plot(shap_values, X, plot_type='bar', show=False)
        plt.savefig(output_path / 'summary_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Dependence plots for top features
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_features_idx = mean_abs_shap.argsort()[-5:][::-1]

        for idx in top_features_idx:
            feature_name = X.columns[idx] if isinstance(X, pd.DataFrame) else f'feature_{idx}'
            shap.dependence_plot(idx, shap_values, X, show=False)
            plt.savefig(output_path / f'dependence_{feature_name}.png',
                       dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizations exported to {output_path}")

    def export_explanations_log(self, filepath='explanations_log.json'):
        """Export logged explanations to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.explanations_log, f, indent=2)
        print(f"Explanations log exported to {filepath}")

    def get_global_importance(self, shap_values, feature_names=None):
        """
        Compute global feature importance from SHAP values.

        Returns:
            DataFrame with feature importance rankings
        """
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(mean_abs_shap))]

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        return importance_df

# Production usage example
prod_explainer = ProductionSHAPExplainer(
    model=xgb_model,
    background_data=X_train_df.sample(100),
    model_type='tree',
    cache_dir='./shap_cache',
    model_version='2.1'
)

# Explain single prediction (API endpoint use case)
explanation = prod_explainer.explain_single(
    X_test_df.iloc[[0]],
    feature_names=feature_names,
    return_json=True
)
print(json.dumps(explanation['top_5_features'], indent=2))

# Batch processing
shap_values_batch = prod_explainer.explain_batch(X_test_df, batch_size=50)

# Export visualizations
prod_explainer.export_visualizations(
    shap_values_batch if not isinstance(shap_values_batch, list) else shap_values_batch[1],
    X_test_df,
    output_dir='./production_shap_plots'
)

# Global importance
global_importance = prod_explainer.get_global_importance(
    shap_values_batch if not isinstance(shap_values_batch, list) else shap_values_batch[1],
    feature_names=feature_names
)
print("\nGlobal Feature Importance:")
print(global_importance.head(10))

# Export audit log
prod_explainer.export_explanations_log('shap_audit_log.json')
```

---

## 4. LIME (Local Interpretable Model-agnostic Explanations)

### 4.1 How LIME Works

**LIME** explains individual predictions by fitting an interpretable model (linear regression) locally around the prediction.

**Algorithm**:
1. Take instance x to explain
2. Generate perturbed samples around x
3. Get model predictions for perturbed samples
4. Weight samples by proximity to x
5. Train interpretable model (linear) on weighted samples
6. Coefficients of linear model = explanation

**Key Insight**: Complex models may be non-linear globally, but are approximately linear locally.

```python
from lime import lime_tabular
from lime.lime_text import LimeTextExplainer
from lime.lime_image import LimeImageExplainer

# LIME for tabular data
lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    mode='classification',
    discretize_continuous=True  # Bin continuous features for better explanations
)

# Explain single prediction
instance_idx = 0
lime_exp = lime_explainer.explain_instance(
    data_row=X_test[instance_idx],
    predict_fn=xgb_model.predict_proba,
    num_features=10,  # Top 10 features
    num_samples=5000  # Number of perturbed samples
)

# Display explanation
print("LIME explanation:")
print(lime_exp.as_list())

# Visualize
lime_exp.show_in_notebook(show_table=True)
lime_exp.save_to_file('lime_explanation.html')
```

**Understanding LIME output**:

```python
# Get explanation as list of (feature, weight) tuples
explanation_list = lime_exp.as_list()

for feature_range, weight in explanation_list:
    print(f"{feature_range}: {weight:+.4f}")
    # Example: "feature_0 <= 0.5": +0.23 means this condition increases probability by 0.23

# Get feature importance map
feature_importance = dict(lime_exp.as_list())

# Local linear approximation coefficients
local_coefs = lime_exp.local_exp[1]  # For class 1
print("\nLocal linear model coefficients:")
for feature_idx, coef in local_coefs:
    print(f"{feature_names[feature_idx]}: {coef:+.4f}")
```

### 4.2 Tabular Explanations

Complete LIME implementation for tabular data with hyperparameter tuning.

```python
class TabularLIMEExplainer:
    """
    Production LIME explainer for tabular data.
    """

    def __init__(self, X_train, feature_names, class_names=None,
                 mode='classification', categorical_features=None):
        """
        Args:
            X_train: Training data for reference distribution
            feature_names: List of feature names
            class_names: List of class names
            mode: 'classification' or 'regression'
            categorical_features: Indices of categorical features
        """
        self.feature_names = feature_names
        self.class_names = class_names or ['Class 0', 'Class 1']
        self.mode = mode

        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=self.class_names,
            mode=mode,
            categorical_features=categorical_features,
            discretize_continuous=True,
            random_state=42
        )

    def explain(self, instance, predict_fn, num_features=10, num_samples=5000):
        """
        Explain single prediction.

        Args:
            instance: Instance to explain (1D array)
            predict_fn: Model prediction function
            num_features: Number of features in explanation
            num_samples: Number of perturbed samples

        Returns:
            LIME explanation object
        """
        explanation = self.explainer.explain_instance(
            data_row=instance,
            predict_fn=predict_fn,
            num_features=num_features,
            num_samples=num_samples
        )

        return explanation

    def explain_to_dict(self, instance, predict_fn, num_features=10):
        """Get explanation as JSON-serializable dict"""
        exp = self.explain(instance, predict_fn, num_features)

        # Get prediction
        prediction = predict_fn(instance.reshape(1, -1))[0]

        # Get feature contributions
        contributions = []
        for feature_desc, weight in exp.as_list():
            contributions.append({
                'feature_description': feature_desc,
                'weight': float(weight)
            })

        return {
            'prediction': prediction.tolist() if hasattr(prediction, 'tolist') else prediction,
            'feature_contributions': contributions,
            'intercept': float(exp.intercept[1]) if self.mode == 'classification' else float(exp.intercept),
            'score': float(exp.score) if hasattr(exp, 'score') else None
        }

    def explain_batch(self, X, predict_fn, num_features=10, num_samples=5000):
        """Explain multiple instances"""
        explanations = []

        for i in range(X.shape[0]):
            exp = self.explain(X[i], predict_fn, num_features, num_samples)
            explanations.append(exp)

            if (i + 1) % 10 == 0:
                print(f"Explained {i + 1}/{X.shape[0]} instances")

        return explanations

    def compare_explanations(self, instances, predict_fn, labels=None, num_features=10):
        """
        Compare explanations for multiple instances side-by-side.

        Args:
            instances: List of instances to explain
            predict_fn: Model prediction function
            labels: Optional labels for instances
            num_features: Number of features to show
        """
        explanations = []

        for i, instance in enumerate(instances):
            exp = self.explain(instance, predict_fn, num_features)
            explanations.append(exp)

        # Create comparison visualization
        fig, axes = plt.subplots(1, len(instances), figsize=(6*len(instances), 4))
        if len(instances) == 1:
            axes = [axes]

        for i, (exp, ax) in enumerate(zip(explanations, axes)):
            # Extract features and weights
            features_weights = exp.as_list()
            features = [fw[0].split('<=')[0].split('>')[0].strip() for fw in features_weights]
            weights = [fw[1] for fw in features_weights]

            # Plot
            colors = ['green' if w > 0 else 'red' for w in weights]
            ax.barh(features, weights, color=colors)
            ax.set_xlabel('LIME Weight')
            title = f'Instance {i}' if labels is None else labels[i]
            ax.set_title(title)
            ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)

        plt.tight_layout()
        return fig

# Usage
tabular_lime = TabularLIMEExplainer(
    X_train=X_train,
    feature_names=feature_names,
    class_names=['Negative', 'Positive'],
    mode='classification'
)

# Single explanation
exp_dict = tabular_lime.explain_to_dict(
    instance=X_test[0],
    predict_fn=xgb_model.predict_proba,
    num_features=8
)

print("LIME Explanation (JSON):")
print(json.dumps(exp_dict, indent=2))

# Compare multiple instances
fig = tabular_lime.compare_explanations(
    instances=[X_test[0], X_test[5], X_test[10]],
    predict_fn=xgb_model.predict_proba,
    labels=['Sample 0', 'Sample 5', 'Sample 10'],
    num_features=8
)
plt.savefig('lime_comparison.png', dpi=300, bbox_inches='tight')
```

**Hyperparameter Impact**:

```python
# Demonstrate effect of num_samples
def compare_lime_stability(instance, predict_fn, num_features=5, trials=10):
    """Test LIME stability with different num_samples"""
    sample_sizes = [100, 500, 1000, 5000]

    results = {size: [] for size in sample_sizes}

    for size in sample_sizes:
        for trial in range(trials):
            exp = lime_explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=size
            )

            # Extract top feature weights
            weights = [w for _, w in exp.as_list()[:num_features]]
            results[size].append(weights)

    # Compute standard deviation (lower = more stable)
    for size, weight_lists in results.items():
        weight_array = np.array(weight_lists)
        mean_std = weight_array.std(axis=0).mean()
        print(f"num_samples={size}: mean std={mean_std:.4f}")

compare_lime_stability(X_test[0], xgb_model.predict_proba, trials=10)

# Output shows higher num_samples --> lower std --> more stable explanations
# Trade-off: stability vs computation time
```

### 4.3 Text Explanations

LIME for NLP models - explains which words/tokens contribute to predictions.

```python
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Example: Sentiment classification
texts_train = [
    "This movie was absolutely amazing and wonderful",
    "Terrible film, waste of time",
    "Great acting and beautiful cinematography",
    "Boring and predictable plot",
    "Loved every minute of it",
    "Awful experience, very disappointing"
] * 100  # Repeat for larger dataset

labels_train = [1, 0, 1, 0, 1, 0] * 100  # 1=positive, 0=negative

# Train text classifier
vectorizer = TfidfVectorizer(max_features=1000)
text_clf = make_pipeline(vectorizer, MultinomialNB())
text_clf.fit(texts_train, labels_train)

# LIME text explainer
text_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Explain prediction
test_text = "This movie had great acting but the plot was terrible and boring"

explanation = text_explainer.explain_instance(
    test_text,
    text_clf.predict_proba,
    num_features=10,
    num_samples=1000
)

print("Text explanation:")
print(explanation.as_list())

# Visualize
explanation.show_in_notebook(text=test_text)
explanation.save_to_file('lime_text_explanation.html')
```

**Advanced text explanation with transformers**:

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load pre-trained sentiment model
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create prediction function
def predict_sentiment(texts):
    """Prediction function for LIME"""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.numpy()

# LIME explainer for transformer
lime_transformer_explainer = LimeTextExplainer(class_names=['Negative', 'Positive'])

# Explain
test_text = "The movie started well but became increasingly disappointing and tedious"
explanation_transformer = lime_transformer_explainer.explain_instance(
    test_text,
    predict_sentiment,
    num_features=10,
    num_samples=500
)

print("\nTransformer explanation:")
for word, weight in explanation_transformer.as_list():
    print(f"{word}: {weight:+.4f}")

# Save
explanation_transformer.save_to_file('lime_transformer_explanation.html')
```

### 4.4 Image Explanations

LIME for computer vision - highlights image regions that contribute to predictions.

```python
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Load pre-trained image classifier
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Load model
resnet_model = ResNet50(weights='imagenet')

# Load and preprocess image
def load_image(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    return img_array

# Prediction function for LIME
def predict_fn(images):
    """
    LIME requires batch prediction function.
    images: shape (n_samples, height, width, channels)
    """
    processed = preprocess_input(images.copy())
    predictions = resnet_model.predict(processed, verbose=0)
    return predictions

# Create image explainer
image_explainer = LimeImageExplainer()

# Load image (use any image file)
# img = load_image('path/to/image.jpg')
# For demonstration, create random image
img = np.random.randint(0, 255, size=(224, 224, 3)).astype(np.float32)

# Explain prediction
explanation_image = image_explainer.explain_instance(
    img,
    predict_fn,
    top_labels=5,  # Explain top 5 predicted classes
    hide_color=0,  # Color for hidden regions
    num_samples=1000  # Number of perturbed images
)

# Get top predicted class
top_class = explanation_image.top_labels[0]

# Visualize explanation
temp_img, mask = explanation_image.get_image_and_mask(
    top_class,
    positive_only=True,  # Show only positive contributions
    num_features=10,  # Number of superpixels to highlight
    hide_rest=False  # Don't hide rest of image
)

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img.astype(np.uint8))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(mark_boundaries(temp_img, mask))
axes[1].set_title('LIME Explanation (Positive)')
axes[1].axis('off')

# Show negative contributions
temp_img_neg, mask_neg = explanation_image.get_image_and_mask(
    top_class,
    positive_only=False,
    num_features=10,
    hide_rest=False
)
axes[2].imshow(mark_boundaries(temp_img_neg, mask_neg))
axes[2].set_title('LIME Explanation (All)')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('lime_image_explanation.png', dpi=300, bbox_inches='tight')
```

**Custom image explanation class**:

```python
class ImageLIMEExplainer:
    """Production LIME explainer for images"""

    def __init__(self, model, preprocess_fn=None):
        """
        Args:
            model: Image classification model
            preprocess_fn: Optional preprocessing function
        """
        self.model = model
        self.preprocess_fn = preprocess_fn or (lambda x: x)
        self.explainer = LimeImageExplainer()

    def predict_fn(self, images):
        """Prediction function wrapper"""
        processed = self.preprocess_fn(images.copy())
        return self.model.predict(processed, verbose=0)

    def explain(self, image, top_labels=5, num_samples=1000, num_features=10):
        """Explain image prediction"""
        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=top_labels,
            num_samples=num_samples
        )
        return explanation

    def visualize_explanation(self, image, explanation, class_idx,
                             show_positive_only=True, num_features=10):
        """Create visualization of explanation"""
        temp_img, mask = explanation.get_image_and_mask(
            class_idx,
            positive_only=show_positive_only,
            num_features=num_features,
            hide_rest=False
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].imshow(image.astype(np.uint8))
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mark_boundaries(temp_img, mask))
        axes[1].set_title(f'LIME Explanation (Class {class_idx})')
        axes[1].axis('off')

        plt.tight_layout()
        return fig

# Usage
image_lime = ImageLIMEExplainer(resnet_model, preprocess_fn=preprocess_input)
exp = image_lime.explain(img, top_labels=3, num_samples=500)
fig = image_lime.visualize_explanation(img, exp, exp.top_labels[0])
plt.savefig('custom_lime_image.png', dpi=300)
```

### 4.5 Complete LIME Implementation

Unified LIME class supporting tabular, text, and image data.

```python
class UnifiedLIMEExplainer:
    """
    Unified LIME explainer supporting multiple data types.
    Automatically detects data type and applies appropriate explanation method.
    """

    def __init__(self, model, data_type='auto', **kwargs):
        """
        Args:
            model: Trained model
            data_type: 'tabular', 'text', 'image', or 'auto'
            **kwargs: Data-specific parameters
        """
        self.model = model
        self.data_type = data_type
        self.kwargs = kwargs
        self.explainer = None
        self.explanations_cache = {}

    def _initialize_explainer(self, data):
        """Initialize appropriate LIME explainer based on data type"""
        if self.data_type == 'auto':
            self.data_type = self._detect_data_type(data)

        if self.data_type == 'tabular':
            self.explainer = lime_tabular.LimeTabularExplainer(
                training_data=self.kwargs.get('training_data'),
                feature_names=self.kwargs.get('feature_names'),
                class_names=self.kwargs.get('class_names', ['0', '1']),
                mode=self.kwargs.get('mode', 'classification')
            )
        elif self.data_type == 'text':
            self.explainer = LimeTextExplainer(
                class_names=self.kwargs.get('class_names', ['0', '1'])
            )
        elif self.data_type == 'image':
            self.explainer = LimeImageExplainer()
        else:
            raise ValueError(f"Unknown data_type: {self.data_type}")

    def _detect_data_type(self, data):
        """Auto-detect data type"""
        if isinstance(data, str):
            return 'text'
        elif isinstance(data, np.ndarray):
            if data.ndim == 3 or data.ndim == 4:  # Image
                return 'image'
            else:  # Tabular
                return 'tabular'
        else:
            return 'tabular'

    def explain(self, instance, predict_fn=None, num_features=10, num_samples=5000):
        """
        Explain instance (auto-detects type).

        Args:
            instance: Data instance to explain
            predict_fn: Model prediction function (uses self.model if None)
            num_features: Number of features to include
            num_samples: Number of perturbed samples

        Returns:
            LIME explanation object
        """
        if self.explainer is None:
            self._initialize_explainer(instance)

        if predict_fn is None:
            if hasattr(self.model, 'predict_proba'):
                predict_fn = self.model.predict_proba
            else:
                predict_fn = self.model.predict

        # Explain based on data type
        if self.data_type == 'tabular':
            explanation = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
        elif self.data_type == 'text':
            explanation = self.explainer.explain_instance(
                instance,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples
            )
        elif self.data_type == 'image':
            explanation = self.explainer.explain_instance(
                instance,
                predict_fn,
                top_labels=num_features,
                num_samples=num_samples
            )

        return explanation

    def explain_to_api_response(self, instance, predict_fn=None, num_features=5):
        """
        Generate API-friendly explanation response.

        Returns:
            JSON-serializable dict
        """
        explanation = self.explain(instance, predict_fn, num_features)

        if self.data_type in ['tabular', 'text']:
            contributions = [
                {'feature': feat, 'weight': float(weight)}
                for feat, weight in explanation.as_list()
            ]

            return {
                'data_type': self.data_type,
                'prediction': self._get_prediction(instance, predict_fn),
                'contributions': contributions,
                'num_features': num_features
            }
        elif self.data_type == 'image':
            return {
                'data_type': 'image',
                'prediction': self._get_prediction(instance, predict_fn),
                'top_classes': explanation.top_labels,
                'message': 'Image explanation requires visualization'
            }

    def _get_prediction(self, instance, predict_fn):
        """Get model prediction"""
        if predict_fn is None:
            predict_fn = self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict

        if self.data_type == 'image':
            instance_batch = instance.reshape(1, *instance.shape)
        elif self.data_type == 'text':
            instance_batch = [instance]
        else:
            instance_batch = instance.reshape(1, -1)

        pred = predict_fn(instance_batch)
        return pred[0].tolist() if hasattr(pred[0], 'tolist') else float(pred[0])

# Usage examples
# Tabular
unified_tabular = UnifiedLIMEExplainer(
    model=xgb_model,
    data_type='tabular',
    training_data=X_train,
    feature_names=feature_names,
    class_names=['Negative', 'Positive']
)
tabular_exp = unified_tabular.explain_to_api_response(X_test[0], num_features=5)
print("Tabular explanation:")
print(json.dumps(tabular_exp, indent=2))

# Text
unified_text = UnifiedLIMEExplainer(
    model=text_clf,
    data_type='text',
    class_names=['Negative', 'Positive']
)
text_exp = unified_text.explain_to_api_response(
    "Great movie with excellent acting",
    num_features=5
)
print("\nText explanation:")
print(json.dumps(text_exp, indent=2))
```

---

## 5. Grad-CAM and Saliency Maps

### 5.1 Gradient-Based Visualization Theory

**Gradient-based methods** explain CNN predictions by computing gradients of the output with respect to input pixels. High gradients indicate important regions.

**Key Concept**: If changing a pixel significantly changes the prediction, that pixel is important.

**Types**:
1. **Vanilla Gradients**: Raw gradients of output w.r.t. input
2. **Saliency Maps**: Absolute value of gradients
3. **Grad-CAM**: Gradients of output w.r.t. feature maps (class activation mapping)
4. **Guided Backpropagation**: Modified gradients filtering negative values
5. **Integrated Gradients**: Path-integrated gradients (section 6)

### 5.2 Vanilla Gradients and Saliency Maps

**Saliency maps** show which pixels have the highest impact on the prediction.

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Load pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Load and preprocess image
def load_and_preprocess_image(img_path, size=(224, 224)):
    """Load image and prepare for model"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dummy image for demonstration
    img = Image.fromarray((np.random.rand(224, 224, 3) * 255).astype(np.uint8))
    img_tensor = transform(img).unsqueeze(0)
    img_tensor.requires_grad = True

    return img_tensor, img

img_tensor, original_img = load_and_preprocess_image('dummy_path.jpg')

# Forward pass
output = model(img_tensor)
predicted_class = output.argmax(dim=1).item()

# Backward pass to get gradients
model.zero_grad()
output[0, predicted_class].backward()

# Get gradients
gradients = img_tensor.grad.data.squeeze().numpy()

# Create saliency map (take max across color channels)
saliency_map = np.max(np.abs(gradients), axis=0)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(original_img)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(saliency_map, cmap='hot')
axes[1].set_title('Saliency Map')
axes[1].axis('off')

# Overlay
axes[2].imshow(original_img)
axes[2].imshow(saliency_map, cmap='hot', alpha=0.5)
axes[2].set_title('Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('saliency_map.png', dpi=300, bbox_inches='tight')
```

**Production saliency map class**:

```python
class SaliencyMapGenerator:
    """Generate saliency maps for CNN models (PyTorch)"""

    def __init__(self, model, device='cpu'):
        """
        Args:
            model: PyTorch CNN model
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def generate_saliency(self, image_tensor, target_class=None):
        """
        Generate saliency map for image.

        Args:
            image_tensor: Input image tensor (1, C, H, W) or (C, H, W)
            target_class: Target class index (uses predicted class if None)

        Returns:
            saliency_map: numpy array (H, W)
        """
        # Ensure correct shape
        if image_tensor.ndim == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Move to device and require gradients
        image_tensor = image_tensor.to(self.device)
        image_tensor.requires_grad = True

        # Forward pass
        output = self.model(image_tensor)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients
        gradients = image_tensor.grad.data.squeeze().cpu().numpy()

        # Take max across channels
        saliency_map = np.max(np.abs(gradients), axis=0)

        return saliency_map, target_class

    def generate_smooth_saliency(self, image_tensor, target_class=None,
                                 noise_level=0.2, num_samples=50):
        """
        Generate smoothed saliency map (SmoothGrad).
        Averages saliency maps from noisy versions of image.

        Args:
            image_tensor: Input image
            target_class: Target class
            noise_level: Std of Gaussian noise
            num_samples: Number of noisy samples

        Returns:
            Smoothed saliency map
        """
        saliency_maps = []

        for _ in range(num_samples):
            # Add Gaussian noise
            noise = torch.randn_like(image_tensor) * noise_level
            noisy_image = image_tensor + noise

            # Generate saliency
            saliency, _ = self.generate_saliency(noisy_image, target_class)
            saliency_maps.append(saliency)

        # Average saliency maps
        smooth_saliency = np.mean(saliency_maps, axis=0)

        return smooth_saliency

    def visualize(self, image, saliency_map, title='Saliency Map'):
        """Visualize saliency map"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Convert image tensor to numpy if needed
        if torch.is_tensor(image):
            image_np = image.squeeze().permute(1, 2, 0).cpu().numpy()
            # Denormalize if needed
            image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        else:
            image_np = image

        axes[0].imshow(image_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(saliency_map, cmap='hot')
        axes[1].set_title(title)
        axes[1].axis('off')

        axes[2].imshow(image_np)
        axes[2].imshow(saliency_map, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

# Usage
device = 'cuda' if torch.cuda.is_available() else 'cpu'
saliency_gen = SaliencyMapGenerator(model, device=device)

# Generate saliency
saliency, pred_class = saliency_gen.generate_saliency(img_tensor)
print(f"Predicted class: {pred_class}")

# Generate smooth saliency
smooth_saliency = saliency_gen.generate_smooth_saliency(img_tensor, num_samples=30)

# Visualize
fig1 = saliency_gen.visualize(img_tensor, saliency, 'Vanilla Saliency')
fig2 = saliency_gen.visualize(img_tensor, smooth_saliency, 'Smooth Saliency (SmoothGrad)')

fig1.savefig('vanilla_saliency.png', dpi=300, bbox_inches='tight')
fig2.savefig('smooth_saliency.png', dpi=300, bbox_inches='tight')
```

### 5.3 Grad-CAM Implementation

**Grad-CAM (Gradient-weighted Class Activation Mapping)** produces visual explanations by using gradients of the target class flowing into the final convolutional layer.

**Algorithm**:
1. Forward pass: Get feature maps from final conv layer
2. Backward pass: Get gradients of target class w.r.t. feature maps
3. Global average pooling of gradients --> weights
4. Weighted combination of feature maps
5. ReLU to keep only positive influences
6. Upsample to input image size

```python
class GradCAM:
    """
    Grad-CAM implementation for PyTorch models.
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch model
            target_layer: Target convolutional layer for Grad-CAM
        """
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        # Hooks to capture activations and gradients
        self.activations = None
        self.gradients = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            target_class: Target class index (uses predicted if None)

        Returns:
            cam: Grad-CAM heatmap (H, W)
            prediction: Model prediction
        """
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get activations and gradients
        activations = self.activations  # Shape: (1, C, H', W')
        gradients = self.gradients      # Shape: (1, C, H', W')

        # Global average pooling of gradients --> weights
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # Shape: (1, C, 1, 1)

        # Weighted combination
        cam = (weights * activations).sum(dim=1).squeeze()  # Shape: (H', W')

        # ReLU
        cam = torch.clamp(cam, min=0)

        # Normalize
        cam = cam / cam.max() if cam.max() > 0 else cam

        return cam.cpu().numpy(), target_class

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap='jet'):
        """
        Overlay heatmap on original image.

        Args:
            heatmap: Grad-CAM heatmap (H, W)
            image: Original image (H, W, 3)
            alpha: Transparency of heatmap
            colormap: Matplotlib colormap

        Returns:
            Overlayed image
        """
        # Resize heatmap to image size
        from skimage.transform import resize
        heatmap_resized = resize(heatmap, image.shape[:2])

        # Apply colormap
        import matplotlib.cm as cm
        cmap = cm.get_cmap(colormap)
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB only

        # Normalize image
        image_normalized = (image - image.min()) / (image.max() - image.min() + 1e-8)

        # Overlay
        overlayed = alpha * heatmap_colored + (1 - alpha) * image_normalized
        overlayed = np.clip(overlayed, 0, 1)

        return overlayed

# Usage with ResNet50
resnet = models.resnet50(pretrained=True)
resnet.eval()

# Target layer: last convolutional layer in ResNet50
target_layer = resnet.layer4[-1].conv3  # or resnet.layer4[-1]

# Create Grad-CAM
gradcam = GradCAM(resnet, target_layer)

# Generate CAM
cam, pred_class = gradcam.generate_cam(img_tensor)

print(f"Predicted class: {pred_class}")
print(f"CAM shape: {cam.shape}")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
original_np = original_img if not torch.is_tensor(original_img) else original_img.squeeze().permute(1, 2, 0).numpy()

axes[0].imshow(original_np)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(cam, cmap='jet')
axes[1].set_title('Grad-CAM')
axes[1].axis('off')

# Overlay
overlayed = gradcam.overlay_heatmap(cam, original_np)
axes[2].imshow(overlayed)
axes[2].set_title('Grad-CAM Overlay')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('gradcam_visualization.png', dpi=300, bbox_inches='tight')
```

**Grad-CAM for different architectures**:

```python
def get_target_layer(model_name):
    """
    Get appropriate target layer for different architectures.

    Returns last convolutional layer for common architectures.
    """
    if 'resnet' in model_name.lower():
        return model.layer4[-1]
    elif 'vgg' in model_name.lower():
        return model.features[-1]
    elif 'densenet' in model_name.lower():
        return model.features.denseblock4
    elif 'efficientnet' in model_name.lower():
        return model.features[-1]
    elif 'mobilenet' in model_name.lower():
        return model.features[-1]
    else:
        raise ValueError(f"Unknown architecture: {model_name}")

# Example: VGG16
vgg16 = models.vgg16(pretrained=True)
vgg16.eval()

target_layer_vgg = get_target_layer('vgg16')
gradcam_vgg = GradCAM(vgg16, target_layer_vgg)
cam_vgg, _ = gradcam_vgg.generate_cam(img_tensor)

# Example: DenseNet
densenet = models.densenet121(pretrained=True)
densenet.eval()

target_layer_densenet = get_target_layer('densenet121')
gradcam_densenet = GradCAM(densenet, target_layer_densenet)
cam_densenet, _ = gradcam_densenet.generate_cam(img_tensor)
```

### 5.4 Guided Grad-CAM

**Guided Grad-CAM** combines Grad-CAM with Guided Backpropagation for high-resolution, class-discriminative visualizations.

```python
class GuidedBackpropagation:
    """Guided Backpropagation for fine-grained visualizations"""

    def __init__(self, model):
        self.model = model
        self.model.eval()

        # Store original ReLU backward functions
        self.relu_outputs = []
        self.update_relus()

    def update_relus(self):
        """Update ReLU layers to use guided backprop"""
        def relu_backward_hook(module, grad_in, grad_out):
            """Only backpropagate positive gradients"""
            return (torch.clamp(grad_in[0], min=0.0),)

        # Find all ReLU layers and add hooks
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                module.register_backward_hook(relu_backward_hook)

    def generate_gradients(self, input_tensor, target_class=None):
        """Generate guided gradients"""
        input_tensor.requires_grad = True

        # Forward
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients
        gradients = input_tensor.grad.data.squeeze().cpu().numpy()

        return gradients

class GuidedGradCAM:
    """Combine Grad-CAM with Guided Backpropagation"""

    def __init__(self, model, target_layer):
        self.gradcam = GradCAM(model, target_layer)
        self.guided_backprop = GuidedBackpropagation(model)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Guided Grad-CAM.

        Returns:
            guided_gradcam: High-resolution class-discriminative visualization
        """
        # Get Grad-CAM
        cam, pred_class = self.gradcam.generate_cam(input_tensor, target_class)

        # Get Guided Backprop
        guided_grads = self.guided_backprop.generate_gradients(input_tensor, pred_class)

        # Resize CAM to input size
        from skimage.transform import resize
        cam_resized = resize(cam, guided_grads.shape[1:])

        # Element-wise multiplication
        guided_gradcam = cam_resized[np.newaxis, :, :] * guided_grads

        return guided_gradcam, pred_class

# Usage
guided_gradcam = GuidedGradCAM(resnet, target_layer)
ggcam, pred_class = guided_gradcam.generate(img_tensor)

# Visualize
fig, axes = plt.subplots(1, 4, figsize=(20, 5))

axes[0].imshow(original_np)
axes[0].set_title('Original')
axes[0].axis('off')

axes[1].imshow(cam, cmap='jet')
axes[1].set_title('Grad-CAM')
axes[1].axis('off')

# Guided backprop (take max across channels)
guided_bp = guided_gradcam.guided_backprop.generate_gradients(img_tensor)
guided_bp_vis = np.max(np.abs(guided_bp), axis=0)
axes[2].imshow(guided_bp_vis, cmap='gray')
axes[2].set_title('Guided Backprop')
axes[2].axis('off')

# Guided Grad-CAM
ggcam_vis = np.max(np.abs(ggcam), axis=0)
axes[3].imshow(ggcam_vis, cmap='hot')
axes[3].set_title('Guided Grad-CAM')
axes[3].axis('off')

plt.tight_layout()
plt.savefig('guided_gradcam.png', dpi=300, bbox_inches='tight')
```

---

## 6. Integrated Gradients

### 6.1 Attribution Theory

**Integrated Gradients (IG)** is an attribution method that assigns importance scores to input features by integrating gradients along a straight path from a baseline to the input.

**Key Properties**:
1. **Sensitivity**: If input and baseline differ in one feature and have different predictions, that feature gets non-zero attribution
2. **Implementation Invariance**: Attributions are the same for functionally equivalent models
3. **Completeness**: Attributions sum to difference between output at input and baseline

**Formula**:

```
IG_i(x) = (x_i - x'_i) * integral from 0 to 1 of:
    (d/dx_i) F(x' + alpha * (x - x'))_alpha d_alpha
```

Where:
- x = input
- x' = baseline (typically zero vector or random noise)
- F = model function
- i = feature index

**Approximation** (Riemann sum):

```
IG_i(x) approx= (x_i - x'_i) * sum from k=1 to m of:
    (d/dx_i) F(x' + k/m * (x - x')) / m
```

Where m = number of steps (typically 50-300).

### 6.2 Implementation for Neural Networks

```python
class IntegratedGradients:
    """
    Integrated Gradients implementation for PyTorch models.
    """

    def __init__(self, model):
        """
        Args:
            model: PyTorch model
        """
        self.model = model
        self.model.eval()

    def generate_attributions(self, input_tensor, target_class=None,
                              baseline=None, steps=50):
        """
        Generate integrated gradients attributions.

        Args:
            input_tensor: Input to explain (1, C, H, W) or (1, D)
            target_class: Target class index (uses predicted if None)
            baseline: Baseline input (uses zeros if None)
            steps: Number of interpolation steps

        Returns:
            attributions: Same shape as input_tensor
        """
        # Set baseline to zeros if not provided
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Ensure baseline doesn't require grad
        baseline = baseline.detach()

        # Get prediction for target class
        with torch.no_grad():
            output = self.model(input_tensor)
            if target_class is None:
                target_class = output.argmax(dim=1).item()

        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, steps + 1).to(input_tensor.device)

        # Compute gradients for each interpolated input
        gradients = []

        for alpha in alphas:
            # Interpolated input
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad = True

            # Forward pass
            output = self.model(interpolated)

            # Backward pass
            self.model.zero_grad()
            output[0, target_class].backward()

            # Store gradient
            gradients.append(interpolated.grad.detach())

        # Average gradients (trapezoidal rule)
        gradients = torch.stack(gradients)
        avg_gradients = gradients.mean(dim=0)

        # Integrated gradients = (input - baseline) * avg_gradients
        integrated_grads = (input_tensor - baseline) * avg_gradients

        return integrated_grads.detach(), target_class

    def visualize_attributions(self, attributions, original_image, title='Integrated Gradients'):
        """Visualize attributions"""
        # Convert to numpy
        if torch.is_tensor(attributions):
            attr_np = attributions.squeeze().cpu().numpy()
        else:
            attr_np = attributions

        if torch.is_tensor(original_image):
            img_np = original_image.squeeze().permute(1, 2, 0).cpu().numpy()
            img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
        else:
            img_np = original_image

        # For images, take max across channels
        if attr_np.ndim == 3:
            attr_vis = np.max(np.abs(attr_np), axis=0)
        else:
            attr_vis = np.abs(attr_np)

        # Normalize
        attr_vis = (attr_vis - attr_vis.min()) / (attr_vis.max() - attr_vis.min() + 1e-8)

        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(img_np)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(attr_vis, cmap='hot')
        axes[1].set_title(title)
        axes[1].axis('off')

        axes[2].imshow(img_np)
        axes[2].imshow(attr_vis, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        return fig

# Usage with image model
ig_generator = IntegratedGradients(resnet)

# Generate attributions
attributions, pred_class = ig_generator.generate_attributions(
    img_tensor,
    steps=100  # More steps = more accurate
)

print(f"Predicted class: {pred_class}")
print(f"Attributions shape: {attributions.shape}")

# Visualize
fig = ig_generator.visualize_attributions(attributions, img_tensor)
fig.savefig('integrated_gradients.png', dpi=300, bbox_inches='tight')
```

**Comparing different baselines**:

```python
def compare_baselines(model, input_tensor, baselines_dict, steps=50):
    """
    Compare Integrated Gradients with different baselines.

    Args:
        model: PyTorch model
        input_tensor: Input to explain
        baselines_dict: Dict of {name: baseline_tensor}
        steps: IG steps
    """
    ig = IntegratedGradients(model)

    results = {}

    for name, baseline in baselines_dict.items():
        attr, pred_class = ig.generate_attributions(
            input_tensor,
            baseline=baseline,
            steps=steps
        )
        results[name] = attr

    # Visualize comparison
    n_baselines = len(baselines_dict)
    fig, axes = plt.subplots(1, n_baselines + 1, figsize=(5 * (n_baselines + 1), 5))

    # Original image
    img_np = input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    axes[0].imshow(img_np)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Attributions for each baseline
    for i, (name, attr) in enumerate(results.items(), 1):
        attr_vis = np.max(np.abs(attr.squeeze().cpu().numpy()), axis=0)
        attr_vis = (attr_vis - attr_vis.min()) / (attr_vis.max() - attr_vis.min() + 1e-8)

        axes[i].imshow(attr_vis, cmap='hot')
        axes[i].set_title(f'IG - {name}')
        axes[i].axis('off')

    plt.tight_layout()
    return fig, results

# Define different baselines
baselines = {
    'zeros': torch.zeros_like(img_tensor),
    'random_noise': torch.randn_like(img_tensor) * 0.1,
    'mean_image': torch.ones_like(img_tensor) * 0.5,
    'blurred': torch.from_numpy(
        np.random.rand(*img_tensor.shape).astype(np.float32)
    )  # Placeholder for actual blur
}

fig_baselines, results_baselines = compare_baselines(
    resnet,
    img_tensor,
    baselines,
    steps=50
)
fig_baselines.savefig('ig_baseline_comparison.png', dpi=300, bbox_inches='tight')
```

### 6.3 Integrated Gradients for NLP

**Integrated Gradients for text** - attributions at token/word level.

```python
class IntegratedGradientsText:
    """Integrated Gradients for NLP models"""

    def __init__(self, model, tokenizer, device='cpu'):
        """
        Args:
            model: Transformer model (BERT, GPT, etc.)
            tokenizer: Corresponding tokenizer
            device: 'cpu' or 'cuda'
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    def generate_attributions(self, text, target_class=None, steps=50):
        """
        Generate integrated gradients for text input.

        Args:
            text: Input text string
            target_class: Target class index
            steps: Number of integration steps

        Returns:
            attributions: List of (token, attribution_score) tuples
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Get embeddings
        embeddings = self.model.get_input_embeddings()(inputs['input_ids'])

        # Baseline: zero embeddings
        baseline_embeddings = torch.zeros_like(embeddings)

        # Get target class
        with torch.no_grad():
            output = self.model(**inputs).logits
            if target_class is None:
                target_class = output.argmax(dim=-1).item()

        # Integrated gradients
        alphas = torch.linspace(0, 1, steps + 1).to(self.device)
        gradients = []

        for alpha in alphas:
            # Interpolated embeddings
            interpolated_emb = baseline_embeddings + alpha * (embeddings - baseline_embeddings)
            interpolated_emb.requires_grad = True

            # Forward pass (replace input_ids with embeddings)
            # This requires custom forward - simplified here
            outputs = self.model.base_model(inputs_embeds=interpolated_emb,
                                           attention_mask=inputs['attention_mask'])
            logits = self.model.classifier(outputs.last_hidden_state[:, 0, :])  # CLS token

            # Backward
            self.model.zero_grad()
            logits[0, target_class].backward()

            gradients.append(interpolated_emb.grad.detach())

        # Average gradients
        gradients = torch.stack(gradients)
        avg_gradients = gradients.mean(dim=0)

        # Integrated gradients
        integrated_grads = (embeddings - baseline_embeddings) * avg_gradients

        # Sum across embedding dimension
        token_attributions = integrated_grads.sum(dim=-1).squeeze()

        # Convert to list with tokens
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        attributions = [(token, float(attr)) for token, attr in zip(tokens, token_attributions)]

        return attributions, target_class

    def visualize_text_attributions(self, attributions, normalize=True):
        """
        Visualize text attributions with color coding.

        Args:
            attributions: List of (token, score) tuples
            normalize: Normalize scores to [0, 1]
        """
        tokens = [a[0] for a in attributions]
        scores = np.array([a[1] for a in attributions])

        if normalize:
            scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Create HTML visualization
        html = "<div style='font-family: monospace; line-height: 2;'>"

        for token, score in zip(tokens, scores):
            # Color: red for negative, green for positive
            if score > 0:
                color = f'rgba(0, 255, 0, {abs(score)})'
            else:
                color = f'rgba(255, 0, 0, {abs(score)})'

            html += f"<span style='background-color: {color}; padding: 2px;'>{token}</span> "

        html += "</div>"

        return html

# Example usage (simplified - requires proper model setup)
# from transformers import AutoModel, AutoTokenizer
#
# model = AutoModel.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
#
# ig_text = IntegratedGradientsText(model, tokenizer)
# attributions, pred_class = ig_text.generate_attributions(
#     "This movie was absolutely fantastic and entertaining"
# )
#
# print("Token attributions:")
# for token, score in attributions:
#     print(f"{token}: {score:+.4f}")
#
# html_viz = ig_text.visualize_text_attributions(attributions)
# with open('text_ig.html', 'w') as f:
#     f.write(html_viz)
```

**Production-ready IG class with caching**:

```python
import hashlib
import pickle

class ProductionIntegratedGradients:
    """
    Production IG with caching and batch processing.
    """

    def __init__(self, model, cache_dir='./ig_cache', device='cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.ig = IntegratedGradients(model)

    def _get_cache_key(self, input_tensor, target_class, baseline_type, steps):
        """Generate cache key from inputs"""
        input_hash = hashlib.md5(input_tensor.cpu().numpy().tobytes()).hexdigest()
        key = f"{input_hash}_{target_class}_{baseline_type}_{steps}"
        return key

    def generate_with_cache(self, input_tensor, target_class=None,
                           baseline=None, steps=50, use_cache=True):
        """Generate attributions with optional caching"""
        baseline_type = 'zeros' if baseline is None else 'custom'
        cache_key = self._get_cache_key(input_tensor, target_class, baseline_type, steps)
        cache_path = self.cache_dir / f"{cache_key}.pkl"

        # Check cache
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        # Compute attributions
        attr, pred_class = self.ig.generate_attributions(
            input_tensor, target_class, baseline, steps
        )

        result = {'attributions': attr, 'predicted_class': pred_class}

        # Cache result
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump(result, f)

        return result

    def batch_generate(self, input_batch, steps=50, batch_size=4):
        """Generate attributions for batch of inputs"""
        n_samples = input_batch.shape[0]
        results = []

        for i in range(0, n_samples, batch_size):
            batch = input_batch[i:min(i+batch_size, n_samples)]

            for sample in batch:
                result = self.generate_with_cache(
                    sample.unsqueeze(0),
                    steps=steps,
                    use_cache=True
                )
                results.append(result)

            print(f"Processed {min(i+batch_size, n_samples)}/{n_samples}")

        return results

# Usage
prod_ig = ProductionIntegratedGradients(resnet, device=device)

# Single with caching
result = prod_ig.generate_with_cache(img_tensor, steps=100, use_cache=True)
print(f"Cached result - class: {result['predicted_class']}")

# Batch processing
batch_results = prod_ig.batch_generate(
    torch.randn(10, 3, 224, 224),  # 10 samples
    steps=50,
    batch_size=2
)
print(f"Processed {len(batch_results)} samples")
```

---

## 7. Counterfactual Explanations

### 7.1 What are Counterfactuals?

**Counterfactual explanations** answer "What would need to change for a different outcome?" They provide actionable insights by showing minimal changes to input features that would flip the prediction.

**Example**: "Your loan was denied. If your income were $65K instead of $50K, the loan would be approved."

**Properties of good counterfactuals**:
1. **Validity**: Changed instance produces desired outcome
2. **Proximity**: Minimal changes from original instance
3. **Sparsity**: Few features changed
4. **Actionability**: Changes are feasible (can't change age from 60 to 25)
5. **Diversity**: Multiple different counterfactual paths

**Use Cases**:
- Credit/loan decisions - show path to approval
- Healthcare - alternative treatment outcomes
- Recourse - how to change unfavorable decisions
- Model debugging - find decision boundaries

### 7.2 DiCE (Diverse Counterfactual Explanations)

**DiCE** generates diverse counterfactual explanations using optimization.

```python
import dice_ml
from dice_ml import Data, Model, Dice

# Prepare dataset for DiCE
df = pd.DataFrame(X_train, columns=feature_names)
df['target'] = y_train

# Define continuous and categorical features
continuous_features = feature_names[:8]  # Assume first 8 are continuous
categorical_features = feature_names[8:]  # Rest are categorical

# Create DiCE data object
d = dice_ml.Data(
    dataframe=df,
    continuous_features=continuous_features,
    outcome_name='target'
)

# Create DiCE model object
m = dice_ml.Model(model=xgb_model, backend='sklearn', model_type='classifier')

# Create DiCE explainer
dice_exp = Dice(d, m, method='random')  # Methods: 'random', 'genetic', 'kdtree'

# Generate counterfactuals for test instance
query_instance = pd.DataFrame(X_test[0:1], columns=feature_names)

counterfactuals = dice_exp.generate_counterfactuals(
    query_instance,
    total_CFs=5,  # Number of counterfactuals
    desired_class='opposite',  # 'opposite' or specific class
    proximity_weight=0.5,  # Weight for proximity vs diversity
    diversity_weight=1.0
)

# Visualize
counterfactuals.visualize_as_dataframe(show_only_changes=True)
```

**Complete DiCE implementation with constraints**:

```python
class CounterfactualExplainer:
    """
    Production counterfactual explainer with DiCE.
    Supports feature constraints and actionability.
    """

    def __init__(self, model, data, continuous_features, categorical_features=None,
                 outcome_name='target', backend='sklearn'):
        """
        Args:
            model: Trained model
            data: Training dataframe (with outcome column)
            continuous_features: List of continuous feature names
            categorical_features: List of categorical feature names
            outcome_name: Target column name
            backend: 'sklearn', 'tensorflow', or 'pytorch'
        """
        self.model = model
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features or []
        self.outcome_name = outcome_name

        # Create DiCE objects
        self.d = dice_ml.Data(
            dataframe=data,
            continuous_features=continuous_features,
            outcome_name=outcome_name
        )

        self.m = dice_ml.Model(
            model=model,
            backend=backend,
            model_type='classifier'
        )

        # Initialize explainer (will be created with specific method)
        self.explainer = None

    def generate_counterfactuals(self, query_instance, method='random',
                                 total_CFs=5, desired_class='opposite',
                                 features_to_vary='all', permitted_range=None,
                                 feature_weights=None):
        """
        Generate diverse counterfactual explanations.

        Args:
            query_instance: Instance to explain (DataFrame or dict)
            method: 'random', 'genetic', or 'kdtree'
            total_CFs: Number of counterfactuals to generate
            desired_class: Target class ('opposite' or specific class)
            features_to_vary: List of features allowed to change ('all' or list)
            permitted_range: Dict of {feature: [min, max]} constraints
            feature_weights: Dict of {feature: weight} for optimization

        Returns:
            DiCE counterfactuals object
        """
        # Create explainer with specified method
        self.explainer = Dice(self.d, self.m, method=method)

        # Convert query to DataFrame if needed
        if not isinstance(query_instance, pd.DataFrame):
            query_instance = pd.DataFrame([query_instance], columns=self.continuous_features + self.categorical_features)

        # Generate counterfactuals
        cf = self.explainer.generate_counterfactuals(
            query_instance,
            total_CFs=total_CFs,
            desired_class=desired_class,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range,
            feature_weights=feature_weights
        )

        return cf

    def explain_with_constraints(self, query_instance, immutable_features=None,
                                 monotonic_increase=None, monotonic_decrease=None,
                                 total_CFs=3):
        """
        Generate counterfactuals with actionability constraints.

        Args:
            query_instance: Instance to explain
            immutable_features: List of features that cannot change (e.g., age, race)
            monotonic_increase: Features that can only increase (e.g., education)
            monotonic_decrease: Features that can only decrease (e.g., debt)
            total_CFs: Number of counterfactuals

        Returns:
            Constrained counterfactuals
        """
        # Build features_to_vary (exclude immutable)
        all_features = self.continuous_features + self.categorical_features
        if immutable_features:
            features_to_vary = [f for f in all_features if f not in immutable_features]
        else:
            features_to_vary = 'all'

        # Build permitted ranges
        permitted_range = {}

        if isinstance(query_instance, pd.DataFrame):
            query_values = query_instance.iloc[0]
        else:
            query_values = query_instance

        if monotonic_increase:
            for feat in monotonic_increase:
                if feat in query_values:
                    # Can only increase
                    permitted_range[feat] = [float(query_values[feat]), float('inf')]

        if monotonic_decrease:
            for feat in monotonic_decrease:
                if feat in query_values:
                    # Can only decrease
                    permitted_range[feat] = [float('-inf'), float(query_values[feat])]

        # Generate counterfactuals
        cf = self.generate_counterfactuals(
            query_instance,
            method='genetic',  # Better for constraints
            total_CFs=total_CFs,
            features_to_vary=features_to_vary,
            permitted_range=permitted_range if permitted_range else None
        )

        return cf

    def compare_counterfactuals(self, original, counterfactuals_df):
        """
        Compare original instance with counterfactuals.

        Returns:
            DataFrame showing changes
        """
        changes = []

        for idx, cf_row in counterfactuals_df.iterrows():
            change_dict = {'counterfactual_id': idx}

            for feature in original.index:
                if feature == self.outcome_name:
                    continue

                orig_val = original[feature]
                cf_val = cf_row[feature]

                if orig_val != cf_val:
                    change_dict[feature] = {
                        'original': orig_val,
                        'counterfactual': cf_val,
                        'change': cf_val - orig_val if isinstance(cf_val, (int, float)) else 'categorical_change'
                    }

            changes.append(change_dict)

        return changes

    def generate_actionable_recourse(self, query_instance, total_CFs=3):
        """
        Generate actionable recourse - practical steps to change outcome.

        Returns:
            List of actionable recommendations
        """
        # Define typically immutable features
        immutable = ['age', 'race', 'gender', 'ethnicity']  # Adjust based on dataset

        # Define features that should only increase (positive direction)
        increase_only = ['education_level', 'years_experience', 'savings']

        # Generate constrained counterfactuals
        cf = self.explain_with_constraints(
            query_instance,
            immutable_features=immutable,
            monotonic_increase=increase_only,
            total_CFs=total_CFs
        )

        # Extract as DataFrame
        cf_df = cf.cf_examples_list[0].final_cfs_df

        # Compare changes
        if isinstance(query_instance, pd.DataFrame):
            original = query_instance.iloc[0]
        else:
            original = pd.Series(query_instance)

        changes = self.compare_counterfactuals(original, cf_df)

        # Create actionable recommendations
        recommendations = []

        for i, change in enumerate(changes):
            rec = {
                'recourse_option': i + 1,
                'actions': []
            }

            for feature, change_info in change.items():
                if feature == 'counterfactual_id':
                    continue

                if isinstance(change_info, dict):
                    action = f"Change {feature} from {change_info['original']:.2f} to {change_info['counterfactual']:.2f}"
                    if 'change' in change_info and isinstance(change_info['change'], (int, float)):
                        action += f" (change: {change_info['change']:+.2f})"

                    rec['actions'].append(action)

            recommendations.append(rec)

        return recommendations

# Complete example with realistic data
# Create sample credit scoring dataset
np.random.seed(42)
n_samples = 1000

credit_data = pd.DataFrame({
    'income': np.random.randint(20000, 150000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'debt': np.random.randint(0, 50000, n_samples),
    'years_employed': np.random.randint(0, 40, n_samples),
    'age': np.random.randint(18, 70, n_samples),
    'num_credit_cards': np.random.randint(0, 10, n_samples)
})

# Create target based on rules
credit_data['approved'] = (
    (credit_data['income'] > 50000) &
    (credit_data['credit_score'] > 650) &
    (credit_data['debt'] < 30000)
).astype(int)

# Train model
X_credit = credit_data.drop('approved', axis=1)
y_credit = credit_data['approved']

credit_model = RandomForestClassifier(n_estimators=100, random_state=42)
credit_model.fit(X_credit, y_credit)

# Create counterfactual explainer
cf_explainer = CounterfactualExplainer(
    model=credit_model,
    data=credit_data,
    continuous_features=['income', 'credit_score', 'debt', 'years_employed', 'age', 'num_credit_cards'],
    outcome_name='approved',
    backend='sklearn'
)

# Example: Denied loan applicant
denied_applicant = {
    'income': 45000,
    'credit_score': 620,
    'debt': 35000,
    'years_employed': 3,
    'age': 28,
    'num_credit_cards': 4
}

# Generate counterfactuals with constraints
cf_result = cf_explainer.explain_with_constraints(
    denied_applicant,
    immutable_features=['age'],  # Can't change age
    monotonic_increase=['income', 'credit_score', 'years_employed'],  # Should only increase
    monotonic_decrease=['debt'],  # Should only decrease
    total_CFs=3
)

print("Original prediction:", credit_model.predict([list(denied_applicant.values())])[0])
print("\nCounterfactual explanations:")
cf_result.visualize_as_dataframe(show_only_changes=True)

# Get actionable recourse
recourse = cf_explainer.generate_actionable_recourse(denied_applicant, total_CFs=3)

print("\nActionable Recourse Options:")
for rec in recourse:
    print(f"\nOption {rec['recourse_option']}:")
    for action in rec['actions']:
        print(f"  - {action}")
```

### 7.3 Custom Counterfactual Generation

**Optimization-based counterfactual generation** using custom objectives.

```python
from scipy.optimize import minimize

class CustomCounterfactualGenerator:
    """
    Custom counterfactual generator using optimization.
    Allows fine-grained control over objective function.
    """

    def __init__(self, model, feature_names, feature_ranges=None):
        """
        Args:
            model: Trained model with predict_proba method
            feature_names: List of feature names
            feature_ranges: Dict of {feature_idx: (min, max)}
        """
        self.model = model
        self.feature_names = feature_names
        self.feature_ranges = feature_ranges or {}

    def generate_counterfactual(self, original_instance, target_class,
                                lambda_proximity=1.0, lambda_sparsity=0.1,
                                lambda_plausibility=0.5, max_iterations=1000):
        """
        Generate single counterfactual using optimization.

        Args:
            original_instance: Original instance (1D array)
            target_class: Desired class
            lambda_proximity: Weight for proximity loss
            lambda_sparsity: Weight for sparsity (fewer changes)
            lambda_plausibility: Weight for plausibility (realistic values)
            max_iterations: Maximum optimization iterations

        Returns:
            Counterfactual instance
        """
        def objective(x_cf):
            """
            Objective function to minimize:
            L = prediction_loss + proximity_loss + sparsity_loss + plausibility_loss
            """
            # Reshape for model
            x_cf_2d = x_cf.reshape(1, -1)

            # Prediction loss (want high probability for target class)
            pred_proba = self.model.predict_proba(x_cf_2d)[0, target_class]
            prediction_loss = -np.log(pred_proba + 1e-10)  # Negative log likelihood

            # Proximity loss (L2 distance from original)
            proximity_loss = np.sum((x_cf - original_instance) ** 2)

            # Sparsity loss (L1 distance - encourages few changes)
            sparsity_loss = np.sum(np.abs(x_cf - original_instance))

            # Plausibility loss (distance from feature ranges)
            plausibility_loss = 0
            for feat_idx, (min_val, max_val) in self.feature_ranges.items():
                if x_cf[feat_idx] < min_val:
                    plausibility_loss += (min_val - x_cf[feat_idx]) ** 2
                elif x_cf[feat_idx] > max_val:
                    plausibility_loss += (x_cf[feat_idx] - max_val) ** 2

            # Total loss
            total_loss = (prediction_loss +
                         lambda_proximity * proximity_loss +
                         lambda_sparsity * sparsity_loss +
                         lambda_plausibility * plausibility_loss)

            return total_loss

        # Constraints
        bounds = []
        for i in range(len(original_instance)):
            if i in self.feature_ranges:
                bounds.append(self.feature_ranges[i])
            else:
                # Default: allow +/- 50% change
                lower = original_instance[i] * 0.5
                upper = original_instance[i] * 1.5
                bounds.append((lower, upper))

        # Optimize
        result = minimize(
            objective,
            x0=original_instance,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )

        if result.success:
            return result.x
        else:
            print(f"Optimization failed: {result.message}")
            return original_instance

    def generate_diverse_counterfactuals(self, original_instance, target_class,
                                        n_counterfactuals=5, diversity_weight=1.0):
        """
        Generate multiple diverse counterfactuals.

        Args:
            original_instance: Original instance
            target_class: Desired class
            n_counterfactuals: Number to generate
            diversity_weight: Weight for diversity among counterfactuals

        Returns:
            List of counterfactual instances
        """
        counterfactuals = []

        for i in range(n_counterfactuals):
            # Add random perturbation to initial point for diversity
            if i == 0:
                init = original_instance
            else:
                # Start from random perturbation
                init = original_instance + np.random.randn(len(original_instance)) * 0.1

            # Generate counterfactual
            cf = self.generate_counterfactual(
                init,
                target_class,
                lambda_proximity=1.0 - (diversity_weight * i / n_counterfactuals),
                lambda_sparsity=0.1,
                max_iterations=500
            )

            counterfactuals.append(cf)

        return counterfactuals

    def explain_changes(self, original, counterfactual, top_n=5):
        """
        Explain what changed from original to counterfactual.

        Returns:
            List of feature changes sorted by magnitude
        """
        changes = []

        for i, feature in enumerate(self.feature_names):
            orig_val = original[i]
            cf_val = counterfactual[i]
            change = cf_val - orig_val

            if abs(change) > 1e-6:  # Non-zero change
                changes.append({
                    'feature': feature,
                    'original': orig_val,
                    'counterfactual': cf_val,
                    'change': change,
                    'percent_change': (change / orig_val * 100) if orig_val != 0 else float('inf')
                })

        # Sort by absolute change magnitude
        changes.sort(key=lambda x: abs(x['change']), reverse=True)

        return changes[:top_n]

# Usage
custom_cf_gen = CustomCounterfactualGenerator(
    model=credit_model,
    feature_names=['income', 'credit_score', 'debt', 'years_employed', 'age', 'num_credit_cards'],
    feature_ranges={
        0: (0, 500000),  # income
        1: (300, 850),   # credit_score
        2: (0, 100000),  # debt
        3: (0, 50),      # years_employed
        4: (18, 100),    # age
        5: (0, 20)       # num_credit_cards
    }
)

# Original denied instance
denied_features = np.array([45000, 620, 35000, 3, 28, 4])

print("Original prediction:", credit_model.predict([denied_features])[0])
print("Original probability:", credit_model.predict_proba([denied_features])[0])

# Generate counterfactual for approval (class 1)
cf = custom_cf_gen.generate_counterfactual(
    denied_features,
    target_class=1,
    lambda_proximity=1.0,
    lambda_sparsity=0.2
)

print("\nCounterfactual prediction:", credit_model.predict([cf])[0])
print("Counterfactual probability:", credit_model.predict_proba([cf])[0])

# Explain changes
changes = custom_cf_gen.explain_changes(denied_features, cf, top_n=5)

print("\nRequired changes for approval:")
for change in changes:
    print(f"{change['feature']}:")
    print(f"  Original: {change['original']:.2f}")
    print(f"  Counterfactual: {change['counterfactual']:.2f}")
    print(f"  Change: {change['change']:+.2f} ({change['percent_change']:+.1f}%)")

# Generate diverse counterfactuals
diverse_cfs = custom_cf_gen.generate_diverse_counterfactuals(
    denied_features,
    target_class=1,
    n_counterfactuals=5,
    diversity_weight=0.3
)

print(f"\nGenerated {len(diverse_cfs)} diverse counterfactuals")
for i, cf in enumerate(diverse_cfs):
    print(f"CF {i+1} probability: {credit_model.predict_proba([cf])[0, 1]:.3f}")
```

---

## 8. Attention Visualization

### 8.1 Transformer Attention Mechanisms

**Attention mechanisms** in transformers learn to focus on relevant parts of the input. Visualizing attention weights can reveal what the model considers important.

**Types of attention**:
1. **Self-attention**: Relates different positions in a single sequence
2. **Cross-attention**: Relates positions between encoder and decoder
3. **Multi-head attention**: Multiple attention patterns learned in parallel

**Visualization challenges**:
- Multiple layers (12-24 in BERT/GPT)
- Multiple heads per layer (8-16 heads)
- Attention is not always interpretable
- High attention != high importance (attention is not explanation)

```python
from transformers import AutoModel, AutoTokenizer
import torch

# Load pre-trained BERT
model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()

# Input text
text = "The quick brown fox jumps over the lazy dog"
inputs = tokenizer(text, return_tensors='pt')

# Forward pass with attention outputs
with torch.no_grad():
    outputs = model(**inputs)

# Extract attention weights
attentions = outputs.attentions  # Tuple of length num_layers
# Each element shape: (batch_size, num_heads, seq_len, seq_len)

print(f"Number of layers: {len(attentions)}")
print(f"Attention shape per layer: {attentions[0].shape}")

# Get tokens
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print(f"Tokens: {tokens}")

# Visualize attention from last layer, first head
last_layer_attention = attentions[-1][0, 0].numpy()  # Shape: (seq_len, seq_len)

plt.figure(figsize=(10, 8))
plt.imshow(last_layer_attention, cmap='viridis')
plt.xticks(range(len(tokens)), tokens, rotation=90)
plt.yticks(range(len(tokens)), tokens)
plt.xlabel('Key (attended to)')
plt.ylabel('Query (attending from)')
plt.title('BERT Last Layer, Head 0 - Attention Weights')
plt.colorbar()
plt.tight_layout()
plt.savefig('bert_attention_heatmap.png', dpi=300)
```

**Average attention across heads and layers**:

```python
def visualize_average_attention(attentions, tokens, layer_idx=-1):
    """
    Visualize average attention across all heads in a layer.

    Args:
        attentions: Tuple of attention tensors
        tokens: List of token strings
        layer_idx: Layer index (-1 for last layer)
    """
    # Get layer attention: (batch, heads, seq_len, seq_len)
    layer_attention = attentions[layer_idx][0]  # Remove batch dim

    # Average across heads
    avg_attention = layer_attention.mean(dim=0).numpy()  # (seq_len, seq_len)

    # Plot
    plt.figure(figsize=(12, 10))
    plt.imshow(avg_attention, cmap='hot')
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)
    plt.xlabel('Key')
    plt.ylabel('Query')
    plt.title(f'Average Attention - Layer {layer_idx}')
    plt.colorbar(label='Attention Weight')
    plt.tight_layout()

    return avg_attention

avg_attn = visualize_average_attention(attentions, tokens, layer_idx=-1)
plt.savefig('bert_avg_attention.png', dpi=300)
```

### 8.2 BertViz for Interactive Attention Visualization

**BertViz** provides interactive visualizations for transformer attention.

```python
from bertviz import head_view, model_view

# Prepare inputs for BertViz
inputs = tokenizer.encode_plus(text, return_tensors='pt', add_special_tokens=True)
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']

# Get attention
with torch.no_grad():
    outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True)

attentions = outputs.attentions  # Tuple of tensors

# Get tokens
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

# Head view - shows attention patterns for each head
html_head_view = head_view(attentions, tokens)

# Save to file
with open('bertviz_head_view.html', 'w') as f:
    f.write(html_head_view.data)

print("BertViz head view saved to bertviz_head_view.html")

# Model view - shows attention across all layers
html_model_view = model_view(attentions, tokens)

with open('bertviz_model_view.html', 'w') as f:
    f.write(html_model_view.data)

print("BertViz model view saved to bertviz_model_view.html")
```

**Custom attention visualization for specific tokens**:

```python
class AttentionVisualizer:
    """Custom attention visualization for transformers"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.model.eval()

    def get_attention_weights(self, text):
        """Get attention weights for text"""
        inputs = self.tokenizer(text, return_tensors='pt')

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attentions = outputs.attentions

        return tokens, attentions

    def visualize_token_attention(self, text, token_idx, layer_idx=-1, head_idx=0):
        """
        Visualize attention from a specific token.

        Args:
            text: Input text
            token_idx: Index of token to visualize (query)
            layer_idx: Layer index
            head_idx: Head index
        """
        tokens, attentions = self.get_attention_weights(text)

        # Get attention for specific head
        attention = attentions[layer_idx][0, head_idx].numpy()  # (seq_len, seq_len)

        # Get attention from token_idx to all other tokens
        token_attention = attention[token_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.bar(range(len(tokens)), token_attention)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_ylabel('Attention Weight')
        ax.set_title(f'Attention from "{tokens[token_idx]}" - Layer {layer_idx}, Head {head_idx}')
        plt.tight_layout()

        return fig

    def find_most_attended_tokens(self, text, target_token_idx, layer_idx=-1, top_n=5):
        """
        Find tokens that receive most attention from target token.

        Returns:
            List of (token, attention_weight) tuples
        """
        tokens, attentions = self.get_attention_weights(text)

        # Average attention across all heads in layer
        layer_attn = attentions[layer_idx][0].mean(dim=0).numpy()  # (seq_len, seq_len)

        # Get attention from target token
        target_attention = layer_attn[target_token_idx]

        # Get top N
        top_indices = target_attention.argsort()[-top_n:][::-1]

        results = [
            (tokens[i], float(target_attention[i]))
            for i in top_indices
        ]

        return results

    def visualize_all_heads(self, text, layer_idx=-1):
        """Visualize all attention heads in a layer"""
        tokens, attentions = self.get_attention_weights(text)

        num_heads = attentions[layer_idx].shape[1]
        attention = attentions[layer_idx][0].numpy()  # (num_heads, seq_len, seq_len)

        # Create grid of subplots
        n_rows = int(np.ceil(num_heads / 4))
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows))
        axes = axes.flatten()

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            im = ax.imshow(attention[head_idx], cmap='viridis', aspect='auto')
            ax.set_title(f'Head {head_idx}')
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            ax.set_xticklabels(tokens, rotation=90, fontsize=8)
            ax.set_yticklabels(tokens, fontsize=8)
            plt.colorbar(im, ax=ax)

        # Hide unused subplots
        for idx in range(num_heads, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        return fig

# Usage
attn_viz = AttentionVisualizer(model, tokenizer)

# Visualize attention from specific token
fig1 = attn_viz.visualize_token_attention(
    text="The model learns to attend to important words",
    token_idx=5,  # "attend"
    layer_idx=-1,
    head_idx=0
)
fig1.savefig('token_attention.png', dpi=300, bbox_inches='tight')

# Find most attended tokens
most_attended = attn_viz.find_most_attended_tokens(
    text="The model learns to attend to important words",
    target_token_idx=5,  # "attend"
    layer_idx=-1,
    top_n=5
)

print("Most attended tokens from 'attend':")
for token, weight in most_attended:
    print(f"  {token}: {weight:.4f}")

# Visualize all heads in last layer
fig2 = attn_viz.visualize_all_heads(text, layer_idx=-1)
fig2.savefig('all_heads_attention.png', dpi=300, bbox_inches='tight')
```

### 8.3 Limitations of Attention as Explanation

**Critical: Attention is NOT explanation!**

Research has shown that attention weights don't always correlate with feature importance. High attention doesn't necessarily mean high influence on prediction.

**Reasons**:
1. **Attention is one part of computation**: Final prediction depends on value vectors, not just attention weights
2. **Attention can be uniform**: Model may attend uniformly but still make strong predictions
3. **Counterfactual attention**: Changing attention doesn't always change prediction
4. **Multiple heads**: Different heads may capture different patterns; one head's attention may be misleading

**Proper use of attention visualization**:
- Use as exploratory tool, not definitive explanation
- Combine with gradient-based methods (Integrated Gradients)
- Validate with perturbation studies
- Use attention alongside SHAP/LIME

```python
def validate_attention_importance(model, tokenizer, text, layer_idx=-1):
    """
    Validate if high attention correlates with high importance.
    Use integrated gradients as ground truth.
    """
    # Get attention weights
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attentions = outputs.attentions

    # Average attention across heads
    avg_attention = attentions[layer_idx][0].mean(dim=0).numpy()
    # Average attention received by each token
    token_attention = avg_attention.mean(axis=0)

    # Get Integrated Gradients importance
    # (Simplified - actual IG for transformers is more complex)
    embeddings = model.get_input_embeddings()(inputs['input_ids'])
    embeddings.requires_grad = True

    # Forward
    outputs_ig = model(inputs_embeds=embeddings)

    # Backward (using first output logit as target)
    model.zero_grad()
    outputs_ig.last_hidden_state[:, 0, 0].backward()

    # Get gradients
    gradients = embeddings.grad.data.squeeze().norm(dim=-1).numpy()

    # Compare correlation
    correlation = np.corrcoef(token_attention, gradients)[0, 1]

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(range(len(tokens)), token_attention)
    axes[0].set_xticks(range(len(tokens)))
    axes[0].set_xticklabels(tokens, rotation=45, ha='right')
    axes[0].set_ylabel('Average Attention Weight')
    axes[0].set_title('Attention-Based Importance')

    axes[1].bar(range(len(tokens)), gradients)
    axes[1].set_xticks(range(len(tokens)))
    axes[1].set_xticklabels(tokens, rotation=45, ha='right')
    axes[1].set_ylabel('Gradient Norm')
    axes[1].set_title('Gradient-Based Importance')

    fig.suptitle(f'Attention vs Gradient Importance (correlation: {correlation:.3f})')
    plt.tight_layout()

    return fig, correlation

# Validate
fig_validation, corr = validate_attention_importance(
    model, tokenizer,
    "This is an important sentence for testing attention"
)
fig_validation.savefig('attention_validation.png', dpi=300, bbox_inches='tight')
print(f"Correlation between attention and gradients: {corr:.3f}")
```

---

## 9. Model-Specific Interpretability

### 9.1 Decision Tree Visualization

Decision trees are **inherently interpretable** - the structure itself is the explanation.

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn import tree
import graphviz

# Train decision tree
dt_model = DecisionTreeClassifier(
    max_depth=4,  # Limit depth for interpretability
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt_model.fit(X_train, y_train)

# Visualize tree with matplotlib
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    filled=True,  # Color by class
    rounded=True,
    fontsize=10
)
plt.title('Decision Tree Visualization')
plt.tight_layout()
plt.savefig('decision_tree_viz.png', dpi=300, bbox_inches='tight')

# Export as text rules
tree_rules = export_text(dt_model, feature_names=feature_names)
print("Decision Tree Rules:")
print(tree_rules)

# Export as Graphviz (DOT format)
dot_data = tree.export_graphviz(
    dt_model,
    out_file=None,
    feature_names=feature_names,
    class_names=['Class 0', 'Class 1'],
    filled=True,
    rounded=True
)

# Render with graphviz
graph = graphviz.Source(dot_data)
graph.render('decision_tree', format='png', cleanup=True)
print("Graphviz tree saved to decision_tree.png")
```

**Extract interpretable rules programmatically**:

```python
def extract_rules_from_tree(tree_model, feature_names, class_names=None):
    """
    Extract human-readable rules from decision tree.

    Returns:
        List of rule strings
    """
    tree_ = tree_model.tree_
    feature_name = [
        feature_names[i] if i != tree.TREE_UNDEFINED else "undefined"
        for i in tree_.feature
    ]

    if class_names is None:
        class_names = [f'Class {i}' for i in range(tree_.n_classes[0])]

    def recurse(node, depth, path_conditions):
        """Recursively traverse tree and build rules"""
        indent = "  " * depth

        if tree_.feature[node] != tree.TREE_UNDEFINED:
            # Internal node
            name = feature_name[node]
            threshold = tree_.threshold[node]

            # Left branch (<=)
            left_conditions = path_conditions + [f"{name} <= {threshold:.2f}"]
            recurse(tree_.children_left[node], depth + 1, left_conditions)

            # Right branch (>)
            right_conditions = path_conditions + [f"{name} > {threshold:.2f}"]
            recurse(tree_.children_right[node], depth + 1, right_conditions)
        else:
            # Leaf node
            class_idx = np.argmax(tree_.value[node])
            n_samples = tree_.n_node_samples[node]
            class_prob = tree_.value[node][0, class_idx] / n_samples

            rule = " AND ".join(path_conditions)
            prediction = class_names[class_idx]

            rules.append({
                'conditions': rule,
                'prediction': prediction,
                'probability': class_prob,
                'n_samples': n_samples
            })

    rules = []
    recurse(0, 0, [])
    return rules

# Extract rules
rules = extract_rules_from_tree(dt_model, feature_names, class_names=['Negative', 'Positive'])

print("\nExtracted Rules:")
for i, rule in enumerate(rules, 1):
    print(f"\nRule {i}:")
    print(f"  IF {rule['conditions']}")
    print(f"  THEN {rule['prediction']}")
    print(f"  (Probability: {rule['probability']:.2f}, Samples: {rule['n_samples']})")

# Save rules to JSON
import json
with open('decision_tree_rules.json', 'w') as f:
    json.dump(rules, f, indent=2)
```

**Decision path for specific instance**:

```python
def explain_tree_prediction(tree_model, instance, feature_names):
    """
    Explain prediction by showing decision path.

    Args:
        tree_model: Trained decision tree
        instance: Single instance (1D array)
        feature_names: List of feature names

    Returns:
        Explanation string
    """
    # Get decision path
    path = tree_model.decision_path(instance.reshape(1, -1))
    leaf_id = tree_model.apply(instance.reshape(1, -1))[0]

    # Get tree structure
    tree_ = tree_model.tree_

    # Extract path
    node_indicator = path.toarray()[0]
    node_indices = np.where(node_indicator)[0]

    explanation = "Decision Path:\n"
    explanation += "=" * 50 + "\n"

    for i, node_id in enumerate(node_indices):
        # If not leaf
        if node_id != leaf_id:
            feature_id = tree_.feature[node_id]
            threshold = tree_.threshold[node_id]
            feature_value = instance[feature_id]
            feature = feature_names[feature_id]

            if feature_value <= threshold:
                direction = "<="
            else:
                direction = ">"

            explanation += f"{i+1}. {feature} = {feature_value:.2f} {direction} {threshold:.2f}\n"

    # Leaf node prediction
    class_probs = tree_.value[leaf_id][0]
    class_probs /= class_probs.sum()
    predicted_class = np.argmax(class_probs)

    explanation += "=" * 50 + "\n"
    explanation += f"Prediction: Class {predicted_class}\n"
    explanation += f"Probabilities: {class_probs}\n"
    explanation += f"Samples in leaf: {tree_.n_node_samples[leaf_id]}\n"

    return explanation

# Example
test_instance = X_test[0]
explanation = explain_tree_prediction(dt_model, test_instance, feature_names)
print(explanation)
```

### 9.2 Linear Model Coefficients

Linear models (Linear Regression, Logistic Regression, Linear SVM) have **coefficients as direct explanations**.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Standardize features for fair coefficient comparison
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Extract coefficients
coefficients = lr_model.coef_[0]
intercept = lr_model.intercept_[0]

# Create DataFrame
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coefficients
}).sort_values('coefficient', key=abs, ascending=False)

print("Logistic Regression Coefficients:")
print(coef_df)

# Visualize
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in coef_df['coefficient']]
plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.title('Logistic Regression Feature Coefficients\n(Positive = increases P(Class 1), Negative = decreases)')
plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
plt.tight_layout()
plt.savefig('logistic_regression_coefficients.png', dpi=300, bbox_inches='tight')

# Interpret coefficients
print("\nInterpretation:")
print(f"Intercept (baseline log-odds): {intercept:.4f}")
for _, row in coef_df.iterrows():
    feature = row['feature']
    coef = row['coefficient']
    odds_ratio = np.exp(coef)

    print(f"\n{feature}:")
    print(f"  Coefficient: {coef:.4f}")
    print(f"  Odds Ratio: {odds_ratio:.4f}")
    print(f"  Interpretation: 1 unit increase --> {(odds_ratio-1)*100:+.1f}% change in odds")
```

**Explain individual predictions with linear model**:

```python
def explain_linear_prediction(model, scaler, instance, feature_names):
    """
    Explain linear model prediction by showing feature contributions.

    Args:
        model: Trained linear model
        scaler: StandardScaler used for features
        instance: Single instance (original scale)
        feature_names: List of feature names

    Returns:
        Explanation dict
    """
    # Scale instance
    instance_scaled = scaler.transform(instance.reshape(1, -1))[0]

    # Get coefficients
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]

    # Compute contributions
    contributions = instance_scaled * coefficients

    # Prediction
    log_odds = intercept + contributions.sum()
    probability = 1 / (1 + np.exp(-log_odds))

    # Create explanation
    explanation = {
        'intercept': intercept,
        'log_odds': log_odds,
        'probability': probability,
        'predicted_class': int(probability >= 0.5),
        'feature_contributions': []
    }

    for i, (feature, coef, contrib, value) in enumerate(
        zip(feature_names, coefficients, contributions, instance_scaled)
    ):
        explanation['feature_contributions'].append({
            'feature': feature,
            'value_scaled': value,
            'coefficient': coef,
            'contribution': contrib
        })

    # Sort by absolute contribution
    explanation['feature_contributions'].sort(
        key=lambda x: abs(x['contribution']), reverse=True
    )

    return explanation

# Example
test_instance = X_test[0]
linear_explanation = explain_linear_prediction(
    lr_model, scaler, test_instance, feature_names
)

print("Linear Model Explanation:")
print(f"Intercept: {linear_explanation['intercept']:.4f}")
print(f"Log-odds: {linear_explanation['log_odds']:.4f}")
print(f"Probability: {linear_explanation['probability']:.4f}")
print(f"Prediction: Class {linear_explanation['predicted_class']}")

print("\nTop 5 Feature Contributions:")
for contrib in linear_explanation['feature_contributions'][:5]:
    print(f"{contrib['feature']}:")
    print(f"  Scaled value: {contrib['value_scaled']:.4f}")
    print(f"  Coefficient: {contrib['coefficient']:.4f}")
    print(f"  Contribution: {contrib['contribution']:.4f}")
```

### 9.3 Rule Extraction from Black-Box Models

**Extract interpretable rules from complex models** using surrogate decision trees.

```python
class RuleExtractor:
    """
    Extract interpretable rules from black-box models.
    Uses decision tree as global surrogate model.
    """

    def __init__(self, black_box_model, max_depth=5, min_samples_leaf=50):
        """
        Args:
            black_box_model: Complex model to approximate
            max_depth: Max depth of surrogate tree
            min_samples_leaf: Min samples per leaf
        """
        self.black_box_model = black_box_model
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.surrogate_tree = None
        self.fidelity_score = None

    def extract_rules(self, X, feature_names, class_names=None):
        """
        Extract rules by training surrogate tree.

        Args:
            X: Feature data
            feature_names: List of feature names
            class_names: Optional class names

        Returns:
            List of extracted rules
        """
        # Get black-box predictions
        y_pred = self.black_box_model.predict(X)

        # Train surrogate decision tree
        self.surrogate_tree = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.surrogate_tree.fit(X, y_pred)

        # Measure fidelity (agreement with black-box)
        surrogate_pred = self.surrogate_tree.predict(X)
        self.fidelity_score = (y_pred == surrogate_pred).mean()

        print(f"Surrogate model fidelity: {self.fidelity_score:.2%}")

        # Extract rules
        rules = extract_rules_from_tree(
            self.surrogate_tree,
            feature_names,
            class_names
        )

        return rules

    def visualize_surrogate(self, feature_names, class_names=None):
        """Visualize surrogate tree"""
        if self.surrogate_tree is None:
            raise ValueError("Must call extract_rules first")

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.surrogate_tree,
            feature_names=feature_names,
            class_names=class_names or ['0', '1'],
            filled=True,
            rounded=True,
            fontsize=10
        )
        plt.title(f'Surrogate Decision Tree (Fidelity: {self.fidelity_score:.2%})')
        plt.tight_layout()

        return plt.gcf()

# Example: Extract rules from XGBoost
rule_extractor = RuleExtractor(xgb_model, max_depth=4, min_samples_leaf=30)

extracted_rules = rule_extractor.extract_rules(
    X_train,
    feature_names,
    class_names=['Negative', 'Positive']
)

print(f"\nExtracted {len(extracted_rules)} rules from XGBoost")

print("\nTop 5 rules:")
for i, rule in enumerate(extracted_rules[:5], 1):
    print(f"\n{i}. IF {rule['conditions']}")
    print(f"   THEN {rule['prediction']} (prob={rule['probability']:.2f}, n={rule['n_samples']})")

# Visualize
fig = rule_extractor.visualize_surrogate(feature_names, ['Negative', 'Positive'])
fig.savefig('surrogate_tree_rules.png', dpi=300, bbox_inches='tight')
```

---

## 10. Global vs Local Explanations

### 10.1 Understanding the Distinction

**Local explanations**: Explain individual predictions
- LIME: Explains single instance
- SHAP for one sample: Individual feature contributions
- Counterfactuals: Changes needed for specific instance
- Use case: "Why was THIS loan denied?"

**Global explanations**: Explain overall model behavior
- Feature importance: Which features matter most overall?
- Partial dependence plots: How does feature affect predictions?
- Aggregated SHAP: Average impact across dataset
- Use case: "What factors drive loan approvals in general?"

### 10.2 Aggregating Local Explanations

**Convert local explanations to global insights** by aggregating across many instances.

```python
class GlobalLocalExplainer:
    """
    Generate both global and local explanations.
    Aggregate local explanations for global insights.
    """

    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names

        # Create SHAP explainer
        self.shap_explainer = shap.TreeExplainer(model)

    def global_feature_importance(self, X):
        """
        Global feature importance via mean |SHAP|.

        Args:
            X: Dataset to analyze

        Returns:
            DataFrame with global importance
        """
        # Compute SHAP values
        shap_values = self.shap_explainer.shap_values(X)

        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_abs_shap
        }).sort_values('importance', ascending=False)

        return importance_df

    def local_explanation(self, instance):
        """
        Local explanation for single instance.

        Args:
            instance: Single instance

        Returns:
            Explanation dict
        """
        shap_values = self.shap_explainer.shap_values(instance.reshape(1, -1))

        explanation = {
            'prediction': self.model.predict(instance.reshape(1, -1))[0],
            'base_value': self.shap_explainer.expected_value,
            'shap_values': dict(zip(self.feature_names, shap_values[0])),
            'feature_values': dict(zip(self.feature_names, instance))
        }

        # Sort by absolute SHAP value
        sorted_features = sorted(
            explanation['shap_values'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        explanation['top_features'] = sorted_features[:5]

        return explanation

    def partial_dependence(self, feature_idx, grid_resolution=100):
        """
        Partial dependence plot for a feature.

        Args:
            feature_idx: Index of feature
            grid_resolution: Number of grid points

        Returns:
            Grid values and average predictions
        """
        # Create grid of feature values
        feature_min = self.X_train[:, feature_idx].min()
        feature_max = self.X_train[:, feature_idx].max()
        grid = np.linspace(feature_min, feature_max, grid_resolution)

        # For each grid point, average predictions over dataset
        avg_predictions = []

        for value in grid:
            # Copy dataset and replace feature
            X_modified = self.X_train.copy()
            X_modified[:, feature_idx] = value

            # Predict
            preds = self.model.predict_proba(X_modified)[:, 1]
            avg_predictions.append(preds.mean())

        return grid, np.array(avg_predictions)

    def compare_global_local(self, instance, X_global):
        """
        Compare local explanation with global importance.

        Args:
            instance: Single instance to explain
            X_global: Dataset for global importance

        Returns:
            Comparison visualization
        """
        # Global importance
        global_imp = self.global_feature_importance(X_global)

        # Local explanation
        local_exp = self.local_explanation(instance)

        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Global importance (top 10)
        top_global = global_imp.head(10)
        axes[0].barh(top_global['feature'], top_global['importance'])
        axes[0].set_xlabel('Mean |SHAP Value|')
        axes[0].set_title('Global Feature Importance')
        axes[0].invert_yaxis()

        # Local importance (top 10)
        local_features = [f[0] for f in local_exp['top_features'][:10]]
        local_values = [abs(f[1]) for f in local_exp['top_features'][:10]]

        axes[1].barh(local_features, local_values)
        axes[1].set_xlabel('|SHAP Value|')
        axes[1].set_title('Local Feature Importance (This Instance)')
        axes[1].invert_yaxis()

        plt.tight_layout()
        return fig

# Usage
global_local_explainer = GlobalLocalExplainer(xgb_model, X_train, feature_names)

# Global importance
global_importance = global_local_explainer.global_feature_importance(X_test)
print("Global Feature Importance:")
print(global_importance.head(10))

# Local explanation
local_exp = global_local_explainer.local_explanation(X_test[0])
print(f"\nLocal Explanation for instance 0:")
print(f"Prediction: {local_exp['prediction']}")
print(f"Top features:")
for feature, shap_val in local_exp['top_features']:
    print(f"  {feature}: {shap_val:+.4f}")

# Comparison
fig = global_local_explainer.compare_global_local(X_test[0], X_test)
fig.savefig('global_vs_local_comparison.png', dpi=300, bbox_inches='tight')

# Partial dependence
feature_idx = 0
grid, pd_values = global_local_explainer.partial_dependence(feature_idx)

plt.figure(figsize=(10, 6))
plt.plot(grid, pd_values)
plt.xlabel(feature_names[feature_idx])
plt.ylabel('Average Predicted Probability')
plt.title(f'Partial Dependence Plot - {feature_names[feature_idx]}')
plt.grid(alpha=0.3)
plt.savefig('partial_dependence.png', dpi=300, bbox_inches='tight')
```

### 10.3 When to Use Global vs Local

**Use Local Explanations when**:
- Explaining specific decisions to individuals (loan denials, medical diagnoses)
- Debugging specific model failures
- Providing recourse for unfavorable decisions
- Regulatory compliance (right to explanation)

**Use Global Explanations when**:
- Understanding overall model behavior
- Model validation and debugging
- Feature selection and engineering
- Communicating model insights to stakeholders
- Detecting systemic bias

**Best Practice**: Use both! Global for understanding, local for decisions.

---

## 11. Explainability in Production

### 11.1 Logging Explanations

**Production explainability requires systematic logging** for audit trails and debugging.

```python
import logging
from datetime import datetime
import uuid

class ProductionExplainabilityLogger:
    """
    Log explanations for production ML systems.
    Supports audit trails, debugging, and compliance.
    """

    def __init__(self, model, explainer, log_dir='./explanation_logs'):
        """
        Args:
            model: Production model
            explainer: Explainer object (SHAP, LIME, etc.)
            log_dir: Directory for log files
        """
        self.model = model
        self.explainer = explainer
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Setup logging
        self.logger = logging.getLogger('ExplainabilityLogger')
        self.logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(self.log_dir / 'explanations.log')
        fh.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

        # Database for structured storage (using JSON files for simplicity)
        self.db_path = self.log_dir / 'explanations_db.jsonl'

    def log_prediction_with_explanation(self, instance, user_id=None,
                                       request_id=None, metadata=None):
        """
        Log prediction with full explanation.

        Args:
            instance: Input instance
            user_id: Optional user identifier
            request_id: Optional request ID
            metadata: Optional metadata dict

        Returns:
            Logged record with explanation
        """
        # Generate IDs
        if request_id is None:
            request_id = str(uuid.uuid4())

        timestamp = datetime.now().isoformat()

        # Make prediction
        prediction = self.model.predict(instance.reshape(1, -1))[0]
        prediction_proba = self.model.predict_proba(instance.reshape(1, -1))[0]

        # Generate explanation
        shap_values = self.explainer.shap_values(instance.reshape(1, -1))

        # Create record
        record = {
            'request_id': request_id,
            'timestamp': timestamp,
            'user_id': user_id,
            'prediction': int(prediction),
            'prediction_probability': prediction_proba.tolist(),
            'feature_values': instance.tolist(),
            'shap_values': shap_values[0].tolist() if isinstance(shap_values, np.ndarray) else shap_values,
            'base_value': float(self.explainer.expected_value),
            'metadata': metadata or {}
        }

        # Log to file
        self.logger.info(f"Prediction logged - ID: {request_id}, User: {user_id}, Prediction: {prediction}")

        # Save to database
        with open(self.db_path, 'a') as f:
            f.write(json.dumps(record) + '\n')

        return record

    def get_explanations_by_user(self, user_id):
        """Retrieve all explanations for a user (GDPR compliance)"""
        explanations = []

        if not self.db_path.exists():
            return explanations

        with open(self.db_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record.get('user_id') == user_id:
                    explanations.append(record)

        return explanations

    def get_explanation_by_request(self, request_id):
        """Retrieve specific explanation by request ID"""
        if not self.db_path.exists():
            return None

        with open(self.db_path, 'r') as f:
            for line in f:
                record = json.loads(line)
                if record['request_id'] == request_id:
                    return record

        return None

    def generate_compliance_report(self, user_id):
        """
        Generate GDPR Article 15 compliance report.

        Returns:
            Human-readable report of all decisions for user
        """
        explanations = self.get_explanations_by_user(user_id)

        report = f"""
GDPR ARTICLE 15 - RIGHT OF ACCESS REPORT
=========================================

User ID: {user_id}
Report Generated: {datetime.now().isoformat()}
Total Decisions: {len(explanations)}

DECISION HISTORY:
"""

        for i, exp in enumerate(explanations, 1):
            report += f"\n{i}. Decision ID: {exp['request_id']}\n"
            report += f"   Date: {exp['timestamp']}\n"
            report += f"   Prediction: {'Approved' if exp['prediction'] == 1 else 'Denied'}\n"
            report += f"   Confidence: {max(exp['prediction_probability']):.2%}\n"

            # Top 3 contributing features
            shap_vals = np.array(exp['shap_values'])
            top_indices = np.argsort(np.abs(shap_vals))[-3:][::-1]

            report += f"   Top Contributing Factors:\n"
            for idx in top_indices:
                report += f"     - Feature {idx}: SHAP value {shap_vals[idx]:+.4f}\n"

        report += "\n" + "=" * 50 + "\n"
        report += "You have the right to:\n"
        report += "1. Request human review of any decision\n"
        report += "2. Request correction of inaccurate data\n"
        report += "3. Request deletion of your data (right to be forgotten)\n"
        report += "\nContact: privacy@company.com\n"

        return report

    def monitor_explanation_drift(self, window_size=100):
        """
        Monitor drift in feature contributions over time.
        Alerts if explanation patterns change significantly.
        """
        if not self.db_path.exists():
            return None

        # Load recent explanations
        recent_explanations = []

        with open(self.db_path, 'r') as f:
            for line in f:
                recent_explanations.append(json.loads(line))

        if len(recent_explanations) < window_size:
            return None

        # Take last window_size explanations
        recent = recent_explanations[-window_size:]
        previous = recent_explanations[-2*window_size:-window_size]

        # Compute average SHAP values
        recent_shap = np.array([r['shap_values'] for r in recent]).mean(axis=0)
        previous_shap = np.array([r['shap_values'] for r in previous]).mean(axis=0)

        # Compute drift (L2 distance)
        drift = np.linalg.norm(recent_shap - previous_shap)

        drift_report = {
            'timestamp': datetime.now().isoformat(),
            'drift_score': float(drift),
            'recent_avg_shap': recent_shap.tolist(),
            'previous_avg_shap': previous_shap.tolist(),
            'alert': drift > 0.5  # Threshold for alert
        }

        if drift_report['alert']:
            self.logger.warning(f"Explanation drift detected! Score: {drift:.4f}")

        return drift_report

# Usage
prod_logger = ProductionExplainabilityLogger(
    model=xgb_model,
    explainer=shap.TreeExplainer(xgb_model),
    log_dir='./production_logs'
)

# Log predictions
for i in range(10):
    record = prod_logger.log_prediction_with_explanation(
        instance=X_test[i],
        user_id=f'user_{i % 3}',  # Simulate multiple users
        metadata={'source': 'api', 'version': '1.0'}
    )

print(f"Logged {i+1} predictions")

# Retrieve user explanations
user_explanations = prod_logger.get_explanations_by_user('user_0')
print(f"\nUser 'user_0' has {len(user_explanations)} decisions")

# Generate compliance report
report = prod_logger.generate_compliance_report('user_0')
print(report)

# Save report
with open('./production_logs/compliance_report_user_0.txt', 'w') as f:
    f.write(report)

# Monitor drift
drift_report = prod_logger.monitor_explanation_drift(window_size=5)
if drift_report:
    print(f"\nDrift monitoring: {drift_report}")
```

### 11.2 Real-Time Explanation APIs

**Serve explanations via REST API** for production systems.

```python
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Global model and explainer (loaded once at startup)
global_model = xgb_model
global_explainer = shap.TreeExplainer(global_model)
global_feature_names = feature_names

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint with explanation.

    Request:
    {
        "features": [val1, val2, ...],
        "user_id": "user_123",
        "include_explanation": true
    }

    Response:
    {
        "prediction": 0 or 1,
        "probability": [p0, p1],
        "explanation": {...}
    }
    """
    try:
        data = request.get_json()

        # Validate input
        if 'features' not in data:
            return jsonify({'error': 'Missing features'}), 400

        features = np.array(data['features']).reshape(1, -1)
        user_id = data.get('user_id')
        include_explanation = data.get('include_explanation', True)

        # Prediction
        prediction = int(global_model.predict(features)[0])
        probability = global_model.predict_proba(features)[0].tolist()

        response = {
            'prediction': prediction,
            'probability': probability
        }

        # Add explanation if requested
        if include_explanation:
            shap_values = global_explainer.shap_values(features)[0]

            # Top 5 features by absolute SHAP value
            top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]

            explanation = {
                'base_value': float(global_explainer.expected_value),
                'top_features': [
                    {
                        'feature': global_feature_names[i],
                        'value': float(features[0, i]),
                        'shap_value': float(shap_values[i]),
                        'contribution': 'positive' if shap_values[i] > 0 else 'negative'
                    }
                    for i in top_indices
                ]
            }

            response['explanation'] = explanation

        # Log (in production, use proper logging)
        # prod_logger.log_prediction_with_explanation(features[0], user_id)

        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/explain/<request_id>', methods=['GET'])
def get_explanation(request_id):
    """
    Retrieve explanation for previous prediction.

    Response:
    {
        "request_id": "...",
        "timestamp": "...",
        "prediction": ...,
        "explanation": {...}
    }
    """
    # In production, retrieve from database
    # explanation = prod_logger.get_explanation_by_request(request_id)

    # Placeholder
    return jsonify({'message': 'Explanation retrieval endpoint'}), 200

@app.route('/user/<user_id>/explanations', methods=['GET'])
def get_user_explanations(user_id):
    """
    GDPR Article 15: Get all explanations for a user.
    """
    # In production, retrieve from database
    # explanations = prod_logger.get_explanations_by_user(user_id)

    # Placeholder
    return jsonify({'message': f'Explanations for user {user_id}'}), 200

# Run API (development mode)
# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

print("Flask API endpoints defined:")
print("  POST /predict - Get prediction with explanation")
print("  GET /explain/<request_id> - Retrieve stored explanation")
print("  GET /user/<user_id>/explanations - GDPR compliance endpoint")
```

### 11.3 Monitoring Feature Contributions

**Monitor feature contributions over time** to detect model degradation and data drift.

```python
class FeatureContributionMonitor:
    """
    Monitor feature contributions in production.
    Detect shifts in feature importance and data distribution.
    """

    def __init__(self, model, explainer, feature_names, baseline_data):
        """
        Args:
            model: Production model
            explainer: SHAP explainer
            feature_names: List of feature names
            baseline_data: Baseline dataset for comparison
        """
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names

        # Compute baseline SHAP values
        self.baseline_shap = explainer.shap_values(baseline_data)
        self.baseline_mean_abs_shap = np.abs(self.baseline_shap).mean(axis=0)

        # Storage for production SHAP values
        self.production_shap_values = []

    def log_production_prediction(self, instance):
        """Log SHAP values for production prediction"""
        shap_val = self.explainer.shap_values(instance.reshape(1, -1))[0]
        self.production_shap_values.append({
            'timestamp': datetime.now(),
            'shap_values': shap_val
        })

    def compute_feature_importance_drift(self):
        """
        Compute drift in feature importance between baseline and production.

        Returns:
            Drift metrics
        """
        if len(self.production_shap_values) == 0:
            return None

        # Get production SHAP values
        prod_shap = np.array([p['shap_values'] for p in self.production_shap_values])
        prod_mean_abs_shap = np.abs(prod_shap).mean(axis=0)

        # Compute drift metrics
        # 1. L2 distance
        l2_drift = np.linalg.norm(prod_mean_abs_shap - self.baseline_mean_abs_shap)

        # 2. Rank correlation (Spearman)
        from scipy.stats import spearmanr
        rank_corr, _ = spearmanr(prod_mean_abs_shap, self.baseline_mean_abs_shap)

        # 3. Per-feature drift
        feature_drift = []
        for i, feature in enumerate(self.feature_names):
            drift = {
                'feature': feature,
                'baseline_importance': float(self.baseline_mean_abs_shap[i]),
                'production_importance': float(prod_mean_abs_shap[i]),
                'absolute_drift': float(abs(prod_mean_abs_shap[i] - self.baseline_mean_abs_shap[i])),
                'relative_drift': float((prod_mean_abs_shap[i] - self.baseline_mean_abs_shap[i]) /
                                       (self.baseline_mean_abs_shap[i] + 1e-10))
            }
            feature_drift.append(drift)

        # Sort by absolute drift
        feature_drift.sort(key=lambda x: x['absolute_drift'], reverse=True)

        return {
            'l2_drift': float(l2_drift),
            'rank_correlation': float(rank_corr),
            'n_production_samples': len(self.production_shap_values),
            'top_drifting_features': feature_drift[:10],
            'timestamp': datetime.now().isoformat()
        }

    def visualize_drift(self):
        """Visualize feature importance drift"""
        drift_metrics = self.compute_feature_importance_drift()

        if drift_metrics is None:
            print("No production data logged yet")
            return

        # Extract data
        features = [f['feature'] for f in drift_metrics['top_drifting_features']]
        baseline_imp = [f['baseline_importance'] for f in drift_metrics['top_drifting_features']]
        prod_imp = [f['production_importance'] for f in drift_metrics['top_drifting_features']]

        # Plot
        x = np.arange(len(features))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, baseline_imp, width, label='Baseline', alpha=0.8)
        ax.bar(x + width/2, prod_imp, width, label='Production', alpha=0.8)

        ax.set_ylabel('Mean |SHAP Value|')
        ax.set_title(f'Feature Importance Drift\n(L2={drift_metrics["l2_drift"]:.4f}, Correlation={drift_metrics["rank_correlation"]:.4f})')
        ax.set_xticks(x)
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.legend()

        plt.tight_layout()
        return fig

# Usage
monitor = FeatureContributionMonitor(
    model=xgb_model,
    explainer=global_explainer,
    feature_names=feature_names,
    baseline_data=X_train[:1000]
)

# Simulate production predictions
print("Logging production predictions...")
for i in range(50):
    monitor.log_production_prediction(X_test[i])

# Compute drift
drift_metrics = monitor.compute_feature_importance_drift()

print("\nFeature Importance Drift Metrics:")
print(f"L2 Drift: {drift_metrics['l2_drift']:.4f}")
print(f"Rank Correlation: {drift_metrics['rank_correlation']:.4f}")
print(f"\nTop 5 Drifting Features:")
for feat in drift_metrics['top_drifting_features'][:5]:
    print(f"{feat['feature']}:")
    print(f"  Baseline: {feat['baseline_importance']:.4f}")
    print(f"  Production: {feat['production_importance']:.4f}")
    print(f"  Drift: {feat['relative_drift']:+.2%}")

# Visualize
fig = monitor.visualize_drift()
fig.savefig('feature_importance_drift.png', dpi=300, bbox_inches='tight')
```

---

## 12. Resources and References

### 12.1 Key Papers

**SHAP**:
- Lundberg, S. M., & Lee, S. I. (2017). "A Unified Approach to Interpreting Model Predictions." NeurIPS.
- Lundberg, S. M., et al. (2020). "From local explanations to global understanding with explainable AI for trees." Nature Machine Intelligence.

**LIME**:
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?: Explaining the Predictions of Any Classifier." KDD.

**Integrated Gradients**:
- Sundararajan, M., Taly, A., & Yan, Q. (2017). "Axiomatic Attribution for Deep Networks." ICML.

**Grad-CAM**:
- Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV.

**Attention as Explanation**:
- Jain, S., & Wallace, B. C. (2019). "Attention is not Explanation." NAACL.
- Wiegreffe, S., & Pinter, Y. (2019). "Attention is not not Explanation." EMNLP.

**Counterfactual Explanations**:
- Mothilal, R. K., et al. (2020). "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations." FAT*.

### 12.2 Libraries and Tools

**Python Libraries**:
```python
# Core explainability libraries
pip install shap lime dice-ml

# Deep learning interpretability
pip install captum  # PyTorch interpretability
pip install tf-explain  # TensorFlow interpretability

# Visualization
pip install bertviz  # Transformer attention
pip install alibi  # Model-agnostic explanations
pip install interpret  # Microsoft InterpretML

# Fairness and bias
pip install fairlearn aif360
```

**Production Tools**:
- **Seldon Alibi**: Production explainability for deployed models
- **AWS SageMaker Clarify**: Explainability and bias detection
- **Google Explainable AI**: Integrated with Vertex AI
- **Azure InterpretML**: Model interpretability toolkit

### 12.3 Best Practices Summary

**1. Choose the Right Method**:
- Tree models --> TreeSHAP (fast and exact)
- Any model --> KernelSHAP or LIME (slower but general)
- Deep learning --> Integrated Gradients, Grad-CAM
- NLP --> Attention + Integrated Gradients (attention alone is insufficient)

**2. Validate Explanations**:
- Use multiple methods and compare
- Test on known synthetic data
- Perform sanity checks (e.g., random model should have uniform importance)

**3. Consider Your Audience**:
- Technical users: SHAP values, coefficients, technical metrics
- Non-technical users: Natural language, visualizations, simple rules
- Regulators: Audit trails, compliance reports, human review options

**4. Production Deployment**:
- Cache explainers to avoid recomputation
- Log all explanations for audit trails
- Monitor explanation drift
- Provide API endpoints for explanation retrieval
- Implement GDPR Article 15 compliance

**5. Avoid Common Pitfalls**:
- Don't rely solely on attention for transformers
- Don't use impurity-based importance with high-cardinality features
- Don't ignore computational costs in production
- Don't confuse correlation with causation
- Don't skip validation of explanation methods

### 12.4 Regulatory Compliance Checklist

**GDPR Article 15 - Right of Access**:
- [ ] Store all predictions with explanations
- [ ] Enable retrieval of user's decision history
- [ ] Provide human-readable explanations
- [ ] Include contact information for human review

**EU AI Act Article 13 - Transparency**:
- [ ] Log model version and parameters
- [ ] Document explanation methodology
- [ ] Provide confidence scores with predictions
- [ ] Enable human oversight capabilities

**FCRA/ECOA (Credit Decisions)**:
- [ ] Generate adverse action notices
- [ ] List top 4 factors for denials
- [ ] Ensure explanations are specific (not generic)
- [ ] Provide recourse information

### 12.5 Code Repository Template

Complete template for production explainability system:

```python
# explainability_system.py

import numpy as np
import pandas as pd
import shap
from pathlib import Path
import json
import logging
from datetime import datetime

class ProductionExplainabilitySystem:
    """
    Complete production explainability system.

    Features:
    - Multiple explanation methods (SHAP, LIME, feature importance)
    - Logging and audit trails
    - GDPR compliance
    - Drift monitoring
    - API-ready outputs
    """

    def __init__(self, model, model_type='tree', feature_names=None,
                 background_data=None, log_dir='./explainability_logs'):
        """
        Initialize explainability system.

        Args:
            model: Trained model
            model_type: 'tree', 'linear', 'neural', or 'general'
            feature_names: List of feature names
            background_data: Background dataset for SHAP
            log_dir: Directory for logs
        """
        self.model = model
        self.model_type = model_type
        self.feature_names = feature_names or [f'feature_{i}' for i in range(model.n_features_in_)]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # Initialize explainer based on model type
        if model_type == 'tree':
            self.explainer = shap.TreeExplainer(model)
        elif model_type == 'general' and background_data is not None:
            self.explainer = shap.KernelExplainer(model.predict_proba, background_data)
        else:
            self.explainer = shap.Explainer(model)

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging infrastructure"""
        self.logger = logging.getLogger('ExplainabilitySystem')
        self.logger.setLevel(logging.INFO)

        fh = logging.FileHandler(self.log_dir / 'system.log')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def explain(self, instance, user_id=None, return_format='json'):
        """
        Generate explanation for instance.

        Args:
            instance: Input instance
            user_id: Optional user ID for logging
            return_format: 'json', 'dict', or 'text'

        Returns:
            Explanation in requested format
        """
        # Ensure 2D shape
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Prediction
        prediction = self.model.predict(instance)[0]
        prediction_proba = self.model.predict_proba(instance)[0] if hasattr(self.model, 'predict_proba') else None

        # SHAP values
        shap_values = self.explainer.shap_values(instance)
        if isinstance(shap_values, list):  # Multi-class
            shap_values = shap_values[1]  # Positive class

        # Build explanation
        explanation = {
            'prediction': int(prediction),
            'probability': prediction_proba.tolist() if prediction_proba is not None else None,
            'base_value': float(self.explainer.expected_value),
            'features': []
        }

        # Sort features by SHAP value magnitude
        shap_array = shap_values[0] if shap_values.ndim > 1 else shap_values
        sorted_indices = np.argsort(np.abs(shap_array))[::-1]

        for idx in sorted_indices[:10]:  # Top 10
            explanation['features'].append({
                'name': self.feature_names[idx],
                'value': float(instance[0, idx]),
                'shap_value': float(shap_array[idx]),
                'contribution': 'positive' if shap_array[idx] > 0 else 'negative'
            })

        # Log
        self._log_explanation(explanation, user_id)

        # Return in requested format
        if return_format == 'json':
            return json.dumps(explanation, indent=2)
        elif return_format == 'text':
            return self._format_text_explanation(explanation)
        else:
            return explanation

    def _format_text_explanation(self, explanation):
        """Format explanation as human-readable text"""
        text = f"Prediction: {'Approved' if explanation['prediction'] == 1 else 'Denied'}\n"
        if explanation['probability']:
            text += f"Confidence: {max(explanation['probability']):.1%}\n\n"

        text += "Top Contributing Factors:\n"
        for i, feat in enumerate(explanation['features'][:5], 1):
            text += f"{i}. {feat['name']}: {feat['value']:.2f} "
            text += f"({'increases' if feat['contribution'] == 'positive' else 'decreases'} likelihood)\n"

        return text

    def _log_explanation(self, explanation, user_id):
        """Log explanation to database"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'explanation': explanation
        }

        with open(self.log_dir / 'explanations.jsonl', 'a') as f:
            f.write(json.dumps(record) + '\n')

        self.logger.info(f"Explanation logged for user {user_id}")

    def generate_compliance_report(self, user_id):
        """Generate GDPR Article 15 compliance report"""
        # Implementation similar to previous examples
        pass

    def monitor_drift(self, window_size=100):
        """Monitor explanation drift over time"""
        # Implementation similar to previous examples
        pass

# Usage example
if __name__ == "__main__":
    # Initialize system
    system = ProductionExplainabilitySystem(
        model=xgb_model,
        model_type='tree',
        feature_names=feature_names,
        log_dir='./explainability_production'
    )

    # Explain instance
    explanation = system.explain(X_test[0], user_id='user_123', return_format='text')
    print(explanation)

    print("\nProduction explainability system ready!")
    print("Features: SHAP explanations, logging, compliance, monitoring")
```

### 12.6 Additional Resources

**Online Courses**:
- Coursera: "Interpretable Machine Learning" by Christoph Molnar
- fast.ai: "Practical Deep Learning - Interpretability"

**Books**:
- "Interpretable Machine Learning" by Christoph Molnar (free online)
- "Explainable AI" by Leilani H. Gilpin

**Community**:
- SHAP GitHub: https://github.com/slundberg/shap
- LIME GitHub: https://github.com/marcotcr/lime
- Explainable AI Blog: https://christophm.github.io/interpretable-ml-book/

**Conferences**:
- FAT* (Fairness, Accountability, Transparency)
- ICML Workshop on Interpretable ML
- NeurIPS Workshop on Human-Centric AI

---

**End of Model Explainability Guide**

This comprehensive guide covers explainability from theory to production deployment. Key takeaways:

1. **Regulatory compliance is mandatory** - GDPR, EU AI Act, FCRA require explanations
2. **Use multiple methods** - SHAP for accuracy, LIME for flexibility, Grad-CAM for vision
3. **Production requires infrastructure** - logging, monitoring, APIs, audit trails
4. **Validate explanations** - attention is not explanation, test methods rigorously
5. **Consider your audience** - technical vs non-technical, global vs local

**Production Checklist**:
- [ ] Explainer integrated with model pipeline
- [ ] Logging and audit trails implemented
- [ ] GDPR Article 15 compliance endpoint
- [ ] Drift monitoring active
- [ ] Human review process established
- [ ] Documentation and user guides created

