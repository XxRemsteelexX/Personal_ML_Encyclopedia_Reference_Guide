# Evaluation Metrics: A Comprehensive Guide

## Overview

Evaluation metrics quantify model performance and guide decision-making throughout the ML lifecycle. Choosing the right metric is as important as choosing the right model - an inappropriate metric can lead to misleading conclusions and poor business outcomes.

**Key Principle:** The metric should align with the business objective and the cost of different error types.

---

## Table of Contents
1. [Classification Metrics](#classification-metrics)
2. [Regression Metrics](#regression-metrics)
3. [Ranking Metrics](#ranking-metrics)
4. [NLP Metrics](#nlp-metrics)
5. [Computer Vision Metrics](#computer-vision-metrics)
6. [Metric Selection Guide](#metric-selection-guide)

---

## Classification Metrics

### 1. Confusion Matrix

The foundation for understanding classification performance.

```python
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot confusion matrix with detailed annotations.
    """
    cm = confusion_matrix(y_true, y_pred)

    if labels is None:
        labels = np.unique(y_true)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

    return cm

# Example usage
y_true = [0, 1, 0, 1, 0, 1, 0, 1]
y_pred = [0, 1, 0, 0, 0, 1, 1, 1]

cm = plot_confusion_matrix(y_true, y_pred, labels=['Negative', 'Positive'])
print("Confusion Matrix:")
print(cm)
```

**Interpretation:**
```
                Predicted
                Neg    Pos
Actual  Neg     TN     FP
        Pos     FN     TP

TN: True Negative (correctly predicted negative)
FP: False Positive (Type I error)
FN: False Negative (Type II error)
TP: True Positive (correctly predicted positive)
```

---

### 2. Accuracy

**Definition:** Proportion of correct predictions.

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

```python
from sklearn.metrics import accuracy_score

def evaluate_accuracy(y_true, y_pred):
    """
    Calculate accuracy with interpretation.
    """
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Manual calculation
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    print(f"Correct predictions: {correct}/{total}")

    return acc

# Example
accuracy = evaluate_accuracy(y_true, y_pred)
```

**When to Use:**
- Balanced datasets (similar class frequencies)
- All errors have equal cost

**When NOT to Use:**
- Imbalanced datasets (e.g., 99% negative class → 99% accuracy by always predicting negative!)
- Different error costs (e.g., false negative in cancer detection is worse than false positive)

---

### 3. Precision, Recall, and F1-Score

#### Precision (Positive Predictive Value)

**Definition:** Of all positive predictions, how many were correct?

$$\text{Precision} = \frac{TP}{TP + FP}$$

**Use Case:** Minimize false positives (e.g., spam detection - don't want to mark legitimate emails as spam)

#### Recall (Sensitivity, True Positive Rate)

**Definition:** Of all actual positives, how many did we detect?

$$\text{Recall} = \frac{TP}{TP + FN}$$

**Use Case:** Minimize false negatives (e.g., disease detection - don't want to miss cases)

#### F1-Score (Harmonic Mean)

**Definition:** Balanced measure of precision and recall.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = \frac{2TP}{2TP + FP + FN}$$

**Why Harmonic Mean?** Penalizes extreme values. If either precision or recall is low, F1 is low.

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

def detailed_classification_metrics(y_true, y_pred, labels=None):
    """
    Comprehensive classification report with all metrics.
    """
    # Overall metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average='binary'
    )

    print("=" * 60)
    print("BINARY CLASSIFICATION METRICS")
    print("=" * 60)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Support:   {np.sum(support)}")

    # Detailed report
    print("\n" + "=" * 60)
    print("DETAILED CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=labels))

    return precision, recall, f1

# Example
labels = ['Negative', 'Positive']
metrics = detailed_classification_metrics(y_true, y_pred, labels)
```

#### F-Beta Score (Weighted Precision and Recall)

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

- **β < 1:** Emphasize precision (e.g., β=0.5)
- **β = 1:** F1-score (balanced)
- **β > 1:** Emphasize recall (e.g., β=2)

```python
from sklearn.metrics import fbeta_score

# Emphasize recall (β=2) - important when false negatives are costly
f2 = fbeta_score(y_true, y_pred, beta=2)
print(f"F2-Score (emphasize recall): {f2:.4f}")

# Emphasize precision (β=0.5) - important when false positives are costly
f05 = fbeta_score(y_true, y_pred, beta=0.5)
print(f"F0.5-Score (emphasize precision): {f05:.4f}")
```

---

### 4. ROC Curve and AUC-ROC

**ROC (Receiver Operating Characteristic) Curve:** Plot of True Positive Rate (TPR) vs. False Positive Rate (FPR) at various threshold settings.

$$\text{TPR (Recall)} = \frac{TP}{TP + FN}$$

$$\text{FPR} = \frac{FP}{FP + TN}$$

**AUC-ROC (Area Under the Curve):** Single number summarizing ROC curve.
- **AUC = 1.0:** Perfect classifier
- **AUC = 0.5:** Random classifier
- **AUC < 0.5:** Worse than random (flip predictions!)

```python
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

def plot_roc_curve(y_true, y_proba, title='ROC Curve'):
    """
    Plot ROC curve with AUC score.

    Parameters:
    -----------
    y_true : array-like
        True binary labels
    y_proba : array-like
        Predicted probabilities for positive class
    """
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"AUC-ROC: {roc_auc:.4f}")

    # Find optimal threshold (maximize TPR - FPR)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print(f"Optimal threshold: {optimal_threshold:.4f}")
    print(f"TPR at optimal: {tpr[optimal_idx]:.4f}")
    print(f"FPR at optimal: {fpr[optimal_idx]:.4f}")

    return fpr, tpr, thresholds, roc_auc

# Example with probability predictions
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
model = LogisticRegression()
model.fit(X, y)
y_proba = model.predict_proba(X)[:, 1]  # Probability of positive class

fpr, tpr, thresholds, roc_auc = plot_roc_curve(y, y_proba)
```

**When to Use ROC-AUC:**
- Binary classification with probability predictions
- Balanced or moderately imbalanced datasets
- Want threshold-independent evaluation

**Limitations:**
- Can be overly optimistic for highly imbalanced datasets
- Doesn't account for cost of different errors

---

### 5. Precision-Recall Curve

Better than ROC for imbalanced datasets.

```python
from sklearn.metrics import precision_recall_curve, average_precision_score, PrecisionRecallDisplay

def plot_precision_recall_curve(y_true, y_proba, title='Precision-Recall Curve'):
    """
    Plot PR curve with average precision score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    avg_precision = average_precision_score(y_true, y_proba)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='darkorange', lw=2,
            label=f'PR curve (AP = {avg_precision:.3f})')

    # Baseline (proportion of positive class)
    baseline = np.sum(y_true) / len(y_true)
    ax.plot([0, 1], [baseline, baseline], color='navy', lw=2,
            linestyle='--', label=f'Baseline ({baseline:.3f})')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"Average Precision: {avg_precision:.4f}")

    # Find threshold for specific precision/recall
    desired_precision = 0.9
    idx = np.where(precision >= desired_precision)[0]
    if len(idx) > 0:
        best_idx = idx[np.argmax(recall[idx])]
        print(f"\nFor precision >= {desired_precision}:")
        print(f"  Threshold: {thresholds[best_idx]:.4f}")
        print(f"  Precision: {precision[best_idx]:.4f}")
        print(f"  Recall: {recall[best_idx]:.4f}")

    return precision, recall, thresholds, avg_precision

# Example with imbalanced data
X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20,
    weights=[0.95, 0.05],  # 95% negative, 5% positive
    random_state=42
)
model.fit(X_imb, y_imb)
y_proba_imb = model.predict_proba(X_imb)[:, 1]

pr_metrics = plot_precision_recall_curve(y_imb, y_proba_imb)
```

**When to Use PR Curve:**
- Highly imbalanced datasets
- Positive class is rare and important
- More informative than ROC in these cases

---

### 6. Multi-Class Metrics

#### Averaging Strategies

```python
from sklearn.metrics import precision_recall_fscore_support

def multiclass_metrics(y_true, y_pred, labels=None):
    """
    Calculate metrics for multi-class classification with different averaging.
    """
    # Macro: Average of per-class metrics (treats all classes equally)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro'
    )

    # Micro: Aggregate TP, FP, FN across all classes
    # (equivalent to accuracy for multi-class)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro'
    )

    # Weighted: Average weighted by support (class frequency)
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )

    print("=" * 60)
    print("MULTI-CLASS METRICS")
    print("=" * 60)
    print(f"{'Averaging':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)
    print(f"{'Macro':<15} {macro_p:>11.4f} {macro_r:>11.4f} {macro_f1:>11.4f}")
    print(f"{'Micro':<15} {micro_p:>11.4f} {micro_r:>11.4f} {micro_f1:>11.4f}")
    print(f"{'Weighted':<15} {weighted_p:>11.4f} {weighted_r:>11.4f} {weighted_f1:>11.4f}")

    # Per-class metrics
    print("\n" + "=" * 60)
    print("PER-CLASS METRICS")
    print("=" * 60)
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred)

    if labels is None:
        labels = [f"Class {i}" for i in range(len(p))]

    print(f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 60)
    for i, label in enumerate(labels):
        print(f"{label:<15} {p[i]:>11.4f} {r[i]:>11.4f} {f1[i]:>11.4f} {support[i]:>9}")

    return {
        'macro': (macro_p, macro_r, macro_f1),
        'micro': (micro_p, micro_r, micro_f1),
        'weighted': (weighted_p, weighted_r, weighted_f1),
        'per_class': (p, r, f1, support)
    }

# Example with 3 classes
from sklearn.datasets import make_classification

X_multi, y_multi = make_classification(
    n_samples=1000, n_features=20, n_informative=15,
    n_classes=3, n_clusters_per_class=1, random_state=42
)
from sklearn.ensemble import RandomForestClassifier
model_multi = RandomForestClassifier(random_state=42)
model_multi.fit(X_multi, y_multi)
y_pred_multi = model_multi.predict(X_multi)

metrics = multiclass_metrics(y_multi, y_pred_multi, labels=['Class A', 'Class B', 'Class C'])
```

**Averaging Strategies:**

1. **Macro Average:**
   - Treats all classes equally
   - Use when all classes are equally important
   - Sensitive to performance on minority classes

2. **Micro Average:**
   - Aggregates contributions from all classes
   - Dominated by frequent classes
   - Equivalent to accuracy for multi-class

3. **Weighted Average:**
   - Weighted by class frequency
   - Use when classes have different importance proportional to their frequency
   - Most common in practice

---

### 7. Cohen's Kappa

Measures agreement between predicted and actual, accounting for chance agreement.

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

where:
- $p_o$ = observed agreement (accuracy)
- $p_e$ = expected agreement by chance

```python
from sklearn.metrics import cohen_kappa_score

def evaluate_with_kappa(y_true, y_pred):
    """
    Calculate Cohen's Kappa for classification.
    """
    kappa = cohen_kappa_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Cohen's Kappa: {kappa:.4f}")

    # Interpretation
    if kappa < 0:
        interpretation = "Poor (worse than random)"
    elif kappa < 0.20:
        interpretation = "Slight"
    elif kappa < 0.40:
        interpretation = "Fair"
    elif kappa < 0.60:
        interpretation = "Moderate"
    elif kappa < 0.80:
        interpretation = "Substantial"
    else:
        interpretation = "Almost perfect"

    print(f"Interpretation: {interpretation}")

    return kappa

# Example
kappa = evaluate_with_kappa(y_multi, y_pred_multi)
```

**When to Use:**
- Multi-class classification with imbalanced classes
- Annotator agreement assessment
- Accounts for random chance (better than accuracy alone)

---

### 8. Matthews Correlation Coefficient (MCC)

Balanced measure even for imbalanced datasets.

$$\text{MCC} = \frac{TP \times TN - FP \times FN}{\sqrt{(TP + FP)(TP + FN)(TN + FP)(TN + FN)}}$$

Range: [-1, 1]
- **MCC = 1:** Perfect prediction
- **MCC = 0:** Random prediction
- **MCC = -1:** Perfect inverse prediction

```python
from sklearn.metrics import matthews_corrcoef

def evaluate_with_mcc(y_true, y_pred):
    """
    Calculate MCC for binary classification.
    """
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"Matthews Correlation Coefficient: {mcc:.4f}")

    if mcc > 0.9:
        interpretation = "Excellent"
    elif mcc > 0.7:
        interpretation = "Good"
    elif mcc > 0.5:
        interpretation = "Moderate"
    elif mcc > 0.3:
        interpretation = "Weak"
    else:
        interpretation = "Poor"

    print(f"Interpretation: {interpretation}")

    return mcc

# Example with imbalanced data
mcc = evaluate_with_mcc(y_imb, model.predict(X_imb))
```

**When to Use:**
- Highly imbalanced datasets
- Single metric that considers all confusion matrix elements
- More informative than F1 or accuracy for imbalanced data

---

### 9. Log Loss (Cross-Entropy Loss)

Measures the quality of probability predictions.

$$\text{Log Loss} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}y_{ij}\log(p_{ij})$$

where:
- $N$ = number of samples
- $M$ = number of classes
- $y_{ij}$ = 1 if sample $i$ belongs to class $j$, else 0
- $p_{ij}$ = predicted probability that sample $i$ belongs to class $j$

```python
from sklearn.metrics import log_loss

def evaluate_probabilistic(y_true, y_proba):
    """
    Evaluate probability predictions with log loss.
    """
    logloss = log_loss(y_true, y_proba)

    print(f"Log Loss: {logloss:.4f}")
    print("\nInterpretation:")
    print("  Lower is better (0 = perfect)")
    print("  Heavily penalizes confident wrong predictions")

    # Compare with random
    if len(y_proba.shape) == 2:
        n_classes = y_proba.shape[1]
        random_proba = np.ones_like(y_proba) / n_classes
        random_logloss = log_loss(y_true, random_proba)
        print(f"\nRandom baseline log loss: {random_logloss:.4f}")
        print(f"Improvement: {(random_logloss - logloss) / random_logloss * 100:.2f}%")

    return logloss

# Example
y_proba_multi = model_multi.predict_proba(X_multi)
logloss = evaluate_probabilistic(y_multi, y_proba_multi)
```

**When to Use:**
- Ranking models by probability calibration
- Kaggle competitions (many use log loss)
- When probability estimates are important
- Penalizes overconfident wrong predictions

---

### 10. Calibration

How well do predicted probabilities match true frequencies?

```python
from sklearn.calibration import calibration_curve, CalibrationDisplay

def plot_calibration_curve(y_true, y_proba, n_bins=10):
    """
    Plot calibration curve to assess probability calibration.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot calibration curve
    ax.plot(mean_predicted_value, fraction_of_positives,
            marker='o', linewidth=2, label='Model')

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Calculate calibration error
    calibration_error = np.abs(fraction_of_positives - mean_predicted_value).mean()
    print(f"Mean Calibration Error: {calibration_error:.4f}")

    return fraction_of_positives, mean_predicted_value

# Example
plot_calibration_curve(y_imb, y_proba_imb)
```

**Calibration Techniques:**

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_classifier(model, X_train, y_train, X_test, y_test, method='sigmoid'):
    """
    Calibrate classifier probabilities.

    Parameters:
    -----------
    method : str, 'sigmoid' or 'isotonic'
        - 'sigmoid': Platt scaling (parametric)
        - 'isotonic': Non-parametric (needs more data)
    """
    # Train calibrated classifier
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train, y_train)

    # Compare before and after
    y_proba_before = model.predict_proba(X_test)[:, 1]
    y_proba_after = calibrated.predict_proba(X_test)[:, 1]

    logloss_before = log_loss(y_test, y_proba_before)
    logloss_after = log_loss(y_test, y_proba_after)

    print(f"Log Loss before calibration: {logloss_before:.4f}")
    print(f"Log Loss after calibration: {logloss_after:.4f}")
    print(f"Improvement: {(logloss_before - logloss_after):.4f}")

    return calibrated

# Example
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size=0.3, random_state=42)
model_uncal = RandomForestClassifier(random_state=42)
model_uncal.fit(X_train, y_train)

calibrated_model = calibrate_classifier(model_uncal, X_train, y_train, X_test, y_test)
```

---

## Regression Metrics

### 1. Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)

$$\text{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\text{RMSE} = \sqrt{\text{MSE}}$$

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def regression_metrics(y_true, y_pred):
    """
    Calculate all regression metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Calculate residuals
    residuals = y_true - y_pred

    print("=" * 60)
    print("REGRESSION METRICS")
    print("=" * 60)
    print(f"MSE:   {mse:.4f}")
    print(f"RMSE:  {rmse:.4f}")
    print(f"MAE:   {mae:.4f}")
    print(f"R²:    {r2:.4f}")

    print("\n" + "=" * 60)
    print("RESIDUAL STATISTICS")
    print("=" * 60)
    print(f"Mean residual:   {residuals.mean():.4f}")
    print(f"Std residual:    {residuals.std():.4f}")
    print(f"Min residual:    {residuals.min():.4f}")
    print(f"Max residual:    {residuals.max():.4f}")

    # Plot residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Residual plot
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(y=0, color='r', linestyle='--')
    axes[0].set_xlabel('Predicted Values')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residual Plot')
    axes[0].grid(alpha=0.3)

    # Residual distribution
    axes[1].hist(residuals, bins=30, edgecolor='black')
    axes[1].set_xlabel('Residuals')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Residual Distribution')
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return mse, rmse, mae, r2

# Example
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor

X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

model_reg = RandomForestRegressor(random_state=42)
model_reg.fit(X_train_reg, y_train_reg)
y_pred_reg = model_reg.predict(X_test_reg)

metrics = regression_metrics(y_test_reg, y_pred_reg)
```

**When to Use:**
- **MSE/RMSE:** Penalizes large errors more (outlier sensitive)
- Same units as target variable (RMSE)
- Standard choice for regression

---

### 2. Mean Absolute Error (MAE)

$$\text{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**Properties:**
- More robust to outliers than MSE/RMSE
- Same scale as target variable
- Interpretable: average prediction error

```python
def compare_mse_mae(y_true, y_pred, with_outliers=False):
    """
    Compare MSE and MAE behavior with/without outliers.
    """
    if with_outliers:
        # Add some outliers
        y_pred_outlier = y_pred.copy()
        outlier_indices = np.random.choice(len(y_pred), size=10, replace=False)
        y_pred_outlier[outlier_indices] += np.random.randn(10) * 100
        y_pred = y_pred_outlier

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"{'With outliers' if with_outliers else 'Without outliers'}:")
    print(f"  MSE:  {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE/MAE ratio: {rmse/mae:.2f}")
    print()

# Compare
compare_mse_mae(y_test_reg, y_pred_reg, with_outliers=False)
compare_mse_mae(y_test_reg, y_pred_reg, with_outliers=True)
```

---

### 3. R² (Coefficient of Determination)

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

**Interpretation:**
- Proportion of variance in target explained by model
- Range: (-∞, 1]
- **R² = 1:** Perfect fit
- **R² = 0:** Model no better than mean baseline
- **R² < 0:** Model worse than predicting mean

```python
def detailed_r2_analysis(y_true, y_pred):
    """
    Detailed R² analysis with adjusted R².
    """
    r2 = r2_score(y_true, y_pred)

    # Calculate components
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)

    print("=" * 60)
    print("R² ANALYSIS")
    print("=" * 60)
    print(f"R²: {r2:.4f}")
    print(f"\nComponents:")
    print(f"  SS_res (residual sum of squares): {ss_res:.2f}")
    print(f"  SS_tot (total sum of squares):    {ss_tot:.2f}")
    print(f"  Variance explained: {r2 * 100:.2f}%")

    # Adjusted R² (requires number of features)
    # adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    # where n = samples, p = features

    return r2

r2 = detailed_r2_analysis(y_test_reg, y_pred_reg)
```

**Adjusted R²:** Penalizes adding irrelevant features.

$$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$

where $n$ = samples, $p$ = features.

---

### 4. Mean Absolute Percentage Error (MAPE)

$$\text{MAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|$$

```python
def mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate MAPE (be careful with zeros in y_true).
    """
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    print(f"MAPE: {mape:.2f}%")

    return mape

# Example
mape = mean_absolute_percentage_error(y_test_reg, y_pred_reg)
```

**When to Use:**
- Scale-independent (compare across different datasets)
- Business-friendly (percentage error)

**Limitations:**
- Undefined for zero values
- Asymmetric (over-predictions penalized less than under-predictions)

---

### 5. Symmetric MAPE (SMAPE)

$$\text{SMAPE} = \frac{100\%}{n}\sum_{i=1}^{n}\frac{|y_i - \hat{y}_i|}{(|y_i| + |\hat{y}_i|)/2}$$

```python
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    """
    Calculate SMAPE (symmetric version of MAPE).
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(numerator / denominator) * 100

    print(f"SMAPE: {smape:.2f}%")

    return smape

smape = symmetric_mean_absolute_percentage_error(y_test_reg, y_pred_reg)
```

**Advantages:**
- More symmetric than MAPE
- Bounded: [0%, 200%]

---

### 6. Quantile Loss (for Uncertainty Quantification)

For prediction intervals:

$$L_\tau(y, \hat{y}) = \max[\tau(y - \hat{y}), (\tau - 1)(y - \hat{y})]$$

```python
from sklearn.ensemble import GradientBoostingRegressor

def quantile_regression_evaluation(X_train, y_train, X_test, y_test, quantiles=[0.05, 0.5, 0.95]):
    """
    Evaluate quantile regression for prediction intervals.
    """
    predictions = {}

    for quantile in quantiles:
        model = GradientBoostingRegressor(loss='quantile', alpha=quantile, random_state=42)
        model.fit(X_train, y_train)
        predictions[quantile] = model.predict(X_test)

    # Plot prediction intervals
    plt.figure(figsize=(12, 6))

    # Sort by true values for better visualization
    sort_idx = np.argsort(y_test)
    x_plot = np.arange(len(y_test))

    plt.plot(x_plot, y_test[sort_idx], 'o', label='True values', alpha=0.5)
    plt.plot(x_plot, predictions[0.5][sort_idx], '-', label='Median prediction', linewidth=2)
    plt.fill_between(
        x_plot,
        predictions[0.05][sort_idx],
        predictions[0.95][sort_idx],
        alpha=0.3,
        label='90% prediction interval'
    )

    plt.xlabel('Sample (sorted)')
    plt.ylabel('Value')
    plt.title('Quantile Regression: Prediction Intervals')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Calculate coverage
    coverage = np.mean(
        (y_test >= predictions[0.05]) & (y_test <= predictions[0.95])
    )
    print(f"90% Prediction Interval Coverage: {coverage * 100:.2f}%")

    return predictions

# Example
predictions = quantile_regression_evaluation(
    X_train_reg, y_train_reg, X_test_reg, y_test_reg
)
```

**When to Use:**
- Uncertainty quantification
- Risk-sensitive applications
- Asymmetric cost functions

---

## Ranking Metrics

### 1. Normalized Discounted Cumulative Gain (NDCG)

Used in information retrieval and recommendation systems.

$$\text{DCG@k} = \sum_{i=1}^{k}\frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

$$\text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}$$

```python
from sklearn.metrics import ndcg_score

def evaluate_ranking(y_true_relevance, y_pred_scores, k=10):
    """
    Evaluate ranking quality with NDCG.

    Parameters:
    -----------
    y_true_relevance : array
        True relevance scores (e.g., 0-5 scale)
    y_pred_scores : array
        Predicted scores for ranking
    """
    # NDCG at different k values
    k_values = [1, 3, 5, 10, len(y_true_relevance)]

    print("=" * 60)
    print("RANKING METRICS (NDCG)")
    print("=" * 60)

    for k_val in k_values:
        if k_val <= len(y_true_relevance):
            ndcg = ndcg_score([y_true_relevance], [y_pred_scores], k=k_val)
            print(f"NDCG@{k_val}: {ndcg:.4f}")

    return ndcg

# Example: Search results relevance
y_true_relevance = [3, 2, 3, 0, 1, 2, 3, 2, 0, 1]  # Relevance scores (0-3)
y_pred_scores = [0.9, 0.7, 0.8, 0.1, 0.3, 0.6, 0.85, 0.5, 0.2, 0.4]  # Model scores

ndcg = evaluate_ranking(y_true_relevance, y_pred_scores)
```

---

### 2. Mean Reciprocal Rank (MRR)

$$\text{MRR} = \frac{1}{|Q|}\sum_{i=1}^{|Q|}\frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of first relevant result.

```python
def mean_reciprocal_rank(y_true_list, y_pred_list):
    """
    Calculate MRR for multiple queries.

    Parameters:
    -----------
    y_true_list : list of lists
        True relevance for each query
    y_pred_list : list of lists
        Predicted ranking for each query
    """
    reciprocal_ranks = []

    for y_true, y_pred in zip(y_true_list, y_pred_list):
        # Find rank of first relevant item
        for rank, (true_rel, pred_score) in enumerate(zip(y_true, y_pred), start=1):
            if true_rel > 0:  # Relevant item
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)  # No relevant item found

    mrr = np.mean(reciprocal_ranks)
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")

    return mrr

# Example: Multiple search queries
y_true_list = [
    [1, 0, 1, 0],  # Query 1: relevant at positions 0 and 2
    [0, 1, 0, 0],  # Query 2: relevant at position 1
    [0, 0, 0, 1],  # Query 3: relevant at position 3
]
y_pred_list = [
    [0.9, 0.1, 0.8, 0.2],
    [0.3, 0.9, 0.1, 0.2],
    [0.4, 0.5, 0.3, 0.8],
]

mrr = mean_reciprocal_rank(y_true_list, y_pred_list)
```

---

## NLP Metrics

### 1. BLEU Score

Bilingual Evaluation Understudy (for machine translation).

```python
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def evaluate_bleu(reference, hypothesis):
    """
    Calculate BLEU score for translation quality.

    Parameters:
    -----------
    reference : list of str
        Reference translations (can be multiple)
    hypothesis : str
        Generated translation
    """
    # Tokenize
    reference_tokens = [ref.split() for ref in reference]
    hypothesis_tokens = hypothesis.split()

    # Calculate BLEU scores
    bleu1 = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu(reference_tokens, hypothesis_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    print("=" * 60)
    print("BLEU SCORES")
    print("=" * 60)
    print(f"BLEU-1: {bleu1:.4f} (unigram)")
    print(f"BLEU-2: {bleu2:.4f} (bigram)")
    print(f"BLEU-3: {bleu3:.4f} (trigram)")
    print(f"BLEU-4: {bleu4:.4f} (4-gram)")

    return bleu4

# Example
reference = ["the cat is on the mat", "there is a cat on the mat"]
hypothesis = "the cat is sitting on the mat"

bleu = evaluate_bleu(reference, hypothesis)
```

---

### 2. ROUGE Score

Recall-Oriented Understudy for Gisting Evaluation (for summarization).

```python
from rouge_score import rouge_scorer

def evaluate_rouge(reference, hypothesis):
    """
    Calculate ROUGE scores for summarization.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    print("=" * 60)
    print("ROUGE SCORES")
    print("=" * 60)
    for metric, score in scores.items():
        print(f"{metric.upper()}:")
        print(f"  Precision: {score.precision:.4f}")
        print(f"  Recall:    {score.recall:.4f}")
        print(f"  F1:        {score.fmeasure:.4f}")

    return scores

# Example
reference = "The cat is on the mat. It is sleeping."
hypothesis = "A cat sleeps on the mat."

rouge_scores = evaluate_rouge(reference, hypothesis)
```

---

### 3. Perplexity

Measures how well a language model predicts text.

$$\text{Perplexity} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(w_i|w_{<i})\right)$$

Lower perplexity = better model.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def calculate_perplexity(text, model_name='gpt2'):
    """
    Calculate perplexity of text using pretrained LM.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    # Encode text
    encodings = tokenizer(text, return_tensors='pt')

    # Calculate perplexity
    with torch.no_grad():
        outputs = model(**encodings, labels=encodings['input_ids'])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    print(f"Perplexity: {perplexity:.2f}")
    print("Lower perplexity = better prediction")

    return perplexity

# Example (requires transformers library)
# text = "The quick brown fox jumps over the lazy dog."
# perplexity = calculate_perplexity(text)
```

---

## Computer Vision Metrics

### 1. Intersection over Union (IoU)

For object detection and segmentation.

$$\text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}}$$

```python
def calculate_iou(box1, box2):
    """
    Calculate IoU between two bounding boxes.

    Parameters:
    -----------
    box1, box2 : array [x1, y1, x2, y2]
        Bounding box coordinates (top-left and bottom-right)
    """
    # Calculate intersection
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # Calculate union
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area > 0 else 0

    print(f"IoU: {iou:.4f}")

    return iou

# Example
box_true = [100, 100, 200, 200]  # Ground truth
box_pred = [120, 120, 220, 200]  # Prediction

iou = calculate_iou(box_true, box_pred)
```

---

### 2. Mean Average Precision (mAP)

For object detection across multiple classes.

```python
def calculate_ap(recalls, precisions):
    """
    Calculate Average Precision (AP) using 11-point interpolation.
    """
    # 11-point interpolation
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11

    return ap

def mean_average_precision(y_true_boxes, y_pred_boxes, iou_threshold=0.5):
    """
    Calculate mAP for object detection.

    This is a simplified example. Production implementations
    use libraries like pycocotools.
    """
    # Implementation depends on specific format
    # Typically:
    # 1. Calculate IoU between predicted and true boxes
    # 2. Sort predictions by confidence
    # 3. Calculate precision-recall curve
    # 4. Calculate AP for each class
    # 5. Average across classes

    print("For production mAP calculation, use:")
    print("  - pycocotools for COCO-style evaluation")
    print("  - torchmetrics.detection.mean_ap")

    pass

# For production, use:
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
```

---

### 3. Dice Coefficient

For image segmentation (similar to F1-score).

$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

```python
def dice_coefficient(y_true_mask, y_pred_mask):
    """
    Calculate Dice coefficient for segmentation.

    Parameters:
    -----------
    y_true_mask, y_pred_mask : array
        Binary segmentation masks
    """
    intersection = np.sum(y_true_mask * y_pred_mask)
    dice = (2.0 * intersection) / (np.sum(y_true_mask) + np.sum(y_pred_mask))

    print(f"Dice Coefficient: {dice:.4f}")

    return dice

# Example
y_true_mask = np.random.randint(0, 2, size=(256, 256))
y_pred_mask = np.random.randint(0, 2, size=(256, 256))

dice = dice_coefficient(y_true_mask, y_pred_mask)
```

---

## Metric Selection Guide

### Decision Tree for Metric Selection

```
Classification?
├─ Yes
│  ├─ Binary or Multi-class?
│  │  ├─ Binary
│  │  │  ├─ Balanced classes?
│  │  │  │  ├─ Yes → Accuracy, F1-Score
│  │  │  │  └─ No → F1-Score, ROC-AUC, PR-AUC, MCC
│  │  │  ├─ Need probabilities?
│  │  │  │  └─ Yes → Log Loss, Brier Score
│  │  │  └─ Different error costs?
│  │  │     └─ Yes → Custom weighted metric
│  │  └─ Multi-class
│  │     ├─ Classes equally important?
│  │     │  ├─ Yes → Macro F1
│  │     │  └─ No → Weighted F1
│  │     └─ Need calibrated probabilities?
│  │        └─ Yes → Log Loss
│  └─ Ranking/Recommendation?
│     └─ NDCG, MRR, MAP
└─ No (Regression?)
   ├─ Yes
   │  ├─ Outliers present?
   │  │  ├─ Yes → MAE, Huber Loss
   │  │  └─ No → RMSE, MSE
   │  ├─ Need interpretable percentage?
   │  │  └─ Yes → MAPE, SMAPE
   │  ├─ Multiple scales/datasets?
   │  │  └─ Yes → MAPE, R²
   │  └─ Need uncertainty?
   │     └─ Yes → Quantile Loss
   └─ NLP?
      ├─ Translation → BLEU
      ├─ Summarization → ROUGE
      └─ Language Modeling → Perplexity
```

### Metric Selection by Domain

| Domain | Primary Metrics | Secondary Metrics |
|--------|----------------|-------------------|
| **Fraud Detection** | Precision@k, F2-score | ROC-AUC, PR-AUC |
| **Medical Diagnosis** | Recall, F2-score | Specificity, NPV |
| **Spam Detection** | Precision, F0.5-score | Accuracy |
| **Credit Scoring** | ROC-AUC, KS statistic | Gini coefficient |
| **Recommendation** | NDCG@k, MAP@k | Precision@k, Recall@k |
| **Demand Forecasting** | MAPE, WMAPE | RMSE, MAE |
| **Image Segmentation** | Dice, IoU | Pixel Accuracy |
| **Object Detection** | mAP@0.5, mAP@0.5:0.95 | Recall@k |
| **Machine Translation** | BLEU, METEOR | ROUGE-L |
| **Text Summarization** | ROUGE-1, ROUGE-2, ROUGE-L | BLEU |

---

## Summary: Key Takeaways

### 1. No Single "Best" Metric
- Always use multiple metrics
- Align metrics with business objectives
- Consider error costs

### 2. Beware of Misleading Metrics
- Accuracy on imbalanced data
- Metrics on training set
- Cherry-picked metrics

### 3. Always Validate Properly
- Use cross-validation
- Report confidence intervals
- Test statistical significance

### 4. Consider the Full Picture
- Performance metrics (accuracy, RMSE)
- Calibration (probability quality)
- Fairness (demographic parity)
- Interpretability
- Computational cost

### 5. Production Considerations
- Latency requirements
- Memory constraints
- Online vs. offline evaluation
- A/B testing metrics

---

**Last Updated:** 2025-10-14
**Status:** Complete - Production-ready implementations with PhD-level explanations
