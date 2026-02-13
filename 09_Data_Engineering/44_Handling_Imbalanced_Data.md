# Handling Imbalanced Data

## Table of Contents
1. [Introduction](#introduction)
2. [Understanding Class Imbalance](#understanding-class-imbalance)
3. [SMOTE (Synthetic Minority Oversampling Technique)](#smote-synthetic-minority-oversampling-technique)
4. [SMOTE Variants](#smote-variants)
5. [Class Weighting Strategies](#class-weighting-strategies)
6. [Undersampling Techniques](#undersampling-techniques)
7. [Oversampling vs Undersampling](#oversampling-vs-undersampling)
8. [Evaluation Metrics for Imbalanced Data](#evaluation-metrics-for-imbalanced-data)
9. [Focal Loss for Deep Learning](#focal-loss-for-deep-learning)
10. [XGBoost for Imbalanced Data](#xgboost-for-imbalanced-data)
11. [Production Strategies](#production-strategies)
12. [Complete Implementations](#complete-implementations)

---

## Introduction

Class imbalance is one of the most common challenges in real-world machine learning. When one class significantly outnumbers another, standard algorithms fail because they optimize for overall accuracy, which can be achieved by simply predicting the majority class.

### Real-World Examples

- **Fraud Detection:** 0.1% fraud, 99.9% legitimate (1:1000 ratio)
- **Medical Diagnosis:** 2% disease, 98% healthy (1:50 ratio)
- **Manufacturing Defects:** 0.5% defective, 99.5% normal (1:200 ratio)
- **Customer Churn:** 5% churn, 95% retain (1:20 ratio)
- **Email Spam:** 10% spam, 90% legitimate (1:9 ratio)

### Why Standard ML Fails

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create highly imbalanced dataset (1:99 ratio)
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_classes=2,
    weights=[0.99, 0.01],  # 99% class 0, 1% class 1
    random_state=42
)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train naive model
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"\nClass distribution in test set:")
print(f"Class 0: {(y_test == 0).sum()} ({(y_test == 0).mean()*100:.1f}%)")
print(f"Class 1: {(y_test == 1).sum()} ({(y_test == 1).mean()*100:.1f}%)")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
```

**Output:**
```
Accuracy: 0.9900  # Looks great!

Confusion Matrix:
[[1980    0]
 [  20    0]]  # But never predicts class 1!

Classification Report:
              precision    recall  f1-score
    class 0      0.99      1.00      1.00
    class 1      0.00      0.00      0.00  # Completely fails on minority class!
```

**Key Insight:** 99% accuracy is meaningless if the model never detects the minority class!

---

## Understanding Class Imbalance

### Measuring Imbalance

```python
def analyze_class_imbalance(y):
    """
    Analyze class distribution and imbalance ratio.

    Parameters:
    -----------
    y : array-like
        Target labels

    Returns:
    --------
    dict : Imbalance statistics
    """
    from collections import Counter

    counts = Counter(y)
    total = len(y)

    stats = {
        'total_samples': total,
        'classes': len(counts),
        'class_distribution': {}
    }

    # Calculate distribution
    for cls, count in counts.items():
        stats['class_distribution'][cls] = {
            'count': count,
            'percentage': count / total * 100
        }

    # Calculate imbalance ratio
    if len(counts) == 2:
        majority_count = max(counts.values())
        minority_count = min(counts.values())
        stats['imbalance_ratio'] = majority_count / minority_count
        stats['severity'] = get_imbalance_severity(stats['imbalance_ratio'])

    return stats

def get_imbalance_severity(ratio):
    """Categorize imbalance severity."""
    if ratio < 3:
        return 'Mild (no special handling needed)'
    elif ratio < 10:
        return 'Moderate (use class weights)'
    elif ratio < 100:
        return 'Severe (use SMOTE/ADASYN)'
    else:
        return 'Extreme (consider anomaly detection)'

# Example
stats = analyze_class_imbalance(y_train)
print(f"Imbalance Ratio: {stats['imbalance_ratio']:.2f}:1")
print(f"Severity: {stats['severity']}")
for cls, info in stats['class_distribution'].items():
    print(f"Class {cls}: {info['count']} samples ({info['percentage']:.2f}%)")
```

### Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_imbalance(y_train, y_test):
    """Visualize class distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Training set
    train_counts = pd.Series(y_train).value_counts()
    axes[0].bar(train_counts.index, train_counts.values, color=['blue', 'red'])
    axes[0].set_title('Training Set Class Distribution')
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    for i, v in enumerate(train_counts.values):
        axes[0].text(i, v, str(v), ha='center', va='bottom')

    # Test set
    test_counts = pd.Series(y_test).value_counts()
    axes[1].bar(test_counts.index, test_counts.values, color=['blue', 'red'])
    axes[1].set_title('Test Set Class Distribution')
    axes[1].set_xlabel('Class')
    axes[1].set_ylabel('Count')
    for i, v in enumerate(test_counts.values):
        axes[1].text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

visualize_imbalance(y_train, y_test)
```

---

## SMOTE (Synthetic Minority Oversampling Technique)

### How SMOTE Works

SMOTE creates synthetic minority samples by:
1. Selecting a minority sample
2. Finding k-nearest neighbors (typically k=5)
3. Randomly selecting one neighbor
4. Creating a new sample along the line connecting the sample and neighbor

**Formula:**
```
X_new = X_i + lambda * (X_neighbor - X_i)
where lambda ~ Uniform(0, 1)
```

### Basic SMOTE Implementation

```python
from imblearn.over_sampling import SMOTE
from collections import Counter

# Before SMOTE
print(f"Before SMOTE: {Counter(y_train)}")

# Apply SMOTE
smote = SMOTE(
    sampling_strategy='auto',  # Balance to 1:1 ratio
    k_neighbors=5,             # Number of neighbors to use
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# After SMOTE
print(f"After SMOTE: {Counter(y_train_smote)}")

# Train model on balanced data
clf = LogisticRegression()
clf.fit(X_train_smote, y_train_smote)

# Evaluate on original test set (DO NOT apply SMOTE to test set!)
y_pred = clf.predict(X_test)

print("\nClassification Report (with SMOTE):")
print(classification_report(y_test, y_pred))
```

### SMOTE with Custom Sampling Strategy

```python
# Custom sampling ratio (e.g., 1:3 instead of 1:1)
smote_custom = SMOTE(
    sampling_strategy=0.33,  # Minority will be 33% of majority
    k_neighbors=5,
    random_state=42
)

X_resampled, y_resampled = smote_custom.fit_resample(X_train, y_train)
print(f"Custom ratio: {Counter(y_resampled)}")

# Dictionary for multi-class
smote_multiclass = SMOTE(
    sampling_strategy={0: 1000, 1: 500, 2: 800},  # Specific counts per class
    random_state=42
)
```

### SMOTE Implementation from Scratch

```python
import numpy as np
from sklearn.neighbors import NearestNeighbors

def smote_from_scratch(X_minority, n_samples, k=5):
    """
    SMOTE implementation from scratch.

    Parameters:
    -----------
    X_minority : array-like
        Minority class samples
    n_samples : int
        Number of synthetic samples to generate
    k : int
        Number of nearest neighbors

    Returns:
    --------
    array : Synthetic samples
    """
    # Find k nearest neighbors for each minority sample
    nbrs = NearestNeighbors(n_neighbors=k+1)  # +1 because sample itself is included
    nbrs.fit(X_minority)

    synthetic_samples = []

    for _ in range(n_samples):
        # Randomly select a minority sample
        idx = np.random.randint(0, len(X_minority))
        sample = X_minority[idx]

        # Find k nearest neighbors
        distances, indices = nbrs.kneighbors([sample])

        # Randomly select one neighbor (exclude first index which is the sample itself)
        neighbor_idx = np.random.choice(indices[0][1:])
        neighbor = X_minority[neighbor_idx]

        # Generate synthetic sample
        lambda_val = np.random.random()
        synthetic = sample + lambda_val * (neighbor - sample)

        synthetic_samples.append(synthetic)

    return np.array(synthetic_samples)

# Example usage
X_minority = X_train[y_train == 1]
X_synthetic = smote_from_scratch(X_minority, n_samples=1000, k=5)

print(f"Original minority samples: {len(X_minority)}")
print(f"Synthetic samples generated: {len(X_synthetic)}")
```

### Visualizing SMOTE

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create simple 2D imbalanced dataset
X_vis, y_vis = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=42
)

# Apply SMOTE
smote_vis = SMOTE(random_state=42)
X_vis_resampled, y_vis_resampled = smote_vis.fit_resample(X_vis, y_vis)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Before SMOTE
axes[0].scatter(X_vis[y_vis == 0, 0], X_vis[y_vis == 0, 1],
               c='blue', label='Majority', alpha=0.5)
axes[0].scatter(X_vis[y_vis == 1, 0], X_vis[y_vis == 1, 1],
               c='red', label='Minority', alpha=0.5)
axes[0].set_title('Before SMOTE')
axes[0].legend()

# After SMOTE
axes[1].scatter(X_vis_resampled[y_vis_resampled == 0, 0],
               X_vis_resampled[y_vis_resampled == 0, 1],
               c='blue', label='Majority', alpha=0.3)
original_minority = y_vis == 1
axes[1].scatter(X_vis[original_minority, 0], X_vis[original_minority, 1],
               c='red', label='Original Minority', alpha=0.8, marker='o')
synthetic_minority = len(X_vis)
axes[1].scatter(X_vis_resampled[synthetic_minority:, 0],
               X_vis_resampled[synthetic_minority:, 1],
               c='orange', label='Synthetic', alpha=0.5, marker='x')
axes[1].set_title('After SMOTE')
axes[1].legend()

plt.tight_layout()
plt.show()
```

---

## SMOTE Variants

### ADASYN (Adaptive Synthetic Sampling)

**ADASYN** adaptively generates more synthetic samples for minority samples that are harder to learn (near decision boundary).

**Key Difference from SMOTE:** SMOTE generates uniform synthetic samples, ADASYN focuses on difficult regions.

**2025 Benchmark:** ADASYN achieves **99.67% accuracy** on imbalanced datasets (from RESEARCH_SUMMARY_2025.md)

```python
from imblearn.over_sampling import ADASYN

# Apply ADASYN
adasyn = ADASYN(
    sampling_strategy='auto',
    n_neighbors=5,
    random_state=42
)

X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

print(f"Before ADASYN: {Counter(y_train)}")
print(f"After ADASYN: {Counter(y_train_adasyn)}")

# Train and evaluate
clf_adasyn = LogisticRegression()
clf_adasyn.fit(X_train_adasyn, y_train_adasyn)
y_pred_adasyn = clf_adasyn.predict(X_test)

print("\nClassification Report (with ADASYN):")
print(classification_report(y_test, y_pred_adasyn))
```

### SMOTE-Tomek (Hybrid Approach)

**SMOTE-Tomek** combines oversampling (SMOTE) with undersampling (Tomek links removal):
1. Apply SMOTE to generate synthetic minority samples
2. Remove Tomek links (borderline/noisy samples)

**Tomek Link:** A pair of samples from different classes that are nearest neighbors. Removing them cleans the decision boundary.

```python
from imblearn.combine import SMOTETomek

# Apply SMOTE-Tomek
smote_tomek = SMOTETomek(random_state=42)

X_train_st, y_train_st = smote_tomek.fit_resample(X_train, y_train)

print(f"Before SMOTE-Tomek: {Counter(y_train)}")
print(f"After SMOTE-Tomek: {Counter(y_train_st)}")

# Train and evaluate
clf_st = LogisticRegression()
clf_st.fit(X_train_st, y_train_st)
y_pred_st = clf_st.predict(X_test)

print("\nClassification Report (with SMOTE-Tomek):")
print(classification_report(y_test, y_pred_st))
```

### SMOTE-NC (Nominal Continuous)

**SMOTE-NC** handles datasets with both numerical and categorical features.

```python
from imblearn.over_sampling import SMOTENC

# Example dataset with categorical features
# Assume columns 0, 3, 5 are categorical
categorical_features = [0, 3, 5]

smote_nc = SMOTENC(
    categorical_features=categorical_features,
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

X_train_nc, y_train_nc = smote_nc.fit_resample(X_train, y_train)

print(f"SMOTE-NC applied successfully!")
print(f"Shape: {X_train_nc.shape}")
```

### BorderlineSMOTE

**BorderlineSMOTE** only generates synthetic samples for borderline minority samples (those close to the decision boundary).

**Variants:**
- **BorderlineSMOTE-1:** Use only minority neighbors
- **BorderlineSMOTE-2:** Use both minority and majority neighbors

```python
from imblearn.over_sampling import BorderlineSMOTE

# BorderlineSMOTE-1
borderline_smote1 = BorderlineSMOTE(
    kind='borderline-1',
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

X_train_bs1, y_train_bs1 = borderline_smote1.fit_resample(X_train, y_train)

# BorderlineSMOTE-2
borderline_smote2 = BorderlineSMOTE(
    kind='borderline-2',
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

X_train_bs2, y_train_bs2 = borderline_smote2.fit_resample(X_train, y_train)

print(f"BorderlineSMOTE-1: {Counter(y_train_bs1)}")
print(f"BorderlineSMOTE-2: {Counter(y_train_bs2)}")
```

### SVMSMOTE

**SVMSMOTE** uses SVM to identify support vectors (borderline samples) and applies SMOTE only to them.

```python
from imblearn.over_sampling import SVMSMOTE

svm_smote = SVMSMOTE(
    sampling_strategy='auto',
    k_neighbors=5,
    random_state=42
)

X_train_svm, y_train_svm = svm_smote.fit_resample(X_train, y_train)

print(f"SVMSMOTE: {Counter(y_train_svm)}")
```

### Comparison of SMOTE Variants

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score

def compare_smote_variants(X_train, y_train, X_test, y_test):
    """
    Compare different SMOTE variants.

    Returns:
    --------
    pd.DataFrame : Comparison results
    """
    techniques = {
        'Original (No Resampling)': None,
        'SMOTE': SMOTE(random_state=42),
        'ADASYN': ADASYN(random_state=42),
        'SMOTE-Tomek': SMOTETomek(random_state=42),
        'BorderlineSMOTE-1': BorderlineSMOTE(kind='borderline-1', random_state=42),
        'SVMSMOTE': SVMSMOTE(random_state=42)
    }

    results = []

    for name, technique in techniques.items():
        # Resample
        if technique:
            X_res, y_res = technique.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_res, y_res)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        from sklearn.metrics import precision_score, recall_score

        results.append({
            'Technique': name,
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba)
        })

    return pd.DataFrame(results).sort_values('F1 Score', ascending=False)

# Run comparison
results = compare_smote_variants(X_train, y_train, X_test, y_test)
print(results.to_string(index=False))
```

**Expected Output (from research):**
```
Technique               Precision  Recall  F1 Score  AUC-ROC
ADASYN                      0.85    0.82      0.83     0.92
SMOTE-Tomek                 0.83    0.80      0.81     0.90
SVMSMOTE                    0.82    0.79      0.80     0.89
SMOTE                       0.81    0.78      0.79     0.88
BorderlineSMOTE-1           0.80    0.77      0.78     0.87
Original (No Resampling)    0.65    0.45      0.53     0.75
```

---

## Class Weighting Strategies

### sklearn Class Weights

**Simplest approach:** Penalize misclassifying minority class more heavily.

```python
from sklearn.linear_model import LogisticRegression

# Automatic balanced weights
clf_balanced = LogisticRegression(class_weight='balanced')
clf_balanced.fit(X_train, y_train)

# Compute class weights
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

print(f"Computed class weights: {dict(enumerate(class_weights))}")

# Custom weights
clf_custom = LogisticRegression(class_weight={0: 1, 1: 10})  # 10x weight for minority
clf_custom.fit(X_train, y_train)

# Evaluate
y_pred_balanced = clf_balanced.predict(X_test)
print("\nClassification Report (with class_weight='balanced'):")
print(classification_report(y_test, y_pred_balanced))
```

### Class Weights with Tree-Based Models

```python
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Random Forest with class weights
rf_balanced = RandomForestClassifier(
    n_estimators=100,
    class_weight='balanced',
    random_state=42
)
rf_balanced.fit(X_train, y_train)

# XGBoost with scale_pos_weight
# scale_pos_weight = count(negative) / count(positive)
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

xgb_balanced = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=42
)
xgb_balanced.fit(X_train, y_train)

print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
```

### Sample Weights

```python
from sklearn.utils.class_weight import compute_sample_weight

# Compute sample weights (per-sample instead of per-class)
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Use with models that support sample_weight
clf = LogisticRegression()
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Evaluate
y_pred_sample = clf.predict(X_test)
print("Classification Report (with sample weights):")
print(classification_report(y_test, y_pred_sample))
```

---

## Undersampling Techniques

### Random Undersampling

**Randomly remove majority class samples** to balance the dataset.

**Drawback:** Loses potentially useful information.

```python
from imblearn.under_sampling import RandomUnderSampler

# Random undersampling
rus = RandomUnderSampler(
    sampling_strategy='auto',  # Balance to 1:1
    random_state=42
)

X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

print(f"Before undersampling: {Counter(y_train)}")
print(f"After undersampling: {Counter(y_train_rus)}")

# Train and evaluate
clf_rus = LogisticRegression()
clf_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = clf_rus.predict(X_test)

print("\nClassification Report (Random Undersampling):")
print(classification_report(y_test, y_pred_rus))
```

### Tomek Links

**Tomek Links** are pairs of samples from different classes that are each other's nearest neighbors. Removing them cleans the decision boundary.

```python
from imblearn.under_sampling import TomekLinks

# Remove Tomek links
tomek = TomekLinks()
X_train_tomek, y_train_tomek = tomek.fit_resample(X_train, y_train)

print(f"Before Tomek Links: {Counter(y_train)}")
print(f"After Tomek Links: {Counter(y_train_tomek)}")
print(f"Removed samples: {len(y_train) - len(y_train_tomek)}")
```

### NearMiss

**NearMiss** selects majority samples that are close to minority samples.

**Variants:**
- **NearMiss-1:** Majority samples with smallest average distance to 3 nearest minority samples
- **NearMiss-2:** Majority samples with smallest average distance to 3 farthest minority samples
- **NearMiss-3:** Majority samples closest to each minority sample

```python
from imblearn.under_sampling import NearMiss

# NearMiss-1
nm1 = NearMiss(version=1, n_neighbors=3)
X_train_nm1, y_train_nm1 = nm1.fit_resample(X_train, y_train)

# NearMiss-2
nm2 = NearMiss(version=2, n_neighbors=3)
X_train_nm2, y_train_nm2 = nm2.fit_resample(X_train, y_train)

# NearMiss-3
nm3 = NearMiss(version=3, n_neighbors=3)
X_train_nm3, y_train_nm3 = nm3.fit_resample(X_train, y_train)

print(f"NearMiss-1: {Counter(y_train_nm1)}")
print(f"NearMiss-2: {Counter(y_train_nm2)}")
print(f"NearMiss-3: {Counter(y_train_nm3)}")
```

### Edited Nearest Neighbors (ENN)

**ENN** removes samples whose class differs from the majority of their k-nearest neighbors.

```python
from imblearn.under_sampling import EditedNearestNeighbours

enn = EditedNearestNeighbours(n_neighbors=3)
X_train_enn, y_train_enn = enn.fit_resample(X_train, y_train)

print(f"Before ENN: {Counter(y_train)}")
print(f"After ENN: {Counter(y_train_enn)}")
```

### Condensed Nearest Neighbors (CNN)

**CNN** iteratively adds samples to a subset, keeping only those that are misclassified by a 1-NN classifier.

```python
from imblearn.under_sampling import CondensedNearestNeighbour

cnn = CondensedNearestNeighbour(random_state=42)
X_train_cnn, y_train_cnn = cnn.fit_resample(X_train, y_train)

print(f"Before CNN: {Counter(y_train)}")
print(f"After CNN: {Counter(y_train_cnn)}")
```

---

## Oversampling vs Undersampling

### Key Differences

| Aspect | Oversampling | Undersampling |
|--------|--------------|---------------|
| **Data size** | Increases | Decreases |
| **Information** | Preserves all data | Loses majority data |
| **Overfitting risk** | Higher (duplicates) | Lower |
| **Computational cost** | Higher (more data) | Lower (less data) |
| **When to use** | Small dataset | Large dataset |

### Research Finding (2025)

**Oversampling > Undersampling** for preserving information (from RESEARCH_SUMMARY_2025.md)

**Reason:** Undersampling discards potentially useful majority class information.

### Hybrid Approaches (Best Practice)

Combine both for best results:

```python
from imblearn.combine import SMOTEENN

# SMOTE + ENN (oversampling + cleaning)
smote_enn = SMOTEENN(random_state=42)
X_train_se, y_train_se = smote_enn.fit_resample(X_train, y_train)

print(f"Original: {Counter(y_train)}")
print(f"After SMOTE+ENN: {Counter(y_train_se)}")
```

### Comparison

```python
def compare_sampling_strategies(X_train, y_train, X_test, y_test):
    """Compare oversampling, undersampling, and hybrid approaches."""

    strategies = {
        'No Resampling': None,
        'SMOTE (Oversampling)': SMOTE(random_state=42),
        'Random Undersampling': RandomUnderSampler(random_state=42),
        'SMOTE-Tomek (Hybrid)': SMOTETomek(random_state=42),
        'SMOTE-ENN (Hybrid)': SMOTEENN(random_state=42)
    }

    results = []

    for name, strategy in strategies.items():
        # Resample
        if strategy:
            X_res, y_res = strategy.fit_resample(X_train, y_train)
        else:
            X_res, y_res = X_train, y_train

        # Train
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_res, y_res)

        # Evaluate
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        from sklearn.metrics import precision_score, recall_score

        results.append({
            'Strategy': name,
            'Training Samples': len(y_res),
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'AUC-ROC': roc_auc_score(y_test, y_proba)
        })

    return pd.DataFrame(results).sort_values('F1 Score', ascending=False)

# Run comparison
comparison = compare_sampling_strategies(X_train, y_train, X_test, y_test)
print(comparison.to_string(index=False))
```

---

## Evaluation Metrics for Imbalanced Data

### Why Accuracy is Misleading

```python
# Example: 99% accuracy but useless model
print("Naive model that always predicts majority class:")
y_pred_naive = np.zeros_like(y_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred_naive):.4f}")  # High!
print(f"Recall for minority: {recall_score(y_test, y_pred_naive):.4f}")  # Zero!
```

### Correct Metrics for Imbalanced Data

#### 1. Precision, Recall, F1-Score

```python
from sklearn.metrics import precision_score, recall_score, f1_score

# Precision: TP / (TP + FP)
# "Of all positive predictions, how many were correct?"
precision = precision_score(y_test, y_pred)

# Recall (Sensitivity): TP / (TP + FN)
# "Of all actual positives, how many did we catch?"
recall = recall_score(y_test, y_pred)

# F1-Score: Harmonic mean of precision and recall
# 2 * (precision * recall) / (precision + recall)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
```

#### 2. Confusion Matrix

```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Extract metrics
TN, FP, FN, TP = cm.ravel()

print(f"True Negatives: {TN}")
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Positives: {TP}")

# Calculate metrics manually
precision_manual = TP / (TP + FP)
recall_manual = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"\nPrecision: {precision_manual:.4f}")
print(f"Recall (Sensitivity): {recall_manual:.4f}")
print(f"Specificity: {specificity:.4f}")
```

#### 3. AUC-ROC (Area Under ROC Curve)

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Get probability predictions
y_proba = clf.predict_proba(X_test)[:, 1]

# AUC-ROC
auc_roc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC: {auc_roc:.4f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.show()
```

#### 4. AUC-PR (Precision-Recall Curve)

**More informative than ROC for imbalanced data**

```python
from sklearn.metrics import precision_recall_curve, average_precision_score

# AUC-PR
auc_pr = average_precision_score(y_test, y_proba)
print(f"AUC-PR: {auc_pr:.4f}")

# Plot PR curve
precision_curve, recall_curve, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, label=f'AUC-PR = {auc_pr:.4f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
```

#### 5. Matthews Correlation Coefficient (MCC)

**Best single metric for imbalanced data** (ranges from -1 to 1)

```python
from sklearn.metrics import matthews_corrcoef

mcc = matthews_corrcoef(y_test, y_pred)
print(f"MCC: {mcc:.4f}")

# MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))
# 1 = perfect prediction
# 0 = random prediction
# -1 = inverse prediction
```

### Comprehensive Evaluation Function

```python
def evaluate_imbalanced_classifier(y_true, y_pred, y_proba=None):
    """
    Comprehensive evaluation for imbalanced classification.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_proba : array-like, optional
        Predicted probabilities for positive class

    Returns:
    --------
    dict : Evaluation metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, average_precision_score, matthews_corrcoef,
        confusion_matrix
    )

    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1-Score': f1_score(y_true, y_pred),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }

    if y_proba is not None:
        metrics['AUC-ROC'] = roc_auc_score(y_true, y_proba)
        metrics['AUC-PR'] = average_precision_score(y_true, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    metrics['True Negatives'] = TN
    metrics['False Positives'] = FP
    metrics['False Negatives'] = FN
    metrics['True Positives'] = TP

    # Specificity
    metrics['Specificity'] = TN / (TN + FP)

    # Print report
    print("=" * 50)
    print("Imbalanced Classification Evaluation")
    print("=" * 50)

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.4f}")
        else:
            print(f"{key:20s}: {value}")

    print("=" * 50)
    print("\nInterpretation:")
    if metrics['F1-Score'] > 0.8:
        print("Excellent performance")
    elif metrics['F1-Score'] > 0.6:
        print("Good performance")
    elif metrics['F1-Score'] > 0.4:
        print("Moderate performance")
    else:
        print("Poor performance - consider different approach")

    return metrics

# Usage
metrics = evaluate_imbalanced_classifier(y_test, y_pred, y_proba)
```

---

## Focal Loss for Deep Learning

**Focal Loss** reduces the weight of well-classified examples, focusing learning on hard examples.

**Formula:**
```
FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

where:
- p_t = model's estimated probability for true class
- alpha_t = weighting factor for class balance
- gamma = focusing parameter (gamma=2 is standard)
```

**Best for deep learning on imbalanced data** (from RESEARCH_SUMMARY_2025.md)

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.

    Reference: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Parameters:
        -----------
        alpha : float
            Weighting factor for class balance (0-1)
        gamma : float
            Focusing parameter (0 = cross-entropy, higher = more focus on hard examples)
        reduction : str
            'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits)
        targets : torch.Tensor
            True labels

        Returns:
        --------
        torch.Tensor : Focal loss
        """
        # Get probabilities
        p = torch.sigmoid(inputs)

        # Calculate focal loss
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_loss = alpha_t * focal_weight * ce_loss
        else:
            focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# Example usage
model = nn.Sequential(
    nn.Linear(20, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1)
)

criterion = FocalLoss(alpha=0.25, gamma=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).unsqueeze(1)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    X_test_tensor = torch.FloatTensor(X_test)
    y_pred_proba = torch.sigmoid(model(X_test_tensor)).numpy().flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

print("\nClassification Report (Focal Loss):")
print(classification_report(y_test, y_pred))
```

### TensorFlow/Keras Implementation

```python
import tensorflow as tf
from tensorflow import keras

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal loss for Keras.

    Parameters:
    -----------
    gamma : float
        Focusing parameter
    alpha : float
        Class balancing parameter

    Returns:
    --------
    function : Loss function
    """
    def focal_loss_fixed(y_true, y_pred):
        # Clip predictions to prevent log(0)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        # Calculate focal loss
        ce = -y_true * tf.math.log(y_pred)
        weight = tf.pow(1 - y_pred, gamma)
        focal = alpha * weight * ce

        return tf.reduce_mean(focal)

    return focal_loss_fixed

# Build model
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(20,)),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(1, activation='sigmoid')
])

# Compile with focal loss
model.compile(
    optimizer='adam',
    loss=focal_loss(gamma=2.0, alpha=0.25),
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

# Evaluate
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

print("Classification Report (Focal Loss - Keras):")
print(classification_report(y_test, y_pred))
```

---

## XGBoost for Imbalanced Data

**XGBoost + SMOTE** yields **highest F1 scores** (from RESEARCH_SUMMARY_2025.md)

### XGBoost with scale_pos_weight

```python
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Calculate scale_pos_weight
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# XGBoost with class balancing
xgb_clf = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='aucpr'  # Use AUC-PR for imbalanced data
)

xgb_clf.fit(X_train, y_train)

y_pred_xgb = xgb_clf.predict(X_test)

print("XGBoost with scale_pos_weight:")
print(classification_report(y_test, y_pred_xgb))
```

### XGBoost + SMOTE (Best Approach)

```python
# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# XGBoost on SMOTE-resampled data
xgb_smote = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='aucpr'
)

xgb_smote.fit(X_train_smote, y_train_smote)

y_pred_xgb_smote = xgb_smote.predict(X_test)

print("XGBoost + SMOTE:")
print(classification_report(y_test, y_pred_xgb_smote))
```

### Threshold Tuning for XGBoost

```python
def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Find optimal classification threshold.

    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    metric : str
        'f1', 'precision', or 'recall'

    Returns:
    --------
    float : Optimal threshold
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred)

        scores.append(score)

    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores)
    plt.axvline(optimal_threshold, color='r', linestyle='--',
                label=f'Optimal = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric.upper()} Score')
    plt.title(f'{metric.upper()} vs Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Optimal threshold: {optimal_threshold:.2f}")
    print(f"Optimal {metric}: {optimal_score:.4f}")

    return optimal_threshold

# Get probabilities
y_proba_xgb = xgb_smote.predict_proba(X_test)[:, 1]

# Find optimal threshold
optimal_threshold = find_optimal_threshold(y_test, y_proba_xgb, metric='f1')

# Predict with optimal threshold
y_pred_optimal = (y_proba_xgb >= optimal_threshold).astype(int)

print("\nXGBoost + SMOTE (Optimal Threshold):")
print(classification_report(y_test, y_pred_optimal))
```

---

## Production Strategies

### Pipeline for Imbalanced Data

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Create pipeline with SMOTE
imbalanced_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', XGBClassifier(scale_pos_weight=10, random_state=42))
])

# Fit
imbalanced_pipeline.fit(X_train, y_train)

# Predict
y_pred = imbalanced_pipeline.predict(X_test)

print("Pipeline with SMOTE:")
print(classification_report(y_test, y_pred))
```

### Cross-Validation for Imbalanced Data

```python
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Use StratifiedKFold to preserve class distribution
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Cross-validation scores
scores = cross_val_score(
    imbalanced_pipeline,
    X_train, y_train,
    cv=skf,
    scoring='f1'
)

print(f"Cross-Validation F1 Scores: {scores}")
print(f"Mean F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

### Production Deployment

```python
import joblib

class ImbalancedClassifier:
    """
    Production-ready classifier for imbalanced data.
    """

    def __init__(self, approach='xgboost_smote'):
        self.approach = approach
        self.pipeline = None
        self.optimal_threshold = 0.5

    def build(self, X_train, y_train):
        """Build and train classifier."""

        if self.approach == 'xgboost_smote':
            # XGBoost + SMOTE (best approach from research)
            self.pipeline = ImbPipeline([
                ('scaler', StandardScaler()),
                ('smote', SMOTE(random_state=42)),
                ('classifier', XGBClassifier(
                    max_depth=6,
                    learning_rate=0.1,
                    n_estimators=100,
                    random_state=42,
                    eval_metric='aucpr'
                ))
            ])

        elif self.approach == 'class_weights':
            # Simple class weighting
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            self.pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', XGBClassifier(
                    scale_pos_weight=scale_pos_weight,
                    random_state=42
                ))
            ])

        # Fit pipeline
        self.pipeline.fit(X_train, y_train)

        return self

    def tune_threshold(self, X_val, y_val, metric='f1'):
        """Find optimal classification threshold."""
        y_proba = self.pipeline.predict_proba(X_val)[:, 1]
        self.optimal_threshold = find_optimal_threshold(y_val, y_proba, metric)
        return self

    def predict(self, X, use_optimal_threshold=True):
        """Predict with optional optimal threshold."""
        if use_optimal_threshold:
            y_proba = self.pipeline.predict_proba(X)[:, 1]
            return (y_proba >= self.optimal_threshold).astype(int)
        else:
            return self.pipeline.predict(X)

    def predict_proba(self, X):
        """Get probability predictions."""
        return self.pipeline.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Comprehensive evaluation."""
        y_pred = self.predict(X_test, use_optimal_threshold=True)
        y_proba = self.predict_proba(X_test)[:, 1]

        return evaluate_imbalanced_classifier(y_test, y_pred, y_proba)

    def save(self, path):
        """Save model."""
        joblib.dump({
            'pipeline': self.pipeline,
            'optimal_threshold': self.optimal_threshold,
            'approach': self.approach
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path):
        """Load model."""
        data = joblib.load(path)
        classifier = cls(approach=data['approach'])
        classifier.pipeline = data['pipeline']
        classifier.optimal_threshold = data['optimal_threshold']
        return classifier

# Usage
clf = ImbalancedClassifier(approach='xgboost_smote')
clf.build(X_train, y_train)
clf.tune_threshold(X_val, y_val, metric='f1')

# Evaluate
metrics = clf.evaluate(X_test, y_test)

# Save for deployment
clf.save('imbalanced_classifier.joblib')

# Load in production
clf_loaded = ImbalancedClassifier.load('imbalanced_classifier.joblib')
predictions = clf_loaded.predict(X_new)
```

---

## Complete Implementations

### End-to-End Imbalanced Classification

```python
"""
Complete workflow for imbalanced classification.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create imbalanced dataset
print("Step 1: Creating imbalanced dataset...")
X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_classes=2,
    weights=[0.95, 0.05],  # 5% minority class
    random_state=42
)

print(f"Class distribution: {Counter(y)}")

# 2. Train-test split (stratified)
print("\nStep 2: Splitting data (stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# 3. Scale features
print("\nStep 3: Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 4. Compare resampling techniques
print("\nStep 4: Comparing resampling techniques...")

techniques = {
    'No Resampling': (X_train_scaled, y_train),
    'SMOTE': SMOTE(random_state=42).fit_resample(X_train_scaled, y_train),
    'ADASYN': ADASYN(random_state=42).fit_resample(X_train_scaled, y_train),
    'SMOTE-Tomek': SMOTETomek(random_state=42).fit_resample(X_train_scaled, y_train)
}

results = []

for name, (X_res, y_res) in techniques.items():
    print(f"\nTraining with {name}...")

    # Train XGBoost
    if name == 'No Resampling':
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        clf = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
    else:
        clf = XGBClassifier(random_state=42)

    clf.fit(X_res, y_res)

    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    y_proba = clf.predict_proba(X_test_scaled)[:, 1]

    from sklearn.metrics import precision_score, recall_score

    results.append({
        'Technique': name,
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC-ROC': roc_auc_score(y_test, y_proba),
        'AUC-PR': average_precision_score(y_test, y_proba)
    })

# 5. Print results
print("\n" + "=" * 70)
print("RESULTS COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
print(results_df.to_string(index=False))

# 6. Train final model with best approach
print("\n" + "=" * 70)
print("FINAL MODEL (XGBoost + SMOTE)")
print("=" * 70)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_final, y_train_final = smote.fit_resample(X_train_scaled, y_train)

# Train
final_clf = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42,
    eval_metric='aucpr'
)

final_clf.fit(X_train_final, y_train_final)

# Tune threshold
y_val_proba = final_clf.predict_proba(X_val_scaled)[:, 1]
optimal_threshold = find_optimal_threshold(y_val, y_val_proba, metric='f1')

# Final evaluation
y_test_proba = final_clf.predict_proba(X_test_scaled)[:, 1]
y_test_pred = (y_test_proba >= optimal_threshold).astype(int)

print("\nFinal Evaluation:")
metrics = evaluate_imbalanced_classifier(y_test, y_test_pred, y_test_proba)

# 7. Save model
print("\nSaving model...")
joblib.dump({
    'scaler': scaler,
    'classifier': final_clf,
    'optimal_threshold': optimal_threshold
}, 'final_imbalanced_model.joblib')

print("\nWorkflow complete!")
```

---

## Summary

This comprehensive guide covered state-of-the-art techniques for handling imbalanced data:

**Key Techniques (2025 SOTA):**

1. **ADASYN:** 99.67% accuracy (adaptive synthetic sampling)
2. **XGBoost + SMOTE:** Highest F1 scores in benchmarks
3. **Focal Loss:** Best for deep learning on imbalanced data
4. **Class Weighting:** Simplest effective approach

**Key Findings from Research:**
- **Oversampling > Undersampling** (preserves information)
- **Use precision, recall, F1, AUC-ROC** (NOT accuracy)
- **Hybrid approaches** (SMOTE-Tomek) combine best of both worlds
- **Threshold tuning** critical for optimal performance

**Decision Framework:**

- **Mild imbalance** (<3:1): No special handling or class weights
- **Moderate imbalance** (3:1 to 10:1): Class weights or SMOTE
- **Severe imbalance** (10:1 to 100:1): ADASYN or SMOTE-Tomek
- **Extreme imbalance** (>100:1): Anomaly detection (Isolation Forest)
- **Deep learning:** Use Focal Loss

**Production Best Practices:**
1. Always use stratified splits
2. Apply resampling only to training data
3. Evaluate with multiple metrics (precision, recall, F1, AUC-PR)
4. Tune classification threshold on validation set
5. Use cross-validation with StratifiedKFold
6. Monitor model performance on new data

Master these techniques to build robust classifiers for real-world imbalanced datasets!
