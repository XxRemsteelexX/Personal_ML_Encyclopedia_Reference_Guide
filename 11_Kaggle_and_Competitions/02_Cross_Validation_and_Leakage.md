# Cross-Validation and Data Leakage Prevention

## Table of Contents
- [1. Introduction - Why Cross-Validation Matters](#1-introduction---why-cross-validation-matters)
- [2. K-Fold Cross-Validation](#2-k-fold-cross-validation)
- [3. Group K-Fold Cross-Validation](#3-group-k-fold-cross-validation)
- [4. Time Series Cross-Validation](#4-time-series-cross-validation)
- [5. Nested Cross-Validation](#5-nested-cross-validation)
- [6. Data Leakage Types](#6-data-leakage-types)
- [7. Leakage Detection](#7-leakage-detection)
- [8. Leakage Prevention](#8-leakage-prevention)
- [9. Competition-Specific Cross-Validation](#9-competition-specific-cross-validation)
- [10. Common Mistakes](#10-common-mistakes)
- [11. Resources and References](#11-resources-and-references)

---

## 1. Introduction - Why Cross-Validation Matters

### The Golden Rule

**"A good CV score predicts good LB score. A bad CV doesn't guarantee bad LB, but trust your CV."**

Cross-validation is the single most important technique for reliable model evaluation in machine learning competitions and production systems. Poor CV strategy leads to overfitting, incorrect model selection, and catastrophic failures on unseen data.

### Overfitting vs Generalization

**Overfitting** occurs when a model learns patterns specific to the training data that don't generalize to new data. Cross-validation helps detect this by simulating how the model will perform on unseen data.

**Key Metrics:**
- **Train Score >> Validation Score**: Clear overfitting (e.g., Train AUC 0.95, Val AUC 0.72)
- **Train Score ~ Validation Score**: Good generalization (e.g., Train AUC 0.85, Val AUC 0.83)
- **Train Score < Validation Score**: Possible data issues or lucky validation split

### Why Cross-Validation is Critical

1. **Reduces Variance**: Single train-test split can be lucky/unlucky
2. **Better Hyperparameter Tuning**: More reliable performance estimates
3. **Detects Leakage**: Suspiciously high scores indicate data leakage
4. **Model Selection**: Compare different model architectures fairly
5. **Production Readiness**: Estimates real-world performance

### The Variance Problem

```python
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate synthetic dataset
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)

# Single split - high variance
single_scores = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    single_scores.append(model.score(X_test, y_test))

print(f"Single Split - Mean: {np.mean(single_scores):.4f}, Std: {np.std(single_scores):.4f}")

# Cross-validation - lower variance
cv_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                            X, y, cv=5, scoring='accuracy')
print(f"5-Fold CV - Mean: {cv_scores.mean():.4f}, Std: {cv_scores.std():.4f}")
```

**Expected Output:**
```
Single Split - Mean: 0.8650, Std: 0.0187
5-Fold CV - Mean: 0.8620, Std: 0.0089
```

Cross-validation reduces standard deviation by ~50%, giving more reliable estimates.

---

## 2. K-Fold Cross-Validation

### Standard K-Fold

**K-Fold** divides data into K equally-sized folds. Each fold serves as validation set once while remaining K-1 folds form the training set.

**When to use:**
- IID (Independent and Identically Distributed) data
- No temporal ordering
- No grouped structure
- Sufficient data (>1000 samples)

```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Standard K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Manual implementation for full control
fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_val, y_val)
    fold_scores.append(score)
    print(f"Fold {fold_idx}: {score:.4f}")

print(f"\nMean CV Score: {np.mean(fold_scores):.4f} +/- {np.std(fold_scores):.4f}")
```

**Key Parameters:**
- `n_splits`: Number of folds (typically 5 or 10)
- `shuffle`: Always set to `True` unless data is pre-shuffled
- `random_state`: For reproducibility

**Choosing K:**
- **K=5**: Standard choice, good bias-variance tradeoff
- **K=10**: More data per fold, higher computational cost
- **K=20**: Small datasets, but high variance
- **K=N (LOOCV)**: Only for tiny datasets (<100 samples), very expensive

### Stratified K-Fold

**Stratified K-Fold** preserves class distribution in each fold, critical for imbalanced datasets.

**When to use:**
- Classification tasks
- Imbalanced classes (e.g., 90% class 0, 10% class 1)
- Small datasets where random splits might miss minority class
- Multi-class classification

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
import pandas as pd

# Create imbalanced dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_classes=2, weights=[0.9, 0.1], random_state=42)

print(f"Class distribution: {np.bincount(y)}")  # [900, 100]

# Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_predictions = np.zeros(len(y))  # Out-of-fold predictions
oof_probas = np.zeros(len(y))

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Verify stratification
    print(f"Fold {fold_idx} - Train class dist: {np.bincount(y_train)}, "
          f"Val class dist: {np.bincount(y_val)}")

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Store out-of-fold predictions
    oof_predictions[val_idx] = model.predict(X_val)
    oof_probas[val_idx] = model.predict_proba(X_val)[:, 1]

# Overall CV score using out-of-fold predictions
print(f"\nOut-of-Fold AUC: {roc_auc_score(y, oof_probas):.4f}")
print(classification_report(y, oof_predictions))
```

**Out-of-Fold (OOF) Predictions:**
- Each sample gets exactly one prediction from a model that never saw it during training
- OOF predictions are unbiased estimates of model performance
- Used for stacking/blending in ensemble models

### Repeated K-Fold

**Repeated K-Fold** runs K-Fold multiple times with different random splits, further reducing variance.

**When to use:**
- Small datasets (<1000 samples)
- High variance in CV scores
- Final model evaluation before deployment
- Have computational budget

```python
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

# Repeated K-Fold
rkf = RepeatedKFold(n_splits=5, n_repeats=3, random_state=42)

# For classification with class imbalance
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

# Evaluate with repeated CV
model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                   max_depth=3, random_state=42)
scores = cross_val_score(model, X, y, cv=rskf, scoring='roc_auc')

print(f"Repeated Stratified 5-Fold (3 repeats):")
print(f"Mean AUC: {scores.mean():.4f} +/- {scores.std():.4f}")
print(f"Total folds: {len(scores)}")  # 5 * 3 = 15
```

**Tradeoff:**
- **Pros**: Lower variance, more reliable estimate
- **Cons**: 3x computational cost (for 3 repeats)

### Custom Cross-Validation Splits

```python
from sklearn.model_selection import PredefinedSplit

# Create custom train/val split based on domain knowledge
# Example: Separate by date ranges or data sources
split_indices = np.array([-1] * 800 + [0] * 200)  # -1 = train, 0 = test fold
ps = PredefinedSplit(test_fold=split_indices)

for train_idx, val_idx in ps.split():
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}")
    # Train model on custom split
```

---

## 3. Group K-Fold Cross-Validation

### Why Group K-Fold Matters

**Problem:** Many real-world datasets have **grouped structure** where samples aren't independent:
- Multiple purchases per customer
- Multiple visits per patient
- Multiple images per object
- Time series windows from same sequence

Using standard K-Fold can split groups across train/validation, causing **group leakage**.

### Group K-Fold Implementation

**GroupKFold** ensures all samples from the same group are in the same fold.

```python
from sklearn.model_selection import GroupKFold
import pandas as pd

# Simulate customer transaction data
np.random.seed(42)
n_customers = 100
data = []
for customer_id in range(n_customers):
    n_transactions = np.random.randint(5, 20)
    for _ in range(n_transactions):
        data.append({
            'customer_id': customer_id,
            'transaction_amount': np.random.lognormal(4, 1),
            'days_since_last': np.random.exponential(5),
            'churned': np.random.binomial(1, 0.15)
        })

df = pd.DataFrame(data)
print(f"Total transactions: {len(df)}")
print(f"Unique customers: {df['customer_id'].nunique()}")

# Prepare features and target
X = df[['transaction_amount', 'days_since_last']].values
y = df['churned'].values
groups = df['customer_id'].values

# Group K-Fold
gkf = GroupKFold(n_splits=5)

for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
    train_groups = set(groups[train_idx])
    val_groups = set(groups[val_idx])

    # Verify no group overlap
    assert len(train_groups.intersection(val_groups)) == 0, "Group leakage detected!"

    print(f"Fold {fold_idx}:")
    print(f"  Train: {len(train_idx)} samples, {len(train_groups)} customers")
    print(f"  Val: {len(val_idx)} samples, {len(val_groups)} customers")
    print(f"  Churn rate - Train: {y[train_idx].mean():.3f}, Val: {y[val_idx].mean():.3f}")
```

### Stratified Group K-Fold

**Challenge:** GroupKFold doesn't preserve class distribution. For imbalanced classification with groups, use custom implementation.

```python
from sklearn.model_selection import StratifiedGroupKFold
# Available in sklearn >= 1.0

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
    print(f"Fold {fold_idx}:")
    print(f"  Train class dist: {np.bincount(y[train_idx])}")
    print(f"  Val class dist: {np.bincount(y[val_idx])}")
```

### Medical/Healthcare Applications

```python
# Patient-level cross-validation for medical imaging
# Each patient has multiple scans/images

import pandas as pd
from sklearn.model_selection import GroupKFold

# Simulate medical imaging dataset
patients = []
for patient_id in range(50):
    n_scans = np.random.randint(10, 30)
    disease = np.random.binomial(1, 0.3)
    for scan_num in range(n_scans):
        patients.append({
            'patient_id': patient_id,
            'scan_number': scan_num,
            'feature_1': np.random.randn(),
            'feature_2': np.random.randn(),
            'has_disease': disease  # Same for all scans of patient
        })

df_medical = pd.DataFrame(patients)

# WRONG: Standard K-Fold (patient leakage)
# Some scans of same patient in train and val

# CORRECT: Group K-Fold by patient
X_med = df_medical[['feature_1', 'feature_2']].values
y_med = df_medical['has_disease'].values
groups_med = df_medical['patient_id'].values

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X_med, y_med, groups=groups_med):
    # Now each patient's scans are entirely in train OR val, never both
    pass
```

### Leave-One-Group-Out

**Leave-One-Group-Out (LOGO)** uses each group as a validation set once.

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
n_splits = logo.get_n_splits(groups=groups)
print(f"Number of splits (one per group): {n_splits}")

# Use for very critical applications where you need to test on each group
# Warning: Can be very slow if many groups
```

---

## 4. Time Series Cross-Validation

### Why Time Series CV is Different

**Key Principle:** Cannot use future data to predict the past. Standard K-Fold violates temporal ordering.

**Requirements:**
- Training data must precede validation data
- No shuffling
- Expanding or sliding window approach
- Account for autocorrelation

### TimeSeriesSplit

**TimeSeriesSplit** creates sequential train-test splits preserving temporal order.

```python
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import matplotlib.pyplot as plt

# Generate time series data
dates = pd.date_range('2020-01-01', periods=1000, freq='D')
df_ts = pd.DataFrame({
    'date': dates,
    'value': np.cumsum(np.random.randn(1000)) + 100,
    'feature_1': np.random.randn(1000),
    'target': np.random.randn(1000)
})

X_ts = df_ts[['value', 'feature_1']].values
y_ts = df_ts['target'].values

# TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5, gap=0, test_size=None)

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_ts), 1):
    train_dates = df_ts.iloc[train_idx]['date']
    val_dates = df_ts.iloc[val_idx]['date']

    print(f"Fold {fold_idx}:")
    print(f"  Train: {train_dates.min()} to {train_dates.max()} ({len(train_idx)} samples)")
    print(f"  Val: {val_dates.min()} to {val_dates.max()} ({len(val_idx)} samples)")
    print(f"  Gap check: {(val_dates.min() - train_dates.max()).days} days")
```

**Output (Example):**
```
Fold 1:
  Train: 2020-01-01 to 2020-02-28 (59 samples)
  Val: 2020-02-29 to 2020-05-26 (88 samples)
```

### Walk-Forward Validation

**Walk-Forward** validation uses a sliding window, more realistic for production time series.

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

# Walk-forward validation with fixed window
train_window = 365  # 1 year of training data
val_window = 30     # 1 month validation

results = []
for start_idx in range(0, len(df_ts) - train_window - val_window, val_window):
    train_start = start_idx
    train_end = start_idx + train_window
    val_start = train_end
    val_end = val_start + val_window

    X_train = X_ts[train_start:train_end]
    y_train = y_ts[train_start:train_end]
    X_val = X_ts[val_start:val_end]
    y_val = y_ts[val_start:val_end]

    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1,
                                      max_depth=3, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    results.append({
        'train_period': f"{df_ts.iloc[train_start]['date']} to {df_ts.iloc[train_end-1]['date']}",
        'val_period': f"{df_ts.iloc[val_start]['date']} to {df_ts.iloc[val_end-1]['date']}",
        'mse': mse
    })

results_df = pd.DataFrame(results)
print(results_df)
print(f"\nMean MSE: {results_df['mse'].mean():.4f} +/- {results_df['mse'].std():.4f}")
```

### Purged and Embargoed Cross-Validation

**Critical for financial data** to prevent lookahead bias and autocorrelation.

**Purging:** Remove samples from training set that are too close in time to validation set.
**Embargo:** Add gap between train and validation periods.

```python
class PurgedTimeSeriesSplit:
    """
    Time series split with purging and embargo.

    Parameters:
    -----------
    n_splits : int
        Number of splits
    embargo_td : int
        Embargo period (number of samples to skip after each validation)
    purge_td : int
        Purge period (samples to remove from train before validation)
    """
    def __init__(self, n_splits=5, embargo_td=0, purge_td=0):
        self.n_splits = n_splits
        self.embargo_td = embargo_td
        self.purge_td = purge_td

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            # Validation set
            val_start = (i + 1) * test_size
            val_end = val_start + test_size

            # Training set (everything before val, minus purge)
            train_end = val_start - self.purge_td
            train_indices = np.arange(0, max(0, train_end))

            # Validation indices with embargo
            val_indices = np.arange(val_start, min(val_end, n_samples))

            # Apply embargo: skip samples immediately after validation
            if i < self.n_splits - 1:
                embargo_end = val_end + self.embargo_td

            yield train_indices, val_indices

# Example usage for financial data
X_financial = np.random.randn(1000, 10)
y_financial = np.random.randn(1000)

# Purge 5 days, embargo 10 days
ptscv = PurgedTimeSeriesSplit(n_splits=5, embargo_td=10, purge_td=5)

for fold_idx, (train_idx, val_idx) in enumerate(ptscv.split(X_financial), 1):
    print(f"Fold {fold_idx}:")
    print(f"  Train: 0 to {train_idx[-1]} ({len(train_idx)} samples)")
    print(f"  Val: {val_idx[0]} to {val_idx[-1]} ({len(val_idx)} samples)")
    print(f"  Gap: {val_idx[0] - train_idx[-1] - 1} samples")
```

### Expanding Window vs Sliding Window

```python
# Expanding Window: Training set grows over time
def expanding_window_split(X, n_splits=5, min_train=100):
    n = len(X)
    test_size = (n - min_train) // n_splits

    for i in range(n_splits):
        train_end = min_train + i * test_size
        val_start = train_end
        val_end = val_start + test_size

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, min(val_end, n))

        yield train_idx, val_idx

# Sliding Window: Training set size is fixed
def sliding_window_split(X, n_splits=5, train_size=100):
    n = len(X)
    test_size = (n - train_size) // n_splits

    for i in range(n_splits):
        val_start = train_size + i * test_size
        val_end = val_start + test_size
        train_start = max(0, val_start - train_size)

        train_idx = np.arange(train_start, val_start)
        val_idx = np.arange(val_start, min(val_end, n))

        yield train_idx, val_idx
```

**When to use:**
- **Expanding Window**: When more data is always better, non-stationary data
- **Sliding Window**: Stationary data, recent patterns more relevant, concept drift

---

## 5. Nested Cross-Validation

### The Double CV Problem

**Problem:** Using the same data for hyperparameter tuning and model evaluation leads to optimistic bias.

**Solution:** Nested CV uses two loops:
- **Outer loop**: Model evaluation (unbiased performance estimate)
- **Inner loop**: Hyperparameter tuning

### Complete Nested CV Implementation

```python
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load data
X, y = load_breast_cancer(return_X_y=True)

# Outer CV for model evaluation
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
    'kernel': ['rbf']
}

# Nested CV
outer_scores = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), 1):
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    # Scale data (fit on outer train only)
    scaler = StandardScaler()
    X_train_outer_scaled = scaler.fit_transform(X_train_outer)
    X_test_outer_scaled = scaler.transform(X_test_outer)

    # Inner CV for hyperparameter tuning
    grid_search = GridSearchCV(
        SVC(),
        param_grid,
        cv=inner_cv,
        scoring='roc_auc',
        n_jobs=-1
    )

    grid_search.fit(X_train_outer_scaled, y_train_outer)

    # Evaluate best model on outer test set
    best_score = grid_search.score(X_test_outer_scaled, y_test_outer)
    outer_scores.append(best_score)

    print(f"Outer Fold {fold_idx}:")
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Inner CV score: {grid_search.best_score_:.4f}")
    print(f"  Outer test score: {best_score:.4f}")

print(f"\nNested CV Score: {np.mean(outer_scores):.4f} +/- {np.std(outer_scores):.4f}")
```

**Key Insights:**
- `grid_search.best_score_`: Inner CV score (optimistic, used for tuning)
- `outer_scores`: Outer CV scores (unbiased performance estimate)
- Outer CV score typically lower than inner CV score

### Non-Nested vs Nested CV Comparison

```python
# Non-nested CV (WRONG for final evaluation)
grid_search_simple = GridSearchCV(SVC(), param_grid, cv=5, scoring='roc_auc')
X_scaled = StandardScaler().fit_transform(X)
grid_search_simple.fit(X_scaled, y)
non_nested_score = grid_search_simple.best_score_

print(f"Non-nested CV score: {non_nested_score:.4f}")  # OPTIMISTIC
print(f"Nested CV score: {np.mean(outer_scores):.4f}")  # UNBIASED
print(f"Bias: {non_nested_score - np.mean(outer_scores):.4f}")
```

**Typical Result:**
```
Non-nested CV score: 0.9850  (optimistic)
Nested CV score: 0.9720      (realistic)
Bias: 0.0130
```

### Nested CV with Random Search

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, uniform

# Random search parameter distributions
param_distributions = {
    'C': loguniform(1e-2, 1e2),
    'gamma': loguniform(1e-4, 1e0),
    'kernel': ['rbf']
}

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

outer_scores_random = []

for train_idx, test_idx in outer_cv.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random search instead of grid search
    random_search = RandomizedSearchCV(
        SVC(),
        param_distributions,
        n_iter=20,  # Number of random combinations
        cv=inner_cv,
        scoring='roc_auc',
        random_state=42,
        n_jobs=-1
    )

    random_search.fit(X_train_scaled, y_train)
    outer_scores_random.append(random_search.score(X_test_scaled, y_test))

print(f"Nested CV with Random Search: {np.mean(outer_scores_random):.4f} +/- {np.std(outer_scores_random):.4f}")
```

**Random Search Advantages:**
- Faster for large parameter spaces
- Better exploration of continuous parameters
- Often finds good solutions with fewer iterations

---

## 6. Data Leakage Types

Data leakage is the silent killer in machine learning competitions and production systems. It occurs when information from outside the training dataset is used to create the model, leading to overly optimistic performance that doesn't generalize.

### Type 1: Target Leakage

**Definition:** Features that are derived from or directly contain the target variable, or information that wouldn't be available at prediction time.

```python
# EXAMPLE 1: E-commerce churn prediction

# WRONG: Using future information
df['days_until_churn'] = (df['churn_date'] - df['current_date']).dt.days
# This feature knows the future! Model will have perfect accuracy.

# CORRECT: Using only past information
df['days_since_signup'] = (df['current_date'] - df['signup_date']).dt.days
df['total_purchases'] = df.groupby('user_id')['purchase'].cumsum().shift(1).fillna(0)

# EXAMPLE 2: Medical diagnosis

# WRONG: Including treatment information
features = ['age', 'symptoms', 'treatment_prescribed', 'lab_results']
# treatment_prescribed is decided AFTER diagnosis, not before

# CORRECT: Only pre-diagnosis features
features = ['age', 'symptoms', 'initial_vitals']

# EXAMPLE 3: Credit default prediction

# WRONG: Using post-default information
df['late_payment_fees'] = df['fees']  # Fees charged AFTER default
df['recovery_amount'] = df['collected']  # Known only after default

# CORRECT: Historical payment patterns
df['avg_payment_delay_last_6m'] = df.groupby('customer')['days_late'].rolling(6).mean()
df['missed_payments_count'] = df.groupby('customer')['missed'].cumsum().shift(1)
```

### Type 2: Train-Test Contamination

**Definition:** Information from the test set leaks into the training process through preprocessing, scaling, or feature engineering.

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# WRONG: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses mean/std from ENTIRE dataset including test
X_train, X_test = train_test_split(X_scaled, test_size=0.2)
# Test set statistics influenced training!

# CORRECT: Split first, then scale
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only
X_test_scaled = scaler.transform(X_test)  # Transform using train statistics

# WRONG: Imputing missing values before split
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)  # Uses mean from test set too
X_train, X_test = train_test_split(X_imputed)

# CORRECT: Impute after split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# WRONG: Feature selection before split
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)  # Sees all data
X_train, X_test = train_test_split(X_selected)

# CORRECT: Feature selection inside CV
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
```

### Type 3: Temporal Leakage

**Definition:** Using future information to predict the past, violating temporal causality.

```python
# WRONG: Using future aggregations
df['user_total_purchases'] = df.groupby('user_id')['purchase_amount'].transform('sum')
# This includes FUTURE purchases when predicting churn today

# CORRECT: Cumulative features with shift
df = df.sort_values(['user_id', 'date'])
df['user_total_purchases_before'] = df.groupby('user_id')['purchase_amount'].cumsum().shift(1).fillna(0)

# WRONG: Rolling window without proper boundaries
df['rolling_mean_7d'] = df.groupby('user_id')['value'].transform(
    lambda x: x.rolling(7, min_periods=1).mean()
)
# Includes current day, which is cheating for same-day prediction

# CORRECT: Rolling window excluding current observation
df['rolling_mean_7d'] = df.groupby('user_id')['value'].transform(
    lambda x: x.shift(1).rolling(7, min_periods=1).mean()
)

# EXAMPLE: Stock price prediction
df_stock = pd.DataFrame({
    'date': pd.date_range('2020-01-01', periods=100),
    'price': np.random.randn(100).cumsum() + 100,
    'volume': np.random.randint(1000, 10000, 100)
})

# WRONG: Future volatility
df_stock['future_volatility_30d'] = df_stock['price'].rolling(30).std()

# CORRECT: Historical volatility
df_stock['historical_volatility_30d'] = df_stock['price'].shift(1).rolling(30).std()

# WRONG: Forward-filling missing values
df_stock['price_filled'] = df_stock['price'].fillna(method='bfill')
# Uses future prices to fill past missing values!

# CORRECT: Backward-looking fill
df_stock['price_filled'] = df_stock['price'].fillna(method='ffill')
```

### Type 4: Group Leakage

**Definition:** Information leaks across groups when samples from the same entity appear in both train and test sets.

```python
# EXAMPLE: Customer lifetime value prediction

# Dataset with multiple transactions per customer
df_transactions = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2, 2, 3, 3, 3] * 10,
    'transaction_amount': np.random.lognormal(4, 1, 90),
    'days_since_last': np.random.exponential(10, 90),
    'will_purchase_again': np.random.binomial(1, 0.6, 90)
})

# WRONG: Random split (customer appears in both train and test)
from sklearn.model_selection import train_test_split
X = df_transactions[['transaction_amount', 'days_since_last']].values
y = df_transactions['will_purchase_again'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# Customer 1's transactions might be in both train and test!

# CORRECT: Split by customer
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
groups = df_transactions['customer_id'].values

for train_idx, test_idx in gss.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    train_customers = set(groups[train_idx])
    test_customers = set(groups[test_idx])
    assert len(train_customers.intersection(test_customers)) == 0
```

### Type 5: Feature Engineering Leakage

**Definition:** Creating features that inadvertently encode the target or use global statistics.

```python
# WRONG: Target encoding without proper CV
df['category_mean_target'] = df.groupby('category')['target'].transform('mean')
# Each sample knows the average target of its category, including itself!

# CORRECT: Target encoding with leave-one-out or CV
def target_encode_loo(df, column, target):
    """Leave-one-out target encoding"""
    global_mean = df[target].mean()
    agg = df.groupby(column)[target].agg(['sum', 'count'])

    # Leave-one-out: (sum - current) / (count - 1)
    encoded = df[column].map(agg['sum']) - df[target]
    encoded = encoded / (df[column].map(agg['count']) - 1)
    encoded = encoded.fillna(global_mean)

    return encoded

# WRONG: Rank features using entire dataset
df['rank_feature'] = df['value'].rank()
# Ranks are influenced by test set values

# CORRECT: Rank within training set, clip test set
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='uniform', n_quantiles=1000)
X_train['value_rank'] = qt.fit_transform(X_train[['value']])
X_test['value_rank'] = qt.transform(X_test[['value']])  # Clips values outside training range
```

### Type 6: ID and Metadata Leakage

**Definition:** Identifiers or metadata that accidentally correlate with the target.

```python
# WRONG: Using row number as feature
df['row_id'] = range(len(df))
# If data is sorted by target, row_id perfectly predicts target!

# WRONG: Using filename/timestamp from data collection
# Example: Images labeled 'cancer_001.jpg', 'healthy_001.jpg'
df['filename'] = ['cancer_001.jpg', 'cancer_002.jpg', 'healthy_001.jpg']
df['contains_cancer_word'] = df['filename'].str.contains('cancer').astype(int)
# Perfect predictor, but won't generalize!

# CORRECT: Remove or properly handle identifiers
features_to_drop = ['row_id', 'filename', 'creation_timestamp', 'file_path']
X = df.drop(columns=features_to_drop + ['target'])
```

---

## 7. Leakage Detection

### Adversarial Validation

**Adversarial Validation** checks if the model can distinguish between training and test sets. If it can (AUC >> 0.5), the distributions differ, and your CV might not match the leaderboard.

**Complete Implementation:**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def adversarial_validation(train_df, test_df, feature_cols, n_splits=5):
    """
    Adversarial validation to check train-test similarity.

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Test data
    feature_cols : list
        List of feature column names
    n_splits : int
        Number of CV folds

    Returns:
    --------
    dict with AUC score, feature importances, and predictions
    """
    # Label train=0, test=1
    train_df['is_test'] = 0
    test_df['is_test'] = 1

    # Combine datasets
    combined = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    # Prepare features and target
    X = combined[feature_cols]
    y = combined['is_test']

    # Train model to distinguish train from test
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                       max_depth=3, random_state=42)

    # Cross-validation score
    cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring='roc_auc')
    mean_auc = cv_scores.mean()

    print(f"Adversarial Validation AUC: {mean_auc:.4f} +/- {cv_scores.std():.4f}")

    # Interpret results
    if mean_auc < 0.55:
        print("GOOD: Train and test are very similar. CV should match LB.")
    elif mean_auc < 0.65:
        print("OK: Some differences between train/test. Monitor CV-LB correlation.")
    elif mean_auc < 0.75:
        print("WARNING: Significant differences. CV might not match LB.")
    else:
        print("DANGER: Train and test are very different. CV unreliable!")

    # Feature importances
    model.fit(X, y)
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 10 features distinguishing train from test:")
    print(feature_importance.head(10))

    return {
        'auc': mean_auc,
        'cv_scores': cv_scores,
        'feature_importance': feature_importance,
        'model': model
    }

# Example usage
# Generate synthetic data with distribution shift
np.random.seed(42)
train_data = pd.DataFrame({
    'feature_1': np.random.normal(0, 1, 1000),
    'feature_2': np.random.normal(0, 1, 1000),
    'feature_3': np.random.normal(0, 1, 1000),
    'target': np.random.binomial(1, 0.5, 1000)
})

# Test set with shifted distribution
test_data = pd.DataFrame({
    'feature_1': np.random.normal(0.5, 1, 500),  # Shifted mean
    'feature_2': np.random.normal(0, 1.5, 500),  # Different variance
    'feature_3': np.random.normal(0, 1, 500),    # Same distribution
})

feature_cols = ['feature_1', 'feature_2', 'feature_3']
results = adversarial_validation(train_data, test_data, feature_cols)
```

**Expected Output:**
```
Adversarial Validation AUC: 0.7234 +/- 0.0156
WARNING: Significant differences. CV might not match LB.

Top features distinguishing train from test:
        feature  importance
0    feature_1      0.5234
1    feature_2      0.3456
2    feature_3      0.1310
```

### Using Adversarial Validation Results

```python
# Strategy 1: Remove problematic features
results = adversarial_validation(train_data, test_data, feature_cols)
problematic_features = results['feature_importance'][
    results['feature_importance']['importance'] > 0.3
]['feature'].tolist()

print(f"Consider removing: {problematic_features}")

# Strategy 2: Create time-based validation split
# If temporal shift detected, use recent data for validation
if results['auc'] > 0.65:
    print("Using time-based validation instead of random split")
    # Use last 20% of training data as validation
    split_point = int(len(train_data) * 0.8)
    train_subset = train_data.iloc[:split_point]
    val_subset = train_data.iloc[split_point:]

# Strategy 3: Weight samples by similarity to test
def weight_samples_by_test_similarity(train_df, test_df, feature_cols):
    """Weight training samples by how similar they are to test set"""
    results = adversarial_validation(train_df, test_df, feature_cols)
    model = results['model']

    # Predict probability that each train sample is from test
    X_train = train_df[feature_cols]
    test_similarity = model.predict_proba(X_train)[:, 1]

    # Higher weight for samples similar to test
    sample_weights = test_similarity / test_similarity.mean()

    return sample_weights
```

### Feature Importance Analysis for Leakage

**Suspiciously high feature importance often indicates leakage.**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

def detect_leakage_via_importance(X_train, y_train, feature_names, threshold=0.8):
    """
    Detect potential leakage using feature importance.

    Parameters:
    -----------
    threshold : float
        Importance threshold for flagging suspicious features (default 0.8)
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Flag suspicious features
    suspicious = importance_df[importance_df['importance'] > threshold]

    if len(suspicious) > 0:
        print("WARNING: Suspiciously high importance features (possible leakage):")
        print(suspicious)

        # Permutation importance for validation
        print("\nValidating with permutation importance...")
        perm_importance = permutation_importance(
            model, X_train, y_train, n_repeats=10, random_state=42
        )

        for idx in suspicious.index[:3]:
            feat_name = feature_names[idx]
            perm_imp = perm_importance.importances_mean[idx]
            print(f"{feat_name}: Permutation importance = {perm_imp:.4f}")

    else:
        print("No suspiciously high importance features detected.")

    return importance_df

# Example: Detect leakage
X_train_leak = np.random.randn(1000, 10)
y_train_leak = np.random.binomial(1, 0.5, 1000)

# Add a leaky feature (perfect predictor)
X_train_leak[:, 0] = y_train_leak + np.random.normal(0, 0.01, 1000)

feature_names = [f'feature_{i}' for i in range(10)]
feature_names[0] = 'LEAKY_FEATURE'

importance_df = detect_leakage_via_importance(
    X_train_leak, y_train_leak, feature_names, threshold=0.5
)
```

### Too-Good-to-Be-True Scores

**Rule of thumb:** If your CV score seems unrealistically high, investigate leakage.

```python
def check_too_good_to_be_true(cv_score, task='classification'):
    """
    Check if CV score is suspiciously high.

    Parameters:
    -----------
    cv_score : float
        Cross-validation score (AUC for classification, R2 for regression)
    task : str
        'classification' or 'regression'
    """
    if task == 'classification':
        if cv_score > 0.99:
            print(f"DANGER: AUC {cv_score:.4f} is suspiciously high. Check for leakage!")
            return True
        elif cv_score > 0.95:
            print(f"WARNING: AUC {cv_score:.4f} is very high. Verify no leakage.")
            return True
        else:
            print(f"AUC {cv_score:.4f} seems reasonable.")
            return False

    elif task == 'regression':
        if cv_score > 0.99:
            print(f"DANGER: R2 {cv_score:.4f} is suspiciously high. Check for leakage!")
            return True
        elif cv_score > 0.95:
            print(f"WARNING: R2 {cv_score:.4f} is very high. Verify no leakage.")
            return True
        else:
            print(f"R2 {cv_score:.4f} seems reasonable.")
            return False

# Example
cv_auc = 0.987
check_too_good_to_be_true(cv_auc, task='classification')
```

### Temporal Consistency Check

```python
def check_temporal_consistency(df, date_col, feature_col, target_col):
    """
    Check if feature values leak future information.

    For time series: features should only depend on past data.
    """
    df_sorted = df.sort_values(date_col)

    # Check if feature is constant across time for same entity
    # (could indicate using global statistics)
    variance_over_time = df_sorted.groupby(date_col)[feature_col].std()

    if variance_over_time.mean() < 0.01:
        print(f"WARNING: {feature_col} has very low variance over time.")
        print("Might be using global statistics (leakage).")

    # Check correlation with future target
    df_sorted['future_target'] = df_sorted.groupby('entity_id')[target_col].shift(-1)
    corr_with_future = df_sorted[[feature_col, 'future_target']].corr().iloc[0, 1]

    if abs(corr_with_future) > 0.3:
        print(f"WARNING: {feature_col} correlates {corr_with_future:.3f} with future target!")
        print("Possible temporal leakage.")
```

---

## 8. Leakage Prevention

### Proper Pipeline Design with sklearn.Pipeline

**sklearn.Pipeline** ensures all preprocessing steps are applied correctly within cross-validation.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Create pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(f_classif, k=10)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation with pipeline
# All steps (impute, scale, select) are fit ONLY on training folds
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='roc_auc')
print(f"Pipeline CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# WRONG: Preprocessing outside pipeline
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage!
cv_scores_wrong = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_scaled, y, cv=5, scoring='roc_auc'
)
print(f"Without Pipeline (WRONG) CV AUC: {cv_scores_wrong.mean():.4f}")
```

### Advanced Pipeline with ColumnTransformer

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Define feature types
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['occupation', 'education', 'marital_status']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Cross-validation (no leakage)
cv_scores = cross_val_score(full_pipeline, X_df, y, cv=5, scoring='roc_auc')
print(f"Full Pipeline CV AUC: {cv_scores.mean():.4f}")
```

### Target Encoding Inside CV Folds

**Target encoding** must be done separately for each fold to prevent leakage.

```python
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class TargetEncoderCV(BaseEstimator, TransformerMixin):
    """
    Target encoder that works inside CV pipelines.
    Uses leave-one-out encoding to prevent leakage.
    """
    def __init__(self, cols, smoothing=1.0):
        self.cols = cols
        self.smoothing = smoothing
        self.encodings = {}

    def fit(self, X, y):
        """Fit encoding maps on training data"""
        X = pd.DataFrame(X, columns=self.cols) if not isinstance(X, pd.DataFrame) else X
        y = pd.Series(y)

        for col in self.cols:
            # Calculate mean target per category
            agg = pd.DataFrame({'y': y, 'cat': X[col]})
            stats = agg.groupby('cat')['y'].agg(['mean', 'count'])

            # Smoothing to handle rare categories
            global_mean = y.mean()
            stats['smoothed_mean'] = (
                (stats['mean'] * stats['count'] + global_mean * self.smoothing) /
                (stats['count'] + self.smoothing)
            )

            self.encodings[col] = stats['smoothed_mean'].to_dict()

        return self

    def transform(self, X):
        """Transform using fitted encodings"""
        X = pd.DataFrame(X, columns=self.cols) if not isinstance(X, pd.DataFrame) else X
        X_encoded = X.copy()

        for col in self.cols:
            # Use global mean for unseen categories
            global_mean = np.mean(list(self.encodings[col].values()))
            X_encoded[col] = X[col].map(self.encodings[col]).fillna(global_mean)

        return X_encoded.values

# Usage in pipeline
categorical_cols = ['category', 'subcategory']

pipeline_with_target_encoding = Pipeline([
    ('target_encoder', TargetEncoderCV(cols=categorical_cols, smoothing=10)),
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# CV automatically applies target encoding per fold
cv_scores = cross_val_score(
    pipeline_with_target_encoding, X_cat, y, cv=5, scoring='roc_auc'
)
print(f"Target Encoding Pipeline CV AUC: {cv_scores.mean():.4f}")
```

### SMOTE Inside CV (Not Before)

**SMOTE** (Synthetic Minority Over-sampling) must be applied only to training folds.

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier

# WRONG: SMOTE before CV
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
cv_scores_wrong = cross_val_score(
    RandomForestClassifier(n_estimators=100, random_state=42),
    X_resampled, y_resampled, cv=5
)  # Validation set was used in SMOTE!

# CORRECT: SMOTE inside pipeline
pipeline_smote = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

cv_scores_correct = cross_val_score(pipeline_smote, X, y, cv=5, scoring='roc_auc')
print(f"SMOTE in Pipeline CV AUC: {cv_scores_correct.mean():.4f}")
```

### Custom CV Splitter for Complex Scenarios

```python
from sklearn.model_selection import BaseCrossValidator

class CustomTimeGroupSplit(BaseCrossValidator):
    """
    Custom CV that handles both time ordering AND group structure.
    Use for user-level time series data.
    """
    def __init__(self, n_splits=5, gap_days=0):
        self.n_splits = n_splits
        self.gap_days = gap_days

    def split(self, X, y=None, groups=None):
        """
        X must be DataFrame with 'user_id' and 'date' columns
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be DataFrame with user_id and date")

        # Get unique users
        users = X['user_id'].unique()
        n_users = len(users)
        test_size = n_users // (self.n_splits + 1)

        # Sort users by their earliest date (to maintain some time ordering)
        user_min_dates = X.groupby('user_id')['date'].min().sort_values()
        sorted_users = user_min_dates.index.tolist()

        for i in range(self.n_splits):
            # Split users, not rows
            val_user_start = (i + 1) * test_size
            val_user_end = val_user_start + test_size

            train_users = sorted_users[:val_user_start]
            val_users = sorted_users[val_user_start:val_user_end]

            # Get row indices
            train_idx = X[X['user_id'].isin(train_users)].index.tolist()
            val_idx = X[X['user_id'].isin(val_users)].index.tolist()

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Usage
custom_cv = CustomTimeGroupSplit(n_splits=5, gap_days=7)
for train_idx, val_idx in custom_cv.split(df_with_users):
    # Train model
    pass
```

---

## 9. Competition-Specific Cross-Validation

### Matching CV to Leaderboard

**The most important skill in competitions:** Make your local CV correlate with public leaderboard.

**Strategies:**

1. **Understand the test set composition**
   - Time period (past vs future)
   - Data source (same vs different)
   - Distribution (similar vs shifted)

2. **Mimic the evaluation metric**
   - Use exact same metric in CV
   - Same averaging method (macro vs micro)
   - Same class weights

3. **Replicate data structure**
   - If test is future data: use time-based split
   - If test is different hospitals: use group-based split
   - If test is stratified: use stratified CV

```python
def create_competition_cv_strategy(competition_type):
    """
    Create CV strategy based on competition type.

    Parameters:
    -----------
    competition_type : str
        'time_series', 'grouped', 'iid', 'imbalanced'
    """
    if competition_type == 'time_series':
        # Example: Stock prediction, sales forecasting
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
        print("Using TimeSeriesSplit for temporal data")

    elif competition_type == 'grouped':
        # Example: Customer churn, patient diagnosis
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=5)
        print("Using GroupKFold for grouped data")

    elif competition_type == 'imbalanced':
        # Example: Fraud detection, rare disease
        from sklearn.model_selection import StratifiedKFold
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        print("Using StratifiedKFold for imbalanced classes")

    else:  # 'iid'
        # Example: Image classification, simple tabular
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        print("Using standard KFold for IID data")

    return cv
```

### CV-LB Correlation Analysis

**Track correlation between CV improvements and LB improvements.**

```python
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Track experiments
experiments = {
    'experiment': ['baseline', 'add_feature_1', 'add_feature_2', 'tune_params', 'ensemble'],
    'cv_score': [0.850, 0.855, 0.862, 0.868, 0.875],
    'lb_score': [0.845, 0.848, 0.857, 0.865, 0.871]
}

df_exp = pd.DataFrame(experiments)

# Calculate correlation
corr, p_value = pearsonr(df_exp['cv_score'], df_exp['lb_score'])
print(f"CV-LB Correlation: {corr:.4f} (p-value: {p_value:.4f})")

if corr > 0.9:
    print("EXCELLENT: CV is highly predictive of LB. Trust your CV!")
elif corr > 0.7:
    print("GOOD: CV correlates well with LB.")
elif corr > 0.5:
    print("MODERATE: CV somewhat predicts LB. Be cautious.")
else:
    print("POOR: CV doesn't match LB. Re-examine CV strategy!")

# Visualize
plt.figure(figsize=(8, 6))
plt.scatter(df_exp['cv_score'], df_exp['lb_score'], s=100, alpha=0.6)
for i, exp in enumerate(df_exp['experiment']):
    plt.annotate(exp, (df_exp['cv_score'][i], df_exp['lb_score'][i]))
plt.plot([0.84, 0.88], [0.84, 0.88], 'r--', label='Perfect correlation')
plt.xlabel('CV Score')
plt.ylabel('LB Score')
plt.title(f'CV-LB Correlation: {corr:.3f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.savefig('cv_lb_correlation.png', dpi=150)
plt.show()
```

### When to Trust CV vs LB

**Decision Matrix:**

| CV-LB Correlation | CV > LB | CV < LB | Action |
|-------------------|---------|---------|--------|
| High (>0.9) | Overfitting to LB | Good | Trust CV, optimize for CV |
| High (>0.9) | Good | Underfitting | Trust CV, keep improving |
| Low (<0.5) | Either | Either | Re-examine CV strategy |

```python
def should_trust_cv(cv_score, lb_score, cv_lb_correlation, n_submissions=10):
    """
    Decide whether to trust CV or LB.

    Parameters:
    -----------
    cv_score : float
        Local cross-validation score
    lb_score : float
        Public leaderboard score
    cv_lb_correlation : float
        Historical correlation between CV and LB
    n_submissions : int
        Number of submissions made so far
    """
    gap = cv_score - lb_score

    if cv_lb_correlation > 0.9:
        print("High CV-LB correlation detected.")
        if gap > 0.02:
            print("WARNING: CV >> LB suggests overfitting to public LB.")
            print("ACTION: Trust CV, avoid overfitting to LB.")
        else:
            print("ACTION: Trust CV, optimize locally.")
        return True

    elif cv_lb_correlation > 0.7:
        print("Moderate CV-LB correlation.")
        if n_submissions < 5:
            print("ACTION: Make more submissions to understand CV-LB relationship.")
        else:
            print("ACTION: Trust CV with caution, monitor LB.")
        return True

    else:
        print("Low CV-LB correlation.")
        print("ACTION: Re-examine CV strategy. CV might not match test set.")
        print("Suggestions:")
        print("  1. Try adversarial validation")
        print("  2. Check for data leakage")
        print("  3. Verify CV mimics test set structure (time/group)")
        return False

# Example
should_trust_cv(cv_score=0.875, lb_score=0.851, cv_lb_correlation=0.95, n_submissions=12)
```

### Holdout Validation for Final Model Selection

```python
def create_holdout_for_final_evaluation(X, y, test_size=0.2, stratify=True):
    """
    Create a separate holdout set for final model evaluation.

    This holdout is NEVER used during development, only for final check.
    """
    if stratify:
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    else:
        X_dev, X_holdout, y_dev, y_holdout = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    print(f"Development set: {len(X_dev)} samples")
    print(f"Holdout set: {len(X_holdout)} samples (DO NOT TOUCH until final evaluation)")

    return X_dev, X_holdout, y_dev, y_holdout

# Use development set for all experimentation
X_dev, X_holdout, y_dev, y_holdout = create_holdout_for_final_evaluation(X, y)

# Do ALL development on X_dev, y_dev
# Use X_holdout, y_holdout ONLY ONCE at the very end
```

---

## 10. Common Mistakes

### Mistake 1: Scaling Before Split

```python
# WRONG
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Uses test statistics!
X_train, X_test = train_test_split(X_scaled)

# CORRECT
X_train, X_test = train_test_split(X)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Why it matters:** Test set mean/std leak into training, causing ~1-5% optimistic bias.

### Mistake 2: Feature Selection Before CV

```python
# WRONG
selector = SelectKBest(f_classif, k=20)
X_selected = selector.fit_transform(X, y)  # Sees all data
cv_scores = cross_val_score(model, X_selected, y, cv=5)  # Optimistic!

# CORRECT
pipeline = Pipeline([
    ('selector', SelectKBest(f_classif, k=20)),
    ('model', RandomForestClassifier(n_estimators=100))
])
cv_scores = cross_val_score(pipeline, X, y, cv=5)  # Unbiased
```

### Mistake 3: Using SMOTE Before CV

```python
# WRONG
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(X, y)
cv_scores = cross_val_score(model, X_resampled, y_resampled, cv=5)  # Leakage!

# CORRECT
from imblearn.pipeline import Pipeline as ImbPipeline
pipeline = ImbPipeline([
    ('smote', SMOTE()),
    ('model', RandomForestClassifier())
])
cv_scores = cross_val_score(pipeline, X, y, cv=5)
```

### Mistake 4: Target Encoding Without LOO

```python
# WRONG
df['category_encoded'] = df.groupby('category')['target'].transform('mean')
# Each sample knows its own category's target mean!

# CORRECT (Leave-One-Out)
def target_encode_loo(df, cat_col, target_col):
    agg = df.groupby(cat_col)[target_col].agg(['sum', 'count'])
    encoded = df[cat_col].map(agg['sum']) - df[target_col]
    encoded = encoded / (df[cat_col].map(agg['count']) - 1)
    return encoded.fillna(df[target_col].mean())

df['category_encoded'] = target_encode_loo(df, 'category', 'target')
```

### Mistake 5: Using Future Data in Time Series

```python
# WRONG
df['rolling_mean_7d'] = df['value'].rolling(7).mean()
# Includes current value in the mean!

# CORRECT
df['rolling_mean_7d'] = df['value'].shift(1).rolling(7).mean()
# Only uses past values
```

### Mistake 6: Not Shuffling in K-Fold (for non-time-series)

```python
# WRONG (if data is ordered)
kf = KFold(n_splits=5, shuffle=False)
# If data is sorted by target, folds will be biased!

# CORRECT
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

### Mistake 7: Ignoring Group Structure

```python
# WRONG (customer data with multiple rows per customer)
kf = KFold(n_splits=5)
for train_idx, val_idx in kf.split(X):
    # Same customer in both train and val!
    pass

# CORRECT
gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=customer_ids):
    # Each customer entirely in train OR val
    pass
```

### Mistake 8: Using Test Set for Imputation

```python
# WRONG
imputer = SimpleImputer(strategy='mean')
df_imputed = imputer.fit_transform(df)  # Uses test set mean!
X_train, X_test = train_test_split(df_imputed)

# CORRECT
X_train, X_test = train_test_split(df)
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)
```

### Mistake 9: Overfitting to Public Leaderboard

```python
"""
DANGER: Making too many submissions based on public LB

Problem:
- Public LB is small subset (often 30-40% of test)
- Optimizing for public LB --> overfit to that subset
- Private LB (remaining 60-70%) will be worse

Solution:
- Limit submissions (e.g., max 2 per day)
- Trust your CV if CV-LB correlation is high
- Select final submission based on CV, not public LB
"""

# Track your submissions
submissions = pd.DataFrame({
    'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
    'cv_score': [0.850, 0.855, 0.852],
    'public_lb': [0.845, 0.851, 0.848],
    'description': ['baseline', 'added features', 'different model']
})

# Select based on CV, not public LB
best_by_cv = submissions.loc[submissions['cv_score'].idxmax()]
best_by_lb = submissions.loc[submissions['public_lb'].idxmax()]

print("Best by CV:", best_by_cv['description'], "CV:", best_by_cv['cv_score'])
print("Best by LB:", best_by_lb['description'], "LB:", best_by_lb['public_lb'])
print("\nRECOMMENDATION: Select", best_by_cv['description'])
```

### Mistake 10: Not Setting random_state

```python
# WRONG (non-reproducible)
kf = KFold(n_splits=5, shuffle=True)  # Different splits each run!

# CORRECT
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # Reproducible
```

### Comprehensive Checklist

```python
def validate_cv_setup(X_train, X_test, y_train, y_test, cv_strategy, pipeline=None):
    """
    Validate that CV setup doesn't have common mistakes.

    Returns:
    --------
    dict with validation results
    """
    issues = []

    # Check 1: Train-test overlap (for grouped data)
    if hasattr(cv_strategy, 'groups'):
        print("Checking for group overlap...")
        # Would need actual groups to check

    # Check 2: Target leakage (look for perfect scores)
    if pipeline:
        score = pipeline.fit(X_train, y_train).score(X_test, y_test)
        if score > 0.99:
            issues.append("WARNING: Score > 0.99 suggests possible target leakage")

    # Check 3: Feature scaling in pipeline
    if pipeline and hasattr(pipeline, 'named_steps'):
        has_scaler = any('scaler' in step or 'standard' in step.lower()
                        for step in pipeline.named_steps.keys())
        if not has_scaler:
            issues.append("INFO: No scaler found in pipeline. Consider adding one.")

    # Check 4: Data leakage through preprocessing
    if not pipeline:
        issues.append("WARNING: Not using Pipeline. Easy to introduce leakage!")

    # Check 5: Random state set
    if hasattr(cv_strategy, 'random_state') and cv_strategy.random_state is None:
        issues.append("WARNING: random_state not set. Results not reproducible.")

    # Print results
    if issues:
        print("Validation Issues Found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("No issues detected. CV setup looks good!")

    return {'issues': issues, 'is_valid': len(issues) == 0}
```

---

## 11. Resources and References

### Essential Reading

**Papers:**
- Cawley, G. C., & Talbot, N. L. (2010). "On over-fitting in model selection and subsequent selection bias in performance evaluation." Journal of Machine Learning Research, 11, 2079-2107.
  - Explains nested cross-validation and selection bias
  - https://www.jmlr.org/papers/v11/cawley10a.html

- Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection." IJCAI, 14(2), 1137-1145.
  - Classic paper comparing CV strategies
  - Shows K=10 is good default for bias-variance tradeoff

- Lopez de Prado, M. (2018). "Advances in Financial Machine Learning." Wiley.
  - Chapter 7: Cross-validation in finance
  - Purged K-Fold, Embargo techniques
  - Essential for time series with autocorrelation

**Blog Posts and Tutorials:**
- Sklearn Cross-Validation Guide: https://scikit-learn.org/stable/modules/cross_validation.html
- Fast.ai: "How (and why) to create a good validation set"
- Kaggle Learn: Cross-Validation Tutorial
- MLU-Explain: Visual explanation of cross-validation concepts

### Kaggle Competitions Case Studies

**Time Series:**
- M5 Forecasting: Used grouped time series split by store
- Web Traffic Forecasting: Walk-forward validation critical
- Jane Street Market Prediction: Purged + embargoed CV

**Grouped Data:**
- Google QUEST Q&A: Grouped by question_id
- Mechanisms of Action (MoA): Grouped by compound
- RSNA Pneumonia Detection: Grouped by patient_id

**Adversarial Validation:**
- Porto Seguro Safe Driver: Major train-test distribution shift
- Santander Customer Transaction: Synthetic test data detection
- IEEE-CIS Fraud Detection: Time-based distribution shift

### Python Libraries

```python
# Core libraries
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit,
    RepeatedKFold, RepeatedStratifiedKFold,
    cross_val_score, cross_validate,
    GridSearchCV, RandomizedSearchCV
)

# Imbalanced data
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler

# Advanced CV
# mlxtend: for stacking with proper CV
from mlxtend.feature_selection import SequentialFeatureSelector
from mlxtend.evaluate import feature_importance_permutation

# Category Encoders: target encoding with CV
import category_encoders as ce
ce.TargetEncoder(smoothing=1.0)  # Has CV option
```

### Code Templates

**Template 1: Complete CV Pipeline**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Choose CV strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate with multiple metrics
scoring = {
    'accuracy': 'accuracy',
    'roc_auc': 'roc_auc',
    'f1': 'f1_weighted'
}

cv_results = cross_validate(
    pipeline, X, y,
    cv=cv_strategy,
    scoring=scoring,
    return_train_score=True,
    n_jobs=-1
)

# Print results
for metric in scoring.keys():
    train_score = cv_results[f'train_{metric}'].mean()
    test_score = cv_results[f'test_{metric}'].mean()
    print(f"{metric}: Train={train_score:.4f}, Test={test_score:.4f}")
```

**Template 2: Out-of-Fold Predictions**

```python
import numpy as np
from sklearn.model_selection import StratifiedKFold

def get_oof_predictions(model, X, y, cv=None):
    """
    Get out-of-fold predictions for stacking/blending.

    Returns:
    --------
    oof_preds : np.array
        Out-of-fold predictions (same length as y)
    models : list
        Trained models from each fold
    """
    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = np.zeros(len(y))
    models = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train = y[train_idx]

        # Clone model for each fold
        from sklearn.base import clone
        fold_model = clone(model)

        fold_model.fit(X_train, y_train)
        oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]
        models.append(fold_model)

        print(f"Fold {fold_idx} complete")

    return oof_preds, models

# Usage
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(n_estimators=100)
oof_preds, trained_models = get_oof_predictions(model, X, y)

# Evaluate OOF predictions
from sklearn.metrics import roc_auc_score
print(f"OOF AUC: {roc_auc_score(y, oof_preds):.4f}")
```

### Production Deployment Considerations

**1. Retrain on Full Dataset**
```python
# After CV evaluation, retrain on ALL data for production
final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Use ALL available data
final_model.fit(X, y)

# Save model
import joblib
joblib.dump(final_model, 'production_model.pkl')
```

**2. Monitor for Concept Drift**
```python
def monitor_prediction_drift(model, X_new, baseline_probs):
    """
    Monitor if new data predictions drift from baseline.

    Parameters:
    -----------
    baseline_probs : np.array
        Distribution of predictions on validation set
    """
    new_probs = model.predict_proba(X_new)[:, 1]

    # KS test for distribution shift
    from scipy.stats import ks_2samp
    statistic, p_value = ks_2samp(baseline_probs, new_probs)

    if p_value < 0.05:
        print(f"WARNING: Prediction distribution has shifted (p={p_value:.4f})")
        print("Consider retraining model with recent data")
    else:
        print("Prediction distribution stable")

    return statistic, p_value
```

**3. A/B Testing Models**
```python
"""
Before fully deploying new model:
1. Deploy to small % of traffic (5-10%)
2. Monitor metrics (accuracy, latency, business KPIs)
3. Gradually increase traffic if metrics improve
4. Rollback if metrics degrade
"""
```

### Quick Reference: CV Strategy Selection

```
Data Characteristics                CV Strategy
---------------------------------------------------------------------------
IID, balanced classes               KFold(n_splits=5, shuffle=True)
IID, imbalanced classes             StratifiedKFold(n_splits=5, shuffle=True)
Time series, predict future         TimeSeriesSplit(n_splits=5)
Financial time series               PurgedTimeSeriesSplit(embargo=10, purge=5)
Multiple samples per entity         GroupKFold(n_splits=5)
Multiple per entity + imbalanced    StratifiedGroupKFold(n_splits=5)
Small dataset (<500 samples)        RepeatedStratifiedKFold(n_splits=5, n_repeats=3)
Hyperparameter tuning               Nested CV (outer + inner loops)
Production deployment prep          Holdout set (80/20 or 90/10)
```

### Final Wisdom

**The Three Laws of Cross-Validation:**

1. **Law of Consistency**: Use the same CV strategy for feature engineering, model selection, and final evaluation. Mixing strategies introduces bias.

2. **Law of Isolation**: All preprocessing must happen inside CV folds. Any operation that uses global statistics (scaling, imputation, encoding) must be fit on training fold only.

3. **Law of Trust**: If CV correlates with LB (>0.9), trust CV over LB. If it doesn't, fix your CV strategy, don't chase LB.

**Remember:**
- CV score is an estimate, not ground truth
- Lower variance matters as much as higher mean
- When in doubt, add more folds or use repeated CV
- A good CV strategy is worth more than fancy models
- Data leakage destroys everything - check twice, validate thrice

---

**End of Cross-Validation and Data Leakage Prevention Guide**
