# Cross-Validation: Robust Model Evaluation

## Overview

Cross-validation is a resampling technique used to evaluate model performance and ensure generalization to unseen data. It's essential for reliable model assessment, hyperparameter tuning, and detecting overfitting.

**Key Principle:** Never evaluate your model on data it has seen during training. Cross-validation helps estimate how well your model will perform on independent data.

---

## Table of Contents
1. [Why Cross-Validation?](#why-cross-validation)
2. [K-Fold Cross-Validation](#k-fold-cross-validation)
3. [Stratified K-Fold](#stratified-k-fold)
4. [Time Series Cross-Validation](#time-series-cross-validation)
5. [Leave-One-Out (LOO)](#leave-one-out)
6. [Nested Cross-Validation](#nested-cross-validation)
7. [Group K-Fold](#group-k-fold)
8. [Custom Cross-Validation](#custom-cross-validation)
9. [Statistical Significance Testing](#statistical-significance-testing)
10. [Best Practices](#best-practices)

---

## Why Cross-Validation?

### Problems with Simple Train/Test Split

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Generate dataset
X, y = make_classification(n_samples=100, n_features=20, random_state=42)

# Single train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)

print(f"Test accuracy: {score:.4f}")

# Problem: This score depends heavily on the random split!
# Different random_state → different score
```

**Issues:**
1. **High Variance:** Performance estimate depends on the specific split
2. **Data Inefficiency:** Some data only used for training, some only for testing
3. **Risk of Unlucky Split:** Test set might be easier/harder than representative

### Solution: Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# K-fold cross-validation (5 folds)
scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
```

**Benefits:**
- More robust performance estimate
- Better use of available data
- Quantifies uncertainty (standard deviation)
- Reduces risk of overfitting to validation set

---

## K-Fold Cross-Validation

### Concept

1. Split data into K equal-sized folds
2. For each fold k:
   - Use fold k as validation set
   - Use remaining K-1 folds as training set
   - Train model and evaluate
3. Average results across all K folds

```
Fold 1: [Val][Train][Train][Train][Train]
Fold 2: [Train][Val][Train][Train][Train]
Fold 3: [Train][Train][Val][Train][Train]
Fold 4: [Train][Train][Train][Val][Train]
Fold 5: [Train][Train][Train][Train][Val]
```

### Implementation

```python
from sklearn.model_selection import KFold, cross_validate
import matplotlib.pyplot as plt

def kfold_cross_validation(model, X, y, k=5, scoring='accuracy', random_state=42):
    """
    Perform K-fold cross-validation with detailed reporting.

    Parameters:
    -----------
    model : estimator
        scikit-learn compatible model
    X : array-like
        Features
    y : array-like
        Target
    k : int
        Number of folds
    scoring : str or list
        Metric(s) to evaluate
    """
    # Define K-fold splitter
    kfold = KFold(n_splits=k, shuffle=True, random_state=random_state)

    # Perform cross-validation
    cv_results = cross_validate(
        model, X, y,
        cv=kfold,
        scoring=scoring,
        return_train_score=True,
        return_estimator=True
    )

    # Print results
    print("=" * 70)
    print(f"K-FOLD CROSS-VALIDATION (K={k})")
    print("=" * 70)

    test_scores = cv_results['test_score']
    train_scores = cv_results['train_score']

    print(f"\nFold Results:")
    print(f"{'Fold':<8} {'Train Score':<15} {'Test Score':<15}")
    print("-" * 70)
    for i, (train, test) in enumerate(zip(train_scores, test_scores), 1):
        print(f"{i:<8} {train:>14.4f} {test:>14.4f}")

    print("-" * 70)
    print(f"{'Mean':<8} {train_scores.mean():>14.4f} {test_scores.mean():>14.4f}")
    print(f"{'Std':<8} {train_scores.std():>14.4f} {test_scores.std():>14.4f}")

    # Check for overfitting
    train_test_gap = train_scores.mean() - test_scores.mean()
    print(f"\nTrain-Test Gap: {train_test_gap:.4f}")
    if train_test_gap > 0.1:
        print("⚠️  Warning: Potential overfitting detected!")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Box plot
    axes[0].boxplot([train_scores, test_scores], labels=['Train', 'Test'])
    axes[0].set_ylabel('Score')
    axes[0].set_title('Score Distribution Across Folds')
    axes[0].grid(alpha=0.3)

    # Line plot
    fold_indices = np.arange(1, k + 1)
    axes[1].plot(fold_indices, train_scores, 'o-', label='Train', linewidth=2)
    axes[1].plot(fold_indices, test_scores, 's-', label='Test', linewidth=2)
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Score per Fold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

    return cv_results

# Example
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

iris = load_iris()
X, y = iris.data, iris.target

model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_results = kfold_cross_validation(model, X, y, k=5)
```

### Choosing K

**Common Values:**
- **K=5:** Good balance, commonly used
- **K=10:** More folds, less bias but higher variance
- **K=n (LOO):** Maximum folds, computationally expensive

**Trade-offs:**
- **Smaller K:** Less computation, higher bias, lower variance
- **Larger K:** More computation, lower bias, higher variance

```python
def compare_k_values(model, X, y, k_values=[3, 5, 10, 20]):
    """
    Compare cross-validation results for different K values.
    """
    results = {}

    for k in k_values:
        scores = cross_val_score(model, X, y, cv=k, scoring='accuracy')
        results[k] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }

    print("=" * 60)
    print("COMPARISON OF K VALUES")
    print("=" * 60)
    print(f"{'K':<10} {'Mean Accuracy':<20} {'Std Dev':<15}")
    print("-" * 60)

    for k, result in results.items():
        print(f"{k:<10} {result['mean']:>19.4f} {result['std']:>14.4f}")

    return results

# Compare
comparison = compare_k_values(model, X, y)
```

---

## Stratified K-Fold

Maintains class distribution in each fold (essential for imbalanced data).

```python
from sklearn.model_selection import StratifiedKFold

def stratified_kfold_cv(model, X, y, k=5):
    """
    Stratified K-fold preserves class distribution.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    # Check class distribution in each fold
    print("=" * 70)
    print("CLASS DISTRIBUTION PER FOLD")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        train_dist = np.bincount(y[train_idx]) / len(train_idx)
        val_dist = np.bincount(y[val_idx]) / len(val_idx)

        print(f"\nFold {fold}:")
        print(f"  Train distribution: {train_dist}")
        print(f"  Val distribution:   {val_dist}")

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    print("\n" + "=" * 70)
    print("STRATIFIED K-FOLD RESULTS")
    print("=" * 70)
    print(f"Mean accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores

# Example with imbalanced data
from sklearn.datasets import make_classification

X_imb, y_imb = make_classification(
    n_samples=1000, n_features=20,
    weights=[0.9, 0.1],  # 90% class 0, 10% class 1
    random_state=42
)

print("Overall class distribution:")
print(f"Class 0: {np.sum(y_imb == 0) / len(y_imb) * 100:.1f}%")
print(f"Class 1: {np.sum(y_imb == 1) / len(y_imb) * 100:.1f}%")

scores_stratified = stratified_kfold_cv(model, X_imb, y_imb, k=5)
```

### Stratified vs. Regular K-Fold

```python
def compare_stratified_vs_regular(model, X, y, k=5):
    """
    Compare stratified and regular K-fold.
    """
    # Regular K-fold
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores_regular = cross_val_score(model, X, y, cv=kf, scoring='f1')

    # Stratified K-fold
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores_stratified = cross_val_score(model, X, y, cv=skf, scoring='f1')

    print("=" * 70)
    print("STRATIFIED VS. REGULAR K-FOLD")
    print("=" * 70)
    print(f"{'Method':<25} {'Mean F1':<15} {'Std Dev':<15}")
    print("-" * 70)
    print(f"{'Regular K-Fold':<25} {scores_regular.mean():>14.4f} {scores_regular.std():>14.4f}")
    print(f"{'Stratified K-Fold':<25} {scores_stratified.mean():>14.4f} {scores_stratified.std():>14.4f}")

    return scores_regular, scores_stratified

# Compare on imbalanced data
comparison = compare_stratified_vs_regular(model, X_imb, y_imb)
```

**When to Use Stratified K-Fold:**
- Imbalanced datasets (always!)
- Multi-class classification
- Small datasets where class distribution matters

---

## Time Series Cross-Validation

**Critical Rule:** Cannot shuffle time series data! Must respect temporal order.

### 1. Rolling Window (Time Series Split)

```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(model, X, y, n_splits=5):
    """
    Time series cross-validation with expanding window.

    Fold 1: [Train------][Test]
    Fold 2: [Train-----------][Test]
    Fold 3: [Train----------------][Test]
    Fold 4: [Train---------------------][Test]
    Fold 5: [Train--------------------------][Test]
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    print("=" * 70)
    print(f"TIME SERIES CROSS-VALIDATION (Expanding Window)")
    print("=" * 70)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        # Train and evaluate
        model.fit(X[train_idx], y[train_idx])
        score = model.score(X[test_idx], y[test_idx])
        scores.append(score)

        print(f"\nFold {fold}:")
        print(f"  Train: {train_idx[0]:5d} to {train_idx[-1]:5d} (n={len(train_idx):5d})")
        print(f"  Test:  {test_idx[0]:5d} to {test_idx[-1]:5d} (n={len(test_idx):5d})")
        print(f"  Score: {score:.4f}")

    scores = np.array(scores)
    print("\n" + "=" * 70)
    print(f"Mean Score: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores

# Example with time series data
from sklearn.datasets import make_regression

X_ts, y_ts = make_regression(n_samples=1000, n_features=10, noise=10, random_state=42)

from sklearn.ensemble import RandomForestRegressor
model_ts = RandomForestRegressor(n_estimators=100, random_state=42)

ts_scores = time_series_cv(model_ts, X_ts, y_ts, n_splits=5)
```

### 2. Rolling Window with Fixed Size

```python
def rolling_window_cv(X, y, train_size, test_size, step=1):
    """
    Rolling window cross-validation with fixed train/test sizes.

    Window 1: [Train][Test]
    Window 2:    [Train][Test]
    Window 3:       [Train][Test]
    """
    n_samples = len(X)
    window_size = train_size + test_size

    if window_size > n_samples:
        raise ValueError("Window size exceeds data length")

    print("=" * 70)
    print("ROLLING WINDOW CROSS-VALIDATION (Fixed Size)")
    print("=" * 70)

    scores = []
    window_num = 1

    for start in range(0, n_samples - window_size + 1, step):
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size

        # Get windows
        X_train_window = X[train_start:train_end]
        y_train_window = y[train_start:train_end]
        X_test_window = X[test_start:test_end]
        y_test_window = y[test_start:test_end]

        # Train and evaluate
        model_ts.fit(X_train_window, y_train_window)
        score = model_ts.score(X_test_window, y_test_window)
        scores.append(score)

        print(f"\nWindow {window_num}:")
        print(f"  Train: {train_start:5d} to {train_end-1:5d}")
        print(f"  Test:  {test_start:5d} to {test_end-1:5d}")
        print(f"  Score: {score:.4f}")

        window_num += 1

    scores = np.array(scores)
    print("\n" + "=" * 70)
    print(f"Mean Score: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores

# Example
rolling_scores = rolling_window_cv(X_ts, y_ts, train_size=500, test_size=100, step=100)
```

### 3. Walk-Forward Validation

```python
def walk_forward_validation(X, y, initial_train_size, test_size):
    """
    Walk-forward validation: retrain after each test period.

    Step 1: [Train------------][Test]
    Step 2: [Train-----------------][Test]
    Step 3: [Train----------------------][Test]
    """
    n_samples = len(X)
    scores = []

    print("=" * 70)
    print("WALK-FORWARD VALIDATION")
    print("=" * 70)

    train_end = initial_train_size
    step_num = 1

    while train_end + test_size <= n_samples:
        test_start = train_end
        test_end = test_start + test_size

        # Get data
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_test = X[test_start:test_end]
        y_test = y[test_start:test_end]

        # Train and evaluate
        model_ts.fit(X_train, y_train)
        score = model_ts.score(X_test, y_test)
        scores.append(score)

        print(f"\nStep {step_num}:")
        print(f"  Train: 0 to {train_end-1:5d} (n={train_end})")
        print(f"  Test:  {test_start:5d} to {test_end-1:5d} (n={test_size})")
        print(f"  Score: {score:.4f}")

        # Move forward
        train_end = test_end
        step_num += 1

    scores = np.array(scores)
    print("\n" + "=" * 70)
    print(f"Mean Score: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores

# Example
wf_scores = walk_forward_validation(X_ts, y_ts, initial_train_size=500, test_size=100)
```

**Time Series CV Best Practices:**
- Always respect temporal order
- Use expanding or rolling window
- Consider seasonality (test on full seasons)
- Account for auto-correlation in standard errors

---

## Leave-One-Out (LOO)

Special case of K-fold where K = n (number of samples).

```python
from sklearn.model_selection import LeaveOneOut

def leave_one_out_cv(model, X, y):
    """
    Leave-One-Out cross-validation.

    Each sample is used once as test set.
    """
    loo = LeaveOneOut()
    n_samples = X.shape[0]

    print("=" * 70)
    print(f"LEAVE-ONE-OUT CROSS-VALIDATION (n={n_samples})")
    print("=" * 70)

    # For large datasets, use cross_val_score
    if n_samples > 100:
        print("Dataset large, using cross_val_score...")
        scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
        print(f"LOO Accuracy: {scores.mean():.4f} (n={len(scores)} iterations)")
    else:
        # For small datasets, show details
        predictions = []
        actuals = []

        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            predictions.append(pred[0])
            actuals.append(y_test[0])

        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(actuals, predictions)

        print(f"LOO Accuracy: {accuracy:.4f}")
        print(f"Errors: {np.sum(np.array(actuals) != np.array(predictions))} / {n_samples}")

    return scores if n_samples > 100 else accuracy

# Example with small dataset
from sklearn.datasets import load_iris

iris = load_iris()
X_small, y_small = iris.data[:50], iris.target[:50]  # Small subset

loo_score = leave_one_out_cv(model, X_small, y_small)
```

**When to Use LOO:**
- Very small datasets (n < 100)
- Maximum use of training data
- Low bias but high variance

**When NOT to Use:**
- Large datasets (computationally expensive: n iterations)
- High variance in estimates
- Correlated samples

---

## Nested Cross-Validation

For unbiased hyperparameter tuning and model selection.

```python
from sklearn.model_selection import GridSearchCV

def nested_cross_validation(X, y, param_grid, outer_cv=5, inner_cv=3):
    """
    Nested cross-validation for hyperparameter tuning.

    Outer loop: Model evaluation (unbiased performance estimate)
    Inner loop: Hyperparameter tuning

    This prevents optimistic bias from tuning on test set.
    """
    from sklearn.ensemble import RandomForestClassifier

    # Outer CV for model evaluation
    outer_cv_splitter = StratifiedKFold(n_splits=outer_cv, shuffle=True, random_state=42)

    # Inner CV for hyperparameter tuning
    inner_cv_splitter = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=42)

    print("=" * 70)
    print(f"NESTED CROSS-VALIDATION")
    print(f"Outer CV: {outer_cv} folds (performance evaluation)")
    print(f"Inner CV: {inner_cv} folds (hyperparameter tuning)")
    print("=" * 70)

    outer_scores = []
    best_params_list = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv_splitter.split(X, y), 1):
        print(f"\n{'='*70}")
        print(f"OUTER FOLD {fold}/{outer_cv}")
        print('='*70)

        # Split data
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        # Inner CV: Hyperparameter tuning
        model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            model,
            param_grid,
            cv=inner_cv_splitter,
            scoring='accuracy',
            n_jobs=-1
        )

        grid_search.fit(X_train_outer, y_train_outer)

        # Best model from inner CV
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_params_list.append(best_params)

        print(f"\nBest parameters (inner CV): {best_params}")
        print(f"Best inner CV score: {grid_search.best_score_:.4f}")

        # Evaluate on outer test set
        outer_score = best_model.score(X_test_outer, y_test_outer)
        outer_scores.append(outer_score)

        print(f"Outer test score: {outer_score:.4f}")

    outer_scores = np.array(outer_scores)

    print("\n" + "=" * 70)
    print("NESTED CV RESULTS")
    print("=" * 70)
    print(f"Outer CV Scores: {outer_scores}")
    print(f"Mean Score: {outer_scores.mean():.4f} ± {outer_scores.std():.4f}")

    print("\n" + "=" * 70)
    print("BEST PARAMETERS PER FOLD")
    print("=" * 70)
    for fold, params in enumerate(best_params_list, 1):
        print(f"Fold {fold}: {params}")

    return outer_scores, best_params_list

# Example
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

nested_scores, best_params = nested_cross_validation(
    X_imb, y_imb, param_grid, outer_cv=5, inner_cv=3
)
```

**Why Nested CV?**
- Regular CV with hyperparameter tuning is optimistically biased
- Nested CV provides unbiased performance estimate
- Essential for comparing different model types

**Structure:**
```
Outer CV (Performance Evaluation)
├── Fold 1
│   └── Inner CV (Hyperparameter Tuning)
│       ├── Fold 1
│       ├── Fold 2
│       └── Fold 3
├── Fold 2
│   └── Inner CV
│       ├── Fold 1
│       ├── Fold 2
│       └── Fold 3
...
```

---

## Group K-Fold

For data with natural groupings (e.g., multiple samples per patient, location, etc.).

```python
from sklearn.model_selection import GroupKFold

def group_kfold_cv(X, y, groups, n_splits=5):
    """
    Group K-Fold ensures samples from same group stay together.

    Use when:
    - Multiple measurements per subject
    - Spatial/temporal clusters
    - Hierarchical data structure
    """
    gkf = GroupKFold(n_splits=n_splits)

    print("=" * 70)
    print(f"GROUP K-FOLD CROSS-VALIDATION (K={n_splits})")
    print("=" * 70)

    scores = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), 1):
        # Check group separation
        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])
        overlap = train_groups.intersection(test_groups)

        print(f"\nFold {fold}:")
        print(f"  Train groups: {sorted(train_groups)}")
        print(f"  Test groups:  {sorted(test_groups)}")
        print(f"  Overlap: {overlap if overlap else 'None (correct!)'}")

        # Train and evaluate
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)

        print(f"  Score: {score:.4f}")

    scores = np.array(scores)
    print("\n" + "=" * 70)
    print(f"Mean Score: {scores.mean():.4f} ± {scores.std():.4f}")

    return scores

# Example: Medical data with multiple measurements per patient
n_samples = 300
n_patients = 30

# Each patient has ~10 measurements
groups = np.repeat(np.arange(n_patients), n_samples // n_patients)

# Generate data
X_grouped, y_grouped = make_classification(
    n_samples=n_samples, n_features=20, random_state=42
)

group_scores = group_kfold_cv(X_grouped, y_grouped, groups, n_splits=5)
```

**When to Use Group K-Fold:**
- Medical data (multiple visits per patient)
- Video data (multiple frames per video)
- Spatial data (multiple measurements per location)
- Any hierarchical/clustered data structure

**Critical:** Prevents data leakage when samples within groups are correlated.

---

## Custom Cross-Validation

Define your own splitting logic.

```python
from sklearn.model_selection import BaseCrossValidator

class StratifiedGroupKFold(BaseCrossValidator):
    """
    Custom CV: Stratified + Grouped K-Fold.

    Maintains both:
    - Class balance (stratified)
    - Group separation (grouped)
    """
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X, y, groups):
        """Generate train/test indices."""
        from sklearn.model_selection import StratifiedKFold

        # Get unique groups
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Calculate group-level targets (majority class)
        group_labels = np.array([
            np.bincount(y[groups == g]).argmax()
            for g in unique_groups
        ])

        # Stratified K-fold on groups
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )

        # Generate splits
        for train_groups_idx, test_groups_idx in skf.split(unique_groups, group_labels):
            train_groups = unique_groups[train_groups_idx]
            test_groups = unique_groups[test_groups_idx]

            # Convert to sample indices
            train_idx = np.where(np.isin(groups, train_groups))[0]
            test_idx = np.where(np.isin(groups, test_groups))[0]

            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Example usage
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X_grouped, y_grouped, cv=sgkf, groups=groups)
print(f"Stratified Group K-Fold Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

### Custom Time Series CV with Gaps

```python
class TimeSeriesGapCV:
    """
    Time series CV with gap between train and test.

    Prevents leakage when there's temporal dependence.

    [Train------][Gap][Test]
    [Train-----------][Gap][Test]
    """
    def __init__(self, n_splits=5, gap=10):
        self.n_splits = n_splits
        self.gap = gap

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        test_size = n_samples // (self.n_splits + 1)

        for i in range(self.n_splits):
            test_start = (i + 1) * test_size
            test_end = test_start + test_size

            # Add gap before test set
            train_end = test_start - self.gap

            if train_end > 0 and test_end <= n_samples:
                train_idx = np.arange(0, train_end)
                test_idx = np.arange(test_start, test_end)

                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Example
tsgap_cv = TimeSeriesGapCV(n_splits=5, gap=20)

scores = cross_val_score(model_ts, X_ts, y_ts, cv=tsgap_cv)
print(f"Time Series Gap CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

## Statistical Significance Testing

### 1. Paired T-Test (Compare Two Models)

```python
from scipy.stats import ttest_rel

def compare_models_ttest(model1, model2, X, y, cv=5):
    """
    Compare two models using paired t-test on CV scores.

    H0: Models have equal performance
    H1: Models have different performance
    """
    kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    # Get CV scores for both models
    scores1 = cross_val_score(model1, X, y, cv=kfold, scoring='accuracy')
    scores2 = cross_val_score(model2, X, y, cv=kfold, scoring='accuracy')

    # Paired t-test
    t_stat, p_value = ttest_rel(scores1, scores2)

    print("=" * 70)
    print("PAIRED T-TEST: MODEL COMPARISON")
    print("=" * 70)
    print(f"Model 1 mean: {scores1.mean():.4f} ± {scores1.std():.4f}")
    print(f"Model 2 mean: {scores2.mean():.4f} ± {scores2.std():.4f}")
    print(f"\nDifference: {(scores1.mean() - scores2.mean()):.4f}")
    print(f"t-statistic: {t_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n✓ Significant difference (p < {alpha})")
        if scores1.mean() > scores2.mean():
            print("  Model 1 is significantly better")
        else:
            print("  Model 2 is significantly better")
    else:
        print(f"\n✗ No significant difference (p >= {alpha})")

    return t_stat, p_value, scores1, scores2

# Example: Compare Random Forest vs. Logistic Regression
from sklearn.linear_model import LogisticRegression

model_rf = RandomForestClassifier(random_state=42)
model_lr = LogisticRegression(random_state=42, max_iter=1000)

t_stat, p_value, scores_rf, scores_lr = compare_models_ttest(
    model_rf, model_lr, X_imb, y_imb, cv=10
)
```

### 2. McNemar's Test (Binary Classification)

```python
from statsmodels.stats.contingency_tables import mcnemar

def mcnemar_test(y_true, y_pred1, y_pred2):
    """
    McNemar's test for comparing two classifiers.

    Tests if disagreements are systematic or random.
    """
    # Create contingency table
    # Rows: Model 1, Cols: Model 2
    # Cells: both correct, M1 correct M2 wrong, M1 wrong M2 correct, both wrong

    both_correct = np.sum((y_pred1 == y_true) & (y_pred2 == y_true))
    m1_correct_m2_wrong = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    m1_wrong_m2_correct = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    both_wrong = np.sum((y_pred1 != y_true) & (y_pred2 != y_true))

    # Contingency table for McNemar
    table = [[both_correct, m1_wrong_m2_correct],
             [m1_correct_m2_wrong, both_wrong]]

    print("=" * 70)
    print("McNEMAR'S TEST")
    print("=" * 70)
    print("\nContingency Table:")
    print(f"                  Model 2 Correct   Model 2 Wrong")
    print(f"Model 1 Correct      {both_correct:8d}        {m1_correct_m2_wrong:8d}")
    print(f"Model 1 Wrong        {m1_wrong_m2_correct:8d}        {both_wrong:8d}")

    # Run McNemar's test
    result = mcnemar(table, exact=True)

    print(f"\nStatistic: {result.statistic:.4f}")
    print(f"p-value: {result.pvalue:.4f}")

    alpha = 0.05
    if result.pvalue < alpha:
        print(f"\n✓ Significant difference (p < {alpha})")
    else:
        print(f"\n✗ No significant difference (p >= {alpha})")

    return result

# Example
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_imb, y_imb, test_size=0.3, random_state=42
)

model_rf.fit(X_train, y_train)
model_lr.fit(X_train, y_train)

y_pred_rf = model_rf.predict(X_test)
y_pred_lr = model_lr.predict(X_test)

mcnemar_result = mcnemar_test(y_test, y_pred_rf, y_pred_lr)
```

### 3. Friedman Test (Multiple Models)

```python
from scipy.stats import friedmanchisquare

def friedman_test(*model_scores_list):
    """
    Friedman test for comparing multiple models.

    Non-parametric alternative to repeated measures ANOVA.
    """
    # Each argument is CV scores for one model
    stat, p_value = friedmanchisquare(*model_scores_list)

    print("=" * 70)
    print("FRIEDMAN TEST (Multiple Models)")
    print("=" * 70)

    for i, scores in enumerate(model_scores_list, 1):
        print(f"Model {i}: {scores.mean():.4f} ± {scores.std():.4f}")

    print(f"\nFriedman statistic: {stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"\n✓ Significant difference among models (p < {alpha})")
        print("Proceed with post-hoc tests (e.g., Nemenyi)")
    else:
        print(f"\n✗ No significant difference (p >= {alpha})")

    return stat, p_value

# Example: Compare 3 models
from sklearn.svm import SVC

model_svm = SVC(random_state=42)

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores_rf = cross_val_score(model_rf, X_imb, y_imb, cv=kfold)
scores_lr = cross_val_score(model_lr, X_imb, y_imb, cv=kfold)
scores_svm = cross_val_score(model_svm, X_imb, y_imb, cv=kfold)

friedman_result = friedman_test(scores_rf, scores_lr, scores_svm)
```

---

## Best Practices

### 1. Always Use Cross-Validation

```python
# ❌ WRONG: Single train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y)
model.fit(X_train, y_train)
score = model.score(X_test, y_test)  # High variance!

# ✅ CORRECT: Cross-validation
scores = cross_val_score(model, X, y, cv=5)
print(f"Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

### 2. Stratify for Classification

```python
# ✅ CORRECT: Stratified for classification
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf)
```

### 3. Respect Temporal Order for Time Series

```python
# ❌ WRONG: Shuffling time series
kfold = KFold(n_splits=5, shuffle=True)  # NO!

# ✅ CORRECT: Time series split
tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_ts, y_ts, cv=tscv)
```

### 4. Use Nested CV for Hyperparameter Tuning

```python
# ❌ WRONG: Tuning on same data used for evaluation
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
score = grid_search.best_score_  # Optimistically biased!

# ✅ CORRECT: Nested CV
outer_scores = nested_cross_validation(X, y, param_grid)
```

### 5. Use Groups When Data Has Structure

```python
# ✅ CORRECT: Group K-fold for hierarchical data
gkf = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf, groups=groups)
```

### 6. Report Confidence Intervals

```python
def report_with_ci(scores, confidence=0.95):
    """
    Report mean with confidence interval.
    """
    from scipy import stats

    mean = scores.mean()
    std_err = scores.std() / np.sqrt(len(scores))
    ci = stats.t.interval(confidence, len(scores)-1, loc=mean, scale=std_err)

    print(f"Mean: {mean:.4f}")
    print(f"95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return mean, ci

scores = cross_val_score(model, X, y, cv=10)
mean, ci = report_with_ci(scores)
```

### 7. Choose Appropriate K

| Dataset Size | Recommended K | Reasoning |
|--------------|---------------|-----------|
| n < 100 | LOO or 5-fold | Maximize training data |
| 100 ≤ n < 1000 | 5-10 fold | Good balance |
| n ≥ 1000 | 5 fold | Efficient, low variance |
| Very large | 3 fold | Computational efficiency |

### 8. Avoid Data Leakage

```python
# ❌ WRONG: Preprocessing before split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Leakage!
scores = cross_val_score(model, X_scaled, y, cv=5)

# ✅ CORRECT: Preprocessing within CV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', model)
])
scores = cross_val_score(pipeline, X, y, cv=5)
```

---

## Summary

### Quick Reference Table

| Scenario | CV Method | Key Parameters |
|----------|-----------|----------------|
| General classification | `StratifiedKFold` | `n_splits=5`, `shuffle=True` |
| General regression | `KFold` | `n_splits=5`, `shuffle=True` |
| Time series | `TimeSeriesSplit` | `n_splits=5` |
| Grouped data | `GroupKFold` | `n_splits=5` |
| Small datasets | `LeaveOneOut` | - |
| Hyperparameter tuning | Nested CV | `outer_cv=5`, `inner_cv=3` |
| Imbalanced data | `StratifiedKFold` | Always stratify! |

### Key Takeaways

1. **Always cross-validate** - never trust a single train/test split
2. **Stratify for classification** - maintains class distribution
3. **Respect temporal order** - for time series data
4. **Use nested CV** - for unbiased hyperparameter tuning
5. **Consider groups** - when data has hierarchical structure
6. **Test significance** - compare models statistically
7. **Report uncertainty** - mean ± std or confidence intervals
8. **Avoid leakage** - preprocess within CV folds

---

**Last Updated:** 2025-10-14
**Status:** Complete - Production-ready implementations with best practices
