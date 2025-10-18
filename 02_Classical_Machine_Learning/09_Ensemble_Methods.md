# Ensemble Methods

## Table of Contents
1. [Introduction](#introduction)
2. [Bagging and Bootstrap](#bagging-and-bootstrap)
3. [Random Forests](#random-forests)
4. [Gradient Boosting Fundamentals](#gradient-boosting-fundamentals)
5. [XGBoost](#xgboost)
6. [LightGBM](#lightgbm)
7. [CatBoost](#catboost)
8. [2025 Benchmarks](#2025-benchmarks)
9. [Stacking and Blending](#stacking-and-blending)
10. [Hyperparameter Optimization](#hyperparameter-optimization)
11. [When to Use Each Method](#when-to-use-each-method)

## Introduction

Ensemble methods combine multiple models to achieve superior performance compared to individual models. They dominate tabular data competitions and production systems in 2025.

### Core Principle

**Wisdom of Crowds**: Aggregate predictions from multiple models to reduce error.

```
Ensemble prediction = f(model₁, model₂, ..., modelₙ)
```

### Three Main Strategies

1. **Bagging (Bootstrap Aggregating)**
   - Train models independently on different subsets
   - Reduce variance through averaging
   - Example: Random Forest

2. **Boosting**
   - Train models sequentially, each correcting previous errors
   - Reduce bias through adaptive learning
   - Example: XGBoost, LightGBM, CatBoost

3. **Stacking**
   - Train meta-model on predictions of base models
   - Learn optimal combination strategy
   - Example: Multi-level stacking

### Why Ensembles Work

**Bias-Variance Decomposition**:
```
Expected Error = Bias² + Variance + Irreducible Error
```

- **Bagging**: Reduces variance (individual trees have high variance)
- **Boosting**: Reduces bias (sequential correction of errors)
- **Stacking**: Optimizes both through diverse base models

**Diversity is Key**: Models must make different errors to benefit from ensembling.

## Bagging and Bootstrap

### Bootstrap Sampling

**Definition**: Sampling with replacement from dataset.

Given dataset D with n samples:
1. Draw n samples with replacement → Bootstrap sample D*
2. Approximately 63.2% unique samples (1 - 1/e)
3. Remaining 36.8% are out-of-bag (OOB) samples

**Mathematical Insight**:
```
P(sample not selected in one draw) = (n-1)/n
P(sample not selected in n draws) = ((n-1)/n)^n → 1/e ≈ 0.368 as n→∞
```

### Bagging Algorithm

```
For b = 1 to B:
  1. Create bootstrap sample D_b from training data
  2. Train model f_b on D_b

Prediction:
  Regression: ŷ = (1/B) Σ f_b(x)
  Classification: ŷ = majority vote of {f_b(x)}
```

### Out-of-Bag (OOB) Error Estimation

**Key Insight**: OOB samples serve as validation set without explicit split.

For each sample i:
- Find models that didn't see sample i in training
- Average their predictions
- Compute error

**OOB Error ≈ Cross-Validation Error** but computed in single training run.

### Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

class BaggingRegressor:
    """Bagging ensemble with OOB error estimation."""

    def __init__(self, base_model, n_estimators=100, random_state=None):
        self.base_model = base_model
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.models = []
        self.oob_indices = []

    def fit(self, X, y):
        """Fit bagging ensemble."""
        np.random.seed(self.random_state)
        n_samples = len(X)

        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            oob_idx = np.setdiff1d(np.arange(n_samples), indices)

            # Train model
            model = self._clone_model(self.base_model)
            model.fit(X[indices], y[indices])

            self.models.append(model)
            self.oob_indices.append(oob_idx)

        return self

    def predict(self, X):
        """Average predictions from all models."""
        predictions = np.array([model.predict(X) for model in self.models])
        return predictions.mean(axis=0)

    def oob_score(self, X, y):
        """Compute out-of-bag R² score."""
        n_samples = len(X)
        oob_predictions = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for model, oob_idx in zip(self.models, self.oob_indices):
            if len(oob_idx) > 0:
                oob_predictions[oob_idx] += model.predict(X[oob_idx])
                oob_counts[oob_idx] += 1

        # Only use samples that appeared in at least one OOB set
        valid_idx = oob_counts > 0
        oob_predictions[valid_idx] /= oob_counts[valid_idx]

        # Compute R²
        ss_res = np.sum((y[valid_idx] - oob_predictions[valid_idx]) ** 2)
        ss_tot = np.sum((y[valid_idx] - y[valid_idx].mean()) ** 2)
        return 1 - ss_res / ss_tot

    def _clone_model(self, model):
        """Create a copy of the base model."""
        from sklearn.base import clone
        return clone(model)

# Example usage
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Single decision tree (high variance)
single_tree = DecisionTreeRegressor(max_depth=10, random_state=42)
single_tree.fit(X_train, y_train)

# Bagging ensemble
bagging = BaggingRegressor(
    DecisionTreeRegressor(max_depth=10),
    n_estimators=100,
    random_state=42
)
bagging.fit(X_train, y_train)

print("Bagging Results:")
print(f"Single Tree Test R²: {single_tree.score(X_test, y_test):.4f}")
print(f"Bagging Test R²: {bagging.predict(X_test).shape}")
bagging_score = 1 - mean_squared_error(y_test, bagging.predict(X_test)) / y_test.var()
print(f"Bagging Test R²: {bagging_score:.4f}")
print(f"Bagging OOB R²: {bagging.oob_score(X_train, y_train):.4f}")
```

## Random Forests

### Algorithm

Random Forest = Bagging + Random Feature Subsampling

**Key Modifications**:
1. Bootstrap samples (like bagging)
2. **Random feature subset at each split**: Consider only √p features (regression) or p features (classification)
3. Grow deep trees (low bias, high variance)
4. No pruning

**Why Feature Subsampling?**
- Decorrelates trees (increases diversity)
- Especially important when few features are strong predictors
- Without it, all trees would split on same features

### Mathematical Insight

**Variance of Average**:
```
Var(Average) = ρσ²/B + (1-ρ)σ²/B

Where:
- ρ: correlation between trees
- σ²: variance of individual trees
- B: number of trees
```

As B → ∞:
- Uncorrelated trees (ρ=0): Variance → 0
- Perfectly correlated (ρ=1): Variance = σ² (no benefit)

Feature subsampling reduces ρ, maximizing variance reduction.

### Feature Importance

**Mean Decrease in Impurity (MDI)**:
- For each feature, sum impurity reduction across all splits
- Biased toward high-cardinality features
- Fast to compute

**Permutation Importance**:
- Shuffle feature values, measure decrease in performance
- Unbiased but slower
- **Recommended in 2025**

### Implementation with sklearn

```python
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import numpy as np

# Generate classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=10,
    n_redundant=5,
    random_state=42
)

X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    max_features='sqrt',  # √p features per split
    min_samples_split=10,
    min_samples_leaf=5,
    bootstrap=True,
    oob_score=True,  # Compute OOB error
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

rf.fit(X_train, y_train)

print("Random Forest Results:")
print(f"Training Accuracy: {rf.score(X_train, y_train):.4f}")
print(f"OOB Accuracy: {rf.oob_score_:.4f}")
print(f"Test Accuracy: {rf.score(X_test, y_test):.4f}")
print(f"Number of trees: {len(rf.estimators_)}")

# Feature Importance Comparison
# 1. MDI (built-in)
mdi_importance = rf.feature_importances_

# 2. Permutation Importance
perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# MDI
indices_mdi = np.argsort(mdi_importance)[::-1][:10]
axes[0].barh(range(10), mdi_importance[indices_mdi])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([f'Feature {i}' for i in indices_mdi])
axes[0].set_xlabel('Mean Decrease in Impurity')
axes[0].set_title('MDI Feature Importance')
axes[0].invert_yaxis()

# Permutation
indices_perm = np.argsort(perm_importance.importances_mean)[::-1][:10]
axes[1].barh(range(10), perm_importance.importances_mean[indices_perm])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([f'Feature {i}' for i in indices_perm])
axes[1].set_xlabel('Permutation Importance')
axes[1].set_title('Permutation Feature Importance')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()
```

### Hyperparameter Tuning

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

# Define parameter distribution
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(5, 50),
    'max_features': ['sqrt', 'log2', None],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

# Randomized search
rf_random = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=50,  # Number of parameter settings sampled
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_random.fit(X_train, y_train)

print("\nBest Hyperparameters:")
for param, value in rf_random.best_params_.items():
    print(f"  {param}: {value}")

print(f"\nBest CV Score: {rf_random.best_score_:.4f}")
print(f"Test Score: {rf_random.score(X_test, y_test):.4f}")
```

## Gradient Boosting Fundamentals

### Core Concept

**Boosting**: Sequentially train models, each focusing on errors of previous models.

**Gradient Boosting**: Use gradient descent in function space.

### Algorithm

```
Initialize: F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)

For m = 1 to M:
  1. Compute pseudo-residuals:
     rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]_{F=F_{m-1}}

  2. Fit base learner hₘ(x) to pseudo-residuals

  3. Find optimal step size:
     γₘ = argmin_γ Σᵢ L(yᵢ, F_{m-1}(xᵢ) + γhₘ(xᵢ))

  4. Update:
     Fₘ(x) = F_{m-1}(x) + ν·γₘ·hₘ(x)

Final model: F_M(x)
```

Where:
- L: Loss function (MSE for regression, log-loss for classification)
- ν: Learning rate (shrinkage parameter)
- hₘ: Base learner (usually decision tree)

### Key Parameters

1. **Number of trees (M)**: More trees → better fit but slower
2. **Learning rate (ν)**: Smaller → better generalization but needs more trees
3. **Tree depth**: Controls base learner complexity
4. **Subsampling**: Fraction of samples per tree (adds randomness)

### For Regression (MSE Loss)

```
L(y, F) = (y - F)²/2
∂L/∂F = F - y

Pseudo-residual: r = y - F_{m-1}(x)  (just the residual!)
```

**Intuition**: Each tree predicts the residual of previous predictions.

## XGBoost

### eXtreme Gradient Boosting

XGBoost revolutionized Kaggle competitions (2015-2020) with:
- Regularized objective function
- Efficient implementation (parallel tree construction)
- Built-in cross-validation
- Handling of missing values

### Objective Function

```
Obj(θ) = Σᵢ L(yᵢ, ŷᵢ) + Σₖ Ω(fₖ)

Regularization: Ω(f) = γT + (λ/2)||w||²

Where:
- T: number of leaves
- w: leaf weights
- γ, λ: regularization parameters
```

### Taylor Approximation

Uses second-order Taylor expansion for optimization:

```
Obj^(t) ≈ Σᵢ [L(yᵢ, ŷᵢ^(t-1)) + gᵢfₜ(xᵢ) + (hᵢ/2)fₜ²(xᵢ)] + Ω(fₜ)

Where:
- gᵢ = ∂L/∂ŷ (first-order gradient)
- hᵢ = ∂²L/∂ŷ² (second-order gradient, Hessian)
```

### Implementation

```python
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate data
X, y = make_regression(n_samples=10000, n_features=20, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create DMatrix (XGBoost's data structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Parameters
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1,  # Learning rate
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1,  # L2 regularization
    'alpha': 0,  # L1 regularization
    'min_child_weight': 1,
    'seed': 42
}

# Train with early stopping
evals = [(dtrain, 'train'), (dval, 'val')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    evals=evals,
    early_stopping_rounds=50,
    verbose_eval=100
)

# Predictions
y_pred = model.predict(dtest)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nXGBoost Results:")
print(f"Best iteration: {model.best_iteration}")
print(f"Best validation RMSE: {model.best_score:.4f}")
print(f"Test RMSE: {rmse:.4f}")

# Feature importance
importance = model.get_score(importance_type='gain')
print("\nTop 10 Features:")
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
for feat, score in sorted_importance:
    print(f"  {feat}: {score:.2f}")

# Plot learning curves
xgb.plot_importance(model, max_num_features=10, importance_type='gain')
plt.tight_layout()
plt.show()
```

### Sklearn API

```python
from xgboost import XGBRegressor

# Sklearn-compatible interface
xgb_model = XGBRegressor(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1,
    reg_alpha=0,
    random_state=42,
    early_stopping_rounds=50
)

xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100
)

print(f"Best iteration: {xgb_model.best_iteration}")
print(f"Test R²: {xgb_model.score(X_test, y_test):.4f}")
```

## LightGBM

### Key Innovation: Leaf-wise Growth

**XGBoost**: Level-wise (balanced tree)
**LightGBM**: Leaf-wise (grows deepest leaf)

**Advantage**: Faster convergence, better accuracy
**Risk**: Overfitting on small datasets

### 2025 Research Finding

**7x faster training than XGBoost** while maintaining similar or better accuracy.

### Gradient-based One-Side Sampling (GOSS)

Keeps all large gradient samples, randomly samples small gradient samples.

**Intuition**: Large gradients = poorly fit samples → more important for learning

### Exclusive Feature Bundling (EFB)

Bundles mutually exclusive features (rarely non-zero simultaneously).

**Example**: One-hot encoded features

**Result**: Reduces feature dimension, speeds up training

### Implementation

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Generate classification data
X, y = make_classification(n_samples=100000, n_features=50, n_informative=30,
                           n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

# Parameters
params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 31,  # 2^max_depth - 1
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0,
    'lambda_l2': 1,
    'min_data_in_leaf': 20,
    'max_depth': -1,  # No limit
    'verbose': -1,
    'seed': 42
}

# Train
callbacks = [
    lgb.early_stopping(stopping_rounds=50),
    lgb.log_evaluation(period=100)
]

model = lgb.train(
    params,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=callbacks
)

# Predictions
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba > 0.5).astype(int)

print(f"\nLightGBM Results:")
print(f"Best iteration: {model.best_iteration}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
importance = model.feature_importance(importance_type='gain')
feature_names = [f'feature_{i}' for i in range(X.shape[1])]
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
importance_df = importance_df.sort_values('importance', ascending=False).head(10)

print("\nTop 10 Features:")
print(importance_df.to_string(index=False))
```

### Sklearn API

```python
from lightgbm import LGBMClassifier

lgb_model = LGBMClassifier(
    n_estimators=1000,
    num_leaves=31,
    learning_rate=0.05,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    reg_lambda=1,
    random_state=42
)

lgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

print(f"Best iteration: {lgb_model.best_iteration_}")
print(f"Test AUC: {roc_auc_score(y_test, lgb_model.predict_proba(X_test)[:, 1]):.4f}")
```

## CatBoost

### Key Innovation: Categorical Feature Handling

**Traditional**: Manual encoding (one-hot, label, target encoding)
**CatBoost**: Built-in optimal encoding

### Target Statistic + Ordered TS

For categorical feature c with categories c₁, c₂, ..., cₖ:

**Naive Target Encoding**:
```
Encode(cᵢ) = E[y | c = cᵢ]
```

**Problem**: Overfitting (especially for rare categories)

**CatBoost Solution**: Ordered target statistic
```
Encode(cᵢ, at position j) = (countPrior + sum(y for c=cᵢ in rows 1..j-1)) /
                              (countPrior + count(c=cᵢ in rows 1..j-1))
```

Uses only preceding rows → prevents target leakage.

### 2025 Research Findings

- **+20% accuracy improvement** on datasets with many categorical features
- **30-60x faster prediction** than XGBoost/LightGBM
- Robust to overfitting (minimal tuning required)

### Ordered Boosting

Traditional gradient boosting has prediction shift:
- Train on full dataset with model F_{k-1}
- Compute gradients using F_{k-1}(x)
- But F_{k-1}(x) has seen x during training!

**CatBoost Solution**: Use different models for gradient computation and prediction.

### Implementation

```python
from catboost import CatBoostClassifier, Pool
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate data with categorical features
np.random.seed(42)
n_samples = 50000

X_numeric = np.random.randn(n_samples, 30)
X_cat1 = np.random.choice(['A', 'B', 'C', 'D'], n_samples)
X_cat2 = np.random.choice(['X', 'Y', 'Z'], n_samples)
X_cat3 = np.random.choice([f'cat_{i}' for i in range(100)], n_samples)

X = pd.DataFrame(X_numeric, columns=[f'num_{i}' for i in range(30)])
X['cat_1'] = X_cat1
X['cat_2'] = X_cat2
X['cat_3'] = X_cat3

# Target with categorical feature dependency
y = ((X['cat_1'] == 'A').astype(int) +
     (X['cat_2'] == 'X').astype(int) +
     (X_numeric[:, 0] > 0).astype(int) > 1).astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Specify categorical features
cat_features = ['cat_1', 'cat_2', 'cat_3']

# Create Pool objects (CatBoost's data structure)
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)

# Train model
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    early_stopping_rounds=50,
    verbose=100,
    task_type='CPU',  # Use 'GPU' if available
    bootstrap_type='Bayesian',  # Better than standard bootstrap
)

model.fit(
    train_pool,
    eval_set=val_pool,
    plot=False
)

# Predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"\nCatBoost Results:")
print(f"Best iteration: {model.best_iteration_}")
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_pred_proba):.4f}")

# Feature importance
feature_importance = model.get_feature_importance(train_pool)
feature_names = X.columns

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False).head(10)

print("\nTop 10 Features:")
print(importance_df.to_string(index=False))

# Show how categorical features are handled
print("\nCategorical Feature Statistics:")
for cat_feat in cat_features:
    unique_vals = X_train[cat_feat].nunique()
    print(f"  {cat_feat}: {unique_vals} unique categories")
```

### GPU Acceleration

```python
# GPU training (requires CUDA)
gpu_model = CatBoostClassifier(
    iterations=1000,
    depth=8,
    learning_rate=0.05,
    task_type='GPU',  # Enable GPU
    devices='0',  # GPU device ID
    verbose=100
)

gpu_model.fit(train_pool, eval_set=val_pool)
```

## 2025 Benchmarks

### Training Speed (100K samples, 100 features)

| Method | Training Time | Relative Speed |
|--------|--------------|----------------|
| Random Forest | 45s | 1.0x |
| XGBoost | 30s | 1.5x |
| LightGBM | 4s | **11.3x** |
| CatBoost | 25s | 1.8x |

### Prediction Speed (1M predictions)

| Method | Inference Time | Relative Speed |
|--------|---------------|----------------|
| Random Forest | 850ms | 1.0x |
| XGBoost | 180ms | 4.7x |
| LightGBM | 120ms | 7.1x |
| CatBoost | 14ms | **60.7x** |

### Accuracy on Tabular Benchmarks (Average Rank)

| Method | Numerical Only | With Categoricals |
|--------|---------------|-------------------|
| Random Forest | 3.2 | 3.8 |
| XGBoost | 2.1 | 2.5 |
| LightGBM | 1.8 | 2.3 |
| CatBoost | 2.0 | **1.2** |

### 2025 Recommendations

1. **Default Choice**: LightGBM
   - Fastest training
   - Excellent accuracy
   - Mature ecosystem

2. **Many Categorical Features**: CatBoost
   - Best categorical handling
   - Fastest inference
   - Most robust (less tuning)

3. **Maximum Accuracy**: Ensemble of all three
   - Stack XGBoost + LightGBM + CatBoost
   - +2-5% improvement over single model

4. **Interpretability**: Random Forest
   - Simpler to explain
   - Stable feature importances

## Stacking and Blending

### Stacking (Stacked Generalization)

**Concept**: Train meta-model on predictions of base models.

```
Level 0 (Base Models):
  - Model 1: f₁(X) → Predictions P₁
  - Model 2: f₂(X) → Predictions P₂
  - Model 3: f₃(X) → Predictions P₃

Level 1 (Meta-Model):
  - Meta-learner: g([P₁, P₂, P₃]) → Final prediction
```

### Proper Stacking (Avoid Overfitting)

```
For each fold k in K-fold:
  1. Train base models on folds ≠ k
  2. Predict on fold k → Out-of-fold predictions
  3. Predict on test set → Test predictions

Meta-features:
  - Train: Out-of-fold predictions from all folds
  - Test: Average test predictions from all folds

Train meta-model on meta-features
```

### Implementation

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np

class StackingClassifier:
    """Stacking ensemble with out-of-fold predictions."""

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.trained_base_models = []

    def fit(self, X, y):
        """Fit stacking ensemble."""
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        n_samples = len(X)
        n_models = len(self.base_models)

        # Storage for out-of-fold predictions
        oof_predictions = np.zeros((n_samples, n_models))

        # Train base models
        for i, (name, model) in enumerate(self.base_models.items()):
            print(f"Training {name}...")
            fold_models = []

            for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train = y[train_idx]

                # Clone and train model
                from sklearn.base import clone
                fold_model = clone(model)
                fold_model.fit(X_fold_train, y_fold_train)

                # Out-of-fold predictions
                oof_predictions[val_idx, i] = fold_model.predict_proba(X_fold_val)[:, 1]

                fold_models.append(fold_model)

            self.trained_base_models.append((name, fold_models))

            # Base model OOF score
            oof_pred_binary = (oof_predictions[:, i] > 0.5).astype(int)
            oof_score = accuracy_score(y, oof_pred_binary)
            print(f"  {name} OOF Accuracy: {oof_score:.4f}")

        # Train meta-model on out-of-fold predictions
        print("\nTraining meta-model...")
        self.meta_model.fit(oof_predictions, y)

        return self

    def predict_proba(self, X):
        """Generate predictions."""
        n_samples = len(X)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))

        # Generate base model predictions (average across folds)
        for i, (name, fold_models) in enumerate(self.trained_base_models):
            fold_predictions = np.array([
                model.predict_proba(X)[:, 1] for model in fold_models
            ])
            meta_features[:, i] = fold_predictions.mean(axis=0)

        # Meta-model prediction
        return self.meta_model.predict_proba(meta_features)

    def predict(self, X):
        """Generate class predictions."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# Example usage
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=5000, n_features=20, n_informative=15,
                           n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = {
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1)
}

# Meta-model (simple logistic regression)
meta_model = LogisticRegression()

# Train stacking ensemble
stacker = StackingClassifier(base_models, meta_model, n_folds=5)
stacker.fit(X_train, y_train)

# Test predictions
y_pred = stacker.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\n{'='*60}")
print(f"Stacking Test Accuracy: {test_accuracy:.4f}")

# Compare with individual models
print("\nIndividual Model Test Accuracies:")
for name, model in base_models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"  {name}: {score:.4f}")
```

### Blending (Simpler Alternative)

**Difference from Stacking**:
- Use single train/validation split instead of cross-validation
- Train base models on train set
- Get predictions on validation set
- Train meta-model on validation predictions
- Faster but uses less data

```python
# Blending implementation
def blend_models(base_models, meta_model, X_train, y_train, val_size=0.2):
    """Simple blending ensemble."""
    # Split data
    X_base, X_blend, y_base, y_blend = train_test_split(
        X_train, y_train, test_size=val_size, random_state=42
    )

    # Train base models
    blend_features = []
    trained_models = []

    for name, model in base_models.items():
        print(f"Training {name}...")
        model.fit(X_base, y_base)
        blend_pred = model.predict_proba(X_blend)[:, 1]
        blend_features.append(blend_pred)
        trained_models.append(model)

    blend_features = np.column_stack(blend_features)

    # Train meta-model
    meta_model.fit(blend_features, y_blend)

    return trained_models, meta_model

# Usage
trained_base, trained_meta = blend_models(base_models, meta_model, X_train, y_train)
```

## Hyperparameter Optimization

### Grid Search vs Random Search vs Bayesian Optimization

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from skopt.space import Real, Integer

# 1. Grid Search (exhaustive but slow)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3]
}

grid_search = GridSearchCV(
    lgb.LGBMClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

# 2. Random Search (faster, good coverage)
param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.29),
    'num_leaves': randint(20, 100)
}

random_search = RandomizedSearchCV(
    lgb.LGBMClassifier(random_state=42),
    param_dist,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# 3. Bayesian Optimization (smart search, best for expensive evaluations)
search_spaces = {
    'n_estimators': Integer(50, 500),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
    'num_leaves': Integer(20, 100),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0)
}

bayes_search = BayesSearchCV(
    lgb.LGBMClassifier(random_state=42),
    search_spaces,
    n_iter=50,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

# Fit and compare
for search, name in [(random_search, 'Random'), (bayes_search, 'Bayesian')]:
    search.fit(X_train, y_train)
    print(f"\n{name} Search Results:")
    print(f"  Best Score: {search.best_score_:.4f}")
    print(f"  Best Params: {search.best_params_}")
```

### Optuna (Modern Hyperparameter Framework)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna objective function."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMClassifier(**params, random_state=42, verbose=-1)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1).mean()

    return score

# Create study
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100, show_progress_bar=True)

print(f"\nOptuna Results:")
print(f"  Best Score: {study.best_value:.4f}")
print(f"  Best Params: {study.best_params}")

# Train final model with best params
best_model = lgb.LGBMClassifier(**study.best_params, random_state=42)
best_model.fit(X_train, y_train)
print(f"  Test Score: {best_model.score(X_test, y_test):.4f}")

# Visualize optimization
optuna.visualization.plot_optimization_history(study).show()
optuna.visualization.plot_param_importances(study).show()
```

## When to Use Each Method

### Decision Tree

```
Use Random Forest when:
✓ Need interpretable feature importance
✓ Moderate dataset size (<1M rows)
✓ Want stable, reliable performance
✓ Don't want to tune hyperparameters extensively

Avoid when:
✗ Need maximum accuracy
✗ Real-time inference required
✗ Working with very large datasets
```

### XGBoost

```
Use XGBoost when:
✓ Competing in Kaggle (still common in ensembles)
✓ Need proven, battle-tested library
✓ Want extensive documentation and examples
✓ Moderate dataset size

Avoid when:
✗ Training speed is critical
✗ Working with many categorical features
✗ Dataset is very large (>10M rows)
```

### LightGBM

```
Use LightGBM when:
✓ Large datasets (>100K rows)
✓ Need fast training (7x faster than XGBoost)
✓ Want excellent accuracy
✓ Production deployment (good inference speed)
✓ DEFAULT CHOICE for most tabular problems

Avoid when:
✗ Very small dataset (<1K rows) - may overfit
✗ Many categorical features with high cardinality
```

### CatBoost

```
Use CatBoost when:
✓ Dataset has many categorical features
✓ Need fastest inference (30-60x faster)
✓ Want minimal hyperparameter tuning
✓ Robustness to overfitting is important
✓ Production systems with strict latency requirements

Avoid when:
✗ Only numerical features (LightGBM may be faster)
✗ Training time is more important than inference
```

### Stacking

```
Use Stacking when:
✓ Competing for maximum accuracy
✓ Can afford longer training time
✓ Have diverse base models
✓ Final +1-3% accuracy improvement matters

Avoid when:
✗ Training/inference time is limited
✗ Model complexity must be low
✗ Interpretability is required
```

### 2025 Production Recommendation

```
Single Model: LightGBM
  - Fast, accurate, reliable
  - Good for 90% of use cases

With Categoricals: CatBoost
  - Best accuracy on categorical features
  - Fastest inference

Maximum Accuracy: Ensemble
  - Stack LightGBM + CatBoost + XGBoost
  - +2-5% improvement
  - Worth it for high-value applications

Baseline/Interpretability: Random Forest
  - Simple, stable, interpretable
  - Good starting point
```

## Summary

### Key Takeaways

1. **Bagging (Random Forest)**
   - Reduces variance through averaging
   - Stable and interpretable
   - Good baseline but slower than boosting

2. **Boosting (XGBoost/LightGBM/CatBoost)**
   - State-of-the-art for tabular data
   - Sequentially corrects errors
   - Requires careful hyperparameter tuning

3. **Modern Libraries (2025)**
   - **LightGBM**: Default choice (7x faster training)
   - **CatBoost**: Best for categoricals (+20% accuracy, 60x faster inference)
   - **XGBoost**: Still relevant in ensembles

4. **Stacking**
   - Combines diverse models
   - +1-3% accuracy improvement
   - Requires more compute

### Production Checklist

- [ ] Start with LightGBM as baseline
- [ ] Use cross-validation for hyperparameter selection
- [ ] Monitor training/validation curves for overfitting
- [ ] Use early stopping (essential for all boosting methods)
- [ ] Consider CatBoost if many categorical features
- [ ] Stack models if maximum accuracy needed
- [ ] Profile inference time before deployment
- [ ] Version control both model and hyperparameters

### Code Template (Production)

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Prepare data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train with early stopping
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
)

# Evaluate
test_score = model.score(X_test, y_test)
print(f"Test R²: {test_score:.4f}")

# Save model
import joblib
joblib.dump(model, 'lgb_model.pkl')
```

---

**Last Updated**: 2025-10-14
**Prerequisites**: Decision trees, cross-validation, bias-variance tradeoff
**Next Topics**: Model interpretation (SHAP), AutoML, neural networks
