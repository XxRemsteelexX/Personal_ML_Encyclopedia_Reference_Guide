# 7. Tree-Based Models: Decision Trees, Random Forest, XGBoost

## Overview

Tree-based models are among the most powerful and widely used algorithms in machine learning. This guide covers when to use them, when NOT to use them, and critical best practices including encoding strategies.

**Models Covered:**
- Decision Trees
- Random Forest
- XGBoost / LightGBM
- Gradient Boosting Machines

---

## 7.1 Decision Trees

### What They Are
Decision trees split data recursively based on feature values to create rules for prediction. They're the building blocks of ensemble methods.

###  When to Use Decision Trees

1. **Need interpretability**
   - Easy to visualize and explain to non-technical stakeholders
   - Can extract clear business rules

2. **Non-linear relationships**
   - Captures interactions between features automatically
   - No assumptions about data distribution

3. **Mixed data types**
   - Handles both numerical and categorical features
   - No need for extensive preprocessing

4. **Quick baseline**
   - Fast to train
   - Good starting point before ensemble methods

###  When NOT to Use Decision Trees

1. **Need high accuracy**
   - Single trees overfit easily
   - High variance (small data changes = big tree changes)
   - **Solution:** Use ensemble methods (Random Forest, XGBoost)

2. **Linear relationships**
   - Trees inefficiently approximate linear patterns
   - **Better alternative:** Linear/Logistic Regression

3. **Extrapolation needed**
   - Trees cannot predict beyond training data range
   - **Example:** If max training value is 100, tree cannot predict 150

4. **High-dimensional sparse data**
   - Performs poorly with many features and few samples
   - **Better alternative:** Regularized linear models

5. **Need probabilistic predictions**
   - Trees give discrete splits, not smooth probabilities
   - **Better alternative:** Logistic regression, neural nets

---

## 7.2 Random Forest

### What It Is
An ensemble of decision trees trained on random subsets of data and features. Predictions are averaged (regression) or voted (classification).

###  When to Use Random Forest

1. **Tabular data with moderate feature count**
   - Sweet spot: 10-1000 features
   - Excellent for structured/tabular datasets

2. **Need robustness**
   - Resistant to overfitting (via averaging)
   - Handles noisy data well through ensemble learning

3. **Mixed data types**
   - Works with categorical + numerical without extensive preprocessing
   - Can handle missing values (with some implementations)

4. **Feature importance needed**
   - Provides feature importance scores
   - Good for feature selection

5. **Imbalanced classes**
   - Can adjust class weights
   - Use `class_weight='balanced'` in sklearn

6. **Non-linear relationships**
   - Captures complex interactions automatically
   - No need for feature engineering

7. **Need baseline quickly**
   - Works well out-of-the-box with minimal tuning
   - Robust default hyperparameters

###  When NOT to Use Random Forest

1. **High-cardinality categorical features**
   - Struggles with 100+ categories
   - **Better alternative:** XGBoost with target encoding

2. **Need extrapolation**
   - Cannot predict beyond training range
   - **Example:** Predicting future prices beyond historical max

3. **Real-time low-latency predictions**
   - 100+ trees = slower inference
   - **Better alternative:** Single optimized model or neural net

4. **Memory constraints**
   - Stores many trees in memory
   - Large datasets + many trees = high RAM usage

5. **Linear relationships dominate**
   - Inefficient for simple linear patterns
   - **Better alternative:** Linear regression, regularized models

6. **Very large datasets (10M+ rows)**
   - Training becomes slow
   - **Better alternative:** XGBoost, LightGBM (faster)

7. **Need probabilistic calibration**
   - Probabilities not well-calibrated
   - **Solution:** Apply Platt scaling or isotonic regression

8. **Sparse high-dimensional data (text, images)**
   - Inefficient with 10,000+ features
   - **Better alternative:** Linear models, deep learning

---

## 7.3 XGBoost / LightGBM / Gradient Boosting

### What They Are
Gradient boosting builds trees sequentially, each correcting errors of previous trees. XGBoost and LightGBM are optimized implementations.

###  When to Use XGBoost/LightGBM

1. **Kaggle competitions / maximum performance**
   - Often wins competitions on structured data
   - State-of-the-art for tabular data

2. **Large datasets (1M+ rows)**
   - Faster than Random Forest on large data
   - Efficient memory usage

3. **High-cardinality categoricals**
   - Handles 100s of categories better than RF
   - XGBoost 1.7+ has native categorical support

4. **Need best accuracy**
   - Better performance than Random Forest (with tuning)
   - Sequential learning captures subtle patterns

5. **Imbalanced data**
   - `scale_pos_weight` parameter helps
   - Better than Random Forest for severe imbalance

6. **Missing values**
   - XGBoost learns optimal handling automatically
   - No need to impute

7. **Have time to tune**
   - Many hyperparameters to optimize
   - Tuning yields significant gains

###  When NOT to Use XGBoost

1. **Limited training time**
   - Requires careful hyperparameter tuning
   - **Better alternative:** Random Forest (good defaults)

2. **Need interpretability**
   - Harder to explain than single trees
   - Many trees, complex interactions
   - **Better alternative:** Decision tree, linear models

3. **Small datasets (<1000 rows)**
   - Easily overfits
   - **Better alternative:** Regularized models, simple models

4. **Linear relationships**
   - Overkill for simple patterns
   - **Better alternative:** Linear regression

5. **Need fast inference**
   - Hundreds/thousands of trees = slower
   - **Better alternative:** Simplified models, distillation

6. **GPU unavailable and data is huge**
   - CPU training can be slow
   - **Better alternative:** LightGBM (faster on CPU)

---

## 7.4 CRITICAL: Encoding for Tree-Based Models

### The Encoding Problem

**One-hot encoding often HURTS tree-based model performance!**

### Encoding Comparison

| Encoding Type | Random Forest | XGBoost | When to Use | Pros | Cons |
|--------------|---------------|---------|-------------|------|------|
| **Label Encoding** |  Good |  Good | Low-med cardinality (<50) | Fast, low memory, no dimensionality increase | Creates spurious ordering |
| **One-Hot Encoding** | [WARNING] Avoid | [WARNING] Avoid | Very low cardinality (<10) | No false ordering | Explodes dimensions, slower training, splits become inefficient |
| **Target Encoding** |  Excellent |  Excellent | High cardinality (50+) | Captures target relationship, single feature | Risk of leakage, needs CV |
| **Frequency Encoding** |  Good |  Good | Any cardinality | Simple, no leakage | Less informative |
| **Native Categorical** |  Not available |  XGBoost 1.7+ | XGBoost only | Optimal splits | Still experimental |

---

### Encoding Best Practices

#### 1. Label Encoding for Tree Models (Default Choice)

```python
from sklearn.preprocessing import LabelEncoder

# For tree-based models, label encoding works well
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

**Why it works:**
- Trees split on single feature values
- No increase in dimensionality
- Faster training
- Less memory usage

**The "false ordering" myth:**
- Trees don't care about numeric ordering
- They split on thresholds: `feature < 3` vs `feature >= 3`
- No assumption of ordinal relationship

#### 2. Target Encoding (Best for High Cardinality)

```python
# Target encoding with cross-validation to prevent leakage
from category_encoders import TargetEncoder

# CRITICAL: Use cross-validation to prevent leakage
encoder = TargetEncoder(cols=['high_cardinality_feature'])

# On training data
X_train_encoded = encoder.fit_transform(X_train, y_train)

# On test data
X_test_encoded = encoder.transform(X_test)
```

**When to use:**
- High cardinality (50+ categories)
- Categories have different target rates
- Example: ZIP codes, user IDs, product IDs

**Pros:**
- Captures category --> target relationship
- Single numeric feature (no dimension explosion)
- Often improves performance significantly

**Cons:**
- **LEAKAGE RISK:** Must use cross-validation encoding
- Can overfit on small categories

**Best practice:**
```python
# Add smoothing to prevent overfitting on rare categories
encoder = TargetEncoder(
    cols=['category'],
    smoothing=1.0,  # Blend with global mean
    min_samples_leaf=20  # Minimum samples for category
)
```

#### 3. Frequency Encoding (Simple, No Leakage)

```python
# Count how often each category appears
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)
```

**When to use:**
- Want something simple and safe
- Frequency correlates with target
- Example: Rare products might be niche/expensive

#### 4.  Avoid One-Hot Encoding for Trees

**Why it hurts performance:**

1. **Dimensionality explosion**
   - 100 categories --> 100 features
   - Slows training significantly
   - Increases memory usage

2. **Inefficient splits**
   - One-hot creates many sparse binary features
   - Trees must check each binary column separately
   - Example: Instead of 1 split on original feature, needs 100 splits

3. **Feature importance gets diluted**
   - Importance spread across many dummy variables
   - Harder to interpret

**Only use one-hot when:**
- Nominal categories with NO ordering and very low cardinality (<10)
- Using linear models (which DO need one-hot)

---

### Encoding Decision Flow

```
Is cardinality < 10?
+--- Yes --> Label Encoding (safe default)
|
+--- No --> Is cardinality < 50?
    +--- Yes --> Label Encoding or Target Encoding
    |
    +--- No (50+ categories) --> Target Encoding (with CV)
        +--- Alternative: Frequency Encoding (safer, simpler)
        +--- XGBoost only: Try native categorical support
```

---

## 7.5 Hyperparameter Tuning Guide

### Random Forest Key Parameters

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,        # ^ = better but slower (100-500 typical)
    max_depth=None,          # v = prevent overfitting (try 10-50)
    min_samples_split=2,     # ^ = prevent overfitting (try 5-20)
    min_samples_leaf=1,      # ^ = prevent overfitting (try 2-10)
    max_features='sqrt',     # sqrt for classification, 1/3 for regression
    bootstrap=True,          # Always True for RF
    class_weight='balanced', # For imbalanced data
    n_jobs=-1,              # Use all CPU cores
    random_state=42
)
```

**Tuning priority:**
1. `n_estimators` (more is better, diminishing returns after 200-300)
2. `max_depth` (prevent overfitting)
3. `min_samples_leaf` (prevent overfitting)
4. `max_features` (controls randomness)

### XGBoost Key Parameters

```python
import xgboost as xgb

xgb_model = xgb.XGBClassifier(
    n_estimators=1000,           # High value, use early stopping
    learning_rate=0.01,          # v = better but needs more trees (0.01-0.3)
    max_depth=6,                 # v = prevent overfitting (3-10)
    min_child_weight=1,          # ^ = prevent overfitting (1-10)
    subsample=0.8,               # Row sampling (0.5-1.0)
    colsample_bytree=0.8,        # Column sampling (0.5-1.0)
    gamma=0,                     # ^ = prevent overfitting (0-5)
    reg_alpha=0,                 # L1 regularization (0-10)
    reg_lambda=1,                # L2 regularization (0-10)
    scale_pos_weight=1,          # For imbalanced: sum(negative)/sum(positive)
    tree_method='hist',          # Faster training
    enable_categorical=True,     # XGBoost 1.7+ native categorical
    early_stopping_rounds=50,
    eval_metric='logloss'
)
```

**Tuning priority:**
1. `learning_rate` + `n_estimators` (tune together)
2. `max_depth` (single most important)
3. `subsample` + `colsample_bytree` (regularization)
4. `min_child_weight` (prevent overfitting)

**Pro tip:**
- Start with `learning_rate=0.1`, find best `n_estimators` with early stopping
- Then reduce `learning_rate` to 0.01 and increase `n_estimators` proportionally
- Lower learning rate = better performance but longer training

---

## 7.6 Common Pitfalls & Solutions

### Pitfall 1: Data Leakage in Target Encoding

 **Wrong:**
```python
# This causes leakage!
encoder = TargetEncoder()
df['encoded'] = encoder.fit_transform(df[['category']], df['target'])

# Then split train/test
X_train, X_test = train_test_split(df)
```

 **Correct:**
```python
# Split FIRST
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Then encode (using only training data)
encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)  # No fit!
```

### Pitfall 2: Not Using Early Stopping (XGBoost)

 **Wrong:**
```python
model = xgb.XGBClassifier(n_estimators=1000)
model.fit(X_train, y_train)  # Might overfit!
```

 **Correct:**
```python
model = xgb.XGBClassifier(
    n_estimators=10000,  # Set high
    early_stopping_rounds=50
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
# Stops when validation performance plateaus
```

### Pitfall 3: Using One-Hot for High Cardinality

 **Wrong:**
```python
# 100 categories --> 100 features!
df_encoded = pd.get_dummies(df, columns=['high_card_feature'])
rf.fit(df_encoded, y)  # Slow, worse performance
```

 **Correct:**
```python
# Use label or target encoding
le = LabelEncoder()
df['encoded'] = le.fit_transform(df['high_card_feature'])
rf.fit(df[['encoded']], y)  # Fast, better performance
```

### Pitfall 4: Not Scaling for Tree Models

 **Good news:** Tree models don't need feature scaling!

```python
# NO NEED for StandardScaler or MinMaxScaler
# Trees split on thresholds, scale doesn't matter
rf.fit(X, y)  # Works fine with mixed scales
```

### Pitfall 5: Ignoring Class Imbalance

 **Wrong:**
```python
rf = RandomForestClassifier()  # Biased toward majority class
```

 **Correct:**
```python
# Random Forest
rf = RandomForestClassifier(class_weight='balanced')

# XGBoost
scale_pos_weight = sum(y == 0) / sum(y == 1)
xgb_model = xgb.XGBClassifier(scale_pos_weight=scale_pos_weight)
```

---

## 7.7 Quick Reference: Model Selection

```
Start here: What's your goal?

+--- Need interpretability?
|  +--- Use: Decision Tree (single)
|
+--- Need quick baseline?
|  +--- Use: Random Forest (good defaults)
|
+--- Want maximum accuracy (have time to tune)?
|  +--- Use: XGBoost or LightGBM
|
+--- Large dataset (1M+ rows)?
|  +--- Use: LightGBM (fastest)
|
+--- High-cardinality categoricals?
|  +--- Use: XGBoost with target encoding
|
+--- Linear relationship obvious?
   +--- DON'T use trees! Use linear model
```

---

## 7.8 Summary Checklist

### Before Training:
- [ ] Encode categoricals (label or target, NOT one-hot)
- [ ] Handle missing values (or use XGBoost native handling)
- [ ] Check for class imbalance
- [ ] Split data BEFORE encoding (prevent leakage)

### Model Choice:
- [ ] Quick baseline --> Random Forest
- [ ] Maximum performance --> XGBoost
- [ ] Need interpretability --> Single Decision Tree
- [ ] Large data --> LightGBM

### During Training:
- [ ] Use cross-validation
- [ ] Set `random_state` for reproducibility
- [ ] Use early stopping (XGBoost)
- [ ] Monitor validation metrics

### After Training:
- [ ] Check feature importances
- [ ] Validate on holdout test set
- [ ] Check for overfitting (train vs test scores)
- [ ] Calibrate probabilities if needed

---

## Resources & Further Reading

**XGBoost Documentation:**
- https://xgboost.readthedocs.io/

**Encoding Libraries:**
- category_encoders: https://contrib.scikit-learn.org/category_encoders/

**Hyperparameter Tuning:**
- Optuna (automated tuning): https://optuna.org/

**Key Papers:**
- Random Forests (Breiman, 2001)
- XGBoost (Chen & Guestrin, 2016)

---

**Last Updated:** 2025-10-12
**Next Section:** Linear Models & SVMs (Phase 2)
