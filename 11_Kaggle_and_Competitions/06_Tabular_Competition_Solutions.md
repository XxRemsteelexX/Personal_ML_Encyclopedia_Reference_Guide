# Tabular Competition Solutions

## Table of Contents
- [Introduction](#introduction)
- [Gradient Boosting Frameworks](#gradient-boosting-frameworks)
- [Feature Engineering Patterns](#feature-engineering-patterns)
- [Feature Selection](#feature-selection)
- [Cross-Validation Strategies](#cross-validation-strategies)
- [Winning Solution Breakdowns](#winning-solution-breakdowns)
- [Ensemble Methods](#ensemble-methods)
- [Post-Processing Tricks](#post-processing-tricks)
- [Neural Networks for Tabular](#neural-networks-for-tabular)
- [Imbalanced Data Handling](#imbalanced-data-handling)
- [Resources](#resources)

---

## Introduction

Tabular machine learning competitions remain the most common format on Kaggle and similar platforms. Despite advances in deep learning, **gradient boosting methods** (XGBoost, LightGBM, CatBoost) dominate tabular competitions, winning approximately 80-90% of top placements.

### Why Gradient Boosting Dominates

**Advantages of gradient boosting for tabular data:**
- Handles mixed data types (numerical and categorical) naturally
- Robust to outliers and missing values
- Built-in feature importance and interactions
- Fast training with parallelization
- Minimal preprocessing required
- Strong performance out-of-the-box

**When neural networks win:**
- Very large datasets (millions of rows)
- Complex non-linear interactions
- High-cardinality categorical features
- Time-series or sequential patterns
- When combined with gradient boosting in ensembles

### Competition Landscape

**Typical tabular competition characteristics:**
- 50k to 5M training examples
- 50 to 1000+ features
- Binary classification or regression most common
- Custom evaluation metrics (AUC, RMSE, F1, LogLoss)
- Domain-specific constraints (finance, healthcare, retail)

---

## Gradient Boosting Frameworks

### XGBoost

**XGBoost** (Extreme Gradient Boosting) is the most widely used framework, known for speed and performance.

**Key hyperparameters:**

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost parameters for classification
xgb_params = {
    # Tree structure
    'max_depth': 7,                    # 6-8 typical range, deeper = more complex
    'min_child_weight': 1,             # Minimum sum of instance weight in child
    'gamma': 0,                        # Minimum loss reduction for split

    # Boosting parameters
    'learning_rate': 0.03,             # 0.01-0.05 for competitions
    'n_estimators': 5000,              # Large value with early stopping

    # Sampling
    'subsample': 0.8,                  # Row sampling, 0.7-0.8 prevents overfitting
    'colsample_bytree': 0.8,           # Column sampling per tree
    'colsample_bylevel': 1.0,          # Column sampling per level
    'colsample_bynode': 1.0,           # Column sampling per split

    # Regularization
    'reg_alpha': 0.1,                  # L1 regularization
    'reg_lambda': 1.0,                 # L2 regularization (default 1.0)

    # Other
    'objective': 'binary:logistic',    # Loss function
    'eval_metric': 'auc',              # Evaluation metric
    'tree_method': 'hist',             # 'hist' or 'gpu_hist' for speed
    'random_state': 42,
    'n_jobs': -1
}

# Train with early stopping
model = xgb.XGBClassifier(**xgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    early_stopping_rounds=100,
    verbose=50
)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]

# Feature importance
importance = model.get_booster().get_score(importance_type='gain')
```

**XGBoost regression example:**

```python
xgb_reg_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 3000,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 3,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'tree_method': 'hist',
    'random_state': 42
}

model = xgb.XGBRegressor(**xgb_reg_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=100,
    verbose=50
)
```

---

### LightGBM

**LightGBM** uses leaf-wise growth (vs. level-wise in XGBoost), often faster and more accurate.

**Key hyperparameters:**

```python
import lightgbm as lgb

# LightGBM parameters for classification
lgb_params = {
    # Tree structure
    'num_leaves': 63,                  # 31-127 typical, max leaves (2^max_depth - 1)
    'max_depth': -1,                   # -1 for no limit, or 6-10
    'min_child_samples': 20,           # Minimum data in leaf
    'min_child_weight': 1e-3,          # Minimum sum of hessian in leaf
    'min_split_gain': 0.0,             # Minimum gain to split

    # Boosting parameters
    'learning_rate': 0.03,
    'n_estimators': 5000,
    'boosting_type': 'gbdt',           # 'gbdt', 'dart', 'goss'

    # Sampling
    'subsample': 0.8,                  # Also called 'bagging_fraction'
    'bagging_freq': 1,                 # Frequency for bagging
    'feature_fraction': 0.8,           # Also called 'colsample_bytree'

    # Regularization
    'lambda_l1': 0.1,                  # L1 regularization
    'lambda_l2': 1.0,                  # L2 regularization

    # Other
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Train with early stopping
model = lgb.LGBMClassifier(**lgb_params)
model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]
```

**DART mode for better generalization:**

```python
# DART (Dropouts meet Multiple Additive Regression Trees)
lgb_dart_params = {
    'boosting_type': 'dart',
    'drop_rate': 0.1,                  # Dropout rate for trees
    'max_drop': 50,                    # Max number of dropped trees
    'skip_drop': 0.5,                  # Probability of skipping dropout
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 3000,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42
}

model_dart = lgb.LGBMClassifier(**lgb_dart_params)
model_dart.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(100)]
)
```

---

### CatBoost

**CatBoost** excels with categorical features, handling them natively without encoding.

**Key hyperparameters:**

```python
from catboost import CatBoostClassifier, Pool

# CatBoost parameters
cat_params = {
    # Tree structure
    'depth': 8,                        # 6-10 typical range
    'min_data_in_leaf': 1,

    # Boosting parameters
    'iterations': 5000,
    'learning_rate': 0.03,

    # Sampling
    'bootstrap_type': 'Bayesian',      # 'Bayesian', 'Bernoulli', 'MVS'
    'bagging_temperature': 1.0,        # For Bayesian bootstrap
    'subsample': 0.8,                  # For Bernoulli/MVS

    # Regularization
    'l2_leaf_reg': 3.0,                # L2 regularization
    'random_strength': 1.0,            # Randomness for scoring splits

    # Growing policy
    'grow_policy': 'SymmetricTree',    # 'SymmetricTree', 'Depthwise', 'Lossguide'

    # Other
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'random_seed': 42,
    'task_type': 'CPU',                # 'CPU' or 'GPU'
    'verbose': 50
}

# Specify categorical features
cat_features = ['cat_col1', 'cat_col2', 'cat_col3']

# Create Pool objects for efficiency
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Train with early stopping
model = CatBoostClassifier(**cat_params)
model.fit(
    train_pool,
    eval_set=val_pool,
    early_stopping_rounds=100,
    verbose=50
)

# Predict
y_pred = model.predict_proba(X_test)[:, 1]
```

---

### Framework Comparison

| Feature | XGBoost | LightGBM | CatBoost |
|---------|---------|----------|----------|
| **Tree Growth** | Level-wise | Leaf-wise | Level-wise (default) |
| **Speed** | Fast | Fastest | Medium |
| **Categorical Handling** | Manual encoding | Manual encoding | Native support |
| **Memory Usage** | Medium | Low | High |
| **GPU Support** | Yes | Yes | Yes |
| **Default Performance** | Good | Good | Best |
| **Hyperparameter Tuning** | Moderate | Moderate | Minimal needed |
| **Overfitting Risk** | Medium | Higher | Lower |
| **Best For** | General purpose | Large datasets, speed | Categorical features |

**Recommendation:** Start with LightGBM for speed, use CatBoost if many categorical features, use XGBoost for final ensembles.

---

## Feature Engineering Patterns

Feature engineering is the most important factor in tabular competitions. Good features can improve performance more than hyperparameter tuning.

### Aggregation Features

**Groupby aggregations** create powerful features by summarizing data at different levels.

```python
import pandas as pd
import numpy as np

def create_aggregation_features(df, group_cols, agg_cols, agg_funcs):
    """
    Create aggregation features for tabular data.

    Args:
        df: DataFrame
        group_cols: Columns to group by (list)
        agg_cols: Columns to aggregate (list)
        agg_funcs: Aggregation functions (list)

    Returns:
        DataFrame with aggregation features
    """
    agg_features = pd.DataFrame()

    for col in agg_cols:
        for func in agg_funcs:
            # Create aggregation
            agg_name = f'{col}_by_{"_".join(group_cols)}_{func}'
            agg_result = df.groupby(group_cols)[col].transform(func)
            agg_features[agg_name] = agg_result

            # Difference from group statistic
            if func in ['mean', 'median']:
                diff_name = f'{col}_diff_{func}_{"_".join(group_cols)}'
                agg_features[diff_name] = df[col] - agg_result

            # Ratio to group statistic
            if func in ['mean', 'median', 'max']:
                ratio_name = f'{col}_ratio_{func}_{"_".join(group_cols)}'
                agg_features[ratio_name] = df[col] / (agg_result + 1e-5)

    return agg_features

# Example usage
group_cols = ['customer_id']
agg_cols = ['transaction_amount', 'transaction_count']
agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'sum']

agg_feats = create_aggregation_features(df, group_cols, agg_cols, agg_funcs)
df = pd.concat([df, agg_feats], axis=1)

# Multi-level aggregations
# Customer-level
customer_feats = df.groupby('customer_id').agg({
    'transaction_amount': ['mean', 'std', 'min', 'max', 'sum', 'count'],
    'days_since_last': ['mean', 'min'],
    'is_fraud': ['sum', 'mean']
}).reset_index()
customer_feats.columns = ['_'.join(col).strip('_') for col in customer_feats.columns]

# Merchant-level
merchant_feats = df.groupby('merchant_id').agg({
    'transaction_amount': ['mean', 'std', 'median'],
    'customer_id': ['nunique']
}).reset_index()
merchant_feats.columns = ['_'.join(col).strip('_') for col in merchant_feats.columns]

# Merge back
df = df.merge(customer_feats, on='customer_id', how='left')
df = df.merge(merchant_feats, on='merchant_id', how='left')
```

---

### Time-Based Features

**Extract temporal patterns** from datetime columns.

```python
def create_time_features(df, date_col):
    """
    Create comprehensive time-based features.

    Args:
        df: DataFrame with datetime column
        date_col: Name of datetime column

    Returns:
        DataFrame with time features
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Basic temporal features
    df[f'{date_col}_year'] = df[date_col].dt.year
    df[f'{date_col}_month'] = df[date_col].dt.month
    df[f'{date_col}_day'] = df[date_col].dt.day
    df[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
    df[f'{date_col}_dayofyear'] = df[date_col].dt.dayofyear
    df[f'{date_col}_hour'] = df[date_col].dt.hour
    df[f'{date_col}_minute'] = df[date_col].dt.minute

    # Cyclical encoding for periodic features
    df[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
    df[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
    df[f'{date_col}_day_sin'] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
    df[f'{date_col}_day_cos'] = np.cos(2 * np.pi * df[date_col].dt.day / 31)
    df[f'{date_col}_hour_sin'] = np.sin(2 * np.pi * df[date_col].dt.hour / 24)
    df[f'{date_col}_hour_cos'] = np.cos(2 * np.pi * df[date_col].dt.hour / 24)

    # Binary indicators
    df[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
    df[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
    df[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)
    df[f'{date_col}_is_quarter_start'] = df[date_col].dt.is_quarter_start.astype(int)
    df[f'{date_col}_is_quarter_end'] = df[date_col].dt.is_quarter_end.astype(int)

    # Relative time features
    df[f'{date_col}_days_since_start'] = (df[date_col] - df[date_col].min()).dt.days
    df[f'{date_col}_days_until_end'] = (df[date_col].max() - df[date_col]).dt.days

    return df

# Lag and rolling window features
def create_lag_features(df, group_col, value_col, lags=[1, 2, 3, 7, 14, 30]):
    """Create lag features for time series data."""
    df = df.sort_values(['customer_id', 'date'])

    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df.groupby(group_col)[value_col].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 30]:
        df[f'{value_col}_rolling_mean_{window}'] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'{value_col}_rolling_std_{window}'] = df.groupby(group_col)[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    return df
```

---

### Interaction Features

**Create features from combinations** of existing features.

```python
def create_interaction_features(df, numeric_cols, categorical_cols=None):
    """
    Create interaction features between numeric and categorical columns.

    Args:
        df: DataFrame
        numeric_cols: List of numeric columns
        categorical_cols: List of categorical columns (optional)

    Returns:
        DataFrame with interaction features
    """
    df_new = df.copy()

    # Numeric-Numeric interactions
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            # Arithmetic operations
            df_new[f'{col1}_plus_{col2}'] = df[col1] + df[col2]
            df_new[f'{col1}_minus_{col2}'] = df[col1] - df[col2]
            df_new[f'{col1}_times_{col2}'] = df[col1] * df[col2]
            df_new[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-5)

            # Ratios (both directions)
            df_new[f'{col1}_ratio_{col2}'] = df[col1] / (df[col2] + 1e-5)
            df_new[f'{col2}_ratio_{col1}'] = df[col2] / (df[col1] + 1e-5)

    # Polynomial features (squared, cubed)
    for col in numeric_cols:
        df_new[f'{col}_squared'] = df[col] ** 2
        df_new[f'{col}_cubed'] = df[col] ** 3
        df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
        df_new[f'{col}_log'] = np.log1p(np.abs(df[col]))

    # Categorical-Numeric interactions
    if categorical_cols:
        for cat_col in categorical_cols:
            for num_col in numeric_cols:
                # Group statistics already created in aggregation features
                pass

    return df_new

# Example: Domain-specific interactions
# For transaction data
df['amount_per_item'] = df['total_amount'] / (df['item_count'] + 1)
df['amount_deviation'] = df['amount'] - df['amount_mean_by_customer']
df['velocity'] = df['distance'] / (df['time'] + 1e-5)

# For credit scoring
df['debt_to_income'] = df['total_debt'] / (df['annual_income'] + 1)
df['credit_utilization'] = df['credit_used'] / (df['credit_limit'] + 1)
```

---

### Target Encoding

**Target encoding** maps categorical values to target statistics, powerful but requires care to avoid leakage.

```python
from sklearn.model_selection import KFold

def target_encode_cv(train_df, test_df, cat_cols, target_col, n_splits=5, smoothing=10):
    """
    Target encoding with cross-validation to prevent leakage.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        cat_cols: List of categorical columns to encode
        target_col: Target column name
        n_splits: Number of CV folds
        smoothing: Smoothing parameter (higher = more regularization)

    Returns:
        train_encoded, test_encoded DataFrames
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Global mean for smoothing
    global_mean = train_df[target_col].mean()

    for col in cat_cols:
        # Initialize encoded column
        train_encoded[f'{col}_target_enc'] = 0.0
        test_encoded[f'{col}_target_enc'] = 0.0

        # CV encoding for train
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for train_idx, val_idx in kf.split(train_df):
            # Calculate statistics on train fold
            stats = train_df.iloc[train_idx].groupby(col)[target_col].agg(['mean', 'count'])

            # Smoothed encoding
            # Formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            stats['smoothed_mean'] = (
                (stats['count'] * stats['mean'] + smoothing * global_mean) /
                (stats['count'] + smoothing)
            )

            # Map to validation fold
            train_encoded.loc[val_idx, f'{col}_target_enc'] = (
                train_df.iloc[val_idx][col].map(stats['smoothed_mean'])
            )

        # Fill missing with global mean
        train_encoded[f'{col}_target_enc'].fillna(global_mean, inplace=True)

        # Encoding for test (use all training data)
        stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
        stats['smoothed_mean'] = (
            (stats['count'] * stats['mean'] + smoothing * global_mean) /
            (stats['count'] + smoothing)
        )
        test_encoded[f'{col}_target_enc'] = test_df[col].map(stats['smoothed_mean'])
        test_encoded[f'{col}_target_enc'].fillna(global_mean, inplace=True)

    return train_encoded, test_encoded

# Example usage
cat_cols = ['category_1', 'category_2', 'category_3']
train_encoded, test_encoded = target_encode_cv(
    train_df, test_df,
    cat_cols=cat_cols,
    target_col='target',
    n_splits=5,
    smoothing=10
)
```

---

### Frequency and Label Encoding

```python
def frequency_encoding(train_df, test_df, cat_cols):
    """
    Encode categorical variables by their frequency.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        cat_cols: List of categorical columns

    Returns:
        train_encoded, test_encoded DataFrames
    """
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_cols:
        # Calculate frequencies
        freq = train_df[col].value_counts(normalize=True).to_dict()

        # Map frequencies
        train_encoded[f'{col}_freq'] = train_df[col].map(freq)
        test_encoded[f'{col}_freq'] = test_df[col].map(freq)

        # Fill unseen categories with 0
        train_encoded[f'{col}_freq'].fillna(0, inplace=True)
        test_encoded[f'{col}_freq'].fillna(0, inplace=True)

    return train_encoded, test_encoded

# Label encoding for ordinal or tree-based models
from sklearn.preprocessing import LabelEncoder

def label_encode(train_df, test_df, cat_cols):
    """Label encode categorical columns."""
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    for col in cat_cols:
        le = LabelEncoder()

        # Fit on train
        train_encoded[f'{col}_label'] = le.fit_transform(train_df[col].astype(str))

        # Transform test (handle unseen categories)
        test_encoded[f'{col}_label'] = test_df[col].astype(str).map(
            {label: idx for idx, label in enumerate(le.classes_)}
        )
        test_encoded[f'{col}_label'].fillna(-1, inplace=True)

    return train_encoded, test_encoded

# One-hot encoding decision rule:
# Use when:
# - Low cardinality (< 10-20 unique values)
# - Linear models (Logistic Regression, Linear SVM)
# - Neural networks
#
# Avoid when:
# - High cardinality (creates too many features)
# - Tree-based models (prefer label encoding or target encoding)
```

---

### NLP Features in Tabular Data

**Extract features from text columns** in tabular datasets.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import re

def create_text_features(df, text_col):
    """
    Create NLP features from text column.

    Args:
        df: DataFrame
        text_col: Name of text column

    Returns:
        DataFrame with text features
    """
    df = df.copy()

    # Basic text statistics
    df[f'{text_col}_length'] = df[text_col].astype(str).str.len()
    df[f'{text_col}_word_count'] = df[text_col].astype(str).str.split().str.len()
    df[f'{text_col}_unique_words'] = df[text_col].astype(str).apply(
        lambda x: len(set(x.lower().split()))
    )
    df[f'{text_col}_avg_word_length'] = (
        df[f'{text_col}_length'] / (df[f'{text_col}_word_count'] + 1)
    )

    # Character-level features
    df[f'{text_col}_num_digits'] = df[text_col].astype(str).str.count(r'\d')
    df[f'{text_col}_num_uppercase'] = df[text_col].astype(str).str.count(r'[A-Z]')
    df[f'{text_col}_num_punctuation'] = df[text_col].astype(str).str.count(r'[^\w\s]')
    df[f'{text_col}_num_special'] = df[text_col].astype(str).str.count(r'[@#$%^&*]')

    # Lexical features
    df[f'{text_col}_has_url'] = df[text_col].astype(str).str.contains(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
        regex=True
    ).astype(int)
    df[f'{text_col}_has_email'] = df[text_col].astype(str).str.contains(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        regex=True
    ).astype(int)

    return df

# TF-IDF features
def create_tfidf_features(train_df, test_df, text_col, n_components=50):
    """
    Create TF-IDF features and reduce dimensionality.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        text_col: Text column name
        n_components: Number of TF-IDF components to keep

    Returns:
        train_tfidf, test_tfidf DataFrames
    """
    # TF-IDF vectorization
    tfidf = TfidfVectorizer(
        max_features=n_components,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        strip_accents='unicode',
        lowercase=True,
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english'
    )

    # Fit on train
    train_tfidf = tfidf.fit_transform(train_df[text_col].astype(str))
    test_tfidf = tfidf.transform(test_df[text_col].astype(str))

    # Convert to DataFrame
    tfidf_cols = [f'{text_col}_tfidf_{i}' for i in range(n_components)]
    train_tfidf_df = pd.DataFrame(
        train_tfidf.toarray()[:, :n_components],
        columns=tfidf_cols,
        index=train_df.index
    )
    test_tfidf_df = pd.DataFrame(
        test_tfidf.toarray()[:, :n_components],
        columns=tfidf_cols,
        index=test_df.index
    )

    return train_tfidf_df, test_tfidf_df
```

---

### Complete Feature Engineering Pipeline

```python
class FeatureEngineer:
    """Complete feature engineering pipeline for tabular data."""

    def __init__(self, numeric_cols, categorical_cols, text_cols=None, date_cols=None):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.text_cols = text_cols or []
        self.date_cols = date_cols or []

    def fit_transform(self, train_df, target_col):
        """Fit on train and transform."""
        self.train_df = train_df.copy()
        self.target_col = target_col

        # Time features
        for date_col in self.date_cols:
            train_df = create_time_features(train_df, date_col)

        # Text features
        for text_col in self.text_cols:
            train_df = create_text_features(train_df, text_col)

        # Interaction features
        train_df = create_interaction_features(train_df, self.numeric_cols)

        # Target encoding
        train_df, _ = target_encode_cv(
            train_df, train_df,  # Dummy test
            self.categorical_cols,
            target_col,
            n_splits=5
        )

        # Frequency encoding
        train_df, _ = frequency_encoding(train_df, train_df, self.categorical_cols)

        return train_df

    def transform(self, test_df):
        """Transform test data."""
        # Time features
        for date_col in self.date_cols:
            test_df = create_time_features(test_df, date_col)

        # Text features
        for text_col in self.text_cols:
            test_df = create_text_features(test_df, text_col)

        # Interaction features
        test_df = create_interaction_features(test_df, self.numeric_cols)

        # Target encoding
        _, test_df = target_encode_cv(
            self.train_df, test_df,
            self.categorical_cols,
            self.target_col,
            n_splits=5
        )

        # Frequency encoding
        _, test_df = frequency_encoding(self.train_df, test_df, self.categorical_cols)

        return test_df
```

---

## Feature Selection

Feature selection reduces overfitting, speeds up training, and improves model performance.

### Null Importance

**Null importance** identifies features that perform no better than random noise.

```python
import warnings
warnings.filterwarnings('ignore')

def get_null_importance(X_train, y_train, n_runs=10, n_estimators=100):
    """
    Calculate null importance by training on shuffled targets.

    Args:
        X_train: Training features
        y_train: Training target
        n_runs: Number of random shuffles
        n_estimators: Number of trees in model

    Returns:
        actual_importance, null_importance DataFrames
    """
    import lightgbm as lgb

    # Train on actual data
    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    actual_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })

    # Train on shuffled targets
    null_importances = []
    for run in range(n_runs):
        # Shuffle target
        y_shuffled = y_train.copy()
        np.random.shuffle(y_shuffled)

        # Train model
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=0.05,
            num_leaves=31,
            random_state=run,
            n_jobs=-1
        )
        model.fit(X_train, y_shuffled)

        # Store importances
        null_importances.append(model.feature_importances_)

    # Combine null importances
    null_importance = pd.DataFrame(
        null_importances,
        columns=X_train.columns
    )

    return actual_importance, null_importance

def select_features_null_importance(actual_imp, null_imp, threshold=75):
    """
    Select features based on null importance.

    Args:
        actual_imp: Actual importance DataFrame
        null_imp: Null importance DataFrame
        threshold: Percentile threshold (e.g., 75 = keep if actual > 75th percentile of null)

    Returns:
        List of selected features
    """
    selected_features = []

    for feature in actual_imp['feature']:
        actual_value = actual_imp[actual_imp['feature'] == feature]['importance'].values[0]
        null_values = null_imp[feature].values
        null_threshold = np.percentile(null_values, threshold)

        if actual_value > null_threshold:
            selected_features.append(feature)

    return selected_features

# Usage
actual_imp, null_imp = get_null_importance(X_train, y_train, n_runs=10)
selected_features = select_features_null_importance(actual_imp, null_imp, threshold=75)
print(f"Selected {len(selected_features)} out of {len(X_train.columns)} features")

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
```

---

### Permutation Importance

**Permutation importance** measures feature importance by shuffling each feature and measuring performance drop.

```python
from sklearn.metrics import roc_auc_score

def permutation_importance(model, X_val, y_val, metric=roc_auc_score, n_repeats=10):
    """
    Calculate permutation importance.

    Args:
        model: Trained model
        X_val: Validation features
        y_val: Validation target
        metric: Scoring metric
        n_repeats: Number of times to permute each feature

    Returns:
        DataFrame with permutation importances
    """
    # Baseline score
    y_pred = model.predict_proba(X_val)[:, 1]
    baseline_score = metric(y_val, y_pred)

    importances = []

    for col in X_val.columns:
        scores = []

        for _ in range(n_repeats):
            # Permute feature
            X_permuted = X_val.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col].values)

            # Calculate score
            y_pred_permuted = model.predict_proba(X_permuted)[:, 1]
            permuted_score = metric(y_val, y_pred_permuted)

            # Importance = drop in performance
            scores.append(baseline_score - permuted_score)

        importances.append({
            'feature': col,
            'importance_mean': np.mean(scores),
            'importance_std': np.std(scores)
        })

    importance_df = pd.DataFrame(importances).sort_values(
        'importance_mean',
        ascending=False
    )

    return importance_df

# Usage
perm_imp = permutation_importance(model, X_val, y_val, n_repeats=5)
important_features = perm_imp[perm_imp['importance_mean'] > 0]['feature'].tolist()
```

---

### Adversarial Validation

**Adversarial validation** detects train/test distribution shift and identifies problematic features.

```python
from sklearn.model_selection import cross_val_score

def adversarial_validation(train_df, test_df, features, n_splits=5):
    """
    Perform adversarial validation to check train/test similarity.

    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        features: List of feature columns
        n_splits: Number of CV folds

    Returns:
        auc_score, feature_importance DataFrame
    """
    import lightgbm as lgb

    # Create labels: 0 for train, 1 for test
    train_df['is_test'] = 0
    test_df['is_test'] = 1

    # Combine datasets
    combined = pd.concat([train_df[features + ['is_test']],
                          test_df[features + ['is_test']]],
                         axis=0, ignore_index=True)

    X = combined[features]
    y = combined['is_test']

    # Train classifier to distinguish train from test
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42
    )

    # Cross-validation AUC
    # AUC close to 0.5 = train and test are similar (good)
    # AUC close to 1.0 = train and test are very different (bad)
    auc_scores = cross_val_score(
        model, X, y,
        cv=n_splits,
        scoring='roc_auc',
        n_jobs=-1
    )
    auc_score = np.mean(auc_scores)

    # Feature importance shows which features differ most
    model.fit(X, y)
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return auc_score, importance_df

# Usage
auc_score, adv_importance = adversarial_validation(train, test, feature_cols)
print(f"Adversarial Validation AUC: {auc_score:.4f}")
print("Top features causing distribution shift:")
print(adv_importance.head(10))

# If AUC > 0.7, consider:
# 1. Removing features with high adversarial importance
# 2. Using different CV strategy (e.g., stratify by problematic features)
# 3. Time-based split if temporal drift
```

---

## Cross-Validation Strategies

Proper cross-validation is critical for reliable performance estimation and preventing overfitting to the leaderboard.

### Standard Strategies

```python
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, TimeSeriesSplit, KFold
)

# 1. StratifiedKFold - preserves class distribution
# Use for: Classification with imbalanced classes
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train model
    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)

    # Validate
    y_pred = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, y_pred)
    print(f"Fold {fold}: AUC = {score:.4f}")

# 2. GroupKFold - prevents data leakage for grouped data
# Use for: Multiple rows per entity (customer, user, etc.)
# Example: Customer transactions - all transactions from same customer in same fold
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df['customer_id'])):
    # Ensures all transactions from a customer are in either train or validation, not both
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

# 3. TimeSeriesSplit - for temporal data
# Use for: Time series, when test data is in the future
tscv = TimeSeriesSplit(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    # Training set always comes before validation set
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
```

---

### Purged and Embargoed Cross-Validation

**For financial data** with temporal dependencies, use purged and embargoed CV to prevent lookahead bias.

```python
def purged_embargoed_cv(df, date_col, n_splits=5, embargo_days=7, purge_days=3):
    """
    Purged and embargoed cross-validation for financial time series.

    Args:
        df: DataFrame with datetime index or column
        date_col: Date column name
        n_splits: Number of folds
        embargo_days: Days to embargo after validation set
        purge_days: Days to purge before validation set

    Yields:
        train_idx, val_idx for each fold
    """
    df = df.sort_values(date_col).reset_index(drop=True)

    # Split into date ranges
    date_range = df[date_col].max() - df[date_col].min()
    fold_size = date_range / n_splits

    for fold in range(n_splits):
        # Validation period
        val_start = df[date_col].min() + fold * fold_size
        val_end = val_start + fold_size

        # Purge period (remove data just before validation)
        purge_start = val_start - pd.Timedelta(days=purge_days)

        # Embargo period (remove data just after validation)
        embargo_end = val_end + pd.Timedelta(days=embargo_days)

        # Indices
        train_idx = df[
            (df[date_col] < purge_start) | (df[date_col] > embargo_end)
        ].index.tolist()

        val_idx = df[
            (df[date_col] >= val_start) & (df[date_col] < val_end)
        ].index.tolist()

        yield train_idx, val_idx

# Usage
for fold, (train_idx, val_idx) in enumerate(
    purged_embargoed_cv(df, 'transaction_date', n_splits=5, embargo_days=7)
):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print(f"Fold {fold}: Train size = {len(train_idx)}, Val size = {len(val_idx)}")
```

---

### Complete CV Pipeline

```python
class CVTrainer:
    """Cross-validation training pipeline."""

    def __init__(self, model_class, params, cv_strategy, metric):
        self.model_class = model_class
        self.params = params
        self.cv_strategy = cv_strategy
        self.metric = metric
        self.models = []
        self.oof_predictions = None
        self.scores = []

    def fit(self, X, y):
        """Train with cross-validation."""
        self.oof_predictions = np.zeros(len(X))

        for fold, (train_idx, val_idx) in enumerate(self.cv_strategy.split(X, y)):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}")
            print(f"{'='*50}")

            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train model
            model = self.model_class(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
            )

            # Predict
            y_pred = model.predict_proba(X_val)[:, 1]
            self.oof_predictions[val_idx] = y_pred

            # Score
            score = self.metric(y_val, y_pred)
            self.scores.append(score)
            print(f"Fold {fold + 1} Score: {score:.4f}")

            # Save model
            self.models.append(model)

        # Overall score
        overall_score = self.metric(y, self.oof_predictions)
        print(f"\n{'='*50}")
        print(f"Overall CV Score: {overall_score:.4f}")
        print(f"Std: {np.std(self.scores):.4f}")
        print(f"{'='*50}")

        return self

    def predict(self, X_test):
        """Predict using all fold models."""
        predictions = np.zeros(len(X_test))

        for model in self.models:
            predictions += model.predict_proba(X_test)[:, 1]

        return predictions / len(self.models)

# Usage
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

params = {
    'n_estimators': 5000,
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'subsample': 0.8,
    'feature_fraction': 0.8,
    'random_state': 42,
    'verbose': -1
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

trainer = CVTrainer(
    model_class=lgb.LGBMClassifier,
    params=params,
    cv_strategy=cv_strategy,
    metric=roc_auc_score
)

trainer.fit(X_train, y_train)
test_predictions = trainer.predict(X_test)
```

---

## Winning Solution Breakdowns

### Amex Default Prediction (2022)

**Competition:** Predict credit default on a large-scale dataset with temporal features.

**Winning Approach:**

**1. Feature Engineering:**
```python
# Aggregation features across customer statements
def create_amex_features(df):
    """Create features for Amex Default Prediction."""
    # Customer-level aggregations
    agg_funcs = ['mean', 'std', 'min', 'max', 'last', 'first']

    # Numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    customer_feats = df.groupby('customer_id')[num_cols].agg(agg_funcs)
    customer_feats.columns = ['_'.join(col).strip() for col in customer_feats.columns]

    # Difference features (last - first)
    for col in num_cols:
        customer_feats[f'{col}_diff'] = (
            customer_feats[f'{col}_last'] - customer_feats[f'{col}_first']
        )

    # Trend features (linear regression slope)
    def calculate_trend(x):
        if len(x) < 2:
            return 0
        return np.polyfit(range(len(x)), x, 1)[0]

    trend_feats = df.groupby('customer_id')[num_cols].apply(calculate_trend)
    trend_feats.columns = [f'{col}_trend' for col in num_cols]

    return pd.concat([customer_feats, trend_feats], axis=1)
```

**2. Model Configuration:**
```python
# DART LightGBM for better generalization
lgb_params_amex = {
    'boosting_type': 'dart',
    'drop_rate': 0.05,
    'max_drop': 50,
    'skip_drop': 0.5,
    'num_leaves': 127,
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'min_child_samples': 40,
    'lambda_l1': 0.5,
    'lambda_l2': 2.0,
    'objective': 'binary',
    'metric': 'auc',
    'random_state': 42
}

# Multiple seeds for diversity
seeds = [42, 123, 456, 789, 2023]
models = []

for seed in seeds:
    lgb_params_amex['random_state'] = seed
    model = lgb.LGBMClassifier(**lgb_params_amex)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200)]
    )
    models.append(model)

# Average predictions
predictions = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
```

**3. Key Insights:**
- Aggregation across time (13 statements per customer) was crucial
- DART mode reduced overfitting better than standard GBDT
- Customer-level sequence patterns (first, last, difference, trend)
- 800+ engineered features
- Ensemble of 5 models with different seeds

---

### Home Credit Default Risk (2018)

**Competition:** Predict loan default using heterogeneous data from multiple tables.

**Winning Approach:**

**1. Massive Feature Engineering:**
```python
# Feature engineering from multiple tables
def home_credit_features(application_df, bureau_df, prev_application_df,
                         credit_card_df, installments_df):
    """
    Create features from Home Credit dataset.

    Tables:
    - application: Main application data
    - bureau: Credit bureau data
    - previous_application: Previous loan applications
    - credit_card_balance: Credit card balance history
    - installments_payments: Payment history
    """

    # Bureau features
    bureau_agg = bureau_df.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'CREDIT_DAY_OVERDUE': ['mean', 'max', 'sum'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'sum', 'max'],
        'CREDIT_TYPE': ['nunique'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean']
    })
    bureau_agg.columns = ['_'.join(col).strip() for col in bureau_agg.columns]

    # Bureau active credits
    bureau_active = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active']
    bureau_active_agg = bureau_active.groupby('SK_ID_CURR').agg({
        'AMT_CREDIT_SUM': ['sum', 'mean', 'max'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean', 'max']
    })
    bureau_active_agg.columns = ['active_' + '_'.join(col) for col in bureau_active_agg.columns]

    # Previous applications
    prev_agg = prev_application_df.groupby('SK_ID_CURR').agg({
        'AMT_ANNUITY': ['mean', 'max', 'min'],
        'AMT_APPLICATION': ['mean', 'max', 'min'],
        'AMT_CREDIT': ['mean', 'max', 'min'],
        'AMT_DOWN_PAYMENT': ['mean', 'max'],
        'HOUR_APPR_PROCESS_START': ['mean', 'min', 'max'],
        'RATE_DOWN_PAYMENT': ['mean', 'max', 'min'],
        'DAYS_DECISION': ['mean', 'min', 'max'],
        'CNT_PAYMENT': ['mean', 'sum', 'max']
    })
    prev_agg.columns = ['prev_' + '_'.join(col) for col in prev_agg.columns]

    # Approved vs rejected
    prev_approved = prev_application_df[prev_application_df['NAME_CONTRACT_STATUS'] == 'Approved']
    prev_refused = prev_application_df[prev_application_df['NAME_CONTRACT_STATUS'] == 'Refused']

    # Credit card balance
    cc_agg = credit_card_df.groupby('SK_ID_CURR').agg({
        'AMT_BALANCE': ['mean', 'max', 'min', 'var'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean', 'max'],
        'AMT_DRAWINGS_ATM_CURRENT': ['mean', 'sum', 'max'],
        'AMT_DRAWINGS_CURRENT': ['mean', 'sum', 'max'],
        'AMT_INST_MIN_REGULARITY': ['mean', 'max'],
        'AMT_PAYMENT_CURRENT': ['mean', 'sum', 'max'],
        'CNT_DRAWINGS_ATM_CURRENT': ['mean', 'sum', 'max'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'max', 'sum'],
        'SK_DPD': ['mean', 'max', 'sum'],
        'SK_DPD_DEF': ['mean', 'max', 'sum']
    })
    cc_agg.columns = ['cc_' + '_'.join(col) for col in cc_agg.columns]

    # Credit utilization
    credit_card_df['utilization'] = (
        credit_card_df['AMT_BALANCE'] / credit_card_df['AMT_CREDIT_LIMIT_ACTUAL']
    )
    util_agg = credit_card_df.groupby('SK_ID_CURR')['utilization'].agg(['mean', 'max', 'min'])

    # Installments
    inst_agg = installments_df.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION': ['nunique', 'max'],
        'NUM_INSTALMENT_NUMBER': ['max', 'mean'],
        'DAYS_INSTALMENT': ['max', 'mean', 'min'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'min'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['max', 'mean', 'sum']
    })
    inst_agg.columns = ['inst_' + '_'.join(col) for col in inst_agg.columns]

    # Payment differences
    installments_df['payment_diff'] = installments_df['AMT_PAYMENT'] - installments_df['AMT_INSTALMENT']
    installments_df['payment_ratio'] = installments_df['AMT_PAYMENT'] / (installments_df['AMT_INSTALMENT'] + 1)
    installments_df['days_diff'] = installments_df['DAYS_ENTRY_PAYMENT'] - installments_df['DAYS_INSTALMENT']

    # Merge all features
    application_df = application_df.merge(bureau_agg, left_on='SK_ID_CURR', right_index=True, how='left')
    application_df = application_df.merge(bureau_active_agg, left_on='SK_ID_CURR', right_index=True, how='left')
    application_df = application_df.merge(prev_agg, left_on='SK_ID_CURR', right_index=True, how='left')
    application_df = application_df.merge(cc_agg, left_on='SK_ID_CURR', right_index=True, how='left')
    application_df = application_df.merge(inst_agg, left_on='SK_ID_CURR', right_index=True, how='left')

    return application_df
```

**2. Model Stacking:**
```python
# Layer 1: Base models
base_models = [
    lgb.LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=3000),
    xgb.XGBClassifier(max_depth=7, learning_rate=0.05, n_estimators=3000),
    CatBoostClassifier(depth=6, learning_rate=0.05, iterations=3000)
]

# Layer 2: Meta model (logistic regression)
from sklearn.linear_model import LogisticRegression
meta_model = LogisticRegression()

# Train with stacking
from sklearn.model_selection import cross_val_predict

meta_features_train = []
for model in base_models:
    # Out-of-fold predictions
    oof_preds = cross_val_predict(
        model, X_train, y_train,
        cv=5, method='predict_proba', n_jobs=-1
    )[:, 1]
    meta_features_train.append(oof_preds)

meta_X_train = np.column_stack(meta_features_train)
meta_model.fit(meta_X_train, y_train)
```

**3. Key Insights:**
- 800+ features from multiple tables
- Heavy aggregation at customer level
- Stacking LightGBM + XGBoost + CatBoost
- Feature selection using permutation importance
- Careful handling of missing values (domain-specific)

---

### IEEE-CIS Fraud Detection (2019)

**Competition:** Detect fraudulent transactions using transaction and identity data.

**Winning Approach:**

**1. Time-Based Features:**
```python
def create_fraud_time_features(df):
    """Create time-based features for fraud detection."""

    # Time deltas from reference
    df['TransactionDT_day'] = df['TransactionDT'] / (24 * 60 * 60)
    df['TransactionDT_hour'] = (df['TransactionDT'] / 3600) % 24
    df['TransactionDT_dow'] = (df['TransactionDT'] / (24 * 60 * 60)) % 7

    # Frequency of transactions
    df['uid'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['uid_count'] = df.groupby('uid')['TransactionID'].transform('count')
    df['uid_TransactionAmt_mean'] = df.groupby('uid')['TransactionAmt'].transform('mean')
    df['uid_TransactionAmt_std'] = df.groupby('uid')['TransactionAmt'].transform('std')

    # Time since last transaction
    df = df.sort_values(['uid', 'TransactionDT'])
    df['time_since_last'] = df.groupby('uid')['TransactionDT'].diff()
    df['time_to_next'] = df.groupby('uid')['TransactionDT'].diff(-1).abs()

    # Velocity features
    df['transaction_velocity_1h'] = df.groupby('uid')['TransactionDT'].transform(
        lambda x: ((x.max() - x.min()) < 3600).sum()
    )
    df['transaction_velocity_24h'] = df.groupby('uid')['TransactionDT'].transform(
        lambda x: ((x.max() - x.min()) < 86400).sum()
    )

    return df

def create_entity_linking_features(df):
    """Entity linking to find related transactions."""

    # Create UIDs from different combinations
    uids = []

    # Card-based
    uids.append(df['card1'].astype(str) + '_' + df['card2'].astype(str))
    uids.append(df['card1'].astype(str) + '_' + df['addr1'].astype(str))

    # Email-based
    uids.append(df['P_emaildomain'].astype(str) + '_' + df['card1'].astype(str))

    # Device-based
    uids.append(df['DeviceInfo'].astype(str) + '_' + df['card1'].astype(str))

    # Address-based
    uids.append(df['addr1'].astype(str) + '_' + df['addr2'].astype(str))

    # Aggregate fraud rate for each UID
    for i, uid in enumerate(uids):
        uid_name = f'uid_{i}'
        df[uid_name] = uid

        # Historical fraud rate (use time-based split to avoid leakage)
        df[f'{uid_name}_fraud_rate'] = df.groupby(uid_name)['isFraud'].transform('mean')
        df[f'{uid_name}_transaction_count'] = df.groupby(uid_name)['TransactionID'].transform('count')

    return df
```

**2. Magic Features:**
```python
# "Magic" features found through EDA
def create_magic_features(df):
    """Create magic features discovered through EDA."""

    # Transaction amount patterns
    df['TransactionAmt_decimal'] = ((df['TransactionAmt'] - df['TransactionAmt'].astype(int)) * 1000).astype(int)
    df['TransactionAmt_is_round'] = (df['TransactionAmt'] == df['TransactionAmt'].round()).astype(int)

    # Card combinations
    df['card1_card2'] = df['card1'].astype(str) + '_' + df['card2'].astype(str)
    df['card3_card5'] = df['card3'].astype(str) + '_' + df['card5'].astype(str)

    # D columns (anonymous temporal features)
    d_cols = [col for col in df.columns if col.startswith('D')]
    for col in d_cols:
        df[f'{col}_isna'] = df[col].isna().astype(int)

    # V columns (Vesta engineered features)
    v_cols = [col for col in df.columns if col.startswith('V')]

    # Count missing V columns
    df['V_missing_count'] = df[v_cols].isna().sum(axis=1)

    # V column patterns
    for col in v_cols[:50]:  # First 50 V columns
        if df[col].dtype in [np.float64, np.int64]:
            df[f'{col}_to_mean_card1'] = df[col] / (df.groupby('card1')[col].transform('mean') + 1e-5)
            df[f'{col}_to_std_card1'] = df[col] / (df.groupby('card1')[col].transform('std') + 1e-5)

    return df
```

**3. Model Configuration:**
```python
# LightGBM with custom parameters
lgb_params_fraud = {
    'num_leaves': 256,
    'max_depth': -1,
    'learning_rate': 0.01,
    'n_estimators': 10000,
    'objective': 'binary',
    'metric': 'auc',
    'subsample': 0.9,
    'subsample_freq': 1,
    'feature_fraction': 0.9,
    'min_child_samples': 100,
    'lambda_l1': 1.0,
    'lambda_l2': 1.0,
    'random_state': 42
}

# Group K-Fold by card1 to prevent leakage
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)

for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=df['card1'])):
    model = lgb.LGBMClassifier(**lgb_params_fraud)
    model.fit(
        X.iloc[train_idx], y.iloc[train_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(200)]
    )
```

**4. Key Insights:**
- Entity linking (finding related transactions)
- Time-based features (velocity, time since last)
- Magic features from EDA (decimal patterns, missing patterns)
- GroupKFold by card1 to prevent leakage
- Heavy feature engineering on anonymous columns

---

### Santander Customer Transaction (2019)

**Competition:** Predict customer transactions with anonymized features.

**Winning Approach:**

**1. Augmentation Tricks:**
```python
def augment_data(X_train, y_train, augmentation_ratio=0.2):
    """
    Data augmentation for tabular data.
    Add Gaussian noise to positive class to balance dataset.
    """
    X_pos = X_train[y_train == 1]
    y_pos = y_train[y_train == 1]

    n_augment = int(len(X_pos) * augmentation_ratio)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, (n_augment, X_pos.shape[1]))
    X_augmented = X_pos.sample(n_augment, replace=True).values + noise
    y_augmented = np.ones(n_augment)

    # Combine
    X_train_aug = np.vstack([X_train, X_augmented])
    y_train_aug = np.concatenate([y_train, y_augmented])

    return X_train_aug, y_train_aug
```

**2. Magic Features:**
```python
def santander_magic_features(df):
    """
    Create magic features for Santander competition.
    Many features were normal distributions with different parameters.
    """
    # Count unique values per row (many rows were duplicates or near-duplicates)
    df['unique_count'] = df.nunique(axis=1)

    # Sum and mean of all features
    df['feature_sum'] = df.sum(axis=1)
    df['feature_mean'] = df.mean(axis=1)
    df['feature_std'] = df.std(axis=1)
    df['feature_max'] = df.max(axis=1)
    df['feature_min'] = df.min(axis=1)

    # Count zeros
    df['zero_count'] = (df == 0).sum(axis=1)

    # Count positive/negative
    df['positive_count'] = (df > 0).sum(axis=1)
    df['negative_count'] = (df < 0).sum(axis=1)

    # Statistical moments
    from scipy.stats import skew, kurtosis
    df['feature_skew'] = df.apply(lambda x: skew(x), axis=1)
    df['feature_kurtosis'] = df.apply(lambda x: kurtosis(x), axis=1)

    # Find "real vs fake" features
    # Some features were synthetic/augmented by organizers
    real_features = []
    for col in df.columns:
        if df[col].value_counts().iloc[0] / len(df) < 0.1:  # Not too many duplicates
            real_features.append(col)

    df['real_feature_mean'] = df[real_features].mean(axis=1)
    df['real_feature_std'] = df[real_features].std(axis=1)

    return df
```

**3. Adversarial Validation and Pseudo-Labeling:**
```python
# Remove train samples that are too similar to test
def remove_similar_to_test(X_train, y_train, X_test, threshold=0.7):
    """Use adversarial validation to remove train samples similar to test."""
    from sklearn.ensemble import RandomForestClassifier

    # Train adversarial model
    X_adv = pd.concat([X_train, X_test], axis=0)
    y_adv = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_test))])

    adv_model = RandomForestClassifier(n_estimators=100, random_state=42)
    adv_model.fit(X_adv, y_adv)

    # Predict probability that train samples are "test-like"
    train_pred = adv_model.predict_proba(X_train)[:, 1]

    # Keep only train samples that are not too test-like
    keep_idx = train_pred < threshold

    return X_train[keep_idx], y_train[keep_idx]

# Pseudo-labeling
def pseudo_labeling(X_train, y_train, X_test, threshold=0.99):
    """Add confident test predictions to training set."""
    # Train initial model
    model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Predict on test
    test_pred = model.predict_proba(X_test)[:, 1]

    # Get confident predictions
    confident_pos = test_pred > threshold
    confident_neg = test_pred < (1 - threshold)

    # Add to training set
    X_pseudo = X_test[confident_pos | confident_neg]
    y_pseudo = (test_pred[confident_pos | confident_neg] > 0.5).astype(int)

    X_train_new = pd.concat([X_train, X_pseudo], axis=0)
    y_train_new = np.concatenate([y_train, y_pseudo])

    return X_train_new, y_train_new
```

**4. Key Insights:**
- Data augmentation with Gaussian noise
- Statistical features (skew, kurtosis, moments)
- Identifying "real" vs "synthetic" features
- Adversarial validation to remove problematic samples
- Pseudo-labeling with high confidence threshold

---

## Ensemble Methods

Ensembling combines multiple models to improve performance and reduce variance.

### Averaging

**Simple averaging** and **weighted averaging** of predictions.

```python
# Simple averaging
def simple_average(predictions_list):
    """
    Simple average of predictions.

    Args:
        predictions_list: List of prediction arrays

    Returns:
        Averaged predictions
    """
    return np.mean(predictions_list, axis=0)

# Weighted averaging
def weighted_average(predictions_list, weights=None):
    """
    Weighted average of predictions.

    Args:
        predictions_list: List of prediction arrays
        weights: List of weights (must sum to 1)

    Returns:
        Weighted averaged predictions
    """
    if weights is None:
        weights = np.ones(len(predictions_list)) / len(predictions_list)

    return np.average(predictions_list, axis=0, weights=weights)

# Example
lgb_pred = lgb_model.predict_proba(X_test)[:, 1]
xgb_pred = xgb_model.predict_proba(X_test)[:, 1]
cat_pred = cat_model.predict_proba(X_test)[:, 1]

# Simple average
simple_avg_pred = simple_average([lgb_pred, xgb_pred, cat_pred])

# Weighted average (tuned on validation set)
weighted_avg_pred = weighted_average(
    [lgb_pred, xgb_pred, cat_pred],
    weights=[0.4, 0.35, 0.25]
)
```

---

### Rank Averaging

**Rank averaging** is robust to different scales and distributions.

```python
from scipy.stats import rankdata

def rank_average(predictions_list):
    """
    Rank average of predictions.
    Converts each prediction to ranks, then averages ranks.

    Args:
        predictions_list: List of prediction arrays

    Returns:
        Rank-averaged predictions
    """
    ranked_predictions = []

    for preds in predictions_list:
        # Convert to ranks (normalized to [0, 1])
        ranks = rankdata(preds) / len(preds)
        ranked_predictions.append(ranks)

    # Average ranks
    avg_ranks = np.mean(ranked_predictions, axis=0)

    return avg_ranks

# Example
rank_avg_pred = rank_average([lgb_pred, xgb_pred, cat_pred])
```

---

### Stacking

**Stacking** trains a meta-model on out-of-fold predictions from base models.

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Ridge

class StackingClassifier:
    """
    Two-layer stacking classifier.

    Layer 1: Base models
    Layer 2: Meta model
    """

    def __init__(self, base_models, meta_model, n_folds=5):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds

    def fit(self, X, y):
        """Train stacking ensemble."""
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Store out-of-fold predictions for meta model
        oof_predictions = np.zeros((len(X), len(self.base_models)))

        # Store fitted base models
        self.fitted_base_models = []

        # Train base models with CV
        for i, model in enumerate(self.base_models):
            print(f"Training base model {i+1}/{len(self.base_models)}")

            fold_models = []
            oof_preds = np.zeros(len(X))

            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Clone and fit model
                from copy import deepcopy
                fold_model = deepcopy(model)
                fold_model.fit(X_train, y_train)

                # Out-of-fold predictions
                oof_preds[val_idx] = fold_model.predict_proba(X_val)[:, 1]

                fold_models.append(fold_model)

            oof_predictions[:, i] = oof_preds
            self.fitted_base_models.append(fold_models)

        # Train meta model on out-of-fold predictions
        print("Training meta model")
        self.meta_model.fit(oof_predictions, y)

        return self

    def predict_proba(self, X):
        """Predict using stacking ensemble."""
        # Get predictions from base models
        base_predictions = np.zeros((len(X), len(self.base_models)))

        for i, fold_models in enumerate(self.fitted_base_models):
            # Average predictions across folds
            fold_preds = np.zeros(len(X))
            for fold_model in fold_models:
                fold_preds += fold_model.predict_proba(X)[:, 1]
            base_predictions[:, i] = fold_preds / len(fold_models)

        # Meta model prediction
        meta_pred = self.meta_model.predict_proba(base_predictions)

        return meta_pred

    def predict(self, X):
        """Predict class labels."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

# Example usage
base_models = [
    lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=31),
    xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=6),
    CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, verbose=0)
]

meta_model = LogisticRegression(max_iter=1000, C=1.0)

stacker = StackingClassifier(base_models, meta_model, n_folds=5)
stacker.fit(X_train, y_train)
stacked_pred = stacker.predict_proba(X_test)[:, 1]
```

---

### Hill Climbing for Optimal Weights

**Hill climbing** finds optimal ensemble weights through iterative search.

```python
def hill_climbing_weights(predictions_list, y_true, metric, n_iterations=1000):
    """
    Find optimal weights for ensemble using hill climbing.

    Args:
        predictions_list: List of prediction arrays
        y_true: True labels
        metric: Scoring function (higher is better)
        n_iterations: Number of hill climbing iterations

    Returns:
        optimal_weights, best_score
    """
    n_models = len(predictions_list)

    # Initialize with equal weights
    best_weights = np.ones(n_models) / n_models
    best_pred = weighted_average(predictions_list, best_weights)
    best_score = metric(y_true, best_pred)

    for iteration in range(n_iterations):
        # Random perturbation
        new_weights = best_weights + np.random.normal(0, 0.01, n_models)

        # Ensure weights are positive and sum to 1
        new_weights = np.maximum(new_weights, 0)
        new_weights = new_weights / np.sum(new_weights)

        # Evaluate
        new_pred = weighted_average(predictions_list, new_weights)
        new_score = metric(y_true, new_pred)

        # Update if better
        if new_score > best_score:
            best_weights = new_weights
            best_score = new_score
            print(f"Iteration {iteration}: Score = {best_score:.6f}, Weights = {best_weights}")

    return best_weights, best_score

# Example usage
from sklearn.metrics import roc_auc_score

optimal_weights, optimal_score = hill_climbing_weights(
    predictions_list=[lgb_pred_val, xgb_pred_val, cat_pred_val],
    y_true=y_val,
    metric=roc_auc_score,
    n_iterations=1000
)

print(f"Optimal weights: {optimal_weights}")
print(f"Optimal score: {optimal_score:.6f}")

# Apply to test predictions
final_pred = weighted_average(
    [lgb_pred_test, xgb_pred_test, cat_pred_test],
    weights=optimal_weights
)
```

---

## Post-Processing Tricks

Post-processing can squeeze out extra performance without retraining models.

### Threshold Optimization

**Optimize classification threshold** for metrics like F1, precision, recall.

```python
from sklearn.metrics import f1_score, precision_score, recall_score

def optimize_threshold(y_true, y_pred_proba, metric=f1_score):
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Scoring function

    Returns:
        optimal_threshold, best_score
    """
    best_threshold = 0.5
    best_score = 0

    # Search thresholds from 0.1 to 0.9
    for threshold in np.linspace(0.1, 0.9, 81):
        y_pred = (y_pred_proba >= threshold).astype(int)
        score = metric(y_true, y_pred)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score

# Example
optimal_threshold, f1 = optimize_threshold(y_val, val_pred_proba, metric=f1_score)
print(f"Optimal threshold: {optimal_threshold:.3f}, F1: {f1:.4f}")

# Apply to test
test_pred_binary = (test_pred_proba >= optimal_threshold).astype(int)
```

---

### Calibration

**Calibration** adjusts predicted probabilities to match true frequencies.

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression

# Platt scaling (logistic calibration)
def platt_scaling(y_true, y_pred_proba):
    """
    Calibrate probabilities using Platt scaling (logistic regression).

    Args:
        y_true: True labels
        y_pred_proba: Uncalibrated probabilities

    Returns:
        Calibrated probabilities
    """
    lr = LogisticRegression()
    lr.fit(y_pred_proba.reshape(-1, 1), y_true)
    calibrated = lr.predict_proba(y_pred_proba.reshape(-1, 1))[:, 1]
    return calibrated

# Isotonic regression
def isotonic_calibration(y_true, y_pred_proba):
    """
    Calibrate probabilities using isotonic regression.

    Args:
        y_true: True labels
        y_pred_proba: Uncalibrated probabilities

    Returns:
        Calibrated probabilities
    """
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(y_pred_proba, y_true)
    calibrated = iso.predict(y_pred_proba)
    return calibrated

# Example
val_pred_calibrated = platt_scaling(y_val, val_pred_proba)

# Apply same calibration to test
lr = LogisticRegression()
lr.fit(val_pred_proba.reshape(-1, 1), y_val)
test_pred_calibrated = lr.predict_proba(test_pred_proba.reshape(-1, 1))[:, 1]
```

---

### Distribution Matching

**Match test prediction distribution** to historical target distribution.

```python
def match_distribution(test_pred, target_mean):
    """
    Adjust predictions to match target mean.

    Args:
        test_pred: Test predictions
        target_mean: Target mean (from training data)

    Returns:
        Adjusted predictions
    """
    current_mean = test_pred.mean()

    # Shift predictions to match target mean
    shift = target_mean - current_mean
    adjusted = test_pred + shift

    # Clip to [0, 1]
    adjusted = np.clip(adjusted, 0, 1)

    return adjusted

# Example
train_mean = y_train.mean()
test_pred_adjusted = match_distribution(test_pred, train_mean)
```

---

## Neural Networks for Tabular

While gradient boosting dominates, neural networks can win on specific tabular tasks.

### TabNet

**TabNet** uses sequential attention for feature selection.

```python
# Install: pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetClassifier
import torch

# TabNet parameters
tabnet_params = {
    'n_d': 64,                      # Width of decision prediction layer
    'n_a': 64,                      # Width of attention embedding
    'n_steps': 5,                   # Number of steps in architecture
    'gamma': 1.5,                   # Coefficient for feature reusage
    'n_independent': 2,             # Number of independent GLU layers
    'n_shared': 2,                  # Number of shared GLU layers
    'lambda_sparse': 1e-4,          # Sparsity regularization
    'optimizer_fn': torch.optim.Adam,
    'optimizer_params': {'lr': 2e-2},
    'scheduler_params': {
        'step_size': 50,
        'gamma': 0.9
    },
    'scheduler_fn': torch.optim.lr_scheduler.StepLR,
    'mask_type': 'entmax',          # 'sparsemax' or 'entmax'
    'seed': 42,
    'verbose': 10
}

# Train TabNet
tabnet = TabNetClassifier(**tabnet_params)
tabnet.fit(
    X_train.values, y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_metric=['auc'],
    max_epochs=200,
    patience=50,
    batch_size=1024,
    virtual_batch_size=128
)

# Predict
tabnet_pred = tabnet.predict_proba(X_test.values)[:, 1]

# Feature importance
feature_importance = tabnet.feature_importances_
```

---

### When Neural Networks Beat GBMs

**Neural networks can outperform gradient boosting when:**

1. **Very large datasets** (millions of rows)
   - Neural networks scale better with data size
   - Example: 10M+ rows with 100+ features

2. **High-cardinality categorical features**
   - Embeddings learn better representations
   - Example: User IDs, product IDs with 100k+ unique values

3. **Complex interactions**
   - Deep networks capture higher-order interactions
   - Example: Recommendation systems, NLP-heavy tabular

4. **Sequential/temporal patterns**
   - RNNs/Transformers for time-series tabular data
   - Example: Customer behavior sequences

5. **Multi-modal inputs**
   - Combining tabular with images, text, audio
   - Example: Product matching with text + features

**Best practices for neural networks on tabular:**
- Use embeddings for categorical features
- Batch normalization after each layer
- Dropout for regularization (0.1-0.3)
- Learning rate scheduling
- Ensemble with gradient boosting

```python
# Simple neural network for tabular data
import torch
import torch.nn as nn

class TabularNN(nn.Module):
    def __init__(self, num_features, hidden_sizes=[256, 128, 64], dropout=0.2):
        super(TabularNN, self).__init__()

        layers = []
        in_size = num_features

        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_size = hidden_size

        layers.append(nn.Linear(in_size, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Training loop
model = TabularNN(num_features=X_train.shape[1])
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    optimizer.zero_grad()

    outputs = model(torch.FloatTensor(X_train.values))
    loss = criterion(outputs, torch.FloatTensor(y_train.values).unsqueeze(1))
    loss.backward()
    optimizer.step()
```

---

## Imbalanced Data Handling

Most tabular competitions have imbalanced target distributions.

### SMOTE and ADASYN

**SMOTE** (Synthetic Minority Over-sampling Technique) generates synthetic examples.

```python
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# SMOTE
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# ADASYN (Adaptive Synthetic Sampling)
adasyn = ADASYN(sampling_strategy=0.5, random_state=42)
X_resampled, y_resampled = adasyn.fit_resample(X_train, y_train)

# Combined over and under sampling
over_under = Pipeline([
    ('over', SMOTE(sampling_strategy=0.5)),
    ('under', RandomUnderSampler(sampling_strategy=0.8))
])
X_resampled, y_resampled = over_under.fit_resample(X_train, y_train)
```

---

### Class Weighting in Gradient Boosting

```python
# XGBoost: scale_pos_weight
from sklearn.utils.class_weight import compute_class_weight

# Calculate scale_pos_weight
scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

xgb_params = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 3000,
    'scale_pos_weight': scale_pos_weight,  # Weight for positive class
    'eval_metric': 'auc'
}

# LightGBM: is_unbalance or scale_pos_weight
lgb_params = {
    'num_leaves': 31,
    'learning_rate': 0.05,
    'n_estimators': 3000,
    'is_unbalance': True,  # Automatically balance weights
    # OR
    'scale_pos_weight': scale_pos_weight,
    'metric': 'auc'
}

# CatBoost: class_weights
class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
cat_params = {
    'depth': 6,
    'learning_rate': 0.05,
    'iterations': 3000,
    'class_weights': class_weights,
    'eval_metric': 'AUC'
}
```

---

### Focal Loss

**Focal loss** focuses on hard-to-classify examples.

```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.

    Args:
        y_true: True labels
        y_pred: Predicted probabilities
        alpha: Balance parameter (0.25 is common)
        gamma: Focusing parameter (2.0 is common)

    Returns:
        Focal loss
    """
    epsilon = 1e-7
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    cross_entropy = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    focal_weight = alpha * y_true * (1 - y_pred) ** gamma + \
                   (1 - alpha) * (1 - y_true) * y_pred ** gamma

    return focal_weight * cross_entropy

# Custom objective for LightGBM
def lgb_focal_loss(y_true, y_pred):
    """Focal loss objective for LightGBM."""
    alpha = 0.25
    gamma = 2.0

    epsilon = 1e-7
    y_pred = 1 / (1 + np.exp(-y_pred))  # Sigmoid
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # Gradient
    grad = alpha * (1 - y_true) * y_pred ** gamma * (gamma * (1 - y_pred) * np.log(1 - y_pred) + y_pred) - \
           (1 - alpha) * y_true * (1 - y_pred) ** gamma * (gamma * y_pred * np.log(y_pred) + (1 - y_pred))

    # Hessian (approximate)
    hess = alpha * (1 - y_true) * y_pred ** (gamma - 1) * (gamma * (1 - 2 * y_pred) + y_pred) + \
           (1 - alpha) * y_true * (1 - y_pred) ** (gamma - 1) * (gamma * (2 * y_pred - 1) + (1 - y_pred))

    return grad, hess

# Use with LightGBM
model = lgb.train(
    params,
    train_set,
    valid_sets=[val_set],
    fobj=lgb_focal_loss,
    num_boost_round=3000
)
```

---

## Resources

### Documentation
- **XGBoost:** https://xgboost.readthedocs.io/
- **LightGBM:** https://lightgbm.readthedocs.io/
- **CatBoost:** https://catboost.ai/docs/
- **TabNet:** https://github.com/dreamquark-ai/tabnet

### Kaggle Winning Solutions
- **Kaggle Solutions:** https://www.kaggle.com/competitions (Solutions tab)
- **Kaggle Past Solutions:** https://github.com/interviewBubble/Kaggle-Competition-Solutions-and-Ideas

### Books
- **Feature Engineering for Machine Learning** by Alice Zheng
- **The Kaggle Book** by Konrad Banachewicz
- **Approaching (Almost) Any Machine Learning Problem** by Abhishek Thakur

### Tools
- **AutoML:** H2O AutoML, Auto-sklearn, FLAML
- **Feature Engineering:** Featuretools, tsfresh
- **Imbalanced Learning:** imbalanced-learn

