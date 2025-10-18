# Data Preprocessing for Machine Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Data Loading and Validation](#data-loading-and-validation)
3. [Missing Value Strategies](#missing-value-strategies)
4. [Outlier Detection and Handling](#outlier-detection-and-handling)
5. [Feature Scaling](#feature-scaling)
6. [Encoding Categorical Variables](#encoding-categorical-variables)
7. [Feature Engineering Pipelines](#feature-engineering-pipelines)
8. [sklearn Pipeline and ColumnTransformer](#sklearn-pipeline-and-columntransformer)
9. [Custom Transformers](#custom-transformers)
10. [Data Validation with Great Expectations](#data-validation-with-great-expectations)
11. [Production Preprocessing Pipelines](#production-preprocessing-pipelines)
12. [Complete Implementations](#complete-implementations)

---

## Introduction

Data preprocessing is the foundation of machine learning success. Poor data quality leads to poor models, regardless of algorithm sophistication. This guide covers production-ready preprocessing techniques used in industry and research as of 2025.

**Key Principle:** "Garbage in, garbage out" - preprocessing can make or break your model.

### Why Preprocessing Matters

- **Models expect clean, scaled data:** Most algorithms assume features are on similar scales
- **Missing values break algorithms:** sklearn raises errors on NaN values
- **Categorical data needs encoding:** Algorithms require numerical inputs
- **Outliers skew learning:** Can dominate loss functions and gradients
- **Consistency across train/test:** Same transformations must apply to new data

### 2025 Best Practices

1. **Always validate data before training** (Great Expectations, Pandera)
2. **Fit transformers only on training data** (prevent data leakage)
3. **Use sklearn Pipeline** for reproducibility
4. **Version preprocessing code** alongside models
5. **Monitor data drift in production** (Evidently AI, WhyLabs)

---

## Data Loading and Validation

### Loading Data

```python
import pandas as pd
import numpy as np
from pathlib import Path

# CSV files
df = pd.read_csv('data.csv',
                 parse_dates=['date_column'],
                 dtype={'id': str, 'amount': float},
                 na_values=['NA', 'missing', ''])

# Parquet (preferred for large datasets - 10x faster, 10x smaller)
df = pd.read_parquet('data.parquet')

# SQL database
from sqlalchemy import create_engine
engine = create_engine('postgresql://user:password@localhost/db')
df = pd.read_sql_query("SELECT * FROM table WHERE date > '2024-01-01'", engine)

# Multiple files efficiently
from glob import glob
files = glob('data/*.csv')
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
```

### Basic Data Validation

```python
def validate_dataframe(df, required_columns, date_columns=None):
    """
    Validate DataFrame structure and basic properties.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    required_columns : list
        Columns that must be present
    date_columns : list, optional
        Columns that should be datetime

    Returns:
    --------
    dict : Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': []
    }

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        results['valid'] = False
        results['errors'].append(f"Missing columns: {missing_cols}")

    # Check for duplicate rows
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        results['warnings'].append(f"Found {n_duplicates} duplicate rows")

    # Check date columns
    if date_columns:
        for col in date_columns:
            if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
                results['warnings'].append(f"Column {col} is not datetime type")

    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        results['warnings'].append(f"Columns with all null values: {null_cols}")

    # Basic statistics
    results['n_rows'] = len(df)
    results['n_cols'] = len(df.columns)
    results['memory_mb'] = df.memory_usage(deep=True).sum() / 1024**2

    return results

# Example usage
validation = validate_dataframe(
    df,
    required_columns=['id', 'feature1', 'target'],
    date_columns=['created_at', 'updated_at']
)

if not validation['valid']:
    raise ValueError(f"Data validation failed: {validation['errors']}")
```

### Data Profiling

```python
def profile_data(df):
    """Generate comprehensive data profile."""

    profile = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicates': df.duplicated().sum(),
        'columns': {}
    }

    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'missing_count': df[col].isnull().sum(),
            'missing_pct': df[col].isnull().mean() * 100,
            'unique_count': df[col].nunique(),
            'unique_pct': df[col].nunique() / len(df) * 100
        }

        # Numeric columns
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info.update({
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'median': df[col].median(),
                'skew': df[col].skew(),
                'kurtosis': df[col].kurtosis()
            })

        # Categorical columns
        elif pd.api.types.is_object_dtype(df[col]):
            value_counts = df[col].value_counts()
            col_info.update({
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else None,
                'cardinality': df[col].nunique()
            })

        profile['columns'][col] = col_info

    return profile

# Usage
profile = profile_data(df)

# Print summary
print(f"Dataset shape: {profile['shape']}")
print(f"Memory usage: {profile['memory_usage_mb']:.2f} MB")
print(f"Duplicate rows: {profile['duplicates']}")

# Check high missing values
high_missing = {col: info['missing_pct']
                for col, info in profile['columns'].items()
                if info['missing_pct'] > 50}
if high_missing:
    print(f"\nColumns with >50% missing: {high_missing}")
```

---

## Missing Value Strategies

### Understanding Missing Data Types

1. **MCAR (Missing Completely at Random):** Missing values are random, no pattern
2. **MAR (Missing at Random):** Missingness depends on observed data
3. **MNAR (Missing Not at Random):** Missingness depends on unobserved data

**Strategy depends on missing type and percentage.**

### Simple Imputation

```python
from sklearn.impute import SimpleImputer

# Mean imputation (for numeric features, MCAR)
imputer_mean = SimpleImputer(strategy='mean')
df['age_imputed'] = imputer_mean.fit_transform(df[['age']])

# Median imputation (robust to outliers)
imputer_median = SimpleImputer(strategy='median')
df['income_imputed'] = imputer_median.fit_transform(df[['income']])

# Most frequent (for categorical)
imputer_mode = SimpleImputer(strategy='most_frequent')
df['category_imputed'] = imputer_mode.fit_transform(df[['category']])

# Constant value
imputer_constant = SimpleImputer(strategy='constant', fill_value='Unknown')
df['status_imputed'] = imputer_constant.fit_transform(df[['status']])
```

### Advanced Imputation

```python
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor

# KNN Imputation (uses similar samples)
# Best for: Numeric data with patterns, moderate missing values (<20%)
knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
X_imputed_knn = knn_imputer.fit_transform(X_numeric)

# Iterative Imputation (MICE - Multivariate Imputation by Chained Equations)
# Best for: Complex relationships, MAR data
iterative_imputer = IterativeImputer(
    estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    max_iter=10,
    random_state=42
)
X_imputed_iterative = iterative_imputer.fit_transform(X_numeric)

# Example: Compare imputation methods
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# Create data with missing values
X, y = make_regression(n_samples=1000, n_features=5, random_state=42)
X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])

# Introduce missing values (20% MCAR)
mask = np.random.random(X_df.shape) < 0.2
X_missing = X_df.copy()
X_missing[mask] = np.nan

# Apply different imputation strategies
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'Median': SimpleImputer(strategy='median'),
    'KNN': KNNImputer(n_neighbors=5),
    'Iterative': IterativeImputer(max_iter=10, random_state=42)
}

results = {}
for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    # Calculate RMSE vs original
    rmse = np.sqrt(np.mean((X - X_imputed)**2))
    results[name] = rmse
    print(f"{name} imputation RMSE: {rmse:.4f}")
```

### Missing Value Indicator

```python
from sklearn.impute import MissingIndicator

# Add binary indicator for missing values (useful information)
indicator = MissingIndicator()
missing_mask = indicator.fit_transform(X)

# Combine imputation with indicator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

imputer_with_indicator = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', add_indicator=True))
])

X_with_indicator = imputer_with_indicator.fit_transform(X)
# Now X_with_indicator contains both imputed values and missing indicators
```

### When to Drop vs Impute

```python
def handle_missing_values(df, threshold=0.5):
    """
    Strategy for handling missing values.

    Parameters:
    -----------
    df : pd.DataFrame
    threshold : float
        Drop columns with missing % above threshold

    Returns:
    --------
    pd.DataFrame : Processed DataFrame
    """
    # 1. Drop columns with too many missing values
    missing_pct = df.isnull().mean()
    cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()

    if cols_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # 2. Drop rows if few missing
    rows_with_missing = df.isnull().any(axis=1).sum()
    if rows_with_missing / len(df) < 0.01:  # Less than 1%
        print(f"Dropping {rows_with_missing} rows with missing values (<1%)")
        df = df.dropna()
    else:
        # 3. Impute remaining missing values
        print("Imputing remaining missing values")

        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Impute numeric with median
        if len(numeric_cols) > 0:
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

        # Impute categorical with mode
        if len(categorical_cols) > 0:
            df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

    return df
```

---

## Outlier Detection and Handling

### Statistical Methods

```python
def detect_outliers_iqr(df, column, k=1.5):
    """
    Detect outliers using Interquartile Range (IQR) method.

    Parameters:
    -----------
    df : pd.DataFrame
    column : str
    k : float
        IQR multiplier (1.5 is standard, 3.0 is more conservative)

    Returns:
    --------
    pd.Series : Boolean mask of outliers
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)

    print(f"Column: {column}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    print(f"  Outliers: {outliers.sum()} ({outliers.mean()*100:.2f}%)")

    return outliers

def detect_outliers_zscore(df, column, threshold=3):
    """
    Detect outliers using Z-score method.

    Parameters:
    -----------
    df : pd.DataFrame
    column : str
    threshold : float
        Number of standard deviations (3 is standard)

    Returns:
    --------
    pd.Series : Boolean mask of outliers
    """
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    outliers = z_scores > threshold

    print(f"Column: {column}")
    print(f"  Mean: {df[column].mean():.2f}, Std: {df[column].std():.2f}")
    print(f"  Outliers: {outliers.sum()} ({outliers.mean()*100:.2f}%)")

    return outliers

# Example usage
outliers_iqr = detect_outliers_iqr(df, 'income', k=1.5)
outliers_zscore = detect_outliers_zscore(df, 'income', threshold=3)
```

### Machine Learning Methods

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope

# Isolation Forest (best for high-dimensional data)
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of outliers
    random_state=42,
    n_estimators=100
)
outliers_iso = iso_forest.fit_predict(X)  # -1 for outliers, 1 for inliers

# Local Outlier Factor (density-based)
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1
)
outliers_lof = lof.fit_predict(X)

# Elliptic Envelope (assumes Gaussian distribution)
elliptic = EllipticEnvelope(contamination=0.1, random_state=42)
outliers_elliptic = elliptic.fit_predict(X)

# Visualize outliers (2D example)
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

methods = [
    ('Isolation Forest', outliers_iso),
    ('Local Outlier Factor', outliers_lof),
    ('Elliptic Envelope', outliers_elliptic)
]

for ax, (name, outliers) in zip(axes, methods):
    inliers = outliers == 1
    ax.scatter(X[inliers, 0], X[inliers, 1], c='blue', label='Inliers', alpha=0.6)
    ax.scatter(X[~inliers, 0], X[~inliers, 1], c='red', label='Outliers', alpha=0.6)
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.show()
```

### Handling Outliers

```python
def handle_outliers(df, column, method='clip', k=1.5):
    """
    Handle outliers using various strategies.

    Parameters:
    -----------
    df : pd.DataFrame
    column : str
    method : str
        'remove', 'clip', 'transform', 'winsorize'
    k : float
        IQR multiplier

    Returns:
    --------
    pd.DataFrame : Processed DataFrame
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR

    if method == 'remove':
        # Remove outliers
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    elif method == 'clip':
        # Clip to bounds (recommended for most cases)
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    elif method == 'transform':
        # Log transformation (for right-skewed data)
        df[column] = np.log1p(df[column])

    elif method == 'winsorize':
        # Replace outliers with percentile values
        from scipy.stats import mstats
        df[column] = mstats.winsorize(df[column], limits=[0.05, 0.05])

    return df

# Example: Try different methods
df_clipped = handle_outliers(df.copy(), 'income', method='clip')
df_transformed = handle_outliers(df.copy(), 'income', method='transform')
df_winsorized = handle_outliers(df.copy(), 'income', method='winsorize')
```

---

## Feature Scaling

### Why Scaling Matters

- **Gradient-based algorithms:** (Linear/Logistic Regression, Neural Networks) converge faster
- **Distance-based algorithms:** (KNN, SVM, K-Means) require features on same scale
- **Regularization:** (Lasso, Ridge) penalizes features unequally if scales differ
- **Tree-based algorithms:** (Random Forest, XGBoost) DON'T require scaling

### StandardScaler (Z-score Normalization)

```python
from sklearn.preprocessing import StandardScaler

# Standardize features: mean=0, std=1
scaler = StandardScaler()

# Fit on training data only!
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Or fit_transform for training (shorthand)
X_train_scaled = scaler.fit_transform(X_train)

# Check results
print(f"Mean: {X_train_scaled.mean(axis=0)}")  # Should be ~0
print(f"Std: {X_train_scaled.std(axis=0)}")    # Should be ~1

# When to use: Default choice for most algorithms (Linear, NN, SVM)
```

### MinMaxScaler (Normalization)

```python
from sklearn.preprocessing import MinMaxScaler

# Scale features to [0, 1] range (or custom range)
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X_train)

# Custom range (e.g., [-1, 1] for neural networks with tanh)
scaler_tanh = MinMaxScaler(feature_range=(-1, 1))
X_scaled_tanh = scaler_tanh.fit_transform(X_train)

# Formula: X_scaled = (X - X_min) / (X_max - X_min)

# When to use:
# - Features have bounded distribution
# - Neural networks (faster convergence in [0,1] or [-1,1])
# - No outliers (outliers will compress most values to small range)
```

### RobustScaler (Robust to Outliers)

```python
from sklearn.preprocessing import RobustScaler

# Scale using median and IQR (robust to outliers)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_train)

# Formula: X_scaled = (X - median) / IQR

# When to use:
# - Data contains outliers that you want to preserve
# - Prefer over StandardScaler when outliers present
```

### MaxAbsScaler (For Sparse Data)

```python
from sklearn.preprocessing import MaxAbsScaler

# Scale by maximum absolute value (preserves sparsity)
scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_train_sparse)

# Formula: X_scaled = X / max(abs(X))
# Range: [-1, 1]

# When to use:
# - Sparse data (doesn't destroy sparsity like StandardScaler)
# - Already centered at zero
```

### Normalizer (Row-wise Scaling)

```python
from sklearn.preprocessing import Normalizer

# Scale each sample (row) to unit norm
normalizer = Normalizer(norm='l2')  # 'l1', 'l2', or 'max'
X_normalized = normalizer.fit_transform(X)

# Formula (L2): X_normalized = X / sqrt(sum(X^2))

# When to use:
# - Text classification (TF-IDF vectors)
# - Neural embeddings
# - Focus on direction rather than magnitude
```

### Comparison Example

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import (StandardScaler, MinMaxScaler,
                                   RobustScaler, Normalizer)

# Generate sample data with outliers
np.random.seed(42)
X = np.concatenate([
    np.random.randn(100, 1) * 10 + 50,  # Normal data
    np.array([[200], [250], [-50]])      # Outliers
])

# Apply different scalers
scalers = {
    'Original': None,
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

fig, axes = plt.subplots(1, len(scalers), figsize=(20, 4))

for ax, (name, scaler) in zip(axes, scalers.items()):
    if scaler:
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X

    ax.hist(X_scaled, bins=30, edgecolor='black')
    ax.set_title(f'{name}\nMean: {X_scaled.mean():.2f}, Std: {X_scaled.std():.2f}')
    ax.axvline(X_scaled.mean(), color='red', linestyle='--', label='Mean')

plt.tight_layout()
plt.show()
```

---

## Encoding Categorical Variables

### One-Hot Encoding

```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Manual with pandas (simple)
df_encoded = pd.get_dummies(df, columns=['category', 'region'], drop_first=False)

# With sklearn (more control, works in pipelines)
encoder = OneHotEncoder(
    drop='first',            # Drop first category to avoid multicollinearity
    sparse_output=False,     # Return dense array (sparse=True for high cardinality)
    handle_unknown='ignore'  # Ignore unknown categories in test set
)

X_encoded = encoder.fit_transform(X_categorical)

# Get feature names
feature_names = encoder.get_feature_names_out(['category', 'region'])

# When to use:
# - Low cardinality (<10 categories)
# - Tree-based models (can handle high cardinality)
# - No ordinal relationship
```

### Label Encoding

```python
from sklearn.preprocessing import LabelEncoder

# Encode labels as integers
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Example
categories = ['red', 'blue', 'green', 'blue', 'red']
encoded = encoder.fit_transform(categories)
print(encoded)  # [2, 0, 1, 0, 2]

# Decode back
decoded = encoder.inverse_transform(encoded)
print(decoded)  # ['red', 'blue', 'green', 'blue', 'red']

# When to use:
# - Target variable encoding (classification)
# - Ordinal features (low, medium, high)
# - NOT for nominal features (creates false ordering)
```

### Ordinal Encoding

```python
from sklearn.preprocessing import OrdinalEncoder

# Encode with explicit ordering
encoder = OrdinalEncoder(
    categories=[['low', 'medium', 'high']],
    handle_unknown='use_encoded_value',
    unknown_value=-1
)

X_ordinal = encoder.fit_transform(df[['priority']])

# When to use:
# - Features with natural ordering
# - Tree-based models (better than one-hot for high cardinality)
```

### Target Encoding (Mean Encoding)

```python
from category_encoders import TargetEncoder

# Encode based on target mean for each category
encoder = TargetEncoder(
    cols=['category', 'region'],
    smoothing=1.0  # Regularization (higher = more smoothing)
)

# Fit on train only
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)

# When to use:
# - High cardinality features (hundreds of categories)
# - Strong relationship between category and target
# - Be careful of overfitting (use cross-validation)

# Example with cross-validation to prevent overfitting
from sklearn.model_selection import KFold

def target_encode_cv(X, y, column, n_splits=5):
    """Target encoding with cross-validation to prevent overfitting."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_encoded = X.copy()

    # Global mean as fallback
    global_mean = y.mean()

    for train_idx, val_idx in kf.split(X):
        # Calculate means on training fold
        means = X.iloc[train_idx].groupby(column)[column].apply(
            lambda x: y.iloc[train_idx][x.index].mean()
        )

        # Apply to validation fold
        X_encoded.loc[val_idx, f'{column}_encoded'] = (
            X.loc[val_idx, column].map(means).fillna(global_mean)
        )

    return X_encoded
```

### Frequency Encoding

```python
def frequency_encoding(df, column):
    """
    Encode categories by their frequency.

    When to use:
    - High cardinality
    - Frequency correlates with target
    """
    freq = df[column].value_counts(normalize=True)
    df[f'{column}_freq'] = df[column].map(freq)
    return df

# Example
df = frequency_encoding(df, 'product_id')
```

### Binary Encoding

```python
from category_encoders import BinaryEncoder

# Convert to binary representation (more compact than one-hot)
encoder = BinaryEncoder(cols=['high_cardinality_feature'])
X_binary = encoder.fit_transform(X)

# Example: 5 categories need only 3 binary columns (vs 5 for one-hot)
# Category A = 001, B = 010, C = 011, D = 100, E = 101

# When to use:
# - Medium cardinality (10-100 categories)
# - Want to reduce dimensionality vs one-hot
```

### Hash Encoding

```python
from category_encoders import HashingEncoder

# Hash categories to fixed number of columns
encoder = HashingEncoder(n_components=10, cols=['category'])
X_hashed = encoder.fit_transform(X)

# When to use:
# - Very high cardinality (1000s of categories)
# - Memory constrained
# - Acceptable to have collisions
```

---

## Feature Engineering Pipelines

### Creating New Features

```python
import pandas as pd
import numpy as np

def create_features(df):
    """
    Comprehensive feature engineering.

    Techniques:
    - Interactions
    - Aggregations
    - Date features
    - Domain-specific
    """
    df = df.copy()

    # 1. Interaction features
    df['feature1_x_feature2'] = df['feature1'] * df['feature2']
    df['feature1_div_feature2'] = df['feature1'] / (df['feature2'] + 1e-8)
    df['feature1_plus_feature2'] = df['feature1'] + df['feature2']

    # 2. Polynomial features
    df['feature1_squared'] = df['feature1'] ** 2
    df['feature1_cubed'] = df['feature1'] ** 3
    df['feature1_sqrt'] = np.sqrt(np.abs(df['feature1']))

    # 3. Date features
    if 'date' in df.columns:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # 4. Binning continuous features
    df['age_bin'] = pd.cut(df['age'], bins=[0, 18, 30, 50, 100],
                           labels=['young', 'adult', 'middle', 'senior'])

    # 5. Aggregations (if grouped data)
    if 'customer_id' in df.columns:
        customer_stats = df.groupby('customer_id')['purchase_amount'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).add_prefix('customer_')
        df = df.merge(customer_stats, left_on='customer_id', right_index=True)

    # 6. Lag features (for time series)
    if 'date' in df.columns:
        df = df.sort_values('date')
        df['value_lag1'] = df['value'].shift(1)
        df['value_lag7'] = df['value'].shift(7)
        df['value_rolling_mean_7'] = df['value'].rolling(window=7).mean()

    # 7. Domain-specific (e.g., e-commerce)
    if 'price' in df.columns and 'quantity' in df.columns:
        df['total_value'] = df['price'] * df['quantity']
        df['is_bulk_purchase'] = (df['quantity'] > 10).astype(int)

    return df

# Apply feature engineering
df_engineered = create_features(df)
```

### Polynomial Features

```python
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial and interaction features
poly = PolynomialFeatures(
    degree=2,              # Polynomial degree
    interaction_only=False, # Include polynomial terms
    include_bias=False     # Don't include intercept column
)

X_poly = poly.fit_transform(X)

# Get feature names
feature_names = poly.get_feature_names_out(['x1', 'x2'])
print(feature_names)  # ['x1', 'x2', 'x1^2', 'x1 x2', 'x2^2']

# Warning: Can create many features
# Original: 10 features -> Degree 2: 65 features -> Degree 3: 285 features
```

---

## sklearn Pipeline and ColumnTransformer

### Basic Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Simple pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression())
])

# Fit and predict
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

# Access steps
scaler = pipeline.named_steps['scaler']
classifier = pipeline.named_steps['classifier']
```

### ColumnTransformer (Different Transformations for Different Columns)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column groups
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['gender', 'region', 'education']

# Define transformations
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough'  # Keep remaining columns unchanged
)

# Full pipeline with model
full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

# Fit
full_pipeline.fit(X_train, y_train)

# Evaluate
score = full_pipeline.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

### Advanced Pipeline with Feature Engineering

```python
from sklearn.base import BaseEstimator, TransformerMixin

# Custom transformer for feature engineering (see next section)
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['feature1_x_feature2'] = X['feature1'] * X['feature2']
        X['feature1_squared'] = X['feature1'] ** 2
        return X

# Complete pipeline
preprocessing_pipeline = Pipeline([
    ('engineer', FeatureEngineer()),
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

model_pipeline = Pipeline([
    ('preprocessing', preprocessing_pipeline),
    ('classifier', LogisticRegression())
])

# Grid search with pipeline
from sklearn.model_selection import GridSearchCV

param_grid = {
    'preprocessing__scaler': [StandardScaler(), RobustScaler()],
    'classifier__C': [0.1, 1.0, 10.0],
    'classifier__penalty': ['l1', 'l2']
}

grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
```

---

## Custom Transformers

### Basic Custom Transformer

```python
from sklearn.base import BaseEstimator, TransformerMixin

class LogTransformer(BaseEstimator, TransformerMixin):
    """Apply log transformation to skewed features."""

    def __init__(self, columns=None):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.columns is None:
            self.columns = X.select_dtypes(include=[np.number]).columns

        for col in self.columns:
            X[col] = np.log1p(X[col])  # log1p handles zeros

        return X

# Usage
log_transformer = LogTransformer(columns=['income', 'price'])
X_transformed = log_transformer.fit_transform(X)
```

### Advanced Custom Transformer with State

```python
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip outliers based on IQR computed from training data."""

    def __init__(self, k=1.5):
        self.k = k
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)

        for col in X.select_dtypes(include=[np.number]).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1

            self.bounds_[col] = {
                'lower': Q1 - self.k * IQR,
                'upper': Q3 + self.k * IQR
            }

        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()

        for col, bounds in self.bounds_.items():
            if col in X.columns:
                X[col] = X[col].clip(
                    lower=bounds['lower'],
                    upper=bounds['upper']
                )

        return X

# Usage in pipeline
pipeline = Pipeline([
    ('clipper', OutlierClipper(k=1.5)),
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
```

### Functional Transformer

```python
from sklearn.preprocessing import FunctionTransformer

# Simple function
def add_features(X):
    X = pd.DataFrame(X)
    X['new_feature'] = X.iloc[:, 0] * X.iloc[:, 1]
    return X

# Wrap in FunctionTransformer
feature_adder = FunctionTransformer(
    func=add_features,
    validate=False
)

# Use in pipeline
pipeline = Pipeline([
    ('add_features', feature_adder),
    ('scaler', StandardScaler())
])
```

### Custom Transformer for Date Features

```python
class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract date features from datetime column."""

    def __init__(self, date_column='date'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        date_col = pd.to_datetime(X[self.date_column])

        X['year'] = date_col.dt.year
        X['month'] = date_col.dt.month
        X['day'] = date_col.dt.day
        X['dayofweek'] = date_col.dt.dayofweek
        X['quarter'] = date_col.dt.quarter
        X['is_weekend'] = date_col.dt.dayofweek.isin([5, 6]).astype(int)
        X['is_month_start'] = date_col.dt.is_month_start.astype(int)
        X['is_month_end'] = date_col.dt.is_month_end.astype(int)
        X['days_in_month'] = date_col.dt.days_in_month

        # Drop original date column
        X = X.drop(columns=[self.date_column])

        return X

# Usage
date_extractor = DateFeatureExtractor(date_column='transaction_date')
X_with_date_features = date_extractor.fit_transform(df)
```

---

## Data Validation with Great Expectations

### Installation and Setup

```python
# Install
# pip install great-expectations

import great_expectations as ge
import pandas as pd

# Convert pandas DataFrame to Great Expectations DataFrame
df_ge = ge.from_pandas(df)

# Or read directly
df_ge = ge.read_csv('data.csv')
```

### Basic Expectations

```python
# Expect column to exist
result = df_ge.expect_column_to_exist('customer_id')
print(result)

# Expect no null values
result = df_ge.expect_column_values_to_not_be_null('customer_id')

# Expect values in set
result = df_ge.expect_column_values_to_be_in_set(
    'status',
    ['active', 'inactive', 'pending']
)

# Expect numeric range
result = df_ge.expect_column_values_to_be_between(
    'age',
    min_value=0,
    max_value=120
)

# Expect unique values
result = df_ge.expect_column_values_to_be_unique('email')

# Expect regex match
result = df_ge.expect_column_values_to_match_regex(
    'email',
    r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
)

# Expect column mean in range
result = df_ge.expect_column_mean_to_be_between(
    'purchase_amount',
    min_value=10,
    max_value=1000
)
```

### Production Data Validation Suite

```python
def create_validation_suite(df):
    """
    Create comprehensive validation suite for production data.

    Returns:
    --------
    dict : Validation results
    """
    df_ge = ge.from_pandas(df)

    validations = []

    # 1. Schema validation
    expected_columns = ['id', 'created_at', 'amount', 'status', 'customer_id']
    for col in expected_columns:
        validations.append(
            df_ge.expect_column_to_exist(col)
        )

    # 2. Completeness validation
    required_columns = ['id', 'customer_id', 'amount']
    for col in required_columns:
        validations.append(
            df_ge.expect_column_values_to_not_be_null(col)
        )

    # 3. Value range validation
    validations.append(
        df_ge.expect_column_values_to_be_between('amount', min_value=0, max_value=100000)
    )

    # 4. Categorical values validation
    validations.append(
        df_ge.expect_column_values_to_be_in_set(
            'status',
            ['active', 'inactive', 'pending', 'cancelled']
        )
    )

    # 5. Uniqueness validation
    validations.append(
        df_ge.expect_column_values_to_be_unique('id')
    )

    # 6. Statistical validation
    validations.append(
        df_ge.expect_column_mean_to_be_between('amount', min_value=50, max_value=500)
    )

    # 7. Row count validation
    validations.append(
        df_ge.expect_table_row_count_to_be_between(min_value=100, max_value=1000000)
    )

    # Check if all validations passed
    all_passed = all(v['success'] for v in validations)

    return {
        'all_passed': all_passed,
        'total_checks': len(validations),
        'passed': sum(v['success'] for v in validations),
        'failed': sum(not v['success'] for v in validations),
        'validations': validations
    }

# Run validation
results = create_validation_suite(df)

if not results['all_passed']:
    print(f"Validation failed: {results['failed']} checks failed")
    for v in results['validations']:
        if not v['success']:
            print(f"  - {v['expectation_config']['expectation_type']}")
else:
    print("All validation checks passed!")
```

### Automated Data Quality Monitoring

```python
class DataQualityMonitor:
    """Monitor data quality over time."""

    def __init__(self, reference_df):
        self.reference_stats = self._compute_stats(reference_df)

    def _compute_stats(self, df):
        """Compute reference statistics."""
        stats = {}

        for col in df.select_dtypes(include=[np.number]).columns:
            stats[col] = {
                'mean': df[col].mean(),
                'std': df[col].std(),
                'min': df[col].min(),
                'max': df[col].max(),
                'missing_pct': df[col].isnull().mean()
            }

        return stats

    def check_drift(self, new_df, threshold=0.1):
        """
        Check if new data has drifted from reference.

        Parameters:
        -----------
        new_df : pd.DataFrame
        threshold : float
            Maximum allowed relative change

        Returns:
        --------
        dict : Drift report
        """
        new_stats = self._compute_stats(new_df)
        drift_report = {'drifted_features': []}

        for col in self.reference_stats:
            if col not in new_stats:
                drift_report['drifted_features'].append({
                    'feature': col,
                    'reason': 'missing_in_new_data'
                })
                continue

            ref = self.reference_stats[col]
            new = new_stats[col]

            # Check mean drift
            if ref['mean'] != 0:
                mean_change = abs((new['mean'] - ref['mean']) / ref['mean'])
                if mean_change > threshold:
                    drift_report['drifted_features'].append({
                        'feature': col,
                        'metric': 'mean',
                        'reference': ref['mean'],
                        'current': new['mean'],
                        'change_pct': mean_change * 100
                    })

            # Check std drift
            if ref['std'] != 0:
                std_change = abs((new['std'] - ref['std']) / ref['std'])
                if std_change > threshold:
                    drift_report['drifted_features'].append({
                        'feature': col,
                        'metric': 'std',
                        'reference': ref['std'],
                        'current': new['std'],
                        'change_pct': std_change * 100
                    })

        drift_report['has_drift'] = len(drift_report['drifted_features']) > 0

        return drift_report

# Usage
monitor = DataQualityMonitor(df_train)

# Check new batch
drift_report = monitor.check_drift(df_new_batch, threshold=0.15)

if drift_report['has_drift']:
    print("Data drift detected!")
    for drift in drift_report['drifted_features']:
        print(f"  {drift['feature']} ({drift['metric']}): "
              f"{drift['change_pct']:.2f}% change")
```

---

## Production Preprocessing Pipelines

### Complete Production Pipeline

```python
import joblib
from pathlib import Path

class ProductionPreprocessor:
    """
    Production-ready preprocessing pipeline with versioning and monitoring.
    """

    def __init__(self, version='1.0'):
        self.version = version
        self.pipeline = None
        self.feature_names = None
        self.metadata = {}

    def build_pipeline(self, numeric_features, categorical_features):
        """Build preprocessing pipeline."""

        # Numeric transformer
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('outlier_clipper', OutlierClipper(k=1.5)),
            ('scaler', RobustScaler())
        ])

        # Categorical transformer
        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
        ])

        # Combine
        self.pipeline = ColumnTransformer([
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

        self.metadata['numeric_features'] = numeric_features
        self.metadata['categorical_features'] = categorical_features

        return self

    def fit(self, X, y=None):
        """Fit pipeline on training data."""
        self.pipeline.fit(X)

        # Store feature names after transformation
        self.feature_names = self._get_feature_names()

        # Store fit metadata
        self.metadata['fit_time'] = pd.Timestamp.now().isoformat()
        self.metadata['n_samples'] = len(X)
        self.metadata['n_features_in'] = X.shape[1]
        self.metadata['n_features_out'] = len(self.feature_names)

        return self

    def transform(self, X):
        """Transform new data."""
        return self.pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """Fit and transform in one step."""
        self.fit(X, y)
        return self.transform(X)

    def _get_feature_names(self):
        """Extract feature names from pipeline."""
        feature_names = []

        for name, transformer, features in self.pipeline.transformers_:
            if name == 'num':
                feature_names.extend(features)
            elif name == 'cat':
                # Get one-hot encoded feature names
                encoder = transformer.named_steps['encoder']
                cat_features = encoder.get_feature_names_out(features)
                feature_names.extend(cat_features)

        return feature_names

    def save(self, path):
        """Save pipeline to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save pipeline
        joblib.dump(self.pipeline, path / f'pipeline_v{self.version}.joblib')

        # Save metadata
        metadata_path = path / f'metadata_v{self.version}.json'
        with open(metadata_path, 'w') as f:
            import json
            json.dump(self.metadata, f, indent=2)

        print(f"Pipeline saved to {path}")

    @classmethod
    def load(cls, path, version='1.0'):
        """Load pipeline from disk."""
        path = Path(path)

        preprocessor = cls(version=version)
        preprocessor.pipeline = joblib.load(path / f'pipeline_v{version}.joblib')

        # Load metadata
        metadata_path = path / f'metadata_v{version}.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                import json
                preprocessor.metadata = json.load(f)

        return preprocessor

# Usage
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['gender', 'region', 'education']

# Build and fit
preprocessor = ProductionPreprocessor(version='1.0')
preprocessor.build_pipeline(numeric_features, categorical_features)
preprocessor.fit(X_train)

# Transform
X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save for deployment
preprocessor.save('models/preprocessor/')

# Load in production
preprocessor_loaded = ProductionPreprocessor.load('models/preprocessor/', version='1.0')
X_new_processed = preprocessor_loaded.transform(X_new)
```

### Preprocessing with Error Handling

```python
class RobustPreprocessor:
    """Preprocessing pipeline with comprehensive error handling."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.error_log = []

    def transform_with_validation(self, X):
        """Transform with validation and error handling."""
        try:
            # Validate input
            self._validate_input(X)

            # Transform
            X_transformed = self.pipeline.transform(X)

            # Validate output
            self._validate_output(X_transformed)

            return X_transformed

        except Exception as e:
            self.error_log.append({
                'timestamp': pd.Timestamp.now(),
                'error': str(e),
                'input_shape': X.shape if hasattr(X, 'shape') else None
            })
            raise

    def _validate_input(self, X):
        """Validate input data."""
        # Check for required columns
        if isinstance(X, pd.DataFrame):
            expected_cols = (self.pipeline.transformers_[0][2] +
                           self.pipeline.transformers_[1][2])
            missing_cols = set(expected_cols) - set(X.columns)
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")

        # Check for NaN in entire DataFrame
        if isinstance(X, np.ndarray):
            if np.isnan(X).all():
                raise ValueError("Input contains only NaN values")
        elif isinstance(X, pd.DataFrame):
            if X.isnull().all().all():
                raise ValueError("Input contains only null values")

        # Check for infinite values
        if isinstance(X, np.ndarray):
            if np.isinf(X).any():
                raise ValueError("Input contains infinite values")
        elif isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if np.isinf(X[numeric_cols]).any().any():
                raise ValueError("Input contains infinite values")

    def _validate_output(self, X_transformed):
        """Validate transformed data."""
        # Check for NaN (should be imputed)
        if np.isnan(X_transformed).any():
            raise ValueError("Transformation produced NaN values")

        # Check for infinite values
        if np.isinf(X_transformed).any():
            raise ValueError("Transformation produced infinite values")

        # Check shape
        if len(X_transformed) == 0:
            raise ValueError("Transformation produced empty array")

# Usage
robust_preprocessor = RobustPreprocessor(preprocessor.pipeline)

try:
    X_processed = robust_preprocessor.transform_with_validation(X_new)
except ValueError as e:
    print(f"Preprocessing failed: {e}")
    print(f"Error log: {robust_preprocessor.error_log}")
```

---

## Complete Implementations

### End-to-End Preprocessing Example

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load data
df = pd.read_csv('data.csv')

# 2. Initial validation
print("Initial Data Profile:")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"Dtypes:\n{df.dtypes}")

# 3. Split features and target
X = df.drop('target', axis=1)
y = df['target']

# 4. Train-test split (stratified for classification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Define feature types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# 6. Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 7. Create full pipeline with model
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 8. Fit pipeline
print("\nFitting pipeline...")
model_pipeline.fit(X_train, y_train)

# 9. Evaluate
y_pred = model_pipeline.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 10. Save pipeline
joblib.dump(model_pipeline, 'model_pipeline.joblib')
print("\nPipeline saved!")

# 11. Load and use in production
loaded_pipeline = joblib.load('model_pipeline.joblib')
new_predictions = loaded_pipeline.predict(X_new)
```

### Kaggle Competition Preprocessing Template

```python
"""
Production-ready preprocessing template for Kaggle competitions.
Handles numeric, categorical, text, and date features.
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

class KagglePreprocessor:
    """Comprehensive preprocessor for Kaggle competitions."""

    def __init__(self, target_col='target'):
        self.target_col = target_col
        self.numeric_features = []
        self.categorical_low_card = []
        self.categorical_high_card = []
        self.date_features = []
        self.text_features = []
        self.pipeline = None

    def analyze_features(self, df):
        """Automatically detect feature types."""
        for col in df.columns:
            if col == self.target_col:
                continue

            # Numeric
            if pd.api.types.is_numeric_dtype(df[col]):
                self.numeric_features.append(col)

            # Datetime
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                self.date_features.append(col)

            # Categorical
            elif pd.api.types.is_object_dtype(df[col]):
                unique_ratio = df[col].nunique() / len(df)

                # Check if text (high unique ratio, long strings)
                if unique_ratio > 0.8 and df[col].str.len().mean() > 20:
                    self.text_features.append(col)
                # Low cardinality categorical
                elif df[col].nunique() < 10:
                    self.categorical_low_card.append(col)
                # High cardinality categorical
                else:
                    self.categorical_high_card.append(col)

        print(f"Numeric features ({len(self.numeric_features)}): {self.numeric_features}")
        print(f"Low cardinality categorical ({len(self.categorical_low_card)}): {self.categorical_low_card}")
        print(f"High cardinality categorical ({len(self.categorical_high_card)}): {self.categorical_high_card}")
        print(f"Date features ({len(self.date_features)}): {self.date_features}")
        print(f"Text features ({len(self.text_features)}): {self.text_features}")

    def build_pipeline(self):
        """Build comprehensive preprocessing pipeline."""
        transformers = []

        # Numeric features
        if self.numeric_features:
            numeric_transformer = Pipeline([
                ('imputer', KNNImputer(n_neighbors=5)),
                ('scaler', RobustScaler())
            ])
            transformers.append(('num', numeric_transformer, self.numeric_features))

        # Low cardinality categorical (one-hot encode)
        if self.categorical_low_card:
            cat_low_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ])
            transformers.append(('cat_low', cat_low_transformer, self.categorical_low_card))

        # High cardinality categorical (target encode)
        if self.categorical_high_card:
            cat_high_transformer = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', TargetEncoder(smoothing=10))
            ])
            transformers.append(('cat_high', cat_high_transformer, self.categorical_high_card))

        self.pipeline = ColumnTransformer(
            transformers=transformers,
            remainder='drop'
        )

        return self.pipeline

    def fit_transform(self, df):
        """Fit and transform data."""
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]

        # Analyze features
        self.analyze_features(df)

        # Build pipeline
        self.build_pipeline()

        # Extract date features before fitting pipeline
        if self.date_features:
            for col in self.date_features:
                X = self._extract_date_features(X, col)
                self.numeric_features.extend([
                    f'{col}_year', f'{col}_month', f'{col}_day',
                    f'{col}_dayofweek', f'{col}_quarter'
                ])
            # Rebuild pipeline with new numeric features
            self.build_pipeline()

        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X, y)

        return X_transformed, y

    def transform(self, df):
        """Transform new data."""
        X = df.drop(columns=[self.target_col], errors='ignore')

        # Extract date features
        if self.date_features:
            for col in self.date_features:
                X = self._extract_date_features(X, col)

        # Transform
        X_transformed = self.pipeline.transform(X)

        return X_transformed

    def _extract_date_features(self, df, col):
        """Extract features from date column."""
        df = df.copy()
        date_col = pd.to_datetime(df[col], errors='coerce')

        df[f'{col}_year'] = date_col.dt.year
        df[f'{col}_month'] = date_col.dt.month
        df[f'{col}_day'] = date_col.dt.day
        df[f'{col}_dayofweek'] = date_col.dt.dayofweek
        df[f'{col}_quarter'] = date_col.dt.quarter
        df[f'{col}_is_weekend'] = date_col.dt.dayofweek.isin([5, 6]).astype(int)

        df = df.drop(columns=[col])

        return df

# Usage
df = pd.read_csv('train.csv')

# Initialize preprocessor
preprocessor = KagglePreprocessor(target_col='target')

# Fit and transform
X_train, y_train = preprocessor.fit_transform(df)

# Transform test data
df_test = pd.read_csv('test.csv')
X_test = preprocessor.transform(df_test)

print(f"\nTransformed shape: {X_train.shape}")
print("Ready for modeling!")
```

---

## Summary

This comprehensive guide covered:

1. **Data Loading & Validation:** Efficient loading, profiling, and validation
2. **Missing Values:** Simple, KNN, iterative imputation strategies
3. **Outlier Detection:** Statistical (IQR, Z-score) and ML methods (Isolation Forest, LOF)
4. **Feature Scaling:** StandardScaler, MinMaxScaler, RobustScaler, when to use each
5. **Encoding:** One-hot, label, ordinal, target, frequency, binary, hash encoding
6. **Feature Engineering:** Interactions, polynomials, date features, domain-specific
7. **sklearn Pipeline:** Composable, reproducible preprocessing
8. **ColumnTransformer:** Different transformations for different feature types
9. **Custom Transformers:** Building custom preprocessing components
10. **Great Expectations:** Production data validation
11. **Production Pipelines:** Versioning, error handling, monitoring

**Key Takeaways:**

- **Always fit on training data only** to prevent data leakage
- **Use pipelines** for reproducibility and deployment
- **Choose scaling based on algorithm:** Neural nets need scaling, trees don't
- **Validate data continuously** in production (Great Expectations, custom monitors)
- **Version preprocessing code** alongside models
- **Monitor for data drift** to detect when to retrain

**2025 Best Practices:**
- Great Expectations for automated validation
- Feature stores (Feast, Tecton) for feature reuse
- Continuous monitoring with Evidently AI, WhyLabs
- Automated retraining on drift detection
- Preprocessing as code in version control

Master these techniques to build robust ML systems that perform reliably in production!
