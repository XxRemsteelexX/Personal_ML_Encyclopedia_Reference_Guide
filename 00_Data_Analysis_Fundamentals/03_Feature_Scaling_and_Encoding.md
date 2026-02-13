# 3. Feature Scaling and Encoding

## Overview

**Why This Matters:** Many ML algorithms require features on similar scales. Distance-based algorithms (KNN, SVM, neural networks) and gradient descent-based models are especially sensitive.

**2025 Best Practice:** Always scale features AFTER train/test split to avoid data leakage!

---

## 3.1 Feature Scaling Methods

### When to Scale Features

**Scale Required:**
- Neural Networks (gradient descent sensitive)
- KNN, K-Means (distance-based)
- SVM (especially with RBF kernel)
- Linear/Logistic Regression with regularization
- PCA, t-SNE, UMAP
- Gradient Boosting (helps convergence)

**Scale NOT Required:**
- Tree-based models (decision trees, random forest, XGBoost)
- Naive Bayes

---

### 1. Min-Max Normalization (0-1 Scaling)

**Formula:** X_scaled = (X - X_min) / (X_max - X_min)

**Result:** Values between 0 and 1

**When to Use:**
 When you need bounded values
 Neural networks (especially with sigmoid/tanh)
 Image data (pixel values)

**Disadvantages:**
 Sensitive to outliers (they determine min/max)

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Create scaler
scaler = MinMaxScaler()

# Fit on training data ONLY
scaler.fit(X_train)

# Transform both train and test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# With pandas DataFrame
scaler = MinMaxScaler()
df[['feature1', 'feature2']] = scaler.fit_transform(df[['feature1', 'feature2']])

# Custom range (e.g., -1 to 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_scaled = scaler.fit_transform(X)

# Manual implementation
def min_max_scale(X):
    return (X - X.min()) / (X.max() - X.min())
```

---

### 2. Standardization (Z-score Normalization)

**Formula:** X_scaled = (X - mu) / sigma

**Result:** Mean = 0, Standard Deviation = 1

**When to Use:**
 **Most common choice** (default for many algorithms)
 When data follows normal distribution
 Less sensitive to outliers than Min-Max
 Linear regression, logistic regression, neural networks
 PCA, clustering

```python
from sklearn.preprocessing import StandardScaler

# Create scaler
scaler = StandardScaler()

# Fit on training data
scaler.fit(X_train)

# Transform
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# One step (for train only)
X_train_scaled = scaler.fit_transform(X_train)

# Access mean and std
print("Means:", scaler.mean_)
print("Stds:", scaler.scale_)

# Inverse transform (get original values back)
X_original = scaler.inverse_transform(X_scaled)
```

---

### 3. Robust Scaler (Outlier-Resistant)

**Formula:** X_scaled = (X - median) / IQR

**When to Use:**
 **When you have outliers**
 More robust than StandardScaler
 Uses median and IQR instead of mean and std

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Custom quantile range (default is 25%-75%)
scaler = RobustScaler(quantile_range=(10, 90))
X_scaled = scaler.fit_transform(X)
```

---

### 4. MaxAbs Scaler (Sparse Data)

**Formula:** X_scaled = X / |X_max|

**Result:** Values between -1 and 1

**When to Use:**
 Sparse data (many zeros)
 Data already centered at zero
 Preserves sparsity (doesn't densify sparse matrices)

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)  # Works with sparse matrices
```

---

### 5. Unit Vector Scaling (Normalization)

**Formula:** X_scaled = X / ||X||

**Result:** Each sample has unit norm (length = 1)

**When to Use:**
 Text data (TF-IDF vectors)
 When direction matters more than magnitude
 Cosine similarity computations

```python
from sklearn.preprocessing import Normalizer

# L2 norm (Euclidean)
scaler = Normalizer(norm='l2')
X_normalized = scaler.transform(X)

# L1 norm (Manhattan)
scaler = Normalizer(norm='l1')
X_normalized = scaler.transform(X)

# Manual L2 normalization
X_normalized = X / np.linalg.norm(X, axis=1, keepdims=True)
```

---

### 6. Log Transformation (Skewed Data)

**When to Use:**
 **Highly skewed distributions**
 Right-skewed data (long tail on right)
 Make distribution more normal
 Income, prices, counts

```python
import numpy as np

# Log transform (use log1p for values near 0)
df['log_feature'] = np.log1p(df['feature'])  # log(1 + x)

# Square root transform (less aggressive)
df['sqrt_feature'] = np.sqrt(df['feature'])

# Box-Cox transformation (automatically finds best power)
from scipy import stats

transformed, lambda_param = stats.boxcox(df['feature'])
print(f"Optimal lambda: {lambda_param}")

# Yeo-Johnson (works with negative values)
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson')
df['transformed'] = pt.fit_transform(df[['feature']])
```

---

### Comparison Table

| Method | Range | Outlier Sensitive | Use Case |
|--------|-------|-------------------|----------|
| **Min-Max** | [0, 1] |  Yes | Neural networks, bounded values |
| **Standardization** | Mean=0, Std=1 | [WARNING] Moderate | **Default choice**, most algorithms |
| **Robust** | Varies |  No | **When outliers present** |
| **MaxAbs** | [-1, 1] |  Yes | Sparse data |
| **Normalizer** | Unit norm |  No | Text vectors, cosine similarity |
| **Log** | > 0 |  No | Skewed distributions |

---

## 3.2 Categorical Encoding

### Why Encode?

ML algorithms require numeric input. Categorical variables (text) must be converted to numbers.

---

### 1. Label Encoding (Ordinal)

**Use When:** Categories have natural order (Low < Medium < High)

```python
from sklearn.preprocessing import LabelEncoder

# Example: education level
df['education'] = ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor']

le = LabelEncoder()
df['education_encoded'] = le.fit_transform(df['education'])

# Result: Bachelor=0, High School=1, Master=2, PhD=3
print(df['education_encoded'].values)

# Get original labels back
df['education_decoded'] = le.inverse_transform(df['education_encoded'])

# Manual ordinal encoding with custom order
education_map = {
    'High School': 0,
    'Bachelor': 1,
    'Master': 2,
    'PhD': 3
}
df['education_encoded'] = df['education'].map(education_map)
```

**[WARNING] Warning:** Don't use for nominal categories (no order) - model might assume order!

---

### 2. One-Hot Encoding (Nominal)

**Use When:** Categories have NO natural order (color, country, brand)

**Creates binary columns for each category**

```python
import pandas as pd

# Example: colors
df = pd.DataFrame({'color': ['red', 'blue', 'green', 'red', 'blue']})

# pandas get_dummies (easy)
df_encoded = pd.get_dummies(df, columns=['color'], prefix='color')

# Result: color_red, color_blue, color_green columns (0s and 1s)
print(df_encoded)

# Drop first category to avoid multicollinearity (for linear models)
df_encoded = pd.get_dummies(df, columns=['color'], drop_first=True)

# sklearn OneHotEncoder
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop first category
encoded = ohe.fit_transform(df[['color']])

# Get feature names
feature_names = ohe.get_feature_names_out(['color'])
df_encoded = pd.DataFrame(encoded, columns=feature_names)
```

**Disadvantages:**
-  **High cardinality problem**: 1000 categories --> 1000 columns!
-  Sparse matrices (memory intensive)

---

### 3. Frequency Encoding (High Cardinality)

**Replace category with its frequency count**

**Use When:** Many unique categories (100+)

```python
# Calculate frequency
freq_map = df['category'].value_counts(normalize=True).to_dict()

# Encode
df['category_freq'] = df['category'].map(freq_map)

# Example
# category: ['A', 'B', 'A', 'C', 'A']
# Frequencies: A=0.6, B=0.2, C=0.2
# Encoded: [0.6, 0.2, 0.6, 0.2, 0.6]
```

---

### 4. Target Encoding (Mean Encoding)

**Replace category with mean of target variable for that category**

**[WARNING] Caution:** Can cause overfitting! Use with cross-validation.

```python
# Calculate mean target per category
target_means = df.groupby('category')['target'].mean()

# Encode
df['category_encoded'] = df['category'].map(target_means)

# With regularization (smoothing to prevent overfitting)
def target_encode_with_smoothing(df, category_col, target_col, smoothing=10):
    # Global mean
    global_mean = df[target_col].mean()

    # Category stats
    agg = df.groupby(category_col)[target_col].agg(['mean', 'count'])

    # Smoothed mean
    smoothed = (agg['mean'] * agg['count'] + global_mean * smoothing) / (agg['count'] + smoothing)

    return df[category_col].map(smoothed)

df['category_encoded'] = target_encode_with_smoothing(df, 'category', 'target')
```

**Better approach: Use cross-validation to prevent leakage**

```python
from sklearn.model_selection import KFold

def target_encode_cv(df, category_col, target_col, n_splits=5):
    encoded = np.zeros(len(df))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(df):
        # Calculate mean on train fold
        target_means = df.iloc[train_idx].groupby(category_col)[target_col].mean()

        # Apply to validation fold
        encoded[val_idx] = df.iloc[val_idx][category_col].map(target_means)

    return encoded

df['category_encoded'] = target_encode_cv(df, 'category', 'target')
```

---

### 5. Binary Encoding (Medium Cardinality)

**Converts to binary, then splits into columns**

**Use When:** 10-100 unique categories

```python
import category_encoders as ce

# Binary encoding
encoder = ce.BinaryEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df)

# Example: 5 categories --> 3 binary columns (2^3 = 8 > 5)
# More compact than one-hot
```

---

### 6. Hash Encoding (Very High Cardinality)

**Uses hash function to map categories to fixed number of buckets**

**Use When:** 1000+ categories, memory constraints

```python
from sklearn.feature_extraction import FeatureHasher

# Hash to 10 features
hasher = FeatureHasher(n_features=10, input_type='string')
hashed = hasher.transform(df['category'].apply(lambda x: [x]))

# Collision possible but rare with good hash function
```

---

### 7. Embedding (Deep Learning)

**Learn representations during training**

**Use When:** Very high cardinality, deep learning model

```python
import torch
import torch.nn as nn

# Example: 1000 unique categories --> 50-dimensional embeddings
embedding_layer = nn.Embedding(num_embeddings=1000, embedding_dim=50)

# Input: category indices
category_indices = torch.LongTensor([0, 15, 234, 999])

# Output: 50-dimensional vectors
embeddings = embedding_layer(category_indices)
print(embeddings.shape)  # torch.Size([4, 50])
```

---

### Encoding Decision Tree

```
How many unique categories?
|
+--- < 10: One-Hot Encoding
|
+--- 10-100:
|   +--- Tree-based model: Label Encoding
|   +--- Linear model: Binary Encoding or Target Encoding
|
+--- > 100:
    +--- Frequency Encoding
    +--- Hash Encoding
    +--- Target Encoding (with CV)
    +--- Embeddings (deep learning)
```

---

## 3.3 DateTime Features

```python
# Convert to datetime
df['date'] = pd.to_datetime(df['date'])

# Extract components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0
df['day_of_year'] = df['date'].dt.dayofyear
df['week_of_year'] = df['date'].dt.isocalendar().week
df['quarter'] = df['date'].dt.quarter

# Is weekend?
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Hour (for datetime)
df['hour'] = df['timestamp'].dt.hour
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

# Cyclical encoding (for cyclical features like month, hour)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Time since reference
reference_date = pd.Timestamp('2020-01-01')
df['days_since_ref'] = (df['date'] - reference_date).dt.days
```

---

## 3.4 Complete Preprocessing Pipeline (2025)

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define column types
numeric_features = ['age', 'income', 'score']
categorical_features = ['city', 'education', 'occupation']

# Numeric pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Fit and transform
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Use in full pipeline with model
from sklearn.ensemble import RandomForestClassifier

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

full_pipeline.fit(X_train, y_train)
predictions = full_pipeline.predict(X_test)
```

---

## 3.5 Best Practices (2025)

### 1. Always Split First, Then Scale

```python
#  CORRECT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on train
X_test_scaled = scaler.transform(X_test)        # Transform test

#  WRONG (data leakage!)
X_scaled = scaler.fit_transform(X)  # Information from test leaks to train!
X_train, X_test = train_test_split(X_scaled)
```

---

### 2. Save Scalers for Production

```python
import joblib

# Save scaler
scaler = StandardScaler()
scaler.fit(X_train)
joblib.dump(scaler, 'scaler.pkl')

# Load in production
scaler = joblib.load('scaler.pkl')
X_new_scaled = scaler.transform(X_new)
```

---

### 3. Handle Unknown Categories

```python
# One-hot encoding with unknown handling
ohe = OneHotEncoder(handle_unknown='ignore')  # Ignores new categories
ohe.fit(X_train[['category']])

# Transform with new category
X_test_encoded = ohe.transform(X_test[['category']])
```

---

### 4. Check for Data Leakage

```python
# Feature should NOT be scaled before split
# Target encoding should use cross-validation
# Time series: use temporal split, not random
```

---

## 3.6 Feature Scaling Checklist

**Before Scaling:**
- [ ] Split into train/test first
- [ ] Check for outliers (consider RobustScaler)
- [ ] Check distribution (consider log transform)

**During Scaling:**
- [ ] Fit scaler on training data only
- [ ] Transform both train and test
- [ ] Save scaler for production

**After Scaling:**
- [ ] Verify mean ~= 0, std ~= 1 (for StandardScaler)
- [ ] Check no data leakage occurred
- [ ] Document which scaler used

---

## 3.7 Categorical Encoding Checklist

**Choosing Encoding:**
- [ ] Count unique categories (cardinality)
- [ ] Check if ordinal (natural order) or nominal
- [ ] Consider model type (tree vs linear)

**Implementation:**
- [ ] Handle unknown categories
- [ ] Use CV for target encoding
- [ ] Check for high cardinality issues
- [ ] Document encoding strategy

---

## Resources

**Libraries:**
- scikit-learn (StandardScaler, OneHotEncoder)
- category_encoders (Binary, Target, Hashing)
- feature_engine (advanced transformers)

**Documentation:**
- sklearn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
- category_encoders: https://contrib.scikit-learn.org/category_encoders/

**Best Practices:**
- Always fit on train, transform on test
- Save scalers/encoders for production
- Use pipelines to prevent leakage
- Document all transformations
