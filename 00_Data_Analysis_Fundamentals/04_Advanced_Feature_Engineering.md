# 3. Feature Engineering & Encoding

## Overview

Feature engineering is the art of creating new features or transforming existing ones to improve model performance. It's often the difference between mediocre and excellent models.

**"Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering."** - Andrew Ng

This guide covers encoding strategies, transformations, and when to use each technique.

---

## 3.1 Categorical Encoding: Complete Guide

### Encoding Comparison Table

| Encoding | Tree Models | Linear Models | When to Use | Pros | Cons |
|----------|-------------|---------------|-------------|------|------|
| **Label** |  Excellent |  Avoid | Trees only, low-med cardinality | Fast, no dimensions added | Creates false ordering for linear |
| **One-Hot** |  Avoid |  Required | Linear models, low cardinality (<10) | No false ordering | Explodes dimensions, slow for trees |
| **Target** |  Excellent |  Good | High cardinality (50+) | Captures target signal | Leakage risk, needs CV |
| **Frequency** |  Good |  Good | Any cardinality | Simple, no leakage | Less informative |
| **Binary** |  Good |  Good | 2 categories | Simple, efficient | Only for binary |
| **Ordinal** |  Good |  Good | Natural ordering exists | Preserves order | Assumes equal spacing |
| **Hash** |  Good | [WARNING] Okay | Very high cardinality (1000+) | Fixed dimensions | Collisions, not reversible |

---

### 3.1.1 Label Encoding

**What:** Assign integer to each category (Cat=0, Dog=1, Bird=2)

**Code:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['pet_encoded'] = le.fit_transform(df['pet'])
```

** When to Use:**
- Tree-based models (Random Forest, XGBoost)
- Low to medium cardinality (<50 categories)
- Want minimal memory usage

** When NOT to Use:**
- Linear models (creates false ordinal relationship)
- Categories have no natural ordering
- Need interpretability with linear models

**Why it works for trees:**
Trees split on thresholds (`feature < 1.5`), not numeric relationships. The ordering doesn't matter.

---

### 3.1.2 One-Hot Encoding

**What:** Create binary column for each category

**Code:**
```python
# Pandas
df_encoded = pd.get_dummies(df, columns=['pet'], drop_first=True)

# Sklearn
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse=False)
encoded = encoder.fit_transform(df[['pet']])
```

** When to Use:**
- **Linear models (REQUIRED)**
- **Neural networks**
- Low cardinality (<10 categories)
- No natural ordering

** When NOT to Use:**
- Tree-based models (hurts performance)
- High cardinality (50+ categories) --> dimension explosion
- Memory constrained

**drop_first=True:** Avoids multicollinearity (dummy variable trap)

---

### 3.1.3 Target Encoding

**What:** Replace category with mean of target variable

**Code:**
```python
from category_encoders import TargetEncoder

# CRITICAL: Use CV to prevent leakage
encoder = TargetEncoder(cols=['city'], smoothing=1.0)

# Fit on train only
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)  # No fit!
```

** When to Use:**
- High cardinality (50+ categories)
- Categories have different target rates
- Tree models or linear models
- Examples: ZIP codes, user IDs, product IDs

** When NOT to Use:**
- Small dataset (overfits rare categories)
- Can't do proper cross-validation
- Need simple, explainable model

**CRITICAL: Prevent Leakage**
```python
#  WRONG - causes leakage!
df['city_encoded'] = df.groupby('city')['target'].transform('mean')

#  CORRECT - use CV
from category_encoders import TargetEncoder
encoder = TargetEncoder(smoothing=1.0, min_samples_leaf=20)
```

**Smoothing:** Blends category mean with global mean to prevent overfitting

---

### 3.1.4 Frequency Encoding

**What:** Replace category with its frequency/count

**Code:**
```python
# Simple frequency encoding
freq_map = df['category'].value_counts(normalize=True).to_dict()
df['category_freq'] = df['category'].map(freq_map)

# Or count encoding
count_map = df['category'].value_counts().to_dict()
df['category_count'] = df['category'].map(count_map)
```

** When to Use:**
- Quick and safe (no leakage)
- Frequency correlates with target
- High cardinality
- Examples: Rare products might be niche/expensive

** When NOT to Use:**
- Frequency unrelated to target
- Need to capture category --> target relationship (use target encoding)

---

### 3.1.5 Ordinal Encoding

**What:** Assign integers respecting natural order

**Code:**
```python
from sklearn.preprocessing import OrdinalEncoder

# Define order
education_order = ['High School', 'Bachelor', 'Master', 'PhD']

encoder = OrdinalEncoder(categories=[education_order])
df['education_encoded'] = encoder.fit_transform(df[['education']])
```

** When to Use:**
- Natural ordering exists (Low/Medium/High, Small/Large)
- Order matters for target
- Both tree and linear models

** When NOT to Use:**
- No natural order (colors, cities)
- Order not equal spacing (Small=1, Medium=2, Large=10?)

---

### 3.1.6 Hash Encoding

**What:** Use hash function to map categories to fixed number of bins

**Code:**
```python
from category_encoders import HashingEncoder

encoder = HashingEncoder(cols=['high_card_feature'], n_components=10)
df_encoded = encoder.fit_transform(df)
```

** When to Use:**
- Very high cardinality (1000+ categories)
- Memory constrained
- Production with new categories appearing

** When NOT to Use:**
- Low cardinality (overkill)
- Collisions are problematic
- Need to reverse encoding

---

### 3.1.7 Binary Encoding

**What:** Convert category to binary, then split into separate columns

**Code:**
```python
from category_encoders import BinaryEncoder

encoder = BinaryEncoder(cols=['category'])
df_encoded = encoder.fit_transform(df)
```

** When to Use:**
- Medium cardinality (10-100 categories)
- Balance between one-hot (too many dims) and label (false ordering)

**Example:**
- 100 categories --> 7 binary columns (2^7 = 128)
- vs. 100 one-hot columns

---

## 3.2 Numerical Transformations

### 3.2.1 Log Transformation

**What:** Apply log(x) or log(x+1) to reduce skewness

**Code:**
```python
import numpy as np

# For positive values
df['feature_log'] = np.log(df['feature'])

# For values including zero
df['feature_log1p'] = np.log1p(df['feature'])  # log(1 + x)
```

** When to Use:**
- Right-skewed distributions (long tail)
- Wide range (e.g., income: $10K to $10M)
- Multiplicative relationships
- **Linear models benefit most**

** When NOT to Use:**
- Negative values (undefined)
- Already normally distributed
- Tree models (handle skewness naturally)

**Visual check:**
```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
df['income'].hist(bins=50, ax=axes[0])
axes[0].set_title('Original')
np.log1p(df['income']).hist(bins=50, ax=axes[1])
axes[1].set_title('Log Transformed')
```

---

### 3.2.2 Square Root & Power Transformations

**Code:**
```python
# Square root (moderate skewness)
df['feature_sqrt'] = np.sqrt(df['feature'])

# Box-Cox (optimizes transformation)
from scipy.stats import boxcox
df['feature_boxcox'], lambda_param = boxcox(df['feature'])

# Yeo-Johnson (handles negatives)
from sklearn.preprocessing import PowerTransformer
pt = PowerTransformer(method='yeo-johnson')
df['feature_transformed'] = pt.fit_transform(df[['feature']])
```

** When to Use:**
- Moderate skewness (sqrt)
- Want optimal transformation (Box-Cox)
- Have negative values (Yeo-Johnson)
- **Linear models and neural nets benefit**

---

### 3.2.3 Scaling Transformations

**Code:**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# StandardScaler (mean=0, std=1)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# MinMaxScaler (range [0, 1])
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# RobustScaler (uses median, IQR - robust to outliers)
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)
```

**When to Use Each:**
- **StandardScaler**: Default for linear models, neural nets, SVM
- **MinMaxScaler**: Need specific range [0, 1], neural nets
- **RobustScaler**: Outliers present

**Never for:** Tree-based models (don't need scaling)

---

## 3.3 DateTime Feature Engineering

### 3.3.1 Basic Components

**Code:**
```python
df['datetime'] = pd.to_datetime(df['datetime'])

# Basic extraction
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df['hour'] = df['datetime'].dt.hour
df['minute'] = df['datetime'].dt.minute
df['day_of_week'] = df['datetime'].dt.dayofweek  # 0=Monday
df['day_of_year'] = df['datetime'].dt.dayofyear
df['week_of_year'] = df['datetime'].dt.isocalendar().week
df['quarter'] = df['datetime'].dt.quarter
```

** When to Use:**
- Seasonality patterns (sales by month)
- Time-of-day effects (traffic by hour)
- Weekday vs weekend patterns

---

### 3.3.2 Binary Indicators

**Code:**
```python
# Weekend indicator
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Business hours
df['is_business_hours'] = df['hour'].between(9, 17).astype(int)

# Month-end
df['is_month_end'] = (df['datetime'].dt.is_month_end).astype(int)

# Holiday (requires holidays library)
import holidays
us_holidays = holidays.US()
df['is_holiday'] = df['datetime'].dt.date.isin(us_holidays).astype(int)
```

** When to Use:**
- Clear binary patterns (weekend sales drop)
- Simpler than full categorical (7 days --> 1 binary)

---

### 3.3.3 Cyclical Encoding

**Why:** Hour 23 is close to hour 0, but numerically far (23 vs 0)

**Code:**
```python
import numpy as np

# Hour (24-hour cycle)
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

# Day of week (7-day cycle)
df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

# Month (12-month cycle)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
```

** When to Use:**
- Cyclical patterns (hour, day, month)
- **Linear models and neural nets** (trees handle it either way)
- Need smooth transitions (23 --> 0)

** When NOT to Use:**
- Tree models (can handle raw values)
- No cyclical pattern (year is not cyclical)

---

### 3.3.4 Time-Based Aggregations

**Code:**
```python
# Purchases in last 30 days
df['purchases_last_30d'] = df.groupby('user_id')['datetime'].transform(
    lambda x: x.rolling('30D').count()
)

# Days since last purchase
df['days_since_last_purchase'] = (
    df.groupby('user_id')['datetime'].diff().dt.days
)

# Days until next holiday
df['days_to_holiday'] = (df['next_holiday_date'] - df['datetime']).dt.days
```

** When to Use:**
- Recency matters (e.g., days since last login predicts churn)
- Aggregated behavior (purchases/month)

---

## 3.4 Polynomial & Interaction Features

### 3.4.1 Polynomial Features

**What:** Create x^2, x^3, etc. to capture non-linear relationships

**Code:**
```python
from sklearn.preprocessing import PolynomialFeatures

# Degree 2: x, y, x^2, xy, y^2
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Get feature names
poly.get_feature_names_out(['x', 'y'])
# Output: ['x', 'y', 'x^2', 'x y', 'y^2']
```

** When to Use:**
- **Linear models** show underfitting
- Curved relationship suspected
- Small number of features (2-10)
- Degree 2 or 3 (rarely higher)

** When NOT to Use:**
- Tree models (find interactions automatically)
- High-dimensional data (exponential growth)
- Degree > 3 (overfitting risk)

**Dimension explosion:**
- 10 features, degree 2 --> 55 features
- 10 features, degree 3 --> 220 features

---

### 3.4.2 Interaction Features (Manual)

**What:** Multiply features to capture joint effects

**Code:**
```python
# Manual interactions
df['sqft_per_bedroom'] = df['sqft'] / df['bedrooms']
df['price_per_sqft'] = df['price'] / df['sqft']
df['income_to_loan_ratio'] = df['income'] / df['loan_amount']

# Categorical x numerical
df['premium_customer_spending'] = df['is_premium'] * df['total_spent']
```

** When to Use:**
- Domain knowledge suggests interaction
- Example: House value depends on sqft AND location
- Linear models
- Small number of specific interactions

** When NOT to Use:**
- Tree models (less benefit)
- No domain knowledge (creates noise)
- High-dimensional already

---

## 3.5 Binning / Discretization

### What It Is

Convert continuous variable into categorical bins.

### Methods

**Equal-Width Binning:**
```python
# Same bin width
df['age_binned'] = pd.cut(df['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
```

**Equal-Frequency (Quantile) Binning:**
```python
# Same number of samples per bin
df['income_binned'] = pd.qcut(df['income'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

**Custom Bins:**
```python
# Domain-driven bins
bins = [0, 18, 35, 50, 65, 100]
labels = ['<18', '18-35', '35-50', '50-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
```

**KBinsDiscretizer (sklearn):**
```python
from sklearn.preprocessing import KBinsDiscretizer

# uniform, quantile, or kmeans
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
df['feature_binned'] = discretizer.fit_transform(df[['feature']])
```

---

###  When to Use Binning

1. **Handle outliers without removal**
   - Extreme values grouped into bins
   - Less sensitive than raw values

2. **Capture non-linear patterns in linear models**
   - Age: 18-25 behaves differently than linear
   - Binning + one-hot creates "age group" features

3. **Simplify model**
   - Reduce noise
   - More interpretable (age groups vs exact age)

4. **Known business thresholds**
   - Credit score: <600, 600-700, >700
   - Use domain knowledge

5. **Improve specific algorithms**
   - Naive Bayes prefers categorical
   - Some tree models with binned features

---

###  When NOT to Use Binning

1. **Loss of information**
   - Age 25 and 35 both become "young"
   - Loses granularity

2. **Tree-based models**
   - Trees find optimal splits automatically
   - Binning prevents finer splits

3. **Small datasets**
   - Bins with few samples --> overfitting

4. **Arbitrary bins**
   - Without domain knowledge, bins might be meaningless

5. **Already discrete/categorical**
   - Don't bin count variables with few values

---

## 3.6 Domain-Specific Features

### 3.6.1 Text Features (Simple)

**Code:**
```python
# Length features
df['text_length'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text_length'] / df['word_count']

# Character features
df['uppercase_count'] = df['text'].str.count(r'[A-Z]')
df['digit_count'] = df['text'].str.count(r'\d')
df['special_char_count'] = df['text'].str.count(r'[^a-zA-Z0-9\s]')

# Binary indicators
df['contains_email'] = df['text'].str.contains(r'\S+@\S+').astype(int)
df['contains_url'] = df['text'].str.contains(r'http').astype(int)
```

---

### 3.6.2 Geospatial Features

**Code:**
```python
# Distance between two points (Haversine formula)
from sklearn.metrics.pairwise import haversine_distances
import numpy as np

def haversine_distance(lat1, lon1, lat2, lon2):
    """Return distance in kilometers"""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km

df['distance_to_center'] = haversine_distance(
    df['lat'], df['lon'],
    city_center_lat, city_center_lon
)

# Clustering (neighborhoods)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=10)
df['geo_cluster'] = kmeans.fit_predict(df[['lat', 'lon']])
```

---

### 3.6.3 E-Commerce Features

**Code:**
```python
# Customer behavior
df['avg_order_value'] = df['total_spent'] / df['num_orders']
df['days_since_first_purchase'] = (df['current_date'] - df['first_purchase_date']).dt.days
df['purchase_frequency'] = df['num_orders'] / df['days_as_customer']

# Product features
df['discount_percentage'] = (df['original_price'] - df['sale_price']) / df['original_price']
df['is_on_sale'] = (df['discount_percentage'] > 0).astype(int)
```

---

## 3.7 Feature Engineering Decision Flow

```
What type of feature?

+--- CATEGORICAL
|  +--- For tree models?
|  |  +--- Low cardinality (<50) --> Label Encoding
|  |  +--- High cardinality (50+) --> Target Encoding (with CV)
|  |
|  +--- For linear models?
|     +--- Low cardinality (<10) --> One-Hot Encoding
|     +--- Medium (10-50) --> Binary Encoding or Target
|     +--- High (50+) --> Target Encoding or Hash
|
+--- NUMERICAL
|  +--- Skewed distribution?
|  |  +--- For linear models --> Log/Box-Cox transform
|  |
|  +--- Different scales?
|  |  +--- For linear models --> StandardScaler
|  |
|  +--- Non-linear relationship (linear model)?
|  |  +--- Polynomial features (degree 2-3)
|  |
|  +--- Tree models --> Use raw values
|
+--- DATETIME
|  +--- Extract: year, month, day, hour, dow
|  +--- Create: is_weekend, is_holiday, is_business_hours
|  +--- Cyclical: hour_sin/cos, month_sin/cos (for linear models)
|  +--- Aggregations: days_since_last_event, events_last_30d
|
+--- TEXT (simple features)
   +--- Length: char_count, word_count
   +--- Counts: uppercase_count, digit_count
   +--- Binary: contains_url, contains_email
```

---

## 3.8 Common Pitfalls & Solutions

### Pitfall 1: Data Leakage in Target Encoding

 **Wrong:**
```python
# Leakage: using full dataset
df['city_encoded'] = df.groupby('city')['target'].transform('mean')
X_train, X_test = train_test_split(df)
```

 **Correct:**
```python
# Split first, then encode
X_train, X_test, y_train, y_test = train_test_split(X, y)

encoder = TargetEncoder()
X_train_encoded = encoder.fit_transform(X_train, y_train)
X_test_encoded = encoder.transform(X_test)
```

---

### Pitfall 2: Fitting Scaler on Full Data

 **Wrong:**
```python
scaler.fit(pd.concat([X_train, X_test]))  # Leakage!
```

 **Correct:**
```python
scaler.fit(X_train)  # Only training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### Pitfall 3: Creating Too Many Polynomial Features

 **Wrong:**
```python
# 100 features, degree 3 --> 176,851 features!
poly = PolynomialFeatures(degree=3)
X_poly = poly.fit_transform(X_100_features)
```

 **Correct:**
```python
# Select few features, use degree 2
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X[['feature1', 'feature2', 'feature3']])
```

---

### Pitfall 4: Arbitrary Binning

 **Wrong:**
```python
# No reasoning
df['age_binned'] = pd.cut(df['age'], bins=5)  # Why 5?
```

 **Correct:**
```python
# Use domain knowledge or data-driven
bins = [0, 18, 35, 50, 65, 100]  # Meaningful age groups
df['age_binned'] = pd.cut(df['age'], bins=bins)

# OR use quantiles for equal distribution
df['age_binned'] = pd.qcut(df['age'], q=4)
```

---

### Pitfall 5: Over-Engineering

**Don't create features just because you can!**

-  Start simple
-  Add features based on domain knowledge
-  Validate each feature improves model
-  Don't blindly create hundreds of features

---

## 3.9 Feature Engineering Checklist

### Before Engineering:
- [ ] Understand domain and business problem
- [ ] Explore data (EDA first!)
- [ ] Identify feature types (categorical, numerical, datetime, text)
- [ ] Check for missing values and outliers

### Categorical Features:
- [ ] Choose encoding based on model type
- [ ] Use label encoding for trees
- [ ] Use one-hot for linear models (low cardinality)
- [ ] Use target encoding for high cardinality (with CV)
- [ ] Avoid data leakage in target encoding

### Numerical Features:
- [ ] Handle skewness (log transform for linear models)
- [ ] Scale features (for linear models, neural nets, SVM)
- [ ] Consider polynomial features (for linear models, if underfitting)
- [ ] Create domain-specific ratios/interactions

### DateTime Features:
- [ ] Extract basic components (year, month, hour, dow)
- [ ] Create binary indicators (weekend, holiday, business hours)
- [ ] Use cyclical encoding for linear models
- [ ] Create time-based aggregations (recency, frequency)

### Validation:
- [ ] Split data BEFORE feature engineering
- [ ] Use pipelines to prevent leakage
- [ ] Validate features improve model performance
- [ ] Remove features that don't help

---

## Resources & Further Reading

**Libraries:**
- category_encoders: https://contrib.scikit-learn.org/category_encoders/
- Feature-engine: https://feature-engine.readthedocs.io/
- scikit-learn preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html

**Books:**
- "Feature Engineering for Machine Learning" by Alice Zheng
- "Feature Engineering and Selection" by Kuhn & Johnson

**Key Concepts:**
- Always prevent data leakage
- Understand your model's requirements
- Domain knowledge > blind feature creation
- Validate every feature's impact

---

**Last Updated:** 2025-10-12
**Next Section:** Statistical Tests (Phase 4)
