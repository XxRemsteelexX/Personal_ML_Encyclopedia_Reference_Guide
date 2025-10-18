# 2. Data Cleaning and Preparation

## Overview

**Data cleaning is 80% of data analysis work.** Poor data quality leads to incorrect decisions, business inefficiencies, and financial losses.

**2025 Truth:** AI-driven automation is transforming data cleaning, but human judgment remains critical.

---

## 2.1 The Data Cleaning Process

### EDA → Clean → Validate Loop

```
1. Exploratory Data Analysis (identify issues)
   ↓
2. Data Cleaning (fix issues)
   ↓
3. Validation (confirm fixes worked)
   ↓
4. Repeat if needed
```

**Critical Rule:** **ALWAYS make a copy of original data before cleaning!**

```python
import pandas as pd
import numpy as np

# ALWAYS DO THIS FIRST
df_original = pd.read_csv('data.csv')
df = df_original.copy()  # Work on copy

# Save original
df_original.to_csv('data_original_backup.csv', index=False)
```

---

## 2.2 Missing Value Handling

### Step 1: Understand WHY Data is Missing

**Three Types of Missingness:**

1. **MCAR (Missing Completely At Random)**
   - No pattern to missingness
   - Example: Random sensor failures
   - **Safe to delete or impute**

2. **MAR (Missing At Random)**
   - Missingness depends on observed data
   - Example: Older people less likely to report income
   - **Can impute using related variables**

3. **MNAR (Missing Not At Random)**
   - Missingness depends on unobserved data
   - Example: High earners don't report income
   - **Most problematic - need domain expertise**

```python
# Analyze missingness patterns
import missingno as msno

# Matrix plot
msno.matrix(df)

# Correlation of missingness
msno.heatmap(df)

# Dendrogram (clusters of missingness)
msno.dendrogram(df)
```

---

### Step 2: Decide on Strategy

```python
# Missing value analysis
missing_info = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum(),
    'Missing_Pct': (df.isnull().sum() / len(df)) * 100,
    'Data_Type': df.dtypes
})
missing_info = missing_info[missing_info['Missing_Count'] > 0]
missing_info = missing_info.sort_values('Missing_Pct', ascending=False)

print(missing_info)

# Decision matrix
def missing_value_strategy(missing_pct):
    if missing_pct > 50:
        return "Consider dropping column"
    elif missing_pct > 20:
        return "Impute carefully or flag"
    elif missing_pct > 5:
        return "Impute with domain knowledge"
    else:
        return "Simple imputation OK"

missing_info['Strategy'] = missing_info['Missing_Pct'].apply(missing_value_strategy)
print(missing_info)
```

---

### Step 3: Handle Missing Values

#### Option 1: Deletion

```python
# Drop rows with ANY missing values (use sparingly)
df_complete = df.dropna()

# Drop rows with missing in SPECIFIC columns
df_cleaned = df.dropna(subset=['important_column1', 'important_column2'])

# Drop rows with MORE than threshold missing
threshold = 0.5  # 50% missing
df_cleaned = df.dropna(thresh=int(threshold * len(df.columns)))

# Drop columns with too many missing
threshold = 0.3  # 30% missing
df_cleaned = df.dropna(axis=1, thresh=int(threshold * len(df)))
```

#### Option 2: Simple Imputation

```python
# Mean (numeric, no outliers)
df['column'].fillna(df['column'].mean(), inplace=True)

# Median (numeric, with outliers)
df['column'].fillna(df['column'].median(), inplace=True)

# Mode (categorical)
df['column'].fillna(df['column'].mode()[0], inplace=True)

# Forward fill (time series)
df['column'].fillna(method='ffill', inplace=True)

# Backward fill
df['column'].fillna(method='bfill', inplace=True)

# Constant value
df['column'].fillna(0, inplace=True)
df['category_col'].fillna('Unknown', inplace=True)
```

#### Option 3: Advanced Imputation

```python
# KNN Imputation
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

# Iterative Imputation (MICE)
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    imputer.fit_transform(df[numeric_cols]),
    columns=numeric_cols
)

# Use predictive model
from sklearn.ensemble import RandomForestRegressor

def impute_with_model(df, target_col):
    """Impute using Random Forest"""
    # Split into rows with/without missing
    df_missing = df[df[target_col].isnull()]
    df_not_missing = df[df[target_col].notnull()]

    if len(df_missing) == 0:
        return df

    # Features (other columns)
    feature_cols = [c for c in df.columns if c != target_col]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    X_train = df_not_missing[feature_cols]
    y_train = df_not_missing[target_col]
    rf.fit(X_train, y_train)

    # Predict missing
    X_predict = df_missing[feature_cols]
    predictions = rf.predict(X_predict)

    # Fill in predictions
    df.loc[df[target_col].isnull(), target_col] = predictions

    return df
```

#### Option 4: Create Missing Indicator

```python
# Add binary column for missingness
df['column_was_missing'] = df['column'].isnull().astype(int)

# Then impute
df['column'].fillna(df['column'].median(), inplace=True)

# Now model can learn if missingness is informative
```

---

## 2.3 Duplicate Removal

```python
# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# See duplicates
duplicates = df[df.duplicated(keep=False)]
print(duplicates.sort_values(by=df.columns.tolist()))

# Remove exact duplicates
df_cleaned = df.drop_duplicates()

# Remove duplicates based on specific columns
df_cleaned = df.drop_duplicates(subset=['id', 'date'], keep='first')

# Keep last occurrence
df_cleaned = df.drop_duplicates(subset=['id'], keep='last')

# Fuzzy duplicate detection (e.g., similar names)
from fuzzywuzzy import fuzz

def find_fuzzy_duplicates(df, column, threshold=90):
    """Find similar strings using Levenshtein distance"""
    duplicates = []
    values = df[column].unique()

    for i, val1 in enumerate(values):
        for val2 in values[i+1:]:
            similarity = fuzz.ratio(str(val1), str(val2))
            if similarity >= threshold:
                duplicates.append((val1, val2, similarity))

    return pd.DataFrame(duplicates, columns=['Value1', 'Value2', 'Similarity'])

# Usage
fuzzy_dups = find_fuzzy_duplicates(df, 'company_name', threshold=85)
print(fuzzy_dups)
```

---

## 2.4 Data Type Corrections

```python
# Check current types
print(df.dtypes)

# Convert to numeric (handle errors)
df['numeric_col'] = pd.to_numeric(df['numeric_col'], errors='coerce')

# Convert to datetime
df['date_col'] = pd.to_datetime(df['date_col'], errors='coerce')

# Multiple date formats
df['date_col'] = pd.to_datetime(df['date_col'], infer_datetime_format=True)

# Convert to categorical (saves memory)
df['category_col'] = df['category_col'].astype('category')

# Boolean conversion
df['bool_col'] = df['bool_col'].map({'Yes': True, 'No': False})

# String cleaning
df['text_col'] = df['text_col'].str.strip()  # Remove whitespace
df['text_col'] = df['text_col'].str.lower()  # Lowercase
df['text_col'] = df['text_col'].str.replace('[^a-zA-Z0-9]', '', regex=True)  # Remove special chars
```

---

## 2.5 Handling Outliers

### Detect Outliers

```python
# IQR Method
def detect_outliers_iqr(df, column, multiplier=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - multiplier * IQR
    upper = Q3 + multiplier * IQR

    outliers = df[(df[column] < lower) | (df[column] > upper)]
    return outliers, lower, upper

# Z-score Method
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[z_scores > threshold]
    return outliers

# Isolation Forest (multivariate)
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination=0.1, random_state=42)
outlier_labels = iso_forest.fit_predict(df[numeric_cols])
df['is_outlier'] = outlier_labels == -1
```

### Handle Outliers

```python
# Option 1: Remove
outliers, lower, upper = detect_outliers_iqr(df, 'column')
df_cleaned = df[~df.index.isin(outliers.index)]

# Option 2: Cap (Winsorization)
df['column_capped'] = df['column'].clip(lower=lower, upper=upper)

# Option 3: Transform
df['column_log'] = np.log1p(df['column'])  # Log transform
df['column_sqrt'] = np.sqrt(df['column'])  # Square root

# Option 4: Flag and keep
df['is_outlier'] = False
df.loc[outliers.index, 'is_outlier'] = True
```

---

## 2.6 Standardization & Normalization

```python
# Standardize column names
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)

# Standardize text values
df['country'] = df['country'].str.lower().str.strip()
df['country'] = df['country'].replace({
    'usa': 'united states',
    'u.s.a.': 'united states',
    'us': 'united states'
})

# Standardize dates
df['date'] = pd.to_datetime(df['date']).dt.date

# Standardize numeric formats
df['price'] = df['price'].str.replace('$', '').str.replace(',', '').astype(float)
df['percentage'] = df['percentage'].str.replace('%', '').astype(float) / 100
```

---

## 2.7 Data Validation

### Check Logical Relationships

```python
# Age can't exceed 150
assert df['age'].max() <= 150, "Invalid age found"

# Start date before end date
assert (df['end_date'] >= df['start_date']).all(), "End date before start date"

# Non-negative counts
assert (df['count'] >= 0).all(), "Negative count found"

# Valid email format
import re
email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
df['valid_email'] = df['email'].str.match(email_pattern)
print(f"Invalid emails: {(~df['valid_email']).sum()}")

# Referential integrity (foreign keys)
valid_customer_ids = customers_df['customer_id'].unique()
invalid = df[~df['customer_id'].isin(valid_customer_ids)]
print(f"Invalid customer_ids: {len(invalid)}")
```

---

### Check Business Rules

```python
# Revenue = price × quantity
df['calculated_revenue'] = df['price'] * df['quantity']
discrepancy = df['revenue'] != df['calculated_revenue']
print(f"Revenue discrepancies: {discrepancy.sum()}")

# Total should equal sum of parts
df['total_calculated'] = df[['part1', 'part2', 'part3']].sum(axis=1)
diff = abs(df['total'] - df['total_calculated'])
print(f"Total mismatches: {(diff > 0.01).sum()}")
```

---

## 2.8 Feature Engineering During Cleaning

```python
# Extract date components
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Create bins
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 50, 65, 100],
                          labels=['<18', '18-35', '35-50', '50-65', '65+'])

# Extract from text
df['domain'] = df['email'].str.split('@').str[1]
df['area_code'] = df['phone'].str[:3]

# Combine columns
df['full_name'] = df['first_name'] + ' ' + df['last_name']

# Flag creation
df['high_value'] = (df['purchase_amount'] > df['purchase_amount'].quantile(0.9)).astype(int)
```

---

## 2.9 Data Cleaning Pipeline (2025 Best Practice)

```python
class DataCleaningPipeline:
    """Reusable data cleaning pipeline"""

    def __init__(self, df):
        self.df_original = df.copy()
        self.df = df.copy()
        self.cleaning_log = []

    def log_step(self, step_name, rows_before, rows_after):
        self.cleaning_log.append({
            'step': step_name,
            'rows_before': rows_before,
            'rows_after': rows_after,
            'rows_removed': rows_before - rows_after,
            'pct_removed': ((rows_before - rows_after) / rows_before) * 100 if rows_before > 0 else 0
        })

    def remove_duplicates(self, subset=None):
        rows_before = len(self.df)
        self.df = self.df.drop_duplicates(subset=subset)
        rows_after = len(self.df)
        self.log_step('Remove Duplicates', rows_before, rows_after)
        return self

    def handle_missing(self, strategy='drop', threshold=0.5):
        rows_before = len(self.df)

        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'drop_columns':
            self.df = self.df.dropna(axis=1, thresh=int(threshold * len(self.df)))
        elif strategy == 'median':
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())

        rows_after = len(self.df)
        self.log_step('Handle Missing', rows_before, rows_after)
        return self

    def remove_outliers(self, columns, method='iqr', multiplier=1.5):
        rows_before = len(self.df)

        for col in columns:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR

                self.df = self.df[(self.df[col] >= lower) & (self.df[col] <= upper)]

        rows_after = len(self.df)
        self.log_step('Remove Outliers', rows_before, rows_after)
        return self

    def standardize_text(self, columns):
        rows_before = len(self.df)

        for col in columns:
            self.df[col] = self.df[col].str.lower().str.strip()

        rows_after = len(self.df)
        self.log_step('Standardize Text', rows_before, rows_after)
        return self

    def get_cleaned_data(self):
        return self.df

    def get_report(self):
        return pd.DataFrame(self.cleaning_log)

# Usage
pipeline = DataCleaningPipeline(df)
cleaned_df = (pipeline
              .remove_duplicates()
              .handle_missing(strategy='median')
              .remove_outliers(['age', 'income'], method='iqr')
              .standardize_text(['name', 'city'])
              .get_cleaned_data())

# See what was done
report = pipeline.get_report()
print(report)
```

---

## 2.10 Data Cleaning Checklist (2025)

### Before You Start:
- [ ] **Make backup of original data**
- [ ] Understand business context
- [ ] Document assumptions

### Cleaning Steps:
- [ ] Remove exact duplicates
- [ ] Check for fuzzy duplicates
- [ ] Handle missing values (understand WHY first)
- [ ] Correct data types
- [ ] Standardize formats (dates, text, numbers)
- [ ] Detect and handle outliers
- [ ] Validate logical relationships
- [ ] Check business rules

### After Cleaning:
- [ ] **Re-run EDA to validate**
- [ ] Compare distributions before/after
- [ ] Document all changes made
- [ ] Save cleaned data with timestamp
- [ ] Create data cleaning report

---

## 2.11 Automated Data Cleaning (2025)

```python
# Great Expectations (data validation framework)
import great_expectations as gx

# Create expectation suite
suite = gx.core.ExpectationSuite(expectation_suite_name="my_suite")

# Add expectations
suite.add_expectation(
    gx.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_not_be_null",
        kwargs={"column": "customer_id"}
    )
)

suite.add_expectation(
    gx.core.ExpectationConfiguration(
        expectation_type="expect_column_values_to_be_between",
        kwargs={"column": "age", "min_value": 0, "max_value": 150}
    )
)

# Validate data
results = context.run_validation_operator(
    "action_list_operator",
    assets_to_validate=[batch],
    run_id="my_run_id"
)
```

```python
# PyCaret for automated preprocessing
from pycaret.classification import setup

# One line to handle missing, encoding, scaling, etc.
clf_setup = setup(
    data=df,
    target='target_column',
    session_id=42,
    normalize=True,
    transformation=True,
    ignore_low_variance=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.9
)
```

---

## 2.12 Common Mistakes to Avoid

### ❌ Don't Do This:

1. **Clean without understanding**
   ```python
   df.dropna()  # Might delete 90% of data!
   ```

2. **Impute everything with mean**
   ```python
   df.fillna(df.mean())  # Destroys relationships
   ```

3. **Delete outliers without investigation**
   ```python
   df = df[df['column'] < df['column'].quantile(0.99)]  # Might be real VIPs!
   ```

4. **Forget to document**
   ```python
   # No comments, no log, can't reproduce
   ```

### ✅ Do This Instead:

1. **Understand before cleaning**
2. **Use domain knowledge**
3. **Document every decision**
4. **Validate after cleaning**
5. **Keep original data**

---

## Resources

**Books:**
- "Data Cleaning" by Ihaka & Gentleman
- "Bad Data Handbook" by McCallum

**Libraries (2025):**
- pandas (core)
- Great Expectations (validation)
- PyCaret (auto preprocessing)
- pandas-profiling (quality reports)

**Tools:**
- OpenRefine (interactive cleaning)
- Trifacta Wrangler
- Talend Data Preparation

**Best Practices:**
- Document everything in code
- Version control (git)
- Unit tests for data quality
- Automated pipelines for production
