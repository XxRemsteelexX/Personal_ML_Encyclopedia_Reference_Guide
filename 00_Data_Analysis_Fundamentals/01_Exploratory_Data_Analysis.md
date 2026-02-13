# 1. Exploratory Data Analysis (EDA)

## Overview

**EDA is the critical first step** in any data analysis project. It helps you understand your data before making assumptions, identify errors, discover patterns, detect outliers, and find relationships between variables.

**Core Philosophy:** "Look at your data before you model it"

---

## 1.1 Why EDA Matters (2025 Perspective)

### Business Impact of Poor Data Quality:
-  Incorrect decision-making
-  Business inefficiencies
-  Decreased customer satisfaction
-  Financial losses
-  Damaged credibility of BI reports

### EDA Benefits:
 Understand data structure and distributions
 Identify data quality issues early
 Discover unexpected patterns
 Inform modeling decisions
 Validate assumptions
 Generate hypotheses

---

## 1.2 The EDA Process

### Step 1: Initial Data Inspection

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('data.csv')

# FIRST LOOK: Always start here
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nBasic Info:")
print(df.info())

print("\nMemory Usage:")
print(df.memory_usage(deep=True))
```

---

### Step 2: Summary Statistics

```python
# Descriptive statistics
print("Numeric Columns Summary:")
print(df.describe())

# For categorical columns
print("\nCategorical Columns Summary:")
print(df.describe(include='object'))

# Custom statistics
print("\nCustom Statistics:")
print(df.agg({
    'column1': ['min', 'max', 'median', 'skew'],
    'column2': ['mean', 'std', 'var']
}))

# Quantiles
print("\nQuantiles:")
print(df.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]))
```

---

### Step 3: Missing Value Analysis

```python
# Missing values count
print("Missing Values:")
missing = df.isnull().sum()
missing_pct = (missing / len(df)) * 100
missing_df = pd.DataFrame({
    'Missing_Count': missing,
    'Percentage': missing_pct
})
print(missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False))

# Visualize missing values
import missingno as msno

# Matrix visualization
msno.matrix(df)
plt.title('Missing Value Matrix')
plt.show()

# Heatmap (correlations of missingness)
msno.heatmap(df)
plt.title('Missing Value Correlations')
plt.show()

# Dendrogram (clusters of missingness)
msno.dendrogram(df)
plt.title('Missing Value Clustering')
plt.show()
```

---

### Step 4: Distribution Analysis

```python
# Histograms for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

fig, axes = plt.subplots(len(numeric_cols), 2, figsize=(15, 5*len(numeric_cols)))

for i, col in enumerate(numeric_cols):
    # Histogram
    df[col].hist(bins=50, ax=axes[i, 0], edgecolor='black')
    axes[i, 0].set_title(f'{col} - Histogram')
    axes[i, 0].set_xlabel(col)
    axes[i, 0].set_ylabel('Frequency')

    # Box plot
    df.boxplot(column=col, ax=axes[i, 1])
    axes[i, 1].set_title(f'{col} - Box Plot')

plt.tight_layout()
plt.show()

# Distribution statistics
for col in numeric_cols:
    print(f"\n{col}:")
    print(f"  Skewness: {df[col].skew():.3f}")
    print(f"  Kurtosis: {df[col].kurtosis():.3f}")

    # Normality test (Shapiro-Wilk for small samples)
    if len(df) < 5000:
        from scipy import stats
        stat, p = stats.shapiro(df[col].dropna())
        print(f"  Shapiro-Wilk: stat={stat:.3f}, p-value={p:.4f}")
        print(f"  Normal? {'Yes' if p > 0.05 else 'No'}")
```

---

### Step 5: Categorical Variable Analysis

```python
# Value counts for categorical variables
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

for col in categorical_cols:
    print(f"\n{col} Value Counts:")
    print(df[col].value_counts())
    print(f"Unique values: {df[col].nunique()}")

    # Bar plot
    plt.figure(figsize=(10, 6))
    df[col].value_counts().head(20).plot(kind='bar')
    plt.title(f'{col} - Top 20 Values')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Cardinality analysis
print("\nCardinality Analysis:")
for col in categorical_cols:
    cardinality = df[col].nunique()
    cardinality_pct = (cardinality / len(df)) * 100
    print(f"{col}: {cardinality} unique ({cardinality_pct:.2f}%)")
```

---

### Step 6: Outlier Detection

```python
# IQR Method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    print(f"\n{column}:")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    return outliers

# Apply to all numeric columns
for col in numeric_cols:
    outliers = detect_outliers_iqr(df, col)

# Z-score method
from scipy import stats

def detect_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column].dropna()))
    outliers = df[column][z_scores > threshold]

    print(f"\n{column} (Z-score > {threshold}):")
    print(f"  Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.2f}%)")

    return outliers

# Apply Z-score method
for col in numeric_cols:
    outliers = detect_outliers_zscore(df, col)
```

---

### Step 7: Correlation Analysis

```python
# Correlation matrix
corr = df[numeric_cols].corr()

# Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.show()

# Strong correlations (|r| > 0.7)
print("\nStrong Correlations (|r| > 0.7):")
strong_corr = []
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        if abs(corr.iloc[i, j]) > 0.7:
            strong_corr.append({
                'Variable 1': corr.columns[i],
                'Variable 2': corr.columns[j],
                'Correlation': corr.iloc[i, j]
            })

strong_corr_df = pd.DataFrame(strong_corr)
print(strong_corr_df.sort_values('Correlation', ascending=False))

# Pairplot for selected variables
sns.pairplot(df[numeric_cols].sample(min(1000, len(df))))
plt.suptitle('Pairplot of Numeric Variables', y=1.01)
plt.show()
```

---

### Step 8: Relationship with Target Variable

```python
# Assuming 'target' is your target variable

# Numeric features vs target
for col in numeric_cols:
    if col != 'target':
        plt.figure(figsize=(12, 4))

        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(df[col], df['target'], alpha=0.5)
        plt.xlabel(col)
        plt.ylabel('Target')
        plt.title(f'{col} vs Target')

        # Box plot by target bins
        plt.subplot(1, 2, 2)
        df_binned = df.copy()
        df_binned['target_bin'] = pd.cut(df['target'], bins=5)
        df_binned.boxplot(column=col, by='target_bin')
        plt.xlabel('Target Bin')
        plt.ylabel(col)
        plt.title(f'{col} Distribution by Target')
        plt.suptitle('')

        plt.tight_layout()
        plt.show()

# Categorical features vs target
for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    df.groupby(col)['target'].mean().sort_values().plot(kind='barh')
    plt.xlabel('Average Target')
    plt.ylabel(col)
    plt.title(f'Average Target by {col}')
    plt.tight_layout()
    plt.show()
```

---

## 1.3 Advanced EDA Techniques (2025)

### Automated EDA Libraries

```python
# Pandas Profiling (now ydata-profiling)
from ydata_profiling import ProfileReport

profile = ProfileReport(df, title="Data Profiling Report", explorative=True)
profile.to_file("data_report.html")

# Key features:
# - Automatic type inference
# - Missing value analysis
# - Correlation analysis
# - Duplicate detection
# - Warnings for data quality issues
```

```python
# Sweetviz
import sweetviz as sv

report = sv.analyze(df)
report.show_html("sweetviz_report.html")

# Compare datasets
train_df = df.sample(frac=0.8)
test_df = df.drop(train_df.index)
compare_report = sv.compare([train_df, "Train"], [test_df, "Test"])
compare_report.show_html("comparison_report.html")
```

```python
# D-Tale (Interactive)
import dtale

d = dtale.show(df)
d.open_browser()

# Features:
# - Interactive filtering
# - Column analysis
# - Correlations
# - Charts & graphs
# - Data export
```

---

### Statistical Tests for EDA

```python
from scipy import stats

# Test for normality
def test_normality(df, column):
    """Test if data is normally distributed"""
    # Shapiro-Wilk (n < 5000)
    if len(df) < 5000:
        stat, p = stats.shapiro(df[column].dropna())
        print(f"Shapiro-Wilk: stat={stat:.4f}, p={p:.4f}")

    # Kolmogorov-Smirnov (large samples)
    stat, p = stats.kstest(df[column].dropna(), 'norm')
    print(f"K-S Test: stat={stat:.4f}, p={p:.4f}")

    # Anderson-Darling
    result = stats.anderson(df[column].dropna())
    print(f"Anderson-Darling: stat={result.statistic:.4f}")

# Test for independence (categorical variables)
def test_independence(df, col1, col2):
    """Chi-square test of independence"""
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = stats.chi2_contingency(contingency)

    print(f"Chi-square: {chi2:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"DOF: {dof}")
    print(f"Independent: {'Yes' if p > 0.05 else 'No'}")

# Test for correlation significance
def test_correlation(df, col1, col2):
    """Test if correlation is significant"""
    r, p = stats.pearsonr(df[col1].dropna(), df[col2].dropna())

    print(f"Pearson r: {r:.4f}")
    print(f"P-value: {p:.4f}")
    print(f"Significant: {'Yes' if p < 0.05 else 'No'}")
```

---

### Time Series EDA

```python
# For time series data
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df['value'], model='additive', period=12)

fig, axes = plt.subplots(4, 1, figsize=(15, 12))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()

# Stationarity test (Augmented Dickey-Fuller)
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    result = adfuller(timeseries.dropna())

    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    if result[1] < 0.05:
        print("[x] Series is stationary")
    else:
        print("[ ] Series is non-stationary")

test_stationarity(df['value'])
```

---

## 1.4 EDA Best Practices (2025)

### 1. Document Everything

```python
# Create markdown report as you go
with open('eda_findings.md', 'w') as f:
    f.write("# EDA Findings\n\n")
    f.write(f"## Dataset Overview\n")
    f.write(f"- Shape: {df.shape}\n")
    f.write(f"- Date: {pd.Timestamp.now()}\n\n")

    f.write("## Key Findings\n")
    f.write("1. Missing values in columns X, Y, Z\n")
    f.write("2. Strong correlation between A and B\n")
    f.write("3. Outliers detected in column C\n")
```

---

### 2. Version Control Your Analysis

```python
# Save processed data with timestamp
timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
df.to_csv(f'data_processed_{timestamp}.csv', index=False)

# Save EDA notebook with version
# Jupyter: File --> Save As --> notebook_eda_v1.ipynb
```

---

### 3. Create Reusable Functions

```python
def eda_report(df):
    """Generate comprehensive EDA report"""

    report = {
        'shape': df.shape,
        'memory_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'duplicates': df.duplicated().sum(),
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().sum(),
        'missing_pct': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    }

    return report

# Usage
summary = eda_report(df)
print(pd.Series(summary))
```

---

### 4. Validate Your Findings

```python
# After cleaning, re-run checks
def validate_cleaning(df_before, df_after):
    """Validate data cleaning didn't introduce issues"""

    print("Validation Report:")
    print(f"Shape before: {df_before.shape}")
    print(f"Shape after: {df_after.shape}")
    print(f"Rows removed: {df_before.shape[0] - df_after.shape[0]}")

    # Check distributions didn't change dramatically
    for col in df_before.select_dtypes(include=[np.number]).columns:
        mean_before = df_before[col].mean()
        mean_after = df_after[col].mean()
        pct_change = ((mean_after - mean_before) / mean_before) * 100

        if abs(pct_change) > 10:
            print(f"[WARNING] {col} mean changed by {pct_change:.1f}%")
```

---

## 1.5 EDA Checklist

Use this checklist for every dataset:

### Data Understanding
- [ ] Load data and check shape
- [ ] Inspect first/last rows
- [ ] Check data types
- [ ] Understand each column's meaning

### Data Quality
- [ ] Check for missing values
- [ ] Identify duplicates
- [ ] Detect outliers
- [ ] Validate logical relationships (e.g., age < 150, dates in valid range)

### Distributions
- [ ] Plot histograms for numeric variables
- [ ] Check skewness and kurtosis
- [ ] Test for normality if needed
- [ ] Analyze categorical value counts

### Relationships
- [ ] Correlation matrix
- [ ] Scatter plots
- [ ] Group-by analyses
- [ ] Target variable relationships

### Documentation
- [ ] Document key findings
- [ ] Note data quality issues
- [ ] List assumptions made
- [ ] Save processed data

---

## Resources

**Books:**
- "Exploratory Data Analysis" by Tukey (1977) - Classic
- "R for Data Science" by Wickham & Grolemund
- "Python Data Science Handbook" by VanderPlas

**Libraries (2025):**
- pandas, numpy, matplotlib, seaborn (core)
- ydata-profiling (automated reports)
- sweetviz (visual comparisons)
- dtale (interactive)
- missingno (missing value visualization)

**Online:**
- Kaggle Learn: Data Visualization
- DataCamp: Exploratory Data Analysis in Python
- Coursera: Applied Data Science with Python
