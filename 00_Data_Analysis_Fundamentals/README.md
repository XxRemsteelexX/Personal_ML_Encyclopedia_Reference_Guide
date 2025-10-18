# Data Analysis Fundamentals - Complete Guide

## Overview

This folder contains **comprehensive data analyst and data scientist fundamentals** - everything you need to go from raw data to model-ready datasets.

**Coverage:** EDA → Cleaning → Feature Engineering → Scaling → Encoding

---

## 📁 Files in This Folder

### 1. **01_Exploratory_Data_Analysis.md**

**Complete EDA Process:**
- Initial data inspection (shape, types, info)
- Summary statistics (describe, quantiles)
- Missing value analysis (missingno visualizations)
- Distribution analysis (histograms, box plots, skewness, kurtosis)
- Categorical variable analysis (value counts, cardinality)
- Outlier detection (IQR, Z-score, Isolation Forest)
- Correlation analysis (heatmaps, pairplots)
- Target variable relationships

**Advanced Techniques:**
- Automated EDA (pandas-profiling, sweetviz, dtale)
- Statistical tests (normality, independence, correlation)
- Time series EDA (decomposition, stationarity)

**Best Practices:**
- Document everything
- Version control
- Reusable functions
- Validation after cleaning

**Includes:** Complete Python examples, visualization code, EDA checklist

---

### 2. **02_Data_Cleaning.md**

**Complete Data Cleaning Process:**

**Missing Values:**
- Understanding missingness (MCAR, MAR, MNAR)
- Deletion strategies
- Simple imputation (mean, median, mode, forward/backward fill)
- Advanced imputation (KNN, MICE, predictive models)
- Missing indicator features

**Duplicates:**
- Exact duplicate removal
- Fuzzy duplicate detection (Levenshtein distance)
- Conditional duplicates

**Data Types:**
- Numeric conversion with error handling
- Datetime parsing
- Categorical conversion
- String cleaning

**Outliers:**
- Detection: IQR, Z-score, Isolation Forest
- Handling: Remove, cap (winsorization), transform, flag

**Standardization:**
- Column names
- Text values
- Date/numeric formats

**Validation:**
- Logical relationships (age < 150, start < end date)
- Business rules (revenue = price × quantity)
- Referential integrity

**Automation:**
- Reusable DataCleaningPipeline class
- Great Expectations framework
- PyCaret auto preprocessing

**Includes:** Complete examples, pipeline code, validation checks, best practices

---

### 3. **03_Feature_Scaling_and_Encoding.md**

**Feature Scaling Methods:**

1. **Min-Max Normalization** (0-1 scaling)
   - When: Neural networks, bounded values
   - Sensitive to outliers

2. **Standardization** (Z-score)
   - **Most common choice**
   - Mean=0, Std=1
   - Less sensitive to outliers

3. **Robust Scaler**
   - **Best for outliers**
   - Uses median and IQR

4. **MaxAbs Scaler**
   - Sparse data
   - Preserves sparsity

5. **Unit Vector Scaling**
   - Text data, cosine similarity
   - Normalizes to unit length

6. **Log Transformation**
   - Skewed distributions
   - Box-Cox, Yeo-Johnson

**Categorical Encoding Methods:**

1. **Label Encoding**
   - Ordinal categories (Low < Medium < High)

2. **One-Hot Encoding**
   - Nominal categories (no order)
   - **Most common for <10 categories**

3. **Frequency Encoding**
   - High cardinality (100+ categories)

4. **Target Encoding**
   - Mean target per category
   - ⚠️ Use cross-validation!

5. **Binary Encoding**
   - Medium cardinality (10-100)
   - More compact than one-hot

6. **Hash Encoding**
   - Very high cardinality (1000+)
   - Memory efficient

7. **Embeddings**
   - Deep learning
   - Learns representations

**DateTime Features:**
- Extract: year, month, day, day_of_week, hour
- Engineer: is_weekend, is_business_hours
- Cyclical encoding (sin/cos for months/hours)

**Complete Pipeline:**
- ColumnTransformer for mixed data types
- Pipeline with imputation + scaling + encoding
- Production-ready code

**Best Practices:**
- Split BEFORE scaling (avoid data leakage!)
- Save scalers for production
- Handle unknown categories
- Document all transformations

**Includes:** Comparison tables, decision trees, complete pipeline examples

---

## 🎯 Quick Start Guide

### For Data Analysts:

**Step 1:** Start with EDA
```python
# Read file 01_Exploratory_Data_Analysis.md
# Follow the 8-step EDA process
# Use automated tools (pandas-profiling, sweetviz)
```

**Step 2:** Clean your data
```python
# Read file 02_Data_Cleaning.md
# Use DataCleaningPipeline class
# Document all changes
```

**Step 3:** Prepare features
```python
# Read file 03_Feature_Scaling_and_Encoding.md
# Scale numeric features (StandardScaler)
# Encode categorical features (OneHotEncoder)
```

---

### For Data Scientists:

**Use all three files in sequence:**

1. **EDA** → Understand data, identify issues
2. **Cleaning** → Fix issues, validate
3. **Scaling/Encoding** → Prepare for modeling

**Then move to:**
- `01_Statistical_Foundations/` for A/B testing, hypothesis testing
- `02_Classical_Machine_Learning/` for modeling
- `05_NLP_and_Transformers/` for text data
- `06_Generative_Models/` for generative AI

---

## 📊 What's Covered

### Data Quality Issues:
✅ Missing values (all types and solutions)
✅ Duplicates (exact and fuzzy)
✅ Outliers (detection and handling)
✅ Data type errors
✅ Inconsistent formats

### Feature Engineering:
✅ Scaling (6 methods with comparisons)
✅ Encoding (7 methods with decision tree)
✅ DateTime features
✅ Cyclical features

### Validation:
✅ Logical relationship checks
✅ Business rule validation
✅ Distribution verification
✅ Data leakage prevention

### Automation:
✅ Reusable pipelines
✅ Great Expectations
✅ PyCaret
✅ Production-ready code

---

## 🔍 Key Concepts

### EDA (01_Exploratory_Data_Analysis.md):
- **8-step process** for thorough data understanding
- **Automated tools** for quick insights
- **Statistical tests** for validation
- **Visualization** best practices

### Cleaning (02_Data_Cleaning.md):
- **EDA → Clean → Validate loop**
- **ALWAYS backup original data**
- **Document every decision**
- **Reusable pipeline approach**

### Scaling/Encoding (03_Feature_Scaling_and_Encoding.md):
- **Split FIRST, scale SECOND** (avoid leakage!)
- **Choose encoding based on cardinality**
- **Save transformers for production**
- **Pipeline everything**

---

## 💡 Best Practices Across All Files

### 1. Always Work on a Copy
```python
df_original = pd.read_csv('data.csv')
df = df_original.copy()  # Work on this
```

### 2. Split Before Scaling
```python
# ✅ Correct
X_train, X_test = train_test_split(X, y)
scaler.fit(X_train)  # Fit on train only!
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ❌ Wrong (data leakage!)
X_scaled = scaler.fit_transform(X)  # Test info leaks!
X_train, X_test = train_test_split(X_scaled)
```

### 3. Document Everything
```python
# Keep a cleaning log
cleaning_log = []
cleaning_log.append({
    'step': 'Remove duplicates',
    'rows_before': len(df),
    'rows_after': len(df_cleaned)
})
```

### 4. Use Pipelines
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier())
])
```

### 5. Save for Production
```python
import joblib

# Save pipeline
joblib.dump(pipeline, 'pipeline.pkl')

# Load and use
pipeline = joblib.load('pipeline.pkl')
predictions = pipeline.predict(new_data)
```

---

## 🎓 Learning Path

### Beginner:
1. Read 01_EDA (focus on sections 1.1-1.7)
2. Practice with pandas-profiling for automated EDA
3. Read 02_Cleaning (sections 2.1-2.6)
4. Use simple imputation and StandardScaler
5. Master One-Hot Encoding

### Intermediate:
1. Advanced EDA (statistical tests, time series)
2. Advanced imputation (KNN, MICE)
3. All scaling methods
4. Target encoding, frequency encoding
5. Build reusable pipelines

### Advanced:
1. DataCleaningPipeline class customization
2. Great Expectations for validation
3. Custom transformers
4. Feature engineering automation
5. Production deployment patterns

---

## 📚 Related Files

**After mastering this folder, move to:**

- `01_Statistical_Foundations/` - A/B testing, hypothesis testing, Bayesian methods
- `02_Classical_Machine_Learning/` - Linear models, trees, ensembles
- `05_NLP_and_Transformers/` - Text data specific techniques
- `RESEARCH_SUMMARY_2025.md` - All latest research findings

---

## ✅ Checklists

### EDA Checklist (from file 01):
- [ ] Load and inspect data
- [ ] Check for missing values
- [ ] Analyze distributions
- [ ] Detect outliers
- [ ] Correlation analysis
- [ ] Document findings

### Cleaning Checklist (from file 02):
- [ ] Backup original data
- [ ] Handle missing values
- [ ] Remove duplicates
- [ ] Fix data types
- [ ] Handle outliers
- [ ] Validate results

### Preprocessing Checklist (from file 03):
- [ ] Split train/test first
- [ ] Scale numeric features
- [ ] Encode categorical features
- [ ] Handle datetime features
- [ ] Save transformers
- [ ] Validate no leakage

---

## 🏆 What Makes This Unique

### Comprehensive Coverage:
✅ **Everything** from raw data to model-ready
✅ **2025 best practices** (Great Expectations, PyCaret)
✅ **Production-ready** code examples
✅ **Decision frameworks** (when to use what)

### Practical Focus:
✅ Working code you can copy-paste
✅ Common pitfalls highlighted
✅ Reusable classes and functions
✅ Real-world examples

### Data Analyst + Data Scientist:
✅ EDA for analysts
✅ Statistical validation
✅ Feature engineering for modeling
✅ Production deployment patterns

---

## 🚀 Quick Reference

**Need to...**

**Understand your data?** → 01_EDA.md (8-step process)

**Handle missing values?** → 02_Cleaning.md (section 2.2, 9 methods)

**Remove outliers?** → 01_EDA.md (section 1.7) + 02_Cleaning.md (section 2.5)

**Scale features?** → 03_Scaling_Encoding.md (section 3.1, 6 methods + comparison)

**Encode categories?** → 03_Scaling_Encoding.md (section 3.2, 7 methods + decision tree)

**Build pipeline?** → 03_Scaling_Encoding.md (section 3.4, complete example)

**Validate data quality?** → 02_Cleaning.md (section 2.7 + Great Expectations)

---

**You now have COMPLETE data analysis and preprocessing coverage for 2025! 🎉**

**Total Content:** 3 comprehensive files covering every aspect of data preparation
**Quality:** Production-ready, best practices, 2025 state-of-the-art
**Format:** Working code, checklists, decision frameworks
