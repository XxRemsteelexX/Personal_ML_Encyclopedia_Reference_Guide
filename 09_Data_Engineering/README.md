# Data Engineering for Machine Learning

## Overview

This section covers the essential data engineering practices and technologies that form the foundation of production-grade machine learning systems. Data engineering is the critical bridge between raw data and ML models, encompassing preprocessing, augmentation, handling imbalanced datasets, and working with big data technologies.

**Target Audience:** Data Scientists, ML Engineers, Data Engineers working on ML systems

**Prerequisites:**
- Python programming proficiency
- Understanding of pandas, numpy, scikit-learn
- Basic SQL and data manipulation concepts
- Familiarity with ML workflows

---

## Section Contents

### 41. Data Preprocessing
**File:** `41_Data_Preprocessing.md`

Comprehensive guide to production-ready data preprocessing:
- Data loading and validation
- Missing value imputation strategies
- Outlier detection and handling (IQR, Z-score, Isolation Forest)
- Feature scaling (StandardScaler, MinMaxScaler, RobustScaler)
- Categorical encoding (one-hot, label, target encoding)
- Feature engineering pipelines
- sklearn Pipeline and ColumnTransformer
- Custom transformers
- Data validation with Great Expectations
- Production deployment patterns

**Key Tools:** scikit-learn, pandas, Great Expectations, category_encoders

**When to Use:** Every ML project requires robust preprocessing

---

### 43. Data Augmentation
**File:** `43_Data_Augmentation.md`

State-of-the-art augmentation techniques for computer vision and NLP:
- **Computer Vision:** Mixup, CutMix, Cutout, Random Erasing, AutoAugment
- **NLP:** Synonym replacement, back-translation, random operations
- **GAN-based synthetic data generation**
- **2025 advanced techniques:** AugMax, RandAugment
- Production implementation with albumentations and nlpaug

**Key Tools:** albumentations, imgaug, nlpaug, torch

**When to Use:** Limited training data, overfitting, or class imbalance

---

### 44. Handling Imbalanced Data
**File:** `44_Handling_Imbalanced_Data.md`

Research-backed techniques for imbalanced classification (2025 SOTA):

**Key Techniques:**
- **SMOTE:** Synthetic Minority Oversampling (k-NN based)
- **ADASYN:** Adaptive Synthetic Sampling (99.67% accuracy benchmark)
- **SMOTE-TOMEK:** Hybrid over/undersampling
- **SMOTE-NC:** Handles categorical features
- **BorderlineSMOTE:** Focuses on decision boundary
- **Class weighting:** Simple and effective baseline
- **Focal Loss:** Best for deep learning on imbalanced data
- **XGBoost + SMOTE:** Highest F1 scores in benchmarks

**Research Findings:**
- Oversampling > undersampling (preserves information)
- Use precision, recall, F1, AUC-ROC (NOT accuracy)
- ADASYN adapts to local density for better synthetic samples
- Focal Loss reduces weight of well-classified examples

**Key Tools:** imbalanced-learn, scikit-learn, XGBoost, TensorFlow/PyTorch

**When to Use:** Fraud detection, medical diagnosis, rare event prediction, any classification with class imbalance ratio > 10:1

---

### 45. Big Data Technologies
**File:** `45_Big_Data_Technologies.md`

Production big data frameworks for ML at scale (2025 benchmarks):

**Technologies Covered:**

1. **Apache Hadoop**
   - HDFS distributed storage
   - MapReduce batch processing
   - Fault tolerance and reliability
   - When to use: Massive batch processing, archival storage

2. **Apache Spark**
   - In-memory processing (10-100x faster than MapReduce)
   - Spark SQL, DataFrames, RDDs
   - MLlib for distributed ML
   - PySpark for Python integration
   - When to use: Large-scale analytics, iterative ML algorithms

3. **Dask (2025 RECOMMENDED)**
   - **50% faster than Spark** on standard benchmarks
   - Python-native and lightweight
   - Seamless pandas/numpy/scikit-learn integration
   - Scales XGBoost, LightGBM training
   - When to use: Python ML workflows, medium-to-large data

4. **Distributed ML Frameworks**
   - Ray: General-purpose distributed computing
   - Horovod: Distributed deep learning (multi-GPU/multi-node)
   - Kubernetes deployment patterns

5. **Cloud Platforms (2025)**
   - AWS: EMR, Glue, SageMaker
   - Azure: HDInsight, Databricks, Synapse
   - GCP: Dataproc, BigQuery, Vertex AI

**Key Benchmarks:**
- Dask: 50% faster than Spark (2025)
- Spark: 10-100x faster than Hadoop MapReduce
- Horovod: Near-linear scaling to 1000s of GPUs

**When to Use Each:**
- Data < 10GB: pandas, single machine
- Data 10GB-1TB: Dask (Python ecosystem) or Spark (mature ecosystem)
- Data > 1TB: Spark or cloud-native solutions (BigQuery, Snowflake)
- Deep Learning at Scale: Horovod, Ray

---

## 2025 State-of-the-Art Highlights

### Data Preprocessing
- **Great Expectations** for automated data quality checks
- **Feature stores** (Feast, Tecton) for consistent feature engineering
- **Data validation in CI/CD pipelines**
- **Automated outlier detection** with Isolation Forest and Local Outlier Factor

### Imbalanced Data
- **ADASYN** achieves 99.67% accuracy in benchmarks (2025)
- **Focal Loss** now standard for imbalanced deep learning
- **XGBoost + SMOTE** yields highest F1 scores
- **Ensemble methods** reduce pseudo-labeling noise

### Big Data
- **Dask surpasses Spark** by 50% on Python ML workloads (2025)
- **Cloud-native** solutions dominate (BigQuery, Snowflake, Databricks)
- **Kubernetes** standard for deploying all big data frameworks
- **Edge computing** emerging for real-time ML inference

---

## Technology Stack Reference

### Data Preprocessing
```python
# Core libraries
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
import great_expectations as ge
from category_encoders import TargetEncoder
```

### Imbalanced Data
```python
# Imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks, NearMiss
from collections import Counter

# Focal Loss (PyTorch)
import torch
import torch.nn as nn
```

### Big Data
```python
# Dask
import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.preprocessing import StandardScaler

# PySpark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier

# Ray
import ray
from ray import tune
from ray.train import ScalingConfig
```

---

## Common Workflows

### 1. Standard Preprocessing Pipeline
```
Raw Data --> Validation --> Missing Value Handling --> Outlier Detection -->
Feature Scaling --> Encoding --> Feature Engineering --> Train/Val/Test Split --> Model Training
```

### 2. Imbalanced Classification Workflow
```
Load Data --> EDA (check class distribution) --> Choose resampling strategy (SMOTE/ADASYN) -->
Apply resampling --> Train model with class weights --> Evaluate with precision/recall/F1 -->
Tune threshold --> Production deployment
```

### 3. Big Data ML Pipeline
```
Data in HDFS/S3 --> Load with Dask/Spark --> Distributed preprocessing -->
Feature engineering at scale --> Train on subset or full data -->
Distributed hyperparameter tuning --> Model deployment to Kubernetes
```

---

## Performance Benchmarks (2025)

### Preprocessing
- **Pipeline automation:** 60% reduction in preprocessing errors
- **Great Expectations:** Catches 95%+ of data quality issues before training
- **Feature stores:** 40% faster model development with reusable features

### Imbalanced Data
| Technique | F1 Score | Recall | Precision |
|-----------|----------|--------|-----------|
| Baseline (no resampling) | 0.45 | 0.32 | 0.78 |
| SMOTE | 0.72 | 0.68 | 0.76 |
| ADASYN | 0.78 | 0.75 | 0.81 |
| XGBoost + SMOTE | 0.82 | 0.79 | 0.85 |
| Focal Loss (DL) | 0.85 | 0.83 | 0.87 |

*Source: RESEARCH_SUMMARY_2025.md benchmarks*

### Big Data Technologies
| Framework | Processing Speed | Best Use Case |
|-----------|------------------|---------------|
| Hadoop MapReduce | 1x (baseline) | Archival batch processing |
| Spark | 10-100x | Large-scale iterative ML |
| Dask | 150x (50% faster than Spark) | Python ML workflows |
| Ray | Variable | Distributed RL, hyperparameter tuning |

---

## Best Practices

### Data Preprocessing
1. **Always validate data before training** (Great Expectations, Pandera)
2. **Fit scalers/encoders only on training data** (prevent data leakage)
3. **Use ColumnTransformer** for heterogeneous feature types
4. **Save preprocessing pipelines with models** (joblib, pickle)
5. **Monitor data drift in production** (Evidently AI, WhyLabs)

### Imbalanced Data
1. **Never use accuracy as the primary metric**
2. **Start with class weighting** (simplest approach)
3. **Try SMOTE/ADASYN** if class weighting insufficient
4. **Use stratified splits** to preserve class distribution
5. **Evaluate on precision-recall curve** and AUC-PR
6. **Consider Focal Loss for deep learning**

### Big Data
1. **Start small:** Prototype on subset with pandas
2. **Choose framework based on ecosystem:**
   - Python-first: Use Dask
   - Mature ecosystem with lots of integrations: Use Spark
   - Deep learning at scale: Use Horovod or Ray
3. **Optimize data formats:** Parquet > CSV for big data
4. **Partition data intelligently** (by date, category)
5. **Monitor cluster resource usage** (CPU, memory, network)
6. **Use cloud-native solutions** for ease of scaling

---

## Decision Trees

### When to Resample for Imbalanced Data?
```
Class imbalance ratio < 3:1 --> No resampling needed, use class weights
Class imbalance ratio 3:1 to 10:1 --> Try class weights first, then SMOTE
Class imbalance ratio > 10:1 --> Use ADASYN or SMOTE-TOMEK
Deep learning model --> Use Focal Loss
Extremely imbalanced (>100:1) --> Anomaly detection approach (Isolation Forest)
```

### When to Use Big Data Technologies?
```
Data size < 10GB --> pandas (single machine)
Data size 10-100GB --> Dask (if Python-centric) or Spark
Data size 100GB-1TB --> Spark or Dask with distributed cluster
Data size > 1TB --> Cloud-native (BigQuery, Snowflake) or Spark
Real-time streaming --> Spark Structured Streaming or Flink
Deep learning distributed training --> Horovod (multi-GPU) or Ray
```

---

## Common Pitfalls to Avoid

### Data Preprocessing
 Fitting scalers on entire dataset (data leakage)
 Fit only on training set, transform val/test

 Not handling missing values systematically
 Use imputation strategies appropriate for data type

 One-hot encoding high-cardinality features
 Use target encoding or embeddings

 Not validating data in production
 Continuous validation with Great Expectations

### Imbalanced Data
 Using accuracy as primary metric
 Use precision, recall, F1, AUC-ROC, AUC-PR

 Only oversampling minority class
 Consider hybrid (SMOTE-TOMEK) or just class weights

 Applying SMOTE before cross-validation
 Apply SMOTE inside each CV fold

 Not stratifying train/test splits
 Always use stratified splits

### Big Data
 Using big data tools for small data
 Prototype locally, scale when necessary

 Loading entire dataset into memory
 Use lazy evaluation (Dask, Spark)

 Not partitioning data properly
 Partition by relevant keys (date, region)

 Using CSV for large datasets
 Use Parquet or ORC (10x smaller, 100x faster)

---

## Integration with Other Sections

- **01_Statistical_Foundations:** Hypothesis testing for detecting data drift
- **02_Classical_ML:** Feature engineering complements classical algorithms
- **03_Deep_Learning:** Data augmentation critical for deep learning
- **04_Computer_Vision:** Augmentation techniques (Mixup, CutMix)
- **05_NLP_Transformers:** Text augmentation and tokenization
- **08_MLOps_Production:** Deployment of preprocessing pipelines
- **10_Model_Evaluation:** Evaluation metrics for imbalanced data

---

## Further Reading

### Books
- "Designing Data-Intensive Applications" by Martin Kleppmann
- "Data Pipelines Pocket Reference" by James Densmore
- "Learning Spark" by Jules S. Damji et al.

### Online Resources
- imbalanced-learn documentation: https://imbalanced-learn.org/
- Dask documentation: https://docs.dask.org/
- Apache Spark documentation: https://spark.apache.org/docs/latest/
- Great Expectations: https://greatexpectations.io/

### Papers
- SMOTE: "SMOTE: Synthetic Minority Over-sampling Technique" (Chawla et al., 2002)
- ADASYN: "ADASYN: Adaptive Synthetic Sampling Approach" (He et al., 2008)
- Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)

---

## Quick Start Examples

### Preprocessing Pipeline
```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Define transformers for numeric and categorical features
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
```

### SMOTE for Imbalanced Data
```python
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_resampled, y_resampled)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Dask for Big Data
```python
import dask.dataframe as dd
from dask_ml.model_selection import train_test_split
from dask_ml.linear_model import LogisticRegression

# Load large dataset
df = dd.read_parquet('s3://bucket/large_data/*.parquet')

# Preprocessing
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = LogisticRegression()
clf.fit(X_train, y_train)

# Evaluate
score = clf.score(X_test, y_test)
print(f"Accuracy: {score:.4f}")
```

---

## Summary

This Data Engineering section equips you with production-ready techniques for:

1. **Preprocessing:** Robust pipelines with validation and scaling
2. **Augmentation:** State-of-the-art techniques for CV and NLP
3. **Imbalanced Data:** Research-backed resampling (ADASYN, SMOTE) and evaluation
4. **Big Data:** Modern frameworks (Dask, Spark) with 2025 benchmarks

**Key Takeaways:**
- Data quality determines model quality
- Imbalanced data requires specialized techniques (ADASYN, Focal Loss)
- Dask is 50% faster than Spark for Python ML workloads (2025)
- Preprocessing pipelines should be versioned and deployed with models
- Use precision/recall/F1 for imbalanced classification (not accuracy)

Master these techniques to build robust, scalable ML systems that perform well in production.
