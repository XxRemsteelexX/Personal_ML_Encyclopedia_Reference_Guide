# 12. Master Model Selection Decision Tree

## Overview

This is the **ultimate reference guide** integrating all previous sections. Use this to select the right model, preprocessing, encoding, and evaluation strategy for any ML/AI problem.

**What This Guide Covers:**
- Complete end-to-end ML workflow
- Model selection decision trees
- Data type --> model mapping
- Performance vs complexity tradeoffs
- Quick reference tables
- Common scenarios and solutions

---

## 12.1 The Complete ML Pipeline

```
1. UNDERSTAND THE PROBLEM
   +--- What are you predicting?
   +--- Classification, regression, generation?
   +--- What's the success metric?

2. EXPLORE YOUR DATA
   +--- How many samples? (100? 1K? 1M?)
   +--- How many features? (10? 100? 10K?)
   +--- What types? (numerical, categorical, text, images?)
   +--- Missing values? Outliers? Imbalance?
   +--- Run statistical tests (see Phase 4)

3. FEATURE ENGINEERING
   +--- Handle missing values
   +--- Encode categoricals (see Phase 3)
   |  +--- Tree models --> Label/Target encoding
   |  +--- Linear models --> One-hot encoding
   +--- Scale features (if using linear/neural nets)
   +--- Create derived features

4. MODEL SELECTION (THIS GUIDE)
   +--- Choose based on:
   |  +--- Data size
   |  +--- Data type
   |  +--- Problem type
   |  +--- Constraints (time, compute, interpretability)

5. TRAINING
   +--- Split data (train/val/test)
   +--- Handle class imbalance
   +--- Hyperparameter tuning
   +--- Cross-validation

6. EVALUATION
   +--- Right metrics for problem
   +--- Test on holdout set
   +--- Check for overfitting
   +--- Statistical significance tests

7. DEPLOYMENT
   +--- Model serving
   +--- Monitoring
   +--- A/B testing (see Phase 4)
```

---

## 12.2 Master Decision Tree: Problem Type First

### START HERE

```
What type of problem do you have?

+----- SUPERVISED LEARNING
|    +--- Predicting a category (classification)
|    |  +--- See Section 12.3
|    |
|    +--- Predicting a number (regression)
|       +--- See Section 12.4
|
+----- UNSUPERVISED LEARNING
|    +--- Finding groups (clustering)
|    +--- Reducing dimensions (PCA, t-SNE)
|    +--- Anomaly detection
|       +--- See Section 12.5
|
+----- GENERATIVE
|    +--- Generate images
|    +--- Generate text
|    +--- Generate other data
|       +--- See Section 12.6
|
+----- NATURAL LANGUAGE PROCESSING
|    +--- Text classification
|    +--- Text generation
|    +--- NER, Q&A, etc.
|    +--- See Section 12.7
|
+----- TIME SERIES
     +--- Forecasting
     +--- Anomaly detection
     +--- See Section 12.8
```

---

## 12.3 Classification: Complete Decision Flow

### Step 1: Data Type?

```
What type of features do you have?

+--- TABULAR/STRUCTURED DATA (most common)
|  +--- Go to Step 2: Sample Size
|
+--- IMAGES
|  +--- <10K samples --> Use pretrained CNN (transfer learning)
|  +--- 10K-100K --> Fine-tune CNN (ResNet, EfficientNet)
|  +--- 100K+ --> Train CNN from scratch (rare) or fine-tune
|
+--- TEXT
|  +--- <1K samples --> TF-IDF + Logistic Regression OR few-shot LLM
|  +--- 1K-100K --> Fine-tune BERT/RoBERTa
|  +--- 100K+ --> Fine-tune large transformer
|
+--- AUDIO
|  +--- Extract features --> Spectrograms --> CNN
|
+--- MIXED (tabular + text/images)
   +--- Multi-modal model or separate models + ensemble
```

### Step 2: Tabular Classification - Sample Size

```
How many training samples?

+--- < 1,000 samples (SMALL)
|  +--- Try: Logistic Regression (baseline)
|  +--- Then: Random Forest (usually best for small data)
|  +--- Avoid: Deep learning (will overfit)
|
+--- 1,000 - 10,000 samples (MEDIUM)
|  +--- Start: Random Forest (good defaults)
|  +--- Best: XGBoost (with tuning)
|  +--- Alternative: Logistic Regression with L1/L2
|
+--- 10,000 - 100,000 samples (LARGE)
|  +--- Best: XGBoost or LightGBM (winner 90% of time)
|  +--- Alternative: Random Forest
|  +--- Try: Neural network (if features complex)
|
+--- 100,000+ samples (VERY LARGE)
   +--- Best: LightGBM (fastest)
   +--- Alternative: XGBoost
   +--- Deep learning: Only if non-linear relationships
```

### Step 3: Feature Count

```
How many features?

+--- < 10 features (LOW)
|  +--- Start with Logistic Regression
|  +--- Visualize relationships
|  +--- Try: Random Forest
|
+--- 10 - 100 features (MEDIUM)
|  +--- XGBoost / Random Forest (sweet spot)
|  +--- Check feature importance
|
+--- 100 - 1,000 features (HIGH)
|  +--- XGBoost with feature selection
|  +--- OR: L1 regularized logistic regression
|  +--- Reduce dimensions if needed
|
+--- 1,000+ features (VERY HIGH)
   +--- L1 Logistic Regression (feature selection)
   +--- Deep learning (if enough samples)
   +--- Consider: PCA or feature selection first
```

### Step 4: Class Imbalance?

```
Is your dataset imbalanced?

+--- Balanced (40-60% split)
|  +--- No special handling needed
|
+--- Moderate (20-40% or 60-80%)
|  +--- Random Forest: class_weight='balanced'
|  +--- XGBoost: scale_pos_weight
|
+--- Severe (<20% or >80%)
   +--- 1. Resample (SMOTE, RandomOverSampler)
   +--- 2. Use class weights
   +--- 3. Use F1/precision/recall (NOT accuracy)
   +--- 4. Try: XGBoost (handles imbalance well)
```

### Step 5: Other Considerations

```
Special requirements?

+--- Need interpretability?
|  +--- Best: Logistic Regression or Decision Tree
|  +--- OK: Random Forest (feature importance)
|  +--- Avoid: XGBoost, Neural Nets (black box)
|
+--- Need fast inference (<1ms)?
|  +--- Best: Logistic Regression
|  +--- OK: Small Decision Tree
|  +--- Avoid: Large ensemble (100+ trees)
|
+--- Need fast training?
|  +--- Best: Logistic Regression
|  +--- OK: Random Forest
|  +--- Avoid: XGBoost (needs tuning)
|
+--- Limited compute (no GPU)?
   +--- Best: LightGBM or Random Forest
   +--- Avoid: Deep learning
```

---

## 12.4 Regression: Complete Decision Flow

### Step 1: Linear vs Non-linear?

```
Is relationship linear?

+--- Check with scatter plots
+--- Calculate correlation
+--- If mostly linear --> Use linear model
   If non-linear --> Use tree-based or neural net
```

### Step 2: Regression Model Selection

```
Based on data characteristics:

+--- LINEAR RELATIONSHIP
|  +--- < 100 features:
|  |  +--- Linear Regression (OLS)
|  |
|  +--- 100-1000 features or some noise:
|  |  +--- Ridge Regression (L2) - default choice
|  |  +--- Lasso (L1) - if need feature selection
|  |
|  +--- 1000+ features:
|     +--- ElasticNet (L1 + L2)
|
+--- NON-LINEAR RELATIONSHIP
|  +--- < 10K samples:
|  |  +--- Random Forest Regressor
|  |
|  +--- 10K - 100K samples:
|  |  +--- XGBoost / LightGBM (best choice)
|  |
|  +--- 100K+ samples + complex:
|     +--- LightGBM (fastest)
|     +--- Neural Network (if very complex)
|
+--- TIME SERIES
|  +--- See Section 12.8
|
+--- NEED EXTRAPOLATION (predict beyond training range)
   +--- Linear models (can extrapolate)
   +--- Avoid: Trees (cannot extrapolate)
```

### Step 3: Regression Metrics

```
Choose evaluation metric:

+--- MAE (Mean Absolute Error)
|  +--- Use when: Outliers should not dominate
|
+--- MSE (Mean Squared Error)
|  +--- Use when: Large errors are very bad
|
+--- RMSE (Root Mean Squared Error)
|  +--- Use when: Same units as target
|
+--- R^2 (R-squared)
|  +--- Use when: Need % variance explained
|
+--- MAPE (Mean Absolute Percentage Error)
   +--- Use when: Relative errors matter
```

---

## 12.5 Unsupervised Learning

### Clustering

```
How many samples?

+--- < 10K samples
|  +--- K-Means (default, fast)
|  +--- DBSCAN (if arbitrary shapes)
|  +--- Hierarchical (if need dendrogram)
|
+--- 10K+ samples
   +--- K-Means or Mini-Batch K-Means
   +--- HDBSCAN (better than DBSCAN)
```

### Dimensionality Reduction

```
What's your goal?

+--- Visualization (2D/3D)
|  +--- t-SNE (best for visualization)
|  +--- UMAP (faster, preserves global structure)
|
+--- Feature reduction for model
|  +--- PCA (linear)
|  +--- Feature selection (see Phase 3)
|
+--- Manifold learning
   +--- UMAP or Autoencoder
```

### Anomaly Detection

```
What type of anomalies?

+--- Point anomalies (single outliers)
|  +--- Isolation Forest (default choice)
|  +--- One-Class SVM
|  +--- LOF (Local Outlier Factor)
|
+--- Contextual anomalies
|  +--- Autoencoder or VAE
|
+--- Time series anomalies
   +--- See Section 12.8
```

---

## 12.6 Generative Models

### Image Generation

```
What do you want to generate?

+--- Text-to-image (2025 BEST)
|  +--- Stable Diffusion (open-source)
|  +--- DALL-E 3 (highest quality, API)
|  +--- Midjourney (artistic)
|
+--- Image-to-image
|  +--- Need quality: Stable Diffusion
|  +--- Need speed: CycleGAN / Pix2Pix
|
+--- Face generation
|  +--- StyleGAN3
|
+--- Super-resolution (upscaling)
|  +--- ESRGAN or Diffusion upscaler
|
+--- Data augmentation
   +--- Simple transforms (rotation, flip) - START HERE
   +--- GAN (only if need realism)
```

### Text Generation

```
What type of text?

+--- Creative writing, chat
|  +--- GPT-4 / GPT-3.5 (API)
|  +--- Claude (API)
|  +--- Llama / Mistral (open-source)
|
+--- Code generation
|  +--- CodeLlama or GPT-4
|
+--- Domain-specific
   +--- Fine-tune LLM on domain data
```

### Feature Learning / Anomaly Detection

```
What's your goal?

+--- Learn features for downstream task
|  +--- VAE (interpretable latent space)
|
+--- Anomaly detection
|  +--- Autoencoder (simple)
|  +--- VAE (probabilistic)
|
+--- Data compression
   +--- Autoencoder
```

---

## 12.7 Natural Language Processing (NLP)

### Quick NLP Decision Tree

```
What's your NLP task?

+--- CLASSIFICATION (sentiment, topic, etc.)
|  +--- < 1K samples:
|  |  +--- TF-IDF + Logistic Regression (baseline)
|  |  +--- Few-shot GPT-4 (if budget allows)
|  |
|  +--- 1K - 10K samples:
|  |  +--- Fine-tune DistilBERT or RoBERTa
|  |
|  +--- 10K+ samples:
|     +--- Fine-tune RoBERTa-large or DeBERTa
|
+--- GENERATION (stories, articles, etc.)
|  +--- Best quality: GPT-4
|  +--- Cost-effective: GPT-3.5
|  +--- Self-hosted: Llama 3, Mistral
|
+--- NAMED ENTITY RECOGNITION (NER)
|  +--- Fine-tune BERT or RoBERTa
|
+--- QUESTION ANSWERING
|  +--- Extractive (answer in passage):
|  |  +--- Fine-tune BERT
|  |
|  +--- Generative (create answer):
|     +--- T5, Flan-T5, or GPT
|
+--- SUMMARIZATION
|  +--- BART or T5 (open-source)
|  +--- GPT-4 (best quality)
|
+--- TRANSLATION
|  +--- MarianMT (fast, specific pairs)
|  +--- T5 or GPT-4 (multi-language)
|
+--- SEMANTIC SIMILARITY
   +--- Sentence-BERT (SBERT)
```

---

## 12.8 Time Series

### Forecasting

```
What type of time series?

+--- UNIVARIATE (single variable)
|  +--- Short-term (<100 points):
|  |  +--- ARIMA or Exponential Smoothing
|  |
|  +--- Medium-term (100-10K points):
|  |  +--- Prophet (Facebook - easy to use)
|  |  +--- SARIMA (seasonal)
|  |
|  +--- Long-term (10K+ points):
|     +--- LSTM (deep learning)
|     +--- XGBoost (with lag features)
|
+--- MULTIVARIATE (many variables)
|  +--- VAR (Vector AutoRegression)
|  +--- LSTM (deep learning)
|  +--- XGBoost (with feature engineering)
|
+--- HIGH FREQUENCY (tick data)
   +--- Statistical models + feature engineering
```

### Time Series Anomaly Detection

```
What kind of anomalies?

+--- Point anomalies (spikes)
|  +--- Statistical methods (Z-score, IQR)
|
+--- Pattern anomalies (unusual sequences)
|  +--- LSTM Autoencoder
|  +--- Isolation Forest (with lag features)
|
+--- Change point detection
   +--- Bayesian methods or Prophet
```

---

## 12.9 Categorical Encoding Decision Tree

**CRITICAL:** Encoding choice dramatically affects model performance!

```
What model are you using?

+--- TREE-BASED (Random Forest, XGBoost, LightGBM)
|  |
|  +--- Cardinality < 50?
|  |  +--- Use: Label Encoding (default)
|  |
|  +--- Cardinality >= 50 (high cardinality)?
|     +--- Use: Target Encoding (with CV to prevent leakage)
|     +--- Alternative: Frequency Encoding
|  |
|  +---  AVOID: One-Hot Encoding (hurts performance!)
|
+--- LINEAR MODELS (Logistic, Linear, SVM)
|  |
|  +--- Cardinality < 10?
|  |  +--- Use: One-Hot Encoding
|  |
|  +--- Cardinality 10-50?
|  |  +--- Use: One-Hot with dimensionality reduction
|  |  +--- OR: Target Encoding + Linear model
|  |
|  +--- Cardinality > 50?
|     +--- Use: Target Encoding OR
|     +--- Feature selection on one-hot
|
+--- DEEP LEARNING (Neural Networks)
   |
   +--- Low cardinality (<100)?
   |  +--- Use: Embedding layer
   |
   +--- High cardinality (>=100)?
      +--- Use: Embedding layer with dropout
```

### Encoding Summary Table

| Encoding | Random Forest | XGBoost | Linear Models | Neural Nets |
|----------|--------------|---------|---------------|-------------|
| **Label** |  Excellent |  Excellent |  Bad (false ordering) | [WARNING] OK (use embedding better) |
| **One-Hot** |  Bad (slow, worse) |  Bad (slow, worse) |  Excellent | [WARNING] OK (high-dim) |
| **Target** |  Best |  Best |  Good |  Good |
| **Frequency** |  Good |  Good |  Good |  Good |
| **Embedding** | N/A | N/A | N/A |  Best |

---

## 12.10 Feature Scaling Decision Tree

```
What model are you using?

+--- TREE-BASED (Random Forest, XGBoost, Decision Tree)
|  +---  NO SCALING NEEDED
|     Trees split on thresholds, scale doesn't matter
|
+--- LINEAR MODELS (Linear/Logistic Regression, SVM)
|  +--- [WARNING] SCALING REQUIRED
|     +--- StandardScaler (default choice)
|     |  +--- Features --> mean=0, std=1
|     |
|     +--- MinMaxScaler (if need [0,1] range)
|     |  +--- For neural nets with sigmoid
|     |
|     +--- RobustScaler (if many outliers)
|        +--- Uses median, IQR (robust to outliers)
|
+--- NEURAL NETWORKS
|  +--- [WARNING] SCALING REQUIRED
|     +--- StandardScaler (most common)
|     +--- MinMaxScaler ([0,1] for images/sigmoid)
|
+--- K-MEANS, KNN, SVM
|  +--- [WARNING] SCALING CRITICAL
|     Distance-based algorithms very sensitive to scale
|
+--- NAIVE BAYES
   +---  NO SCALING NEEDED
```

---

## 12.11 Quick Reference: Model Comparison

### Performance vs Complexity

| Model | Accuracy | Training Time | Inference Time | Interpretability | Tuning Needed |
|-------|----------|--------------|----------------|------------------|---------------|
| **Logistic Regression** |  |  Fast |  Fast |  High |  Minimal |
| **Decision Tree** |  |  Fast |  Fast |  High |  Minimal |
| **Random Forest** |  |  Medium |  Medium |  Medium |  Minimal |
| **XGBoost** |  |  Medium |  Medium |  Low |  High |
| **LightGBM** |  |  Fast |  Fast |  Low |  High |
| **SVM** |  |  Slow |  Medium |  Low |  Medium |
| **Neural Network** |  |  Slow |  Fast |  Very Low |  High |
| **BERT (NLP)** |  |  Slow |  Medium |  Very Low |  Medium |
| **Diffusion (images)** |  |  Slow |  Very Slow |  Very Low |  Medium |

---

## 12.12 Common Scenarios and Solutions

### Scenario 1: Small Dataset (<1K samples)

**Problem:** Not enough data for complex models

**Solution:**
```
1. Start with simple models (Logistic Regression, Random Forest)
2. Use cross-validation (NOT train/test split)
3. Avoid deep learning (will overfit)
4. Consider:
   - Data augmentation (if images)
   - Few-shot learning (if text with LLM)
   - Transfer learning (if images)
5. Feature engineering is CRITICAL
```

### Scenario 2: Imbalanced Classes (95% vs 5%)

**Problem:** Model predicts majority class always

**Solution:**
```
1. Use stratified split (preserve class ratios)
2. Resample:
   - SMOTE (synthetic minority oversampling)
   - RandomOverSampler
   - RandomUnderSampler (if lots of data)
3. Use class weights:
   - Random Forest: class_weight='balanced'
   - XGBoost: scale_pos_weight
4. Use right metrics:
   - F1-score, precision, recall (NOT accuracy)
   - ROC-AUC, PR-AUC
5. Try XGBoost (handles imbalance well)
```

### Scenario 3: High Cardinality Categoricals (1000+ categories)

**Problem:** One-hot creates 1000+ features, crashes model

**Solution:**
```
1. For Tree Models:
   - Use Target Encoding (with CV)
   - Or Frequency Encoding

2. For Linear Models:
   - Target Encoding
   - Or: Feature selection on one-hot
   - Or: Group rare categories into "Other"

3. For Neural Nets:
   - Use Embedding layer (learns representations)

4. Consider: Is feature worth it?
   - Remove if low importance
```

### Scenario 4: Need Interpretability

**Problem:** Stakeholders need to understand model

**Solution:**
```
1. Use interpretable models:
   - Logistic Regression (best)
   - Decision Tree (rules)
   - Linear Regression

2. If need performance + interpretability:
   - Random Forest + SHAP values
   - XGBoost + SHAP values

3. Document:
   - Feature importance
   - Decision rules
   - Example predictions

4. Avoid:
   - Deep neural networks
   - Large ensembles (100+ trees)
```

### Scenario 5: Real-Time Predictions (<10ms latency)

**Problem:** Need instant predictions

**Solution:**
```
1. Use fast models:
   - Logistic Regression (fastest)
   - Small Decision Tree
   - Small Random Forest (<10 trees)

2. Optimize inference:
   - Reduce features (feature selection)
   - Quantize model (if neural net)
   - Use ONNX runtime

3. Architecture:
   - Cache predictions if possible
   - Use efficient serving (FastAPI, TorchServe)

4. Avoid:
   - Large ensembles (100+ trees)
   - Deep neural networks (unless optimized)
   - Transformer models
```

### Scenario 6: Many Missing Values

**Problem:** 20-50% of data missing

**Solution:**
```
1. Understand missingness:
   - MCAR (Missing Completely At Random) --> OK to drop/impute
   - MAR (Missing At Random) --> Impute carefully
   - MNAR (Missing Not At Random) --> Model missingness

2. Imputation strategies:
   - Mean/Median (numerical)
   - Mode (categorical)
   - KNN imputation (more sophisticated)
   - XGBoost (handles missing natively!)

3. Add indicator features:
   - Create 'is_missing' binary feature
   - Model can learn patterns

4. Use models that handle missing:
   - XGBoost (learns optimal handling)
   - LightGBM
```

### Scenario 7: Need to Predict Beyond Training Range (Extrapolation)

**Problem:** Training data: 0-100, need to predict 150

**Solution:**
```
1. Use models that extrapolate:
   - Linear Regression 
   - Polynomial Regression 
   - ARIMA (time series) 

2. AVOID models that DON'T extrapolate:
   - Random Forest 
   - XGBoost 
   - K-NN 
   - Neural Nets 

3. Alternative:
   - Domain knowledge (physics-based models)
   - Collect more data in range
```

### Scenario 8: Non-linear Relationships but Need Interpretability

**Problem:** Data is complex but stakeholders need understanding

**Solution:**
```
1. Use Random Forest + SHAP:
   - Good performance
   - SHAP explains predictions
   - Feature importance clear

2. Or: Decision Tree (max_depth=5-10)
   - Not as accurate as RF
   - But very interpretable

3. Or: Polynomial Features + Linear Model
   - Feature engineering captures non-linearity
   - Model itself is linear (interpretable)

4. Document everything:
   - Visualize decision boundaries
   - Show example predictions
   - Explain feature importance
```

---

## 12.13 Model Selection Flowchart (Printable)

```
+-----------------------------------------------------------------+
|                    START: CHOOSE YOUR MODEL                  |
+--------------------------------+--------------------------------+
                               |
                     What's your data type?
                               |
        +------------------------+------------------------+
        |                      |                      |
    TABULAR                  TEXT                 IMAGES
        |                      |                      |
        |                      |                      |
   How many samples?    How many samples?       How many samples?
        |                      |                      |
   +------+------+           +------+------+            +------+------+
   |         |           |         |            |         |
 <1K     1K-10K       <1K      1K-10K         <10K     10K+
   |         |           |         |            |         |
LogReg   XGBoost     TF-IDF    BERT      Pretrained  Fine-tune
  OR        OR         OR       OR        CNN        CNN
  RF       LightGBM  GPT-4  RoBERTa    (Transfer)  (ResNet)


TABULAR DETAIL:

<1K samples:
  --> Logistic Regression (baseline)
  --> Random Forest (usually best)

1K-10K samples:
  --> Random Forest (good defaults)
  --> XGBoost (best with tuning)

10K-100K samples:
  --> XGBoost or LightGBM (winner 90%)
  --> Neural Net (if complex)

100K+ samples:
  --> LightGBM (fastest)
  --> XGBoost
  --> Deep Learning (if justified)


SPECIAL CONSIDERATIONS:

Need interpretability?
  --> Logistic Regression or Decision Tree

Need fast inference?
  --> Logistic Regression or small model

High-dimensional (1000+ features)?
  --> L1 Logistic Regression or XGBoost with feature selection

Imbalanced classes?
  --> XGBoost with scale_pos_weight
  --> Random Forest with class_weight='balanced'

Many categorical features?
  --> XGBoost with Target Encoding (NOT one-hot)
```

---

## 12.14 Evaluation Metrics Cheat Sheet

### Classification Metrics

| Metric | When to Use | Formula/Notes |
|--------|-------------|---------------|
| **Accuracy** | Balanced classes only | (TP + TN) / Total |
| **Precision** | False positives are costly | TP / (TP + FP) |
| **Recall** | False negatives are costly | TP / (TP + FN) |
| **F1-Score** | Imbalanced classes, balance P & R | 2 x (P x R) / (P + R) |
| **ROC-AUC** | Binary classification, threshold-agnostic | Area under ROC curve |
| **PR-AUC** | Imbalanced classes (better than ROC-AUC) | Area under precision-recall curve |
| **Log Loss** | Probabilistic predictions | Penalizes wrong confidence |

### Regression Metrics

| Metric | When to Use | Formula/Notes |
|--------|-------------|---------------|
| **MAE** | Outliers should not dominate | Mean Absolute Error |
| **MSE** | Large errors very bad | Mean Squared Error |
| **RMSE** | Same units as target | sqrtMSE |
| **R^2** | % variance explained | 1 - (SS_res / SS_tot) |
| **MAPE** | Relative errors matter | Mean Absolute % Error |

### Which Metric for Which Problem?

```
Classification:

+--- Balanced classes?
|  +--- Use: Accuracy

+--- Imbalanced classes?
|  +--- Use: F1-Score or PR-AUC

+--- False positives very bad? (e.g., spam filter)
|  +--- Use: Precision

+--- False negatives very bad? (e.g., fraud detection)
|  +--- Use: Recall

+--- Need probability calibration?
   +--- Use: Log Loss or Brier Score


Regression:

+--- Outliers present?
|  +--- Use: MAE (robust)

+--- Large errors critical?
|  +--- Use: MSE or RMSE

+--- Need interpretability?
|  +--- Use: R^2

+--- Relative errors matter?
   +--- Use: MAPE
```

---

## 12.15 Statistical Testing for Model Selection

### When to Use Statistical Tests

```
Scenario 1: Comparing Two Models

+--- Use: Paired t-test or McNemar's test
|
+--- Steps:
|  1. Get predictions from both models on same test set
|  2. Calculate metric for each fold (cross-validation)
|  3. Run paired t-test
|  4. If p < 0.05 --> significant difference
|
+--- Example:
   Model A: [0.85, 0.87, 0.86, 0.84, 0.88] (5-fold CV)
   Model B: [0.82, 0.84, 0.83, 0.81, 0.85]
   --> Run paired t-test --> Is difference significant?


Scenario 2: A/B Testing Model in Production

+--- Use: Two-sample test (see Phase 4)
|
+--- Steps:
|  1. Deploy Model A to 50% users, Model B to 50%
|  2. Collect metrics (conversion rate, accuracy, etc.)
|  3. Run two-sample t-test
|  4. Calculate required sample size (power analysis)
|
+--- See: Phase 4 - Statistical Tests for details


Scenario 3: Feature Selection

+--- Use: Chi-square test (categorical) or correlation (numerical)
|
+--- Steps:
   1. Calculate feature-target relationship
   2. Test significance
   3. Remove features with p > 0.05 (not significant)
```

---

## 12.16 Hyperparameter Tuning Strategy

### Tuning Priority by Model

**Random Forest:**
```
1. n_estimators (100-500)
2. max_depth (None or 10-50)
3. min_samples_leaf (1-10)
4. max_features ('sqrt' for classification)
```

**XGBoost:**
```
1. learning_rate + n_estimators (tune together)
2. max_depth (3-10)
3. subsample + colsample_bytree (0.5-1.0)
4. min_child_weight (1-10)
```

**Neural Network:**
```
1. Learning rate (most important)
2. Batch size (32, 64, 128)
3. Architecture (layers, neurons)
4. Dropout rate (0.2-0.5)
```

**BERT (NLP):**
```
1. Learning rate (1e-5 to 5e-5)
2. Batch size (8, 16, 32)
3. Epochs (2-5)
4. Warmup steps (500-1000)
```

### Tuning Methods

| Method | When to Use | Pros | Cons |
|--------|-------------|------|------|
| **Grid Search** | <5 hyperparameters, fast training | Exhaustive | Slow, combinatorial explosion |
| **Random Search** | Many hyperparameters | Faster than grid | Random (not optimal) |
| **Optuna** | Complex tuning, time available | Smart search, best results | Setup overhead |
| **Manual** | Know domain, iterative | Flexible | Time-consuming |

**Recommendation:** Use Optuna for XGBoost, manual tuning for Neural Nets/BERT.

---

## 12.17 Final Checklist: Before Deployment

### Data Pipeline
- [ ] Handle missing values consistently
- [ ] Encode categoricals correctly for model type
- [ ] Scale features if using linear/neural models
- [ ] Feature engineering documented
- [ ] Data leakage checked (no test data in training)

### Model
- [ ] Cross-validation used (not just train/test split)
- [ ] Appropriate metric chosen for problem
- [ ] Model tuned (hyperparameters optimized)
- [ ] Overfitting checked (train vs test performance)
- [ ] Feature importance analyzed

### Evaluation
- [ ] Tested on holdout set (never seen before)
- [ ] Statistical significance tested (if comparing models)
- [ ] Error analysis performed (where does model fail?)
- [ ] Calibration checked (probabilities accurate?)
- [ ] Fairness/bias checked (if applicable)

### Deployment
- [ ] Inference time acceptable (<target latency)
- [ ] Model size acceptable (memory constraints)
- [ ] Serving infrastructure ready
- [ ] Monitoring in place (data drift, performance)
- [ ] A/B testing plan (gradual rollout)
- [ ] Rollback plan (if model fails)

---

## 12.18 Quick Reference: Problem --> Solution

| Problem | Model | Encoding | Scaling | Metric |
|---------|-------|----------|---------|--------|
| **Tabular Classification (1K-100K rows)** | XGBoost | Label/Target | No | F1-Score |
| **Tabular Regression** | XGBoost | Label/Target | No | RMSE |
| **Small Dataset (<1K)** | Random Forest | Label | No | CV F1 |
| **Text Classification** | Fine-tune BERT | Tokenizer | No | F1-Score |
| **Image Classification** | Fine-tune CNN | None | Normalize | Accuracy |
| **Text Generation** | GPT / Llama | Tokenizer | No | Perplexity |
| **Image Generation** | Stable Diffusion | None | Normalize | FID |
| **Time Series Forecasting** | Prophet / XGBoost | N/A | Optional | MAE/RMSE |
| **Anomaly Detection** | Isolation Forest | Depends | Yes | Precision/Recall |
| **Clustering** | K-Means | One-hot | Yes | Silhouette |

---

## 12.19 Resources for Further Learning

### Books
- "Hands-On Machine Learning" (Aurelien Geron) - Practical ML
- "The Elements of Statistical Learning" - Theory
- "Interpretable Machine Learning" (Christoph Molnar) - Explainability

### Online Courses
- Fast.ai - Practical deep learning
- Coursera ML (Andrew Ng) - Foundations
- HuggingFace Course - NLP/Transformers

### Tools
- scikit-learn: Classical ML
- XGBoost/LightGBM: Gradient boosting
- PyTorch/TensorFlow: Deep learning
- HuggingFace: NLP/Transformers
- Optuna: Hyperparameter tuning
- SHAP: Model interpretability

### Competitions (Practice)
- Kaggle - ML competitions
- DrivenData - Social impact
- Google Colab - Free GPU

---

## 12.20 Summary: The Golden Rules

1. **Start Simple:** Baseline with logistic regression or simple model
2. **Understand Your Data:** Explore before modeling (EDA is critical)
3. **Right Encoding:** Tree models --> Label/Target, Linear --> One-hot
4. **Right Metric:** F1 for imbalanced, RMSE for regression, etc.
5. **Cross-Validate:** Always use CV, never just train/test
6. **XGBoost Usually Wins:** For tabular data (90% of cases)
7. **Transformers for NLP:** BERT/GPT for text (2025 standard)
8. **Diffusion for Images:** Stable Diffusion for generation
9. **Tune Carefully:** Hyperparameters matter (especially XGBoost)
10. **Test Statistically:** Ensure improvements are significant
11. **Monitor in Production:** Data drift kills models
12. **Document Everything:** Future you will thank present you

---

**Last Updated:** 2025-10-12

**This completes the ML Encyclopedia! You now have:**
- Phase 1: Tree-Based Models 
- Phase 2: Linear Models & SVMs 
- Phase 3: Feature Engineering & Encoding 
- Phase 4: Statistical Tests 
- Phase 5: Deep Learning Fundamentals 
- Phase 6: NLP & Transformers 
- Phase 7: GANs & Generative Models 
- Phase 8: Master Model Selection (this document) 

**Print this guide and keep it as a reference for all your ML projects!**
