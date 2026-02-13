# 8. Linear Models & Support Vector Machines (SVMs)

## Overview

Linear models are the foundation of machine learning. They're interpretable, fast, and work well when relationships are linear. This guide covers when to use them, when NOT to use them, and critical preprocessing requirements.

**Models Covered:**
- Linear Regression
- Logistic Regression
- Ridge Regression (L2 regularization)
- Lasso Regression (L1 regularization)
- ElasticNet (L1 + L2)
- Support Vector Machines (SVMs)

**Key Difference from Tree Models:**
-  Linear models NEED feature scaling
-  Linear models NEED one-hot encoding for categoricals
-  Linear models assume linear relationships

---

## 8.1 Linear Regression

### What It Is
Predicts continuous outcomes by fitting a line (or hyperplane) through data points. Minimizes squared error between predictions and actual values.

###  When to Use Linear Regression

1. **Linear relationship exists**
   - Features have linear relationship with target
   - Check with scatter plots or correlation matrix

2. **Need interpretability**
   - Coefficients show feature importance
   - Easy to explain to non-technical stakeholders
   - Example: "Each year of education increases salary by $5,000"

3. **Quick baseline**
   - Fast to train
   - Good starting point before trying complex models

4. **Small to medium datasets**
   - Works well with limited data (100s to 10,000s of samples)
   - Less prone to overfitting than complex models

5. **Need extrapolation**
   - Unlike trees, can predict beyond training data range
   - Example: Predict sales for year 2026 when trained on 2020-2025

6. **Feature selection needed**
   - Use Lasso to identify important features
   - Coefficients shrink irrelevant features to zero

###  When NOT to Use Linear Regression

1. **Non-linear relationships**
   - Curved, exponential, or complex patterns
   - **Better alternative:** Polynomial features + linear model, or tree-based models

2. **Outliers present**
   - Squared loss is sensitive to outliers
   - **Better alternative:** Huber loss, quantile regression, or tree models

3. **High-dimensional data (p > n)**
   - More features than samples
   - **Better alternative:** Ridge or Lasso regularization

4. **Multicollinearity**
   - Highly correlated features
   - **Better alternative:** Ridge regression, PCA, or drop correlated features

5. **Categorical target variable**
   - Predicting classes, not continuous values
   - **Better alternative:** Logistic regression, tree models

6. **Heteroscedasticity (non-constant variance)**
   - Variance of errors changes with X
   - **Solution:** Transform target (log, sqrt), or use weighted least squares

---

## 8.2 Logistic Regression

### What It Is
Classification algorithm that predicts probability of binary outcomes using sigmoid function. Outputs values between 0 and 1.

###  When to Use Logistic Regression

1. **Binary classification**
   - Two-class problems (yes/no, fraud/not fraud)
   - Can extend to multiclass with one-vs-rest or multinomial

2. **Need probability estimates**
   - Outputs calibrated probabilities (unlike trees)
   - Example: "Customer has 73% chance of churning"

3. **Need interpretability**
   - Coefficients show log-odds impact
   - Example: "Each additional login reduces churn odds by 15%"

4. **Linear decision boundary**
   - Classes are linearly separable
   - Check with 2D scatter plots

5. **High-dimensional sparse data**
   - Works well with text data (after TF-IDF)
   - Use L1 regularization for feature selection

6. **Baseline for classification**
   - Fast, reliable starting point
   - Benchmark for comparing complex models

7. **Need calibrated probabilities**
   - Better calibrated than Random Forest or SVM
   - Important for decision-making (e.g., medical diagnosis)

###  When NOT to Use Logistic Regression

1. **Non-linear decision boundary**
   - Classes not linearly separable
   - **Better alternative:** SVM with RBF kernel, tree models, neural nets

2. **Many irrelevant features**
   - Without regularization, overfits
   - **Solution:** Use L1 (Lasso) or L2 (Ridge) regularization

3. **Multicollinearity**
   - Unstable coefficients
   - **Better alternative:** Ridge logistic regression

4. **Imbalanced classes**
   - Biases toward majority class
   - **Solution:** Use `class_weight='balanced'` or SMOTE

5. **Need maximum accuracy**
   - Simpler model, may underperform
   - **Better alternative:** XGBoost, neural nets (but less interpretable)

---

## 8.3 Ridge Regression (L2 Regularization)

### What It Is
Linear regression with L2 penalty: minimizes `RSS + lambda * sum(beta^2)`. Shrinks coefficients but never exactly to zero.

###  When to Use Ridge

1. **Multicollinearity present**
   - Highly correlated features
   - Ridge stabilizes coefficient estimates

2. **Many features of similar importance**
   - Most features contribute to prediction
   - Ridge keeps all features, just shrinks them

3. **Prevent overfitting**
   - Model too complex for data size
   - Ridge reduces variance

4. **p > n (more features than samples)**
   - Ridge handles this better than OLS
   - Common in genomics, text analysis

5. **Want all features in model**
   - Need all features for interpretability
   - Don't want automatic feature selection

###  When NOT to Use Ridge

1. **Need feature selection**
   - Ridge doesn't zero out coefficients
   - **Better alternative:** Lasso

2. **Many irrelevant features**
   - Ridge keeps all features (small but non-zero)
   - **Better alternative:** Lasso or ElasticNet

3. **No regularization needed**
   - n >> p and no multicollinearity
   - **Better alternative:** Ordinary Linear Regression (faster)

---

## 8.4 Lasso Regression (L1 Regularization)

### What It Is
Linear regression with L1 penalty: minimizes `RSS + lambda * sum|beta|`. Shrinks some coefficients exactly to zero (feature selection).

###  When to Use Lasso

1. **Need automatic feature selection**
   - Many irrelevant features
   - Lasso zeros out unimportant features

2. **Few important features**
   - Sparse solution desired
   - Example: Only 10 out of 1000 genes matter

3. **Interpretability + simplicity**
   - Want minimal set of features
   - Easier to explain model with fewer features

4. **High-dimensional data**
   - p >> n (more features than samples)
   - Lasso selects relevant subset

5. **Debugging feature importance**
   - Identify which features actually matter
   - Use Lasso path to see feature selection order

###  When NOT to Use Lasso

1. **Grouped correlated features**
   - Lasso picks one, zeros out others randomly
   - **Example:** If price_usd and price_eur are correlated, Lasso picks one arbitrarily
   - **Better alternative:** ElasticNet or group Lasso

2. **n < p and highly correlated features**
   - Lasso selects at most n features (saturates)
   - **Better alternative:** ElasticNet

3. **All features are important**
   - Don't want to lose any features
   - **Better alternative:** Ridge

4. **Need stable feature selection**
   - Small data changes = different features selected
   - **Better alternative:** ElasticNet or stability selection

---

## 8.5 ElasticNet (L1 + L2)

### What It Is
Combines Ridge (L2) and Lasso (L1): minimizes `RSS + lambda_1 * sum|beta| + lambda_2 * sum(beta^2)`. Gets best of both worlds.

###  When to Use ElasticNet

1. **Grouped correlated features**
   - ElasticNet tends to select groups together
   - Example: Stock prices of companies in same sector

2. **p >> n (high-dimensional data)**
   - More features than samples
   - ElasticNet more stable than Lasso alone

3. **Want feature selection + stability**
   - Need to zero out features (like Lasso)
   - But want stable selection (like Ridge)

4. **Don't know which regularization to use**
   - ElasticNet is safest default
   - Combines benefits of both

5. **Medium cardinality with correlation**
   - 50-500 features with multicollinearity
   - ElasticNet handles both issues

###  When NOT to Use ElasticNet

1. **Two hyperparameters to tune**
   - `alpha` (L1 ratio) and `lambda` (penalty strength)
   - Takes longer to optimize
   - **Alternative:** Use Ridge or Lasso if clear which is better

2. **Interpretability critical**
   - Harder to explain two penalties
   - **Better alternative:** Lasso (simpler, still selects features)

3. **Very small dataset**
   - Risk of overfitting hyperparameters
   - **Better alternative:** Ridge with fixed lambda

---

## 8.6 Support Vector Machines (SVMs)

### What It Is
Finds optimal hyperplane to separate classes with maximum margin. Can handle non-linear boundaries via kernel trick.

###  When to Use SVM

1. **Clear margin of separation**
   - Classes are well-separated
   - SVM finds optimal boundary

2. **High-dimensional data**
   - Works well when p > n
   - Example: Text classification, gene expression

3. **Non-linear boundaries (with kernel)**
   - RBF kernel captures complex patterns
   - Polynomial kernel for polynomial relationships

4. **Outlier resistant (with soft margin)**
   - C parameter controls tolerance to misclassification
   - More robust than logistic regression

5. **Small to medium datasets**
   - Efficient on 100s to 10,000s of samples
   - Excellent performance with proper tuning

6. **Binary classification**
   - SVM naturally designed for two-class problems
   - Can extend to multiclass (one-vs-one, one-vs-rest)

###  When NOT to Use SVM

1. **Large datasets (100K+ samples)**
   - Training complexity: O(n^2) to O(n^3)
   - Very slow on large data
   - **Better alternative:** Logistic regression, XGBoost

2. **Need probability estimates**
   - SVM outputs distances, not probabilities
   - Platt scaling can convert, but less reliable
   - **Better alternative:** Logistic regression

3. **Many noisy features**
   - SVM can overfit in high noise
   - **Better alternative:** Tree-based models (handle noise better)

4. **Need interpretability**
   - Hard to explain kernel transformations
   - Support vectors don't provide clear feature importance
   - **Better alternative:** Linear models, decision trees

5. **Imbalanced classes**
   - Biases toward majority class
   - **Solution:** Adjust `class_weight`, but still challenging

6. **Limited computational resources**
   - RBF kernel training is expensive
   - Prediction also slower than linear models
   - **Better alternative:** Linear SVM, logistic regression

7. **Multiclass with many classes**
   - One-vs-one: O(k^2) classifiers for k classes
   - Becomes unwieldy with 10+ classes
   - **Better alternative:** Multiclass logistic, tree models

---

## 8.7 SVM Kernel Selection Guide

### Kernel Types

| Kernel | When to Use | Complexity | Hyperparameters |
|--------|-------------|------------|-----------------|
| **Linear** | Linearly separable, p >> n | Low (fast) | C only |
| **RBF (Gaussian)** | Non-linear, default choice | High | C, gamma |
| **Polynomial** | Polynomial relationships | Medium | C, degree, coef0 |
| **Sigmoid** | Neural network-like | Medium | C, gamma, coef0 |

### Selection Strategy

```
Start simple, work up:

1. Try LINEAR kernel first
   +--- If accuracy good --> DONE (use linear)
   +--- If p >> n --> DEFINITELY use linear
   +--- If poor --> Continue to step 2

2. Try RBF kernel
   +--- Default choice for non-linear
   +--- Works well in most cases
   +--- Tune C and gamma with GridSearch

3. Try Polynomial (rare)
   +--- Only if you know relationship is polynomial

4. Avoid Sigmoid
   +--- Rarely used, neural nets are better
```

### Hyperparameter Guidelines

**For Linear SVM:**
```python
from sklearn.svm import SVC

svm = SVC(
    kernel='linear',
    C=1.0,           # ^ = less regularization, more complex
    class_weight='balanced'  # For imbalanced data
)
```

**For RBF SVM:**
```python
svm = SVC(
    kernel='rbf',
    C=1.0,           # ^ = less regularization
    gamma='scale',   # v = wider influence (smoother)
                     # ^ = narrow influence (more complex)
    class_weight='balanced'
)
```

**Tuning Priority:**
1. **C (Regularization)**: Controls margin vs misclassification trade-off
   - Small C: Wide margin, more misclassification (underfit)
   - Large C: Narrow margin, fewer misclassifications (overfit)

2. **gamma (RBF only)**: Controls influence of single training point
   - Small gamma: Far-reaching influence (smoother, underfit)
   - Large gamma: Close-range influence (more complex, overfit)

**Pro tip:**
```python
# Use GridSearch with cross-validation
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid.fit(X_train, y_train)
print(f"Best: {grid.best_params_}")
```

---

## 8.8 CRITICAL: Preprocessing for Linear Models

### [WARNING] Linear Models != Tree Models

Linear models require different preprocessing than tree-based models!

| Preprocessing | Linear Models | Tree Models |
|---------------|---------------|-------------|
| **Feature Scaling** |  REQUIRED |  Not needed |
| **One-Hot Encoding** |  REQUIRED |  Avoid (use label/target) |
| **Handle Missing** |  REQUIRED | [WARNING] XGBoost can handle |
| **Outlier Handling** |  Important | [WARNING] Less sensitive |
| **Feature Engineering** |  Critical | [WARNING] Trees find interactions |

---

### 1. Feature Scaling (REQUIRED)

**Why:** Linear models use gradient descent or distance metrics. Features with larger scales dominate.

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (most common)
scaler = StandardScaler()  # Mean=0, Std=1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use same scaler!

# MinMaxScaler (for bounded features)
scaler = MinMaxScaler()  # Scale to [0, 1]
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to use each:**
- **StandardScaler**: Default choice, works for most cases
- **MinMaxScaler**: When features need to be in specific range [0, 1]
- **RobustScaler**: When outliers present (uses median/IQR)

**CRITICAL:** Always fit on training data only, then transform test data!

 **Wrong:**
```python
# This leaks information!
scaler.fit(X_train_and_test)
```

 **Correct:**
```python
# Fit on train, transform both
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

### 2. One-Hot Encoding (REQUIRED for Categoricals)

**Why:** Linear models interpret numeric values as ordered. Label encoding creates false ordinal relationships.

```python
# For LINEAR models, use ONE-HOT encoding
X_encoded = pd.get_dummies(X, columns=['category_col'], drop_first=True)

# OR use sklearn
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse=False)
X_encoded = encoder.fit_transform(X[['category_col']])
```

**drop_first=True**: Avoids multicollinearity (dummy variable trap)

**High cardinality problem:**
- 100 categories --> 99 dummy variables
- Solution: Group rare categories, use target encoding with regularization, or try tree models

---

### 3. Handle Missing Values (REQUIRED)

Linear models cannot handle NaN values.

```python
from sklearn.impute import SimpleImputer

# Numeric: Use median (robust to outliers)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Categorical: Use most frequent
imputer = SimpleImputer(strategy='most_frequent')
```

**Or add missing indicator:**
```python
df['feature_was_missing'] = df['feature'].isnull().astype(int)
df['feature'].fillna(df['feature'].median(), inplace=True)
```

---

### 4. Complete Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# Define transformers
numeric_features = ['age', 'income', 'credit_score']
categorical_features = ['gender', 'city', 'job']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

# Fit and predict
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

**Benefits:**
- No data leakage
- Reproducible
- Easy to deploy
- Prevents errors

---

## 8.9 Regularization Parameter Selection

### Lambda (alpha) Selection Strategy

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Automatic alpha selection with CV
ridge = RidgeCV(alphas=[0.001, 0.01, 0.1, 1, 10, 100], cv=5)
ridge.fit(X_train, y_train)
print(f"Best alpha: {ridge.alpha_}")

# For Lasso
lasso = LassoCV(alphas=None, cv=5, max_iter=10000)
lasso.fit(X_train, y_train)

# For ElasticNet
elastic = ElasticNetCV(
    alphas=None,
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],  # L1 vs L2 mix
    cv=5
)
elastic.fit(X_train, y_train)
```

**Guidelines:**
- Start with default CV search
- For small datasets: Use 5-10 fold CV
- For large datasets: Use 3-5 fold CV or validation set

---

## 8.10 Common Pitfalls & Solutions

### Pitfall 1: Forgetting to Scale

 **Wrong:**
```python
model = LogisticRegression()
model.fit(X_train, y_train)  # Features have different scales!
```

 **Correct:**
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
```

### Pitfall 2: Using Label Encoding for Linear Models

 **Wrong:**
```python
# Creates false ordering: Red=0, Blue=1, Green=2
df['color'] = LabelEncoder().fit_transform(df['color'])
# Linear model thinks Green > Blue > Red!
```

 **Correct:**
```python
# One-hot encoding
df = pd.get_dummies(df, columns=['color'], drop_first=True)
```

### Pitfall 3: Not Handling Multicollinearity

**Check VIF (Variance Inflation Factor):**
```python
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data)
# VIF > 10: High multicollinearity
```

**Solutions:**
1. Drop one of correlated features
2. Use Ridge regression
3. Use PCA
4. Combine correlated features

### Pitfall 4: Wrong Regularization Choice

**Decision tree:**
```
Need feature selection?
+--- Yes --> Lasso or ElasticNet
+--- No --> Ridge

Correlated features?
+--- Yes --> ElasticNet or Ridge
+--- No --> Lasso (if need selection) or Ridge
```

### Pitfall 5: Overfitting with SVM RBF

 **Wrong:**
```python
svm = SVC(kernel='rbf', C=1000, gamma=10)  # Very complex!
svm.fit(X_train, y_train)
```

 **Correct:**
```python
# Start with defaults and tune
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
# Then use GridSearch
```

---

## 8.11 Quick Reference: Model Selection

```
What's your problem?

+--- Continuous target?
|  +--- Linear relationship? --> Linear Regression
|  +--- Need regularization?
|  |  +--- Feature selection needed? --> Lasso
|  |  +--- All features important? --> Ridge
|  |  +--- Unsure? --> ElasticNet
|  +--- Non-linear? --> Use tree models or polynomial features
|
+--- Binary classification?
|  +--- Linear boundary? --> Logistic Regression
|  +--- Non-linear boundary?
|  |  +--- High-dimensional? --> Linear SVM first
|  |  +--- Then try --> SVM with RBF kernel
|  |  +--- Large data? --> Tree models (faster)
|  +--- Need probabilities? --> Logistic (NOT SVM)
|
+--- Need interpretability?
   +--- Linear/Logistic Regression (NOT SVM with kernel)
```

---

## 8.12 Summary Checklist

### Before Training Linear Models:
- [ ] Scale features (StandardScaler)
- [ ] One-hot encode categoricals
- [ ] Handle missing values
- [ ] Check for multicollinearity (VIF)
- [ ] Handle outliers (if present)

### Model Selection:
- [ ] Regression: Start with Linear/Ridge/Lasso
- [ ] Classification: Start with Logistic
- [ ] High-dimensional: Try Lasso or Linear SVM
- [ ] Non-linear: Try SVM with RBF or tree models

### During Training:
- [ ] Use cross-validation
- [ ] Tune regularization (alpha/C)
- [ ] Check convergence warnings
- [ ] Monitor train vs validation score

### After Training:
- [ ] Evaluate on test set
- [ ] Check coefficient magnitudes
- [ ] Interpret feature importance
- [ ] Verify no data leakage

---

## Resources & Further Reading

**Scikit-learn Documentation:**
- Linear Models: https://scikit-learn.org/stable/modules/linear_model.html
- SVM: https://scikit-learn.org/stable/modules/svm.html

**Books:**
- "Introduction to Statistical Learning" (Chapter 3, 6, 9)
- "Elements of Statistical Learning" (Chapter 3, 4, 12)

**Key Papers:**
- Ridge/Lasso: Tibshirani (1996)
- ElasticNet: Zou & Hastie (2005)
- SVM: Cortes & Vapnik (1995)

---

**Last Updated:** 2025-10-12
**Next Section:** Feature Engineering & Encoding (Phase 3)
