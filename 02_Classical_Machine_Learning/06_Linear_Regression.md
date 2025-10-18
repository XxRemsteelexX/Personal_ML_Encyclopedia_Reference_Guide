# Linear Regression

## Table of Contents
1. [Introduction](#introduction)
2. [Simple Linear Regression](#simple-linear-regression)
3. [Multiple Linear Regression](#multiple-linear-regression)
4. [Assumptions (LINE)](#assumptions-line)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Multicollinearity](#multicollinearity)
7. [Polynomial Regression](#polynomial-regression)
8. [Regularization](#regularization)
9. [Implementation](#implementation)
10. [Diagnostics](#diagnostics)
11. [When to Use](#when-to-use)

## Introduction

Linear regression is the foundation of supervised learning, modeling the relationship between features and a continuous target variable. Despite its simplicity, it remains essential in 2025 for:

- **Baseline modeling**: Establishing performance benchmarks
- **Interpretability**: Understanding feature importance and relationships
- **Inference**: Statistical hypothesis testing and confidence intervals
- **Real-time systems**: Fastest prediction times (<1ms)
- **Regularized variants**: Competitive with modern methods on high-dimensional data

### Mathematical Foundation

Linear regression assumes a linear relationship between features X and target y:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

Where:
- y: target variable (continuous)
- β₀: intercept (bias term)
- βⱼ: coefficient for feature j
- xⱼ: feature j
- ε: random error term (irreducible error)

**Goal**: Find coefficients β that minimize prediction error on training data while generalizing to new data.

## Simple Linear Regression

### Model Definition

For one feature x and target y:

```
y = β₀ + β₁x + ε
```

### Ordinary Least Squares (OLS) Solution

**Objective**: Minimize sum of squared residuals (RSS):

```
RSS(β₀, β₁) = Σᵢ(yᵢ - ŷᵢ)² = Σᵢ(yᵢ - β₀ - β₁xᵢ)²
```

**Analytical Solution** (derived by setting ∂RSS/∂β = 0):

```
β₁ = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²
   = Cov(X, Y) / Var(X)
   = r · (σy / σx)

β₀ = ȳ - β₁x̄
```

Where:
- x̄, ȳ: sample means
- r: Pearson correlation coefficient
- σx, σy: standard deviations

### Geometric Interpretation

- β₁: Slope - change in y per unit change in x
- β₀: Intercept - predicted y when x = 0
- Regression line: y = β₀ + β₁x minimizes vertical distances to points

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

class SimpleLinearRegression:
    """Simple linear regression with statistical inference."""

    def __init__(self):
        self.beta_0 = None
        self.beta_1 = None
        self.residuals = None
        self.se_beta_0 = None
        self.se_beta_1 = None

    def fit(self, X, y):
        """
        Fit simple linear regression using OLS.

        Args:
            X: 1D array of features
            y: 1D array of targets
        """
        X = np.asarray(X).flatten()
        y = np.asarray(y).flatten()

        n = len(X)
        x_mean = X.mean()
        y_mean = y.mean()

        # Calculate coefficients
        numerator = ((X - x_mean) * (y - y_mean)).sum()
        denominator = ((X - x_mean) ** 2).sum()

        self.beta_1 = numerator / denominator
        self.beta_0 = y_mean - self.beta_1 * x_mean

        # Calculate residuals
        y_pred = self.predict(X)
        self.residuals = y - y_pred

        # Standard errors for inference
        rss = (self.residuals ** 2).sum()
        se_residual = np.sqrt(rss / (n - 2))  # n-2 degrees of freedom

        self.se_beta_1 = se_residual / np.sqrt(((X - x_mean) ** 2).sum())
        self.se_beta_0 = se_residual * np.sqrt(1/n + x_mean**2 / ((X - x_mean)**2).sum())

        return self

    def predict(self, X):
        """Make predictions."""
        X = np.asarray(X).flatten()
        return self.beta_0 + self.beta_1 * X

    def summary(self, X, y):
        """Print regression summary with statistics."""
        n = len(X)

        # T-statistics and p-values
        t_beta_1 = self.beta_1 / self.se_beta_1
        t_beta_0 = self.beta_0 / self.se_beta_0

        p_beta_1 = 2 * (1 - stats.t.cdf(abs(t_beta_1), n - 2))
        p_beta_0 = 2 * (1 - stats.t.cdf(abs(t_beta_0), n - 2))

        # R-squared
        y_pred = self.predict(X)
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r_squared = 1 - (ss_res / ss_tot)

        print("Simple Linear Regression Summary")
        print("=" * 60)
        print(f"Number of observations: {n}")
        print(f"R-squared: {r_squared:.4f}")
        print(f"Residual standard error: {np.sqrt(ss_res / (n - 2)):.4f}")
        print("\nCoefficients:")
        print(f"{'Parameter':<12} {'Estimate':>10} {'Std Error':>10} {'t-value':>10} {'p-value':>10}")
        print("-" * 60)
        print(f"{'Intercept':<12} {self.beta_0:>10.4f} {self.se_beta_0:>10.4f} {t_beta_0:>10.4f} {p_beta_0:>10.4e}")
        print(f"{'Slope':<12} {self.beta_1:>10.4f} {self.se_beta_1:>10.4f} {t_beta_1:>10.4f} {p_beta_1:>10.4e}")

# Example usage
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 + 3 * X + np.random.normal(0, 2, 100)

model = SimpleLinearRegression()
model.fit(X, y)
model.summary(X, y)
```

## Multiple Linear Regression

### Model Definition

For p features:

```
y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε
```

**Matrix Form**:

```
Y = Xβ + ε

Where:
- Y: n×1 vector of targets
- X: n×(p+1) design matrix (includes intercept column)
- β: (p+1)×1 coefficient vector
- ε: n×1 error vector
```

### OLS Solution (Matrix Form)

**Objective**: Minimize RSS = ||Y - Xβ||²

**Normal Equations**:

```
X^T X β = X^T Y
```

**Closed-Form Solution**:

```
β̂ = (X^T X)^(-1) X^T Y
```

**Properties**:
- Unique solution if X^T X is invertible (requires p < n and no perfect multicollinearity)
- Unbiased: E[β̂] = β
- Minimum variance among linear unbiased estimators (Gauss-Markov Theorem)

### Derivation

To minimize RSS(β) = (Y - Xβ)^T(Y - Xβ):

```
RSS(β) = Y^T Y - 2β^T X^T Y + β^T X^T X β

∂RSS/∂β = -2X^T Y + 2X^T X β = 0

X^T X β = X^T Y

β̂ = (X^T X)^(-1) X^T Y
```

### Implementation with sklearn

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

# Generate synthetic data
np.random.seed(42)
n_samples = 1000
n_features = 5

X = np.random.randn(n_samples, n_features)
true_coef = np.array([2.5, -1.0, 3.5, 0.0, -2.0])
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Fit model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Evaluation
print("Multiple Linear Regression Results")
print("=" * 60)
print("\nCoefficients:")
for i, coef in enumerate(model.coef_):
    print(f"  β{i+1} = {coef:.4f} (true: {true_coef[i]:.4f})")
print(f"  β0 (intercept) = {model.intercept_:.4f}")

print("\nTraining Metrics:")
print(f"  R² = {r2_score(y_train, y_train_pred):.4f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_train, y_train_pred)):.4f}")
print(f"  MAE = {mean_absolute_error(y_train, y_train_pred):.4f}")

print("\nTest Metrics:")
print(f"  R² = {r2_score(y_test, y_test_pred):.4f}")
print(f"  RMSE = {np.sqrt(mean_squared_error(y_test, y_test_pred)):.4f}")
print(f"  MAE = {mean_absolute_error(y_test, y_test_pred):.4f}")
```

## Assumptions (LINE)

Linear regression requires four key assumptions (LINE):

### 1. Linearity

**Assumption**: Relationship between X and y is linear.

**Check**: Residual plots should show no pattern.

```python
import matplotlib.pyplot as plt

def check_linearity(model, X, y):
    """Plot residuals vs fitted values."""
    y_pred = model.predict(X)
    residuals = y - y_pred

    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')
    plt.show()

    # Should see random scatter around 0
    # Patterns indicate non-linearity
```

**Fix**: Transform features (log, sqrt, polynomial) or use non-linear models.

### 2. Independence

**Assumption**: Observations are independent (no autocorrelation in residuals).

**Check**: Durbin-Watson test for time series data.

```python
from statsmodels.stats.stattools import durbin_watson

def check_independence(residuals):
    """Test for autocorrelation in residuals."""
    dw_stat = durbin_watson(residuals)

    print(f"Durbin-Watson statistic: {dw_stat:.4f}")
    print("  <1.5: Positive autocorrelation")
    print("  1.5-2.5: No autocorrelation (good)")
    print("  >2.5: Negative autocorrelation")

    return dw_stat
```

**Fix**: Use time series models (ARIMA) or include lagged variables.

### 3. Normality

**Assumption**: Residuals are normally distributed.

**Check**: Q-Q plot and Shapiro-Wilk test.

```python
from scipy.stats import shapiro, probplot

def check_normality(residuals):
    """Test normality of residuals."""
    # Shapiro-Wilk test
    stat, p_value = shapiro(residuals)
    print(f"Shapiro-Wilk test: statistic={stat:.4f}, p-value={p_value:.4f}")

    if p_value > 0.05:
        print("  Residuals appear normally distributed (p > 0.05)")
    else:
        print("  Residuals may not be normally distributed (p < 0.05)")

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(8, 6))
    probplot(residuals, dist="norm", plot=ax)
    ax.set_title("Q-Q Plot")
    plt.show()
```

**Fix**: Transform target variable (log, Box-Cox) or use robust regression.

### 4. Equal Variance (Homoscedasticity)

**Assumption**: Residuals have constant variance across all fitted values.

**Check**: Scale-location plot and Breusch-Pagan test.

```python
from statsmodels.stats.diagnostic import het_breuschpagan

def check_homoscedasticity(model, X, y):
    """Test for heteroscedasticity."""
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Breusch-Pagan test
    lm_stat, lm_pvalue, fstat, f_pvalue = het_breuschpagan(residuals, X)

    print(f"Breusch-Pagan test: p-value={lm_pvalue:.4f}")
    if lm_pvalue > 0.05:
        print("  Homoscedasticity assumption satisfied (p > 0.05)")
    else:
        print("  Heteroscedasticity detected (p < 0.05)")

    # Scale-location plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, np.sqrt(np.abs(residuals)), alpha=0.5)
    plt.xlabel('Fitted values')
    plt.ylabel('√|Residuals|')
    plt.title('Scale-Location Plot')
    plt.show()
```

**Fix**: Use weighted least squares or robust standard errors.

## Evaluation Metrics

### R-Squared (R²)

**Definition**: Proportion of variance in y explained by the model.

```
R² = 1 - (SS_res / SS_tot)

Where:
- SS_res = Σ(yᵢ - ŷᵢ)²  (residual sum of squares)
- SS_tot = Σ(yᵢ - ȳ)²    (total sum of squares)
```

**Properties**:
- Range: [0, 1] (can be negative for bad models)
- R² = 1: Perfect fit
- R² = 0: Model no better than mean
- Always increases when adding features (even irrelevant ones)

### Adjusted R-Squared

**Definition**: R² penalized for number of features.

```
R²_adj = 1 - [(1 - R²) · (n - 1) / (n - p - 1)]

Where:
- n: number of samples
- p: number of features
```

**Use**: Compare models with different numbers of features. Only increases if new feature improves model more than would be expected by chance.

### Root Mean Squared Error (RMSE)

**Definition**: Square root of average squared error.

```
RMSE = √[Σ(yᵢ - ŷᵢ)² / n]
```

**Properties**:
- Same units as target variable
- Penalizes large errors more than MAE
- Standard metric for regression competitions

### Mean Absolute Error (MAE)

**Definition**: Average absolute error.

```
MAE = Σ|yᵢ - ŷᵢ| / n
```

**Properties**:
- Same units as target variable
- More robust to outliers than RMSE
- Easier to interpret

### When to Use Each Metric

```python
def evaluate_regression(y_true, y_pred, X):
    """Comprehensive regression evaluation."""
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

    n = len(y_true)
    p = X.shape[1]

    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)

    print("Regression Metrics:")
    print(f"  R² = {r2:.4f}")
    print(f"  Adjusted R² = {adj_r2:.4f}")
    print(f"  RMSE = {rmse:.4f}")
    print(f"  MAE = {mae:.4f}")

    # Interpretation
    print("\nInterpretation:")
    print(f"  Model explains {r2*100:.1f}% of variance")
    print(f"  Average prediction error: {mae:.4f} (MAE)")
    print(f"  Typical prediction error: {rmse:.4f} (RMSE)")

    return {'r2': r2, 'adj_r2': adj_r2, 'rmse': rmse, 'mae': mae}
```

## Multicollinearity

### Definition

Multicollinearity occurs when features are highly correlated, making it difficult to isolate individual feature effects.

**Problems**:
- Unstable coefficient estimates (high variance)
- Difficulty interpreting feature importance
- Poor generalization

### Detection: Variance Inflation Factor (VIF)

**Definition**: Measures how much variance of β̂ⱼ is inflated due to correlation with other features.

```
VIF_j = 1 / (1 - R²_j)

Where R²_j is from regressing x_j on all other features
```

**Interpretation**:
- VIF = 1: No correlation
- VIF < 5: Moderate correlation (acceptable)
- VIF > 10: High multicollinearity (problematic)
- VIF > 100: Severe multicollinearity

### Implementation

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

def calculate_vif(X, feature_names=None):
    """
    Calculate VIF for each feature.

    Args:
        X: Feature matrix (numpy array or pandas DataFrame)
        feature_names: List of feature names (optional)

    Returns:
        DataFrame with VIF values
    """
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
        X = X.values
    elif feature_names is None:
        feature_names = [f'X{i}' for i in range(X.shape[1])]

    vif_data = pd.DataFrame()
    vif_data["Feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    vif_data = vif_data.sort_values('VIF', ascending=False)

    print("Variance Inflation Factors:")
    print(vif_data.to_string(index=False))
    print("\nInterpretation:")
    print(f"  Features with VIF > 10: {(vif_data['VIF'] > 10).sum()}")
    print(f"  Features with VIF > 5: {(vif_data['VIF'] > 5).sum()}")

    return vif_data

# Example: Create multicollinear data
np.random.seed(42)
X1 = np.random.randn(100)
X2 = X1 + np.random.randn(100) * 0.1  # Highly correlated with X1
X3 = np.random.randn(100)
X4 = 2 * X1 + 3 * X3  # Perfect linear combination

X = np.column_stack([X1, X2, X3, X4])
vif_df = calculate_vif(X, ['X1', 'X2', 'X3', 'X4'])
```

### Solutions to Multicollinearity

1. **Remove correlated features**: Use correlation matrix or VIF
2. **Combine features**: PCA or domain-specific combinations
3. **Use regularization**: Ridge regression (see below)
4. **Collect more data**: Reduces coefficient variance

## Polynomial Regression

### Concept

Model non-linear relationships using polynomial features while maintaining linear model structure.

**Degree 2 (Quadratic)**:
```
y = β₀ + β₁x + β₂x² + ε
```

**Degree d**:
```
y = β₀ + β₁x + β₂x² + ... + βₐx^d + ε
```

### Implementation

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Generate non-linear data
np.random.seed(42)
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 0.5 * X.ravel()**2 - 3 * X.ravel() + 2 + np.random.randn(100) * 3

# Fit polynomial regression of different degrees
degrees = [1, 2, 3, 5, 10]
plt.figure(figsize=(15, 10))

for i, degree in enumerate(degrees, 1):
    # Create polynomial features and fit
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X, y)

    # Predictions
    X_plot = np.linspace(0, 10, 300).reshape(-1, 1)
    y_plot = poly_model.predict(X_plot)

    # Plot
    plt.subplot(2, 3, i)
    plt.scatter(X, y, alpha=0.5, label='Data')
    plt.plot(X_plot, y_plot, 'r-', label=f'Degree {degree}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(f'Polynomial Degree {degree}\nR² = {poly_model.score(X, y):.4f}')
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nPolynomial Regression Results:")
for degree in degrees:
    poly_model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
    poly_model.fit(X, y)
    train_score = poly_model.score(X, y)
    print(f"  Degree {degree:2d}: R² = {train_score:.4f}")
```

### Choosing Polynomial Degree

**Underfitting (degree too low)**:
- High bias, low variance
- Poor training and test performance

**Overfitting (degree too high)**:
- Low bias, high variance
- Excellent training performance, poor test performance

**Optimal degree**:
- Use cross-validation to select
- Balance between bias and variance
- Regularization helps with high-degree polynomials

```python
from sklearn.model_selection import cross_val_score

def select_polynomial_degree(X, y, max_degree=10, cv=5):
    """Select optimal polynomial degree using cross-validation."""
    degrees = range(1, max_degree + 1)
    cv_scores = []

    for degree in degrees:
        poly_model = Pipeline([
            ('poly', PolynomialFeatures(degree=degree)),
            ('linear', LinearRegression())
        ])
        scores = cross_val_score(poly_model, X, y, cv=cv,
                                 scoring='neg_mean_squared_error')
        cv_scores.append(-scores.mean())  # Convert to positive RMSE

    best_degree = degrees[np.argmin(cv_scores)]

    plt.figure(figsize=(10, 6))
    plt.plot(degrees, cv_scores, 'bo-')
    plt.axvline(best_degree, color='r', linestyle='--',
                label=f'Best degree: {best_degree}')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Cross-Validated MSE')
    plt.title('Model Selection via Cross-Validation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    print(f"Optimal polynomial degree: {best_degree}")
    return best_degree
```

## Regularization

Regularization adds penalty terms to prevent overfitting, especially crucial for:
- High-dimensional data (p ≈ n or p > n)
- Multicollinear features
- Polynomial regression with high degrees

### Ridge Regression (L2 Regularization)

**Objective**: Minimize RSS + L2 penalty

```
L(β) = ||Y - Xβ||² + λ||β||²
     = Σ(yᵢ - ŷᵢ)² + λΣβⱼ²
```

**Solution**:
```
β̂_ridge = (X^T X + λI)^(-1) X^T Y
```

**Properties**:
- Shrinks coefficients toward zero (but not exactly zero)
- Reduces variance at cost of increased bias
- Handles multicollinearity by stabilizing (X^T X)^(-1)
- All features retained

**Hyperparameter λ**:
- λ = 0: Standard OLS
- λ → ∞: β → 0
- Choose via cross-validation

### Lasso Regression (L1 Regularization)

**Objective**: Minimize RSS + L1 penalty

```
L(β) = ||Y - Xβ||² + λ||β||₁
     = Σ(yᵢ - ŷᵢ)² + λΣ|βⱼ|
```

**Properties**:
- Shrinks some coefficients exactly to zero (sparse solutions)
- Performs automatic feature selection
- No closed-form solution (requires iterative optimization)
- Unstable when features are correlated

**Use Case**: Feature selection and interpretability

### Elastic Net (L1 + L2)

**Objective**: Combine Ridge and Lasso

```
L(β) = ||Y - Xβ||² + λ₁||β||₁ + λ₂||β||²
     = Σ(yᵢ - ŷᵢ)² + α·λ·Σ|βⱼ| + (1-α)·λ·Σβⱼ²
```

Where α ∈ [0, 1] controls the mix:
- α = 0: Pure Ridge
- α = 1: Pure Lasso
- α = 0.5: Equal mix

**Properties**:
- Combines benefits of Ridge and Lasso
- Handles correlated features better than Lasso
- Performs feature selection like Lasso
- **Recommended default** for most applications

### Implementation and Comparison

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import numpy as np

# Generate high-dimensional data with correlated features
np.random.seed(42)
n_samples, n_features = 100, 50
X = np.random.randn(n_samples, n_features)
# Create some correlations
X[:, 1] = X[:, 0] + np.random.randn(n_samples) * 0.1
X[:, 2] = X[:, 0] + np.random.randn(n_samples) * 0.1

# True model: only 5 features matter
true_coef = np.zeros(n_features)
true_coef[:5] = [3, 2, -4, 1.5, -2.5]
y = X @ true_coef + np.random.randn(n_samples) * 0.5

# Split data
X_train, X_test = X[:80], X[80:]
y_train, y_test = y[:80], y[80:]

# IMPORTANT: Scale features for regularization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models with cross-validated hyperparameter selection
alphas = np.logspace(-3, 3, 100)

# Ridge
ridge = GridSearchCV(
    Ridge(),
    {'alpha': alphas},
    cv=5,
    scoring='neg_mean_squared_error'
)
ridge.fit(X_train_scaled, y_train)

# Lasso
lasso = GridSearchCV(
    Lasso(max_iter=10000),
    {'alpha': alphas},
    cv=5,
    scoring='neg_mean_squared_error'
)
lasso.fit(X_train_scaled, y_train)

# Elastic Net
elastic = GridSearchCV(
    ElasticNet(max_iter=10000),
    {'alpha': alphas, 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]},
    cv=5,
    scoring='neg_mean_squared_error'
)
elastic.fit(X_train_scaled, y_train)

# Compare results
models = {
    'Ridge': ridge.best_estimator_,
    'Lasso': lasso.best_estimator_,
    'Elastic Net': elastic.best_estimator_
}

print("Regularized Regression Comparison")
print("=" * 80)

for name, model in models.items():
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)

    print(f"\n{name}:")
    print(f"  Best alpha: {model.alpha if hasattr(model, 'alpha') else 'N/A':.4f}")
    if name == 'Elastic Net':
        print(f"  L1 ratio: {model.l1_ratio:.2f}")
    print(f"  Train R²: {train_score:.4f}")
    print(f"  Test R²: {test_score:.4f}")
    print(f"  Non-zero coefficients: {n_nonzero}/{n_features}")
    print(f"  Top 5 coefficients: {np.abs(model.coef_).argsort()[-5:][::-1]}")

# Visualize coefficient paths
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (name, alphas_range) in zip(axes, [
    ('Ridge', alphas),
    ('Lasso', alphas),
    ('Elastic Net', alphas)
]):
    coefs = []
    for alpha in alphas_range:
        if name == 'Ridge':
            model = Ridge(alpha=alpha)
        elif name == 'Lasso':
            model = Lasso(alpha=alpha, max_iter=10000)
        else:
            model = ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=10000)

        model.fit(X_train_scaled, y_train)
        coefs.append(model.coef_)

    coefs = np.array(coefs)

    for i in range(min(10, n_features)):  # Plot first 10 features
        ax.plot(alphas_range, coefs[:, i], alpha=0.7)

    ax.set_xscale('log')
    ax.set_xlabel('Alpha (regularization strength)')
    ax.set_ylabel('Coefficient value')
    ax.set_title(f'{name} Coefficient Paths')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.show()
```

### Cross-Validation for Regularization

```python
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

# Built-in cross-validation (more efficient)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000)
elastic_cv = ElasticNetCV(alphas=alphas, l1_ratio=[0.1, 0.5, 0.9],
                          cv=5, max_iter=10000)

# Fit
ridge_cv.fit(X_train_scaled, y_train)
lasso_cv.fit(X_train_scaled, y_train)
elastic_cv.fit(X_train_scaled, y_train)

print("Cross-Validated Regularization Parameters:")
print(f"  Ridge alpha: {ridge_cv.alpha_:.4f}")
print(f"  Lasso alpha: {lasso_cv.alpha_:.4f}")
print(f"  Elastic Net alpha: {elastic_cv.alpha_:.4f}, l1_ratio: {elastic_cv.l1_ratio_:.2f}")
```

## Diagnostics and Residual Analysis

### Comprehensive Diagnostic Suite

```python
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def regression_diagnostics(model, X, y, feature_names=None):
    """
    Comprehensive regression diagnostics.

    Args:
        model: Fitted sklearn model
        X: Feature matrix
        y: Target vector
        feature_names: List of feature names (optional)
    """
    y_pred = model.predict(X)
    residuals = y - y_pred

    # Standardized residuals
    residuals_std = residuals / residuals.std()

    # Leverage (hat values)
    from scipy.linalg import inv
    X_with_intercept = np.column_stack([np.ones(len(X)), X])
    hat_matrix = X_with_intercept @ inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T
    leverage = np.diag(hat_matrix)

    # Cook's distance
    n, p = X.shape
    cooks_d = (residuals_std**2 / p) * (leverage / (1 - leverage)**2)

    # Create diagnostic plots
    fig = plt.figure(figsize=(16, 12))

    # 1. Residuals vs Fitted
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Fitted values')
    ax1.set_ylabel('Residuals')
    ax1.set_title('Residuals vs Fitted')
    ax1.grid(True, alpha=0.3)

    # 2. Q-Q Plot
    ax2 = plt.subplot(2, 3, 2)
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Normal Q-Q Plot')
    ax2.grid(True, alpha=0.3)

    # 3. Scale-Location
    ax3 = plt.subplot(2, 3, 3)
    ax3.scatter(y_pred, np.sqrt(np.abs(residuals_std)), alpha=0.5)
    ax3.set_xlabel('Fitted values')
    ax3.set_ylabel('√|Standardized Residuals|')
    ax3.set_title('Scale-Location')
    ax3.grid(True, alpha=0.3)

    # 4. Residuals vs Leverage
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(leverage, residuals_std, alpha=0.5)
    ax4.axhline(y=0, color='r', linestyle='--')
    ax4.set_xlabel('Leverage')
    ax4.set_ylabel('Standardized Residuals')
    ax4.set_title('Residuals vs Leverage')
    ax4.grid(True, alpha=0.3)

    # Mark high leverage points
    high_leverage = leverage > 2 * (p + 1) / n
    ax4.scatter(leverage[high_leverage], residuals_std[high_leverage],
               color='red', s=100, facecolors='none', edgecolors='r')

    # 5. Cook's Distance
    ax5 = plt.subplot(2, 3, 5)
    ax5.stem(range(len(cooks_d)), cooks_d, markerfmt=',')
    ax5.axhline(y=4/n, color='r', linestyle='--', label="Cook's D threshold")
    ax5.set_xlabel('Observation index')
    ax5.set_ylabel("Cook's Distance")
    ax5.set_title("Cook's Distance")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. Histogram of Residuals
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Residuals')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Histogram of Residuals')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print("Regression Diagnostics Summary")
    print("=" * 80)

    # Normality test
    shapiro_stat, shapiro_p = stats.shapiro(residuals)
    print(f"\nNormality (Shapiro-Wilk test):")
    print(f"  Statistic: {shapiro_stat:.4f}, p-value: {shapiro_p:.4f}")
    print(f"  {'✓' if shapiro_p > 0.05 else '✗'} Residuals appear normal" if shapiro_p > 0.05
          else "  ✗ Residuals may not be normal")

    # Influential points
    influential = np.where(cooks_d > 4/n)[0]
    print(f"\nInfluential Points (Cook's D > {4/n:.4f}):")
    print(f"  {len(influential)} observations ({len(influential)/n*100:.1f}%)")
    if len(influential) > 0:
        print(f"  Indices: {influential[:10]}{'...' if len(influential) > 10 else ''}")

    # High leverage points
    high_lev_indices = np.where(high_leverage)[0]
    print(f"\nHigh Leverage Points (leverage > {2*(p+1)/n:.4f}):")
    print(f"  {len(high_lev_indices)} observations ({len(high_lev_indices)/n*100:.1f}%)")

    return {
        'residuals': residuals,
        'leverage': leverage,
        'cooks_distance': cooks_d,
        'influential_points': influential
    }

# Example usage
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
model = LinearRegression()
model.fit(X, y)

diagnostics = regression_diagnostics(model, X, y)
```

## When to Use Linear Regression

### Ideal Use Cases

1. **Interpretability is critical**
   - Healthcare: Understanding feature impact on outcomes
   - Finance: Regulatory compliance requires explainable models
   - Scientific research: Testing hypotheses about relationships

2. **Fast inference required**
   - Real-time systems (<1ms latency)
   - Edge devices with limited compute
   - High-throughput APIs (millions of predictions/second)

3. **Small to medium tabular datasets**
   - n < 100,000 samples
   - p < 100 features
   - Linear or polynomial relationships

4. **Baseline modeling**
   - Quick prototypes to establish performance floor
   - Sanity checks before complex models
   - Ablation studies

### When NOT to Use

1. **Complex non-linear relationships**
   - Use: Gradient boosting, neural networks
   - Example: Image recognition, NLP

2. **Very high-dimensional data (p >> n)**
   - Use: Regularized regression (Lasso, Elastic Net) or dimensionality reduction
   - Example: Genomics, text classification

3. **Categorical targets**
   - Use: Logistic regression, classification algorithms
   - Example: Binary or multi-class classification

4. **Time series with complex patterns**
   - Use: ARIMA, Prophet, LSTMs
   - Example: Stock prices, weather forecasting

### 2025 Recommendations

**Structured/Tabular Data**:
1. Baseline: Linear regression (Ridge/Lasso/Elastic Net)
2. Production: LightGBM or CatBoost
3. Interpretability: Linear regression + SHAP values
4. Real-time: Linear regression or lightweight boosting

**Model Stack**:
- Use linear regression as one component in stacking ensemble
- Captures linear relationships while other models handle non-linearity
- Improves overall robustness

## Summary

Linear regression remains essential in 2025 for:

**Strengths**:
- Interpretability and statistical inference
- Fast training (<1 second for n < 1M)
- Fast prediction (<1ms)
- No hyperparameter tuning for OLS
- Solid baseline for any regression task

**Limitations**:
- Assumes linear relationships
- Sensitive to outliers
- Requires LINE assumptions
- Poor with high multicollinearity (without regularization)

**Best Practices**:
1. Always check assumptions (LINE)
2. Use regularization for high-dimensional data
3. Scale features for regularized methods
4. Cross-validate regularization parameters
5. Perform diagnostic checks
6. Consider as baseline or ensemble component

**Modern Variants (2025)**:
- Elastic Net: Default regularized choice
- Polynomial features: For non-linear relationships
- Stacking component: Complement gradient boosting
- SHAP values: Enhanced interpretability

**Code Template**:
```python
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Production-ready pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('regression', ElasticNetCV(
        l1_ratio=[0.1, 0.5, 0.9],
        alphas=np.logspace(-3, 3, 100),
        cv=5,
        max_iter=10000
    ))
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

---

**Last Updated**: 2025-10-14
**Prerequisites**: Linear algebra, calculus, probability
**Next Topics**: Logistic regression, decision trees, ensemble methods
