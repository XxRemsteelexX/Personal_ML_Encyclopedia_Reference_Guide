# Kaggle Competition Winning Strategies - Grandmaster Playbook

## Overview

**Kaggle Grandmaster techniques** distilled from hundreds of competition wins. These battle-tested strategies consistently place in top 1-5%.

**Key Insight:** Winning isn't about one trick - it's about systematic application of proven techniques.

---

## The Winning Formula

```
Problem Understanding (20%)
    v
Strong Baseline (20%)
    v
Feature Engineering (25%)
    v
Model Ensembles (25%)
    v
Final Refinements (10%)
    v
[trophy] Top 1% Finish
```

---

## 1. Smarter Exploratory Data Analysis (EDA)

### Beyond Basic Checks

**Standard EDA (What everyone does):**
- Check data types
- Look for missing values
- Plot distributions

**Grandmaster EDA (What winners do):**
- **Train vs Test distribution analysis**
- **Temporal patterns** (data leakage detection)
- **Target variable analysis** (distribution, correlations)
- **Competition-specific nuances**

### Train vs Test Distribution Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def compare_train_test_distributions(train_df, test_df, features):
    """
    Critical check: Are train and test from same distribution?
    """
    results = []

    for feature in features:
        # Kolmogorov-Smirnov test
        stat, pvalue = stats.ks_2samp(
            train_df[feature].dropna(),
            test_df[feature].dropna()
        )

        results.append({
            'feature': feature,
            'ks_statistic': stat,
            'p_value': pvalue,
            'different_distribution': pvalue < 0.05
        })

        # Visual comparison
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        train_df[feature].hist(bins=50, alpha=0.5, label='Train', density=True)
        test_df[feature].hist(bins=50, alpha=0.5, label='Test', density=True)
        plt.legend()
        plt.title(f'{feature} - Distribution Comparison')

        plt.subplot(1, 2, 2)
        sns.boxplot(data=pd.DataFrame({
            'Train': train_df[feature],
            'Test': test_df[feature]
        }))
        plt.title(f'{feature} - Box Plot')

        plt.tight_layout()
        plt.show()

    return pd.DataFrame(results)

# Usage
distribution_report = compare_train_test_distributions(
    train_df,
    test_df,
    numeric_features
)

# Flag problematic features
different_dist = distribution_report[distribution_report['different_distribution']]
print(f"[WARNING] Features with different distributions: {len(different_dist)}")
print(different_dist[['feature', 'ks_statistic', 'p_value']])
```

**Why this matters:**
- If train/test distributions differ --> model will fail
- Can reveal data leakage or time-based splits
- Informs feature engineering strategy

---

### Temporal Pattern Detection

```python
def detect_temporal_patterns(train_df, test_df, date_column='date'):
    """
    Check for time-based data leakage
    """
    # Create time index if not exists
    if date_column not in train_df.columns:
        train_df['time_index'] = range(len(train_df))
        test_df['time_index'] = range(len(train_df), len(train_df) + len(test_df))
    else:
        train_df['time_index'] = pd.to_datetime(train_df[date_column])
        test_df['time_index'] = pd.to_datetime(test_df[date_column])

    # Check if test follows train chronologically
    train_max_time = train_df['time_index'].max()
    test_min_time = test_df['time_index'].min()

    print(f"Train max time: {train_max_time}")
    print(f"Test min time: {test_min_time}")

    if test_min_time > train_max_time:
        print("[x] Chronological split (test is future)")
        cv_strategy = "TimeSeriesSplit"
    else:
        print("[x] Random split")
        cv_strategy = "KFold or StratifiedKFold"

    # Check for trends over time
    for col in train_df.select_dtypes(include=[np.number]).columns:
        correlation = train_df[[col, 'time_index']].corr().iloc[0, 1]
        if abs(correlation) > 0.3:
            print(f"[WARNING] {col} has temporal trend (corr={correlation:.3f})")

    return cv_strategy
```

---

### Competition-Specific Analysis

```python
def competition_specific_eda(train_df, target_col, competition_type):
    """
    Tailored EDA based on competition type
    """
    if competition_type == 'classification':
        # Class balance
        print("Class distribution:")
        print(train_df[target_col].value_counts(normalize=True))

        # Check for class imbalance
        class_counts = train_df[target_col].value_counts()
        imbalance_ratio = class_counts.max() / class_counts.min()

        if imbalance_ratio > 10:
            print(f"[WARNING] Severe class imbalance (ratio: {imbalance_ratio:.1f})")
            print("--> Consider: SMOTE, class weights, focal loss")

    elif competition_type == 'regression':
        # Target distribution
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        train_df[target_col].hist(bins=50, edgecolor='black')
        plt.title('Target Distribution')

        plt.subplot(1, 3, 2)
        stats.probplot(train_df[target_col], dist="norm", plot=plt)
        plt.title('Q-Q Plot')

        plt.subplot(1, 3, 3)
        np.log1p(train_df[target_col]).hist(bins=50, edgecolor='black')
        plt.title('Log-Transformed Target')

        plt.tight_layout()
        plt.show()

        # Check if log transform helps
        skew_original = train_df[target_col].skew()
        skew_log = np.log1p(train_df[target_col]).skew()

        print(f"Skewness - Original: {skew_original:.2f}, Log: {skew_log:.2f}")
        if abs(skew_log) < abs(skew_original):
            print("--> Consider log-transforming target")

    elif competition_type == 'time_series':
        # Stationarity check
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(train_df[target_col].dropna())
        print(f"ADF Statistic: {result[0]:.4f}")
        print(f"p-value: {result[1]:.4f}")

        if result[1] > 0.05:
            print("[WARNING] Non-stationary series")
            print("--> Consider: Differencing, detrending, seasonal decomposition")
```

---

## 2. Diverse Baseline Models

### The Baseline Strategy

**Goal:** Quickly understand which model family works best

**Approach:** Train 5-10 diverse models in parallel

```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

def create_diverse_baselines(X, y, task='classification'):
    """
    Create diverse baseline models
    """
    if task == 'classification':
        models = {
            'Logistic': LogisticRegression(max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'GBM': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
            'MLP': MLPClassifier(hidden_layers=(100, 50), max_iter=500, random_state=42)
        }
    else:  # regression
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from xgboost import XGBRegressor
        from lightgbm import LGBMRegressor
        from catboost import CatBoostRegressor

        models = {
            'Ridge': Ridge(),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GBM': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'XGBoost': XGBRegressor(n_estimators=100, random_state=42),
            'LightGBM': LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
            'CatBoost': CatBoostRegressor(n_estimators=100, random_state=42, verbose=0)
        }

    # Train and evaluate all models
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5,
                                scoring='accuracy' if task == 'classification' else 'r2')
        results[name] = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        print(f"{name:15} - Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Identify best model family
    best_model = max(results, key=lambda x: results[x]['mean_score'])
    print(f"\n[trophy] Best baseline: {best_model}")

    return results, models[best_model]
```

**Key Insights:**
- **Tree-based models (XGB, LGBM, CatBoost)** --> Usually best for tabular data
- **Neural networks** --> Good for complex interactions, large datasets
- **Linear models** --> Fast baseline, good for high-dimensional sparse data

---

## 3. Feature Engineering - The 25% Advantage

### Systematic Feature Generation

```python
class FeatureEngineer:
    """
    Systematic feature engineering pipeline
    """

    def __init__(self):
        self.generated_features = []

    def create_interaction_features(self, df, numeric_cols, max_combinations=2):
        """
        Create interaction features (multiplication, division, addition)
        """
        from itertools import combinations

        new_features = {}

        for col1, col2 in combinations(numeric_cols, max_combinations):
            # Multiplication
            new_features[f'{col1}_X_{col2}'] = df[col1] * df[col2]

            # Division (avoid division by zero)
            new_features[f'{col1}_DIV_{col2}'] = df[col1] / (df[col2] + 1e-5)
            new_features[f'{col2}_DIV_{col1}'] = df[col2] / (df[col1] + 1e-5)

            # Addition
            new_features[f'{col1}_PLUS_{col2}'] = df[col1] + df[col2]

            # Subtraction
            new_features[f'{col1}_MINUS_{col2}'] = df[col1] - df[col2]

        self.generated_features.extend(new_features.keys())
        return pd.DataFrame(new_features)

    def create_aggregation_features(self, df, group_cols, agg_cols):
        """
        GroupBy aggregations - MOST POWERFUL technique
        """
        new_features = {}

        for group_col in group_cols:
            for agg_col in agg_cols:
                # Mean
                new_features[f'{agg_col}_mean_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('mean')

                # Std
                new_features[f'{agg_col}_std_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('std')

                # Min/Max
                new_features[f'{agg_col}_min_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('min')
                new_features[f'{agg_col}_max_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('max')

                # Median
                new_features[f'{agg_col}_median_by_{group_col}'] = df.groupby(group_col)[agg_col].transform('median')

                # Rank within group
                new_features[f'{agg_col}_rank_by_{group_col}'] = df.groupby(group_col)[agg_col].rank()

        self.generated_features.extend(new_features.keys())
        return pd.DataFrame(new_features)

    def create_categorical_features(self, df, cat_cols):
        """
        Categorical feature engineering
        """
        new_features = {}

        for col in cat_cols:
            # Frequency encoding
            freq_map = df[col].value_counts(normalize=True).to_dict()
            new_features[f'{col}_frequency'] = df[col].map(freq_map)

            # Count encoding
            count_map = df[col].value_counts().to_dict()
            new_features[f'{col}_count'] = df[col].map(count_map)

            # Nunique per category (if applicable with another categorical)
            for other_col in cat_cols:
                if col != other_col:
                    nunique_map = df.groupby(col)[other_col].nunique().to_dict()
                    new_features[f'{other_col}_nunique_per_{col}'] = df[col].map(nunique_map)

        self.generated_features.extend(new_features.keys())
        return pd.DataFrame(new_features)

    def create_time_features(self, df, date_col):
        """
        Time-based features
        """
        df[date_col] = pd.to_datetime(df[date_col])
        new_features = {}

        new_features[f'{date_col}_year'] = df[date_col].dt.year
        new_features[f'{date_col}_month'] = df[date_col].dt.month
        new_features[f'{date_col}_day'] = df[date_col].dt.day
        new_features[f'{date_col}_dayofweek'] = df[date_col].dt.dayofweek
        new_features[f'{date_col}_quarter'] = df[date_col].dt.quarter
        new_features[f'{date_col}_is_weekend'] = (df[date_col].dt.dayofweek >= 5).astype(int)
        new_features[f'{date_col}_is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        new_features[f'{date_col}_is_month_end'] = df[date_col].dt.is_month_end.astype(int)

        # Cyclical encoding
        new_features[f'{date_col}_month_sin'] = np.sin(2 * np.pi * df[date_col].dt.month / 12)
        new_features[f'{date_col}_month_cos'] = np.cos(2 * np.pi * df[date_col].dt.month / 12)
        new_features[f'{date_col}_day_sin'] = np.sin(2 * np.pi * df[date_col].dt.day / 31)
        new_features[f'{date_col}_day_cos'] = np.cos(2 * np.pi * df[date_col].dt.day / 31)

        self.generated_features.extend(new_features.keys())
        return pd.DataFrame(new_features)

# Usage
fe = FeatureEngineer()

# Generate all features
interaction_features = fe.create_interaction_features(df, numeric_cols)
aggregation_features = fe.create_aggregation_features(df, group_cols=['category'], agg_cols=numeric_cols)
categorical_features = fe.create_categorical_features(df, cat_cols)
time_features = fe.create_time_features(df, 'date')

# Combine
df_enriched = pd.concat([df, interaction_features, aggregation_features, categorical_features, time_features], axis=1)

print(f"Original features: {len(df.columns)}")
print(f"Generated features: {len(fe.generated_features)}")
print(f"Total features: {len(df_enriched.columns)}")
```

**Impact:** Feature engineering alone can improve scores by 5-15%

---

## 4. Hill Climbing Ensembles

### Greedy Ensemble Building

```python
def hill_climbing_ensemble(models, X_val, y_val, metric='accuracy'):
    """
    Systematically add models to ensemble
    """
    from sklearn.metrics import accuracy_score, mean_squared_error

    # Get predictions from all models
    predictions = {}
    for name, model in models.items():
        predictions[name] = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)

    # Start with best single model
    scores = {}
    for name, pred in predictions.items():
        if metric == 'accuracy':
            scores[name] = accuracy_score(y_val, (pred > 0.5).astype(int))
        else:
            scores[name] = -mean_squared_error(y_val, pred)

    best_model = max(scores, key=scores.get)
    ensemble = [best_model]
    ensemble_pred = predictions[best_model]
    best_score = scores[best_model]

    print(f"Starting with {best_model}: {best_score:.4f}")

    # Iteratively add models
    remaining_models = set(predictions.keys()) - {best_model}

    while remaining_models:
        improved = False

        for candidate in remaining_models:
            # Try adding this model
            new_ensemble_pred = (ensemble_pred + predictions[candidate]) / 2

            if metric == 'accuracy':
                new_score = accuracy_score(y_val, (new_ensemble_pred > 0.5).astype(int))
            else:
                new_score = -mean_squared_error(y_val, new_ensemble_pred)

            if new_score > best_score:
                best_score = new_score
                ensemble.append(candidate)
                ensemble_pred = new_ensemble_pred
                remaining_models.remove(candidate)
                improved = True
                print(f"Added {candidate}: {best_score:.4f}")
                break

        if not improved:
            break

    print(f"\nFinal ensemble: {ensemble}")
    print(f"Final score: {best_score:.4f}")

    return ensemble, ensemble_pred
```

---

## 5. Stacking - The Grandmaster Technique

### Multi-Level Stacking

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import numpy as np

def create_stacking_features(models, X_train, y_train, X_test, n_folds=5):
    """
    Create out-of-fold predictions for stacking
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Initialize arrays
    train_meta_features = np.zeros((X_train.shape[0], len(models)))
    test_meta_features = np.zeros((X_test.shape[0], len(models)))

    for i, (name, model) in enumerate(models.items()):
        # Out-of-fold predictions for train
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            model.fit(X_tr, y_tr)
            val_pred = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val)
            train_meta_features[val_idx, i] = val_pred

        # Full train predictions for test
        model.fit(X_train, y_train)
        test_pred = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
        test_meta_features[:, i] = test_pred

        print(f"Completed {name}")

    return train_meta_features, test_meta_features

# Level 1: Base models
base_models = {
    'XGBoost': XGBClassifier(n_estimators=300, learning_rate=0.05, random_state=42),
    'LightGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=-1),
    'CatBoost': CatBoostClassifier(n_estimators=300, learning_rate=0.05, random_state=42, verbose=0),
    'RandomForest': RandomForestClassifier(n_estimators=300, random_state=42)
}

# Create meta features
train_meta, test_meta = create_stacking_features(base_models, X_train, y_train, X_test)

# Level 2: Meta model
meta_model = Ridge()  # Simple meta-learner often works best
meta_model.fit(train_meta, y_train)

# Final predictions
final_predictions = meta_model.predict(test_meta)
```

**Why stacking wins:**
- Captures complementary strengths of different models
- Meta-model learns optimal combination
- Often +2-5% improvement over best single model

---

## Key Takeaways

1. **EDA is strategic** - Focus on train/test differences, temporal patterns
2. **Diverse baselines** - Quickly identify best model family
3. **Feature engineering** - GroupBy aggregations are most powerful
4. **Ensembles are essential** - Hill climbing or stacking, never submit single model
5. **Systematic approach** - Follow proven playbook, don't reinvent

**Competition Winning Formula:**
- Solid EDA (find quirks/leakage) --> +5-10%
- Strong feature engineering --> +10-20%
- Well-tuned ensemble --> +5-10%
- **Total edge: 20-40% over naive approaches**

**Next:** `02_Advanced_Ensembles.md` - Deep dive into stacking, blending, and weighted ensembles
