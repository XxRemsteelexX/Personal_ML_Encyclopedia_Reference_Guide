# Time Series Competition Solutions

## Table of Contents

- [Introduction](#introduction)
- [Feature Engineering for Time Series](#feature-engineering-for-time-series)
  - [Lag Features](#lag-features)
  - [Rolling Window Statistics](#rolling-window-statistics)
  - [Expanding Window Features](#expanding-window-features)
  - [Date Decomposition](#date-decomposition)
  - [Fourier Features for Seasonality](#fourier-features-for-seasonality)
  - [Target Encoding with Time-Based CV](#target-encoding-with-time-based-cv)
  - [Difference and Ratio Features](#difference-and-ratio-features)
  - [Complete Feature Engineering Pipeline](#complete-feature-engineering-pipeline)
- [Validation Strategies](#validation-strategies)
  - [Walk-Forward Validation](#walk-forward-validation)
  - [Sliding Window Validation](#sliding-window-validation)
  - [Purged Group Time Series Split](#purged-group-time-series-split)
  - [Embargo Period for Financial Data](#embargo-period-for-financial-data)
  - [Multi-Step Ahead Validation](#multi-step-ahead-validation)
  - [Custom TimeSeriesSplit with Purging](#custom-timeseriessplit-with-purging)
- [Winning Solution Breakdowns](#winning-solution-breakdowns)
  - [M5 Forecasting Accuracy 2020](#m5-forecasting-accuracy-2020)
  - [Optiver Realized Volatility 2021](#optiver-realized-volatility-2021)
  - [Jane Street Market Prediction 2021](#jane-street-market-prediction-2021)
  - [Ubiquant Market Prediction 2022](#ubiquant-market-prediction-2022)
  - [G-Research Crypto Forecasting 2022](#g-research-crypto-forecasting-2022)
  - [Walmart Sales Forecasting](#walmart-sales-forecasting)
- [GBDT Approaches](#gbdt-approaches)
  - [LightGBM for Time Series](#lightgbm-for-time-series)
  - [Feature Importance Selection](#feature-importance-selection)
  - [Multi-Step Forecasting Strategies](#multi-step-forecasting-strategies)
  - [Quantile Regression for Prediction Intervals](#quantile-regression-for-prediction-intervals)
  - [LightGBM Time Series Pipeline](#lightgbm-time-series-pipeline)
- [Neural Approaches](#neural-approaches)
  - [N-BEATS](#n-beats)
  - [Temporal Fusion Transformer](#temporal-fusion-transformer)
  - [PatchTST](#patchtst)
  - [DeepAR](#deepar)
  - [WaveNet-Style Dilated Convolutions](#wavenet-style-dilated-convolutions)
  - [LSTM and GRU Baselines](#lstm-and-gru-baselines)
  - [Neural Forecasting Code](#neural-forecasting-code)
- [Statistical Baselines](#statistical-baselines)
  - [ARIMA and SARIMA](#arima-and-sarima)
  - [Prophet](#prophet)
  - [Exponential Smoothing ETS](#exponential-smoothing-ets)
  - [Theta Method](#theta-method)
  - [When Statistical Methods Beat ML](#when-statistical-methods-beat-ml)
  - [Quick Baseline with StatsForecast](#quick-baseline-with-statsforecast)
- [Hierarchical Forecasting](#hierarchical-forecasting)
  - [Aggregation Approaches](#aggregation-approaches)
  - [Reconciliation Methods](#reconciliation-methods)
  - [M5-Style Hierarchy](#m5-style-hierarchy)
  - [Hierarchical Reconciliation Code](#hierarchical-reconciliation-code)
- [Financial Time Series Specifics](#financial-time-series-specifics)
  - [Feature Neutralization](#feature-neutralization)
  - [Era-Based Cross-Validation](#era-based-cross-validation)
  - [Sharpe Ratio Optimization](#sharpe-ratio-optimization)
  - [Online Learning and Incremental Updates](#online-learning-and-incremental-updates)
  - [Dealing with Regime Changes](#dealing-with-regime-changes)
  - [Feature Neutralization Code](#feature-neutralization-code)
- [Post-Processing](#post-processing)
  - [Clipping Predictions](#clipping-predictions)
  - [Rounding for Count Data](#rounding-for-count-data)
  - [Ensemble of Different Horizons](#ensemble-of-different-horizons)
  - [Conformal Prediction for Intervals](#conformal-prediction-for-intervals)
  - [Post-Processing Pipeline Code](#post-processing-pipeline-code)
- [Resources](#resources)

---

## Introduction

Time series competitions consistently rank among the most challenging on Kaggle and similar platforms because they introduce **temporal dependencies** that violate the i.i.d. assumption underlying standard ML pipelines. The leaderboard is dominated by solutions that combine **gradient-boosted decision trees** (LightGBM, XGBoost, CatBoost) with meticulously crafted lag and rolling-window features, validated through rigorous walk-forward or purged cross-validation schemes. Between 2018 and 2024, LightGBM appeared in over 80% of gold-medal time series solutions, while neural approaches (N-BEATS, Temporal Fusion Transformer, PatchTST) gained traction primarily in competitions with very long histories or multiple related series.

The key distinguishing factor from tabular competitions is that **information leakage is catastrophic and subtle**. A single feature computed using future data -- for example, a rolling mean that includes the target period -- can produce a validation score that is 10x better than the true generalization error. Competitions like M5 Forecasting (30,490 series, 1,941 days of Walmart sales), Optiver Realized Volatility (112 stocks, 10-minute buckets), and Jane Street Market Prediction (anonymized financial features, online submission) each demand domain-specific validation and feature engineering strategies. This guide covers the concrete techniques, hyperparameters, and code patterns that separate top-10 finishes from median-ranking submissions.

---

## Feature Engineering for Time Series

### Lag Features

**Lag features** are the single most important feature class in tabular time series competitions. They encode the value of the target variable at prior timesteps, letting tree-based models learn autoregressive patterns without explicit recurrence. Standard lag choices depend on the data frequency: for daily retail data, lags at 7, 14, 21, 28 (weekly periodicity), 91, 182, 365 (seasonal periodicity) are standard. For hourly data, lags at 1, 2, 3, 6, 12, 24, 48, 168 (one week in hours) cover the relevant cycles. The M5 competition 1st-place solution used **lags 1 through 28 plus 364 and 365** to capture both short-term momentum and year-over-year patterns.

```python
import pandas as pd
import numpy as np

def add_lag_features(df, target_col, group_col, lags):
    """
    Add lag features grouped by entity (e.g., item_id, store_id).

    Parameters
    ----------
    df : pd.DataFrame with columns [group_col, target_col], sorted by date.
    target_col : str, name of the target variable.
    group_col : str or list[str], grouping column(s).
    lags : list[int], lag offsets in timesteps.

    Returns
    -------
    df with new columns named {target_col}_lag_{k} for each k in lags.
    """
    for lag in lags:
        col_name = f"{target_col}_lag_{lag}"
        df[col_name] = df.groupby(group_col)[target_col].shift(lag)
    return df

# Example usage for daily retail sales data
DAILY_LAGS = [1, 2, 3, 7, 14, 21, 28, 35, 42, 49, 56, 91, 182, 364, 365]
df = add_lag_features(df, target_col="sales", group_col="item_store_id", lags=DAILY_LAGS)
```

A critical pitfall: when forecasting **h steps ahead**, lags 1 through h-1 are unavailable at inference time. In the M5 competition the forecast horizon was 28 days, so the minimum usable lag for non-recursive approaches was lag 28. Solutions that used lags 1-27 required **recursive prediction** -- predicting day t, plugging the prediction back in as a lag feature, then predicting day t+1. Recursive approaches accumulate error but capture short-term momentum that direct approaches miss.

### Rolling Window Statistics

**Rolling window features** summarize recent history with aggregations computed over a fixed trailing window. The standard set includes mean, standard deviation, minimum, maximum, median, and quantiles (10th, 25th, 75th, 90th percentiles). Window sizes typically match the lag structure: 7, 14, 28, 56, 91, 182, 365 days for daily data. The M5 1st-place solution computed rolling means and standard deviations over windows of 7, 14, 30, 60, 180 days, all shifted by the forecast horizon to prevent leakage.

```python
def add_rolling_features(df, target_col, group_col, windows, min_periods=1, shift=1):
    """
    Add rolling window statistics grouped by entity.

    Parameters
    ----------
    df : pd.DataFrame, sorted by date within each group.
    target_col : str, name of target variable.
    group_col : str or list[str], grouping columns.
    windows : list[int], window sizes in timesteps.
    min_periods : int, minimum observations for valid result.
    shift : int, shift to prevent leakage (>= forecast horizon for direct).

    Returns
    -------
    df with rolling mean, std, min, max, q10, q90 columns.
    """
    grouped = df.groupby(group_col)[target_col]
    for w in windows:
        rolled = grouped.transform(
            lambda x: x.shift(shift).rolling(window=w, min_periods=min_periods)
        )
        # rolling() returns a Rolling object inside transform; handle explicitly
        shifted = df.groupby(group_col)[target_col].shift(shift)
        for stat_name, func in [
            ("mean", lambda s: s.rolling(w, min_periods=min_periods).mean()),
            ("std", lambda s: s.rolling(w, min_periods=min_periods).std()),
            ("min", lambda s: s.rolling(w, min_periods=min_periods).min()),
            ("max", lambda s: s.rolling(w, min_periods=min_periods).max()),
            ("q10", lambda s: s.rolling(w, min_periods=min_periods).quantile(0.10)),
            ("q90", lambda s: s.rolling(w, min_periods=min_periods).quantile(0.90)),
        ]:
            col_name = f"{target_col}_roll_{stat_name}_{w}"
            df[col_name] = shifted.groupby(df[group_col]).transform(
                lambda x: func(x)
            )
    return df

WINDOWS = [7, 14, 28, 56, 91, 182, 365]
df = add_rolling_features(df, "sales", "item_store_id", WINDOWS, shift=28)
```

### Expanding Window Features

**Expanding window features** use all available history up to a given point rather than a fixed trailing window. They capture long-run statistics such as the historical mean, cumulative sum, lifetime minimum and maximum, and running count of non-zero observations. Expanding features are especially useful for series with structural changes because they reflect the full trajectory of a series. In the M5 competition, expanding mean sales (computed from the first available date up to 28 days before the target) was a top-10 importance feature across multiple winning solutions.

```python
def add_expanding_features(df, target_col, group_col, shift=1):
    """
    Add expanding (cumulative) window features.
    """
    shifted = df.groupby(group_col)[target_col].shift(shift)
    grouped_shifted = shifted.groupby(df[group_col])

    df[f"{target_col}_expanding_mean"] = grouped_shifted.transform(
        lambda x: x.expanding(min_periods=1).mean()
    )
    df[f"{target_col}_expanding_std"] = grouped_shifted.transform(
        lambda x: x.expanding(min_periods=1).std()
    )
    df[f"{target_col}_expanding_count_nonzero"] = grouped_shifted.transform(
        lambda x: (x > 0).expanding(min_periods=1).sum()
    )
    df[f"{target_col}_expanding_max"] = grouped_shifted.transform(
        lambda x: x.expanding(min_periods=1).max()
    )
    return df
```

### Date Decomposition

**Date decomposition features** extract calendar attributes from the timestamp and encode them as integers or cyclical features. The standard set includes: `day_of_week` (0-6), `day_of_month` (1-31), `day_of_year` (1-366), `week_of_year` (1-53), `month` (1-12), `quarter` (1-4), `year`, `is_weekend` (boolean), `is_month_start`, `is_month_end`, `is_quarter_start`, `is_quarter_end`. For retail forecasting, holiday indicators are essential: US federal holidays, Black Friday, Christmas Eve, Super Bowl Sunday, and state-specific holidays. The M5 competition data included SNAP (food stamp) payment schedules per state, which were among the most predictive calendar features.

```python
def add_date_features(df, date_col="date"):
    """
    Extract calendar features from a datetime column.
    """
    dt = pd.to_datetime(df[date_col])
    df["day_of_week"] = dt.dt.dayofweek          # 0=Monday, 6=Sunday
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["is_month_start"] = dt.dt.is_month_start.astype(int)
    df["is_month_end"] = dt.dt.is_month_end.astype(int)
    df["is_quarter_start"] = dt.dt.is_quarter_start.astype(int)
    df["is_quarter_end"] = dt.dt.is_quarter_end.astype(int)
    return df

def add_holiday_features(df, date_col="date", country="US"):
    """
    Add holiday indicators using the holidays library.
    Requires: pip install holidays
    """
    import holidays
    holiday_dates = holidays.country_holidays(country, years=df[date_col].dt.year.unique())
    df["is_holiday"] = pd.to_datetime(df[date_col]).isin(holiday_dates).astype(int)

    # Days until / since nearest holiday
    holiday_series = pd.Series(sorted(holiday_dates.keys()))
    dates = pd.to_datetime(df[date_col])
    df["days_to_next_holiday"] = dates.apply(
        lambda d: min((h - d.date()).days for h in holiday_dates if h >= d.date())
        if any(h >= d.date() for h in holiday_dates) else 999
    )
    return df
```

### Fourier Features for Seasonality

**Fourier features** encode periodic patterns as pairs of sine and cosine terms at specified frequencies. For a period P and harmonic order k, the features are `sin(2 * pi * k * t / P)` and `cos(2 * pi * k * t / P)`. Standard choices: for weekly seasonality use P=7 with k=1,2,3; for annual seasonality use P=365.25 with k=1,2,3,4,5. Prophet uses 10 Fourier terms for annual seasonality and 3 for weekly seasonality by default. In tree-based pipelines, Fourier features are less critical (trees can learn step functions from integer calendar features), but they are essential for linear models and neural networks that benefit from smooth periodic inputs.

```python
def add_fourier_features(df, date_col="date", periods=None):
    """
    Add Fourier sine/cosine features for specified periodicities.

    Parameters
    ----------
    periods : dict, mapping period_name -> (period_length, n_harmonics)
              e.g., {"weekly": (7, 3), "annual": (365.25, 5)}
    """
    if periods is None:
        periods = {"weekly": (7, 3), "annual": (365.25, 5)}

    dt = pd.to_datetime(df[date_col])
    day_number = (dt - dt.min()).dt.days.values.astype(float)

    for name, (period, n_harmonics) in periods.items():
        for k in range(1, n_harmonics + 1):
            df[f"fourier_{name}_sin_{k}"] = np.sin(2 * np.pi * k * day_number / period)
            df[f"fourier_{name}_cos_{k}"] = np.cos(2 * np.pi * k * day_number / period)
    return df
```

### Target Encoding with Time-Based CV

**Target encoding** replaces a categorical variable with the mean (or other statistic) of the target for that category. In time series, computing this naively over the entire dataset causes severe leakage. The correct approach uses **cumulative (expanding) target encoding**: for each row at time t, the encoded value is the mean of the target for that category using only data from times strictly before t. This is equivalent to an expanding-window group mean with shift=1. Regularization via additive smoothing with a global prior (typically alpha=10 to 100) prevents extreme values for categories with few observations.

```python
def time_based_target_encoding(df, cat_col, target_col, alpha=20):
    """
    Expanding target encoding: at time t, use only data from t-1 and earlier.
    Assumes df is sorted by date.

    Parameters
    ----------
    alpha : float, smoothing parameter. Higher = more regularization toward global mean.
    """
    global_mean = df[target_col].mean()
    cumsum = df.groupby(cat_col)[target_col].cumsum() - df[target_col]
    cumcount = df.groupby(cat_col).cumcount()
    # Smoothed encoding: (cumsum + alpha * global_mean) / (cumcount + alpha)
    df[f"{cat_col}_target_enc"] = (cumsum + alpha * global_mean) / (cumcount + alpha)
    return df
```

### Difference and Ratio Features

**Difference features** capture the rate of change in the target, analogous to discrete derivatives. First-order differences (value_t - value_{t-1}) measure momentum; second-order differences (diff_t - diff_{t-1}) measure acceleration. **Ratio features** normalize the current value by a baseline such as the rolling mean: `sales_t / rolling_mean_28` produces a multiplicative deviation that is scale-invariant across series. In the M5 competition, the ratio of current sales to the rolling 28-day mean was a top-5 importance feature because it captured whether a product was selling above or below its recent baseline, independent of the absolute sales volume.

```python
def add_diff_ratio_features(df, target_col, group_col, shift=1):
    """
    Add first-order diff, second-order diff, and ratio-to-rolling-mean features.
    """
    grouped = df.groupby(group_col)[target_col]

    # First-order difference
    df[f"{target_col}_diff_1"] = grouped.diff(shift)
    # Second-order difference
    df[f"{target_col}_diff_2"] = df.groupby(group_col)[f"{target_col}_diff_1"].diff(shift)

    # Ratio to rolling mean (28-day window, shifted to prevent leakage)
    rolling_mean_28 = grouped.transform(
        lambda x: x.shift(shift).rolling(28, min_periods=1).mean()
    )
    df[f"{target_col}_ratio_roll28"] = df[target_col].shift(shift) / rolling_mean_28.replace(0, np.nan)

    # Ratio to same day last week
    df[f"{target_col}_ratio_lag7"] = df[target_col].shift(shift) / grouped.shift(7 + shift - 1).replace(0, np.nan)

    return df
```

### Complete Feature Engineering Pipeline

```python
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TimeSeriesFeatureEngineer:
    """
    Complete feature engineering pipeline for time series competitions.
    Designed for daily frequency retail/demand forecasting.
    """

    def __init__(self, target_col="sales", group_col="item_store_id",
                 date_col="date", forecast_horizon=28):
        self.target_col = target_col
        self.group_col = group_col
        self.date_col = date_col
        self.h = forecast_horizon  # minimum shift to prevent leakage

    def fit_transform(self, df):
        df = df.sort_values([self.group_col, self.date_col]).reset_index(drop=True)
        df = self._add_date_features(df)
        df = self._add_lag_features(df)
        df = self._add_rolling_features(df)
        df = self._add_expanding_features(df)
        df = self._add_diff_ratio_features(df)
        df = self._add_fourier_features(df)
        return df

    def _add_date_features(self, df):
        dt = pd.to_datetime(df[self.date_col])
        df["day_of_week"] = dt.dt.dayofweek
        df["day_of_month"] = dt.dt.day
        df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        df["month"] = dt.dt.month
        df["quarter"] = dt.dt.quarter
        df["year"] = dt.dt.year
        df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
        return df

    def _add_lag_features(self, df):
        safe_lags = [self.h, self.h + 1, self.h + 2, self.h + 6,
                     self.h + 13, self.h + 27, 56, 91, 182, 364, 365]
        safe_lags = sorted(set(l for l in safe_lags if l >= self.h))
        for lag in safe_lags:
            df[f"lag_{lag}"] = df.groupby(self.group_col)[self.target_col].shift(lag)
        return df

    def _add_rolling_features(self, df):
        windows = [7, 14, 28, 56, 91, 182]
        for w in windows:
            shifted = df.groupby(self.group_col)[self.target_col].shift(self.h)
            for stat, func_name in [("mean", "mean"), ("std", "std")]:
                col = f"roll_{stat}_{w}"
                df[col] = shifted.groupby(df[self.group_col]).transform(
                    lambda x: getattr(x.rolling(w, min_periods=1), func_name)()
                )
        return df

    def _add_expanding_features(self, df):
        shifted = df.groupby(self.group_col)[self.target_col].shift(self.h)
        df["expanding_mean"] = shifted.groupby(df[self.group_col]).transform(
            lambda x: x.expanding(min_periods=1).mean()
        )
        return df

    def _add_diff_ratio_features(self, df):
        grouped = df.groupby(self.group_col)[self.target_col]
        df["diff_h"] = grouped.diff(self.h)
        roll_mean = df.get(f"roll_mean_28")
        if roll_mean is not None:
            df["ratio_to_roll28"] = grouped.shift(self.h) / roll_mean.replace(0, np.nan)
        return df

    def _add_fourier_features(self, df):
        dt = pd.to_datetime(df[self.date_col])
        t = (dt - dt.min()).dt.days.values.astype(float)
        for k in range(1, 4):
            df[f"fourier_w_sin_{k}"] = np.sin(2 * np.pi * k * t / 7)
            df[f"fourier_w_cos_{k}"] = np.cos(2 * np.pi * k * t / 7)
        for k in range(1, 6):
            df[f"fourier_y_sin_{k}"] = np.sin(2 * np.pi * k * t / 365.25)
            df[f"fourier_y_cos_{k}"] = np.cos(2 * np.pi * k * t / 365.25)
        return df

# Usage:
# fe = TimeSeriesFeatureEngineer(target_col="sales", group_col="item_store_id",
#                                 date_col="date", forecast_horizon=28)
# df_featured = fe.fit_transform(df)
```

---

## Validation Strategies

### Walk-Forward Validation

**Walk-forward (expanding window) validation** is the gold standard for time series. The training set starts at the beginning of the data and expands forward in time; the validation set is always the next h timesteps after training. For N splits, fold i trains on data from time 0 to T_i and validates on T_i+1 to T_i+h. This mimics the actual production scenario where the model is retrained periodically on all available data. In the M5 competition, top solutions typically used 3 to 5 walk-forward folds with a 28-day validation window, matching the 28-day competition forecast horizon.

```python
from sklearn.model_selection import TimeSeriesSplit

# sklearn's TimeSeriesSplit implements walk-forward with expanding window
tscv = TimeSeriesSplit(n_splits=5, test_size=28)  # 28-day validation sets

for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
    print(f"Fold {fold}: train {len(train_idx)}, val {len(val_idx)}")
```

### Sliding Window Validation

**Sliding window validation** uses a fixed-size training window that slides forward in time, discarding old data as new data enters. This is preferable when the data-generating process is non-stationary and older data hurts performance. The training window size W is a hyperparameter: too small loses useful patterns; too large includes stale data. Typical choices are W = 180 to 730 days for daily data. In the Optiver competition, many top solutions found that using only the most recent 60 to 90 days of training data outperformed using the full history, because market microstructure features had short-lived predictive power.

```python
def sliding_window_split(dates, train_window, val_window, step):
    """
    Generate (train_indices, val_indices) for sliding window CV.

    Parameters
    ----------
    dates : pd.Series of datetime, sorted ascending.
    train_window : int, number of days in training window.
    val_window : int, number of days in validation window.
    step : int, number of days to slide between folds.
    """
    unique_dates = sorted(dates.unique())
    folds = []
    start = 0
    while start + train_window + val_window <= len(unique_dates):
        train_dates = unique_dates[start:start + train_window]
        val_dates = unique_dates[start + train_window:start + train_window + val_window]
        train_idx = dates.isin(train_dates)
        val_idx = dates.isin(val_dates)
        folds.append((np.where(train_idx)[0], np.where(val_idx)[0]))
        start += step
    return folds
```

### Purged Group Time Series Split

**Purged group time series split** introduces a gap (purge window) between the training and validation sets to prevent information leakage from features that span multiple timesteps (e.g., rolling windows, lag features computed at feature-engineering time). The purge window should be at least as large as the largest lookback window used in feature engineering. For the M5 competition with 28-day lags and 28-day rolling windows, a purge gap of 28 days was standard. For financial competitions like Jane Street, where features might encode multi-day returns, a purge of 5 to 20 trading days was typical. Additionally, an **embargo** period after the validation set prevents leakage from features in the training data that use future validation-period data in their computation.

### Embargo Period for Financial Data

In financial time series, the **embargo period** is critical because overlapping returns create label leakage between adjacent folds. If the target is a 5-day forward return, then training on day t uses information from days t through t+5 -- so the validation set must start no earlier than day t+6. Marcos Lopez de Prado's "Advances in Financial Machine Learning" formalized this as the **combinatorial purged cross-validation (CPCV)** method. The embargo is computed as `embargo_days = max(label_horizon, max_feature_lookback)`. For the Jane Street competition with what appeared to be 1-day forward returns and features with up to 20-day lookbacks, top solutions used purge=5 and embargo=5 trading days, giving a total gap of 10 days between train and validation.

### Multi-Step Ahead Validation

When the competition requires forecasting multiple steps ahead (e.g., 28 days in M5, or 10-minute buckets spanning 10 minutes in Optiver), the validation must evaluate the **full multi-step forecast** rather than single-step accuracy. This means the validation metric is computed over the entire forecast horizon as a single unit. For WRMSSE (Weighted Root Mean Squared Scaled Error) in M5, the metric weighted each series by its revenue contribution, so optimizing per-series RMSE independently would give suboptimal results. The validation loop must simulate the exact inference procedure: if recursive prediction is used, the validation must also use recursive prediction rather than teacher-forcing.

### Custom TimeSeriesSplit with Purging

```python
import numpy as np
from sklearn.model_selection import BaseCrossValidator

class PurgedTimeSeriesSplit(BaseCrossValidator):
    """
    Walk-forward CV with purge gap and embargo for financial time series.

    Parameters
    ----------
    n_splits : int, number of CV folds.
    purge_gap : int, number of timesteps to exclude between train and val.
    embargo_pct : float, fraction of val size to exclude after val end
                  from subsequent training folds (only relevant for CPCV).
    val_size : int, number of timesteps in each validation fold.
    """

    def __init__(self, n_splits=5, purge_gap=0, embargo_pct=0.0, val_size=None):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.val_size = val_size

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        groups : array-like of shape (n_samples,), temporal group indices
                 (e.g., date integers). If None, uses row indices.
        """
        n_samples = len(X)
        if groups is not None:
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
        else:
            unique_groups = np.arange(n_samples)
            n_groups = n_samples
            groups = unique_groups

        if self.val_size is None:
            val_size = n_groups // (self.n_splits + 1)
        else:
            val_size = self.val_size

        embargo_size = int(val_size * self.embargo_pct)

        for i in range(self.n_splits):
            val_end_group_idx = n_groups - (self.n_splits - 1 - i) * val_size
            val_start_group_idx = val_end_group_idx - val_size
            train_end_group_idx = val_start_group_idx - self.purge_gap

            if train_end_group_idx <= 0:
                continue

            train_groups = unique_groups[:train_end_group_idx]
            val_groups = unique_groups[val_start_group_idx:val_end_group_idx]

            train_idx = np.where(np.isin(groups, train_groups))[0]
            val_idx = np.where(np.isin(groups, val_groups))[0]

            yield train_idx, val_idx

# Usage with date-based groups:
# cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=5, embargo_pct=0.01, val_size=20)
# for train_idx, val_idx in cv.split(X, groups=df["date_int"].values):
#     model.fit(X.iloc[train_idx], y.iloc[train_idx])
#     preds = model.predict(X.iloc[val_idx])
```

---

## Winning Solution Breakdowns

### M5 Forecasting Accuracy 2020

The **M5 Forecasting Accuracy** competition on Kaggle asked participants to predict 28 days of daily unit sales for 30,490 products across 10 Walmart stores (3,049 products per store). The evaluation metric was **WRMSSE** (Weighted Root Mean Squared Scaled Error), which weights each series by its dollar sales contribution and scales the error by the historical in-sample naive forecast error.

**1st place (Monsarator, team)** used a pure **LightGBM** approach with 3 separate models for different aggregation levels. Key techniques: (1) **recursive prediction** with lag features as short as lag-1, predicting one day at a time and feeding predictions back as features; (2) **hierarchical features** encoding the mean sales at item, department, category, store, and state levels; (3) **price features** including current price, price relative to historical max/min, price momentum (change over last 1/4/12 weeks); (4) **SNAP features** (food stamp eligibility calendar per state); (5) a custom loss function approximating WRMSSE. LightGBM parameters: `num_leaves=63, learning_rate=0.03, feature_fraction=0.5, bagging_fraction=0.5, bagging_freq=1, max_bin=255, n_estimators=2000` with early stopping at patience=50. Training on ~35M rows took approximately 2 hours on a single machine.

**2nd place** combined LightGBM with a **DeepAR** model (probabilistic LSTM from Amazon). The DeepAR model used 3 LSTM layers with hidden_size=40, trained for 100 epochs with learning_rate=1e-3 on Adam. The final submission was a 0.7/0.3 weighted average of LightGBM and DeepAR predictions.

**Key takeaway**: The M5 demonstrated that meticulously engineered features on LightGBM beat deep learning on tabular time series. The winning solution did not use any neural networks.

### Optiver Realized Volatility 2021

The **Optiver Realized Volatility Prediction** competition required predicting 10-minute realized volatility for 112 stocks using order book and trade data. The target was `sqrt(sum(log_returns^2))` computed over the next 10-minute window.

**1st place** used an ensemble of **LightGBM + CatBoost + 2-layer MLP**. The critical innovation was feature engineering from raw order book snapshots: (1) **weighted average price (WAP)** computed from bid/ask prices and volumes at the top of book; (2) **log returns** of WAP over 1-second intervals; (3) **realized volatility** over trailing 2, 5, and 10-minute windows; (4) **bid-ask spread** statistics (mean, std, max over the window); (5) **order imbalance** = (bid_size - ask_size) / (bid_size + ask_size); (6) **trade statistics** including volume-weighted price, trade count, trade volume statistics. Total of approximately 300 engineered features per stock-window.

The **neural component** was a 2-layer MLP with dimensions [300, 128, 64, 1], BatchNorm, Dropout(0.3), ReLU activation, trained with Adam (lr=1e-3, weight_decay=1e-5) for 50 epochs. The MLP received both per-stock features and cross-stock features (correlation of returns with SPY proxy, sector-average volatility). The final ensemble weighted LightGBM at 0.4, CatBoost at 0.3, and MLP at 0.3.

### Jane Street Market Prediction 2021

The **Jane Street Market Prediction** competition was unique for its **online submission format**: the test set was revealed row-by-row through a Kaggle API, and models had to make predictions in real-time with a 15-minute timeout per batch. The target was a binary decision (trade / no trade) to maximize a utility function based on weighted returns.

**Top solutions** relied on: (1) **online learning** -- updating model weights or thresholds as new test data arrived, since the test set spanned a different time period than training; (2) **feature neutralization** -- residualizing predictions against dominant market factors to reduce correlation with the market beta; (3) **era-aware cross-validation** -- the data was organized into "eras" (time periods), and CV was performed by holding out entire eras with purge gaps; (4) **gradient-boosted trees** for the base model, often LightGBM with `objective='binary', num_leaves=31, learning_rate=0.05, min_child_samples=100`.

A distinctive technique was **online threshold adjustment**: the model output a continuous probability, and the trade/no-trade threshold was updated during test-time based on the running Sharpe ratio. Solutions that adapted during test time gained 5-10% improvement over static models.

### Ubiquant Market Prediction 2022

The **Ubiquant Market Prediction** competition involved 3,000+ anonymized stocks with 300 anonymized features, predicting future returns. The metric was **mean Pearson correlation** between predictions and targets across time periods.

**Top solutions** used: (1) **group-based features** -- computing per-timestep cross-sectional statistics (rank, z-score, percentile) of each feature across all stocks; (2) **purged cross-validation** with purge_gap=5 and 5 folds; (3) **LightGBM** with `num_leaves=31, learning_rate=0.01, n_estimators=5000, reg_alpha=0.1, reg_lambda=1.0, colsample_bytree=0.6, subsample=0.8`; (4) **neural networks** using a 3-layer MLP with hidden dimensions [512, 256, 128], BatchNorm, SiLU activation, Dropout(0.1), trained with AdamW (lr=1e-3, weight_decay=1e-2); (5) **blending** of LightGBM, XGBoost, CatBoost, and MLP with weights optimized by Nelder-Mead on the CV score.

### G-Research Crypto Forecasting 2022

The **G-Research Crypto Forecasting** competition required predicting 15-minute returns for 14 cryptocurrencies. The metric was weighted Pearson correlation.

**Key techniques from top solutions**: (1) **multi-asset features** -- using BTC and ETH returns as features for all altcoins, since crypto markets are highly correlated; (2) **temporal embedding** -- encoding the time-of-day and day-of-week as learned embeddings in neural models, capturing intraday volume patterns (crypto trades 24/7 but activity peaks during US/Asia market hours); (3) **volatility-adjusted targets** -- normalizing returns by recent realized volatility to stabilize the target distribution; (4) **LightGBM with dart boosting** (`boosting_type='dart', drop_rate=0.1, max_drop=50`) which provided better generalization than standard GBDT on this noisy data; (5) **feature selection via null importance** -- training models on shuffled targets and keeping only features whose actual importance exceeded their null importance at the 95th percentile.

### Walmart Sales Forecasting

The **Walmart Recruiting - Store Sales Forecasting** competition (an older Kaggle classic) required predicting weekly sales for 45 stores across 99 departments. The evaluation metric was **weighted MAE** with 5x weight on holiday weeks.

**Winning approaches** combined: (1) **ARIMA per series** -- fitting auto.arima (via R's forecast package or Python's pmdarima with `seasonal=True, m=52, stepwise=True, max_p=3, max_q=3`) individually to each of the 4,455 store-department combinations; (2) **LightGBM with holiday features** -- binary indicators for Super Bowl, Labor Day, Thanksgiving, Christmas, plus `weeks_to_holiday` and `weeks_since_holiday` counters; (3) **hybrid ensemble** -- using ARIMA predictions as a feature in the LightGBM model, which allowed the tree model to learn when to trust the statistical forecast and when to override it based on contextual features; (4) **markdown features** -- Walmart provided promotional markdown data which were strong predictors during holiday periods.

---

## GBDT Approaches

### LightGBM for Time Series

**LightGBM** dominates time series competitions for several reasons: it handles the large feature spaces created by lag/rolling features efficiently; its leaf-wise growth strategy captures non-linear temporal patterns; it natively supports categorical features without one-hot encoding; and it trains orders of magnitude faster than neural networks. The recommended parameter configuration for time series differs from tabular classification: lower learning rates (0.01-0.05), more leaves (63-255), feature subsampling to decorrelate trees across temporal features, and careful early stopping on a time-ordered validation set.

```python
import lightgbm as lgb

LIGHTGBM_TS_PARAMS = {
    "objective": "regression",       # or "tweedie" for count data with zeros
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 127,               # higher for complex temporal patterns
    "learning_rate": 0.03,
    "feature_fraction": 0.5,         # column subsampling per tree
    "bagging_fraction": 0.66,        # row subsampling per tree
    "bagging_freq": 1,
    "min_child_samples": 50,         # prevent overfitting to rare events
    "max_bin": 255,
    "lambda_l1": 0.1,               # L1 regularization
    "lambda_l2": 1.0,               # L2 regularization
    "min_gain_to_split": 0.01,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

# For count data with many zeros (e.g., M5 daily sales):
TWEEDIE_PARAMS = {
    **LIGHTGBM_TS_PARAMS,
    "objective": "tweedie",
    "tweedie_variance_power": 1.1,  # between 1 (Poisson) and 2 (Gamma)
    "metric": "custom",              # use competition metric
}
```

### Feature Importance Selection

With hundreds of lag, rolling, calendar, and interaction features, **feature selection** is critical to prevent overfitting and reduce training time. The standard pipeline: (1) train a LightGBM model with all features; (2) extract `feature_importances_` (gain-based or split-based); (3) keep the top-K features (typically K=100-300); (4) optionally run **null importance testing** -- shuffle the target 5-10 times, train models, and keep only features whose actual importance exceeds the 95th percentile of their null importance distribution.

```python
def select_features_by_importance(model, feature_names, top_k=200):
    """
    Select top-K features by LightGBM gain importance.
    """
    importance = model.feature_importance(importance_type="gain")
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=False)

    selected = importance_df.head(top_k)["feature"].tolist()
    print(f"Selected {len(selected)} features. "
          f"Top 10: {importance_df.head(10)['feature'].tolist()}")
    return selected

def null_importance_selection(X, y, model_params, n_shuffles=5, percentile=95):
    """
    Null importance test: keep features whose actual importance exceeds
    the 95th percentile of their shuffled-target importance distribution.
    """
    # Actual importance
    model = lgb.LGBMRegressor(**model_params)
    model.fit(X, y)
    actual_imp = pd.Series(model.feature_importance(importance_type="gain"),
                           index=X.columns)
    # Null importances
    null_imps = pd.DataFrame(index=X.columns)
    for i in range(n_shuffles):
        y_shuffled = y.sample(frac=1.0, random_state=i).values
        model.fit(X, y_shuffled)
        null_imps[f"shuffle_{i}"] = model.feature_importance(importance_type="gain")

    threshold = null_imps.quantile(percentile / 100.0, axis=1)
    selected = actual_imp[actual_imp > threshold].index.tolist()
    return selected
```

### Multi-Step Forecasting Strategies

Three strategies exist for predicting multiple future timesteps:

**Recursive (iterated) forecasting**: Train a single model to predict one step ahead. At inference, predict step t+1, append it to the feature set, then predict t+2, and so on. Advantages: captures short-term autoregressive dynamics; only one model to train. Disadvantages: error accumulation -- prediction errors in early steps propagate to later steps. The M5 1st-place solution used recursive prediction and still won, suggesting that with high-quality lag features, error accumulation is manageable over 28 steps.

**Direct forecasting**: Train H separate models, one for each forecast horizon. Model_h predicts y_{t+h} using features available at time t. Advantages: no error accumulation; each model specializes in its horizon. Disadvantages: H models to train and store; ignores the dependency structure between horizons. This is the safer default for competitions.

**Multi-output (MIMO) forecasting**: Train a single model that outputs all H steps simultaneously. For LightGBM, this requires training H separate LightGBM models (one per horizon) since LightGBM does not natively support multi-output. For neural networks, the output layer has H neurons. Advantages: captures inter-horizon dependencies. Disadvantages: harder to train; requires more data.

```python
def train_direct_models(X_train, y_train_dict, X_val, y_val_dict, params, horizons):
    """
    Train one LightGBM model per forecast horizon (direct strategy).

    Parameters
    ----------
    y_train_dict : dict[int, pd.Series], mapping horizon h -> target at t+h.
    horizons : list[int], e.g., [1, 2, ..., 28].
    """
    models = {}
    for h in horizons:
        dtrain = lgb.Dataset(X_train, label=y_train_dict[h])
        dval = lgb.Dataset(X_val, label=y_val_dict[h])
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        models[h] = model
        print(f"Horizon {h}: best iteration {model.best_iteration}, "
              f"best score {model.best_score['valid_0'][params['metric']]:.6f}")
    return models
```

### Quantile Regression for Prediction Intervals

**Quantile regression** produces prediction intervals rather than point estimates. LightGBM supports this via `objective='quantile'` with `alpha` specifying the quantile level. Training three models (alpha=0.1, 0.5, 0.9) gives the 10th percentile, median, and 90th percentile, forming an 80% prediction interval. The M5 companion competition (M5 Uncertainty) required participants to submit quantile forecasts at the 0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995 levels.

```python
def train_quantile_models(X_train, y_train, X_val, y_val, base_params,
                          quantiles=(0.1, 0.5, 0.9)):
    """
    Train separate LightGBM models for each quantile.
    """
    models = {}
    for q in quantiles:
        params = {**base_params, "objective": "quantile", "alpha": q, "metric": "quantile"}
        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)
        model = lgb.train(params, dtrain, num_boost_round=1500,
                          valid_sets=[dval],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)])
        models[q] = model
    return models

# Inference:
# predictions = {q: models[q].predict(X_test) for q in quantiles}
```

### LightGBM Time Series Pipeline

```python
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

def lightgbm_ts_pipeline(df, target_col, group_col, date_col,
                          feature_cols, n_folds=3, forecast_horizon=28):
    """
    End-to-end LightGBM pipeline for time series competition.
    Uses walk-forward validation with purge gap.
    """
    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "num_leaves": 127,
        "learning_rate": 0.03,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.66,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
        "n_jobs": -1,
    }

    dates = sorted(df[date_col].unique())
    val_size = forecast_horizon
    purge_gap = forecast_horizon  # gap = forecast horizon to prevent leakage

    oof_preds = np.full(len(df), np.nan)
    models = []
    scores = []

    for fold in range(n_folds):
        val_end_idx = len(dates) - fold * val_size
        val_start_idx = val_end_idx - val_size
        train_end_idx = val_start_idx - purge_gap

        if train_end_idx <= 0:
            break

        train_dates = dates[:train_end_idx]
        val_dates = dates[val_start_idx:val_end_idx]

        train_mask = df[date_col].isin(train_dates)
        val_mask = df[date_col].isin(val_dates)

        X_train = df.loc[train_mask, feature_cols]
        y_train = df.loc[train_mask, target_col]
        X_val = df.loc[val_mask, feature_cols]
        y_val = df.loc[val_mask, target_col]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val)

        model = lgb.train(
            params, dtrain, num_boost_round=3000,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
        )

        preds = model.predict(X_val)
        preds = np.clip(preds, 0, None)  # sales cannot be negative

        rmse = np.sqrt(mean_squared_error(y_val, preds))
        scores.append(rmse)
        models.append(model)
        oof_preds[val_mask.values] = preds

        print(f"Fold {fold}: RMSE = {rmse:.4f}, best_iter = {model.best_iteration}")

    print(f"\nMean CV RMSE: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")
    return models, oof_preds, scores
```

---

## Neural Approaches

### N-BEATS

**N-BEATS** (Neural Basis Expansion Analysis for Interpretable Time Series Forecasting) is a pure deep learning architecture for univariate forecasting proposed by Oreshkin et al. (2020). It consists of a stack of **fully connected blocks**, each with a **backcast** (reconstruction of the input) and a **forecast** branch. The architecture uses no recurrence, no convolution, and no attention -- only feedforward layers with ReLU activation and residual connections between blocks. The input is a lookback window of length L, and the output is a forecast of length H.

Key architectural details: each block has 4 FC layers of width 256 or 512 with ReLU, followed by two linear projection heads (one for backcast of dimension L, one for forecast of dimension H). Blocks are organized into **stacks** (typically 2 stacks of 3 blocks each). In the **interpretable** variant, the basis functions are constrained: the trend stack uses polynomial basis (degree 2-3), and the seasonality stack uses Fourier basis (harmonics up to H/2). In the **generic** variant, the basis is learned. N-BEATS won the M4 competition with an ensemble of generic and interpretable variants.

Training parameters: batch_size=1024, learning_rate=1e-3 with Adam, L=5*H (lookback = 5x forecast horizon), 100 epochs with early stopping, ensemble of 10 models with different random initializations.

### Temporal Fusion Transformer

The **Temporal Fusion Transformer (TFT)** by Lim et al. (2021) is a multi-horizon forecasting model that handles static covariates, known future inputs (e.g., holidays), and observed past inputs. It uses **Variable Selection Networks** (VSN) to learn which features matter, **Gated Residual Networks** (GRN) for nonlinear processing, a **seq2seq LSTM encoder-decoder** for temporal processing, and **multi-head attention** over the encoder outputs for long-range dependencies. The output is a set of quantile forecasts (e.g., 10th, 50th, 90th percentiles).

Architecture details: `hidden_size=128, lstm_layers=2, num_attention_heads=4, dropout=0.1`. The VSN produces per-timestep feature importance weights, making the model interpretable. TFT is available in PyTorch Forecasting (`pytorch_forecasting.models.TemporalFusionTransformer`) and Darts (`darts.models.TFTModel`). TFT excels on datasets with rich metadata (many known future covariates and static features) but is slower to train than N-BEATS and requires more hyperparameter tuning.

### PatchTST

**PatchTST** (Patch Time Series Transformer, Nie et al., 2023) applies the Vision Transformer (ViT) patching strategy to time series. Instead of feeding individual timesteps, the input series is divided into non-overlapping **patches** of length P (typically P=16 or 24), and each patch is linearly projected into a D-dimensional embedding. A standard Transformer encoder (multi-head self-attention + feedforward) processes the sequence of patch embeddings. Key design choice: **channel independence** -- each variate (channel) is processed independently through the same Transformer, then the outputs are concatenated. This dramatically reduces the parameter count and improves generalization.

Architecture: `patch_len=16, stride=8 (overlapping patches), d_model=128, n_heads=16, d_ff=256, n_layers=3, dropout=0.2`. PatchTST achieved state-of-the-art on long-term forecasting benchmarks (ETTh1, ETTh2, ETTm1, Weather, Electricity) and is competitive with LightGBM when sufficient training data is available (10,000+ timesteps per series).

### DeepAR

**DeepAR** (Salinas et al., 2020, Amazon) is a probabilistic forecasting model based on autoregressive LSTMs. It models the conditional distribution of the next timestep given history using a parametric likelihood (Gaussian, Negative Binomial, or Student-t). At each timestep, the LSTM hidden state is mapped to the distribution parameters (e.g., mu and sigma for Gaussian), and samples are drawn autoregressively during prediction. DeepAR handles multiple related time series by learning a shared model with per-series scaling.

Architecture: `num_layers=3, hidden_size=40, cell_type='lstm', dropout_rate=0.1, embedding_dimension=[50, 50]` for categorical features. Training: Adam with lr=1e-3, batch_size=32, epochs=100. DeepAR is available in GluonTS (`gluonts.mx.model.deepar.DeepAREstimator`) and PyTorch Forecasting. The 2nd-place M5 solution used DeepAR as part of a LightGBM+DeepAR ensemble, with DeepAR contributing approximately 30% of the blend weight.

### WaveNet-Style Dilated Convolutions

**Dilated causal convolutions** (inspired by WaveNet, van den Oord et al., 2016) use exponentially increasing dilation rates to capture long-range dependencies without recurrence. A stack of 1D convolutions with dilation rates [1, 2, 4, 8, 16, 32, 64] and kernel_size=2 gives a receptive field of 128 timesteps with only 7 layers. Combined with **residual connections** and **gated activation** (tanh * sigmoid), this architecture trains faster than LSTMs and parallelizes better.

```python
import torch
import torch.nn as nn

class DilatedCausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size,
            dilation=dilation, padding=dilation  # causal padding
        )
        self.gate = nn.Conv1d(
            in_channels, 2 * out_channels, kernel_size,
            dilation=dilation, padding=dilation
        )
        self.residual = nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, x):
        # x: (batch, channels, time)
        h = self.conv(x)[:, :, :x.size(2)]  # trim for causal
        h_filter, h_gate = h.chunk(2, dim=1)
        h = torch.tanh(h_filter) * torch.sigmoid(h_gate)
        residual = self.residual(h) + x
        return residual, h
```

### LSTM and GRU Baselines

A simple **LSTM or GRU baseline** remains a strong neural starting point. The architecture: an embedding layer for categorical features (item_id, store_id), concatenated with numerical features, fed into a 2-layer LSTM (hidden_size=128, dropout=0.2), followed by a linear head projecting to H outputs (one per forecast horizon). Training with Adam (lr=1e-3), MSE loss, batch_size=128, and 50 epochs with early stopping typically achieves 85-90% of the best neural model's performance while being 5-10x faster to train and tune.

```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, n_layers=2,
                 forecast_horizon=28, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden_dim, forecast_horizon)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)       # (batch, seq_len, hidden_dim)
        last_hidden = lstm_out[:, -1, :] # (batch, hidden_dim)
        out = self.head(self.dropout(last_hidden))  # (batch, forecast_horizon)
        return out
```

### Neural Forecasting Code

```python
# Using neuralforecast library (Nixtla) for N-BEATS and TFT
# pip install neuralforecast

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, TFT, PatchTST
from neuralforecast.losses.pytorch import MAE, QuantileLoss
import pandas as pd

# Data must be in long format: unique_id | ds | y
# unique_id: series identifier, ds: datetime, y: target value

# N-BEATS configuration
nbeats_model = NBEATS(
    h=28,                          # forecast horizon
    input_size=5 * 28,             # lookback = 5 * horizon
    stack_types=["trend", "seasonality", "identity"],
    n_blocks=[3, 3, 1],
    mlp_units=[[512, 512], [512, 512], [512, 512]],
    n_harmonics=5,                 # for seasonality stack
    n_polynomials=3,               # for trend stack
    learning_rate=1e-3,
    loss=MAE(),
    max_steps=1000,
    batch_size=1024,
    scaler_type="standard",
    random_seed=42,
)

# TFT configuration
tft_model = TFT(
    h=28,
    input_size=5 * 28,
    hidden_size=128,
    n_head=4,
    learning_rate=1e-3,
    loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
    max_steps=1000,
    batch_size=256,
    scaler_type="standard",
    stat_exog_list=["store_id_enc", "item_category_enc"],  # static features
    futr_exog_list=["day_of_week", "is_holiday"],           # known future
    random_seed=42,
)

# PatchTST configuration
patchtst_model = PatchTST(
    h=28,
    input_size=512,                # longer lookback for transformer
    patch_len=16,
    stride=8,
    hidden_size=128,
    n_heads=16,
    learning_rate=1e-4,
    loss=MAE(),
    max_steps=1000,
    batch_size=128,
    scaler_type="standard",
    random_seed=42,
)

# Train and predict
nf = NeuralForecast(
    models=[nbeats_model, tft_model, patchtst_model],
    freq="D"  # daily frequency
)
nf.fit(df=train_df)
forecasts = nf.predict()
# forecasts has columns: unique_id, ds, NBEATS, TFT, PatchTST

# Ensemble: simple average
forecasts["ensemble"] = (
    forecasts["NBEATS"] + forecasts["TFT"] + forecasts["PatchTST"]
) / 3.0
```

---

## Statistical Baselines

### ARIMA and SARIMA

**ARIMA(p,d,q)** models the time series as a linear combination of p autoregressive terms, d differences (to achieve stationarity), and q moving average terms. **SARIMA(p,d,q)(P,D,Q,m)** extends this with seasonal components at period m. Auto-selection via AIC/BIC (implemented in `pmdarima.auto_arima` or `statsforecast.models.AutoARIMA`) fits multiple (p,d,q) combinations and selects the best. Typical settings: `max_p=5, max_q=5, max_d=2, seasonal=True, m=7` for weekly seasonality in daily data; `m=12` for monthly data with annual seasonality.

ARIMA's strength is on **short, univariate series** with clear linear structure. On the M3 competition (3,003 monthly/quarterly/yearly series), ARIMA variants outperformed early neural methods. However, on large-scale competitions like M5 (30,490 series), fitting individual ARIMA models is computationally expensive and typically underperforms LightGBM by 10-20% on the competition metric.

```python
import pmdarima as pm

# Auto ARIMA for a single series
model = pm.auto_arima(
    y_train,
    seasonal=True,
    m=7,                        # weekly seasonality for daily data
    max_p=5, max_q=5, max_d=2,
    max_P=2, max_Q=2, max_D=1,
    stepwise=True,              # faster search (not exhaustive)
    suppress_warnings=True,
    error_action="ignore",
    information_criterion="aic",
    trace=False,
)
print(model.summary())
forecast = model.predict(n_periods=28)
```

### Prophet

**Prophet** (Taylor and Letham, 2018, Meta/Facebook) decomposes a time series into trend + seasonality + holidays + residual. It is an additive regression model where the trend is a piecewise linear or logistic growth curve with automatic changepoint detection, seasonality is modeled with Fourier series (10 terms for annual, 3 for weekly by default), and holidays are specified as a DataFrame. Prophet is designed for **business time series** with strong seasonal effects, missing data, and holiday effects.

Key parameters: `changepoint_prior_scale=0.05` (flexibility of trend; increase to 0.5 for more flexible trends), `seasonality_prior_scale=10.0` (regularization on Fourier coefficients), `holidays_prior_scale=10.0`, `seasonality_mode='additive'` or `'multiplicative'`. Prophet fits in seconds per series using Stan (MAP estimation by default) and handles missing data gracefully. It is an excellent **baseline model** for competitions but rarely wins alone -- top teams use Prophet predictions as features in a LightGBM model.

```python
from prophet import Prophet

m = Prophet(
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    holidays_prior_scale=10.0,
    seasonality_mode="multiplicative",  # for series with growing amplitude
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
)
# Add country holidays
m.add_country_holidays(country_name="US")
m.fit(train_df)  # train_df must have columns 'ds' and 'y'
future = m.make_future_dataframe(periods=28)
forecast = m.predict(future)
# forecast has columns: ds, yhat, yhat_lower, yhat_upper, trend, weekly, yearly
```

### Exponential Smoothing ETS

**Exponential Smoothing (ETS)** models decompose a series into Error, Trend, and Seasonal components, each of which can be None, Additive, or Multiplicative. The standard taxonomy yields models like ETS(A,A,A) = additive error, additive trend, additive seasonality (equivalent to Holt-Winters additive). The **damped trend** variant (ETS(A,Ad,A)) adds a damping parameter phi (typically 0.8-0.98) that prevents the trend from extrapolating indefinitely, which consistently improves forecast accuracy on long horizons. In the M4 competition, the damped exponential smoothing model was one of the strongest pure statistical baselines, outperforming ARIMA on 60% of the series.

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

model = ExponentialSmoothing(
    y_train,
    trend="add",
    seasonal="mul",
    seasonal_periods=7,
    damped_trend=True,
    initialization_method="estimated",
)
fitted = model.fit(optimized=True)
forecast = fitted.forecast(28)
```

### Theta Method

The **Theta method** (Assimakopoulos and Nikolopoulos, 2000) decomposes the series into two "theta lines" by applying different amounts of curvature modification. The original series is decomposed into a linear trend (theta=0) and an amplified version (theta=2). The trend line is extrapolated linearly, while the amplified version is forecast with Simple Exponential Smoothing (SES). The final forecast is the average of the two. Despite its simplicity, the Theta method won the M3 competition and remains a strong baseline. It is available in `statsforecast.models.Theta` and `statsmodels.tsa.forecasting.theta.ThetaModel`.

### When Statistical Methods Beat ML

Statistical methods outperform ML (LightGBM, neural networks) in several scenarios: (1) **short series** with fewer than 100 observations, where feature engineering produces mostly NaN values and ML models overfit; (2) **low-frequency data** (monthly, quarterly) where the total sample size is small; (3) **simple patterns** with dominant linear trend and regular seasonality, where ARIMA or ETS capture the data-generating process exactly; (4) **many independent series** with no shared structure, where global ML models cannot leverage cross-series patterns; (5) **extreme forecast horizons** (>2x the series length), where statistical models' parametric extrapolation is more stable than ML models' pattern-matching.

The **M4 competition** (100,000 series at multiple frequencies) showed that statistical ensembles (combining ETS, ARIMA, Theta) achieved MASE of approximately 11.4, while pure neural methods scored worse at approximately 13.0. However, the winning solution (ES-RNN by Smyl) hybridized exponential smoothing with an RNN, achieving MASE of 9.4 -- suggesting that the optimal approach combines statistical and ML components.

### Quick Baseline with StatsForecast

```python
# pip install statsforecast
from statsforecast import StatsForecast
from statsforecast.models import (
    AutoARIMA,
    AutoETS,
    AutoTheta,
    SeasonalNaive,
    WindowAverage,
    SeasonalWindowAverage,
)

# Data format: unique_id | ds | y (long format, same as neuralforecast)
models = [
    AutoARIMA(season_length=7),
    AutoETS(season_length=7),
    AutoTheta(season_length=7),
    SeasonalNaive(season_length=7),       # repeat last week
    WindowAverage(window_size=28),         # 28-day moving average
    SeasonalWindowAverage(
        season_length=7, window_size=4     # average of last 4 same-weekdays
    ),
]

sf = StatsForecast(
    models=models,
    freq="D",
    n_jobs=-1,
)
sf.fit(df=train_df)
forecasts = sf.predict(h=28)
# forecasts has columns: unique_id, ds, AutoARIMA, AutoETS, AutoTheta,
#                         SeasonalNaive, WindowAverage, SeasWA

# Ensemble baseline: simple average of all models
model_cols = [c for c in forecasts.columns if c not in ["unique_id", "ds"]]
forecasts["ensemble"] = forecasts[model_cols].mean(axis=1)
```

---

## Hierarchical Forecasting

### Aggregation Approaches

**Hierarchical forecasting** arises when the target variable can be aggregated at multiple levels. In the M5 competition, the hierarchy had 12 levels: individual item-store (30,490 series), item (3,049), department-store (70), category-store (30), store (10), department (7), category (3), state-item (9,147), state-department (21), state-category (9), state (3), and total (1). Forecasts must be **coherent** -- the sum of item-level forecasts for a store should equal the store-level forecast.

Three classical approaches: (1) **Bottom-up**: forecast at the most granular level and aggregate upward. Advantage: captures item-level patterns. Disadvantage: noisy at the bottom level, errors accumulate. (2) **Top-down**: forecast at the aggregate level and disaggregate downward using historical proportions. Advantage: aggregate series are smoother and easier to forecast. Disadvantage: loses item-level dynamics. (3) **Middle-out**: forecast at an intermediate level (e.g., department-store), aggregate upward for higher levels, disaggregate downward for lower levels. This is often the pragmatic choice when the intermediate level has enough data for reliable forecasts but captures meaningful variation.

### Reconciliation Methods

**Forecast reconciliation** produces coherent forecasts from independently generated base forecasts at each level. The idea: generate "base" forecasts independently at each level of the hierarchy, then adjust them to ensure coherence using an optimization procedure.

**OLS reconciliation** (Hyndman et al., 2011) minimizes the sum of squared adjustments subject to the coherence constraint. **MinTrace reconciliation** (Wickramanayake et al., 2019) minimizes the trace of the forecast error covariance matrix and has variants: `mint_shrink` (shrinkage estimator for the covariance), `ols` (identity covariance), `wls_struct` (diagonal covariance proportional to the number of series below each node), `wls_var` (diagonal covariance from base forecast variances). **MinTrace with shrinkage** (`mint_shrink`) is generally the best-performing reconciliation method across competitions and benchmarks.

### M5-Style Hierarchy

The M5 hierarchy structure in descending order of aggregation:

- **Level 1**: Total (1 series) -- all sales across all stores
- **Level 2**: State (3 series) -- CA, TX, WI
- **Level 3**: Store (10 series) -- CA_1 through CA_4, TX_1 through TX_3, WI_1 through WI_3
- **Level 4**: Category (3 series) -- HOBBIES, HOUSEHOLD, FOODS
- **Level 5**: Department (7 series) -- HOBBIES_1, HOBBIES_2, HOUSEHOLD_1, HOUSEHOLD_2, FOODS_1, FOODS_2, FOODS_3
- **Level 6-12**: Cross-products of state/store/category/department/item

The WRMSSE metric weighted each of the 42,840 series (across 12 levels) by its dollar sales contribution. This meant that high-revenue items in the FOODS_3 department (which included staples like milk, eggs, bread) had disproportionate influence. Top solutions tailored their models to over-index on high-WRMSSE-weight series, sometimes training separate specialized models for the top-100 series by weight.

### Hierarchical Reconciliation Code

```python
# pip install hierarchicalforecast
from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import (
    BottomUp,
    TopDown,
    MiddleOut,
    MinTrace,
    OptimalCombination,
)
import pandas as pd
import numpy as np

def build_m5_hierarchy_tags(df):
    """
    Build hierarchy specification for M5-style data.
    Returns a dict mapping hierarchy level names to arrays of group labels.
    """
    tags = {
        "total": np.array(["total"] * len(df)),
        "state": df["state_id"].values,
        "store": df["store_id"].values,
        "category": df["cat_id"].values,
        "department": df["dept_id"].values,
        "item": df["item_id"].values,
        "state_category": (df["state_id"] + "_" + df["cat_id"]).values,
        "store_department": (df["store_id"] + "_" + df["dept_id"]).values,
        "item_store": (df["item_id"] + "_" + df["store_id"]).values,
    }
    return tags

# Reconciliation with MinTrace (shrinkage)
def reconcile_forecasts(base_forecasts, S_matrix, method="mint_shrink"):
    """
    Reconcile base forecasts to ensure hierarchical coherence.

    Parameters
    ----------
    base_forecasts : pd.DataFrame with columns [unique_id, ds, yhat]
    S_matrix : np.ndarray, summing matrix defining the hierarchy.
    method : str, reconciliation method.
    """
    reconciler = HierarchicalReconciliation(
        reconcilers=[
            MinTrace(method="mint_shrink"),  # or "ols", "wls_struct"
        ]
    )
    # reconciler.reconcile expects base forecasts and the hierarchy spec
    reconciled = reconciler.reconcile(
        Y_hat_df=base_forecasts,
        S=S_matrix,
        tags=hierarchy_tags,
    )
    return reconciled

# Manual bottom-up reconciliation (simpler approach)
def bottom_up_reconcile(item_store_forecasts, hierarchy_map):
    """
    Aggregate item-store level forecasts to all higher levels.

    Parameters
    ----------
    item_store_forecasts : pd.DataFrame with columns
        [item_id, store_id, date, yhat]
    hierarchy_map : pd.DataFrame mapping item_id to dept_id, cat_id, state_id.
    """
    df = item_store_forecasts.merge(hierarchy_map, on="item_id")

    # Store level
    store_fc = df.groupby(["store_id", "date"])["yhat"].sum().reset_index()
    # Department level
    dept_fc = df.groupby(["dept_id", "date"])["yhat"].sum().reset_index()
    # Category level
    cat_fc = df.groupby(["cat_id", "date"])["yhat"].sum().reset_index()
    # State level
    state_fc = df.groupby(["state_id", "date"])["yhat"].sum().reset_index()
    # Total level
    total_fc = df.groupby("date")["yhat"].sum().reset_index()
    total_fc["level"] = "total"

    return {
        "item_store": item_store_forecasts,
        "store": store_fc,
        "department": dept_fc,
        "category": cat_fc,
        "state": state_fc,
        "total": total_fc,
    }
```

---

## Financial Time Series Specifics

### Feature Neutralization

**Feature neutralization** removes the linear component of predictions explained by common market factors, leaving only the idiosyncratic (stock-specific) signal. This technique was essential in the Jane Street, Numerai, and Ubiquant competitions. The procedure: at each timestep, regress the raw prediction vector (across all stocks) on a set of factor exposures (market beta, sector, size, momentum), then use the residual as the neutralized prediction. This reduces correlation with common factors that tend to be crowded and unstable.

Mathematically: given predictions `p` (n_stocks x 1) and factor exposures `F` (n_stocks x k), the neutralized predictions are `p_neutral = p - F @ (F^T F)^{-1} F^T @ p`. This is equivalent to projecting p onto the null space of F.

### Era-Based Cross-Validation

In financial competitions, data is organized into **eras** (non-overlapping time periods, typically 1 week or 1 month). Era-based CV holds out entire eras rather than individual rows, because rows within an era share market conditions and are not independent. The standard protocol from Numerai: split eras into 5 contiguous groups, use each group as validation in turn (5-fold walk-forward on eras), with a purge gap of 2-4 eras between train and validation. The CV metric is the **mean correlation across eras**, not the overall correlation -- this prevents high-volatility eras from dominating the metric.

### Sharpe Ratio Optimization

The **Sharpe ratio** (mean return / std of returns) is the standard risk-adjusted performance metric in finance. Directly optimizing Sharpe is non-trivial because it is not decomposable across samples. Approximations used in competitions: (1) optimize mean correlation (proxy for returns) while monitoring correlation stability (proxy for Sharpe); (2) use a differentiable Sharpe loss `L = -mean(r) / std(r)` where r = prediction * target, which can be backpropagated through a neural network; (3) post-hoc, scale predictions to maximize in-sample Sharpe and use portfolio optimization (mean-variance with constraints) to allocate capital.

### Online Learning and Incremental Updates

**Online learning** updates model parameters as new data arrives during the test phase. In the Jane Street competition (where test data arrived row-by-row through an API), top solutions implemented: (1) **running statistics updates** -- maintaining exponential moving averages of feature means and variances to normalize new data; (2) **threshold adaptation** -- adjusting the trade/no-trade threshold based on recent model accuracy; (3) **incremental gradient updates** for linear models or shallow neural networks using SGD with a small learning rate (1e-5 to 1e-4) on each new batch.

```python
class OnlineNormalizer:
    """
    Exponential moving average normalizer for online/streaming data.
    """
    def __init__(self, n_features, decay=0.99):
        self.decay = decay
        self.mean = np.zeros(n_features)
        self.var = np.ones(n_features)
        self.initialized = False

    def update_and_transform(self, X):
        """
        Update running statistics and return normalized data.
        X: np.ndarray of shape (batch_size, n_features)
        """
        batch_mean = X.mean(axis=0)
        batch_var = X.var(axis=0)

        if not self.initialized:
            self.mean = batch_mean
            self.var = batch_var
            self.initialized = True
        else:
            self.mean = self.decay * self.mean + (1 - self.decay) * batch_mean
            self.var = self.decay * self.var + (1 - self.decay) * batch_var

        X_normalized = (X - self.mean) / np.sqrt(self.var + 1e-8)
        return X_normalized
```

### Dealing with Regime Changes

**Regime changes** (structural breaks in the data-generating process) are the primary challenge in financial forecasting. Techniques to handle them: (1) **regime detection** using Hidden Markov Models (HMM with 2-3 states for bull/bear/sideways) or change-point detection algorithms (e.g., PELT, BinSeg from the `ruptures` library); (2) **adaptive training windows** -- shortening the training window during detected regime changes; (3) **regime-conditional models** -- training separate models for each regime and switching between them; (4) **robust feature engineering** -- using rank-transformed or z-scored features that are invariant to scale shifts; (5) **ensemble diversity** -- combining models trained on different time windows (short/medium/long) to hedge against regime dependence.

```python
import ruptures

def detect_regimes(series, model="rbf", min_size=20, penalty=10):
    """
    Detect regime change points using the PELT algorithm.

    Parameters
    ----------
    series : np.ndarray, 1D time series.
    model : str, cost function ('rbf', 'l2', 'l1', 'normal').
    min_size : int, minimum segment length.
    penalty : float, penalty for adding a changepoint (higher = fewer breaks).

    Returns
    -------
    breakpoints : list[int], indices of detected change points.
    """
    algo = ruptures.Pelt(model=model, min_size=min_size).fit(series)
    breakpoints = algo.predict(pen=penalty)
    return breakpoints  # last element is len(series)
```

### Feature Neutralization Code

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def neutralize_predictions(predictions, factor_exposures, proportion=1.0):
    """
    Neutralize predictions against factor exposures at each timestep.

    Parameters
    ----------
    predictions : pd.Series, raw model predictions indexed by (era, stock_id).
    factor_exposures : pd.DataFrame, columns are factor names,
                       indexed matching predictions.
    proportion : float in [0,1], how much to neutralize.
                 1.0 = full neutralization, 0.0 = no change.

    Returns
    -------
    pd.Series, neutralized predictions.
    """
    exposures = factor_exposures.values
    # Project predictions onto factor space
    reg = LinearRegression(fit_intercept=True)
    reg.fit(exposures, predictions.values)
    factor_component = reg.predict(exposures)
    neutralized = predictions.values - proportion * factor_component
    return pd.Series(neutralized, index=predictions.index)

def cross_sectional_neutralize(df, pred_col, factor_cols, era_col, proportion=1.0):
    """
    Neutralize predictions within each era (cross-sectional).
    Standard approach for Numerai / Jane Street / Ubiquant.

    Parameters
    ----------
    df : pd.DataFrame with columns [era_col, pred_col] + factor_cols.
    """
    neutralized = df.copy()
    for era, group in df.groupby(era_col):
        if len(group) < 10:
            continue
        preds = group[pred_col].values
        factors = group[factor_cols].values

        # Handle NaN in factors
        mask = ~np.any(np.isnan(factors), axis=1) & ~np.isnan(preds)
        if mask.sum() < 10:
            continue

        reg = LinearRegression(fit_intercept=True)
        reg.fit(factors[mask], preds[mask])
        factor_component = np.full_like(preds, np.nan)
        factor_component[mask] = reg.predict(factors[mask])
        neutralized.loc[group.index, pred_col] = preds - proportion * factor_component

    return neutralized

# Example usage for Numerai-style competition:
# df["prediction_neutralized"] = cross_sectional_neutralize(
#     df, pred_col="prediction", factor_cols=["feature_1", "feature_2", "feature_3"],
#     era_col="era", proportion=0.5
# )["prediction"]

def rank_normalize(series):
    """
    Rank-normalize predictions to uniform [0, 1] distribution.
    Standard post-processing for financial competitions.
    """
    return series.rank(pct=True)

def era_correlation(df, pred_col, target_col, era_col):
    """
    Compute per-era Pearson correlation and summary statistics.
    The primary evaluation metric for Numerai-style competitions.
    """
    corrs = df.groupby(era_col).apply(
        lambda g: g[pred_col].corr(g[target_col])
    )
    print(f"Mean era correlation: {corrs.mean():.4f}")
    print(f"Std era correlation:  {corrs.std():.4f}")
    print(f"Sharpe (corr):        {corrs.mean() / corrs.std():.4f}")
    print(f"Max drawdown:         {corrs.cumsum().cummax().sub(corrs.cumsum()).max():.4f}")
    return corrs
```

---

## Post-Processing

### Clipping Predictions

**Clipping** constrains predictions to the valid range of the target variable. For count data (unit sales), predictions must be non-negative: `np.clip(preds, 0, None)`. For bounded targets (e.g., probabilities in [0,1], or percentage changes in [-100, inf]), clip to the domain. For the M5 competition, clipping negative LightGBM predictions to 0 improved the WRMSSE by approximately 0.5% because the Tweedie loss occasionally produced slightly negative predictions for zero-dominated series. For financial return predictions, clipping to the 1st and 99th percentile of the training target distribution prevents extreme predictions that could destabilize portfolio optimization.

### Rounding for Count Data

When the target is integer-valued (unit sales, event counts), **rounding** predictions can improve metrics that penalize fractional predictions. Strategies: (1) simple `np.round(preds)` for symmetric metrics; (2) `np.floor(preds)` for metrics that penalize over-prediction (e.g., inventory cost asymmetry); (3) **probabilistic rounding** -- round up with probability equal to the fractional part, preserving the expected value: `rounded = np.floor(preds) + (np.random.random(len(preds)) < (preds - np.floor(preds)))`. For the M5 competition, rounding was not beneficial because WRMSSE uses continuous RMSE; however, for competitions with MAE or MAPE on integer targets, rounding typically improves the score by 0.1-0.5%.

### Ensemble of Different Horizons

When predicting multiple horizons (e.g., 28 days), different models may excel at different horizons. A **horizon-specific ensemble** assigns different weights to each model at each horizon. For example, an ARIMA model might receive weight 0.4 at horizons 1-7 (short-term, where its autoregressive structure excels) but weight 0.1 at horizons 21-28 (long-term, where LightGBM's cross-series learning dominates). Weights can be optimized per-horizon on the validation set using `scipy.optimize.minimize` with the competition metric as the objective.

```python
from scipy.optimize import minimize

def optimize_ensemble_weights(val_preds_list, y_val, metric_fn, n_models):
    """
    Optimize ensemble weights to minimize a competition metric.

    Parameters
    ----------
    val_preds_list : list of np.ndarray, each of shape (n_samples,).
                     Predictions from each model on the validation set.
    y_val : np.ndarray, true values.
    metric_fn : callable(y_true, y_pred) -> float, lower is better.
    n_models : int, number of models.
    """
    def objective(weights):
        # Normalize weights to sum to 1
        w = np.abs(weights) / np.abs(weights).sum()
        blended = sum(w[i] * val_preds_list[i] for i in range(n_models))
        return metric_fn(y_val, blended)

    # Initialize with equal weights
    x0 = np.ones(n_models) / n_models
    result = minimize(objective, x0, method="Nelder-Mead",
                      options={"maxiter": 1000, "xatol": 1e-6})
    optimal_weights = np.abs(result.x) / np.abs(result.x).sum()
    print(f"Optimal weights: {optimal_weights}")
    print(f"Optimal metric: {result.fun:.6f}")
    return optimal_weights
```

### Conformal Prediction for Intervals

**Conformal prediction** provides distribution-free prediction intervals with guaranteed coverage. Unlike quantile regression, conformal prediction makes no distributional assumptions and provides finite-sample coverage guarantees. The procedure: (1) fit a point prediction model on the training set; (2) compute the conformity scores (absolute residuals) on a calibration set; (3) for a desired coverage level alpha, set the interval width to the (1-alpha) quantile of the calibration residuals; (4) prediction interval = [point_prediction - q, point_prediction + q]. For time series, the calibration set must be temporally after the training set. **Adaptive conformal inference** (Gibbs and Candes, 2021) updates the quantile online as new data arrives, maintaining coverage under distribution shift.

```python
def conformal_prediction_intervals(y_cal, preds_cal, preds_test, alpha=0.1):
    """
    Compute conformal prediction intervals.

    Parameters
    ----------
    y_cal : np.ndarray, true values on calibration set.
    preds_cal : np.ndarray, point predictions on calibration set.
    preds_test : np.ndarray, point predictions on test set.
    alpha : float, miscoverage rate. 0.1 = 90% coverage.

    Returns
    -------
    lower, upper : np.ndarray, prediction interval bounds.
    """
    # Conformity scores = absolute residuals
    scores = np.abs(y_cal - preds_cal)
    # Quantile with finite-sample correction
    n = len(scores)
    q_level = np.ceil((1 - alpha) * (n + 1)) / n
    q = np.quantile(scores, min(q_level, 1.0))
    lower = preds_test - q
    upper = preds_test + q
    print(f"Conformal interval width: +/- {q:.4f}")
    print(f"Calibration coverage: {np.mean((y_cal >= preds_cal - q) & (y_cal <= preds_cal + q)):.3f}")
    return lower, upper

class AdaptiveConformal:
    """
    Adaptive Conformal Inference (ACI) for online prediction intervals.
    Updates the quantile level based on observed coverage.
    """
    def __init__(self, alpha=0.1, gamma=0.01):
        self.alpha_target = alpha
        self.alpha_t = alpha
        self.gamma = gamma  # step size for adaptation

    def update(self, y_true, lower, upper):
        """
        Update after observing true value. Adjust alpha if coverage is off.
        """
        covered = (y_true >= lower) and (y_true <= upper)
        # If covered, increase alpha (tighter intervals next time)
        # If not covered, decrease alpha (wider intervals next time)
        err_t = 1.0 - float(covered)
        self.alpha_t = self.alpha_t + self.gamma * (self.alpha_target - err_t)
        self.alpha_t = np.clip(self.alpha_t, 0.01, 0.5)
        return self.alpha_t
```

### Post-Processing Pipeline Code

```python
import numpy as np
import pandas as pd

class PostProcessor:
    """
    Complete post-processing pipeline for time series competition submissions.
    """

    def __init__(self, clip_lower=0, clip_upper=None, round_integers=False,
                 smooth_window=None, blend_with_baseline=None,
                 baseline_weight=0.0):
        """
        Parameters
        ----------
        clip_lower : float or None, lower bound for clipping.
        clip_upper : float or None, upper bound for clipping.
        round_integers : bool, whether to round to nearest integer.
        smooth_window : int or None, if set, apply moving average smoothing
                        across forecast horizons.
        blend_with_baseline : np.ndarray or None, baseline predictions to blend.
        baseline_weight : float, weight for baseline in blend (0 to 1).
        """
        self.clip_lower = clip_lower
        self.clip_upper = clip_upper
        self.round_integers = round_integers
        self.smooth_window = smooth_window
        self.blend_with_baseline = blend_with_baseline
        self.baseline_weight = baseline_weight

    def transform(self, predictions):
        """
        Apply all post-processing steps in order.

        Parameters
        ----------
        predictions : np.ndarray of shape (n_samples,) or (n_samples, n_horizons)

        Returns
        -------
        np.ndarray, post-processed predictions.
        """
        preds = predictions.copy().astype(float)

        # Step 1: Blend with baseline (e.g., seasonal naive)
        if self.blend_with_baseline is not None:
            preds = (1 - self.baseline_weight) * preds + \
                    self.baseline_weight * self.blend_with_baseline

        # Step 2: Smooth across horizons (for multi-step forecasts)
        if self.smooth_window is not None and preds.ndim == 2:
            smoothed = pd.DataFrame(preds).T.rolling(
                self.smooth_window, min_periods=1, center=True
            ).mean().T.values
            preds = smoothed

        # Step 3: Clip to valid range
        preds = np.clip(preds, self.clip_lower, self.clip_upper)

        # Step 4: Round for integer targets
        if self.round_integers:
            preds = np.round(preds).astype(int)

        return preds

# Example usage:
# pp = PostProcessor(clip_lower=0, clip_upper=1000, round_integers=False,
#                    smooth_window=3, blend_with_baseline=seasonal_naive_preds,
#                    baseline_weight=0.1)
# final_preds = pp.transform(raw_model_preds)

def create_submission(predictions, sample_submission_path, output_path,
                      id_col="id", target_col="demand"):
    """
    Create a competition submission file with post-processing checks.
    """
    sub = pd.read_csv(sample_submission_path)
    assert len(predictions) == len(sub), \
        f"Prediction length {len(predictions)} != submission length {len(sub)}"

    sub[target_col] = predictions

    # Sanity checks
    assert not sub[target_col].isna().any(), "NaN found in predictions"
    assert not np.isinf(sub[target_col]).any(), "Inf found in predictions"
    print(f"Prediction statistics:")
    print(f"  Min:    {sub[target_col].min():.4f}")
    print(f"  Max:    {sub[target_col].max():.4f}")
    print(f"  Mean:   {sub[target_col].mean():.4f}")
    print(f"  Median: {sub[target_col].median():.4f}")
    print(f"  Std:    {sub[target_col].std():.4f}")
    print(f"  Zeros:  {(sub[target_col] == 0).sum()} ({(sub[target_col] == 0).mean()*100:.1f}%)")

    sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path} ({len(sub)} rows)")
    return sub
```

---

## Resources

**Competitions (Kaggle)**:
- M5 Forecasting Accuracy: www.kaggle.com/c/m5-forecasting-accuracy
- M5 Forecasting Uncertainty: www.kaggle.com/c/m5-forecasting-uncertainty
- Optiver Realized Volatility Prediction: www.kaggle.com/c/optiver-realized-volatility-prediction
- Jane Street Market Prediction: www.kaggle.com/c/jane-street-market-prediction
- Ubiquant Market Prediction: www.kaggle.com/c/ubiquant-market-prediction
- G-Research Crypto Forecasting: www.kaggle.com/c/g-research-crypto-forecasting
- Walmart Recruiting Store Sales Forecasting: www.kaggle.com/c/walmart-recruiting-store-sales-forecasting

**Libraries**:
- LightGBM: github.com/microsoft/LightGBM (primary GBDT for time series competitions)
- neuralforecast (Nixtla): github.com/Nixtla/neuralforecast (N-BEATS, TFT, PatchTST, DeepAR)
- statsforecast (Nixtla): github.com/Nixtla/statsforecast (AutoARIMA, ETS, Theta, fast baselines)
- hierarchicalforecast (Nixtla): github.com/Nixtla/hierarchicalforecast (reconciliation methods)
- pytorch-forecasting: github.com/jdb78/pytorch-forecasting (TFT, DeepAR, N-BEATS in PyTorch)
- GluonTS (Amazon): github.com/awslabs/gluonts (DeepAR, Transformer, probabilistic models)
- Prophet (Meta): github.com/facebook/prophet
- Darts (Unit8): github.com/unit8co/darts (unified API for statistical and neural models)
- pmdarima: github.com/alkaline-ml/pmdarima (Python wrapper for auto.arima)
- tsfeatures: github.com/Nixtla/tsfeatures (automatic feature extraction for time series)
- MAPIE: github.com/scikit-learn-contrib/MAPIE (conformal prediction for time series)

**Papers**:
- Oreshkin et al. "N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting" (ICLR 2020)
- Lim et al. "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting" (IJF 2021)
- Nie et al. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers" (ICLR 2023) -- PatchTST
- Salinas et al. "DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks" (IJF 2020)
- Smyl "A Hybrid Method of Exponential Smoothing and Recurrent Neural Networks for Time Series Forecasting" (IJF 2020) -- M4 winner
- Makridakis et al. "M5 Accuracy Competition: Results, Findings, and Conclusions" (IJF 2022)
- Hyndman et al. "Optimal Combination Forecasts for Hierarchical Time Series" (Computational Statistics and Data Analysis 2011)
- Lopez de Prado "Advances in Financial Machine Learning" (Wiley 2018) -- purged CV, feature importance
- Gibbs and Candes "Adaptive Conformal Inference Under Distribution Shift" (NeurIPS 2021)

**Key Blogs and Discussions**:
- M5 1st place solution write-up: www.kaggle.com/c/m5-forecasting-accuracy/discussion/163684
- Optiver 1st place solution: www.kaggle.com/c/optiver-realized-volatility-prediction/discussion/274970
- Numerai documentation on feature neutralization: docs.numer.ai
- Rob Hyndman's forecasting textbook (free online): otexts.com/fpp3
