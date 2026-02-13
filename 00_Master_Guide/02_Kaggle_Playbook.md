# Kaggle Competition Playbook

## Table of Contents

- [Introduction](#introduction)
- [Competition Lifecycle](#competition-lifecycle)
- [Hardware and Compute Setup](#hardware-and-compute-setup)
- [Evaluation Metric Mastery](#evaluation-metric-mastery)
- [Validation Strategy](#validation-strategy)
- [Feature Engineering Mindset](#feature-engineering-mindset)
- [Ensemble Strategies](#ensemble-strategies)
- [Submission Strategy](#submission-strategy)
- [Team Collaboration](#team-collaboration)
- [Medal Strategy by Tier](#medal-strategy-by-tier)
- [Common Mistakes](#common-mistakes)
- [Time Management Templates](#time-management-templates)
- [See Also](#see-also)

---

## Introduction

Kaggle competitions are not academic exercises. They are adversarial optimization
problems where the margin between gold and silver is often less than 0.001 on the
evaluation metric. What separates top finishers from the rest is not knowledge of
exotic algorithms but **disciplined execution of fundamentals**.

**Core principles that separate top finishers:**

- **Trust your local CV above everything.** The public leaderboard is a noisy
  estimate on a small fraction of the test data. Competitors who chase LB scores
  get burned by shake-up. Build a robust validation scheme in the first week and
  never abandon it.
- **Read everything before writing code.** The competition description, the data
  description, the evaluation metric explanation, every forum post, and every
  public notebook. The winners almost always cite a forum post or data insight
  that guided their approach.
- **Iterate relentlessly.** A single model trained once will never win. The
  winning approach is dozens of experiments per day, each tracked and compared
  against local CV. The feedback loop speed is the bottleneck.
- **Ensemble aggressively.** Almost every winning solution is an ensemble. If
  you are not blending at least 3-5 diverse models, you are leaving performance
  on the table.
- **Post-processing matters.** Rounding, thresholding, clipping predictions to
  valid ranges, adjusting for known test distribution -- these small steps often
  close the gap between top 10 and top 100.

The mindset is that of an engineer, not a scientist. You are not trying to
understand the data generating process. You are trying to minimize a specific loss
function on a specific hidden test set within a deadline. Every decision should
serve that goal.

---

## Competition Lifecycle

A well-run competition follows a predictable arc. Deviating from this timeline
is the single most common reason competitors underperform relative to their
skill level.

### Day 1-3: Read Everything

Do not write a single line of model code. Spend the first three days absorbing
information.

- Read the **competition description** word by word. Understand exactly what is
  being predicted and why. Business context matters because it constrains what
  features make sense.
- Study the **evaluation metric**. Understand its mathematical properties. Is it
  threshold-dependent? Does it penalize false positives more than false negatives?
  Is it sensitive to calibration?
- Read the **data description**. Understand every column. Note which are
  categorical, which are continuous, which have missing values, which are IDs.
- Read **every forum post**. Sort by most votes. The host often clarifies
  ambiguities in the forum. Other competitors share critical insights about data
  quality issues.
- Read **public notebooks**. Do not fork and submit them. Read them to understand
  what approaches others are trying and what baseline scores look like.
- Check for **external data** rules. Some competitions allow external data, which
  changes everything.

### Day 3-5: EDA Notebook

Build a comprehensive EDA notebook that you will reference throughout the
competition.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def eda_report(train, test, target_col, id_col=None):
    """Generate a comprehensive EDA report for a Kaggle competition."""

    exclude_cols = [target_col]
    if id_col:
        exclude_cols.append(id_col)

    feature_cols = [c for c in train.columns if c not in exclude_cols]

    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Train shape: {train.shape}")
    print(f"Test shape:  {test.shape}")
    print(f"Train columns not in test: {set(train.columns) - set(test.columns)}")
    print(f"Test columns not in train: {set(test.columns) - set(train.columns)}")

    print(f"\nTarget distribution:")
    print(train[target_col].describe())
    print(f"Target nunique: {train[target_col].nunique()}")

    print("\n" + "=" * 60)
    print("MISSING VALUES")
    print("=" * 60)
    train_missing = train[feature_cols].isnull().sum()
    test_missing = test[feature_cols].isnull().sum()
    missing_df = pd.DataFrame({
        "train_missing": train_missing,
        "train_pct": train_missing / len(train) * 100,
        "test_missing": test_missing,
        "test_pct": test_missing / len(test) * 100,
    })
    missing_df = missing_df[missing_df["train_missing"] > 0].sort_values(
        "train_pct", ascending=False
    )
    print(missing_df.to_string())

    print("\n" + "=" * 60)
    print("FEATURE TYPES")
    print("=" * 60)
    for col in feature_cols:
        nunique = train[col].nunique()
        dtype = train[col].dtype
        category = "numeric" if dtype in ["int64", "float64"] else "categorical"
        if category == "numeric" and nunique < 20:
            category = "low_cardinality_numeric"
        print(f"  {col:40s} {str(dtype):10s} nunique={nunique:8d}  [{category}]")

    print("\n" + "=" * 60)
    print("TRAIN/TEST DISTRIBUTION DRIFT")
    print("=" * 60)
    for col in feature_cols:
        if train[col].dtype in ["int64", "float64"]:
            tr = train[col].dropna().values
            te = test[col].dropna().values
            if len(tr) > 0 and len(te) > 0:
                ks_stat, ks_pval = stats.ks_2samp(tr, te)
                if ks_pval < 0.01:
                    print(f"  DRIFT: {col:40s} KS={ks_stat:.4f} p={ks_pval:.6f}")

    return missing_df


def plot_target_distribution(train, target_col):
    """Plot target variable distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if train[target_col].nunique() <= 20:
        train[target_col].value_counts().sort_index().plot(
            kind="bar", ax=axes[0], title="Target Value Counts"
        )
    else:
        train[target_col].hist(bins=50, ax=axes[0])
        axes[0].set_title("Target Distribution")

    if train[target_col].dtype in ["int64", "float64"]:
        stats.probplot(train[target_col].dropna(), plot=axes[1])
        axes[1].set_title("Target Q-Q Plot")

    plt.tight_layout()
    plt.savefig("target_distribution.png", dpi=150, bbox_inches="tight")
    plt.show()
```

### Day 5-7: Strong Baseline

Get a strong baseline submitted as early as possible. The baseline should be
embarrassingly simple but competent.

**For tabular data:** LightGBM with default parameters and minimal feature
engineering. This is almost always the right first model.

**For computer vision:** A pretrained EfficientNet or ConvNeXt fine-tuned for
a few epochs. Do not architect from scratch.

**For NLP:** A pretrained DeBERTa or similar transformer fine-tuned on the
competition data. Start with the smallest model that fits in memory.

**For time series:** LightGBM with lag features and rolling statistics.

The baseline score tells you where you stand and gives you a reference point
for every future experiment.

### Week 2: Feature Engineering Iteration

This is where competitions are won. Spend the bulk of your time here.

- Run 5-10 experiments per day.
- Track every experiment: features used, hyperparameters, CV score, LB score.
- Use feature importance from your best model to guide the next round of
  feature engineering.
- Try domain-specific features inspired by forum discussion.
- Automate the experiment loop so each run takes minutes, not hours.

### Week 3+: Ensemble and Post-Processing

- Train diverse models: LightGBM, XGBoost, CatBoost, neural networks.
- Use different feature subsets for diversity.
- Stack or blend OOF predictions.
- Tune post-processing: thresholds, rounding, clipping.

### Final Days: Submit and Trust CV

- Select your two final submissions carefully.
- One should be your best CV score (safe pick).
- One can be a more aggressive ensemble or an experiment.
- **Do not change your submission in the last hour based on public LB.**
- Trust your CV. Go to sleep.

---

## Hardware and Compute Setup

Compute is a competitive advantage. Faster iteration means more experiments,
and more experiments mean better scores.

### Kaggle Notebooks

- **Free tier:** 30 hours/week GPU (T4 or P100), 20 hours/week TPU.
- **Limitations:** 9-hour max session, limited disk, no internet during
  submission (for code competitions).
- **Best for:** Quick experiments, submitting in code competitions, leveraging
  Kaggle datasets without download.
- **Tip:** Use Kaggle datasets as a package manager. Upload your preprocessed
  data, trained models, and utility scripts as datasets, then import them.

### Google Colab Pro+

- **Cost:** Roughly $50/month.
- **Hardware:** A100 GPU (40GB), high-memory VMs, longer runtimes.
- **Best for:** Training large models, NLP transformers that need >16GB VRAM.
- **Tip:** Mount Google Drive for persistent storage. Save checkpoints
  frequently because sessions can still disconnect.

### Local GPU Setup

- **RTX 3090 (24GB):** Excellent price-to-performance for competitions. Handles
  most tabular and medium CV/NLP workloads.
- **RTX 4090 (24GB):** Faster than 3090 for training, better for large batch
  sizes due to higher bandwidth.
- **RTX 5090 (32GB):** The current best consumer option. Extra VRAM matters for
  NLP and large image models.
- **Setup essentials:** Ubuntu/Linux, CUDA toolkit, cuDNN, conda environment
  per competition.

```python
# Check GPU availability and memory
import torch

def print_gpu_info():
    """Print GPU information for debugging compute setup."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_mem / (1024 ** 3)
            print(f"GPU {i}: {props.name}")
            print(f"  Total memory: {total_mem:.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Multi-processor count: {props.multi_processor_count}")
    else:
        print("No CUDA GPU available")

print_gpu_info()
```

### Cloud Instances

| Provider | Instance     | GPU         | VRAM  | Approx. Cost/hr |
|----------|-------------|-------------|-------|------------------|
| AWS      | p3.2xlarge  | 1x V100     | 16GB  | $3.06            |
| AWS      | p4d.24xlarge| 8x A100     | 320GB | $32.77           |
| GCP      | n1 + T4     | 1x T4       | 16GB  | $1.40            |
| GCP      | a2-highgpu  | 1x A100     | 40GB  | $3.67            |
| Lambda   | 1x A100     | 1x A100     | 40GB  | $1.10            |

**Cost optimization strategies:**

- Use **spot/preemptible instances** for training (60-80% discount). Save
  checkpoints every epoch so preemption does not lose progress.
- **Start local, scale to cloud.** Develop and debug locally, then run full
  training on cloud.
- **Shut down instances** the moment training finishes. Use scripts that
  auto-terminate after the job completes.
- **Profile first.** Often the bottleneck is data loading, not GPU compute.
  Fix the bottleneck before buying more GPU.

### Multi-GPU Training

```python
# PyTorch DistributedDataParallel minimal setup
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    """Initialize distributed training process group."""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training."""
    dist.destroy_process_group()

def train_worker(rank, world_size, model_cls, dataset, epochs, batch_size):
    """Worker function for distributed training."""
    setup(rank, world_size)

    model = model_cls().to(rank)
    model = DDP(model, device_ids=[rank])

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()

    cleanup()

# Launch: mp.spawn(train_worker, args=(world_size, ...), nprocs=world_size)
```

---

## Evaluation Metric Mastery

You cannot optimize what you do not understand. Every competition has a specific
evaluation metric, and your entire pipeline -- from loss function to
post-processing -- should be aligned with it.

### AUC-ROC

**Area Under the Receiver Operating Characteristic curve.** Measures the
probability that a randomly chosen positive example is ranked higher than a
randomly chosen negative example.

- **Threshold-independent.** Your model only needs to rank correctly, not
  produce calibrated probabilities.
- **Insensitive to class imbalance** in the sense that it does not depend on
  the prevalence ratio, but can be misleading when negatives vastly outnumber
  positives.
- **Optimization:** Use binary cross-entropy as the training loss. It directly
  optimizes for ranking quality.

### Log Loss (Binary Cross-Entropy)

- **Probabilistic.** Penalizes confident wrong predictions severely.
- **Requires calibration.** A model that ranks well but outputs poorly
  calibrated probabilities will score badly.
- **Optimization:** Train with log loss directly. Consider Platt scaling or
  isotonic regression for post-hoc calibration.

### F1 and F-Beta Scores

- **Threshold-dependent.** You must choose a classification threshold.
- **F1:** Harmonic mean of precision and recall (beta=1).
- **F-beta:** Weights recall beta times more than precision.
- **Optimization:** Train with log loss for good probability estimates, then
  optimize the threshold on your validation set.

```python
import numpy as np
from scipy.optimize import minimize_scalar

def optimize_f1_threshold(y_true, y_prob):
    """Find the threshold that maximizes F1 score."""
    def neg_f1(threshold):
        y_pred = (y_prob >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        return -f1

    result = minimize_scalar(neg_f1, bounds=(0.01, 0.99), method="bounded")
    best_threshold = result.x
    best_f1 = -result.fun
    return best_threshold, best_f1
```

### RMSE and MAE

- **RMSE (Root Mean Squared Error):** Penalizes large errors quadratically.
  Sensitive to outliers. Optimized by MSE loss.
- **MAE (Mean Absolute Error):** Treats all error magnitudes linearly.
  Robust to outliers. Optimized by MAE/Huber loss.
- **Practical note:** If the metric is RMSE, consider log-transforming the
  target if it is right-skewed. Predict log(target), then exponentiate.

### QWK (Quadratic Weighted Kappa)

Used in ordinal classification (e.g., rating predictions 1-5).

- Measures agreement between predicted and actual ratings, weighted by the
  squared distance between them.
- **Optimization:** Train as regression, then optimize rounding thresholds.

```python
from sklearn.metrics import cohen_kappa_score
from scipy.optimize import minimize

def optimize_qwk_thresholds(y_true, y_pred_continuous, num_classes):
    """Optimize rounding thresholds for QWK."""
    def neg_qwk(thresholds):
        thresholds = np.sort(thresholds)
        y_pred_rounded = np.digitize(y_pred_continuous, thresholds)
        return -cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")

    # Initial thresholds: evenly spaced
    init_thresholds = np.linspace(0.5, num_classes - 0.5, num_classes - 1)
    result = minimize(neg_qwk, init_thresholds, method="Nelder-Mead")
    optimal_thresholds = np.sort(result.x)
    best_qwk = -result.fun
    return optimal_thresholds, best_qwk
```

### MAP@K (Mean Average Precision at K)

Used in retrieval and recommendation competitions.

- For each query, compute average precision of the top K predictions.
- Average across all queries.

```python
def apk(actual, predicted, k=10):
    """Compute Average Precision at K for a single query."""
    if not actual:
        return 0.0
    predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k)

def mapk(actual_list, predicted_list, k=10):
    """Compute Mean Average Precision at K across all queries."""
    return np.mean([apk(a, p, k) for a, p in zip(actual_list, predicted_list)])
```

### Dice / IoU (Segmentation)

- **Dice coefficient:** 2 * |A intersection B| / (|A| + |B|). Ranges 0 to 1.
- **IoU (Jaccard):** |A intersection B| / |A union B|.
- **Optimization:** Use Dice loss or a combination of Dice + BCE as the
  training loss for segmentation tasks.

### Custom Metrics as Sklearn Scorers

```python
from sklearn.metrics import make_scorer

def custom_metric(y_true, y_pred):
    """Define your competition-specific metric here."""
    # Example: weighted combination
    return np.mean(np.abs(y_true - y_pred) * np.where(y_true > 0, 2.0, 1.0))

custom_scorer = make_scorer(custom_metric, greater_is_better=False)
# Use in cross_val_score: cross_val_score(model, X, y, scoring=custom_scorer)
```

---

## Validation Strategy

**Validation is the foundation of everything.** If your validation is wrong,
every decision you make will be wrong. Spend serious time getting this right.

### Rule 1: Trust Local CV, Not Public LB

The public leaderboard shows your score on a subset (often 20-30%) of the test
data. It is a noisy estimate. Your local CV, if properly constructed, averages
over multiple folds and is far more reliable.

**If local CV improves but public LB drops, keep the change.** The final
private LB will likely agree with your CV.

### Rule 2: CV Must Mimic Test Distribution

Your cross-validation scheme must create train/validation splits that resemble
the train/test split the competition organizers created. If the test data is
from a different time period, use time-based splits. If the test data groups
entities differently, use group-based splits.

### Stratified K-Fold

The default choice for classification. Ensures each fold has approximately the
same target distribution as the full training set.

```python
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import numpy as np

def stratified_kfold_lgbm(X, y, params, n_splits=5, seed=42):
    """Train LightGBM with stratified k-fold and return OOF predictions."""
    oof_preds = np.zeros(len(X))
    models = []
    scores = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100),
                lgb.log_evaluation(200),
            ],
        )

        oof_preds[val_idx] = model.predict(X_val)
        models.append(model)

        fold_score = params.get("metric", "auc")
        print(f"Fold {fold}: best_iteration={model.best_iteration}")

    return oof_preds, models, scores
```

### Group K-Fold

Use when there are groups in the data (e.g., multiple rows per user, per
patient, per store). All rows belonging to the same group must be in the same
fold to prevent leakage.

```python
from sklearn.model_selection import GroupKFold

def group_kfold_split(X, y, groups, n_splits=5):
    """Generate group-aware cross-validation splits."""
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, val_idx in gkf.split(X, y, groups):
        # Verify no group leakage
        train_groups = set(groups.iloc[train_idx])
        val_groups = set(groups.iloc[val_idx])
        assert len(train_groups & val_groups) == 0, "Group leakage detected!"
        splits.append((train_idx, val_idx))
    return splits
```

### Time-Based Split

For time series or any data with temporal ordering. Never train on future data
to predict the past.

```python
def time_based_split(df, time_col, n_splits=5, gap_days=0):
    """Generate time-based cross-validation splits with optional gap."""
    import pandas as pd

    df = df.sort_values(time_col).reset_index(drop=True)
    unique_dates = sorted(df[time_col].unique())
    fold_size = len(unique_dates) // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = unique_dates[(i + 1) * fold_size]
        if gap_days > 0:
            val_start = train_end + pd.Timedelta(days=gap_days)
        else:
            val_start = train_end
        val_end = unique_dates[min((i + 2) * fold_size, len(unique_dates) - 1)]

        train_idx = df[df[time_col] < train_end].index.values
        val_idx = df[
            (df[time_col] >= val_start) & (df[time_col] <= val_end)
        ].index.values

        if len(val_idx) > 0:
            splits.append((train_idx, val_idx))

    return splits
```

### Adversarial Validation

Detects whether the train and test distributions differ. Train a classifier to
distinguish train from test. If AUC is close to 0.5, they are similar. If AUC
is high, there is distribution shift, and you should design your CV to account
for it.

```python
def adversarial_validation(train_df, test_df, feature_cols, n_splits=5):
    """Detect train/test distribution shift using adversarial validation."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    # Label: 0 = train, 1 = test
    combined = pd.concat([
        train_df[feature_cols].assign(is_test=0),
        test_df[feature_cols].assign(is_test=1),
    ], ignore_index=True)

    X = combined[feature_cols]
    y = combined["is_test"]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        dtrain = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx])
        dval = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

        model = lgb.train(
            {"objective": "binary", "metric": "auc", "verbose": -1},
            dtrain,
            num_boost_round=200,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
        )

        preds = model.predict(X.iloc[val_idx])
        auc = roc_auc_score(y.iloc[val_idx], preds)
        aucs.append(auc)

    mean_auc = np.mean(aucs)
    print(f"Adversarial Validation AUC: {mean_auc:.4f}")
    if mean_auc > 0.7:
        print("WARNING: Significant train/test distribution shift detected.")
        print("Consider using adversarial features or time-based splits.")
    else:
        print("Train and test distributions appear similar.")

    return mean_auc
```

### Nested CV for Hyperparameter Tuning

Use nested CV when you want unbiased estimates of generalization performance
while also tuning hyperparameters.

- **Outer loop:** Evaluates model performance (5-fold).
- **Inner loop:** Tunes hyperparameters (3-fold within each outer fold).
- **When to use:** When you suspect your hyperparameter tuning is overfitting
  to your validation set. In practice, most Kaggle competitors skip nested CV
  and tune on a single holdout, accepting the slight bias.

### 5-Fold vs 10-Fold

- **5-fold:** Standard choice. Good balance of bias and variance. Faster to run.
  Use this unless you have a specific reason not to.
- **10-fold:** Lower bias, but higher variance. Takes twice as long. Use when
  the dataset is small (under 10k samples) and you want more stable estimates.
- **Repeated K-fold:** Run 5-fold three times with different seeds. Averages
  out the randomness in fold assignment. Use for final model selection.

---

## Feature Engineering Mindset

Feature engineering is the highest-leverage activity in most tabular
competitions. A mediocre model with great features will beat a great model with
mediocre features almost every time.

### Understand the Domain First

Before generating features mechanically, understand what the data represents.
Read the competition description. Read relevant Wikipedia articles. If it is a
healthcare competition, learn the clinical context. If it is a financial
competition, understand the products involved.

**The best features encode domain knowledge.** They are not discovered by brute
force. They are hypothesized by someone who understands the problem and then
validated by the model.

### Start Simple, Iterate

Do not spend three days building a feature engineering pipeline before you have
a baseline model. Start with raw features. Get a score. Then add features one
at a time (or in small groups) and measure the impact on CV.

### Feature Importance to Guide Next Features

After each model run, examine feature importance. This tells you:

- Which features the model finds useful (high importance).
- Which features are noise (zero or near-zero importance, candidates for removal).
- Where to focus next (if aggregates of feature X are important, try more
  aggregates of X).

### Aggregate Features (Group-By Statistics)

The single most powerful category of engineered features for tabular data.

```python
def create_aggregate_features(df, group_col, agg_col, agg_funcs=None):
    """Create group-by aggregate features."""
    if agg_funcs is None:
        agg_funcs = ["mean", "std", "min", "max", "median", "count"]

    agg_dict = {agg_col: agg_funcs}
    grouped = df.groupby(group_col).agg(agg_dict)
    grouped.columns = [f"{group_col}_{agg_col}_{func}" for func in agg_funcs]
    grouped = grouped.reset_index()

    df = df.merge(grouped, on=group_col, how="left")

    # Deviation from group mean
    mean_col = f"{group_col}_{agg_col}_mean"
    df[f"{agg_col}_dev_from_{group_col}_mean"] = df[agg_col] - df[mean_col]

    # Ratio to group mean
    df[f"{agg_col}_ratio_to_{group_col}_mean"] = df[agg_col] / (
        df[mean_col] + 1e-10
    )

    return df
```

### Target Encoding

Powerful but dangerous. Encodes a categorical variable as the mean of the
target for that category. **Must be done within cross-validation folds to
prevent leakage.**

```python
def target_encode_cv(train, test, cat_col, target_col, n_splits=5, smooth=10):
    """Target encode with cross-validation to prevent leakage."""
    from sklearn.model_selection import KFold

    global_mean = train[target_col].mean()
    train[f"{cat_col}_target_enc"] = np.nan
    test_encodings = []

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(train):
        # Compute encoding from training fold only
        fold_train = train.iloc[train_idx]
        agg = fold_train.groupby(cat_col)[target_col].agg(["mean", "count"])
        # Smoothed encoding
        agg["smooth_mean"] = (
            (agg["mean"] * agg["count"] + global_mean * smooth)
            / (agg["count"] + smooth)
        )
        encoding_map = agg["smooth_mean"].to_dict()

        # Apply to validation fold
        train.iloc[val_idx, train.columns.get_loc(f"{cat_col}_target_enc")] = (
            train.iloc[val_idx][cat_col].map(encoding_map).fillna(global_mean)
        )

        test_encodings.append(
            test[cat_col].map(encoding_map).fillna(global_mean)
        )

    # Average test encodings across folds
    test[f"{cat_col}_target_enc"] = np.mean(test_encodings, axis=0)

    return train, test
```

### Null Handling as a Feature

Missing values often carry information. Before imputing, create indicator
columns.

```python
def add_null_features(df, cols=None):
    """Add null indicator features and null count."""
    if cols is None:
        cols = df.columns[df.isnull().any()].tolist()

    for col in cols:
        df[f"{col}_is_null"] = df[col].isnull().astype(int)

    df["null_count"] = df[cols].isnull().sum(axis=1)
    return df
```

### Feature Interactions

Simple arithmetic combinations of features can capture non-linear
relationships that tree models might miss (or find more efficiently).

```python
def create_interactions(df, col_pairs):
    """Create interaction features for specified column pairs.

    Args:
        df: DataFrame.
        col_pairs: List of (col_a, col_b) tuples.
    """
    for col_a, col_b in col_pairs:
        df[f"{col_a}_times_{col_b}"] = df[col_a] * df[col_b]
        df[f"{col_a}_div_{col_b}"] = df[col_a] / (df[col_b] + 1e-10)
        df[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]
        df[f"{col_a}_plus_{col_b}"] = df[col_a] + df[col_b]
    return df
```

### Text Features in Tabular Competitions

When a tabular dataset includes free-text columns, extract simple features
before resorting to NLP models.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

def extract_text_features(train_text, test_text, max_features=200):
    """Extract text features for tabular models."""
    # Basic stats
    def text_stats(series):
        stats_df = pd.DataFrame()
        stats_df["text_len"] = series.str.len()
        stats_df["word_count"] = series.str.split().str.len()
        stats_df["unique_words"] = series.apply(
            lambda x: len(set(str(x).split())) if pd.notna(x) else 0
        )
        stats_df["avg_word_len"] = series.apply(
            lambda x: np.mean([len(w) for w in str(x).split()]) if pd.notna(x) else 0
        )
        stats_df["uppercase_ratio"] = series.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1)
        )
        return stats_df

    train_stats = text_stats(train_text)
    test_stats = text_stats(test_text)

    # TF-IDF
    tfidf = TfidfVectorizer(max_features=max_features, stop_words="english")
    train_tfidf = tfidf.fit_transform(train_text.fillna(""))
    test_tfidf = tfidf.transform(test_text.fillna(""))

    return train_stats, test_stats, train_tfidf, test_tfidf
```

---

## Ensemble Strategies

Almost every winning Kaggle solution uses an ensemble. The key insight is that
**diversity matters more than individual model quality.** Three mediocre models
that make different errors will outperform three copies of the best single model.

### Sources of Diversity

- **Different algorithms:** LightGBM + XGBoost + CatBoost + neural network.
- **Different feature sets:** Each model sees a different subset of features.
- **Different preprocessing:** One model on raw features, another on normalized,
  another on rank-transformed.
- **Different random seeds:** Even the same algorithm with different seeds
  produces slightly different models.
- **Different hyperparameters:** Shallow trees vs deep trees, high learning
  rate vs low learning rate.

### Weighted Averaging

The simplest ensemble method. Weight each model's predictions and average.

```python
from scipy.optimize import minimize

def optimize_ensemble_weights(oof_predictions, y_true, metric_fn, maximize=True):
    """Find optimal weights for ensemble averaging.

    Args:
        oof_predictions: List of numpy arrays, one per model (OOF predictions).
        y_true: True labels.
        metric_fn: Function(y_true, y_pred) -> score.
        maximize: If True, maximize metric; if False, minimize.

    Returns:
        Optimal weights, best score.
    """
    n_models = len(oof_predictions)

    def objective(weights):
        # Normalize weights to sum to 1
        weights = np.abs(weights)
        weights = weights / weights.sum()
        blended = np.zeros_like(oof_predictions[0])
        for w, pred in zip(weights, oof_predictions):
            blended += w * pred
        score = metric_fn(y_true, blended)
        return -score if maximize else score

    # Start with equal weights
    init_weights = np.ones(n_models) / n_models
    constraints = {"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - 1.0}

    result = minimize(
        objective,
        init_weights,
        method="Nelder-Mead",
        options={"maxiter": 10000},
    )

    best_weights = np.abs(result.x)
    best_weights = best_weights / best_weights.sum()
    best_score = -result.fun if maximize else result.fun

    print("Optimal ensemble weights:")
    for i, w in enumerate(best_weights):
        print(f"  Model {i}: {w:.4f}")
    print(f"Ensemble score: {best_score:.6f}")

    return best_weights, best_score
```

### Stacking

Use **out-of-fold (OOF) predictions** from base models as input features for a
meta-learner. This is more powerful than simple averaging because the
meta-learner can learn which models to trust in which regions of the input space.

```python
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_predict

def build_stacking_ensemble(oof_predictions, y_true, test_predictions, task="regression"):
    """Build a stacking ensemble with a simple meta-learner.

    Args:
        oof_predictions: List of OOF prediction arrays from base models.
        y_true: True target values.
        test_predictions: List of test prediction arrays from base models.
        task: 'regression' or 'classification'.

    Returns:
        Meta-model OOF predictions, meta-model test predictions.
    """
    # Stack OOF predictions as meta-features
    meta_train = np.column_stack(oof_predictions)
    meta_test = np.column_stack(test_predictions)

    if task == "regression":
        meta_model = Ridge(alpha=1.0)
    else:
        meta_model = LogisticRegression(C=1.0, max_iter=1000)

    # OOF predictions from meta-model (to evaluate stacking performance)
    meta_oof = cross_val_predict(meta_model, meta_train, y_true, cv=5)

    # Fit meta-model on all OOF predictions and predict test
    meta_model.fit(meta_train, y_true)
    meta_test_preds = meta_model.predict(meta_test)

    return meta_oof, meta_test_preds
```

### Blending

Simpler than stacking. Hold out a portion of the training data, generate
predictions from base models on the holdout, then train the meta-learner on
those predictions.

- **Advantage:** No risk of OOF leakage. Conceptually simpler.
- **Disadvantage:** Wastes data. The holdout is not used for base model training.
- **When to use:** When you do not trust your OOF generation pipeline or when
  the dataset is large enough that wasting 20% is acceptable.

### Rank Averaging

When models output predictions on different scales (e.g., one model outputs
probabilities 0-1, another outputs logits -5 to 5), convert to ranks before
averaging.

```python
from scipy.stats import rankdata

def rank_average(predictions_list):
    """Average predictions after converting to ranks.

    Args:
        predictions_list: List of prediction arrays from different models.

    Returns:
        Rank-averaged predictions (normalized to 0-1 range).
    """
    ranked = []
    for preds in predictions_list:
        ranks = rankdata(preds) / len(preds)
        ranked.append(ranks)
    return np.mean(ranked, axis=0)
```

### Hill Climbing

Greedy forward selection of ensemble members. Start with the best single model.
Then iteratively add the model that improves the ensemble the most. Stop when no
model improves the ensemble.

```python
def hill_climbing_ensemble(oof_predictions, y_true, metric_fn, maximize=True):
    """Greedy forward selection of ensemble members.

    Args:
        oof_predictions: Dict of {model_name: oof_array}.
        y_true: True labels.
        metric_fn: Function(y_true, y_pred) -> score.
        maximize: Whether higher metric is better.

    Returns:
        Selected model names, weights, final score.
    """
    model_names = list(oof_predictions.keys())
    compare = (lambda a, b: a > b) if maximize else (lambda a, b: a < b)

    # Find best single model
    best_score = None
    best_model = None
    for name in model_names:
        score = metric_fn(y_true, oof_predictions[name])
        if best_score is None or compare(score, best_score):
            best_score = score
            best_model = name

    selected = [best_model]
    selected_preds = [oof_predictions[best_model]]
    current_blend = oof_predictions[best_model].copy()

    print(f"Start: {best_model} -> {best_score:.6f}")

    improved = True
    while improved:
        improved = False
        best_candidate = None
        best_new_score = best_score

        for name in model_names:
            if name in selected:
                continue
            # Try adding this model with equal weight
            n = len(selected_preds) + 1
            candidate_blend = (
                current_blend * (n - 1) / n + oof_predictions[name] / n
            )
            score = metric_fn(y_true, candidate_blend)
            if compare(score, best_new_score):
                best_new_score = score
                best_candidate = name

        if best_candidate is not None:
            selected.append(best_candidate)
            selected_preds.append(oof_predictions[best_candidate])
            n = len(selected_preds)
            current_blend = np.mean(selected_preds, axis=0)
            best_score = best_new_score
            improved = True
            print(f"  + {best_candidate} -> {best_score:.6f} ({n} models)")

    print(f"\nFinal ensemble: {len(selected)} models, score={best_score:.6f}")
    weights = {name: 1.0 / len(selected) for name in selected}
    return selected, weights, best_score
```

### When to Stop Adding Models

- When the marginal improvement drops below your estimated LB noise (typically
  0.0001 or less).
- When all remaining candidate models are highly correlated with models already
  in the ensemble (check pairwise correlation of OOF predictions).
- When adding more models makes your pipeline too slow or complex to manage in
  the final submission window.

---

## Submission Strategy

The final submission selection can make or break your competition result. You
get two final submissions (in most competitions), and the private LB score of
your selected submissions determines your final rank.

### The Two-Submission Rule

- **Submission 1 (Safe):** Your best local CV score. This should be the model
  or ensemble you trust the most based on rigorous cross-validation.
- **Submission 2 (Aggressive):** Either your best public LB score (if different
  from Submission 1) or a more aggressive ensemble that might score higher but
  has more variance.

### Shake-Up Analysis

**Shake-up** is when competitors' rankings change dramatically between the
public and private leaderboards. Shake-up is most severe when:

- The public test set is small (high variance in public LB scores).
- The metric is sensitive to a few examples (e.g., log loss with outliers).
- Competitors overfit to the public LB.

**Estimate shake-up risk:** Compare your public LB score to your local CV score.
If public LB is significantly better than CV, you may be overfitting to the
public portion of the test set.

### Overfitting to Public LB

**Signs you are overfitting to public LB:**

- You make changes that hurt CV but improve public LB, and you keep them.
- You submit more than 3-5 times per day.
- You select models based on public LB rather than CV.
- Your public LB score is suspiciously better than your CV score.

**Prevention:** Limit yourself to 1-2 submissions per day. Always record your CV
score alongside your LB score. If they diverge, trust CV.

### Late Submission Strategy

If you join late (final week), focus on:

1. Read the top public notebooks and forum posts.
2. Build the strongest single model you can (no time for elaborate ensembles).
3. Apply any post-processing tricks mentioned in the forum.
4. Select submissions conservatively (trust CV).

---

## Team Collaboration

### When to Team Up

- **Early in the competition:** Risky. You do not know if your approaches are
  complementary. Wait until you each have a strong single model.
- **Mid-competition:** Ideal. You have established baselines and can immediately
  benefit from model diversity.
- **Late in the competition:** Can work if you each bring a strong, different
  model to the table.

**Team merge checklist:**

- Compare OOF predictions. Low correlation means high ensemble potential.
- Verify compatible validation schemes. You must agree on how to evaluate.
- Agree on code sharing and experiment tracking conventions.

### Code Sharing and Notebook Organization

```
competition_name/
    data/
        raw/
        processed/
        external/
    notebooks/
        eda/
        experiments/
    src/
        features/
            feature_engineering.py
            target_encoding.py
        models/
            lgbm_trainer.py
            nn_trainer.py
        ensemble/
            stacking.py
            blending.py
        utils/
            metrics.py
            data_loader.py
    configs/
        lgbm_v1.yaml
        lgbm_v2.yaml
    submissions/
    oof/
    logs/
    README.md
```

### Diverse Skillsets

The best teams combine:

- **Feature engineer:** Deep domain understanding, creative feature construction.
- **Model specialist:** Hyperparameter tuning, architecture design, training tricks.
- **Ensemble expert:** Stacking, blending, weight optimization, submission selection.

In practice, with 2-3 person teams, each person covers multiple roles.

### Communication Workflow

- **Daily sync:** Share CV scores, discuss what worked and what did not. Five
  minutes is enough.
- **Shared experiment tracker:** Google Sheet or Notion table with columns for
  experiment name, features, model, CV score, LB score, notes.
- **Version control:** Git repo shared via GitHub. Each person works on a branch.
- **Shared OOF predictions:** Upload OOF predictions and test predictions to a
  shared folder. This allows anyone on the team to run ensemble experiments.

---

## Medal Strategy by Tier

### Bronze Medal

- Submit anything above the default baseline.
- A simple LightGBM with default parameters and basic feature engineering is
  often enough for bronze.
- Focus on learning the workflow rather than optimizing the score.
- **Minimum effort:** Fork a public notebook, make a small improvement, submit.

### Silver Medal

- A well-tuned single model with thoughtful feature engineering.
- Basic ensemble: average 2-3 models (LightGBM + XGBoost + CatBoost).
- Proper validation strategy.
- Some post-processing (threshold optimization, clipping).
- **Time investment:** 20-40 hours over the competition.

### Gold Medal

- Innovative feature engineering that goes beyond what is in public notebooks.
- Diverse ensemble of 5+ models including at least one neural network.
- Perfect cross-validation that you trust completely.
- Post-processing tuned to the specific metric.
- Active forum participation (both reading and contributing).
- **Time investment:** 60-100+ hours over the competition.

### Grandmaster Path

- **Competitions Master:** 5 gold medals, 1 solo gold.
- **Consistency is key.** Competing in 2-3 competitions per year, finishing in
  the top 1% each time.
- **Specialize, then generalize.** Start by dominating one competition type
  (e.g., tabular), then expand to CV and NLP.
- **Build reusable infrastructure.** Your competition template should let you
  set up a new competition in under an hour.
- **Team strategically.** Form teams with other strong competitors for gold
  medal pushes.

---

## Common Mistakes

### Spending Too Long on EDA Before Modeling

EDA is important but has diminishing returns. You will learn more about the data
from modeling it than from plotting it. Get a model running by day 5 at the
latest. Then use model insights (feature importance, error analysis) to guide
further exploration.

### Not Reading the Forums

The Kaggle forums are where hosts clarify rules, competitors share data
insights, and key ideas get discussed. Competitors who ignore the forum are
flying blind. Check the forum daily.

### Overfitting to Public Leaderboard

This is the number one cause of shake-up disappointment. The public LB is a
noisy estimate. If you make decisions based on it instead of your local CV, you
will regret it when the private LB is revealed. Build a trustworthy CV scheme
and let it guide every decision.

### Wrong Validation Strategy

If your CV does not correlate with the public LB at all, something is wrong
with your validation scheme. Common causes:

- Not respecting group structure (group leakage).
- Not respecting temporal ordering (data leakage).
- Using random splits when the test set was selected non-randomly.
- Too few folds (high variance in CV estimate).

### Not Ensembling Enough

Many competitors stop at a single model. Even a simple average of LightGBM +
XGBoost + CatBoost with default parameters will often gain 0.002-0.005 on the
metric, which can translate to hundreds of ranks.

### Starting Too Late

If the competition runs for 3 months, starting in the final week means you skip
the most important phase: iterative feature engineering. You cannot compensate
with a fancy model. Start early and iterate.

### Ignoring Post-Processing

Small post-processing steps can have outsized impact:

- **Threshold optimization** for classification metrics.
- **Clipping** predictions to valid ranges.
- **Rounding** for ordinal/integer targets.
- **Adjusting** predictions to match known test set statistics.
- **Label smoothing** in the opposite direction: if you used label smoothing
  during training, adjust predictions to be more confident at test time.

---

## Time Management Templates

### 2-Week Sprint vs 3-Month Marathon

| Phase                    | 2-Week Sprint    | 3-Month Marathon  |
|--------------------------|------------------|-------------------|
| Read rules and forum     | Day 1            | Days 1-3          |
| EDA                      | Day 1-2          | Days 3-7          |
| Baseline model           | Day 2-3          | Days 5-10         |
| Feature engineering v1   | Day 3-5          | Weeks 2-3         |
| Feature engineering v2   | Day 5-7          | Weeks 3-5         |
| Hyperparameter tuning    | Day 7-8          | Weeks 5-6         |
| Additional model types   | Day 8-9          | Weeks 6-8         |
| Ensemble building        | Day 9-11         | Weeks 8-10        |
| Post-processing          | Day 11-12        | Weeks 10-11       |
| Submission selection     | Day 13-14        | Week 12           |
| Buffer for surprises     | None             | Week 12           |

### Daily Routine During Active Competition

**Weekday (2-3 hours available):**

1. **Morning (15 min):** Check forum for new posts, review overnight
   experiment results.
2. **Evening session (2 hours):**
   - Review results from previous experiments.
   - Design and queue 2-3 new experiments.
   - Update experiment tracker.
   - Submit best result if improved.

**Weekend (6-8 hours available):**

1. **Morning (3-4 hours):**
   - Deep feature engineering session.
   - Run longer experiments (full hyperparameter search, multiple seeds).
2. **Afternoon (2-3 hours):**
   - Ensemble experiments.
   - Error analysis on worst predictions.
   - Write up findings for team (if applicable).
3. **Evening (1 hour):**
   - Queue overnight experiments.
   - Plan next week's priorities.

### When to Sleep vs When to Push

- **Sleep** when you are waiting for long-running experiments. Start training
  before bed, review results in the morning.
- **Sleep** when you are making mistakes. Tired coding leads to bugs that waste
  more time than sleeping would have cost.
- **Push** on the final weekend before the deadline. This is when ensemble
  tuning and submission selection happen.
- **Push** when you have a breakthrough idea. If a new feature improved CV by
  0.005, immediately try variations of it.
- **Never** pull an all-nighter for the final submission. Make your final
  selections by 6 PM on the last day and walk away. Panic changes at midnight
  almost always make things worse.

---

## See Also

- `../01_Supervised_Learning/` -- Detailed guides on individual algorithms
  (LightGBM, XGBoost, CatBoost, neural networks) referenced in this playbook.
- `../02_Feature_Engineering/` -- Extended feature engineering techniques
  beyond what is covered in this playbook.
- `../03_Model_Evaluation/` -- Deep dive into evaluation metrics, calibration,
  and threshold optimization.
- `../04_Ensemble_Methods/` -- Advanced ensemble techniques including
  multi-level stacking and Bayesian model combination.
- `../05_Competition_Solutions/` -- Writeups of specific competition solutions
  that demonstrate these principles in practice.
- `../06_NLP/` -- NLP-specific competition strategies including transformer
  fine-tuning and text preprocessing.
- `../07_Computer_Vision/` -- CV-specific competition strategies including
  augmentation, TTA, and architecture selection.
- `../08_Time_Series/` -- Time series competition strategies including lag
  features, temporal validation, and forecasting ensembles.
