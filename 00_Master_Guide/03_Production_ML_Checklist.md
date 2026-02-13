# Production ML Checklist

A comprehensive, opinionated set of checklists and templates for taking machine learning
models from prototype to production and keeping them healthy once deployed. Inspired by
Google's seminal paper *Hidden Technical Debt in Machine Learning Systems* (Sculley et al.,
2015), which demonstrated that the actual ML code in a production system is a surprisingly
small fraction of the total infrastructure required.

## Table of Contents

- [Introduction](#introduction)
- [Pre-Development Checklist](#pre-development-checklist)
- [Data Pipeline Checklist](#data-pipeline-checklist)
- [Model Development Checklist](#model-development-checklist)
- [Model Card Template](#model-card-template)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [A/B Testing Setup](#ab-testing-setup)
- [Monitoring Checklist](#monitoring-checklist)
- [Alert Thresholds Guide](#alert-thresholds-guide)
- [Retraining Strategy](#retraining-strategy)
- [Incident Response](#incident-response)
- [Documentation Checklist](#documentation-checklist)
- [See Also](#see-also)

---

## Introduction

Most ML projects fail not because the model is bad, but because the system around the
model is incomplete. Checklists enforce discipline at every stage of the lifecycle and
prevent the slow accumulation of technical debt that eventually makes a system
unmaintainable.

**Why checklists matter:**

- **Reproducibility** -- Without documented steps, replicating results months later is
  nearly impossible. Checklists capture the implicit knowledge that lives in a
  practitioner's head.
- **Risk reduction** -- A missed validation step or an undocumented data dependency can
  cause silent failures in production that go undetected for weeks.
- **Team coordination** -- Checklists create a shared language between data scientists,
  ML engineers, and platform teams about what "production-ready" actually means.
- **Regulatory compliance** -- In regulated industries (finance, healthcare), auditable
  processes are not optional. Checklists provide the paper trail.

**Key insight from Google's ML Technical Debt paper:** The authors identified several
categories of hidden debt -- data dependencies, configuration debt, entanglement (changing
anything changes everything), undeclared consumers, and pipeline jungles. Every checklist
item below is designed to prevent or mitigate one of these debt categories.

**How to use this guide:**

1. Treat each checklist as a gate. Do not proceed to the next stage until every item is
   addressed (either completed or explicitly marked as not applicable with justification).
2. Copy the relevant checklist into your project tracker at the start of each phase.
3. Assign an owner to each item. Unowned items do not get done.
4. Review completed checklists with at least one peer before moving on.

---

## Pre-Development Checklist

Before writing any model code, establish the foundations. Skipping this phase is the
single most common cause of wasted effort in ML projects.

- [ ] **Problem is well-defined with clear success metrics**
  - Write a one-sentence problem statement
  - Define the primary metric (e.g., precision@k, RMSE, AUC-ROC)
  - Define at least one guardrail metric (a metric that must not degrade)
  - Specify the minimum acceptable performance threshold

- [ ] **Baseline performance established**
  - Implement a rule-based or heuristic baseline (e.g., most-frequent-class, moving average)
  - Implement a simple model baseline (e.g., logistic regression, gradient boosted trees)
  - Record baseline metrics -- these are your floor, not your ceiling
  - Document the baseline so future team members understand the starting point

- [ ] **Data availability and quality assessed**
  - Inventory all data sources (internal databases, third-party APIs, public datasets)
  - Measure data volume (is there enough for the chosen approach?)
  - Assess label quality (who labeled it, what was the inter-annotator agreement?)
  - Identify data freshness requirements (real-time, daily, weekly?)
  - Check for known biases in the data collection process

- [ ] **Ethical review completed**
  - Identify protected attributes in the data (race, gender, age, location)
  - Assess potential for disparate impact across subgroups
  - Review privacy implications (does the model memorize PII?)
  - Document intended use and foreseeable misuse cases
  - Consult with legal/compliance if operating in a regulated domain

- [ ] **Stakeholder expectations aligned**
  - Present baseline results to stakeholders before promising ML improvements
  - Agree on the definition of "good enough" performance
  - Clarify how the model output will be consumed (API, batch, embedded)
  - Set expectations about iteration timelines and uncertainty

- [ ] **Timeline and compute budget estimated**
  - Estimate training compute (GPU hours, cloud cost)
  - Estimate inference compute (requests per second, latency requirements)
  - Estimate storage requirements (training data, model artifacts, logs)
  - Build in buffer time for data issues (they always take longer than expected)
  - Identify long-lead items (data access approvals, infrastructure provisioning)

---

## Data Pipeline Checklist

Data pipelines are the foundation of every ML system. A flawed pipeline produces a flawed
model, and the model will not tell you the pipeline is broken.

- [ ] **Data schema documented and versioned**
  - Every column/field has a name, type, description, and allowed range
  - Schema changes are versioned (schema v1, v2, etc.)
  - Breaking schema changes trigger pipeline failures, not silent errors

- [ ] **Data validation rules defined**
  - Null/missing rate thresholds per column
  - Range checks for numeric fields
  - Categorical value checks (no unexpected categories)
  - Row count expectations (sudden drops indicate upstream issues)
  - Cross-column consistency checks (e.g., end_date >= start_date)

- [ ] **Missing value strategy documented**
  - For each feature, document: drop, impute (mean/median/mode), or flag
  - Justify each choice (not just "we used mean imputation everywhere")
  - Ensure imputation statistics are computed on training data only

- [ ] **Feature engineering reproducible**
  - All transformations expressed in code (no manual spreadsheet steps)
  - Feature computation logic shared between training and serving
  - Feature store or shared library to prevent training-serving skew

- [ ] **Train/test split documented**
  - Split strategy documented (random, temporal, stratified, group-based)
  - No data leakage verified (future data not leaking into training)
  - Hold-out test set created and locked (not used during development)
  - If time-series: split is strictly temporal

- [ ] **Data versioning in place**
  - Training datasets are versioned and immutable once created
  - Tool in use (DVC, Delta Lake, LakeFS, or cloud-native versioning)
  - Every model artifact is linked to the exact dataset version it was trained on

- [ ] **Sensitive data handling**
  - PII identified and either removed or encrypted
  - Access controls in place (not everyone needs access to raw data)
  - Data retention policies documented and enforced
  - Anonymization or pseudonymization applied where required

**Data validation template using Great Expectations:**

```python
import great_expectations as gx

context = gx.get_context()

# Define expectations for a training dataset
validator = context.sources.pandas_default.read_csv(
    "training_data.csv"
)

# Schema and completeness checks
validator.expect_table_columns_to_match_ordered_list(
    column_list=["user_id", "feature_1", "feature_2", "label"]
)
validator.expect_column_values_to_not_be_null("user_id")
validator.expect_column_values_to_not_be_null("label")

# Range checks
validator.expect_column_values_to_be_between(
    "feature_1", min_value=0.0, max_value=1.0
)
validator.expect_column_values_to_be_in_set(
    "label", value_set=[0, 1]
)

# Distribution checks
validator.expect_column_mean_to_be_between(
    "feature_1", min_value=0.3, max_value=0.7
)
validator.expect_column_proportion_of_unique_values_to_be_between(
    "user_id", min_value=0.99, max_value=1.0
)

# Row count sanity check
validator.expect_table_row_count_to_be_between(
    min_value=10000, max_value=10000000
)

results = validator.validate()
if not results.success:
    raise ValueError(f"Data validation failed: {results}")
```

---

## Model Development Checklist

Systematic model development avoids the trap of endlessly tuning a single architecture
while ignoring more impactful changes (better data, simpler model, different framing).

- [ ] **Multiple model families evaluated**
  - Test at least 2-3 fundamentally different approaches (e.g., linear, tree-based, neural)
  - Compare on the same validation set with the same metric
  - Document why the selected model was chosen (not just accuracy -- consider latency,
    interpretability, maintenance burden)

- [ ] **Hyperparameter tuning with proper cross-validation**
  - Use k-fold CV (typically k=5) or time-series CV for temporal data
  - Search strategy documented (grid, random, Bayesian)
  - Tuning performed on validation folds only, never on the test set
  - Best hyperparameters recorded and committed to version control

- [ ] **Overfitting checked**
  - Training metric vs. validation metric gap is small and stable
  - Learning curves plotted (performance vs. training set size)
  - Regularization applied if gap is significant
  - Model complexity justified relative to dataset size

- [ ] **Feature importance analyzed**
  - SHAP values or permutation importance computed
  - Top features reviewed for plausibility (no leakage signals)
  - Features contributing near-zero importance considered for removal
  - Feature interactions examined for the most important features

- [ ] **Model interpretability documented**
  - Global explanations: which features matter overall
  - Local explanations: why did the model make this specific prediction
  - Edge case explanations: what happens at the boundaries
  - Tool used documented (SHAP, LIME, InterpretML, built-in feature importance)

- [ ] **Error analysis performed**
  - Confusion matrix or residual analysis by segment
  - Worst-performing subgroups identified
  - Common failure modes categorized (e.g., "fails on short text", "fails on new users")
  - Targeted improvements planned for the most impactful failure modes

- [ ] **Fairness metrics computed across subgroups**
  - Define protected groups relevant to the problem
  - Compute metrics per group (e.g., TPR, FPR, precision by demographic)
  - Check for disparate impact (80% rule or statistical parity)
  - Document any acceptable trade-offs and their justification

- [ ] **Model card created**
  - See the Model Card Template section below

---

## Model Card Template

Model cards (Mitchell et al., 2019) provide structured documentation that travels with
the model artifact. Every model deployed to production must have a model card.

**Minimum required fields:**

| Field | Description |
|---|---|
| Model Name | Human-readable identifier |
| Version | Semantic version (e.g., 2.1.0) |
| Date | Training completion date |
| Owner | Team or individual responsible |
| Intended Use | What the model is designed for |
| Out-of-Scope Use | What the model should NOT be used for |
| Training Data | Description, version, size, date range |
| Evaluation Data | Description, version, size, date range |
| Metrics | Primary and secondary metrics on eval data |
| Subgroup Performance | Metrics broken down by relevant subgroups |
| Ethical Considerations | Known biases, limitations, risks |
| Caveats and Recommendations | Conditions where performance degrades |
| Maintenance Plan | Retraining schedule, monitoring owner |

**Model card YAML template:**

```yaml
model_card:
  model_name: "customer_churn_predictor"
  version: "2.1.0"
  date: "2025-03-15"
  owner: "ml-platform-team"
  contact: "ml-team@company.com"

  intended_use:
    primary_use: "Predict 30-day churn probability for active subscribers"
    primary_users: "Retention marketing team"
    out_of_scope:
      - "Predicting churn for trial users (not in training data)"
      - "Predicting churn beyond 30-day window"
      - "Individual-level causal inference (this is correlational)"

  training_data:
    description: "12 months of subscriber activity data"
    version: "dataset-v3.2"
    size: "2.4M rows, 847K unique users"
    date_range: "2024-03-01 to 2025-02-28"
    known_issues:
      - "Under-represents users who joined in last 30 days"
      - "Geographic skew toward US users (78% of data)"

  evaluation_data:
    description: "Hold-out set from same distribution"
    version: "dataset-v3.2-test"
    size: "600K rows, 212K unique users"
    date_range: "2025-01-01 to 2025-02-28"

  metrics:
    primary:
      - metric: "AUC-ROC"
        value: 0.847
      - metric: "Precision@10%"
        value: 0.62
    secondary:
      - metric: "Recall@10%"
        value: 0.38
      - metric: "Brier Score"
        value: 0.089

  subgroup_performance:
    - group: "tenure < 6 months"
      auc_roc: 0.791
      note: "Lower performance for newer users"
    - group: "tenure >= 6 months"
      auc_roc: 0.872
    - group: "US users"
      auc_roc: 0.851
    - group: "Non-US users"
      auc_roc: 0.823

  ethical_considerations:
    - "Model uses behavioral features only; no demographic features"
    - "Risk of feedback loop: targeted retention offers may change behavior"
    - "Users are not informed that their churn risk is being predicted"

  maintenance:
    retraining_schedule: "Monthly"
    drift_monitoring: "Weekly PSI check on top 10 features"
    owner: "ml-platform-team"
    escalation: "ml-oncall@company.com"
```

---

## Pre-Deployment Checklist

The gap between "works on my laptop" and "works in production" is where most ML projects
die. This checklist bridges that gap.

- [ ] **Model serialized in portable format**
  - Format chosen based on serving infrastructure (ONNX, TorchScript, SavedModel, pickle)
  - Serialized model tested: load it in a clean environment and verify predictions match
  - Model artifact includes version, training date, and dataset version in metadata
  - Artifact stored in a model registry (MLflow, Weights & Biases, SageMaker, or similar)

- [ ] **Inference latency tested**
  - p50, p95, and p99 latency measured under realistic conditions
  - Latency meets SLA requirements (document the SLA)
  - Latency tested with realistic input sizes (not just toy examples)
  - Batch vs. real-time inference decision documented

- [ ] **Memory footprint measured**
  - Model size on disk documented
  - Peak memory usage during inference measured
  - Memory usage stable over time (no memory leaks over thousands of requests)
  - Fits within the allocated container/instance resources

- [ ] **API endpoint tested with edge cases**
  - Empty input, null values, extremely long input
  - Out-of-distribution input (values outside training range)
  - Malformed requests (wrong types, missing fields)
  - Concurrent requests (race conditions)
  - All edge cases return appropriate error codes, not crashes

- [ ] **Load testing completed**
  - Sustained load at expected QPS for at least 10 minutes
  - Spike load at 3-5x expected QPS
  - Measure degradation curve (at what QPS does latency become unacceptable?)
  - Auto-scaling tested if applicable

- [ ] **Rollback plan documented**
  - Previous model version available and tested
  - Rollback can be executed in under 5 minutes
  - Rollback procedure does not require code deployment (config change or model swap)
  - Rollback has been tested at least once (not just documented)

- [ ] **Logging configured**
  - Every prediction logged: input features, output, model version, timestamp
  - Latency logged per request
  - Errors logged with full stack traces
  - Log volume estimated and storage planned
  - Sampling strategy defined if full logging is too expensive

- [ ] **A/B test framework ready**
  - Traffic splitting mechanism in place
  - Metrics pipeline connected to the experiment platform
  - See A/B Testing Setup section for details

**FastAPI deployment template:**

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, validator
import numpy as np
import joblib
import time
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Service",
    version="2.1.0",
    description="Customer churn prediction API",
)

# Load model at startup
MODEL_PATH = os.getenv("MODEL_PATH", "model_v2.1.0.joblib")
model = joblib.load(MODEL_PATH)
MODEL_VERSION = "2.1.0"


class PredictionRequest(BaseModel):
    user_id: str
    features: list[float]

    @validator("features")
    def validate_features(cls, v):
        if len(v) != 15:
            raise ValueError(f"Expected 15 features, got {len(v)}")
        return v


class PredictionResponse(BaseModel):
    user_id: str
    prediction: float
    model_version: str
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": MODEL_VERSION}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    start_time = time.time()

    try:
        features = np.array(request.features).reshape(1, -1)
        prediction = float(model.predict_proba(features)[0, 1])
    except Exception as e:
        logger.error(f"Prediction failed for user {request.user_id}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

    latency_ms = (time.time() - start_time) * 1000

    logger.info(
        f"prediction user_id={request.user_id} "
        f"score={prediction:.4f} "
        f"model={MODEL_VERSION} "
        f"latency_ms={latency_ms:.1f}"
    )

    return PredictionResponse(
        user_id=request.user_id,
        prediction=prediction,
        model_version=MODEL_VERSION,
        latency_ms=round(latency_ms, 1),
    )


@app.post("/predict_batch")
def predict_batch(requests: list[PredictionRequest]):
    start_time = time.time()

    features = np.array([r.features for r in requests])
    predictions = model.predict_proba(features)[:, 1].tolist()

    latency_ms = (time.time() - start_time) * 1000
    logger.info(
        f"batch_prediction count={len(requests)} "
        f"model={MODEL_VERSION} "
        f"latency_ms={latency_ms:.1f}"
    )

    return [
        PredictionResponse(
            user_id=r.user_id,
            prediction=round(p, 4),
            model_version=MODEL_VERSION,
            latency_ms=round(latency_ms / len(requests), 1),
        )
        for r, p in zip(requests, predictions)
    ]
```

---

## A/B Testing Setup

A/B testing is how you prove that your model actually improves the business metric, not
just the offline metric. Without it, you are guessing.

### Traffic Splitting Strategy

| Strategy | When to Use | Considerations |
|---|---|---|
| Random per-request | Stateless predictions (search ranking) | Simplest; user may see inconsistent results |
| User-based hash | Personalization models | Consistent experience per user; use stable hash |
| Geographic | Region-specific models | Useful for localization; confounded by regional differences |
| Time-based | When user bucketing is impractical | Beware of time-of-day and day-of-week effects |
| Stratified | When subgroup balance matters | More complex setup; better statistical properties |

### Minimum Sample Size Calculation

Before starting the experiment, compute the required sample size. Running an experiment
that is too small wastes time; running one that is too large wastes traffic.

```
n = (Z_alpha/2 + Z_beta)^2 * (p1*(1-p1) + p2*(1-p2)) / (p1 - p2)^2

Where:
  Z_alpha/2 = 1.96 for 95% confidence
  Z_beta    = 0.84 for 80% power
  p1        = baseline conversion rate
  p2        = expected conversion rate with new model
```

**Rule of thumb:** For a 1% absolute lift on a 10% baseline rate, you need approximately
15,000 samples per group.

### Metrics Framework

| Category | Examples | Purpose |
|---|---|---|
| Primary metric | Conversion rate, revenue per user, CTR | The metric the experiment is designed to improve |
| Guardrail metrics | Latency, error rate, user complaints | Metrics that must NOT degrade |
| Diagnostic metrics | Feature distributions, prediction distributions | Help debug unexpected results |

### Duration Estimation

- Minimum: 1 full business cycle (usually 1 week to capture day-of-week effects)
- Recommended: 2-4 weeks for stable estimates
- Never peek at results and stop early when significant (this inflates false positive rate)
- Use sequential testing methods if early stopping is required

### Decision Framework

| Outcome | Primary Metric | Guardrails | Decision |
|---|---|---|---|
| Significant positive | Improved | Held | Ship the new model |
| Significant positive | Improved | Degraded | Investigate; do not ship without fixing guardrails |
| Not significant | Flat | Held | Extend experiment or iterate on model |
| Significant negative | Degraded | Any | Kill the experiment; analyze what went wrong |

**A/B test analysis template:**

```python
import numpy as np
from scipy import stats


def ab_test_analysis(
    control_conversions: int,
    control_total: int,
    treatment_conversions: int,
    treatment_total: int,
    alpha: float = 0.05,
) -> dict:
    """
    Analyze an A/B test with a binary outcome (e.g., conversion).

    Returns a dictionary with rates, lift, confidence interval,
    p-value, and a human-readable recommendation.
    """
    p_control = control_conversions / control_total
    p_treatment = treatment_conversions / treatment_total

    # Absolute and relative lift
    absolute_lift = p_treatment - p_control
    relative_lift = absolute_lift / p_control if p_control > 0 else float("inf")

    # Pooled standard error
    p_pooled = (control_conversions + treatment_conversions) / (
        control_total + treatment_total
    )
    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/control_total + 1/treatment_total))

    # Z-test
    z_stat = absolute_lift / se if se > 0 else 0
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    # Confidence interval for the difference
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_lower = absolute_lift - z_crit * se
    ci_upper = absolute_lift + z_crit * se

    # Decision
    if p_value < alpha and absolute_lift > 0:
        recommendation = "SHIP: Statistically significant positive result"
    elif p_value < alpha and absolute_lift < 0:
        recommendation = "KILL: Statistically significant negative result"
    else:
        recommendation = "INCONCLUSIVE: Not enough evidence to decide"

    return {
        "control_rate": round(p_control, 4),
        "treatment_rate": round(p_treatment, 4),
        "absolute_lift": round(absolute_lift, 4),
        "relative_lift_pct": round(relative_lift * 100, 2),
        "p_value": round(p_value, 4),
        "confidence_interval": (round(ci_lower, 4), round(ci_upper, 4)),
        "significant": p_value < alpha,
        "recommendation": recommendation,
    }


# Example usage
results = ab_test_analysis(
    control_conversions=450,
    control_total=5000,
    treatment_conversions=510,
    treatment_total=5000,
    alpha=0.05,
)

for key, value in results.items():
    print(f"{key}: {value}")
```

---

## Monitoring Checklist

A model without monitoring is a model you cannot trust. Production models degrade
silently. By the time someone notices, the damage is done.

- [ ] **Data drift detection**
  - Population Stability Index (PSI) computed for each input feature
  - Kolmogorov-Smirnov test for continuous features
  - Chi-squared test for categorical features
  - Jensen-Shannon divergence as a symmetric alternative to KL divergence
  - Reference distribution stored from training data

- [ ] **Prediction drift monitoring**
  - Distribution of model outputs tracked over time
  - Sudden shifts in prediction mean or variance trigger alerts
  - Useful even when ground truth labels are delayed

- [ ] **Feature distribution monitoring**
  - Summary statistics (mean, std, min, max, null rate) tracked per feature
  - New categories in categorical features detected
  - Correlation structure between features monitored (entanglement detection)

- [ ] **Model performance tracking**
  - When ground truth is available: compute primary metric on a rolling basis
  - When ground truth is delayed: use proxy metrics or prediction confidence
  - Performance tracked per subgroup, not just in aggregate

- [ ] **Latency and throughput monitoring**
  - p50, p95, p99 latency tracked per endpoint
  - Requests per second tracked
  - Queue depth tracked if using async inference

- [ ] **Error rate tracking**
  - HTTP 5xx rate tracked
  - Model-specific errors tracked (input validation failures, NaN predictions)
  - Upstream dependency failures tracked

- [ ] **Alert thresholds defined**
  - See Alert Thresholds Guide below
  - Every alert has a documented response procedure
  - Alerts are tested periodically (fire drill)

- [ ] **Escalation procedures documented**
  - On-call rotation defined
  - Escalation path: on-call -> team lead -> engineering manager
  - Communication channels defined (Slack channel, PagerDuty, email)

**Drift detection example using Evidently:**

```python
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DataDriftTable,
    ColumnDriftMetric,
)


def detect_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    feature_columns: list[str],
    drift_threshold: float = 0.05,
) -> dict:
    """
    Compare current production data against training reference data.

    Uses Evidently to compute per-feature drift scores and an
    overall dataset drift assessment.
    """
    report = Report(
        metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ]
    )
    report.run(
        reference_data=reference_data[feature_columns],
        current_data=current_data[feature_columns],
    )

    result = report.as_dict()

    # Extract per-feature drift results
    drift_results = {}
    dataset_drift = result["metrics"][0]["result"]["dataset_drift"]
    drift_table = result["metrics"][1]["result"]["drift_by_columns"]

    for col_name, col_result in drift_table.items():
        drift_results[col_name] = {
            "drift_detected": col_result["drift_detected"],
            "drift_score": col_result["drift_score"],
            "stattest_name": col_result["stattest_name"],
        }

    return {
        "dataset_drift_detected": dataset_drift,
        "drifted_feature_count": sum(
            1 for v in drift_results.values() if v["drift_detected"]
        ),
        "total_features": len(drift_results),
        "per_feature": drift_results,
    }


# Example usage
reference = pd.read_csv("training_data.csv")
current = pd.read_csv("last_7_days_production.csv")

features = ["feature_1", "feature_2", "feature_3", "feature_4"]
drift = detect_drift(reference, current, features)

if drift["dataset_drift_detected"]:
    print(f"ALERT: Dataset drift detected!")
    print(f"  Drifted features: {drift['drifted_feature_count']}/{drift['total_features']}")
    for feat, info in drift["per_feature"].items():
        if info["drift_detected"]:
            print(f"  - {feat}: score={info['drift_score']:.4f} ({info['stattest_name']})")
```

---

## Alert Thresholds Guide

Thresholds should be calibrated to your system. The values below are reasonable defaults
for most ML systems. Adjust based on your domain and risk tolerance.

### Data Drift Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| PSI (per feature) | > 0.1 | > 0.25 | Warning: investigate. Critical: halt predictions, retrain. |
| KS statistic | > 0.05 | > 0.1 | Warning: monitor closely. Critical: investigate data source. |
| JS divergence | > 0.05 | > 0.1 | Same as KS. |
| Null rate increase | > 2x baseline | > 5x baseline | Check upstream data pipeline. |
| New categories | Any new category | > 5% of traffic | Update feature encoding, retrain if significant. |

### Model Performance Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| Primary metric degradation | > 5% relative drop | > 10% relative drop | Warning: investigate root cause. Critical: rollback. |
| Prediction mean shift | > 2 std from baseline | > 3 std from baseline | Check for data or concept drift. |
| Prediction variance change | > 50% increase | > 100% increase | Model confidence is unstable. |
| Calibration drift | Brier score +0.02 | Brier score +0.05 | Recalibrate or retrain. |

### Infrastructure Thresholds

| Metric | Warning | Critical | Action |
|---|---|---|---|
| p99 latency | > 2x baseline | > 5x baseline | Scale up or optimize model. |
| p50 latency | > 1.5x baseline | > 3x baseline | Investigate resource contention. |
| Error rate (5xx) | > 1% | > 5% | Warning: investigate. Critical: page on-call. |
| Memory usage | > 80% of limit | > 95% of limit | Scale up or investigate memory leak. |
| CPU usage | > 70% sustained | > 90% sustained | Scale out or optimize. |
| Queue depth | > 100 messages | > 1000 messages | Consumer is falling behind; scale or shed load. |

### PSI Reference Table

Population Stability Index (PSI) is the most common drift metric. Interpretation:

| PSI Value | Interpretation |
|---|---|
| < 0.1 | No significant change |
| 0.1 - 0.25 | Moderate change, investigate |
| > 0.25 | Significant change, action required |

```
PSI = SUM( (actual_pct - expected_pct) * ln(actual_pct / expected_pct) )

Where:
  actual_pct   = proportion in each bin for current data
  expected_pct = proportion in each bin for reference data
  Bins are typically 10 equal-width or equal-frequency bins
```

---

## Retraining Strategy

Models decay. The world changes, user behavior shifts, and data distributions evolve.
A retraining strategy keeps your model relevant.

### Scheduled Retraining

- **Weekly:** Appropriate for fast-moving domains (ad click prediction, fraud detection)
- **Monthly:** Appropriate for moderately stable domains (churn prediction, demand forecasting)
- **Quarterly:** Appropriate for slow-moving domains (credit scoring, medical diagnosis)
- **Key principle:** Automate everything. If retraining requires manual steps, it will
  eventually stop happening.

### Triggered Retraining

Retrain when monitoring detects drift, rather than on a fixed schedule. This avoids
unnecessary retraining when the world is stable and ensures fast response when it changes.

**Triggers:**
- PSI exceeds critical threshold on multiple features
- Primary metric drops below acceptable threshold
- Significant new data source becomes available
- Business logic changes (new product categories, new user segments)

### Champion-Challenger Framework

Always compare the new model (challenger) against the current model (champion) before
promoting.

```
1. Train challenger model on latest data
2. Evaluate challenger on held-out test set
3. Compare challenger vs champion on same test set
4. If challenger wins on primary metric AND does not degrade guardrails:
   a. Deploy challenger to shadow mode (receives traffic, predictions not served)
   b. Monitor shadow predictions for 1-7 days
   c. If stable, promote challenger to A/B test
   d. If A/B test positive, promote to champion
5. If challenger loses, investigate why and iterate
```

### Deployment Patterns

**Shadow deployment:**
- New model receives production traffic but predictions are not served to users
- Compare predictions against the current model
- Zero risk to users; full exposure to production data
- Use when: validating a new model architecture or a major retraining

**Blue-green deployment:**
- Two identical production environments: blue (current) and green (new)
- Switch traffic from blue to green atomically
- Instant rollback by switching back to blue
- Use when: deploying a tested model with high confidence

**Canary deployment:**
- Route a small percentage of traffic (1-5%) to the new model
- Monitor metrics closely on the canary population
- Gradually increase traffic if metrics are healthy (5% -> 25% -> 50% -> 100%)
- Use when: deploying with moderate confidence; want gradual validation

```
Canary rollout schedule (example):

Day 1:   1% traffic  -> Monitor for 4 hours minimum
Day 1-2: 5% traffic  -> Monitor for 24 hours
Day 2-3: 25% traffic -> Monitor for 24 hours
Day 3-5: 50% traffic -> Monitor for 48 hours
Day 5:   100% traffic -> Full deployment
```

---

## Incident Response

When a model fails in production, speed and clarity matter. Having a pre-defined incident
response process prevents panic and ad-hoc decision making.

### Model Failure Classification

| Category | Examples | Typical Cause | First Response |
|---|---|---|---|
| Data issue | Missing features, schema change, null spike | Upstream pipeline failure | Check data sources, roll back data pipeline |
| Model degradation | Accuracy drop, prediction drift | Concept drift, data drift | Evaluate drift metrics, consider retraining |
| Infrastructure | High latency, OOM errors, service down | Resource limits, dependency failure | Scale up, restart, check dependencies |
| Integration | Wrong model version served, stale cache | Deployment error, config issue | Verify deployment, clear caches |

### Rollback Procedure

```
Step 1: DETECT
  - Alert fires (automated) or user reports issue (manual)
  - On-call acknowledges alert within 5 minutes

Step 2: ASSESS
  - Check monitoring dashboards (latency, error rate, prediction distribution)
  - Check recent deployments or config changes
  - Classify the failure (data, model, infra, integration)
  - Estimate user impact (how many users affected, how severely)

Step 3: MITIGATE
  - If model issue: roll back to previous model version
    Command: update model registry pointer to previous version
    Verify: check /health endpoint returns previous version
  - If data issue: switch to fallback data source or cached features
  - If infra issue: scale up, restart, or fail over to backup
  - Target: mitigation within 15 minutes of detection

Step 4: COMMUNICATE
  - Post in incident channel: what happened, what was done, current status
  - Update status page if user-facing impact
  - Notify stakeholders (product, business)

Step 5: RESOLVE
  - Root cause analysis (may take hours or days)
  - Fix the underlying issue
  - Deploy fix through normal process (not hotfix)
  - Verify fix with monitoring
```

### Post-Incident Review Template

Complete this template within 5 business days of every production incident:

```
INCIDENT REVIEW
===============
Date: YYYY-MM-DD
Severity: P1 / P2 / P3 / P4
Duration: X hours Y minutes (detection to resolution)

TIMELINE
--------
- HH:MM - Alert fired / issue reported
- HH:MM - On-call acknowledged
- HH:MM - Root cause identified
- HH:MM - Mitigation applied
- HH:MM - Full resolution confirmed

IMPACT
------
- Users affected: N
- Revenue impact: $X (estimated)
- Predictions affected: N
- Data loss: Yes / No

ROOT CAUSE
----------
[1-2 paragraph description of what went wrong and why]

WHAT WENT WELL
--------------
- [Monitoring detected the issue quickly]
- [Rollback worked as expected]
- [Team coordinated effectively]

WHAT WENT POORLY
----------------
- [Alert was noisy and initially ignored]
- [Rollback took longer than expected due to X]
- [Communication to stakeholders was delayed]

ACTION ITEMS
------------
- [ ] [Action item 1] - Owner: @name - Due: YYYY-MM-DD
- [ ] [Action item 2] - Owner: @name - Due: YYYY-MM-DD
- [ ] [Action item 3] - Owner: @name - Due: YYYY-MM-DD
```

### Communication Plan

| Audience | Channel | Timing | Content |
|---|---|---|---|
| Engineering team | Incident Slack channel | Immediately | Technical details, status updates |
| Product/business | Email or Slack | Within 30 minutes | Impact summary, ETA for resolution |
| End users (if applicable) | Status page | Within 1 hour | Plain-language description, workarounds |
| Leadership | Summary email | Within 24 hours | Impact, root cause, action items |

---

## Documentation Checklist

Documentation is not optional. Undocumented systems are unmaintainable systems. If the
person who built the model leaves, the documentation is all that remains.

- [ ] **README with setup instructions**
  - How to set up the development environment (dependencies, env vars)
  - How to run training
  - How to run inference locally
  - How to run tests
  - Tested on a clean machine by someone other than the author

- [ ] **Architecture diagram**
  - Data flow from source to model to serving
  - All external dependencies shown
  - Storage locations for data, models, and logs
  - Updated when architecture changes (stale diagrams are worse than none)

- [ ] **API documentation**
  - OpenAPI/Swagger spec generated from code (not hand-written)
  - Every endpoint documented with request/response examples
  - Error codes and their meanings documented
  - Rate limits and authentication documented
  - Versioning policy documented

- [ ] **Runbook for operations**
  - How to deploy a new model version
  - How to roll back
  - How to investigate common alerts
  - How to trigger manual retraining
  - How to access logs and monitoring dashboards
  - Contact information for on-call and escalation

- [ ] **Model card**
  - See Model Card Template section above
  - Stored alongside the model artifact in the model registry
  - Updated with every model version

- [ ] **Data dictionary**
  - Every feature: name, type, description, source, business meaning
  - Every label: definition, how it was collected, known issues
  - Updated when features are added or removed

- [ ] **Change log**
  - Every model version: what changed and why
  - Every data pipeline change: what changed and why
  - Every infrastructure change: what changed and why
  - Format: date, version, author, description of change

---

## See Also

- **MLOps** -- For detailed coverage of CI/CD pipelines for ML, infrastructure patterns,
  feature stores, model registries, and orchestration tools (Kubeflow, Airflow, Prefect).
- **Model Evaluation** -- For deep dives into evaluation metrics, cross-validation
  strategies, calibration, and statistical testing of model comparisons.
- **Model Selection Decision Tree** -- For guidance on choosing the right model family
  based on data characteristics, problem type, and constraints.

**Key references:**

- Sculley, D. et al. (2015). *Hidden Technical Debt in Machine Learning Systems.* NeurIPS.
- Mitchell, M. et al. (2019). *Model Cards for Model Reporting.* FAT* Conference.
- Breck, E. et al. (2017). *The ML Test Score: A Rubric for ML Production Readiness.* IEEE BigData.
- Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning: A Survey of Case Studies.* ACM Computing Surveys.
