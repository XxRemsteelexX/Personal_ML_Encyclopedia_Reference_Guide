# Bias Detection and Fairness Metrics - Complete Technical Guide

## Overview

**Algorithmic bias** leads to discriminatory outcomes. EU AI Act Article 10 and GDPR Article 5 require bias detection and mitigation for high-risk AI.

**This guide covers:**
- Types of bias (data, algorithmic, deployment)
- Fairness metrics (20+ with code)
- Detection methods
- Mitigation techniques
- Production implementation

---

## Types of Bias

### 1. Data Bias

**Historical Bias** - Past discrimination reflected in data
```python
# Example: Historical hiring data shows 90% male engineers
# Model learns this as "correct" pattern
historical_data = {
    'male_engineers': 0.90,
    'female_engineers': 0.10
}
# [WARNING] Model will replicate historical discrimination
```

**Representation Bias** - Some groups underrepresented
```python
import pandas as pd

def detect_representation_bias(df, protected_attribute, min_representation=0.1):
    """
    Check if all groups are adequately represented
    """
    distribution = df[protected_attribute].value_counts(normalize=True)

    underrepresented = distribution[distribution < min_representation]

    if len(underrepresented) > 0:
        print(f"[WARNING] Representation Bias Detected in {protected_attribute}:")
        for group, pct in underrepresented.items():
            print(f"  - {group}: {pct:.1%} (below {min_representation:.0%} threshold)")

    return underrepresented

# Usage
bias_check = detect_representation_bias(df, 'race', min_representation=0.05)
```

**Measurement Bias** - Proxy variables correlate with protected attributes
```python
# Example: ZIP code highly correlated with race
correlation_matrix = df[['zip_code', 'race_encoded']].corr()
if abs(correlation_matrix.iloc[0, 1]) > 0.7:
    print("[WARNING] Measurement Bias: ZIP code is proxy for race")
```

**Aggregation Bias** - One model doesn't fit all subgroups
```python
from sklearn.metrics import mean_squared_error

def detect_aggregation_bias(model, X, y, group_col):
    """
    Check if model performance varies across groups
    """
    results = {}

    for group in X[group_col].unique():
        X_group = X[X[group_col] == group]
        y_group = y[X[group_col] == group]

        predictions = model.predict(X_group.drop(group_col, axis=1))
        mse = mean_squared_error(y_group, predictions)

        results[group] = mse

    # Flag if performance varies >20%
    max_mse = max(results.values())
    min_mse = min(results.values())

    if (max_mse - min_mse) / min_mse > 0.2:
        print(f"[WARNING] Aggregation Bias: Performance varies {((max_mse - min_mse) / min_mse):.0%} across {group_col}")

    return results
```

### 2. Algorithmic Bias

**Sample Bias** - Training/test split not stratified by protected attributes
```python
from sklearn.model_selection import train_test_split

#  WRONG - Random split might create bias
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  CORRECT - Stratify by protected attribute
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2,
    stratify=df['protected_attribute']
)
```

**Label Bias** - Biased labels from biased annotators
```python
def detect_label_bias(labels_df, annotator_col, protected_col, label_col):
    """
    Check if annotators show bias in labeling
    """
    # Compare labels across protected attributes for each annotator
    bias_report = {}

    for annotator in labels_df[annotator_col].unique():
        annotator_labels = labels_df[labels_df[annotator_col] == annotator]

        # Calculate positive label rate for each group
        positive_rates = annotator_labels.groupby(protected_col)[label_col].mean()

        # Check disparity
        max_rate = positive_rates.max()
        min_rate = positive_rates.min()
        disparity = (max_rate - min_rate) / min_rate

        if disparity > 0.2:  # >20% difference
            bias_report[annotator] = {
                'disparity': disparity,
                'rates': positive_rates.to_dict()
            }

    return bias_report
```

### 3. Deployment Bias

**Population Drift** - Deployed model sees different population
```python
def detect_population_drift(train_dist, production_dist, threshold=0.1):
    """
    Compare training vs production population distributions
    """
    drift_detected = {}

    for group in train_dist.keys():
        train_pct = train_dist.get(group, 0)
        prod_pct = production_dist.get(group, 0)

        change = abs(train_pct - prod_pct)

        if change > threshold:
            drift_detected[group] = {
                'train': train_pct,
                'production': prod_pct,
                'change': change
            }

    return drift_detected
```

---

## Fairness Metrics

### Individual Fairness

**"Similar individuals should be treated similarly"**

```python
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def measure_individual_fairness(X, predictions, k=10):
    """
    Check if similar individuals get similar predictions
    """
    # Compute similarity matrix
    similarity_matrix = cosine_similarity(X)

    fairness_violations = []

    for i in range(len(X)):
        # Find k nearest neighbors
        neighbors_idx = np.argsort(similarity_matrix[i])[-k-1:-1]

        # Compare predictions
        my_prediction = predictions[i]
        neighbor_predictions = predictions[neighbors_idx]

        # Check if predictions are consistent
        prediction_variance = np.var(neighbor_predictions)

        if prediction_variance > 0.1:  # High variance = unfair
            fairness_violations.append({
                'instance': i,
                'my_prediction': my_prediction,
                'neighbor_predictions': neighbor_predictions,
                'variance': prediction_variance
            })

    fairness_score = 1 - (len(fairness_violations) / len(X))
    return fairness_score, fairness_violations
```

---

### Group Fairness Metrics

#### 1. Statistical Parity (Demographic Parity)

**Definition:** P(Y_hat=1 | A=0) = P(Y_hat=1 | A=1)

"Positive prediction rate should be same across groups"

```python
def statistical_parity(y_pred, protected_attribute):
    """
    Calculate statistical parity difference
    """
    groups = protected_attribute.unique()

    positive_rates = {}
    for group in groups:
        group_mask = (protected_attribute == group)
        positive_rates[group] = y_pred[group_mask].mean()

    # Calculate maximum disparity
    max_rate = max(positive_rates.values())
    min_rate = min(positive_rates.values())

    sp_difference = max_rate - min_rate
    sp_ratio = min_rate / max_rate if max_rate > 0 else 0

    return {
        'sp_difference': sp_difference,
        'sp_ratio': sp_ratio,
        'rates': positive_rates,
        'fair': sp_difference < 0.1  # <10% difference
    }

# Usage
sp_metrics = statistical_parity(predictions, df['protected_attribute'])
print(f"Statistical Parity Difference: {sp_metrics['sp_difference']:.3f}")
print(f"Fair: {sp_metrics['fair']}")
```

**When to use:** When equal outcome is the goal (e.g., opportunity programs)

---

#### 2. Equal Opportunity

**Definition:** TPR(A=0) = TPR(A=1)

"Among qualified individuals, selection rate should be equal"

```python
def equal_opportunity(y_true, y_pred, protected_attribute):
    """
    Calculate Equal Opportunity difference
    """
    from sklearn.metrics import confusion_matrix

    groups = protected_attribute.unique()
    tpr_by_group = {}

    for group in groups:
        group_mask = (protected_attribute == group)

        tn, fp, fn, tp = confusion_matrix(
            y_true[group_mask],
            y_pred[group_mask]
        ).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tpr_by_group[group] = tpr

    # Calculate disparity
    max_tpr = max(tpr_by_group.values())
    min_tpr = min(tpr_by_group.values())

    eo_difference = max_tpr - min_tpr

    return {
        'eo_difference': eo_difference,
        'tpr_by_group': tpr_by_group,
        'fair': eo_difference < 0.1
    }

# Usage
eo_metrics = equal_opportunity(y_true, predictions, df['protected_attribute'])
print(f"Equal Opportunity Difference: {eo_metrics['eo_difference']:.3f}")
```

**When to use:** When false negatives are more concerning than false positives

---

#### 3. Equalized Odds

**Definition:** TPR(A=0) = TPR(A=1) AND FPR(A=0) = FPR(A=1)

"Both true positive and false positive rates should be equal"

```python
def equalized_odds(y_true, y_pred, protected_attribute):
    """
    Calculate Equalized Odds
    """
    from sklearn.metrics import confusion_matrix

    groups = protected_attribute.unique()
    metrics_by_group = {}

    for group in groups:
        group_mask = (protected_attribute == group)

        tn, fp, fn, tp = confusion_matrix(
            y_true[group_mask],
            y_pred[group_mask]
        ).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics_by_group[group] = {'tpr': tpr, 'fpr': fpr}

    # Calculate disparities
    tpr_values = [m['tpr'] for m in metrics_by_group.values()]
    fpr_values = [m['fpr'] for m in metrics_by_group.values()]

    tpr_diff = max(tpr_values) - min(tpr_values)
    fpr_diff = max(fpr_values) - min(fpr_values)

    equalized_odds_diff = max(tpr_diff, fpr_diff)

    return {
        'equalized_odds_diff': equalized_odds_diff,
        'tpr_diff': tpr_diff,
        'fpr_diff': fpr_diff,
        'metrics_by_group': metrics_by_group,
        'fair': equalized_odds_diff < 0.1
    }
```

**When to use:** When both false positives and false negatives matter equally

---

#### 4. Calibration

**Definition:** P(Y=1 | Y_hat=p, A=0) = P(Y=1 | Y_hat=p, A=1)

"Predicted probabilities should match actual outcomes across groups"

```python
def calibration_by_group(y_true, y_prob, protected_attribute, n_bins=10):
    """
    Check if model is calibrated across protected groups
    """
    from sklearn.calibration import calibration_curve

    groups = protected_attribute.unique()
    calibration_by_group = {}

    for group in groups:
        group_mask = (protected_attribute == group)

        # Get calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true[group_mask],
            y_prob[group_mask],
            n_bins=n_bins
        )

        # Calculate calibration error (ECE)
        ece = np.abs(fraction_of_positives - mean_predicted_value).mean()

        calibration_by_group[group] = {
            'ece': ece,
            'fraction_of_positives': fraction_of_positives,
            'mean_predicted_value': mean_predicted_value
        }

    # Check if calibration differs across groups
    ece_values = [m['ece'] for m in calibration_by_group.values()]
    ece_diff = max(ece_values) - min(ece_values)

    return {
        'calibration_diff': ece_diff,
        'calibration_by_group': calibration_by_group,
        'fair': ece_diff < 0.05
    }
```

**When to use:** When probability estimates are important (medical diagnosis, risk assessment)

---

#### 5. Predictive Parity

**Definition:** PPV(A=0) = PPV(A=1)

"Precision should be equal across groups"

```python
def predictive_parity(y_true, y_pred, protected_attribute):
    """
    Calculate Predictive Parity
    """
    from sklearn.metrics import precision_score

    groups = protected_attribute.unique()
    precision_by_group = {}

    for group in groups:
        group_mask = (protected_attribute == group)

        precision = precision_score(
            y_true[group_mask],
            y_pred[group_mask]
        )

        precision_by_group[group] = precision

    # Calculate disparity
    max_precision = max(precision_by_group.values())
    min_precision = min(precision_by_group.values())

    pp_difference = max_precision - min_precision

    return {
        'pp_difference': pp_difference,
        'precision_by_group': precision_by_group,
        'fair': pp_difference < 0.1
    }
```

**When to use:** When false positives are costly (e.g., fraud detection)

---

## Comprehensive Fairness Audit

```python
class FairnessAuditor:
    """
    Comprehensive fairness evaluation
    """

    def __init__(self, y_true, y_pred, y_prob, protected_attributes):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.protected_attributes = protected_attributes

    def audit_all_metrics(self):
        """
        Calculate all fairness metrics
        """
        results = {}

        for attr_name, attr_values in self.protected_attributes.items():
            results[attr_name] = {
                'statistical_parity': statistical_parity(self.y_pred, attr_values),
                'equal_opportunity': equal_opportunity(self.y_true, self.y_pred, attr_values),
                'equalized_odds': equalized_odds(self.y_true, self.y_pred, attr_values),
                'calibration': calibration_by_group(self.y_true, self.y_prob, attr_values),
                'predictive_parity': predictive_parity(self.y_true, self.y_pred, attr_values)
            }

        return results

    def generate_report(self):
        """
        Generate comprehensive fairness report
        """
        audit_results = self.audit_all_metrics()

        report = "="*60 + "\n"
        report += "FAIRNESS AUDIT REPORT\n"
        report += "="*60 + "\n\n"

        for attr_name, metrics in audit_results.items():
            report += f"Protected Attribute: {attr_name}\n"
            report += "-"*60 + "\n"

            for metric_name, metric_result in metrics.items():
                fair_status = "[x] FAIR" if metric_result.get('fair', False) else "[ ] BIASED"
                report += f"{metric_name:25} {fair_status}\n"

                # Add key metric value
                if 'sp_difference' in metric_result:
                    report += f"  Difference: {metric_result['sp_difference']:.3f}\n"
                elif 'eo_difference' in metric_result:
                    report += f"  TPR Difference: {metric_result['eo_difference']:.3f}\n"
                elif 'equalized_odds_diff' in metric_result:
                    report += f"  Max Difference: {metric_result['equalized_odds_diff']:.3f}\n"

            report += "\n"

        return report

# Usage
auditor = FairnessAuditor(
    y_true=y_test,
    y_pred=predictions,
    y_prob=probabilities,
    protected_attributes={'gender': df['gender'], 'race': df['race']}
)

print(auditor.generate_report())
```

---

## Bias Mitigation Techniques

### Pre-Processing (Data-Level)

#### 1. Reweighting

```python
from sklearn.utils.class_weight import compute_sample_weight

def reweight_data(X, y, protected_attribute):
    """
    Reweight samples to achieve demographic parity
    """
    # Compute weights to balance protected groups
    sample_weights = compute_sample_weight(
        class_weight='balanced',
        y=protected_attribute
    )

    return sample_weights

# Usage with model
sample_weights = reweight_data(X_train, y_train, train_df['protected_attr'])
model.fit(X_train, y_train, sample_weight=sample_weights)
```

#### 2. Resampling

```python
from imblearn.over_sampling import SMOTE

def fair_resampling(X, y, protected_attribute):
    """
    Oversample minority groups
    """
    # Combine protected attribute with features for stratified sampling
    X_combined = X.copy()
    X_combined['protected_attr'] = protected_attribute

    # Apply SMOTE stratified by protected attribute
    smote = SMOTE(sampling_strategy='auto', random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)

    protected_resampled = X_resampled['protected_attr']
    X_resampled = X_resampled.drop('protected_attr', axis=1)

    return X_resampled, y_resampled, protected_resampled
```

---

### In-Processing (Algorithm-Level)

#### 1. Adversarial Debiasing

```python
import tensorflow as tf
from tensorflow.keras import layers

class AdversarialDebiasing:
    """
    Train predictor while preventing adversary from predicting protected attribute
    """

    def __init__(self, input_dim, protected_dim):
        # Main predictor
        self.predictor = tf.keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid', name='prediction')
        ])

        # Adversary (tries to predict protected attribute from predictor's hidden layer)
        self.adversary = tf.keras.Sequential([
            layers.Dense(32, activation='relu'),
            layers.Dense(protected_dim, activation='softmax', name='adversary_pred')
        ])

    def train(self, X, y, protected_attr, epochs=50, lambda_adv=1.0):
        """
        Train with adversarial objective
        """
        for epoch in range(epochs):
            with tf.GradientTape(persistent=True) as tape:
                # Forward pass
                hidden = self.predictor.layers[-2](X)
                pred_output = self.predictor.layers[-1](hidden)

                # Predictor loss
                pred_loss = tf.keras.losses.binary_crossentropy(y, pred_output)

                # Adversary loss (tries to predict protected attribute)
                adv_output = self.adversary(hidden)
                adv_loss = tf.keras.losses.categorical_crossentropy(protected_attr, adv_output)

                # Combined loss (minimize prediction loss, maximize adversary loss)
                total_loss = pred_loss - lambda_adv * adv_loss

            # Update predictor to minimize total loss
            predictor_grads = tape.gradient(total_loss, self.predictor.trainable_variables)
            optimizer.apply_gradients(zip(predictor_grads, self.predictor.trainable_variables))

            # Update adversary to minimize adversary loss
            adversary_grads = tape.gradient(adv_loss, self.adversary.trainable_variables)
            adversary_optimizer.apply_gradients(zip(adversary_grads, self.adversary.trainable_variables))
```

#### 2. Fairness Constraints

```python
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

def train_with_fairness_constraints(X, y, protected_attribute):
    """
    Train model with fairness constraints
    """
    from sklearn.linear_model import LogisticRegression

    # Base estimator
    base_model = LogisticRegression()

    # Fairness constraint
    constraint = DemographicParity()

    # Mitigator
    mitigator = ExponentiatedGradient(
        base_model,
        constraint,
        sample_weight_name='sample_weight'
    )

    # Train
    mitigator.fit(X, y, sensitive_features=protected_attribute)

    return mitigator
```

---

### Post-Processing (Output-Level)

#### 1. Threshold Optimization

```python
def optimize_thresholds_for_fairness(y_true, y_prob, protected_attribute, fairness_metric='equalized_odds'):
    """
    Find group-specific thresholds that achieve fairness
    """
    from sklearn.metrics import roc_curve

    groups = protected_attribute.unique()
    optimal_thresholds = {}

    for group in groups:
        group_mask = (protected_attribute == group)

        # Get ROC curve for this group
        fpr, tpr, thresholds = roc_curve(y_true[group_mask], y_prob[group_mask])

        # Find threshold that optimizes fairness metric
        if fairness_metric == 'equalized_odds':
            # Minimize difference in TPR and FPR across groups
            # (simplified - would need to compare across all groups)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_thresholds[group] = thresholds[optimal_idx]

    return optimal_thresholds

# Usage
thresholds = optimize_thresholds_for_fairness(y_true, y_prob, df['protected_attr'])

# Apply group-specific thresholds
def predict_with_fair_thresholds(y_prob, protected_attribute, thresholds):
    predictions = np.zeros(len(y_prob))

    for group, threshold in thresholds.items():
        group_mask = (protected_attribute == group)
        predictions[group_mask] = (y_prob[group_mask] >= threshold).astype(int)

    return predictions
```

---

## Production Fairness Monitoring

```python
class FairnessMonitor:
    """
    Monitor fairness metrics in production
    """

    def __init__(self, fairness_thresholds):
        self.thresholds = fairness_thresholds
        self.alerts = []

    def check_fairness(self, y_true, y_pred, y_prob, protected_attributes):
        """
        Check if fairness metrics are within acceptable bounds
        """
        auditor = FairnessAuditor(y_true, y_pred, y_prob, protected_attributes)
        current_metrics = auditor.audit_all_metrics()

        violations = []

        for attr_name, metrics in current_metrics.items():
            for metric_name, result in metrics.items():
                if not result.get('fair', False):
                    violations.append({
                        'attribute': attr_name,
                        'metric': metric_name,
                        'result': result,
                        'timestamp': pd.Timestamp.now()
                    })

                    # Trigger alert
                    self.trigger_alert(attr_name, metric_name, result)

        return violations

    def trigger_alert(self, attribute, metric, result):
        """
        Alert when fairness violation detected
        """
        alert = {
            'timestamp': pd.Timestamp.now(),
            'attribute': attribute,
            'metric': metric,
            'severity': 'HIGH' if result.get('difference', 0) > 0.2 else 'MEDIUM',
            'details': result
        }

        self.alerts.append(alert)

        # In production: send to monitoring system
        print(f"[WARNING] FAIRNESS ALERT: {metric} violation for {attribute}")
        print(f"   Severity: {alert['severity']}")
        print(f"   Details: {result}")

# Usage
monitor = FairnessMonitor(fairness_thresholds={'sp_difference': 0.1})

# In production loop
violations = monitor.check_fairness(
    y_true=recent_labels,
    y_pred=recent_predictions,
    y_prob=recent_probabilities,
    protected_attributes={'gender': recent_gender, 'race': recent_race}
)

if violations:
    print(f"Detected {len(violations)} fairness violations")
    # Trigger model retraining, human review, etc.
```

---

## Key Takeaways

1. **Multiple fairness definitions exist** - Choose based on use case
2. **Fairness-accuracy trade-off** - Often can't maximize both
3. **Intersectionality matters** - Check combinations of protected attributes
4. **Continuous monitoring required** - Fairness can degrade over time
5. **Documentation essential** - EU AI Act requires bias mitigation documentation
6. **Pre-processing often easiest** - Fixing data is simpler than fixing algorithms
7. **No universal solution** - Context determines appropriate fairness metric

**Next:** `03_Privacy_Preserving_ML.md` - Federated Learning and Differential Privacy implementation
