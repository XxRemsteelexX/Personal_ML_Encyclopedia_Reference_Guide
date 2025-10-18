# Monitoring and Drift Detection (2025 Best Practices)

## Table of Contents
1. [Introduction](#introduction)
2. [Monitoring Fundamentals](#monitoring-fundamentals)
3. [Data Drift Detection](#data-drift-detection)
4. [Concept Drift vs Data Drift](#concept-drift-vs-data-drift)
5. [Performance Monitoring](#performance-monitoring)
6. [Alerting Systems](#alerting-systems)
7. [Production Monitoring Tools](#production-monitoring-tools)
8. [Automated Retraining](#automated-retraining)
9. [Production Examples](#production-examples)
10. [Best Practices](#best-practices)

---

## Introduction

Production ML models degrade over time due to changes in data distributions, user behavior, and external factors. According to 2025 industry research, proper monitoring and drift detection lead to a **40% reduction in production incidents**.

### Why Monitoring is Critical

- **Silent Failures:** Models can degrade without obvious errors
- **Business Impact:** Poor predictions lead to lost revenue and user trust
- **Regulatory Compliance:** Many industries require model monitoring (EU AI Act)
- **Cost Optimization:** Identify inefficiencies and resource waste

### Key Monitoring Objectives

1. **Model Performance:** Accuracy, precision, recall, business metrics
2. **Data Quality:** Missing values, outliers, distribution changes
3. **System Health:** Latency, throughput, error rates
4. **Resource Usage:** CPU, memory, GPU utilization, costs
5. **Fairness:** Bias detection across demographic groups

### 2025 Trends

- **Automated Remediation:** Self-healing systems that retrain automatically
- **Real-time Drift Detection:** Sub-minute detection of distribution shifts
- **Explainable Monitoring:** Understanding why models fail
- **Multi-model Observability:** Tracking hundreds of models centrally
- **Carbon Monitoring:** Tracking ML system environmental impact

---

## Monitoring Fundamentals

### What to Monitor

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np
from datetime import datetime

@dataclass
class ModelMetrics:
    """Core metrics to monitor for ML models."""

    # Performance metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float

    # Business metrics
    conversion_rate: float
    revenue_impact: float
    user_satisfaction: float

    # System metrics
    latency_p50: float  # ms
    latency_p95: float  # ms
    latency_p99: float  # ms
    throughput: float   # requests/sec
    error_rate: float   # percentage

    # Resource metrics
    cpu_usage: float    # percentage
    memory_usage: float # GB
    gpu_usage: float    # percentage
    cost_per_1k_requests: float

    # Data quality
    missing_values_pct: float
    outliers_pct: float
    feature_correlation_change: float

    timestamp: datetime

class MetricsCollector:
    """Collect and store model metrics over time."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.predictions = []
        self.ground_truth = []
        self.latencies = []
        self.feature_stats = {}

    def log_prediction(
        self,
        features: np.ndarray,
        prediction: float,
        ground_truth: float = None,
        latency: float = None
    ):
        """Log a single prediction with metadata."""
        self.predictions.append(prediction)

        if ground_truth is not None:
            self.ground_truth.append(ground_truth)

        if latency is not None:
            self.latencies.append(latency)

        # Update feature statistics
        for i, value in enumerate(features):
            feature_name = f"feature_{i}"
            if feature_name not in self.feature_stats:
                self.feature_stats[feature_name] = []

            self.feature_stats[feature_name].append(value)

            # Keep only recent window
            if len(self.feature_stats[feature_name]) > self.window_size:
                self.feature_stats[feature_name].pop(0)

    def compute_metrics(self) -> ModelMetrics:
        """Compute current metrics from collected data."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

        if len(self.ground_truth) < 10:
            return None  # Not enough data

        # Convert predictions to binary (assuming threshold 0.5)
        y_pred = (np.array(self.predictions) > 0.5).astype(int)
        y_true = np.array(self.ground_truth)

        # Compute metrics
        metrics = ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1_score=f1_score(y_true, y_pred, zero_division=0),
            auc_roc=roc_auc_score(y_true, self.predictions),

            # Placeholder for business metrics (requires integration)
            conversion_rate=0.0,
            revenue_impact=0.0,
            user_satisfaction=0.0,

            # Latency metrics
            latency_p50=np.percentile(self.latencies, 50) if self.latencies else 0,
            latency_p95=np.percentile(self.latencies, 95) if self.latencies else 0,
            latency_p99=np.percentile(self.latencies, 99) if self.latencies else 0,

            # Placeholder for throughput and resource metrics
            throughput=0.0,
            error_rate=0.0,
            cpu_usage=0.0,
            memory_usage=0.0,
            gpu_usage=0.0,
            cost_per_1k_requests=0.0,

            # Data quality
            missing_values_pct=0.0,
            outliers_pct=0.0,
            feature_correlation_change=0.0,

            timestamp=datetime.now()
        )

        return metrics

    def get_feature_distribution(self, feature_name: str) -> Dict:
        """Get distribution statistics for a feature."""
        if feature_name not in self.feature_stats:
            return None

        values = np.array(self.feature_stats[feature_name])

        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75)
        }
```

---

## Data Drift Detection

Data drift occurs when the statistical properties of input features change over time. This can significantly degrade model performance.

### Statistical Tests for Drift

```python
from scipy import stats
from typing import Tuple
import numpy as np

class DriftDetector:
    """Detect data drift using statistical tests."""

    def __init__(self, reference_data: np.ndarray, alpha: float = 0.05):
        """
        Args:
            reference_data: Training data distribution (reference)
            alpha: Significance level for statistical tests
        """
        self.reference_data = reference_data
        self.alpha = alpha

        # Store reference statistics
        self.reference_mean = np.mean(reference_data, axis=0)
        self.reference_std = np.std(reference_data, axis=0)

    def kolmogorov_smirnov_test(
        self,
        current_data: np.ndarray,
        feature_idx: int = None
    ) -> Tuple[float, bool]:
        """
        Kolmogorov-Smirnov test for distribution drift.

        Returns:
            (p_value, is_drift_detected)
        """
        if feature_idx is not None:
            ref = self.reference_data[:, feature_idx]
            cur = current_data[:, feature_idx]
        else:
            ref = self.reference_data.flatten()
            cur = current_data.flatten()

        statistic, p_value = stats.ks_2samp(ref, cur)

        is_drift = p_value < self.alpha

        return p_value, is_drift

    def population_stability_index(
        self,
        current_data: np.ndarray,
        feature_idx: int,
        n_bins: int = 10
    ) -> Tuple[float, bool]:
        """
        Population Stability Index (PSI) for drift detection.

        PSI < 0.1: No significant change
        0.1 <= PSI < 0.2: Moderate change
        PSI >= 0.2: Significant change (drift)

        Returns:
            (psi_value, is_drift_detected)
        """
        ref = self.reference_data[:, feature_idx]
        cur = current_data[:, feature_idx]

        # Create bins from reference data
        bins = np.linspace(ref.min(), ref.max(), n_bins + 1)
        bins[0] = -np.inf
        bins[-1] = np.inf

        # Calculate distributions
        ref_dist, _ = np.histogram(ref, bins=bins)
        cur_dist, _ = np.histogram(cur, bins=bins)

        # Convert to percentages
        ref_pct = ref_dist / len(ref)
        cur_pct = cur_dist / len(cur)

        # Avoid division by zero
        ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
        cur_pct = np.where(cur_pct == 0, 0.0001, cur_pct)

        # Calculate PSI
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        is_drift = psi >= 0.2

        return psi, is_drift

    def kl_divergence(
        self,
        current_data: np.ndarray,
        feature_idx: int,
        n_bins: int = 50
    ) -> Tuple[float, bool]:
        """
        Kullback-Leibler divergence for drift detection.

        Returns:
            (kl_value, is_drift_detected)
        """
        ref = self.reference_data[:, feature_idx]
        cur = current_data[:, feature_idx]

        # Create bins
        bins = np.linspace(
            min(ref.min(), cur.min()),
            max(ref.max(), cur.max()),
            n_bins + 1
        )

        # Calculate distributions
        ref_hist, _ = np.histogram(ref, bins=bins)
        cur_hist, _ = np.histogram(cur, bins=bins)

        # Normalize to probabilities
        ref_prob = ref_hist / ref_hist.sum()
        cur_prob = cur_hist / cur_hist.sum()

        # Avoid log(0)
        ref_prob = np.where(ref_prob == 0, 1e-10, ref_prob)
        cur_prob = np.where(cur_prob == 0, 1e-10, cur_prob)

        # Calculate KL divergence
        kl_div = np.sum(cur_prob * np.log(cur_prob / ref_prob))

        # Threshold for drift (can be tuned)
        threshold = 0.1
        is_drift = kl_div > threshold

        return kl_div, is_drift

    def detect_drift_all_features(
        self,
        current_data: np.ndarray,
        method: str = 'psi'
    ) -> Dict:
        """
        Detect drift across all features.

        Args:
            current_data: Current production data
            method: 'ks' (Kolmogorov-Smirnov), 'psi' (PSI), or 'kl' (KL divergence)

        Returns:
            Dictionary with drift results per feature
        """
        n_features = self.reference_data.shape[1]
        results = {}

        for i in range(n_features):
            if method == 'ks':
                value, is_drift = self.kolmogorov_smirnov_test(current_data, i)
                metric_name = 'p_value'
            elif method == 'psi':
                value, is_drift = self.population_stability_index(current_data, i)
                metric_name = 'psi'
            elif method == 'kl':
                value, is_drift = self.kl_divergence(current_data, i)
                metric_name = 'kl_divergence'
            else:
                raise ValueError(f"Unknown method: {method}")

            results[f'feature_{i}'] = {
                metric_name: value,
                'drift_detected': is_drift
            }

        return results

# Usage Example
reference_data = np.random.randn(1000, 10)  # Training data
detector = DriftDetector(reference_data, alpha=0.05)

# Production data (with drift in feature 0)
current_data = np.random.randn(500, 10)
current_data[:, 0] += 2.0  # Introduce drift

# Detect drift
drift_results = detector.detect_drift_all_features(current_data, method='psi')

for feature, result in drift_results.items():
    if result['drift_detected']:
        print(f"⚠️  Drift detected in {feature}: PSI = {result['psi']:.3f}")
```

### Advanced Drift Detection with ADWIN

```python
from river import drift
import numpy as np

class AdaptiveDriftDetector:
    """
    Adaptive drift detection using River library.
    Detects both gradual and sudden drift in real-time.
    """

    def __init__(self, delta: float = 0.002):
        """
        Args:
            delta: Confidence level (lower = more sensitive)
        """
        self.adwin = drift.ADWIN(delta=delta)
        self.page_hinkley = drift.PageHinkley()
        self.kswin = drift.KSWIN(alpha=0.005)

    def update(self, value: float) -> dict:
        """
        Update drift detectors with new value.

        Returns:
            Dictionary with drift detection results
        """
        # Update all detectors
        adwin_drift = self.adwin.update(value)
        ph_drift = self.page_hinkley.update(value)
        kswin_drift = self.kswin.update(value)

        return {
            'adwin_drift': adwin_drift,
            'page_hinkley_drift': ph_drift,
            'kswin_drift': kswin_drift,
            'any_drift': adwin_drift or ph_drift or kswin_drift
        }

# Usage in streaming scenario
adaptive_detector = AdaptiveDriftDetector(delta=0.002)

for i in range(1000):
    # Simulate prediction error rate
    if i < 500:
        error = np.random.beta(2, 8)  # Low error rate
    else:
        error = np.random.beta(8, 2)  # High error rate (drift)

    result = adaptive_detector.update(error)

    if result['any_drift']:
        print(f"🚨 Drift detected at sample {i}: {result}")
```

---

## Concept Drift vs Data Drift

### Data Drift (Covariate Shift)

**Definition:** Distribution of input features P(X) changes, but P(Y|X) remains the same.

**Example:** User demographics change, but the relationship between features and outcome is stable.

**Detection:**
```python
class DataDriftMonitor:
    """Monitor for data drift (P(X) changes)."""

    def __init__(self, reference_features: np.ndarray):
        self.reference_features = reference_features
        self.detector = DriftDetector(reference_features)

    def check_drift(self, current_features: np.ndarray) -> dict:
        """Check for data drift in features."""
        return self.detector.detect_drift_all_features(
            current_features,
            method='psi'
        )
```

### Concept Drift

**Definition:** Relationship between features and target P(Y|X) changes, even if P(X) stays the same.

**Example:** Customer preferences change over time (same features, different outcomes).

**Detection:**
```python
class ConceptDriftMonitor:
    """Monitor for concept drift (P(Y|X) changes)."""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.errors = []

    def update(self, prediction: float, ground_truth: float):
        """Update with new prediction and ground truth."""
        error = abs(prediction - ground_truth)
        self.errors.append(error)

        if len(self.errors) > self.window_size:
            self.errors.pop(0)

    def detect_drift(self) -> bool:
        """Detect if error rate is increasing (concept drift)."""
        if len(self.errors) < self.window_size:
            return False

        # Split into two windows
        mid = len(self.errors) // 2
        old_errors = self.errors[:mid]
        recent_errors = self.errors[mid:]

        # Compare error rates
        old_mean = np.mean(old_errors)
        recent_mean = np.mean(recent_errors)

        # Use t-test
        statistic, p_value = stats.ttest_ind(old_errors, recent_errors)

        # Drift if recent errors significantly higher
        is_drift = (p_value < 0.05) and (recent_mean > old_mean * 1.1)

        return is_drift

    def get_error_trend(self) -> dict:
        """Get error trend statistics."""
        if len(self.errors) < 100:
            return None

        return {
            'current_error': np.mean(self.errors[-100:]),
            'baseline_error': np.mean(self.errors[:100]),
            'trend': 'increasing' if np.mean(self.errors[-100:]) > np.mean(self.errors[:100]) else 'stable'
        }
```

### Combined Monitoring

```python
class ComprehensiveDriftMonitor:
    """Monitor both data drift and concept drift."""

    def __init__(
        self,
        reference_features: np.ndarray,
        window_size: int = 1000
    ):
        self.data_drift_monitor = DataDriftMonitor(reference_features)
        self.concept_drift_monitor = ConceptDriftMonitor(window_size)

    def update(
        self,
        features: np.ndarray,
        prediction: float,
        ground_truth: float = None
    ):
        """Update monitors with new data."""
        if ground_truth is not None:
            self.concept_drift_monitor.update(prediction, ground_truth)

    def check_all_drift(self, current_features: np.ndarray) -> dict:
        """Check for all types of drift."""
        # Data drift
        data_drift = self.data_drift_monitor.check_drift(current_features)
        data_drift_detected = any(
            result['drift_detected']
            for result in data_drift.values()
        )

        # Concept drift
        concept_drift_detected = self.concept_drift_monitor.detect_drift()

        return {
            'data_drift': {
                'detected': data_drift_detected,
                'features': data_drift
            },
            'concept_drift': {
                'detected': concept_drift_detected,
                'trend': self.concept_drift_monitor.get_error_trend()
            },
            'action_required': data_drift_detected or concept_drift_detected
        }
```

---

## Performance Monitoring

### System Performance Metrics

```python
import psutil
import time
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
from typing import Callable
import functools

# Prometheus metrics
REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    'model_requests_total',
    'Total model requests',
    ['model_name', 'status'],
    registry=REGISTRY
)

REQUEST_LATENCY = Histogram(
    'model_request_latency_seconds',
    'Request latency',
    ['model_name'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=REGISTRY
)

PREDICTION_VALUE = Histogram(
    'model_prediction_value',
    'Prediction values distribution',
    ['model_name'],
    registry=REGISTRY
)

CPU_USAGE = Gauge(
    'model_cpu_usage_percent',
    'CPU usage percentage',
    ['model_name'],
    registry=REGISTRY
)

MEMORY_USAGE = Gauge(
    'model_memory_usage_bytes',
    'Memory usage in bytes',
    ['model_name'],
    registry=REGISTRY
)

GPU_MEMORY = Gauge(
    'model_gpu_memory_bytes',
    'GPU memory usage in bytes',
    ['model_name'],
    registry=REGISTRY
)

class PerformanceMonitor:
    """Monitor system and model performance."""

    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.process = psutil.Process()

    def monitor_latency(self, func: Callable) -> Callable:
        """Decorator to monitor function latency."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                status = "success"
                return result

            except Exception as e:
                status = "error"
                raise e

            finally:
                latency = time.time() - start_time

                # Update Prometheus metrics
                REQUEST_COUNT.labels(
                    model_name=self.model_name,
                    status=status
                ).inc()

                REQUEST_LATENCY.labels(
                    model_name=self.model_name
                ).observe(latency)

        return wrapper

    def update_resource_metrics(self):
        """Update CPU and memory metrics."""
        # CPU usage
        cpu_percent = self.process.cpu_percent(interval=0.1)
        CPU_USAGE.labels(model_name=self.model_name).set(cpu_percent)

        # Memory usage
        memory_info = self.process.memory_info()
        MEMORY_USAGE.labels(model_name=self.model_name).set(memory_info.rss)

        # GPU memory (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated()
                GPU_MEMORY.labels(model_name=self.model_name).set(gpu_memory)
        except:
            pass

    def log_prediction(self, prediction_value: float):
        """Log prediction value for distribution monitoring."""
        PREDICTION_VALUE.labels(model_name=self.model_name).observe(prediction_value)

# Usage
monitor = PerformanceMonitor(model_name="fraud_detection")

@monitor.monitor_latency
def predict(features):
    # Your prediction logic
    prediction = model(features)
    monitor.log_prediction(prediction)
    return prediction

# Update resource metrics periodically
import threading

def update_metrics_loop():
    while True:
        monitor.update_resource_metrics()
        time.sleep(5)  # Update every 5 seconds

metrics_thread = threading.Thread(target=update_metrics_loop, daemon=True)
metrics_thread.start()
```

---

## Alerting Systems

### Alert Configuration

```python
from enum import Enum
from dataclasses import dataclass
from typing import List, Callable
import smtplib
from email.mime.text import MIMEText

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """Define alert rule."""
    name: str
    metric: str
    threshold: float
    comparison: str  # 'gt', 'lt', 'eq'
    severity: AlertSeverity
    cooldown_seconds: int = 300  # Minimum time between alerts

@dataclass
class Alert:
    """Alert object."""
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: datetime
    metric_value: float

class AlertManager:
    """Manage alerts and notifications."""

    def __init__(self):
        self.rules: List[AlertRule] = []
        self.last_alert_times = {}
        self.handlers: List[Callable] = []

    def add_rule(self, rule: AlertRule):
        """Add alert rule."""
        self.rules.append(rule)

    def add_handler(self, handler: Callable):
        """Add alert handler (e.g., email, Slack, PagerDuty)."""
        self.handlers.append(handler)

    def check_rules(self, metrics: dict):
        """Check all rules against current metrics."""
        alerts = []

        for rule in self.rules:
            if rule.metric not in metrics:
                continue

            metric_value = metrics[rule.metric]

            # Check threshold
            triggered = False
            if rule.comparison == 'gt' and metric_value > rule.threshold:
                triggered = True
            elif rule.comparison == 'lt' and metric_value < rule.threshold:
                triggered = True
            elif rule.comparison == 'eq' and metric_value == rule.threshold:
                triggered = True

            if not triggered:
                continue

            # Check cooldown
            last_alert = self.last_alert_times.get(rule.name, None)
            if last_alert:
                time_since = (datetime.now() - last_alert).total_seconds()
                if time_since < rule.cooldown_seconds:
                    continue  # Skip due to cooldown

            # Create alert
            alert = Alert(
                rule_name=rule.name,
                severity=rule.severity,
                message=f"{rule.name}: {rule.metric} = {metric_value:.3f} (threshold: {rule.threshold})",
                timestamp=datetime.now(),
                metric_value=metric_value
            )

            alerts.append(alert)
            self.last_alert_times[rule.name] = datetime.now()

        # Send alerts
        for alert in alerts:
            self._send_alert(alert)

        return alerts

    def _send_alert(self, alert: Alert):
        """Send alert to all handlers."""
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Error sending alert: {e}")

# Alert handlers
def email_handler(alert: Alert):
    """Send email alert."""
    # Configure your SMTP settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    sender = "alerts@yourcompany.com"
    recipients = ["oncall@yourcompany.com"]
    password = "your_password"

    msg = MIMEText(f"""
    Alert: {alert.rule_name}
    Severity: {alert.severity.value}
    Time: {alert.timestamp}
    Message: {alert.message}
    """)

    msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        print(f"✅ Email alert sent: {alert.rule_name}")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def slack_handler(alert: Alert):
    """Send Slack alert."""
    import requests

    webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

    emoji = {
        AlertSeverity.INFO: ":information_source:",
        AlertSeverity.WARNING: ":warning:",
        AlertSeverity.CRITICAL: ":rotating_light:"
    }

    payload = {
        "text": f"{emoji[alert.severity]} *{alert.rule_name}*",
        "attachments": [{
            "color": "danger" if alert.severity == AlertSeverity.CRITICAL else "warning",
            "fields": [
                {"title": "Message", "value": alert.message, "short": False},
                {"title": "Time", "value": str(alert.timestamp), "short": True},
                {"title": "Value", "value": f"{alert.metric_value:.3f}", "short": True}
            ]
        }]
    }

    try:
        response = requests.post(webhook_url, json=payload)
        response.raise_for_status()
        print(f"✅ Slack alert sent: {alert.rule_name}")
    except Exception as e:
        print(f"❌ Failed to send Slack alert: {e}")

# Usage
alert_manager = AlertManager()

# Add rules
alert_manager.add_rule(AlertRule(
    name="High Error Rate",
    metric="error_rate",
    threshold=0.05,  # 5%
    comparison='gt',
    severity=AlertSeverity.CRITICAL,
    cooldown_seconds=600
))

alert_manager.add_rule(AlertRule(
    name="Model Accuracy Drop",
    metric="accuracy",
    threshold=0.85,
    comparison='lt',
    severity=AlertSeverity.WARNING,
    cooldown_seconds=300
))

alert_manager.add_rule(AlertRule(
    name="High Latency",
    metric="latency_p95",
    threshold=500,  # ms
    comparison='gt',
    severity=AlertSeverity.WARNING,
    cooldown_seconds=300
))

# Add handlers
alert_manager.add_handler(email_handler)
alert_manager.add_handler(slack_handler)

# Check metrics
current_metrics = {
    'error_rate': 0.08,  # Above threshold
    'accuracy': 0.82,    # Below threshold
    'latency_p95': 450
}

alerts = alert_manager.check_rules(current_metrics)
```

---

## Production Monitoring Tools

### 1. Prometheus Integration

```python
from prometheus_client import start_http_server, generate_latest
from flask import Flask, Response

app = Flask(__name__)

@app.route('/metrics')
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(REGISTRY), mimetype='text/plain')

# Start Prometheus metrics server
# start_http_server(8001)  # Metrics available at :8001/metrics
```

**Prometheus Configuration (prometheus.yml):**

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'ml_model'
    static_configs:
      - targets: ['ml-model-service:8000']
    metrics_path: '/metrics'

rule_files:
  - 'alert_rules.yml'

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

**Alert Rules (alert_rules.yml):**

```yaml
groups:
  - name: ml_model_alerts
    interval: 30s
    rules:
      - alert: HighErrorRate
        expr: rate(model_requests_total{status="error"}[5m]) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }}"

      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(model_request_latency_seconds_bucket[5m])) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P95 latency is {{ $value }}s"

      - alert: DataDriftDetected
        expr: drift_psi_value > 0.2
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "Data drift detected"
          description: "PSI value is {{ $value }}"
```

### 2. WhyLabs Integration (ML-Specific Monitoring)

```python
import whylogs as why
from whylogs.api.writer.whylabs import WhyLabsWriter
import pandas as pd

class WhyLabsMonitor:
    """Monitor with WhyLabs for ML-specific observability."""

    def __init__(self, org_id: str, api_key: str, dataset_id: str):
        self.org_id = org_id
        self.api_key = api_key
        self.dataset_id = dataset_id

    def log_dataframe(self, df: pd.DataFrame, timestamp: datetime = None):
        """Log dataframe to WhyLabs."""
        # Create profile
        results = why.log(df)

        # Write to WhyLabs
        writer = WhyLabsWriter(
            org_id=self.org_id,
            api_key=self.api_key,
            dataset_id=self.dataset_id
        )

        writer.write(profile=results.profile())

        print(f"✅ Logged {len(df)} records to WhyLabs")

    def log_predictions(
        self,
        features: pd.DataFrame,
        predictions: np.ndarray,
        ground_truth: np.ndarray = None
    ):
        """Log predictions with features and outcomes."""
        df = features.copy()
        df['prediction'] = predictions

        if ground_truth is not None:
            df['ground_truth'] = ground_truth
            df['error'] = np.abs(predictions - ground_truth)

        self.log_dataframe(df)

# Usage
whylabs = WhyLabsMonitor(
    org_id="org-123",
    api_key="api-key-xyz",
    dataset_id="model-prod"
)

# Log production data
features = pd.DataFrame(...)
predictions = model.predict(features)
whylabs.log_predictions(features, predictions)
```

### 3. Datadog Integration

```python
from datadog import initialize, statsd
from datadog_api_client import ApiClient, Configuration
from datadog_api_client.v1.api.metrics_api import MetricsApi
from datadog_api_client.v1.model.metrics_payload import MetricsPayload

class DatadogMonitor:
    """Monitor with Datadog."""

    def __init__(self, api_key: str, app_key: str):
        # Initialize Datadog
        initialize(api_key=api_key, app_key=app_key)

        self.configuration = Configuration()
        self.configuration.api_key['apiKeyAuth'] = api_key
        self.configuration.api_key['appKeyAuth'] = app_key

    def send_metric(self, metric_name: str, value: float, tags: List[str] = None):
        """Send metric to Datadog."""
        statsd.gauge(metric_name, value, tags=tags or [])

    def send_distribution(self, metric_name: str, value: float, tags: List[str] = None):
        """Send distribution metric."""
        statsd.distribution(metric_name, value, tags=tags or [])

    def send_event(self, title: str, text: str, alert_type: str = 'info'):
        """Send event to Datadog."""
        from datadog import api
        api.Event.create(
            title=title,
            text=text,
            alert_type=alert_type,
            tags=['ml_model', 'production']
        )

# Usage
dd = DatadogMonitor(api_key="your-api-key", app_key="your-app-key")

# Send metrics
dd.send_metric('model.accuracy', 0.95, tags=['model:fraud_detection', 'version:v3'])
dd.send_distribution('model.latency', 45.2, tags=['model:fraud_detection'])

# Send drift event
dd.send_event(
    title='Data Drift Detected',
    text='PSI value exceeded threshold in feature_5',
    alert_type='warning'
)
```

---

## Automated Retraining

```python
from typing import Optional
import schedule
import time

class AutomatedRetrainingPipeline:
    """Automated model retraining pipeline with drift detection."""

    def __init__(
        self,
        drift_detector: DriftDetector,
        concept_drift_monitor: ConceptDriftMonitor,
        retrain_function: Callable,
        deploy_function: Callable,
        min_retrain_interval_hours: int = 24
    ):
        self.drift_detector = drift_detector
        self.concept_drift_monitor = concept_drift_monitor
        self.retrain_function = retrain_function
        self.deploy_function = deploy_function
        self.min_retrain_interval_hours = min_retrain_interval_hours
        self.last_retrain_time = None

    def should_retrain(self, current_features: np.ndarray) -> tuple:
        """
        Determine if retraining is needed.

        Returns:
            (should_retrain: bool, reason: str)
        """
        # Check data drift
        drift_results = self.drift_detector.detect_drift_all_features(
            current_features,
            method='psi'
        )

        data_drift_detected = any(
            result['drift_detected']
            for result in drift_results.values()
        )

        # Check concept drift
        concept_drift_detected = self.concept_drift_monitor.detect_drift()

        # Check time since last retrain
        if self.last_retrain_time:
            hours_since = (datetime.now() - self.last_retrain_time).total_seconds() / 3600
            if hours_since < self.min_retrain_interval_hours:
                return False, f"Too soon since last retrain ({hours_since:.1f}h)"

        if data_drift_detected:
            return True, "Data drift detected"
        elif concept_drift_detected:
            return True, "Concept drift detected"
        else:
            return False, "No drift detected"

    def retrain_and_deploy(self, reason: str):
        """Execute retraining and deployment."""
        print(f"🔄 Starting retraining: {reason}")

        try:
            # Retrain model
            new_model = self.retrain_function()

            # Validate new model
            if self._validate_model(new_model):
                # Deploy new model
                self.deploy_function(new_model)

                self.last_retrain_time = datetime.now()
                print(f"✅ Retraining complete and deployed")

                # Send notification
                self._notify_retraining(reason, success=True)
            else:
                print(f"❌ New model failed validation")
                self._notify_retraining(reason, success=False)

        except Exception as e:
            print(f"❌ Retraining failed: {e}")
            self._notify_retraining(reason, success=False, error=str(e))

    def _validate_model(self, model) -> bool:
        """Validate new model before deployment."""
        # Implement validation logic
        # e.g., test on holdout set, compare with current model
        return True

    def _notify_retraining(self, reason: str, success: bool, error: str = None):
        """Send notification about retraining."""
        # Implement notification (email, Slack, etc.)
        pass

    def run_continuous_monitoring(self, check_interval_minutes: int = 60):
        """Run continuous monitoring and auto-retrain."""
        def check_and_retrain():
            # Fetch recent production data
            current_features = self._fetch_recent_data()

            # Check if retraining needed
            should_retrain, reason = self.should_retrain(current_features)

            if should_retrain:
                self.retrain_and_deploy(reason)

        # Schedule periodic checks
        schedule.every(check_interval_minutes).minutes.do(check_and_retrain)

        print(f"🚀 Starting continuous monitoring (check every {check_interval_minutes} min)")

        while True:
            schedule.run_pending()
            time.sleep(60)

    def _fetch_recent_data(self) -> np.ndarray:
        """Fetch recent production data for drift detection."""
        # Implement data fetching from production logs
        pass

# Usage
def retrain_model():
    """Retrain model with latest data."""
    # Your retraining logic
    print("Retraining model...")
    return trained_model

def deploy_model(model):
    """Deploy new model."""
    # Your deployment logic
    print("Deploying model...")

pipeline = AutomatedRetrainingPipeline(
    drift_detector=detector,
    concept_drift_monitor=concept_monitor,
    retrain_function=retrain_model,
    deploy_function=deploy_model,
    min_retrain_interval_hours=24
)

# Run continuous monitoring
pipeline.run_continuous_monitoring(check_interval_minutes=60)
```

---

## Best Practices

### 1. Comprehensive Monitoring Checklist

- **Model Performance:**
  - Track accuracy, precision, recall, F1, AUC-ROC
  - Monitor business metrics (conversion, revenue, satisfaction)
  - Set up degradation alerts

- **Data Quality:**
  - Check for missing values, outliers, invalid ranges
  - Monitor feature distributions (PSI, KL divergence)
  - Validate data schema and types

- **System Health:**
  - Track latency (P50, P95, P99)
  - Monitor throughput and error rates
  - Resource usage (CPU, memory, GPU)

- **Drift Detection:**
  - Implement both data drift and concept drift monitoring
  - Use multiple detection methods (PSI, KS test, KL divergence)
  - Set appropriate thresholds and cooldown periods

- **Alerting:**
  - Define clear alert rules with severity levels
  - Implement alerting channels (email, Slack, PagerDuty)
  - Avoid alert fatigue with cooldowns

### 2. Monitoring Dashboard Example

```python
# Grafana dashboard JSON (import into Grafana)
dashboard_json = {
    "title": "ML Model Monitoring",
    "panels": [
        {
            "title": "Request Rate",
            "targets": [{"expr": "rate(model_requests_total[5m])"}]
        },
        {
            "title": "Error Rate",
            "targets": [{"expr": "rate(model_requests_total{status='error'}[5m]) / rate(model_requests_total[5m])"}]
        },
        {
            "title": "Latency Distribution",
            "targets": [
                {"expr": "histogram_quantile(0.50, rate(model_request_latency_seconds_bucket[5m]))", "legendFormat": "P50"},
                {"expr": "histogram_quantile(0.95, rate(model_request_latency_seconds_bucket[5m]))", "legendFormat": "P95"},
                {"expr": "histogram_quantile(0.99, rate(model_request_latency_seconds_bucket[5m]))", "legendFormat": "P99"}
            ]
        },
        {
            "title": "Model Accuracy (Sliding Window)",
            "targets": [{"expr": "model_accuracy"}]
        },
        {
            "title": "Data Drift (PSI)",
            "targets": [{"expr": "drift_psi_value"}]
        }
    ]
}
```

---

## Summary

Effective monitoring and drift detection in 2025 requires:

1. **Multi-faceted Monitoring:** Track model performance, data quality, system health, and resources
2. **Proactive Drift Detection:** Implement both data drift and concept drift monitoring
3. **Automated Alerting:** Set up intelligent alerts with proper thresholds and cooldowns
4. **Modern Tools:** Use Prometheus, Datadog, WhyLabs for comprehensive observability
5. **Automated Remediation:** Implement auto-retraining when drift is detected

**Key Metrics:**
- 40% reduction in production incidents with proper monitoring
- Sub-minute drift detection with modern tools
- Automated retraining reduces manual intervention by 80%

**Next Steps:**
- Set up deployment pipeline (Section 36)
- Implement pipeline orchestration (Section 38)
- Enable experiment tracking (Section 40)
