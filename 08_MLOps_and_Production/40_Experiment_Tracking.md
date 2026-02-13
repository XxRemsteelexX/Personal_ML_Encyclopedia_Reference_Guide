# Experiment Tracking (2025 Best Practices)

## Table of Contents
1. [Introduction](#introduction)
2. [Tracking Fundamentals](#tracking-fundamentals)
3. [MLflow](#mlflow)
4. [Weights & Biases](#weights--biases)
5. [TensorBoard](#tensorboard)
6. [Comparison and Selection](#comparison-and-selection)
7. [Reproducibility Best Practices](#reproducibility-best-practices)
8. [Model Registry](#model-registry)
9. [Production Integration](#production-integration)
10. [Best Practices](#best-practices)

---

## Introduction

Experiment tracking is the systematic recording of machine learning experiments, including parameters, metrics, artifacts, and metadata. In 2025, robust experiment tracking is essential for reproducibility, collaboration, and model governance.

### Why Experiment Tracking

- **Reproducibility:** Recreate exact experiment conditions months later
- **Collaboration:** Share results and insights with team members
- **Model Selection:** Compare hundreds of experiments systematically
- **Debugging:** Understand why experiments succeed or fail
- **Compliance:** Meet regulatory requirements for model governance
- **Knowledge Transfer:** Document decisions for future team members

### Key Challenges

1. **Scale:** Tracking thousands of experiments without performance degradation
2. **Organization:** Structuring experiments for easy retrieval
3. **Artifact Management:** Storing large models and datasets efficiently
4. **Versioning:** Linking code, data, and model versions
5. **Cost:** Balancing storage costs with retention requirements

### 2025 Landscape

- **LLM Experiment Tracking:** Specialized tools for tracking prompts, completions, tokens
- **Multi-Modal Tracking:** Images, text, audio, video in single experiments
- **Edge Deployment Tracking:** Tracking on-device model performance
- **Green ML:** Carbon footprint tracking integrated into experiments
- **Automated Insights:** AI-powered experiment analysis and recommendations

---

## Tracking Fundamentals

### What to Track

```python
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import hashlib

@dataclass
class Experiment:
    """Comprehensive experiment tracking."""

    # Identification
    experiment_id: str
    experiment_name: str
    run_id: str
    timestamp: datetime = field(default_factory=datetime.now)

    # Code versioning
    git_commit: Optional[str] = None
    git_branch: Optional[str] = None
    code_version: Optional[str] = None

    # Environment
    python_version: str = None
    cuda_version: Optional[str] = None
    gpu_type: Optional[str] = None
    os_version: str = None

    # Data versioning
    dataset_name: str = None
    dataset_version: str = None
    dataset_hash: Optional[str] = None
    train_size: int = 0
    val_size: int = 0
    test_size: int = 0

    # Model configuration
    model_type: str = None
    model_architecture: Dict[str, Any] = field(default_factory=dict)

    # Hyperparameters
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Training configuration
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    optimizer: str = "adam"
    loss_function: str = None

    # Metrics (tracked over time)
    metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Final results
    final_metrics: Dict[str, float] = field(default_factory=dict)

    # Resource usage
    training_time_seconds: float = 0.0
    gpu_hours: float = 0.0
    peak_memory_gb: float = 0.0
    carbon_emissions_kg: Optional[float] = None

    # Artifacts
    model_path: Optional[str] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    artifact_uris: Dict[str, str] = field(default_factory=dict)

    # Tags and notes
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    status: str = "running"  # running, completed, failed

    def to_dict(self) -> dict:
        """Convert to dictionary for storage."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_name': self.experiment_name,
            'run_id': self.run_id,
            'timestamp': self.timestamp.isoformat(),
            'git_commit': self.git_commit,
            'hyperparameters': self.hyperparameters,
            'final_metrics': self.final_metrics,
            'training_time_seconds': self.training_time_seconds,
            'status': self.status,
            'tags': self.tags
        }

    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log a metric value."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def compute_dataset_hash(self, data_path: str) -> str:
        """Compute hash of dataset for versioning."""
        import hashlib

        hash_md5 = hashlib.md5()
        with open(data_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class SimpleExperimentTracker:
    """Simple local experiment tracker."""

    def __init__(self, storage_path: str = "./experiments"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.current_experiment: Optional[Experiment] = None

    def start_experiment(
        self,
        experiment_name: str,
        hyperparameters: dict,
        tags: List[str] = None
    ) -> Experiment:
        """Start new experiment."""
        import uuid
        from pathlib import Path

        experiment = Experiment(
            experiment_id=str(uuid.uuid4()),
            experiment_name=experiment_name,
            run_id=str(uuid.uuid4())[:8],
            hyperparameters=hyperparameters,
            tags=tags or [],

            # Auto-detect environment
            python_version=sys.version.split()[0],
            os_version=platform.platform()
        )

        # Try to get git info
        try:
            import subprocess
            experiment.git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('utf-8').strip()
            experiment.git_branch = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).decode('utf-8').strip()
        except:
            pass

        self.current_experiment = experiment
        return experiment

    def log_metric(self, metric_name: str, value: float, step: int = None):
        """Log metric to current experiment."""
        if self.current_experiment:
            self.current_experiment.log_metric(metric_name, value, step)

    def log_params(self, params: dict):
        """Log parameters."""
        if self.current_experiment:
            self.current_experiment.hyperparameters.update(params)

    def log_artifact(self, artifact_path: str, artifact_name: str = None):
        """Log artifact file."""
        if self.current_experiment:
            if artifact_name is None:
                artifact_name = Path(artifact_path).name

            # Copy to experiment directory
            exp_dir = self.storage_path / self.current_experiment.run_id
            exp_dir.mkdir(exist_ok=True)

            dest_path = exp_dir / artifact_name
            shutil.copy(artifact_path, dest_path)

            self.current_experiment.artifact_uris[artifact_name] = str(dest_path)

    def end_experiment(self, status: str = "completed"):
        """End current experiment and save."""
        if self.current_experiment:
            self.current_experiment.status = status

            # Save experiment data
            exp_dir = self.storage_path / self.current_experiment.run_id
            exp_dir.mkdir(exist_ok=True)

            with open(exp_dir / 'experiment.json', 'w') as f:
                json.dump(self.current_experiment.to_dict(), f, indent=2)

            print(f" Experiment {self.current_experiment.run_id} completed")
            self.current_experiment = None

    def list_experiments(self, limit: int = 10) -> List[dict]:
        """List recent experiments."""
        experiments = []

        for exp_dir in sorted(self.storage_path.iterdir(), reverse=True)[:limit]:
            exp_file = exp_dir / 'experiment.json'
            if exp_file.exists():
                with open(exp_file) as f:
                    experiments.append(json.load(f))

        return experiments

# Usage
tracker = SimpleExperimentTracker()

# Start experiment
experiment = tracker.start_experiment(
    experiment_name="fraud_detection_xgboost",
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 10,
        'learning_rate': 0.1
    },
    tags=['xgboost', 'fraud', 'production']
)

# Training loop
for epoch in range(10):
    # ... training code ...
    train_loss = 0.5 - epoch * 0.02
    val_loss = 0.6 - epoch * 0.018

    tracker.log_metric('train_loss', train_loss, step=epoch)
    tracker.log_metric('val_loss', val_loss, step=epoch)

# Save model
tracker.log_artifact('/tmp/model.pkl', 'model.pkl')

# End experiment
tracker.end_experiment(status='completed')

# List experiments
experiments = tracker.list_experiments()
for exp in experiments:
    print(f"{exp['run_id']}: {exp['experiment_name']} - {exp['status']}")
```

---

## MLflow

MLflow is the most widely adopted open-source experiment tracking platform in 2025.

### MLflow Setup

```bash
# Install MLflow
pip install mlflow

# Start tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# With PostgreSQL and S3
mlflow server \
    --backend-store-uri postgresql://user:password@localhost/mlflow \
    --default-artifact-root s3://my-bucket/mlruns \
    --host 0.0.0.0 \
    --port 5000
```

### Basic MLflow Tracking

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import pandas as pd

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("fraud_detection")

# Start run
with mlflow.start_run(run_name="rf_baseline"):

    # Log parameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'random_state': 42
    }
    mlflow.log_params(params)

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("auc", auc)

    # Log model
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="fraud_detection_rf"
    )

    # Log additional artifacts
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    importance_df.to_csv("feature_importance.csv", index=False)
    mlflow.log_artifact("feature_importance.csv")

    # Log dataset info
    mlflow.log_param("train_size", len(X_train))
    mlflow.log_param("test_size", len(X_test))

    # Log tags
    mlflow.set_tags({
        "model_type": "random_forest",
        "task": "classification",
        "dataset": "fraud_transactions",
        "author": "data_science_team"
    })

    print(f" Run ID: {mlflow.active_run().info.run_id}")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   AUC: {auc:.4f}")
```

### Advanced MLflow: Nested Runs

```python
import mlflow
from sklearn.model_selection import cross_val_score

mlflow.set_experiment("hyperparameter_tuning")

# Parent run
with mlflow.start_run(run_name="grid_search") as parent_run:

    # Grid search parameters
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 15],
        'min_samples_split': [2, 5, 10]
    }

    best_score = 0
    best_params = None

    # Nested runs for each parameter combination
    for n_est in param_grid['n_estimators']:
        for depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:

                # Child run
                with mlflow.start_run(
                    run_name=f"n{n_est}_d{depth}_s{min_split}",
                    nested=True
                ) as child_run:

                    params = {
                        'n_estimators': n_est,
                        'max_depth': depth,
                        'min_samples_split': min_split
                    }

                    # Log parameters
                    mlflow.log_params(params)

                    # Train and evaluate
                    model = RandomForestClassifier(**params, random_state=42)
                    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')

                    mean_score = scores.mean()
                    std_score = scores.std()

                    # Log metrics
                    mlflow.log_metric("cv_auc_mean", mean_score)
                    mlflow.log_metric("cv_auc_std", std_score)

                    # Update best
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = params

    # Log best results to parent run
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow.log_metric("best_cv_auc", best_score)

    print(f" Best parameters: {best_params}")
    print(f"   Best AUC: {best_score:.4f}")
```

### MLflow Autologging

```python
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.pytorch

# Enable autologging for sklearn
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

# Enable autologging for XGBoost
mlflow.xgboost.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True
)

# Enable autologging for PyTorch
mlflow.pytorch.autolog(
    log_every_n_epoch=1,
    log_models=True
)

# Now training automatically logs everything
with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Automatically logged:
    # - Parameters
    # - Metrics (if using sklearn's cross_val_score)
    # - Model
    # - Model signature
    # - Input example
```

### MLflow Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:5000")

# Register model
model_name = "fraud_detection"

# Get latest run
experiment = client.get_experiment_by_name("fraud_detection")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.auc DESC"],
    max_results=1
)

best_run = runs[0]

# Register model version
model_uri = f"runs:/{best_run.info.run_id}/model"
model_version = mlflow.register_model(model_uri, model_name)

print(f"Registered model version: {model_version.version}")

# Transition to staging
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Staging"
)

# Add description
client.update_model_version(
    name=model_name,
    version=model_version.version,
    description="Random Forest with n_estimators=100, max_depth=10. AUC=0.95"
)

# Promote to production
client.transition_model_version_stage(
    name=model_name,
    version=model_version.version,
    stage="Production",
    archive_existing_versions=True  # Archive previous production versions
)

# Load production model
production_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")

# Make predictions
predictions = production_model.predict(X_test)
```

### MLflow Projects

```yaml
# MLproject
name: fraud_detection

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "/data/train.csv"}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}

    command: "python train.py --data-path {data_path} --n-estimators {n_estimators} --max-depth {max_depth}"

  evaluate:
    parameters:
      model_uri: {type: str}
      test_data: {type: str}

    command: "python evaluate.py --model-uri {model_uri} --test-data {test_data}"
```

```python
# train.py
import mlflow
import argparse

def train(data_path, n_estimators, max_depth):
    """Training function."""

    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Load data
        df = pd.read_csv(data_path)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        model.fit(X, y)

        # Log model
        mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)

    args = parser.parse_args()
    train(args.data_path, args.n_estimators, args.max_depth)
```

```bash
# Run MLflow project
mlflow run . \
    -P data_path=/data/train.csv \
    -P n_estimators=200 \
    -P max_depth=15

# Run from GitHub
mlflow run https://github.com/org/project.git \
    -P n_estimators=200
```

---

## Weights & Biases

W&B is the leading commercial experiment tracking platform, with superior visualization and collaboration features.

### W&B Setup

```bash
# Install W&B
pip install wandb

# Login (get API key from wandb.ai)
wandb login
```

### Basic W&B Tracking

```python
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Initialize W&B run
run = wandb.init(
    project="fraud_detection",
    name="resnet_baseline",
    config={
        "architecture": "ResNet50",
        "learning_rate": 0.001,
        "batch_size": 64,
        "epochs": 50,
        "optimizer": "Adam",
        "dataset": "fraud_transactions"
    },
    tags=["baseline", "resnet", "production"]
)

# Access config
config = wandb.config

# Training loop
model = ResNet50(num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.CrossEntropyLoss()

for epoch in range(config.epochs):
    model.train()
    train_loss = 0
    train_acc = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Log metrics every N batches
        if batch_idx % 10 == 0:
            wandb.log({
                "batch_train_loss": loss.item(),
                "batch": epoch * len(train_loader) + batch_idx
            })

    # Validation
    model.eval()
    val_loss = 0
    val_acc = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item()

    # Log epoch metrics
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss / len(train_loader),
        "val_loss": val_loss / len(val_loader),
        "train_acc": train_acc,
        "val_acc": val_acc,
        "learning_rate": optimizer.param_groups[0]['lr']
    })

# Log final model
wandb.save("model_final.pth")
torch.save(model.state_dict(), "model_final.pth")

# Finish run
wandb.finish()
```

### Advanced W&B: Hyperparameter Sweeps

```python
# sweep_config.yaml
program: train.py
method: bayes  # or 'grid', 'random'

metric:
  name: val_auc
  goal: maximize

parameters:
  learning_rate:
    distribution: log_uniform_values
    min: 0.0001
    max: 0.1

  batch_size:
    values: [32, 64, 128, 256]

  optimizer:
    values: ['adam', 'sgd', 'adamw']

  hidden_size:
    distribution: int_uniform
    min: 64
    max: 512

  dropout:
    distribution: uniform
    min: 0.1
    max: 0.5

early_terminate:
  type: hyperband
  min_iter: 10
  s: 2
```

```python
# train.py for sweeps
import wandb

def train():
    """Training function for sweep."""

    # Initialize W&B (config is provided by sweep)
    run = wandb.init()
    config = wandb.config

    # Build model with config
    model = build_model(
        hidden_size=config.hidden_size,
        dropout=config.dropout
    )

    # Optimizer from config
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Training loop
    for epoch in range(50):
        # ... training code ...

        # Log metrics
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_auc': val_auc
        })

    # Return final metric for sweep
    return val_auc

# Run sweep
sweep_id = wandb.sweep(sweep_config, project="fraud_detection")
wandb.agent(sweep_id, function=train, count=50)  # Run 50 trials
```

### W&B Artifacts

```python
import wandb

run = wandb.init(project="fraud_detection")

# Log dataset as artifact
dataset_artifact = wandb.Artifact(
    name="fraud_dataset",
    type="dataset",
    description="Fraud detection dataset v2.0",
    metadata={
        "source": "internal_db",
        "created": "2025-10-01",
        "rows": 1000000,
        "features": 50
    }
)

# Add files to artifact
dataset_artifact.add_file("train.csv")
dataset_artifact.add_file("test.csv")
dataset_artifact.add_dir("processed/")

# Log artifact
run.log_artifact(dataset_artifact)

# Use artifact in another run
run = wandb.init(project="fraud_detection")

# Download and use artifact
artifact = run.use_artifact('fraud_dataset:latest')
artifact_dir = artifact.download()

# Load data
import pandas as pd
train_df = pd.read_csv(f"{artifact_dir}/train.csv")

# Log model as artifact
model_artifact = wandb.Artifact(
    name="fraud_model",
    type="model",
    description="Random Forest fraud detection model"
)

model_artifact.add_file("model.pkl")
run.log_artifact(model_artifact)
```

### W&B Visualization

```python
import wandb
import numpy as np
import matplotlib.pyplot as plt

run = wandb.init(project="fraud_detection")

# Log images
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
wandb.log({"chart": wandb.Image(fig)})

# Log confusion matrix
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test,
        preds=y_pred,
        class_names=["legitimate", "fraud"]
    )
})

# Log ROC curve
wandb.log({
    "roc": wandb.plot.roc_curve(
        y_test,
        y_prob,
        labels=["legitimate", "fraud"]
    )
})

# Log PR curve
wandb.log({
    "pr": wandb.plot.pr_curve(
        y_test,
        y_prob,
        labels=["legitimate", "fraud"]
    )
})

# Log histogram
wandb.log({
    "predictions_histogram": wandb.Histogram(y_prob)
})

# Log table
table = wandb.Table(
    columns=["epoch", "train_loss", "val_loss", "val_auc"],
    data=[
        [1, 0.5, 0.6, 0.85],
        [2, 0.4, 0.55, 0.87],
        [3, 0.35, 0.52, 0.89]
    ]
)
wandb.log({"results_table": table})
```

---

## TensorBoard

TensorBoard is PyTorch and TensorFlow's built-in visualization tool.

### TensorBoard Setup

```python
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

# Create writer
writer = SummaryWriter(log_dir='runs/fraud_detection_experiment_1')

# Training loop
model = build_model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

for epoch in range(100):
    # Training
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log training loss
        global_step = epoch * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train', loss.item(), global_step)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = correct / len(val_loader.dataset)

    # Log validation metrics
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)

    # Log learning rate
    writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

    # Log histograms of weights
    for name, param in model.named_parameters():
        writer.add_histogram(name, param, epoch)
        if param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

    # Log images (every 10 epochs)
    if epoch % 10 == 0:
        # Get sample batch
        sample_data, _ = next(iter(val_loader))
        img_grid = torchvision.utils.make_grid(sample_data[:16])
        writer.add_image('Sample_images', img_grid, epoch)

# Log model graph
sample_input, _ = next(iter(train_loader))
writer.add_graph(model, sample_input)

# Log hyperparameters
hparams = {
    'lr': 0.001,
    'batch_size': 64,
    'hidden_size': 256,
    'dropout': 0.3
}

metrics = {
    'final_val_loss': val_loss,
    'final_val_acc': val_acc
}

writer.add_hparams(hparams, metrics)

# Close writer
writer.close()
```

```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Open in browser: http://localhost:6006
```

### TensorBoard with Multiple Runs

```python
from torch.utils.tensorboard import SummaryWriter

def train_model(config, run_name):
    """Train model and log to TensorBoard."""

    writer = SummaryWriter(log_dir=f'runs/{run_name}')

    # Training loop
    for epoch in range(config['epochs']):
        # ... training ...

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

    writer.close()

    return val_acc

# Run multiple experiments
configs = [
    {'lr': 0.001, 'epochs': 50, 'hidden_size': 128},
    {'lr': 0.01, 'epochs': 50, 'hidden_size': 128},
    {'lr': 0.001, 'epochs': 50, 'hidden_size': 256},
]

for i, config in enumerate(configs):
    run_name = f"experiment_{i}_lr{config['lr']}_hs{config['hidden_size']}"
    train_model(config, run_name)
```

### TensorBoard Projector (Embeddings)

```python
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import datasets, transforms

writer = SummaryWriter()

# Get embeddings
model.eval()
embeddings = []
labels = []
images = []

with torch.no_grad():
    for data, target in test_loader:
        # Get embeddings (output of penultimate layer)
        embedding = model.get_embedding(data)
        embeddings.append(embedding)
        labels.append(target)
        images.append(data)

embeddings = torch.cat(embeddings)
labels = torch.cat(labels)
images = torch.cat(images)

# Log embeddings
writer.add_embedding(
    embeddings,
    metadata=labels,
    label_img=images,
    global_step=epoch,
    tag='fraud_embeddings'
)

writer.close()
```

---

## Comparison and Selection

### Feature Comparison

| Feature | MLflow | W&B | TensorBoard |
|---------|--------|-----|-------------|
| **Open Source** | Yes | No (free tier) | Yes |
| **Self-hosted** | Yes | No | Yes |
| **Ease of Use** | Medium | Easy | Medium |
| **Visualization** | Basic | Excellent | Good |
| **Collaboration** | Basic | Excellent | Basic |
| **Model Registry** | Yes | Yes | No |
| **Hyperparameter Sweeps** | No | Yes | No |
| **Artifacts** | Yes | Yes | Limited |
| **Scalability** | High | High | Medium |
| **Cost** | Free | Free tier + paid | Free |
| **Best For** | Open-source, full lifecycle | Teams, visualization | PyTorch/TF users |

### Selection Guide

```python
class ExperimentTrackingSelector:
    """Help select the right tracking tool."""

    @staticmethod
    def recommend(
        team_size: str,
        budget: str,
        framework: str,
        hosting: str
    ) -> str:
        """
        Recommend tracking tool.

        Args:
            team_size: 'individual', 'small' (2-10), 'large' (>10)
            budget: 'free', 'limited', 'enterprise'
            framework: 'pytorch', 'tensorflow', 'sklearn', 'mixed'
            hosting: 'cloud', 'self-hosted', 'either'
        """

        if budget == 'free' and hosting == 'self-hosted':
            if framework in ['pytorch', 'tensorflow']:
                return "TensorBoard (built-in, simple)"
            else:
                return "MLflow (open-source, full-featured)"

        elif team_size == 'large' and budget != 'free':
            return "Weights & Biases (best collaboration)"

        elif framework == 'sklearn':
            return "MLflow (best sklearn integration)"

        elif hosting == 'self-hosted':
            return "MLflow (flexible self-hosting)"

        else:
            return "Weights & Biases free tier or MLflow"

# Usage
recommendation = ExperimentTrackingSelector.recommend(
    team_size='small',
    budget='limited',
    framework='pytorch',
    hosting='either'
)
print(f"Recommended: {recommendation}")
```

---

## Reproducibility Best Practices

### Complete Reproducibility

```python
import torch
import numpy as np
import random
import os
import mlflow
import hashlib
import json
from datetime import datetime

class ReproducibleExperiment:
    """Ensure complete experiment reproducibility."""

    def __init__(self, experiment_name: str, seed: int = 42):
        self.experiment_name = experiment_name
        self.seed = seed
        self.run_id = None

        # Set all random seeds
        self._set_seeds()

        # Collect environment info
        self.environment = self._collect_environment()

    def _set_seeds(self):
        """Set all random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Make PyTorch deterministic
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set environment variable for PYTHONHASHSEED
        os.environ['PYTHONHASHSEED'] = str(self.seed)

    def _collect_environment(self) -> dict:
        """Collect complete environment information."""
        import platform
        import sys

        env = {
            # Python environment
            'python_version': sys.version,
            'python_executable': sys.executable,

            # System info
            'platform': platform.platform(),
            'processor': platform.processor(),

            # Package versions
            'numpy_version': np.__version__,
            'torch_version': torch.__version__,

            # CUDA info
            'cuda_available': torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            env['cuda_version'] = torch.version.cuda
            env['cudnn_version'] = torch.backends.cudnn.version()
            env['gpu_name'] = torch.cuda.get_device_name(0)
            env['gpu_count'] = torch.cuda.device_count()

        return env

    def start_run(self, config: dict):
        """Start reproducible run."""
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run() as run:
            self.run_id = run.info.run_id

            # Log config
            mlflow.log_params(config)

            # Log random seed
            mlflow.log_param("random_seed", self.seed)

            # Log environment
            for key, value in self.environment.items():
                mlflow.log_param(f"env_{key}", str(value))

            # Log git info
            git_info = self._get_git_info()
            if git_info:
                mlflow.log_params(git_info)

            # Save requirements
            self._save_requirements()

            return run

    def _get_git_info(self) -> dict:
        """Get git repository information."""
        try:
            import subprocess

            git_info = {}

            # Commit hash
            git_info['git_commit'] = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD']
            ).decode('utf-8').strip()

            # Branch
            git_info['git_branch'] = subprocess.check_output(
                ['git', 'rev-parse', '--abbrev-ref', 'HEAD']
            ).decode('utf-8').strip()

            # Check for uncommitted changes
            git_status = subprocess.check_output(
                ['git', 'status', '--porcelain']
            ).decode('utf-8').strip()

            git_info['git_dirty'] = len(git_status) > 0

            # Remote URL
            git_info['git_remote'] = subprocess.check_output(
                ['git', 'config', '--get', 'remote.origin.url']
            ).decode('utf-8').strip()

            return git_info

        except:
            return None

    def _save_requirements(self):
        """Save exact package versions."""
        import subprocess

        # Freeze requirements
        requirements = subprocess.check_output(
            ['pip', 'freeze']
        ).decode('utf-8')

        # Save to file
        with open('requirements_frozen.txt', 'w') as f:
            f.write(requirements)

        # Log as artifact
        mlflow.log_artifact('requirements_frozen.txt')

    def hash_dataset(self, data_path: str) -> str:
        """Compute dataset hash for versioning."""
        hash_md5 = hashlib.md5()

        with open(data_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        dataset_hash = hash_md5.hexdigest()

        # Log dataset hash
        mlflow.log_param('dataset_hash', dataset_hash)

        return dataset_hash

# Usage
experiment = ReproducibleExperiment(
    experiment_name="fraud_detection",
    seed=42
)

config = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'epochs': 50
}

with experiment.start_run(config):
    # Hash dataset
    dataset_hash = experiment.hash_dataset('/data/train.csv')

    # Training code
    # ... guaranteed to be reproducible ...

    # Log results
    mlflow.log_metric('accuracy', 0.95)
```

---

## Model Registry

### Production Model Management

```python
from mlflow.tracking import MlflowClient
from datetime import datetime
import json

class ProductionModelRegistry:
    """Manage production model lifecycle."""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()

    def register_model(
        self,
        run_id: str,
        model_name: str,
        description: str = None
    ) -> int:
        """Register model from run."""

        model_uri = f"runs:/{run_id}/model"

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)

        # Add description
        if description:
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=description
            )

        print(f" Registered {model_name} version {model_version.version}")

        return model_version.version

    def promote_to_staging(
        self,
        model_name: str,
        version: int,
        approval_required: bool = True
    ):
        """Promote model to staging."""

        if approval_required:
            # Get approver
            approval = input(f"Approve promotion of {model_name} v{version} to Staging? (yes/no): ")
            if approval.lower() != 'yes':
                print(" Promotion cancelled")
                return

        # Transition to staging
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Staging"
        )

        # Add transition metadata
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="promoted_to_staging_at",
            value=datetime.now().isoformat()
        )

        print(f" Promoted to Staging")

    def promote_to_production(
        self,
        model_name: str,
        version: int,
        metrics_threshold: dict = None,
        approval_required: bool = True
    ):
        """Promote model to production with validation."""

        # Validate metrics if threshold provided
        if metrics_threshold:
            run = self.client.get_run(
                self.client.get_model_version(model_name, version).run_id
            )

            for metric, threshold in metrics_threshold.items():
                actual_value = run.data.metrics.get(metric)

                if actual_value is None:
                    print(f" Metric {metric} not found")
                    return

                if actual_value < threshold:
                    print(f" {metric}={actual_value:.4f} below threshold {threshold}")
                    return

        # Require approval
        if approval_required:
            approval = input(f"Approve promotion of {model_name} v{version} to Production? (yes/no): ")
            if approval.lower() != 'yes':
                print(" Promotion cancelled")
                return

        # Transition to production, archiving existing
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        # Add metadata
        self.client.set_model_version_tag(
            name=model_name,
            version=version,
            key="promoted_to_production_at",
            value=datetime.now().isoformat()
        )

        print(f" Promoted to Production")

    def get_production_model(self, model_name: str):
        """Get current production model."""

        versions = self.client.get_latest_versions(
            model_name,
            stages=["Production"]
        )

        if not versions:
            return None

        return versions[0]

    def rollback_production(self, model_name: str):
        """Rollback to previous production version."""

        # Get current production version
        current = self.get_production_model(model_name)

        if not current:
            print("No production model to rollback")
            return

        # Get archived versions
        archived = self.client.get_latest_versions(
            model_name,
            stages=["Archived"]
        )

        if not archived:
            print("No archived model to rollback to")
            return

        # Get most recent archived (previous production)
        previous = max(archived, key=lambda v: v.last_updated_timestamp)

        # Promote previous to production
        self.client.transition_model_version_stage(
            name=model_name,
            version=previous.version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f" Rolled back to version {previous.version}")
        print(f"   Current version {current.version} archived")

# Usage
registry = ProductionModelRegistry()

# Register model
version = registry.register_model(
    run_id="abc123",
    model_name="fraud_detection",
    description="XGBoost model trained on Oct 2025 data"
)

# Promote to staging
registry.promote_to_staging("fraud_detection", version)

# Promote to production with validation
registry.promote_to_production(
    model_name="fraud_detection",
    version=version,
    metrics_threshold={'auc': 0.95, 'accuracy': 0.90}
)

# Get production model
prod_model = registry.get_production_model("fraud_detection")
print(f"Production version: {prod_model.version}")

# Rollback if needed
registry.rollback_production("fraud_detection")
```

---

## Production Integration

### Integrating Tracking with Deployment

```python
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

# Load model from registry
mlflow.set_tracking_uri("http://mlflow:5000")
model = mlflow.pyfunc.load_model("models:/fraud_detection/Production")

# Get model version info
client = mlflow.tracking.MlflowClient()
production_versions = client.get_latest_versions("fraud_detection", stages=["Production"])
model_version = production_versions[0].version
model_run_id = production_versions[0].run_id

# Create new run for production monitoring
production_run = mlflow.start_run(run_name=f"production_serving_v{model_version}")

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction and log to MLflow."""

    # Make prediction
    features_df = pd.DataFrame([request.features])
    prediction = model.predict(features_df)[0]
    probability = model.predict_proba(features_df)[0][1]

    # Log prediction (for monitoring)
    with mlflow.start_run(run_id=production_run.info.run_id):
        mlflow.log_metric("predictions_count", 1)
        mlflow.log_metric("prediction_value", prediction)

    return PredictionResponse(
        prediction=int(prediction),
        probability=float(probability),
        model_version=model_version
    )

@app.get("/model_info")
async def model_info():
    """Get production model information."""
    return {
        "model_version": model_version,
        "run_id": model_run_id,
        "stage": "Production"
    }
```

---

## Best Practices

### 1. Experiment Organization

```python
# Use hierarchical experiment names
mlflow.set_experiment("fraud_detection/baseline")
mlflow.set_experiment("fraud_detection/hyperparameter_tuning")
mlflow.set_experiment("fraud_detection/production")

# Use consistent tags
tags = {
    "task": "classification",
    "dataset": "fraud_v2",
    "author": "data_science_team",
    "priority": "high",
    "deployment_target": "production"
}
mlflow.set_tags(tags)

# Use meaningful run names
run_name = f"xgboost_n{n_estimators}_d{max_depth}_{datetime.now().strftime('%Y%m%d_%H%M')}"
```

### 2. Comprehensive Logging

```python
def comprehensive_logging(model, X_train, X_test, y_train, y_test, config):
    """Log everything needed for reproducibility."""

    with mlflow.start_run():
        # 1. Parameters
        mlflow.log_params(config)

        # 2. Environment
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("cuda_available", torch.cuda.is_available())

        # 3. Data info
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # 4. Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # 5. Metrics
        y_pred = model.predict(X_test)
        mlflow.log_metrics({
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "training_time_seconds": training_time
        })

        # 6. Model
        mlflow.sklearn.log_model(model, "model")

        # 7. Artifacts
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure()
        sns.heatmap(cm, annot=True, fmt='d')
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")

        # Feature importance
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': X_train.columns,
                'importance': model.feature_importances_
            })
            importance_df.to_csv("feature_importance.csv")
            mlflow.log_artifact("feature_importance.csv")
```

---

## Summary

Experiment tracking in 2025:

1. **Choose Your Tool:**
   - **MLflow:** Open-source, self-hosted, full ML lifecycle
   - **W&B:** Best visualization and collaboration, cloud-based
   - **TensorBoard:** Built-in for PyTorch/TensorFlow, simple

2. **Track Everything:**
   - Parameters, metrics, artifacts
   - Environment, git info, data versions
   - Training time, resource usage

3. **Ensure Reproducibility:**
   - Set random seeds
   - Log environment details
   - Version data and code together
   - Save exact package versions

4. **Use Model Registry:**
   - Version all models
   - Stage-based promotion (Staging -> Production)
   - Metadata and approval workflows

5. **Integrate with Production:**
   - Link experiments to deployments
   - Monitor production predictions
   - Enable rollback capabilities

**Next Steps:**
- Set up deployment pipeline (Section 36)
- Implement monitoring (Section 37)
- Automate with orchestration (Section 38)