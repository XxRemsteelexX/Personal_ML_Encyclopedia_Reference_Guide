# ML Pipeline Orchestration (2025 Best Practices)

## Table of Contents
1. [Introduction](#introduction)
2. [Pipeline Fundamentals](#pipeline-fundamentals)
3. [Apache Airflow](#apache-airflow)
4. [Kubeflow Pipelines](#kubeflow-pipelines)
5. [MLflow Projects](#mlflow-projects)
6. [Modern Alternatives](#modern-alternatives)
7. [Data Versioning with DVC](#data-versioning-with-dvc)
8. [Feature Stores](#feature-stores)
9. [End-to-End Examples](#end-to-end-examples)
10. [Best Practices](#best-practices)

---

## Introduction

ML pipeline orchestration automates the end-to-end machine learning workflow, from data ingestion to model deployment. Proper orchestration is essential for production ML systems in 2025.

### Why Orchestration Matters

- **Reproducibility:** Ensure experiments and deployments are consistent
- **Automation:** Reduce manual intervention and errors
- **Scalability:** Handle large-scale data and complex workflows
- **Monitoring:** Track pipeline execution and detect failures
- **Collaboration:** Enable team members to understand and modify workflows

### Key Components

1. **Data Ingestion:** Collect and validate raw data
2. **Preprocessing:** Clean, transform, feature engineering
3. **Training:** Model training with hyperparameter tuning
4. **Evaluation:** Validate model performance
5. **Deployment:** Deploy to production
6. **Monitoring:** Track model and pipeline health

### 2025 Landscape

- **DAG-based Orchestration:** Still dominant (Airflow, Prefect, Dagster)
- **Kubernetes-Native:** Kubeflow, Argo Workflows for cloud-native ML
- **Serverless Pipelines:** AWS Step Functions, GCP Workflows
- **Feature Stores:** Centralized feature management (Feast, Tecton)
- **GitOps for ML:** Infrastructure and pipelines as code

---

## Pipeline Fundamentals

### Directed Acyclic Graph (DAG)

ML pipelines are typically represented as DAGs, where:
- **Nodes** represent tasks (data loading, training, etc.)
- **Edges** represent dependencies between tasks
- **Acyclic** ensures no circular dependencies

```python
from dataclasses import dataclass
from typing import List, Callable, Any, Dict
from enum import Enum
import logging

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class Task:
    """Represents a single task in the pipeline."""
    name: str
    function: Callable
    dependencies: List[str] = None
    params: Dict[str, Any] = None
    status: TaskStatus = TaskStatus.PENDING

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.params is None:
            self.params = {}

class SimplePipeline:
    """Simple DAG-based pipeline executor."""

    def __init__(self, name: str):
        self.name = name
        self.tasks: Dict[str, Task] = {}
        self.logger = logging.getLogger(__name__)

    def add_task(self, task: Task):
        """Add task to pipeline."""
        self.tasks[task.name] = task

    def validate_dag(self) -> bool:
        """Validate that pipeline is a valid DAG (no cycles)."""
        visited = set()
        rec_stack = set()

        def has_cycle(task_name: str) -> bool:
            visited.add(task_name)
            rec_stack.add(task_name)

            task = self.tasks[task_name]
            for dep in task.dependencies:
                if dep not in visited:
                    if has_cycle(dep):
                        return True
                elif dep in rec_stack:
                    return True

            rec_stack.remove(task_name)
            return False

        for task_name in self.tasks:
            if task_name not in visited:
                if has_cycle(task_name):
                    return False

        return True

    def get_execution_order(self) -> List[str]:
        """Get topological sort of tasks."""
        in_degree = {name: 0 for name in self.tasks}

        # Calculate in-degrees
        for task in self.tasks.values():
            for dep in task.dependencies:
                in_degree[task.name] += 1

        # Find tasks with no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        execution_order = []

        while queue:
            task_name = queue.pop(0)
            execution_order.append(task_name)

            # Update in-degrees
            for other_task in self.tasks.values():
                if task_name in other_task.dependencies:
                    in_degree[other_task.name] -= 1
                    if in_degree[other_task.name] == 0:
                        queue.append(other_task.name)

        return execution_order

    def run(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute pipeline."""
        if not self.validate_dag():
            raise ValueError("Pipeline contains cycles!")

        if context is None:
            context = {}

        execution_order = self.get_execution_order()
        results = {}

        self.logger.info(f"Starting pipeline: {self.name}")
        self.logger.info(f"Execution order: {execution_order}")

        for task_name in execution_order:
            task = self.tasks[task_name]

            # Check if dependencies succeeded
            deps_failed = any(
                self.tasks[dep].status == TaskStatus.FAILED
                for dep in task.dependencies
            )

            if deps_failed:
                task.status = TaskStatus.SKIPPED
                self.logger.warning(f"Skipping {task_name} due to failed dependencies")
                continue

            # Execute task
            try:
                task.status = TaskStatus.RUNNING
                self.logger.info(f"Running task: {task_name}")

                result = task.function(context, results, **task.params)
                results[task_name] = result

                task.status = TaskStatus.SUCCESS
                self.logger.info(f" Task {task_name} completed")

            except Exception as e:
                task.status = TaskStatus.FAILED
                self.logger.error(f" Task {task_name} failed: {e}")
                results[task_name] = None

        # Pipeline summary
        success_count = sum(1 for t in self.tasks.values() if t.status == TaskStatus.SUCCESS)
        total_count = len(self.tasks)

        self.logger.info(f"Pipeline complete: {success_count}/{total_count} tasks succeeded")

        return results

# Example usage
def load_data(context, results, data_path):
    """Load data task."""
    import pandas as pd
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df

def preprocess_data(context, results):
    """Preprocess data task."""
    df = results['load_data']
    # Preprocessing logic
    print(f"Preprocessed {len(df)} records")
    return df

def train_model(context, results, model_type='rf'):
    """Train model task."""
    df = results['preprocess_data']
    # Training logic
    print(f"Trained {model_type} model")
    return {'model_type': model_type, 'accuracy': 0.95}

# Build pipeline
pipeline = SimplePipeline(name="ml_training_pipeline")

pipeline.add_task(Task(
    name="load_data",
    function=load_data,
    params={'data_path': '/data/train.csv'}
))

pipeline.add_task(Task(
    name="preprocess_data",
    function=preprocess_data,
    dependencies=["load_data"]
))

pipeline.add_task(Task(
    name="train_model",
    function=train_model,
    dependencies=["preprocess_data"],
    params={'model_type': 'xgboost'}
))

# Run pipeline
results = pipeline.run()
```

---

## Apache Airflow

Airflow is the most widely adopted ML orchestration tool in 2025. It uses Python DAGs to define workflows.

### Installation and Setup

```bash
# Install Airflow
pip install apache-airflow==2.7.0
pip install apache-airflow-providers-amazon
pip install apache-airflow-providers-google

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Start webserver
airflow webserver --port 8080

# Start scheduler (in separate terminal)
airflow scheduler
```

### Basic ML Pipeline with Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import pandas as pd
import joblib

# Default arguments
default_args = {
    'owner': 'ml_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='Daily model training pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    catchup=False,
    tags=['ml', 'production']
)

def extract_data(**context):
    """Extract data from S3."""
    execution_date = context['execution_date']

    s3_hook = S3Hook(aws_conn_id='aws_default')

    # Download data from S3
    file_key = f"data/raw/{execution_date.strftime('%Y-%m-%d')}/data.csv"
    local_path = f"/tmp/data_{execution_date.strftime('%Y%m%d')}.csv"

    s3_hook.download_file(
        key=file_key,
        bucket_name='ml-data-bucket',
        local_path=local_path
    )

    # Push path to XCom
    context['task_instance'].xcom_push(key='data_path', value=local_path)

    print(f" Downloaded data: {file_key}")

def preprocess_data(**context):
    """Preprocess data."""
    # Pull data path from XCom
    ti = context['task_instance']
    data_path = ti.xcom_pull(task_ids='extract_data', key='data_path')

    # Load and preprocess
    df = pd.read_csv(data_path)

    # Preprocessing logic
    df = df.dropna()
    df = df[df['value'] > 0]

    # Feature engineering
    df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-6)

    # Save preprocessed data
    processed_path = data_path.replace('data_', 'processed_')
    df.to_parquet(processed_path, index=False)

    ti.xcom_push(key='processed_data_path', value=processed_path)

    print(f" Preprocessed {len(df)} records")

def train_model(**context):
    """Train ML model."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score

    ti = context['task_instance']
    data_path = ti.xcom_pull(task_ids='preprocess_data', key='processed_data_path')

    # Load data
    df = pd.read_parquet(data_path)

    # Split features and target
    X = df.drop('target', axis=1)
    y = df['target']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f" Model trained - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

    # Save model
    execution_date = context['execution_date']
    model_path = f"/tmp/model_{execution_date.strftime('%Y%m%d')}.pkl"
    joblib.dump(model, model_path)

    # Push metrics and model path
    ti.xcom_push(key='model_path', value=model_path)
    ti.xcom_push(key='accuracy', value=accuracy)
    ti.xcom_push(key='auc', value=auc)

def validate_model(**context):
    """Validate model meets quality thresholds."""
    ti = context['task_instance']
    accuracy = ti.xcom_pull(task_ids='train_model', key='accuracy')
    auc = ti.xcom_pull(task_ids='train_model', key='auc')

    # Validation thresholds
    min_accuracy = 0.85
    min_auc = 0.90

    if accuracy < min_accuracy or auc < min_auc:
        raise ValueError(
            f"Model quality below threshold: "
            f"Accuracy={accuracy:.3f} (min={min_accuracy}), "
            f"AUC={auc:.3f} (min={min_auc})"
        )

    print(f" Model validation passed")

def deploy_model(**context):
    """Deploy model to S3."""
    ti = context['task_instance']
    model_path = ti.xcom_pull(task_ids='train_model', key='model_path')
    execution_date = context['execution_date']

    s3_hook = S3Hook(aws_conn_id='aws_default')

    # Upload to S3
    s3_key = f"models/production/model_{execution_date.strftime('%Y%m%d')}.pkl"
    s3_hook.load_file(
        filename=model_path,
        key=s3_key,
        bucket_name='ml-models-bucket',
        replace=True
    )

    print(f" Model deployed: {s3_key}")

# Define tasks
task_extract = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    provide_context=True,
    dag=dag
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    provide_context=True,
    dag=dag
)

task_train = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    provide_context=True,
    dag=dag
)

task_validate = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    provide_context=True,
    dag=dag
)

task_deploy = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    provide_context=True,
    dag=dag
)

task_notify = BashOperator(
    task_id='send_notification',
    bash_command='echo "Pipeline completed successfully" | mail -s "ML Pipeline Success" ml-team@company.com',
    dag=dag
)

# Define dependencies
task_extract >> task_preprocess >> task_train >> task_validate >> task_deploy >> task_notify
```

### Advanced Airflow: Dynamic Task Generation

```python
from airflow.decorators import dag, task
from datetime import datetime
from typing import List

@dag(
    schedule_interval='@daily',
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'dynamic']
)
def dynamic_ml_pipeline():
    """Dynamic pipeline with multiple models."""

    @task
    def get_models() -> List[str]:
        """Return list of models to train."""
        return ['random_forest', 'xgboost', 'lightgbm', 'catboost']

    @task
    def train_single_model(model_name: str) -> dict:
        """Train a single model."""
        print(f"Training {model_name}...")

        # Training logic
        accuracy = 0.90 + (hash(model_name) % 10) / 100

        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'model_path': f'/models/{model_name}.pkl'
        }

    @task
    def select_best_model(model_results: List[dict]) -> dict:
        """Select best performing model."""
        best_model = max(model_results, key=lambda x: x['accuracy'])
        print(f"Best model: {best_model['model_name']} (accuracy={best_model['accuracy']:.3f})")
        return best_model

    @task
    def deploy_best_model(best_model: dict):
        """Deploy the best model."""
        print(f"Deploying {best_model['model_name']}...")
        # Deployment logic
        print(f" Deployed {best_model['model_name']}")

    # Pipeline flow
    models = get_models()
    model_results = train_single_model.expand(model_name=models)
    best_model = select_best_model(model_results)
    deploy_best_model(best_model)

# Instantiate DAG
dynamic_pipeline = dynamic_ml_pipeline()
```

---

## Kubeflow Pipelines

Kubeflow Pipelines is Kubernetes-native ML orchestration, ideal for cloud deployments.

### Installation

```bash
# Install Kubeflow Pipelines SDK
pip install kfp

# Deploy Kubeflow Pipelines to Kubernetes cluster
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=2.0.0"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/platform-agnostic?ref=2.0.0"
```

### Kubeflow Pipeline Example

```python
from kfp import dsl, compiler
from kfp.dsl import component, Input, Output, Dataset, Model, Metrics

# Define components as containerized functions
@component(
    base_image='python:3.11',
    packages_to_install=['pandas', 'scikit-learn']
)
def load_data(
    data_path: str,
    output_dataset: Output[Dataset]
):
    """Load data component."""
    import pandas as pd

    df = pd.read_csv(data_path)
    df.to_csv(output_dataset.path, index=False)

    print(f"Loaded {len(df)} records")

@component(
    base_image='python:3.11',
    packages_to_install=['pandas', 'scikit-learn']
)
def preprocess_data(
    input_dataset: Input[Dataset],
    output_dataset: Output[Dataset]
):
    """Preprocess data component."""
    import pandas as pd

    df = pd.read_csv(input_dataset.path)

    # Preprocessing
    df = df.dropna()
    df = df[df['value'] > 0]

    df.to_csv(output_dataset.path, index=False)

    print(f"Preprocessed {len(df)} records")

@component(
    base_image='python:3.11',
    packages_to_install=['pandas', 'scikit-learn', 'joblib']
)
def train_model(
    input_dataset: Input[Dataset],
    output_model: Output[Model],
    metrics: Output[Metrics],
    n_estimators: int = 100
):
    """Train model component."""
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import joblib

    # Load data
    df = pd.read_csv(input_dataset.path)

    X = df.drop('target', axis=1)
    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    # Save model
    joblib.dump(model, output_model.path)

    # Log metrics
    metrics.log_metric('accuracy', accuracy)
    metrics.log_metric('auc', auc)

    print(f"Model trained - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

@component(
    base_image='python:3.11',
    packages_to_install=['joblib']
)
def deploy_model(
    input_model: Input[Model],
    model_name: str = "production_model"
):
    """Deploy model component."""
    import joblib
    import shutil

    # Copy model to deployment location
    deployment_path = f"/models/{model_name}.pkl"
    shutil.copy(input_model.path, deployment_path)

    print(f"Model deployed to {deployment_path}")

# Define pipeline
@dsl.pipeline(
    name='ML Training Pipeline',
    description='End-to-end ML training pipeline'
)
def ml_training_pipeline(
    data_path: str = 's3://bucket/data.csv',
    n_estimators: int = 100
):
    """ML training pipeline."""

    # Load data
    load_task = load_data(data_path=data_path)

    # Preprocess
    preprocess_task = preprocess_data(
        input_dataset=load_task.outputs['output_dataset']
    )

    # Train
    train_task = train_model(
        input_dataset=preprocess_task.outputs['output_dataset'],
        n_estimators=n_estimators
    )

    # Deploy
    deploy_task = deploy_model(
        input_model=train_task.outputs['output_model'],
        model_name='fraud_detection_v3'
    )

# Compile pipeline
compiler.Compiler().compile(
    pipeline_func=ml_training_pipeline,
    package_path='ml_pipeline.yaml'
)

# Run pipeline
from kfp import Client

client = Client(host='http://kubeflow-pipelines:8080')

run = client.create_run_from_pipeline_func(
    ml_training_pipeline,
    arguments={
        'data_path': 's3://ml-data/train.csv',
        'n_estimators': 200
    }
)

print(f"Pipeline run created: {run.run_id}")
```

---

## MLflow Projects

MLflow Projects package ML code for reproducible runs.

### MLflow Project Structure

```yaml
# MLproject file
name: fraud_detection_training

conda_env: conda.yaml

entry_points:
  main:
    parameters:
      data_path: {type: str, default: "/data/train.csv"}
      learning_rate: {type: float, default: 0.01}
      n_estimators: {type: int, default: 100}
      max_depth: {type: int, default: 10}
    command: "python train.py --data-path {data_path} --learning-rate {learning_rate} --n-estimators {n_estimators} --max-depth {max_depth}"

  evaluate:
    parameters:
      model_path: {type: str}
      test_data: {type: str}
    command: "python evaluate.py --model-path {model_path} --test-data {test_data}"
```

```yaml
# conda.yaml
name: ml_env
channels:
  - defaults
  - conda-forge
dependencies:
  - python=3.11
  - pip
  - pip:
    - mlflow==2.8.0
    - pandas==2.0.0
    - scikit-learn==1.3.0
    - xgboost==2.0.0
```

```python
# train.py
import mlflow
import mlflow.sklearn
import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

def train(data_path, learning_rate, n_estimators, max_depth):
    """Train model with MLflow tracking."""

    # Load data
    df = pd.read_csv(data_path)
    X = df.drop('target', axis=1)
    y = df['target']

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("data_path", data_path)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
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
        mlflow.sklearn.log_model(model, "model")

        print(f" Training complete - Accuracy: {accuracy:.3f}, AUC: {auc:.3f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=10)

    args = parser.parse_args()
    train(args.data_path, args.learning_rate, args.n_estimators, args.max_depth)
```

**Run MLflow Project:**

```bash
# Run locally
mlflow run . \
    -P data_path=/data/train.csv \
    -P n_estimators=200 \
    -P max_depth=15

# Run from Git
mlflow run https://github.com/myorg/ml-project.git \
    -v main \
    -P data_path=s3://bucket/data.csv

# Run on Kubernetes
mlflow run . \
    --backend kubernetes \
    --backend-config kubernetes_config.json
```

---

## Modern Alternatives

### 1. Prefect (Modern Airflow Alternative)

```python
from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import timedelta
import pandas as pd

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def load_data(data_path: str) -> pd.DataFrame:
    """Load data with caching."""
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} records")
    return df

@task
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess data."""
    df = df.dropna()
    print(f"Preprocessed {len(df)} records")
    return df

@task(retries=3, retry_delay_seconds=60)
def train_model(df: pd.DataFrame, model_type: str = 'rf') -> dict:
    """Train model with retries."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    return {'model': model, 'accuracy': accuracy}

@task
def deploy_model(model_result: dict):
    """Deploy model."""
    if model_result['accuracy'] > 0.85:
        print(f" Deploying model with accuracy {model_result['accuracy']:.3f}")
    else:
        raise ValueError(f"Model accuracy {model_result['accuracy']:.3f} below threshold")

@flow(name="ml-training-pipeline")
def ml_pipeline(data_path: str = "/data/train.csv"):
    """ML training pipeline with Prefect."""

    # Execute tasks
    df = load_data(data_path)
    df_processed = preprocess_data(df)
    model_result = train_model(df_processed)
    deploy_model(model_result)

# Run pipeline
if __name__ == "__main__":
    ml_pipeline()
```

**Deploy to Prefect Cloud:**

```python
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule

deployment = Deployment.build_from_flow(
    flow=ml_pipeline,
    name="daily-ml-training",
    schedule=CronSchedule(cron="0 2 * * *"),  # 2 AM daily
    work_pool_name="ml-pool"
)

deployment.apply()
```

### 2. Dagster (Asset-Based Orchestration)

```python
from dagster import asset, AssetExecutionContext, Definitions
import pandas as pd

@asset
def raw_data(context: AssetExecutionContext) -> pd.DataFrame:
    """Load raw data asset."""
    df = pd.read_csv("/data/train.csv")
    context.log.info(f"Loaded {len(df)} records")
    return df

@asset
def processed_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """Processed data asset."""
    df = raw_data.dropna()
    df = df[df['value'] > 0]
    return df

@asset
def trained_model(context: AssetExecutionContext, processed_data: pd.DataFrame) -> dict:
    """Trained model asset."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X = processed_data.drop('target', axis=1)
    y = processed_data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))

    context.log.info(f"Model accuracy: {accuracy:.3f}")

    return {'model': model, 'accuracy': accuracy}

# Define assets
defs = Definitions(
    assets=[raw_data, processed_data, trained_model]
)
```

---

## Data Versioning with DVC

DVC (Data Version Control) tracks data and models alongside code.

### Setup DVC

```bash
# Install DVC
pip install dvc dvc-s3

# Initialize DVC in git repo
git init
dvc init

# Configure remote storage
dvc remote add -d myremote s3://my-bucket/dvc-storage

# Track data files
dvc add data/train.csv
git add data/train.csv.dvc data/.gitignore
git commit -m "Track training data"

# Push data to remote
dvc push
```

### DVC Pipeline

```yaml
# dvc.yaml - Define pipeline stages
stages:
  preprocess:
    cmd: python src/preprocess.py --input data/raw.csv --output data/processed.csv
    deps:
      - src/preprocess.py
      - data/raw.csv
    outs:
      - data/processed.csv

  train:
    cmd: python src/train.py --data data/processed.csv --output models/model.pkl
    deps:
      - src/train.py
      - data/processed.csv
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py --model models/model.pkl --data data/test.csv
    deps:
      - src/evaluate.py
      - models/model.pkl
      - data/test.csv
    metrics:
      - eval_metrics.json:
          cache: false
```

```bash
# Run pipeline
dvc repro

# Show metrics
dvc metrics show

# Compare experiments
dvc metrics diff main

# Pull data and models
dvc pull
```

---

## Feature Stores

Feature stores centralize feature engineering and serving.

### Feast (Open-Source Feature Store)

```python
from feast import FeatureStore, Entity, FeatureView, Field
from feast.types import Float32, Int64
from feast.data_source import FileSource
from datetime import timedelta

# Define entity
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity"
)

# Define feature view
user_features_source = FileSource(
    path="/data/user_features.parquet",
    timestamp_field="event_timestamp"
)

user_features_view = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="total_purchases", dtype=Int64),
        Field(name="avg_transaction_value", dtype=Float32),
        Field(name="days_since_signup", dtype=Int64),
    ],
    source=user_features_source,
    ttl=timedelta(days=7)
)

# Initialize feature store
store = FeatureStore(repo_path="/feast_repo")

# Materialize features to online store
store.materialize_incremental(end_date=datetime.now())

# Fetch features for training
from feast import FeatureService

training_fs = FeatureService(
    name="training_features",
    features=[user_features_view]
)

# Get training data
entity_df = pd.DataFrame({
    "user_id": [1, 2, 3, 4, 5],
    "event_timestamp": [datetime.now()] * 5
})

training_data = store.get_historical_features(
    entity_df=entity_df,
    features=training_fs
).to_df()

# Online feature retrieval (for inference)
features = store.get_online_features(
    features=[
        "user_features:total_purchases",
        "user_features:avg_transaction_value"
    ],
    entity_rows=[{"user_id": 123}]
).to_dict()

print(f"User 123 features: {features}")
```

---

## End-to-End Examples

### Complete Production Pipeline

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

class MLPipeline:
    """Complete ML pipeline with DVC, MLflow, and Airflow."""

    def __init__(self, run_date):
        self.run_date = run_date
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("fraud_detection")

    def extract_data(self):
        """Extract data with DVC."""
        import subprocess

        # Pull latest data
        subprocess.run(["dvc", "pull"], check=True)

        # Load data
        df = pd.read_csv("data/raw.csv")
        return df

    def preprocess(self, df):
        """Preprocess with feature store."""
        from feast import FeatureStore

        # Clean data
        df = df.dropna()

        # Fetch features from feature store
        store = FeatureStore(repo_path="/feast_repo")

        entity_df = df[['user_id', 'timestamp']].rename(
            columns={'timestamp': 'event_timestamp'}
        )

        features = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "user_features:total_purchases",
                "user_features:avg_transaction_value",
                "transaction_features:amount",
                "transaction_features:merchant_category"
            ]
        ).to_df()

        # Merge with original data
        df = df.merge(features, on='user_id')

        return df

    def train(self, df):
        """Train with MLflow tracking."""
        with mlflow.start_run():
            # Log parameters
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 10
            }
            mlflow.log_params(params)

            # Split data
            X = df.drop('is_fraud', axis=1)
            y = df['is_fraud']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, stratify=y
            )

            # Train
            model = RandomForestClassifier(**params, random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'auc': roc_auc_score(y_test, y_prob)
            }
            mlflow.log_metrics(metrics)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="fraud_detection"
            )

            # Save with DVC
            model_path = f"models/model_{self.run_date}.pkl"
            joblib.dump(model, model_path)

            import subprocess
            subprocess.run(["dvc", "add", model_path], check=True)

            return model, metrics

    def validate(self, metrics):
        """Validate model quality."""
        if metrics['accuracy'] < 0.90 or metrics['auc'] < 0.95:
            raise ValueError(f"Model quality below threshold: {metrics}")

        return True

    def deploy(self, model):
        """Deploy model to production."""
        # Promote to production in MLflow
        client = mlflow.tracking.MlflowClient()

        # Get latest version
        versions = client.get_latest_versions("fraud_detection", stages=["None"])
        latest_version = versions[0].version

        # Promote to production
        client.transition_model_version_stage(
            name="fraud_detection",
            version=latest_version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f" Model v{latest_version} promoted to Production")

# Airflow DAG
default_args = {
    'owner': 'ml_team',
    'start_date': datetime(2025, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'complete_ml_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

def run_pipeline(**context):
    """Execute complete pipeline."""
    run_date = context['ds']

    pipeline = MLPipeline(run_date)

    # Extract
    df = pipeline.extract_data()

    # Preprocess
    df_processed = pipeline.preprocess(df)

    # Train
    model, metrics = pipeline.train(df_processed)

    # Validate
    pipeline.validate(metrics)

    # Deploy
    pipeline.deploy(model)

task = PythonOperator(
    task_id='run_ml_pipeline',
    python_callable=run_pipeline,
    provide_context=True,
    dag=dag
)
```

---

## Best Practices

### 1. Idempotent Tasks

```python
@task
def idempotent_preprocessing(input_path: str, output_path: str):
    """Idempotent task - same input always produces same output."""
    import hashlib

    # Check if output exists and matches input hash
    input_hash = hashlib.md5(open(input_path, 'rb').read()).hexdigest()

    if os.path.exists(output_path):
        # Check metadata
        metadata_path = f"{output_path}.metadata"
        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                cached_hash = f.read().strip()

            if cached_hash == input_hash:
                print("Using cached result")
                return

    # Process data
    df = pd.read_csv(input_path)
    df_processed = df.dropna()
    df_processed.to_csv(output_path, index=False)

    # Save metadata
    with open(f"{output_path}.metadata", 'w') as f:
        f.write(input_hash)
```

### 2. Error Handling and Retries

```python
from airflow.exceptions import AirflowException
import time

@task(retries=3, retry_delay=timedelta(minutes=5))
def robust_task(**context):
    """Task with proper error handling."""
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            # Task logic
            result = perform_operation()
            return result

        except TransientError as e:
            if attempt < max_attempts - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise AirflowException(f"Task failed after {max_attempts} attempts: {e}")

        except PermanentError as e:
            raise AirflowException(f"Permanent error, not retrying: {e}")
```

### 3. Monitoring and Alerting

```python
from airflow.operators.python import get_current_context
from airflow.hooks.base import BaseHook

def send_slack_alert(message: str, status: str = "info"):
    """Send Slack alert."""
    import requests

    webhook_url = BaseHook.get_connection('slack').password

    color = {"info": "#36a64f", "warning": "#ff9900", "error": "#ff0000"}

    payload = {
        "attachments": [{
            "color": color.get(status, "#808080"),
            "text": message
        }]
    }

    requests.post(webhook_url, json=payload)

@task
def monitored_task():
    """Task with monitoring."""
    context = get_current_context()

    try:
        # Task logic
        result = perform_work()

        # Success notification
        send_slack_alert(
            f"Task {context['task_instance'].task_id} succeeded",
            status="info"
        )

        return result

    except Exception as e:
        # Error notification
        send_slack_alert(
            f"Task {context['task_instance'].task_id} failed: {str(e)}",
            status="error"
        )
        raise
```

---

## Summary

ML pipeline orchestration in 2025 requires:

1. **Choose the Right Tool:**
   - **Airflow:** Most mature, large ecosystem, Python-based
   - **Kubeflow:** Kubernetes-native, cloud-focused
   - **Prefect/Dagster:** Modern alternatives with better UX
   - **MLflow:** Focused on ML lifecycle, less orchestration

2. **Version Everything:**
   - Code (Git)
   - Data (DVC)
   - Models (MLflow Registry)
   - Pipelines (Git + CI/CD)

3. **Centralize Features:**
   - Use feature stores (Feast, Tecton)
   - Ensure consistency between training and serving
   - Enable feature reuse across models

4. **Monitor Pipelines:**
   - Track execution time and failure rates
   - Set up alerts for pipeline failures
   - Log data quality metrics

5. **Design for Failure:**
   - Idempotent tasks
   - Proper retries with backoff
   - Graceful degradation

**Next Steps:**
- Explore AutoML for automation (Section 39)
- Implement experiment tracking (Section 40)
- Deploy with CI/CD (Section 36)
