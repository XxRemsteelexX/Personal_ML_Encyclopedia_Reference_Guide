# AutoML and Neural Architecture Search (2025 State-of-the-Art)

## Table of Contents
1. [Introduction](#introduction)
2. [AutoML Fundamentals](#automl-fundamentals)
3. [AutoML Frameworks](#automl-frameworks)
4. [Neural Architecture Search](#neural-architecture-search)
5. [Hyperparameter Optimization](#hyperparameter-optimization)
6. [Automated Feature Engineering](#automated-feature-engineering)
7. [Production Deployment](#production-deployment)
8. [Best Practices](#best-practices)

---

## Introduction

Automated Machine Learning (AutoML) automates the process of applying machine learning to real-world problems. In 2025, AutoML has evolved to include sophisticated techniques for model selection, hyperparameter tuning, and neural architecture search.

### Why AutoML

- **Democratization:** Enables non-experts to build ML models
- **Efficiency:** Reduces time from weeks to hours
- **Performance:** Often outperforms manual approaches
- **Exploration:** Discovers architectures humans might miss
- **Consistency:** Standardized, reproducible workflows

### When to Use AutoML

**Good Use Cases:**
- Rapid prototyping and baseline establishment
- Tabular data with well-defined supervised learning tasks
- Limited ML expertise on team
- Time-constrained projects
- Hyperparameter optimization for existing models

**When NOT to Use:**
- Highly specialized domains requiring custom architectures
- Explainability is critical (AutoML can be black-box)
- Very limited computational budget
- Research requiring novel architectures

### 2025 Landscape

- **Foundation Model Fine-Tuning:** AutoML for LLM adaptation
- **Multi-Modal AutoML:** Automatic model selection for text, image, audio
- **Green AutoML:** Energy-efficient architecture search
- **Federated AutoML:** Privacy-preserving automated learning
- **Small Language Models:** Automated compression and distillation

---

## AutoML Fundamentals

### AutoML Pipeline Components

```python
from dataclasses import dataclass
from typing import List, Dict, Any
from enum import Enum

class AutoMLComponent(Enum):
    """Components of AutoML pipeline."""
    DATA_PREPROCESSING = "data_preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_SELECTION = "model_selection"
    HYPERPARAMETER_TUNING = "hyperparameter_tuning"
    ENSEMBLE = "ensemble"

@dataclass
class AutoMLConfig:
    """Configuration for AutoML pipeline."""

    # Task configuration
    task_type: str  # 'classification' or 'regression'
    metric: str  # 'accuracy', 'auc', 'rmse', etc.

    # Time and resource constraints
    max_runtime_seconds: int = 3600  # 1 hour default
    max_trials: int = 100
    n_jobs: int = -1  # Use all available cores

    # Components to enable
    enable_preprocessing: bool = True
    enable_feature_engineering: bool = True
    enable_ensemble: bool = True

    # Advanced options
    cross_validation_folds: int = 5
    early_stopping_rounds: int = 10
    optimization_metric: str = 'auto'

    # Model constraints
    max_model_size_mb: float = 100.0
    max_inference_latency_ms: float = 100.0

class SimpleAutoML:
    """Simple AutoML implementation for understanding concepts."""

    def __init__(self, config: AutoMLConfig):
        self.config = config
        self.models = self._get_model_search_space()
        self.best_model = None
        self.best_score = -np.inf if 'max' in config.metric else np.inf

    def _get_model_search_space(self) -> List[tuple]:
        """Define search space of models and hyperparameters."""
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from xgboost import XGBClassifier
        from lightgbm import LGBMClassifier

        if self.config.task_type == 'classification':
            return [
                ('logistic_regression', LogisticRegression, {
                    'C': [0.001, 0.01, 0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }),
                ('random_forest', RandomForestClassifier, {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [10, 20, 30, None],
                    'min_samples_split': [2, 5, 10]
                }),
                ('xgboost', XGBClassifier, {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'subsample': [0.8, 1.0]
                }),
                ('lightgbm', LGBMClassifier, {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15],
                    'learning_rate': [0.01, 0.1, 0.3],
                    'num_leaves': [31, 63, 127]
                })
            ]

    def fit(self, X, y):
        """Automated model selection and hyperparameter tuning."""
        from sklearn.model_selection import cross_val_score
        import itertools

        print(f"Starting AutoML with {len(self.models)} model types...")

        for model_name, model_class, param_grid in self.models:
            print(f"\nTesting {model_name}...")

            # Generate all combinations of hyperparameters
            param_names = list(param_grid.keys())
            param_values = list(param_grid.values())

            for param_combo in itertools.product(*param_values):
                params = dict(zip(param_names, param_combo))

                try:
                    # Create and train model
                    model = model_class(**params, random_state=42)

                    # Cross-validation
                    scores = cross_val_score(
                        model, X, y,
                        cv=self.config.cross_validation_folds,
                        scoring=self.config.metric,
                        n_jobs=self.config.n_jobs
                    )

                    mean_score = scores.mean()

                    # Update best model
                    if mean_score > self.best_score:
                        self.best_score = mean_score
                        self.best_model = model
                        self.best_params = params
                        self.best_model_name = model_name

                        print(f"  New best: {model_name} - {self.config.metric}={mean_score:.4f}")

                except Exception as e:
                    print(f"  Error with {params}: {e}")
                    continue

        # Train final model on full data
        print(f"\n✅ Best model: {self.best_model_name}")
        print(f"   Parameters: {self.best_params}")
        print(f"   {self.config.metric}: {self.best_score:.4f}")

        self.best_model.fit(X, y)

        return self

    def predict(self, X):
        """Make predictions with best model."""
        return self.best_model.predict(X)

# Usage example
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Configure AutoML
config = AutoMLConfig(
    task_type='classification',
    metric='accuracy',
    max_runtime_seconds=300,  # 5 minutes
    cross_validation_folds=5
)

# Run AutoML
automl = SimpleAutoML(config)
automl.fit(X_train, y_train)

# Predict
predictions = automl.predict(X_test)
print(f"\nTest accuracy: {(predictions == y_test).mean():.4f}")
```

---

## AutoML Frameworks

### 1. AutoGluon (Recommended for 2025)

AutoGluon is Amazon's AutoML framework, excelling at tabular, text, and multimodal data.

```python
from autogluon.tabular import TabularPredictor
import pandas as pd

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Define predictor
predictor = TabularPredictor(
    label='target',
    problem_type='binary',  # or 'multiclass', 'regression'
    eval_metric='roc_auc',
    path='./autogluon_models'
)

# Train with auto configuration
predictor.fit(
    train_data,
    time_limit=3600,  # 1 hour
    presets='best_quality',  # or 'medium_quality', 'optimize_for_deployment'
    num_bag_folds=5,  # K-fold bagging for ensembles
    num_bag_sets=1,
    num_stack_levels=1  # Stacking layers
)

# Evaluate
test_score = predictor.evaluate(test_data)
print(f"Test score: {test_score}")

# Get leaderboard
leaderboard = predictor.leaderboard(test_data, silent=True)
print(leaderboard)

# Predictions
predictions = predictor.predict(test_data.drop('target', axis=1))
probabilities = predictor.predict_proba(test_data.drop('target', axis=1))

# Feature importance
feature_importance = predictor.feature_importance(test_data)
print(feature_importance)

# Model info
model_info = predictor.info()
print(f"Best model: {model_info['best_model']}")
```

**Advanced AutoGluon:**

```python
from autogluon.tabular import TabularPredictor

# Custom hyperparameter search space
hyperparameters = {
    'XGB': {},  # Use default XGBoost params
    'NN_TORCH': {
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'activation': 'relu',
        'dropout_prob': [0.1, 0.3, 0.5],  # Search over dropout
    },
    'GBM': [  # LightGBM
        {'num_boost_round': 100, 'num_leaves': 31},
        {'num_boost_round': 200, 'num_leaves': 63}
    ],
    'CAT': {},  # CatBoost
    'RF': [
        {'n_estimators': 100},
        {'n_estimators': 200, 'max_depth': 15}
    ]
}

predictor = TabularPredictor(label='target', path='./models')

predictor.fit(
    train_data,
    hyperparameters=hyperparameters,
    time_limit=7200,  # 2 hours
    presets='best_quality',
    num_bag_folds=10,
    num_stack_levels=2,  # Two-level stacking

    # Advanced options
    feature_metadata='infer',  # Auto-detect feature types
    auto_stack=True,
    hyperparameter_tune_kwargs={
        'scheduler': 'local',
        'searcher': 'auto',
        'num_trials': 100
    }
)

# Persist for deployment
predictor.save()

# Load for inference
loaded_predictor = TabularPredictor.load('./models')
```

### 2. H2O AutoML (Enterprise-Grade)

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O
h2o.init(max_mem_size='16G', nthreads=-1)

# Load data
train = h2o.import_file('train.csv')
test = h2o.import_file('test.csv')

# Define features and target
x = train.columns
y = 'target'
x.remove(y)

# Configure AutoML
aml = H2OAutoML(
    max_runtime_secs=3600,  # 1 hour
    max_models=50,
    seed=42,

    # Algorithm selection
    exclude_algos=['DeepLearning'],  # Exclude specific algorithms

    # Stopping criteria
    stopping_metric='AUC',
    stopping_tolerance=0.001,
    stopping_rounds=3,

    # Ensemble configuration
    keep_cross_validation_predictions=True,
    keep_cross_validation_models=True,

    # Resource allocation
    nfolds=5,
    balance_classes=True,  # For imbalanced data
    max_after_balance_size=3.0
)

# Train
aml.train(x=x, y=y, training_frame=train)

# Leaderboard
lb = aml.leaderboard
print(lb.head())

# Best model
best_model = aml.leader
print(f"Best model: {best_model.model_id}")
print(f"AUC: {best_model.auc(valid=True):.4f}")

# Predictions
predictions = best_model.predict(test)

# Explain
explain = best_model.explain(test)

# Model interpretability
varimp = best_model.varimp(use_pandas=True)
print(varimp.head(10))

# Save model
model_path = h2o.save_model(model=best_model, path='./h2o_models', force=True)

# Load for inference
loaded_model = h2o.load_model(model_path)

# Shutdown H2O
h2o.shutdown(prompt=False)
```

### 3. Auto-sklearn (Scikit-learn Compatible)

```python
from autosklearn.classification import AutoSklearnClassifier
from autosklearn.metrics import accuracy, roc_auc
import sklearn.datasets
import sklearn.metrics

# Load data
X_train, X_test, y_train, y_test = ...

# Configure AutoML
automl = AutoSklearnClassifier(
    time_left_for_this_task=3600,  # 1 hour
    per_run_time_limit=300,  # 5 minutes per model

    # Resource allocation
    n_jobs=8,
    memory_limit=16384,  # MB

    # Ensemble configuration
    ensemble_size=50,
    ensemble_nbest=200,

    # Resampling
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},

    # Metric
    metric=roc_auc,

    # Advanced
    include={
        'classifier': ['random_forest', 'extra_trees', 'gradient_boosting', 'xgradient_boosting'],
        'feature_preprocessor': ['no_preprocessing', 'pca', 'kernel_pca']
    },

    # Debugging
    tmp_folder='/tmp/autosklearn_tmp',
    delete_tmp_folder_after_terminate=False
)

# Train
automl.fit(X_train, y_train, dataset_name='my_dataset')

# Show models
print(automl.show_models())

# Sprint statistics
print(automl.sprint_statistics())

# Predictions
predictions = automl.predict(X_test)
probabilities = automl.predict_proba(X_test)

# Evaluate
score = sklearn.metrics.roc_auc_score(y_test, probabilities[:, 1])
print(f"Test AUC: {score:.4f}")

# Get best pipeline
print(automl.get_models_with_weights())

# Refit on full data
automl.refit(X_train, y_train)
```

### 4. TPOT (Genetic Programming)

```python
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Configure TPOT
tpot = TPOTClassifier(
    generations=50,  # Number of evolutionary iterations
    population_size=100,  # Number of individuals in population

    # Resource allocation
    n_jobs=-1,
    max_time_mins=60,  # 1 hour
    max_eval_time_mins=5,  # 5 min per pipeline

    # Evolutionary algorithm
    offspring_size=None,  # Default: population_size
    mutation_rate=0.9,
    crossover_rate=0.1,

    # Configuration
    scoring='roc_auc',
    cv=5,

    # Pipeline constraints
    config_dict='TPOT light',  # or 'TPOT MDR', 'TPOT sparse', None (default)

    # Output
    verbosity=2,
    random_state=42,

    # Early stopping
    early_stop=10,  # Stop if no improvement for 10 generations

    # Memory
    memory='auto'
)

# Train
tpot.fit(X_train, y_train)

# Evaluate
score = tpot.score(X_test, y_test)
print(f"Test score: {score:.4f}")

# Export best pipeline
tpot.export('best_pipeline.py')

# The exported pipeline looks like:
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

exported_pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(bootstrap=False, max_features=0.6, min_samples_leaf=5, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
"""
```

---

## Neural Architecture Search

### NAS Fundamentals

Neural Architecture Search automates the design of neural network architectures.

```python
import torch
import torch.nn as nn
from typing import List, Tuple

class SearchSpace:
    """Define NAS search space."""

    # Possible operations
    OPS = {
        'none': lambda C, stride: Zero(stride),
        'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1),
        'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
        'skip_connect': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
        'sep_conv_3x3': lambda C, stride: SepConv(C, C, 3, stride, 1),
        'sep_conv_5x5': lambda C, stride: SepConv(C, C, 5, stride, 2),
        'dil_conv_3x3': lambda C, stride: DilConv(C, C, 3, stride, 2, 2),
        'dil_conv_5x5': lambda C, stride: DilConv(C, C, 5, stride, 4, 2),
    }

    @staticmethod
    def get_op(op_name: str, C: int, stride: int):
        """Get operation by name."""
        return SearchSpace.OPS[op_name](C, stride)

class MixedOp(nn.Module):
    """Mixed operation for differentiable NAS."""

    def __init__(self, C, stride):
        super().__init__()
        self.ops = nn.ModuleList()

        for op_name in SearchSpace.OPS.keys():
            op = SearchSpace.get_op(op_name, C, stride)
            self.ops.append(op)

    def forward(self, x, weights):
        """Weighted sum of all operations."""
        return sum(w * op(x) for w, op in zip(weights, self.ops))

class DARTSCell(nn.Module):
    """DARTS cell with architecture parameters."""

    def __init__(self, n_nodes, C_prev_prev, C_prev, C, reduction):
        super().__init__()
        self.n_nodes = n_nodes

        if reduction:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        # Initialize architecture parameters
        self.n_ops = len(SearchSpace.OPS)
        self.alpha = nn.Parameter(torch.randn(n_nodes, self.n_ops))

        # Mixed operations for each edge
        self.ops = nn.ModuleList()
        for i in range(n_nodes):
            for j in range(i + 2):
                stride = 2 if reduction and j < 2 else 1
                op = MixedOp(C, stride)
                self.ops.append(op)

    def forward(self, s0, s1):
        """Forward pass through cell."""
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self.n_nodes):
            # Get weighted sum of all inputs to this node
            s = sum(
                self.ops[offset + j](h, self.alpha[i])
                for j, h in enumerate(states)
            )
            offset += len(states)
            states.append(s)

        # Concatenate all intermediate nodes
        return torch.cat(states[2:], dim=1)
```

### DARTS (Differentiable Architecture Search)

```python
class DARTSSearcher:
    """DARTS neural architecture search."""

    def __init__(
        self,
        n_classes: int = 10,
        n_layers: int = 8,
        n_nodes: int = 4,
        C: int = 16
    ):
        self.model = Network(n_classes, n_layers, n_nodes, C)
        self.architect = Architect(self.model)

    def search(
        self,
        train_loader,
        valid_loader,
        epochs: int = 50,
        learning_rate: float = 0.025
    ):
        """Search for best architecture."""
        import torch.optim as optim

        # Optimizer for model weights
        optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            weight_decay=3e-4
        )

        # Optimizer for architecture parameters
        arch_optimizer = optim.Adam(
            self.model.arch_parameters(),
            lr=3e-4,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )

        for epoch in range(epochs):
            # Train model weights
            train_loss = self._train_epoch(
                train_loader,
                valid_loader,
                optimizer,
                arch_optimizer
            )

            # Validate
            val_loss, val_acc = self._validate(valid_loader)

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

            # Print current architecture
            if (epoch + 1) % 10 == 0:
                self._print_genotype()

        # Get final architecture
        genotype = self.model.genotype()
        return genotype

    def _train_epoch(self, train_loader, valid_loader, optimizer, arch_optimizer):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0

        train_iter = iter(train_loader)
        valid_iter = iter(valid_loader)

        for step, (input, target) in enumerate(train_iter):
            # Update architecture
            try:
                input_search, target_search = next(valid_iter)
            except StopIteration:
                valid_iter = iter(valid_loader)
                input_search, target_search = next(valid_iter)

            self.architect.step(
                input, target,
                input_search, target_search,
                optimizer
            )

            # Update model weights
            optimizer.zero_grad()
            logits = self.model(input)
            loss = nn.CrossEntropyLoss()(logits, target)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def _validate(self, valid_loader):
        """Validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for input, target in valid_loader:
                logits = self.model(input)
                loss = nn.CrossEntropyLoss()(logits, target)
                total_loss += loss.item()

                _, predicted = logits.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return total_loss / len(valid_loader), correct / total

    def _print_genotype(self):
        """Print current architecture."""
        genotype = self.model.genotype()
        print(f"Current architecture: {genotype}")

# Usage
searcher = DARTSSearcher(
    n_classes=10,
    n_layers=8,
    n_nodes=4,
    C=16
)

# Search
best_architecture = searcher.search(
    train_loader,
    valid_loader,
    epochs=50
)

print(f"Best architecture found: {best_architecture}")
```

### Efficient NAS with Once-for-All

```python
class OFASearcher:
    """Once-for-All Network for efficient NAS."""

    def __init__(self):
        # Use pre-trained OFA network
        from ofa.model_zoo import ofa_net

        self.ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.0', pretrained=True)

    def search_subnet(
        self,
        target_latency_ms: float = 20,
        target_device: str = 'cpu'
    ):
        """Search for subnet meeting latency constraint."""
        from ofa.nas.efficiency_predictor import LatencyPredictor

        # Initialize latency predictor
        latency_predictor = LatencyPredictor(device=target_device)

        # Search configuration
        config = {
            'image_size': [192, 224, 256],
            'ks': [3, 5, 7],  # Kernel sizes
            'e': [3, 4, 6],   # Expansion ratios
            'd': [2, 3, 4]    # Depths
        }

        best_subnet = None
        best_acc = 0

        # Random search (can use evolutionary algorithm)
        for _ in range(1000):
            # Sample random subnet
            subnet_config = self._sample_subnet(config)

            # Predict latency
            latency = latency_predictor.predict_latency(subnet_config)

            if latency <= target_latency_ms:
                # Evaluate accuracy (can use predictor)
                acc = self._evaluate_subnet(subnet_config)

                if acc > best_acc:
                    best_acc = acc
                    best_subnet = subnet_config

        return best_subnet, best_acc

    def _sample_subnet(self, config):
        """Sample random subnet configuration."""
        import random

        return {
            'image_size': random.choice(config['image_size']),
            'ks': [random.choice(config['ks']) for _ in range(20)],
            'e': [random.choice(config['e']) for _ in range(20)],
            'd': [random.choice(config['d']) for _ in range(5)]
        }

    def _evaluate_subnet(self, config):
        """Evaluate subnet accuracy."""
        # Extract subnet from OFA network
        self.ofa_network.set_active_subnet(**config)
        subnet = self.ofa_network.get_active_subnet()

        # Evaluate on validation set
        # ... evaluation code ...

        return accuracy
```

---

## Hyperparameter Optimization

### 1. Optuna (State-of-the-Art HPO)

```python
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import xgboost as xgb
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Objective function for Optuna."""

    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True)
    }

    # Train model
    model = xgb.XGBClassifier(**params, random_state=42, use_label_encoder=False)

    # Cross-validation with pruning
    scores = []
    for fold in range(5):
        # Train-validation split for this fold
        X_train_fold, X_val_fold, y_train_fold, y_val_fold = ...

        model.fit(
            X_train_fold, y_train_fold,
            eval_set=[(X_val_fold, y_val_fold)],
            early_stopping_rounds=10,
            verbose=False
        )

        score = model.score(X_val_fold, y_val_fold)
        scores.append(score)

        # Report intermediate value for pruning
        trial.report(score, fold)

        # Prune trial if not promising
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(scores)

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5)
)

# Optimize
study.optimize(
    objective,
    n_trials=100,
    timeout=3600,  # 1 hour
    n_jobs=-1,  # Parallel trials
    show_progress_bar=True
)

# Best parameters
print("Best parameters:")
print(study.best_params)
print(f"Best value: {study.best_value:.4f}")

# Visualization
import optuna.visualization as vis

# Plot optimization history
fig = vis.plot_optimization_history(study)
fig.show()

# Plot parameter importances
fig = vis.plot_param_importances(study)
fig.show()

# Plot parallel coordinate
fig = vis.plot_parallel_coordinate(study)
fig.show()

# Slice plot
fig = vis.plot_slice(study)
fig.show()
```

### 2. Ray Tune (Distributed HPO)

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
import torch.nn as nn

def train_model(config):
    """Training function for Ray Tune."""

    # Build model with config
    model = nn.Sequential(
        nn.Linear(10, config['hidden_size']),
        nn.ReLU(),
        nn.Dropout(config['dropout']),
        nn.Linear(config['hidden_size'], config['hidden_size']),
        nn.ReLU(),
        nn.Dropout(config['dropout']),
        nn.Linear(config['hidden_size'], 1)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )

    # Training loop
    for epoch in range(100):
        # ... training code ...

        # Report metrics to Ray Tune
        tune.report(loss=val_loss, accuracy=val_acc)

# Define search space
search_space = {
    'lr': tune.loguniform(1e-4, 1e-1),
    'hidden_size': tune.choice([64, 128, 256, 512]),
    'dropout': tune.uniform(0.1, 0.5),
    'weight_decay': tune.loguniform(1e-5, 1e-2),
    'batch_size': tune.choice([32, 64, 128])
}

# Configure scheduler (early stopping)
scheduler = ASHAScheduler(
    max_t=100,  # Maximum epochs
    grace_period=10,  # Minimum epochs before stopping
    reduction_factor=2
)

# Configure search algorithm
search_alg = OptunaSearch(
    metric="accuracy",
    mode="max"
)

# Run hyperparameter search
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=100,  # Number of trials
    scheduler=scheduler,
    search_alg=search_alg,
    resources_per_trial={'cpu': 2, 'gpu': 0.5},  # Resource allocation
    verbose=1
)

# Get best config
best_config = analysis.get_best_config(metric="accuracy", mode="max")
print(f"Best config: {best_config}")

# Get results dataframe
df = analysis.dataframe()
print(df.head())
```

---

## Automated Feature Engineering

### TPOT for Feature Engineering

```python
from tpot import TPOTClassifier

# TPOT automatically tries feature engineering techniques
tpot_config = {
    'sklearn.preprocessing.Binarizer': {
        'threshold': np.arange(0.0, 1.01, 0.05)
    },
    'sklearn.decomposition.PCA': {
        'n_components': range(1, 11),
        'iterated_power': [1, 2, 3]
    },
    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2, 3],
        'include_bias': [False],
        'interaction_only': [False, True]
    },
    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.0, 1.01, 0.05)
    }
}

tpot = TPOTClassifier(
    config_dict=tpot_config,
    generations=50,
    population_size=50,
    verbosity=2
)

tpot.fit(X_train, y_train)
```

### Featuretools for Automated Feature Engineering

```python
import featuretools as ft
import pandas as pd

# Create EntitySet
es = ft.EntitySet(id='customers')

# Add dataframes
es = es.add_dataframe(
    dataframe_name='customers',
    dataframe=customers_df,
    index='customer_id'
)

es = es.add_dataframe(
    dataframe_name='transactions',
    dataframe=transactions_df,
    index='transaction_id',
    time_index='transaction_time'
)

# Define relationships
es = es.add_relationship('customers', 'customer_id', 'transactions', 'customer_id')

# Deep feature synthesis
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    agg_primitives=['sum', 'mean', 'max', 'min', 'std', 'count'],
    trans_primitives=['day', 'month', 'year', 'weekday'],
    max_depth=2,
    verbose=True
)

print(f"Generated {len(feature_defs)} features")
print(feature_matrix.head())

# Use with AutoML
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='target')
predictor.fit(feature_matrix)
```

---

## Production Deployment

### Deploying AutoML Models

```python
class AutoMLModelServer:
    """Production server for AutoML models."""

    def __init__(self, model_path: str):
        # Load AutoGluon model
        from autogluon.tabular import TabularPredictor
        self.predictor = TabularPredictor.load(model_path)

    def predict(self, features: dict) -> dict:
        """Make prediction."""
        import pandas as pd

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Predict
        prediction = self.predictor.predict(df)[0]
        probability = self.predictor.predict_proba(df).iloc[0].to_dict()

        return {
            'prediction': int(prediction),
            'probabilities': probability,
            'model_version': self.predictor.version
        }

# FastAPI deployment
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load model at startup
model_server = AutoMLModelServer('/models/autogluon_model')

class PredictionRequest(BaseModel):
    features: dict

class PredictionResponse(BaseModel):
    prediction: int
    probabilities: dict
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Prediction endpoint."""
    try:
        result = model_server.predict(request.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy"}
```

---

## Best Practices

### 1. AutoML Selection Guide

```python
class AutoMLSelector:
    """Help select the right AutoML tool."""

    @staticmethod
    def recommend(
        data_type: str,
        dataset_size: str,
        time_budget: str,
        deployment_target: str
    ) -> str:
        """
        Recommend AutoML tool based on requirements.

        Args:
            data_type: 'tabular', 'image', 'text', 'multimodal'
            dataset_size: 'small' (<10K), 'medium' (10K-1M), 'large' (>1M)
            time_budget: 'fast' (<1h), 'medium' (1-24h), 'slow' (>24h)
            deployment_target: 'cloud', 'edge', 'mobile'
        """

        if data_type == 'tabular':
            if time_budget == 'fast':
                return "AutoGluon with presets='medium_quality'"
            elif dataset_size == 'large':
                return "H2O AutoML (distributed)"
            else:
                return "AutoGluon with presets='best_quality'"

        elif data_type in ['image', 'text', 'multimodal']:
            return "AutoGluon (supports multimodal)"

        elif deployment_target == 'edge':
            return "TPOT (generates sklearn pipelines, easy to deploy)"

        return "AutoGluon (most versatile)"

# Usage
recommendation = AutoMLSelector.recommend(
    data_type='tabular',
    dataset_size='medium',
    time_budget='medium',
    deployment_target='cloud'
)
print(f"Recommended: {recommendation}")
```

### 2. Monitoring AutoML in Production

```python
from prometheus_client import Counter, Histogram

# Metrics
AUTOML_PREDICTIONS = Counter(
    'automl_predictions_total',
    'Total AutoML predictions',
    ['model_type', 'status']
)

AUTOML_LATENCY = Histogram(
    'automl_prediction_latency_seconds',
    'AutoML prediction latency',
    ['model_type']
)

def predict_with_monitoring(model, features):
    """Prediction with monitoring."""
    import time

    start = time.time()
    model_type = type(model).__name__

    try:
        prediction = model.predict(features)

        # Update metrics
        AUTOML_PREDICTIONS.labels(
            model_type=model_type,
            status='success'
        ).inc()

        AUTOML_LATENCY.labels(model_type=model_type).observe(
            time.time() - start
        )

        return prediction

    except Exception as e:
        AUTOML_PREDICTIONS.labels(
            model_type=model_type,
            status='error'
        ).inc()
        raise e
```

---

## Summary

AutoML and NAS in 2025:

1. **Choose the Right Tool:**
   - **AutoGluon:** Best overall, especially for tabular and multimodal
   - **H2O AutoML:** Enterprise-grade, distributed computing
   - **Auto-sklearn:** Scikit-learn compatible, research-focused
   - **TPOT:** Genetic programming, interpretable pipelines

2. **HPO Tools:**
   - **Optuna:** State-of-the-art, flexible, visualization
   - **Ray Tune:** Distributed, scalable, deep learning focus

3. **NAS:**
   - **DARTS:** Differentiable, efficient
   - **Once-for-All:** Pre-trained networks, fast subnet search

4. **Production:**
   - Monitor AutoML models same as manual models
   - Export interpretable pipelines when possible
   - Version AutoML configurations and results
   - A/B test AutoML vs manual models

**Next Steps:**
- Implement experiment tracking (Section 40)
- Deploy models (Section 36)
- Monitor in production (Section 37)
