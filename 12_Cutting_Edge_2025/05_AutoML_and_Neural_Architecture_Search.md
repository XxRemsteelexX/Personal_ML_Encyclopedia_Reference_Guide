# AutoML & Neural Architecture Search - Automated ML Pipeline Design

## Overview

**AutoML automates the entire ML pipeline:** From data preprocessing to model selection, hyperparameter tuning, and architecture design.

**2024-2025 State:**
- AutoGluon with foundation models (NeurIPS 2024)
- Neural Architecture Search (NAS) with efficient search
- Hyperparameter optimization at scale
- Zero-code ML with LLM agents

---

## AutoML Frameworks

### AutoGluon (AWS - 2024 SOTA)

**NeurIPS 2024 Update:** Foundation models for tabular data (TabPFN-Mix) and time series (Chronos).

```python
from autogluon.tabular import TabularPredictor
from autogluon.timeseries import TimeSeriesPredictor
import pandas as pd

# TABULAR DATA
# -----------

# Load data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Create predictor (ONE LINE!)
predictor = TabularPredictor(
    label='target',
    eval_metric='roc_auc',
    problem_type='binary'  # auto-detects if not specified
)

# Fit (automatic feature engineering, model selection, ensembling)
predictor.fit(
    train_data=train_data,
    time_limit=3600,  # 1 hour
    presets='best_quality'  # or 'medium_quality', 'optimize_for_deployment'
)

# Predictions with uncertainty
predictions = predictor.predict(test_data)
pred_probs = predictor.predict_proba(test_data)

# Leaderboard
leaderboard = predictor.leaderboard(test_data)
print(leaderboard)

"""
                        model   score_val  pred_time_val   fit_time
0   WeightedEnsemble_L2      0.9234         0.52          156.3
1   LightGBM_BAG_L1         0.9201         0.18           45.2
2   CatBoost_BAG_L1         0.9187         0.31           67.8
3   NeuralNetFastAI_BAG     0.9156         0.89          124.5
"""

# Feature importance
importance = predictor.feature_importance(test_data)


# TIME SERIES
# -----------

ts_predictor = TimeSeriesPredictor(
    prediction_length=24,  # Forecast horizon
    target='sales',
    eval_metric='MAPE'
)

ts_predictor.fit(
    train_data=ts_train,
    time_limit=1800,
    presets='best_quality'
)

forecast = ts_predictor.predict(ts_test)


# MULTIMODAL (Text + Tabular + Images)
# -----------

from autogluon.multimodal import MultiModalPredictor

mm_predictor = MultiModalPredictor(label='label')

# Data with text, images, and tabular features
train_data = pd.DataFrame({
    'text': ['Product is great!', 'Terrible quality'],
    'image': ['img1.jpg', 'img2.jpg'],
    'price': [29.99, 49.99],
    'label': [1, 0]
})

mm_predictor.fit(train_data, time_limit=3600)
predictions = mm_predictor.predict(test_data)


# ADVANCED: Custom models and ensembles
# -----------

predictor.fit(
    train_data=train_data,
    hyperparameters={
        'GBM': [
            {'num_boost_round': 100, 'learning_rate': 0.03},
            {'num_boost_round': 200, 'learning_rate': 0.01}
        ],
        'NN_TORCH': {},
        'CAT': {}
    },
    num_stack_levels=2,  # Stacking depth
    num_bag_folds=8      # Bagging folds
)
```

**Key Features:**
- **TabPFN-Mix:** Foundation model for tabular data (zero-shot learning)
- **Chronos:** Foundation model for time series
- **Multi-layer stacking:** Automatic ensemble stacking
- **4-line ML:** Load --> Fit --> Predict --> Done

**Performance (2025 Benchmark):** AutoGluon scored 0.61+ across 21 datasets, outperforming most frameworks.

---

### H2O AutoML

**Java-based, enterprise-grade:** Stacked ensembles with fast random search.

```python
import h2o
from h2o.automl import H2OAutoML

# Initialize H2O cluster
h2o.init()

# Load data
df = h2o.import_file('train.csv')
train, valid, test = df.split_frame([0.7, 0.15])

# Identify predictors and response
x = df.columns
y = 'target'
x.remove(y)

# AutoML
aml = H2OAutoML(
    max_models=20,
    max_runtime_secs=3600,
    seed=1,

    # Leaderboard settings
    sort_metric='AUC',

    # Advanced
    nfolds=5,
    keep_cross_validation_predictions=True,
    keep_cross_validation_models=True
)

aml.train(x=x, y=y, training_frame=train, validation_frame=valid)

# Leaderboard
lb = aml.leaderboard
print(lb)

# Best model
best_model = aml.leader
print(best_model)

# Predictions
preds = best_model.predict(test)

# Explain
best_model.explain(test)

# Export
h2o.save_model(best_model, path='./h2o_models/')
```

**Unique Features:**
- **Stacked ensembles:** Automatically creates meta-learners
- **Explainability:** Built-in model explanations (SHAP, PDP)
- **Scalability:** Distributed training on clusters
- **Production:** Export to MOJO/POJO for Java deployment

---

### Auto-sklearn (Academic Standard)

**Bayesian optimization + meta-learning:** Learns from past performance on similar datasets.

```python
import autosklearn.classification
import autosklearn.regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Classification
automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=3600,
    per_run_time_limit=300,

    # Ensemble settings
    ensemble_size=50,
    ensemble_nbest=200,

    # Resampling
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 5},

    # Resources
    n_jobs=-1,
    memory_limit=8192  # MB
)

automl.fit(X_train, y_train)

# Predictions
y_pred = automl.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Show models and hyperparameters
print(automl.show_models())

# Statistics
print(automl.sprint_statistics())

# Refit on full data
automl.refit(X_train, y_train)
```

**Advantages:**
- **Meta-learning:** Uses performance on similar datasets to warm-start optimization
- **Structured search space:** 110 hyperparameters across 15 classifiers
- **Ensemble selection:** Pruned ensemble for efficiency

---

### PyCaret (Rapid Prototyping)

**Low-code ML:** Entire pipeline in <10 lines.

```python
from pycaret.classification import *

# Setup (handles missing values, encoding, scaling, feature selection)
clf = setup(
    data=train_data,
    target='target',
    session_id=123,

    # Auto feature engineering
    feature_selection=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,

    # Preprocessing
    normalize=True,
    transformation=True,

    # Sampling
    fix_imbalance=True
)

# Compare all models
best_models = compare_models(n_select=3)  # Top 3

# Tune top model
tuned_model = tune_model(best_models[0], optimize='AUC')

# Ensemble
bagged_model = ensemble_model(tuned_model, method='Bagging')
boosted_model = ensemble_model(tuned_model, method='Boosting')

# Blend top 3
blender = blend_models(best_models)

# Stack
stacker = stack_models(best_models)

# Final predictions
predictions = predict_model(stacker, data=test_data)

# Deploy
save_model(stacker, 'final_model')
```

**Best For:**
- Rapid prototyping
- Business analysts / citizen data scientists
- Kaggle competitions (quick baseline)

---

## Neural Architecture Search (NAS)

### DARTS (Differentiable Architecture Search)

**Breakthrough:** Continuous relaxation of discrete architecture search --> gradient descent.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """Mixture of operations (weighted by architecture parameters)"""

    def __init__(self, C, stride):
        super().__init__()
        self._ops = nn.ModuleList()

        # Candidate operations
        for primitive in PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        # Weighted sum of all operations
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):
    """DARTS cell with architecture parameters"""

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C):
        super().__init__()

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._steps = steps
        self._multiplier = multiplier

        self._ops = nn.ModuleList()
        for i in range(self._steps):
            for j in range(2 + i):
                stride = 1
                op = MixedOp(C, stride)
                self._ops.append(op)

    def forward(self, s0, s1, weights):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0

        for i in range(self._steps):
            # Mix all previous states
            s = sum(self._ops[offset + j](h, weights[offset + j])
                   for j, h in enumerate(states))
            offset += len(states)
            states.append(s)

        # Concatenate intermediate states
        return torch.cat(states[-self._multiplier:], dim=1)


class DARTSNetwork(nn.Module):
    """Full DARTS searchable network"""

    def __init__(self, C, num_classes, layers):
        super().__init__()

        self._C = C
        self._num_classes = num_classes
        self._layers = layers

        # Architecture parameters (learnable!)
        k = sum(1 for i in range(steps) for j in range(2 + i))
        num_ops = len(PRIMITIVES)

        self.alphas_normal = nn.Parameter(torch.randn(k, num_ops))
        self.alphas_reduce = nn.Parameter(torch.randn(k, num_ops))

        # Network cells
        self.cells = nn.ModuleList()
        for i in range(layers):
            cell = Cell(steps=4, multiplier=4, C_prev_prev=C, C_prev=C, C=C)
            self.cells.append(cell)

    def forward(self, x):
        # Softmax over operations
        weights_normal = F.softmax(self.alphas_normal, dim=-1)
        weights_reduce = F.softmax(self.alphas_reduce, dim=-1)

        s0 = s1 = x
        for i, cell in enumerate(self.cells):
            weights = weights_reduce if reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.global_pooling(s1)
        logits = self.classifier(out)

        return logits

    def arch_parameters(self):
        return [self.alphas_normal, self.alphas_reduce]

    def genotype(self):
        """Extract final architecture"""

        def _parse(weights):
            gene = []
            n = 2
            start = 0

            for i in range(steps):
                end = start + n
                W = weights[start:end].copy()

                # Select top-2 operations
                edges = sorted(range(i + 2),
                             key=lambda x: -max(W[x][k] for k in range(len(W[x]))))[:2]

                for j in edges:
                    k_best = max(range(len(W[j])), key=lambda k: W[j][k])
                    gene.append((PRIMITIVES[k_best], j))

                start = end
                n += 1

            return gene

        gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
        gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

        return Genotype(normal=gene_normal, reduce=gene_reduce)


# Bi-level optimization
def train_darts(model, train_data, valid_data, epochs=50):
    """Alternate between weights and architecture"""

    architect = Architect(model, args)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.025,
        momentum=0.9,
        weight_decay=3e-4
    )

    for epoch in range(epochs):
        # 1. Update architecture parameters
        for step, (input_train, target_train) in enumerate(train_data):
            input_valid, target_valid = next(iter(valid_data))

            # Update alpha (architecture parameters)
            architect.step(input_train, target_train,
                          input_valid, target_valid,
                          optimizer)

        # 2. Update network weights
        for step, (input, target) in enumerate(train_data):
            logits = model(input)
            loss = criterion(logits, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print discovered architecture
        genotype = model.genotype()
        print(f'Epoch {epoch}: {genotype}')

    return model.genotype()
```

**Results:** Discovers architectures matching hand-designed models in <1 GPU-day (vs. thousands of GPU-days for RL-based NAS).

---

### ENAS (Efficient NAS)

**Parameter sharing:** All child models share weights --> 1000x speedup.

```python
class ENASController(nn.Module):
    """RNN controller that generates architectures"""

    def __init__(self, num_layers=12, num_branches=6):
        super().__init__()

        self.num_layers = num_layers
        self.num_branches = num_branches

        # LSTM controller
        self.lstm = nn.LSTMCell(hidden_size=100, input_size=100)

        # Embedding for previous decisions
        self.g_emb = nn.Embedding(1, 100)

        # Outputs
        self.w_soft = nn.Linear(100, num_branches)  # Operation
        self.w_attn = nn.Linear(100, 100)          # Connection

    def sample(self):
        """Sample an architecture"""

        inputs = self.g_emb.weight
        h, c = torch.zeros(1, 100), torch.zeros(1, 100)

        entropies = []
        log_probs = []
        sampled_arc = []

        for layer in range(self.num_layers):
            # Predict operation
            h, c = self.lstm(inputs, (h, c))
            logits = self.w_soft(h)

            # Sample
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1)
            sampled_arc.append(action.item())

            # Track for REINFORCE
            log_prob = F.log_softmax(logits, dim=-1)
            entropies.append(-(log_prob * probs).sum())
            log_probs.append(log_prob.gather(1, action))

            # Next input
            inputs = action

        return sampled_arc, log_probs, entropies


def train_enas(controller, shared_model, train_data, valid_data):
    """Train controller with REINFORCE"""

    controller_optimizer = torch.optim.Adam(controller.parameters(), lr=3.5e-4)

    for epoch in range(epochs):
        # 1. Sample architectures and train shared weights
        for step in range(100):
            arc, _, _ = controller.sample()

            # Train shared model with this architecture
            loss = train_shared_model(shared_model, arc, train_data)

        # 2. Update controller
        arc, log_probs, entropies = controller.sample()

        # Evaluate architecture
        with torch.no_grad():
            reward = evaluate_architecture(shared_model, arc, valid_data)

        # REINFORCE
        policy_loss = -sum(lp * reward for lp in log_probs)
        entropy_bonus = -sum(entropies)

        total_loss = policy_loss + 0.001 * entropy_bonus

        controller_optimizer.zero_grad()
        total_loss.backward()
        controller_optimizer.step()

        print(f"Epoch {epoch}: Reward={reward:.4f}, Arch={arc}")
```

**Speed:** 1 GPU-day vs. 2000+ GPU-days for NASNet.

---

## Hyperparameter Optimization

### Optuna (Tree-Structured Parzen Estimator)

**Bayesian optimization:** Sample promising hyperparameters based on past trials.

```python
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    # Sample hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
    }

    # Train model
    clf = RandomForestClassifier(**params, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    return accuracy

# Optimize
study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(),
    pruner=optuna.pruners.MedianPruner()
)

study.optimize(objective, n_trials=100)

# Best params
print(f"Best accuracy: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualize
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

### Ray Tune (Distributed HPO)

**Scalable across clusters:** Population-based training, HyperBand/ASHA.

```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

def train_model(config):
    """Training function with hyperparameters from config"""

    model = NeuralNetwork(
        hidden_size=config['hidden_size'],
        lr=config['lr'],
        dropout=config['dropout']
    )

    for epoch in range(100):
        train_loss = train_epoch(model)
        val_loss = validate(model)

        # Report to Ray Tune
        tune.report(loss=val_loss, accuracy=val_acc)

# Search space
search_space = {
    'hidden_size': tune.choice([64, 128, 256, 512]),
    'lr': tune.loguniform(1e-4, 1e-1),
    'dropout': tune.uniform(0.1, 0.5),
    'batch_size': tune.choice([32, 64, 128])
}

# Scheduler (early stopping)
scheduler = ASHAScheduler(
    max_t=100,
    grace_period=10,
    reduction_factor=2
)

# Search algorithm (Optuna backend)
search_alg = OptunaSearch()

# Run
analysis = tune.run(
    train_model,
    config=search_space,
    num_samples=100,
    scheduler=scheduler,
    search_alg=search_alg,
    resources_per_trial={'cpu': 2, 'gpu': 0.5}
)

# Best config
best_config = analysis.get_best_config(metric='loss', mode='min')
print(f"Best config: {best_config}")
```

**Integration:** Ray Tune + Optuna = distributed Bayesian optimization.

---

## Production Pipeline

```python
class AutoMLPipeline:
    """End-to-end AutoML pipeline"""

    def __init__(self, task='classification'):
        self.task = task
        self.predictor = None

    def auto_train(self, train_data, target, time_limit=3600):
        """Automatic training with AutoGluon"""

        from autogluon.tabular import TabularPredictor

        self.predictor = TabularPredictor(
            label=target,
            problem_type=self.task
        )

        self.predictor.fit(
            train_data=train_data,
            time_limit=time_limit,
            presets='best_quality'
        )

        # Model summary
        leaderboard = self.predictor.leaderboard()
        print(leaderboard)

        return self.predictor

    def deploy(self, model_name='automl_model'):
        """Save for production"""

        self.predictor.save(model_name)

        # Inference code
        inference_code = f"""
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load('{model_name}')

def predict(data):
    return predictor.predict(data)
        """

        with open('inference.py', 'w') as f:
            f.write(inference_code)

        print(f"Model saved to {model_name}/")

    def explain(self, test_data):
        """Explainability"""

        importance = self.predictor.feature_importance(test_data)
        print(importance)

        return importance
```

---

## Best Practices

1. **Start with AutoGluon** - Best performance with minimal code
2. **Use Ray Tune for distributed HPO** - Scale across clusters
3. **NAS for novel architectures** - DARTS for efficiency
4. **Monitor compute budget** - Set time_limit to avoid overspending
5. **Ensemble everything** - Stacking almost always improves performance
6. **Explain predictions** - Use SHAP/LIME for interpretability

**Key Takeaway:** AutoML democratizes ML, but understanding fundamentals is still crucial for debugging and custom solutions.

**Performance (2025):**
- AutoGluon: 0.61+ across benchmarks
- DARTS: Discovers competitive architectures in <1 GPU-day
- Ray Tune + Optuna: 10-100x speedup with distribution
