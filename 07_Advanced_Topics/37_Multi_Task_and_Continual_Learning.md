# 37. Multi-Task Learning and Continual Learning

## Table of Contents

- [37.1 MTL Fundamentals](#371-mtl-fundamentals)
- [37.2 MTL Architectures](#372-mtl-architectures)
- [37.3 Loss Balancing](#373-loss-balancing)
- [37.4 MTL Applications](#374-mtl-applications)
- [37.5 Practical Tips for MTL](#375-practical-tips-for-mtl)
- [37.6 The Catastrophic Forgetting Problem](#376-the-catastrophic-forgetting-problem)
- [37.7 Regularization-Based Methods](#377-regularization-based-methods)
- [37.8 Replay-Based Methods](#378-replay-based-methods)
- [37.9 Architecture-Based Methods](#379-architecture-based-methods)
- [37.10 Continual Learning Benchmarks](#3710-continual-learning-benchmarks)
- [37.11 Curriculum Learning](#3711-curriculum-learning)
- [37.12 Meta-Continual Learning](#3712-meta-continual-learning)
- [37.13 Comparison and Selection Guide](#3713-comparison-and-selection-guide)
- [See Also](#see-also)
- [Resources](#resources)

---

## 37.1 MTL Fundamentals

**Multi-Task Learning (MTL)** is a learning paradigm in which a model is trained on multiple
related tasks simultaneously, leveraging shared representations to improve generalization
across all tasks. Rather than training isolated models for each task, MTL exploits the
inductive bias that related tasks share common structure.

### Definition and Motivation

MTL operates on three core principles:

- **Shared Representations**: By forcing a single model to solve multiple tasks, the learned
  features must capture the underlying structure common to all tasks. This produces more
  general and robust representations than single-task training.

- **Implicit Regularization**: Additional tasks act as a regularizer. The model cannot
  overfit to noise specific to any single task because it must simultaneously satisfy
  constraints from all tasks. This effect is strongest when tasks are genuinely related.

- **Implicit Data Augmentation**: Each task provides its own training signal. Even if
  Task A has limited data, Task B may provide complementary supervision that improves
  the shared representation. This is particularly valuable in low-data regimes.

### Inductive Bias from Related Tasks

The key insight behind MTL is that related tasks share a common hypothesis space. When
Task A and Task B share underlying causal factors, jointly learning both tasks constrains
the model to prefer hypotheses that explain both tasks well. Formally, if the true data
generating process for both tasks depends on a shared latent variable z, then MTL
encourages the model to learn representations that capture z.

The **task relatedness assumption** states that tasks sharing statistical structure will
benefit from joint training. This can manifest as:

- Shared input features (same images, different labels)
- Shared causal mechanisms (same underlying physics, different measurements)
- Hierarchical structure (coarse and fine-grained classification)
- Complementary supervision (segmentation helps detection, detection helps segmentation)

### When MTL Helps vs Hurts

**MTL helps when**:
- Tasks are genuinely related and share underlying structure
- Individual tasks have limited training data
- Tasks provide complementary gradients to the shared representation
- The shared representation capacity is sufficient for all tasks

**MTL hurts when**:
- Tasks are unrelated or conflicting (negative transfer)
- One dominant task overwhelms others during training
- Task difficulty is highly imbalanced
- The shared architecture bottleneck constrains per-task performance

### Positive Transfer vs Negative Transfer

**Positive transfer** occurs when training on Task B improves performance on Task A
compared to training on Task A alone. This is the desired outcome of MTL.

**Negative transfer** occurs when joint training degrades performance on one or more
tasks. Common causes include:

- **Task interference**: Gradients from different tasks push shared parameters in
  conflicting directions
- **Capacity limitations**: The shared network lacks capacity to represent all tasks
- **Optimization conflicts**: Different tasks converge at different rates, causing
  some tasks to dominate early training
- **Label space conflicts**: Tasks with contradictory labeling schemes

Detecting negative transfer requires comparing MTL performance against single-task
baselines for each task individually.

---

## 37.2 MTL Architectures

### Hard Parameter Sharing

**Hard parameter sharing** is the most common MTL architecture. A shared backbone
(encoder) processes inputs, and task-specific heads branch off to produce per-task
outputs. All tasks share the same lower-level features while learning task-specific
transformations in their heads.

Advantages:
- Simple to implement and scale
- Strong regularization effect from sharing most parameters
- Reduced total parameter count compared to separate models
- Efficient inference when tasks share the same input

Disadvantages:
- All tasks forced through the same bottleneck representation
- Negative transfer if tasks require fundamentally different features
- Limited flexibility in choosing what to share

### Soft Parameter Sharing

In **soft parameter sharing**, each task has its own model, but a regularization term
encourages the parameters of different task models to remain similar. Common approaches
include L2 distance between parameters, trace norm regularization, or constraining
models to share a low-rank structure.

The loss function becomes:

```
L_total = sum_t L_t(theta_t) + lambda * sum_{t,t'} D(theta_t, theta_t')
```

where D is a distance measure between parameter sets.

### Cross-Stitch Networks

**Cross-stitch networks** (Misra et al., 2016) learn linear combinations of features
from different task-specific networks at each layer. A cross-stitch unit at layer l
computes:

```
x_A^{l+1} = alpha_AA * x_A^l + alpha_AB * x_B^l
x_B^{l+1} = alpha_BA * x_A^l + alpha_BB * x_B^l
```

The alpha values are learned end-to-end, allowing the network to discover optimal
sharing patterns automatically.

### Multi-gate Mixture-of-Experts (MMoE)

**MMoE** (Ma et al., 2018) replaces the shared backbone with a set of expert
sub-networks. Each task has its own gating network that produces a softmax distribution
over experts. The task-specific input to each tower is a weighted combination of
expert outputs:

```
f_t(x) = sum_i g_t_i(x) * E_i(x)
```

where E_i is expert i and g_t is the gating network for task t. This allows each
task to selectively use different experts, naturally handling task conflicts.

### Progressive Layered Extraction (PLE)

**PLE** (Tang et al., 2020) extends MMoE by introducing both shared experts and
task-specific experts at each extraction layer. This addresses the seesaw phenomenon
in MMoE where improving one task can degrade another. PLE stacks multiple extraction
layers, each containing:

- Shared expert modules accessible to all tasks
- Task-specific expert modules private to each task
- Task-specific gating networks that combine both shared and private experts

### Code: Hard Parameter Sharing in PyTorch

```python
import torch
import torch.nn as nn
from typing import Dict, List, Tuple


class SharedBackbone(nn.Module):
    """Shared feature extractor for multi-task learning."""

    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            prev_dim = dim
        self.network = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TaskHead(nn.Module):
    """Task-specific prediction head."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 task_type: str = "classification"):
        super().__init__()
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class HardParameterSharingMTL(nn.Module):
    """
    Multi-task model with hard parameter sharing.
    A shared backbone feeds into task-specific heads.
    """

    def __init__(
        self,
        input_dim: int,
        shared_dims: List[int],
        task_configs: Dict[str, Dict],
    ):
        """
        Args:
            input_dim: Dimension of input features.
            shared_dims: Hidden dimensions for the shared backbone.
            task_configs: Dict mapping task_name -> {
                'hidden_dim': int,
                'output_dim': int,
                'task_type': 'classification' or 'regression',
                'loss_fn': nn.Module (optional)
            }
        """
        super().__init__()
        self.backbone = SharedBackbone(input_dim, shared_dims)
        self.task_heads = nn.ModuleDict()
        self.loss_fns = {}

        for name, config in task_configs.items():
            self.task_heads[name] = TaskHead(
                input_dim=self.backbone.output_dim,
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                task_type=config.get("task_type", "classification"),
            )
            if "loss_fn" in config:
                self.loss_fns[name] = config["loss_fn"]
            elif config.get("task_type") == "regression":
                self.loss_fns[name] = nn.MSELoss()
            else:
                self.loss_fns[name] = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        shared_features = self.backbone(x)
        outputs = {}
        for name, head in self.task_heads.items():
            outputs[name] = head(shared_features)
        return outputs

    def compute_losses(
        self, outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        for name in outputs:
            if name in targets:
                losses[name] = self.loss_fns[name](outputs[name], targets[name])
        return losses


# Usage example
if __name__ == "__main__":
    model = HardParameterSharingMTL(
        input_dim=128,
        shared_dims=[256, 128, 64],
        task_configs={
            "sentiment": {"hidden_dim": 32, "output_dim": 3, "task_type": "classification"},
            "ner": {"hidden_dim": 32, "output_dim": 9, "task_type": "classification"},
            "rating": {"hidden_dim": 32, "output_dim": 1, "task_type": "regression"},
        },
    )

    x = torch.randn(16, 128)
    outputs = model(x)
    targets = {
        "sentiment": torch.randint(0, 3, (16,)),
        "ner": torch.randint(0, 9, (16,)),
        "rating": torch.randn(16, 1),
    }
    losses = model.compute_losses(outputs, targets)
    total_loss = sum(losses.values())
    total_loss.backward()
    print(f"Task losses: { {k: v.item() for k, v in losses.items()} }")
```

### Code: MMoE Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class Expert(nn.Module):
    """Single expert network."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GatingNetwork(nn.Module):
    """Task-specific gating network that weights expert outputs."""

    def __init__(self, input_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.gate(x), dim=-1)


class MMoE(nn.Module):
    """
    Multi-gate Mixture-of-Experts for multi-task learning.

    Each task has its own gating network that learns which experts
    to attend to. This allows tasks to share experts when beneficial
    and use different experts when tasks conflict.
    """

    def __init__(
        self,
        input_dim: int,
        expert_hidden_dim: int,
        expert_output_dim: int,
        num_experts: int,
        task_configs: Dict[str, Dict],
    ):
        super().__init__()
        self.num_experts = num_experts

        # Shared expert pool
        self.experts = nn.ModuleList([
            Expert(input_dim, expert_output_dim, expert_hidden_dim)
            for _ in range(num_experts)
        ])

        # Per-task gating networks and towers
        self.gates = nn.ModuleDict()
        self.towers = nn.ModuleDict()

        for name, config in task_configs.items():
            self.gates[name] = GatingNetwork(input_dim, num_experts)
            self.towers[name] = nn.Sequential(
                nn.Linear(expert_output_dim, config["hidden_dim"]),
                nn.ReLU(inplace=True),
                nn.Linear(config["hidden_dim"], config["output_dim"]),
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Compute all expert outputs: (num_experts, batch_size, expert_dim)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)

        outputs = {}
        for name in self.towers:
            # Gate weights: (batch_size, num_experts)
            gate_weights = self.gates[name](x)
            # Weighted combination: (batch_size, expert_dim)
            gated_output = torch.bmm(
                gate_weights.unsqueeze(1),  # (batch, 1, num_experts)
                expert_outputs,              # (batch, num_experts, expert_dim)
            ).squeeze(1)
            outputs[name] = self.towers[name](gated_output)

        return outputs


# Usage example
if __name__ == "__main__":
    model = MMoE(
        input_dim=64,
        expert_hidden_dim=128,
        expert_output_dim=64,
        num_experts=8,
        task_configs={
            "ctr": {"hidden_dim": 32, "output_dim": 1},
            "cvr": {"hidden_dim": 32, "output_dim": 1},
            "engagement": {"hidden_dim": 32, "output_dim": 1},
        },
    )

    x = torch.randn(32, 64)
    outputs = model(x)
    for name, out in outputs.items():
        print(f"{name}: {out.shape}")

    # Inspect gating distributions
    with torch.no_grad():
        for name in model.gates:
            weights = model.gates[name](x)
            print(f"{name} gate entropy: {-(weights * weights.log()).sum(-1).mean():.3f}")
```

---

## 37.3 Loss Balancing

When training a multi-task model, the naive approach of summing all task losses with
equal weights often produces suboptimal results. Tasks may have losses on different
scales, converge at different rates, or produce conflicting gradients. **Loss balancing**
strategies address these issues by dynamically adjusting the contribution of each task.

### Uniform Weighting (Naive)

The simplest approach sums task losses with equal weights:

```
L_total = sum_t L_t
```

This fails when:
- Task losses have different magnitudes (one task dominates)
- Tasks converge at different rates
- Tasks conflict in gradient space

### Uncertainty Weighting (Kendall et al., 2018)

**Uncertainty weighting** derives task weights from the homoscedastic (task-dependent)
uncertainty of each task. For regression tasks, the weighted loss becomes:

```
L_total = sum_t (1 / (2 * sigma_t^2)) * L_t + log(sigma_t)
```

where sigma_t is a learnable parameter representing the observation noise for task t.
The log(sigma_t) term acts as a regularizer preventing sigma from growing unbounded
(which would drive the loss coefficient to zero).

For classification tasks, the formulation adjusts to:

```
L_total = sum_t (1 / sigma_t^2) * L_t + log(sigma_t)
```

Key properties:
- Tasks with high uncertainty (large sigma) get lower weight
- The log regularizer prevents collapse
- No hyperparameter tuning needed for task weights
- Weights adapt automatically during training

### GradNorm

**GradNorm** (Chen et al., 2018) directly normalizes gradient magnitudes across tasks.
It maintains learnable loss weights w_t and adjusts them so that each task's gradient
norm relative to the shared parameters matches a target. The target is based on each
task's relative training rate: tasks that train slowly get higher gradient norms.

The GradNorm loss is:

```
L_grad = sum_t |G_t(w) - G_bar * r_t^alpha|
```

where G_t(w) is the gradient norm for task t, G_bar is the average gradient norm,
r_t is the relative inverse training rate for task t, and alpha controls the
strength of the restoring force.

### PCGrad (Project Conflicting Gradients)

**PCGrad** (Yu et al., 2020) detects and resolves gradient conflicts between tasks.
When two tasks have gradients that point in conflicting directions (negative cosine
similarity), PCGrad projects one gradient onto the normal plane of the other, removing
the conflicting component.

Algorithm:
1. Compute per-task gradients g_1, g_2, ..., g_T
2. For each pair (g_i, g_j): if g_i . g_j < 0, replace g_i with
   g_i - (g_i . g_j / ||g_j||^2) * g_j
3. Use the modified gradients for the update

### MGDA (Multiple Gradient Descent Algorithm)

**MGDA** frames multi-task optimization as finding a Pareto-optimal update direction.
It solves a constrained optimization problem to find the minimum-norm point in the
convex hull of task gradients. This guarantees that the resulting update direction
either improves all tasks or is Pareto-stationary.

### Dynamic Weight Average (DWA)

**DWA** (Liu et al., 2019) adjusts task weights based on the rate of change of each
task's loss. Tasks whose loss decreased more slowly in recent epochs get higher weights:

```
w_t(epoch) = T * exp(r_t(epoch-1) / temperature) / sum_t' exp(r_t'(epoch-1) / temperature)
```

where r_t is the ratio L_t(epoch-1) / L_t(epoch-2) and T is the number of tasks.

### Code: Uncertainty Weighting

```python
import torch
import torch.nn as nn
from typing import Dict


class UncertaintyWeightedLoss(nn.Module):
    """
    Automatically learn task weights from homoscedastic uncertainty.

    Each task has a learnable log-variance parameter. Tasks with high
    uncertainty (noisy) are automatically down-weighted.

    Reference: Kendall et al., "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics" (2018)
    """

    def __init__(self, task_names: list, task_types: Dict[str, str] = None):
        """
        Args:
            task_names: List of task identifiers.
            task_types: Dict mapping task_name -> 'classification' or 'regression'.
                        Defaults to 'classification' for all tasks.
        """
        super().__init__()
        self.task_names = task_names
        self.task_types = task_types or {name: "classification" for name in task_names}

        # log(sigma^2) for each task, initialized to 0 (sigma=1)
        self.log_vars = nn.ParameterDict({
            name: nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute the uncertainty-weighted total loss.

        Args:
            losses: Dict mapping task_name -> unweighted scalar loss.

        Returns:
            Weighted total loss.
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

        for name in self.task_names:
            if name not in losses:
                continue

            log_var = self.log_vars[name]

            if self.task_types.get(name) == "regression":
                # (1 / 2*sigma^2) * L + log(sigma) = (1/2) * exp(-log_var) * L + (1/2) * log_var
                precision = torch.exp(-log_var)
                total_loss += 0.5 * (precision * losses[name] + log_var)
            else:
                # (1 / sigma^2) * L + log(sigma) = exp(-log_var) * L + (1/2) * log_var
                precision = torch.exp(-log_var)
                total_loss += precision * losses[name] + 0.5 * log_var

        return total_loss

    def get_weights(self) -> Dict[str, float]:
        """Return current effective weights (1/sigma^2) for each task."""
        weights = {}
        for name in self.task_names:
            log_var = self.log_vars[name].item()
            weights[name] = float(torch.exp(torch.tensor(-log_var)))
        return weights


# Usage example
if __name__ == "__main__":
    task_names = ["detection", "segmentation", "depth"]
    task_types = {
        "detection": "classification",
        "segmentation": "classification",
        "depth": "regression",
    }
    criterion = UncertaintyWeightedLoss(task_names, task_types)

    # Simulated per-task losses
    losses = {
        "detection": torch.tensor(2.5, requires_grad=True),
        "segmentation": torch.tensor(0.8, requires_grad=True),
        "depth": torch.tensor(15.0, requires_grad=True),
    }

    total = criterion(losses)
    total.backward()

    print(f"Total loss: {total.item():.4f}")
    print(f"Task weights: {criterion.get_weights()}")
    for name in task_names:
        print(f"  {name} log_var: {criterion.log_vars[name].item():.4f}")
```

### Code: PCGrad Implementation

```python
import torch
import torch.nn as nn
from typing import List, Dict
import copy
import random


class PCGrad:
    """
    Projecting Conflicting Gradients for multi-task learning.

    When gradients from two tasks conflict (negative cosine similarity),
    project one onto the normal plane of the other to remove the
    conflicting component.

    Reference: Yu et al., "Gradient Surgery for Multi-Task Learning" (2020)
    """

    def __init__(self, optimizer: torch.optim.Optimizer):
        self.optimizer = optimizer

    def step(self, task_losses: List[torch.Tensor], shared_params: List[nn.Parameter]):
        """
        Compute PCGrad update and apply it.

        Args:
            task_losses: List of scalar losses, one per task.
            shared_params: List of shared parameters to update.
        """
        # Compute per-task gradients for shared parameters
        task_grads = []
        for loss in task_losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for p in shared_params:
                if p.grad is not None:
                    grads.append(p.grad.clone().flatten())
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device))
            task_grads.append(torch.cat(grads))

        # Apply PCGrad: project conflicting gradients
        num_tasks = len(task_grads)
        projected_grads = [g.clone() for g in task_grads]

        for i in range(num_tasks):
            # Randomize the order of other tasks
            order = list(range(num_tasks))
            random.shuffle(order)

            for j in order:
                if i == j:
                    continue
                dot = torch.dot(projected_grads[i], task_grads[j])
                if dot < 0:
                    # Project: remove the conflicting component
                    proj = dot / (torch.dot(task_grads[j], task_grads[j]) + 1e-12)
                    projected_grads[i] -= proj * task_grads[j]

        # Average the projected gradients
        final_grad = torch.stack(projected_grads).mean(dim=0)

        # Assign the computed gradient back to parameters
        self.optimizer.zero_grad()
        offset = 0
        for p in shared_params:
            numel = p.numel()
            p.grad = final_grad[offset:offset + numel].view(p.shape).clone()
            offset += numel

        self.optimizer.step()

    def get_conflict_stats(
        self, task_losses: List[torch.Tensor], shared_params: List[nn.Parameter]
    ) -> Dict[str, float]:
        """Compute statistics about gradient conflicts between tasks."""
        task_grads = []
        for loss in task_losses:
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            grads = []
            for p in shared_params:
                if p.grad is not None:
                    grads.append(p.grad.clone().flatten())
                else:
                    grads.append(torch.zeros(p.numel(), device=p.device))
            task_grads.append(torch.cat(grads))

        num_tasks = len(task_grads)
        num_conflicts = 0
        total_pairs = 0
        avg_cosine = 0.0

        for i in range(num_tasks):
            for j in range(i + 1, num_tasks):
                cos = torch.nn.functional.cosine_similarity(
                    task_grads[i].unsqueeze(0), task_grads[j].unsqueeze(0)
                ).item()
                avg_cosine += cos
                total_pairs += 1
                if cos < 0:
                    num_conflicts += 1

        return {
            "conflict_ratio": num_conflicts / max(total_pairs, 1),
            "avg_cosine_similarity": avg_cosine / max(total_pairs, 1),
            "num_conflicts": num_conflicts,
            "total_pairs": total_pairs,
        }


# Usage example
if __name__ == "__main__":
    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
    head_a = nn.Linear(32, 1)
    head_b = nn.Linear(32, 1)

    all_params = list(model.parameters()) + list(head_a.parameters()) + list(head_b.parameters())
    optimizer = torch.optim.Adam(all_params, lr=1e-3)
    pcgrad = PCGrad(optimizer)

    x = torch.randn(16, 32)
    shared_out = model(x)

    loss_a = nn.functional.mse_loss(head_a(shared_out), torch.randn(16, 1))
    loss_b = nn.functional.mse_loss(head_b(shared_out), torch.randn(16, 1))

    shared_params = list(model.parameters())
    stats = pcgrad.get_conflict_stats([loss_a, loss_b], shared_params)
    print(f"Gradient conflict stats: {stats}")

    # Recompute losses (graphs consumed by stat computation)
    shared_out = model(x)
    loss_a = nn.functional.mse_loss(head_a(shared_out), torch.randn(16, 1))
    loss_b = nn.functional.mse_loss(head_b(shared_out), torch.randn(16, 1))
    pcgrad.step([loss_a, loss_b], shared_params)
    print("PCGrad step completed.")
```

---

## 37.4 MTL Applications

### NLP: Joint NER + POS Tagging + Sentiment

In natural language processing, many tasks operate over the same text and share
linguistic features. A shared encoder (e.g., BERT) can feed into task-specific
heads for Named Entity Recognition, Part-of-Speech tagging, and sentiment analysis.

Key observations:
- **Syntactic tasks** (POS tagging, dependency parsing) share lower-level features
- **Semantic tasks** (NER, sentiment, relation extraction) benefit from mid-level
  representations
- Joint training of syntactic and semantic tasks often improves both
- Token-level and sentence-level tasks can share a backbone with different pooling

### CV: Joint Detection + Segmentation + Depth Estimation

Computer vision MTL models process a single image through a shared encoder
(e.g., ResNet, EfficientNet) and produce multiple outputs:

- **Object detection**: bounding boxes and class labels
- **Semantic segmentation**: per-pixel class labels
- **Depth estimation**: per-pixel depth values
- **Surface normal prediction**: per-pixel orientation vectors

Notable systems:
- **MultiNet**: shared encoder for detection, segmentation, and classification
- **PAD-Net**: multi-task with multi-scale feature aggregation
- **MTI-Net**: multi-scale task interaction network

### RecSys: CTR Prediction with Multiple Objectives

Recommendation systems naturally involve multiple correlated objectives:

- **Click-Through Rate (CTR)**: will the user click?
- **Conversion Rate (CVR)**: will the user purchase?
- **Engagement metrics**: dwell time, likes, shares, comments
- **User satisfaction**: long-term retention signals

MMoE and PLE architectures originated in this domain (Google, Tencent). The key
challenge is that different objectives may conflict: maximizing clicks can reduce
long-term engagement. Multi-objective optimization through MTL provides a principled
way to balance these trade-offs.

### Autonomous Driving: Multi-Task Perception

Self-driving systems must simultaneously perform:
- 2D/3D object detection
- Lane detection
- Semantic segmentation
- Depth estimation
- Motion prediction
- Traffic sign recognition

MTL is critical here for both accuracy (shared features improve robustness) and
efficiency (a single forward pass for all tasks reduces latency). Real-time
constraints make the parameter-sharing benefits of MTL essential.

---

## 37.5 Practical Tips for MTL

### Task Grouping Strategies

Not all tasks should be trained together. Strategies for grouping:

- **Correlation-based**: Measure task gradient similarity. Group tasks with
  positive cosine similarity between gradients.
- **Domain-based**: Group tasks operating on the same input domain.
- **Difficulty-based**: Group tasks of similar difficulty to avoid dominance.
- **Automatic grouping**: Use methods like TAG (Task Affinity Grouping) to
  automatically discover optimal groups based on inter-task affinity.

### Gradient Accumulation Per Task

When tasks have different dataset sizes or batch sizes, accumulate gradients
per task before updating:

```python
# Pseudocode for per-task gradient accumulation
for epoch in range(num_epochs):
    for task_batch in task_sampler:
        task_name, (inputs, targets) = task_batch
        outputs = model(inputs)
        loss = loss_fns[task_name](outputs[task_name], targets)
        (loss / accumulation_steps[task_name]).backward()

        if step % accumulation_steps[task_name] == 0:
            optimizer.step()
            optimizer.zero_grad()
```

### Task Sampling Strategies

When tasks have different dataset sizes, how to sample batches matters:

- **Proportional sampling**: Sample from each task proportional to its dataset size.
  Large tasks dominate.
- **Equal sampling**: Sample equally from all tasks. Small tasks get upsampled.
- **Temperature-based sampling**: p(task_t) proportional to N_t^(1/T). Temperature T=1
  gives proportional, T=infinity gives equal. T in [2,5] often works well.
- **Annealed sampling**: Start with equal sampling, gradually shift to proportional.

### When to Share and When to Separate

Guidelines for architecture decisions:

- **Share early layers**: Low-level features (edges, textures in CV; morphology in NLP)
  are almost always shareable.
- **Separate later layers**: High-level, task-specific features should be private.
- **Monitor gradient conflict**: If gradient cosine similarity between tasks drops
  below -0.1 consistently, consider separating those tasks.
- **Use capacity indicators**: If adding more shared capacity (wider layers) consistently
  helps, the current architecture may be bottlenecking.

### Debugging Negative Transfer

Step-by-step debugging process:

1. **Establish baselines**: Train each task in isolation. Record per-task performance.
2. **Add tasks incrementally**: Start with two tasks, measure transfer. Add tasks one
   by one to identify which cause negative transfer.
3. **Check gradient conflicts**: Use PCGrad's conflict stats to quantify interference.
4. **Vary sharing depth**: Try sharing fewer or more layers.
5. **Scale up capacity**: Negative transfer sometimes indicates insufficient model capacity.
6. **Use soft sharing**: Switch from hard to soft parameter sharing for conflicting tasks.
7. **Apply loss balancing**: Try uncertainty weighting or GradNorm before giving up on MTL.

---

## 37.6 The Catastrophic Forgetting Problem

**Catastrophic forgetting** (also called catastrophic interference) is the phenomenon
where a neural network, upon learning a new task, abruptly loses performance on
previously learned tasks. This is the central challenge in **continual learning**
(also called lifelong learning or sequential learning).

### Definition

When a model trained on Task A is subsequently trained on Task B, the parameter updates
for Task B overwrite the parameters that were important for Task A. Unlike biological
neural networks, which can accumulate knowledge over a lifetime, standard neural networks
are prone to this destructive interference.

### Stability-Plasticity Dilemma

The core tension in continual learning:

- **Stability**: The model should retain knowledge from previous tasks (resist forgetting).
- **Plasticity**: The model should be able to learn new tasks (adapt to new data).

Perfect stability means no learning; perfect plasticity means no retention. All
continual learning methods navigate this trade-off.

### Measuring Forgetting

Key metrics for continual learning evaluation:

- **Backward Transfer (BWT)**: Average change in performance on task t after learning
  subsequent tasks. Negative BWT indicates forgetting.
  ```
  BWT = (1/T-1) * sum_{t=1}^{T-1} (R_{T,t} - R_{t,t})
  ```
  where R_{i,j} is performance on task j after learning task i.

- **Forward Transfer (FWT)**: Average performance improvement on task t from learning
  previous tasks (compared to random initialization).
  ```
  FWT = (1/T-1) * sum_{t=2}^{T} (R_{t-1,t} - R_{0,t})
  ```

- **Average Accuracy**: Mean accuracy across all tasks after learning all tasks.
  ```
  ACC = (1/T) * sum_{t=1}^{T} R_{T,t}
  ```

- **Forgetting Measure**: Maximum performance achieved minus final performance.
  ```
  F_t = max_{i in {1,...,T-1}} (R_{i,t} - R_{T,t})
  ```

### Continual Learning Scenarios

The three standard scenarios, ordered by difficulty:

1. **Task-Incremental Learning (Task-IL)**: The task identity is provided at test time.
   The model only needs to solve the current task. Easiest scenario. Example: multi-head
   classifier where the correct head is selected at inference.

2. **Domain-Incremental Learning (Domain-IL)**: The task identity is not provided, but
   the task structure remains the same (same output space). Example: classifying digits
   where the distribution shifts over time.

3. **Class-Incremental Learning (Class-IL)**: The task identity is not provided, and
   new classes are added over time. The model must distinguish among all classes seen
   so far. Hardest scenario. Example: learning to recognize 10 new animal species
   each month while retaining all previous species.

---

## 37.7 Regularization-Based Methods

Regularization-based methods add a penalty term to the loss function that discourages
large changes to parameters that were important for previous tasks.

### Elastic Weight Consolidation (EWC)

**EWC** (Kirkpatrick et al., 2017) uses the Fisher information matrix to identify
which parameters are important for previously learned tasks. The loss for learning
task B after task A becomes:

```
L_total = L_B(theta) + (lambda/2) * sum_i F_i * (theta_i - theta_A_i)^2
```

where F_i is the diagonal of the Fisher information matrix (approximated as the
expected squared gradient), theta_A are the optimal parameters for task A, and
lambda controls the regularization strength.

Key properties:
- F_i is large for parameters important to Task A --> those parameters are penalized
  more for changing
- F_i is small for unimportant parameters --> those are free to change for Task B
- Diagonal approximation makes computation tractable
- Online EWC accumulates Fisher information across all previous tasks

### Synaptic Intelligence (SI)

**SI** (Zenke et al., 2017) computes parameter importance online during training,
rather than after training. For each parameter, SI tracks the cumulative contribution
to the loss reduction:

```
omega_i = sum_t (Delta_L_t / (Delta_theta_i)^2 + epsilon)
```

The regularization loss mirrors EWC but uses omega instead of Fisher information.
Advantages over EWC: computed online (no separate Fisher computation phase), considers
the entire training trajectory rather than just the final point.

### Memory Aware Synapses (MAS)

**MAS** (Aljundi et al., 2018) estimates parameter importance based on the sensitivity
of the learned function's output to parameter changes, rather than the loss. Importance
is measured as:

```
omega_i = (1/N) * sum_n |partial f(x_n) / partial theta_i|
```

This is task-agnostic (does not require labels) and can be computed on unlabeled data,
making it suitable for unsupervised continual learning.

### Learning without Forgetting (LwF)

**LwF** (Li & Hoiem, 2017) uses knowledge distillation to preserve old task performance
without storing any old task data. Before training on the new task:

1. Record the model's outputs on the new task's data for the old task heads
2. Use these recorded outputs as soft targets during new task training

The loss becomes:

```
L_total = L_new(theta) + lambda * L_distill(f_old(x), f_new(x))
```

where L_distill is typically KL divergence between old and new task outputs.
Limitation: does not work well when old and new tasks have very different data
distributions.

### Code: EWC Implementation in PyTorch

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional
import copy


class EWC:
    """
    Elastic Weight Consolidation for continual learning.

    After learning each task, computes the Fisher information matrix
    to identify important parameters. During subsequent task training,
    penalizes changes to important parameters.

    Reference: Kirkpatrick et al., "Overcoming catastrophic forgetting
    in neural networks" (2017)
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        online: bool = False,
        gamma: float = 1.0,
    ):
        """
        Args:
            model: The neural network model.
            ewc_lambda: Regularization strength. Higher values preserve
                        old tasks more but reduce plasticity.
            online: If True, use online EWC (running Fisher estimate).
            gamma: Decay factor for online EWC (0 < gamma <= 1).
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.online = online
        self.gamma = gamma

        # Store Fisher information and parameter snapshots per task
        self.fisher_matrices: List[Dict[str, torch.Tensor]] = []
        self.param_snapshots: List[Dict[str, torch.Tensor]] = []

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the diagonal Fisher information matrix.

        Uses empirical Fisher: average of squared gradients of the
        log-likelihood over training data.
        """
        fisher = {
            name: torch.zeros_like(param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        self.model.eval()
        count = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if num_samples is not None and count >= num_samples:
                break

            device = next(self.model.parameters()).device
            inputs, targets = inputs.to(device), targets.to(device)

            self.model.zero_grad()
            outputs = self.model(inputs)

            # Use log-likelihood (negative cross-entropy per sample)
            loss = nn.functional.cross_entropy(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach() ** 2

            count += inputs.size(0)

        # Normalize
        for name in fisher:
            fisher[name] /= max(count, 1)

        return fisher

    def register_task(self, dataloader: DataLoader, num_samples: Optional[int] = None):
        """
        Register a completed task: compute Fisher and store parameters.

        Call this after training on each task.
        """
        fisher = self.compute_fisher(dataloader, num_samples)

        if self.online and len(self.fisher_matrices) > 0:
            # Online EWC: accumulate Fisher with decay
            prev_fisher = self.fisher_matrices[-1]
            for name in fisher:
                fisher[name] = self.gamma * prev_fisher[name] + fisher[name]
            self.fisher_matrices[-1] = fisher
            self.param_snapshots[-1] = {
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            }
        else:
            self.fisher_matrices.append(fisher)
            self.param_snapshots.append({
                name: param.detach().clone()
                for name, param in self.model.named_parameters()
                if param.requires_grad
            })

    def penalty(self) -> torch.Tensor:
        """
        Compute the EWC penalty term.

        Returns:
            Scalar tensor representing the EWC regularization loss.
        """
        if len(self.fisher_matrices) == 0:
            return torch.tensor(0.0, device=next(self.model.parameters()).device)

        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)

        for fisher, params_old in zip(self.fisher_matrices, self.param_snapshots):
            for name, param in self.model.named_parameters():
                if name in fisher:
                    loss += (fisher[name] * (param - params_old[name]) ** 2).sum()

        return (self.ewc_lambda / 2.0) * loss


def train_with_ewc(
    model: nn.Module,
    ewc: EWC,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
):
    """Training loop with EWC regularization."""
    device = next(model.parameters()).device
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        total_task_loss = 0.0
        total_ewc_loss = 0.0

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            task_loss = nn.functional.cross_entropy(outputs, targets)
            ewc_loss = ewc.penalty()

            loss = task_loss + ewc_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            total_ewc_loss += ewc_loss.item()

        num_batches = len(dataloader)
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Loss: {total_loss/num_batches:.4f} "
            f"(task: {total_task_loss/num_batches:.4f}, "
            f"ewc: {total_ewc_loss/num_batches:.4f})"
        )
```

---

## 37.8 Replay-Based Methods

Replay-based methods maintain a memory buffer of examples from previous tasks and
interleave them during training on new tasks. This directly addresses forgetting by
periodically revisiting old data.

### Experience Replay

The simplest replay approach stores a fixed-size buffer of examples from previous tasks.
During training on a new task, each mini-batch includes both new task data and randomly
sampled old task data.

**Buffer management strategies**:
- **Reservoir sampling**: Each example has equal probability of being in the buffer,
  regardless of when it was seen. Maintains a representative sample.
- **Ring buffer**: Allocate equal space per task. New examples from a task replace the
  oldest examples from that task.
- **Herding**: Select buffer examples that best approximate the class mean in feature
  space (used in iCaRL).

### Generative Replay

Instead of storing real examples, **generative replay** (Shin et al., 2017) trains a
generative model (VAE or GAN) alongside the main model. When learning a new task, the
generator produces synthetic examples from previous tasks.

Advantages:
- No need to store real data (privacy-preserving)
- Buffer size is effectively unlimited

Disadvantages:
- Generator quality limits replay quality
- Generator itself suffers from forgetting
- Computationally expensive

### Dark Experience Replay (DER/DER++)

**DER** (Buzzega et al., 2020) stores not just inputs and labels but also the model's
logit outputs at the time of storage. During replay, the model is trained to match both
the stored labels and the stored logits:

```
L = L_ce(f(x_new), y_new) + alpha * L_mse(f(x_buf), logits_buf) + beta * L_ce(f(x_buf), y_buf)
```

DER++ adds the classification loss on buffer samples (the beta term). This combines
the benefits of experience replay and knowledge distillation.

### Gradient Episodic Memory (GEM)

**GEM** (Lopez-Paz & Ranzato, 2017) uses the replay buffer not for direct rehearsal
but as a constraint. It ensures that the gradient update for the new task does not
increase the loss on any previous task's buffer examples.

Formally, GEM solves:
```
minimize ||g - g_new||^2
subject to <g, g_t> >= 0  for all previous tasks t
```

where g_new is the gradient from the new task and g_t are gradients computed on
buffer samples from task t.

### A-GEM (Averaged GEM)

**A-GEM** (Chaudhry et al., 2019) simplifies GEM by replacing per-task constraints
with a single averaged constraint. Instead of ensuring non-negative inner product with
each task gradient individually, A-GEM constrains the update against the average
gradient over all buffer samples:

```
if <g_new, g_ref> < 0:
    g = g_new - (<g_new, g_ref> / <g_ref, g_ref>) * g_ref
```

This is much cheaper than solving GEM's quadratic program and scales better.

### Code: Experience Replay Buffer for Continual Learning

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Tuple, Optional, Dict
import random


class ReplayBuffer:
    """
    Experience replay buffer for continual learning.

    Supports reservoir sampling for balanced representation
    across all tasks, and ring buffer for equal per-task allocation.
    """

    def __init__(
        self,
        max_size: int = 5000,
        strategy: str = "reservoir",
        store_logits: bool = False,
    ):
        """
        Args:
            max_size: Maximum number of examples in the buffer.
            strategy: 'reservoir' for reservoir sampling, 'ring' for
                      ring buffer with equal per-task allocation.
            store_logits: If True, also store model logits (for DER).
        """
        self.max_size = max_size
        self.strategy = strategy
        self.store_logits = store_logits

        self.buffer_x: list = []
        self.buffer_y: list = []
        self.buffer_logits: list = []
        self.buffer_task_ids: list = []
        self.seen_count = 0  # Total examples seen (for reservoir sampling)
        self.task_counts: Dict[int, int] = {}  # Per-task count in buffer

    def add_task_data(
        self,
        dataloader: DataLoader,
        task_id: int,
        model: Optional[nn.Module] = None,
    ):
        """
        Add examples from a completed task to the buffer.

        Args:
            dataloader: DataLoader for the completed task.
            task_id: Integer identifier for this task.
            model: Current model (required if store_logits=True).
        """
        if self.strategy == "reservoir":
            self._reservoir_add(dataloader, task_id, model)
        elif self.strategy == "ring":
            self._ring_add(dataloader, task_id, model)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _reservoir_add(
        self, dataloader: DataLoader, task_id: int,
        model: Optional[nn.Module] = None,
    ):
        """Reservoir sampling: each example has equal probability of being stored."""
        device = "cpu"
        if model is not None:
            model.eval()
            device = next(model.parameters()).device

        for inputs, targets in dataloader:
            for i in range(inputs.size(0)):
                self.seen_count += 1

                if len(self.buffer_x) < self.max_size:
                    # Buffer not full: add directly
                    self.buffer_x.append(inputs[i].cpu())
                    self.buffer_y.append(targets[i].cpu())
                    self.buffer_task_ids.append(task_id)

                    if self.store_logits and model is not None:
                        with torch.no_grad():
                            logit = model(inputs[i:i+1].to(device)).cpu().squeeze(0)
                        self.buffer_logits.append(logit)
                else:
                    # Buffer full: replace with probability max_size / seen_count
                    idx = random.randint(0, self.seen_count - 1)
                    if idx < self.max_size:
                        old_task = self.buffer_task_ids[idx]
                        self.task_counts[old_task] = self.task_counts.get(old_task, 1) - 1

                        self.buffer_x[idx] = inputs[i].cpu()
                        self.buffer_y[idx] = targets[i].cpu()
                        self.buffer_task_ids[idx] = task_id

                        if self.store_logits and model is not None:
                            with torch.no_grad():
                                logit = model(inputs[i:i+1].to(device)).cpu().squeeze(0)
                            self.buffer_logits[idx] = logit

        self.task_counts[task_id] = sum(1 for t in self.buffer_task_ids if t == task_id)

    def _ring_add(
        self, dataloader: DataLoader, task_id: int,
        model: Optional[nn.Module] = None,
    ):
        """Ring buffer: equal allocation per task."""
        num_tasks = len(self.task_counts) + 1
        per_task = self.max_size // num_tasks

        # Collect all new task examples
        new_x, new_y = [], []
        for inputs, targets in dataloader:
            new_x.append(inputs)
            new_y.append(targets)
        new_x = torch.cat(new_x, dim=0)
        new_y = torch.cat(new_y, dim=0)

        # Randomly select per_task examples from the new task
        indices = torch.randperm(len(new_x))[:per_task]
        selected_x = [new_x[i].cpu() for i in indices]
        selected_y = [new_y[i].cpu() for i in indices]

        # Trim existing tasks to per_task examples each
        trimmed_x, trimmed_y, trimmed_tasks = [], [], []
        for t_id in self.task_counts:
            task_indices = [
                j for j, tid in enumerate(self.buffer_task_ids) if tid == t_id
            ]
            keep = task_indices[:per_task]
            for j in keep:
                trimmed_x.append(self.buffer_x[j])
                trimmed_y.append(self.buffer_y[j])
                trimmed_tasks.append(t_id)

        # Rebuild buffer
        self.buffer_x = trimmed_x + selected_x
        self.buffer_y = trimmed_y + selected_y
        self.buffer_task_ids = trimmed_tasks + [task_id] * len(selected_x)
        self.task_counts[task_id] = len(selected_x)
        for t_id in list(self.task_counts.keys()):
            if t_id != task_id:
                self.task_counts[t_id] = min(self.task_counts[t_id], per_task)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch from the buffer."""
        if len(self.buffer_x) == 0:
            raise ValueError("Buffer is empty.")

        indices = random.sample(
            range(len(self.buffer_x)),
            min(batch_size, len(self.buffer_x)),
        )
        x = torch.stack([self.buffer_x[i] for i in indices])
        y = torch.stack([self.buffer_y[i] for i in indices])
        return x, y

    def sample_with_logits(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample batch including stored logits (for DER)."""
        if not self.store_logits or len(self.buffer_logits) == 0:
            raise ValueError("Logits not available.")

        indices = random.sample(
            range(len(self.buffer_x)),
            min(batch_size, len(self.buffer_x)),
        )
        x = torch.stack([self.buffer_x[i] for i in indices])
        y = torch.stack([self.buffer_y[i] for i in indices])
        logits = torch.stack([self.buffer_logits[i] for i in indices])
        return x, y, logits

    @property
    def size(self) -> int:
        return len(self.buffer_x)

    def stats(self) -> Dict:
        return {
            "total_size": self.size,
            "max_size": self.max_size,
            "task_distribution": dict(self.task_counts),
            "seen_count": self.seen_count,
        }


def train_with_replay(
    model: nn.Module,
    new_task_loader: DataLoader,
    replay_buffer: ReplayBuffer,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = 10,
    replay_batch_size: int = 32,
    replay_weight: float = 1.0,
):
    """Training loop with experience replay."""
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, targets in new_task_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # New task loss
            outputs = model(inputs)
            new_loss = criterion(outputs, targets)

            # Replay loss
            replay_loss = torch.tensor(0.0, device=device)
            if replay_buffer.size > 0:
                buf_x, buf_y = replay_buffer.sample(replay_batch_size)
                buf_x, buf_y = buf_x.to(device), buf_y.to(device)
                buf_outputs = model(buf_x)
                replay_loss = criterion(buf_outputs, buf_y)

            loss = new_loss + replay_weight * replay_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(new_task_loader)
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
```

---

## 37.9 Architecture-Based Methods

Architecture-based methods modify the network structure to accommodate new tasks,
either by adding new components or by isolating subsets of parameters for each task.

### Progressive Neural Networks

**Progressive Neural Networks** (Rusu et al., 2016) add a new "column" (a complete
network) for each new task. Lateral connections from all previous columns feed into
the new column, enabling forward transfer. Previous columns are frozen, preventing
forgetting entirely.

Structure:
- Task 1: Column 1 (trained, then frozen)
- Task 2: Column 2 + lateral connections from Column 1 (trained, then frozen)
- Task k: Column k + lateral connections from Columns 1..k-1

Advantages: Zero forgetting (frozen columns), positive forward transfer (lateral
connections). Disadvantages: Linear growth in parameters with number of tasks,
no backward transfer.

### PackNet

**PackNet** (Mallya & Lazebnik, 2018) uses iterative pruning and weight freezing:

1. Train on Task 1 using the full network
2. Prune unimportant weights (e.g., smallest magnitude) to free capacity
3. Freeze remaining weights for Task 1
4. Train on Task 2 using only the freed weights
5. Prune and freeze for Task 2
6. Repeat

This maintains a single network with task-specific binary masks. At inference,
the appropriate mask selects which weights to use. The total capacity is fixed,
so the number of tasks is bounded by the pruning rate.

### DEN (Dynamically Expandable Networks)

**DEN** (Yoon et al., 2018) dynamically expands the network capacity as needed:

1. Attempt to learn the new task using selective retraining (only retrain relevant neurons)
2. If performance is insufficient, split neurons that were important for both old and new tasks
3. If still insufficient, add new neurons and connections
4. Apply group sparse regularization to keep the expansion efficient

### HAT (Hard Attention to the Task)

**HAT** (Serra et al., 2018) learns a binary attention mask per task that gates
the activations of each layer. During training on task t, the mask m_t is learned
alongside the weights. Previous task masks are used to protect important units:

- Units used by previous tasks (mask value close to 1) have their gradients blocked
- Units unused by previous tasks (mask value close to 0) are available for the new task

The attention is implemented using a sigmoid with learned embeddings per task, and
an annealing schedule drives the sigmoid towards binary values.

### Adapter Modules for Continual Learning

**Adapter-based methods** insert small task-specific modules (adapters) into a frozen
pre-trained backbone:

- Each adapter is a bottleneck: down-projection, nonlinearity, up-projection
- The backbone remains frozen (no forgetting)
- Each new task adds its own set of adapters
- Parameter growth is minimal (adapters are typically 1-5% of backbone size)

This approach is especially effective with large pre-trained models (BERT, ViT)
where the backbone features are already general-purpose.

### Code: Progressive Networks Concept

```python
import torch
import torch.nn as nn
from typing import List, Optional


class ProgressiveColumn(nn.Module):
    """
    A single column in a Progressive Neural Network.

    Receives lateral connections from all previous columns and
    combines them with its own forward computation.
    """

    def __init__(
        self,
        layer_dims: List[int],
        prev_columns: Optional[List["ProgressiveColumn"]] = None,
    ):
        super().__init__()
        self.layer_dims = layer_dims
        self.num_layers = len(layer_dims) - 1

        # Main layers for this column
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

        # Lateral connections from previous columns
        self.lateral_connections = nn.ModuleList()
        if prev_columns:
            for layer_idx in range(self.num_layers):
                lateral_for_layer = nn.ModuleList()
                for prev_col in prev_columns:
                    # Connect from prev column's layer output to this column's layer input
                    prev_dim = prev_col.layer_dims[layer_idx + 1]
                    curr_dim = layer_dims[layer_idx + 1]
                    lateral_for_layer.append(nn.Linear(prev_dim, curr_dim))
                self.lateral_connections.append(lateral_for_layer)

        self.activation = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        prev_activations: Optional[List[List[torch.Tensor]]] = None,
    ) -> List[torch.Tensor]:
        """
        Forward pass through this column.

        Args:
            x: Input tensor.
            prev_activations: List of activation lists from previous columns.
                              prev_activations[col_idx][layer_idx] = tensor.

        Returns:
            List of activations at each layer (for use by future columns).
        """
        activations = []
        h = x

        for layer_idx in range(self.num_layers):
            h = self.layers[layer_idx](h)

            # Add lateral connections
            if prev_activations and len(self.lateral_connections) > layer_idx:
                for col_idx, lateral in enumerate(self.lateral_connections[layer_idx]):
                    prev_h = prev_activations[col_idx][layer_idx]
                    h = h + lateral(prev_h)

            if layer_idx < self.num_layers - 1:
                h = self.activation(h)

            activations.append(h)

        return activations


class ProgressiveNetwork(nn.Module):
    """
    Progressive Neural Network for continual learning.

    Adds a new column for each task. Previous columns are frozen.
    Lateral connections enable forward transfer.

    Reference: Rusu et al., "Progressive Neural Networks" (2016)
    """

    def __init__(self, layer_dims: List[int]):
        """
        Args:
            layer_dims: Dimensions for each layer [input, h1, h2, ..., output].
        """
        super().__init__()
        self.layer_dims = layer_dims
        self.columns: nn.ModuleList = nn.ModuleList()
        self.num_tasks = 0

    def add_task(self) -> int:
        """
        Add a new column for a new task.

        Returns:
            Task index for the new task.
        """
        prev_columns = list(self.columns) if len(self.columns) > 0 else None
        new_column = ProgressiveColumn(self.layer_dims, prev_columns)

        # Freeze all previous columns
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False

        self.columns.append(new_column)
        self.num_tasks += 1
        return self.num_tasks - 1

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass for a specific task.

        Args:
            x: Input tensor.
            task_id: Which task's output to produce.

        Returns:
            Output tensor from the specified task's column.
        """
        all_activations = []

        for col_idx in range(task_id + 1):
            prev_acts = all_activations[:col_idx] if col_idx > 0 else None
            activations = self.columns[col_idx](x, prev_acts)
            all_activations.append(activations)

        # Return the final layer activation from the requested task's column
        return all_activations[task_id][-1]

    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return parameters of the latest (unfrozen) column only."""
        if len(self.columns) == 0:
            return []
        return list(self.columns[-1].parameters())


# Usage example
if __name__ == "__main__":
    prog_net = ProgressiveNetwork(layer_dims=[784, 256, 128, 10])

    # Task 0
    task_0 = prog_net.add_task()
    optimizer_0 = torch.optim.Adam(prog_net.get_trainable_params(), lr=1e-3)

    x = torch.randn(32, 784)
    out = prog_net(x, task_id=0)
    print(f"Task 0 output shape: {out.shape}")  # (32, 10)

    # Task 1 -- previous column is frozen, new column added
    task_1 = prog_net.add_task()
    optimizer_1 = torch.optim.Adam(prog_net.get_trainable_params(), lr=1e-3)

    out = prog_net(x, task_id=1)
    print(f"Task 1 output shape: {out.shape}")  # (32, 10)

    # Verify freezing
    frozen_params = sum(1 for p in prog_net.columns[0].parameters() if not p.requires_grad)
    trainable_params = sum(1 for p in prog_net.columns[1].parameters() if p.requires_grad)
    print(f"Column 0 frozen params: {frozen_params}")
    print(f"Column 1 trainable params: {trainable_params}")
```

---

## 37.10 Continual Learning Benchmarks

### Split MNIST / Split CIFAR-10/100

The most common CL benchmarks split a classification dataset into sequential tasks:

- **Split MNIST**: 10 digits split into 5 tasks of 2 classes each (0/1, 2/3, 4/5, 6/7, 8/9)
- **Split CIFAR-10**: 10 classes split into 5 tasks of 2 classes each
- **Split CIFAR-100**: 100 classes split into 10 or 20 tasks

These are simple but widely used for initial method validation. Limitations:
tasks are clearly separated, data distribution shifts are abrupt, and the
datasets are relatively small.

### Permuted MNIST

Each task uses the same MNIST dataset but with a fixed random permutation applied
to pixel locations. This creates domain-incremental learning: the task structure
(digit classification) stays the same, but the input distribution changes.

Useful because:
- Tasks all have the same difficulty
- Unlimited number of tasks (different permutations)
- Tests domain adaptation rather than class separation

### CORe50 (Continuous Object Recognition)

**CORe50** is a benchmark specifically designed for continual learning research
with real-world conditions:

- 50 objects from 10 categories
- 11 distinct sessions (different backgrounds, lighting, poses)
- Supports New Instances (NI), New Classes (NC), and NI+NC scenarios
- Temporal correlation within sessions (video frames)

### CLEAR Benchmark

**CLEAR** (Continual LEArning on Real-World imagery) uses data from YFCC100M
spanning 2004-2014, providing naturally evolving visual concepts:

- 10 time-stamped buckets of images
- Natural distribution shift over time
- Tests both in-distribution and out-of-distribution generalization
- More realistic than artificial splits

### Avalanche Library

**Avalanche** is the most comprehensive framework for continual learning research.
Built on PyTorch, it provides:

- Standardized benchmarks (all of the above and more)
- Pre-implemented strategies (EWC, SI, LwF, GEM, Replay, etc.)
- Evaluation protocols with standard metrics
- Plugin system for combining methods
- Logging and visualization

### Code: Continual Learning with Avalanche

```python
import torch
import torch.nn as nn
from typing import List

# Avalanche imports
try:
    from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR100
    from avalanche.models import SimpleMLP
    from avalanche.training.supervised import (
        Naive,
        EWC,
        Replay,
        GEM,
        LwF,
    )
    from avalanche.evaluation.metrics import (
        accuracy_metrics,
        forgetting_metrics,
        loss_metrics,
        timing_metrics,
    )
    from avalanche.logging import InteractiveLogger, TextLogger
    from avalanche.training.plugins import EvaluationPlugin
    AVALANCHE_AVAILABLE = True
except ImportError:
    AVALANCHE_AVAILABLE = False
    print("Avalanche not installed. Install with: pip install avalanche-lib")


def create_benchmark(name: str = "split_mnist", n_experiences: int = 5):
    """Create a continual learning benchmark."""
    if name == "split_mnist":
        return SplitMNIST(
            n_experiences=n_experiences,
            seed=42,
            return_task_id=True,
        )
    elif name == "split_cifar100":
        return SplitCIFAR100(
            n_experiences=n_experiences,
            seed=42,
            return_task_id=True,
        )
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def create_evaluation_plugin(log_dir: str = "./logs"):
    """Create standard CL evaluation plugin."""
    return EvaluationPlugin(
        accuracy_metrics(
            minibatch=False,
            epoch=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        timing_metrics(epoch=True),
        loggers=[InteractiveLogger(), TextLogger(open(f"{log_dir}/log.txt", "w"))],
    )


def run_continual_learning_experiment(
    strategy_name: str = "ewc",
    benchmark_name: str = "split_mnist",
    n_experiences: int = 5,
    epochs_per_task: int = 5,
    lr: float = 0.001,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """
    Run a complete continual learning experiment.

    Args:
        strategy_name: One of 'naive', 'ewc', 'replay', 'gem', 'lwf'.
        benchmark_name: One of 'split_mnist', 'split_cifar100'.
        n_experiences: Number of sequential tasks.
        epochs_per_task: Training epochs per task.
        lr: Learning rate.
        device: Device for training.
    """
    if not AVALANCHE_AVAILABLE:
        print("Avalanche is required for this experiment.")
        return

    # Create benchmark
    benchmark = create_benchmark(benchmark_name, n_experiences)

    # Create model
    if benchmark_name == "split_mnist":
        model = SimpleMLP(num_classes=10, input_size=28 * 28)
    else:
        model = SimpleMLP(num_classes=100, input_size=3 * 32 * 32)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    eval_plugin = create_evaluation_plugin()

    # Create strategy
    strategy_map = {
        "naive": lambda: Naive(
            model, optimizer, criterion,
            train_epochs=epochs_per_task,
            device=device,
            evaluator=eval_plugin,
        ),
        "ewc": lambda: EWC(
            model, optimizer, criterion,
            ewc_lambda=1000.0,
            train_epochs=epochs_per_task,
            device=device,
            evaluator=eval_plugin,
        ),
        "replay": lambda: Replay(
            model, optimizer, criterion,
            mem_size=500,
            train_epochs=epochs_per_task,
            device=device,
            evaluator=eval_plugin,
        ),
        "gem": lambda: GEM(
            model, optimizer, criterion,
            patterns_per_exp=256,
            memory_strength=0.5,
            train_epochs=epochs_per_task,
            device=device,
            evaluator=eval_plugin,
        ),
        "lwf": lambda: LwF(
            model, optimizer, criterion,
            alpha=1.0,
            temperature=2.0,
            train_epochs=epochs_per_task,
            device=device,
            evaluator=eval_plugin,
        ),
    }

    if strategy_name not in strategy_map:
        raise ValueError(f"Unknown strategy: {strategy_name}. Choose from {list(strategy_map)}")

    strategy = strategy_map[strategy_name]()

    # Train sequentially on each experience
    print(f"\n{'='*60}")
    print(f"Strategy: {strategy_name.upper()}")
    print(f"Benchmark: {benchmark_name}")
    print(f"{'='*60}\n")

    results = []
    for exp_id, experience in enumerate(benchmark.train_stream):
        print(f"\n--- Training on Experience {exp_id} ---")
        res = strategy.train(experience)

        print(f"\n--- Evaluating after Experience {exp_id} ---")
        eval_res = strategy.eval(benchmark.test_stream)
        results.append(eval_res)

    return results


if __name__ == "__main__" and AVALANCHE_AVAILABLE:
    # Compare strategies
    for strategy in ["naive", "ewc", "replay"]:
        run_continual_learning_experiment(
            strategy_name=strategy,
            benchmark_name="split_mnist",
            n_experiences=5,
            epochs_per_task=3,
        )
```

---

## 37.11 Curriculum Learning

**Curriculum learning** (Bengio et al., 2009) organizes the training data in a
meaningful order, typically from easy to hard examples, rather than presenting
data in random order. This mirrors how humans learn: starting with fundamentals
before tackling complex material.

### Self-Paced Learning

**Self-paced learning** (Kumar et al., 2010) lets the model itself decide which
examples are easy. Examples with low loss are considered easy and are prioritized.
A threshold parameter lambda controls the difficulty boundary, and lambda increases
over training to gradually include harder examples.

The objective is:

```
min_{w, v} sum_i (v_i * L(x_i, y_i; w) - lambda * v_i)
```

where v_i in {0, 1} indicates whether example i is selected. The closed-form
solution selects examples with loss below lambda.

### Teacher-Student Curriculum

An external "teacher" model or heuristic defines the curriculum. The teacher
evaluates example difficulty and schedules them for the student:

- **Loss-based**: Teacher ranks examples by the student's loss
- **Confidence-based**: Teacher ranks examples by prediction confidence
- **Transfer-based**: Use a pre-trained model's uncertainty as difficulty proxy

### Competence-Based Curriculum

The curriculum adapts to the learner's current competence level c(t):

```
c(t) = min(1, sqrt(t * (1 - c_0^2) / T + c_0^2))
```

At each step, only examples with difficulty below c(t) are eligible for sampling.
Difficulty can be measured by sentence length (NLP), image complexity (CV), or
any task-appropriate metric.

### Baby-Step Curriculum

A strict ordering that introduces examples in difficulty buckets:
1. Train only on easiest bucket until convergence
2. Add the next difficulty bucket
3. Repeat until all data is included

This is the most structured form of curriculum learning. It works well when
difficulty can be reliably estimated but can be sensitive to the difficulty metric.

### Anti-Curriculum (Hard Examples First)

Counterintuitively, some work shows that starting with the hardest examples can
also improve training, particularly for:

- Tasks where hard examples are the most informative
- Models that need to learn decision boundaries early
- Imbalanced datasets where minority classes are "hard"

**Focal loss** (Lin et al., 2017) can be viewed as a soft anti-curriculum: it
upweights hard (misclassified) examples throughout training.

### Code: Curriculum Learning Scheduler

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
import numpy as np
from typing import Callable, Optional, List
import math


class DifficultyScorer:
    """Compute difficulty scores for training examples."""

    def __init__(self, method: str = "loss"):
        """
        Args:
            method: 'loss' (model loss), 'length' (sequence length),
                    'confidence' (1 - max prob), or 'precomputed'.
        """
        self.method = method
        self.scores: Optional[np.ndarray] = None

    @torch.no_grad()
    def compute_scores(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: str = "cpu",
    ) -> np.ndarray:
        """Compute difficulty scores for all examples in the dataset."""
        model.eval()
        all_scores = []

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if self.method == "loss":
                # Per-example loss (higher = harder)
                losses = nn.functional.cross_entropy(
                    outputs, targets, reduction="none"
                )
                all_scores.append(losses.cpu().numpy())
            elif self.method == "confidence":
                # 1 - max probability (higher = harder)
                probs = torch.softmax(outputs, dim=-1)
                confidence = 1.0 - probs.max(dim=-1).values
                all_scores.append(confidence.cpu().numpy())

        self.scores = np.concatenate(all_scores)
        return self.scores

    def set_precomputed_scores(self, scores: np.ndarray):
        """Set externally computed difficulty scores."""
        self.scores = scores
        self.method = "precomputed"


class CurriculumSampler(Sampler):
    """
    Sampler that implements curriculum learning by controlling
    which examples are available at each epoch.
    """

    def __init__(
        self,
        difficulty_scores: np.ndarray,
        total_epochs: int,
        strategy: str = "linear",
        initial_fraction: float = 0.2,
        anti_curriculum: bool = False,
    ):
        """
        Args:
            difficulty_scores: Array of difficulty scores per example.
            total_epochs: Total training epochs (used for scheduling).
            strategy: 'linear', 'sqrt', 'step', or 'exponential'.
            initial_fraction: Fraction of data available at epoch 0.
            anti_curriculum: If True, start with hard examples instead.
        """
        self.scores = difficulty_scores
        self.total_epochs = total_epochs
        self.strategy = strategy
        self.initial_fraction = initial_fraction
        self.anti_curriculum = anti_curriculum
        self.current_epoch = 0
        self.num_examples = len(difficulty_scores)

        # Sort indices by difficulty
        self.sorted_indices = np.argsort(difficulty_scores)
        if anti_curriculum:
            self.sorted_indices = self.sorted_indices[::-1].copy()

    def set_epoch(self, epoch: int):
        """Update the current epoch for the schedule."""
        self.current_epoch = epoch

    def _get_fraction(self) -> float:
        """Compute the fraction of data to use at current epoch."""
        t = self.current_epoch / max(self.total_epochs - 1, 1)

        if self.strategy == "linear":
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * t
        elif self.strategy == "sqrt":
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * math.sqrt(t)
        elif self.strategy == "exponential":
            frac = 1.0 - (1.0 - self.initial_fraction) * math.exp(-3.0 * t)
        elif self.strategy == "step":
            num_steps = 5
            step = min(int(t * num_steps), num_steps - 1)
            frac = self.initial_fraction + (1.0 - self.initial_fraction) * step / (num_steps - 1)
        else:
            frac = 1.0

        return min(frac, 1.0)

    def __iter__(self):
        frac = self._get_fraction()
        num_available = max(1, int(frac * self.num_examples))

        # Select the easiest (or hardest, if anti-curriculum) examples
        available_indices = self.sorted_indices[:num_available]

        # Shuffle the available indices
        perm = np.random.permutation(len(available_indices))
        yield from available_indices[perm].tolist()

    def __len__(self):
        frac = self._get_fraction()
        return max(1, int(frac * self.num_examples))


class CurriculumTrainer:
    """Complete curriculum learning training loop."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scorer: DifficultyScorer,
        strategy: str = "linear",
        initial_fraction: float = 0.2,
        score_update_freq: int = 5,
        device: str = "cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scorer = scorer
        self.strategy = strategy
        self.initial_fraction = initial_fraction
        self.score_update_freq = score_update_freq
        self.device = device

    def train(
        self,
        dataset: Dataset,
        num_epochs: int,
        batch_size: int = 64,
    ):
        """Run curriculum-guided training."""
        # Initial difficulty scoring
        init_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        scores = self.scorer.compute_scores(self.model, init_loader, self.device)

        sampler = CurriculumSampler(
            difficulty_scores=scores,
            total_epochs=num_epochs,
            strategy=self.strategy,
            initial_fraction=self.initial_fraction,
        )

        for epoch in range(num_epochs):
            sampler.set_epoch(epoch)
            loader = DataLoader(
                dataset, batch_size=batch_size, sampler=sampler
            )

            self.model.train()
            epoch_loss = 0.0
            num_batches = 0

            for inputs, targets in loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            frac = sampler._get_fraction()
            avg_loss = epoch_loss / max(num_batches, 1)
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Loss: {avg_loss:.4f} - "
                f"Data fraction: {frac:.2%} ({len(sampler)} examples)"
            )

            # Periodically re-score difficulty
            if (epoch + 1) % self.score_update_freq == 0 and epoch < num_epochs - 1:
                rescore_loader = DataLoader(
                    dataset, batch_size=batch_size, shuffle=False
                )
                scores = self.scorer.compute_scores(
                    self.model, rescore_loader, self.device
                )
                sampler = CurriculumSampler(
                    difficulty_scores=scores,
                    total_epochs=num_epochs,
                    strategy=self.strategy,
                    initial_fraction=self.initial_fraction,
                )
                sampler.set_epoch(epoch + 1)
                print(f"  [Difficulty scores updated]")
```

---

## 37.12 Meta-Continual Learning

**Meta-continual learning** combines meta-learning and continual learning, using
meta-learning to learn how to learn continually. The idea is to meta-train a model
that can quickly adapt to new tasks without forgetting old ones.

### ANML (A Neuromodulated Meta-Learning Algorithm)

**ANML** (Beaulieu et al., 2020) uses a neuromodulatory network to gate the
activations of a prediction network. The neuromodulatory network learns to
selectively activate or suppress neurons for each task, creating task-specific
pathways through a shared network.

Architecture:
- **Prediction network**: Standard feedforward network
- **Neuromodulation network**: Takes input and produces a gating mask
- Combined: output = prediction_net(input * gate(input))

Meta-training with MAML-style optimization teaches the neuromodulator to create
masks that protect previously learned pathways while enabling new learning.

### OML (Online Meta-Learning)

**OML** (Javed & White, 2019) meta-learns a representation that is suitable for
continual learning. The key insight is to meta-train the feature extractor such
that a simple online learner (e.g., linear classifier with SGD) can learn new
tasks sequentially without forgetting.

The representation learning network (RLN) is trained using MAML to:
1. Learn a new task from few examples
2. Retain performance on previously seen tasks

After meta-training, the RLN is frozen and only the online learner is updated
during deployment.

### Connection to Few-Shot Learning

Meta-continual learning is closely related to few-shot learning:

- Both learn from limited data
- Both use meta-learning to achieve fast adaptation
- The key difference: continual learning requires not forgetting, while few-shot
  learning typically does not track performance on previous episodes

Methods like **Prototypical Networks** can be adapted for continual learning by
maintaining and updating class prototypes as new data arrives. **MAML** derivatives
that incorporate replay or regularization can handle the continual setting.

The progression from few-shot to continual learning:
1. **Few-shot**: Adapt quickly to new task, no memory requirement
2. **Continual few-shot**: Adapt quickly AND remember all previous tasks
3. **Meta-continual**: Learn the learning algorithm that achieves continual few-shot

---

## 37.13 Comparison and Selection Guide

### Methods Comparison Table

| Method | Category | Memory Cost | Compute Cost | Forgetting | Forward Transfer | Scalability |
|--------|----------|-------------|--------------|------------|-----------------|-------------|
| Fine-tuning (baseline) | - | None | Low | Severe | None | High |
| EWC | Regularization | O(params) | Medium | Moderate | Low | High |
| SI | Regularization | O(params) | Low | Moderate | Low | High |
| MAS | Regularization | O(params) | Medium | Moderate | Low | High |
| LwF | Regularization | None | Medium | Moderate | Low | High |
| Experience Replay | Replay | O(buffer) | Low | Low | Low | High |
| DER++ | Replay | O(buffer) | Low | Very Low | Low | High |
| GEM | Replay | O(buffer) | High | Low | Low | Medium |
| A-GEM | Replay | O(buffer) | Medium | Moderate | Low | High |
| Progressive Nets | Architecture | O(T * params) | High | None | High | Low |
| PackNet | Architecture | O(masks) | Medium | None | Moderate | Medium |
| HAT | Architecture | O(masks) | Medium | Very Low | Moderate | Medium |
| Adapters | Architecture | O(T * adapter) | Low | None | Low | High |

T = number of tasks, buffer = replay buffer size, params = model parameters.

### When to Use Each Approach

**Use regularization methods (EWC, SI, MAS) when**:
- You cannot store old task data (privacy, storage constraints)
- You need a simple baseline with reasonable performance
- The number of tasks is moderate (performance degrades with many tasks)
- You prefer a single model without architectural changes

**Use replay methods (Experience Replay, DER++) when**:
- You can store a small buffer of old data
- You need strong performance with many tasks
- You want the simplest method that works well (replay is often the strongest)
- Computational overhead must be minimal

**Use architecture methods (Progressive, PackNet, HAT) when**:
- Zero forgetting is required (safety-critical applications)
- You know the total number of tasks in advance (for capacity planning)
- Forward transfer is important (Progressive Nets)
- Task identity is available at test time

**Use LwF when**:
- You cannot store data AND cannot modify the architecture
- Old and new task data come from similar distributions
- You want a distillation-only approach with no extra memory

### Practical Recommendations by Scenario

**Scenario 1: Few tasks (<5), clear task boundaries**
- Start with Experience Replay (simple and effective)
- Add EWC if replay buffer is too small
- Consider MTL if tasks can be trained jointly

**Scenario 2: Many tasks (>20), streaming data**
- Use A-GEM or DER++ (scalable replay)
- Online EWC with Fisher accumulation
- Consider adapter-based approaches with pre-trained backbones

**Scenario 3: Privacy constraints (no data storage)**
- LwF as the primary approach
- EWC or SI for regularization
- Generative Replay if you can afford training a generator

**Scenario 4: Safety-critical (zero forgetting required)**
- Progressive Neural Networks (guaranteed no forgetting)
- PackNet or HAT with frozen weights
- Adapter modules on a frozen backbone

**Scenario 5: Pre-trained foundation model**
- Adapter modules (parameter-efficient, no forgetting in backbone)
- Prompt tuning (learn task-specific prompts)
- LoRA with task-specific adapters

---

## See Also

- **Chapter 31: Transfer Learning** - Pre-training and fine-tuning strategies that
  form the foundation for continual learning with pre-trained models
- **Chapter 32: Meta-Learning** - Learning to learn, which directly connects to
  meta-continual learning approaches (ANML, OML)
- **Chapter 33: Reinforcement Learning** - Continual RL presents unique challenges
  with non-stationary environments and reward structures
- **Chapter 12: Regularization Techniques** - General regularization methods that
  underpin EWC, SI, and weight-based continual learning
- **Chapter 15: Optimization** - Gradient manipulation techniques (gradient projection,
  multi-objective optimization) used in PCGrad and GEM
- **Knowledge Distillation** - The core technique behind LwF and DER approaches

---

## Resources

### Foundational Papers

- Caruana, R. (1997). "Multitask Learning." Machine Learning, 28(1), 41-75.
  The original MTL paper establishing the field.

- Kirkpatrick, J. et al. (2017). "Overcoming catastrophic forgetting in neural
  networks." PNAS. Introduced EWC.

- Kendall, A., Gal, Y., & Cipolla, R. (2018). "Multi-Task Learning Using
  Uncertainty to Weigh Losses for Scene Geometry and Semantics." CVPR.

- Ma, J. et al. (2018). "Modeling Task Relationships in Multi-Task Learning
  with Multi-Gate Mixture-of-Experts." KDD. The MMoE architecture.

- Yu, T. et al. (2020). "Gradient Surgery for Multi-Task Learning." NeurIPS.
  The PCGrad method.

- Lopez-Paz, D. & Ranzato, M. (2017). "Gradient Episodic Memory for Continual
  Learning." NeurIPS. Introduced GEM.

- Buzzega, P. et al. (2020). "Dark Experience for General Continual Learning:
  a Strong, Simple Baseline." NeurIPS. Introduced DER/DER++.

- Rusu, A.A. et al. (2016). "Progressive Neural Networks." arXiv:1606.04671.

- Serra, J. et al. (2018). "Overcoming Catastrophic Forgetting with Hard
  Attention to the Task." ICML. Introduced HAT.

### Survey Papers

- Vandenhende, S. et al. (2021). "Multi-Task Learning for Dense Prediction
  Tasks: A Survey." TPAMI. Comprehensive CV MTL survey.

- De Lange, M. et al. (2021). "A Continual Learning Survey: Defying Forgetting
  in Classification Tasks." TPAMI. Extensive CL benchmark comparison.

- Parisi, G.I. et al. (2019). "Continual Lifelong Learning with Neural Networks:
  A Review." Neural Networks.

### Libraries and Frameworks

- **Avalanche** (https://avalanche.continualai.org/): Comprehensive continual
  learning framework built on PyTorch. Provides benchmarks, strategies, metrics,
  and evaluation protocols.

- **LibMTL** (https://github.com/median-research-group/LibMTL): PyTorch library
  for multi-task learning with implementations of loss balancing and architecture
  methods.

- **Sequoia** (https://github.com/lebrice/Sequoia): Research framework for
  continual learning with standardized settings.

### Courses and Tutorials

- "Continual Learning in Neural Networks" by German Prestes (continualai.org tutorials)
- Stanford CS330: Multi-Task and Meta-Learning (Finn, 2020-present)
- "Multi-Task Learning in NLP" tutorial at ACL conferences

### Benchmarks and Competitions

- ContinualAI benchmark suite (continualai.org)
- CLVISION workshop challenges at CVPR
- CORe50 benchmark (vlomonaco.github.io/core50)
- CLEAR benchmark for realistic continual learning evaluation
