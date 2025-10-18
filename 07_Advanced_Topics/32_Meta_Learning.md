# Meta-Learning: Learning to Learn

Meta-learning (learning to learn) trains models to quickly adapt to new tasks with minimal data by learning across a distribution of tasks. Instead of learning a single task, meta-learning algorithms learn an inductive bias that facilitates rapid learning of new tasks.

## Table of Contents
1. [Fundamentals of Meta-Learning](#fundamentals)
2. [Model-Agnostic Meta-Learning (MAML)](#maml)
3. [MAML Variants](#maml-variants)
4. [Metric-Based Meta-Learning](#metric-based)
5. [Optimization-Based Meta-Learning](#optimization-based)
6. [Applications](#applications)
7. [2025 Trends](#trends-2025)
8. [When to Use Meta-Learning](#when-to-use)

---

## Fundamentals of Meta-Learning {#fundamentals}

### Core Concept: Learning to Learn

**Traditional Learning**: Given dataset D, learn parameters θ* that minimize loss L(θ; D).

**Meta-Learning**: Given task distribution p(T), learn meta-parameters φ such that a model can quickly adapt to new tasks sampled from p(T).

**Intuition**: Learn how to learn efficiently, rather than learning specific tasks.

### Mathematical Framework

**Task Distribution**: p(T) where each task T_i consists of:
- Training set (support set): S_i = {(x_j, y_j)}_{j=1}^K
- Test set (query set): Q_i = {(x_j, y_j)}_{j=1}^M

**Meta-Learning Objective**:

```
φ* = argmin_φ E_{T~p(T)} [ L_T(f_φ(S_T)) ]
```

Where L_T evaluates task-specific model f_φ(S_T) adapted from meta-parameters φ using support set S_T.

### Two-Loop Optimization Structure

Meta-learning typically involves nested optimization loops:

**Inner Loop (Task-Level)**:
- Adapt to specific task using support set
- Fast adaptation with few gradient steps
- Produces task-specific parameters θ_i

**Outer Loop (Meta-Level)**:
- Update meta-parameters across tasks
- Learn adaptation strategy
- Uses query set performance

```
For each iteration:
    Sample batch of tasks {T_i} ~ p(T)
    For each task T_i:
        # Inner loop: adapt to task
        θ_i = adapt(φ, S_i)  # Few-shot learning

        # Evaluate on query set
        L_i = loss(θ_i, Q_i)

    # Outer loop: update meta-parameters
    φ = φ - α * ∇_φ Σ_i L_i
```

### Support Set and Query Set

**Support Set (S)**: Few examples used for adaptation (analogous to training set)
- K-shot learning: K examples per class
- Used in inner loop

**Query Set (Q)**: Examples used for evaluation (analogous to test set)
- Measures how well model adapted
- Used to compute meta-gradient

**Example**: 5-way 1-shot classification
- Support set: 1 example from each of 5 classes (5 examples total)
- Query set: 15 examples from same 5 classes (3 per class)

### Types of Meta-Learning

**1. Metric-Based**: Learn embedding space where similar examples cluster
- Examples: Prototypical Networks, Matching Networks, Siamese Networks
- No gradient updates at test time

**2. Model-Based**: Learn model with internal architecture for rapid learning
- Examples: Memory-Augmented Neural Networks, Meta Networks
- Use recurrent/attention mechanisms

**3. Optimization-Based**: Learn optimization algorithm or initialization
- Examples: MAML, Reptile, Meta-SGD
- Learn how to update parameters efficiently

---

## Model-Agnostic Meta-Learning (MAML) {#maml}

MAML (Finn et al., 2017) is the most influential optimization-based meta-learning algorithm.

### Core Idea

Learn initialization parameters φ such that one or few gradient steps on a new task lead to optimal performance.

**Key Insight**: Good meta-initialization is sensitive to task-specific training, such that small parameter changes lead to large improvements on any task.

### Mathematical Formulation

**Inner Loop (Task Adaptation)**:

For task T_i with support set S_i, perform K gradient steps:

```
θ_i = φ - α * ∇_φ L_{S_i}(φ)
```

Where α is inner loop learning rate (typically small, e.g., 0.01).

For multiple steps:
```
θ_i^(0) = φ
θ_i^(k+1) = θ_i^(k) - α * ∇_{θ_i^(k)} L_{S_i}(θ_i^(k))
```

**Outer Loop (Meta-Update)**:

Update meta-parameters using query set performance:

```
φ = φ - β * ∇_φ Σ_{T_i ~ p(T)} L_{Q_i}(θ_i)
```

Where β is outer loop learning rate (meta-learning rate).

### Complete Derivation

**Objective**: Minimize expected loss on query sets after adaptation:

```
min_φ E_{T~p(T)} [ L_T^{query}(θ_T') ]

where θ_T' = φ - α * ∇_φ L_T^{support}(φ)
```

**Meta-Gradient (Single Inner Step)**:

```
∇_φ L_T^{query}(θ_T') = ∇_φ L_T^{query}(φ - α * ∇_φ L_T^{support}(φ))
```

By chain rule:
```
= ∇_{θ_T'} L_T^{query}(θ_T') * ∂θ_T'/∂φ

= ∇_{θ_T'} L_T^{query}(θ_T') * (I - α * ∂²L_T^{support}/∂φ²)
```

The second-order term ∂²L_T^{support}/∂φ² (Hessian) is expensive to compute!

**Computational Complexity**:
- First-order: O(|θ|) where |θ| is number of parameters
- Second-order: O(|θ|²) due to Hessian

### PyTorch Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

class MAML:
    """
    Model-Agnostic Meta-Learning (MAML).

    Args:
        model: Neural network model
        inner_lr: Learning rate for task adaptation (α)
        outer_lr: Learning rate for meta-update (β)
        num_inner_steps: Number of gradient steps for adaptation
        first_order: If True, use first-order approximation (faster)
    """
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001,
                 num_inner_steps=5, first_order=False):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps
        self.first_order = first_order

        # Meta-optimizer for outer loop
        self.meta_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=outer_lr
        )

    def inner_loop(self, task_data):
        """
        Adapt model to a single task.

        Args:
            task_data: Dict with 'support' and 'query' sets

        Returns:
            adapted_params: Task-specific parameters
            query_loss: Loss on query set
        """
        support_x, support_y = task_data['support']
        query_x, query_y = task_data['query']

        # Clone current parameters for task-specific adaptation
        adapted_params = [p.clone() for p in self.model.parameters()]

        # Inner loop: Adapt to task using support set
        for step in range(self.num_inner_steps):
            # Forward pass with current adapted parameters
            support_logits = self.model.functional_forward(
                support_x, adapted_params
            )

            # Compute support loss
            support_loss = F.cross_entropy(support_logits, support_y)

            # Compute gradients w.r.t. adapted parameters
            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=not self.first_order  # 2nd order if not first_order
            )

            # Gradient descent step
            adapted_params = [
                p - self.inner_lr * g
                for p, g in zip(adapted_params, grads)
            ]

        # Evaluate on query set with adapted parameters
        query_logits = self.model.functional_forward(query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, query_y)

        return adapted_params, query_loss

    def outer_loop(self, task_batch):
        """
        Meta-update across batch of tasks.

        Args:
            task_batch: List of task data dicts

        Returns:
            meta_loss: Average loss across tasks
            meta_acc: Average accuracy across tasks
        """
        meta_loss = 0.0
        meta_acc = 0.0

        # Process each task in batch
        task_losses = []
        for task_data in task_batch:
            # Inner loop adaptation
            adapted_params, query_loss = self.inner_loop(task_data)
            task_losses.append(query_loss)

            # Compute accuracy
            query_x, query_y = task_data['query']
            query_logits = self.model.functional_forward(query_x, adapted_params)
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()
            meta_acc += acc.item()

        # Average loss across tasks
        meta_loss = torch.stack(task_losses).mean()
        meta_acc /= len(task_batch)

        # Meta-gradient step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item(), meta_acc

    def train(self, task_distribution, num_iterations, tasks_per_batch=4):
        """
        Complete MAML training loop.

        Args:
            task_distribution: Iterator yielding tasks
            num_iterations: Number of meta-training iterations
            tasks_per_batch: Number of tasks per meta-update
        """
        for iteration in range(num_iterations):
            # Sample batch of tasks
            task_batch = [next(task_distribution) for _ in range(tasks_per_batch)]

            # Meta-update
            meta_loss, meta_acc = self.outer_loop(task_batch)

            if iteration % 100 == 0:
                print(f"Iteration {iteration}: "
                      f"Meta-Loss = {meta_loss:.4f}, "
                      f"Meta-Acc = {meta_acc:.4f}")

    def evaluate(self, test_tasks, num_adaptation_steps=None):
        """
        Evaluate on test tasks.

        Args:
            test_tasks: List of test task data
            num_adaptation_steps: Override inner steps for evaluation
        """
        if num_adaptation_steps is not None:
            original_steps = self.num_inner_steps
            self.num_inner_steps = num_adaptation_steps

        total_acc = 0.0
        for task_data in test_tasks:
            adapted_params, query_loss = self.inner_loop(task_data)

            query_x, query_y = task_data['query']
            query_logits = self.model.functional_forward(query_x, adapted_params)
            pred = query_logits.argmax(dim=1)
            acc = (pred == query_y).float().mean()
            total_acc += acc.item()

        avg_acc = total_acc / len(test_tasks)

        if num_adaptation_steps is not None:
            self.num_inner_steps = original_steps

        return avg_acc

# Model with functional forward (required for MAML)
class FunctionalModel(nn.Module):
    """
    Neural network with functional forward pass.
    Allows forward pass with arbitrary parameters (not self.parameters()).
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Standard forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def functional_forward(self, x, params):
        """
        Forward pass with external parameters.

        Args:
            x: Input tensor
            params: List of parameters [W1, b1, W2, b2, W3, b3]
        """
        # Manual forward pass with provided parameters
        x = F.linear(x, params[0], params[1])  # fc1
        x = F.relu(x)
        x = F.linear(x, params[2], params[3])  # fc2
        x = F.relu(x)
        x = F.linear(x, params[4], params[5])  # fc3
        return x

# Usage example
def sample_sine_task(amplitude_range=(0.1, 5.0), phase_range=(0, 3.14)):
    """Sample a sinusoidal regression task."""
    amplitude = torch.rand(1) * (amplitude_range[1] - amplitude_range[0]) + amplitude_range[0]
    phase = torch.rand(1) * (phase_range[1] - phase_range[0]) + phase_range[0]

    # Generate support and query sets
    support_x = torch.rand(10, 1) * 10 - 5  # 10 points in [-5, 5]
    support_y = amplitude * torch.sin(support_x + phase)

    query_x = torch.rand(10, 1) * 10 - 5
    query_y = amplitude * torch.sin(query_x + phase)

    return {
        'support': (support_x, support_y),
        'query': (query_x, query_y)
    }

# Training
model = FunctionalModel(input_dim=1, hidden_dim=40, output_dim=1)
maml = MAML(model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5)

task_generator = (sample_sine_task() for _ in range(100000))
maml.train(task_generator, num_iterations=10000, tasks_per_batch=4)
```

### Higher-Order Library (Recommended)

For production use, leverage the `higher` library for efficient MAML implementation:

```python
import higher

class MAMLWithHigher:
    """MAML using higher library for efficient higher-order gradients."""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.num_inner_steps = num_inner_steps
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)

    def meta_train_step(self, task_batch):
        """Single meta-training step."""
        self.meta_optimizer.zero_grad()
        meta_loss = 0.0

        for task_data in task_batch:
            support_x, support_y = task_data['support']
            query_x, query_y = task_data['query']

            # Create differentiable optimizer for inner loop
            with higher.innerloop_ctx(
                self.model,
                self.meta_optimizer,
                copy_initial_weights=False
            ) as (fmodel, diffopt):

                # Inner loop: adapt to task
                for _ in range(self.num_inner_steps):
                    support_logits = fmodel(support_x)
                    support_loss = F.cross_entropy(support_logits, support_y)
                    diffopt.step(support_loss)

                # Evaluate on query set
                query_logits = fmodel(query_x)
                query_loss = F.cross_entropy(query_logits, query_y)
                meta_loss += query_loss

        # Meta-update
        meta_loss /= len(task_batch)
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
```

---

## MAML Variants {#maml-variants}

### MAML++ (Antoniou et al., 2019)

**Improvements over MAML**:
1. **Multi-Step Loss**: Use loss at all inner loop steps, not just final
2. **Per-Parameter Learning Rates**: Learn separate α for each parameter
3. **Batch Normalization**: Transductive batch norm statistics
4. **Cosine Annealing**: Anneal learning rates during inner loop

```python
class MAMLPlusPlus(MAML):
    """MAML++ with improved training."""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001,
                 num_inner_steps=5, multi_step_loss=True):
        super().__init__(model, inner_lr, outer_lr, num_inner_steps)
        self.multi_step_loss = multi_step_loss

        # Learnable per-parameter learning rates
        self.inner_lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

        # Add to meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.inner_lrs),
            lr=outer_lr
        )

    def inner_loop(self, task_data):
        """Inner loop with multi-step loss."""
        support_x, support_y = task_data['support']
        query_x, query_y = task_data['query']

        adapted_params = [p.clone() for p in self.model.parameters()]
        query_losses = []

        # Inner loop with loss at each step
        for step in range(self.num_inner_steps):
            # Support loss
            support_logits = self.model.functional_forward(support_x, adapted_params)
            support_loss = F.cross_entropy(support_logits, support_y)

            # Gradients
            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=True
            )

            # Update with per-parameter learning rates
            adapted_params = [
                p - lr * g
                for p, lr, g in zip(adapted_params, self.inner_lrs, grads)
            ]

            # Query loss at this step (for multi-step loss)
            if self.multi_step_loss:
                query_logits = self.model.functional_forward(query_x, adapted_params)
                query_loss = F.cross_entropy(query_logits, query_y)
                query_losses.append(query_loss)

        # Final query loss
        if not self.multi_step_loss:
            query_logits = self.model.functional_forward(query_x, adapted_params)
            query_loss = F.cross_entropy(query_logits, query_y)
            return adapted_params, query_loss
        else:
            # Average loss across all steps
            return adapted_params, torch.stack(query_losses).mean()
```

### Reptile (Nichol et al., 2018)

**Idea**: First-order approximation of MAML. Simpler and faster.

**Algorithm**:
1. Sample task
2. Perform multiple SGD steps on task
3. Move meta-parameters toward task-specific parameters

**Update Rule**:
```
φ = φ + β * (θ_i - φ)
```

Where θ_i are parameters after K steps on task i.

**Key Difference from MAML**: No second-order derivatives, no query set needed!

```python
class Reptile:
    """Reptile: First-order meta-learning."""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.1, num_inner_steps=5):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.num_inner_steps = num_inner_steps

    def train_step(self, task_batch):
        """Single Reptile meta-update."""
        # Store initial meta-parameters
        meta_params = [p.clone() for p in self.model.parameters()]

        for task_data in task_batch:
            support_x, support_y = task_data['support']

            # Reset to meta-parameters
            for p, meta_p in zip(self.model.parameters(), meta_params):
                p.data.copy_(meta_p.data)

            # SGD on task (no query set needed!)
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)

            for _ in range(self.num_inner_steps):
                optimizer.zero_grad()
                logits = self.model(support_x)
                loss = F.cross_entropy(logits, support_y)
                loss.backward()
                optimizer.step()

            # Interpolate toward task-specific parameters
            with torch.no_grad():
                for meta_p, task_p in zip(meta_params, self.model.parameters()):
                    meta_p.data.add_(
                        self.outer_lr * (task_p.data - meta_p.data)
                    )

        # Update model to new meta-parameters
        for p, meta_p in zip(self.model.parameters(), meta_params):
            p.data.copy_(meta_p.data)
```

**Reptile vs MAML**:
- **Reptile**: First-order, no query set, simpler, faster
- **MAML**: Second-order, requires query set, theoretically better
- **Practice**: Reptile often performs comparably to MAML

### ANIL (Almost No Inner Loop)

**Finding**: In MAML, only the final layer needs to be adapted; early layers can be frozen.

**Benefit**: Faster adaptation, fewer parameters to update.

```python
class ANIL(MAML):
    """ANIL: Only adapt final layer."""
    def inner_loop(self, task_data):
        support_x, support_y = task_data['support']
        query_x, query_y = task_data['query']

        # Extract features with frozen feature extractor
        with torch.no_grad():
            support_features = self.model.feature_extractor(support_x)
            query_features = self.model.feature_extractor(query_x)

        # Only adapt classifier head
        adapted_classifier = [
            p.clone() for p in self.model.classifier.parameters()
        ]

        for _ in range(self.num_inner_steps):
            support_logits = F.linear(
                support_features,
                adapted_classifier[0],
                adapted_classifier[1]
            )
            support_loss = F.cross_entropy(support_logits, support_y)

            grads = torch.autograd.grad(
                support_loss,
                adapted_classifier,
                create_graph=True
            )

            adapted_classifier = [
                p - self.inner_lr * g
                for p, g in zip(adapted_classifier, grads)
            ]

        # Query loss
        query_logits = F.linear(
            query_features,
            adapted_classifier[0],
            adapted_classifier[1]
        )
        query_loss = F.cross_entropy(query_logits, query_y)

        return adapted_classifier, query_loss
```

---

## Metric-Based Meta-Learning {#metric-based}

Metric-based methods learn an embedding space where classification is performed by comparing distances.

### Prototypical Networks (Detailed)

**Algorithm**:
1. Embed all examples: e_i = f_θ(x_i)
2. Compute class prototypes: c_k = (1/|S_k|) ∑_{(x_i,y_i)∈S_k} f_θ(x_i)
3. Classify by nearest prototype in embedding space

**Loss** (negative log-probability based on softmax over distances):

```
L = -log(exp(-d(f_θ(x), c_y)) / ∑_k exp(-d(f_θ(x), c_k)))
```

Where d(.,.) is distance metric (typically squared Euclidean).

```python
class ProtoNet(nn.Module):
    """Prototypical Networks with Euclidean distance."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def compute_prototypes(self, support_embeddings, support_labels, n_way):
        """Compute class prototypes."""
        prototypes = torch.zeros(n_way, support_embeddings.size(1)).to(support_embeddings.device)

        for k in range(n_way):
            class_mask = (support_labels == k)
            prototypes[k] = support_embeddings[class_mask].mean(dim=0)

        return prototypes

    def forward(self, support_x, support_y, query_x, n_way):
        # Embed
        support_embeds = self.encoder(support_x)
        query_embeds = self.encoder(query_x)

        # Prototypes
        prototypes = self.compute_prototypes(support_embeds, support_y, n_way)

        # Squared Euclidean distance
        distances = torch.cdist(query_embeds, prototypes, p=2) ** 2

        # Negative distance as logits
        return -distances

# Training
def train_protonet(model, task_loader, num_episodes):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        # Sample task
        support_x, support_y, query_x, query_y, n_way = next(task_loader)

        # Forward
        logits = model(support_x, support_y, query_x, n_way)

        # Loss
        loss = F.cross_entropy(logits, query_y)

        # Update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 100 == 0:
            acc = (logits.argmax(1) == query_y).float().mean()
            print(f"Episode {episode}: Loss={loss.item():.4f}, Acc={acc.item():.4f}")
```

### Relation Networks

**Idea**: Learn a relation module to compare query and support examples.

**Architecture**:
```
Query → Encoder → e_q
Support → Encoder → e_s
Relation Module(e_q, e_s) → similarity score
```

```python
class RelationNetwork(nn.Module):
    """Relation Networks for few-shot learning."""
    def __init__(self, encoder, relation_module):
        super().__init__()
        self.encoder = encoder
        self.relation_module = relation_module

    def forward(self, support_x, support_y, query_x, n_way, k_shot):
        # Encode
        support_embeds = self.encoder(support_x)  # [n_way*k_shot, d]
        query_embeds = self.encoder(query_x)      # [n_query, d]

        # Compute prototypes
        support_embeds = support_embeds.view(n_way, k_shot, -1).mean(dim=1)  # [n_way, d]

        n_query = query_embeds.size(0)
        embed_dim = query_embeds.size(1)

        # Expand for pairwise comparison
        query_embeds_expanded = query_embeds.unsqueeze(1).expand(-1, n_way, -1)  # [n_query, n_way, d]
        support_embeds_expanded = support_embeds.unsqueeze(0).expand(n_query, -1, -1)  # [n_query, n_way, d]

        # Concatenate query and support embeddings
        pairs = torch.cat([
            query_embeds_expanded,
            support_embeds_expanded
        ], dim=2)  # [n_query, n_way, 2*d]

        # Relation scores
        pairs_flat = pairs.view(-1, 2 * embed_dim)
        relation_scores = self.relation_module(pairs_flat)
        relation_scores = relation_scores.view(n_query, n_way)

        return relation_scores

# Relation module (MLP)
class RelationModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
```

---

## Optimization-Based Meta-Learning {#optimization-based}

### Meta-SGD (Li et al., 2017)

**Idea**: Learn per-parameter learning rates along with initialization.

**Update**:
```
θ_i = φ - α ⊙ ∇_φ L_S(φ)
```

Where α is a vector of learnable learning rates (one per parameter), ⊙ is element-wise multiplication.

```python
class MetaSGD(MAML):
    """Meta-SGD: Learn per-parameter learning rates."""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5):
        super().__init__(model, inner_lr, outer_lr, num_inner_steps)

        # Learnable learning rates (same shape as parameters)
        self.meta_lrs = nn.ParameterList([
            nn.Parameter(torch.ones_like(p) * inner_lr)
            for p in model.parameters()
        ])

        # Meta-optimizer includes meta-lrs
        self.meta_optimizer = torch.optim.Adam(
            list(model.parameters()) + list(self.meta_lrs),
            lr=outer_lr
        )

    def inner_loop(self, task_data):
        support_x, support_y = task_data['support']
        query_x, query_y = task_data['query']

        adapted_params = [p.clone() for p in self.model.parameters()]

        for _ in range(self.num_inner_steps):
            support_logits = self.model.functional_forward(support_x, adapted_params)
            support_loss = F.cross_entropy(support_logits, support_y)

            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=True
            )

            # Update with learnable per-parameter learning rates
            adapted_params = [
                p - lr * g
                for p, lr, g in zip(adapted_params, self.meta_lrs, grads)
            ]

        # Query loss
        query_logits = self.model.functional_forward(query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, query_y)

        return adapted_params, query_loss
```

### Meta-Curvature (Park & Oliva, 2019)

**Idea**: Use second-order curvature information for faster adaptation.

Learns a preconditioner matrix for gradient descent.

```python
class MetaCurvature(MAML):
    """Meta-learning with curvature information."""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, num_inner_steps=5):
        super().__init__(model, inner_lr, outer_lr, num_inner_steps)

        # Learnable preconditioner (low-rank approximation for efficiency)
        self.rank = 10
        param_dims = [p.numel() for p in model.parameters()]
        total_params = sum(param_dims)

        # Low-rank preconditioner: P = I + UV^T
        self.U = nn.Parameter(torch.randn(total_params, self.rank) * 0.01)
        self.V = nn.Parameter(torch.randn(total_params, self.rank) * 0.01)

    def precondition_gradient(self, grads):
        """Apply learned preconditioner to gradients."""
        # Flatten gradients
        flat_grad = torch.cat([g.view(-1) for g in grads])

        # Precondition: (I + UV^T) @ grad
        preconditioned = flat_grad + self.U @ (self.V.T @ flat_grad)

        # Unflatten
        result = []
        offset = 0
        for g in grads:
            numel = g.numel()
            result.append(preconditioned[offset:offset+numel].view_as(g))
            offset += numel

        return result

    def inner_loop(self, task_data):
        support_x, support_y = task_data['support']
        query_x, query_y = task_data['query']

        adapted_params = [p.clone() for p in self.model.parameters()]

        for _ in range(self.num_inner_steps):
            support_logits = self.model.functional_forward(support_x, adapted_params)
            support_loss = F.cross_entropy(support_logits, support_y)

            grads = torch.autograd.grad(
                support_loss,
                adapted_params,
                create_graph=True
            )

            # Precondition gradients with learned curvature
            preconditioned_grads = self.precondition_gradient(grads)

            # Update
            adapted_params = [
                p - self.inner_lr * g
                for p, g in zip(adapted_params, preconditioned_grads)
            ]

        # Query loss
        query_logits = self.model.functional_forward(query_x, adapted_params)
        query_loss = F.cross_entropy(query_logits, query_y)

        return adapted_params, query_loss
```

---

## Applications {#applications}

### Few-Shot Image Classification

Standard benchmark: Omniglot, miniImageNet

```python
# Complete few-shot classification pipeline
class FewShotClassifier:
    """Production few-shot classifier."""
    def __init__(self, backbone='resnet18', method='maml', n_way=5, k_shot=5):
        self.n_way = n_way
        self.k_shot = k_shot

        # Backbone encoder
        if backbone == 'resnet18':
            from torchvision.models import resnet18
            resnet = resnet18(pretrained=True)
            self.encoder = nn.Sequential(*list(resnet.children())[:-1])
            feature_dim = 512
        else:
            self.encoder = ConvEncoder()
            feature_dim = 64

        # Classifier
        self.classifier = nn.Linear(feature_dim, n_way)

        # Meta-learner
        if method == 'maml':
            self.meta_learner = MAML(self, inner_lr=0.01, outer_lr=0.001)
        elif method == 'protonet':
            self.meta_learner = ProtoNet(self.encoder)
        else:
            raise ValueError(f"Unknown method: {method}")

    def extract_features(self, x):
        features = self.encoder(x)
        return features.view(features.size(0), -1)

    def forward(self, x):
        features = self.extract_features(x)
        return self.classifier(features)

    def adapt_and_predict(self, support_x, support_y, query_x):
        """Fast adaptation at test time."""
        if isinstance(self.meta_learner, MAML):
            task_data = {
                'support': (support_x, support_y),
                'query': (query_x, torch.zeros(query_x.size(0)).long())  # Dummy
            }
            adapted_params, _ = self.meta_learner.inner_loop(task_data)

            # Predict with adapted parameters
            query_logits = self.functional_forward(query_x, adapted_params)
            return query_logits

        else:  # ProtoNet
            return self.meta_learner(support_x, support_y, query_x, self.n_way)
```

### Reinforcement Learning (RL²)

**Idea**: Meta-learn RL algorithm that can quickly adapt to new tasks.

**Approach**: Treat RL task as sequence modeling problem. RNN learns to explore and exploit.

```python
class MetaRL:
    """Meta-RL with recurrent policy."""
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        # Recurrent policy network
        self.rnn = nn.GRU(
            input_size=state_dim + action_dim + 1,  # state + prev_action + reward
            hidden_size=hidden_dim,
            num_layers=2
        )
        self.policy_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, prev_action, prev_reward, hidden):
        """
        Forward pass.

        Args:
            state: Current state [batch, state_dim]
            prev_action: Previous action [batch, action_dim]
            prev_reward: Previous reward [batch, 1]
            hidden: RNN hidden state

        Returns:
            action_logits, value, hidden
        """
        # Concatenate inputs
        rnn_input = torch.cat([state, prev_action, prev_reward], dim=-1)
        rnn_input = rnn_input.unsqueeze(0)  # [1, batch, input_dim]

        # RNN forward
        rnn_out, hidden = self.rnn(rnn_input, hidden)
        rnn_out = rnn_out.squeeze(0)  # [batch, hidden_dim]

        # Policy and value
        action_logits = self.policy_head(rnn_out)
        value = self.value_head(rnn_out)

        return action_logits, value, hidden

    def meta_train(self, task_distribution, num_iterations):
        """Meta-training across task distribution."""
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        for iteration in range(num_iterations):
            # Sample batch of tasks
            task_batch = [task_distribution.sample() for _ in range(4)]

            total_loss = 0
            for task in task_batch:
                # Rollout episode with current policy
                states, actions, rewards = self.rollout_episode(task)

                # Compute returns
                returns = self.compute_returns(rewards)

                # Policy gradient loss
                loss = self.compute_pg_loss(states, actions, returns)
                total_loss += loss

            # Meta-update
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
```

### Personalization

**Application**: Personalize models to individual users with their limited data.

```python
class PersonalizedRecommender:
    """Meta-learned personalized recommendation system."""
    def __init__(self, num_items, embedding_dim=64):
        # Shared item embeddings (meta-parameters)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)

        # User-specific parameters (adapted via MAML)
        self.user_encoder = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

        self.maml = MAML(self, inner_lr=0.01, outer_lr=0.001, num_inner_steps=3)

    def forward(self, user_history, candidate_items):
        """
        Predict scores for candidate items.

        Args:
            user_history: List of item IDs user interacted with
            candidate_items: Tensor of candidate item IDs

        Returns:
            scores: Predicted scores for candidates
        """
        # Encode user from history
        history_embeds = self.item_embeddings(user_history)
        user_embed = self.user_encoder(history_embeds.mean(dim=0))

        # Get candidate embeddings
        candidate_embeds = self.item_embeddings(candidate_items)

        # Compute scores (dot product)
        scores = (user_embed.unsqueeze(0) * candidate_embeds).sum(dim=1)
        return scores

    def personalize(self, user_interactions):
        """
        Adapt model to specific user.

        Args:
            user_interactions: List of (item_id, rating) tuples

        Returns:
            personalized_model: User-specific model
        """
        # Prepare task data
        item_ids = torch.tensor([item for item, _ in user_interactions])
        ratings = torch.tensor([rating for _, rating in user_interactions])

        task_data = {
            'support': (item_ids[:10], ratings[:10]),  # First 10 for adaptation
            'query': (item_ids[10:], ratings[10:])     # Rest for evaluation
        }

        # MAML inner loop adaptation
        personalized_params, _ = self.maml.inner_loop(task_data)

        return personalized_params
```

---

## 2025 Trends {#trends-2025}

### Foundation Models Reduce Meta-Learning Need

**Observation**: Large language models (GPT-4, Claude, Gemini) perform few-shot learning through **in-context learning** without parameter updates.

**Mechanism**: Provide examples in prompt, model adapts behavior.

```python
# In-context learning with LLM (no gradient updates!)
prompt = """
Classify the sentiment of movie reviews.

Examples:
Review: "This movie was amazing!" → Positive
Review: "Waste of time and money." → Negative
Review: "Pretty good, but could be better." → Neutral

Now classify:
Review: "Absolutely loved it! Best film this year."
Sentiment:
"""

# Model performs few-shot classification from prompt alone
response = llm.generate(prompt)  # "Positive"
```

**Impact**: For many NLP tasks, in-context learning with LLMs is simpler and more effective than meta-learning.

### Hybrid Approaches: Meta-Learning + Foundation Models

**Idea**: Meta-learn how to construct optimal prompts or examples for foundation models.

```python
class MetaPromptLearner:
    """Meta-learn prompt construction for LLMs."""
    def __init__(self, llm, prompt_encoder):
        self.llm = llm  # Frozen foundation model
        self.prompt_encoder = prompt_encoder  # Learnable

    def construct_prompt(self, task_examples, query):
        """
        Learn to select and order examples for optimal prompting.

        Args:
            task_examples: Pool of example (input, output) pairs
            query: New input to classify

        Returns:
            prompt: Constructed prompt string
        """
        # Encode examples
        example_embeddings = self.prompt_encoder(task_examples)

        # Select most relevant examples (learnable selection)
        relevance_scores = self.compute_relevance(example_embeddings, query)
        top_k_examples = task_examples[relevance_scores.topk(k=5).indices]

        # Construct prompt
        prompt = "Classify the following:\n\n"
        for input_text, output_text in top_k_examples:
            prompt += f"Input: {input_text}\nOutput: {output_text}\n\n"
        prompt += f"Input: {query}\nOutput:"

        return prompt

    def meta_train(self, task_distribution):
        """Meta-learn prompt construction strategy."""
        optimizer = torch.optim.Adam(self.prompt_encoder.parameters(), lr=1e-3)

        for task in task_distribution:
            # Construct prompts for task queries
            prompts = [
                self.construct_prompt(task.support_set, query)
                for query in task.query_set
            ]

            # Get LLM predictions
            predictions = [self.llm.generate(p) for p in prompts]

            # Compute task loss (accuracy)
            loss = self.compute_loss(predictions, task.query_labels)

            # Update prompt encoder
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

### Task-Aware Fine-Tuning

**Idea**: Combine benefits of meta-learning (fast adaptation) with fine-tuning (task-specific optimization).

**Approach**: Meta-learn initialization, then fine-tune on target task.

```python
class TaskAwareFineTuner:
    """Meta-learned initialization + task-specific fine-tuning."""
    def __init__(self, base_model):
        self.base_model = base_model

        # Meta-learn initialization via MAML
        self.maml = MAML(base_model, inner_lr=0.01, outer_lr=0.001)

    def meta_pretrain(self, task_distribution, num_iterations):
        """Meta-learn good initialization."""
        self.maml.train(task_distribution, num_iterations)

    def adapt_to_task(self, task_data, num_fine_tune_steps=100):
        """
        Adapt to new task: MAML initialization + fine-tuning.

        Args:
            task_data: Target task dataset
            num_fine_tune_steps: Number of fine-tuning steps
        """
        # Phase 1: MAML rapid adaptation (5 steps)
        quick_adapt_data = {
            'support': (task_data.train_x[:50], task_data.train_y[:50]),
            'query': (task_data.val_x, task_data.val_y)
        }
        adapted_params, _ = self.maml.inner_loop(quick_adapt_data)

        # Load adapted parameters
        for p, adapted_p in zip(self.base_model.parameters(), adapted_params):
            p.data.copy_(adapted_p.data)

        # Phase 2: Fine-tune on full task dataset
        optimizer = torch.optim.Adam(self.base_model.parameters(), lr=1e-4)

        for step in range(num_fine_tune_steps):
            batch_x, batch_y = task_data.sample_batch()
            logits = self.base_model(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.base_model
```

### Meta-Learning for Hyperparameter Optimization

**Application**: Learn hyperparameters (learning rates, architectures) that transfer across tasks.

```python
class MetaHyperparameterOptimizer:
    """Meta-learn hyperparameters."""
    def __init__(self):
        # Learnable hyperparameters
        self.learning_rate = nn.Parameter(torch.tensor(1e-3))
        self.weight_decay = nn.Parameter(torch.tensor(1e-4))
        self.dropout_rate = nn.Parameter(torch.tensor(0.5))

    def create_model(self, task):
        """Create model with meta-learned hyperparameters."""
        model = TaskSpecificModel(dropout=self.dropout_rate.item())
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.learning_rate.item(),
            weight_decay=self.weight_decay.item()
        )
        return model, optimizer

    def meta_train(self, task_distribution):
        """Meta-learn hyperparameters across tasks."""
        meta_optimizer = torch.optim.Adam([
            self.learning_rate,
            self.weight_decay,
            self.dropout_rate
        ], lr=1e-4)

        for task in task_distribution:
            # Create model with current hyperparameters
            model, optimizer = self.create_model(task)

            # Train on task
            for _ in range(100):
                loss = train_step(model, optimizer, task.train_data)

            # Evaluate on validation
            val_loss = evaluate(model, task.val_data)

            # Meta-update hyperparameters
            meta_optimizer.zero_grad()
            val_loss.backward()
            meta_optimizer.step()

            # Project to valid ranges
            with torch.no_grad():
                self.learning_rate.clamp_(1e-5, 1e-1)
                self.weight_decay.clamp_(0, 1e-2)
                self.dropout_rate.clamp_(0, 0.9)
```

---

## When to Use Meta-Learning {#when-to-use}

### Use Meta-Learning When:

1. **Many Related Tasks**: You have access to distribution of related tasks
   - Example: Personalization across thousands of users
   - Example: Medical diagnosis across multiple diseases

2. **Few Examples Per Task**: Each task has limited labeled data
   - Typical: 1-20 examples per class
   - Few-shot learning scenarios

3. **Rapid Adaptation Required**: Need to quickly adapt to new tasks
   - Example: Robotics adapting to new environments
   - Example: Recommendation system for new users

4. **Cannot Use Foundation Models**: Domain-specific tasks where LLMs don't apply
   - Example: Control systems, robotics
   - Example: Specialized computer vision (medical, satellite)

5. **Personalization at Scale**: Customize model for many users/contexts
   - Example: Personal assistants
   - Example: Adaptive user interfaces

### Don't Use Meta-Learning When:

1. **Sufficient Data Available**: If you have 10K+ examples per task, standard learning works better

2. **Single Task**: No task distribution available for meta-training

3. **LLM In-Context Learning Sufficient**: For NLP tasks, GPT-4/Claude with examples often better

4. **Computational Constraints**: Meta-learning requires training across many tasks (expensive)

5. **Transfer Learning Sufficient**: If pretrained model + fine-tuning works well

---

## Comparison: Meta-Learning vs Alternatives

| Approach | Data Requirement | Adaptation Speed | When to Use |
|----------|-----------------|------------------|-------------|
| **Standard Learning** | 10K+ examples | N/A | Plenty of data |
| **Transfer Learning** | 1K+ examples | Medium (fine-tuning) | Pretrained model available |
| **Meta-Learning (MAML)** | 5-50 examples | Fast (few steps) | Many related tasks |
| **Few-Shot (ProtoNet)** | 1-20 examples | Very fast (no gradients) | Metric learning applicable |
| **In-Context Learning (LLM)** | 1-10 examples | Instant (no updates) | NLP tasks, LLM available |

---

## Best Practices (2025)

### Algorithm Selection
- **MAML**: When second-order gradients feasible and query sets available
- **Reptile**: When computational efficiency critical
- **ProtoNet**: For classification with good embedding space
- **MAML++**: State-of-the-art for most vision tasks

### Implementation
- Use `higher` library for efficient MAML implementation
- Leverage pretrained backbones (ImageNet for vision)
- Monitor both support and query performance
- Use appropriate train/val/test task splits

### Hyperparameters
- **Inner LR** (α): 0.001 - 0.01 (smaller for pretrained models)
- **Outer LR** (β): 0.0001 - 0.001
- **Inner Steps**: 1-10 (more steps = better adaptation but slower)
- **Tasks per batch**: 2-8 (limited by memory)

### Debugging
- First verify on simple task (sinusoid regression)
- Check that inner loop adaptation improves task performance
- Ensure query loss decreases across meta-training
- Visualize learned representations (t-SNE, UMAP)

---

## Summary

Meta-learning enables rapid adaptation to new tasks by learning across task distributions:

- **MAML**: Learn initialization for fast gradient-based adaptation
- **Metric-based**: Learn embedding space for non-parametric classification
- **Optimization-based**: Learn learning algorithms or hyperparameters

**2025 Context**: While foundation models with in-context learning reduce meta-learning need for NLP, meta-learning remains crucial for:
- Specialized domains (robotics, control, medical imaging)
- Personalization at scale
- Tasks requiring parameter updates (not just prompting)
- Learning with limited compute (can't afford LLM inference)

**Key Insight**: Meta-learning teaches models how to learn efficiently, enabling practical few-shot learning across vision, RL, and personalization domains.
