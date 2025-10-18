# Optimization in Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Gradient Descent Fundamentals](#gradient-descent-fundamentals)
3. [Momentum-Based Optimizers](#momentum-based-optimizers)
4. [Adaptive Learning Rate Optimizers](#adaptive-learning-rate-optimizers)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Gradient Clipping](#gradient-clipping)
7. [Optimization Landscape](#optimization-landscape)
8. [Optimizer Selection Guide](#optimizer-selection-guide)
9. [Complete Implementations](#complete-implementations)
10. [2025 Best Practices](#2025-best-practices)

---

## Introduction

Optimization is the process of finding parameters θ* that minimize a loss function:

```
θ* = argmin_θ L(θ)
```

In deep learning, this is challenging because:
- **Non-convex:** Multiple local minima and saddle points
- **High-dimensional:** Millions to billions of parameters
- **Stochastic:** Training on mini-batches introduces noise
- **Ill-conditioned:** Different parameters have different scales

**Goal:** Find good parameters efficiently, not necessarily the global minimum.

---

## Gradient Descent Fundamentals

### The Core Algorithm

**Gradient descent update rule:**
```
θ_{t+1} = θ_t - α ∇L(θ_t)
```

Where:
- **θ_t:** Parameters at iteration t
- **α:** Learning rate (step size)
- **∇L(θ_t):** Gradient of loss with respect to parameters

**Intuition:** Move in the direction opposite to the gradient (steepest descent).

### Mathematical Foundation

**First-order Taylor expansion:**
```
L(θ + Δθ) ≈ L(θ) + ∇L(θ)^T Δθ
```

To minimize L(θ + Δθ), choose:
```
Δθ = -α ∇L(θ)
```

This gives:
```
L(θ - α∇L(θ)) ≈ L(θ) - α ||∇L(θ)||^2
```

For small α, this guarantees loss decrease.

### Batch Gradient Descent

**Full batch update:**
```
θ_{t+1} = θ_t - α (1/N) Σ_{i=1}^N ∇L_i(θ_t)
```

Where:
- **N:** Total number of training examples
- **L_i:** Loss on example i

**Advantages:**
- Exact gradient computation
- Stable convergence
- Theoretical guarantees

**Disadvantages:**
- Computationally expensive (requires full dataset pass)
- Memory intensive (all data in memory)
- Slow updates (one update per epoch)
- Can get stuck in local minima

**When to use:** Small datasets that fit in memory, when exact gradients are needed.

### Stochastic Gradient Descent (SGD)

**Single example update:**
```
θ_{t+1} = θ_t - α ∇L_i(θ_t)
```

Where i is randomly sampled from {1, ..., N}.

**Properties:**
- **Unbiased:** E[∇L_i(θ)] = ∇L(θ)
- **High variance:** Individual gradients are noisy
- **Exploration:** Noise helps escape local minima

**Advantages:**
- Fast updates (one per example)
- Can escape local minima (noise acts as exploration)
- Online learning possible
- Lower memory requirements

**Disadvantages:**
- High variance in updates
- Slower convergence
- Requires careful learning rate tuning

### Mini-Batch Gradient Descent

**Mini-batch update (modern standard):**
```
θ_{t+1} = θ_t - α (1/B) Σ_{i∈B_t} ∇L_i(θ_t)
```

Where:
- **B_t:** Mini-batch of size B at iteration t
- **B:** Batch size (typically 32, 64, 128, 256)

**Advantages:**
- Balances computational efficiency and gradient accuracy
- Vectorization on GPUs (parallel computation)
- Stable convergence with reasonable variance
- Memory efficient

**Disadvantages:**
- Still noisy (but less than SGD)
- Requires batch size tuning

**Batch Size Selection:**

**Small batches (32-64):**
- More noise → better exploration
- Better generalization (implicit regularization)
- More frequent updates
- Lower memory usage

**Large batches (256-1024):**
- Less noise → faster convergence
- Better hardware utilization (GPUs)
- Fewer updates per epoch
- May generalize worse (sharp minima)

**2025 Best Practice:** Start with 32-64, increase if GPU underutilized, use gradient accumulation for effective large batches.

---

## Momentum-Based Optimizers

### SGD with Momentum

**Core idea:** Accumulate past gradients to build velocity, smoothing out oscillations.

**Update equations:**
```
v_t = β v_{t-1} + ∇L(θ_t)
θ_{t+1} = θ_t - α v_t
```

Where:
- **v_t:** Velocity (exponentially weighted average of gradients)
- **β:** Momentum coefficient (typically 0.9)

**Physical analogy:** A ball rolling down a hill accumulates momentum.

**Mathematical derivation:**

Expanding the recursion:
```
v_t = ∇L(θ_t) + β∇L(θ_{t-1}) + β^2∇L(θ_{t-2}) + ...
    = Σ_{i=0}^t β^i ∇L(θ_{t-i})
```

This is an exponentially weighted moving average with decay β.

**Properties:**
- **Accelerates convergence** in relevant directions
- **Dampens oscillations** in high-curvature directions
- **Passes through small local minima** due to velocity
- **Effective learning rate:** α/(1-β) for constant gradients

**Advantages:**
- Faster convergence than vanilla SGD
- Reduces oscillations
- Works well in practice

**Disadvantages:**
- Introduces hyperparameter β
- Can overshoot minima

**Optimal momentum value:**

**Theorem (Polyak, 1964):** For quadratic loss, optimal β for condition number κ is:
```
β* = (√κ - 1) / (√κ + 1)
```

In practice, β = 0.9 works well for most problems.

### Nesterov Accelerated Gradient (NAG)

**Core idea:** "Look ahead" before computing gradient.

**Update equations:**
```
v_t = β v_{t-1} + ∇L(θ_t - β v_{t-1})  # Lookahead gradient
θ_{t+1} = θ_t - α v_t
```

**Intuition:**
1. Make a "trial" move: θ_t - β v_{t-1}
2. Compute gradient at trial position
3. Update velocity based on lookahead gradient
4. Update parameters

**Advantages over standard momentum:**
- Better convergence rate: O(1/t^2) vs O(1/t) for strongly convex functions
- More responsive to gradient changes
- Reduces overshooting

**PyTorch implementation:**

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    nesterov=True  # Enables NAG
)
```

---

## Adaptive Learning Rate Optimizers

### AdaGrad (Adaptive Gradient)

**Core idea:** Adapt learning rate per parameter based on historical gradients.

**Update equations:**
```
g_t = ∇L(θ_t)
G_t = G_{t-1} + g_t ⊙ g_t  # Accumulate squared gradients
θ_{t+1} = θ_t - α / (√G_t + ε) ⊙ g_t
```

Where:
- **G_t:** Accumulated squared gradients (element-wise)
- **ε:** Small constant for numerical stability (typically 1e-8)
- **⊙:** Element-wise multiplication

**Properties:**
- **Large gradients:** Learning rate decreases (G_t large)
- **Small gradients:** Learning rate increases (G_t small)
- **Per-parameter adaptation:** Each parameter has its own effective learning rate

**Advantages:**
- No manual learning rate tuning
- Works well for sparse features (NLP, recommender systems)
- Guarantees for convex optimization

**Disadvantages:**
- Learning rate monotonically decreases
- Can stop learning too early (G_t grows without bound)
- Not recommended for deep learning (2025)

**When to use:** Sparse data (word embeddings), convex problems.

### RMSprop (Root Mean Square Propagation)

**Core idea:** Fix AdaGrad's aggressive learning rate decay using exponential moving average.

**Update equations:**
```
g_t = ∇L(θ_t)
E[g^2]_t = β E[g^2]_{t-1} + (1-β) g_t ⊙ g_t
θ_{t+1} = θ_t - α / (√E[g^2]_t + ε) ⊙ g_t
```

Where:
- **E[g^2]_t:** Exponential moving average of squared gradients
- **β:** Decay rate (typically 0.9 or 0.99)

**Key difference from AdaGrad:** Uses moving average instead of cumulative sum.

**Advantages:**
- Resolves AdaGrad's learning rate decay problem
- Works well for non-stationary objectives (RNNs)
- Suitable for online learning

**Disadvantages:**
- Still has hyperparameters (α, β)
- No momentum component
- Can be unstable with large gradients

**Historical note:** Proposed by Geoff Hinton in a Coursera lecture (not formally published).

### Adam (Adaptive Moment Estimation)

**Core idea:** Combine momentum (first moment) and RMSprop (second moment).

**Update equations:**
```
g_t = ∇L(θ_t)
m_t = β_1 m_{t-1} + (1-β_1) g_t              # First moment (mean)
v_t = β_2 v_{t-1} + (1-β_2) g_t ⊙ g_t        # Second moment (variance)

# Bias correction
m_hat_t = m_t / (1 - β_1^t)
v_hat_t = v_t / (1 - β_2^t)

# Update
θ_{t+1} = θ_t - α m_hat_t / (√v_hat_t + ε)
```

Where:
- **m_t:** First moment estimate (momentum)
- **v_t:** Second moment estimate (RMSprop)
- **β_1, β_2:** Decay rates (defaults: 0.9, 0.999)
- **α:** Learning rate (default: 0.001)
- **ε:** Numerical stability (default: 1e-8)

**Bias correction:** Needed because m_t and v_t are initialized at 0, leading to bias toward zero in early iterations.

**Mathematical justification:**

Without bias correction, initial estimates are biased:
```
E[m_1] = (1-β_1) E[g_1] ≠ E[g_1]  # Biased toward 0
```

With bias correction:
```
E[m_hat_1] = E[m_1] / (1-β_1) = E[g_1]  # Unbiased
```

**Advantages:**
- Combines benefits of momentum and adaptive learning rates
- Works well across a wide range of problems
- Default choice for many practitioners
- Relatively robust to hyperparameter choices

**Disadvantages:**
- Can converge to suboptimal solutions (sharp minima)
- Weight decay not properly implemented (fixed in AdamW)
- Many hyperparameters

**When to use:** General-purpose default optimizer (2025).

### AdamW (Adam with Decoupled Weight Decay)

**Core idea:** Fix Adam's weight decay implementation.

**The problem with Adam + L2 regularization:**

In vanilla Adam with L2 regularization:
```
L_total = L(θ) + (λ/2) ||θ||^2
∇L_total = ∇L(θ) + λθ

# Adam applies adaptive learning rate to weight decay term
θ_{t+1} = θ_t - α m_hat_t / (√v_hat_t + ε)  # where m_hat includes λθ
```

This is **NOT equivalent** to proper L2 regularization because the adaptive learning rate scales the weight decay.

**AdamW solution (decoupled weight decay):**
```
g_t = ∇L(θ_t)
m_t = β_1 m_{t-1} + (1-β_1) g_t
v_t = β_2 v_{t-1} + (1-β_2) g_t ⊙ g_t

m_hat_t = m_t / (1 - β_1^t)
v_hat_t = v_t / (1 - β_2^t)

# Update with DECOUPLED weight decay
θ_{t+1} = θ_t - α m_hat_t / (√v_hat_t + ε) - α λ θ_t
                                               ^^^^^^^^
                                            Weight decay applied
                                            separately
```

**Key difference:** Weight decay is applied directly to parameters, not included in gradient.

**Mathematical comparison:**

**Adam + L2 reg:** Effective weight decay varies per parameter based on gradient history
**AdamW:** Uniform weight decay independent of gradients

**Empirical results:** AdamW consistently outperforms Adam, especially for transformers.

**2025 Best Practice:** Always use AdamW instead of Adam.

### Other Variants

**AdaMax:**
```
v_t = max(β_2 v_{t-1}, |g_t|)  # L∞ norm instead of L2
θ_{t+1} = θ_t - α m_hat_t / v_t
```

**Advantages:** More stable to large gradients, less sensitive to β_2.

**Nadam (Nesterov + Adam):**
```
Combines Nesterov momentum with Adam's adaptive learning rates
```

**When to use:** Rarely better than AdamW in practice.

**RAdam (Rectified Adam):**
```
Adaptive warm-up for Adam to fix early training instability
```

**When to use:** Training very deep networks from scratch.

---

## Learning Rate Scheduling

### Why Schedule Learning Rates?

**Early training:** Need large learning rates to make progress
**Late training:** Need small learning rates to fine-tune and converge

**Benefits:**
- Faster convergence
- Better final performance
- Escape plateaus
- Avoid overshooting

### Step Decay

**Definition:**
```
α_t = α_0 * γ^(⌊t/k⌋)
```

Where:
- **α_0:** Initial learning rate
- **γ:** Decay factor (typically 0.1 or 0.5)
- **k:** Step size (epochs between decays)

**Example:** α_0=0.1, γ=0.1, k=30
- Epochs 0-29: α = 0.1
- Epochs 30-59: α = 0.01
- Epochs 60-89: α = 0.001

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=30,
    gamma=0.1
)
```

**Advantages:**
- Simple to implement
- Predictable behavior
- Works well for many tasks

**Disadvantages:**
- Requires tuning step size
- Abrupt changes in learning rate
- Not adaptive to training dynamics

### Exponential Decay

**Definition:**
```
α_t = α_0 * e^(-λt)
```

Or in discrete form:
```
α_t = α_0 * γ^t
```

Where γ = e^(-λ) ≈ 0.95 - 0.99.

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=0.95
)
```

**Advantages:**
- Smooth decay
- No hyperparameter for step size

**Disadvantages:**
- Can decay too quickly or slowly
- Still requires tuning decay rate

### Cosine Annealing

**Definition:**
```
α_t = α_min + (α_max - α_min) * (1 + cos(πt/T)) / 2
```

Where:
- **T:** Total number of iterations/epochs
- **α_min:** Minimum learning rate (often 0)
- **α_max:** Maximum learning rate

**Shape:** Smooth cosine curve from α_max to α_min.

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=epochs,
    eta_min=0
)
```

**Advantages:**
- Smooth, gradual decay
- No abrupt changes
- Widely used in modern architectures (ResNets, Transformers)
- Works well empirically

**Disadvantages:**
- Requires knowing total training time
- Learning rate never truly reaches zero (in finite time)

**2025 Best Practice:** Cosine annealing is the default choice for many modern models.

### Cosine Annealing with Warm Restarts (SGDR)

**Definition:**
```
α_t = α_min + (α_max - α_min) * (1 + cos(πt_i/T_i)) / 2
```

Where:
- **t_i:** Iterations since last restart
- **T_i:** Iterations until next restart

**Warm restart:** Periodically reset learning rate to α_max.

**Schedule:**
```
T_i = T_0 * T_mult^i

Example (T_0=10, T_mult=2):
Restart at epochs: 0, 10, 30, 70, 150, ...
```

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=10,        # First restart after 10 epochs
    T_mult=2,      # Double restart period each time
    eta_min=0
)
```

**Advantages:**
- Escapes local minima via restarts
- Explores multiple solutions
- Can improve generalization

**Disadvantages:**
- More hyperparameters
- Can be unstable
- Not always better than simple cosine annealing

### One-Cycle Policy

**Core idea:** Gradually increase learning rate, then decrease (triangular schedule).

**Schedule:**
1. **Warm-up phase (first 30% of training):** Increase LR from α_min to α_max
2. **Annealing phase (remaining 70%):** Decrease LR from α_max to α_min

**Additional features:**
- Momentum inversely related to learning rate
- Final learning rate much lower than initial

**Mathematical form:**
```
# Warm-up phase (t < pct_start * T)
α_t = α_min + (α_max - α_min) * (t / (pct_start * T))

# Annealing phase (t ≥ pct_start * T)
α_t = α_max - (α_max - α_min) * ((t - pct_start*T) / ((1-pct_start)*T))
```

**PyTorch implementation:**
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.1,
    epochs=100,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warm-up
    anneal_strategy='cos'
)
```

**Advantages:**
- Very fast convergence
- Often achieves best results
- Regularization effect from high learning rates

**Disadvantages:**
- Sensitive to max_lr choice
- Requires careful tuning
- Not suitable for all tasks

**When to use:** Time-constrained training, when you can tune hyperparameters.

### Warm-Up Strategies

**Problem:** Large learning rates at initialization can cause instability.

**Solution:** Gradually increase learning rate from near-zero to target value.

**Linear warm-up:**
```
α_t = α_target * min(1, t / T_warmup)
```

**Typical warm-up duration:** 1000-10000 steps, or 1-5 epochs.

**Why warm-up helps:**
1. Parameters initialized randomly, large updates can be harmful
2. Batch normalization statistics not yet stable
3. Adaptive optimizers (Adam) need time to estimate moments

**PyTorch implementation:**
```python
def warmup_lr_scheduler(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
```

**2025 Best Practice:** Always use warm-up for transformers and large models.

### Learning Rate Finder

**Idea:** Automatically find good learning rate before training.

**Algorithm (Smith, 2015):**
1. Start with very small learning rate (e.g., 1e-7)
2. Train for a few hundred iterations
3. Exponentially increase learning rate after each batch
4. Plot loss vs learning rate
5. Select learning rate where loss decreases fastest

**Implementation:**
```python
def find_lr(model, train_loader, optimizer, criterion, device, start_lr=1e-7, end_lr=10, num_iter=100):
    """
    Learning rate range test.

    Returns:
        lrs: List of learning rates tested
        losses: Corresponding losses
    """
    model.train()

    lrs = []
    losses = []

    # Exponential growth rate
    mult = (end_lr / start_lr) ** (1 / num_iter)
    lr = start_lr

    optimizer.param_groups[0]['lr'] = lr

    avg_loss = 0
    best_loss = float('inf')
    batch_num = 0

    iterator = iter(train_loader)

    for iteration in range(num_iter):
        batch_num += 1

        try:
            data, target = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            data, target = next(iterator)

        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)

        # Compute smoothed loss
        avg_loss = 0.9 * avg_loss + 0.1 * loss.item()
        smoothed_loss = avg_loss / (1 - 0.9 ** batch_num)

        # Stop if loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            break

        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss

        # Record
        lrs.append(lr)
        losses.append(smoothed_loss)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update learning rate
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr

    return lrs, losses

# Usage and plotting
lrs, losses = find_lr(model, train_loader, optimizer, criterion, device)

plt.figure(figsize=(10, 6))
plt.plot(lrs, losses)
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.title('Learning Rate Finder')
plt.grid(True)
plt.show()

# Select LR where loss decreases fastest (steepest negative slope)
# Typically 1-2 orders of magnitude before minimum
```

**Interpretation:**
- **Too small:** Loss decreases slowly
- **Good range:** Loss decreases rapidly (steepest descent)
- **Too large:** Loss increases or becomes unstable

**Best learning rate:** Typically where slope is steepest, NOT where loss is minimum.

---

## Gradient Clipping

### Why Gradient Clipping?

**Problem:** Exploding gradients (especially in RNNs, very deep networks).

**Symptoms:**
- Loss becomes NaN
- Weights become NaN or Inf
- Unstable training

**Solution:** Limit gradient magnitude.

### Gradient Clipping by Value

**Definition:**
```
g_clipped = max(min(g, threshold), -threshold)
```

**Applies element-wise clipping.**

**PyTorch implementation:**
```python
torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
```

**Disadvantages:**
- Changes gradient direction
- Threshold is arbitrary and problem-dependent

### Gradient Clipping by Norm (Recommended)

**Definition:**
```
g_clipped = g * min(1, threshold / ||g||)
```

Where ||g|| is the L2 norm of the gradient vector.

**Properties:**
- Preserves gradient direction
- Only scales magnitude if too large
- More theoretically justified

**Derivation:**
```
If ||g|| ≤ threshold: g_clipped = g
If ||g|| > threshold: g_clipped = g * (threshold / ||g||)
                      → ||g_clipped|| = threshold
```

**PyTorch implementation:**
```python
# Compute gradients
loss.backward()

# Clip gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Update parameters
optimizer.step()
```

**Typical threshold values:**
- **MLPs, CNNs:** 1.0 - 5.0
- **RNNs, LSTMs:** 0.5 - 1.0
- **Transformers:** 1.0

**2025 Best Practice:** Use gradient clipping by norm with max_norm=1.0 as default.

### Adaptive Clipping

**Idea:** Adjust clipping threshold based on gradient statistics.

**Example:** Clip to 99th percentile of historical gradient norms.

```python
class AdaptiveGradientClipper:
    def __init__(self, percentile=99, window_size=1000):
        self.percentile = percentile
        self.grad_norms = []
        self.window_size = window_size

    def clip(self, parameters):
        # Compute current gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, float('inf'))

        # Update history
        self.grad_norms.append(total_norm.item())
        if len(self.grad_norms) > self.window_size:
            self.grad_norms.pop(0)

        # Compute adaptive threshold
        if len(self.grad_norms) >= 100:
            threshold = np.percentile(self.grad_norms, self.percentile)
            torch.nn.utils.clip_grad_norm_(parameters, threshold)

        return total_norm
```

---

## Optimization Landscape

### Challenges in Deep Learning Optimization

**1. Non-convexity:**
- Multiple local minima
- Saddle points
- Flat regions (plateaus)

**2. High dimensionality:**
- Curse of dimensionality
- Many saddle points (exponential in dimension)

**3. Ill-conditioning:**
- Different parameters have different scales
- Hessian eigenvalues vary widely (high condition number)

**4. Stochasticity:**
- Mini-batch noise
- Helps escape sharp minima (implicit regularization)

### Loss Surface Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_loss_surface_2d(model, data, target, criterion, param1_name, param2_name,
                         param1_range=(-1, 1), param2_range=(-1, 1), steps=50):
    """
    Visualize loss surface by varying two parameters.

    WARNING: Only feasible for small models / visualization purposes.
    """
    model.eval()

    # Get original parameters
    original_params = {}
    for name, param in model.named_parameters():
        original_params[name] = param.data.clone()

    # Create grid
    p1_values = np.linspace(param1_range[0], param1_range[1], steps)
    p2_values = np.linspace(param2_range[0], param2_range[1], steps)
    P1, P2 = np.meshgrid(p1_values, p2_values)

    losses = np.zeros((steps, steps))

    # Compute loss for each point in grid
    for i in range(steps):
        for j in range(steps):
            # Set parameters
            for name, param in model.named_parameters():
                if name == param1_name:
                    param.data = original_params[name] + p1_values[i]
                elif name == param2_name:
                    param.data = original_params[name] + p2_values[j]
                else:
                    param.data = original_params[name]

            # Compute loss
            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                losses[i, j] = loss.item()

    # Restore original parameters
    for name, param in model.named_parameters():
        param.data = original_params[name]

    # Plot
    fig = plt.figure(figsize=(12, 5))

    # 3D surface
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(P1, P2, losses, cmap='viridis', alpha=0.8)
    ax1.set_xlabel(param1_name)
    ax1.set_ylabel(param2_name)
    ax1.set_zlabel('Loss')
    ax1.set_title('Loss Surface (3D)')

    # 2D contour
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(P1, P2, losses, levels=20, cmap='viridis')
    ax2.set_xlabel(param1_name)
    ax2.set_ylabel(param2_name)
    ax2.set_title('Loss Surface (Contour)')
    plt.colorbar(contour, ax=ax2)

    plt.tight_layout()
    plt.show()
```

### Sharp vs Flat Minima

**Sharp minimum:**
```
High curvature → Small perturbations increase loss significantly
→ Poor generalization
```

**Flat minimum:**
```
Low curvature → Perturbations don't increase loss much
→ Better generalization
```

**Relationship to batch size:**
- **Small batches:** Noisy gradients → Find flat minima → Better generalization
- **Large batches:** Accurate gradients → Find sharp minima → Worse generalization

**Solutions for large batch training:**
1. Learning rate scaling: α_large = α_small * (B_large / B_small)
2. Warm-up
3. Ghost batch normalization
4. Layer-wise adaptive learning rates

---

## Optimizer Selection Guide

### Decision Tree (2025)

```
Task Type?
│
├─ General Deep Learning (CNNs, MLPs)
│  └─ Use AdamW (lr=1e-3, weight_decay=1e-4)
│     ├─ With cosine annealing
│     └─ With warm-up for deep models
│
├─ Transformers / NLP
│  └─ Use AdamW (lr=1e-4 to 5e-5, weight_decay=0.01)
│     ├─ With linear warm-up (5-10% of steps)
│     └─ With cosine or linear decay
│
├─ Computer Vision (from scratch)
│  └─ Use SGD with momentum (lr=0.1, momentum=0.9, weight_decay=1e-4)
│     └─ With step decay or cosine annealing
│
├─ Fine-tuning Pre-trained Models
│  └─ Use AdamW with very small LR (1e-5 to 1e-6)
│     └─ Optional: Discriminative learning rates (different LR per layer)
│
└─ Reinforcement Learning
   └─ Use Adam or RMSprop (depending on algorithm)
```

### Hyperparameter Recommendations

| Optimizer | Learning Rate | Weight Decay | Other Parameters |
|-----------|---------------|--------------|------------------|
| **SGD** | 0.01 - 0.1 | 1e-4 | momentum=0.9 |
| **SGD (with momentum)** | 0.01 - 0.1 | 1e-4 | momentum=0.9, nesterov=True |
| **Adam** | 1e-3 - 1e-4 | 0 (or use AdamW) | β1=0.9, β2=0.999, ε=1e-8 |
| **AdamW** | 1e-3 - 1e-4 | 1e-4 - 1e-2 | β1=0.9, β2=0.999, ε=1e-8 |
| **RMSprop** | 1e-3 - 1e-4 | 0 | β=0.9 |

### When to Use Each Optimizer

**SGD with Momentum:**
- Training CNNs from scratch (ResNets, VGGs)
- When computational resources are limited
- When you have time to tune learning rate schedule
- Empirically can achieve best final performance (but requires tuning)

**AdamW:**
- Default choice for most tasks (2025)
- Transformers and NLP models
- Fine-tuning pre-trained models
- When you want good results without extensive tuning
- Faster convergence than SGD

**RMSprop:**
- Recurrent neural networks (RNNs, LSTMs)
- Reinforcement learning (though Adam often works too)
- Online learning / non-stationary objectives

**AdaGrad:**
- Sparse data (word embeddings, recommender systems)
- When features have very different frequencies

---

## Complete Implementations

### Optimizer from Scratch

```python
import torch
import numpy as np
from typing import List, Dict, Callable

class Optimizer:
    """Base class for optimizers."""

    def __init__(self, parameters, lr: float):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        """Zero out gradients."""
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        """Update parameters. Must be implemented by subclasses."""
        raise NotImplementedError


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with optional momentum.

    Args:
        parameters: Model parameters
        lr: Learning rate
        momentum: Momentum factor (0 for vanilla SGD)
        weight_decay: L2 regularization coefficient
        nesterov: Enable Nesterov momentum
    """

    def __init__(self, parameters, lr: float = 0.01, momentum: float = 0,
                 weight_decay: float = 0, nesterov: bool = False):
        super().__init__(parameters, lr)
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        # Initialize velocity
        self.velocity = [torch.zeros_like(p.data) for p in self.parameters]

    def step(self):
        """Perform single optimization step."""
        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Weight decay (L2 regularization)
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Momentum
            if self.momentum != 0:
                velocity = self.velocity[i]
                velocity.mul_(self.momentum).add_(grad)

                if self.nesterov:
                    # Nesterov momentum: gradient + momentum
                    grad = grad.add(velocity, alpha=self.momentum)
                else:
                    # Standard momentum
                    grad = velocity

            # Update parameters
            param.data.add_(grad, alpha=-self.lr)


class Adam(Optimizer):
    """
    Adam optimizer.

    Args:
        parameters: Model parameters
        lr: Learning rate
        betas: Coefficients for computing running averages (β1, β2)
        eps: Term for numerical stability
        weight_decay: L2 regularization (NOT recommended, use AdamW instead)
    """

    def __init__(self, parameters, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moments
        self.m = [torch.zeros_like(p.data) for p in self.parameters]  # First moment
        self.v = [torch.zeros_like(p.data) for p in self.parameters]  # Second moment
        self.t = 0  # Time step

    def step(self):
        """Perform single optimization step."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Weight decay (L2 regularization) - NOT recommended, use AdamW
            if self.weight_decay != 0:
                grad = grad.add(param.data, alpha=self.weight_decay)

            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Update biased second moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters
            param.data.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)


class AdamW(Optimizer):
    """
    AdamW optimizer (Adam with decoupled weight decay).

    Args:
        parameters: Model parameters
        lr: Learning rate
        betas: Coefficients for computing running averages (β1, β2)
        eps: Term for numerical stability
        weight_decay: Weight decay coefficient (decoupled from gradient)
    """

    def __init__(self, parameters, lr: float = 1e-3,
                 betas: tuple = (0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 1e-2):
        super().__init__(parameters, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        # Initialize moments
        self.m = [torch.zeros_like(p.data) for p in self.parameters]
        self.v = [torch.zeros_like(p.data) for p in self.parameters]
        self.t = 0

    def step(self):
        """Perform single optimization step."""
        self.t += 1

        for i, param in enumerate(self.parameters):
            if param.grad is None:
                continue

            grad = param.grad.data

            # Update biased first moment estimate
            self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)

            # Update biased second moment estimate
            self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            # Update parameters with Adam step
            param.data.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

            # Decoupled weight decay (applied separately!)
            if self.weight_decay != 0:
                param.data.add_(param.data, alpha=-self.lr * self.weight_decay)


# Testing and comparison
if __name__ == "__main__":
    # Create simple model
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 1)
    )

    # Generate data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)

    # Test custom optimizer
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.6f}")

    print("\nCustom AdamW optimizer test completed!")
```

### Complete Training Loop with Modern Practices

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List

class ModernTrainer:
    """
    Modern training loop with 2025 best practices:
    - Mixed precision training
    - Gradient accumulation
    - Learning rate scheduling with warm-up
    - Gradient clipping
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True,
        grad_clip: Optional[float] = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == 'cuda'
        self.grad_clip = grad_clip

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }

    def train_epoch(
        self,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        accumulation_steps: int = 1
    ) -> float:
        """
        Train for one epoch with gradient accumulation.

        Args:
            train_loader: Training data loader
            optimizer: Optimizer
            criterion: Loss function
            accumulation_steps: Number of steps to accumulate gradients

        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0

        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)

            # Mixed precision forward pass
            with autocast(enabled=self.use_amp):
                output = self.model(data)
                loss = criterion(output, target)
                # Scale loss for gradient accumulation
                loss = loss / accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # Gradient clipping
                if self.grad_clip is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip
                    )

                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps * data.size(0)

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(
        self,
        data_loader: DataLoader,
        criterion: nn.Module
    ) -> float:
        """Evaluate model on data."""
        self.model.eval()
        total_loss = 0.0

        for data, target in data_loader:
            data, target = data.to(self.device), target.to(self.device)

            output = self.model(data)
            loss = criterion(output, target)
            total_loss += loss.item() * data.size(0)

        return total_loss / len(data_loader.dataset)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        optimizer_name: str = 'adamw',
        scheduler_name: str = 'cosine',
        warmup_epochs: int = 5,
        accumulation_steps: int = 1,
        early_stopping_patience: Optional[int] = 10,
        save_best: bool = True,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Complete training loop with all modern features.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            lr: Learning rate
            weight_decay: Weight decay coefficient
            optimizer_name: 'sgd', 'adam', 'adamw'
            scheduler_name: 'cosine', 'step', 'onecycle'
            warmup_epochs: Number of warm-up epochs
            accumulation_steps: Gradient accumulation steps
            early_stopping_patience: Early stopping patience (None = disabled)
            save_best: Save best model checkpoint
            verbose: Print progress

        Returns:
            Training history
        """
        # Create optimizer
        if optimizer_name == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=0.9,
                weight_decay=weight_decay,
                nesterov=True
            )
        elif optimizer_name == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Create loss function
        criterion = nn.MSELoss()

        # Create learning rate scheduler with warm-up
        def warmup_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            return 1.0

        warmup_scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda
        )

        if scheduler_name == 'cosine':
            main_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
                eta_min=lr * 0.01
            )
        elif scheduler_name == 'step':
            main_scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            )
        elif scheduler_name == 'onecycle':
            main_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=lr,
                epochs=epochs,
                steps_per_epoch=len(train_loader) // accumulation_steps
            )
        else:
            main_scheduler = None

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(
                train_loader,
                optimizer,
                criterion,
                accumulation_steps
            )
            self.history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
            else:
                val_loss = train_loss

            # Learning rate scheduling
            current_lr = optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)

            if epoch < warmup_epochs:
                warmup_scheduler.step()
            elif main_scheduler is not None and scheduler_name != 'onecycle':
                main_scheduler.step()

            # Early stopping
            if early_stopping_patience is not None and val_loader is not None:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if save_best:
                        torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"\nEarly stopping at epoch {epoch}")
                        break

            # Print progress
            if verbose and epoch % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"Val Loss: {val_loss:.6f} | "
                          f"LR: {current_lr:.6f}")
                else:
                    print(f"Epoch {epoch:3d}/{epochs} | "
                          f"Train Loss: {train_loss:.6f} | "
                          f"LR: {current_lr:.6f}")

        # Load best model
        if save_best and val_loader is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))

        return self.history

    def plot_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Loss
        axes[0].plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            axes[0].plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training History', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Learning rate
        axes[1].plot(self.history['lr'], color='red', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 10000
    X = torch.randn(n_samples, 20)
    y = (X[:, 0]**2 + X[:, 1]**2 + torch.randn(n_samples) * 0.1).unsqueeze(1)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Create model
    model = nn.Sequential(
        nn.Linear(20, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1)
    )

    # Train with modern practices
    trainer = ModernTrainer(
        model,
        use_amp=True,
        grad_clip=1.0
    )

    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        lr=1e-3,
        weight_decay=1e-4,
        optimizer_name='adamw',
        scheduler_name='cosine',
        warmup_epochs=5,
        accumulation_steps=1,
        early_stopping_patience=10,
        save_best=True,
        verbose=True
    )

    # Plot results
    trainer.plot_history()

    print("\nTraining complete!")
```

---

## 2025 Best Practices

### Summary of Recommendations

**1. Optimizer Selection:**
- **Default:** AdamW (lr=1e-3, weight_decay=1e-4)
- **Computer Vision (from scratch):** SGD with momentum (lr=0.1, momentum=0.9)
- **Transformers:** AdamW (lr=1e-4 to 5e-5, weight_decay=0.01)

**2. Learning Rate Scheduling:**
- **Default:** Cosine annealing with warm-up (5-10 epochs)
- **Alternative:** One-cycle policy for fast training
- **Always:** Use warm-up for large models

**3. Gradient Clipping:**
- **Always use** gradient clipping by norm (max_norm=1.0)
- **Essential for:** RNNs, Transformers, very deep networks

**4. Mixed Precision Training:**
- **Always enable** on modern GPUs (Ampere, Hopper)
- Use BF16 if available (better range than FP16)
- Essential for large models

**5. Batch Size:**
- **Start with:** 32-64
- **Increase if:** GPU underutilized
- **Use gradient accumulation** for effective large batches

**6. Weight Decay:**
- **Vision:** 1e-4
- **NLP/Transformers:** 1e-2
- **Always use with AdamW**, not Adam

### Common Mistakes to Avoid

1. **Using Adam instead of AdamW** (weight decay implementation is wrong)
2. **No warm-up for large models** (causes instability)
3. **Learning rate too large or too small** (use LR finder)
4. **Not using gradient clipping** (exploding gradients)
5. **Ignoring mixed precision training** (slower, more memory)
6. **Fixed learning rate** (use scheduling)
7. **Wrong batch size** (too small = slow, too large = poor generalization)

### Hyperparameter Tuning Order

**Priority 1 (tune first):**
1. Learning rate (most important!)
2. Batch size

**Priority 2 (tune if time allows):**
3. Weight decay
4. Learning rate schedule
5. Warm-up duration

**Priority 3 (usually use defaults):**
6. Adam β values
7. Gradient clipping threshold
8. Optimizer choice (AdamW for most tasks)

---

## Summary

**Key Insights:**

1. **Optimization is non-convex and stochastic** in deep learning
2. **Adaptive optimizers (AdamW) are the modern default** (2025)
3. **Learning rate scheduling is essential** for best performance
4. **Gradient clipping prevents training instabilities**
5. **Mixed precision training is standard practice**

**2025 Recommendations:**

- **Optimizer:** AdamW (not Adam!)
- **Learning rate:** 1e-3 (tune with LR finder)
- **Schedule:** Cosine annealing with warm-up
- **Gradient clipping:** max_norm=1.0
- **Mixed precision:** Always enable
- **Batch size:** 32-64, increase if GPU allows

**Next Steps:**

- Study regularization techniques (File 16)
- Learn about training strategies (File 17)
- Experiment with different optimizers on your tasks
- Use learning rate finder to tune hyperparameters

---

## References

1. Kingma & Ba (2014). "Adam: A Method for Stochastic Optimization"
2. Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization" (AdamW)
3. Smith (2017). "Cyclical Learning Rates for Training Neural Networks"
4. Loshchilov & Hutter (2016). "SGDR: Stochastic Gradient Descent with Warm Restarts"
5. Polyak (1964). "Some methods of speeding up the convergence of iteration methods"
6. Nesterov (1983). "A method for unconstrained convex minimization problem with the rate of convergence O(1/k^2)"
7. You et al. (2017). "Large Batch Training of Convolutional Networks"
8. Goyal et al. (2017). "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour"
