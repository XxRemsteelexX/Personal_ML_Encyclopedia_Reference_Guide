# Activation Functions

## Table of Contents
1. [Introduction](#introduction)
2. [Why Activation Functions Matter](#why-activation-functions-matter)
3. [Classical Activation Functions](#classical-activation-functions)
4. [The ReLU Revolution](#the-relu-revolution)
5. [Modern Activation Functions (2025)](#modern-activation-functions-2025)
6. [Gradient Dynamics](#gradient-dynamics)
7. [Activation Function Selection Guide](#activation-function-selection-guide)
8. [Complete Implementations](#complete-implementations)
9. [Visualization and Analysis](#visualization-and-analysis)
10. [Practical Considerations](#practical-considerations)

---

## Introduction

Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns. Without activation functions (or with only linear activations), a neural network would be equivalent to a single linear transformation, regardless of depth.

**Fundamental Property:**
```
f(W_L(...W_2(W_1 x + b_1) + b_2...) + b_L) = W_total x + b_total
```

With non-linear activations:
```
f(W_L φ(...φ(W_2 φ(W_1 x + b_1) + b_2)...) + b_L) ≠ Linear function
```

---

## Why Activation Functions Matter

### Universal Approximation

**Theorem:** A network with at least one hidden layer and a non-linear activation can approximate any continuous function (under mild conditions).

**Key insight:** The activation function must be non-linear and non-polynomial.

### Gradient Flow

Activation functions determine how gradients flow during backpropagation:

```
∂L/∂z^[l] = ∂L/∂a^[l] ⊙ φ'(z^[l])
```

If φ'(z) → 0 (vanishing gradient) or φ'(z) → ∞ (exploding gradient), training fails.

### Computational Efficiency

Modern activations are designed for:
- Fast forward computation
- Fast gradient computation
- GPU-friendly operations
- Numerical stability

---

## Classical Activation Functions

### Sigmoid Function

**Definition:**
```
σ(z) = 1 / (1 + e^(-z))
```

**Derivative:**
```
σ'(z) = σ(z)(1 - σ(z))
```

**Properties:**
- **Range:** (0, 1)
- **Output interpretation:** Probability
- **Gradient:** Maximum at z=0 (σ'(0) = 0.25)
- **Saturation:** For |z| > 4, gradient ≈ 0

**Mathematical Analysis:**

Limits:
```
lim_{z→∞} σ(z) = 1
lim_{z→-∞} σ(z) = 0
σ(0) = 0.5
```

Taylor expansion around z=0:
```
σ(z) ≈ 0.5 + 0.25z - 0.0416z^3 + ...
```

**Advantages:**
- Smooth and differentiable
- Clear probabilistic interpretation
- Historically important

**Disadvantages:**
- **Vanishing gradient problem:** For large |z|, σ'(z) ≈ 0
- **Not zero-centered:** Output always positive (causes zig-zagging in gradient descent)
- **Expensive computation:** Exponential function

**When to use:**
- Output layer for binary classification
- Gate mechanisms (e.g., LSTM)
- NOT recommended for hidden layers (2025)

### Hyperbolic Tangent (tanh)

**Definition:**
```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z)) = 2σ(2z) - 1
```

**Derivative:**
```
tanh'(z) = 1 - tanh^2(z)
```

**Properties:**
- **Range:** (-1, 1)
- **Zero-centered:** Output has mean ≈ 0
- **Gradient:** Maximum at z=0 (tanh'(0) = 1)
- **Saturation:** For |z| > 2, gradient ≈ 0

**Mathematical Analysis:**

Relationship to sigmoid:
```
tanh(z) = 2σ(2z) - 1
```

Taylor expansion:
```
tanh(z) ≈ z - z^3/3 + 2z^5/15 - ...
```

**Advantages:**
- Zero-centered (better than sigmoid)
- Stronger gradients than sigmoid
- Smooth and differentiable

**Disadvantages:**
- Still suffers from vanishing gradients
- Expensive computation
- Can saturate

**When to use:**
- Hidden layers (better than sigmoid, but ReLU preferred)
- Recurrent networks (LSTMs, GRUs)
- Output normalization when range (-1, 1) desired

### Softmax Function

**Definition (for vector z ∈ ℝ^K):**
```
softmax(z)_i = e^(z_i) / Σ_j e^(z_j)
```

**Properties:**
- **Range:** (0, 1) for each component
- **Sum:** Σ_i softmax(z)_i = 1
- **Interpretation:** Probability distribution

**Derivative (Jacobian):**
```
∂softmax(z)_i / ∂z_j = softmax(z)_i (δ_ij - softmax(z)_j)
```

Where δ_ij is the Kronecker delta.

**Numerical Stability:**

Naive implementation can overflow. Use this trick:
```
softmax(z) = softmax(z - max(z))  # Shift for stability
```

**Proof of equivalence:**
```
softmax(z - c)_i = e^(z_i - c) / Σ_j e^(z_j - c)
                 = e^z_i * e^(-c) / (Σ_j e^z_j * e^(-c))
                 = e^z_i / Σ_j e^z_j
                 = softmax(z)_i
```

**When to use:**
- Output layer for multi-class classification
- Attention mechanisms
- Mixture of experts

---

## The ReLU Revolution

### Rectified Linear Unit (ReLU)

**Definition:**
```
ReLU(z) = max(0, z) = {z if z > 0
                      {0 if z ≤ 0
```

**Derivative:**
```
ReLU'(z) = {1 if z > 0
           {0 if z ≤ 0
           {undefined at z = 0
```

In practice, define ReLU'(0) = 0 or 1 (doesn't matter much).

**Properties:**
- **Range:** [0, ∞)
- **Non-saturating:** No vanishing gradient for z > 0
- **Sparse activation:** ~50% of neurons are 0
- **Computational efficiency:** Simple max operation

**Why ReLU Works (Theoretical Insights):**

1. **Linear regime:** For z > 0, behaves linearly (easy optimization)
2. **Non-linear overall:** Combination is non-linear
3. **Biological plausibility:** Similar to neural activation patterns
4. **Gradient flow:** Gradient is 1 (no vanishing) for active neurons

**Advantages:**
- Fast computation (no exponentials)
- No vanishing gradient for positive values
- Induces sparsity (regularization effect)
- Empirically works very well
- Accelerates convergence (6x faster, Krizhevsky et al. 2012)

**Disadvantages:**
- **Dying ReLU problem:** Neurons can "die" (always output 0)
- Not zero-centered
- Unbounded output (can lead to instability)

**Dying ReLU Problem:**

If a neuron's input is always negative:
```
z = w^T x + b < 0  for all x
```

Then:
```
ReLU(z) = 0
∂ReLU(z)/∂z = 0
→ No gradient flow
→ Weights never update
→ Neuron is "dead"
```

**Causes:**
- Poor initialization (too negative bias)
- Large learning rate (causes large weight updates)
- Data distribution shift

**Mitigation:**
- Proper initialization (He initialization)
- Moderate learning rates
- Use variants (Leaky ReLU, PReLU)

### Leaky ReLU

**Definition:**
```
LeakyReLU(z) = {z      if z > 0
               {αz     if z ≤ 0
```

Where α ∈ (0, 1), typically α = 0.01 or 0.2.

**Derivative:**
```
LeakyReLU'(z) = {1   if z > 0
                {α   if z ≤ 0
```

**Advantages over ReLU:**
- Never dies (always has gradient)
- Small negative slope allows negative values

**Disadvantages:**
- Introduces hyperparameter α
- Not always better than ReLU empirically

**When to use:**
- When experiencing dying ReLU problem
- In GANs (commonly used in discriminator)

### Parametric ReLU (PReLU)

**Definition:**
```
PReLU(z) = {z   if z > 0
           {αz  if z ≤ 0
```

Where α is a **learnable parameter** (updated via backpropagation).

**Gradient with respect to α:**
```
∂L/∂α = Σ_{z<0} ∂L/∂PReLU(z) * z
```

**Advantages:**
- Automatically learns best α
- Can have different α for each channel/neuron
- Often outperforms fixed ReLU/Leaky ReLU

**Disadvantages:**
- Additional parameters to learn
- Risk of overfitting with too many parameters

**Variants:**
- **Channel-wise PReLU:** One α per channel
- **Element-wise PReLU:** One α per neuron

### Exponential Linear Unit (ELU)

**Definition:**
```
ELU(z) = {z              if z > 0
         {α(e^z - 1)     if z ≤ 0
```

Where α > 0 (typically α = 1).

**Derivative:**
```
ELU'(z) = {1              if z > 0
          {ELU(z) + α     if z ≤ 0
          {α e^z          if z ≤ 0
```

**Properties:**
- **Smooth:** No kink at z = 0 (unlike ReLU)
- **Negative values:** Pushes mean activation toward zero
- **Saturation:** For large negative z, approaches -α

**Advantages:**
- No dying ReLU problem
- Mean activations closer to zero (faster convergence)
- Smooth gradient

**Disadvantages:**
- Exponential computation (slower than ReLU)
- Still has saturation for negative values

**Scaled ELU (SELU):**

**Definition:**
```
SELU(z) = λ * {z              if z > 0
              {α(e^z - 1)     if z ≤ 0
```

Where λ ≈ 1.0507 and α ≈ 1.6733 (specifically chosen values).

**Special property:** Self-normalizing (activations converge to mean 0, variance 1) under specific conditions:
- Weights initialized with LeCun normal
- Sequential fully-connected layers
- Input normalized

---

## Modern Activation Functions (2025)

### Gaussian Error Linear Unit (GELU)

**Definition:**
```
GELU(z) = z * Φ(z)
```

Where Φ(z) is the CDF of standard normal distribution:
```
Φ(z) = P(Z ≤ z) where Z ~ N(0, 1)
     = (1/2)[1 + erf(z/√2)]
```

**Intuition:** Stochastically gates the input based on its value. Larger inputs are more likely to be "passed through."

**Approximations:**

**Hyperbolic tangent approximation (fast):**
```
GELU(z) ≈ 0.5z(1 + tanh[√(2/π)(z + 0.044715z^3)])
```

**Sigmoid approximation:**
```
GELU(z) ≈ z σ(1.702z)
```

**Derivative:**
```
GELU'(z) = Φ(z) + z φ(z)
```

Where φ(z) is the PDF of standard normal.

**Properties:**
- **Smooth:** Infinitely differentiable
- **Non-monotonic:** Slight "bump" around z = 0
- **Stochastic interpretation:** Probabilistic gating

**Advantages:**
- State-of-the-art for transformers (BERT, GPT)
- Better empirical performance than ReLU in many cases
- Smooth gradient flow

**Disadvantages:**
- More expensive than ReLU
- Less interpretable

**When to use (2025 standard):**
- **Transformers:** BERT, GPT, Vision Transformers
- **Large language models**
- **Modern vision architectures**

### Swish / SiLU (Sigmoid Linear Unit)

**Definition:**
```
Swish(z) = z * σ(z) = z / (1 + e^(-z))
```

Also called SiLU (Sigmoid Linear Unit) or Swish-1.

**Parametric version:**
```
Swish_β(z) = z * σ(βz)
```

Where β is a learnable or fixed parameter (β = 1 for standard Swish).

**Derivative:**
```
Swish'(z) = σ(z) + z σ(z)(1 - σ(z))
           = σ(z)(1 + z(1 - σ(z)))
```

**Properties:**
- **Smooth:** Infinitely differentiable
- **Self-gated:** Input modulates itself
- **Non-monotonic:** Slight dip for negative values

**Advantages:**
- Outperforms ReLU in many architectures
- Smooth and differentiable
- Simple form

**Disadvantages:**
- More expensive than ReLU (sigmoid computation)
- Can have vanishing gradients for very negative values

**When to use:**
- EfficientNet and mobile architectures
- When computational budget allows
- Alternative to ReLU for better performance

### Mish

**Definition:**
```
Mish(z) = z * tanh(softplus(z))
        = z * tanh(ln(1 + e^z))
```

**Derivative:**
```
Mish'(z) = sech^2(softplus(z)) + z * σ(z) * sech^2(softplus(z))
```

Where sech(z) = 1/cosh(z).

**Properties:**
- **Smooth:** Infinitely differentiable
- **Unbounded above, bounded below:** Similar to Swish
- **Self-regularizing:** Small negative values allowed

**Advantages:**
- Strong empirical performance
- Smooth gradient landscape
- Regularization effect

**Disadvantages:**
- Most expensive to compute
- Requires softplus and tanh

**When to use:**
- When seeking maximum performance
- Computer vision tasks
- When computational cost is not critical

### Comparison of Modern Activations

| Function | Smoothness | Computational Cost | Transformers | Vision | Empirical Performance |
|----------|------------|-------------------|--------------|--------|----------------------|
| ReLU     | Kinked     | Very Low          | Good         | Good   | Baseline             |
| GELU     | Smooth     | Medium            | Excellent    | Good   | SOTA for Transformers|
| Swish    | Smooth     | Medium            | Good         | Excellent | Better than ReLU  |
| Mish     | Smooth     | High              | Good         | Excellent | Often best        |

---

## Gradient Dynamics

### Vanishing Gradient Problem

**Problem:** Gradients become exponentially small in deep networks.

**Mathematical Analysis:**

For a network with L layers:
```
∂L/∂W^[1] = ∂L/∂z^[L] * ∏_{l=2}^L (W^[l])^T * φ'(z^[l-1])
```

If |φ'(z)| < 1 for most z (e.g., sigmoid), then:
```
|∂L/∂W^[1]| ≈ |∂L/∂z^[L]| * ∏_{l=2}^L |W^[l]| * |φ'(z^[l-1])|
```

As L increases, this product can become very small.

**Example (sigmoid):**
```
max(σ'(z)) = 0.25

After 10 layers: 0.25^10 ≈ 10^(-6)
```

**Solutions:**
- Use ReLU (φ'(z) = 1 for z > 0)
- Batch normalization
- Residual connections (skip connections)
- Proper initialization

### Exploding Gradient Problem

**Problem:** Gradients become exponentially large.

**Occurs when:**
- Large weights
- Certain activations (unbounded derivatives)
- Poor initialization

**Solutions:**
- Gradient clipping
- Proper initialization (Xavier, He)
- Batch normalization
- Lower learning rates

### Dead Neurons Problem

**Specific to ReLU:**

A neuron is "dead" if:
```
ReLU(w^T x + b) = 0  for all x in dataset
```

**Statistics:**

In practice, 20-40% of neurons can die with poor initialization or large learning rates.

**Solutions:**
- Leaky ReLU, PReLU, ELU
- Lower learning rates
- Proper initialization
- Batch normalization

---

## Activation Function Selection Guide

### Decision Tree (2025 Best Practices)

```
Task Type?
│
├─ Transformers / NLP
│  └─ Use GELU (standard for BERT, GPT)
│
├─ Computer Vision (CNNs)
│  ├─ Default: ReLU (fast and reliable)
│  ├─ Better performance: Swish or Mish
│  └─ Mobile/Efficient: Swish (EfficientNet)
│
├─ Recurrent Networks (RNNs, LSTMs)
│  └─ Use tanh (for gates), sigmoid (for gates)
│
├─ GANs
│  ├─ Generator: ReLU or Leaky ReLU
│  └─ Discriminator: Leaky ReLU
│
└─ Output Layer
   ├─ Binary Classification: Sigmoid
   ├─ Multi-class Classification: Softmax
   └─ Regression: Linear (no activation)
```

### Hidden Layer Recommendations

**2025 Standards:**

1. **First choice: ReLU**
   - Fast, simple, works well
   - Good default for most tasks

2. **Second choice: GELU**
   - For transformers and large models
   - Better empirical performance

3. **Third choice: Swish/Mish**
   - When seeking maximum performance
   - If computational budget allows

4. **Special cases:**
   - **Dying ReLU problem:** Leaky ReLU or PReLU
   - **Self-normalizing networks:** SELU
   - **Smooth gradients needed:** ELU or Swish

### Output Layer Recommendations

| Task | Activation | Loss Function |
|------|-----------|---------------|
| Binary Classification | Sigmoid | Binary Cross-Entropy |
| Multi-class Classification | Softmax | Categorical Cross-Entropy |
| Multi-label Classification | Sigmoid (per class) | Binary Cross-Entropy |
| Regression (unbounded) | Linear | MSE / MAE |
| Regression (bounded [0,1]) | Sigmoid | MSE |
| Regression (bounded [-1,1]) | tanh | MSE |

---

## Complete Implementations

### PyTorch Implementations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class ActivationFunctions:
    """
    Complete implementation of all major activation functions.
    Includes both forward and derivative computations.
    """

    # ========== Classical Activations ==========

    @staticmethod
    def sigmoid(z: torch.Tensor) -> torch.Tensor:
        """Sigmoid: σ(z) = 1 / (1 + e^(-z))"""
        return torch.sigmoid(z)

    @staticmethod
    def sigmoid_derivative(z: torch.Tensor) -> torch.Tensor:
        """Derivative: σ'(z) = σ(z)(1 - σ(z))"""
        s = torch.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def tanh(z: torch.Tensor) -> torch.Tensor:
        """Hyperbolic tangent"""
        return torch.tanh(z)

    @staticmethod
    def tanh_derivative(z: torch.Tensor) -> torch.Tensor:
        """Derivative: tanh'(z) = 1 - tanh^2(z)"""
        t = torch.tanh(z)
        return 1 - t ** 2

    @staticmethod
    def softmax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """
        Numerically stable softmax.

        softmax(z)_i = e^z_i / Σ_j e^z_j
        """
        # Subtract max for numerical stability
        z_shifted = z - torch.max(z, dim=dim, keepdim=True)[0]
        exp_z = torch.exp(z_shifted)
        return exp_z / torch.sum(exp_z, dim=dim, keepdim=True)

    # ========== ReLU Family ==========

    @staticmethod
    def relu(z: torch.Tensor) -> torch.Tensor:
        """ReLU: max(0, z)"""
        return F.relu(z)

    @staticmethod
    def relu_derivative(z: torch.Tensor) -> torch.Tensor:
        """Derivative: 1 if z > 0, else 0"""
        return (z > 0).float()

    @staticmethod
    def leaky_relu(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        """Leaky ReLU: max(αz, z)"""
        return F.leaky_relu(z, negative_slope=alpha)

    @staticmethod
    def leaky_relu_derivative(z: torch.Tensor, alpha: float = 0.01) -> torch.Tensor:
        """Derivative: 1 if z > 0, else α"""
        return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, alpha))

    @staticmethod
    def elu(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """ELU: z if z > 0, else α(e^z - 1)"""
        return F.elu(z, alpha=alpha)

    @staticmethod
    def elu_derivative(z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """Derivative: 1 if z > 0, else α*e^z"""
        return torch.where(z > 0, torch.ones_like(z), alpha * torch.exp(z))

    @staticmethod
    def selu(z: torch.Tensor) -> torch.Tensor:
        """
        Scaled ELU: λ * [z if z > 0, else α(e^z - 1)]

        λ ≈ 1.0507, α ≈ 1.6733 (specifically chosen for self-normalization)
        """
        return F.selu(z)

    # ========== Modern Activations ==========

    @staticmethod
    def gelu(z: torch.Tensor, approximate: bool = False) -> torch.Tensor:
        """
        GELU: z * Φ(z) where Φ is the CDF of N(0,1)

        Args:
            approximate: Use tanh approximation (faster)
        """
        if approximate:
            # Approximation: 0.5 * z * (1 + tanh(√(2/π) * (z + 0.044715 * z^3)))
            return 0.5 * z * (1 + torch.tanh(
                np.sqrt(2 / np.pi) * (z + 0.044715 * torch.pow(z, 3))
            ))
        else:
            # Exact: z * Φ(z)
            return z * 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

    @staticmethod
    def gelu_derivative(z: torch.Tensor) -> torch.Tensor:
        """Derivative: Φ(z) + z * φ(z)"""
        cdf = 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))
        pdf = torch.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
        return cdf + z * pdf

    @staticmethod
    def swish(z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """
        Swish/SiLU: z * σ(βz)

        Args:
            beta: Scaling parameter (β = 1 for standard Swish)
        """
        return z * torch.sigmoid(beta * z)

    @staticmethod
    def swish_derivative(z: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
        """Derivative: σ(βz) + βz*σ(βz)*(1 - σ(βz))"""
        sigmoid_z = torch.sigmoid(beta * z)
        return sigmoid_z + beta * z * sigmoid_z * (1 - sigmoid_z)

    @staticmethod
    def mish(z: torch.Tensor) -> torch.Tensor:
        """
        Mish: z * tanh(softplus(z))
             = z * tanh(ln(1 + e^z))
        """
        return z * torch.tanh(F.softplus(z))

    @staticmethod
    def mish_derivative(z: torch.Tensor) -> torch.Tensor:
        """Derivative of Mish"""
        softplus_z = F.softplus(z)
        tanh_softplus = torch.tanh(softplus_z)
        sigmoid_z = torch.sigmoid(z)
        sech2_softplus = 1 / (torch.cosh(softplus_z) ** 2)

        return tanh_softplus + z * sigmoid_z * sech2_softplus


class PReLU(nn.Module):
    """
    Parametric ReLU with learnable negative slope.

    Args:
        num_parameters: Number of α parameters (1 for shared, num_channels for channel-wise)
        init: Initial value of α
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25):
        super().__init__()
        self.num_parameters = num_parameters
        self.alpha = nn.Parameter(torch.ones(num_parameters) * init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.where(x > 0, x, self.alpha * x)


# ========== Visualization ==========

def plot_activations():
    """Plot all activation functions for comparison."""
    z = torch.linspace(-5, 5, 1000)

    activations = {
        'Sigmoid': ActivationFunctions.sigmoid,
        'Tanh': ActivationFunctions.tanh,
        'ReLU': ActivationFunctions.relu,
        'Leaky ReLU': lambda z: ActivationFunctions.leaky_relu(z, 0.1),
        'ELU': ActivationFunctions.elu,
        'GELU': ActivationFunctions.gelu,
        'Swish': ActivationFunctions.swish,
        'Mish': ActivationFunctions.mish,
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(activations.items()):
        with torch.no_grad():
            y = func(z)

        axes[idx].plot(z.numpy(), y.numpy(), linewidth=2)
        axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[idx].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('z')
        axes[idx].set_ylabel('φ(z)')

    plt.tight_layout()
    plt.savefig('activation_functions.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_derivatives():
    """Plot derivatives of activation functions."""
    z = torch.linspace(-5, 5, 1000)

    derivatives = {
        'Sigmoid': ActivationFunctions.sigmoid_derivative,
        'Tanh': ActivationFunctions.tanh_derivative,
        'ReLU': ActivationFunctions.relu_derivative,
        'Leaky ReLU': lambda z: ActivationFunctions.leaky_relu_derivative(z, 0.1),
        'ELU': ActivationFunctions.elu_derivative,
        'GELU': ActivationFunctions.gelu_derivative,
        'Swish': ActivationFunctions.swish_derivative,
        'Mish': ActivationFunctions.mish_derivative,
    }

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, (name, func) in enumerate(derivatives.items()):
        with torch.no_grad():
            dy = func(z)

        axes[idx].plot(z.numpy(), dy.numpy(), linewidth=2, color='red')
        axes[idx].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[idx].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[idx].axhline(y=1, color='g', linestyle='--', alpha=0.3, label="1.0")
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_title(f"{name} Derivative", fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('z')
        axes[idx].set_ylabel("φ'(z)")

    plt.tight_layout()
    plt.savefig('activation_derivatives.png', dpi=300, bbox_inches='tight')
    plt.show()


def compare_gradient_flow():
    """
    Demonstrate gradient flow through deep networks with different activations.
    """
    depth = 50  # Number of layers
    activations_to_test = {
        'Sigmoid': ActivationFunctions.sigmoid_derivative,
        'Tanh': ActivationFunctions.tanh_derivative,
        'ReLU': ActivationFunctions.relu_derivative,
        'GELU': ActivationFunctions.gelu_derivative,
        'Swish': ActivationFunctions.swish_derivative,
    }

    results = {}

    for name, derivative_func in activations_to_test.items():
        # Simulate gradient flow through layers
        gradient = torch.ones(1)  # Start with gradient = 1
        gradient_history = [gradient.item()]

        for layer in range(depth):
            # Random pre-activation (simulating forward pass)
            z = torch.randn(1) * 0.5  # Small variance

            # Multiply by derivative (backpropagation)
            with torch.no_grad():
                gradient = gradient * derivative_func(z)

            gradient_history.append(gradient.item())

        results[name] = gradient_history

    # Plot results
    plt.figure(figsize=(12, 6))
    for name, history in results.items():
        plt.semilogy(history, label=name, linewidth=2)

    plt.xlabel('Layer (from output to input)', fontsize=12)
    plt.ylabel('Gradient Magnitude (log scale)', fontsize=12)
    plt.title('Gradient Flow Through 50-Layer Network', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('gradient_flow.png', dpi=300, bbox_inches='tight')
    plt.show()


# ========== Empirical Comparison ==========

class ActivationComparison(nn.Module):
    """
    Network for comparing different activation functions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        activation_name: str = 'relu'
    ):
        super().__init__()

        # Select activation
        if activation_name == 'relu':
            self.activation = nn.ReLU()
        elif activation_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation_name == 'elu':
            self.activation = nn.ELU()
        elif activation_name == 'gelu':
            self.activation = nn.GELU()
        elif activation_name == 'silu':  # Swish
            self.activation = nn.SiLU()
        elif activation_name == 'mish':
            self.activation = nn.Mish()
        elif activation_name == 'tanh':
            self.activation = nn.Tanh()
        elif activation_name == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(f"Unknown activation: {activation_name}")

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(self.activation)

        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def benchmark_activations():
    """
    Empirically compare activation functions on a simple task.
    """
    import time
    from torch.utils.data import TensorDataset, DataLoader

    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 10000
    X = torch.randn(n_samples, 20)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 + torch.randn(n_samples) * 0.1).unsqueeze(1)

    # Split data
    train_size = int(0.8 * n_samples)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Activations to test
    activations = ['relu', 'leaky_relu', 'elu', 'gelu', 'silu', 'mish']

    results = {}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for activation_name in activations:
        print(f"\nTesting {activation_name}...")

        # Create model
        model = ActivationComparison(
            input_dim=20,
            hidden_dims=[64, 64, 32],
            output_dim=1,
            activation_name=activation_name
        ).to(device)

        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # Training
        start_time = time.time()
        losses = []

        for epoch in range(50):
            model.train()
            epoch_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= len(train_loader.dataset)
            losses.append(epoch_loss)

        training_time = time.time() - start_time

        # Evaluation
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            y_test_device = y_test.to(device)
            test_predictions = model(X_test_device)
            test_loss = criterion(test_predictions, y_test_device).item()

        results[activation_name] = {
            'final_train_loss': losses[-1],
            'test_loss': test_loss,
            'training_time': training_time,
            'loss_history': losses
        }

        print(f"  Train Loss: {losses[-1]:.6f}")
        print(f"  Test Loss: {test_loss:.6f}")
        print(f"  Time: {training_time:.2f}s")

    # Plot comparison
    plt.figure(figsize=(15, 5))

    # Loss curves
    plt.subplot(1, 3, 1)
    for name, result in results.items():
        plt.plot(result['loss_history'], label=name, linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final losses
    plt.subplot(1, 3, 2)
    names = list(results.keys())
    train_losses = [results[name]['final_train_loss'] for name in names]
    test_losses = [results[name]['test_loss'] for name in names]

    x = np.arange(len(names))
    width = 0.35
    plt.bar(x - width/2, train_losses, width, label='Train', alpha=0.8)
    plt.bar(x + width/2, test_losses, width, label='Test', alpha=0.8)
    plt.xlabel('Activation')
    plt.ylabel('Loss')
    plt.title('Final Loss Comparison')
    plt.xticks(x, names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # Training time
    plt.subplot(1, 3, 3)
    times = [results[name]['training_time'] for name in names]
    plt.bar(names, times, color='steelblue', alpha=0.8)
    plt.xlabel('Activation')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('activation_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()

    return results


if __name__ == "__main__":
    print("Activation Functions Demo\n" + "="*50)

    # 1. Plot all activation functions
    print("\n1. Plotting activation functions...")
    plot_activations()

    # 2. Plot derivatives
    print("\n2. Plotting derivatives...")
    plot_derivatives()

    # 3. Compare gradient flow
    print("\n3. Analyzing gradient flow...")
    compare_gradient_flow()

    # 4. Benchmark activations
    print("\n4. Benchmarking activations...")
    results = benchmark_activations()

    print("\n" + "="*50)
    print("Demo complete! Check generated PNG files for visualizations.")
```

---

## Visualization and Analysis

### Activation Function Landscape

```python
def plot_3d_landscape():
    """
    Visualize how different activations transform 2D input space.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    # Create mesh grid
    x1 = np.linspace(-3, 3, 100)
    x2 = np.linspace(-3, 3, 100)
    X1, X2 = np.meshgrid(x1, x2)

    # Compute z = x1^2 + x2^2 (quadratic function)
    Z_input = X1**2 + X2**2

    activations = {
        'Input (z)': lambda z: z,
        'ReLU(z)': lambda z: np.maximum(0, z),
        'Sigmoid(z)': lambda z: 1 / (1 + np.exp(-z)),
        'Tanh(z)': lambda z: np.tanh(z),
    }

    fig = plt.figure(figsize=(16, 12))

    for idx, (name, func) in enumerate(activations.items()):
        ax = fig.add_subplot(2, 2, idx + 1, projection='3d')
        Z_output = func(Z_input)

        surf = ax.plot_surface(X1, X2, Z_output, cmap='viridis', alpha=0.9)
        ax.set_xlabel('x1', fontsize=10)
        ax.set_ylabel('x2', fontsize=10)
        ax.set_zlabel('output', fontsize=10)
        ax.set_title(name, fontsize=12, fontweight='bold')
        fig.colorbar(surf, ax=ax, shrink=0.5)

    plt.tight_layout()
    plt.savefig('activation_3d_landscape.png', dpi=300, bbox_inches='tight')
    plt.show()
```

---

## Practical Considerations

### Computational Cost Analysis

| Activation | Forward FLOPs | Backward FLOPs | Memory | GPU Efficiency |
|------------|---------------|----------------|--------|----------------|
| ReLU       | 1             | 1              | Low    | Excellent      |
| Leaky ReLU | 2             | 2              | Low    | Excellent      |
| ELU        | ~10 (exp)     | ~10            | Medium | Good           |
| GELU       | ~15           | ~20            | Medium | Good           |
| Swish      | ~10           | ~15            | Medium | Good           |
| Mish       | ~20           | ~30            | Medium | Fair           |
| Sigmoid    | ~8 (exp)      | ~10            | Medium | Good           |
| Tanh       | ~10 (exp)     | ~12            | Medium | Good           |

**2025 Recommendation:** Use ReLU by default. Use GELU for transformers. Consider Swish/Mish only when performance gains justify computational cost.

### Numerical Stability

**Common issues:**

1. **Sigmoid/Tanh overflow:**
   ```python
   # Problem
   z = torch.tensor([1000.0])
   sigmoid = 1 / (1 + torch.exp(-z))  # inf

   # Solution: PyTorch's implementation handles this
   sigmoid = torch.sigmoid(z)  # 1.0
   ```

2. **Softmax overflow:**
   ```python
   # Problem
   z = torch.tensor([1000.0, 1001.0, 999.0])
   exp_z = torch.exp(z)  # overflow

   # Solution: subtract max
   z_shifted = z - torch.max(z)
   exp_z = torch.exp(z_shifted)
   softmax = exp_z / torch.sum(exp_z)
   ```

3. **GELU approximation accuracy:**
   ```python
   # Exact (slower)
   gelu_exact = z * 0.5 * (1.0 + torch.erf(z / np.sqrt(2.0)))

   # Approximate (faster, used in transformers)
   gelu_approx = 0.5 * z * (1 + torch.tanh(
       np.sqrt(2/np.pi) * (z + 0.044715 * z**3)
   ))

   # Difference is negligible for training
   ```

### Dying ReLU Diagnosis

```python
def diagnose_dead_neurons(model, data_loader):
    """
    Check percentage of dead ReLU neurons.

    A neuron is "dead" if it never activates (always outputs 0).
    """
    dead_neurons = {}

    model.eval()
    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, nn.ReLU):
                dead_neurons[name] = []

        # Run through dataset
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(next(model.parameters()).device)

            # Forward pass with hooks
            activations = {}

            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output
                return hook

            hooks = []
            for name, module in model.named_modules():
                if isinstance(module, nn.ReLU):
                    hooks.append(module.register_forward_hook(hook_fn(name)))

            model(batch_x)

            # Check which neurons activated
            for name, activation in activations.items():
                # Neuron is active if it outputs > 0
                active = (activation > 0).any(dim=0).float()
                dead_neurons[name].append(active)

            # Remove hooks
            for hook in hooks:
                hook.remove()

    # Compute statistics
    results = {}
    for name, activations_list in dead_neurons.items():
        ever_active = torch.stack(activations_list).any(dim=0)
        dead_percentage = (1 - ever_active.float().mean()).item() * 100
        results[name] = dead_percentage
        print(f"{name}: {dead_percentage:.1f}% dead neurons")

    return results
```

### Hyperparameter Sensitivity

**Learning rate interaction:**

Different activations may require different learning rates:

| Activation | Recommended LR (relative) | Notes |
|------------|---------------------------|-------|
| ReLU       | 1.0× (baseline)           | Standard |
| Leaky ReLU | 1.0×                      | Similar to ReLU |
| ELU        | 1.0-1.5×                  | Can handle higher LR |
| GELU       | 0.8-1.0×                  | Slightly lower |
| Swish      | 0.8-1.0×                  | Similar to GELU |
| Sigmoid    | 0.5×                      | Much lower (rarely used) |
| Tanh       | 0.7×                      | Lower than ReLU |

---

## Summary

**Key Insights:**

1. **Activation functions enable non-linearity** - essential for learning complex patterns

2. **ReLU revolutionized deep learning** - simple, effective, computationally efficient

3. **Modern activations (GELU, Swish, Mish)** - better performance but higher computational cost

4. **Gradient flow is critical** - sigmoid/tanh can cause vanishing gradients, ReLU doesn't (for positive values)

5. **Context matters** - different tasks benefit from different activations

**2025 Best Practices:**

- **Default choice:** ReLU (fast, reliable)
- **Transformers:** GELU (state-of-the-art for NLP)
- **Vision (performance-critical):** Swish or Mish
- **Output layers:** Task-specific (sigmoid, softmax, linear)
- **Avoid:** Sigmoid and tanh in hidden layers (use ReLU family instead)

**Common Mistakes to Avoid:**

1. Using sigmoid/tanh in deep networks (vanishing gradients)
2. Not addressing dying ReLU problem (use Leaky ReLU or proper initialization)
3. Ignoring computational cost of complex activations
4. Not tuning learning rate for different activations
5. Using wrong activation in output layer

**Next Steps:**

- Study optimization algorithms (File 15)
- Learn about normalization techniques (File 16)
- Understand how activations interact with optimization and regularization
- Experiment with different activations on your tasks

---

## References

1. Nair & Hinton (2010). "Rectified Linear Units Improve Restricted Boltzmann Machines"
2. Maas et al. (2013). "Rectifier Nonlinearities Improve Neural Network Acoustic Models"
3. He et al. (2015). "Delving Deep into Rectifiers"
4. Clevert et al. (2015). "Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)"
5. Hendrycks & Gimpel (2016). "Gaussian Error Linear Units (GELUs)"
6. Ramachandran et al. (2017). "Searching for Activation Functions" (Swish)
7. Misra (2019). "Mish: A Self Regularized Non-Monotonic Activation Function"
8. Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks"
