# Neural Network Basics

## Table of Contents
1. [Introduction](#introduction)
2. [Historical Context: From Neurons to Networks](#historical-context)
3. [The Perceptron](#the-perceptron)
4. [Multi-Layer Perceptrons (MLPs)](#multi-layer-perceptrons)
5. [Forward Propagation](#forward-propagation)
6. [Backpropagation Algorithm](#backpropagation-algorithm)
7. [Computational Graphs](#computational-graphs)
8. [Weight Initialization](#weight-initialization)
9. [Bias-Variance Tradeoff in Neural Networks](#bias-variance-tradeoff)
10. [Universal Approximation Theorem](#universal-approximation-theorem)
11. [Complete Implementations](#complete-implementations)
12. [Practical Considerations](#practical-considerations)

---

## Introduction

Neural networks are the foundation of modern deep learning. Despite their biological inspiration, they are fundamentally mathematical function approximators that learn hierarchical representations through gradient-based optimization.

**Key Insight:** A neural network is a parametrized function f(x; θ) where θ represents learnable parameters (weights and biases). Training is the process of finding θ* that minimizes a loss function L(θ) over a dataset.

---

## Historical Context: From Neurons to Networks

### The McCulloch-Pitts Neuron (1943)

The first mathematical model of a neuron:

**Model:**
```
y = 1 if Σ(w_i * x_i) ≥ θ else 0
```

Where:
- x_i: binary inputs
- w_i: weights (fixed, not learned)
- θ: threshold
- y: binary output

**Limitations:**
- Binary inputs and outputs only
- No learning mechanism
- Linear decision boundaries

### The Rosenblatt Perceptron (1958)

**Key Innovation:** Learning rule for adjusting weights

**Perceptron Learning Rule:**
```
w_i(t+1) = w_i(t) + α * (y_true - y_pred) * x_i
```

**Mathematical Formulation:**
```
z = w^T x + b
y = sign(z) = { +1 if z ≥ 0
              { -1 if z < 0
```

**Convergence Theorem:** For linearly separable data, the perceptron algorithm converges in finite steps.

**Proof Sketch:**
Let M be the margin (minimum distance from any point to the decision boundary). After at most (R/M)^2 updates (where R is the maximum norm of any input), the algorithm converges.

**Limitations (Minsky & Papert, 1969):**
- Cannot learn XOR function
- Limited to linearly separable problems
- Led to the first "AI winter"

---

## The Perceptron

### Mathematical Formulation

**Input:** x ∈ ℝ^d (d-dimensional feature vector)
**Parameters:** w ∈ ℝ^d (weights), b ∈ ℝ (bias)
**Pre-activation:** z = w^T x + b = Σ(w_i * x_i) + b
**Activation:** y = φ(z) where φ is an activation function

### Geometric Interpretation

The decision boundary is a hyperplane in ℝ^d:
```
w^T x + b = 0
```

The weight vector w is perpendicular to this hyperplane.

### Limitations

**XOR Problem:**
```
Input: (0,0) → 0
       (0,1) → 1
       (1,0) → 1
       (1,1) → 0
```

No single line can separate the positive and negative classes.

**Solution:** Multi-layer networks with non-linear activations.

---

## Multi-Layer Perceptrons (MLPs)

### Architecture

An MLP consists of:
1. **Input layer:** Receives features x ∈ ℝ^d
2. **Hidden layers:** One or more layers of neurons
3. **Output layer:** Produces predictions y ∈ ℝ^k

### Mathematical Formulation

For an L-layer network:

**Layer 1 (first hidden layer):**
```
z^[1] = W^[1] x + b^[1]
a^[1] = φ^[1](z^[1])
```

**Layer l (general hidden layer):**
```
z^[l] = W^[l] a^[l-1] + b^[l]
a^[l] = φ^[l](z^[l])
```

**Output layer L:**
```
z^[L] = W^[L] a^[L-1] + b^[L]
y_hat = φ^[L](z^[L])
```

### Notation

- **L:** Total number of layers (excluding input)
- **n^[l]:** Number of neurons in layer l
- **W^[l]:** Weight matrix for layer l, shape (n^[l], n^[l-1])
- **b^[l]:** Bias vector for layer l, shape (n^[l], 1)
- **z^[l]:** Pre-activation for layer l
- **a^[l]:** Activation for layer l (a^[0] = x)
- **φ^[l]:** Activation function for layer l

### Why Multiple Layers?

**Theorem (Universal Approximation):** A feedforward network with a single hidden layer containing a finite number of neurons can approximate any continuous function on compact subsets of ℝ^n, under mild assumptions on the activation function.

**However:** Deeper networks can represent the same functions with exponentially fewer neurons.

**Example:** Computing parity of n bits:
- Single hidden layer: Requires O(2^n) neurons
- Deep network: Requires O(n) neurons

---

## Forward Propagation

### Algorithm

**Input:** x ∈ ℝ^d, parameters {W^[l], b^[l]}_{l=1}^L
**Output:** y_hat ∈ ℝ^k

```
1. Set a^[0] = x
2. For l = 1 to L:
   a. Compute z^[l] = W^[l] a^[l-1] + b^[l]
   b. Compute a^[l] = φ^[l](z^[l])
3. Return y_hat = a^[L]
```

### Vectorized Implementation (Batch)

For a batch of m examples X ∈ ℝ^(d × m):

```
Z^[l] = W^[l] A^[l-1] + b^[l]  (broadcasting b^[l])
A^[l] = φ^[l](Z^[l])           (element-wise)
```

Where:
- **A^[l]:** shape (n^[l], m)
- **Z^[l]:** shape (n^[l], m)

### Computational Complexity

For a single forward pass:
- **FLOPs per layer l:** O(n^[l] × n^[l-1])
- **Total FLOPs:** O(Σ n^[l] × n^[l-1])

For typical architectures, this is dominated by matrix multiplications.

---

## Backpropagation Algorithm

### The Core Idea

**Goal:** Compute ∂L/∂W^[l] and ∂L/∂b^[l] for all layers

**Key Insight:** Use the chain rule recursively from output to input

### Mathematical Derivation

**Loss Function:** L(y_hat, y_true)

**Output Layer Error:**
```
δ^[L] = ∂L/∂z^[L] = ∂L/∂a^[L] ⊙ φ'^[L](z^[L])
```

Where ⊙ denotes element-wise multiplication.

**Recursive Error Computation:**
```
δ^[l] = ∂L/∂z^[l] = (W^[l+1])^T δ^[l+1] ⊙ φ'^[l](z^[l])
```

**Proof:**
```
∂L/∂z^[l] = ∂L/∂a^[l] ⊙ φ'^[l](z^[l])        (by chain rule)

∂L/∂a^[l] = Σ_j ∂L/∂z_j^[l+1] * ∂z_j^[l+1]/∂a^[l]
          = Σ_j δ_j^[l+1] * W_j^[l+1]
          = (W^[l+1])^T δ^[l+1]

Therefore:
δ^[l] = (W^[l+1])^T δ^[l+1] ⊙ φ'^[l](z^[l])
```

**Gradient with respect to parameters:**
```
∂L/∂W^[l] = δ^[l] (a^[l-1])^T
∂L/∂b^[l] = δ^[l]
```

**Proof:**
```
∂L/∂W_ij^[l] = ∂L/∂z_i^[l] * ∂z_i^[l]/∂W_ij^[l]
              = δ_i^[l] * a_j^[l-1]

Therefore: ∂L/∂W^[l] = δ^[l] (a^[l-1])^T
```

### Backpropagation Algorithm (Complete)

**Forward Pass:**
```
1. a^[0] = x
2. For l = 1 to L:
   z^[l] = W^[l] a^[l-1] + b^[l]
   a^[l] = φ^[l](z^[l])
   (cache z^[l], a^[l])
```

**Backward Pass:**
```
1. Compute δ^[L] = ∂L/∂a^[L] ⊙ φ'^[L](z^[L])
2. For l = L to 1:
   a. ∂L/∂W^[l] = δ^[l] (a^[l-1])^T
   b. ∂L/∂b^[l] = δ^[l]
   c. If l > 1:
      δ^[l-1] = (W^[l])^T δ^[l] ⊙ φ'^[l-1](z^[l-1])
3. Return gradients {∂L/∂W^[l], ∂L/∂b^[l]}_{l=1}^L
```

### Vectorized Backpropagation (Batch)

For batch size m:

```
dZ^[L] = ∂L/∂A^[L] ⊙ φ'^[L](Z^[L])
dW^[L] = (1/m) dZ^[L] (A^[L-1])^T
db^[L] = (1/m) Σ_i dZ_i^[L]  (sum over batch)

For l = L-1 to 1:
    dZ^[l] = (W^[l+1])^T dZ^[l+1] ⊙ φ'^[l](Z^[l])
    dW^[l] = (1/m) dZ^[l] (A^[l-1])^T
    db^[l] = (1/m) Σ_i dZ_i^[l]
```

### Computational Complexity

Same as forward pass: O(Σ n^[l] × n^[l-1])

**Important:** Backpropagation is not approximation; it's an exact computation of gradients using the chain rule.

---

## Computational Graphs

### Definition

A computational graph is a directed acyclic graph (DAG) where:
- **Nodes:** Operations or variables
- **Edges:** Data flow (tensors)

### Example: Simple Network

```
x → [Linear] → z1 → [ReLU] → a1 → [Linear] → z2 → [Softmax] → y_hat → [Loss] → L
```

### Automatic Differentiation

Modern frameworks (PyTorch, TensorFlow) use computational graphs for automatic differentiation.

**Forward Mode AD:**
- Computes derivatives along with function values
- Efficient for f: ℝ → ℝ^n

**Reverse Mode AD (Backpropagation):**
- Computes derivatives after forward pass
- Efficient for f: ℝ^n → ℝ (typical in ML)

### PyTorch Autograd Example

```python
import torch

# Enable gradient tracking
x = torch.tensor([2.0], requires_grad=True)
w = torch.tensor([3.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# Forward pass (builds computational graph)
z = w * x + b  # z = 7.0
y = z ** 2     # y = 49.0

# Backward pass (traverse graph in reverse)
y.backward()   # Computes gradients

print(f"dy/dx = {x.grad}")  # 42.0
print(f"dy/dw = {w.grad}")  # 28.0
print(f"dy/db = {b.grad}")  # 14.0
```

**Computational Graph:**
```
    x(2)  w(3)
      \   /
       [*]
        |
        z(6)  b(1)
         \    /
          [+]
           |
          z(7)
           |
         [^2]
           |
          y(49)
```

**Reverse Pass:**
```
dy/dy = 1
dy/dz = dy/dy * 2z = 14
dy/db = dy/dz * 1 = 14
dy/dx = dy/dz * w = 42
dy/dw = dy/dz * x = 28
```

---

## Weight Initialization

### Why Initialization Matters

**Poor initialization leads to:**
- Vanishing gradients (weights too small)
- Exploding gradients (weights too large)
- Symmetry breaking issues (all weights equal)

### Naive Approaches (Don't Use)

**All Zeros:**
```python
W = np.zeros((n_out, n_in))
```
**Problem:** All neurons in the same layer compute identical gradients (symmetry problem)

**Too Large:**
```python
W = np.random.randn(n_out, n_in) * 10
```
**Problem:** Activations saturate, gradients vanish

### Xavier/Glorot Initialization (2010)

**Goal:** Maintain variance of activations and gradients across layers

**Derivation:**
For linear layer z = Wx + b with n_in inputs:
```
Var(z_i) = Var(Σ_j W_ij * x_j)
         = Σ_j Var(W_ij) * Var(x_j)    (assuming independence)
         = n_in * Var(W) * Var(x)
```

To maintain Var(z) = Var(x), we need:
```
Var(W) = 1 / n_in
```

Similarly, considering backward pass:
```
Var(W) = 1 / n_out
```

**Xavier Initialization (compromise):**
```
Var(W) = 2 / (n_in + n_out)

W ~ U[-√(6/(n_in + n_out)), √(6/(n_in + n_out))]  (uniform)
or
W ~ N(0, 2/(n_in + n_out))                          (normal)
```

**When to use:** tanh or sigmoid activations

### He Initialization (2015)

**For ReLU activations:**

ReLU zeros out half the neurons on average, so variance is halved.

**Derivation:**
```
E[ReLU(z)] = E[z | z > 0] * P(z > 0) = E[z | z > 0] / 2

Var(ReLU(z)) ≈ Var(z) / 2  (approximately)
```

To compensate:
```
Var(W) = 2 / n_in

W ~ N(0, 2/n_in)
```

**When to use:** ReLU, Leaky ReLU, and variants

### LeCun Initialization

For SELU activation:
```
W ~ N(0, 1/n_in)
```

### Implementation

```python
import torch.nn as nn

# Xavier (for tanh, sigmoid)
nn.init.xavier_uniform_(layer.weight)
nn.init.xavier_normal_(layer.weight)

# He (for ReLU)
nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')

# LeCun (for SELU)
nn.init.normal_(layer.weight, mean=0, std=1/np.sqrt(n_in))
```

### Initialization for Different Layers

**Linear layers:** Use Xavier or He based on activation
**Convolutional layers:** Use He initialization (typically ReLU)
**Recurrent layers:** Orthogonal initialization for hidden-to-hidden weights
**Biases:** Usually initialized to zeros (or small positive for ReLU)

---

## Bias-Variance Tradeoff in Neural Networks

### Classical ML Perspective

**Bias:** Error from incorrect assumptions (underfitting)
**Variance:** Error from sensitivity to training data fluctuations (overfitting)

**Expected Test Error:**
```
E[(y - y_hat)^2] = Bias^2 + Variance + Irreducible Error
```

### Neural Network Perspective

**Capacity:** Determined by:
- Number of parameters (width × depth)
- Activation function expressiveness
- Training procedure

**Classical wisdom:** More capacity → lower bias, higher variance

**Modern deep learning:** Large networks with proper regularization can achieve low bias AND low variance (double descent phenomenon)

### Double Descent

**Observation:** Test error follows a U-shape in classical regime, but continues to decrease past interpolation threshold.

```
Test Error
    |
    |  \    /
    |   \  /  \
    |    \/    \___
    |
    +-----|-----|------> Model Complexity
       classical  interpolation
       regime     threshold
```

**Explanation:**
1. **Underparameterized:** Classical bias-variance tradeoff
2. **Interpolation threshold:** Model perfectly fits training data
3. **Overparameterized:** Implicit regularization from SGD finds smooth solutions

### Practical Implications (2025)

- Use larger models than classically recommended
- Regularize through dropout, weight decay, data augmentation
- Early stopping less critical with proper regularization
- Train longer with learning rate schedules

---

## Universal Approximation Theorem

### Statement (Classical Version, 1989)

**Theorem (Cybenko, Hornik et al.):** Let φ be a non-constant, bounded, and continuous activation function. Then, for any continuous function f: [0,1]^n → ℝ and any ε > 0, there exists a single-hidden-layer network with finite number of neurons that approximates f within ε:

```
|f(x) - f_NN(x)| < ε  for all x ∈ [0,1]^n
```

### Mathematical Formulation

The approximator has the form:
```
f_NN(x) = Σ_{i=1}^N α_i φ(w_i^T x + b_i)
```

Where N is the number of hidden neurons (may be arbitrarily large).

### Modern Extensions

**Depth vs Width:**
- **Wide shallow networks:** Can approximate any function but may require exponentially many neurons
- **Deep narrow networks:** Can represent same functions with polynomially fewer parameters

**Example (Telgarsky, 2016):** There exist functions that require O(2^n) neurons with depth 2, but only O(n) neurons with depth O(n).

### Approximation vs Optimization vs Generalization

**Three fundamental questions:**

1. **Approximation:** Can a network represent the target function?
   - Answer: Yes (UAT guarantees this)

2. **Optimization:** Can we find the parameters via gradient descent?
   - Answer: Non-convex, but works well in practice

3. **Generalization:** Does the learned function generalize to unseen data?
   - Answer: Requires regularization, proper training

**UAT only addresses approximation!**

### Practical Implications

- Neural networks are extremely expressive
- The challenge is learning, not representation
- Depth helps with efficiency (fewer parameters)
- Modern architectures (ResNets, Transformers) designed for both expressiveness and trainability

---

## Complete Implementations

### NumPy Implementation (Educational)

```python
import numpy as np
from typing import List, Tuple, Callable

class NeuralNetwork:
    """
    Multi-layer perceptron implemented from scratch in NumPy.

    Args:
        layer_dims: List of layer dimensions [input_dim, hidden1, hidden2, ..., output_dim]
        activation: Activation function ('relu', 'tanh', 'sigmoid')
        initialization: Weight initialization ('xavier', 'he')
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = 'relu',
        initialization: str = 'he'
    ):
        self.layer_dims = layer_dims
        self.L = len(layer_dims) - 1  # Number of layers (excluding input)
        self.activation = activation

        # Initialize parameters
        self.parameters = {}
        self._initialize_parameters(initialization)

        # Cache for forward pass
        self.cache = {}

    def _initialize_parameters(self, method: str):
        """Initialize weights and biases."""
        np.random.seed(42)

        for l in range(1, self.L + 1):
            n_in = self.layer_dims[l - 1]
            n_out = self.layer_dims[l]

            if method == 'xavier':
                # Xavier initialization
                scale = np.sqrt(2.0 / (n_in + n_out))
            elif method == 'he':
                # He initialization
                scale = np.sqrt(2.0 / n_in)
            else:
                scale = 0.01

            self.parameters[f'W{l}'] = np.random.randn(n_out, n_in) * scale
            self.parameters[f'b{l}'] = np.zeros((n_out, 1))

    def _activation_forward(self, Z: np.ndarray, activation: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply activation function.

        Returns:
            A: Activated values
            activation_cache: Cache for backward pass
        """
        if activation == 'relu':
            A = np.maximum(0, Z)
        elif activation == 'tanh':
            A = np.tanh(Z)
        elif activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return A, Z

    def _activation_backward(self, dA: np.ndarray, Z: np.ndarray, activation: str) -> np.ndarray:
        """Compute gradient of activation."""
        if activation == 'relu':
            dZ = dA * (Z > 0)
        elif activation == 'tanh':
            A = np.tanh(Z)
            dZ = dA * (1 - A ** 2)
        elif activation == 'sigmoid':
            A = 1 / (1 + np.exp(-Z))
            dZ = dA * A * (1 - A)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        return dZ

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward propagation.

        Args:
            X: Input data, shape (n_features, n_examples)

        Returns:
            AL: Output of network, shape (n_output, n_examples)
        """
        A = X
        self.cache['A0'] = X

        # Hidden layers
        for l in range(1, self.L):
            A_prev = A
            W = self.parameters[f'W{l}']
            b = self.parameters[f'b{l}']

            # Linear forward
            Z = np.dot(W, A_prev) + b

            # Activation forward
            A, _ = self._activation_forward(Z, self.activation)

            # Cache values
            self.cache[f'Z{l}'] = Z
            self.cache[f'A{l}'] = A

        # Output layer (linear activation for regression, sigmoid/softmax for classification)
        W = self.parameters[f'W{self.L}']
        b = self.parameters[f'b{self.L}']
        Z = np.dot(W, A) + b
        AL = Z  # Linear output

        self.cache[f'Z{self.L}'] = Z
        self.cache[f'A{self.L}'] = AL

        return AL

    def backward(self, Y: np.ndarray, AL: np.ndarray) -> dict:
        """
        Backward propagation.

        Args:
            Y: True labels, shape (n_output, n_examples)
            AL: Predicted values, shape (n_output, n_examples)

        Returns:
            gradients: Dictionary of gradients
        """
        m = Y.shape[1]
        gradients = {}

        # Output layer gradient (MSE loss)
        dAL = (AL - Y) / m

        # Output layer (linear)
        dZ = dAL
        A_prev = self.cache[f'A{self.L - 1}']

        gradients[f'dW{self.L}'] = np.dot(dZ, A_prev.T)
        gradients[f'db{self.L}'] = np.sum(dZ, axis=1, keepdims=True)
        dA_prev = np.dot(self.parameters[f'W{self.L}'].T, dZ)

        # Hidden layers
        for l in reversed(range(1, self.L)):
            Z = self.cache[f'Z{l}']
            A_prev = self.cache[f'A{l - 1}']

            # Activation backward
            dZ = self._activation_backward(dA_prev, Z, self.activation)

            # Linear backward
            gradients[f'dW{l}'] = np.dot(dZ, A_prev.T)
            gradients[f'db{l}'] = np.sum(dZ, axis=1, keepdims=True)

            if l > 1:
                dA_prev = np.dot(self.parameters[f'W{l}'].T, dZ)

        return gradients

    def update_parameters(self, gradients: dict, learning_rate: float):
        """Update parameters using gradient descent."""
        for l in range(1, self.L + 1):
            self.parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}']
            self.parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}']

    def compute_loss(self, Y: np.ndarray, AL: np.ndarray) -> float:
        """Compute MSE loss."""
        m = Y.shape[1]
        loss = np.sum((AL - Y) ** 2) / (2 * m)
        return loss

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        epochs: int = 1000,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> List[float]:
        """
        Train the network.

        Args:
            X_train: Training data, shape (n_features, n_examples)
            Y_train: Training labels, shape (n_output, n_examples)
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            verbose: Print progress

        Returns:
            losses: List of losses per epoch
        """
        losses = []

        for epoch in range(epochs):
            # Forward pass
            AL = self.forward(X_train)

            # Compute loss
            loss = self.compute_loss(Y_train, AL)
            losses.append(loss)

            # Backward pass
            gradients = self.backward(Y_train, AL)

            # Update parameters
            self.update_parameters(gradients, learning_rate)

            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}/{epochs} - Loss: {loss:.6f}")

        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.forward(X)


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    X = np.random.randn(2, n_samples)  # 2 features
    Y = (X[0] ** 2 + X[1] ** 2).reshape(1, -1)  # y = x1^2 + x2^2

    # Create and train network
    nn = NeuralNetwork(
        layer_dims=[2, 10, 10, 1],  # 2 inputs, 2 hidden layers (10 neurons each), 1 output
        activation='relu',
        initialization='he'
    )

    losses = nn.train(X, Y, epochs=1000, learning_rate=0.01, verbose=True)

    # Make predictions
    Y_pred = nn.predict(X)
    print(f"\nFinal MSE: {np.mean((Y - Y_pred) ** 2):.6f}")
```

### PyTorch Implementation (Production)

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import List, Optional
import matplotlib.pyplot as plt

class MLP(nn.Module):
    """
    Multi-layer perceptron with configurable architecture.

    Args:
        input_dim: Input feature dimension
        hidden_dims: List of hidden layer dimensions
        output_dim: Output dimension
        activation: Activation function ('relu', 'tanh', 'gelu')
        dropout_rate: Dropout probability (0 = no dropout)
        use_batch_norm: Whether to use batch normalization
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        activation: str = 'relu',
        dropout_rate: float = 0.0,
        use_batch_norm: bool = False
    ):
        super().__init__()

        # Activation function
        self.activation_name = activation
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))

            # Batch normalization (optional)
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))

            # Activation
            layers.append(self.activation)

            # Dropout (optional)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer (no activation, dropout, or batch norm)
        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He or Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self.activation_name == 'relu':
                    # He initialization for ReLU
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                else:
                    # Xavier for tanh/sigmoid
                    nn.init.xavier_normal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class MLPTrainer:
    """
    Trainer for MLP with modern best practices.

    Features:
    - Mixed precision training (2025 standard)
    - Learning rate scheduling
    - Gradient clipping
    - Early stopping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_amp: bool = True
    ):
        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp and device == 'cuda'

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': []
        }

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        optimizer_name: str = 'adamw',
        scheduler_name: Optional[str] = 'cosine',
        clip_grad_norm: Optional[float] = 1.0,
        early_stopping_patience: Optional[int] = 10,
        verbose: bool = True
    ):
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of training epochs
            learning_rate: Initial learning rate
            weight_decay: L2 regularization coefficient
            optimizer_name: 'adam', 'adamw', 'sgd'
            scheduler_name: 'cosine', 'step', None
            clip_grad_norm: Max gradient norm (None = no clipping)
            early_stopping_patience: Patience for early stopping (None = disabled)
            verbose: Print progress
        """
        # Optimizer
        if optimizer_name == 'adam':
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'adamw':
            optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        elif optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        if scheduler_name == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif scheduler_name == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        else:
            scheduler = None

        # Loss function
        criterion = nn.MSELoss()

        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0

        # Training loop
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch_x, batch_y in train_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()

                # Mixed precision forward pass
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)

                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()

                # Gradient clipping
                if clip_grad_norm is not None:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)

                # Optimizer step
                self.scaler.step(optimizer)
                self.scaler.update()

                train_loss += loss.item() * batch_x.size(0)

            train_loss /= len(train_loader.dataset)
            self.history['train_loss'].append(train_loss)

            # Validation phase
            if val_loader is not None:
                val_loss = self.evaluate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)

                # Early stopping check
                if early_stopping_patience is not None:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        torch.save(self.model.state_dict(), 'best_model.pth')
                    else:
                        patience_counter += 1
                        if patience_counter >= early_stopping_patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch}")
                            break

            # Learning rate scheduling
            if scheduler is not None:
                scheduler.step()

            # Print progress
            if verbose and epoch % 10 == 0:
                if val_loader is not None:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
                else:
                    print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Load best model if early stopping was used
        if early_stopping_patience is not None and val_loader is not None:
            self.model.load_state_dict(torch.load('best_model.pth'))

    def evaluate(self, data_loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate model on data."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_x.size(0)

        return total_loss / len(data_loader.dataset)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Make predictions."""
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            predictions = self.model(X)
        return predictions.cpu()

    def plot_history(self):
        """Plot training history."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        if self.history['val_loss']:
            plt.plot(self.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training History')
        plt.grid(True)
        plt.show()


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    torch.manual_seed(42)
    n_samples = 10000
    X = torch.randn(n_samples, 10)  # 10 features
    Y = (X[:, 0] ** 2 + X[:, 1] ** 2 + torch.randn(n_samples) * 0.1).unsqueeze(1)

    # Split into train/val
    train_size = int(0.8 * n_samples)
    X_train, X_val = X[:train_size], X[train_size:]
    Y_train, Y_val = Y[:train_size], Y[train_size:]

    # Create data loaders
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Create model
    model = MLP(
        input_dim=10,
        hidden_dims=[64, 64, 32],
        output_dim=1,
        activation='relu',
        dropout_rate=0.1,
        use_batch_norm=True
    )

    # Train model
    trainer = MLPTrainer(model, use_amp=True)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-4,
        optimizer_name='adamw',
        scheduler_name='cosine',
        early_stopping_patience=10,
        verbose=True
    )

    # Plot results
    trainer.plot_history()

    # Make predictions
    Y_pred = trainer.predict(X_val)
    mse = nn.MSELoss()(Y_pred, Y_val)
    print(f"\nFinal Validation MSE: {mse:.6f}")
```

---

## Practical Considerations

### Gradient Checking

Verify backpropagation implementation using numerical gradients:

```python
def gradient_check(model, X, Y, epsilon=1e-7):
    """
    Verify gradients using finite differences.

    Args:
        model: Neural network model
        X: Input data
        Y: Target labels
        epsilon: Small value for numerical gradient

    Returns:
        relative_error: Relative error between analytical and numerical gradients
    """
    # Compute analytical gradients
    model.zero_grad()
    output = model(X)
    loss = nn.MSELoss()(output, Y)
    loss.backward()

    analytical_grads = []
    for param in model.parameters():
        if param.grad is not None:
            analytical_grads.append(param.grad.clone().flatten())
    analytical_grads = torch.cat(analytical_grads)

    # Compute numerical gradients
    numerical_grads = []
    for param in model.parameters():
        param_flat = param.data.flatten()
        param_grad = torch.zeros_like(param_flat)

        for i in range(len(param_flat)):
            # f(x + epsilon)
            param_flat[i] += epsilon
            output_plus = model(X)
            loss_plus = nn.MSELoss()(output_plus, Y).item()

            # f(x - epsilon)
            param_flat[i] -= 2 * epsilon
            output_minus = model(X)
            loss_minus = nn.MSELoss()(output_minus, Y).item()

            # Numerical gradient
            param_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

            # Restore original value
            param_flat[i] += epsilon

        numerical_grads.append(param_grad)

    numerical_grads = torch.cat(numerical_grads)

    # Compute relative error
    numerator = torch.norm(analytical_grads - numerical_grads)
    denominator = torch.norm(analytical_grads) + torch.norm(numerical_grads)
    relative_error = numerator / denominator

    return relative_error.item()

# Usage
# error = gradient_check(model, X_small, Y_small)
# print(f"Relative error: {error}")  # Should be < 1e-7
```

### Debugging Neural Networks

**Common issues and solutions:**

1. **Loss is NaN:**
   - Check for numerical instability (e.g., log(0), division by zero)
   - Reduce learning rate
   - Enable gradient clipping
   - Check data for NaN/Inf values

2. **Loss doesn't decrease:**
   - Verify backpropagation (gradient checking)
   - Try different learning rates (learning rate finder)
   - Check data preprocessing (normalization)
   - Verify loss function is appropriate

3. **Training loss decreases but validation loss increases:**
   - Overfitting: add regularization (dropout, weight decay)
   - Use data augmentation
   - Reduce model capacity
   - Early stopping

4. **Gradients vanish/explode:**
   - Use proper initialization (He for ReLU)
   - Use batch normalization
   - Gradient clipping
   - Consider residual connections

### Performance Optimization

**2025 Best Practices:**

1. **Use mixed precision training** (FP16/BF16)
2. **Enable cuDNN benchmarking** for fixed input sizes
3. **Use DataLoader with multiple workers**
4. **Pin memory for faster GPU transfer**
5. **Compile models** (PyTorch 2.0+)

```python
# Enable cuDNN benchmarking
torch.backends.cudnn.benchmark = True

# Compile model (PyTorch 2.0+)
model = torch.compile(model)

# Efficient data loading
train_loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)
```

---

## Summary

Neural networks are powerful function approximators built on:

1. **Mathematical foundations:** Linear transformations and non-linear activations
2. **Backpropagation:** Efficient gradient computation via the chain rule
3. **Proper initialization:** Xavier/He for stable training
4. **Modern practices:** Mixed precision, adaptive optimizers, regularization

**Key takeaways:**
- Understand the math: backpropagation is not magic, it's calculus
- Initialization matters: use He for ReLU, Xavier for tanh/sigmoid
- Deep networks can represent complex functions efficiently
- Training deep networks requires careful engineering

**Next steps:**
- Study activation functions in detail (File 14)
- Master optimization algorithms (File 15)
- Learn regularization techniques (File 16)
- Implement production training pipelines (File 17)

---

## References

1. Rumelhart, Hinton & Williams (1986). "Learning representations by back-propagating errors"
2. Glorot & Bengio (2010). "Understanding the difficulty of training deep feedforward neural networks"
3. He et al. (2015). "Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
4. Cybenko (1989). "Approximation by superpositions of a sigmoidal function"
5. Hornik, Stinchcombe & White (1989). "Multilayer feedforward networks are universal approximators"
6. Telgarsky (2016). "Benefits of depth in neural networks"
