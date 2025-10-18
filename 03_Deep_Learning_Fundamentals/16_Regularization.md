# Regularization in Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Classical Regularization](#classical-regularization)
3. [Dropout and Variants](#dropout-and-variants)
4. [Normalization Techniques](#normalization-techniques)
5. [Data Augmentation](#data-augmentation)
6. [Modern Regularization Techniques](#modern-regularization-techniques)
7. [Domain-Specific Regularization](#domain-specific-regularization)
8. [Combining Regularization Methods](#combining-regularization-methods)
9. [Complete Implementations](#complete-implementations)
10. [2025 Best Practices](#2025-best-practices)

---

## Introduction

**Regularization** prevents overfitting by constraining model complexity or adding noise during training.

**The fundamental tradeoff:**
```
Test Error = Bias² + Variance + Irreducible Error

Regularization ↑ → Bias ↑, Variance ↓
```

**Goal:** Find the sweet spot where test error is minimized.

### Why Deep Networks Need Regularization

Deep networks have millions of parameters and can easily memorize training data:

**Capacity to overfit:**
- 1M parameters can memorize 1M training examples
- Modern networks: 100M - 1B parameters
- Training sets: 10K - 100M examples

**Surprising observation (Zhang et al., 2017):**
Neural networks can fit **random labels** perfectly, but generalize poorly.

**Conclusion:** Generalization requires regularization, not just capacity control.

---

## Classical Regularization

### L2 Regularization (Weight Decay)

**Objective function:**
```
L_total(θ) = L(θ) + (λ/2) ||θ||²
```

Where:
- **L(θ):** Original loss function
- **λ:** Regularization strength
- **||θ||²:** Sum of squared weights

**Gradient:**
```
∇L_total = ∇L(θ) + λθ
```

**Update rule:**
```
θ_{t+1} = θ_t - α(∇L(θ_t) + λθ_t)
        = (1 - αλ)θ_t - α∇L(θ_t)
```

**Interpretation:** Weights "decay" toward zero by factor (1 - αλ) each step.

**Effect:**
- Penalizes large weights
- Prefers smooth functions
- Bayesian interpretation: Gaussian prior on weights

**PyTorch implementation:**
```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4  # λ = 0.0001
)
```

**Note:** For AdamW, weight decay is decoupled (recommended in 2025).

### L1 Regularization (Lasso)

**Objective function:**
```
L_total(θ) = L(θ) + λ ||θ||₁
```

Where ||θ||₁ = Σ|θᵢ|

**Gradient (subgradient):**
```
∇L_total = ∇L(θ) + λ sign(θ)
```

**Effect:**
- Promotes sparsity (many weights → 0)
- Feature selection
- Bayesian interpretation: Laplacian prior

**When to use:**
- When you want sparse weights
- Feature selection
- Interpretability

**PyTorch implementation:**
```python
# L1 must be added manually
l1_lambda = 1e-5
l1_norm = sum(p.abs().sum() for p in model.parameters())
loss = loss + l1_lambda * l1_norm
```

### Elastic Net (L1 + L2)

**Objective function:**
```
L_total(θ) = L(θ) + λ₁||θ||₁ + λ₂||θ||²
```

**Benefits:**
- Combines sparsity (L1) and smoothness (L2)
- More stable than pure L1

### Early Stopping

**Idea:** Stop training when validation error stops decreasing.

**Algorithm:**
```
1. Split data into train/validation
2. Train model
3. After each epoch:
   - Evaluate on validation set
   - If val_loss improved: save model
   - Else: increment patience counter
4. If patience counter >= max_patience: stop
5. Restore best model
```

**Why it works:**
- Training error decreases monotonically
- Validation error decreases then increases (overfitting)
- Early stopping finds the sweet spot

**Implementation:**
```python
best_val_loss = float('inf')
patience = 0
max_patience = 10

for epoch in range(max_epochs):
    train_loss = train_epoch(...)
    val_loss = evaluate(...)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience = 0
    else:
        patience += 1
        if patience >= max_patience:
            break

load_checkpoint(model)  # Restore best model
```

**Advantages:**
- Simple and effective
- No hyperparameters (except patience)
- Universal (works with any model)

**Disadvantages:**
- Requires validation set
- May stop too early (noisy validation)
- Not suitable for online learning

---

## Dropout and Variants

### Standard Dropout

**Idea:** Randomly drop neurons during training.

**Training:**
```
For each training example:
    For each layer:
        For each neuron:
            Keep neuron with probability p
            Drop (set to 0) with probability (1-p)
```

**Mathematical formulation:**
```
# Training
r ~ Bernoulli(p)  # Dropout mask
y = r ⊙ (Wx + b)   # Element-wise multiplication

# Testing (no dropout)
y = p(Wx + b)      # Scale by p
```

**Inverted Dropout (recommended):**
```
# Training (scale during training)
r ~ Bernoulli(p)
y = (r ⊙ (Wx + b)) / p

# Testing (no scaling needed)
y = Wx + b
```

**Why dropout works:**

1. **Ensemble interpretation:** Training exponentially many networks, testing averages them
2. **Co-adaptation prevention:** Forces neurons to be robust (can't rely on specific other neurons)
3. **Noise regularization:** Adds noise to activations
4. **Implicit model averaging:** Approximates Bayesian model averaging

**Theoretical analysis:**

Expected output during training:
```
E[y] = E[r ⊙ (Wx + b) / p]
     = (Wx + b) E[r] / p
     = (Wx + b) p / p
     = Wx + b
```

Same as test time! (This is why we scale by 1/p)

**Dropout rate selection:**

| Layer Type | Typical Dropout Rate |
|------------|---------------------|
| Input layer | 0.1 - 0.2 |
| Hidden layers (small networks) | 0.5 |
| Hidden layers (large networks) | 0.3 - 0.4 |
| Output layer | 0 (no dropout) |
| Recurrent connections | 0.2 - 0.3 |

**PyTorch implementation:**
```python
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # Applied only during training
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # No dropout on output
        return x

# Training: model.train() enables dropout
# Testing: model.eval() disables dropout
```

**Advantages:**
- Very effective regularization
- Works across many architectures
- Reduces co-adaptation

**Disadvantages:**
- Increases training time (needs more epochs)
- Not suitable for small datasets
- Can hurt performance if too aggressive

### Spatial Dropout (Dropout2D, Dropout3D)

**Problem with standard dropout in CNNs:**
Adjacent pixels are highly correlated, so dropping individual pixels may not be effective.

**Solution:** Drop entire feature maps.

**Implementation:**
```python
# For conv layers with shape (batch, channels, height, width)
nn.Dropout2d(p=0.5)  # Drops entire feature maps

# Example
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.dropout1 = nn.Dropout2d(0.3)  # Spatial dropout
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)  # Drops entire 64 feature maps randomly
        x = F.relu(self.conv2(x))
        x = self.dropout2(x)
        x = x.mean([2, 3])  # Global average pooling
        x = self.fc(x)
        return x
```

**When to use:** CNNs, especially early layers.

### DropConnect

**Idea:** Drop connections (weights) instead of neurons.

**Implementation:**
```
# Dropout: r ⊙ (Wx)
# DropConnect: (r ⊙ W)x

where r is the dropout mask
```

**Effect:** More general than dropout (dropout is special case).

**When to use:** Fully connected layers, can be more effective than dropout.

### DropBlock

**Idea:** Drop contiguous regions (blocks) in feature maps.

**Motivation:**
- Standard dropout: individual pixels
- Spatial dropout: entire feature maps
- DropBlock: contiguous regions (middle ground)

**Algorithm:**
```
1. Sample random locations
2. For each location, drop a block_size × block_size region
3. Keep proportion of activations equal to (1 - drop_rate)
```

**When to use:** CNNs, especially ResNets (2025 recommendation).

**Implementation:**
```python
class DropBlock2D(nn.Module):
    def __init__(self, drop_prob, block_size):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0:
            return x

        # Sample mask
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)

        # Expand mask to blocks
        mask = F.max_pool2d(
            mask,
            kernel_size=self.block_size,
            stride=1,
            padding=self.block_size // 2
        )

        # Invert mask (1 = keep, 0 = drop)
        mask = 1 - mask

        # Normalize and apply
        normalize_factor = mask.numel() / mask.sum()
        return x * mask * normalize_factor
```

---

## Normalization Techniques

### Batch Normalization (BatchNorm)

**Idea:** Normalize activations across the batch dimension.

**Algorithm (training):**
```
# For mini-batch B = {x₁, ..., xₘ}

1. Compute batch statistics:
   μ_B = (1/m) Σᵢ xᵢ
   σ²_B = (1/m) Σᵢ (xᵢ - μ_B)²

2. Normalize:
   x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)

3. Scale and shift (learnable):
   yᵢ = γx̂ᵢ + β
```

Where:
- **γ, β:** Learnable parameters (scale and shift)
- **ε:** Small constant for numerical stability (1e-5)

**Algorithm (testing):**
```
Use running averages computed during training:
   μ_test = moving_avg(μ_B)
   σ²_test = moving_avg(σ²_B)

   y = γ(x - μ_test) / √(σ²_test + ε) + β
```

**Why BatchNorm works:**

1. **Reduces internal covariate shift:** Stabilizes distribution of layer inputs
2. **Allows higher learning rates:** Gradient flow is more stable
3. **Regularization effect:** Adds noise (batch statistics vary)
4. **Reduces sensitivity to initialization:** Normalizes activations

**Mathematical properties:**

**Gradient flow:**
```
∂L/∂x = ∂L/∂x̂ * (1/√(σ² + ε)) * (I - (1/m) - (x̂)(x̂)ᵀ/m)
```

The (I - ...) term prevents gradient explosion.

**PyTorch implementation:**
```python
# For fully connected layers
nn.BatchNorm1d(num_features)

# For convolutional layers (normalizes each channel)
nn.BatchNorm2d(num_channels)

# Example
class BatchNormNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)  # BatchNorm AFTER linear, BEFORE activation
        x = F.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.fc3(x)  # No BN on output layer
        return x
```

**Placement:** BatchNorm typically placed **after** linear/conv layer, **before** activation function.

**Advantages:**
- Faster convergence (can use higher learning rates)
- Reduces sensitivity to initialization
- Acts as regularization
- Standard for CNNs

**Disadvantages:**
- Behavior differs between train/test (requires running statistics)
- Sensitive to batch size (small batches → noisy statistics)
- Not suitable for RNNs (LayerNorm preferred)
- Can hurt performance in some cases (GANs discriminator)

**Batch size considerations:**
- **Large batches (>= 32):** Works well
- **Small batches (<= 16):** Consider Group Normalization
- **Batch size = 1:** Use Layer/Instance/Group Normalization

### Layer Normalization (LayerNorm)

**Idea:** Normalize across the feature dimension (not batch dimension).

**Algorithm:**
```
# For input x with shape (batch, features)

1. Compute statistics for each example:
   μ = (1/d) Σⱼ xⱼ         # Mean across features
   σ² = (1/d) Σⱼ (xⱼ - μ)²  # Variance across features

2. Normalize:
   x̂ = (x - μ) / √(σ² + ε)

3. Scale and shift:
   y = γx̂ + β
```

**Key difference from BatchNorm:**
- **BatchNorm:** Normalizes across batch dimension (different examples)
- **LayerNorm:** Normalizes across feature dimension (single example)

**Why LayerNorm for transformers:**

1. **Batch-independent:** Same computation for train/test
2. **Sequence length invariant:** Works with variable-length sequences
3. **Better for RNNs/Transformers:** No batch statistics to track

**PyTorch implementation:**
```python
nn.LayerNorm(normalized_shape)

# Example
class TransformerBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attention = MultiHeadAttention(d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Pre-LN (modern transformer architecture)
        x = x + self.attention(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
```

**When to use:**
- **Transformers:** Standard choice (2025)
- **RNNs, LSTMs:** Better than BatchNorm
- **Small batch sizes:** Independent of batch
- **Reinforcement learning:** Batch size often 1

### Instance Normalization

**Idea:** Normalize each feature map independently.

**Algorithm:**
```
# For input x with shape (batch, channels, height, width)

Normalize each (batch_i, channel_j) independently:
   μᵢⱼ = mean over (height, width)
   σ²ᵢⱼ = variance over (height, width)

   x̂ᵢⱼ = (xᵢⱼ - μᵢⱼ) / √(σ²ᵢⱼ + ε)
```

**When to use:**
- **Style transfer:** Removes instance-specific contrast
- **GANs:** Sometimes better than BatchNorm

**PyTorch implementation:**
```python
nn.InstanceNorm2d(num_channels)
```

### Group Normalization

**Idea:** Divide channels into groups and normalize within each group.

**Algorithm:**
```
# For input x with shape (batch, channels, height, width)

1. Divide C channels into G groups (each group has C/G channels)
2. Normalize within each group:
   Compute mean and variance over (C/G, H, W) dimensions
   Normalize using these statistics
```

**Special cases:**
- **G = 1:** Layer Normalization
- **G = C:** Instance Normalization

**Advantages:**
- Works well with small batches
- Batch-independent (like LayerNorm)
- Better than BatchNorm for small batches

**PyTorch implementation:**
```python
nn.GroupNorm(num_groups, num_channels)

# Example: 32 channels, 8 groups (4 channels per group)
nn.GroupNorm(8, 32)
```

**When to use:**
- **Small batch training:** Better than BatchNorm
- **Object detection/segmentation:** Standard in modern architectures
- **When batch size is limited:** ResNet with batch size < 16

### Normalization Comparison

| Technique | Normalizes Over | Batch-Independent | Use Case |
|-----------|----------------|-------------------|----------|
| **BatchNorm** | (batch, height, width) | ❌ | CNNs with large batches |
| **LayerNorm** | (features) | ✅ | Transformers, RNNs |
| **InstanceNorm** | (height, width) | ✅ | Style transfer, GANs |
| **GroupNorm** | (group, height, width) | ✅ | Small batch training |

**2025 Recommendation:**
- **CNNs (batch >= 32):** BatchNorm
- **CNNs (batch < 32):** GroupNorm
- **Transformers:** LayerNorm (pre-normalization)
- **RNNs:** LayerNorm
- **Style transfer:** InstanceNorm

---

## Data Augmentation

### Why Data Augmentation?

**Problem:** Limited training data → overfitting

**Solution:** Create new training examples by applying transformations.

**Benefits:**
1. Increases effective dataset size
2. Improves generalization
3. Acts as strong regularization
4. Introduces invariances

**Key principle:** Augmentations should preserve label (or modify it appropriately).

### Computer Vision Augmentations

**Geometric transformations:**
```python
import torchvision.transforms as transforms

# Standard augmentations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Common transformations:**

| Transformation | Effect | When to Use |
|---------------|--------|-------------|
| **Horizontal Flip** | Mirror image | Almost always (not for text) |
| **Rotation** | Rotate ±15° | Natural images |
| **Crop** | Random crops | Almost always |
| **Color Jitter** | Adjust brightness/contrast | Natural images |
| **Gaussian Blur** | Blur image | Robustness to blur |
| **Cutout** | Zero out random patches | Prevent texture bias |
| **Random Erasing** | Erase random rectangles | Occlusion robustness |

**Advanced augmentations (2025):**

**RandAugment:**
```python
from torchvision.transforms import RandAugment

# Automatically selects and applies random augmentations
transform = RandAugment(num_ops=2, magnitude=9)
```

**AutoAugment:**
Learned augmentation policy via reinforcement learning.

**TrivialAugment:**
Simplified, equally effective as AutoAugment.

### NLP Augmentations

**Text-specific augmentations:**

1. **Synonym replacement:** Replace words with synonyms
2. **Random insertion:** Insert random words
3. **Random swap:** Swap word positions
4. **Random deletion:** Delete words randomly
5. **Back-translation:** Translate to another language and back

**Example:**
```python
import nlpaug.augmenter.word as naw

# Synonym replacement using WordNet
aug = naw.SynonymAug(aug_src='wordnet')
text = "The quick brown fox jumps over the lazy dog"
augmented = aug.augment(text)
# "The speedy brown fox jumps over the lazy dog"
```

**2025 NLP Augmentation:**
- **Paraphrasing models:** Use T5/GPT to generate paraphrases
- **Masked language modeling:** Replace tokens with [MASK] and predict

---

## Modern Regularization Techniques

### Label Smoothing

**Problem:** Hard labels (one-hot) can lead to overconfidence.

**Example (3 classes):**
```
Hard label: [0, 1, 0]  # 100% confident
```

**Solution:** Smooth labels toward uniform distribution.

**Label smoothing formula:**
```
y_smooth = (1 - ε) * y_hard + ε / K

where:
  ε: smoothing parameter (typically 0.1)
  K: number of classes
```

**Example (K=3, ε=0.1):**
```
Hard label:   [0, 1, 0]
Smooth label: [0.033, 0.933, 0.033]
```

**Why it works:**
1. Prevents overconfidence
2. Calibrates probabilities better
3. Improves generalization

**PyTorch implementation:**
```python
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        """
        Args:
            pred: (batch_size, num_classes) logits
            target: (batch_size,) class indices
        """
        n_classes = pred.size(-1)
        log_probs = F.log_softmax(pred, dim=-1)

        # Convert target to one-hot
        target_one_hot = torch.zeros_like(log_probs).scatter_(
            1, target.unsqueeze(1), 1
        )

        # Apply label smoothing
        target_smooth = (1 - self.epsilon) * target_one_hot + self.epsilon / n_classes

        # Compute loss
        loss = -(target_smooth * log_probs).sum(dim=-1).mean()
        return loss

# Usage
criterion = LabelSmoothingCrossEntropy(epsilon=0.1)
loss = criterion(logits, targets)
```

**When to use:**
- Classification tasks
- When model is overconfident
- Modern image classification (2025 standard)

**Typical values:** ε = 0.1 for most tasks

### Mixup

**Idea:** Create virtual training examples by mixing two examples.

**Algorithm:**
```
1. Sample two examples: (x₁, y₁), (x₂, y₂)
2. Sample mixing coefficient: λ ~ Beta(α, α)
3. Create mixed example:
   x_mix = λx₁ + (1-λ)x₂
   y_mix = λy₁ + (1-λ)y₂
```

**Typical hyperparameter:** α = 0.2 or α = 1.0

**Effect:**
- Encourages linear behavior between examples
- Regularizes decision boundaries
- Improves calibration

**PyTorch implementation:**
```python
def mixup_data(x, y, alpha=1.0):
    """
    Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Mixup loss function"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Usage in training loop
for x, y in train_loader:
    x, y_a, y_b, lam = mixup_data(x, y, alpha=1.0)
    output = model(x)
    loss = mixup_criterion(criterion, output, y_a, y_b, lam)
    loss.backward()
    optimizer.step()
```

**Advantages:**
- Improves generalization significantly
- Better calibration
- Reduces overfitting

**Disadvantages:**
- Generates unrealistic examples (blend of two images)
- May slow convergence

**When to use:** Image classification (2025 standard for vision)

### CutMix

**Idea:** Instead of blending entire images, cut and paste regions.

**Algorithm:**
```
1. Sample two examples: (x₁, y₁), (x₂, y₂)
2. Sample bounding box B (region to cut)
3. Create mixed example:
   x_mix = x₁ with region B replaced by corresponding region from x₂
   y_mix = λy₁ + (1-λ)y₂  where λ = area(B) / area(image)
```

**PyTorch implementation:**
```python
def cutmix_data(x, y, alpha=1.0):
    """
    CutMix augmentation
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Sample bounding box
    W = x.size(2)
    H = x.size(3)
    cut_ratio = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_ratio)
    cut_h = np.int(H * cut_ratio)

    # Uniform sampling
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # Apply CutMix
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # Adjust lambda to exactly match the area
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam
```

**Advantages:**
- More realistic than Mixup (no blending)
- Forces model to localize objects
- Better than Mixup in many cases

**When to use:** Image classification (2025 best practice)

### CutOut

**Idea:** Randomly mask out square regions of the input during training.

**Effect:**
- Forces model to use entire image (not just discriminative parts)
- Robustness to occlusion

**PyTorch implementation:**
```python
class Cutout(object):
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img: Tensor image of size (C, H, W)
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1:y2, x1:x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

# Usage
transform = transforms.Compose([
    transforms.ToTensor(),
    Cutout(n_holes=1, length=16)
])
```

**When to use:** Image classification, especially CIFAR-10/100

---

## Domain-Specific Regularization

### Stochastic Depth (for ResNets)

**Idea:** Randomly drop entire residual blocks during training.

**Effect:**
- Reduces training time
- Acts as ensemble of shallow networks
- Improves generalization

**Implementation:**
```python
class StochasticDepth(nn.Module):
    def __init__(self, survival_prob=0.8):
        super().__init__()
        self.survival_prob = survival_prob

    def forward(self, x):
        if not self.training:
            return x

        # Randomly drop the entire residual branch
        if torch.rand(1).item() > self.survival_prob:
            return torch.zeros_like(x)
        else:
            return x / self.survival_prob  # Scale to maintain expectation
```

**When to use:** Very deep ResNets (>50 layers)

### Spectral Normalization (for GANs)

**Idea:** Normalize weights by their spectral norm (largest singular value).

**Effect:**
- Stabilizes GAN training
- Lipschitz constraint on discriminator

**PyTorch implementation:**
```python
from torch.nn.utils import spectral_norm

# Apply to any layer
layer = spectral_norm(nn.Conv2d(3, 64, 3))
```

**When to use:** GAN discriminators (2025 standard)

---

## Combining Regularization Methods

### Effective Combinations

**Standard CNN (2025):**
```python
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.BatchNorm2d(64),  # Normalization
    nn.ReLU(),
    nn.Dropout2d(0.1),   # Dropout
    # ... more layers
)

# Training
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Weight decay
# + Data augmentation (Mixup/CutMix)
# + Label smoothing
```

**Transformer (2025):**
```python
class TransformerLayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)  # Layer normalization
        self.attention = MultiHeadAttention(d_model)
        self.dropout1 = nn.Dropout(dropout)  # Dropout
        # ...

# Training
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)  # Weight decay
# + Warm-up learning rate schedule
# + Gradient clipping
# + Label smoothing (for classification)
```

### How Much Regularization?

**Guidelines:**

**Small datasets (< 10K examples):**
- Strong regularization needed
- High dropout (0.5)
- Strong weight decay (1e-3)
- Heavy data augmentation
- Early stopping

**Medium datasets (10K - 1M examples):**
- Moderate regularization
- Moderate dropout (0.3)
- Moderate weight decay (1e-4)
- Standard data augmentation
- Optional early stopping

**Large datasets (> 1M examples):**
- Light regularization
- Light dropout (0.1-0.2)
- Light weight decay (1e-4)
- Light data augmentation
- Train to convergence

**2025 Best Practice:**
Start with moderate regularization, increase if overfitting, decrease if underfitting.

---

## Complete Implementations

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

class RegularizedCNN(nn.Module):
    """
    CNN with all modern regularization techniques.

    Features:
    - Batch Normalization
    - Dropout
    - Label Smoothing
    - Mixup/CutMix support
    """

    def __init__(
        self,
        num_classes=10,
        dropout_rate=0.3,
        use_batchnorm=True
    ):
        super().__init__()

        # Convolutional layers with BatchNorm and Dropout
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.dropout1 = nn.Dropout2d(dropout_rate)

        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.dropout2 = nn.Dropout2d(dropout_rate)

        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Global average pooling (reduces parameters, acts as regularization)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout3(x)

        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.fc(x)
        return x


class RegularizedTrainer:
    """
    Trainer with modern regularization techniques.

    Features:
    - Mixup/CutMix
    - Label smoothing
    - Weight decay (via AdamW)
    - Early stopping
    """

    def __init__(
        self,
        model,
        device='cuda',
        use_mixup=True,
        use_cutmix=True,
        mixup_alpha=1.0,
        cutmix_alpha=1.0,
        label_smoothing=0.1
    ):
        self.model = model.to(device)
        self.device = device
        self.use_mixup = use_mixup
        self.use_cutmix = use_cutmix
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

        # Loss with label smoothing
        self.criterion = LabelSmoothingCrossEntropy(epsilon=label_smoothing)

        # History
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    def mixup_data(self, x, y):
        """Apply Mixup augmentation"""
        if self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def cutmix_data(self, x, y):
        """Apply CutMix augmentation"""
        if self.cutmix_alpha > 0:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        W, H = x.size(2), x.size(3)
        cut_ratio = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_ratio)
        cut_h = np.int(H * cut_ratio)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))

        y_a, y_b = y, y[index]
        return x, y_a, y_b, lam

    def train_epoch(self, train_loader, optimizer):
        """Train for one epoch with augmentations"""
        self.model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(self.device), y.to(self.device)

            # Randomly choose augmentation
            r = np.random.rand()
            if self.use_mixup and r < 0.5:
                x, y_a, y_b, lam = self.mixup_data(x, y)
                output = self.model(x)
                loss = lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)
            elif self.use_cutmix and r >= 0.5:
                x, y_a, y_b, lam = self.cutmix_data(x, y)
                output = self.model(x)
                loss = lam * self.criterion(output, y_a) + (1 - lam) * self.criterion(output, y_b)
            else:
                output = self.model(x)
                loss = self.criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        return total_loss / len(train_loader.dataset)

    @torch.no_grad()
    def evaluate(self, val_loader):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        for x, y in val_loader:
            x, y = x.to(self.device), y.to(self.device)
            output = self.model(x)
            loss = F.cross_entropy(output, y)  # Standard CE for evaluation
            total_loss += loss.item() * x.size(0)

            _, predicted = output.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return total_loss / total, 100. * correct / total


# Example usage
if __name__ == "__main__":
    # Model with regularization
    model = RegularizedCNN(
        num_classes=10,
        dropout_rate=0.3,
        use_batchnorm=True
    )

    # Trainer with augmentations
    trainer = RegularizedTrainer(
        model,
        device='cuda',
        use_mixup=True,
        use_cutmix=True,
        label_smoothing=0.1
    )

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-4
    )

    print("Regularized model ready for training!")
```

---

## 2025 Best Practices

### Computer Vision

**Standard recipe:**
```python
# Model
- BatchNorm after each conv/linear layer
- Dropout2d(0.3) for conv layers
- Dropout(0.5) for fully connected layers

# Training
- Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
- Data augmentation: RandAugment or AutoAugment
- Mixup OR CutMix (α=1.0)
- Label smoothing (ε=0.1)
- Learning rate: Cosine annealing with warm-up
```

### Transformers / NLP

**Standard recipe:**
```python
# Model
- LayerNorm (pre-normalization)
- Dropout(0.1) in attention and FFN
- No BatchNorm

# Training
- Optimizer: AdamW (lr=1e-4, weight_decay=0.01)
- Warm-up: 10% of total steps
- Learning rate: Linear decay after warm-up
- Gradient clipping: max_norm=1.0
- Label smoothing (ε=0.1) for classification
```

### Summary

**Most important regularization techniques (2025):**

1. **Weight decay via AdamW:** Always use
2. **Data augmentation:** Essential for vision
3. **Dropout:** Standard for MLPs, use sparingly for CNNs/Transformers
4. **Normalization:** BatchNorm for CNNs, LayerNorm for Transformers
5. **Mixup/CutMix:** State-of-the-art for image classification
6. **Label smoothing:** Improves calibration

**Key insight:** Combine multiple regularization methods for best results. Modern models use ALL of these techniques together.

---

## References

1. Srivastava et al. (2014). "Dropout: A Simple Way to Prevent Neural Networks from Overfitting"
2. Ioffe & Szegedy (2015). "Batch Normalization: Accelerating Deep Network Training"
3. Ba et al. (2016). "Layer Normalization"
4. Wu & He (2018). "Group Normalization"
5. Zhang et al. (2018). "mixup: Beyond Empirical Risk Minimization"
6. Yun et al. (2019). "CutMix: Regularization Strategy to Train Strong Classifiers"
7. DeVries & Taylor (2017). "Improved Regularization of Convolutional Neural Networks with Cutout"
8. Szegedy et al. (2016). "Rethinking the Inception Architecture" (Label Smoothing)
