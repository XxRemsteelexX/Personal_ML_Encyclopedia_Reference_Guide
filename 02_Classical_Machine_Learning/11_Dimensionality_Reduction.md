# Dimensionality Reduction

## Table of Contents
1. [Introduction](#introduction)
2. [Principal Component Analysis (PCA)](#principal-component-analysis-pca)
3. [t-SNE](#t-sne)
4. [UMAP](#umap)
5. [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
6. [Autoencoders](#autoencoders)
7. [Comparison and Selection](#comparison-and-selection)
8. [Practical Applications](#practical-applications)

## Introduction

Dimensionality reduction transforms high-dimensional data into lower dimensions while preserving important structure. Critical for:

- **Visualization**: Reduce to 2D or 3D for plotting
- **Preprocessing**: Remove noise and redundancy before ML
- **Feature engineering**: Create compact representations
- **Computational efficiency**: Reduce memory and training time
- **Curse of dimensionality**: Improve model performance in high dimensions

### The Curse of Dimensionality

**Problem**: As dimensions increase, data becomes increasingly sparse

```
Volume of unit hypersphere in d dimensions:
V_d ∝ r^d / d!

For r=1:
- d=2: π ≈ 3.14
- d=10: 2.55
- d=100: 10^-40  (essentially zero!)
```

**Consequences**:
- Distance metrics become meaningless (all points equidistant)
- Sample density decreases exponentially
- More data needed to maintain statistical power
- Models overfit easily

### Intrinsic Dimensionality

**Key Insight**: Real-world high-dimensional data often lies on lower-dimensional manifold

**Example**: Images of faces
- Raw pixels: 100×100 = 10,000 dimensions
- Intrinsic dimensions: ~50-100 (pose, lighting, expression, identity)

### Types of Dimensionality Reduction

1. **Linear Methods**
   - PCA: Maximize variance
   - LDA: Maximize class separability
   - Fast, interpretable, but limited expressiveness

2. **Non-linear Methods**
   - t-SNE: Preserve local structure for visualization
   - UMAP: Preserve both local and global structure
   - Autoencoders: Learn complex non-linear mappings

## Principal Component Analysis (PCA)

### Mathematical Foundation

**Objective**: Find orthogonal directions (principal components) that maximize variance

**Formulation**:
```
Given data matrix X (n × d), center columns:
X̃ = X - mean(X)

Covariance matrix:
C = (1/n) X̃ᵀ X̃

Find eigenvectors v and eigenvalues λ:
C v = λ v

Principal components: eigenvectors sorted by eigenvalues
PC₁ has largest λ (most variance), PC₂ second largest, etc.
```

### Algorithm

```
1. Standardize data: X̃ = (X - μ) / σ

2. Compute covariance matrix:
   C = X̃ᵀ X̃ / (n-1)

3. Eigendecomposition:
   C = V Λ Vᵀ
   Where V = eigenvectors, Λ = diagonal(eigenvalues)

4. Select k principal components:
   W = [v₁, v₂, ..., vₖ]  (top k eigenvectors)

5. Transform data:
   Z = X̃ W  (n × k)

6. Reconstruction (optional):
   X̂ = Z Wᵀ + μ
```

### Variance Explained

**How much information retained?**

```
Variance explained by PC_i = λᵢ / Σⱼ λⱼ

Cumulative variance = Σᵢ₌₁ᵏ λᵢ / Σⱼ λⱼ
```

**Rule of thumb**: Keep components explaining 95% of variance

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits

# Load high-dimensional data
digits = load_digits()
X = digits.data  # 1797 samples × 64 features (8×8 images)
y = digits.target

print(f"Original shape: {X.shape}")

# CRITICAL: Standardize before PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=64)  # All components initially
pca.fit(X_scaled)

# Analyze variance explained
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nVariance Explained:")
print(f"  First 10 PCs: {cumulative_variance[9]:.4f}")
print(f"  First 20 PCs: {cumulative_variance[19]:.4f}")
print(f"  First 30 PCs: {cumulative_variance[29]:.4f}")

# Plot variance explained
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Individual variance
axes[0].bar(range(1, 21), explained_variance[:20])
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Variance Explained')
axes[0].set_title('Variance Explained by Each PC')
axes[0].set_xticks(range(1, 21))

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'bo-')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[1].axhline(y=0.99, color='g', linestyle='--', label='99% variance')
axes[1].set_xlabel('Number of Components')
axes[1].set_ylabel('Cumulative Variance Explained')
axes[1].set_title('Cumulative Variance Explained')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Find components for 95% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\nComponents for 95% variance: {n_components_95}")
print(f"Compression ratio: {X.shape[1] / n_components_95:.2f}x")
```

### Scree Plot

**Visual method** to choose number of components

```python
def scree_plot(pca):
    """Create scree plot for PCA."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot eigenvalues
    eigenvalues = pca.explained_variance_
    ax.plot(range(1, len(eigenvalues) + 1), eigenvalues, 'bo-')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Eigenvalue (Variance)')
    ax.set_title('Scree Plot')
    ax.grid(True, alpha=0.3)

    # Find elbow using second derivative
    if len(eigenvalues) > 2:
        second_deriv = np.diff(eigenvalues, n=2)
        elbow = np.argmax(second_deriv) + 2
        ax.axvline(elbow, color='r', linestyle='--',
                  label=f'Elbow at PC {elbow}')
        ax.legend()

    plt.show()

    return elbow

elbow_component = scree_plot(pca)
```

### Visualization with PCA

```python
# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca = pca_2d.fit_transform(X_scaled)

print(f"\n2D PCA:")
print(f"  Shape: {X_pca.shape}")
print(f"  Variance explained: {pca_2d.explained_variance_ratio_.sum():.4f}")

# Plot
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.2%} variance)')
plt.title('PCA: Digits Dataset')
plt.colorbar(scatter, label='Digit')
plt.grid(True, alpha=0.3)
plt.show()
```

### PCA Whitening

**Purpose**: Transform data to have identity covariance matrix

```
Whitening transformation:
Z = X̃ V Λ^(-1/2)

Result:
- Mean = 0
- Variance = 1 for each component
- Uncorrelated
```

**Use cases**: Preprocessing for ICA, some neural networks

```python
# PCA with whitening
pca_white = PCA(n_components=30, whiten=True)
X_whitened = pca_white.fit_transform(X_scaled)

print("Whitened PCA:")
print(f"  Mean: {X_whitened.mean(axis=0)[:5]}")  # Should be ~0
print(f"  Std: {X_whitened.std(axis=0)[:5]}")    # Should be ~1

# Verify whitening: covariance should be identity
cov = np.cov(X_whitened.T)
print(f"  Covariance diagonal: {np.diag(cov)[:5]}")  # Should be ~1
print(f"  Covariance off-diagonal: {cov[0, 1:6]}")  # Should be ~0
```

### Reconstruction and Denoising

```python
# Reduce dimensions then reconstruct
n_components = 20
pca_reduce = PCA(n_components=n_components)
X_reduced = pca_reduce.fit_transform(X_scaled)
X_reconstructed = pca_reduce.inverse_transform(X_reduced)

# Visualize original vs reconstructed
fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    # Original
    axes[0, i].imshow(X[i].reshape(8, 8), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10)

    # Reconstructed (inverse scale)
    X_recon = scaler.inverse_transform(X_reconstructed[i].reshape(1, -1))
    axes[1, i].imshow(X_recon.reshape(8, 8), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title(f'Reconstructed\n({n_components} PCs)', fontsize=10)

plt.tight_layout()
plt.show()

# Reconstruction error
mse = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\nReconstruction MSE: {mse:.6f}")
```

### Incremental PCA

**For Large Datasets**: Process in batches

```python
from sklearn.decomposition import IncrementalPCA

# Simulate large dataset (can't fit in memory)
n_samples_large = 100000
n_features = 1000
batch_size = 1000

# Incremental PCA
ipca = IncrementalPCA(n_components=50, batch_size=batch_size)

# Fit in batches
for i in range(0, n_samples_large, batch_size):
    X_batch = np.random.randn(batch_size, n_features)
    ipca.partial_fit(X_batch)

print("Incremental PCA:")
print(f"  Components: {ipca.n_components}")
print(f"  Variance explained: {ipca.explained_variance_ratio_.sum():.4f}")
```

### Kernel PCA

**Non-linear PCA** using kernel trick

```python
from sklearn.decomposition import KernelPCA

# Generate non-linear data
from sklearn.datasets import make_circles

X_circles, y_circles = make_circles(n_samples=1000, factor=0.3, noise=0.05, random_state=42)

# Standard PCA (linear)
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)

# Kernel PCA with RBF kernel (non-linear)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=10)
X_kpca = kpca.fit_transform(X_circles)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis')
axes[0].set_title('Original Data')

axes[1].scatter(X_pca_linear[:, 0], X_pca_linear[:, 1], c=y_circles, cmap='viridis')
axes[1].set_title('Linear PCA')

axes[2].scatter(X_kpca[:, 0], X_kpca[:, 1], c=y_circles, cmap='viridis')
axes[2].set_title('Kernel PCA (RBF)')

plt.tight_layout()
plt.show()
```

## t-SNE

### t-Distributed Stochastic Neighbor Embedding

**Purpose**: Visualize high-dimensional data in 2D/3D

**Key Idea**:
1. Model pairwise similarities in high-D using Gaussian
2. Model pairwise similarities in low-D using t-distribution
3. Minimize KL divergence between distributions

### Algorithm

**High-dimensional similarities** (Gaussian):
```
p_{j|i} = exp(-||xᵢ - xⱼ||² / 2σᵢ²) / Σₖ exp(-||xᵢ - xₖ||² / 2σᵢ²)

p_{ij} = (p_{j|i} + p_{i|j}) / 2n  (symmetrized)
```

**Low-dimensional similarities** (t-distribution with df=1):
```
q_{ij} = (1 + ||yᵢ - yⱼ||²)⁻¹ / Σₖₗ (1 + ||yₖ - yₗ||²)⁻¹
```

**Objective** (KL divergence):
```
C = KL(P||Q) = Σᵢⱼ pᵢⱼ log(pᵢⱼ / qᵢⱼ)

Minimize using gradient descent
```

### Perplexity Parameter

**Perplexity**: Effective number of neighbors to consider

```
Perplexity = 2^H(Pᵢ)

Where H(Pᵢ) = -Σⱼ pⱼ|ᵢ log₂(pⱼ|ᵢ)  (Shannon entropy)
```

**Guidelines**:
- Range: 5-50 (typical: 30)
- Small perplexity: Focus on local structure
- Large perplexity: Focus on global structure
- Rule of thumb: perplexity < n_samples / 3

### Implementation

```python
from sklearn.manifold import TSNE
import time

# Load data
digits = load_digits()
X = digits.data
y = digits.target

# Preprocess: PCA to 50 dimensions first (recommended for speed)
pca_pre = PCA(n_components=50)
X_pca = pca_pre.fit_transform(StandardScaler().fit_transform(X))

print(f"Pre-PCA: {X.shape} → {X_pca.shape}")

# t-SNE with different perplexities
perplexities = [5, 30, 50]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, perplexity in enumerate(perplexities):
    print(f"\nt-SNE with perplexity={perplexity}...")
    start = time.time()

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate='auto',
        n_iter=1000,
        random_state=42,
        init='pca'  # PCA initialization
    )

    X_tsne = tsne.fit_transform(X_pca)
    elapsed = time.time() - start

    print(f"  Time: {elapsed:.2f}s")
    print(f"  KL divergence: {tsne.kl_divergence_:.4f}")

    # Plot
    scatter = axes[idx].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
    axes[idx].set_title(f't-SNE (perplexity={perplexity})')
    axes[idx].set_xlabel('t-SNE 1')
    axes[idx].set_ylabel('t-SNE 2')

plt.colorbar(scatter, ax=axes, label='Digit')
plt.tight_layout()
plt.show()
```

### Best Practices

**DO**:
- Preprocess with PCA to ~50 dimensions (much faster)
- Try multiple random initializations
- Try multiple perplexity values
- Use for visualization only (not feature engineering)

**DON'T**:
- Interpret distances between clusters (not meaningful)
- Use for downstream ML tasks (use UMAP or PCA instead)
- Compare t-SNE plots from different runs directly
- Use on very large datasets (>10K samples) without PCA first

```python
def tsne_analysis(X, y, perplexities=[5, 30, 50], n_iter=1000):
    """Comprehensive t-SNE analysis."""
    # Preprocess
    X_scaled = StandardScaler().fit_transform(X)

    if X.shape[1] > 50:
        pca = PCA(n_components=50)
        X_prep = pca.fit_transform(X_scaled)
        print(f"PCA preprocessing: {X.shape} → {X_prep.shape}")
        print(f"Variance retained: {pca.explained_variance_ratio_.sum():.4f}")
    else:
        X_prep = X_scaled

    results = {}

    for perplexity in perplexities:
        print(f"\nPerplexity: {perplexity}")
        start = time.time()

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate='auto',
            n_iter=n_iter,
            random_state=42
        )

        X_embedded = tsne.fit_transform(X_prep)
        elapsed = time.time() - start

        results[perplexity] = {
            'embedding': X_embedded,
            'kl_divergence': tsne.kl_divergence_,
            'time': elapsed
        }

        print(f"  KL divergence: {tsne.kl_divergence_:.4f}")
        print(f"  Time: {elapsed:.2f}s")

    return results

# Example
results = tsne_analysis(X, y, perplexities=[10, 30, 50])
```

## UMAP

### Uniform Manifold Approximation and Projection

**2025 Recommendation**: Preferred over t-SNE for most use cases

**Advantages over t-SNE**:
- Faster (minutes vs hours for large datasets)
- Preserves global structure better
- Can be used for downstream ML tasks
- Scales to millions of samples
- Deterministic (with fixed seed)

### Mathematical Foundation

**Based on**: Riemannian geometry and algebraic topology

**Key differences from t-SNE**:
1. Assumes data lies on locally connected manifold
2. Uses fuzzy simplicial set representation
3. Optimizes cross-entropy instead of KL divergence

### Implementation

```python
import umap
import time

# UMAP with default parameters
print("UMAP (default parameters)...")
start = time.time()

umap_model = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    metric='euclidean',
    random_state=42
)

X_umap = umap_model.fit_transform(X)
umap_time = time.time() - start

print(f"  Time: {umap_time:.2f}s")

# Compare with t-SNE
print("\nt-SNE (for comparison)...")
start = time.time()

tsne_model = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate='auto',
    random_state=42
)

X_tsne = tsne_model.fit_transform(X)
tsne_time = time.time() - start

print(f"  Time: {tsne_time:.2f}s")
print(f"  UMAP speedup: {tsne_time/umap_time:.1f}x")

# Visualize comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
axes[0].set_title(f'UMAP ({umap_time:.1f}s)')
axes[0].set_xlabel('UMAP 1')
axes[0].set_ylabel('UMAP 2')

axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
axes[1].set_title(f't-SNE ({tsne_time:.1f}s)')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

### Parameter Tuning

**n_neighbors**: Number of neighbors to consider
- Small (5-15): Local structure
- Large (50-100): Global structure
- Default: 15

**min_dist**: Minimum distance between points in embedding
- Small (0.0-0.1): Tight clusters
- Large (0.3-0.5): Spread out
- Default: 0.1

```python
# Parameter exploration
n_neighbors_values = [5, 15, 50]
min_dist_values = [0.0, 0.1, 0.5]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for i, n_neighbors in enumerate(n_neighbors_values):
    for j, min_dist in enumerate(min_dist_values):
        umap_model = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )

        X_embedded = umap_model.fit_transform(X)

        axes[i, j].scatter(X_embedded[:, 0], X_embedded[:, 1],
                          c=y, cmap='tab10', alpha=0.7, s=10)
        axes[i, j].set_title(f'n_neighbors={n_neighbors}, min_dist={min_dist}')
        axes[i, j].set_xticks([])
        axes[i, j].set_yticks([])

plt.tight_layout()
plt.show()
```

### Supervised UMAP

**Use labels** to guide embedding (better separation)

```python
# Supervised UMAP
umap_supervised = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=2,
    random_state=42
)

# Fit with labels
X_umap_supervised = umap_supervised.fit_transform(X, y=y)

# Compare unsupervised vs supervised
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
axes[0].set_title('UMAP (Unsupervised)')

axes[1].scatter(X_umap_supervised[:, 0], X_umap_supervised[:, 1],
               c=y, cmap='tab10', alpha=0.7)
axes[1].set_title('UMAP (Supervised)')

plt.tight_layout()
plt.show()
```

### Transform New Data

**Unlike t-SNE**, UMAP can transform new samples

```python
# Train on subset
X_train = X[:1500]
y_train = y[:1500]
X_test = X[1500:]
y_test = y[1500:]

# Fit UMAP
umap_model = umap.UMAP(n_components=2, random_state=42)
X_train_umap = umap_model.fit_transform(X_train)

# Transform new data
X_test_umap = umap_model.transform(X_test)

print(f"Train embedding: {X_train_umap.shape}")
print(f"Test embedding: {X_test_umap.shape}")

# Visualize
plt.figure(figsize=(10, 8))
plt.scatter(X_train_umap[:, 0], X_train_umap[:, 1],
           c=y_train, cmap='tab10', alpha=0.5, label='Train')
plt.scatter(X_test_umap[:, 0], X_test_umap[:, 1],
           c=y_test, cmap='tab10', marker='x', s=100, label='Test')
plt.title('UMAP with Train/Test Split')
plt.legend()
plt.show()
```

## Linear Discriminant Analysis (LDA)

### Supervised Dimensionality Reduction

**Unlike PCA**: Uses class labels to find discriminative directions

**Objective**: Maximize between-class variance, minimize within-class variance

### Mathematical Formulation

```
Within-class scatter matrix:
S_W = Σₖ Σᵢ∈Cₖ (xᵢ - μₖ)(xᵢ - μₖ)ᵀ

Between-class scatter matrix:
S_B = Σₖ nₖ(μₖ - μ)(μₖ - μ)ᵀ

Objective:
Maximize J(w) = (wᵀ S_B w) / (wᵀ S_W w)

Solution: Eigenvectors of S_W⁻¹ S_B
```

**Maximum components**: min(n_classes - 1, n_features)

### Implementation

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# LDA for dimensionality reduction
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

print(f"LDA Results:")
print(f"  Original shape: {X.shape}")
print(f"  Reduced shape: {X_lda.shape}")
print(f"  Explained variance ratio: {lda.explained_variance_ratio_}")
print(f"  Total variance explained: {lda.explained_variance_ratio_.sum():.4f}")

# Visualize
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel(f'LD1 ({lda.explained_variance_ratio_[0]:.2%} variance)')
plt.ylabel(f'LD2 ({lda.explained_variance_ratio_[1]:.2%} variance)')
plt.title('LDA: Digits Dataset')
plt.colorbar(scatter, label='Digit')
plt.show()
```

### PCA vs LDA

```python
# Compare PCA and LDA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(StandardScaler().fit_transform(X))

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
axes[0].set_title(f'PCA (unsupervised)')
axes[0].set_xlabel('PC1')
axes[0].set_ylabel('PC2')

axes[1].scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='tab10', alpha=0.7)
axes[1].set_title(f'LDA (supervised)')
axes[1].set_xlabel('LD1')
axes[1].set_ylabel('LD2')

plt.tight_layout()
plt.show()

print("\nComparison:")
print("PCA: Maximizes variance (unsupervised)")
print("LDA: Maximizes class separability (supervised)")
print("Use PCA when: No labels, want to preserve data variance")
print("Use LDA when: Have labels, want to discriminate classes")
```

## Autoencoders

### Neural Network-Based Dimensionality Reduction

**Architecture**: Encoder → Bottleneck → Decoder

```
Input (high-D) → Encoder → Code (low-D) → Decoder → Reconstruction
```

**Training**: Minimize reconstruction error

### Implementation with PyTorch

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

class Autoencoder(nn.Module):
    """Simple autoencoder for dimensionality reduction."""

    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

# Prepare data
X_tensor = torch.FloatTensor(X)
dataset = TensorDataset(X_tensor, X_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize model
input_dim = X.shape[1]
encoding_dim = 10
model = Autoencoder(input_dim, encoding_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
n_epochs = 50
losses = []

print("Training Autoencoder...")
for epoch in range(n_epochs):
    epoch_loss = 0
    for batch_X, batch_y in dataloader:
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.6f}")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.title('Autoencoder Training')
plt.grid(True, alpha=0.3)
plt.show()

# Encode data
model.eval()
with torch.no_grad():
    X_encoded = model.encode(X_tensor).numpy()

print(f"\nEncoded shape: {X_encoded.shape}")

# Visualize (reduce encoding to 2D for plotting)
if encoding_dim > 2:
    pca_vis = PCA(n_components=2)
    X_vis = pca_vis.fit_transform(X_encoded)
else:
    X_vis = X_encoded

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.xlabel('Autoencoder Dim 1')
plt.ylabel('Autoencoder Dim 2')
plt.title(f'Autoencoder ({encoding_dim}D → 2D visualization)')
plt.colorbar(scatter, label='Digit')
plt.show()
```

### Variational Autoencoder (VAE)

**Better regularization** through probabilistic latent space

```python
class VAE(nn.Module):
    """Variational Autoencoder."""

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # Encoder
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder
        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, input_dim)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        return self.fc4(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE loss = reconstruction + KL divergence."""
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_loss

# Train VAE (similar to autoencoder but with VAE loss)
```

## Comparison and Selection

### Comprehensive Comparison

```python
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

def compare_methods(X, y, methods_dict):
    """Compare dimensionality reduction methods."""
    results = []

    # Original (no reduction)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_original = cross_val_score(rf, X, y, cv=5)

    results.append({
        'Method': 'Original',
        'Dimensions': X.shape[1],
        'CV Accuracy': scores_original.mean(),
        'CV Std': scores_original.std(),
        'Time': 0
    })

    # Test each method
    for name, transformer in methods_dict.items():
        print(f"\nTesting {name}...")
        start = time.time()

        # Transform
        if hasattr(transformer, 'fit_transform'):
            X_reduced = transformer.fit_transform(X, y if 'LDA' in name else None)
        else:
            X_reduced = transformer.fit_transform(X)

        fit_time = time.time() - start

        # Evaluate
        scores = cross_val_score(rf, X_reduced, y, cv=5)

        results.append({
            'Method': name,
            'Dimensions': X_reduced.shape[1],
            'CV Accuracy': scores.mean(),
            'CV Std': scores.std(),
            'Time': fit_time
        })

        print(f"  Accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
        print(f"  Time: {fit_time:.2f}s")

    return pd.DataFrame(results)

# Define methods
methods = {
    'PCA (10)': PCA(n_components=10),
    'PCA (30)': PCA(n_components=30),
    'LDA': LinearDiscriminantAnalysis(),
    'UMAP (10)': umap.UMAP(n_components=10, random_state=42),
}

# Preprocess
X_scaled = StandardScaler().fit_transform(X)

# Compare
comparison_df = compare_methods(X_scaled, y, methods)

print("\n\nComparison Summary:")
print("=" * 80)
print(comparison_df.to_string(index=False))

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Accuracy vs Dimensions
axes[0].bar(comparison_df['Method'], comparison_df['CV Accuracy'])
axes[0].axhline(y=comparison_df[comparison_df['Method'] == 'Original']['CV Accuracy'].values[0],
               color='r', linestyle='--', label='Original')
axes[0].set_xlabel('Method')
axes[0].set_ylabel('CV Accuracy')
axes[0].set_title('Accuracy Comparison')
axes[0].legend()
axes[0].tick_params(axis='x', rotation=45)

# Time
axes[1].bar(comparison_df['Method'], comparison_df['Time'])
axes[1].set_xlabel('Method')
axes[1].set_ylabel('Time (seconds)')
axes[1].set_title('Computation Time')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
```

### Decision Matrix

| Method | Speed | Accuracy | Visualization | Supervised | Non-linear | Transform New Data |
|--------|-------|----------|---------------|------------|------------|--------------------|
| PCA | Fast | Good | Fair | No | No | Yes |
| LDA | Fast | Best | Good | Yes | No | Yes |
| t-SNE | Slow | N/A | Excellent | No | Yes | No |
| UMAP | Fast | Good | Excellent | Optional | Yes | Yes |
| Autoencoder | Medium | Good | Fair | No | Yes | Yes |

### Selection Guidelines

```python
def recommend_method(n_samples, n_features, has_labels=False, purpose='ml'):
    """Recommend dimensionality reduction method."""

    print("Dimensionality Reduction Recommendation:")
    print("=" * 60)
    print(f"Dataset: {n_samples} samples × {n_features} features")
    print(f"Labels available: {has_labels}")
    print(f"Purpose: {purpose}")
    print()

    recommendations = []

    if purpose == 'visualization':
        if n_samples < 10000:
            recommendations.append("t-SNE (perplexity=30)")
        recommendations.append("UMAP (n_neighbors=15, min_dist=0.1)")
        if has_labels:
            recommendations.append("Supervised UMAP")
        recommendations.append("PCA (n_components=2) - fast baseline")

    elif purpose == 'ml':
        if has_labels:
            recommendations.append("LDA - best for classification")
        recommendations.append("PCA (keep 95% variance)")
        if n_features > 100:
            recommendations.append("PCA → UMAP pipeline")
        if n_samples > 100000:
            recommendations.append("Incremental PCA")

    elif purpose == 'feature_engineering':
        recommendations.append("PCA (keep 95-99% variance)")
        recommendations.append("UMAP (n_components=10-50)")
        if has_labels:
            recommendations.append("LDA")
        recommendations.append("Autoencoder with reconstruction loss")

    print("Recommended methods (in order of preference):")
    for i, method in enumerate(recommendations, 1):
        print(f"  {i}. {method}")

    return recommendations

# Example recommendations
recommend_method(n_samples=10000, n_features=100, has_labels=True, purpose='ml')
recommend_method(n_samples=5000, n_features=50, has_labels=False, purpose='visualization')
```

## Practical Applications

### Complete Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Production-ready pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # Keep 95% variance
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict
y_pred = pipeline.predict(X_test)

print(f"Pipeline Results:")
print(f"  PCA components selected: {pipeline.named_steps['pca'].n_components_}")
print(f"  Variance explained: {pipeline.named_steps['pca'].explained_variance_ratio_.sum():.4f}")
print(f"  Test accuracy: {(y_pred == y_test).mean():.4f}")
```

### Image Compression

```python
from sklearn.datasets import fetch_olivetti_faces

# Load face images
faces = fetch_olivetti_faces(shuffle=True, random_state=42)
X_faces = faces.data  # 400 images × 4096 pixels (64×64)

print(f"Original: {X_faces.shape}")
print(f"Memory: {X_faces.nbytes / 1024:.2f} KB")

# Compress with PCA
n_components = 100
pca_compress = PCA(n_components=n_components)
X_compressed = pca_compress.fit_transform(X_faces)
X_reconstructed = pca_compress.inverse_transform(X_compressed)

print(f"\nCompressed: {X_compressed.shape}")
print(f"Memory: {X_compressed.nbytes / 1024:.2f} KB")
print(f"Compression ratio: {X_faces.nbytes / X_compressed.nbytes:.1f}x")
print(f"Variance retained: {pca_compress.explained_variance_ratio_.sum():.4f}")

# Visualize
fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i in range(10):
    axes[0, i].imshow(X_faces[i].reshape(64, 64), cmap='gray')
    axes[0, i].axis('off')
    if i == 0:
        axes[0, i].set_title('Original', fontsize=10)

    axes[1, i].imshow(X_reconstructed[i].reshape(64, 64), cmap='gray')
    axes[1, i].axis('off')
    if i == 0:
        axes[1, i].set_title(f'Compressed\n({n_components} PCs)', fontsize=10)

plt.tight_layout()
plt.show()
```

## Summary

### Key Takeaways

1. **PCA (Principal Component Analysis)**
   - Fast, interpretable, linear
   - Maximizes variance
   - Use for: Preprocessing, noise reduction, compression
   - 2025: Still the default first choice

2. **t-SNE**
   - Excellent visualization
   - Slow, non-deterministic
   - Use only for 2D/3D visualization
   - Preprocess with PCA first

3. **UMAP**
   - Faster than t-SNE, better global structure
   - Can transform new data
   - Use for: Visualization AND feature engineering
   - **2025 recommendation**: Replace t-SNE with UMAP

4. **LDA**
   - Supervised, discriminative
   - Best for classification preprocessing
   - Limited to (n_classes - 1) dimensions

5. **Autoencoders**
   - Most flexible, non-linear
   - Requires more data and tuning
   - Use for: Complex manifolds, when others fail

### Production Checklist

- [ ] Scale/normalize features before reduction
- [ ] Use cross-validation to select n_components
- [ ] Save fitted transformer with model
- [ ] Document variance/information retained
- [ ] Monitor reconstruction error in production
- [ ] Consider computational cost (PCA >> UMAP >> t-SNE)
- [ ] Use incremental methods for large datasets

### Code Template

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import joblib

# Production pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('reducer', PCA(n_components=0.95))  # or UMAP, LDA
])

# Fit
pipeline.fit(X_train)

# Transform
X_train_reduced = pipeline.transform(X_train)
X_test_reduced = pipeline.transform(X_test)

# Save
joblib.dump(pipeline, 'dimensionality_reduction.pkl')

# Load and use
pipeline_loaded = joblib.load('dimensionality_reduction.pkl')
X_new_reduced = pipeline_loaded.transform(X_new)
```

---

**Last Updated**: 2025-10-14
**Prerequisites**: Linear algebra, eigenvalue decomposition, manifold learning
**Next Topics**: Manifold learning (Isomap, LLE), sparse coding, dictionary learning
