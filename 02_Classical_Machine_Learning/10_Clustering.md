# Clustering

## Table of Contents
1. [Introduction](#introduction)
2. [K-Means Clustering](#k-means-clustering)
3. [Hierarchical Clustering](#hierarchical-clustering)
4. [DBSCAN](#dbscan)
5. [Gaussian Mixture Models](#gaussian-mixture-models)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Practical Implementations](#practical-implementations)
8. [When to Use Each Method](#when-to-use-each-method)

## Introduction

Clustering is unsupervised learning that groups similar data points without labeled targets. Essential for:

- **Customer segmentation**: Group customers by behavior
- **Anomaly detection**: Identify outliers as single-point clusters
- **Data exploration**: Discover hidden patterns
- **Preprocessing**: Reduce data complexity before supervised learning
- **Feature engineering**: Cluster membership as feature

### Types of Clustering

1. **Partitioning**: K-Means, K-Medoids
2. **Hierarchical**: Agglomerative, Divisive
3. **Density-based**: DBSCAN, OPTICS
4. **Model-based**: Gaussian Mixture Models (GMM)
5. **Grid-based**: STING, CLIQUE

### Key Concepts

**Distance Metrics**: How to measure similarity
- Euclidean: sqrt(sum(x_i - y_i)^2)
- Manhattan: sum|x_i - y_i|
- Cosine: 1 - (x*y)/(||x||*||y||)

**Assumptions**:
- Points in same cluster are similar
- Points in different clusters are dissimilar
- Number of clusters K (may or may not be known)

## K-Means Clustering

### Algorithm

**Objective**: Minimize within-cluster sum of squares (WCSS)

```
WCSS = sum_ksum_iinC_k ||x_i - mu_k||^2

Where:
- C_k: cluster k
- mu_k: centroid of cluster k
- x_i: data point i
```

**Lloyd's Algorithm**:
```
1. Initialize: Select K random centroids mu_1, ..., mu_k
2. Repeat until convergence:
   a. Assignment: Assign each point to nearest centroid
      C_k = {x_i : ||x_i - mu_k|| <= ||x_i - mu_j|| for all j}

   b. Update: Recompute centroids as cluster means
      mu_k = (1/|C_k|) sum_iinC_k x_i

3. Convergence: Stop when assignments don't change
```

### Properties

**Advantages**:
- Simple and fast: O(n*K*i*d) where i = iterations, d = dimensions
- Scales to large datasets
- Guaranteed convergence

**Disadvantages**:
- Requires K to be specified
- Sensitive to initialization
- Assumes spherical clusters of similar size
- Sensitive to outliers

### K-Means++ Initialization

**Problem**: Random initialization can lead to poor local optima

**Solution**: Smart initialization that spreads initial centroids

```
K-Means++ Algorithm:
1. Choose first centroid uniformly at random
2. For each remaining centroid:
   - For each point x, compute D(x) = distance to nearest chosen centroid
   - Choose next centroid with probability proportional to D(x)^2
3. Proceed with standard K-Means
```

**Result**: O(log K) competitive with optimal clustering (in expectation)

### Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

# Generate synthetic data
X, y_true = make_blobs(n_samples=500, centers=4, n_features=2,
                       cluster_std=1.0, random_state=42)

# IMPORTANT: Scale features for K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means with K-Means++ initialization (default in sklearn)
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',  # Default initialization
    n_init=10,  # Run algorithm 10 times, keep best
    max_iter=300,
    random_state=42
)

y_pred = kmeans.fit_predict(X_scaled)

print("K-Means Results:")
print(f"  Number of clusters: {kmeans.n_clusters}")
print(f"  Number of iterations: {kmeans.n_iter_}")
print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")
print(f"  Cluster centers shape: {kmeans.cluster_centers_.shape}")

# Visualize results
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# True labels
axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
axes[0].set_title('True Labels')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')

# K-Means predictions
axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
centers_original = scaler.inverse_transform(kmeans.cluster_centers_)
axes[1].scatter(centers_original[:, 0], centers_original[:, 1],
               c='red', marker='X', s=200, edgecolor='black', linewidth=2,
               label='Centroids')
axes[1].set_title('K-Means Clustering')
axes[1].set_xlabel('Feature 1')
axes[1].set_ylabel('Feature 2')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### Choosing K: Elbow Method

**Concept**: Plot WCSS vs K, look for "elbow" where decrease slows

```python
def elbow_method(X, max_k=10):
    """Find optimal K using elbow method."""
    wcss = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, wcss, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)

    # Find elbow using second derivative
    wcss_diff = np.diff(wcss, n=2)
    elbow_k = np.argmax(wcss_diff) + 2  # +2 due to double diff

    plt.axvline(elbow_k, color='r', linestyle='--',
                label=f'Elbow at K={elbow_k}')
    plt.legend()
    plt.show()

    return elbow_k

optimal_k = elbow_method(X_scaled, max_k=10)
print(f"Optimal K: {optimal_k}")
```

### Silhouette Analysis

**Silhouette Score**: Measures how similar a point is to its cluster vs other clusters

```
For point i in cluster C:
  a(i) = average distance to points in same cluster
  b(i) = min average distance to points in other clusters

Silhouette(i) = (b(i) - a(i)) / max(a(i), b(i))
```

**Range**: [-1, 1]
- +1: Point well matched to cluster
- 0: Point on cluster boundary
- -1: Point likely in wrong cluster

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

def silhouette_analysis(X, max_k=10):
    """Silhouette analysis for multiple K values."""
    silhouette_scores = []
    K_range = range(2, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, silhouette_scores, 'bo-')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')
    plt.grid(True, alpha=0.3)

    best_k = K_range[np.argmax(silhouette_scores)]
    plt.axvline(best_k, color='r', linestyle='--',
                label=f'Best K={best_k} (score={max(silhouette_scores):.3f})')
    plt.legend()
    plt.show()

    return best_k

def plot_silhouette(X, n_clusters):
    """Detailed silhouette plot for specific K."""
    fig, ax = plt.subplots(figsize=(10, 7))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate silhouette scores for samples in cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10

    ax.set_title(f'Silhouette Plot for K={n_clusters}')
    ax.set_xlabel('Silhouette Coefficient')
    ax.set_ylabel('Cluster Label')

    # Average silhouette score
    ax.axvline(x=silhouette_avg, color="red", linestyle="--",
              label=f'Average Score: {silhouette_avg:.3f}')
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    ax.legend()

    plt.show()

    print(f"Average Silhouette Score for K={n_clusters}: {silhouette_avg:.4f}")

best_k = silhouette_analysis(X_scaled, max_k=10)
plot_silhouette(X_scaled, best_k)
```

### Mini-Batch K-Means

**For Large Datasets**: Use mini-batches instead of full dataset

**Algorithm**:
```
1. Sample mini-batch from data
2. Assign samples to nearest centroid
3. Update centroids using mini-batch
4. Repeat until convergence
```

**Advantages**:
- Much faster for large datasets (n > 100K)
- Lower memory requirements
- Slightly lower quality but acceptable

```python
from sklearn.cluster import MiniBatchKMeans
import time

# Generate large dataset
X_large, _ = make_blobs(n_samples=100000, centers=10, n_features=50, random_state=42)
X_large_scaled = StandardScaler().fit_transform(X_large)

# Standard K-Means
start = time.time()
kmeans_standard = KMeans(n_clusters=10, random_state=42, n_init=3)
kmeans_standard.fit(X_large_scaled)
time_standard = time.time() - start

# Mini-Batch K-Means
start = time.time()
kmeans_minibatch = MiniBatchKMeans(n_clusters=10, random_state=42,
                                   batch_size=1000, n_init=3)
kmeans_minibatch.fit(X_large_scaled)
time_minibatch = time.time() - start

print("Large Dataset Clustering:")
print(f"  Standard K-Means:")
print(f"    Time: {time_standard:.2f}s")
print(f"    Inertia: {kmeans_standard.inertia_:.2f}")
print(f"\n  Mini-Batch K-Means:")
print(f"    Time: {time_minibatch:.2f}s")
print(f"    Inertia: {kmeans_minibatch.inertia_:.2f}")
print(f"    Speedup: {time_standard/time_minibatch:.1f}x")
```

## Hierarchical Clustering

### Concept

Build hierarchy of clusters (dendrogram) through iterative merging or splitting.

**Two Approaches**:
1. **Agglomerative** (bottom-up): Start with individual points, merge clusters
2. **Divisive** (top-down): Start with all points, recursively split

### Agglomerative Clustering Algorithm

```
1. Start: Each point is its own cluster
2. Repeat:
   - Find two closest clusters
   - Merge them into single cluster
3. Stop: When desired number of clusters reached or all merged
```

### Linkage Methods

**How to measure distance between clusters**:

1. **Single Linkage**: Minimum distance between points
   ```
   d(C_1, C_2) = min{d(x, y) : x in C_1, y in C_2}
   ```
   - Sensitive to outliers, can create "chains"
   - Good for non-globular clusters

2. **Complete Linkage**: Maximum distance between points
   ```
   d(C_1, C_2) = max{d(x, y) : x in C_1, y in C_2}
   ```
   - Produces compact, spherical clusters
   - Less sensitive to outliers

3. **Average Linkage**: Average distance between all pairs
   ```
   d(C_1, C_2) = (1/|C_1||C_2|) sumsum d(x, y)
   ```
   - Compromise between single and complete
   - Robust to outliers

4. **Ward's Method**: Minimizes within-cluster variance
   ```
   d(C_1, C_2) = increase in SSE from merging
   ```
   - Similar to K-Means objective
   - **Recommended default** for most cases

### Dendrograms

**Visualization**: Tree diagram showing cluster hierarchy

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# Generate data
X, y = make_blobs(n_samples=150, centers=3, n_features=2,
                  cluster_std=0.5, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

# Compute linkage matrix
linkage_methods = ['single', 'complete', 'average', 'ward']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.ravel()

for idx, method in enumerate(linkage_methods):
    # Compute linkage
    Z = linkage(X_scaled, method=method)

    # Plot dendrogram
    axes[idx].set_title(f'{method.capitalize()} Linkage')
    dendrogram(Z, ax=axes[idx], no_labels=True, color_threshold=0)
    axes[idx].set_xlabel('Sample Index')
    axes[idx].set_ylabel('Distance')

plt.tight_layout()
plt.show()
```

### Implementation

```python
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree

# Fit agglomerative clustering
agg_clustering = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward',  # Minimizes variance (recommended)
    compute_distances=True  # Needed for dendrogram
)

y_pred = agg_clustering.fit_predict(X_scaled)

print("Agglomerative Clustering Results:")
print(f"  Number of clusters: {agg_clustering.n_clusters_}")
print(f"  Cluster sizes: {np.bincount(y_pred)}")

# Visualize clustering
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.title('Agglomerative Clustering (Ward Linkage)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(label='Cluster')
plt.show()

# Dendrogram with cut line
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 6))
dendrogram(Z, no_labels=True, color_threshold=2.5)
plt.axhline(y=2.5, color='r', linestyle='--', label='Cut height')
plt.title('Dendrogram (Ward Linkage)')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.legend()
plt.show()
```

### Choosing Number of Clusters

**From Dendrogram**: Look for large vertical distances

```python
def find_optimal_clusters_hierarchical(X, max_k=10):
    """Find optimal K using inconsistency criterion."""
    from scipy.cluster.hierarchy import inconsistent

    Z = linkage(X, method='ward')

    # Compute inconsistency coefficients
    incons = inconsistent(Z, d=2)

    # Look for large jumps in distance
    last = Z[-10:, 2]  # Last 10 merges
    last_rev = last[::-1]
    idxs = np.arange(1, len(last) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(idxs, last_rev, 'bo-')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Distance')
    plt.title('Distance at Each Merge (Last 10)')
    plt.xticks(idxs)
    plt.grid(True, alpha=0.3)

    # Find elbow
    acceleration = np.diff(last_rev, n=2)
    k_optimal = acceleration.argmax() + 2

    plt.axvline(k_optimal, color='r', linestyle='--',
                label=f'Optimal K={k_optimal}')
    plt.legend()
    plt.show()

    return k_optimal

optimal_k_hier = find_optimal_clusters_hierarchical(X_scaled)
print(f"Optimal K (hierarchical): {optimal_k_hier}")
```

## DBSCAN

### Density-Based Spatial Clustering of Applications with Noise

**Key Idea**: Clusters are dense regions separated by sparse regions

**Advantages**:
- Doesn't require K
- Finds arbitrary-shaped clusters
- Identifies outliers as noise
- Robust to outliers

**Disadvantages**:
- Sensitive to parameters (eps, min_samples)
- Struggles with varying density clusters
- High-dimensional data (curse of dimensionality)

### Core Concepts

**Definitions**:
- **epsilon-neighborhood**: Points within distance epsilon
- **Core point**: Has at least min_samples points in epsilon-neighborhood
- **Border point**: In epsilon-neighborhood of core point, but not core itself
- **Noise point**: Neither core nor border

**Density-reachable**: Point q is density-reachable from p if there's a chain of core points from p to q

### Algorithm

```
1. For each point p:
   - Find all points in epsilon-neighborhood
   - If |neighborhood| >= min_samples:
     - Mark p as core point
     - Create new cluster (if p not assigned)
     - Add all density-reachable points to cluster
   - Else:
     - Mark as border or noise (determined later)

2. Border points: In epsilon-neighborhood of core but not core
3. Noise points: Neither core nor border
```

### Parameter Selection

**eps (epsilon)**: Maximum distance for neighborhood
- Too small: Many noise points, fragmented clusters
- Too large: Clusters merge

**min_samples**: Minimum points for core point
- Rule of thumb: min_samples >= d + 1 (d = dimensions)
- Larger values --> denser clusters required

**Finding eps**: k-distance graph

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def find_optimal_eps(X, k=4):
    """Find optimal eps using k-distance graph."""
    # Compute k-nearest neighbors
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Sort distances
    distances = np.sort(distances[:, k-1], axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th Nearest Neighbor Distance')
    plt.title('K-Distance Graph for Optimal eps')
    plt.grid(True, alpha=0.3)

    # Find elbow
    diffs = np.diff(distances)
    knee_idx = np.argmax(diffs)
    optimal_eps = distances[knee_idx]

    plt.axhline(optimal_eps, color='r', linestyle='--',
                label=f'Optimal eps={optimal_eps:.3f}')
    plt.legend()
    plt.show()

    return optimal_eps

# Generate data with clusters of different shapes
from sklearn.datasets import make_moons, make_circles

X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)
X_moons_scaled = StandardScaler().fit_transform(X_moons)

# Find optimal eps
optimal_eps = find_optimal_eps(X_moons_scaled, k=4)
```

### Implementation

```python
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons

# Generate non-convex clusters
X_complex, _ = make_moons(n_samples=500, noise=0.05, random_state=42)
X_complex_scaled = StandardScaler().fit_transform(X_complex)

# Fit DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
y_pred = dbscan.fit_predict(X_complex_scaled)

# Results
n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
n_noise = list(y_pred).count(-1)

print("DBSCAN Results:")
print(f"  Number of clusters: {n_clusters}")
print(f"  Number of noise points: {n_noise} ({n_noise/len(y_pred)*100:.1f}%)")
print(f"  Cluster sizes: {np.bincount(y_pred[y_pred >= 0])}")

# Visualize
plt.figure(figsize=(12, 5))

# DBSCAN results
plt.subplot(1, 2, 1)
unique_labels = set(y_pred)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noise points in black
        col = [0, 0, 0, 1]

    class_member_mask = (y_pred == k)
    xy = X_complex[class_member_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6, alpha=0.6)

plt.title(f'DBSCAN (eps={dbscan.eps}, min_samples={dbscan.min_samples})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Compare with K-Means (fails on non-convex)
plt.subplot(1, 2, 2)
kmeans = KMeans(n_clusters=2, random_state=42)
y_kmeans = kmeans.fit_predict(X_complex_scaled)
plt.scatter(X_complex[:, 0], X_complex[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6)
plt.title('K-Means (K=2)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()
```

### HDBSCAN (Hierarchical DBSCAN)

**2025 Recommended**: Automatically finds optimal eps

```python
import hdbscan

# Fit HDBSCAN
hdb = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=5)
y_pred_hdb = hdb.fit_predict(X_complex_scaled)

print("\nHDBSCAN Results:")
print(f"  Number of clusters: {len(set(y_pred_hdb)) - (1 if -1 in y_pred_hdb else 0)}")
print(f"  Number of noise points: {list(y_pred_hdb).count(-1)}")

# Cluster probabilities (confidence)
print(f"  Average cluster probability: {hdb.probabilities_.mean():.3f}")
```

## Gaussian Mixture Models

### Concept

**Soft Clustering**: Each point has probability of belonging to each cluster

**Assumption**: Data generated from mixture of Gaussian distributions

```
p(x) = sum_k pi_k * N(x | mu_k, sum_k)

Where:
- pi_k: mixing coefficient (weight) for component k
- N(x | mu_k, sum_k): Gaussian distribution with mean mu_k and covariance sum_k
- sum_k pi_k = 1
```

### Expectation-Maximization (EM) Algorithm

**Goal**: Find parameters {pi_k, mu_k, sum_k} that maximize likelihood

```
E-step (Expectation):
  Compute responsibility gamma_i_k = P(cluster k | point i)

  gamma_i_k = pi_k * N(x_i | mu_k, sum_k) / sum_j pi_j * N(x_i | mu_j, sum_j)

M-step (Maximization):
  Update parameters using weighted maximum likelihood

  N_k = sum_i gamma_i_k
  pi_k = N_k / n
  mu_k = (1/N_k) sum_i gamma_i_k * x_i
  sum_k = (1/N_k) sum_i gamma_i_k * (x_i - mu_k)(x_i - mu_k)^T

Repeat E and M steps until convergence
```

### Implementation

```python
from sklearn.mixture import GaussianMixture
from matplotlib.patches import Ellipse

# Generate data
X, y_true = make_blobs(n_samples=500, centers=3, n_features=2,
                       cluster_std=[1.0, 1.5, 0.5], random_state=42)

# Fit GMM
gmm = GaussianMixture(
    n_components=3,
    covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
    max_iter=100,
    random_state=42
)

gmm.fit(X)
y_pred = gmm.predict(X)
proba = gmm.predict_proba(X)

print("Gaussian Mixture Model Results:")
print(f"  Number of components: {gmm.n_components}")
print(f"  Converged: {gmm.converged_}")
print(f"  Number of iterations: {gmm.n_iter_}")
print(f"  Log-likelihood: {gmm.score(X) * len(X):.2f}")
print(f"  BIC: {gmm.bic(X):.2f}")
print(f"  AIC: {gmm.aic(X):.2f}")

# Mixing coefficients
print("\nMixing Coefficients (pi_k):")
for i, weight in enumerate(gmm.weights_):
    print(f"  Component {i}: {weight:.3f}")

# Visualize
def plot_gmm(gmm, X, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Hard clustering
    axes[0].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    axes[0].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='X', s=200, edgecolor='black', linewidth=2)

    # Plot covariance ellipses
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        v, w = np.linalg.eigh(covar)
        v = 2.0 * np.sqrt(2.0) * np.sqrt(v)  # 95% confidence
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180.0 * angle / np.pi

        ell = Ellipse(mean, v[0], v[1], 180.0 + angle,
                     facecolor='none', edgecolor='red', linewidth=2)
        axes[0].add_patch(ell)

    axes[0].set_title('GMM Clustering (Hard Assignment)')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')

    # Soft clustering (probabilities)
    proba = gmm.predict_proba(X)
    axes[1].scatter(X[:, 0], X[:, 1], c=proba.max(axis=1),
                   cmap='viridis', alpha=0.6)
    axes[1].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                   c='red', marker='X', s=200, edgecolor='black', linewidth=2)
    axes[1].set_title('GMM Clustering (Soft Assignment - Max Probability)')
    axes[1].set_xlabel('Feature 1')
    axes[1].set_ylabel('Feature 2')
    plt.colorbar(axes[1].collections[0], ax=axes[1], label='Max Probability')

    plt.tight_layout()
    plt.show()

plot_gmm(gmm, X, y_pred)
```

### Model Selection: BIC and AIC

**How many components?** Use information criteria

**Bayesian Information Criterion (BIC)**:
```
BIC = -2*ln(L) + k*ln(n)

Where:
- L: likelihood
- k: number of parameters
- n: number of samples
```

**Akaike Information Criterion (AIC)**:
```
AIC = -2*ln(L) + 2k
```

**Lower is better** (penalizes model complexity)

```python
def select_gmm_components(X, max_components=10):
    """Select optimal number of components using BIC."""
    bic_scores = []
    aic_scores = []
    n_components_range = range(1, max_components + 1)

    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        bic_scores.append(gmm.bic(X))
        aic_scores.append(gmm.aic(X))

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(n_components_range, bic_scores, 'bo-', label='BIC')
    ax.plot(n_components_range, aic_scores, 'rs-', label='AIC')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Information Criterion')
    ax.set_title('Model Selection for GMM')
    ax.legend()
    ax.grid(True, alpha=0.3)

    best_bic = n_components_range[np.argmin(bic_scores)]
    best_aic = n_components_range[np.argmin(aic_scores)]

    ax.axvline(best_bic, color='b', linestyle='--', alpha=0.5,
              label=f'Best BIC: {best_bic}')
    ax.axvline(best_aic, color='r', linestyle='--', alpha=0.5,
              label=f'Best AIC: {best_aic}')

    plt.legend()
    plt.show()

    print(f"Optimal components (BIC): {best_bic}")
    print(f"Optimal components (AIC): {best_aic}")

    return best_bic

optimal_components = select_gmm_components(X, max_components=10)
```

### Covariance Types

```python
# Compare covariance types
covariance_types = ['spherical', 'diag', 'tied', 'full']

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for idx, cov_type in enumerate(covariance_types):
    gmm = GaussianMixture(n_components=3, covariance_type=cov_type, random_state=42)
    y_pred = gmm.fit_predict(X)

    axes[idx].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    axes[idx].scatter(gmm.means_[:, 0], gmm.means_[:, 1],
                     c='red', marker='X', s=200, edgecolor='black', linewidth=2)
    axes[idx].set_title(f'Covariance Type: {cov_type}\nBIC: {gmm.bic(X):.2f}')
    axes[idx].set_xlabel('Feature 1')
    axes[idx].set_ylabel('Feature 2')

plt.tight_layout()
plt.show()

print("Covariance Types:")
print("  'spherical': sigma^2I (same variance, uncorrelated)")
print("  'diag': diag(sigma_1^2, ..., sigma_a^2) (different variances, uncorrelated)")
print("  'tied': Same covariance for all components")
print("  'full': Each component has own covariance (most flexible)")
```

## Evaluation Metrics

### Internal Metrics (No Ground Truth)

#### Silhouette Score

```python
from sklearn.metrics import silhouette_score

# Already covered above
score = silhouette_score(X, labels)
```

#### Davies-Bouldin Index

**Definition**: Average similarity ratio of each cluster with its most similar cluster

```
DB = (1/K) sum_k max_{k!=j} [(sigma_k + sigma_j) / d(c_k, c_j)]

Where:
- sigma_k: average distance within cluster k
- d(c_k, c_j): distance between cluster centers
```

**Lower is better** (more separated clusters)

```python
from sklearn.metrics import davies_bouldin_score

db_score = davies_bouldin_score(X, labels)
print(f"Davies-Bouldin Index: {db_score:.4f}")
```

#### Calinski-Harabasz Index (Variance Ratio)

**Definition**: Ratio of between-cluster to within-cluster variance

```
CH = [SS_B / (K-1)] / [SS_W / (n-K)]

Where:
- SS_B: between-cluster sum of squares
- SS_W: within-cluster sum of squares
```

**Higher is better** (more separated and compact clusters)

```python
from sklearn.metrics import calinski_harabasz_score

ch_score = calinski_harabasz_score(X, labels)
print(f"Calinski-Harabasz Index: {ch_score:.4f}")
```

### External Metrics (With Ground Truth)

```python
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                            fowlkes_mallows_score, homogeneity_completeness_v_measure)

def evaluate_clustering(y_true, y_pred):
    """Comprehensive clustering evaluation."""
    print("Clustering Evaluation Metrics:")
    print("=" * 60)

    # External metrics (require ground truth)
    if y_true is not None:
        ari = adjusted_rand_score(y_true, y_pred)
        ami = adjusted_mutual_info_score(y_true, y_pred)
        fmi = fowlkes_mallows_score(y_true, y_pred)
        h, c, v = homogeneity_completeness_v_measure(y_true, y_pred)

        print("\nExternal Metrics (with ground truth):")
        print(f"  Adjusted Rand Index (ARI): {ari:.4f}")
        print(f"  Adjusted Mutual Info (AMI): {ami:.4f}")
        print(f"  Fowlkes-Mallows Index (FMI): {fmi:.4f}")
        print(f"  Homogeneity: {h:.4f}")
        print(f"  Completeness: {c:.4f}")
        print(f"  V-Measure: {v:.4f}")

    # Internal metrics (no ground truth needed)
    if len(set(y_pred)) > 1:  # More than one cluster
        silhouette = silhouette_score(X, y_pred)
        db = davies_bouldin_score(X, y_pred)
        ch = calinski_harabasz_score(X, y_pred)

        print("\nInternal Metrics (no ground truth needed):")
        print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
        print(f"  Davies-Bouldin Index: {db:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Index: {ch:.4f} (higher is better)")

# Example
X, y_true = make_blobs(n_samples=500, centers=4, random_state=42)
X_scaled = StandardScaler().fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

evaluate_clustering(y_true, y_pred)
```

## Practical Implementations

### Complete Clustering Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusteringPipeline:
    """Complete clustering pipeline with preprocessing."""

    def __init__(self, algorithm='kmeans', n_clusters=3, **kwargs):
        self.algorithm = algorithm
        self.n_clusters = n_clusters
        self.kwargs = kwargs
        self.pipeline = None
        self.labels_ = None

    def fit(self, X):
        """Fit clustering pipeline."""
        # Choose algorithm
        if self.algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=self.n_clusters, **self.kwargs)
        elif self.algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=self.n_clusters, **self.kwargs)
        elif self.algorithm == 'dbscan':
            clusterer = DBSCAN(**self.kwargs)
        elif self.algorithm == 'gmm':
            clusterer = GaussianMixture(n_components=self.n_clusters, **self.kwargs)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")

        # Build pipeline
        steps = [
            ('scaler', StandardScaler()),
        ]

        # Optional PCA for high-dimensional data
        if X.shape[1] > 50:
            steps.append(('pca', PCA(n_components=0.95)))  # Keep 95% variance

        self.pipeline = Pipeline(steps)

        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X)

        # Cluster
        if hasattr(clusterer, 'fit_predict'):
            self.labels_ = clusterer.fit_predict(X_transformed)
        else:
            clusterer.fit(X_transformed)
            self.labels_ = clusterer.predict(X_transformed)

        self.clusterer = clusterer
        self.X_transformed = X_transformed

        return self

    def evaluate(self, X, y_true=None):
        """Evaluate clustering results."""
        if self.labels_ is None:
            raise ValueError("Must fit before evaluating")

        results = {}

        # Internal metrics
        if len(set(self.labels_)) > 1:
            results['silhouette'] = silhouette_score(self.X_transformed, self.labels_)
            results['davies_bouldin'] = davies_bouldin_score(self.X_transformed, self.labels_)
            results['calinski_harabasz'] = calinski_harabasz_score(self.X_transformed, self.labels_)

        # External metrics
        if y_true is not None:
            results['ari'] = adjusted_rand_score(y_true, self.labels_)
            results['ami'] = adjusted_mutual_info_score(y_true, self.labels_)

        return results

# Example usage
from sklearn.datasets import load_iris

iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# Try different algorithms
algorithms = ['kmeans', 'hierarchical', 'gmm']
results = {}

for algo in algorithms:
    print(f"\n{algo.upper()}:")
    print("=" * 40)

    pipeline = ClusteringPipeline(algorithm=algo, n_clusters=3, random_state=42)
    pipeline.fit(X_iris)

    metrics = pipeline.evaluate(X_iris, y_iris)
    results[algo] = metrics

    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# Compare
print("\n\nComparison:")
print("=" * 60)
import pandas as pd
df_results = pd.DataFrame(results).T
print(df_results.to_string())
```

### Anomaly Detection with Clustering

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def detect_anomalies_clustering(X, contamination=0.1):
    """Detect anomalies using multiple methods."""
    methods = {}

    # 1. DBSCAN (noise points)
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X)
    anomalies_dbscan = (labels == -1)
    methods['DBSCAN'] = anomalies_dbscan

    # 2. GMM (low probability)
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X)
    scores = gmm.score_samples(X)
    threshold = np.percentile(scores, contamination * 100)
    anomalies_gmm = (scores < threshold)
    methods['GMM'] = anomalies_gmm

    # 3. Isolation Forest
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    anomalies_iso = (iso_forest.fit_predict(X) == -1)
    methods['Isolation Forest'] = anomalies_iso

    # 4. Local Outlier Factor
    lof = LocalOutlierFactor(contamination=contamination)
    anomalies_lof = (lof.fit_predict(X) == -1)
    methods['LOF'] = anomalies_lof

    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for idx, (name, anomalies) in enumerate(methods.items()):
        axes[idx].scatter(X[~anomalies, 0], X[~anomalies, 1],
                         c='blue', alpha=0.6, label='Normal')
        axes[idx].scatter(X[anomalies, 0], X[anomalies, 1],
                         c='red', alpha=0.8, label='Anomaly', marker='x')
        axes[idx].set_title(f'{name}: {anomalies.sum()} anomalies')
        axes[idx].set_xlabel('Feature 1')
        axes[idx].set_ylabel('Feature 2')
        axes[idx].legend()

    plt.tight_layout()
    plt.show()

    # Summary
    print("Anomaly Detection Summary:")
    for name, anomalies in methods.items():
        print(f"  {name}: {anomalies.sum()} anomalies ({anomalies.sum()/len(X)*100:.1f}%)")

    return methods

# Generate data with anomalies
X_normal, _ = make_blobs(n_samples=300, centers=3, random_state=42)
X_anomalies = np.random.uniform(low=-10, high=10, size=(30, 2))
X_with_anomalies = np.vstack([X_normal, X_anomalies])
X_scaled = StandardScaler().fit_transform(X_with_anomalies)

anomaly_methods = detect_anomalies_clustering(X_scaled, contamination=0.1)
```

## When to Use Each Method

### K-Means

```
Use K-Means when:
[x] Know number of clusters K
[x] Clusters are spherical and similar size
[x] Large dataset (fast, O(nKid))
[x] Need fast, simple solution

Avoid when:
[ ] Clusters have different shapes
[ ] Clusters have different sizes/densities
[ ] Outliers present
[ ] Don't know K
```

### Hierarchical Clustering

```
Use Hierarchical when:
[x] Want to explore cluster hierarchy
[x] Don't know K (can choose later from dendrogram)
[x] Small to medium dataset (n < 10,000)
[x] Need deterministic results

Avoid when:
[ ] Large dataset (O(n^2 log n) for some methods)
[ ] High-dimensional data
[ ] Need fast clustering
```

### DBSCAN

```
Use DBSCAN when:
[x] Clusters have arbitrary shapes
[x] Outliers/noise present
[x] Don't know K
[x] Varying cluster densities (use HDBSCAN)

Avoid when:
[ ] Clusters have varying densities (use HDBSCAN instead)
[ ] High-dimensional data (curse of dimensionality)
[ ] All points must be assigned to clusters
```

### GMM

```
Use GMM when:
[x] Need soft clustering (probabilities)
[x] Data approximately Gaussian
[x] Want to model data distribution
[x] Clusters have elliptical shapes

Avoid when:
[ ] Non-Gaussian data
[ ] Very large dataset (slower than K-Means)
[ ] Clusters have complex shapes
```

### 2025 Recommendations

**Default Choice**: K-Means (if K known) or HDBSCAN (if K unknown)

**By Use Case**:
- **Customer Segmentation**: K-Means or GMM (interpretable, probabilistic)
- **Anomaly Detection**: DBSCAN or Isolation Forest
- **Image Compression**: K-Means (fast)
- **Gene Expression**: Hierarchical (explore relationships)
- **Document Clustering**: K-Means with cosine distance or spectral clustering

**By Data Characteristics**:
- **Large dataset (n > 100K)**: Mini-Batch K-Means
- **High-dimensional (d > 50)**: PCA + K-Means, or spectral methods
- **Non-convex clusters**: DBSCAN or Spectral Clustering
- **Mixed data types**: K-Prototypes or Gower distance with hierarchical

## Summary

### Key Takeaways

1. **K-Means**
   - Fast and scalable
   - Requires K, assumes spherical clusters
   - Use K-Means++ initialization
   - Elbow method and silhouette for K selection

2. **Hierarchical**
   - Flexible (choose K later)
   - Ward linkage recommended
   - Good for small datasets
   - Dendrograms aid interpretation

3. **DBSCAN**
   - Arbitrary shapes, handles noise
   - No need to specify K
   - Sensitive to parameters
   - Use HDBSCAN for varying densities

4. **GMM**
   - Soft clustering with probabilities
   - Flexible cluster shapes
   - Use BIC/AIC for model selection
   - Slower but more expressive

### Production Checklist

- [ ] Scale features (critical for K-Means, hierarchical)
- [ ] Handle outliers before clustering
- [ ] Use multiple metrics for K selection
- [ ] Validate stability (re-run with different initializations)
- [ ] Consider domain knowledge for K
- [ ] Use appropriate distance metric for data type
- [ ] Dimensionality reduction for high-D data
- [ ] Profile computational requirements

### Code Template

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Preprocessing
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Clustering
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Evaluation
silhouette = silhouette_score(X_scaled, labels)
print(f"Silhouette Score: {silhouette:.4f}")

# Save model
import joblib
joblib.dump({'scaler': scaler, 'kmeans': kmeans}, 'clustering_model.pkl')
```

---

**Last Updated**: 2025-10-14
**Prerequisites**: Distance metrics, optimization, probability
**Next Topics**: Dimensionality reduction, semi-supervised learning, spectral clustering
