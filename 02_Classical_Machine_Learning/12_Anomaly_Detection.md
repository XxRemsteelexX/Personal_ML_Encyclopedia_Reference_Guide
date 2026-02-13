# Anomaly Detection

## Table of Contents
1. [Introduction](#introduction)
2. [Statistical Methods](#statistical-methods)
3. [Isolation Forest](#isolation-forest)
4. [Local Outlier Factor](#local-outlier-factor)
5. [One-Class SVM](#one-class-svm)
6. [DBSCAN for Anomaly Detection](#dbscan-for-anomaly-detection)
7. [Elliptic Envelope](#elliptic-envelope)
8. [Ensemble Anomaly Detection](#ensemble-anomaly-detection)
9. [Time Series Anomaly Detection](#time-series-anomaly-detection)
10. [Evaluation Without Labels](#evaluation-without-labels)
11. [Evaluation With Labels](#evaluation-with-labels)
12. [Production Deployment](#production-deployment)
13. [Algorithm Selection Guide](#algorithm-selection-guide)
14. [See Also](#see-also)
15. [Resources](#resources)

---

## Introduction

**Anomaly detection** (outlier detection, novelty detection) identifies data points that deviate significantly from the majority of the data. Critical for fraud detection, network intrusion, manufacturing defects, system health monitoring, and data quality.

**Anomaly Taxonomy:**
- **Point anomalies**: Individual instances anomalous with respect to the rest of the data (single fraudulent transaction)
- **Contextual anomalies**: Anomalous in a specific context but not otherwise (temperature of 35C is normal in summer, anomalous in winter)
- **Collective anomalies**: Collection of related data instances is anomalous (sequence of failed login attempts)

**Problem Formulations:**
- **Unsupervised**: No labels, assumes anomalies are rare and different from normal (most common)
- **Supervised**: Labeled normal and anomalous examples (rare due to class imbalance)
- **Semi-supervised**: Only normal examples for training (novelty detection)

**Key Challenges:**
- Defining normal behavior boundary
- Adapting to evolving normal behavior (concept drift)
- High-dimensional data (curse of dimensionality)
- Extreme class imbalance (often <1% anomalies)
- Domain-specific anomaly definitions

---

## Statistical Methods

Statistical approaches model expected distribution of data and identify outliers based on deviation measures. Assume underlying probability distribution (often Gaussian).

### Z-Score (Standard Score)

Measures how many standard deviations a data point is from the mean. Assumes Gaussian distribution.

**Formula:**
```
z = (x - mu) / sigma
```

**Threshold:** Typically |z| > 3 (99.7% of data within 3 sigma for normal distribution)

**Limitations:**
- Assumes Gaussian distribution
- Mean and std dev influenced by outliers themselves
- Univariate (separate calculation per feature)

```python
import numpy as np
from scipy import stats

# Univariate Z-score
data = np.random.randn(1000)
data = np.append(data, [10, -10, 15])  # Add outliers

z_scores = np.abs(stats.zscore(data))
threshold = 3
anomalies = np.where(z_scores > threshold)[0]

print(f"Found {len(anomalies)} anomalies")
print(f"Anomaly indices: {anomalies}")
print(f"Anomaly values: {data[anomalies]}")

# Multivariate Z-score (feature-wise)
X = np.random.randn(1000, 5)
X = np.vstack([X, [10, 10, 10, 10, 10]])  # Add outlier

z_scores = np.abs(stats.zscore(X, axis=0))
anomaly_mask = np.any(z_scores > 3, axis=1)
anomaly_indices = np.where(anomaly_mask)[0]

print(f"Multivariate anomalies: {anomaly_indices}")
```

### Modified Z-Score (MAD-based)

Uses **Median Absolute Deviation (MAD)** instead of standard deviation. More robust to outliers.

**Formula:**
```
MAD = median(|x_i - median(x)|)
Modified Z-score = 0.6745 * (x - median(x)) / MAD
```

**Threshold:** Typically |modified z| > 3.5

```python
def modified_z_score(data):
    """Calculate modified z-score using MAD."""
    median = np.median(data)
    mad = np.median(np.abs(data - median))

    # Avoid division by zero
    if mad == 0:
        mad = np.mean(np.abs(data - median))

    modified_z = 0.6745 * (data - median) / mad
    return modified_z

data = np.random.randn(1000)
data = np.append(data, [10, -10, 15])

mod_z_scores = np.abs(modified_z_score(data))
threshold = 3.5
anomalies = np.where(mod_z_scores > threshold)[0]

print(f"MAD-based anomalies: {len(anomalies)}")
```

### Interquartile Range (IQR)

Non-parametric method based on quartiles. Robust to outliers in threshold calculation.

**Formula:**
```
IQR = Q3 - Q1
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
```

**Multiplier:** 1.5 is standard (Tukey's fences), 3.0 for extreme outliers

```python
def iqr_outliers(data, multiplier=1.5):
    """Detect outliers using IQR method."""
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1

    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    outliers = (data < lower_bound) | (data > upper_bound)
    return outliers, lower_bound, upper_bound

data = np.random.randn(1000)
data = np.append(data, [10, -10, 15])

outliers, lower, upper = iqr_outliers(data, multiplier=1.5)
print(f"IQR bounds: [{lower:.2f}, {upper:.2f}]")
print(f"Outliers: {len(np.where(outliers)[0])}")

# Multivariate IQR (feature-wise)
X = np.random.randn(1000, 5)
outlier_mask = np.zeros(X.shape[0], dtype=bool)

for col in range(X.shape[1]):
    outliers, _, _ = iqr_outliers(X[:, col])
    outlier_mask |= outliers

print(f"Multivariate IQR outliers: {np.sum(outlier_mask)}")
```

### Mahalanobis Distance

Measures distance from point to distribution center, accounting for correlations. Multivariate generalization of Z-score.

**Formula:**
```
D_M(x) = sqrt((x - mu)^T * Sigma^-1 * (x - mu))
```

**Threshold:** Chi-squared distribution with p degrees of freedom (p = num features), typically 95th or 99th percentile

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2

def mahalanobis_outliers(X, alpha=0.01):
    """
    Detect outliers using Mahalanobis distance.

    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
    alpha : float, significance level (default 0.01)

    Returns:
    --------
    outliers : boolean array
    distances : Mahalanobis distances
    """
    mean = np.mean(X, axis=0)
    cov = np.cov(X.T)
    inv_cov = np.linalg.inv(cov)

    # Calculate Mahalanobis distance for each point
    distances = np.array([
        mahalanobis(x, mean, inv_cov) for x in X
    ])

    # Chi-squared threshold
    threshold = chi2.ppf(1 - alpha, df=X.shape[1])
    outliers = distances > np.sqrt(threshold)

    return outliers, distances

# Example
np.random.seed(42)
X = np.random.randn(1000, 3)
# Add correlated outliers
X = np.vstack([X, [[5, 5, 5], [-5, -5, -5]]])

outliers, distances = mahalanobis_outliers(X, alpha=0.01)
print(f"Mahalanobis outliers: {np.sum(outliers)}")
print(f"Max distance: {distances.max():.2f}")
print(f"Outlier indices: {np.where(outliers)[0]}")
```

---

## Isolation Forest

**Isolation Forest** isolates anomalies by randomly selecting features and split values. Anomalies are easier to isolate (require fewer splits) than normal points.

**Algorithm:**
1. Randomly select feature and split value
2. Recursively partition data into binary tree
3. Anomalies have shorter average path length across trees
4. Build ensemble of isolation trees (100-300 trees)

**Anomaly Score:**
```
s(x, n) = 2^(-E(h(x)) / c(n))
```
- E(h(x)): Average path length for point x
- c(n): Average path length of unsuccessful search in BST (normalization)
- Score close to 1: Anomaly
- Score close to 0: Normal

**Hyperparameters:**
- **n_estimators**: 100-300 (more trees = more stable, diminishing returns after 200)
- **max_samples**: 256 (original paper), 'auto' uses min(256, n_samples)
- **contamination**: 0.01-0.1 (expected proportion of outliers, 'auto' uses 0.1)
- **max_features**: 1.0 (use all features), can reduce for speed

**Advantages:**
- Linear time complexity O(n log n)
- Low memory requirement
- Works well with high dimensions
- No distance metric needed
- Few hyperparameters

**Limitations:**
- Contamination parameter required
- May miss local anomalies in dense clusters
- Random performance on normal vs normal variation

```python
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate data with outliers
X, _ = make_blobs(n_samples=300, centers=1, n_features=2,
                   random_state=42, cluster_std=1.0)
outliers = np.random.uniform(low=-8, high=8, size=(20, 2))
X = np.vstack([X, outliers])

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=100,
    max_samples='auto',  # 256 or n_samples
    contamination=0.1,   # Expected outlier proportion
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

# Fit and predict (-1 for outliers, 1 for inliers)
y_pred = iso_forest.fit_predict(X)

# Get anomaly scores (lower = more anomalous)
scores = iso_forest.score_samples(X)

print(f"Detected anomalies: {np.sum(y_pred == -1)}")
print(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

# Decision function (negative = outlier)
decision = iso_forest.decision_function(X)
print(f"Decision range: [{decision.min():.3f}, {decision.max():.3f}]")

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
            c='red', label='Anomaly', alpha=0.8)
plt.title('Isolation Forest Predictions')
plt.legend()

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=scores, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='Anomaly Score')
plt.title('Anomaly Scores (lower = more anomalous)')
plt.tight_layout()
```

### Extended Isolation Forest

Addresses limitation of axis-parallel splits using random slopes.

```python
# Extended Isolation Forest (requires eif package)
# pip install eif

try:
    from eif import iForest

    # Create extended isolation forest
    ext_iso = iForest(
        X,
        ntrees=100,
        sample_size=256,
        ExtensionLevel=X.shape[1] - 1  # Use all features for splits
    )

    # Get anomaly scores
    ext_scores = ext_iso.compute_paths(X_test=X)

    print(f"Extended IF score range: [{ext_scores.min():.3f}, {ext_scores.max():.3f}]")

except ImportError:
    print("Extended Isolation Forest requires 'eif' package")
```

---

## Local Outlier Factor

**Local Outlier Factor (LOF)** compares local density of a point to local densities of its neighbors. Detects anomalies in regions of varying density.

**Algorithm:**
1. Find k-nearest neighbors for each point
2. Calculate reachability distance (max of actual distance and k-distance of neighbor)
3. Calculate local reachability density (LRD)
4. LOF = average LRD of neighbors / LRD of point
5. LOF >> 1: Outlier (lower density than neighbors)
6. LOF ~ 1: Normal point

**Hyperparameters:**
- **n_neighbors**: 20 (default), 10-50 typical range, higher = smoother boundary
- **metric**: 'minkowski' (default), 'euclidean', 'manhattan', 'cosine'
- **contamination**: 0.1 (default), expected proportion of outliers
- **novelty**: False (outlier detection on training set), True (novelty detection on new data)

**Complexity:** O(n^2) for distance computation, can use ball tree/KD tree for speedup

**Advantages:**
- Handles varying density regions
- Local perspective (detects local anomalies)
- No assumption about data distribution

**Limitations:**
- Sensitive to k choice
- Computationally expensive for large datasets
- Binary distance metrics (not well-suited for text)

```python
from sklearn.neighbors import LocalOutlierFactor

# Generate data with varying density
np.random.seed(42)
X1 = np.random.randn(200, 2) * 0.5
X2 = np.random.randn(200, 2) * 1.5 + np.array([5, 5])
outliers = np.random.uniform(low=-5, high=10, size=(20, 2))
X = np.vstack([X1, X2, outliers])

# LOF for outlier detection
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    metric='minkowski',
    p=2,  # Euclidean distance
    novelty=False,  # Outlier detection mode
    n_jobs=-1
)

# Fit and predict
y_pred = lof.fit_predict(X)

# Get negative outlier factor (more negative = more outlier)
neg_outlier_factor = lof.negative_outlier_factor_

print(f"Detected anomalies: {np.sum(y_pred == -1)}")
print(f"LOF range: [{-neg_outlier_factor.min():.2f}, {-neg_outlier_factor.max():.2f}]")

# Visualize
plt.figure(figsize=(10, 4))

plt.subplot(121)
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
            c='red', label='Anomaly', alpha=0.8)
plt.title('LOF Predictions')
plt.legend()

plt.subplot(122)
plt.scatter(X[:, 0], X[:, 1], c=-neg_outlier_factor, cmap='RdYlBu', alpha=0.6)
plt.colorbar(label='LOF Score')
plt.title('LOF Scores (higher = more outlier)')
plt.tight_layout()

# LOF for novelty detection
lof_novelty = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.1,
    novelty=True  # Novelty detection mode
)

# Fit on training data
X_train = X[y_pred == 1]  # Use normal points for training
lof_novelty.fit(X_train)

# Predict on new data
X_test = np.random.randn(50, 2) + np.array([2, 2])
y_test_pred = lof_novelty.predict(X_test)
test_scores = lof_novelty.score_samples(X_test)

print(f"\nNovelty detection anomalies: {np.sum(y_test_pred == -1)}")
```

---

## One-Class SVM

**One-Class SVM** learns a decision boundary around normal data in high-dimensional space using kernel trick. Treats anomaly detection as a one-class classification problem.

**Algorithm:**
1. Map data to high-dimensional feature space using kernel
2. Find hyperplane that separates data from origin with maximum margin
3. Points on wrong side of hyperplane (or far from it) are anomalies

**Hyperparameters:**
- **nu**: 0.05-0.1, upper bound on fraction of outliers, lower bound on fraction of support vectors
- **kernel**: 'rbf' (default, most common), 'linear', 'poly', 'sigmoid'
- **gamma**: 'scale' (1/(n_features * X.var())), 'auto' (1/n_features), or float, controls RBF kernel width
- **degree**: 3 (for poly kernel)
- **coef0**: 0.0 (for poly/sigmoid kernel)

**Complexity:** O(n^2 * n_features) to O(n^3 * n_features) depending on solver

**Advantages:**
- Flexible decision boundaries (kernel trick)
- Theoretically well-founded
- Works well with high-dimensional data

**Limitations:**
- Expensive for large datasets (not scalable)
- Sensitive to hyperparameters (nu, gamma)
- Requires careful feature scaling
- Memory intensive (stores support vectors)

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Generate data
np.random.seed(42)
X = np.random.randn(300, 2)
outliers = np.random.uniform(low=-5, high=5, size=(30, 2))
X = np.vstack([X, outliers])

# Scale data (important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# One-Class SVM
oc_svm = OneClassSVM(
    nu=0.1,           # Expected outlier proportion
    kernel='rbf',     # Radial basis function
    gamma='scale',    # 1 / (n_features * X.var())
)

# Fit and predict
y_pred = oc_svm.fit_predict(X_scaled)

# Get decision scores (negative = outlier)
decision_scores = oc_svm.decision_function(X_scaled)

print(f"Detected anomalies: {np.sum(y_pred == -1)}")
print(f"Decision score range: [{decision_scores.min():.2f}, {decision_scores.max():.2f}]")
print(f"Number of support vectors: {len(oc_svm.support_vectors_)}")

# Visualize decision boundary
def plot_decision_boundary(X, y_pred, model, scaler):
    plt.figure(figsize=(10, 8))

    # Create mesh
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Transform mesh
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_scaled = scaler.transform(mesh_points)

    # Predict on mesh
    Z = model.decision_function(mesh_scaled)
    Z = Z.reshape(xx.shape)

    # Plot contours
    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    # Plot points
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
                c='red', label='Anomaly', alpha=0.8)

    plt.title('One-Class SVM Decision Boundary')
    plt.legend()
    plt.colorbar(label='Decision Score')

plot_decision_boundary(X, y_pred, oc_svm, scaler)

# Compare different kernels
kernels = ['linear', 'rbf', 'poly']
plt.figure(figsize=(15, 4))

for i, kernel in enumerate(kernels):
    svm = OneClassSVM(nu=0.1, kernel=kernel, gamma='scale', degree=3)
    y_pred_k = svm.fit_predict(X_scaled)

    plt.subplot(1, 3, i + 1)
    plt.scatter(X[y_pred_k == 1, 0], X[y_pred_k == 1, 1],
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(X[y_pred_k == -1, 0], X[y_pred_k == -1, 1],
                c='red', label='Anomaly', alpha=0.8)
    plt.title(f'Kernel: {kernel}')
    plt.legend()

plt.tight_layout()
```

---

## DBSCAN for Anomaly Detection

**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) identifies clusters and marks points in low-density regions as noise (anomalies).

**Algorithm:**
1. For each point, find neighbors within distance eps
2. Core point: >= min_samples neighbors
3. Border point: < min_samples neighbors but in neighborhood of core point
4. Noise point: Not core or border (marked as anomaly)

**Hyperparameters:**
- **eps**: Maximum distance between neighbors, critical parameter, use k-distance graph
- **min_samples**: Minimum neighbors for core point, typically 2 * n_features or 5-10
- **metric**: 'euclidean' (default), 'manhattan', 'cosine'

**Selecting eps:**
1. Calculate k-distance for each point (distance to k-th nearest neighbor)
2. Sort k-distances
3. Look for "elbow" in k-distance plot
4. eps = distance at elbow

**Advantages:**
- No need to specify number of clusters
- Finds arbitrary-shaped clusters
- Robust to outliers
- Natural anomaly detection

**Limitations:**
- Sensitive to eps and min_samples
- Struggles with varying density clusters
- Not deterministic with border points

```python
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

# Generate data with clusters and outliers
np.random.seed(42)
X1 = np.random.randn(100, 2) * 0.5
X2 = np.random.randn(100, 2) * 0.5 + np.array([3, 3])
X3 = np.random.randn(100, 2) * 0.5 + np.array([0, 3])
outliers = np.random.uniform(low=-3, high=6, size=(20, 2))
X = np.vstack([X1, X2, X3, outliers])

# Find optimal eps using k-distance graph
def find_optimal_eps(X, k=5):
    """Find optimal eps using k-distance graph."""
    neighbors = NearestNeighbors(n_neighbors=k)
    neighbors.fit(X)
    distances, indices = neighbors.kneighbors(X)

    # Sort distances to k-th nearest neighbor
    k_distances = np.sort(distances[:, -1])

    plt.figure(figsize=(10, 4))
    plt.plot(k_distances)
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{k}-th nearest neighbor distance')
    plt.title('K-distance Graph (look for elbow)')
    plt.grid(True)

    return k_distances

k_distances = find_optimal_eps(X, k=5)

# DBSCAN
dbscan = DBSCAN(
    eps=0.5,          # Maximum neighborhood distance
    min_samples=5,    # Minimum points for core point
    metric='euclidean',
    n_jobs=-1
)

# Fit and get labels (-1 for noise/outliers)
labels = dbscan.fit_predict(X)

# Identify anomalies (noise points)
anomalies = labels == -1
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print(f"Number of clusters: {n_clusters}")
print(f"Number of anomalies: {np.sum(anomalies)}")
print(f"Core samples: {len(dbscan.core_sample_indices_)}")

# Visualize
plt.figure(figsize=(12, 4))

plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.scatter(X[anomalies, 0], X[anomalies, 1],
            c='red', marker='x', s=100, label='Anomaly')
plt.title('DBSCAN Clustering')
plt.legend()

plt.subplot(132)
plt.scatter(X[~anomalies, 0], X[~anomalies, 1],
            c='blue', label='Normal', alpha=0.6)
plt.scatter(X[anomalies, 0], X[anomalies, 1],
            c='red', label='Anomaly', alpha=0.8)
plt.title('Anomaly Detection')
plt.legend()

plt.subplot(133)
# Try different eps values
eps_values = [0.3, 0.5, 0.8]
for eps_val in eps_values:
    db = DBSCAN(eps=eps_val, min_samples=5)
    labels_eps = db.fit_predict(X)
    n_outliers = np.sum(labels_eps == -1)
    plt.scatter(eps_val, n_outliers, s=100)

plt.xlabel('eps parameter')
plt.ylabel('Number of outliers')
plt.title('Outliers vs eps')
plt.grid(True)

plt.tight_layout()
```

### HDBSCAN

**HDBSCAN** (Hierarchical DBSCAN) extends DBSCAN by varying density thresholds, more robust parameter selection.

```python
# HDBSCAN (requires hdbscan package)
# pip install hdbscan

try:
    import hdbscan

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=5,
        min_samples=5,
        cluster_selection_epsilon=0.0,
        metric='euclidean',
        core_dist_n_jobs=-1
    )

    labels_hdb = clusterer.fit_predict(X)

    # Outlier scores (GLOSH)
    outlier_scores = clusterer.outlier_scores_

    # Set threshold
    threshold = np.percentile(outlier_scores, 90)
    anomalies_hdb = outlier_scores > threshold

    print(f"HDBSCAN clusters: {len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)}")
    print(f"HDBSCAN anomalies: {np.sum(anomalies_hdb)}")

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=labels_hdb, cmap='viridis', alpha=0.6)
    plt.title('HDBSCAN Clustering')

    plt.subplot(122)
    plt.scatter(X[:, 0], X[:, 1], c=outlier_scores, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(label='Outlier Score')
    plt.title('HDBSCAN Outlier Scores')
    plt.tight_layout()

except ImportError:
    print("HDBSCAN requires 'hdbscan' package")
```

---

## Elliptic Envelope

**Elliptic Envelope** assumes data comes from Gaussian distribution and fits a robust covariance estimate. Outliers are points outside the ellipse defining the normal data region.

**Algorithm:**
1. Estimate robust covariance matrix using **Minimum Covariance Determinant (MCD)**
2. Calculate Mahalanobis distance using robust covariance
3. Fit elliptical boundary around normal data
4. Points outside boundary are anomalies

**Hyperparameters:**
- **contamination**: 0.1 (default), expected proportion of outliers
- **support_fraction**: None (auto), fraction of points to include in MCD, lower = more robust
- **assume_centered**: False, if True assume data is centered at origin

**Advantages:**
- Robust to outliers in covariance estimation
- Fast computation
- Well-suited for Gaussian data
- Interpretable (elliptical boundary)

**Limitations:**
- Assumes Gaussian distribution
- Performance degrades with non-Gaussian data
- Not suitable for high dimensions (>10-20 features)
- Sensitive to contamination parameter

```python
from sklearn.covariance import EllipticEnvelope

# Generate Gaussian data with outliers
np.random.seed(42)
mean = [0, 0]
cov = [[1, 0.5], [0.5, 1]]
X = np.random.multivariate_normal(mean, cov, 300)
outliers = np.random.uniform(low=-6, high=6, size=(30, 2))
X = np.vstack([X, outliers])

# Elliptic Envelope
envelope = EllipticEnvelope(
    contamination=0.1,
    support_fraction=None,  # Auto
    random_state=42
)

# Fit and predict
y_pred = envelope.fit_predict(X)

# Get Mahalanobis distances
mahal_distances = envelope.mahalanobis(X)

print(f"Detected anomalies: {np.sum(y_pred == -1)}")
print(f"Mahalanobis distance range: [{mahal_distances.min():.2f}, {mahal_distances.max():.2f}]")

# Visualize
def plot_elliptic_envelope(X, y_pred, envelope):
    plt.figure(figsize=(12, 4))

    # Plot 1: Predictions
    plt.subplot(131)
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
                c='blue', label='Normal', alpha=0.6)
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
                c='red', label='Anomaly', alpha=0.8)
    plt.title('Elliptic Envelope Predictions')
    plt.legend()

    # Plot 2: Decision boundary
    plt.subplot(132)
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min()-1, X[:, 0].max()+1, 100),
                         np.linspace(X[:, 1].min()-1, X[:, 1].max()+1, 100))
    Z = envelope.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu', alpha=0.6)
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')
    plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1],
                c='blue', alpha=0.4)
    plt.scatter(X[y_pred == -1, 0], X[y_pred == -1, 1],
                c='red', alpha=0.8)
    plt.title('Decision Boundary')
    plt.colorbar(label='Decision Score')

    # Plot 3: Mahalanobis distances
    plt.subplot(133)
    mahal = envelope.mahalanobis(X)
    plt.scatter(X[:, 0], X[:, 1], c=mahal, cmap='RdYlBu_r', alpha=0.6)
    plt.colorbar(label='Mahalanobis Distance')
    plt.title('Mahalanobis Distances')

    plt.tight_layout()

plot_elliptic_envelope(X, y_pred, envelope)

# Compare with standard covariance
from sklearn.covariance import EmpiricalCovariance

emp_cov = EmpiricalCovariance()
emp_cov.fit(X)

robust_location = envelope.location_
robust_covariance = envelope.covariance_
standard_location = emp_cov.location_
standard_covariance = emp_cov.covariance_

print("\nRobust vs Standard Covariance:")
print(f"Robust mean: {robust_location}")
print(f"Standard mean: {standard_location}")
print(f"Robust cov det: {np.linalg.det(robust_covariance):.4f}")
print(f"Standard cov det: {np.linalg.det(standard_covariance):.4f}")
```

---

## 8. Ensemble Anomaly Detection

Combining multiple anomaly detectors improves robustness and reduces false positives. Different detectors capture different types of anomalies.

### Ensemble Strategies

**Voting Methods:**
- **Maximum**: Take maximum anomaly score across detectors
- **Average**: Average anomaly scores (simple or weighted)
- **Majority voting**: Classify as anomaly if majority of detectors agree
- **Threshold combination**: Multiple thresholds with voting

**Feature Bagging:**
- Train detectors on random subsets of features
- Reduces curse of dimensionality
- Captures local and global anomalies

### PyOD Library for Ensemble Detection

PyOD (Python Outlier Detection) provides comprehensive ensemble methods.

```python
# Installation: pip install pyod

import numpy as np
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.combination import average, maximization, majority_vote
from sklearn.datasets import make_classification

# Generate data with outliers
X, _ = make_classification(n_samples=1000, n_features=10,
                           n_informative=8, n_redundant=2,
                           n_clusters_per_class=1, random_state=42)
# Add outliers
rng = np.random.RandomState(42)
X = np.vstack([X, rng.uniform(low=-6, high=6, size=(50, 10))])

# Initialize multiple detectors
detectors = {
    'IForest': IForest(contamination=0.1, random_state=42),
    'LOF': LOF(contamination=0.1),
    'OCSVM': OCSVM(contamination=0.1),
    'KNN': KNN(contamination=0.1)
}

# Train all detectors and collect scores
scores_matrix = np.zeros((len(X), len(detectors)))

for idx, (name, detector) in enumerate(detectors.items()):
    detector.fit(X)
    scores = detector.decision_function(X)
    scores_matrix[:, idx] = scores
    print(f"{name} detected {sum(detector.labels_)} anomalies")

# Combine scores using different methods
avg_scores = average(scores_matrix)
max_scores = maximization(scores_matrix)

# Get predictions from individual detectors
predictions_matrix = np.zeros((len(X), len(detectors)))
for idx, (name, detector) in enumerate(detectors.items()):
    predictions_matrix[:, idx] = detector.labels_

# Majority voting
maj_vote = majority_vote(predictions_matrix, n_classes=2)

print(f"\nEnsemble Results:")
print(f"Average method: {sum(avg_scores > np.percentile(avg_scores, 90))} anomalies")
print(f"Maximum method: {sum(max_scores > np.percentile(max_scores, 90))} anomalies")
print(f"Majority vote: {sum(maj_vote)} anomalies")
```

### Advanced Ensemble with PyOD

```python
from pyod.models.combination import aom, moa
from pyod.models.feature_bagging import FeatureBagging

# Average of Maximum (AOM) - divide detectors into subgroups
aom_scores = aom(scores_matrix, n_buckets=2)
print(f"AOM detected: {sum(aom_scores > np.percentile(aom_scores, 90))} anomalies")

# Maximum of Average (MOA) - more conservative
moa_scores = moa(scores_matrix, n_buckets=2)
print(f"MOA detected: {sum(moa_scores > np.percentile(moa_scores, 90))} anomalies")

# Feature Bagging - train on random feature subsets
fb_detector = FeatureBagging(
    base_estimator=LOF(contamination=0.1),
    n_estimators=10,
    max_features=0.7,  # Use 70% of features
    contamination=0.1,
    random_state=42
)

fb_detector.fit(X)
fb_predictions = fb_detector.predict(X)
fb_scores = fb_detector.decision_function(X)

print(f"Feature Bagging detected: {sum(fb_predictions)} anomalies")
```

### Performance Comparison

```python
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# Create ground truth (last 50 samples are outliers)
y_true = np.zeros(len(X))
y_true[-50:] = 1

# Calculate AUC for each method
methods = {
    'IForest': detectors['IForest'].decision_function(X),
    'LOF': detectors['LOF'].decision_function(X),
    'OCSVM': detectors['OCSVM'].decision_function(X),
    'KNN': detectors['KNN'].decision_function(X),
    'Average': avg_scores,
    'Maximum': max_scores,
    'AOM': aom_scores,
    'MOA': moa_scores,
    'Feature Bagging': fb_scores
}

aucs = {}
for name, scores in methods.items():
    auc = roc_auc_score(y_true, scores)
    aucs[name] = auc
    print(f"{name}: AUC = {auc:.4f}")

# Visualize comparison
plt.figure(figsize=(10, 6))
plt.bar(range(len(aucs)), list(aucs.values()))
plt.xticks(range(len(aucs)), list(aucs.keys()), rotation=45, ha='right')
plt.ylabel('AUC-ROC')
plt.title('Anomaly Detection Performance Comparison')
plt.axhline(y=0.5, color='r', linestyle='--', label='Random')
plt.legend()
plt.tight_layout()
```

---

## 9. Time Series Anomaly Detection

Time series require specialized anomaly detection accounting for temporal dependencies, seasonality, and trends.

### STL Decomposition for Seasonal Anomalies

**STL** (Seasonal and Trend decomposition using Loess) separates time series into trend, seasonal, and residual components. Anomalies detected in residuals.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
import matplotlib.pyplot as plt

# Generate time series with seasonality and anomalies
np.random.seed(42)
n = 365 * 2  # 2 years daily data
t = np.arange(n)

# Components
trend = 0.01 * t
seasonal = 10 * np.sin(2 * np.pi * t / 365)
noise = np.random.randn(n) * 2
ts = trend + seasonal + noise

# Inject anomalies
anomaly_indices = [100, 250, 400, 550]
ts[anomaly_indices] += np.array([20, -25, 30, -20])

# Create DataFrame
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=n, freq='D'),
    'value': ts
})
df.set_index('date', inplace=True)

# STL decomposition
stl = STL(df['value'], seasonal=365, trend=None)
result = stl.fit()

# Detect anomalies in residuals using Z-score
residuals = result.resid
threshold = 3
z_scores = np.abs((residuals - residuals.mean()) / residuals.std())
anomalies = z_scores > threshold

print(f"Detected {sum(anomalies)} anomalies")
print(f"Anomaly dates: {df.index[anomalies].tolist()[:10]}")

# Visualization
fig, axes = plt.subplots(4, 1, figsize=(12, 10))

axes[0].plot(df.index, df['value'])
axes[0].scatter(df.index[anomalies], df['value'][anomalies],
                c='red', s=50, label='Anomaly')
axes[0].set_title('Original Time Series')
axes[0].legend()

axes[1].plot(df.index, result.trend)
axes[1].set_title('Trend')

axes[2].plot(df.index, result.seasonal)
axes[2].set_title('Seasonal')

axes[3].plot(df.index, residuals)
axes[3].scatter(df.index[anomalies], residuals[anomalies],
                c='red', s=50)
axes[3].axhline(y=threshold*residuals.std(), color='r', linestyle='--')
axes[3].axhline(y=-threshold*residuals.std(), color='r', linestyle='--')
axes[3].set_title('Residuals with Anomalies')

plt.tight_layout()
```

### Prophet Anomaly Detection

Facebook Prophet detects anomalies by modeling trend and seasonality, flagging points outside prediction intervals.

```python
# Installation: pip install prophet

from prophet import Prophet
import pandas as pd
import numpy as np

# Prepare data for Prophet (requires 'ds' and 'y' columns)
df_prophet = pd.DataFrame({
    'ds': pd.date_range('2024-01-01', periods=n, freq='D'),
    'y': ts
})

# Fit Prophet model
model = Prophet(
    interval_width=0.99,  # 99% confidence interval
    changepoint_prior_scale=0.05,
    seasonality_prior_scale=10.0,
    yearly_seasonality=True,
    weekly_seasonality=False,
    daily_seasonality=False
)

model.fit(df_prophet)

# Make predictions
forecast = model.predict(df_prophet)

# Detect anomalies outside prediction intervals
df_prophet['yhat'] = forecast['yhat']
df_prophet['yhat_lower'] = forecast['yhat_lower']
df_prophet['yhat_upper'] = forecast['yhat_upper']

# Points outside prediction interval are anomalies
df_prophet['anomaly'] = (
    (df_prophet['y'] < df_prophet['yhat_lower']) |
    (df_prophet['y'] > df_prophet['yhat_upper'])
)

# Calculate anomaly importance (distance from boundary)
df_prophet['importance'] = np.where(
    df_prophet['y'] > df_prophet['yhat_upper'],
    df_prophet['y'] - df_prophet['yhat_upper'],
    np.where(
        df_prophet['y'] < df_prophet['yhat_lower'],
        df_prophet['yhat_lower'] - df_prophet['y'],
        0
    )
)

print(f"Prophet detected {sum(df_prophet['anomaly'])} anomalies")
print("\nTop anomalies:")
print(df_prophet.nlargest(5, 'importance')[['ds', 'y', 'yhat', 'importance']])

# Visualization
fig = model.plot(forecast)
plt.scatter(df_prophet['ds'][df_prophet['anomaly']],
            df_prophet['y'][df_prophet['anomaly']],
            c='red', s=50, zorder=5, label='Anomaly')
plt.legend()
```

### Change Point Detection with Ruptures

**Change point detection** identifies abrupt changes in time series properties (mean, variance).

```python
# Installation: pip install ruptures

import ruptures as rpt
import numpy as np
import matplotlib.pyplot as plt

# Generate signal with change points
n_samples = 1000
signal = np.random.randn(n_samples)
signal[200:400] += 5  # Mean shift
signal[600:800] *= 3  # Variance change

# PELT (Pruned Exact Linear Time) algorithm
algo_pelt = rpt.Pelt(model="rbf", min_size=50, jump=5).fit(signal)
change_points_pelt = algo_pelt.predict(pen=10)

print(f"PELT detected change points at: {change_points_pelt}")

# Binary Segmentation
algo_binseg = rpt.Binseg(model="l2", min_size=50).fit(signal)
change_points_binseg = algo_binseg.predict(n_bkps=3)

print(f"BinSeg detected change points at: {change_points_binseg}")

# Window-based change detection
algo_window = rpt.Window(width=100, model="l2").fit(signal)
change_points_window = algo_window.predict(pen=1)

print(f"Window detected change points at: {change_points_window}")

# Visualization
rpt.display(signal, change_points_pelt, figsize=(10, 6))
plt.title('PELT Change Point Detection')
```

### ADTK (Anomaly Detection Toolkit)

ADTK provides rule-based and machine learning detectors for time series.

```python
# Installation: pip install adtk

from adtk.data import validate_series
from adtk.detector import (
    ThresholdAD, QuantileAD, InterQuartileRangeAD,
    PersistAD, LevelShiftAD, VolatilityShiftAD,
    SeasonalAD, AutoregressionAD
)
from adtk.visualization import plot
import pandas as pd
import numpy as np

# Create time series
dates = pd.date_range('2024-01-01', periods=1000, freq='H')
values = np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000) * 0.1
values[500:510] += 5  # Level shift anomaly
s = pd.Series(values, index=dates)
s = validate_series(s)

# Threshold detector
threshold_detector = ThresholdAD(high=2, low=-2)
anomalies_threshold = threshold_detector.detect(s)

# Inter-Quartile Range detector
iqr_detector = InterQuartileRangeAD(c=3.0)
anomalies_iqr = iqr_detector.detect(s)

# Level shift detector
level_shift_detector = LevelShiftAD(c=5.0, side='both', window=10)
anomalies_level = level_shift_detector.detect(s)

# Seasonal detector (for patterns)
seasonal_detector = SeasonalAD(c=3.0, side='both')
anomalies_seasonal = seasonal_detector.detect(s)

# Autoregression-based detector
ar_detector = AutoregressionAD(n_steps=1, step_size=1, c=3.0)
anomalies_ar = ar_detector.detect(s)

print(f"Threshold: {sum(anomalies_threshold)} anomalies")
print(f"IQR: {sum(anomalies_iqr)} anomalies")
print(f"Level Shift: {sum(anomalies_level)} anomalies")
print(f"Seasonal: {sum(anomalies_seasonal)} anomalies")
print(f"Autoregression: {sum(anomalies_ar)} anomalies")

# Plot results
plot(s, anomaly=anomalies_level, anomaly_color='red',
     anomaly_tag="marker", figsize=(12, 6))
```

### Rolling Statistics for Streaming Detection

```python
import pandas as pd
import numpy as np

class RollingAnomalyDetector:
    """Detect anomalies using rolling statistics"""

    def __init__(self, window_size=50, n_std=3):
        self.window_size = window_size
        self.n_std = n_std

    def detect(self, series):
        """Detect anomalies in time series"""
        rolling_mean = series.rolling(window=self.window_size).mean()
        rolling_std = series.rolling(window=self.window_size).std()

        # Z-score using rolling statistics
        z_score = np.abs((series - rolling_mean) / rolling_std)
        anomalies = z_score > self.n_std

        return anomalies, z_score

# Example usage
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=1000, freq='H')
values = np.random.randn(1000) + np.sin(np.arange(1000) * 2 * np.pi / 100)
values[500] = 10  # Inject anomaly
series = pd.Series(values, index=dates)

detector = RollingAnomalyDetector(window_size=50, n_std=3)
anomalies, scores = detector.detect(series)

print(f"Detected {sum(anomalies)} anomalies")
print(f"Anomaly timestamps: {series.index[anomalies].tolist()}")
```

---

## 10. Evaluation Without Labels

Unsupervised anomaly detection often lacks ground truth labels. Alternative evaluation strategies required.

### Contamination Ratio Estimation

Estimate expected proportion of anomalies in data. Critical parameter for many detectors.

```python
import numpy as np
from sklearn.ensemble import IsolationForest

def estimate_contamination(X, contamination_range=np.linspace(0.01, 0.2, 20)):
    """
    Estimate contamination using silhouette analysis
    """
    from sklearn.metrics import silhouette_score

    scores = []
    for cont in contamination_range:
        clf = IsolationForest(contamination=cont, random_state=42)
        labels = clf.fit_predict(X)

        # Silhouette score (-1 worse, 1 better)
        if len(np.unique(labels)) > 1:
            score = silhouette_score(X, labels)
            scores.append(score)
        else:
            scores.append(-1)

    best_idx = np.argmax(scores)
    best_contamination = contamination_range[best_idx]

    return best_contamination, contamination_range, scores

# Example
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
# Add outliers
rng = np.random.RandomState(42)
X = np.vstack([X, rng.uniform(low=-10, high=10, size=(50, 2))])

best_cont, cont_range, scores = estimate_contamination(X)
print(f"Estimated contamination: {best_cont:.3f} (true: 0.048)")

import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.plot(cont_range, scores, marker='o')
plt.axvline(x=best_cont, color='r', linestyle='--',
            label=f'Best: {best_cont:.3f}')
plt.xlabel('Contamination Ratio')
plt.ylabel('Silhouette Score')
plt.title('Contamination Estimation')
plt.legend()
plt.grid(True)
```

### Silhouette-Based Evaluation

```python
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt

def evaluate_clustering_quality(X, labels):
    """Evaluate anomaly detection as clustering problem"""

    # Overall silhouette score
    score = silhouette_score(X, labels)
    print(f"Overall Silhouette Score: {score:.4f}")

    # Per-sample silhouette scores
    sample_scores = silhouette_samples(X, labels)

    # Separate by cluster
    for label in np.unique(labels):
        cluster_scores = sample_scores[labels == label]
        print(f"Cluster {label}: mean={cluster_scores.mean():.4f}, "
              f"std={cluster_scores.std():.4f}, n={len(cluster_scores)}")

    return score, sample_scores

# Example
clf = IsolationForest(contamination=0.1, random_state=42)
labels = clf.fit_predict(X)
score, sample_scores = evaluate_clustering_quality(X, labels)
```

### Domain Expert Validation Workflow

```python
class ExpertValidationSystem:
    """System for expert review of detected anomalies"""

    def __init__(self, detector, top_k=100):
        self.detector = detector
        self.top_k = top_k
        self.validated_anomalies = []
        self.validated_normal = []

    def get_top_anomalies(self, X):
        """Get top K most anomalous points"""
        scores = self.detector.decision_function(X)
        # Higher score = more anomalous
        top_indices = np.argsort(scores)[-self.top_k:][::-1]
        return top_indices, scores[top_indices]

    def validate_sample(self, indices, X, y_expert):
        """Record expert validation"""
        for idx, label in zip(indices, y_expert):
            if label == 1:  # Expert confirms anomaly
                self.validated_anomalies.append(idx)
            else:
                self.validated_normal.append(idx)

    def estimate_precision(self):
        """Estimate precision from validated samples"""
        total_validated = len(self.validated_anomalies) + len(self.validated_normal)
        if total_validated == 0:
            return 0.0
        return len(self.validated_anomalies) / total_validated

    def get_statistics(self):
        """Get validation statistics"""
        return {
            'total_validated': len(self.validated_anomalies) + len(self.validated_normal),
            'confirmed_anomalies': len(self.validated_anomalies),
            'false_positives': len(self.validated_normal),
            'estimated_precision': self.estimate_precision()
        }

# Example usage
detector = IsolationForest(contamination=0.1, random_state=42)
detector.fit(X)

validator = ExpertValidationSystem(detector, top_k=50)
top_indices, top_scores = validator.get_top_anomalies(X)

# Simulate expert validation (in practice, manual review)
y_expert = np.ones(len(top_indices))  # Assume all confirmed
y_expert[:10] = 0  # Simulate 10 false positives

validator.validate_sample(top_indices, X, y_expert)
stats = validator.get_statistics()
print(f"Validation Stats: {stats}")
```

### Synthetic Anomaly Injection

```python
def inject_synthetic_anomalies(X, contamination=0.1, strategy='uniform'):
    """
    Inject known anomalies to test detector
    """
    n_samples = len(X)
    n_anomalies = int(n_samples * contamination)

    # Copy original data
    X_test = X.copy()
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    if strategy == 'uniform':
        # Random uniform noise in feature space
        mins = X.min(axis=0)
        maxs = X.max(axis=0)
        X_test[anomaly_indices] = np.random.uniform(
            mins - (maxs - mins),
            maxs + (maxs - mins),
            size=(n_anomalies, X.shape[1])
        )
    elif strategy == 'gaussian':
        # Gaussian noise with large variance
        X_test[anomaly_indices] += np.random.randn(n_anomalies, X.shape[1]) * 5
    elif strategy == 'local':
        # Local perturbations
        X_test[anomaly_indices] += np.random.randn(n_anomalies, X.shape[1]) * X.std(axis=0) * 3

    # Create ground truth
    y_true = np.zeros(n_samples)
    y_true[anomaly_indices] = 1

    return X_test, y_true, anomaly_indices

# Test detector with synthetic anomalies
X_normal, _ = make_blobs(n_samples=1000, centers=1, random_state=42)
X_test, y_true, anomaly_idx = inject_synthetic_anomalies(
    X_normal, contamination=0.05, strategy='uniform'
)

# Evaluate detector
detector = IsolationForest(contamination=0.05, random_state=42)
y_pred = detector.fit_predict(X_test)
y_pred_binary = (y_pred == -1).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_true, y_pred_binary))
```

### Metrics When Labels Available

```python
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

def evaluate_with_labels(y_true, y_pred, scores):
    """Comprehensive evaluation with ground truth"""

    # Convert predictions to binary (1=anomaly, 0=normal)
    y_pred_binary = (y_pred == -1).astype(int)

    # Classification metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    # Threshold-based metrics (using decision scores)
    auc_roc = roc_auc_score(y_true, scores)
    auc_pr = average_precision_score(y_true, scores)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)

    print("Classification Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")

    # Plot ROC and PR curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    ax1.plot(fpr, tpr, label=f'AUC = {auc_roc:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curve')
    ax1.legend()
    ax1.grid(True)

    # Precision-Recall curve
    prec, rec, _ = precision_recall_curve(y_true, scores)
    ax2.plot(rec, prec, label=f'AUC = {auc_pr:.4f}')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Precision-Recall Curve')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr
    }

# Example with labeled data
metrics = evaluate_with_labels(y_true, y_pred, -scores)  # Negate for sklearn convention
```

---

## 11. Production Anomaly Detection

Deploying anomaly detection in production requires handling real-time constraints, model updates, and operational concerns.

### Real-time vs Batch Detection Patterns

```python
import time
from collections import deque
import numpy as np

class BatchAnomalyDetector:
    """Process data in batches periodically"""

    def __init__(self, detector, batch_size=1000, retrain_interval=10):
        self.detector = detector
        self.batch_size = batch_size
        self.retrain_interval = retrain_interval
        self.buffer = []
        self.batch_count = 0

    def add_sample(self, x):
        """Add sample to buffer"""
        self.buffer.append(x)

        if len(self.buffer) >= self.batch_size:
            return self.process_batch()
        return []

    def process_batch(self):
        """Process accumulated batch"""
        X = np.array(self.buffer)

        # Retrain periodically
        if self.batch_count % self.retrain_interval == 0:
            self.detector.fit(X)

        predictions = self.detector.predict(X)
        anomalies = np.where(predictions == -1)[0]

        self.buffer = []
        self.batch_count += 1

        return anomalies

class StreamingAnomalyDetector:
    """Process each sample immediately"""

    def __init__(self, window_size=100, n_std=3):
        self.window = deque(maxlen=window_size)
        self.n_std = n_std

    def predict(self, x):
        """Predict if sample is anomaly"""
        if len(self.window) < 10:  # Need minimum samples
            self.window.append(x)
            return False

        # Calculate statistics from window
        window_array = np.array(self.window)
        mean = window_array.mean(axis=0)
        std = window_array.std(axis=0)

        # Check if x is anomaly
        z_score = np.abs((x - mean) / (std + 1e-10))
        is_anomaly = np.any(z_score > self.n_std)

        # Update window
        self.window.append(x)

        return is_anomaly

# Example: Streaming detection
stream_detector = StreamingAnomalyDetector(window_size=100, n_std=3)

# Simulate data stream
np.random.seed(42)
for i in range(1000):
    x = np.random.randn(5)  # Normal sample

    # Inject occasional anomalies
    if i % 100 == 0 and i > 0:
        x += 10

    is_anomaly = stream_detector.predict(x)
    if is_anomaly:
        print(f"Anomaly detected at sample {i}: {x}")
```

### Sliding Window Approach

```python
class SlidingWindowDetector:
    """Anomaly detection with sliding window"""

    def __init__(self, detector, window_size=500, slide_size=100):
        self.detector = detector
        self.window_size = window_size
        self.slide_size = slide_size
        self.window = deque(maxlen=window_size)
        self.anomaly_scores = []

    def update(self, X_new):
        """Update window with new data"""
        for x in X_new:
            self.window.append(x)

        # Train on current window
        if len(self.window) >= self.window_size:
            X_window = np.array(self.window)
            self.detector.fit(X_window)

            # Score new samples
            scores = self.detector.decision_function(X_new)
            self.anomaly_scores.extend(scores)

            return self.detector.predict(X_new)

        return None

# Example
from sklearn.ensemble import IsolationForest

detector = SlidingWindowDetector(
    detector=IsolationForest(contamination=0.1, random_state=42),
    window_size=500,
    slide_size=100
)

# Simulate streaming data
for batch_idx in range(10):
    X_new = np.random.randn(100, 5)

    # Add anomalies to some batches
    if batch_idx % 3 == 0:
        X_new[-5:] += 10

    predictions = detector.update(X_new)

    if predictions is not None:
        n_anomalies = np.sum(predictions == -1)
        print(f"Batch {batch_idx}: {n_anomalies} anomalies detected")
```

### Handling Concept Drift

```python
from sklearn.ensemble import IsolationForest
import numpy as np

class AdaptiveAnomalyDetector:
    """Detector that adapts to concept drift"""

    def __init__(self, base_detector, drift_threshold=0.3, adaptation_rate=0.1):
        self.base_detector = base_detector
        self.drift_threshold = drift_threshold
        self.adaptation_rate = adaptation_rate
        self.reference_scores = None

    def fit(self, X):
        """Initial training"""
        self.base_detector.fit(X)
        self.reference_scores = self.base_detector.decision_function(X)

    def detect_drift(self, X):
        """Detect if data distribution has drifted"""
        current_scores = self.base_detector.decision_function(X)

        # Compare score distributions
        ref_mean = self.reference_scores.mean()
        curr_mean = current_scores.mean()

        drift_score = abs(curr_mean - ref_mean) / (abs(ref_mean) + 1e-10)

        return drift_score > self.drift_threshold, drift_score

    def update(self, X):
        """Update model if drift detected"""
        has_drift, drift_score = self.detect_drift(X)

        if has_drift:
            print(f"Drift detected (score: {drift_score:.4f}), retraining...")

            # Retrain on recent data
            self.base_detector.fit(X)
            self.reference_scores = self.base_detector.decision_function(X)

            return True

        return False

    def predict(self, X):
        """Predict with drift handling"""
        self.update(X)
        return self.base_detector.predict(X)

# Example
detector = AdaptiveAnomalyDetector(
    IsolationForest(contamination=0.1, random_state=42),
    drift_threshold=0.3
)

# Initial training
X_train = np.random.randn(1000, 5)
detector.fit(X_train)

# Simulate drift
for i in range(5):
    # Gradually shift distribution
    X_new = np.random.randn(200, 5) + i * 0.5
    predictions = detector.predict(X_new)
    print(f"Batch {i}: {np.sum(predictions == -1)} anomalies")
```

### Alert Fatigue Reduction

```python
class SmartAlertingSystem:
    """Reduce alert fatigue with intelligent filtering"""

    def __init__(self, min_severity=0.8, suppression_window=3600,
                 max_alerts_per_hour=10):
        self.min_severity = min_severity
        self.suppression_window = suppression_window
        self.max_alerts_per_hour = max_alerts_per_hour
        self.recent_alerts = deque()
        self.suppressed_patterns = set()

    def should_alert(self, anomaly_score, pattern_id, timestamp):
        """Decide if alert should be sent"""

        # Check severity threshold
        if anomaly_score < self.min_severity:
            return False

        # Check if pattern recently suppressed
        if pattern_id in self.suppressed_patterns:
            return False

        # Clean old alerts
        current_time = timestamp
        self.recent_alerts = deque([
            (t, s, p) for t, s, p in self.recent_alerts
            if current_time - t < self.suppression_window
        ])

        # Check rate limit
        if len(self.recent_alerts) >= self.max_alerts_per_hour:
            return False

        # Record alert
        self.recent_alerts.append((timestamp, anomaly_score, pattern_id))

        return True

    def suppress_pattern(self, pattern_id, duration=7200):
        """Temporarily suppress alerts for a pattern"""
        self.suppressed_patterns.add(pattern_id)
        # In production, would use time-based expiration

# Example
alerting = SmartAlertingSystem(
    min_severity=0.8,
    suppression_window=3600,
    max_alerts_per_hour=5
)

# Simulate detections
import time
current_time = time.time()

for i in range(20):
    score = np.random.rand()
    pattern = "pattern_A" if i % 3 == 0 else "pattern_B"

    if alerting.should_alert(score, pattern, current_time + i):
        print(f"ALERT: Score {score:.2f}, Pattern {pattern}")
    else:
        print(f"Suppressed: Score {score:.2f}, Pattern {pattern}")
```

### Threshold Tuning in Production

```python
class DynamicThresholdTuner:
    """Automatically tune detection thresholds"""

    def __init__(self, initial_threshold=0.0, target_fpr=0.01,
                 update_frequency=1000):
        self.threshold = initial_threshold
        self.target_fpr = target_fpr
        self.update_frequency = update_frequency
        self.scores_buffer = []
        self.predictions_buffer = []
        self.sample_count = 0

    def update_threshold(self, scores, feedback=None):
        """Update threshold based on scores and optional feedback"""
        self.scores_buffer.extend(scores)
        self.sample_count += len(scores)

        if self.sample_count >= self.update_frequency:
            # Calculate threshold for target FPR
            sorted_scores = np.sort(self.scores_buffer)
            threshold_idx = int(len(sorted_scores) * (1 - self.target_fpr))
            self.threshold = sorted_scores[threshold_idx]

            # Reset buffer
            self.scores_buffer = []
            self.sample_count = 0

            print(f"Updated threshold to {self.threshold:.4f}")

    def predict(self, scores):
        """Predict anomalies using current threshold"""
        return scores > self.threshold

# Example
tuner = DynamicThresholdTuner(target_fpr=0.05, update_frequency=500)

detector = IsolationForest(contamination=0.1, random_state=42)
X_train = np.random.randn(1000, 5)
detector.fit(X_train)

# Simulate production scoring
for batch_idx in range(10):
    X_new = np.random.randn(100, 5)
    scores = -detector.decision_function(X_new)  # Higher = more anomalous

    predictions = tuner.predict(scores)
    tuner.update_threshold(scores)

    print(f"Batch {batch_idx}: {np.sum(predictions)} anomalies, "
          f"threshold={tuner.threshold:.4f}")
```

---

## 12. Algorithm Selection Guide

Choosing the right anomaly detection algorithm depends on data characteristics, requirements, and constraints.

### Decision Matrix

| Data Characteristic | Recommended Algorithms | Avoid |
|-------------------|----------------------|-------|
| High-dimensional (>50 features) | Isolation Forest, Feature Bagging, Autoencoders | Distance-based (LOF, KNN) |
| Low-dimensional (<10 features) | LOF, DBSCAN, Statistical methods | Deep learning |
| Large dataset (>100K samples) | Isolation Forest, Streaming algorithms | OCSVM (quadratic complexity) |
| Small dataset (<1K samples) | Statistical methods, OCSVM | Deep learning |
| Time series | STL, Prophet, ARIMA-based, ADTK | Standard spatial methods |
| Categorical features | Isolation Forest, One-hot + LOF | Statistical (assumes continuous) |
| Real-time streaming | Rolling statistics, Streaming IF | Batch methods requiring full retrain |
| Needs interpretability | Statistical methods, Decision rules | Deep learning, ensemble |
| Clustered normal data | DBSCAN, LOF | Global methods (Z-score) |
| Labeled data available | Supervised (Random Forest, XGBoost) | Unsupervised |

### Algorithm vs Characteristics Table

| Algorithm | Speed | Scalability | Interpretability | Parameter Sensitivity | Best For |
|-----------|-------|-------------|------------------|----------------------|----------|
| Z-Score | Very Fast | Excellent | High | Low | Univariate, Gaussian data |
| Isolation Forest | Fast | Excellent | Medium | Low | High-dim, mixed types |
| LOF | Slow | Poor | Medium | High (k, threshold) | Clustered data, low-dim |
| One-Class SVM | Medium | Poor | Low | High (kernel, nu) | Small datasets, complex boundaries |
| DBSCAN | Medium | Medium | High | High (eps, min_samples) | Spatial clusters |
| Elliptic Envelope | Fast | Good | Medium | Low | Gaussian distributed |
| Autoencoders | Slow | Good | Low | Very High | Complex patterns, images |
| STL Decomposition | Medium | Good | High | Medium | Seasonal time series |

### When to Use Statistical vs ML Methods

**Use Statistical Methods When:**
- Data follows known distribution (e.g., Gaussian)
- Need high interpretability and explainability
- Small dataset (<1000 samples)
- Simple anomaly patterns (extreme values)
- Domain experts can set thresholds
- Regulatory requirements for transparency

**Use ML Methods When:**
- Complex, high-dimensional data
- Unknown or mixed distributions
- Large dataset (>10K samples)
- Need to capture complex patterns
- Willing to trade interpretability for performance
- Concept drift expected

### Scalability Considerations

```python
import numpy as np
import time
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def benchmark_scalability(algorithms, data_sizes):
    """Benchmark algorithms on different data sizes"""
    results = {name: {'sizes': [], 'train_times': [], 'predict_times': []}
               for name in algorithms.keys()}

    for n_samples in data_sizes:
        print(f"\nTesting with {n_samples} samples...")
        X = np.random.randn(n_samples, 10)

        for name, algo in algorithms.items():
            # Training time
            start = time.time()
            algo.fit(X)
            train_time = time.time() - start

            # Prediction time
            start = time.time()
            predictions = algo.predict(X)
            predict_time = time.time() - start

            results[name]['sizes'].append(n_samples)
            results[name]['train_times'].append(train_time)
            results[name]['predict_times'].append(predict_time)

            print(f"{name}: train={train_time:.3f}s, predict={predict_time:.3f}s")

    return results

# Benchmark
algorithms = {
    'IForest': IsolationForest(n_estimators=100, random_state=42),
    'LOF': LocalOutlierFactor(novelty=True),
    'OCSVM': OneClassSVM(nu=0.1)
}

data_sizes = [1000, 5000, 10000]
results = benchmark_scalability(algorithms, data_sizes)

# Visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

for name, data in results.items():
    ax1.plot(data['sizes'], data['train_times'], marker='o', label=name)
    ax2.plot(data['sizes'], data['predict_times'], marker='o', label=name)

ax1.set_xlabel('Dataset Size')
ax1.set_ylabel('Training Time (s)')
ax1.set_title('Training Scalability')
ax1.legend()
ax1.grid(True)

ax2.set_xlabel('Dataset Size')
ax2.set_ylabel('Prediction Time (s)')
ax2.set_title('Prediction Scalability')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
```

---

## 13. Common Pitfalls

### Training on Contaminated Data

**Problem**: Normal training data contains unlabeled anomalies, biasing the model.

**Solution:**
```python
from sklearn.ensemble import IsolationForest
import numpy as np

def robust_training_with_cleaning(X, initial_contamination=0.1,
                                  iterations=3):
    """Iteratively clean training data"""

    X_clean = X.copy()

    for i in range(iterations):
        # Train on current clean data
        detector = IsolationForest(
            contamination=initial_contamination,
            random_state=42
        )
        detector.fit(X_clean)

        # Remove detected anomalies
        predictions = detector.predict(X_clean)
        X_clean = X_clean[predictions == 1]

        print(f"Iteration {i+1}: {len(X_clean)} samples remain "
              f"({len(X) - len(X_clean)} removed)")

    # Final training on cleaned data
    final_detector = IsolationForest(
        contamination=initial_contamination,
        random_state=42
    )
    final_detector.fit(X_clean)

    return final_detector, X_clean

# Example
X_contaminated = np.random.randn(1000, 5)
X_contaminated = np.vstack([X_contaminated,
                           np.random.uniform(-10, 10, (50, 5))])

detector, X_clean = robust_training_with_cleaning(X_contaminated)
print(f"Final clean dataset: {len(X_clean)} samples")
```

### Feature Scaling Importance

**Problem**: Features with different scales dominate distance-based methods.

**Solution:**
```python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# Generate data with different scales
X = np.column_stack([
    np.random.randn(1000) * 1,      # Small scale
    np.random.randn(1000) * 100,    # Large scale
    np.random.randn(1000) * 0.01    # Very small scale
])

# Add outlier
X = np.vstack([X, [10, 10, 10]])

# Without scaling
lof_unscaled = LocalOutlierFactor(contamination=0.01)
pred_unscaled = lof_unscaled.fit_predict(X)

# With StandardScaler
scaler_std = StandardScaler()
X_scaled_std = scaler_std.fit_transform(X)
lof_std = LocalOutlierFactor(contamination=0.01)
pred_std = lof_std.fit_predict(X_scaled_std)

# With RobustScaler (robust to outliers)
scaler_robust = RobustScaler()
X_scaled_robust = scaler_robust.fit_transform(X)
lof_robust = LocalOutlierFactor(contamination=0.01)
pred_robust = lof_robust.fit_predict(X_scaled_robust)

print(f"Unscaled: {np.sum(pred_unscaled == -1)} anomalies")
print(f"StandardScaler: {np.sum(pred_std == -1)} anomalies")
print(f"RobustScaler: {np.sum(pred_robust == -1)} anomalies")
```

### High-Dimensional Challenges

**Problem**: Curse of dimensionality - all points become equidistant in high dimensions.

**Solution:**
```python
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.random_projection import GaussianRandomProjection
import numpy as np

def handle_high_dimensions(X, method='pca', target_dim=10):
    """Reduce dimensionality before anomaly detection"""

    if method == 'pca':
        reducer = PCA(n_components=target_dim, random_state=42)
    elif method == 'random_projection':
        reducer = GaussianRandomProjection(
            n_components=target_dim,
            random_state=42
        )
    elif method == 'feature_bagging':
        # Use Isolation Forest with feature bagging
        from pyod.models.feature_bagging import FeatureBagging
        detector = FeatureBagging(
            base_estimator=IsolationForest(contamination=0.1),
            n_estimators=10,
            max_features=min(target_dim, X.shape[1]),
            contamination=0.1,
            random_state=42
        )
        return detector, None
    else:
        raise ValueError(f"Unknown method: {method}")

    X_reduced = reducer.fit_transform(X)
    detector = IsolationForest(contamination=0.1, random_state=42)

    return detector, X_reduced

# Example with high-dimensional data
X_high_dim = np.random.randn(1000, 100)  # 100 dimensions

# Different strategies
for method in ['pca', 'random_projection', 'feature_bagging']:
    detector, X_reduced = handle_high_dimensions(X_high_dim, method=method)

    if X_reduced is not None:
        detector.fit(X_reduced)
        predictions = detector.predict(X_reduced)
    else:
        detector.fit(X_high_dim)
        predictions = detector.predict(X_high_dim)

    print(f"{method}: {np.sum(predictions == -1)} anomalies detected")
```

### Ignoring Temporal Structure

**Problem**: Treating time series as independent samples loses temporal context.

**Solution:**
```python
import numpy as np
import pandas as pd

def create_temporal_features(df, value_col, window_sizes=[5, 10, 20]):
    """Add temporal features for time series anomaly detection"""

    df_features = df.copy()

    for window in window_sizes:
        # Rolling statistics
        df_features[f'rolling_mean_{window}'] = (
            df[value_col].rolling(window=window).mean()
        )
        df_features[f'rolling_std_{window}'] = (
            df[value_col].rolling(window=window).std()
        )
        df_features[f'rolling_min_{window}'] = (
            df[value_col].rolling(window=window).min()
        )
        df_features[f'rolling_max_{window}'] = (
            df[value_col].rolling(window=window).max()
        )

        # Lag features
        df_features[f'lag_{window}'] = df[value_col].shift(window)

        # Rate of change
        df_features[f'rate_of_change_{window}'] = (
            df[value_col].pct_change(window)
        )

    # Drop NaN values from rolling windows
    df_features = df_features.dropna()

    return df_features

# Example
dates = pd.date_range('2024-01-01', periods=1000, freq='H')
values = np.sin(np.arange(1000) * 2 * np.pi / 24) + np.random.randn(1000) * 0.1
df = pd.DataFrame({'timestamp': dates, 'value': values})

df_with_features = create_temporal_features(df, 'value')
print(f"Original features: {df.shape[1]}")
print(f"With temporal features: {df_with_features.shape[1]}")
print(f"New features: {df_with_features.columns.tolist()}")
```

### Over-Relying on Unsupervised Scores

**Problem**: Anomaly scores are relative and dataset-dependent, not absolute measures.

**Solution:**
```python
class CalibratedAnomalyDetector:
    """Calibrate anomaly scores using validation set"""

    def __init__(self, detector):
        self.detector = detector
        self.score_mean = None
        self.score_std = None

    def fit(self, X_train, X_val=None):
        """Fit detector and calibrate on validation set"""
        self.detector.fit(X_train)

        # Calibrate on validation set
        if X_val is not None:
            val_scores = self.detector.decision_function(X_val)
            self.score_mean = val_scores.mean()
            self.score_std = val_scores.std()
        else:
            # Use training set for calibration
            train_scores = self.detector.decision_function(X_train)
            self.score_mean = train_scores.mean()
            self.score_std = train_scores.std()

    def predict_proba(self, X):
        """Return calibrated probability-like scores"""
        raw_scores = self.detector.decision_function(X)

        # Standardize scores
        if self.score_std > 0:
            z_scores = (raw_scores - self.score_mean) / self.score_std
        else:
            z_scores = raw_scores - self.score_mean

        # Convert to pseudo-probabilities using sigmoid
        probabilities = 1 / (1 + np.exp(-z_scores))

        return probabilities

    def predict(self, X, threshold=0.9):
        """Predict using calibrated threshold"""
        probabilities = self.predict_proba(X)
        return (probabilities > threshold).astype(int)

# Example
from sklearn.model_selection import train_test_split

X = np.random.randn(1000, 5)
X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

calibrated = CalibratedAnomalyDetector(
    IsolationForest(contamination=0.1, random_state=42)
)
calibrated.fit(X_train, X_val)

# Get calibrated probabilities
X_test = np.random.randn(100, 5)
probabilities = calibrated.predict_proba(X_test)
predictions = calibrated.predict(X_test, threshold=0.9)

print(f"Probability range: [{probabilities.min():.3f}, {probabilities.max():.3f}]")
print(f"Detected {np.sum(predictions)} anomalies with threshold=0.9")
```

---

## 14. Resources and References

### Key Libraries

**PyOD (Python Outlier Detection)**
```bash
pip install pyod
```
- 40+ anomaly detection algorithms
- Unified API (fit, predict, decision_function)
- Ensemble methods (averaging, maximization, AOM, MOA)
- Model combination and feature bagging
- GitHub: https://github.com/yzhao062/pyod

**Scikit-learn**
```bash
pip install scikit-learn
```
- IsolationForest, LocalOutlierFactor, OneClassSVM
- EllipticEnvelope, DBSCAN
- Standard preprocessing and metrics
- Docs: https://scikit-learn.org/stable/modules/outlier_detection.html

**ADTK (Anomaly Detection Toolkit)**
```bash
pip install adtk
```
- Time series anomaly detection
- Rule-based and ML detectors
- Seasonal decomposition, level shift detection
- Pipeline support for complex workflows
- GitHub: https://github.com/arundo/adtk

**Ruptures**
```bash
pip install ruptures
```
- Change point detection
- PELT, Binary Segmentation, Window-based methods
- Multiple cost functions (L2, RBF, linear)
- Docs: https://centre-borelli.github.io/ruptures-docs/

**Prophet**
```bash
pip install prophet
```
- Facebook's forecasting library
- Automatic anomaly detection via prediction intervals
- Handles seasonality and holidays
- Docs: https://facebook.github.io/prophet/

**AnomalyDetection (Twitter)**
```r
install.packages("AnomalyDetection")
```
- R package for time series anomaly detection
- STL-based seasonal decomposition
- GitHub: https://github.com/twitter/AnomalyDetection

### Benchmark Datasets

**KDD Cup 1999**
- Network intrusion detection
- 4.9M samples, 41 features
- Binary (normal/attack) and multi-class labels
- http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html

**NSL-KDD**
- Improved version of KDD Cup 99
- Removes redundant records
- More balanced class distribution
- https://www.unb.ca/cic/datasets/nsl.html

**CICIDS 2017/2018**
- Modern network intrusion dataset
- Realistic background traffic
- Multiple attack types
- https://www.unb.ca/cic/datasets/ids-2017.html

**Numenta Anomaly Benchmark (NAB)**
- Real-world and artificial time series
- 58 labeled time series files
- Multiple domains (AWS server metrics, NYC taxi, etc.)
- https://github.com/numenta/NAB

**ODDS (Outlier Detection DataSets)**
- Collection of 16+ datasets
- Various domains and sizes
- Ground truth labels for evaluation
- http://odds.cs.stonybrook.edu/

**Credit Card Fraud Detection**
- Kaggle dataset with 284K transactions
- Highly imbalanced (0.172% fraud)
- PCA-transformed features for privacy
- https://www.kaggle.com/mlg-ulb/creditcardfraud

### Key Papers and Surveys

**Foundational Papers:**

1. **Isolation Forest** (2008)
   - Liu, F. T., Ting, K. M., & Zhou, Z. H.
   - "Isolation forest" - ICDM 2008
   - Introduced path length for anomaly scoring

2. **Local Outlier Factor** (2000)
   - Breunig, M. M., Kriegel, H. P., Ng, R. T., & Sander, J.
   - "LOF: identifying density-based local outliers" - ACM SIGMOD 2000
   - Density-based local anomaly detection

3. **One-Class SVM** (2001)
   - Scholkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C.
   - "Estimating the support of a high-dimensional distribution" - Neural Computation 2001
   - Kernel-based novelty detection

**Survey Papers:**

1. **Anomaly Detection: A Survey** (2009)
   - Chandola, V., Banerjee, A., & Kumar, V.
   - ACM Computing Surveys 2009
   - Comprehensive taxonomy and methods overview

2. **Deep Learning for Anomaly Detection: A Survey** (2020)
   - Pang, G., Shen, C., Cao, L., & Hengel, A. V. D.
   - ACM Computing Surveys 2020
   - Deep learning approaches and applications

3. **Outlier Detection Techniques** (2010)
   - Hodge, V. J., & Austin, J.
   - Artificial Intelligence Review 2004
   - Classification of detection methods

**Time Series Anomaly Detection:**

1. **STL Decomposition** (1990)
   - Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I.
   - "STL: A seasonal-trend decomposition" - Journal of Official Statistics 1990

2. **Prophet** (2018)
   - Taylor, S. J., & Letham, B.
   - "Forecasting at scale" - The American Statistician 2018
   - Automatic forecasting and anomaly detection

**Evaluation and Benchmarking:**

1. **On the Evaluation of Unsupervised Outlier Detection** (2016)
   - Campos, G. O., et al.
   - ACM Transactions on Knowledge Discovery from Data 2016
   - Evaluation challenges and methodologies

2. **Precision and Recall for Time Series** (2018)
   - Tatbul, N., et al.
   - NeurIPS 2018
   - Time-series-specific evaluation metrics

### Online Resources

**Tutorials and Courses:**
- PyOD Documentation: https://pyod.readthedocs.io/
- Scikit-learn User Guide: https://scikit-learn.org/stable/modules/outlier_detection.html
- Coursera - Unsupervised Learning: https://www.coursera.org/learn/unsupervised-learning
- Fast.ai - Practical Deep Learning: https://course.fast.ai/

**Blogs and Articles:**
- Towards Data Science - Anomaly Detection: https://towardsdatascience.com/tagged/anomaly-detection
- Machine Learning Mastery - Anomaly Detection: https://machinelearningmastery.com/
- Netflix Tech Blog - Anomaly Detection: https://netflixtechblog.com/

**GitHub Repositories:**
- Awesome Anomaly Detection: https://github.com/hoya012/awesome-anomaly-detection
- PyOD Examples: https://github.com/yzhao062/pyod/tree/master/examples
- Time Series Anomaly Detection: https://github.com/rob-med/awesome-TS-anomaly-detection

